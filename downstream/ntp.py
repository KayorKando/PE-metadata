"""Downstream next-token prediction (NTP) on bioRxiv: MAPLE's protocol, plus per-text results and token counts.

MAPLE's protocol (MAPLE repo, biorxiv/utility_eval/fine-tune-nodp-biorxiv_foreval.{sh,py} and ntp_eval.py;
its requirements.txt pins transformers 4.55.4):
  model      prajjwal1/bert-small (29M) as a causal LM: BertLMHeadModel with is_decoder = True
  train set  the survivors of the last PE round (PE_iter_T.csv, column text), text only, no prompt
  recipe     512 tokens per text; 64 texts x 8 accumulated batches = 512 texts per update; 50 epochs = 200 updates
             for 2,000 texts; AdamW, lr 3e-4 constant, weight decay 0.01 (not on bias / LayerNorm), no gradient
             clipping, fp32, seed 42. No model selection: the last weights are evaluated.
  test       biorxiv_test_metadata.csv, 2,000 texts (quota_sweep/test_ref_qwen.csv has the same texts in the same order)
  metric     accuracy = correct next tokens / all tokens of the test set, and perplexity (ntp_eval.py)

What is the same here: model, tokenization, batch composition, optimizer, number of updates, the metric formulas.
The training loop is written out (no Trainer, dp_transformers, datasets or accelerate needed); it follows the
Trainer of transformers 4.55.4 step by step, and tests/test_ntp.py checks that it gives the same weights:
  - an epoch = a random order of the texts, cut into batches of 64, in groups of 8; a last group with fewer than 8
    batches also gets an update;
  - loss of a group = summed cross-entropy of all its predicted tokens / number of its non-pad tokens (the Trainer
    counts the labels before the shift, so [CLS] is in the denominator);
  - --updates 200 works like the Trainer's max_steps: for 2,000 texts it is exactly 50 epochs.
What differs, without changing the numbers: batches are padded to their longest text, not to 512 (padding is masked
in the attention and ignored in the loss); the test set is scored in batches, not one text at a time.
One real difference: is_decoder is given when the model is loaded. MAPLE's training script sets it on the config
after loading; with transformers 4.55.4 a batch in which no text has padding (every text reaches 512 tokens) is
then trained without the causal mask. On bioRxiv this is rare: even if half of the texts reach 512 tokens, the
16-text last batch of an epoch is such a batch about once in 65,000 epochs (64-text batches: never). Here every
batch is causal. MAPLE's evaluation is causal either way (its saved model is reloaded with is_decoder = True).
Even if 90 % of the texts reached the limit, it would be about 10 of the 1,600 batches of a run.

Added to MAPLE's protocol:
  - per-text results (per_text.csv): tokens, correct tokens and summed loss of every test text, also by position
    of the predicted token (0-127, 128-255, 256-383, 384-511). Any split of the test set (tail / head classes,
    length) is then a sum over rows: no new training (downstream/summarize_ntp.py).
  - the number of training tokens of every train set (result.json, token_counts.csv). Survivors of the t5 vote
    are longer than those of a fine-tuned vote, so at 2,000 texts they give more training tokens.
  - variants of the train set:
      full       all texts (MAPLE's protocol)
      tok<B>     token-matched: a random subset of the texts with at most B training tokens in total; B = the
                 smallest train set of the call (rounded down to 2 digits, e.g. 583,412 -> 580,000), kept in
                 <out>/token_budget.json and reused by later calls. Same number of updates, so only the amount
                 of text is equalised. A train set smaller than B keeps all its texts (under_budget in result.json).
      len<L>     every training text cut to its first L tokens (same number of texts for every arm)
  - train sets from a plain CSV (--csv NAME=PATH): n random rows, e.g. real training texts (non-private upper
    bound) or the round-0 pool before any vote (the LLM prior).
  - --train_seeds: several training seeds per train set (order of the texts, dropout) = the noise of the metric.

Output, one dir per (train set, variant, training seed), resumable (a dir with result.json is skipped):
  <out>/<tree>/<run>/round<t>/<variant>-ts<seed>/{result.json, per_text.csv}     PE runs
  <out>/<name>/n<n>_s<sample seed>/<variant>-ts<seed>/...                        --csv train sets
<tree> = the last two parts of --pe_root (quota_sweep/eps4.0, augpe/eps4.0). No model weights are saved.

    python downstream/ntp.py --pe_root $PE_DATA/quota_sweep/eps4.0 $PE_DATA/augpe/eps4.0 \
        --csv real=$PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv --test_csv $PE_DATA/quota_sweep/test_ref_qwen.csv \
        --out $PE_DATA/ntp --rounds 0 last --variants full tok [--shard 0 --num_shards 4]
"""
import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

MODEL = "prajjwal1/bert-small"
SEQ_LEN, BATCH, ACCUM, UPDATES, LR, WD, SEED = 512, 64, 8, 200, 3e-4, 0.01, 42
POS_EDGES = (128, 256, 384)          # buckets of the predicted token's position: <128, <256, <384, >=384
RUN_RE = re.compile(r"(?:lam([0-9.]+)(?:_seed(\d+))?|seed(\d+))$")


# ---------------- model ----------------
def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    return tok


def load_model(name, device, report=None):
    """The pretrained weights as a causal LM. is_decoder is given at load time: the attention layers read it when
    they are built. Set on the config afterwards (MAPLE's training script), transformers 4.55.4 applies the causal
    mask only to batches that contain padding (see the module docstring; tests/test_ntp.py).
    report: called with the names of the weights the checkpoint does not have (they are initialised at random,
    with the seed set before this call, as in MAPLE's script)."""
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(name, is_decoder=True, output_loading_info=True)
    if report:
        report(sorted(info["missing_keys"]))
    return model.to(device)


def encode(tok, texts, max_len=SEQ_LEN):
    """Token ids with the special tokens ([CLS] ... [SEP]), cut to max_len (MAPLE: truncation, max_length 512)."""
    return tok([str(t) for t in texts], truncation=True, max_length=max_len)["input_ids"]


def collate(ids, idx, pad_id, device):
    import torch
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
        m[r, :len(ids[i])] = 1
    return x.to(device), m.to(device)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def no_decay(name):
    """The Trainer's rule: no weight decay on biases and LayerNorm weights."""
    n = name.lower()
    return "bias" in n or "layernorm" in n


def train_lm(model, ids, pad_id, updates=UPDATES, batch=BATCH, accum=ACCUM, lr=LR, wd=WD, seed=SEED, device="cpu",
             shuffle=True, log=None):
    """MAPLE's recipe (see the module docstring). ids: token-id lists of the training texts. Returns training stats."""
    import torch
    import torch.nn.functional as F
    set_seed(seed)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in named if not no_decay(n)], "weight_decay": wd},
         {"params": [p for n, p in named if no_decay(n)], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    n = len(ids)
    per_epoch = math.ceil(math.ceil(n / batch) / accum)          # updates per epoch
    done, losses, t0 = 0, [], time.time()
    while done < updates:
        order = torch.randperm(n, generator=gen).tolist() if shuffle else list(range(n))
        batches = [order[s:s + batch] for s in range(0, n, batch)]
        for g in range(0, len(batches), accum):
            group = [collate(ids, b, pad_id, device) for b in batches[g:g + accum]]
            n_items = sum(int(m.sum()) for _, m in group)      # non-pad tokens of the group ([CLS] included)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for x, m in group:
                logits = model(input_ids=x, attention_mask=m, use_cache=False).logits[:, :-1].float()
                labels = x.masked_fill(m == 0, -100)[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                                       reduction="sum") / n_items
                loss.backward()
                total += loss.item()
            opt.step()
            done += 1
            losses.append(total)
            if log and (done == 1 or done % 50 == 0 or done == updates):
                log(f"    update {done}/{updates}  loss {total:.4f}  ({time.time() - t0:.0f}s)")
            if done >= updates:
                break
    opt.zero_grad(set_to_none=True)
    return dict(updates=done, epochs=round(done / per_epoch, 3), first_train_loss=losses[0],
                final_train_loss=float(np.mean(losses[-per_epoch:])), train_seconds=round(time.time() - t0, 1))


def evaluate(model, ids, pad_id, device="cpu", batch=64):
    """Per test text (row order of ids): n_tokens (with [CLS] and [SEP]: MAPLE's denominator), n_correct and nll
    (summed loss) over its n_tokens - 1 predicted tokens, and k<b> / c<b> = predicted / correct tokens by position
    bucket b of the predicted token (POS_EDGES)."""
    import torch
    import torch.nn.functional as F
    model.eval()
    n, nb = len(ids), len(POS_EDGES) + 1
    order = np.argsort([len(x) for x in ids], kind="stable")
    n_correct, nll = np.zeros(n, np.int64), np.zeros(n, np.float64)
    k, c = np.zeros((n, nb), np.int64), np.zeros((n, nb), np.int64)
    edges = torch.tensor(POS_EDGES, device=device)
    with torch.no_grad():
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x, m = collate(ids, idx, pad_id, device)
            logits = model(input_ids=x, attention_mask=m, use_cache=False).logits[:, :-1].float()   # j predicts j + 1
            tgt, valid = x[:, 1:], m[:, 1:].bool()
            ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none").double() * valid
            hit = (logits.argmax(-1) == tgt) & valid
            n_correct[idx] = hit.sum(1).cpu().numpy()
            nll[idx] = ce.sum(1).cpu().numpy()
            bucket = torch.bucketize(torch.arange(1, x.size(1), device=device), edges, right=True)
            for b in range(nb):
                sel = bucket == b
                k[idx, b] = valid[:, sel].sum(1).cpu().numpy()
                c[idx, b] = hit[:, sel].sum(1).cpu().numpy()
    out = pd.DataFrame({"idx": np.arange(n), "n_tokens": [len(x) for x in ids], "n_correct": n_correct, "nll": nll})
    for b in range(nb):
        out[f"k{b}"], out[f"c{b}"] = k[:, b], c[:, b]
    return out


def metrics(pt):
    """Test-set metrics from per-text rows (any subset of rows gives the metrics of that subset).
    ntp_acc and ppl_maple are MAPLE's ntp_eval.py formulas: accuracy = correct / n_tokens, where n_tokens counts
    [CLS], which is never predicted; perplexity = exp of the n_tokens-weighted mean of the per-text mean loss.
    ntp_acc_pred and loss are per predicted token (n_tokens - 1 per text)."""
    n, c, nll = (pt[col].to_numpy(float) for col in ("n_tokens", "n_correct", "nll"))
    pred = n - 1
    return dict(ntp_acc=float(c.sum() / n.sum()), ppl_maple=float(np.exp(((nll / pred) * n).sum() / n.sum())),
                ntp_acc_pred=float(c.sum() / pred.sum()), loss=float(nll.sum() / pred.sum()),
                n_test_texts=int(len(pt)), n_test_tokens=int(n.sum()))


# ---------------- train sets ----------------
def find_runs(root, patterns=("*",)):
    """PE run dirs under root, one or two levels down, named lam<x>[_seed<s>] (quota_pe.py) or seed<s>
    (augpe_pe.py), that have round files and match one of the glob patterns (relative to root)."""
    root = Path(root)
    ds = sorted({f.parent for pat in ("*/round_*.csv", "*/*/round_*.csv") for f in root.glob(pat)})
    return [d for d in ds if RUN_RE.match(d.name)
            and any(fnmatch.fnmatch(d.relative_to(root).as_posix(), p) for p in patterns)]


def finished_rounds(d):
    rounds = sorted(int(f.stem.split("_")[1]) for f in d.glob("round_*.csv") if f.stem.split("_")[1].isdigit())
    m = d / "metrics.jsonl"
    if m.exists():                       # only rounds whose metrics line was written
        done = len(m.read_text().splitlines())
        rounds = [t for t in rounds if t < done]
    return rounds


def pick_rounds(d, spec, allow_unfinished=False):
    """spec: 'last' (needs the run's DONE file), 'all', or round numbers. -> sorted rounds, last round or None."""
    have = finished_rounds(d)
    last = have[-1] if have and ((d / "DONE").exists() or allow_unfinished) else None
    out = set()
    for s in spec:
        if s == "all":
            out |= set(have)
        elif s == "last":
            if last is not None:
                out.add(last)
        elif int(s) in have:
            out.add(int(s))
    return sorted(out), last


def tree_of(root):
    return "/".join(Path(root).resolve().parts[-2:])


def train_sets(args):
    """All train sets of the call: PE rounds and --csv samples."""
    sets = []
    for root in args.pe_root or []:
        tree = tree_of(root)
        runs = find_runs(root, args.runs)
        if not runs:
            print(f"WARNING: no run dirs match {root} / {args.runs}", flush=True)
        for d in runs:
            rounds, last = pick_rounds(d, args.rounds, args.allow_unfinished)
            if "last" in args.rounds and last is None:
                print(f"WARNING: {d} has no DONE file: 'last' skipped (--allow_unfinished to use its latest round)",
                      flush=True)
            rel = d.relative_to(root).as_posix()
            m = RUN_RE.match(d.name)
            arm = d.parent.name if d.parent.resolve() != Path(root).resolve() else tree.split("/")[0]
            for t in rounds:
                sets.append(dict(kind="pe", tree=tree, arm=arm, run=d.name, lam=float(m.group(1)) if m.group(1) else None,
                                 seed=int(m.group(2) or m.group(3) or 0), round=t, is_last=(t == last),
                                 path=str(d / f"round_{t}.csv"), rel=f"{tree}/{rel}/round{t}"))
    for spec in args.csv or []:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--csv takes NAME=PATH (got '{spec}')")
        for s in args.sample_seeds:
            sets.append(dict(kind="csv", tree=name, arm=name, run=f"n{args.n}_s{s}", lam=None, seed=s, round=None,
                             is_last=False, path=path, n=args.n, rel=f"{name}/n{args.n}_s{s}"))
    return sets


def load_texts(ts):
    """Training texts of a train set (rows without text are dropped). --csv sets: n random rows, fixed by the seed."""
    df = pd.read_csv(ts["path"])
    texts = df["text"].dropna().astype(str)
    texts = texts[texts.str.strip() != ""].tolist()
    if ts["kind"] == "csv" and len(texts) > ts["n"]:
        keep = np.sort(np.random.RandomState(ts["seed"]).permutation(len(texts))[:ts["n"]])
        texts = [texts[i] for i in keep]
    return texts


def token_subset(lengths, budget, seed=0):
    """Indices of a random subset of texts whose token total is at most budget: the texts in a random order
    (fixed by seed) up to the last one that still fits. All texts if they fit."""
    lengths = np.asarray(lengths)
    order = np.random.RandomState(seed).permutation(len(lengths))
    keep = order[:int(np.searchsorted(np.cumsum(lengths[order]), budget, side="right"))]
    return np.sort(keep)


def round_down(n):
    """n rounded down to its two leading digits (583412 -> 580000)."""
    step = 10 ** max(0, len(str(int(n))) - 2)
    return int(n) // step * step


def count_tokens(tok, sets, log=print):
    """Token totals of every train set under the 512-token limit (what variant 'full' trains on)."""
    rows = []
    for ts in sets:
        texts = load_texts(ts)
        full = [len(x) for x in tok(texts, truncation=False, verbose=False)["input_ids"]]
        cut = np.minimum(full, SEQ_LEN)
        rows.append(dict(train_set=ts["rel"], n_texts=len(texts), n_tokens=int(cut.sum()),
                         n_tokens_uncut=int(np.sum(full)), n_truncated=int(np.sum(np.array(full) > SEQ_LEN)),
                         words_mean=round(float(np.mean([len(t.split()) for t in texts])), 1)))
        log(f"  {ts['rel']}: {rows[-1]['n_texts']} texts, {rows[-1]['n_tokens']:,} tokens, "
            f"{rows[-1]['n_truncated']} cut at {SEQ_LEN}")
    return pd.DataFrame(rows)


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def resolve_budget(args, tok, sets, counts=None):
    """Token budget of variant 'tok': --token_budget N, else <out>/token_budget.json, else the smallest train
    set of this call (round_down), which is then written to that file for later calls."""
    out = Path(args.out)
    f = out / "token_budget.json"
    if args.token_budget != "auto":
        return int(args.token_budget)
    if f.exists():
        return int(json.loads(f.read_text())["budget"])
    if counts is None:
        print(f"counting tokens of {len(sets)} train sets (no {f.name} yet)", flush=True)
        counts = count_tokens(tok, sets)
    i = int(counts.n_tokens.idxmin())
    budget = round_down(counts.n_tokens[i])
    write_atomic(out / "token_counts.csv", counts.to_csv(index=False))
    write_atomic(f, json.dumps(dict(budget=budget, smallest_set=counts.train_set[i],
                                    smallest_set_tokens=int(counts.n_tokens[i]), n_sets=len(counts)), indent=2))
    print(f"token budget {budget:,} (smallest: {counts.train_set[i]}, {counts.n_tokens[i]:,}) -> {f}", flush=True)
    return budget


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pe_root", nargs="*", help="run trees, e.g. $PE_DATA/quota_sweep/eps4.0 $PE_DATA/augpe/eps4.0")
    p.add_argument("--runs", nargs="+", default=["*"], help="globs of run dirs under a --pe_root, e.g. 'maple*/lam0*'")
    p.add_argument("--rounds", nargs="+", default=["last"], help="'last', 'all' or round numbers")
    p.add_argument("--allow_unfinished", action="store_true", help="'last' also for runs without a DONE file")
    p.add_argument("--csv", nargs="*", help="NAME=PATH: train on --n random rows of a CSV with a text column")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--sample_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="+", default=["full"], help="full, tok (token-matched), len<L>")
    p.add_argument("--token_budget", default="auto", help="variant tok: token total per train set, or 'auto'")
    p.add_argument("--subsample_seed", type=int, default=0, help="variant tok: which texts are kept")
    p.add_argument("--train_seeds", nargs="+", type=int, default=[SEED])
    p.add_argument("--test_csv", required=True, help="test texts, e.g. $PE_DATA/quota_sweep/test_ref_qwen.csv")
    p.add_argument("--out", required=True, help="output root, e.g. $PE_DATA/ntp")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--accum", type=int, default=ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--count_only", action="store_true", help="only count training tokens and set the token budget")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args(argv)
    for v in args.variants:
        if not re.fullmatch(r"full|tok|len\d+", v):
            p.error(f"unknown variant '{v}' (full, tok, len<L>)")
    for s in args.rounds:
        if s not in ("last", "all") and not s.isdigit():
            p.error(f"--rounds takes 'last', 'all' or numbers (got '{s}')")
    return args


def main(argv=None):
    args = parse(argv)
    sets = train_sets(args)
    if not sets:                         # a job that trains nothing must not look like a success
        sys.exit("no train sets found: check --pe_root / --runs / --rounds / --csv")
    import torch
    out = Path(args.out)
    tok = load_tokenizer(args.model)
    if args.count_only:
        counts = count_tokens(tok, sets)
        write_atomic(out / "token_counts.csv", counts.to_csv(index=False))
        print(f"{len(counts)} train sets -> {out / 'token_counts.csv'}")
        if "tok" in args.variants:
            print(f"token budget: {resolve_budget(args, tok, sets, counts):,}")
        return
    budget = resolve_budget(args, tok, sets) if "tok" in args.variants else None
    names = {v: (f"tok{budget}" if v == "tok" else v) for v in args.variants}
    # variant by variant ('full' first), so that an interrupted job leaves complete variants behind
    jobs = [dict(ts, variant=v, tag=f"{names[v]}-ts{s}", train_seed=s, out=out / ts["rel"] / f"{names[v]}-ts{s}")
            for v in args.variants for s in args.train_seeds for ts in sets]
    mine = jobs[args.shard::args.num_shards]
    todo = [j for j in mine if not (j["out"] / "result.json").exists()]
    print(f"{len(sets)} train sets, {len(jobs)} jobs; shard {args.shard}/{args.num_shards}: {len(mine)} jobs, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = pd.read_csv(args.test_csv)
    test_texts = test["text"].astype(str).tolist()
    test_ids = encode(tok, test_texts)
    test_sha = hashlib.sha1("\n".join(test_texts).encode()).hexdigest()[:16]
    import transformers
    versions = dict(torch=torch.__version__, transformers=transformers.__version__)
    commit = git_commit()
    for i, j in enumerate(todo):
        t0 = time.time()
        texts = load_texts(j)
        max_len = int(j["variant"][3:]) if j["variant"].startswith("len") else SEQ_LEN
        ids = encode(tok, texts, max_len)
        n_all, tokens_all = len(ids), int(sum(len(x) for x in ids))
        if j["variant"] == "tok":
            keep = token_subset([len(x) for x in ids], budget, args.subsample_seed)
            ids, texts = [ids[k] for k in keep], [texts[k] for k in keep]
        if not ids:
            sys.exit(f"{j['rel']} {j['tag']}: no training text left (empty train set, or a token budget below one text)")
        print(f"[{i + 1}/{len(todo)}] {j['rel']} {j['tag']}: {len(ids)} texts, {sum(len(x) for x in ids):,} tokens",
              flush=True)
        set_seed(j["train_seed"])          # before loading: weights missing in the checkpoint get a seeded init
        model = load_model(args.model, device, report=(lambda miss: print(
            f"    weights not in the checkpoint (random init): {miss or 'none'}", flush=True)) if i == 0 else None)
        tr = train_lm(model, ids, tok.pad_token_id, updates=args.updates, batch=args.batch, accum=args.accum,
                      lr=args.lr, seed=j["train_seed"], device=device, log=lambda s: print(s, flush=True))
        pt = evaluate(model, test_ids, tok.pad_token_id, device, args.eval_batch)
        res = dict(train_set=j["rel"], kind=j["kind"], tree=j["tree"], arm=j["arm"], run=j["run"], lam=j["lam"],
                   seed=j["seed"], round=j["round"], is_last=j["is_last"], variant=names[j["variant"]],
                   train_seed=j["train_seed"], source=j["path"],
                   n_train_texts=len(ids), n_train_tokens=int(sum(len(x) for x in ids)),
                   n_train_at_limit=int(sum(len(x) == max_len for x in ids)), train_max_len=max_len,
                   train_words_mean=round(float(np.mean([len(t.split()) for t in texts])), 1),
                   n_texts_full=n_all, n_tokens_full=tokens_all,
                   token_budget=budget if j["variant"] == "tok" else None,
                   under_budget=bool(j["variant"] == "tok" and tokens_all < budget),
                   **tr, **metrics(pt), model=args.model, batch=args.batch, accum=args.accum, lr=args.lr,
                   test_csv=str(args.test_csv), test_sha=test_sha, device=device, commit=commit, **versions)
        j["out"].mkdir(parents=True, exist_ok=True)
        write_atomic(j["out"] / "per_text.csv", pt.to_csv(index=False))
        write_atomic(j["out"] / "result.json", json.dumps(res, indent=2))        # written last: marks the job done
        print(f"    ntp_acc {res['ntp_acc']:.4f}  loss {res['loss']:.3f}  train loss {res['final_train_loss']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
