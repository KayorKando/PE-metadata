"""Vanilla Aug-PE on bioRxiv, with the same drift measurements as the MAPLE runs (quota_sweep/quota_pe.py).

Question: does metadata drift come from the vote (one embedding, top-K), or from MAPLE's metadata prompts?
Aug-PE puts no metadata in any prompt, so if the same blind attributes drift here, the cause is the vote.

The algorithm is MAPLE's own Aug-PE baseline (MAPLE repo, biorxiv/MAPLE/PE.py, main_experiment.sh):
  round 0   RANDOM_API: K * (L + 1) = 14,000 texts from one fixed prompt (no metadata, no examples)
  each round  vote: every private text votes for its nearest candidate under sentence-t5 (max_seq 1024, L2)
              + N(0, sigma^2) per candidate -> top-K by vote -> evaluate the K survivors
              -> VARIATION_API x L on the survivors (no metadata, no length request) -> next candidates
  eps -> (T, sigma) as in MAPLE's main_experiment.sh (noise calibrated by GDP for Aug-PE): eps 4 is T = 9, sigma = 3.214.
Prompts are imported from MAPLE's PE.py, so the generator side is exactly MAPLE's baseline.

Unlike the MAPLE runs, each seed makes its own round-0 pool (seed-dependent vLLM sampling): no metadata means
no shared AIM pool to reuse, and the seed spread then includes round 0.

Measured (same definitions as quota_pe.py, so the two can be compared directly):
  per round, on the K survivors: Qwen labels (eval annotator), jsd_<attr> vs the Qwen-labelled test set
    (word_count measured), jsd_gemini_<attr>, jsd_mean / jsd_blind_mean / jsd_other_mean, MAUVE (sentence-t5)
  extra: jsd_prior_<attr> = JSD(survivors, LLM prior), where the prior is a random sample of the round-0 pool
    before any vote (prior.csv, prior_metrics.json). Drift toward the LLM prior shows as jsd_prior falling
    while jsd (vs real) rises.

Output in --out: pool.csv (round-0 pool), prior.csv, prior_metrics.json, round_<t>.csv (text, vote, eval_<attr>),
metrics.jsonl, qwen_cache.csv, config.json, DONE. Resumable after the last finished round.
mauve_rescore.py works on these run dirs (--root $PE_DATA/augpe/eps4.0 --runs "seed*").

    CUDA_VISIBLE_DEVICES=0 python augpe/augpe_pe.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --seed 0 \
        --test_ref $PE_DATA/quota_sweep/test_ref_qwen.csv --out $PE_DATA/augpe/eps4.0/seed0
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quota_sweep.annotators import ATTRS, CONTENT  # noqa: E402
from vote_correction import jsd  # noqa: E402

# MAPLE main_experiment.sh, PE.py (AugPE) runs: eps -> (PE_T, noise multiplier)
EPS_SETTINGS = {"4.0": (9, 3.214362874574832), "2.0": (9, 5.872135064401613), "1.0": (9, 10.889585799108582),
                "inf": (9, 0.0)}
BLIND = ["word_count", "sample_size", "dominant_data_type"]   # the MAPLE runs' B (Experiment 1)


def top_k(hist, K):
    """Indices of the K largest votes (MAPLE's get_next_iteration, selection_mode='rank'); stable on ties."""
    return np.argsort(-np.asarray(hist, float), kind="stable")[:K]


def round_metrics(ev, ref, gem=None, prior=None, blind=BLIND):
    """JSD of the survivors' labels `ev` ({attr: labels}) against the test set `ref`, MAPLE's Gemini labels of
    the test set `gem` (content attributes), and the LLM prior `prior`. Same keys as quota_pe.py, plus jsd_prior_*."""
    m = {f"jsd_{a}": jsd(ev[a], ref[a]) for a in ATTRS}
    if gem is not None:
        m.update({f"jsd_gemini_{a}": jsd(ev[a], gem[a]) for a in CONTENT})
    m["jsd_mean"] = float(np.mean([m[f"jsd_{a}"] for a in ATTRS]))
    m["jsd_blind_mean"] = float(np.mean([m[f"jsd_{a}"] for a in blind]))
    m["jsd_other_mean"] = float(np.mean([m[f"jsd_{a}"] for a in ATTRS if a not in blind]))
    if prior is not None:
        m.update({f"jsd_prior_{a}": jsd(ev[a], prior[a]) for a in ATTRS})
        m["jsd_prior_blind_mean"] = float(np.mean([m[f"jsd_prior_{a}"] for a in blind]))
        m["jsd_prior_other_mean"] = float(np.mean([m[f"jsd_prior_{a}"] for a in ATTRS if a not in blind]))
    return m


def load_augpe_prompts(maple_dir):
    """create_review_generation_prompt / create_review_variation_prompt from MAPLE's biorxiv/MAPLE/PE.py."""
    code = Path(maple_dir) / "biorxiv" / "MAPLE"
    sys.path.insert(0, str(code))                       # PE.py does `from util import ...`
    spec = importlib.util.spec_from_file_location("maple_augpe", code / "PE.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_review_generation_prompt, mod.create_review_variation_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maple_dir", required=True)
    p.add_argument("--eps", default="4.0", choices=list(EPS_SETTINGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_ref", required=True, help="Qwen-labelled test set (made by quota_pe.py --prep_only)")
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=2000, help="MAPLE PE_num_syn")
    p.add_argument("--L", type=int, default=6, help="MAPLE PE_L_random")
    p.add_argument("--T", type=int, default=None, help="override the number of PE rounds for this eps")
    p.add_argument("--noise", type=float, default=None, help="override the noise multiplier for this eps")
    p.add_argument("--blind", nargs="+", default=BLIND)
    p.add_argument("--prior_n", type=int, default=2000, help="round-0 texts labelled as the LLM-prior reference")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu_mem", type=float, default=0.5, help="vLLM gpu_memory_utilization; the rest is sentence-t5")
    p.add_argument("--emb_batch", type=int, default=32)
    args = p.parse_args()

    T, sigma = EPS_SETTINGS[args.eps]
    T = args.T if args.T is not None else T
    sigma = args.noise if args.noise is not None else sigma
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "method": "augpe", "T": T, "sigma": sigma}, indent=2))
    data_root = Path(os.environ.get("PE_DATA", "."))
    bio = Path(args.maple_dir) / "biorxiv"
    test_path = Path(args.test_ref)
    if not test_path.exists():
        raise SystemExit(f"missing {test_path}: build it with quota_sweep/quota_pe.py --prep_only")

    gen_prompt, var_prompt = load_augpe_prompts(args.maple_dir)
    from util import SuppressChineseLogitsProcessor, set_seed  # noqa: E402  (MAPLE's util.py)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from quota_sweep.annotators import make_annotator
    from quota_sweep.quota_pe import Embedder, load_mauve
    from blindness.embed import embed
    from blindness.schema import clean_labels
    from vote_correction import nearest_pool_index

    set_seed(42 + args.seed)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    llm = LLM(model=args.model_id, max_model_len=32768, gpu_memory_utilization=args.gpu_mem,
              logits_processors=[SuppressChineseLogitsProcessor], seed=args.seed)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)   # MAPLE PE.py defaults
    ann = make_annotator("qwen", args.maple_dir, llm=llm, model=args.model_id)
    ann.load_cache(out / "qwen_cache.csv")
    emb = Embedder(args.emb_batch)
    compute_mauve = load_mauve(args.maple_dir)

    def chat(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(prompts):
        return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]

    # --- private voters (sentence-t5, max_seq 1024, as MAPLE's Aug-PE) and the test reference ---
    priv = pd.read_csv(bio / "biorxiv_train_metadata.csv")
    priv = priv[priv["text"].notna()].reset_index(drop=True)
    X_priv = embed(priv["text"].astype(str).tolist(), "sentence-t5-base", 1024, batch_size=args.emb_batch,
                   cache_dir=str(data_root / "cache"))
    test = pd.read_csv(test_path)
    ref = {a: test[f"ref_{a}"].astype(str).to_numpy() for a in ATTRS}
    gem = clean_labels(test, CONTENT)
    X_test = emb(test["text"].astype(str).tolist())

    # --- round-0 pool (RANDOM_API, before any vote) and the LLM-prior reference ---
    pool_path = out / "pool.csv"
    if not pool_path.exists():
        t0 = time.time()
        n = args.K * (args.L + 1)
        pool = pd.DataFrame({"text": generate([chat(gen_prompt({}, [], 0))] * n)})
        pool.to_csv(pool_path, index=False)
        print(f"round-0 pool: {n} texts in {time.time() - t0:.0f} s", flush=True)
    pool = pd.read_csv(pool_path)
    pool = pool[pool["text"].notna() & (pool["text"].astype(str).str.strip() != "")].reset_index(drop=True)
    prior_path = out / "prior.csv"
    if not prior_path.exists():
        idx = np.random.default_rng(10_000 + args.seed).choice(len(pool), min(args.prior_n, len(pool)), replace=False)
        prior = pool.iloc[np.sort(idx)].reset_index(drop=True)
        pl = ann.label(prior)
        for a in ATTRS:
            prior[f"eval_{a}"] = pl[a]
        pm = round_metrics(pl, ref, gem, blind=args.blind)
        pm["mauve"] = float(compute_mauve(p_features=X_test, q_features=emb(prior["text"].astype(str).tolist())).mauve)
        pm["n"] = len(prior)
        (out / "prior_metrics.json").write_text(json.dumps(pm, indent=2))
        prior.to_csv(prior_path, index=False)
        ann.save_cache(out / "qwen_cache.csv")
        print("prior:", json.dumps({k: round(pm[k], 4) for k in ("mauve", "jsd_blind_mean", "jsd_other_mean")}), flush=True)
    prior = pd.read_csv(prior_path)
    prior_lab = {a: prior[f"eval_{a}"].astype(str).to_numpy() for a in ATTRS}

    # --- resume ---
    mpath = out / "metrics.jsonl"
    done = [json.loads(l) for l in mpath.read_text().splitlines()] if mpath.exists() else []
    start = len(done)
    if start > T:
        print("already finished"); (out / "DONE").write_text(time.strftime("%F %T")); return
    sel_df = pd.read_csv(out / f"round_{start - 1}.csv") if start else None
    for _ in range(start):          # keep the noise stream of an uninterrupted run
        rng.normal(size=args.K * (args.L + 1))
    if start:
        print(f"resuming after round {start - 1}", flush=True)

    for t in range(start, T + 1):
        t0 = time.time()
        if t == 0:
            cand = pool[["text"]].copy()
        else:
            survivors = sel_df[["text"]]
            var = pd.concat([survivors] * args.L, ignore_index=True)
            var["text"] = generate([chat(var_prompt({}, s, [])) for s in var["text"].astype(str)])
            cand = pd.concat([survivors, var], ignore_index=True)
        cand = cand[cand["text"].notna() & (cand["text"].astype(str).str.strip() != "")].reset_index(drop=True)
        Xc = emb(cand["text"].astype(str).tolist())
        hist = np.bincount(nearest_pool_index(X_priv, Xc), minlength=len(cand)).astype(float)
        noise = rng.normal(size=args.K * (args.L + 1))   # fixed length: same stream whatever len(cand) is
        if sigma:
            hist = hist + sigma * np.resize(noise, len(cand))
        sel = top_k(hist, args.K)
        sel_df = cand.iloc[sel].reset_index(drop=True)
        sel_df["vote"] = hist[sel]
        ev = ann.label(sel_df)
        for a in ATTRS:
            sel_df[f"eval_{a}"] = ev[a]
        m = dict(round=t, n_candidates=len(cand), mean_vote=float(hist[sel].mean()),
                 mauve=float(compute_mauve(p_features=X_test, q_features=Xc[sel]).mauve))
        m.update(round_metrics(ev, ref, gem, prior_lab, args.blind))
        m["seconds"] = round(time.time() - t0, 1)
        sel_df.to_csv(out / f"round_{t}.csv", index=False)
        ann.save_cache(out / "qwen_cache.csv")
        with open(mpath, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                          if k in ("round", "mauve", "jsd_blind_mean", "jsd_other_mean", "jsd_word_count",
                                   "jsd_prior_blind_mean", "seconds")}), flush=True)
    (out / "DONE").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    # vLLM keeps an engine subprocess alive; exit hard so a failed run does not hang the job (as quota_pe.py).
    import traceback
    code = 0
    try:
        main()
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            code = e.code or 0
        else:                       # SystemExit("message")
            print(e.code)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
