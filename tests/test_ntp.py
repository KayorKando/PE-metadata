"""downstream/ntp.py and downstream/summarize_ntp.py on a tiny random BERT LM (no download).

What is checked:
  - the model is causal (a later token never changes an earlier prediction) and padding changes nothing;
  - the evaluation gives MAPLE's ntp_eval.py numbers (its formulas are re-typed here, one text at a time);
  - the training loop gives the same model as the transformers Trainer called the way MAPLE calls it
    (skipped without accelerate);
  - token-matched subsets, run discovery, 'last' round, shards, resuming, the token budget file;
  - the summary: per-text rows add up to the run's numbers, groups partition the test set, paired differences.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from downstream import ntp, summarize_ntp  # noqa: E402
from tests.tiny_bert import WORDS, random_texts  # noqa: E402


def tiny_lm(path, dropout=0.1, hidden=32, max_pos=64, seed=0):
    """A random BertLMHeadModel + word-level tokenizer in a local dir. Like prajjwal1/bert-small, the saved
    config does not say is_decoder: the loader has to set it."""
    from transformers import BertConfig, BertLMHeadModel, BertTokenizerFast
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True, model_max_length=max_pos)
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=max_pos, hidden_dropout_prob=dropout,
                     attention_probs_dropout_prob=dropout)
    BertLMHeadModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def logits_of(model, ids, pad_id):
    x, m = ntp.collate(ids, list(range(len(ids))), pad_id, "cpu")
    with torch.no_grad():
        return model(input_ids=x, attention_mask=m).logits


# ---------------- model ----------------
def test_model_is_causal_and_padding_is_neutral(tmp_path):
    path = tiny_lm(tmp_path / "m")
    tok, model = ntp.load_tokenizer(path), ntp.load_model(path, "cpu").eval()
    assert model.config.is_decoder
    a = ntp.encode(tok, ["cell gene protein mouse human tumor neuron brain"])[0]
    b = list(a)
    b[5:] = ntp.encode(tok, ["plant yeast data model method study cohort sample"])[0][5:]     # change tokens 5..
    la, lb = logits_of(model, [a], tok.pad_token_id), logits_of(model, [b], tok.pad_token_id)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=1e-6)            # predictions before the change: the same
    assert not torch.allclose(la[0, 5:], lb[0, 5:], atol=1e-3)
    short = ntp.encode(tok, ["cell gene protein"])[0]
    both = logits_of(model, [a, short], tok.pad_token_id)             # short is padded to a's length
    assert torch.allclose(both[1, :len(short)], logits_of(model, [short], tok.pad_token_id)[0], atol=1e-5)


def test_is_decoder_at_load_vs_maple_set_after_load(tmp_path):
    """MAPLE's training script sets config.is_decoder after loading. With padding in the batch that gives our
    logits. Without any padding, transformers 4.55.4 skips the causal mask for such a model (the attention layers
    were built as an encoder's); ours stays causal, which is why load_model sets is_decoder at load time."""
    path = tiny_lm(tmp_path / "m")
    tok = ntp.load_tokenizer(path)
    ours = ntp.load_model(path, "cpu").eval()
    maple = transformers.AutoModelForCausalLM.from_pretrained(path).eval()
    ids = ntp.encode(tok, random_texts(6, seed=3))                    # different lengths: the batch has padding
    bidirectional = logits_of(maple, ids, tok.pad_token_id)
    maple.config.is_decoder = True                                    # fine-tune-nodp-biorxiv_foreval.py
    assert torch.allclose(logits_of(ours, ids, tok.pad_token_id), logits_of(maple, ids, tok.pad_token_id), atol=1e-6)
    assert not torch.allclose(logits_of(ours, ids, tok.pad_token_id), bidirectional, atol=1e-3)
    full = ntp.encode(tok, [" ".join(WORDS[:8]), " ".join(WORDS[8:16])])      # two texts of one length: no padding
    alt = [list(full[0]), list(full[1])]
    alt[0][6:] = full[1][6:]
    change = (logits_of(ours, full, tok.pad_token_id) - logits_of(ours, alt, tok.pad_token_id))[0, :6].abs().max()
    assert change == 0                                                # ours: still causal


# ---------------- evaluation ----------------
def maple_calculate_metrics(model, tok, texts, max_token_len):
    """MAPLE biorxiv/utility_eval/ntp_eval.py, calculate_metrics with no_prompt=True (same statements)."""
    total_nll = total_tokens = total_correct = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            input_ids = tok(text, return_tensors="pt", truncation=True, max_length=max_token_len).input_ids
            n = input_ids.shape[1]
            outputs = model(input_ids, labels=input_ids.clone())
            total_nll += (outputs.loss * n).item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions[:, :-1] == input_ids[:, 1:]).sum().item()
            total_tokens += n
    return float(np.exp(total_nll / total_tokens)), total_correct / total_tokens


def test_evaluate_matches_maple_ntp_eval(tmp_path):
    path = tiny_lm(tmp_path / "m")
    tok, model = ntp.load_tokenizer(path), ntp.load_model(path, "cpu")
    texts = random_texts(37, seed=5, lo=1, hi=90) + ["cell"]          # some longer than the limit, one very short
    ids = ntp.encode(tok, texts, 48)
    pt = ntp.evaluate(model, ids, tok.pad_token_id, batch=8)
    m = ntp.metrics(pt)
    ppl, acc = maple_calculate_metrics(model, tok, texts, 48)
    assert m["ntp_acc"] == pytest.approx(acc, abs=1e-12)
    assert m["ppl_maple"] == pytest.approx(ppl, rel=1e-5)
    assert list(pt.idx) == list(range(len(texts))) and pt.n_tokens.max() == 48
    assert (pt.n_tokens == [len(x) for x in ids]).all()
    k, c = pt[[f"k{b}" for b in range(4)]].sum(axis=1), pt[[f"c{b}" for b in range(4)]].sum(axis=1)
    assert (k == pt.n_tokens - 1).all() and (c == pt.n_correct).all()        # buckets partition the predicted tokens
    assert m["ntp_acc_pred"] > m["ntp_acc"] and m["loss"] > 0
    half = ntp.metrics(pt.iloc[:10])                                    # a subset of rows = the metrics of that subset
    assert half["ntp_acc"] == pytest.approx(maple_calculate_metrics(model, tok, texts[:10], 48)[1], abs=1e-12)


def test_position_buckets(tmp_path, monkeypatch):
    monkeypatch.setattr(ntp, "POS_EDGES", (4, 8, 12))
    path = tiny_lm(tmp_path / "m")
    tok, model = ntp.load_tokenizer(path), ntp.load_model(path, "cpu")
    ids = ntp.encode(tok, [" ".join(WORDS[:10]), " ".join(WORDS[:2])])    # 12 and 4 tokens with [CLS], [SEP]
    pt = ntp.evaluate(model, ids, tok.pad_token_id)
    assert pt[["k0", "k1", "k2", "k3"]].values.tolist() == [[3, 4, 4, 0], [3, 0, 0, 0]]   # predicted positions 1..n-1


# ---------------- training ----------------
def trainer_as_maple(path, tok, texts, tmp, is_decoder_at_load=False):
    """transformers.Trainer with MAPLE's arguments (small batch, texts in order): the reference of train_lm."""
    from torch.utils.data import SequentialSampler
    ref = transformers.AutoModelForCausalLM.from_pretrained(path, **({"is_decoder": True} if is_decoder_at_load else {}))
    ref.config.is_decoder = True                                                 # MAPLE: set after loading
    enc = tok(texts, padding="max_length", truncation=True, max_length=16)       # MAPLE pads every text to the limit
    data = [{k: v[i] for k, v in enc.items()} for i in range(len(texts))]

    class InOrder(transformers.Trainer):
        def _get_train_sampler(self, *a, **k):
            return SequentialSampler(self.train_dataset)

    targs = transformers.TrainingArguments(
        output_dir=str(tmp), per_device_train_batch_size=4, gradient_accumulation_steps=2, max_steps=7,
        learning_rate=1e-3, lr_scheduler_type="constant", weight_decay=0.01, max_grad_norm=0, seed=0,
        save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True, use_cpu=True)
    InOrder(model=ref, args=targs, train_dataset=data,
            data_collator=transformers.DataCollatorForLanguageModeling(tok, mlm=False)).train()
    return ref


def weight_gap(a, b):
    """Largest difference of any weight, except attention key biases: their true gradient is zero (softmax ignores
    a shift of all keys), so Adam turns rounding noise into steps there, and they do not change the output."""
    sa, sb = a.state_dict(), b.state_dict()
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if "key.bias" not in k)


def test_training_matches_the_trainer_as_maple_calls_it(tmp_path):
    """Same texts in the same order (no shuffling), dropout off: train_lm and transformers.Trainer with MAPLE's
    arguments must end with the same weights. 26 texts, batches of 4, 2 accumulated: 7 batches per epoch, so the
    last group of an epoch has 1 batch; 7 updates stop inside the second epoch (max_steps)."""
    pytest.importorskip("accelerate")
    path = tiny_lm(tmp_path / "m", dropout=0.0)
    tok = ntp.load_tokenizer(path)

    def ours(texts):
        model = ntp.load_model(path, "cpu")
        stats = ntp.train_lm(model, ntp.encode(tok, texts, 16), tok.pad_token_id, updates=7, batch=4, accum=2,
                             lr=1e-3, seed=0, shuffle=False)
        assert stats["updates"] == 7 and stats["epochs"] == pytest.approx(7 / 4)
        return model

    # every text shorter than the limit (as MAPLE's data: a batch of 64 texts always has padding)
    short = random_texts(26, seed=1, lo=2, hi=12)
    a, start = ours(short), ntp.load_model(path, "cpu")
    assert weight_gap(a, start) > 1e-3                                           # training moved the weights
    assert weight_gap(a, trainer_as_maple(path, tok, short, tmp_path / "hf1")) < 1e-5
    # texts that reach the limit (the last batch has no padding): the same, if the Trainer's model is causal there
    mixed = random_texts(26, seed=1, lo=2, hi=20)
    assert any(len(x) < 16 for x in ntp.encode(tok, mixed, 16)[:24]) and all(len(x) == 16 for x in ntp.encode(tok, mixed, 16)[24:])
    assert weight_gap(ours(mixed), trainer_as_maple(path, tok, mixed, tmp_path / "hf2", is_decoder_at_load=True)) < 1e-5


def test_training_learns_and_is_repeatable(tmp_path):
    path = tiny_lm(tmp_path / "m")
    tok = ntp.load_tokenizer(path)
    texts = ["the cell of the mouse is a model for the study of cancer"] * 20 + random_texts(20, seed=2)
    ids = ntp.encode(tok, texts, 32)
    test_ids = ntp.encode(tok, ["the cell of the mouse is a model for the study of cancer"] * 3, 32)
    runs = []
    for _ in range(2):
        model = ntp.load_model(path, "cpu")
        before = ntp.metrics(ntp.evaluate(model, test_ids, tok.pad_token_id))["ntp_acc"]
        stats = ntp.train_lm(model, ids, tok.pad_token_id, updates=60, batch=8, accum=2, lr=3e-3, seed=7)
        runs.append(ntp.metrics(ntp.evaluate(model, test_ids, tok.pad_token_id)))
        assert stats["final_train_loss"] < stats["first_train_loss"]
    assert runs[0]["ntp_acc"] > before + 0.3 and runs[0] == runs[1]      # same seed on CPU: the same numbers


def test_no_decay_groups_follow_the_trainer(tmp_path):
    from transformers import Trainer
    model = ntp.load_model(tiny_lm(tmp_path / "m"), "cpu")
    decay = set(Trainer.get_decay_parameter_names(None, model))        # also lists the tied decoder weight
    names = [n for n, _ in model.named_parameters()]                   # each tensor once, as the optimizer sees them
    ours = {n for n in names if not ntp.no_decay(n)}
    assert ours == {n for n in names if n in decay} and 0 < len(ours) < len(names)


def test_token_subset():
    lengths = np.array([10, 20, 30, 40, 50, 60])
    keep = ntp.token_subset(lengths, 100, seed=0)
    order = np.random.RandomState(0).permutation(6)
    assert lengths[keep].sum() <= 100 and (keep == np.sort(order[:len(keep)])).all()
    assert lengths[order[:len(keep) + 1]].sum() > 100                    # the next text would not fit
    assert (ntp.token_subset(lengths, 100, seed=0) == keep).all()
    assert len(ntp.token_subset(lengths, 10 ** 6)) == 6 and len(ntp.token_subset(lengths, 5)) == 0


# ---------------- the command line, on a small fake tree ----------------
def write_round(d, t, texts):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts, "vote": 1.0}).to_csv(d / f"round_{t}.csv", index=False)


def fake_tree(tmp, T=2):
    """quota_sweep/eps4.0: maple (short and long texts over rounds), maple-ft, classifier/lam0.5; augpe/eps4.0;
    a 'real' CSV; a test CSV with category / ref_* columns. lam0_seed2 of maple is unfinished (no DONE)."""
    data = tmp / "data"
    sweep, aug = data / "quota_sweep" / "eps4.0", data / "augpe" / "eps4.0"
    k = 0
    for arm, lam_dirs, hi in [("maple", ["lam0", "lam0_seed1", "lam0_seed2"], 40), ("maple-ft", ["lam0", "lam0_seed1"], 20),
                              ("classifier", ["lam0.5", "lam0.5_seed1"], 30)]:
        for name in lam_dirs:
            d = sweep / arm / name
            for t in range(T + 1):
                k += 1
                write_round(d, t, random_texts(40, seed=k, lo=5, hi=10 + (hi - 10) * (t + 1) // (T + 1)))
            (d / "metrics.jsonl").write_text("".join(json.dumps({"round": t}) + "\n" for t in range(T + 1)))
            if name != "lam0_seed2":
                (d / "DONE").write_text("x")
    (sweep / "maple" / "notarun").mkdir()
    write_round(sweep / "maple" / "notarun", 0, ["cell"])                 # not a run dir name: ignored
    for s in range(2):
        d = aug / f"seed{s}"
        for t in range(T + 2):
            k += 1
            write_round(d, t, random_texts(40, seed=k, lo=5, hi=25))
        (d / "metrics.jsonl").write_text("".join(json.dumps({"round": t}) + "\n" for t in range(T + 2)))
        (d / "DONE").write_text("x")
    real = pd.DataFrame({"text": random_texts(100, seed=900, lo=5, hi=30) + [None, " "]})
    real.to_csv(data / "real.csv", index=False)
    n = 60
    texts = random_texts(n, seed=901, lo=3, hi=60)
    test = pd.DataFrame({"text": texts, "category": ["neuro"] * 36 + ["cell"] * 18 + ["zoo"] * 6,       # tail: zoo
                         "ref_research_goal": (["a"] * 5 + ["b"] * 4 + ["c"]) * 6,                      # tail: c
                         "ref_sample_size": ["x"] * 30 + ["y"] * 30,                                    # no tail class
                         "word_count": [len(t.split()) for t in texts]})
    test["ref_word_count"] = (test.word_count / 10).round().astype(int) * 10
    test.to_csv(data / "test_ref_qwen.csv", index=False)
    return data


def cli(data, model, *extra, out="ntp"):
    return ["--pe_root", str(data / "quota_sweep" / "eps4.0"), str(data / "augpe" / "eps4.0"),
            "--csv", f"real={data / 'real.csv'}", "--n", "40", "--sample_seeds", "0", "1",
            "--test_csv", str(data / "test_ref_qwen.csv"), "--out", str(data / out), "--model", model,
            "--updates", "3", "--batch", "8", "--accum", "2", *extra]


def results(root):
    return {f.parent.relative_to(root).as_posix(): json.loads(f.read_text()) for f in Path(root).rglob("result.json")}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ntp")
    data, model = fake_tree(tmp), tiny_lm(tmp / "m")
    ntp.main(cli(data, model, "--rounds", "0", "last", "--variants", "full", "tok", "--count_only"))
    for shard in range(2):
        ntp.main(cli(data, model, "--rounds", "0", "last", "--variants", "full", "tok",
                     "--shard", str(shard), "--num_shards", "2"))
    return data, model


def test_cli_train_sets_and_outputs(tree):
    data, model = tree
    res = {k: v for k, v in results(data / "ntp").items() if k.endswith("-ts42")}
    sets = sorted({k.rsplit("/", 1)[0] for k in res})
    # 6 finished runs x (round 0, last) + the unfinished run's round 0 + 2 Aug-PE seeds x (0, last = 3) + 2 real samples
    assert len(sets) == 6 * 2 + 1 + 2 * 2 + 2 and len(res) == 2 * len(sets)               # variants full and tok
    assert "quota_sweep/eps4.0/maple/lam0_seed2/round0" in sets
    assert not any("lam0_seed2/round2" in s or "notarun" in s for s in sets)
    r = res["quota_sweep/eps4.0/maple-ft/lam0_seed1/round2/full-ts42"]
    assert (r["arm"], r["lam"], r["seed"], r["round"], r["is_last"], r["variant"]) == ("maple-ft", 0.0, 1, 2, True, "full")
    assert r["n_train_texts"] == 40 and r["updates"] == 3 and r["n_test_texts"] == 60 and not r["under_budget"]
    a = res["augpe/eps4.0/seed1/round3/full-ts42"]
    assert (a["arm"], a["tree"], a["lam"], a["seed"], a["is_last"]) == ("augpe", "augpe/eps4.0", None, 1, True)
    assert not res["augpe/eps4.0/seed1/round0/full-ts42"]["is_last"]
    real = res["real/n40_s1/full-ts42"]
    assert (real["kind"], real["arm"], real["seed"], real["round"]) == ("csv", "real", 1, None)
    assert real["n_train_texts"] == 40 and real["source"].endswith("real.csv")
    pt = pd.read_csv(data / "ntp" / "real/n40_s1/full-ts42/per_text.csv")
    assert ntp.metrics(pt)["ntp_acc"] == pytest.approx(real["ntp_acc"]) and len(pt) == 60


def test_cli_token_budget(tree):
    data, model = tree
    res = results(data / "ntp")
    counts = pd.read_csv(data / "ntp" / "token_counts.csv").set_index("train_set")
    b = json.loads((data / "ntp" / "token_budget.json").read_text())
    budget, smallest = b["budget"], int(counts.n_tokens.min())
    assert budget == ntp.round_down(smallest) <= smallest and b["smallest_set_tokens"] == smallest
    assert ntp.round_down(583412) == 580000 and ntp.round_down(87) == 87
    full = {k.rsplit("/", 1)[0]: v for k, v in res.items() if k.endswith("/full-ts42")}
    tok = {k.rsplit("/", 1)[0]: v for k, v in res.items() if k.endswith(f"/tok{budget}-ts42")}
    assert set(full) == set(tok) == set(counts.index)
    for k, v in tok.items():
        f = full[k]
        assert f["n_train_tokens"] == counts.n_tokens[k] == v["n_tokens_full"]       # the count is what 'full' trains on
        assert v["n_train_tokens"] <= budget and v["token_budget"] == budget and not v["under_budget"]
        assert v["updates"] == f["updates"] == 3                                     # same number of updates
        if f["n_train_tokens"] - budget > 64:                                        # a longer set loses texts
            assert v["n_train_texts"] < f["n_train_texts"] and v["n_train_tokens"] > budget - 64
    long_, short = tok["quota_sweep/eps4.0/maple/lam0/round2"], tok["quota_sweep/eps4.0/maple-ft/lam0/round2"]
    assert long_["n_train_texts"] < short["n_train_texts"]               # longer texts: fewer of them fit
    # an explicit budget above the set: all texts stay, and the result says so
    ntp.main(cli(data, model, "--rounds", "last", "--runs", "maple-ft/lam0", "--variants", "tok", "--token_budget",
                 "1000000", "--csv", out="ntp_big"))
    (r,) = results(data / "ntp_big").values()
    assert r["under_budget"] and r["variant"] == "tok1000000" and r["n_train_texts"] == 40
    with pytest.raises(SystemExit, match="no training text left"):
        ntp.main(cli(data, model, "--rounds", "last", "--runs", "maple-ft/lam0", "--variants", "tok", "--token_budget",
                     "1", "--csv", out="ntp_zero"))


def test_cli_resumes_and_selects(tree, capsys):
    data, model = tree
    before = {f: f.stat().st_mtime_ns for f in (data / "ntp").rglob("result.json")}
    ntp.main(cli(data, model, "--rounds", "0", "last", "--variants", "full", "tok"))
    assert "0 to run" in capsys.readouterr().out
    assert before == {f: f.stat().st_mtime_ns for f in (data / "ntp").rglob("result.json")}
    ntp.main(cli(data, model, "--rounds", "1", "--runs", "maple/lam0", "seed*", "--train_seeds", "42", "43",
                 "--csv", out="ntp_sel"))
    assert sorted(results(data / "ntp_sel")) == sorted(
        f"{s}/round1/full-ts{t}" for s in ["quota_sweep/eps4.0/maple/lam0", "augpe/eps4.0/seed0", "augpe/eps4.0/seed1"]
        for t in (42, 43))
    ntp.main(cli(data, model, "--rounds", "last", "--runs", "maple/lam0_seed2", "--allow_unfinished", "--csv",
                 out="ntp_unf"))
    assert list(results(data / "ntp_unf")) == ["quota_sweep/eps4.0/maple/lam0_seed2/round2/full-ts42"]
    with pytest.raises(SystemExit, match="no train sets"):
        ntp.main(cli(data, model, "--runs", "nothing*", "--csv", out="ntp_none"))


# ---------------- summary ----------------
def test_tail_classes_rule():
    s = pd.Series(["a"] * 60 + ["b"] * 25 + ["c"] * 11 + ["d"] * 4)      # d: under 10 records, c: 11 %, b would pass 20 %
    assert summarize_ntp.tail_classes(s) == ["c"]
    assert summarize_ntp.tail_classes(pd.Series(["a"] * 50 + ["b"] * 50)) == []


def test_summary(tree):
    data, model = tree
    ntp.main(cli(data, model, "--rounds", "last", "--runs", "maple/lam0", "--train_seeds", "43", "44", "--csv"))
    out = data / "ntp_summary"
    summarize_ntp.main(["--root", str(data / "ntp"), "--test_ref", str(data / "test_ref_qwen.csv"), "--out", str(out),
                        "--min_group", "3"])
    runs, sets = pd.read_csv(out / "ntp_runs.csv"), pd.read_csv(out / "ntp_settings.csv")
    paired, groups = pd.read_csv(out / "ntp_paired.csv"), pd.read_csv(out / "ntp_groups.csv")
    noise = pd.read_csv(out / "ntp_noise.csv")
    assert len(runs) == len(results(data / "ntp")) and (out / "NTP.md").read_text().startswith("# ")
    # settings: mean over PE seeds (train seeds averaged inside a train set first)
    res = results(data / "ntp")
    s = sets[(sets.tree == "quota_sweep/eps4.0") & (sets.arm == "maple") & (sets["round"] == 2) & (sets.variant == "full")]
    seed0 = np.mean([res[f"quota_sweep/eps4.0/maple/lam0/round2/full-ts{t}"]["ntp_acc"] for t in (42, 43, 44)])
    seed1 = res["quota_sweep/eps4.0/maple/lam0_seed1/round2/full-ts42"]["ntp_acc"]
    assert len(s) == 1 and s.n.item() == 2 and s.ntp_acc.item() == pytest.approx((seed0 + seed1) / 2)
    assert bool(s.is_last.item()) and s.n_train_tokens.item() > 0
    # paired with the t5 baseline (maple, lambda 0) by PE seed
    p = paired[(paired.arm == "maple-ft") & (paired["round"] == 2) & (paired.variant == "full") & (paired.metric == "ntp_acc")]
    d = [res[f"quota_sweep/eps4.0/maple-ft/{r}/round2/full-ts42"]["ntp_acc"] for r in ("lam0", "lam0_seed1")]
    assert len(p) == 1 and p.n.item() == 2 and p["diff"].item() == pytest.approx(np.mean([d[0] - seed0, d[1] - seed1]))
    assert p.ci95.item() == pytest.approx(12.706 * np.std([d[0] - seed0, d[1] - seed1], ddof=1) / np.sqrt(2), rel=1e-3)
    # Aug-PE: its last round against MAPLE's last round, and at MAPLE's round number
    a = paired[(paired.arm == "augpe") & (paired.variant == "full") & (paired.metric == "ntp_acc")]
    assert sorted(a.comparison.tolist()) == ["last vs last", "round 0"]     # round 2 of Aug-PE was not trained on
    assert a[a.comparison == "last vs last"][["round", "round_base"]].values.tolist() == [[3, 2]]
    assert "maple" not in set(paired[paired.tree == "quota_sweep/eps4.0"].arm)           # the baseline is not compared with itself
    # groups: tail + head cover the test set; the position buckets cover the predicted tokens
    g = groups[(groups.arm == "maple-ft") & (groups["round"] == 2) & (groups.variant == "full")]
    r0 = res["quota_sweep/eps4.0/maple-ft/lam0/round2/full-ts42"]
    for name in ("category", "ref_research_goal"):
        gg = g[g.grouping == name].set_index("group")
        assert sorted(gg.index) == ["head", "tail"] and gg.n_test_texts.to_dict() == {"head": 54, "tail": 6}
        assert gg.n_test_tokens.sum() == r0["n_test_tokens"] and gg.d_ntp_acc.notna().all()
    assert "ref_sample_size" not in set(groups.grouping) and "ref_word_count" not in set(groups.grouping)
    pt = pd.read_csv(data / "ntp" / "quota_sweep/eps4.0/maple-ft/lam0/round2/full-ts42/per_text.csv")
    tail = [ntp.metrics(pd.read_csv(data / "ntp" / f"quota_sweep/eps4.0/maple-ft/{r}/round2/full-ts42/per_text.csv").iloc[54:])
            ["ntp_acc"] for r in ("lam0", "lam0_seed1")]                 # the 6 'zoo' texts are the last rows
    assert g[(g.grouping == "category") & (g.group == "tail")].ntp_acc.item() == pytest.approx(np.mean(tail))
    assert set(g[g.grouping == "length"].group) <= {"short", "middle", "long"}
    assert g[g.grouping == "length"].n_test_texts.sum() == 60
    pos = g[g.grouping == "position"]
    assert pos.n_test_tokens.sum() == r0["n_test_tokens"] - r0["n_test_texts"] and len(pt) == 60
    # noise: sd over the three training seeds of maple/lam0
    assert len(noise) == 1 and noise.n_train_seeds.item() == 3
    assert noise.ntp_acc_sd.item() == pytest.approx(np.std(
        [res[f"quota_sweep/eps4.0/maple/lam0/round2/full-ts{t}"]["ntp_acc"] for t in (42, 43, 44)], ddof=1))


# ---------------- Nano5 job script ----------------
def test_nano5_ntp_sbatch(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    log = tmp_path / "calls.log"
    stub = "#!/usr/bin/env bash\n"
    for name, body in {"python": stub + f'echo "PY CUDA=$CUDA_VISIBLE_DEVICES OFFLINE=$HF_HUB_OFFLINE $*" >> {log}\n',
                       "module": stub, "nvidia-smi": stub, "git": stub + "echo abc123\n", "conda": stub}.items():
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    work = tmp_path / "work"
    (work / "augpe" / "eps4.0").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "NANO5_DATA": str(work), "USER": "tester",
           "CUDA_VISIBLE_DEVICES": "2,5", "SLURM_JOB_ID": "7", "SLURM_CPUS_PER_TASK": "16", "SLURM_SUBMIT_DIR": str(REPO),
           "NTP_LOGDIR": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "downstream/nano5_ntp.sbatch"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [l.split() for l in log.read_text().splitlines()]
    count, shards, summary = [c for c in calls if "--count_only" in c], [c for c in calls if "--shard" in c], calls[-1]
    assert len(count) == 1 and len(shards) == 2 and "downstream/summarize_ntp.py" in summary
    assert sorted(c[1] for c in shards) == ["CUDA=2", "CUDA=5"] and all(c[2] == "OFFLINE=1" for c in calls)
    for i, c in enumerate(sorted(shards, key=lambda c: c[c.index("--shard") + 1])):
        assert c[c.index("--shard") + 1] == str(i) and c[c.index("--num_shards") + 1] == "2"
        roots = c[c.index("--pe_root") + 1:c.index("--pe_root") + 3]
        assert roots == [f"{work}/quota_sweep/eps4.0", f"{work}/augpe/eps4.0"]
        assert c[c.index("--csv") + 1:c.index("--csv") + 3] == [
            f"real={work}/MAPLE/biorxiv/biorxiv_train_metadata.csv", f"pool_eps4.0={work}/round0/eps4.0/pool.csv"]
        assert c[c.index("--rounds") + 1:c.index("--rounds") + 3] == ["0", "last"]
        assert c[c.index("--variants") + 1:c.index("--variants") + 3] == ["full", "tok"]
        assert c[c.index("--out") + 1] == f"{work}/ntp" and c[c.index("--test_csv") + 1].endswith("test_ref_qwen.csv")
    # narrower call: one source, one variant, extra training seeds
    log.write_text("")
    env.update(SOURCES="maple", VARIANTS="full", ROUNDS="last", TRAIN_SEEDS="43 44", RUNS_GLOB="maple/lam0 maple-ft/lam0",
               CUDA_VISIBLE_DEVICES="0", EPS="2.0")
    r = subprocess.run(["bash", "downstream/nano5_ntp.sbatch"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (c,) = [l.split() for l in log.read_text().splitlines() if "--shard" in l]
    assert "--csv" not in c and c[c.index("--pe_root") + 1] == f"{work}/quota_sweep/eps2.0"
    assert c[c.index("--runs") + 1:c.index("--runs") + 3] == ["maple/lam0", "maple-ft/lam0"]
    assert c[c.index("--train_seeds") + 1:c.index("--train_seeds") + 3] == ["43", "44"]
    assert not any("--count_only" in l for l in log.read_text().splitlines())             # no 'tok' variant: no count
    # a data root of another dataset (NANO5_DATA left over from OpenReview work) is refused
    log.write_text("")
    (work / "DATASET").write_text("openreview\n")
    r = subprocess.run(["bash", "downstream/nano5_ntp.sbatch"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "bioRxiv only" in r.stdout and log.read_text() == ""
