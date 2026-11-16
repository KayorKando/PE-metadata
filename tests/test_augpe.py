"""Aug-PE drift runs (augpe/): selection, metrics, a full tiny run of augpe_pe.py with stub LLM / annotator /
embeddings (round-0 pool, prior, T rounds, resume), the summary, and the Nano5 submit + job scripts."""
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from augpe import augpe_pe, summarize_augpe  # noqa: E402
from quota_sweep.annotators import ATTRS, CONTENT  # noqa: E402

STUB = "#!/usr/bin/env bash\n"


def test_top_k_rank_and_ties():
    h = np.array([1.0, 5.0, 3.0, 5.0, -2.0])
    assert list(augpe_pe.top_k(h, 3)) == [1, 3, 2]          # largest first, ties in index order
    assert len(augpe_pe.top_k(h, 10)) == 5


def test_round_metrics_keys_and_values():
    lab = {a: np.array(["x", "y", "x", "z"]) for a in ATTRS}
    m = augpe_pe.round_metrics(lab, lab, lab, lab)
    assert all(m[f"jsd_{a}"] == 0 and m[f"jsd_prior_{a}"] == 0 for a in ATTRS)
    assert all(f"jsd_gemini_{a}" in m for a in CONTENT) and "jsd_gemini_word_count" not in m
    other = {**lab, "word_count": np.array(["50", "50", "50", "50"])}
    m = augpe_pe.round_metrics(other, lab, None, None)
    assert m["jsd_word_count"] > 0 and m["jsd_blind_mean"] == pytest.approx(m["jsd_word_count"] / 3)
    assert m["jsd_other_mean"] == 0 and "jsd_prior_word_count" not in m


# ---------------------------------------------------------------- full tiny run with stubs
def _h(text, k=8):
    return np.frombuffer(hashlib.sha256(str(text).encode()).digest()[:k], np.uint8).astype(np.float32)


def fake_maple(root):
    code = root / "biorxiv" / "MAPLE"
    code.mkdir(parents=True)
    (code / "PE.py").write_text(
        "def create_review_generation_prompt(new_metadata, examples, n_sample=5):\n"
        "    assert new_metadata == {} and examples == []\n"
        "    return [{'role': 'user', 'content': 'GEN'}]\n"
        "def create_review_variation_prompt(new_metadata, text_to_variate, examples):\n"
        "    assert new_metadata == {}\n"
        "    return [{'role': 'user', 'content': 'VAR ' + text_to_variate}]\n")
    (code / "util.py").write_text("def set_seed(s):\n    pass\nclass SuppressChineseLogitsProcessor:\n    pass\n")
    rng = np.random.default_rng(0)
    words = ["cell", "gene", "mouse", "brain", "data", "model"]
    priv = pd.DataFrame({"text": [" ".join(rng.choice(words, rng.integers(5, 120))) for _ in range(40)]})
    priv.to_csv(root / "biorxiv" / "biorxiv_train_metadata.csv", index=False)
    test = pd.DataFrame({"text": priv["text"][:20]})
    for a in ATTRS:
        test[f"ref_{a}"] = ["a", "b"] * 10
    for a in CONTENT:
        test[a] = ["a", "b"] * 10
    test.to_csv(root.parent / "test_ref.csv", index=False)


class FakeLLM:
    calls = []

    def __init__(self, *a, **k):
        self.rng = np.random.default_rng(k.get("seed", 0))

    def generate(self, prompts, sp, use_tqdm=False):
        FakeLLM.calls.append(len(prompts))
        out = []
        for p in prompts:
            text = " ".join(["w%d" % self.rng.integers(50)] * int(self.rng.integers(3, 200)))
            out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]))
        return out


class FakeAnn:
    def __init__(self):
        self.labelled = 0

    def label(self, df):
        from blindness.schema import word_count_bin
        texts = df["text"].astype(str).tolist()
        self.labelled += len(texts)
        out = {a: np.array(["ab"[int(_h(t)[i % 8]) % 2] for t in texts]) for i, a in enumerate(CONTENT)}
        out["word_count"] = word_count_bin(texts)
        return out

    def load_cache(self, p):
        pass

    def save_cache(self, p):
        Path(p).write_text("text\n")


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    fake_maple(tmp_path / "MAPLE")
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, lambda **k: k
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda m: types.SimpleNamespace(
        apply_chat_template=lambda msgs, tokenize, add_generation_prompt: msgs[-1]["content"]))
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.delitem(sys.modules, "util", raising=False)
    import quota_sweep.annotators as A
    import quota_sweep.quota_pe as Q
    import blindness.embed as E
    ann = FakeAnn()
    monkeypatch.setattr(A, "make_annotator", lambda *a, **k: ann)
    emb = lambda texts: np.stack([_h(t) for t in texts])
    monkeypatch.setattr(Q, "Embedder", lambda batch: emb)
    monkeypatch.setattr(Q, "load_mauve", lambda d: lambda p_features, q_features: types.SimpleNamespace(mauve=0.5))
    monkeypatch.setattr(E, "embed", lambda texts, *a, **k: emb(texts))
    FakeLLM.calls = []
    return tmp_path, ann


def run_main(tmp, *extra):
    argv = ["augpe_pe.py", "--maple_dir", str(tmp / "MAPLE"), "--test_ref", str(tmp / "test_ref.csv"),
            "--out", str(tmp / "run"), "--K", "10", "--L", "2", "--prior_n", "8", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        augpe_pe.main()
    finally:
        sys.argv = old


def test_tiny_run_and_resume(stubbed):
    tmp, ann = stubbed
    run_main(tmp, "--T", "2")
    out = tmp / "run"
    assert FakeLLM.calls == [30, 20, 20]               # pool K(L+1), then L variations of K survivors per round
    pool = pd.read_csv(out / "pool.csv")
    assert len(pool) == 30
    prior = json.loads((out / "prior_metrics.json").read_text())
    assert prior["n"] == 8 and "jsd_blind_mean" in prior and "mauve" in prior
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == [0, 1, 2] and (out / "DONE").exists()
    for r in rows:
        assert r["n_candidates"] <= 30 and all(f"jsd_{a}" in r and f"jsd_prior_{a}" in r for a in ATTRS)
    r1 = pd.read_csv(out / "round_1.csv")
    assert len(r1) == 10 and {"text", "vote"} <= set(r1.columns) and all(f"eval_{a}" in r1 for a in ATTRS)
    assert r1["vote"].is_monotonic_decreasing
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["T"] == 2 and cfg["sigma"] == pytest.approx(3.214362874574832) and cfg["method"] == "augpe"
    # resume: drop the last round and DONE, rerun -> only round 2 is redone, pool and prior are reused
    (out / "DONE").unlink()
    (out / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    FakeLLM.calls = []
    run_main(tmp, "--T", "2")
    assert FakeLLM.calls == [20]
    rows2 = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows2] == [0, 1, 2] and (out / "DONE").exists()


def test_eps_settings_match_maple_augpe():
    assert augpe_pe.EPS_SETTINGS["4.0"] == (9, 3.214362874574832)
    assert augpe_pe.EPS_SETTINGS["inf"][1] == 0.0


# ---------------------------------------------------------------- summary
def write_run(d, rounds, f, prior=None):
    d.mkdir(parents=True)
    with open(d / "metrics.jsonl", "w") as fh:
        for t in rounds:
            m = {"round": t, "mauve": 0.6 + 0.01 * t}
            m.update({f"jsd_{a}": f(a, t) for a in ATTRS})
            m["jsd_blind_mean"] = np.mean([m[f"jsd_{a}"] for a in augpe_pe.BLIND])
            m["jsd_other_mean"] = np.mean([m[f"jsd_{a}"] for a in ATTRS if a not in augpe_pe.BLIND])
            m["jsd_mean"] = np.mean([m[f"jsd_{a}"] for a in ATTRS])
            fh.write(json.dumps(m) + "\n")
    if prior is not None:
        (d / "prior_metrics.json").write_text(json.dumps({"n": 100, "mauve": 0.3, **prior}))


def test_summary(tmp_path):
    rng = np.random.default_rng(1)
    for s in range(3):
        write_run(tmp_path / "augpe" / f"seed{s}", range(10),
                  lambda a, t: (0.1 + 0.04 * t if a == "word_count" else 0.2) + rng.normal(0, 0.005),
                  prior={"jsd_word_count": 0.1})
    (tmp_path / "augpe" / "seed2" / "mauve_gpt2-large.jsonl").write_text(json.dumps({"round": 9, "mauve_gpt2-large": 0.4}))
    write_run(tmp_path / "maple" / "lam0", range(5), lambda a, t: 0.1 + 0.02 * t)
    write_run(tmp_path / "maple" / "lam0_seed1", range(5), lambda a, t: 0.1 + 0.02 * t)
    write_run(tmp_path / "maple" / "lam0.5", range(5), lambda a, t: 9.0)          # not lambda = 0: ignored
    R = summarize_augpe.load(tmp_path / "augpe", tmp_path / "maple")
    assert set(R.method) == {"augpe", "maple"} and R[R.method == "maple"].seed.nunique() == 2
    assert (R["round"] == -1).sum() == 3 and R["jsd_word_count"].max() < 9
    C = summarize_augpe.change(R).set_index("metric")
    assert C.loc["jsd_word_count", "verdict"] == "up" and C.loc["jsd_word_count", "delta_mean"] == pytest.approx(0.36, abs=0.02)
    assert C.loc["jsd_model_organism", "verdict"] == "n.s."
    subprocess.run([sys.executable, "augpe/summarize_augpe.py", "--augpe_root", str(tmp_path / "augpe"),
                    "--maple_root", str(tmp_path / "maple"), "--out", str(tmp_path / "out")], cwd=REPO, check=True,
                   capture_output=True)
    for f in ["drift_runs.csv", "drift_rounds.csv", "drift_change.csv", "DRIFT.md", "drift_curves.png", "drift_attrs.png"]:
        assert (tmp_path / "out" / f).stat().st_size > 0
    md = (tmp_path / "out" / "DRIFT.md").read_text()
    assert "mauve_gpt2-large" in md and "| word_count |" in md and "round 4" in md


# ---------------------------------------------------------------- Nano5 scripts
def stubs(tmp):
    b = tmp / "bin"
    b.mkdir()
    log = tmp / "calls.log"
    files = {"sbatch": STUB + f'echo "SBATCH $* | SEEDS=$SEEDS" >> {log}; echo "Submitted batch job 42"\n',
             "python": STUB + f'echo "PY CUDA=$CUDA_VISIBLE_DEVICES $*" >> {log}\n',
             "module": STUB, "nvidia-smi": STUB, "git": STUB + "echo abc123\n", "conda": STUB}
    for name, body in files.items():
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    w = tmp / "work"
    for f in ["quota_sweep/test_ref_qwen.csv", "MAPLE/biorxiv/biorxiv_train_metadata.csv", "MAPLE/biorxiv/MAPLE/PE.py"]:
        (w / f).parent.mkdir(parents=True, exist_ok=True)
        (w / f).write_text("x")
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "NANO5_DATA": str(w), "USER": "tester"}
    return env, log


def test_submit_and_job(tmp_path):
    env, log = stubs(tmp_path)
    done = tmp_path / "work" / "augpe" / "eps4.0" / "seed3"
    done.mkdir(parents=True)
    (done / "DONE").write_text("x")
    r = subprocess.run(["bash", "augpe/nano5_augpe_submit.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (c,) = [l for l in log.read_text().splitlines() if l.startswith("SBATCH")]
    assert "--gres=gpu:4" in c and "--cpus-per-task=32" in c and "--mem=256G" in c and "--time=03:00:00" in c
    assert c.endswith("SEEDS=0 1 2 4") and "done already: seed3" in r.stdout
    log.write_text("")
    e = {**env, "SEEDS": "0 1 2 4", "CUDA_VISIBLE_DEVICES": "2,3,5,7", "SLURM_JOB_ID": "42",
         "SLURM_CPUS_PER_TASK": "32", "SLURM_SUBMIT_DIR": str(REPO)}
    j = subprocess.run(["bash", "augpe/nano5_augpe.sbatch"], cwd=REPO, env=e, capture_output=True, text=True)
    assert j.returncode == 0, j.stdout + j.stderr
    py = [l.split() for l in log.read_text().splitlines() if "augpe_pe.py" in l]
    assert sorted(p[1] for p in py) == ["CUDA=2", "CUDA=3", "CUDA=5", "CUDA=7"]
    for p in py:
        s = p[p.index("--seed") + 1]
        assert p[p.index("--out") + 1].endswith(f"augpe/eps4.0/seed{s}") and p[p.index("--eps") + 1] == "4.0"


def test_submit_nothing_left_and_missing_input(tmp_path):
    env, log = stubs(tmp_path)
    r = subprocess.run(["bash", "augpe/nano5_augpe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "SEEDS": "0 1 2 3 4 5 6 7 8"})
    assert r.returncode != 0 and "> 8 per node" in r.stdout
    (tmp_path / "work" / "MAPLE" / "biorxiv" / "MAPLE" / "PE.py").unlink()
    r = subprocess.run(["bash", "augpe/nano5_augpe_submit.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "PE.py" in r.stdout and not log.exists()
