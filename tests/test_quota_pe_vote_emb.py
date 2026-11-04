"""Step 4 of the fine-tuned vote embedding: quota_pe.py --vote_emb {t5, ft} on a fake loop.
vLLM, MAPLE's prompt module, the Qwen annotator, sentence-t5 and MAUVE are replaced by small
deterministic fakes; the ft embedder is a tiny local BERT trained by train_embedder.py.
Checks: with t5 MAUVE gets the vote features of the survivors (old path); with ft the vote uses
the ft embedding while MAUVE still gets t5 features; a finished run resumes after its last round."""
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blindness.schema import BIORXIV_SCHEMA
from quota_sweep.annotators import InheritedAnnotator
from tests.tiny_bert import WORDS, make_tiny_bert, random_texts

CONTENT = list(BIORXIV_SCHEMA)
ATTRS = CONTENT + ["word_count"]
T5_DIM = 16


def _h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def fake_t5(texts):
    return np.stack([np.random.default_rng(_h(t)).normal(size=T5_DIM) for t in texts]).astype(np.float32) \
        if len(texts) else np.zeros((0, T5_DIM), np.float32)


class FakeEmbedder:
    def __init__(self, batch=32):
        pass

    def __call__(self, texts):
        return fake_t5(list(texts))


class FakeLLM:
    def __init__(self, model=None, **kw):
        self.n = 0

    def generate(self, prompts, sp, use_tqdm=False):
        out = []
        for p in prompts:
            self.n += 1
            rng = np.random.default_rng(_h(p) + self.n)
            out.append(SimpleNamespace(outputs=[SimpleNamespace(text=" ".join(rng.choice(WORDS, rng.integers(5, 40))))]))
        return out


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]


def make_data(root, rng):
    bio = root / "maple" / "biorxiv"
    bio.mkdir(parents=True)

    def meta(n, numeric_wc):
        df = pd.DataFrame({a: rng.choice(BIORXIV_SCHEMA[a][:3], n) for a in CONTENT})
        df["word_count"] = rng.choice([100, 150, 200], n) if numeric_wc else None
        return df

    priv = meta(150, True).assign(text=random_texts(150, 1))
    priv.to_csv(bio / "biorxiv_train_metadata.csv", index=False)
    meta(150, True).to_csv(bio / "biorxiv_train_metadata_aim_eps4.0_(1,9).csv", index=False)
    pool = meta(120, True).assign(text=random_texts(120, 2))
    pool.to_csv(root / "pool.csv", index=False)
    test = meta(60, True).assign(text=random_texts(60, 3))
    from blindness.schema import word_count_bin
    for a in CONTENT:
        test[f"ref_{a}"] = test[a]
    test["ref_word_count"] = word_count_bin(test["text"])
    test.to_csv(root / "test_ref.csv", index=False)
    # Qwen-labelled pool for training the ft embedder
    pq = pool.copy()
    for a in CONTENT:
        pq[f"map_{a}"] = pq[a]
    pq["word_count_requested"] = pq["word_count"]
    pq.to_csv(root / "pool_qwen.csv", index=False)
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fakes in place, data on disk, PE_DATA = tmp_path. Returns (root, run(**flags) -> out dir, mauve_calls)."""
    import transformers
    import quota_sweep.quota_pe as qp
    root = make_data(tmp_path, np.random.default_rng(0))
    monkeypatch.setenv("PE_DATA", str(root))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "MAPLE", types.SimpleNamespace(
        create_review_variation_prompt=lambda meta, text, ex: [{"role": "user", "content": text}]))
    monkeypatch.setitem(sys.modules, "util", types.SimpleNamespace(SuppressChineseLogitsProcessor=object,
                                                                   set_seed=lambda s: None))
    auto = transformers.AutoTokenizer
    orig = auto.from_pretrained                     # patch the class itself: "fake-qwen" -> FakeTok

    def from_pretrained(name, *a, **k):
        return FakeTok() if name == "fake-qwen" else orig(name, *a, **k)
    monkeypatch.setattr(auto, "from_pretrained", staticmethod(from_pretrained))
    monkeypatch.setattr(qp, "make_annotator", lambda name, *a, **k: InheritedAnnotator())
    monkeypatch.setattr(qp, "Embedder", FakeEmbedder)
    monkeypatch.setattr("blindness.embed.embed", lambda texts, *a, **k: fake_t5(list(texts)))
    calls = []

    def fake_mauve(p_features, q_features):
        calls.append(np.array(q_features))
        assert p_features.shape[1] == q_features.shape[1] == T5_DIM      # MAUVE always on t5 features
        return SimpleNamespace(mauve=float(np.round(np.abs(q_features.mean(0) - p_features.mean(0)).mean(), 6)))
    monkeypatch.setattr(qp, "load_mauve", lambda maple_dir: fake_mauve)

    def run(out, T=1, **flags):
        argv = ["quota_pe.py", "--maple_dir", str(root / "maple"), "--pool_csv", str(root / "pool.csv"),
                "--eps", "4.0", "--lam", "0.5", "--loop_annotator", "inherited", "--test_ref",
                str(root / "test_ref.csv"), "--K", "20", "--L", "2", "--T", str(T), "--model_id", "fake-qwen",
                "--out", str(out)]
        for k, v in flags.items():
            argv += [f"--{k}", str(v)]
        monkeypatch.setattr(sys, "argv", argv)
        qp.main()
        return out
    return root, run, calls


def train_ft(root, monkeypatch):
    from vote_embedding.train_embedder import main as train_main
    bert = make_tiny_bert(root / "tiny_bert", hidden=32, heads=2)
    path = root / "vote_embedding" / "eps4.0" / "normsoftmax"       # the default --vote_emb_path
    monkeypatch.setattr(sys, "argv", ["x", "--pool_qwen", str(root / "pool_qwen.csv"), "--model", bert,
                                      "--epochs", "1", "--max_len", "32", "--out", str(path)])
    train_main()
    return path


def metrics(out):
    return [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]


def test_t5_mauve_gets_vote_features(env):
    root, run, calls = env
    out = run(root / "t5")
    m = metrics(out)
    assert [r["round"] for r in m] == [0, 1] and (out / "DONE").exists()
    for t, q in enumerate(calls):                                        # Xs = Xc[sel] = t5 of the survivors
        surv = pd.read_csv(out / f"round_{t}.csv")["text"].astype(str).tolist()
        assert np.allclose(q, fake_t5(surv))
    assert json.loads((out / "config.json").read_text())["vote_emb"] == "t5"


def test_ft_end_to_end_mauve_on_t5(env, monkeypatch):
    root, run, calls = env
    path = train_ft(root, monkeypatch)
    out = run(root / "ft", vote_emb="ft")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["vote_emb"] == "ft" and cfg["vote_emb_path"] == str(path)
    assert len(metrics(out)) == 2 and len(calls) == 2
    for t, q in enumerate(calls):
        surv = pd.read_csv(out / f"round_{t}.csv")["text"].astype(str).tolist()
        assert len(surv) == 20 and np.allclose(q, fake_t5(surv))        # t5 of the survivors, not ft features
    assert list((root / "cache").glob("emb_ft-normsoftmax-*.npy"))      # X_priv from the ft embedder, cached


def test_ft_vote_differs_from_t5_and_resumes(env, monkeypatch, capsys):
    root, run, calls = env
    train_ft(root, monkeypatch)
    a = run(root / "t5", T=0)
    b = run(root / "ft", T=0, vote_emb="ft")
    sa, sb = (set(pd.read_csv(x / "round_0.csv")["text"]) for x in (a, b))
    assert sa != sb                                                       # the vote embedding changes the selection
    run(root / "ft", T=1, vote_emb="ft")                                  # continue the finished T=0 run
    assert "resuming after round 0" in capsys.readouterr().out
    assert [r["round"] for r in metrics(b)] == [0, 1]
