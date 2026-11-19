"""OpenReview cases. Run by tests/test_openreview.py in a subprocess with PE_DATASET=openreview (the schema and the
dataset settings are fixed at import, so they cannot share a process with the bioRxiv tests).

A tiny fake MAPLE/openreview tree (CSV layout as MAPLE's: "word count" in the private set, no PE.LABEL_ID in the
AIM file, one (area, recommendation) pair only in AIM) and stub vLLM / tokenizer / annotator / embeddings / MAUVE.
Checks: schema cleaning and prompt; loaders; round-0 pool per label id; class-conditional vote and K_g survivors
per label id in quota_pe.py (lambda 0 and 0.5) and augpe_pe.py; blind attributes from blind.json; the k-NN blind
screen; the SciBERT-style embedder trains on the 12 attributes; the Aug-PE summary."""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pe_data  # noqa: E402
from blindness.schema import DATASET, SCHEMA, clean_labels, word_count_bin  # noqa: E402
from pe_data import LABEL_ID, METADATA  # noqa: E402

assert DATASET == "openreview", "run through tests/test_openreview.py (PE_DATASET=openreview)"
CONTENT = list(SCHEMA)
ATTRS = CONTENT + ["word_count"]
AREAS, RECS = SCHEMA["area"][:3], SCHEMA["recommendation"][:3]
PAIRS = [(a, r) for a in AREAS for r in RECS]        # 9 pairs: the private set has the first 8
WORDS = "model data method results experiments baseline novelty clarity theory proof training loss".split()

UTIL = '''import random
import numpy as np
import pandas as pd
def set_seed(seed_value=42):
    random.seed(seed_value); np.random.seed(seed_value)
class SuppressChineseLogitsProcessor:
    pass
def sample_metadata_for_randomapi(df_trn, target_ids, num_sample_per_id, PE_L_random):   # MAPLE util.py, verbatim
    temp_list = []
    for target_id in target_ids:
        candidate = df_trn[df_trn['PE.LABEL_ID']==target_id]
        num_sample_rand = PE_L_random*num_sample_per_id[target_id]
        num_full = num_sample_rand//len(candidate)
        num_remain = num_sample_rand - num_full*len(candidate)
        for _ in range(num_full):
            temp_list.append(candidate)
        if num_remain>0:
            temp_list.append(candidate.sample(n=num_remain,replace=False))
    return pd.concat(temp_list, ignore_index=True)
'''
MAPLE_PY = '''def create_review_variation_prompt(new_metadata, text_to_variate, examples):
    return [{"role": "user", "content": "VAR %d %s" % (int(new_metadata["word_count"]), text_to_variate)}]
class API:
    last = None
    def __init__(self, model_id, sampling_params, examples, n_sample=5, METADATA=None):
        self.examples, self.n_sample, self.METADATA = examples, n_sample, METADATA
        API.last = self
    def RANDOM_API(self, metadata_df, sampling_params=None):
        metadata_df["text"] = ["gen " + " ".join(str(v).split()[0] for v in r) for r in metadata_df[self.METADATA].values]
        return metadata_df
'''
PE_PY = '''def create_review_generation_prompt(new_metadata, examples, n_sample=5):
    assert new_metadata == {} and examples == []
    return [{"role": "user", "content": "GEN"}]
def create_review_variation_prompt(new_metadata, text_to_variate, examples):
    return [{"role": "user", "content": "VAR " + text_to_variate}]
'''


def _h(s):
    return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16)


def text(rng, lo=5, hi=60):
    return " ".join(rng.choice(WORDS, rng.integers(lo, hi)))


def meta(rng, n, pairs):
    idx = rng.integers(len(pairs), size=n)
    df = pd.DataFrame({"area": [pairs[i][0] for i in idx], "recommendation": [pairs[i][1] for i in idx]})
    for a in CONTENT[2:]:
        df[a] = rng.choice(SCHEMA[a][:3], n)
    df["word_count"] = rng.choice([100.0, 150.0, 200.0], n)
    return df


def make_maple(root):
    """root/MAPLE/openreview with MAPLE's file layout; returns the MAPLE dir."""
    rng = np.random.default_rng(0)
    d = root / "MAPLE" / "openreview"
    (d / "MAPLE").mkdir(parents=True)
    (d / "MAPLE" / "util.py").write_text(UTIL)
    (d / "MAPLE" / "MAPLE.py").write_text(MAPLE_PY)
    (d / "MAPLE" / "PE.py").write_text(PE_PY)
    ids = {p: i for i, p in enumerate(PAIRS[:8])}

    def with_ids(df):
        return df.assign(**{LABEL_ID: [ids[(a, r)] for a, r in zip(df.area, df.recommendation)]})

    priv = with_ids(meta(rng, 160, PAIRS[:8])).assign(text=[text(rng) for _ in range(160)])
    priv = priv.rename(columns={"word_count": "word count"})          # as MAPLE's train CSV
    priv.loc[priv.index[:6], "primary_strength"] = np.nan                 # MAPLE's private labels have NaN here
    priv.loc[priv.index[3:12], "primary_weakness"] = np.nan
    priv.to_csv(d / "train_dataset_with_metadata_v4.csv", index=False)
    for name, n in [("val", 60), ("test", 60)]:
        v = with_ids(meta(rng, n, PAIRS[:8])).assign(text=[text(rng) for _ in range(n)])
        v.to_csv(d / f"{name}_dataset_with_metadata_v4.csv", index=False)
    aim = meta(rng, 200, PAIRS)[METADATA]                              # no PE.LABEL_ID; pair 9 is not private
    aim.to_csv(d / "openreview_train_metadata_aim_eps4.0_(1,9).csv", index=False)
    return root / "MAPLE"


class FakeLLM:
    calls = []

    def __init__(self, *a, **k):
        self.n = 0

    def generate(self, prompts, sp, use_tqdm=False):
        FakeLLM.calls.append(len(prompts))
        out = []
        for p in prompts:
            self.n += 1
            rng = np.random.default_rng(_h(p) + self.n)
            out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text(rng, 3, 50))]))
        return out


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]


class FakeAnn:
    """Labels every attribute of the schema from a hash of the text; word_count measured."""
    name = "fake"

    def label(self, df):
        texts = df["text"].astype(str).tolist()
        out = {a: np.array([SCHEMA[a][_h(t + a) % 3] for t in texts]) for a in CONTENT}
        out["word_count"] = word_count_bin(texts)
        return out

    def load_cache(self, p):
        pass

    def save_cache(self, p):
        Path(p).write_text("text\n")


def fake_emb(texts):
    return np.stack([np.random.default_rng(_h(t)).normal(size=8) for t in texts]).astype(np.float32) \
        if len(texts) else np.zeros((0, 8), np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    maple = make_maple(tmp_path)
    (tmp_path / "DATASET").write_text("openreview\n")
    monkeypatch.setenv("PE_DATA", str(tmp_path))
    for m in ("MAPLE", "PE", "util"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw))
    import transformers
    auto = transformers.AutoTokenizer
    orig = auto.from_pretrained
    monkeypatch.setattr(auto, "from_pretrained",
                        staticmethod(lambda name, *a, **k: FakeTok() if name == "fake-qwen" else orig(name, *a, **k)))
    import quota_sweep.annotators as A
    import quota_sweep.quota_pe as Q
    import blindness.embed as E
    from quota_sweep.annotators import InheritedAnnotator
    monkeypatch.setattr(A, "make_annotator", lambda name, *a, **k: InheritedAnnotator() if name == "inherited" else FakeAnn())
    monkeypatch.setattr(Q, "make_annotator", A.make_annotator)
    monkeypatch.setattr(Q, "Embedder", lambda batch=32: fake_emb)
    monkeypatch.setattr(E, "embed", lambda texts, *a, **k: fake_emb(list(texts)))
    monkeypatch.setattr(Q, "load_mauve", lambda d: lambda p_features, q_features: types.SimpleNamespace(mauve=0.5))
    FakeLLM.calls = []
    return tmp_path, maple, monkeypatch


def main_with(monkeypatch, fn, argv):
    monkeypatch.setattr(sys, "argv", ["x"] + [str(a) for a in argv])
    fn()


def k_per_class(maple, K):
    return pe_data.num_sample_per_id(pe_data.load_private(maple)[LABEL_ID], K)


# ---------------------------------------------------------------- schema, prompt, loaders
def test_schema_cleaning_and_prompt():
    df = pd.DataFrame({"actionability_of_feedback": ["Low Actionability", "Highly Actionable (Specific experiments to run)"],
                       "primary_strength": ["Theoretical Analysis", "Strong Empirical Performance"],
                       "novelty_assessment": [None, "Groundbreaking"], "text": ["a b", "c"]})
    lab = clean_labels(df, ["actionability_of_feedback", "primary_strength", "novelty_assessment", "word_count"])
    assert lab["actionability_of_feedback"][0] == "Low Actionability (Vague or fundamental conceptual issues)"
    assert list(lab["primary_strength"]) == ["Other", "Strong Empirical Performance"]
    assert lab["novelty_assessment"][0] == "Unclear/Not Discussed" and list(lab["word_count"]) == ["0", "0"]
    p = pe_data.openreview_prompt("THE REVIEW TEXT")
    assert "THE REVIEW TEXT" in p and all(f'"{a}"' in p for a in CONTENT) and "Poor (Very difficult to understand)" in p
    from reannotate.annotate_qwen import SCHEMA as GUIDED
    assert GUIDED["required"] == CONTENT and GUIDED["properties"]["recommendation"]["enum"] == SCHEMA["recommendation"]
    import annotate_gemini
    assert annotate_gemini.ATTRS == CONTENT and annotate_gemini.load_maple_prompt("/nowhere")(text="x").endswith(
        "**Your output (JSON only):**\n")


def test_settings():
    T, s = pe_data.EPS_SETTINGS["4.0"]
    assert T == 4 and s == pytest.approx(1.876414555237503 * np.sqrt(5 / 3))
    assert 5 / (2 * s ** 2) == pytest.approx(3 / (2 * 1.876414555237503 ** 2))       # same rho_PE as MAPLE's T = 2
    assert pe_data.EPS_SETTINGS["1.0"] == (2, 6.325387702318415) and pe_data.EPS_SETTINGS["inf"] == (9, 0.0)
    assert pe_data.AUGPE_EPS_SETTINGS["4.0"] == (9, 3.000778920378073)
    assert pe_data.BLIND_CANDIDATES == ["primary_strength", "primary_weakness", "keywords_group_1", "keywords_group_2",
                                        "keywords_group_3", "word_count"]
    assert pe_data.FT_BACKBONE == "scibert" and pe_data.DEFAULT_BLIND is None


def test_loaders(tmp_path):
    maple = make_maple(tmp_path)
    priv = pe_data.load_private(maple)
    assert "word_count" in priv and "word count" not in priv and priv[LABEL_ID].nunique() == 8
    aim = pe_data.load_aim(maple, "4.0")
    raw = pd.read_csv(maple / "openreview" / "openreview_train_metadata_aim_eps4.0_(1,9).csv")
    n9 = int(((raw.area == PAIRS[8][0]) & (raw.recommendation == PAIRS[8][1])).sum())
    assert n9 > 0 and len(aim) == len(raw) - n9                         # the non-private pair is dropped
    m = dict(zip(zip(priv.area, priv.recommendation), priv[LABEL_ID]))
    assert all(m[(a, r)] == g for a, r, g in zip(aim.area, aim.recommendation, aim[LABEL_ID]))
    inf = pe_data.load_aim(maple, "inf")
    assert len(inf) == len(priv) and LABEL_ID in inf
    with pytest.raises(SystemExit):
        pe_data.check_data_root(tmp_path)                                # no DATASET file = a bioRxiv root
    (tmp_path / "DATASET").write_text("openreview")
    pe_data.check_data_root(tmp_path)


# ---------------------------------------------------------------- round 0
def test_round0_pool(env):
    tmp, maple, mp = env
    import gen_round0
    out = tmp / "round0" / "eps4.0"
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "4.0", "--num_syn", 40, "--L_random", 2,
                                    "--out", out])
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "4.0", "--merge", "--num_shards", 1, "--out", out])
    pool = pd.read_csv(out / "pool.csv")
    k = k_per_class(maple, 40)
    assert sum(k.values()) == 40 and len(pool) == 3 * 40
    assert pool[LABEL_ID].value_counts().to_dict() == {g: 3 * n for g, n in k.items()}
    assert set(METADATA) <= set(pool.columns) and (pool.word_count_requested == pool.word_count).all()
    import MAPLE as fake
    ex = fake.API.last.examples
    assert len(ex) == 8 + 1 and all(set(m) == set(METADATA) for m, _ in ex)   # one per label id + one more


# ---------------------------------------------------------------- MAPLE PE (quota_pe.py)
def make_pool(tmp, maple, mp):
    import gen_round0
    out = tmp / "round0" / "eps4.0"
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "4.0", "--num_syn", 40, "--L_random", 2, "--out", out])
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "4.0", "--merge", "--out", out])
    return out / "pool.csv"


def run_pe(mp, maple, pool, tmp, out, *extra):
    import quota_sweep.quota_pe as Q
    main_with(mp, Q.main, ["--maple_dir", maple, "--pool_csv", pool, "--eps", "4.0", "--test_ref", tmp / "test_ref.csv",
                           "--K", 40, "--L", 2, "--T", 1, "--model_id", "fake-qwen", "--out", out, *extra])
    return out


def test_quota_pe_class_conditional(env):
    tmp, maple, mp = env
    pool = make_pool(tmp, maple, mp)
    # test reference without blind attributes (the blind screen runs after it in the prep job)
    run_pe(mp, maple, pool, tmp, tmp / "prep", "--lam", 0, "--prep_only")
    ref = pd.read_csv(tmp / "test_ref.csv")
    assert all(f"ref_{a}" in ref for a in ATTRS)
    with pytest.raises(SystemExit):                                      # OpenReview has no default B
        run_pe(mp, maple, pool, tmp, tmp / "x", "--lam", 0, "--loop_annotator", "inherited")
    blind = ["primary_strength", "keywords_group_1", "word_count"]
    (tmp / "blind.json").write_text(json.dumps({"blind": blind}))
    k = k_per_class(maple, 40)
    out = run_pe(mp, maple, pool, tmp, tmp / "maple" / "lam0", "--lam", 0, "--loop_annotator", "inherited",
                 "--blind_json", tmp / "blind.json")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["blind"] == blind and cfg["dataset"] == "openreview" and cfg["T"] == 1
    assert cfg["sigma"] == pytest.approx(pe_data.EPS_SETTINGS["4.0"][1])
    for t in (0, 1):
        r = pd.read_csv(out / f"round_{t}.csv")
        assert r[LABEL_ID].value_counts().to_dict() == k                 # K_g survivors per label id
        assert (r["loop_novelty_assessment"] == "__missing__").all()     # not AIM metadata: no inherited label
        assert r["loop_area"].isin(AREAS).all() and r["eval_novelty_assessment"].isin(SCHEMA["novelty_assessment"]).all()
    for m in [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]:
        other = [a for a in ATTRS if a not in blind and a not in ("area", "recommendation")]
        assert m["jsd_other_mean"] == pytest.approx(np.mean([m[f"jsd_{a}"] for a in other]))
        assert m["jsd_label_mean"] == pytest.approx(np.mean([m["jsd_area"], m["jsd_recommendation"]]))
        assert m["moved"] == 0 and all(f"jsd_gemini_{a}" in m for a in CONTENT)
    cells = json.loads((out / "cells.json").read_text())
    assert set(map(int, cells["k_per_class"])) == set(k) and len(cells["p_marg_class"]) == 8
    # quota inside every label id: still K_g per id, and it moves survivors
    out = run_pe(mp, maple, pool, tmp, tmp / "qwen" / "lam1", "--lam", 1, "--loop_annotator", "qwen", "--blind", *blind)
    for t in (0, 1):
        assert pd.read_csv(out / f"round_{t}.csv")[LABEL_ID].value_counts().to_dict() == k
    ms = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(m["moved"] for m in ms) > 0


# ---------------------------------------------------------------- Aug-PE (augpe_pe.py)
def test_augpe_class_conditional(env):
    tmp, maple, mp = env
    pool = make_pool(tmp, maple, mp)
    run_pe(mp, maple, pool, tmp, tmp / "prep", "--lam", 0, "--prep_only")
    (tmp / "blind.json").write_text(json.dumps({"blind": ["primary_weakness", "keywords_group_2", "word_count"]}))
    from augpe import augpe_pe
    out = tmp / "augpe" / "eps4.0" / "seed0"
    main_with(mp, augpe_pe.main, ["--maple_dir", maple, "--test_ref", tmp / "test_ref.csv", "--out", out, "--K", 40,
                                  "--L", 2, "--T", 1, "--prior_n", 8, "--blind_json", tmp / "blind.json",
                                  "--model_id", "fake-qwen"])
    k = k_per_class(maple, 40)
    p = pd.read_csv(out / "pool.csv")
    assert p[LABEL_ID].value_counts().to_dict() == {g: 3 * n for g, n in k.items()}
    for t in (0, 1):
        assert pd.read_csv(out / f"round_{t}.csv")[LABEL_ID].value_counts().to_dict() == k
    m = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(m) == 2 and "jsd_label_mean" in m[0] and "jsd_prior_area" in m[0]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["sigma"] == pytest.approx(3.000778920378073) and cfg["dataset"] == "openreview"
    from augpe import summarize_augpe
    main_with(mp, summarize_augpe.main, ["--augpe_root", tmp / "augpe" / "eps4.0", "--out", tmp / "sum"])
    md = (tmp / "sum" / "DRIFT.md").read_text()
    assert "jsd_label_mean" in md and "| keywords_group_3 |" in md and (tmp / "sum" / "drift_attrs.png").exists()


# ---------------------------------------------------------------- blind screen, embedder
def test_pick_blind_within_class(env):
    tmp, maple, mp = env
    rng = np.random.default_rng(3)
    n = 400
    pool = meta(rng, n, PAIRS[:8])
    pool[LABEL_ID] = rng.integers(4, size=n)
    pool["text"] = [f"t{i} " + "w " * int(rng.integers(1, 30)) for i in range(n)]
    pool.to_csv(tmp / "pool.csv", index=False)
    codes = {v: i for i, v in enumerate(SCHEMA["primary_strength"])}
    X = np.zeros((n, 10), np.float32)
    X[np.arange(n), [codes[v] for v in pool.primary_strength]] = 5.0       # strength fully visible
    X += rng.normal(size=X.shape).astype(np.float32)
    lookup = dict(zip(pool.text, X))
    import blindness.embed as E
    mp.setattr(E, "embed", lambda texts, *a, **k: np.stack([lookup[t] for t in texts]))
    from blindness import pick_blind
    main_with(mp, pick_blind.main, ["--pool_csv", tmp / "pool.csv", "--out", tmp / "blind.json", "--repeats", 2])
    b = json.loads((tmp / "blind.json").read_text())
    assert b["within_class"] is True and len(b["blind"]) == 3 and "primary_strength" not in b["blind"]
    assert b["kappa"]["primary_strength"] > 0.5 and set(b["kappa"]) == set(pe_data.BLIND_CANDIDATES)
    assert (tmp / "blind.csv").exists()


def test_embedder_trains_on_the_openreview_schema(tmp_path, monkeypatch):
    from tests.tiny_bert import make_tiny_bert
    from vote_embedding.train_embedder import ATTRS as EMB_ATTRS, main as train_main
    assert EMB_ATTRS == CONTENT
    rng = np.random.default_rng(0)
    pool = meta(rng, 120, PAIRS[:8]).assign(text=[text(rng, 3, 20) for _ in range(120)])
    pool["word_count_requested"] = pool["word_count"]
    for a in CONTENT:
        pool[f"map_{a}"] = rng.choice(SCHEMA[a][:2], len(pool))
    pool.to_csv(tmp_path / "pool_qwen.csv", index=False)
    bert = make_tiny_bert(tmp_path / "bert", hidden=32, heads=2)
    out = tmp_path / "emb"
    monkeypatch.setattr(sys, "argv", ["x", "--pool_qwen", str(tmp_path / "pool_qwen.csv"), "--model", bert,
                                      "--epochs", "1", "--max_len", "32", "--out", str(out)])
    train_main()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["attributes"] == CONTENT and (out / "heads.pt").exists()


def test_eps_inf_with_nan_metadata(env):
    """eps = inf uses the private labels as metadata; they have NaN (pandas 3: astype(str) keeps NaN, which broke the
    embedder's prompt groups, Nano5 job 482577). Pool -> PE round -> embedder training must all go through."""
    tmp, maple, mp = env
    import gen_round0
    out = tmp / "round0" / "epsinf"
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "inf", "--num_syn", 40, "--L_random", 2, "--out", out])
    main_with(mp, gen_round0.main, ["--maple_dir", maple, "--eps", "inf", "--merge", "--out", out])
    pool = pd.read_csv(out / "pool.csv")
    assert pool["primary_strength"].isna().any() or pool["primary_weakness"].isna().any()
    run_pe(mp, maple, out / "pool.csv", tmp, tmp / "prep", "--lam", 0, "--prep_only")
    run_pe(mp, maple, out / "pool.csv", tmp, tmp / "inf", "--lam", 0.5, "--loop_annotator", "inherited",
           "--blind", "word_count", "primary_weakness", "primary_strength")
    r = pd.read_csv(tmp / "inf" / "round_1.csv")
    assert len(r) == 40 and r["loop_primary_weakness"].isin(SCHEMA["primary_weakness"]).all()
    from tests.tiny_bert import make_tiny_bert
    from vote_embedding.train_embedder import main as train_main
    rng = np.random.default_rng(1)
    for a in CONTENT:
        pool[f"map_{a}"] = rng.choice(SCHEMA[a][:2], len(pool))
    pool.to_csv(tmp / "pool_qwen.csv", index=False)
    bert = make_tiny_bert(tmp / "bert", hidden=32, heads=2)
    mp.setattr(sys, "argv", ["x", "--pool_qwen", str(tmp / "pool_qwen.csv"), "--model", bert, "--epochs", "1",
                             "--max_len", "32", "--out", str(tmp / "emb")])
    train_main()
    assert (tmp / "emb" / "heads.pt").exists()
