"""ft-t5 arm and third-party MAUVE:
- a T5 backbone in sentence-transformers layout (mean pooling + Dense) loads as encoder + Dense; at init
  the embedding is LN(Dense(mean-pool))/||.||, so the vote starts from sentence-t5's geometry;
  training, save / load and FTEmbedder work end to end with the Dense kept;
- nano5 scripts: fts<N> / ftt5 run dirs and embedder paths; lambda = 0 only needs no classifier spec;
- summarize.py: every family is paired against t5; mauve_<tag>.jsonl is merged in;
- mauve_rescore.py: gpt2 last-token features, one line per finished round, resumable."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reannotate.train_classifier import FineTuned
from tests.test_nano5_scripts import data, env_for, run_job, sbatch_calls, stubs, submit
from tests.tiny_bert import WORDS, random_texts
from vote_embedding.train_embedder import encode_labels, load, make_model, save, st_dense, train

REPO = Path(__file__).resolve().parents[1]
K = {"primary_research_area": 3, "sample_size": 3}


def make_tiny_st_t5(path, d=32, dense_out=24, seed=0, mean=True):
    """Tiny T5 encoder + word-level tokenizer + sentence-transformers modules (Pooling, Dense, Normalize)."""
    from safetensors.torch import save_file
    from transformers import BertTokenizerFast, T5Config, T5EncoderModel
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt"), model_max_length=128).save_pretrained(path)
    torch.manual_seed(seed)
    cfg = T5Config(vocab_size=len(vocab), d_model=d, d_kv=8, d_ff=32, num_layers=1, num_heads=4,
                   pad_token_id=0, decoder_start_token_id=0)
    T5EncoderModel(cfg).save_pretrained(path)
    (path / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
        {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"}]))
    (path / "1_Pooling").mkdir(exist_ok=True)
    (path / "1_Pooling" / "config.json").write_text(json.dumps(
        {"word_embedding_dimension": d, "pooling_mode_mean_tokens": mean, "pooling_mode_cls_token": not mean}))
    (path / "2_Dense").mkdir(exist_ok=True)
    (path / "2_Dense" / "config.json").write_text(json.dumps(
        {"in_features": d, "out_features": dense_out, "bias": False,
         "activation_function": "torch.nn.modules.linear.Identity"}))
    save_file({"linear.weight": torch.randn(dense_out, d) / d ** 0.5}, str(path / "2_Dense" / "model.safetensors"))
    return str(path)


@pytest.fixture(scope="module")
def t5dir(tmp_path_factory):
    return make_tiny_st_t5(tmp_path_factory.mktemp("st_t5"))


def tk_for(name):
    return FineTuned(name, 0, 1, 2e-5, 8, 64, "cpu")


def test_st_dense_and_init_geometry(t5dir, tmp_path):
    from transformers import T5EncoderModel
    lin = st_dense(t5dir)
    assert lin.in_features == 32 and lin.out_features == 24 and lin.bias is None
    assert st_dense(make_tiny_st_t5(tmp_path / "plain")) is not None
    enc, heads = make_model(t5dir, "normsoftmax", K)
    assert isinstance(enc.backbone, T5EncoderModel) and enc.dim == 24 and heads.P["sample_size"].shape == (3, 24)
    tk = tk_for(t5dir)
    ids = tk.encode(random_texts(6))
    b = next(tk._batches(ids, np.arange(6)))
    enc.eval()
    with torch.no_grad():
        x = enc(**b)
        h = T5EncoderModel.from_pretrained(t5dir)(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        z = ((h * m).sum(1) / m.sum(1)) @ lin.weight.T                # sentence-t5 before its Normalize
        ref = torch.nn.functional.layer_norm(z, (24,))
        ref = ref / ref.norm(dim=-1, keepdim=True)
    assert torch.allclose(x, ref, atol=1e-5)


def test_cls_pooling_is_refused(tmp_path):
    with pytest.raises(ValueError):
        st_dense(make_tiny_st_t5(tmp_path / "cls", mean=False))


def test_bert_has_no_dense():
    from tests.tiny_bert import make_tiny_bert
    import tempfile
    assert st_dense(make_tiny_bert(tempfile.mkdtemp(), hidden=16, heads=2)) is None


def test_t5_train_save_load_ftembedder(t5dir, tmp_path):
    from vote_embedding.embedder import FTEmbedder
    texts = random_texts(40, 3)
    lab = {a: np.array([str(len(t.split()) % 3) for t in texts], dtype=object) for a in K}
    tr, va = np.arange(28), np.arange(28, 40)
    classes = {a: sorted(set(lab[a][tr])) for a in K}
    tk = tk_for(t5dir)
    ids = tk.encode(texts)
    torch.manual_seed(0)
    enc, heads = make_model(t5dir, "normsoftmax", {a: len(c) for a, c in classes.items()})
    w0 = enc.proj.weight.detach().clone()
    train(enc, heads, tk, [ids[i] for i in tr], encode_labels(lab, tr, classes), [ids[i] for i in va],
          encode_labels(lab, va, classes), classes, epochs=1, lr=1e-3, log=lambda s: None)
    assert not torch.equal(w0, enc.proj.weight.detach())              # Dense is trained in stage 2
    path = tmp_path / "normsoftmax-t5"
    save(path, enc, heads, tk.tok, classes, {"loss": "normsoftmax", "sigma": 0.05, "max_len": 64})
    assert not (path / "backbone" / "modules.json").exists()           # Dense lives in heads.pt only
    enc2, _, _, _ = load(path)
    assert enc2.proj is not None and torch.equal(enc2.proj.weight, enc.proj.weight.cpu())
    emb = FTEmbedder(path, batch=8, device="cpu", log_overflow=False)
    X = emb(texts)
    enc.eval()
    with torch.no_grad():
        ref = torch.cat([enc(**b) for b in tk._batches(ids, np.arange(len(ids)))]).numpy()
    assert X.shape == (40, 24) and np.allclose(X, ref, atol=1e-4)


# ---------------- nano5 scripts ----------------
def test_submit_embedder_families(tmp_path):
    b, log = stubs(tmp_path)
    work = tmp_path / "work"
    data(work)
    (work / "reannotate/eps4.0/best_annotator_nano5.json").unlink()   # lambda = 0 only: spec not needed
    for v in ["normsoftmax-t5", "normsoftmax-seed1"]:
        (work / "vote_embedding/eps4.0" / v).mkdir(parents=True)
        (work / "vote_embedding/eps4.0" / v / "heads.pt").write_text("x")
    e = env_for(tmp_path, b, GROUP_BY="lam", VOTE_EMBS="ftt5 fts1", LAMS="0", SEEDS="0 1")
    submit(e)
    calls = sbatch_calls(log)
    assert [c.split("--job-name=")[1].split()[0] for c in calls] == ["pe-ftt5-lam0", "pe-fts1-lam0"]
    runs = run_job(e, "ftt5 0 0;ftt5 1 0;fts1 0 0", "0,1,2", log, tmp_path)
    assert sorted(runs) == ["maple-fts1/lam0", "maple-ftt5/lam0", "maple-ftt5/lam0_seed1"]
    for d, (_, py) in runs.items():
        want = "normsoftmax-t5" if "ftt5" in d else "normsoftmax-seed1"
        assert py[py.index("--vote_emb") + 1] == "ft" and py[py.index("--vote_emb_path") + 1].endswith(want)
    # missing embedder / unknown name / classifier spec needed for lambda > 0
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="fts2"))
    assert r.returncode != 0 and "normsoftmax-seed2" in r.stdout
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="bogus"))
    assert r.returncode != 0 and "unknown vote embedding" in r.stdout + r.stderr
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="t5", LAMS="0 0.5"))
    assert r.returncode != 0 and "best_annotator_nano5" in r.stdout


# ---------------- summarize ----------------
def test_summarize_families_and_extra_mauve(tmp_path):
    def run(ann, lam, seed, mauve, gpt):
        d = tmp_path / ann / (f"lam{lam}" + ("" if seed == 0 else f"_seed{seed}"))
        d.mkdir(parents=True)
        rows = [dict(round=t, mauve=mauve, jsd_mean=0.1, jsd_blind_mean=0.2 if ann == "maple" else 0.1,
                     jsd_other_mean=0.05, jsd_word_count=0.3, moved=0, shortfall=0) for t in range(2)]
        (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (d / "mauve_gpt2-large.jsonl").write_text("\n".join(json.dumps({"round": t, "mauve_gpt2-large": gpt})
                                                            for t in range(2)))
    for s in range(3):
        run("maple", 0, s, 0.70 + 0.01 * s, 0.50 + 0.01 * s)
        run("maple-ft", 0, s, 0.68 + 0.01 * s, 0.55 + 0.01 * s)
        run("maple-ftt5", 0, s, 0.72 + 0.01 * s, 0.52 + 0.01 * s)
    subprocess.run([sys.executable, str(REPO / "quota_sweep" / "summarize.py"), "--root", str(tmp_path)],
                   check=True, capture_output=True)
    P = pd.read_csv(tmp_path / "sweep_paired.csv").set_index(["comparison", "metric"])
    assert abs(P.loc[("ft - t5", "mauve_gpt2-large"), "mean"] - 0.05) < 1e-9
    assert abs(P.loc[("ftt5 - t5", "mauve_gpt2-large"), "mean"] - 0.02) < 1e-9
    assert abs(P.loc[("ftt5 - t5", "mauve"), "mean"] - 0.02) < 1e-9
    assert abs(P.loc[("ft - t5", "jsd_blind_mean"), "mean"] + 0.1) < 1e-9
    F = pd.read_csv(tmp_path / "sweep_final.csv")
    assert set(F.annotator) == {"maple", "maple-ft", "maple-ftt5"} and "mauve_gpt2-large" in F
    md = (tmp_path / "SWEEP.md").read_text()
    assert "ftt5 - t5" in md and "mauve_gpt2-large" in md


# ---------------- mauve_rescore ----------------
def test_mauve_rescore(tmp_path, monkeypatch):
    from transformers import BertTokenizerFast, GPT2Config, GPT2Model
    g = tmp_path / "gpt2-tiny"
    g.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (g / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(g / "vocab.txt"), model_max_length=1024).save_pretrained(g)
    torch.manual_seed(0)
    GPT2Model(GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=1024)).save_pretrained(g)
    # fake MAPLE compute_mauve: records the feature shapes, returns mean feature distance
    ue = tmp_path / "MAPLE" / "biorxiv" / "utility_eval"
    ue.mkdir(parents=True)
    (ue / "compute_mauve.py").write_text(
        "import numpy as np\nclass R: pass\n"
        "def compute_mauve(p_features, q_features):\n"
        "    r = R(); r.mauve = float(abs(p_features.mean() - q_features.mean())) + p_features.shape[1]; return r\n")
    pd.DataFrame({"text": random_texts(10, 1)}).to_csv(tmp_path / "test.csv", index=False)
    run = tmp_path / "sweep" / "maple-ft" / "lam0"
    run.mkdir(parents=True)
    for t in range(3):
        pd.DataFrame({"text": random_texts(8, 10 + t)}).to_csv(run / f"round_{t}.csv", index=False)
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"round": t}) for t in range(2)))  # round 2 unfinished
    monkeypatch.setenv("PE_DATA", str(tmp_path))
    cmd = [sys.executable, str(REPO / "quota_sweep" / "mauve_rescore.py"), "--root", str(tmp_path / "sweep"),
           "--maple_dir", str(tmp_path / "MAPLE"), "--test_ref", str(tmp_path / "test.csv"), "--model", str(g)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [json.loads(l) for l in (run / "mauve_gpt2-tiny.jsonl").read_text().splitlines()]
    assert [l["round"] for l in lines] == [0, 1] and all(15.9 < l["mauve_gpt2-tiny"] < 17 for l in lines)
    r = subprocess.run(cmd, capture_output=True, text=True)                 # resumable: nothing left
    assert "0 rounds to score" in r.stdout and len((run / "mauve_gpt2-tiny.jsonl").read_text().splitlines()) == 2


def test_gpt2_last_token_features_ignore_padding(tmp_path):
    from transformers import BertTokenizerFast, GPT2Config, GPT2Model
    from quota_sweep.mauve_rescore import Featurizer
    g = tmp_path / "gpt2-tiny"
    g.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (g / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(g / "vocab.txt"), model_max_length=1024).save_pretrained(g)
    torch.manual_seed(0)
    GPT2Model(GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=2, n_head=2)).save_pretrained(g)
    f = Featurizer(str(g), batch=4, device="cpu")
    texts = random_texts(7, 5)
    X = f(texts)
    single = np.concatenate([f([t]) for t in texts])
    assert np.allclose(X, single, atol=1e-5)


def test_submit_with_after_allows_missing_pool(tmp_path):
    """New eps: the round-0 pool is made by the prep job given as AFTER."""
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    (tmp_path / "work/round0/eps4.0/pool.csv").unlink()
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="t5", LAMS="0"))
    assert r.returncode != 0 and "pool.csv" in r.stdout
    submit(env_for(tmp_path, b, VOTE_EMBS="t5 ft", LAMS="0", SEEDS="0 1", GROUP_BY="lam", AFTER="555"))
    calls = sbatch_calls(log)
    assert len(calls) == 2 and all("--dependency=afterok:555" in c for c in calls)


def test_fp16_checkpoint_loads_as_fp32(tmp_path):
    """sentence-t5-base is stored in fp16; the backbone must train with fp32 weights (bf16 only via autocast)."""
    from transformers import T5EncoderModel
    d = make_tiny_st_t5(tmp_path / "half")
    T5EncoderModel.from_pretrained(d).half().save_pretrained(d)
    enc, _ = make_model(d, "normsoftmax", K)
    assert all(p.dtype == torch.float32 for p in enc.parameters())


def test_ftembedder_falls_back_to_base_tokenizer(t5dir, tmp_path):
    """A tokenizer config written by a newer transformers ('extra_special_tokens' as a list) must not stop the
    vote embedder: it loads the base model's tokenizer instead and gives the same embeddings."""
    from vote_embedding.embedder import FTEmbedder
    enc, heads = make_model(t5dir, "normsoftmax", K)
    tk = tk_for(t5dir)
    path = tmp_path / "emb"
    save(path, enc, heads, tk.tok, {a: [str(i) for i in range(3)] for a in K},
         {"loss": "normsoftmax", "sigma": 0.05, "max_len": 64, "base_model": t5dir})
    texts = random_texts(5, 7)
    X0 = FTEmbedder(path, batch=4, device="cpu", log_overflow=False)(texts)
    cfgf = path / "backbone" / "tokenizer_config.json"
    c = json.loads(cfgf.read_text()); c["extra_special_tokens"] = ["<extra_id_0>"]; cfgf.write_text(json.dumps(c))
    emb = FTEmbedder(path, batch=4, device="cpu", log_overflow=False)
    assert np.allclose(emb(texts), X0, atol=1e-6)


def test_summarize_pairs_family_without_own_t5_dir_and_dedups(tmp_path):
    """maple-ftt5 next to classifier/ + maple/: paired against maple (shown as classifier lambda = 0);
    a round logged twice keeps its last line."""
    def run(ann, lam, seed, blind, dup=False):
        d = tmp_path / ann / (f"lam{lam}" + ("" if seed == 0 else f"_seed{seed}"))
        d.mkdir(parents=True)
        rows = [dict(round=t, mauve=0.7, jsd_mean=0.1, jsd_blind_mean=blind, jsd_other_mean=0.05, moved=0,
                     shortfall=0) for t in range(2)]
        if dup:
            rows.insert(1, dict(rows[1], jsd_blind_mean=9.0))       # stale duplicate of round 1, written first
        (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for s in range(3):
        run("maple", 0, s, 0.30 + 0.01 * s)
        run("classifier", 0.5, s, 0.20)
        run("maple-ftt5", 0, s, 0.15 + 0.01 * s, dup=True)
    r = subprocess.run([sys.executable, str(REPO / "quota_sweep" / "summarize.py"), "--root", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "duplicated rounds" in r.stdout
    P = pd.read_csv(tmp_path / "sweep_paired.csv").set_index(["comparison", "metric"])
    assert abs(P.loc[("ftt5 - t5", "jsd_blind_mean"), "mean"] + 0.15) < 1e-9
    assert P.loc[("ftt5 - t5", "jsd_blind_mean"), "n"] == 3
