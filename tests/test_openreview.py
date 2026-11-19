"""OpenReview support. The dataset-specific cases (tests/openreview_cases.py) run in a subprocess with
PE_DATASET=openreview; here: the dataset-independent pieces (MAPLE's per-id survivor counts, the class-conditional
vote, per-class selection, the gte MAUVE featurizer) and the Nano5 scripts with PE_DATASET=openreview."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pe_data  # noqa: E402

STUB = "#!/usr/bin/env bash\n"


def test_openreview_cases_in_subprocess():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/openreview_cases.py"],
                       cwd=REPO, env={**os.environ, "PE_DATASET": "openreview"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:]


def maple_get_num_sample_per_target_id(priv_ids, num_sample):
    """MAPLE openreview/MAPLE/MAPLE.py, Voting_histogram (id dict + get_num_sample_per_target_id), verbatim logic."""
    id_dict = {}
    for index, i in enumerate(list(priv_ids)):
        if i in id_dict:
            id_dict[i].append(index)
        else:
            id_dict[i] = [index]
    id_list = list(id_dict.keys())
    count_list = np.array([len(id_dict[ID]) for ID in id_list])
    id_ratio = count_list / np.sum(count_list)
    target_num_per_id = id_ratio * num_sample
    num_per_id = np.floor(target_num_per_id).astype(int)
    num_per_id_left = num_sample - np.sum(num_per_id)
    ids = np.argsort(target_num_per_id - num_per_id)[::-1]
    num_per_id[ids[:num_per_id_left]] += 1
    return dict(zip(id_list, num_per_id))


def test_num_sample_per_id_matches_maple():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ids = rng.choice(49, size=int(rng.integers(200, 3000)), p=rng.dirichlet(np.ones(49) * 0.5))
        ids = np.concatenate([ids, np.arange(49)])           # every id present
        a = pe_data.num_sample_per_id(ids, 2000)
        b = maple_get_num_sample_per_target_id(ids, 2000)
        assert list(a) == list(b) and all(a[k] == b[k] for k in a) and sum(a.values()) == 2000


def test_class_vote_only_within_class():
    X_priv = np.array([[0.0, 0], [10, 10], [10.2, 10]], np.float32)
    g_priv = np.array([0, 1, 1])
    X_cand = np.array([[0.0, 0], [10, 10], [0.1, 0], [9, 9]], np.float32)   # cand 0 sits on private 0 but is class 1
    g_cand = np.array([1, 0, 0, 1])
    h = pe_data.class_vote(X_priv, g_priv, X_cand, g_cand)
    assert list(h) == [0, 0, 1, 2]          # private 0 -> cand 2 (not cand 0: other class); privates 1, 2 -> cand 3
    assert list(pe_data.class_vote(X_priv, None, X_cand, None)) == [1, 2, 0, 0]


def test_select_per_class():
    scores = np.array([5, 1, 4, 3, 2, 9.0])
    groups = np.array([0, 0, 1, 1, 1, 0])
    sel, info = pe_data.select_per_class(scores, groups, {1: 2, 0: 1},
                                         lambda s, ci, k: (np.argsort(-s, kind="stable")[:k], {"moved": 1}))
    assert list(sel) == [2, 3, 5] and info == {"shortfall": 0, "moved": 2}


def test_gte_featurizer_uses_cls_and_long_context(tmp_path):
    """gte-base-en-v1.5 is configured by name: CLS pooling, 2048 tokens, trust_remote_code (tiny local stand-in)."""
    import torch
    from tests.tiny_bert import make_tiny_bert
    from quota_sweep.mauve_rescore import Featurizer
    path = make_tiny_bert(tmp_path / "gte-base-en-v1.5", hidden=16, heads=2, max_pos=64)
    f = Featurizer(path, batch=4, device="cpu")
    assert (f.pooling, f.max_len, f.kind) == ("cls", 2048, "enc") and f.key_extra == "|cls|2048"
    texts = ["cell gene", "the model of data and a method"]
    enc = f.tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = f.net(**enc).last_hidden_state[:, 0].numpy()
    assert np.allclose(f(texts), ref, atol=1e-5)
    g = Featurizer(make_tiny_bert(tmp_path / "plain", hidden=16, heads=2), batch=4, device="cpu")
    assert (g.pooling, g.max_len, g.key_extra) == ("mean", 512, "")


# ---------------------------------------------------------------- Nano5 scripts, PE_DATASET=openreview
def stubs(tmp):
    b = tmp / "bin"
    b.mkdir()
    log = tmp / "calls.log"
    files = {"sbatch": STUB + f'echo "SBATCH $* | RUNS=$RUNS SEEDS=$SEEDS DS=$PE_DATASET" >> {log}; echo "Submitted batch job 42"\n',
             "python": STUB + f'echo "PY CUDA=$CUDA_VISIBLE_DEVICES DS=$PE_DATASET DATA=$PE_DATA $*" >> {log}\n',
             "module": STUB, "nvidia-smi": STUB, "git": STUB + "echo abc123\n", "conda": STUB}
    for name, body in files.items():
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    w = tmp / "work"
    for f in ["round0/eps4.0/pool.csv", "quota_sweep/test_ref_qwen.csv", "blind.json",
              "MAPLE/openreview/train_dataset_with_metadata_v4.csv", "MAPLE/openreview/MAPLE/PE.py"]:
        (w / f).parent.mkdir(parents=True, exist_ok=True)
        (w / f).write_text("x")
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "NANO5_DATA": str(w), "USER": "tester",
           "PE_DATASET": "openreview"}
    return env, log, w


def lines(log, key):
    return [l for l in log.read_text().splitlines() if key in l] if log.exists() else []


def test_pe_submit_and_job_openreview(tmp_path):
    env, log, w = stubs(tmp_path)
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "GROUP_BY": "lam", "VOTE_EMBS": "t5", "LAMS": "0 0.5", "SEEDS": "0 1",
                            "ANNOTATOR": "qwen"})
    assert r.returncode == 0, r.stdout + r.stderr
    calls = lines(log, "SBATCH")
    assert len(calls) == 2 and "--job-name=or-pe4.0-t5-lam0 " in calls[0] and calls[0].endswith("DS=openreview")
    log.write_text("")
    e = {**env, "RUNS": "t5 0 0;t5 1 0.5", "ANNOTATOR": "qwen", "CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_ID": "42",
         "SLURM_CPUS_PER_TASK": "16", "SLURM_SUBMIT_DIR": str(REPO)}
    j = subprocess.run(["bash", "quota_sweep/nano5_pe.sbatch"], cwd=REPO, env=e, capture_output=True, text=True)
    assert j.returncode == 0, j.stdout + j.stderr
    py = [l.split() for l in lines(log, "quota_pe.py")]
    assert len(py) == 2
    for p in py:
        assert p[p.index("--blind_json") + 1] == f"{w}/blind.json" and f"DATA={w}" in p
        out = p[p.index("--out") + 1]
        if out.endswith("maple/lam0"):
            assert p[p.index("--loop_annotator") + 1] == "inherited"
        else:
            assert out.endswith("qwen/lam0.5_seed1") and p[p.index("--loop_annotator") + 1] == "qwen"
            assert "--classifier_spec" not in p


def test_pe_submit_openreview_needs_blind_unless_after(tmp_path):
    env, log, w = stubs(tmp_path)
    (w / "blind.json").unlink()
    (w / "round0/eps4.0/pool.csv").unlink()
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "VOTE_EMBS": "t5"})
    assert r.returncode != 0 and "missing" in r.stdout and not lines(log, "SBATCH")
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "VOTE_EMBS": "t5", "AFTER": "11:12"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--dependency=afterok:11:12" in lines(log, "SBATCH")[0]


def test_augpe_submit_and_job_openreview(tmp_path):
    env, log, w = stubs(tmp_path)
    r = subprocess.run(["bash", "augpe/nano5_augpe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "SEEDS": "0 1", "TIME": "05:00:00"})
    assert r.returncode == 0, r.stdout + r.stderr
    (c,) = lines(log, "SBATCH")
    assert "--job-name=or-augpe-eps4.0" in c and "--time=05:00:00" in c
    log.write_text("")
    e = {**env, "SEEDS": "0 1", "CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_ID": "42", "SLURM_CPUS_PER_TASK": "16",
         "SLURM_SUBMIT_DIR": str(REPO)}
    j = subprocess.run(["bash", "augpe/nano5_augpe.sbatch"], cwd=REPO, env=e, capture_output=True, text=True)
    assert j.returncode == 0, j.stdout + j.stderr
    py = [l.split() for l in lines(log, "augpe_pe.py")]
    assert len(py) == 2 and all(p[p.index("--blind_json") + 1] == f"{w}/blind.json" for p in py)


@pytest.mark.parametrize("eps", ["4.0", "2.0"])
def test_prep_job_openreview(tmp_path, eps):
    env, log, w = stubs(tmp_path)
    for f in ["blind.json", "quota_sweep/test_ref_qwen.csv", "round0/eps4.0/pool.csv"]:
        (w / f).unlink()
    e = {**env, "EPS": eps, "SLURM_JOB_ID": "42", "SLURM_SUBMIT_DIR": str(REPO)}
    j = subprocess.run(["bash", "quota_sweep/nano5_prep_eps.sbatch"], cwd=REPO, env=e, capture_output=True, text=True)
    assert j.returncode == 0, j.stdout + j.stderr
    calls = lines(log, "PY ")
    names = [next(t for t in l.split() if t.endswith(".py")) for l in calls]
    base = ["gen_round0.py", "gen_round0.py", "reannotate/annotate_qwen.py"]
    if eps == "4.0":        # the eps-4 job also makes the test reference and blind.json
        assert names == base + ["quota_sweep/quota_pe.py", "blindness/pick_blind.py", "vote_embedding/train_embedder.py"]
        assert "--prep_only" in calls[3] and f"--out {w}/blind.json" in calls[4]
    else:
        assert names == base + ["vote_embedding/train_embedder.py"]
    assert all("DS=openreview" in l and f"DATA={w}" in l for l in calls)


def test_bioRxiv_root_refused_for_openreview(tmp_path):
    """A process with PE_DATASET=openreview must not write into the bioRxiv root (no DATASET file)."""
    r = subprocess.run([sys.executable, "-c", "import pe_data; pe_data.check_data_root()"], cwd=REPO, capture_output=True,
                       text=True, env={**os.environ, "PE_DATASET": "openreview", "PE_DATA": str(tmp_path)})
    assert r.returncode != 0 and "openreview" in (r.stdout + r.stderr)
