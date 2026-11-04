"""Nano5 submit + packed job script with stub sbatch / module / conda / python: how runs are grouped into
jobs (GROUP_BY), the GPU / CPU / memory request, one GPU per run inside a job, the run dirs (lambda = 0 ->
maple[-ft]/ with inherited labels), the flags passed to quota_pe.py, dependency, and skipping finished runs."""
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUB = "#!/usr/bin/env bash\n"


def stubs(tmp):
    b = tmp / "bin"
    b.mkdir()
    log = tmp / "calls.log"
    files = {
        "sbatch": STUB + f'echo "SBATCH $* | RUNS=$RUNS" >> {log}; echo "Submitted batch job 42"\n',
        "python": STUB + f'echo "PY CUDA=$CUDA_VISIBLE_DEVICES $*" >> {log}\n',
        "module": STUB, "nvidia-smi": STUB, "git": STUB + "echo abc123\n", "conda": STUB,
    }
    for name, body in files.items():
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    return b, log


def data(root):
    for f in ["round0/eps4.0/pool.csv", "quota_sweep/test_ref_qwen.csv", "MAPLE/biorxiv/biorxiv_train_metadata.csv",
              "reannotate/eps4.0/best_annotator_nano5.json"]:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x")


def env_for(tmp, b, **extra):
    e = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "NANO5_DATA": str(tmp / "work"), "USER": "tester"}
    e.update(extra)
    return e


def submit(env):
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def sbatch_calls(log):
    return [l for l in log.read_text().splitlines() if l.startswith("SBATCH")]


def run_job(env, runs, gpus, log, tmp):
    """Run nano5_pe.sbatch as Slurm would; -> {run dir: (gpu, argv)}."""
    log.write_text("")
    e = {**env, "RUNS": runs, "CUDA_VISIBLE_DEVICES": gpus, "SLURM_JOB_ID": "42", "SLURM_CPUS_PER_TASK": "16",
         "SLURM_SUBMIT_DIR": str(REPO)}
    t = subprocess.run(["bash", "quota_sweep/nano5_pe.sbatch"], cwd=REPO, env=e, capture_output=True, text=True)
    assert t.returncode == 0, t.stdout + t.stderr
    out = {}
    for l in log.read_text().splitlines():
        if "quota_pe.py" in l:
            py = l.split()
            d = Path(py[py.index("--out") + 1]).relative_to(tmp / "work" / "quota_sweep" / "eps4.0").as_posix()
            out[d] = (py[1].split("=")[1], py)
    return out


def test_group_by_lambda_t5(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    e = env_for(tmp_path, b, GROUP_BY="lam", VOTE_EMBS="t5", LAMS="0 0.5 1", SEEDS="0 1 2")
    submit(e)
    calls = sbatch_calls(log)
    assert len(calls) == 3                                           # one job per lambda
    for c, lam in zip(calls, ["0", "0.5", "1"]):
        assert f"--job-name=pe-t5-lam{lam}" in c and "--gres=gpu:3" in c and "--cpus-per-task=24" in c
        assert "--mem=192G" in c and "--dependency" not in c
        assert c.split("RUNS=")[1] == f"t5 0 {lam};t5 1 {lam};t5 2 {lam}"
    runs = run_job(e, "t5 0 0;t5 1 0;t5 2 0", "3,5,6", log, tmp_path)   # GPUs as Slurm hands them out
    assert sorted(runs) == ["maple/lam0", "maple/lam0_seed1", "maple/lam0_seed2"]
    assert sorted(g for g, _ in runs.values()) == ["3", "5", "6"]       # one GPU per run
    for d, (_, py) in runs.items():
        assert py[py.index("--loop_annotator") + 1] == "inherited" and "--classifier_spec" not in py
    runs = run_job(e, "t5 0 0.5;t5 1 0.5", "0,1", log, tmp_path)
    for d, (_, py) in runs.items():
        assert d.startswith("classifier/lam0.5") and "--classifier_spec" in py and "--vote_emb" not in py


def test_group_all_ft_with_dependency(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    e = env_for(tmp_path, b, GROUP_BY="all", VOTE_EMBS="ft", LAMS="0 0.5 1", SEEDS="0 1", AFTER="777")
    submit(e)
    (c,) = sbatch_calls(log)                                          # one job for all ft runs
    assert "--gres=gpu:6" in c and "--dependency=afterok:777" in c and "--job-name=pe-ft" in c
    runs = run_job(e, c.split("RUNS=")[1], "0,1,2,3,4,5", log, tmp_path)
    assert sorted(runs) == sorted(["maple-ft/lam0", "maple-ft/lam0_seed1", "classifier-ft/lam0.5",
                                   "classifier-ft/lam0.5_seed1", "classifier-ft/lam1", "classifier-ft/lam1_seed1"])
    assert len({g for g, _ in runs.values()}) == 6
    for d, (_, py) in runs.items():
        assert "--vote_emb" in py and py[py.index("--seed") + 1] == ("1" if "seed1" in d else "0")


def test_finished_runs_are_left_out(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    done = tmp_path / "work" / "quota_sweep" / "eps4.0" / "classifier" / "lam0.5_seed1"
    done.mkdir(parents=True)
    (done / "DONE").write_text("x")
    out = submit(env_for(tmp_path, b, GROUP_BY="lam", VOTE_EMBS="t5", LAMS="0.5", SEEDS="0 1 2"))
    (c,) = sbatch_calls(log)
    assert "--gres=gpu:2" in c and c.split("RUNS=")[1] == "t5 0 0.5;t5 2 0.5" and "done already" in out


def test_refuses_ft_without_embedder_and_too_many_gpus(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="ft"))
    assert r.returncode != 0 and "AFTER" in r.stdout
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, GROUP_BY="all", VOTE_EMBS="t5", LAMS="0 0.5 1", SEEDS="0 1 2"))
    assert r.returncode != 0 and "> 8 per node" in r.stdout and not log.exists()
