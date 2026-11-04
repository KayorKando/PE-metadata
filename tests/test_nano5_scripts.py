"""Nano5 submit + array script with stub sbatch / module / conda / python: the grid, the run dirs
(lambda = 0 -> maple[-ft]/ with inherited labels), the flags passed to quota_pe.py, dependency and throttle."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUB = "#!/usr/bin/env bash\n"


def stubs(tmp):
    b = tmp / "bin"
    b.mkdir()
    log = tmp / "calls.log"
    files = {
        "sbatch": STUB + f'echo "SBATCH $* | LAMS=$LAMS SEEDS=$SEEDS VOTE_EMBS=$VOTE_EMBS" >> {log}; echo "Submitted batch job 42"\n',
        "python": STUB + f'echo "PY $*" >> {log}\n',
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


def test_submit_and_tasks(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    e = env_for(tmp_path, b, LAMS="0 0.5 1", SEEDS="0 1", VOTE_EMBS="t5 ft", AFTER="777", MAX_PAR="3")
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    sb = log.read_text()
    assert "--array=0-11%3" in sb and "--dependency=afterok:777" in sb and "--time=02:00:00" in sb

    # run every array task with the same environment the job would get
    runs = {}
    for i in range(12):
        log.write_text("")
        t = subprocess.run(["bash", "quota_sweep/nano5_pe.sbatch"], cwd=REPO, capture_output=True, text=True,
                           env={**e, "SLURM_ARRAY_TASK_ID": str(i), "SLURM_ARRAY_JOB_ID": "42", "SLURM_SUBMIT_DIR": str(REPO)})
        assert t.returncode == 0, t.stdout + t.stderr
        py = [l for l in log.read_text().splitlines() if "quota_pe.py" in l][0].split()
        out = py[py.index("--out") + 1]
        runs[Path(out).relative_to(tmp_path / "work" / "quota_sweep" / "eps4.0").as_posix()] = py
    assert sorted(runs) == sorted(["maple/lam0", "classifier/lam0.5", "classifier/lam1",
                                   "maple/lam0_seed1", "classifier/lam0.5_seed1", "classifier/lam1_seed1",
                                   "maple-ft/lam0", "classifier-ft/lam0.5", "classifier-ft/lam1",
                                   "maple-ft/lam0_seed1", "classifier-ft/lam0.5_seed1", "classifier-ft/lam1_seed1"])
    for d, py in runs.items():
        ann = py[py.index("--loop_annotator") + 1]
        assert ann == ("inherited" if d.startswith("maple") else "classifier")
        assert ("--classifier_spec" in py) == (ann == "classifier")
        assert ("--vote_emb" in py) == ("-ft/" in d)
        assert py[py.index("--seed") + 1] == ("1" if "seed1" in d else "0")


def test_submit_refuses_ft_without_embedder(tmp_path):
    b, log = stubs(tmp_path)
    data(tmp_path / "work")
    r = subprocess.run(["bash", "quota_sweep/nano5_pe_submit.sh"], cwd=REPO, capture_output=True, text=True,
                       env=env_for(tmp_path, b, VOTE_EMBS="ft"))
    assert r.returncode != 0 and "AFTER" in r.stdout and not log.exists()
