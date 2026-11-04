"""summarize.py: the t5 MAPLE baseline is the lambda = 0 point of t5 curves only, maple-ft of -ft curves only;
mean / sd / n over seeds; paired differences vs lambda = 0 and ft - t5 by seed."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KEYS = ["mauve", "jsd_mean", "jsd_blind_mean", "jsd_other_mean"]


def write_run(root, ann, lam, seed, mauve):
    d = root / ann / (f"lam{lam}" + ("" if seed == 0 else f"_seed{seed}"))
    d.mkdir(parents=True)
    rows = [dict(round=t, mauve=mauve - 0.1 * (1 - t), jsd_mean=0.1 + lam, jsd_blind_mean=0.2, jsd_other_mean=0.05,
                 jsd_word_count=0.3, moved=0, shortfall=0) for t in range(2)]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "DONE").write_text("x")


def test_summarize(tmp_path):
    # mauve at the last round = base + 0.01 * seed, so paired differences have sd 0 when the offset is shared
    for s in range(3):
        write_run(tmp_path, "maple", 0, s, 0.60 + 0.01 * s)
        write_run(tmp_path, "classifier", 0.5, s, 0.70 + 0.01 * s)
        write_run(tmp_path, "maple-ft", 0, s, 0.65 + 0.01 * s)
        write_run(tmp_path, "classifier-ft", 0.5, s, 0.80 + 0.02 * s)
    subprocess.run([sys.executable, str(REPO / "quota_sweep" / "summarize.py"), "--root", str(tmp_path)],
                   check=True, capture_output=True)
    F = pd.read_csv(tmp_path / "sweep_final.csv").set_index(["annotator", "lam"])
    assert abs(F.loc[("classifier", 0.0), "mauve"] - 0.61) < 1e-9          # maple (t5)
    assert abs(F.loc[("classifier-ft", 0.0), "mauve"] - 0.66) < 1e-9        # maple-ft, not maple
    assert set(F.index.get_level_values(0)) == {"classifier", "classifier-ft"}
    assert (F.n_seeds == 3).all() and abs(F.loc[("classifier", 0.5), "mauve_sd"] - 0.01) < 1e-9
    P = pd.read_csv(tmp_path / "sweep_paired.csv")
    P = P[P.metric == "mauve"].set_index(["comparison", "annotator", "lam"])
    v = P.loc[("vs lambda=0", "classifier", 0.5)]
    assert abs(v["mean"] - 0.10) < 1e-9 and v.sd < 1e-9 and v.n == 3
    v = P.loc[("vs lambda=0", "classifier-ft", 0.5)]                       # 0.80+0.02s - (0.65+0.01s)
    assert abs(v["mean"] - 0.16) < 1e-9 and abs(v.sd - 0.01) < 1e-9
    w = P.loc[("ft - t5", "classifier-ft", 0.5)]                            # 0.80+0.02s - (0.70+0.01s)
    assert abs(w["mean"] - 0.11) < 1e-9 and w.n == 3 and w.ci95 > 0
    w0 = P.loc[("ft - t5", "classifier-ft", 0.0)]                           # maple-ft - maple
    assert abs(w0["mean"] - 0.05) < 1e-9
    md = (tmp_path / "SWEEP.md").read_text()
    assert "ft - t5" in md and "± " in md and (tmp_path / "frontier.png").exists()
