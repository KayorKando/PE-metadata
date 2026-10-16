"""Collect a lambda sweep: <root>/<loop_annotator>/lam<value>[_seed<s>]/metrics.jsonl

Writes <root>/SWEEP.md, sweep_final.csv (last round of every run), sweep_rounds.csv,
frontier.png (MAUVE vs metadata JSD at the last round) and curves.png (per round).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load(root):
    rows = []
    for f in sorted(Path(root).glob("*/lam*/metrics.jsonl")):
        m = re.match(r"lam([0-9.]+)(?:_seed(\d+))?", f.parent.name)
        if not m:
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r.update(annotator=f.parent.parent.name, lam=float(m.group(1)), seed=int(m.group(2) or 0),
                     done=(f.parent / "DONE").exists())
            rows.append(r)
    R = pd.DataFrame(rows)
    # "maple" = the shared lambda = 0 baseline (the quota is off, so loop labels do not matter):
    # show it as the lambda = 0 point of every annotator's curve.
    if not R.empty and "maple" in set(R.annotator):
        base = R[R.annotator == "maple"]
        others = [a for a in R.annotator.unique() if a != "maple"]
        R = pd.concat([R[R.annotator != "maple"]] + [base.assign(annotator=a) for a in others] +
                      ([] if others else [base]), ignore_index=True)
    return R


def md(df):
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    R = load(root)
    if R.empty:
        print("no runs yet"); return
    R.to_csv(root / "sweep_rounds.csv", index=False)
    last = R.sort_values("round").groupby(["annotator", "lam", "seed"]).tail(1)
    keys = ["mauve", "jsd_mean", "jsd_blind_mean", "jsd_other_mean"]
    jsd_cols = sorted(c for c in R.columns if c.startswith("jsd_") and not c.startswith("jsd_gemini")
                      and c not in keys)
    F = last.groupby(["annotator", "lam"])[keys + jsd_cols + ["round", "done"]].agg(
        lambda x: x.mean() if x.dtype != bool else x.all()).reset_index()
    F.to_csv(root / "sweep_final.csv", index=False)

    parts = ["# Lambda sweep\n", f"Runs: {last.groupby(['annotator', 'lam']).size().to_dict()}\n",
             "## Last round (mean over seeds)\n", md(F[["annotator", "lam", "round", "done"] + keys]), "",
             "## Per-attribute JSD at the last round\n", md(F[["annotator", "lam"] + jsd_cols]), "",
             "## Per round\n",
             md(R.groupby(["annotator", "lam", "round"])[keys + ["moved", "shortfall"]].mean().reset_index()), ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, y in zip(axes, ["jsd_mean", "jsd_blind_mean"]):
        for ann, g in F.groupby("annotator"):
            g = g.sort_values("lam")
            ax.plot(g[y], g["mauve"], "-o", label=f"loop labels: {ann}")
            for _, r in g.iterrows():
                ax.annotate(f"λ={r.lam:g}", (r[y], r["mauve"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(y.replace("_", " ") + " (lower is better)")
        ax.set_ylabel("MAUVE (higher is better)")
        ax.legend(fontsize=8)
    fig.suptitle("MAUVE–JSD frontier at the last PE round")
    fig.tight_layout(); fig.savefig(root / "frontier.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, y in zip(axes, ["mauve", "jsd_blind_mean", "jsd_other_mean"]):
        for (ann, lam), g in R.groupby(["annotator", "lam"]):
            g = g.groupby("round")[y].mean()
            ax.plot(g.index, g.values, "-o", ms=3, label=f"{ann} λ={lam:g}")
        ax.set_xlabel("PE round"); ax.set_title(y.replace("_", " "))
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root / "curves.png", dpi=150); plt.close(fig)
    parts.append("Figures: `frontier.png`, `curves.png`.")
    (root / "SWEEP.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
