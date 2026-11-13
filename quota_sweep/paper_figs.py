"""Paper figures for the lambda sweep, plus the exact numbers behind every mark.

  python quota_sweep/paper_figs.py --root <sweep root> --out figures/paper

Reads <root>/<annotator>/lam*/metrics.jsonl (same loader as summarize.py) and writes to --out:
  rounds.pdf/.png    blind JSD, other JSD, MAUVE per PE round for three settings:
                     MAPLE (t5 vote), t5 vote + quota (lambda = --lam), fine-tuned vote (lambda = 0)
  frontier.pdf/.png  last round, blind JSD vs MAUVE: the t5 lambda path and the ft points
  data_runs.csv      every run x round (the raw data, long format)
  data_rounds.csv    mean / sd / n / 95% CI per setting x round (what rounds.pdf draws)
  data_frontier.csv  the same at the last round for every (embedding, lambda) (what frontier.pdf draws)
Bands and error bars: --band sd (default) or ci (95% t-interval; very wide at n = 2).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import load  # noqa: E402

# Colours: ours = blue, quota = orange (first two slots of a CVD-validated categorical palette);
# the MAPLE baseline is a neutral gray and also dashed, so it never relies on colour alone.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_FT, C_Q, C_BASE = "#2a78d6", "#eb6834", "#8c8b86"
METRICS = [("jsd_blind_mean", "Blind-attribute JSD", "lower is better"),
           ("jsd_other_mean", "Other-attribute JSD", "lower is better"),
           ("mauve", "MAUVE", "higher is better")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def agg(df, by):
    keys = [m for m, _, _ in METRICS]
    g = df.groupby(by)[keys]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("n"))
    for k in keys:
        n = out["n"].to_numpy()
        tq = np.where(n > 1, student_t.ppf(0.975, np.maximum(n - 1, 1)), np.nan)
        out[k + "_ci95"] = tq * out[k + "_sd"] / np.sqrt(n)
    return out.reset_index()


def spread(row, k, band):
    return row[f"{k}_sd" if band == "sd" else f"{k}_ci95"]


def fig_rounds(A, settings, band, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), sharex=True)
    for ax, (k, title, hint) in zip(axes, METRICS):
        for name, ann, lam, color, ls in settings:
            d = A[(A.annotator == ann) & (A.lam == lam)].sort_values("round")
            if d.empty:
                continue
            x, y, s = d["round"], d[k + "_mean"], d.apply(lambda r: spread(r, k, band), axis=1)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.14, lw=0)
            ax.plot(x, y, color=color, ls=ls, lw=1.5, marker="o", ms=3.2,
                    mec="white", mew=0.6, label=f"{name} (n={int(d['n'].min())})", zorder=3)
        arrow = "↓" if hint.startswith("lower") else "↑"
        ax.set_title(f"{title}  {arrow}", loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("PE round")
        ax.set_xticks(sorted(A["round"].unique()))
        ax.margins(x=0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.10), fontsize=7.5,
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"rounds.{ext}", dpi=220)
    plt.close(fig)


def fig_frontier(F, band, out):
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    fams = [("t5 vote + quota (λ path)", lambda a: not a.endswith("-ft"), C_Q, "-"),
            ("fine-tuned vote (+ quota)", lambda a: a.endswith("-ft"), C_FT, "-")]
    for name, sel, color, ls in fams:
        d = F[F.annotator.map(sel)].sort_values("lam")
        if d.empty:
            continue
        x, y = d["jsd_blind_mean_mean"], d["mauve_mean"]
        xe = d.apply(lambda r: spread(r, "jsd_blind_mean", band), axis=1)
        ye = d.apply(lambda r: spread(r, "mauve", band), axis=1)
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=color, elinewidth=0.8, alpha=0.45, zorder=2)
        ax.plot(x, y, color=color, ls=ls, lw=1.2, marker="o", ms=4.5, mec="white", mew=0.7,
                label=f"{name}, n={int(d['n'].min())}", zorder=3)
        ft = d["annotator"].iloc[0].endswith("-ft")
        for xi, yi, lam in zip(x, y, d["lam"]):
            txt = "MAPLE" if (lam == 0 and not ft) else f"λ={lam:g}"
            dx = -5 if ft else 5
            ax.annotate(txt, (xi, yi), xytext=(dx, 4), textcoords="offset points",
                        ha="right" if ft else "left", fontsize=6.5, color=INK if txt == "MAPLE" else INK2)
    ax.annotate("better", xy=(0.03, 0.97), xytext=(0.17, 0.83), xycoords="axes fraction",
                textcoords="axes fraction", fontsize=6.5, color=INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
    ax.set_xlabel("Blind-attribute JSD (lower is better)")
    ax.set_ylabel("MAUVE (higher is better)")
    ax.grid(True, axis="both")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=6.8, borderaxespad=0.2)
    ax.margins(x=0.12, y=0.08)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"frontier.{ext}", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="figures/paper")
    p.add_argument("--lam", type=float, default=0.5, help="quota strength shown for the t5 + quota curve")
    p.add_argument("--band", choices=["sd", "ci"], default="sd")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    R = load(a.root)
    R = R[R.done]
    R.to_csv(out / "data_runs.csv", index=False)
    A = agg(R, ["annotator", "lam", "round"])
    A.to_csv(out / "data_rounds.csv", index=False)
    last = R[R["round"] == R.groupby(["annotator", "lam", "seed"])["round"].transform("max")]
    F = agg(last, ["annotator", "lam"])
    F.to_csv(out / "data_frontier.csv", index=False)

    settings = [("MAPLE (t5 vote)", "classifier", 0.0, C_BASE, "--"),
                (f"t5 vote + quota λ={a.lam:g}", "classifier", a.lam, C_Q, "-"),
                ("fine-tuned vote", "classifier-ft", 0.0, C_FT, "-")]
    fig_rounds(A, settings, a.band, out)
    fig_frontier(F, a.band, out)
    print(f"wrote {out}/: rounds.pdf/.png, frontier.pdf/.png, data_runs.csv, data_rounds.csv, data_frontier.csv")


if __name__ == "__main__":
    main()
