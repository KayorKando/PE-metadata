"""Collect a lambda sweep: <root>/<loop_annotator>[-ft]/lam<value>[_seed<s>]/metrics.jsonl

Writes <root>/SWEEP.md, sweep_final.csv (last round, mean / sd / n over seeds), sweep_rounds.csv,
sweep_paired.csv (paired differences, see below), frontier.png (MAUVE vs metadata JSD at the
last round, error bars = sd over seeds) and curves.png (per round).

Baselines: "maple" = the lambda = 0 run (quota off, so loop labels do not matter); it is shown as
the lambda = 0 point of every t5 curve. "maple-ft" is the same with the fine-tuned vote embedding
and is shown only on the -ft curves (a t5 baseline never stands in for an ft one, or vice versa).

Paired differences (same seed = same privacy-noise and generation seed):
  vs lambda = 0:  metric(annotator, lambda, seed) - metric(annotator, 0, seed)
  ft - t5:        metric(<ann>-ft, lambda, seed) - metric(<ann>, lambda, seed)
reported as mean, sd, n and a 95% t-interval for the mean.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["mauve", "jsd_mean", "jsd_blind_mean", "jsd_other_mean"]
BASES = {"maple": False, "maple-ft": True}      # baseline dir -> belongs to the ft family?


def is_ft(annotator):
    return annotator.endswith("-ft")


def load(root):
    rows = []
    for f in sorted(Path(root).glob("*/lam*/metrics.jsonl")):
        m = re.match(r"lam([0-9.]+)(?:_seed(\d+))?$", f.parent.name)
        if not m:
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r.update(annotator=f.parent.parent.name, lam=float(m.group(1)), seed=int(m.group(2) or 0),
                     done=(f.parent / "DONE").exists())
            rows.append(r)
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    parts = [R[~R.annotator.isin(list(BASES))]]
    for base_name, ft in BASES.items():
        if base_name not in set(R.annotator):
            continue
        base = R[R.annotator == base_name]
        others = [a for a in R.annotator.unique() if a not in BASES and is_ft(a) == ft]
        parts += [base.assign(annotator=a) for a in others] or [base]
    return pd.concat(parts, ignore_index=True)


def md(df):
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def ci95(sd, n):
    from scipy.stats import t
    return np.where(n > 1, t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n), np.nan)


def paired(last, keys):
    """-> long DataFrame: comparison, annotator, lam, metric, mean, sd, n, ci95."""
    out = []
    base = last[last.lam == 0][["annotator", "seed"] + keys]
    P = last[last.lam != 0].merge(base, on=["annotator", "seed"], suffixes=("", "_ref"))
    if len(P):
        out.append(("vs lambda=0", P))
    ft = last[last.annotator.map(is_ft)].copy()
    ft["base"] = ft.annotator.str[:-3]
    Q = ft.merge(last[["annotator", "lam", "seed"] + keys].rename(columns={"annotator": "base"}),
                 on=["base", "lam", "seed"], suffixes=("", "_ref"))
    if len(Q):
        out.append(("ft - t5", Q))
    rows = []
    for name, D in out:
        for k in keys:
            D = D.assign(d=D[k] - D[k + "_ref"])
            for (ann, lam), g in D.groupby(["annotator", "lam"]):
                n, sd = len(g), float(g.d.std()) if len(g) > 1 else float("nan")
                rows.append(dict(comparison=name, annotator=ann, lam=lam, metric=k, mean=float(g.d.mean()), sd=sd,
                                 n=n, ci95=float(ci95(np.array(sd), np.array(n)))))
    return pd.DataFrame(rows)


def pm_table(S, keys):
    """S: groupby(annotator, lam)[keys].agg([mean, std, count]) -> 'mean ± sd' table with n."""
    T = pd.DataFrame({"annotator": [i[0] for i in S.index], "lam": [i[1] for i in S.index],
                      "n_seeds": S[(keys[0], "count")].astype(int).to_numpy()})
    for k in keys:
        T[k] = [f"{m:.4f} ± {s:.4f}" if s == s else f"{m:.4f}" for m, s in zip(S[(k, "mean")], S[(k, "std")])]
    return T


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
    jsd_cols = sorted(c for c in R.columns if c.startswith("jsd_") and not c.startswith("jsd_gemini")
                      and c not in KEYS)
    g = last.groupby(["annotator", "lam"])
    F = g[KEYS + jsd_cols + ["round"]].mean().reset_index()
    sd = g[KEYS].std().reset_index().rename(columns={k: k + "_sd" for k in KEYS})
    F = F.merge(sd, on=["annotator", "lam"]).merge(g.seed.count().rename("n_seeds").reset_index(), on=["annotator", "lam"])
    F = F.merge(g.done.all().reset_index(), on=["annotator", "lam"])
    F.to_csv(root / "sweep_final.csv", index=False)
    PD = paired(last, KEYS)
    PD.to_csv(root / "sweep_paired.csv", index=False)

    parts = ["# Lambda sweep\n", f"Runs (seeds per point): {g.size().to_dict()}\n",
             "## Last round, mean ± sd over seeds\n", md(pm_table(g[KEYS].agg(["mean", "std", "count"]), KEYS)), ""]
    for name in ["vs lambda=0", "ft - t5"]:
        sub = PD[PD.comparison == name] if len(PD) else PD
        if not len(sub):
            continue
        W = sub.assign(cell=[f"{m:+.4f} ± {c:.4f} (n={n})" if c == c else f"{m:+.4f} (n={n})"
                             for m, c, n in zip(sub["mean"], sub.ci95, sub.n)])
        W = W.pivot_table(index=["annotator", "lam"], columns="metric", values="cell", aggfunc="first")[KEYS].reset_index()
        parts += [f"## Paired difference at the last round: {name} (mean ± 95% CI over seeds)\n", md(W), ""]
    parts += ["## Per-attribute JSD at the last round (mean over seeds)\n", md(F[["annotator", "lam"] + jsd_cols]), "",
              "## Per round (mean over seeds)\n",
              md(R.groupby(["annotator", "lam", "round"])[KEYS + ["moved", "shortfall"]].mean().reset_index()), ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, y in zip(axes, ["jsd_mean", "jsd_blind_mean"]):
        for ann, gg in F.groupby("annotator"):
            gg = gg.sort_values("lam")
            ax.errorbar(gg[y], gg["mauve"], xerr=gg[y + "_sd"].fillna(0), yerr=gg["mauve_sd"].fillna(0),
                        fmt="--o" if is_ft(ann) else "-o", capsize=2, ms=4, label=f"{ann}")
            for _, r in gg.iterrows():
                ax.annotate(f"λ={r.lam:g}", (r[y], r["mauve"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(y.replace("_", " ") + " (lower is better)")
        ax.set_ylabel("MAUVE (higher is better)")
        ax.legend(fontsize=8)
    fig.suptitle("MAUVE–JSD frontier at the last PE round (error bars: sd over seeds)")
    fig.tight_layout(); fig.savefig(root / "frontier.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, y in zip(axes, ["mauve", "jsd_blind_mean", "jsd_other_mean"]):
        for (ann, lam), gg in R.groupby(["annotator", "lam"]):
            gg = gg.groupby("round")[y].mean()
            ax.plot(gg.index, gg.values, "--o" if is_ft(ann) else "-o", ms=3, label=f"{ann} λ={lam:g}")
        ax.set_xlabel("PE round"); ax.set_title(y.replace("_", " ") + " (mean over seeds)")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root / "curves.png", dpi=150); plt.close(fig)
    parts.append("Figures: `frontier.png`, `curves.png`.")
    (root / "SWEEP.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
