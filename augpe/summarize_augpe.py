"""Collect the Aug-PE drift runs and put them next to the MAPLE baseline (lambda = 0, t5 vote).

Inputs
  --augpe_root  $PE_DATA/augpe/eps4.0            run dirs seed<s>/ (augpe_pe.py)
  --maple_root  $PE_DATA/quota_sweep/eps4.0/maple run dirs lam0[_seed<s>]/ (quota_pe.py, lambda = 0)   [optional]
Third-party MAUVE (mauve_<tag>.jsonl from quota_sweep/mauve_rescore.py) is merged in when present.
Aug-PE's LLM prior (prior_metrics.json: round-0 pool before any vote) is shown as round -1.

Outputs (in --out, default --augpe_root)
  drift_runs.csv    one row per method x seed x round, all metrics (the data behind every plot)
  drift_rounds.csv  mean / sd / n over seeds per method x round x metric
  drift_change.csv  Aug-PE: per-attribute change of JSD from round 0 to the last round, mean and 95% t-interval
                    over seeds (paired by seed); > 0 with the interval above 0 = drift away from the real data
  DRIFT.md          the tables below; drift_curves.png (blind / other JSD, MAUVE per round), drift_attrs.png
                    (JSD per attribute per round)

    python augpe/summarize_augpe.py --augpe_root $PE_DATA/augpe/eps4.0 --maple_root $PE_DATA/quota_sweep/eps4.0/maple
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import ATTRIBUTES as ATTRS  # noqa: E402  (active dataset, PE_DATASET; + word_count)

MAIN = ["mauve", "jsd_blind_mean", "jsd_other_mean", "jsd_mean"]


def _seed(name, pattern):
    m = re.fullmatch(pattern, name)
    return None if m is None else int(m.group(1) or 0)


def load_run(d):
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    extra = {}
    for g in sorted(d.glob("mauve_*.jsonl")):
        for line in g.read_text().splitlines():
            e = json.loads(line)
            extra.setdefault(e["round"], {}).update({k: v for k, v in e.items() if k.startswith("mauve_")})
    for r in rows:
        r.update(extra.get(r["round"], {}))
    return rows


def load(augpe_root, maple_root=None):
    out = []
    for d in sorted(Path(augpe_root).glob("seed*")):
        s = _seed(d.name, r"seed(\d+)")
        if s is None or not (d / "metrics.jsonl").exists():
            continue
        if (d / "prior_metrics.json").exists():
            pm = json.loads((d / "prior_metrics.json").read_text())
            out.append({**{k: v for k, v in pm.items() if k != "n"}, "method": "augpe", "seed": s, "round": -1})
        out += [{**r, "method": "augpe", "seed": s} for r in load_run(d)]
    if maple_root and Path(maple_root).exists():
        for d in sorted(Path(maple_root).glob("lam0*")):
            s = _seed(d.name, r"lam0(?:\.0)?(?:_seed(\d+))?")
            if s is None or not (d / "metrics.jsonl").exists():
                continue
            out += [{**r, "method": "maple", "seed": s} for r in load_run(d)]
    return pd.DataFrame(out)


def metric_cols(R):
    return [c for c in R.columns if c.startswith(("jsd_", "mauve")) and pd.api.types.is_numeric_dtype(R[c])]


def by_round(R):
    cols = metric_cols(R)
    L = R.melt(id_vars=["method", "seed", "round"], value_vars=cols, var_name="metric").dropna(subset=["value"])
    g = L.groupby(["method", "round", "metric"])["value"]
    return g.agg(mean="mean", sd="std", n="count").reset_index()


def t_interval(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def change(R, method="augpe", start=0):
    """JSD(last round) - JSD(round `start`) per attribute, paired by seed."""
    A = R[(R.method == method) & (R["round"] >= start)]
    rows = []
    for key in [f"jsd_{a}" for a in ATTRS] + ["jsd_blind_mean", "jsd_other_mean", "jsd_label_mean", "mauve"]:
        if key not in A:
            continue
        d = []
        for _, g in A.groupby("seed"):
            g = g.dropna(subset=[key]).sort_values("round")
            if len(g) >= 2 and g["round"].iloc[0] == start:
                d.append(g[key].iloc[-1] - g[key].iloc[0])
        m, lo, hi = t_interval(d)
        rows.append(dict(metric=key, delta_mean=m, ci_lo=lo, ci_hi=hi, n=len(d),
                         verdict=("up" if lo > 0 else "down" if hi < 0 else "n.s.") if len(d) >= 2 else "n<2"))
    return pd.DataFrame(rows)


def fmt(S, method, metric, rnd):
    r = S[(S.method == method) & (S.metric == metric) & (S["round"] == rnd)]
    if not len(r):
        return "—"
    r = r.iloc[0]
    return f"{r['mean']:.3f}" + ("" if np.isnan(r["sd"]) else f" ± {r['sd']:.3f}") + f" (n={int(r['n'])})"


def write_md(R, S, C, path):
    lines = ["# Metadata drift: Aug-PE vs MAPLE (t5 vote, lambda = 0)", "",
             "Mean ± sd over seeds. Round -1 = Aug-PE's round-0 pool before any vote (the LLM prior).", ""]
    rounds = sorted(S["round"].unique())
    main = MAIN + [k for k in ["jsd_label_mean"] if k in set(S.metric)]
    for metric in main + [c for c in sorted(S.metric.unique()) if c.startswith("mauve_")]:
        lines += [f"## {metric}", "", "| round | Aug-PE | MAPLE |", "|---|---|---|"]
        lines += [f"| {t} | {fmt(S, 'augpe', metric, t)} | {fmt(S, 'maple', metric, t)} |" for t in rounds]
        lines.append("")
    for tag, rnd in [("last Aug-PE round", int(R[R.method == "augpe"]["round"].max()) if (R.method == "augpe").any() else None),
                     ("round 4 (MAPLE's last round at eps 4)", 4)]:
        if rnd is None:
            continue
        lines += [f"## Per-attribute JSD vs real test set, {tag} (round {rnd})", "",
                  "| attribute | Aug-PE prior (round -1) | Aug-PE round 0 | Aug-PE | MAPLE |", "|---|---|---|---|---|"]
        for a in ATTRS:
            k = f"jsd_{a}"
            lines.append(f"| {a} | {fmt(S, 'augpe', k, -1)} | {fmt(S, 'augpe', k, 0)} | {fmt(S, 'augpe', k, rnd)} | "
                         f"{fmt(S, 'maple', k, rnd)} |")
        lines.append("")
    if len(C):
        lines += ["## Aug-PE: change from round 0 to the last round (paired by seed, 95% t-interval)", "",
                  "| metric | Δ mean | 95% CI | n | verdict |", "|---|---|---|---|---|"]
        lines += [f"| {r.metric} | {r.delta_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.n} | {r.verdict} |"
                  for r in C.itertuples()]
        lines.append("")
    Path(path).write_text("\n".join(lines))


def plot(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"augpe": ("Aug-PE", "#d1495b"), "maple": ("MAPLE (t5, λ = 0)", "#30638e")}

    def curve(ax, metric):
        for m, (lab, col) in style.items():
            s = S[(S.method == m) & (S.metric == metric)].sort_values("round")
            if not len(s):
                continue
            ax.errorbar(s["round"], s["mean"], yerr=s["sd"].fillna(0), label=lab, color=col, marker="o", ms=3,
                        capsize=2, lw=1.5)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("PE round (-1 = prior)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    keys = ["jsd_blind_mean", "jsd_other_mean", "mauve"] + [c for c in sorted(S.metric.unique()) if c.startswith("mauve_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3))
    for ax, k in zip(np.atleast_1d(axes), keys):
        curve(ax, k)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / "drift_curves.png", dpi=160)
    plt.close(fig)

    ncol = 3 if len(ATTRS) <= 9 else 4
    nrow = int(np.ceil(len(ATTRS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.85 * nrow), squeeze=False)
    for ax in axes.flat[len(ATTRS):]:
        ax.set_visible(False)
    for ax, a in zip(axes.flat, ATTRS):
        curve(ax, f"jsd_{a}")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("JSD vs real test set (Qwen labels; word_count measured)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out) / "drift_attrs.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--augpe_root", required=True)
    p.add_argument("--maple_root", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out or args.augpe_root)
    out.mkdir(parents=True, exist_ok=True)
    R = load(args.augpe_root, args.maple_root)
    if not len(R):
        raise SystemExit(f"no finished rounds under {args.augpe_root}")
    S = by_round(R)
    C = change(R)
    R.sort_values(["method", "seed", "round"]).to_csv(out / "drift_runs.csv", index=False)
    S.to_csv(out / "drift_rounds.csv", index=False)
    C.to_csv(out / "drift_change.csv", index=False)
    write_md(R, S, C, out / "DRIFT.md")
    plot(S, out)
    print((out / "DRIFT.md").read_text())


if __name__ == "__main__":
    main()
