"""Collect a lambda sweep: <root>/<loop_annotator>[-<vemb>]/lam<value>[_seed<s>]/metrics.jsonl
<vemb> = vote-embedding family (none = sentence-t5; ft, fts<N>, ftt5 = fine-tuned, see nano5_lib.sh).
Third-party MAUVE (quota_sweep/mauve_rescore.py) is merged in from mauve_<tag>.jsonl next to
metrics.jsonl, as extra metrics mauve_<tag>.

Writes <root>/SWEEP.md, sweep_final.csv (last round, mean / sd / n over seeds), sweep_rounds.csv,
sweep_paired.csv (paired differences, see below), frontier.png (MAUVE vs metadata JSD at the
last round, error bars = sd over seeds) and curves.png (per round).

Baselines: "maple" = the lambda = 0 run (quota off, so loop labels do not matter); it is shown as
the lambda = 0 point of every t5 curve. "maple-<vemb>" is the same with another vote embedding and
is shown only on the curves of that family (a baseline never stands in for another family's).

Paired differences (same seed = same privacy-noise and generation seed):
  vs lambda = 0:  metric(annotator, lambda, seed) - metric(annotator, 0, seed)
  <vemb> - t5:    metric(<ann>-<vemb>, lambda, seed) - metric(<ann>, lambda, seed)
reported as mean, sd, n and a 95% t-interval for the mean.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["mauve", "jsd_mean", "jsd_blind_mean", "jsd_other_mean"]


def family(annotator):
    """'classifier-ftt5' -> 'ftt5', 'maple' -> '' (sentence-t5)."""
    return annotator.split("-", 1)[1] if "-" in annotator else ""


def is_ft(annotator):
    return family(annotator) != ""


def is_base(annotator):
    return annotator.split("-", 1)[0] == "maple"


def extra_mauve(R):
    return sorted(c for c in R.columns if c.startswith("mauve_")) if len(R) else []


def load(root):
    rows = []
    for f in sorted(Path(root).glob("*/lam*/metrics.jsonl")):
        m = re.match(r"lam([0-9.]+)(?:_seed(\d+))?$", f.parent.name)
        if not m:
            continue
        extra = {}
        for g in sorted(f.parent.glob("mauve_*.jsonl")):     # third-party MAUVE, by round
            for line in g.read_text().splitlines():
                e = json.loads(line)
                extra.setdefault(e["round"], {}).update({k: v for k, v in e.items() if k.startswith("mauve_")})
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r.update(extra.get(r["round"], {}))
            r.update(annotator=f.parent.parent.name, lam=float(m.group(1)), seed=int(m.group(2) or 0),
                     done=(f.parent / "DONE").exists())
            rows.append(r)
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    # a round logged twice (two jobs ran the same run dir at once): keep the last line, warn
    dup = R.duplicated(["annotator", "lam", "seed", "round"], keep="last")
    if dup.any():
        bad = sorted(set(R[dup].apply(lambda r: f"{r.annotator}/lam{r.lam:g} seed {r.seed}", axis=1)))
        print(f"WARNING: duplicated rounds (two writers?) in {bad}; keeping the last line of each round")
        R = R[~dup]
    parts = [R[~R.annotator.map(is_base)]]
    for base_name in sorted(a for a in R.annotator.unique() if is_base(a)):
        base = R[R.annotator == base_name]
        others = [a for a in R.annotator.unique() if not is_base(a) and family(a) == family(base_name)]
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
    ft["base"] = ft.annotator.str.split("-", n=1).str[0]
    # reference = the t5 run of the same loop annotator, lambda and seed; when that annotator has no t5 dir of
    # its own (e.g. maple-ftt5: its t5 baseline maple/ is shown as classifier lambda = 0), any t5 copy will do
    t5 = last[~last.annotator.map(is_ft)]
    exact = t5[["annotator", "lam", "seed"] + keys].rename(columns={"annotator": "base"})
    anyt5 = t5.drop_duplicates(["lam", "seed"])[["lam", "seed"] + keys]
    ft["base"] = ft["base"].where(ft["base"].isin(set(t5.annotator)), "__any__")
    Q = pd.concat([ft[ft.base != "__any__"].merge(exact, on=["base", "lam", "seed"], suffixes=("", "_ref")),
                   ft[ft.base == "__any__"].merge(anyt5, on=["lam", "seed"], suffixes=("", "_ref"))],
                  ignore_index=True)
    for fam in sorted(Q.annotator.map(family).unique()) if len(Q) else []:
        out.append((f"{fam} - t5", Q[Q.annotator.map(family) == fam]))
    rows = []
    for name, D in out:
        for k in keys:
            D = D.assign(d=D[k] - D[k + "_ref"])
            for (ann, lam), g in D.dropna(subset=["d"]).groupby(["annotator", "lam"]):
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
    keys = KEYS + extra_mauve(R)
    jsd_cols = sorted(c for c in R.columns if c.startswith("jsd_") and not c.startswith("jsd_gemini")
                      and c not in KEYS)
    g = last.groupby(["annotator", "lam"])
    F = g[keys + jsd_cols + ["round"]].mean().reset_index()
    sd = g[keys].std().reset_index().rename(columns={k: k + "_sd" for k in keys})
    F = F.merge(sd, on=["annotator", "lam"]).merge(g.seed.count().rename("n_seeds").reset_index(), on=["annotator", "lam"])
    F = F.merge(g.done.all().reset_index(), on=["annotator", "lam"])
    F.to_csv(root / "sweep_final.csv", index=False)
    PD = paired(last, keys)
    PD.to_csv(root / "sweep_paired.csv", index=False)

    parts = ["# Lambda sweep\n", f"Runs (seeds per point): {g.size().to_dict()}\n",
             "## Last round, mean ± sd over seeds\n", md(pm_table(g[keys].agg(["mean", "std", "count"]), keys)), ""]
    for name in (list(dict.fromkeys(PD.comparison)) if len(PD) else []):
        sub = PD[PD.comparison == name]
        if not len(sub):
            continue
        W = sub.assign(cell=[f"{m:+.4f} ± {c:.4f} (n={n})" if c == c else f"{m:+.4f} (n={n})"
                             for m, c, n in zip(sub["mean"], sub.ci95, sub.n)])
        W = W.pivot_table(index=["annotator", "lam"], columns="metric", values="cell", aggfunc="first")
        W = W[[k for k in keys if k in W.columns]].reset_index()
        parts += [f"## Paired difference at the last round: {name} (mean ± 95% CI over seeds)\n", md(W), ""]
    parts += ["## Per-attribute JSD at the last round (mean over seeds)\n", md(F[["annotator", "lam"] + jsd_cols]), "",
              "## Per round (mean over seeds)\n",
              md(R.groupby(["annotator", "lam", "round"])[keys + ["moved", "shortfall"]].mean().reset_index()), ""]

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
