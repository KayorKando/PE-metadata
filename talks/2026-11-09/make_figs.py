"""Figures for the 2026-11-09 advisor talk (dark theme, matches deck.typ).

Data: results/nano5-2026-11-05/eps4.0 (local rsync of Nano5).
t5 arms: 5 seeds complete. ft arms: seeds 0-1 complete, seeds 2-4 only up to round 2-3 locally.
ft last-round means at n = 5 come from summarize.py on Nano5 (2026-11-08):
    ft mean = t5 mean (same seeds 0-4) + paired difference.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # t quantiles, no scipy here

ROOT = Path(__file__).resolve().parents[2] / "results/nano5-2026-11-05/eps4.0"
OUT = Path(__file__).resolve().parent / "figs"; OUT.mkdir(exist_ok=True)

BG, FG, MUT, GRID = "#000000", "#f0efec", "#858585", "#2e2e2e"
C_T5, C_Q, C_FT = "#3987e5", "#d95926", "#199e70"     # dark categorical slots 1-3
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": MUT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 8.5, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2})

BLIND = ["word_count", "dominant_data_type", "sample_size"]
OTHER = ["primary_research_area", "model_organism", "experimental_approach",
         "research_focus_scale", "disease_mention", "research_goal"]

def load():
    rows = []
    for arm_dir, emb in [("maple", "t5"), ("classifier", "t5"), ("maple-ft", "ft"), ("classifier-ft", "ft")]:
        for d in sorted((ROOT / arm_dir).glob("lam*")):
            m = re.match(r"lam([\d.]+)(?:_seed(\d+))?$", d.name)
            lam, seed = float(m.group(1)), int(m.group(2) or 0)
            f = d / "metrics.jsonl"
            if not f.exists(): continue
            for line in f.read_text().splitlines():
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                r.update(emb=emb, lam=lam, seed=seed); rows.append(r)
    return pd.DataFrame(rows)

df = load()
T = 4
t5 = df[df.emb == "t5"]; ft = df[df.emb == "ft"]
print(df.groupby(["emb", "lam", "round"]).seed.nunique().unstack())

# paired ft - t5 at the last round, n = 5 (summarize.py on Nano5, 2026-11-08)
PAIRED = {0.0: dict(mauve=(-0.0351, 0.0537), jsd_blind_mean=(-0.1967, 0.0108), jsd_other_mean=(-0.0392, 0.0055)),
          0.5: dict(mauve=(-0.0527, 0.0455), jsd_blind_mean=(-0.0972, 0.0029), jsd_other_mean=(-0.0216, 0.0032))}

def ci95(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), T975[n - 1] * x.std(ddof=1) / np.sqrt(n)

# ---------- fig 1: drift under t5 MAPLE (lambda = 0) ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(5.9, 2.25), gridspec_kw=dict(width_ratios=[1.6, 1]))
g = t5[t5.lam == 0].groupby("round")
for att in OTHER:
    a.plot(g[f"jsd_{att}"].mean(), color=MUT, lw=0.9, alpha=0.8)
marks = {"word_count": "o", "dominant_data_type": "s", "sample_size": "^"}
for att in BLIND:
    y = g[f"jsd_{att}"].mean()
    a.plot(y, color=FG, lw=1.8, marker=marks[att], ms=4.5)
    a.annotate(att, (T, y.iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=FG)
a.annotate("6 other attributes", (T, g["jsd_research_goal"].mean().iloc[-1]), xytext=(5, -2),
           textcoords="offset points", va="center", fontsize=7.5, color=MUT)
a.set_xticks(range(T + 1)); a.set_xlabel("PE round"); a.set_ylabel("JSD vs real test set")
a.set_title("Metadata JSD per attribute (lower is better)", loc="left"); a.set_xlim(-0.2, T + 2.6)
mu, sd = g["mauve"].mean(), g["mauve"].std()
b.fill_between(mu.index, mu - sd, mu + sd, color=FG, alpha=0.15, lw=0)
b.plot(mu, color=FG, marker="o", ms=4)
b.set_xticks(range(T + 1)); b.set_xlabel("PE round"); b.set_ylim(0.4, 0.9)
b.set_title("MAUVE (higher is better)", loc="left")
fig.tight_layout(); fig.savefig(OUT / "drift_t5.png", dpi=250); plt.close(fig)

# ---------- fig 2: quota under t5, paired vs lambda = 0, last round ----------
last = t5[t5["round"] == T].set_index(["lam", "seed"])
fig, axes = plt.subplots(1, 3, figsize=(5.9, 1.95), sharex=True)
for ax, (k, title) in zip(axes, [("jsd_blind_mean", "Δ blind JSD"), ("jsd_other_mean", "Δ other JSD"), ("mauve", "Δ MAUVE")]):
    lams = [0.25, 0.5, 0.75, 1.0]; mus, cis = [], []
    for l in lams:
        d = last.loc[l, k] - last.loc[0.0, k]
        m_, c_ = ci95(d.dropna()); mus.append(m_); cis.append(c_)
    ax.axhline(0, color=MUT, lw=0.8)
    ax.errorbar(lams, mus, yerr=cis, color=C_Q, marker="o", ms=4.5, lw=1.8, capsize=0, elinewidth=1.6)
    ax.set_title(title, loc="left"); ax.set_xticks(lams); ax.set_xticklabels(["0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("λ")
    print(k, [f"{m_:+.3f}±{c_:.3f}" for m_, c_ in zip(mus, cis)])
fig.tight_layout(); fig.savefig(OUT / "quota_t5.png", dpi=250); plt.close(fig)

# ---------- fig 3: k-NN kappa per attribute (Step 3 check, CHECK.md numbers) ----------
kappa = pd.DataFrame({
    "t5":        [0.533, 0.340, 0.544, 0.326, 0.428, 0.585, 0.234, 0.355, 0.046],
    "frozen":    [0.474, 0.260, 0.592, 0.348, 0.462, 0.423, 0.258, 0.432, 0.090],
    "ft":        [0.586, 0.660, 0.703, 0.567, 0.549, 0.693, 0.454, 0.441, 0.033]},
    index=["primary_research_area", "model_organism", "experimental_approach", "dominant_data_type",
           "research_focus_scale", "disease_mention", "sample_size", "research_goal", "word_count (not trained)"])
kappa = kappa.sort_values("ft")
fig, ax = plt.subplots(figsize=(3.6, 3.1))
yy = np.arange(len(kappa))
for i, (_, r) in enumerate(kappa.iterrows()):
    ax.plot([min(r), max(r)], [i, i], color=GRID, lw=2.5, zorder=1)
for col, c, lab, mk in [("frozen", MUT, "frozen", "D"), ("t5", C_T5, "t5", "o"),
                        ("ft", C_FT, "ft", "o")]:
    ax.scatter(kappa[col], yy, color=c, s=24, marker=mk, label=lab, zorder=3, edgecolors=BG, linewidths=0.8)
ax.set_yticks(yy); ax.set_yticklabels(kappa.index, fontsize=7.5)
for t in ax.get_yticklabels():
    if t.get_text().split(" ")[0] in BLIND: t.set_color(FG)
    else: t.set_color(MUT)
ax.set_xlabel("k-NN κ (k = 5)"); ax.set_xlim(0, 0.75)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, fontsize=7, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
fig.tight_layout(); fig.savefig(OUT / "knn_kappa.png", dpi=250); plt.close(fig)

# ---------- fig 4: ft vs t5, blind JSD per round + paired differences ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(5.9, 2.25), gridspec_kw=dict(width_ratios=[1.15, 1]))
for sub, c, lab in [(t5[t5.lam == 0], C_T5, "t5 (MAPLE)"), (ft[ft.lam == 0], C_FT, "ft")]:
    for s, gs in sub.groupby("seed"):
        a.plot(gs["round"], gs["jsd_blind_mean"], color=c, lw=0.7, alpha=0.45)
    full = sub.groupby("round")["jsd_blind_mean"].mean()
    if lab == "ft":   # last round: n = 5 mean from Nano5 (paired diff + same-seed t5 mean)
        full.loc[T] = last.loc[0.0, "jsd_blind_mean"].mean() + PAIRED[0.0]["jsd_blind_mean"][0]
    a.plot(full.index, full.values, color=c, lw=2.2, marker="o", ms=4)
    a.annotate(lab, (T, full.loc[T]), xytext=(5, 0), textcoords="offset points", va="center", color=FG, fontsize=7.5)
a.set_xticks(range(T + 1)); a.set_xlabel("PE round"); a.set_xlim(-0.2, T + 1.1)
a.set_title("Blind-attribute JSD, λ = 0", loc="left")
rows = [("jsd_blind_mean", "blind JSD ↓"), ("jsd_other_mean", "other JSD ↓"), ("mauve", "MAUVE ↑")]
for j, (lam, mk, fc) in enumerate([(0.0, "o", C_FT), (0.5, "s", BG)]):
    for i, (k, _) in enumerate(rows):
        m_, c_ = PAIRED[lam][k]; y = i + (j - 0.5) * 0.32
        b.errorbar(m_, y, xerr=c_, color=C_FT, marker=mk, ms=5, mfc=fc, mew=1.4, elinewidth=1.6, capsize=0,
                   label=f"λ = {lam:g}" if i == 0 else None)
b.axvline(0, color=MUT, lw=0.8)
b.set_yticks(range(3)); b.set_yticklabels([r[1] for r in rows]); b.invert_yaxis()
b.set_title("ft − t5, last round, n = 5", loc="left"); b.set_xlabel("paired difference (95% CI)")
b.legend(loc="lower left", fontsize=7, handletextpad=0.2)
fig.tight_layout(); fig.savefig(OUT / "ft_vs_t5.png", dpi=250); plt.close(fig)

# ---------- fig 5: visibility model illustration ----------
# candidates: d_{t+1} = rho (1-g) d_t + (1-rho), in units of |pi - p|; survivors (what we measure): e_t = (1-g) d_t
rho, d0, R = 0.6, 0.5, 8
fig, ax = plt.subplots(figsize=(3.4, 2.15))
cols = ["#0d4f3a", "#147a5a", "#199e70", "#5fcca4"]   # one hue, lighter = more visible
for g_, c in zip([0.0, 0.3, 0.6, 0.9], cols):
    d = [d0]
    for _ in range(R): d.append(rho * (1 - g_) * d[-1] + (1 - rho))
    e = [(1 - g_) * x for x in d]
    ax.plot(range(R + 1), e, color=c, marker="o", ms=3)
    ax.annotate(f"g = {g_:g}", (R, e[-1]), xytext=(5, 0), textcoords="offset points", va="center", color=FG, fontsize=7.5)
ax.set_xlabel("PE round"); ax.set_ylabel("‖s_t − p‖ / ‖π − p‖"); ax.set_xlim(-0.3, R + 1.8); ax.set_ylim(0, 1.05)
ax.set_title(f"toy values: ρ = {rho}, d₀ = {d0}", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "model_toy.png", dpi=250); plt.close(fig)

# ---------- fig 6: MAUVE-JSD frontier at the last round ----------
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.45), sharey=True)
t5lams = [0.0, 0.25, 0.5, 0.75, 1.0]
for ax, (k, xl) in zip(axes, [("jsd_blind_mean", "blind JSD (lower is better)"), ("jsd_other_mean", "other JSD (lower is better)")]):
    xs, ys, xe, ye = [], [], [], []
    for l in t5lams:
        d = last.loc[l]
        xs.append(d[k].mean()); ys.append(d["mauve"].mean()); xe.append(d[k].std()); ye.append(d["mauve"].std())
    ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=C_T5, marker="o", ms=4.5, lw=1.6, elinewidth=1, capsize=0, label="t5 + quota")
    for l, x, y in zip(t5lams, xs, ys):
        ax.annotate(f"{l:g}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUT)
    # ft: lambda 0 and 0.5 = n = 5 means (same-seed t5 mean + paired diff); lambda 1 = local seeds 0-1
    fx, fy = [], []
    for l in [0.0, 0.5]:
        ref = last.loc[l]
        fx.append(ref[k].mean() + PAIRED[l][k][0]); fy.append(ref["mauve"].mean() + PAIRED[l]["mauve"][0])
    f1 = ft[(ft.lam == 1.0) & (ft["round"] == T)]
    fx.append(f1[k].mean()); fy.append(f1["mauve"].mean())
    ax.plot(fx, fy, color=C_FT, marker="o", ms=4.5, lw=1.6, label="ft + quota")
    for l, x, y in zip([0, 0.5, 1], fx, fy):
        ax.annotate(f"{l:g}", (x, y), xytext=(4, -9), textcoords="offset points", fontsize=7, color=MUT)
    ax.set_xlabel(xl)
    print(k, "ft", [f"({x:.3f}, {y:.3f})" for x, y in zip(fx, fy)])
axes[0].set_ylabel("MAUVE (higher is better)")
axes[0].legend(loc="lower right", fontsize=7, handletextpad=0.3)
fig.tight_layout(); fig.savefig(OUT / "frontier.png", dpi=250); plt.close(fig)

print("done", sorted(p.name for p in OUT.iterdir()))
