import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984"})
D = pd.read_csv("all_rounds.csv"); D = D[D.lam == 0]
D = D.groupby(["eps", "arm", "seed", "round"]).tail(1)            # ftt5: duplicated rounds -> last writer
C = {"maple": ("#2a78d6", "o", "sentence-t5 (MAPLE)"), "maple-ft": ("#eb6834", "s", "ft BiomedBERT"),
     "maple-ftt5": ("#1baf7a", "^", "ft sentence-t5"), "maple-fts1": ("#eda100", "D", "ft BiomedBERT, seed 1"),
     "maple-fts2": ("#e87ba4", "v", "ft BiomedBERT, seed 2")}
S = D.groupby(["eps", "arm", "round"]).agg(**{f"{k}_{s}": (k, s) for k in ["jsd_blind_mean", "mauve", "mauve_gpt2-large"]
                                               for s in ["mean", "std"]}, n=("seed", "count")).reset_index()
S.to_csv("out/data_rounds_lam0.csv", index=False)

def line(ax, g, arm, y, label=True):
    col, mk, lab = C[arm]
    ax.errorbar(g["round"], g[f"{y}_mean"], yerr=g[f"{y}_std"], color=col, marker=mk, ms=4.5, lw=2, capsize=2,
                label=lab, markeredgecolor="white", markeredgewidth=0.6)
    if label:
        r = g.iloc[-1]; ax.annotate(lab.split(" (")[0], (r["round"], r[f"{y}_mean"]), xytext=(5, 0),
                                    textcoords="offset points", fontsize=7.5, color="#3b3a36", va="center")

# Fig A: blind JSD per eps, t5 vs ft
eps_order = [("eps1.0", "ε = 1"), ("eps2.0", "ε = 2"), ("eps4.0", "ε = 4"), ("epsinf", "ε = ∞ (no privacy)")]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
for ax, (e, title) in zip(axes, eps_order):
    for arm in ["maple", "maple-ft"]:
        line(ax, S[(S.eps == e) & (S.arm == arm)], arm, "jsd_blind_mean", label=False)
    ax.set_title(title); ax.set_xlabel("PE round")
    ax.set_xticks(sorted(S[S.eps == e]["round"].unique()))
axes[0].set_ylabel("blind-attribute JSD (lower = better)")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Metadata drift on blind attributes: sentence-t5 vote vs fine-tuned vote (λ = 0, mean ± sd over 5 seeds)", fontsize=10)
fig.tight_layout(); [fig.savefig(f"out/drift_by_eps.{x}", dpi=200) for x in ("png", "pdf")]; plt.close(fig)

# Fig B: eps 4, all arms: blind JSD, t5-MAUVE, gpt2-MAUVE
arms = ["maple", "maple-ft", "maple-fts1", "maple-fts2", "maple-ftt5"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (y, t) in zip(axes, [("jsd_blind_mean", "blind-attribute JSD (lower = better)"),
                              ("mauve", "MAUVE, sentence-t5 features (= t5 vote space)"),
                              ("mauve_gpt2-large", "MAUVE, gpt2-large features (neutral)")]):
    for arm in arms:
        line(ax, S[(S.eps == "eps4.0") & (S.arm == arm)], arm, y, label=False)
    ax.set_title(t, fontsize=9); ax.set_xlabel("PE round"); ax.set_xticks(range(5))
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("ε = 4, λ = 0: every vote embedding (mean ± sd over 5 seeds)", fontsize=10)
fig.tight_layout(); [fig.savefig(f"out/eps4_arms_rounds.{x}", dpi=200) for x in ("png", "pdf")]; plt.close(fig)
print(S[S.eps == "eps4.0"].round(3).head(3))
