"""Figures for Experiment 1 / 1b (numbers copied from the project doc experiment-1-knn-vs-probe.md).
Run: python3 plot_exp1.py  ->  figures/exp1/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class _R:  # minimal Spearman (no ties in these data)
    def __init__(self, c): self.correlation = c


def spearmanr(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return _R(float(np.corrcoef(rx, ry)[0, 1]))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "exp1")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29d"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})

# ---------------- private data: (k-NN, probe) at n = 2000, 5000, full ----------------
priv = {
    "word_count":            [(-0.017, 0.239), (-0.013, 0.388), (0.016, 0.630)],
    "dominant_data_type":    [(0.183, 0.332), (0.221, 0.382), (0.287, 0.517)],
    "sample_size":           [(0.199, 0.306), (0.223, 0.361), (0.289, 0.490)],
    "research_focus_scale":  [(0.321, 0.467), (0.357, 0.497), (0.407, 0.601)],
    "experimental_approach": [(0.358, 0.480), (0.361, 0.503), (0.419, 0.613)],
    "model_organism":        [(0.297, 0.492), (0.336, 0.516), (0.421, 0.619)],
    "research_goal":         [(0.429, 0.604), (0.440, 0.603), (0.481, 0.666)],
    "primary_research_area": [(0.379, 0.535), (0.411, 0.566), (0.491, 0.629)],
    "disease_mention":       [(0.434, 0.656), (0.467, 0.658), (0.547, 0.730)],
}
SELECTED = {"word_count", "dominant_data_type", "sample_size"}

# round-0 pool (k-NN, probe), measured word_count
pool = {
    "word_count": (0.062, 0.696), "sample_size": (0.297, 0.548),
    "dominant_data_type": (0.334, 0.634), "research_focus_scale": (0.341, 0.544),
    "model_organism": (0.378, 0.622), "primary_research_area": (0.434, 0.529),
    "experimental_approach": (0.503, 0.719), "research_goal": (0.570, 0.781),
    "disease_mention": (0.694, 0.822),
}

def dumbbell(ax, data, title):
    names = list(data)[::-1]
    for i, a in enumerate(names):
        k, p = data[a]
        ax.plot([k, p], [i, i], color=GRAY, lw=2, zorder=1)
        ax.scatter(k, i, s=55, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
        ax.scatter(p, i, s=55, color=ORANGE, zorder=3, edgecolor="white", lw=1.5)
        ax.text(p + 0.02, i, f"gap {p-k:.2f}", va="center", fontsize=8, color=INK2,
                fontweight="bold" if a == "word_count" else "normal")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([("★ " if a in SELECTED else "") + a for a in names])
    ax.set_xlim(-0.05, 1.0); ax.set_xlabel("κ (0 = chance, 1 = perfect)")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)

# Fig 1: dumbbell private (full n) and pool side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
dumbbell(axes[0], {a: v[2] for a, v in priv.items()}, "Private data, full n = 28,846")
dumbbell(axes[1], {a: pool[a] for a in priv}, "Round-0 synthetic pool, n = 14,000 (ε = 4)")
h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="k-NN κ (what the vote uses)"),
     plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="probe κ (what φ encodes)")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.text(0.01, -0.02, "★ = selected as top-3 k-NN-blind. Rows in private-data k-NN order.", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_knn_vs_probe_dumbbell.png"); plt.close(fig)

# Fig 2: scaling with n
ns = [2000, 5000, 28846]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for j, (ax, lab) in enumerate(zip(axes, ["k-NN κ", "probe κ"])):
    for a, v in priv.items():
        y = [t[j] for t in v]
        hi = a == "word_count"
        ax.plot(ns, y, marker="o", ms=5 if hi else 3, lw=2.2 if hi else 1.2,
                color=ORANGE if hi else GRAY, zorder=3 if hi else 2)
        if hi or a == "disease_mention":
            ax.text(ns[-1] * 1.12, y[-1], a, va="center", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(["2k", "5k", "28.8k"])
    ax.minorticks_off(); ax.set_xlim(1700, 90000)
    ax.set_xlabel("n (private points)"); ax.set_title(lab, loc="left", color=INK, fontsize=11)
axes[0].set_ylabel("κ"); axes[0].axhline(0, color=INK2, lw=0.8)
fig.suptitle("word_count: probe rises with n, k-NN stays at chance (gray = other 8 attributes)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_scaling_with_n.png"); plt.close(fig)

# Fig 3: transfer, private k-NN κ vs pool k-NN κ
fig, ax = plt.subplots(figsize=(5.6, 5))
xs = [priv[a][2][0] for a in priv]; ys = [pool[a][0] for a in priv]
ax.plot([0, 0.75], [0, 0.75], color=GRAY, lw=1, ls="--", zorder=1)
for a, x, y in zip(priv, xs, ys):
    c = ORANGE if a in SELECTED else BLUE
    ax.scatter(x, y, s=60, color=c, edgecolor="white", lw=1.5, zorder=3)
    off = (6, -12) if a == "research_focus_scale" else (6, -3)
    ax.annotate(a, (x, y), xytext=off, textcoords="offset points", fontsize=8, color=INK2)
rho = spearmanr(xs, ys).correlation
ax.set_xlim(-0.03, 0.75); ax.set_ylim(-0.03, 0.75)
ax.set_xlabel("k-NN κ on private data (full n)"); ax.set_ylabel("k-NN κ on round-0 pool")
ax.set_title(f"Blindness ranking transfers (Spearman ρ = {rho:.2f})", loc="left", color=INK, fontsize=11)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="selected (top-3 blind)"),
                   plt.Line2D([], [], marker="o", ls="", color=BLUE, label="other"),
                   plt.Line2D([], [], color=GRAY, ls="--", label="y = x")], frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_transfer_private_to_pool.png"); plt.close(fig)

# ---------------- Experiment 1b, probe labels ----------------
corr = {  # correction at sigma = 0, 2.58, 4.75, 6.87
    "disease_mention":       [0.51, 0.49, 0.36, 0.28],
    "primary_research_area": [0.39, 0.33, 0.27, 0.16],
    "model_organism":        [0.33, 0.26, 0.21, 0.15],
    "experimental_approach": [0.18, 0.20, 0.20, 0.18],
    "sample_size":           [0.13, 0.15, 0.14, 0.11],
    "dominant_data_type":    [0.07, 0.06, 0.06, 0.04],
    "research_focus_scale":  [0.04, 0.05, 0.04, 0.00],
    "research_goal":         [-0.35, -0.48, -0.16, -0.23],
}
requested = {"disease_mention": -0.02, "primary_research_area": -0.50, "model_organism": -2.59,
             "experimental_approach": -0.50, "research_goal": 0.02}

# Fig 4: private k-NN κ vs correction (σ = 0)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
for a, c in corr.items():
    x = priv[a][2][0]; y = c[0]
    col = GRAY if a == "research_goal" else (ORANGE if a in SELECTED else BLUE)
    ax.scatter(x, y, s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
    ax.annotate(a, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK2)
ax.axhline(0, color=INK2, lw=0.8)
keep = [a for a in corr if a != "research_goal"]
r7 = spearmanr([priv[a][2][0] for a in keep], [corr[a][0] for a in keep]).correlation
ax.set_xlim(0.25, 0.68)
ax.set_xlabel("k-NN κ on private data (full n)"); ax.set_ylabel("vote correction, σ = 0 (probe labels)")
ax.set_title(f"More visible → more corrected (ρ = {r7:.2f} without research_goal)", loc="left", color=INK, fontsize=11)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="selected (top-3 blind)"),
                   plt.Line2D([], [], marker="o", ls="", color=BLUE, label="other"),
                   plt.Line2D([], [], marker="o", ls="", color=GRAY, label="outlier (made worse)")],
          frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_visibility_vs_correction.png"); plt.close(fig)

# Fig 5: correction vs noise, small multiples
sig = [0, 2.58, 4.75, 6.87]; eps_lab = ["∞", "4", "2", "1"]
order = list(corr)
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
for ax, a in zip(axes.flat, order):
    col = ORANGE if a in SELECTED else (GRAY if a == "research_goal" else BLUE)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(sig, corr[a], marker="o", ms=4, lw=2, color=col)
    ax.set_title(a, fontsize=9, color=INK, loc="left")
    ax.set_xticks(sig); ax.set_xticklabels(eps_lab)
for ax in axes[1]: ax.set_xlabel("ε (σ = 0, 2.58, 4.75, 6.87)")
for ax in axes[:, 0]: ax.set_ylabel("correction")
fig.suptitle("Noise erodes the correction for visible attributes; blind ones stay near 0 (probe labels)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_correction_vs_noise.png"); plt.close(fig)

# Fig 6: requested vs probe labels (σ = 0)
names = list(requested)
fig, ax = plt.subplots(figsize=(7, 3.6))
import numpy as np
y = np.arange(len(names))[::-1]; w = 0.36
ax.barh(y + w/2, [requested[a] for a in names], height=w, color=GRAY, label="requested (AIM) labels")
ax.barh(y - w/2, [corr[a][0] for a in names], height=w, color=BLUE, label="probe-predicted labels")
for yy, a in zip(y, names):
    ax.text(requested[a] - 0.05 if requested[a] < 0 else requested[a] + 0.05, yy + w/2, f"{requested[a]:+.2f}",
            va="center", ha="right" if requested[a] < 0 else "left", fontsize=8, color=INK2)
    ax.text(corr[a][0] + 0.05 if corr[a][0] >= 0 else corr[a][0] - 0.05, yy - w/2, f"{corr[a][0]:+.2f}",
            va="center", ha="left" if corr[a][0] >= 0 else "right", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis="y", visible=False)
ax.set_xlim(-3.0, 0.9); ax.set_xlabel("vote correction, σ = 0")
ax.set_title("“Vote makes content worse” was mostly a requested-label artefact", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_requested_vs_probe_labels.png"); plt.close(fig)
print("ok", sorted(os.listdir(OUT)))
