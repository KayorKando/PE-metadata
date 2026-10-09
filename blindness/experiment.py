"""Paired comparison of k-NN visibility vs linear-probe visibility.

Each repeat: shuffle, split into two disjoint halves (ref, query).
  k-NN : query points look up their k nearest ref points.
  probe: train on ref, test on query.
Both metrics use the same split, so their difference is paired.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import knn_kappa, knn_neighbors, probe_kappa


def split_halves(n, rng, groups=None):
    """Random half/half split. With `groups`, every row of a group lands on the
    same half (group split), so a query point can never find its own twin."""
    if groups is None:
        perm = rng.permutation(n)
        return perm[: n // 2], perm[n // 2:]
    _, inv = np.unique(groups, return_inverse=True)
    order = rng.permutation(inv.max() + 1)
    cut = np.searchsorted(np.cumsum(np.bincount(inv)[order]), n // 2, side="right")
    in_ref = np.zeros(len(order), bool)
    in_ref[order[:cut]] = True
    mask = in_ref[inv]
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def twin_rate(nbr_idx, groups, ref, qry):
    """Share of query points whose 1-NN in ref comes from the same group
    (e.g. generated from the same metadata row). 0 by construction under group split."""
    return float((groups[ref][nbr_idx[:, 0]] == groups[qry]).mean())


def run(X, labels, repeats=20, k=5, seed=0, probe_C=1.0, groups=None, group_split=False):
    """groups: optional per-row group id (same metadata row / same prompt).
    group_split=False keeps the random split but reports `twin_rate`;
    group_split=True keeps each group on one side of the split."""
    rng = np.random.default_rng(seed)
    n = len(X)
    rows = []
    for r in range(repeats):
        ref, qry = split_halves(n, rng, groups if group_split else None)
        nbr = knn_neighbors(X[ref], X[qry], k)  # shared by all attributes
        extra = {} if groups is None else {"twin_rate": twin_rate(nbr, groups, ref, qry)}
        for a, y in labels.items():
            kk, kraw = knn_kappa(nbr, y[ref], y[qry])
            pk, praw = probe_kappa(X[ref], y[ref], X[qry], y[qry], C=probe_C, seed=r)
            rows.append(dict(repeat=r, attribute=a, knn_kappa=kk, knn_agree=kraw,
                             probe_kappa=pk, probe_acc=praw, **extra))
    return pd.DataFrame(rows)


def summarize(per_repeat, labels, top=3, exclude_from_rho=()):
    """exclude_from_rho: attributes left out of the k-NN-vs-probe rank correlation
    (e.g. the two length rows, which would otherwise be counted twice)."""
    g = per_repeat.groupby("attribute")
    s = pd.DataFrame({
        "n_classes": {a: len(np.unique(y)) for a, y in labels.items()},
        "knn_kappa": g.knn_kappa.mean(), "knn_sd": g.knn_kappa.std(),
        "probe_kappa": g.probe_kappa.mean(), "probe_sd": g.probe_kappa.std(),
    })
    s["gap"] = s.probe_kappa - s.knn_kappa  # encoded but not used by the vote
    s["knn_blindness"] = 1 - s.knn_kappa
    s["probe_blindness"] = 1 - s.probe_kappa
    s = s.sort_values("knn_blindness", ascending=False)
    s["selected"] = False
    s.iloc[:top, s.columns.get_loc("selected")] = True
    keep = ~s.index.isin(list(exclude_from_rho))
    rho, p = spearmanr(s.knn_kappa[keep], s.probe_kappa[keep])
    return s, float(rho), float(p)


def plot(summary, rho, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [min(0, summary[["knn_kappa", "probe_kappa"]].min().min() - 0.05), 1]
    ax.plot(lim, lim, color="#999", lw=1, ls="--", zorder=0)
    sel = summary.selected
    ax.errorbar(summary.knn_kappa, summary.probe_kappa, xerr=summary.knn_sd, yerr=summary.probe_sd,
                fmt="none", ecolor="#bbb", lw=1, zorder=1)
    ax.scatter(summary.knn_kappa[~sel], summary.probe_kappa[~sel], color="#4a6fa5", zorder=2,
               label="other attributes")
    ax.scatter(summary.knn_kappa[sel], summary.probe_kappa[sel], color="#c8553d", zorder=2,
               label="selected (most k-NN-blind)")
    for a, r in summary.iterrows():
        ax.annotate(a, (r.knn_kappa, r.probe_kappa), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("k-NN visibility (kappa)  - what the vote can use")
    ax.set_ylabel("linear probe visibility (kappa)  - what phi encodes")
    ax.set_title(f"Attribute visibility under phi   (Spearman rho = {rho:.2f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
