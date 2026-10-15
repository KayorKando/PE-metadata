"""Stratified top-K with metadata quotas (pure numpy, no GPU).

Cells live on the blind attributes B. Each round:
  1. baseline = MAPLE's top-K by the (noisy) vote; R_t(c) = its composition over cells
  2. quotas  K_c = round_LR( K * P_hat(c)^lam * R_t(c)^(1-lam) / Z )      (largest remainder)
  3. in each cell keep the K_c highest-vote candidates (argmax, never reorders within a cell)
  4. water-filling: slots a cell cannot fill go to the best remaining candidates overall
lam = 0 reproduces MAPLE; lam = 1 pins the survivors' cell distribution to P_hat.
"""
from collections import Counter

import numpy as np

OTHER = "__other__"


def cell_keys(labels, attrs):
    """labels: {attr: array of str} -> array of joint keys 'v1|v2|...'."""
    return np.array(["|".join(t) for t in zip(*[np.asarray(labels[a]).astype(str) for a in attrs])])


def build_cell_space(target_labels, attrs, max_cells=30):
    """Target P_hat over joint cells of B (e.g. from AIM records: post-processing).
    Keep the max_cells-1 most frequent cells, merge the rest into OTHER.
    Returns (kept set, P_hat dict over kept cells + OTHER)."""
    keys = cell_keys(target_labels, attrs)
    cnt = Counter(keys)
    ranked = [k for k, _ in cnt.most_common()]
    kept = set(ranked if len(ranked) <= max_cells else ranked[: max_cells - 1])
    p = {k: cnt[k] / len(keys) for k in kept}
    rest = 1.0 - sum(p.values())
    if rest > 1e-12:
        p[OTHER] = rest
    return kept, p


def assign_cells(labels, attrs, kept):
    return np.array([k if k in kept else OTHER for k in cell_keys(labels, attrs)])


def largest_remainder(weights, K):
    """weights: {cell: w >= 0} -> {cell: int}, summing to K (if any weight > 0)."""
    cells = list(weights)
    w = np.array([weights[c] for c in cells], float)
    if w.sum() <= 0:
        return {c: 0 for c in cells}
    x = K * w / w.sum()
    q = np.floor(x).astype(int)
    left = K - q.sum()
    q[np.argsort(-(x - q), kind="stable")[:left]] += 1
    return dict(zip(cells, q.tolist()))


def quotas(p_hat, R, lam, K):
    cells = sorted(set(p_hat) | set(R))
    w = {c: (p_hat.get(c, 0.0) ** lam) * (R.get(c, 0.0) ** (1.0 - lam)) for c in cells}
    return largest_remainder(w, K)


def select(scores, cells, p_hat, lam, K):
    """scores: noisy vote per candidate; cells: cell id per candidate.
    Returns (selected indices, info)."""
    scores = np.asarray(scores, float)
    order = np.argsort(scores)[::-1]          # as MAPLE get_next_iteration(mode='rank')
    base = order[:K]
    R = {c: n / K for c, n in Counter(cells[base]).items()}
    q = quotas(p_hat, R, lam, K)
    chosen = np.zeros(len(scores), bool)
    short = 0
    for c, k in q.items():
        idx = order[cells[order] == c][:k]    # best k of cell c, vote order
        chosen[idx] = True
        short += k - len(idx)
    if short > 0:                              # water-filling
        rest = order[~chosen[order]][:short]
        chosen[rest] = True
    sel = order[chosen[order]]                 # keep vote order
    info = dict(R=R, quota=q, shortfall=int(short),
                moved=int(K - len(np.intersect1d(sel, base))))  # how many survivors differ from MAPLE
    return sel, info
