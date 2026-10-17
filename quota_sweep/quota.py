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


# ---------------- raking: quotas from one-way marginals of B (no merged cells) ----------------
def build_marginals(target_labels, attrs):
    """P_hat_k(v) for every blind attribute k, from the AIM records (post-processing)."""
    out = {}
    for a in attrs:
        c = Counter(np.asarray(target_labels[a]).astype(str))
        n = sum(c.values())
        out[a] = {v: k / n for v, k in c.items()}
    return out


def _marginal(w, cell_vals, j, values):
    m = dict.fromkeys(values, 0.0)
    for c, wc in w.items():
        m[cell_vals[c][j]] = m.get(cell_vals[c][j], 0.0) + wc
    return m


def rake(R, Q, cell_vals, attrs, p_marg, lam, cap=None, iters=300, eps=0.05, tol=1e-6):
    """Joint-cell weights whose one-way marginals are the lambda-mix of the AIM target and
    MAPLE's top-K marginals: m_k(v) ~ P_hat_k(v)^lam * R_k(v)^(1-lam).
    Start from R (MAPLE top-K composition) smoothed with eps * Q (all candidates), so cells
    MAPLE never picked can receive quota; iterative proportional fitting over the attributes.
    cap[c] (fraction of K) = how many candidates cell c actually has: a cell that hits its cap
    is frozen there and the rest of its value's target is spread over the other cells, so the
    quotas are feasible and the marginals stay as close to the targets as the supply allows."""
    cells = list(cell_vals)
    cap = cap or {}
    w = {c: R.get(c, 0.0) + eps * Q.get(c, 0.0) for c in cells}
    z = sum(w.values())
    w = {c: v / z for c, v in w.items()}
    targets = []
    for j, a in enumerate(attrs):
        values = sorted(set(p_marg[a]) | set(cv[j] for cv in cell_vals.values()))
        Rk = _marginal({c: R.get(c, 0.0) for c in cells}, cell_vals, j, values)
        m = {v: (p_marg[a].get(v, 0.0) ** lam) * (Rk.get(v, 0.0) ** (1.0 - lam)) for v in values}
        s = sum(m.values())
        targets.append({v: x / s for v, x in m.items()} if s > 0 else Rk)
    frozen = set()
    for _ in range(iters):
        delta = 0.0
        for j in range(len(attrs)):
            tg = targets[j]
            cur = _marginal(w, cell_vals, j, tg)
            fixed = _marginal({c: w[c] for c in frozen}, cell_vals, j, tg)
            free = {v: cur[v] - fixed[v] for v in tg}
            f = {v: (max(tg[v] - fixed[v], 0.0) / free[v] if free[v] > 1e-15 else 0.0) for v in tg}
            for c in cells:
                if c in frozen:
                    continue
                w[c] *= f.get(cell_vals[c][j], 0.0)
                if c in cap and w[c] > cap[c]:
                    w[c] = cap[c]
                    frozen.add(c)
            delta = max(delta, max(abs(cur[v] - tg[v]) for v in tg))
        if delta < tol:
            break
    return w, targets


def select_raked(scores, labels, attrs, p_marg, lam, K, eps=0.05):
    """Stratified top-K with raked quotas. scores: noisy vote; labels: {attr: array} of the
    candidates (in-loop labels). lam = 0 is exactly MAPLE. Returns (selected indices, info)."""
    scores = np.asarray(scores, float)
    order = np.argsort(scores)[::-1]
    base = order[:K]
    if lam <= 0:
        return base, dict(shortfall=0, moved=0, quota={}, targets={})
    keys = cell_keys(labels, attrs)
    cell_vals = {c: c.split("|") for c in set(keys)}
    R = {c: n / K for c, n in Counter(keys[base]).items()}
    Q = {c: n / len(keys) for c, n in Counter(keys).items()}
    avail = Counter(keys)
    cap = {c: avail[c] / K for c in cell_vals}
    w, targets = rake(R, Q, cell_vals, attrs, p_marg, lam, cap=cap, eps=eps)
    q = largest_remainder(w, K)
    q = {c: min(k, avail[c]) for c, k in q.items()}          # rounding can still overshoot by 1
    chosen = np.zeros(len(scores), bool)
    short = 0
    for c, k in q.items():
        if k == 0:
            continue
        idx = order[keys[order] == c][:k]
        chosen[idx] = True
        short += k - len(idx)
    if short > 0:
        chosen[order[~chosen[order]][:short]] = True
    sel = order[chosen[order]]
    info = dict(quota={c: k for c, k in q.items() if k}, shortfall=int(short),
                capped_cells=int(sum(1 for c in cell_vals if q.get(c, 0) >= avail[c] and avail[c] > 0)),
                moved=int(K - len(np.intersect1d(sel, base))),
                targets={a: targets[j] for j, a in enumerate(attrs)})
    return sel, info


def marginal_jsd(labels, attrs, p_marg):
    """Mean over attributes of JSD(label marginal, P_hat_k)."""
    from scipy.spatial.distance import jensenshannon
    out = {}
    for a in attrs:
        c = Counter(np.asarray(labels[a]).astype(str))
        n = sum(c.values())
        values = sorted(set(c) | set(p_marg[a]))
        p = np.array([c.get(v, 0) / n for v in values])
        qv = np.array([p_marg[a].get(v, 0.0) for v in values])
        out[a] = float(jensenshannon(p, qv))
    return out
