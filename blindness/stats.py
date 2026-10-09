"""How much to trust a Spearman rho computed over ~8 attributes.

With n points the Fisher-z standard error is 1/sqrt(n-3): for n = 8 that is
0.45, so two rhos like 0.95 and 0.64 are usually not distinguishable. These
helpers quantify that, and `leave_one_out` shows which attributes drive rho.
"""
from itertools import permutations

import numpy as np
from scipy.stats import norm, rankdata, spearmanr


def fisher_ci(rho, n, alpha=0.05):
    if n < 4 or abs(rho) >= 1:
        return float("nan"), float("nan")
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    h = norm.ppf(1 - alpha / 2) * se
    return float(np.tanh(z - h)), float(np.tanh(z + h))


def fisher_diff_test(r1, n1, r2, n2):
    """Two-sided p-value for H0: the two (independent) correlations are equal."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(2 * (1 - norm.cdf(abs(z))))


def perm_pvalue(x, y, n_perm=20000, seed=0):
    """Permutation p-value for rho (exact enumeration when n <= 9)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = spearmanr(x, y).correlation
    rx = rankdata(x)
    if len(x) <= 9:
        null = np.array([np.corrcoef(rx, rankdata(p))[0, 1] for p in permutations(y)])
    else:
        rng = np.random.default_rng(seed)
        null = np.array([np.corrcoef(rx, rankdata(rng.permutation(y)))[0, 1] for _ in range(n_perm)])
    return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean())


def boot_ci(x, y, n_boot=5000, seed=0, alpha=0.05):
    """Bootstrap over attributes (resample the points). Wide on purpose: it
    answers "would a different set of 8 attributes give the same rho?"."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            out.append(spearmanr(x[i], y[i]).correlation)
    return tuple(float(v) for v in np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def leave_one_out(x, y, names):
    """rho after dropping each attribute: a big jump means that attribute drives rho."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return {nm: float(spearmanr(np.delete(x, i), np.delete(y, i)).correlation) for i, nm in enumerate(names)}
