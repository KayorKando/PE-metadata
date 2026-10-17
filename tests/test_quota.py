"""Quota selection: lam=0 equals MAPLE top-K; lam=1 matches the target up to rounding;
within a cell the vote order is kept; water-filling keeps |selected| = K."""
import numpy as np

from quota_sweep.quota import OTHER, assign_cells, build_cell_space, quotas, select


def test_quota():
    rng = np.random.default_rng(0)
    n, K = 5000, 500
    lab = {"a": rng.choice(list("xyz"), n, p=[.7, .2, .1]), "b": rng.choice(list("pq"), n)}
    tgt = {"a": rng.choice(list("xyz"), 4000, p=[.3, .3, .4]), "b": rng.choice(list("pq"), 4000)}
    kept, p_hat = build_cell_space(tgt, ["a", "b"], max_cells=5)
    assert len(p_hat) <= 5 and abs(sum(p_hat.values()) - 1) < 1e-9 and OTHER in p_hat
    cells = assign_cells(lab, ["a", "b"], kept)
    scores = rng.normal(size=n) + (lab["a"] == "x")     # the vote prefers x

    sel0, _ = select(scores, cells, p_hat, 0.0, K)
    assert set(sel0) == set(np.argsort(scores)[::-1][:K])           # lam=0 is MAPLE

    sel1, info = select(scores, cells, p_hat, 1.0, K)
    assert len(sel1) == K and len(set(sel1)) == K
    got = {c: np.mean(cells[sel1] == c) for c in p_hat}
    if info["shortfall"] == 0:
        assert max(abs(got[c] - p_hat[c]) for c in p_hat) <= 1.0 / K + 1e-9
    for c in p_hat:                                                  # argmax inside each cell
        in_c = np.flatnonzero(cells == c)
        top = in_c[np.argsort(scores[in_c])[::-1][: np.sum(cells[sel1] == c)]]
        assert set(top) == set(sel1[cells[sel1] == c])

    q = quotas({"u": 1.0}, {"v": 1.0}, 1.0, K)                       # target cell absent: shortfall filled
    assert sum(q.values()) == K
    sel, info = select(scores, np.array(["v"] * n), {"u": 1.0}, 1.0, K)
    assert len(sel) == K and info["shortfall"] == K
    print("ok")




def test_raking():
    from quota_sweep.quota import build_marginals, marginal_jsd, select_raked
    rng = np.random.default_rng(1)
    n, K = 6000, 600
    attrs = ["a", "b", "c"]
    lab = {"a": rng.choice(list("xyz"), n, p=[.7, .2, .1]), "b": rng.choice(list("pq"), n, p=[.8, .2]),
           "c": rng.choice(list("12345"), n)}
    tgt = {"a": rng.choice(list("xyz"), 5000, p=[.3, .3, .4]), "b": rng.choice(list("pq"), 5000, p=[.5, .5]),
           "c": rng.choice(list("12345"), 5000)}
    p_marg = build_marginals(tgt, attrs)
    scores = rng.normal(size=n) + (lab["a"] == "x") + 0.5 * (lab["b"] == "p")   # the vote prefers x, p
    sel0, _ = select_raked(scores, lab, attrs, p_marg, 0.0, K)
    assert set(sel0) == set(np.argsort(scores)[::-1][:K])                       # lam=0 is MAPLE
    sel1, info = select_raked(scores, lab, attrs, p_marg, 1.0, K)
    assert len(sel1) == K == len(set(sel1))
    before = marginal_jsd({a: lab[a][sel0] for a in attrs}, attrs, p_marg)
    after = marginal_jsd({a: lab[a][sel1] for a in attrs}, attrs, p_marg)
    assert after["a"] < 0.03 and after["b"] < 0.03 and before["a"] > 0.2, (before, after)
    for c, k in info["quota"].items():                                           # argmax inside each cell
        vals = c.split("|")
        in_c = np.flatnonzero((lab["a"] == vals[0]) & (lab["b"] == vals[1]) & (lab["c"] == vals[2]))
        picked = np.intersect1d(sel1, in_c)
        top = in_c[np.argsort(scores[in_c])[::-1][: len(picked)]]
        assert set(top) == set(picked)
    sel5, _ = select_raked(scores, lab, attrs, p_marg, 0.5, K)
    mid = marginal_jsd({a: lab[a][sel5] for a in attrs}, attrs, p_marg)
    assert after["a"] < mid["a"] < before["a"]
    print("raking ok", {k: round(v, 3) for k, v in before.items()}, "->", {k: round(v, 3) for k, v in after.items()})


if __name__ == "__main__":
    test_quota()
    test_raking()
