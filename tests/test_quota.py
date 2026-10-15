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


if __name__ == "__main__":
    test_quota()
