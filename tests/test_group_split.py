"""Twins (same prompt -> near-duplicate embeddings) inflate k-NN kappa under a random
split; a group split removes the inflation. The probe is inflated too, less."""
import numpy as np

from blindness.experiment import run, split_halves, summarize


def make_twins(n_groups=400, per_group=4, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n_groups)              # one label per group (as a requested attribute)
    centers = rng.normal(size=(n_groups, d))      # group-specific nuisance direction, unrelated to y
    centers[:, 0] += 0.8 * (y - 1.5)              # modest label signal
    X = np.repeat(centers, per_group, 0) + 0.2 * rng.normal(size=(n_groups * per_group, d))
    groups = np.repeat(np.arange(n_groups), per_group).astype(str)
    return X, {"attr": np.repeat(y, per_group).astype(str)}, groups


def test_split_halves_keeps_groups_together():
    _, _, groups = make_twins()
    ref, qry = split_halves(len(groups), np.random.default_rng(0), groups)
    assert not set(groups[ref]) & set(groups[qry])
    assert abs(len(ref) - len(qry)) <= 4


def test_group_split_removes_twin_inflation():
    X, labels, groups = make_twins()
    rand = summarize(run(X, labels, repeats=3, groups=groups), labels)[0]
    grp = summarize(run(X, labels, repeats=3, groups=groups, group_split=True), labels)[0]
    rand_pr = run(X, labels, repeats=3, groups=groups)
    assert rand_pr.twin_rate.mean() > 0.8
    d_knn = rand.loc["attr", "knn_kappa"] - grp.loc["attr", "knn_kappa"]
    d_probe = rand.loc["attr", "probe_kappa"] - grp.loc["attr", "probe_kappa"]
    assert d_knn > 0.2                 # twins inflate k-NN under a random split
    assert 0 <= d_probe < d_knn        # the probe is inflated too (it can memorize group directions), but less


if __name__ == "__main__":
    test_split_halves_keeps_groups_together()
    test_group_split_removes_twin_inflation()
    print("ok")
