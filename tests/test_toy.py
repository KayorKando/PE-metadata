"""Toy check with known answers.

phi = 64-dim Gaussian. Three attributes:
  loud   : shifts a high-variance direction  -> k-NN sees it, probe sees it
  quiet  : shifts one low-variance direction -> probe sees it, k-NN mostly does not
  absent : independent of X                  -> neither sees it (kappa ~ 0)
"""
import numpy as np

from blindness.experiment import run, summarize


def make_toy(n=2000, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_loud = rng.integers(0, 3, n)
    y_quiet = rng.integers(0, 2, n)
    y_absent = rng.integers(0, 4, n)
    X[:, :8] += 2.0 * (y_loud[:, None] - 1)          # strong, spread over 8 dims
    X[:, 63] = 0.05 * rng.normal(size=n) + 0.3 * y_quiet  # tiny-scale but clean signal
    labels = {"loud": y_loud.astype(str), "quiet": y_quiet.astype(str), "absent": y_absent.astype(str)}
    return X, labels


def test_toy():
    X, labels = make_toy()
    s, rho, _ = summarize(run(X, labels, repeats=5, k=5), labels, top=1)
    print(s.round(3))
    assert s.loc["loud", "knn_kappa"] > 0.8 and s.loc["loud", "probe_kappa"] > 0.8
    assert s.loc["quiet", "probe_kappa"] > 0.8 and s.loc["quiet", "knn_kappa"] < 0.2
    assert abs(s.loc["absent", "knn_kappa"]) < 0.05 and abs(s.loc["absent", "probe_kappa"]) < 0.05
    assert s.index[s.selected].tolist() == ["absent"]


if __name__ == "__main__":
    test_toy()
    print("ok")
