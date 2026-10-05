"""Two ways to ask "does phi see attribute a?".

Both are reported on the same chance-corrected scale (kappa):
0 = no better than chance, 1 = perfect.  blindness = 1 - kappa.

* knn_kappa:   what PE's vote can use. For each query point, the share of its
               k nearest reference points (L2, like MAPLE's vote) with the same
               label; chance level is sum_c p_c^2.
* probe_kappa: what phi encodes at all. A linear probe trained on the
               reference half, scored on the query half with Cohen's kappa.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chance_agreement(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float((p ** 2).sum())


def knn_neighbors(X_ref, X_query, k):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_ref)
    return nn.kneighbors(X_query, return_distance=False)


def knn_kappa(nbr_idx, y_ref, y_query):
    """Same-label proportion among k neighbors, chance-corrected.

    No majority vote, so no ties. Chance uses the pooled label distribution.
    """
    agree = (y_ref[nbr_idx] == y_query[:, None]).mean()
    pe = chance_agreement(np.concatenate([y_ref, y_query]))
    if pe >= 1.0:
        return np.nan, float(agree)
    return float((agree - pe) / (1 - pe)), float(agree)


def probe_kappa(X_tr, y_tr, X_te, y_te, C=1.0, seed=0):
    if len(np.unique(y_tr)) < 2:
        return np.nan, np.nan
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, random_state=seed),
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return float(cohen_kappa_score(y_te, pred)), float((pred == y_te).mean())
