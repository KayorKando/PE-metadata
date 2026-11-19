"""Pick the quota's blind attributes on the round-0 pool: post-processing of DP output (AIM metadata + public
in-context examples), no private text.

Rule (as on bioRxiv, Experiment 1): the --top attributes with the lowest k-NN kappa under the t5 vote embedding
(sentence-t5-base, max_seq_length 1024); k = 5; same-label share among the k neighbours (no majority vote),
chance-corrected; random half/half splits, mean over --repeats. Labels: the requested (AIM) metadata of each
pool row (inherited), word_count measured from the text (as the quota's loop labels). Candidates: the AIM
attributes the quota has a target for, minus the vote's classes (pe_data.BLIND_CANDIDATES).
Class-conditional vote (OpenReview): neighbours are searched only inside the query's own PE.LABEL_ID, as the
vote does, and the chance level is the within-class one: sum_c w_c sum_v p_{c,v}^2 (w_c = share of query
points in class c). Without classes this is blindness/metrics.knn_kappa.

    python blindness/pick_blind.py --pool_csv $PE_DATA/round0/eps4.0/pool.csv --out $PE_DATA/blind.json
Output: blind.json {"blind": [...], "kappa": {attr: mean}, "sd": {attr: sd}, settings} and <out>.csv.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.metrics import chance_agreement, knn_neighbors  # noqa: E402
from blindness.schema import clean_labels  # noqa: E402
from pe_data import BLIND_CANDIDATES, CLASS_CONDITIONAL, LABEL_ID  # noqa: E402


def knn_kappas(X, labels, groups, ref, qry, k):
    """-> {attr: kappa} for one split; neighbours of a query point only among the ref points of its group."""
    agree = {a: 0.0 for a in labels}
    chance = {a: 0.0 for a in labels}
    nq = 0
    for g in np.unique(groups[qry]):
        r, q = ref[groups[ref] == g], qry[groups[qry] == g]
        if len(r) == 0:
            continue
        nbr = knn_neighbors(X[r], X[q], min(k, len(r)))
        for a, y in labels.items():
            agree[a] += float((y[r][nbr] == y[q][:, None]).mean(1).sum())
            chance[a] += chance_agreement(np.concatenate([y[r], y[q]])) * len(q)
        nq += len(q)
    out = {}
    for a in labels:
        ag, pe = agree[a] / nq, chance[a] / nq
        out[a] = float((ag - pe) / (1 - pe)) if pe < 1 else float("nan")
    return out


def blind_table(X, labels, groups=None, k=5, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.zeros(len(X), int) if groups is None else np.asarray(groups)
    rows = []
    for r in range(repeats):
        perm = rng.permutation(len(X))
        ref, qry = perm[: len(X) // 2], perm[len(X) // 2:]
        rows += [dict(repeat=r, attribute=a, knn_kappa=v) for a, v in knn_kappas(X, labels, groups, ref, qry, k).items()]
    R = pd.DataFrame(rows)
    S = R.groupby("attribute").knn_kappa.agg(["mean", "std"]).rename(columns={"mean": "knn_kappa", "std": "sd"})
    return S.sort_values("knn_kappa")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_csv", required=True, help="round-0 pool before the vote (gen_round0.py)")
    p.add_argument("--out", required=True, help="blind.json")
    p.add_argument("--attributes", nargs="+", default=BLIND_CANDIDATES)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="sentence-t5-base")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    from blindness.embed import embed
    pool = pd.read_csv(args.pool_csv)
    pool = pool[pool["text"].notna() & (pool["text"].astype(str).str.strip() != "")].reset_index(drop=True)
    labels = clean_labels(pool, args.attributes)            # word_count: measured from the text
    groups = pool[LABEL_ID].to_numpy() if CLASS_CONDITIONAL else None
    X = embed(pool["text"].astype(str).tolist(), args.model, args.max_seq_length, batch_size=args.batch_size,
              cache_dir=str(Path(os.environ.get("PE_DATA", ".")) / "cache"))
    S = blind_table(X, labels, groups, args.k, args.repeats, args.seed)
    blind = S.index[: args.top].tolist()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    S.assign(selected=S.index.isin(blind)).to_csv(out.with_suffix(".csv"))
    out.write_text(json.dumps({"blind": blind, "kappa": S.knn_kappa.round(4).to_dict(), "sd": S.sd.round(4).to_dict(),
                               "within_class": CLASS_CONDITIONAL, **vars(args), "n_rows": len(pool)}, indent=2))
    print(S.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"blind attributes (lowest k-NN kappa, within class: {CLASS_CONDITIONAL}): {blind} -> {out}")


if __name__ == "__main__":
    main()
