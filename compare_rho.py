"""Is rho(k-NN, probe) really different between two runs (e.g. private vs round-0 pool)?

Reads the summary.csv of two run.py results, drops the length rows, and prints:
  * each rho with a Fisher-z CI, a permutation p-value and a bootstrap-over-attributes CI
  * a Fisher-z test for rho_A == rho_B
  * leave-one-out rho and the rank table, to see WHICH attributes move.

Example:
    python compare_rho.py --a $PE_DATA/results/n0/summary.csv --b $PE_DATA/round0/eps4.0/results/summary.csv \
        --label_a private --label_b pool --exclude word_count word_count_requested
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from blindness.stats import boot_ci, fisher_ci, fisher_diff_test, leave_one_out, perm_pvalue


def load(path, exclude):
    s = pd.read_csv(path, index_col=0)
    return s[~s.index.isin(exclude)][["knn_kappa", "probe_kappa"]]


def report(label, s):
    x, y = s.knn_kappa.to_numpy(), s.probe_kappa.to_numpy()
    rho, p_perm = perm_pvalue(x, y)
    lo, hi = fisher_ci(rho, len(x))
    blo, bhi = boot_ci(x, y)
    print(f"{label}: rho = {rho:.3f} over {len(x)} attributes | Fisher 95% CI [{lo:.2f}, {hi:.2f}] "
          f"| permutation p = {p_perm:.3f} | bootstrap CI [{blo:.2f}, {bhi:.2f}]")
    loo = leave_one_out(x, y, s.index)
    print("  leave-one-out rho: " + ", ".join(f"{k} -> {v:.2f}" for k, v in sorted(loo.items(), key=lambda kv: -abs(kv[1] - rho))))
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A"); ap.add_argument("--label_b", default="B")
    ap.add_argument("--exclude", nargs="*", default=["word_count", "word_count_requested"])
    args = ap.parse_args()

    A, B = load(args.a, args.exclude), load(args.b, args.exclude)
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    print(f"{len(common)} shared attributes (excluded: {args.exclude})\n")
    ra, rb = report(args.label_a, A), report(args.label_b, B)
    print(f"\nFisher-z test rho_{args.label_a} == rho_{args.label_b}: p = {fisher_diff_test(ra, len(A), rb, len(B)):.3f}")

    # rank table: 1 = lowest kappa (most blind)
    t = pd.DataFrame({
        f"knn_rank_{args.label_a}": rankdata(A.knn_kappa), f"probe_rank_{args.label_a}": rankdata(A.probe_kappa),
        f"knn_rank_{args.label_b}": rankdata(B.knn_kappa), f"probe_rank_{args.label_b}": rankdata(B.probe_kappa),
    }, index=common).astype(int)
    t[f"|d|_{args.label_a}"] = (t.iloc[:, 0] - t.iloc[:, 1]).abs()
    t[f"|d|_{args.label_b}"] = (t.iloc[:, 2] - t.iloc[:, 3]).abs()
    print("\nranks (1 = most k-NN-blind / least probe-visible); |d| = k-NN rank minus probe rank:")
    print(t.sort_values(t.columns[-1], ascending=False).to_string())


if __name__ == "__main__":
    main()
