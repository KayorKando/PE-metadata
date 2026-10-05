"""Experiment 1: k-NN blindness vs linear-probe accuracy under the vote embedding phi.

Example:
    python run.py --csv ../MAPLE/biorxiv/biorxiv_train_metadata.csv --n 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blindness.embed import DEFAULT_MODEL, embed
from blindness.experiment import plot, run, summarize
from blindness.schema import BIORXIV_ATTRIBUTES, clean_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with a 'text' column and metadata columns")
    p.add_argument("--attributes", nargs="+", default=BIORXIV_ATTRIBUTES)
    p.add_argument("--n", type=int, default=2000, help="subsample size (0 = all rows)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--top", type=int, default=3, help="how many blindest attributes to select")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["text"].notna()].reset_index(drop=True)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    labels = clean_labels(df, args.attributes)
    X = embed(df["text"].astype(str).tolist(), args.model, args.max_seq_length)

    per_repeat = run(X, labels, repeats=args.repeats, k=args.k, seed=args.seed)
    summary, rho, pval = summarize(per_repeat, labels, top=args.top)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(out / "per_repeat.csv", index=False)
    summary.to_csv(out / "summary.csv")
    plot(summary, rho, out / "knn_vs_probe.png")
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "n_rows_used": len(df), "spearman_rho": rho, "spearman_p": pval}, indent=2))

    pd.set_option("display.width", 160)
    print(summary.round(3).to_string())
    print(f"\nSpearman rho (knn vs probe kappa) = {rho:.3f}  (p = {pval:.3g})")
    print("Selected (most k-NN-blind):", list(summary.index[summary.selected]))


if __name__ == "__main__":
    main()
