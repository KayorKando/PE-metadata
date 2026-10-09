"""Experiment 1: k-NN blindness vs linear-probe accuracy under the vote embedding phi.

Example:
    python run.py --csv ../MAPLE/biorxiv/biorxiv_train_metadata.csv --n 2000

Large, regenerable files (embedding cache, results) go under $PE_DATA
(default: current directory). On the cluster set PE_DATA=/data/<user>/PE-metadata.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from blindness.embed import DEFAULT_MODEL, embed
from blindness.experiment import plot, run, summarize
from blindness.schema import BIORXIV_ATTRIBUTES, clean_labels

# The 9 requested (AIM) metadata columns of a round-0 pool: rows that share all of them
# came from the same metadata row, i.e. the same prompt.
REQUESTED_COLS = [a for a in BIORXIV_ATTRIBUTES if a != "word_count"] + ["word_count_requested"]


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
    data_root = Path(os.environ.get("PE_DATA", "."))
    p.add_argument("--out", default=str(data_root / "results"))
    p.add_argument("--cache_dir", default=str(data_root / "cache"))
    p.add_argument("--batch_size", type=int, default=32, help="embedding batch size")
    p.add_argument("--group_cols", nargs="*", default=None,
                   help="columns whose identical values define a group (e.g. the requested metadata "
                        "of a synthetic pool); 'requested' = the 9 AIM columns. Reports twin_rate.")
    p.add_argument("--group_split", action="store_true",
                   help="keep every group on one side of the split (needs --group_cols)")
    p.add_argument("--exclude_from_rho", nargs="*", default=[],
                   help="attributes left out of the Spearman rho, e.g. word_count word_count_requested")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["text"].notna()].reset_index(drop=True)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    labels = clean_labels(df, args.attributes)
    X = embed(df["text"].astype(str).tolist(), args.model, args.max_seq_length,
              batch_size=args.batch_size, cache_dir=args.cache_dir)

    groups = None
    if args.group_cols:
        cols = REQUESTED_COLS if args.group_cols == ["requested"] else args.group_cols
        groups = df[cols].astype(str).agg("|".join, axis=1).to_numpy()
        sizes = pd.Series(groups).value_counts()
        print(f"groups: {len(sizes)} distinct over {len(df)} rows; "
              f"{int((sizes > 1).sum())} groups have >1 row (max {sizes.max()}); "
              f"group_split={args.group_split}")
    elif args.group_split:
        p.error("--group_split needs --group_cols")

    per_repeat = run(X, labels, repeats=args.repeats, k=args.k, seed=args.seed,
                     groups=groups, group_split=args.group_split)
    summary, rho, pval = summarize(per_repeat, labels, top=args.top, exclude_from_rho=args.exclude_from_rho)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(out / "per_repeat.csv", index=False)
    summary.to_csv(out / "summary.csv")
    plot(summary, rho, out / "knn_vs_probe.png")
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "n_rows_used": len(df), "spearman_rho": rho, "spearman_p": pval}, indent=2))

    pd.set_option("display.width", 160)
    print(summary.round(3).to_string())
    if "twin_rate" in per_repeat:
        tr = per_repeat.groupby("repeat").twin_rate.first()
        print(f"\ntwin_rate (1-NN from the same group) = {tr.mean():.3f} +- {tr.std():.3f}")
    excl = f"  (excluding {args.exclude_from_rho})" if args.exclude_from_rho else ""
    print(f"\nSpearman rho (knn vs probe kappa) = {rho:.3f}  (p = {pval:.3g}){excl}")
    print("Selected (most k-NN-blind):", list(summary.index[summary.selected]))


if __name__ == "__main__":
    main()
