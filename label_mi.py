"""Are the attributes less correlated with each other in the synthetic pool than in
the private data?  (If yes, a high rho on private data partly reflects one shared
"topic" axis, and the lower rho on the pool is the more honest number.)

Computes pairwise normalized mutual information between attribute labels, for the
private CSV (Gemini labels, measured length) and the pool CSV (requested AIM labels).

Example:
    python label_mi.py --private $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
        --pool $PE_DATA/round0/eps4.0/pool.csv --out $PE_DATA/results/label_mi
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from blindness.schema import BIORXIV_SCHEMA, clean_labels

SCHEMA_ATTRS = list(BIORXIV_SCHEMA)


def labels_of(df, length_col):
    lab = clean_labels(df, SCHEMA_ATTRS)
    if length_col == "word_count":           # measured, MAPLE binning
        lab["word_count"] = clean_labels(df, ["word_count"])["word_count"]
    else:                                     # requested AIM bin, as given
        lab["word_count"] = df[length_col].astype(str).to_numpy()
    return lab


def nmi_matrix(lab):
    names = list(lab)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = normalized_mutual_info_score(lab[a], lab[b], average_method="arithmetic")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pool_length", default="word_count_requested",
                    help="pool length column: word_count_requested (AIM label) or word_count (measured)")
    ap.add_argument("--out", default="results/label_mi")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mats = {}
    for name, path, length_col in [("private", args.private, "word_count"), ("pool", args.pool, args.pool_length)]:
        df = pd.read_csv(path)
        df = df[df["text"].notna()].reset_index(drop=True)
        mats[name] = nmi_matrix(labels_of(df, length_col))
        mats[name].to_csv(out / f"nmi_{name}.csv")

    off = lambda m: m.where(~np.eye(len(m), dtype=bool))  # noqa: E731
    per_attr = pd.DataFrame({k: off(m).mean(axis=1) for k, m in mats.items()})
    per_attr["ratio"] = per_attr.pool / per_attr.private
    pd.set_option("display.width", 160)
    print("mean NMI with the other attributes:")
    print(per_attr.round(3).sort_values("ratio").to_string())
    print(f"\noverall mean off-diagonal NMI: private = {off(mats['private']).stack().mean():.3f}, "
          f"pool = {off(mats['pool']).stack().mean():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = max(off(m).max().max() for m in mats.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (k, m) in zip(axes, mats.items()):
        im = ax.imshow(off(m).fillna(np.nan), vmin=0, vmax=vmax, cmap="Blues")
        ax.set_xticks(range(len(m))); ax.set_xticklabels(m.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(f"{k}: pairwise NMI of attribute labels")
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(out / "nmi.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
