"""Pick the best classifier per attribute from the re-annotating results -> best_annotator.json.

"Best" = highest Cohen's kappa vs the Qwen labels on --eval_set (default pool_test_vs_qwen:
held-out synthetic texts, the closest to what the PE loop labels), among models whose
weights were saved. Ties go to the cheaper model (order in COST).

    python reannotate/select_annotator.py --results $PE_DATA/reannotate/eps4.0/results \
        --attributes sample_size dominant_data_type --out $PE_DATA/reannotate/eps4.0/best_annotator.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

COST = ["phi_mh", "phi_mlp", "tfidf", "biomed_mh", "biomed_mlp", "biomedbert", "scibert"]  # cheapest first


def has_weights(res, model, a):
    d = Path(res) / model / "models"
    if model.endswith("_mh"):
        return (d / "multihead.pt").exists()
    return (d / f"{a}.joblib").exists() or (d / f"{a}.pt").exists() or (d / a / "classes.json").exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--attributes", nargs="+", required=True)
    p.add_argument("--eval_set", default="pool_test_vs_qwen")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    M = pd.read_csv(Path(args.results) / "all_metrics.csv")
    M = M[M.eval_set == args.eval_set]
    spec, lines = {}, []
    for a in args.attributes:
        S = M[(M.attribute == a)].copy()
        S = S[[has_weights(args.results, r.model, a) for r in S.itertuples()]]
        if S.empty:
            raise SystemExit(f"no saved model for {a}")
        S["cost"] = S.model.map({m: i for i, m in enumerate(COST)}).fillna(99)
        S = S.sort_values(["kappa", "cost"], ascending=[False, True])
        best = S.iloc[0]
        spec[a] = {"model": best.model, "kappa": float(best.kappa), "marginal_jsd": float(best.marginal_jsd)}
        lines.append(f"{a}: {best.model} (kappa {best.kappa:.3f}); ranking: " +
                     ", ".join(f"{r.model} {r.kappa:.3f}" for r in S.itertuples()))
    out = {"results_dir": str(Path(args.results).resolve()), "eval_set": args.eval_set, "attributes": spec}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
