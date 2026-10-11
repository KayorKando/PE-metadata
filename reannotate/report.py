"""Re-annotating: collect everything into one report (REPORT.md + CSVs).

Tables:
  A. Annotator: Qwen vs Gemini on the same private texts (kappa, marginal JSD),
     and LLM compliance on the pool (requested AIM label vs Qwen label).
  B. Per attribute x model, for each evaluation set: kappa, macro-F1,
     marginal JSD raw / corrected / floor.
  C. Cost: training and prediction time per model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import clean_labels  # noqa: E402
from reannotate.train_classifier import ATTRS, dist, jsd_dist, labels_from_map  # noqa: E402


def md(df, digits=3):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join([d.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="folder with <model>/metrics.csv")
    p.add_argument("--pool_qwen", required=True)
    p.add_argument("--private_csv", required=True)
    p.add_argument("--private_qwen", default=None)
    args = p.parse_args()
    res = Path(args.results)
    parts = ["# Re-annotating report\n"]

    # A. annotator
    pool = pd.read_csv(args.pool_qwen)
    lq = labels_from_map(pool)
    lreq = clean_labels(pool, ATTRS)  # requested AIM labels
    rows = []
    for a in ATTRS:
        ok = np.array([v is not None for v in lq[a]])
        r = {"attribute": a, "pool_compliance_kappa(requested vs qwen)":
             float(cohen_kappa_score(lreq[a][ok], lq[a][ok].astype(str)))}
        rows.append(r)
    A = pd.DataFrame(rows).set_index("attribute")
    if args.private_qwen:
        priv = pd.read_csv(args.private_csv)
        priv = priv[priv["text"].notna()].reset_index(drop=True)
        lg = clean_labels(priv, ATTRS)
        pq = pd.read_csv(args.private_qwen)
        lpq = labels_from_map(pq)
        src = pq["src_index"].to_numpy()
        for a in ATTRS:
            ok = np.array([v is not None for v in lpq[a]])
            g, q = lg[a][src][ok], lpq[a][ok].astype(str)
            cats = sorted(set(g) | set(q))
            A.loc[a, "private_kappa(qwen vs gemini)"] = float(cohen_kappa_score(g, q))
            A.loc[a, "private_marginal_jsd(qwen vs gemini)"] = jsd_dist(dist(q, cats), dist(g, cats))
    A.to_csv(res / "annotator.csv")
    parts += ["## A. Annotator (Qwen) quality\n",
              "Compliance = how often the generated text matches the requested AIM label, judged by Qwen. "
              "Qwen vs Gemini = agreement of the two annotators on the same real (private) texts.\n",
              md(A), ""]
    for f in ["pool_qwen", "private_qwen"]:
        path = getattr(args, f)
        if path and Path(path).with_suffix(".json").exists():
            m = json.loads(Path(path).with_suffix(".json").read_text())
            parts.append(f"- `{Path(path).name}`: {m['n_rows']} rows, {m['n_unparsed']} unparsed, "
                         f"{m['generate_seconds']} s ({m['rows_per_second']} rows/s), model {m['model']}")
    parts.append("")

    # B. classifiers
    ms = [pd.read_csv(f) for f in sorted(res.glob("*/metrics.csv"))]
    if ms:
        M = pd.concat(ms, ignore_index=True)
        M.to_csv(res / "all_metrics.csv", index=False)
        for ev in M["eval_set"].unique():
            sub = M[M.eval_set == ev]
            parts.append(f"## B. Classifiers on `{ev}`\n")
            for metric in ["kappa", "macro_f1", "marginal_jsd", "marginal_jsd_corrected"]:
                T = sub.pivot(index="attribute", columns="model", values=metric)
                if metric.startswith("marginal"):
                    T["floor"] = sub.groupby("attribute")["marginal_jsd_floor"].first()
                T.loc["mean"] = T.mean()
                parts += [f"**{metric}**\n", md(T), ""]

        # C. cost
        rows = []
        for f in sorted(res.glob("*/info.json")):
            i = json.loads(f.read_text())
            per = i["per_attribute"].values()
            rows.append({"model": i["model"], "total_seconds": i["total_seconds"],
                         "train_seconds_sum": sum(x["train_seconds"] for x in per),
                         "predict_seconds_sum": sum(x["predict_seconds"] for x in per)})
        parts += ["## C. Cost\n", md(pd.DataFrame(rows).set_index("model"), 1), ""]

    (res / "REPORT.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
