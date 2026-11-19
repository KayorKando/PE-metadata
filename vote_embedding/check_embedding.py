"""Vote embedding, step 3: is the fine-tuned backbone a better vote embedding? Pool only, no PE run.

Arms: t5 (sentence-t5-base, the current vote embedding, cached), frozen (BiomedBERT mean-pool,
no fine-tuning), ce and normsoftmax (train_embedder.py outputs; arms whose dir is missing are skipped).

On the pool test split (group_split seed 0, unseen by training), repeated random half/half
splits into ref / query (by prompt group, as Experiment 1):
  1. k-NN kappa, k = 5, per attribute incl. the word_count bucket: does the vote see the attribute?
  2. t5 recall@10: share of each query's 10 t5-neighbours (in ref) that are also among its 10
     neighbours under the arm: how much of t5's semantic structure survives (MAUVE stays on t5).
  3. annotator (ce / normsoftmax heads, nearest proxy) vs Qwen on pool test and on the private
     Qwen sample: kappa and marginal JSD (re-annotating's evaluate(), confusion from the val split),
     next to the per-attribute CE BiomedBERT of re-annotating when its metrics.csv is there.
Output: <out>/CHECK.md, knn_kappa.png, knn_recall_per_repeat.csv, annotator.csv.

    python vote_embedding/check_embedding.py \
      --pool_qwen $PE_DATA/reannotate/eps4.0/pool_qwen.csv --private_qwen $PE_DATA/reannotate/eps4.0/private_qwen.csv
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
from blindness.experiment import split_halves  # noqa: E402
from blindness.metrics import knn_kappa, knn_neighbors  # noqa: E402
from blindness.schema import word_count_bin  # noqa: E402
from reannotate.train_classifier import (HF_MODELS, REQUESTED_COLS, confusion, embed_frozen_bert, group_keys,  # noqa: E402
                                         embed_phi, evaluate, group_split, labels_from_map)
from vote_embedding.train_embedder import ATTRS, LOSSES  # noqa: E402

ARMS = ["t5", "frozen"] + list(LOSSES)   # t5, frozen, normsoftmax, ce
COLORS = {"t5": "#a3a29d", "frozen": "#52514e", "ce": "#eb6834", "normsoftmax": "#2a78d6"}
EVAL_SETS = ["pool_test_vs_qwen", "private_sample_vs_qwen"]


def neighbour_metrics(X, labels, groups, repeats, k, recall_k, seed):
    """X: {arm: [n, d]} (must contain t5); labels: {attr: [n]}. -> per-repeat DataFrame."""
    rng = np.random.default_rng(seed)
    n, rows = len(groups), []
    for r in range(repeats):
        ref, qry = split_halves(n, rng, groups)
        kk = min(max(k, recall_k), len(ref))
        nb = {arm: knn_neighbors(Xa[ref], Xa[qry], kk) for arm, Xa in X.items()}
        t5 = nb["t5"][:, :recall_k]
        for arm, nbr in nb.items():
            mine = nbr[:, :recall_k]
            recall = float(np.mean([len(set(a) & set(b)) / recall_k for a, b in zip(mine, t5)]))
            for a, y in labels.items():
                kap, agree = knn_kappa(nbr[:, :k], y[ref], y[qry])
                rows.append(dict(repeat=r, arm=arm, attribute=a, knn_kappa=kap, knn_agree=agree,
                                 recall_t5=recall, n_ref=len(ref), n_query=len(qry)))
    return pd.DataFrame(rows)


def annotator_metrics(ft, y_pool, te, va, texts_pool, priv_texts, y_priv):
    """ft: {arm: FTEmbedder}; -> DataFrame (arm, attribute, eval_set, kappa, marginal_jsd, ...)."""
    rows = []
    for arm, emb in ft.items():
        p_te, p_va = emb.predict([texts_pool[i] for i in te]), emb.predict([texts_pool[i] for i in va])
        p_pr = emb.predict(priv_texts) if priv_texts is not None else None
        for a in ATTRS:
            classes = list(emb.classes[a])
            C = confusion(y_pool[a][va], p_va[a], classes)
            rows.append(dict(arm=arm, attribute=a, eval_set=EVAL_SETS[0], **evaluate(y_pool[a][te], p_te[a], C, classes)))
            if p_pr is not None:
                rows.append(dict(arm=arm, attribute=a, eval_set=EVAL_SETS[1], **evaluate(y_priv[a], p_pr[a], C, classes)))
    return pd.DataFrame(rows)


def reference_metrics(path):
    """Per-attribute CE BiomedBERT of re-annotating (results/biomedbert/metrics.csv), if present."""
    if not path or not Path(path).exists():
        return None
    ref = pd.read_csv(path)
    ref = ref[ref.eval_set.isin(EVAL_SETS)].copy()
    ref["arm"] = "per-attr CE (reannotate)"
    return ref.drop(columns=[c for c in ("model",) if c in ref])


def _pm(m, s):
    return f"{m:.3f} ± {s:.3f}" if s == s else f"{m:.3f}"


def write_report(out, per_rep, ann, info):
    arms = [a for a in ARMS if a in set(per_rep.arm)]
    attrs = list(dict.fromkeys(per_rep.attribute))
    g = per_rep.groupby(["attribute", "arm"]).knn_kappa
    mean, sd = g.mean().unstack(), g.std().unstack()
    L = ["# Step 3: fine-tuned vote embedding, pool-only check", ""]
    L += [f"- {k}: {v}" for k, v in info.items()]
    L += ["", f"## 1. k-NN κ (k = {info['k']}), mean ± sd over {info['repeats']} ref/query splits of the pool test split", "",
          "| attribute | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for a in attrs:
        L.append(f"| {a} | " + " | ".join(_pm(mean.loc[a, arm], sd.loc[a, arm]) for arm in arms) + " |")
    content = [a for a in attrs if a in ATTRS]
    cm = per_rep[per_rep.attribute.isin(content)].groupby(["arm", "repeat"]).knn_kappa.mean().groupby("arm")
    L.append("| **mean of 8 content attributes** | " + " | ".join(_pm(cm.mean()[arm], cm.std()[arm]) for arm in arms) + " |")
    rc = per_rep.groupby(["arm", "repeat"]).recall_t5.first().groupby("arm")
    L += ["", f"## 2. t5 recall@{info['recall_k']} (share of t5's {info['recall_k']} nearest ref points kept by the arm)", "",
          "| " + " | ".join(arms) + " |", "|" + "---|" * len(arms),
          "| " + " | ".join(_pm(rc.mean()[arm], rc.std()[arm]) for arm in arms) + " |",
          "", f"Random neighbours would give ≈ {info['recall_k']}/n_ref ≈ {info['recall_k'] / per_rep.n_ref.iloc[0]:.3f}."]
    if ann is not None and len(ann):
        L += ["", "## 3. Annotator vs Qwen (nearest proxy / argmax): κ, marginal JSD (floor in brackets)", ""]
        for es in EVAL_SETS:
            sub = ann[ann.eval_set == es]
            if not len(sub):
                continue
            aarms = list(dict.fromkeys(sub.arm))
            L += [f"### {es}", "", "| attribute | " + " | ".join(f"{x} κ | {x} JSD" for x in aarms) + " |",
                  "|---|" + "---|---|" * len(aarms)]
            for a in ATTRS:
                cells = []
                for x in aarms:
                    r = sub[(sub.arm == x) & (sub.attribute == a)]
                    cells += [f"{r.kappa.iloc[0]:.3f}", f"{r.marginal_jsd.iloc[0]:.3f} ({r.marginal_jsd_floor.iloc[0]:.3f})"] \
                        if len(r) else ["–", "–"]
                L.append(f"| {a} | " + " | ".join(cells) + " |")
            mk = sub.groupby("arm")[["kappa", "marginal_jsd"]].mean()
            L.append("| **mean** | " + " | ".join(f"**{mk.loc[x, 'kappa']:.3f}** | {mk.loc[x, 'marginal_jsd']:.3f}"
                                                for x in aarms) + " |")
            L.append("")
    L += ["", "![k-NN kappa per attribute](knn_kappa.png)", ""]
    (out / "CHECK.md").write_text("\n".join(L))
    plot(out / "knn_kappa.png", mean, sd, rc, arms, attrs, info["recall_k"])


def plot(path, mean, sd, rc, arms, attrs, recall_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [5, 1]})
    w = 0.8 / len(arms)
    x = np.arange(len(attrs))
    for i, arm in enumerate(arms):
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, mean.loc[attrs, arm], w, yerr=sd.loc[attrs, arm],
               color=COLORS[arm], label=arm, error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(attrs, rotation=30, ha="right")
    ax.set_ylabel("k-NN κ (k = 5)"); ax.axhline(0, color="#52514e", lw=0.8)
    ax.legend(frameon=False, ncol=len(arms), loc="upper left")
    ax2.bar(range(len(arms)), [rc.mean()[a] for a in arms], yerr=[rc.std()[a] for a in arms],
            color=[COLORS[a] for a in arms])
    ax2.set_xticks(range(len(arms))); ax2.set_xticklabels(arms, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05); ax2.set_title(f"t5 recall@{recall_k}")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    pe = Path(os.environ.get("PE_DATA", "."))
    p = argparse.ArgumentParser()
    p.add_argument("--pool_qwen", required=True)
    p.add_argument("--private_qwen", default=None, help="Qwen labels of the private sample (evaluation only)")
    p.add_argument("--eps", default="4.0")
    p.add_argument("--ft_root", default=None, help="dir with ce/ and normsoftmax/ (default $PE_DATA/vote_embedding/eps<eps>)")
    p.add_argument("--out", default=None, help="default <ft_root>/check")
    p.add_argument("--cache_dir", default=None, help="embedding cache (default $PE_DATA/cache)")
    p.add_argument("--frozen_model", default=HF_MODELS["biomedbert"])
    p.add_argument("--ref_metrics", default=None,
                   help="per-attribute CE metrics (default $PE_DATA/reannotate/eps<eps>/results/biomedbert/metrics.csv)")
    p.add_argument("--seed", type=int, default=0, help="group_split seed (must match training)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--recall_k", type=int, default=10)
    args = p.parse_args(argv)
    ft_root = Path(args.ft_root or pe / "vote_embedding" / f"eps{args.eps}")
    out = Path(args.out or ft_root / "check")
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(pe / "cache")

    pool = pd.read_csv(args.pool_qwen)
    texts = pool["text"].astype(str).tolist()
    y_pool = labels_from_map(pool)
    groups = group_keys(pool, [c for c in REQUESTED_COLS if c in pool])
    tr, va, te = group_split(groups, args.seed)
    labelled = np.array([y_pool[ATTRS[0]][i] is not None for i in range(len(pool))])  # missing = whole row
    te, va = te[labelled[te]], va[labelled[va]]
    te_texts = [texts[i] for i in te]
    labels = {a: y_pool[a][te].astype(str) for a in ATTRS}
    labels["word_count"] = word_count_bin(te_texts)

    from vote_embedding.embedder import FTEmbedder
    ft = {arm: FTEmbedder(ft_root / arm, cache_dir=cache_dir) for arm in LOSSES if (ft_root / arm / "heads.pt").exists()}
    print(f"pool test split: {len(te)} labelled texts; fine-tuned arms found: {list(ft) or 'none'}", flush=True)
    X = {"t5": embed_phi(te_texts, cache_dir),
         "frozen": embed_frozen_bert(te_texts, args.frozen_model, cache_dir)}
    X.update({arm: emb(te_texts) for arm, emb in ft.items()})

    per_rep = neighbour_metrics(X, labels, groups[te], args.repeats, args.k, args.recall_k, args.seed)
    per_rep.to_csv(out / "knn_recall_per_repeat.csv", index=False)

    ann = None
    priv_texts = y_priv = None
    if args.private_qwen:
        pq = pd.read_csv(args.private_qwen)
        priv_texts, y_priv = pq["text"].astype(str).tolist(), labels_from_map(pq)
    if ft:
        ann = annotator_metrics(ft, y_pool, te, va, texts, priv_texts, y_priv)
        ref = reference_metrics(args.ref_metrics or pe / "reannotate" / f"eps{args.eps}" / "results" / "biomedbert" / "metrics.csv")
        if ref is not None:
            ann = pd.concat([ann, ref[[c for c in ann.columns if c in ref]]], ignore_index=True)
        ann.to_csv(out / "annotator.csv", index=False)

    info = {"pool test texts": len(te), "repeats": args.repeats, "k": args.k, "recall_k": args.recall_k,
            "arms": ", ".join(X)}
    any_ft = next(iter(ft.values()), None)
    if any_ft is not None:
        from vote_embedding.train_embedder import share_over
        info["share over max_len tokens (pool test)"] = f"{share_over(any_ft.tk.tok, te_texts, any_ft.max_len):.3f}"
        if priv_texts is not None:
            info["share over max_len tokens (private sample)"] = f"{share_over(any_ft.tk.tok, priv_texts, any_ft.max_len):.3f}"
    for arm in ft:
        log = json.loads((ft_root / arm / "train_log.json").read_text()) if (ft_root / arm / "train_log.json").exists() else {}
        info[f"{arm} training"] = (f"first-batch loss {log.get('first_loss', float('nan')):.2f} "
                                   f"(sum ln k {log.get('sum_ln_k', float('nan')):.2f}), best epoch {log.get('best_epoch')}, "
                                   f"val mean κ {log.get('best_val_mean_kappa', float('nan')):.3f}")
    write_report(out, per_rep, ann, info)
    print((out / "CHECK.md").read_text())


if __name__ == "__main__":
    main()
