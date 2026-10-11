"""Re-annotating, step 3: train one small classifier per attribute on Qwen labels
of the round-0 pool, and measure

  (1) agreement with the labels on held-out pool texts: Cohen's kappa, macro-F1, accuracy
  (2) how well the predicted *marginal* matches: JSD(pred, label), and the JSD after
      confusion-matrix correction (p_true ~ solve C^T p = p_pred, C from the val split)
  (3) transfer to real text: the same numbers on private texts against Gemini labels
      (and, with --private_qwen, against Qwen labels on a private sample, which splits the
      gap into "annotator differs" vs "text domain differs"). Private data is used only
      for evaluation (non-private diagnostic); training uses synthetic text only.

Models (one per run, one classifier per attribute):
  tfidf       TF-IDF (1-2 grams) + logistic regression, C picked on val (CPU)
  scibert     allenai/scibert_scivocab_uncased, fine-tuned
  biomedbert  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract, fine-tuned
None of them uses sentence-t5, the vote embedding (no circularity).

Splits: train/val/test = 70/15/15 by prompt group (rows with identical requested
metadata stay on one side). No class weighting (we care about marginals).

Example:
    python reannotate/train_classifier.py --model tfidf \
      --pool_qwen $PE_DATA/reannotate/pool_qwen.csv \
      --private_csv $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
      --private_qwen $PE_DATA/reannotate/private_qwen.csv --out $PE_DATA/reannotate/results
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import BIORXIV_SCHEMA, clean_labels  # noqa: E402

ATTRS = list(BIORXIV_SCHEMA)
HF_MODELS = {"scibert": "allenai/scibert_scivocab_uncased",
             "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"}
REQUESTED_COLS = ATTRS + ["word_count_requested"]


# ---------------- labels, splits, metrics ----------------
def labels_from_map(df):
    """map_<attr> columns -> schema-cleaned labels; None where the annotation is missing."""
    m = df[[f"map_{a}" for a in ATTRS]].rename(columns=lambda c: c[4:])
    missing = m.isna().all(axis=1).to_numpy()
    lab = clean_labels(m, ATTRS)
    return {a: np.where(missing, None, lab[a]) for a in ATTRS}


def group_split(groups, seed, frac=(0.7, 0.15)):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    order = rng.permutation(inv.max() + 1)
    cum = np.cumsum(np.bincount(inv)[order]) / len(groups)
    part = np.zeros(len(order), int)
    part[order[cum > frac[0]]] = 1
    part[order[cum > frac[0] + frac[1]]] = 2
    p = part[inv]
    return np.flatnonzero(p == 0), np.flatnonzero(p == 1), np.flatnonzero(p == 2)


def dist(labels, cats):
    c = Counter(labels)
    v = np.array([c.get(k, 0) for k in cats], float)
    return v / max(v.sum(), 1)


def jsd_dist(p, q):
    return float(jensenshannon(p, q)) if p.sum() > 0 and q.sum() > 0 else float("nan")


def confusion(y_true, y_pred, classes, smooth=0.1):
    idx = {c: i for i, c in enumerate(classes)}
    C = np.full((len(classes), len(classes)), smooth)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            C[idx[t], idx[p]] += 1
    return C / C.sum(1, keepdims=True)  # rows: true class, cols: predicted


def corrected_marginal(y_pred, C, classes):
    """p_pred = C^T p_true  ->  p_true by non-negative least squares, renormalised."""
    p, _ = nnls(C.T, dist(y_pred, classes))
    return p / p.sum() if p.sum() > 0 else p


def evaluate(y_true, y_pred, C, classes, seed=0):
    keep = np.array([t is not None for t in y_true])
    y_true, y_pred = np.asarray(y_true)[keep].astype(str), np.asarray(y_pred)[keep].astype(str)
    cats = sorted(set(classes) | set(y_true))
    p_true = dist(y_true, cats)
    p_corr = dict(zip(classes, corrected_marginal(y_pred, C, classes)))
    rng = np.random.default_rng(seed)  # floor: a resample of the true labels vs themselves
    floor = np.mean([jsd_dist(dist(rng.choice(y_true, len(y_true)), cats), p_true) for _ in range(20)])
    return dict(n=int(len(y_true)),
                kappa=float(cohen_kappa_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
                accuracy=float(accuracy_score(y_true, y_pred)),
                marginal_jsd=jsd_dist(dist(y_pred, cats), p_true),
                marginal_jsd_corrected=jsd_dist(np.array([p_corr.get(k, 0.0) for k in cats]), p_true),
                marginal_jsd_floor=float(floor))


# ---------------- models ----------------
class TfidfLR:
    def __init__(self, seed):
        self.seed = seed
        self.vec = None

    def fit_vectorizer(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        self.vec.fit(texts)

    def fit(self, Xtr, ytr, Xva, yva):
        from sklearn.linear_model import LogisticRegression
        best = None
        for C in [0.3, 1.0, 3.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=3000, random_state=self.seed).fit(Xtr, ytr)
            k = cohen_kappa_score(yva, clf.predict(Xva))
            if best is None or k > best[0]:
                best = (k, C, clf)
        self.clf, self.C = best[2], best[1]
        return {"C": best[1], "val_kappa": float(best[0])}

    def predict(self, X):
        return self.clf.predict(X)


class FineTuned:
    def __init__(self, name, seed, epochs, lr, batch, max_len, device):
        import torch
        from transformers import AutoTokenizer
        self.torch, self.name, self.seed = torch, name, seed
        self.epochs, self.lr, self.batch, self.max_len, self.device = epochs, lr, batch, max_len, device
        self.tok = AutoTokenizer.from_pretrained(name)

    def encode(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_len)["input_ids"]

    def _batches(self, ids, idx):
        for i in range(0, len(idx), self.batch):
            b = [ids[j] for j in idx[i:i + self.batch]]
            enc = self.tok.pad({"input_ids": b}, return_tensors="pt")
            yield {k: v.to(self.device) for k, v in enc.items()}

    def predict_ids(self, ids):
        torch = self.torch
        self.model.eval()
        order = np.argsort([len(x) for x in ids])  # length-sorted batches: less padding
        out = np.empty(len(ids), int)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            for s, enc in zip(range(0, len(order), self.batch), self._batches(ids, order)):
                out[order[s:s + self.batch]] = self.model(**enc).logits.argmax(-1).cpu().numpy()
        return out

    def fit(self, ids_tr, ytr, ids_va, yva, classes):
        torch = self.torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        torch.manual_seed(self.seed)
        cid = {c: i for i, c in enumerate(classes)}
        y = np.array([cid[v] for v in ytr])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=len(classes), ignore_mismatched_sizes=True).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(ids_tr) / self.batch))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        rng = np.random.default_rng(self.seed)
        best, log = None, []
        for ep in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(ids_tr))
            for s, enc in zip(range(0, len(perm), self.batch), self._batches(ids_tr, perm)):
                yb = torch.tensor(y[perm[s:s + self.batch]], device=self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    loss = torch.nn.functional.cross_entropy(self.model(**enc).logits.float(), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            k = cohen_kappa_score(yva, np.array(classes)[self.predict_ids(ids_va)])
            log.append(float(k))
            if best is None or k > best[0]:
                best = (k, ep, {n: p.detach().clone() for n, p in self.model.state_dict().items()})
        self.model.load_state_dict(best[2])
        self.classes = np.array(classes)
        return {"best_epoch": best[1], "val_kappa_per_epoch": log}

    def predict(self, ids):
        return self.classes[self.predict_ids(ids)]


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["tfidf", "scibert", "biomedbert"])
    p.add_argument("--pool_qwen", required=True, help="annotate_qwen.py output on the round-0 pool")
    p.add_argument("--private_csv", required=True, help="private CSV with Gemini labels (evaluation only)")
    p.add_argument("--private_qwen", default=None, help="annotate_qwen.py output on a private sample")
    p.add_argument("--private_n", type=int, default=0, help="evaluate on N random private texts (0 = all)")
    p.add_argument("--attributes", nargs="+", default=ATTRS)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    args = p.parse_args()

    pool = pd.read_csv(args.pool_qwen)
    lab_pool = labels_from_map(pool)
    groups = pool[[c for c in REQUESTED_COLS if c in pool]].astype(str).agg("|".join, axis=1).to_numpy()
    tr, va, te = group_split(groups, args.seed)

    priv = pd.read_csv(args.private_csv)
    priv = priv[priv["text"].notna()].reset_index(drop=True)
    lab_priv = clean_labels(priv, ATTRS)  # Gemini labels
    pidx = np.arange(len(priv))
    if args.private_n and args.private_n < len(priv):
        pidx = np.sort(np.random.default_rng(args.seed).choice(len(priv), args.private_n, replace=False))
    pq = None
    if args.private_qwen:
        pq = pd.read_csv(args.private_qwen)
        lab_pq = labels_from_map(pq)

    sets = {"pool": pool["text"].astype(str).tolist(), "private": priv["text"].astype(str).iloc[pidx].tolist()}
    if pq is not None:
        sets["private_qwen"] = pq["text"].astype(str).tolist()

    t0 = time.time()
    if args.model == "tfidf":
        m = TfidfLR(args.seed)
        m.fit_vectorizer([sets["pool"][i] for i in tr])
        feats = {k: m.vec.transform(v) for k, v in sets.items()}
    else:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = FineTuned(HF_MODELS[args.model], args.seed, args.epochs, args.lr, args.batch_size, args.max_len, dev)
        feats = {k: m.encode(v) for k, v in sets.items()}
    sub = lambda f, idx: f[idx] if args.model == "tfidf" else [f[i] for i in idx]  # noqa: E731

    rows, info, preds = [], {}, {}
    for a in args.attributes:
        y = lab_pool[a]
        ok = lambda idx: idx[np.array([y[i] is not None for i in idx], bool)]  # noqa: E731
        itr, iva, ite = ok(tr), ok(va), ok(te)
        classes = sorted(set(y[itr]))
        ta = time.time()
        if args.model == "tfidf":
            info[a] = m.fit(sub(feats["pool"], itr), y[itr].astype(str), sub(feats["pool"], iva), y[iva].astype(str))
        else:
            info[a] = m.fit(sub(feats["pool"], itr), y[itr], sub(feats["pool"], iva), y[iva].astype(str), classes)
        info[a]["train_seconds"] = round(time.time() - ta, 1)
        C = confusion(y[iva], m.predict(sub(feats["pool"], iva)), classes)

        ti = time.time()
        pred_pool = m.predict(feats["pool"])
        preds[a] = pred_pool
        rows.append(dict(attribute=a, eval_set="pool_test_vs_qwen", **evaluate(y[ite], pred_pool[ite], C, classes)))
        pred_priv = m.predict(feats["private"])
        rows.append(dict(attribute=a, eval_set="private_vs_gemini",
                         **evaluate(lab_priv[a][pidx].astype(object), pred_priv, C, classes)))
        if pq is not None:
            pred_pq = m.predict(feats["private_qwen"])
            rows.append(dict(attribute=a, eval_set="private_sample_vs_qwen", **evaluate(lab_pq[a], pred_pq, C, classes)))
        info[a]["predict_seconds"] = round(time.time() - ti, 1)
        print(f"[{args.model}] {a}: " + ", ".join(f"{r['eval_set']} kappa {r['kappa']:.3f}"
                                                   for r in rows if r["attribute"] == a), flush=True)

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.insert(0, "model", args.model)
    res.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({f"pred_{a}": v for a, v in preds.items()}).to_csv(out / "pool_predictions.csv", index=False)
    (out / "info.json").write_text(json.dumps({**vars(args), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                               "total_seconds": round(time.time() - t0, 1), "per_attribute": info},
                                              indent=2))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
