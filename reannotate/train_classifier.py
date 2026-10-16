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


def mlp_trunk(nn, d, hidden, dropout):
    """Two hidden layers: Linear(d,h) -> ReLU -> Dropout -> Linear(h,h) -> ReLU -> Dropout."""
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))


class EmbeddingMLP:
    """One MLP per attribute on a frozen text embedding: standardise -> two hidden layers
    (256, ReLU, dropout 0.2) -> Linear(256,k). AdamW, early stopping on val kappa."""

    def __init__(self, seed, hidden=256, dropout=0.2, lr=1e-3, wd=1e-4, batch=256, max_epochs=100, patience=10):
        import torch
        self.torch, self.seed = torch, seed
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, wd
        self.batch, self.max_epochs, self.patience = batch, max_epochs, patience
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _t(self, X):
        return self.torch.tensor((X - self.mu) / self.sd, dtype=self.torch.float32, device=self.device)

    def _pred(self, Xt):
        self.net.eval()
        with self.torch.no_grad():
            return self.net(Xt).argmax(-1).cpu().numpy()

    def fit(self, Xtr, ytr, Xva, yva, classes):
        torch = self.torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        cid = {c: i for i, c in enumerate(classes)}
        y = torch.tensor([cid[v] for v in ytr], device=self.device)
        Xt, Xv = self._t(Xtr), self._t(Xva)
        nn = torch.nn
        self.net = nn.Sequential(mlp_trunk(nn, Xtr.shape[1], self.hidden, self.dropout),
                                 nn.Linear(self.hidden, len(classes))).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.classes = np.array(classes)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, bad, log = None, 0, []
        for ep in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g).to(self.device)
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                loss = nn.functional.cross_entropy(self.net(Xt[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
            k = cohen_kappa_score(yva, self.classes[self._pred(Xv)])
            log.append(float(k))
            if best is None or k > best[0]:
                best, bad = (k, ep, {n: p.detach().clone() for n, p in self.net.state_dict().items()}), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best[2])
        return {"best_epoch": best[1], "epochs_run": len(log), "val_kappa": float(best[0])}

    def predict(self, X):
        return self.classes[self._pred(self._t(X))]


class MultiHeadMLP:
    """Shared trunk (same two hidden layers) + one linear head per attribute giving that
    attribute's logits. Trained jointly: sum of per-attribute cross-entropies, rows with a
    missing label ignored for that head. Early stopping on the mean val kappa."""

    def __init__(self, seed, hidden=256, dropout=0.2, lr=1e-3, wd=1e-4, batch=256, max_epochs=100, patience=10):
        import torch
        self.torch, self.seed = torch, seed
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, wd
        self.batch, self.max_epochs, self.patience = batch, max_epochs, patience
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _t(self, X):
        return self.torch.tensor((X - self.mu) / self.sd, dtype=self.torch.float32, device=self.device)

    def _logits(self, Xt):
        self.trunk.eval(); self.heads.eval()
        with self.torch.no_grad():
            h = self.trunk(Xt)
            return {a: self.heads[a](h).argmax(-1).cpu().numpy() for a in self.attrs}

    def fit(self, Xtr, Ytr, Xva, Yva, classes):
        """Ytr / Yva: {attr: label array, None = missing}; classes: {attr: sorted classes}."""
        torch, nn = self.torch, self.torch.nn
        torch.manual_seed(self.seed)
        self.attrs, self.classes = list(classes), {a: np.array(c) for a, c in classes.items()}
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        ys = {}
        for a in self.attrs:
            cid = {c: i for i, c in enumerate(classes[a])}
            ys[a] = torch.tensor([cid.get(v, -100) if v is not None else -100 for v in Ytr[a]], device=self.device)
        Xt, Xv = self._t(Xtr), self._t(Xva)
        self.trunk = mlp_trunk(nn, Xtr.shape[1], self.hidden, self.dropout).to(self.device)
        self.heads = nn.ModuleDict({a: nn.Linear(self.hidden, len(classes[a])) for a in self.attrs}).to(self.device)
        params = list(self.trunk.parameters()) + list(self.heads.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        def val_kappas():
            pr = self._logits(Xv)
            out = {}
            for a in self.attrs:
                ok = np.array([v is not None for v in Yva[a]])
                out[a] = float(cohen_kappa_score(np.asarray(Yva[a])[ok].astype(str), self.classes[a][pr[a][ok]]))
            return out

        best, bad, log = None, 0, []
        for ep in range(self.max_epochs):
            self.trunk.train(); self.heads.train()
            perm = torch.randperm(len(Xt), generator=g).to(self.device)
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                h = self.trunk(Xt[b])
                loss = sum(nn.functional.cross_entropy(self.heads[a](h), ys[a][b], ignore_index=-100)
                           for a in self.attrs if (ys[a][b] >= 0).any())
                opt.zero_grad(); loss.backward(); opt.step()
            vk = val_kappas()
            mk = float(np.mean(list(vk.values())))
            log.append(mk)
            if best is None or mk > best[0]:
                state = ({n: p.detach().clone() for n, p in self.trunk.state_dict().items()},
                         {n: p.detach().clone() for n, p in self.heads.state_dict().items()})
                best, bad = (mk, ep, state, vk), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.trunk.load_state_dict(best[2][0]); self.heads.load_state_dict(best[2][1])
        return {"best_epoch": best[1], "epochs_run": len(log), "val_mean_kappa": best[0], "val_kappa": best[3]}

    def predict(self, X, a):
        return self.classes[a][self._logits(self._t(X))[a]]


class _Head:
    """Per-attribute view of a fitted MultiHeadMLP, so the evaluation loop is shared."""

    def __init__(self, mh, a):
        self.mh, self.a = mh, a

    def predict(self, X):
        return self.mh.predict(X, self.a)


def embed_phi(texts, cache_dir):
    """sentence-t5-base, max_seq_length 1024: the vote embedding (same cache as run.py)."""
    from blindness.embed import embed
    return embed(list(texts), "sentence-t5-base", 1024, batch_size=256, cache_dir=cache_dir)


def embed_frozen_bert(texts, name, cache_dir, max_len=512, batch=64):
    """Frozen encoder, mean of the last hidden states over real tokens. Cached on disk."""
    import hashlib
    path = None
    if cache_dir:
        key = hashlib.sha1(("\n".join(texts) + name + str(max_len)).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"frozen_{name.replace('/', '_')}_{key}.npy"
        if path.exists():
            return np.load(path)
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).to(dev).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
    return out


# ---------------- save / load (for using a trained classifier as the PE annotator) ----------------
def _mlp_from(d, hidden, dropout, k):
    import torch.nn as nn
    return nn.Sequential(mlp_trunk(nn, d, hidden, dropout), nn.Linear(hidden, k))


def save_attr_model(model_name, m, a, mdir):
    """Write what load_predictor needs for attribute a (multi-head models are saved once, see main)."""
    import joblib
    mdir.mkdir(parents=True, exist_ok=True)
    if model_name == "tfidf":
        joblib.dump(m.clf, mdir / f"{a}.joblib")
    elif model_name in HF_MODELS:
        d = mdir / a
        m.model.save_pretrained(d)
        (d / "classes.json").write_text(json.dumps(list(m.classes)))
    elif model_name.endswith("_mlp"):
        import torch
        torch.save({"state": m.net.state_dict(), "mu": m.mu, "sd": m.sd, "classes": list(m.classes),
                    "d": int(len(m.mu)), "hidden": m.hidden, "dropout": m.dropout}, mdir / f"{a}.pt")


def load_predictor(results_dir, model_name, a, device=None):
    """texts -> np.array of labels for attribute a, from results_dir/<model>/models/."""
    import torch
    mdir = Path(results_dir) / model_name / "models"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "tfidf":
        import joblib
        vec, clf = joblib.load(mdir / "vectorizer.joblib"), joblib.load(mdir / f"{a}.joblib")
        return lambda texts: clf.predict(vec.transform(list(texts)))
    if model_name in HF_MODELS:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mdir / "tokenizer")
        net = AutoModelForSequenceClassification.from_pretrained(mdir / a).to(device).eval()
        classes = np.array(json.loads((mdir / a / "classes.json").read_text()))

        def pred(texts, batch=64, max_len=512):
            texts = list(texts)
            order = np.argsort([len(t) for t in texts])
            out = np.empty(len(texts), int)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                for i in range(0, len(order), batch):
                    idx = order[i:i + batch]
                    enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                              return_tensors="pt").to(device)
                    out[idx] = net(**enc).logits.argmax(-1).cpu().numpy()
            return classes[out]
        return pred
    # MLPs on a frozen embedding
    if model_name.startswith("phi"):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-t5-base", device=device)
        st.max_seq_length = 1024
        featurize = lambda texts: st.encode(list(texts), batch_size=256, convert_to_numpy=True)  # noqa: E731
    else:
        featurize = lambda texts: embed_frozen_bert(list(texts), HF_MODELS["biomedbert"], None)  # noqa: E731
    if model_name.endswith("_mh"):
        ck = torch.load(mdir / "multihead.pt", map_location=device, weights_only=False)
        import torch.nn as nn
        trunk = mlp_trunk(nn, ck["d"], ck["hidden"], ck["dropout"]).to(device)
        trunk.load_state_dict(ck["trunk"])
        head = nn.Linear(ck["hidden"], len(ck["classes"][a])).to(device)
        head.load_state_dict({k.split(".", 1)[1]: v for k, v in ck["heads"].items() if k.startswith(a + ".")})
        net, classes = nn.Sequential(trunk, head).eval(), np.array(ck["classes"][a])
    else:
        ck = torch.load(mdir / f"{a}.pt", map_location=device, weights_only=False)
        net = _mlp_from(ck["d"], ck["hidden"], ck["dropout"], len(ck["classes"])).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        classes = np.array(ck["classes"])
    mu, sd = ck["mu"], ck["sd"]

    def pred(texts):
        X = torch.tensor((featurize(texts) - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            return classes[net(X).argmax(-1).cpu().numpy()]
    return pred


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["tfidf", "scibert", "biomedbert", "phi_mlp", "biomed_mlp", "phi_mh", "biomed_mh"])
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
    p.add_argument("--cache_dir", default=None, help="embedding cache (default $PE_DATA/cache)")
    p.add_argument("--save_models", action="store_true", help="save trained models (to use one as the PE annotator)")
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
    import os
    cache_dir = args.cache_dir or str(Path(os.environ.get("PE_DATA", ".")) / "cache")
    array_feats = args.model in ("tfidf", "phi_mlp", "biomed_mlp", "phi_mh", "biomed_mh")
    multihead = args.model.endswith("_mh")
    if args.model in ("phi_mlp", "biomed_mlp", "phi_mh", "biomed_mh"):
        m = MultiHeadMLP(args.seed) if multihead else EmbeddingMLP(args.seed)
        if args.model.startswith("phi"):
            feats = {k: embed_phi(v, cache_dir) for k, v in sets.items()}
        else:
            feats = {k: embed_frozen_bert(v, HF_MODELS["biomedbert"], cache_dir, args.max_len) for k, v in sets.items()}
    elif args.model == "tfidf":
        m = TfidfLR(args.seed)
        m.fit_vectorizer([sets["pool"][i] for i in tr])
        feats = {k: m.vec.transform(v) for k, v in sets.items()}
    else:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = FineTuned(HF_MODELS[args.model], args.seed, args.epochs, args.lr, args.batch_size, args.max_len, dev)
        feats = {k: m.encode(v) for k, v in sets.items()}
    sub = lambda f, idx: f[idx] if array_feats else [f[i] for i in idx]  # noqa: E731

    out = Path(args.out) / args.model
    mdir = out / "models"
    if args.save_models:
        mdir.mkdir(parents=True, exist_ok=True)
        if args.model == "tfidf":
            import joblib
            joblib.dump(m.vec, mdir / "vectorizer.joblib")
        elif args.model in HF_MODELS:
            m.tok.save_pretrained(mdir / "tokenizer")
    rows, info, preds = [], {}, {}
    shared = None
    if multihead:  # one joint fit for all attributes
        ta = time.time()
        nonnull = lambda y, idx: [y[i] for i in idx]  # noqa: E731
        classes_all = {a: sorted(set(v for v in lab_pool[a][tr] if v is not None)) for a in args.attributes}
        shared = m.fit(feats["pool"][tr], {a: nonnull(lab_pool[a], tr) for a in args.attributes},
                       feats["pool"][va], {a: nonnull(lab_pool[a], va) for a in args.attributes}, classes_all)
        shared["train_seconds"] = round(time.time() - ta, 1)
        print(f"[{args.model}] joint fit: {shared['epochs_run']} epochs, val mean kappa {shared['val_mean_kappa']:.3f}",
              flush=True)
        mh = m
        if args.save_models:
            import torch
            torch.save({"trunk": mh.trunk.state_dict(), "heads": mh.heads.state_dict(), "mu": mh.mu, "sd": mh.sd,
                        "classes": {a: list(c) for a, c in mh.classes.items()}, "d": int(len(mh.mu)),
                        "hidden": mh.hidden, "dropout": mh.dropout}, mdir / "multihead.pt")
    for a in args.attributes:
        y = lab_pool[a]
        ok = lambda idx: idx[np.array([y[i] is not None for i in idx], bool)]  # noqa: E731
        itr, iva, ite = ok(tr), ok(va), ok(te)
        classes = sorted(set(y[itr]))
        ta = time.time()
        if multihead:
            m = _Head(mh, a)
            classes = list(mh.classes[a])
            info[a] = {"joint_fit": True, "val_kappa": shared["val_kappa"][a]}
        elif args.model == "tfidf":  # (MLP and fine-tuned models take classes)
            info[a] = m.fit(sub(feats["pool"], itr), y[itr].astype(str), sub(feats["pool"], iva), y[iva].astype(str))
        else:
            info[a] = m.fit(sub(feats["pool"], itr), y[itr], sub(feats["pool"], iva), y[iva].astype(str), classes)
        info[a]["train_seconds"] = round(time.time() - ta, 1) if not multihead else 0.0
        C = confusion(y[iva], m.predict(sub(feats["pool"], iva)), classes)
        if args.save_models and not multihead:
            save_attr_model(args.model, m, a, mdir)

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

    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.insert(0, "model", args.model)
    res.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({f"pred_{a}": v for a, v in preds.items()}).to_csv(out / "pool_predictions.csv", index=False)
    (out / "info.json").write_text(json.dumps({**vars(args), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                               "total_seconds": round(time.time() - t0, 1), "per_attribute": info,
                                               "joint_fit": shared},
                                              indent=2))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
