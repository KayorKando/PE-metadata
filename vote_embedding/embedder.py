"""Vote embedding, step 2: the fine-tuned backbone as a drop-in for quota_pe's sentence-t5 Embedder.

    emb = FTEmbedder("$PE_DATA/vote_embedding/eps4.0/normsoftmax")
    X = emb(texts)          # np.float32 [n, 768]: x = LN(z)/||LN(z)|| (normsoftmax) or z (ce); heads unused
    lab = emb.predict(texts)  # {attr: labels} by nearest proxy (normsoftmax) / argmax (ce)

max_len is the training max_len (512) vs sentence-t5's 1024, so long texts are truncated;
each call logs the share of texts over max_len tokens (log_overflow=False to skip).
With cache_dir set, embeddings are cached on disk keyed like blindness.embed (texts + model
name + max_len); the model name includes a fingerprint of the trained weights' files.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reannotate.train_classifier import FineTuned  # noqa: E402
from vote_embedding.train_embedder import autocast, heads_sizes, load, run_sorted, share_over  # noqa: E402


class FTEmbedder:
    def __init__(self, path, batch=64, device=None, cache_dir=None, log_overflow=True):
        self.path = Path(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc, self.heads, classes, self.cfg = load(self.path, self.device)
        self.classes = {a: np.array(c) for a, c in classes.items()}
        self.max_len = int(self.cfg.get("max_len", 512))
        self.tk = FineTuned(str(self.path / "backbone"), 0, 1, 0.0, batch, self.max_len, self.device)
        self.dim = self.enc.dim
        fp = hashlib.sha1((self.path / "heads.pt").read_bytes() + (self.path / "config.json").read_bytes())
        self.name = f"ft-{self.cfg['loss']}-{fp.hexdigest()[:8]}"
        self.cache_dir, self.log_overflow = cache_dir, log_overflow

    def _cache_path(self, texts):
        key = hashlib.sha1(("\n".join(texts) + self.name + str(self.max_len)).encode()).hexdigest()[:16]
        return Path(self.cache_dir) / f"emb_{self.name}_{key}.npy"

    @torch.no_grad()
    def encode(self, texts):
        """-> (X float32 [n, dim], {attr: predicted labels}) in one pass over length-sorted batches."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), np.float32), {a: c[:0] for a, c in self.classes.items()}
        if self.log_overflow:
            print(f"[FTEmbedder] {100 * share_over(self.tk.tok, texts, self.max_len):.1f}% of {len(texts)} texts "
                  f"over {self.max_len} tokens (truncated)", flush=True)
        ids = self.tk.encode(texts)

        def fn(enc):
            x = self.enc(**enc)
            return torch.cat([x] + list(self.heads(x).values()), -1)

        with autocast(self.device):
            out = run_sorted(fn, self.tk, ids).numpy()
        X, s, pred = out[:, :self.dim].astype(np.float32), self.dim, {}
        for a, k in heads_sizes(self.heads).items():
            pred[a] = self.classes[a][out[:, s:s + k].argmax(-1)]
            s += k
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return X, pred

    def __call__(self, texts):
        texts = list(texts)
        path = self._cache_path(texts) if self.cache_dir else None
        if path is not None and path.exists():
            return np.load(path)
        X = self.encode(texts)[0]
        if path is not None:
            from blindness.embed import save_atomic
            save_atomic(path, X)  # parallel PE runs share the cache
        return X

    def predict(self, texts):
        return self.encode(texts)[1]
