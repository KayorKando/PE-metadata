"""phi = the embedding MAPLE votes with: sentence-t5-base, max_seq_length 1024
(MAPLE PE.py: --emb_name sentence-t5-base, --emb_dim 1024 sets max_seq_length)."""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-t5-base"


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed(texts, model_name=DEFAULT_MODEL, max_seq_length=1024, batch_size=32, cache_dir="cache"):
    key = hashlib.sha1(("\n".join(texts) + model_name + str(max_seq_length)).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=_device())
    model.max_seq_length = max_seq_length
    X = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X
