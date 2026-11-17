"""Third-party MAUVE: re-score finished PE runs with a featurizer that neither arm votes with.

Why: quota_pe.py computes MAUVE on sentence-t5 features, the space the t5 arm votes in, so MAUVE
favours the t5 arm (and the vote's blind spots are MAUVE's blind spots). Here the survivors of each
round (round_<t>.csv, column text) and the test set are re-embedded with --model and MAUVE is
recomputed with MAPLE's compute_mauve (same settings as in the loop; only the features change).

Featurizers (--model):
  gpt2-large (default)  last-token hidden state of the final layer, max 1024 tokens: the featurizer of
                        the original MAUVE paper (Pillutla et al., 2021) and of the mauve-text package
  sentence-t5-base      as in quota_pe.py (sanity check: must reproduce the stored `mauve`)
  other HF encoders     mean-pooled last hidden state, max 512 tokens

--length_match: before scoring, subsample each round's survivors so that their length distribution (50-word
bins, as jsd_word_count) matches the test set's, then score; tag gets "-lenmatch". If the arms keep their
MAUVE ranking after length matching, the ranking is not just a length effect. MAUVE depends on the sample
size, so the matched subsample is then cut to --match_n rows (default 500) for every arm; a round with fewer
matched rows keeps them all and logs the smaller "n" (compare such rounds with care).

Output per run: <run>/mauve_<tag>.jsonl, one line per round {"round": t, "mauve_<tag>": value}
(tag = model name after the last '/', e.g. gpt2-large). Resumable: rounds already there are skipped.
summarize.py merges these files in. Test-set features are cached under $PE_DATA/cache.

    python quota_sweep/mauve_rescore.py --root $PE_DATA/quota_sweep/eps4.0 --maple_dir $PE_DATA/MAPLE \
        --test_ref $PE_DATA/quota_sweep/test_ref_qwen.csv [--shard 0 --num_shards 4] [--last_only]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def tag_of(model):
    return model.rstrip("/").split("/")[-1]


class Featurizer:
    def __init__(self, model, batch=16, device=None):
        import torch
        self.torch, self.model_name, self.batch = torch, model, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kind = "gpt2" if "gpt2" in model else ("st" if "sentence-t5" in model else "enc")
        if self.kind == "st":
            from quota_sweep.quota_pe import Embedder
            self.st = Embedder(batch) if self.device == "cuda" else None
            if self.st is None:
                from sentence_transformers import SentenceTransformer
                self.st_cpu = SentenceTransformer(model, device="cpu")
                self.st_cpu.max_seq_length = 1024
            return
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.max_len = 1024 if self.kind == "gpt2" else 512
        self.net = AutoModel.from_pretrained(model).to(self.device).eval()  # fp32, as mauve-text

    def __call__(self, texts):
        texts = [str(t) for t in texts]
        if self.kind == "st":
            if self.st is not None:
                return self.st(texts)
            return self.st_cpu.encode(texts, batch_size=self.batch, convert_to_numpy=True)
        torch = self.torch
        ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        out = None
        with torch.no_grad():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                enc = self.tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(self.device)
                h = self.net(**enc).last_hidden_state.float()
                m = enc["attention_mask"]
                if self.kind == "gpt2":               # hidden state of the last real token
                    f = h[torch.arange(len(idx), device=h.device), m.sum(1) - 1]
                else:
                    mf = m.unsqueeze(-1).float()
                    f = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
                f = f.cpu().numpy().astype(np.float32)
                if out is None:
                    out = np.zeros((len(texts), f.shape[1]), np.float32)
                out[idx] = f
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out


def cached(feat, texts, cache_dir):
    from blindness.embed import save_atomic
    key = hashlib.sha1(("\n".join(map(str, texts)) + feat.model_name + "mauve").encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"mauvefeat_{tag_of(feat.model_name)}_{key}.npy"
    if path.exists():
        return np.load(path)
    X = feat(texts)
    save_atomic(path, X)
    return X


def word_bins(texts, width=50):
    return np.array([int(round(len(str(t).split()) / width)) for t in texts])


def length_matched_idx(q_texts, p_texts, seed=0, width=50):
    """Indices of a subsample of q whose length-bin distribution equals p's (as far as q allows): the largest
    m with m * p_b <= count_q(b) for every bin b that p uses; then round(m * p_b) rows per bin, sampled
    without replacement. Bins q lacks are dropped from the target (and reported via the returned share)."""
    qb, pb = word_bins(q_texts, width), word_bins(p_texts, width)
    bins, cnt = np.unique(pb, return_counts=True)
    prob = cnt / cnt.sum()
    avail = np.array([np.sum(qb == b) for b in bins])
    keep = avail > 0
    covered = float(prob[keep].sum())
    prob = prob[keep] / prob[keep].sum()
    bins, avail = bins[keep], avail[keep]
    m = float(np.min(avail / prob))
    rng = np.random.default_rng(seed)
    idx = []
    for b, pr, av in zip(bins, prob, avail):
        k = min(int(round(m * pr)), int(av))
        idx += list(rng.choice(np.flatnonzero(qb == b), size=k, replace=False))
    return np.sort(np.array(idx, int)), covered


def run_dirs(root, pattern):
    """Run dirs with round files: lam<x>[_seed<s>] (quota_pe.py) or seed<s> (augpe_pe.py)."""
    ds = sorted({f.parent for f in Path(root).glob(f"{pattern}/round_*.csv")})
    return [d for d in ds if re.match(r"(lam[0-9.]+(_seed\d+)?|seed\d+)$", d.name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="sweep root, e.g. $PE_DATA/quota_sweep/eps4.0")
    p.add_argument("--maple_dir", required=True)
    p.add_argument("--test_ref", required=True)
    p.add_argument("--model", default="gpt2-large")
    p.add_argument("--runs", default="*/lam*", help="glob of run dirs under --root")
    p.add_argument("--last_only", action="store_true", help="only the last round of each run")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--length_match", action="store_true", help="length-matched subsample of the survivors")
    p.add_argument("--match_n", type=int, default=500, help="with --length_match: rows scored per round (all arms)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()
    all_dirs = run_dirs(args.root, args.runs)
    if not all_dirs:                     # a job that scores nothing must not look like a success
        sys.exit(f"no run dirs with round_*.csv match {args.root}/{args.runs}")
    from quota_sweep.quota_pe import load_mauve
    compute_mauve = load_mauve(args.maple_dir)
    tag = tag_of(args.model) + ("-lenmatch" if args.length_match else "")
    key = f"mauve_{tag}"
    cache_dir = Path(os.environ.get("PE_DATA", ".")) / "cache"

    dirs = all_dirs[args.shard::args.num_shards]
    todo = []
    for d in dirs:
        out = d / f"{key}.jsonl"
        have = {json.loads(l)["round"] for l in out.read_text().splitlines()} if out.exists() else set()
        rounds = sorted(int(f.stem.split("_")[1]) for f in d.glob("round_*.csv"))
        if (d / "metrics.jsonl").exists():   # only rounds whose metrics line was written (finished rounds)
            rounds = [t for t in rounds if t < len((d / "metrics.jsonl").read_text().splitlines())]
        if args.last_only and rounds:
            rounds = rounds[-1:]
        todo += [(d, t) for t in rounds if t not in have]
    print(f"{len(dirs)} run dirs in shard {args.shard}/{args.num_shards}, {len(todo)} rounds to score with {args.model}",
          flush=True)
    if not todo:
        return
    feat = Featurizer(args.model, args.batch)
    test = pd.read_csv(args.test_ref)
    X_test = cached(feat, test["text"].astype(str).tolist(), cache_dir)
    for d, t in todo:
        t0 = time.time()
        sel = pd.read_csv(d / f"round_{t}.csv")
        extra = {}
        if args.length_match:
            idx, covered = length_matched_idx(sel["text"].astype(str).tolist(), test["text"].astype(str).tolist(),
                                              seed=t)
            n_matched = len(idx)
            if n_matched > args.match_n:
                idx = np.sort(np.random.default_rng(1000 + t).choice(idx, args.match_n, replace=False))
            sel = sel.iloc[idx].reset_index(drop=True)
            extra = {"test_mass_covered": covered, "n_matched": n_matched}
        Xs = feat(sel["text"].astype(str).tolist())
        v = float(compute_mauve(p_features=X_test, q_features=Xs).mauve)
        with open(d / f"{key}.jsonl", "a") as f:
            f.write(json.dumps({"round": t, key: v, "n": len(sel), **extra}) + "\n")
        print(f"{d.relative_to(args.root)} round {t}: {key} = {v:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
