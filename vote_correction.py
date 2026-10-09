"""Experiment 1b: does MAPLE's vote correct each attribute's distribution?

Experiment 1 measured a proxy (5-NN label agreement inside one dataset).
This measures the real thing. Blindness alone does not create drift: if the
vote ignores attribute a, selection is random w.r.t. a and keeps a's
distribution. What blindness removes is the *restoring force* when the
generator's pool is already off. So for each attribute we ask how much of
the pool's distance to the private distribution the vote closes.

Vote = MAPLE's Voting_histogram: every private point gives 1 vote to its
nearest pool point (L2, sentence-t5-base, max_seq_length 1024); optional
Gaussian noise on the counts; keep the top K (mode 'rank').

Per attribute (JSD = scipy jensenshannon, natural log, as in MAPLE's
compute_mauve_jsd.py; target = the private set the vote uses):
  jsd_pool    : whole pre-vote pool vs private (what the generator gives)
  jsd_random  : K random pool points vs private (a fully blind selection)
  jsd_vote_s  : top-K by the vote with noise multiplier s
  jsd_floor   : K random *private* points vs private (best a size-K sample can do)
  correction_s = (jsd_random - jsd_vote_s) / (jsd_random - jsd_floor)
      0 = no better than blind, 1 = as good as a perfect sample, < 0 = worse.
      NaN when the pool is already within noise of a perfect sample (nothing to correct).

Labels: private = Gemini labels mapped to the schema; pool = requested (AIM)
labels mapped to the schema. word_count is measured from the text on both
sides; word_count_requested (pool) is compared with private measured length.

Embeddings come from the same cache as run.py (same texts -> cache hit).

Example (cluster):
    python vote_correction.py \
      --private_csv $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
      --pool_csv $PE_DATA/round0/eps4.0/pool.csv \
      --exp1_summary $PE_DATA/results/round0_eps4.0/summary.csv \
      --out $PE_DATA/results/vote_correction_eps4.0
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from blindness.schema import BIORXIV_ATTRIBUTES, clean_labels

LENGTH_ATTRS = {"word_count", "word_count_requested"}
# MAPLE main_experiment.sh noise multipliers (MAPLE.py, eps 4 / 2 / 1)
DEFAULT_NOISES = [0.0, 2.5787580697142025, 4.7482491352519505, 6.869194837509232]


def jsd(a, b):
    """MAPLE's compute_category_jsd on two label arrays."""
    ca, cb = Counter(a), Counter(b)
    cats = sorted(set(ca) | set(cb))
    pa = np.array([ca.get(c, 0) for c in cats], float) / len(a)
    pb = np.array([cb.get(c, 0) for c in cats], float) / len(b)
    return float(jensenshannon(pa, pb))


def nearest_pool_index(X_priv, X_pool, batch=4096):
    """For every private point, the index of its nearest pool point (L2)."""
    try:
        import torch
        if torch.cuda.is_available():
            P = torch.from_numpy(X_pool).float().cuda()
            pn = (P * P).sum(1)
            out = []
            for i in range(0, len(X_priv), batch):
                Q = torch.from_numpy(X_priv[i:i + batch]).float().cuda()
                d = pn[None, :] - 2 * Q @ P.T  # |q|^2 is constant per row
                out.append(d.argmin(1).cpu().numpy())
            return np.concatenate(out)
    except ImportError:
        pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X_pool)
    return nn.kneighbors(X_priv, return_distance=False)[:, 0]


def select_top_k(hist, K, noise, rng):
    h = hist.astype(np.float64)
    if noise:
        h = h + rng.normal(scale=noise, size=len(h))
    return np.argsort(h)[::-1][:K]  # as in MAPLE get_next_iteration(mode='rank')


def analyze(X_priv, X_pool, lab_priv, lab_pool, K=2000, noises=(0.0,), seeds=10, seed=0):
    """lab_pool[a] is compared with lab_priv[a] (or lab_priv['word_count'] for
    word_count_requested)."""
    rng = np.random.default_rng(seed)
    nn = nearest_pool_index(X_priv, X_pool)
    hist = np.bincount(nn, minlength=len(X_pool))

    sel_vote = {}
    for s in noises:
        reps = 1 if s == 0 else seeds
        sel_vote[s] = [select_top_k(hist, K, s, rng) for _ in range(reps)]
    sel_rand = [rng.choice(len(X_pool), K, replace=False) for _ in range(seeds)]
    sel_floor = [rng.choice(len(X_priv), K, replace=False) for _ in range(seeds)]

    rows = []
    for a, y_pool in lab_pool.items():
        y_priv = lab_priv["word_count" if a == "word_count_requested" else a]
        r = dict(attribute=a, kind="length" if a in LENGTH_ATTRS else "content",
                 jsd_pool=jsd(y_pool, y_priv))
        rand = [jsd(y_pool[i], y_priv) for i in sel_rand]
        flo = [jsd(y_priv[i], y_priv) for i in sel_floor]
        r.update(jsd_random=np.mean(rand), jsd_random_sd=np.std(rand),
                 jsd_floor=np.mean(flo))
        # If a blind pick is already about as close as a perfect sample, there is
        # nothing to correct and the ratio is just noise: report NaN.
        denom = r["jsd_random"] - r["jsd_floor"]
        ok = denom > 2 * r["jsd_random_sd"]
        for s, sels in sel_vote.items():
            v = [jsd(y_pool[i], y_priv) for i in sels]
            r[f"jsd_vote_{s:g}"] = np.mean(v)
            r[f"correction_{s:g}"] = (r["jsd_random"] - np.mean(v)) / denom if ok else np.nan
        rows.append(r)

    stats = dict(n_private=int(len(X_priv)), n_pool=int(len(X_pool)), K=int(K),
                 pool_points_with_votes=int((hist > 0).sum()),
                 max_votes=int(hist.max()),
                 votes_held_by_top_K=float(np.sort(hist)[::-1][:K].sum() / hist.sum()))
    return pd.DataFrame(rows).set_index("attribute"), stats, hist


def plot(df, noises, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("correction_0").fillna({f"correction_{s:g}": 0.0 for s in noises})
    y = np.arange(len(d))
    h = 0.8 / len(noises)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.5))
    for j, s in enumerate(noises):
        ax.barh(y + (j - (len(noises) - 1) / 2) * h, d[f"correction_{s:g}"], height=h,
                label=f"noise σ = {s:.2f}" if s else "no noise")
    ax.axvline(0, color="#666", lw=1)
    ax.axvline(1, color="#999", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} ({k})" for a, k in zip(d.index, d.kind)], fontsize=8)
    ax.set_xlabel("vote correction  (0 = like a blind random pick, 1 = like a perfect sample)")
    ax.set_title("How much the vote pulls each attribute toward the private distribution")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private_csv", required=True)
    p.add_argument("--pool_csv", required=True)
    p.add_argument("--exp1_summary", default=None, help="run.py summary.csv on the pool, to compare with k-NN kappa")
    p.add_argument("--K", type=int, default=2000)
    p.add_argument("--noises", type=float, nargs="+", default=DEFAULT_NOISES)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="sentence-t5-base")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    data_root = Path(os.environ.get("PE_DATA", "."))
    p.add_argument("--cache_dir", default=str(data_root / "cache"))
    p.add_argument("--out", default=str(data_root / "results" / "vote_correction"))
    args = p.parse_args()

    from blindness.embed import embed

    priv = pd.read_csv(args.private_csv)
    priv = priv[priv["text"].notna()].reset_index(drop=True)
    pool = pd.read_csv(args.pool_csv)
    pool = pool[pool["text"].notna()].reset_index(drop=True)

    lab_priv = clean_labels(priv, BIORXIV_ATTRIBUTES)
    lab_pool = clean_labels(pool, BIORXIV_ATTRIBUTES)
    if "word_count_requested" in pool.columns:
        lab_pool["word_count_requested"] = (
            (pool["word_count_requested"].astype(float) / 50).round() * 50).astype(int).astype(str).to_numpy()

    kw = dict(model_name=args.model, max_seq_length=args.max_seq_length,
              batch_size=args.batch_size, cache_dir=args.cache_dir)
    X_priv = embed(priv["text"].astype(str).tolist(), **kw)
    X_pool = embed(pool["text"].astype(str).tolist(), **kw)

    df, stats, hist = analyze(X_priv, X_pool, lab_priv, lab_pool, K=args.K,
                              noises=args.noises, seeds=args.seeds, seed=args.seed)

    if args.exp1_summary and Path(args.exp1_summary).exists():
        e1 = pd.read_csv(args.exp1_summary, index_col=0)
        df = df.join(e1[["knn_kappa", "probe_kappa"]])
        for s in args.noises:
            both = df.dropna(subset=["knn_kappa", f"correction_{s:g}"])
            for name, sub in [("all", both), ("content", both[both.kind == "content"])]:
                if len(sub) >= 3:
                    rho, pv = spearmanr(sub.knn_kappa, sub[f"correction_{s:g}"])
                    stats[f"spearman_knn_vs_correction_{s:g}_{name}"] = [float(rho), float(pv)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("correction_0")
    df.to_csv(out / "vote_correction.csv")
    np.save(out / "vote_histogram.npy", hist)
    plot(df, args.noises, out / "vote_correction.png")
    (out / "stats.json").write_text(json.dumps({**vars(args), **stats}, indent=2))

    pd.set_option("display.width", 200)
    cols = ["kind", "jsd_pool", "jsd_random", "jsd_floor"] + [f"jsd_vote_{s:g}" for s in args.noises] \
        + [f"correction_{s:g}" for s in args.noises] + [c for c in ["knn_kappa"] if c in df]
    print(df[cols].round(3).to_string())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
