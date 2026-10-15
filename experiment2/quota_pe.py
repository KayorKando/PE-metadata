"""Experiment 2: MAPLE PE with the blind-attribute quota, for one lambda.

Round 0 reuses the pre-vote pool from gen_round0.py (same for every lambda, so runs
differ only in selection). Then for t = 0..T:
  vote (MAPLE: every private point votes for its nearest candidate under phi, + N(0, sigma^2))
  -> quota selection on the blind attributes B (experiment2/quota.py), lambda_t from the schedule
  -> evaluate the K survivors (JSD per attribute vs the test set, MAUVE vs the test set)
  -> VARIATION_API x L on the survivors (MAPLE's prompt) -> next candidates.
Labels used *inside* the loop come from --loop_annotator; evaluation always uses
--eval_annotator (default qwen = ground truth for now). word_count is always measured.

One GPU per run: vLLM (generation + Qwen annotation share one engine) and sentence-t5.
Resumable: finished rounds are reloaded and the run continues after the last one.

Example:
    CUDA_VISIBLE_DEVICES=1 python experiment2/quota_pe.py --maple_dir $PE_DATA/MAPLE \
        --pool_csv $PE_DATA/round0/eps4.0/pool.csv --eps 4.0 --lam 0.5 \
        --test_ref $PE_DATA/exp2/test_ref_qwen.csv --out $PE_DATA/exp2/eps4.0/qwen/lam0.5
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import BIORXIV_SCHEMA, clean_labels  # noqa: E402
from experiment2.annotators import ATTRS, CONTENT, make_annotator  # noqa: E402
from experiment2.quota import assign_cells, build_cell_space, cell_keys, select  # noqa: E402
from vote_correction import jsd, nearest_pool_index  # noqa: E402

METADATA = CONTENT + ["word_count"]   # MAPLE's 9 metadata columns (word_count = requested length)
# MAPLE main_experiment.sh, MAPLE.py runs: eps -> (PE_T, noise multiplier)
EPS_SETTINGS = {"4.0": (4, 2.5787580697142025), "2.0": (4, 4.7482491352519505), "1.0": (2, 6.869194837509232),
                "inf": (9, 0.0)}


def lam_at(t, T, lam, schedule):
    return lam if schedule == "constant" or T == 0 else lam * t / T


def aim_target_labels(aim_df, blind):
    lab = clean_labels(aim_df, [a for a in blind if a != "word_count"])
    if "word_count" in blind:
        lab["word_count"] = ((aim_df["word_count"].astype(float) / 50).round() * 50).astype(int).astype(str).to_numpy()
    return lab


def load_mauve(maple_dir):
    path = Path(maple_dir) / "biorxiv" / "utility_eval" / "compute_mauve.py"
    spec = importlib.util.spec_from_file_location("maple_compute_mauve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_mauve


class Embedder:
    def __init__(self, batch=256):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-t5-base", device="cuda")
        self.m.max_seq_length = 1024
        self.batch = batch

    def __call__(self, texts):
        return self.m.encode(list(texts), batch_size=self.batch, convert_to_numpy=True, show_progress_bar=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maple_dir", required=True)
    p.add_argument("--pool_csv", required=True, help="round-0 candidates before the vote (gen_round0.py)")
    p.add_argument("--eps", default="4.0", choices=list(EPS_SETTINGS))
    p.add_argument("--lam", type=float, required=True, help="lambda (its final value under the linear schedule)")
    p.add_argument("--schedule", default="linear", choices=["linear", "constant"],
                   help="linear: lambda_t = lam * t / T (0 at round 0); constant: lambda_t = lam")
    p.add_argument("--blind", nargs="+", default=["word_count", "sample_size", "dominant_data_type"])
    p.add_argument("--max_cells", type=int, default=30)
    p.add_argument("--loop_annotator", default="qwen", choices=["qwen", "inherited", "classifier"])
    p.add_argument("--eval_annotator", default="qwen", choices=["qwen"])
    p.add_argument("--test_ref", required=True, help="test set with eval-annotator labels (created if missing)")
    p.add_argument("--K", type=int, default=2000)
    p.add_argument("--L", type=int, default=6, help="variations per survivor (MAPLE PE_L_random)")
    p.add_argument("--T", type=int, default=None, help="override MAPLE's number of PE rounds for this eps")
    p.add_argument("--noise", type=float, default=None, help="override MAPLE's noise multiplier for this eps")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu_mem", type=float, default=0.75, help="vLLM gpu_memory_utilization (leave room for sentence-t5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--prep_only", action="store_true", help="only build --test_ref (labels of the test set), then exit")
    args = p.parse_args()

    T, sigma = EPS_SETTINGS[args.eps]
    T = args.T if args.T is not None else T
    sigma = args.noise if args.noise is not None else sigma
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "T": T, "sigma": sigma}, indent=2))
    data_root = Path(os.environ.get("PE_DATA", "."))
    bio = Path(args.maple_dir) / "biorxiv"

    # --- MAPLE pieces ---
    sys.path.insert(0, str(bio / "MAPLE"))
    from MAPLE import create_review_variation_prompt  # noqa: E402
    from util import SuppressChineseLogitsProcessor, set_seed  # noqa: E402
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    set_seed(42 + args.seed)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    llm = LLM(model=args.model_id, max_model_len=32768, gpu_memory_utilization=args.gpu_mem,
              logits_processors=[SuppressChineseLogitsProcessor], seed=args.seed)
    var_sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)

    loop_ann = make_annotator(args.loop_annotator, args.maple_dir, llm=llm, model=args.model_id)
    eval_ann = loop_ann if args.eval_annotator == args.loop_annotator else \
        make_annotator(args.eval_annotator, args.maple_dir, llm=llm, model=args.model_id)
    emb = Embedder()
    compute_mauve = load_mauve(args.maple_dir)

    # --- private voters, AIM target over cells of B, test reference ---
    from blindness.embed import embed
    priv = pd.read_csv(bio / "biorxiv_train_metadata.csv")
    priv = priv[priv["text"].notna()].reset_index(drop=True)
    X_priv = embed(priv["text"].astype(str).tolist(), "sentence-t5-base", 1024, batch_size=256,
                   cache_dir=str(data_root / "cache"))
    aim = pd.read_csv(bio / (f"biorxiv_train_metadata_aim_eps{float(args.eps):.1f}_(1,9).csv"
                             if args.eps != "inf" else "biorxiv_train_metadata.csv"))
    kept, p_hat = build_cell_space(aim_target_labels(aim, args.blind), args.blind, args.max_cells)
    (out / "cells.json").write_text(json.dumps({"blind": args.blind, "p_hat": p_hat}, indent=2))

    test_path = Path(args.test_ref)
    if test_path.exists():
        test = pd.read_csv(test_path)
    else:
        test = pd.read_csv(bio / "biorxiv_test_metadata.csv")
        test = test[test["text"].notna()].reset_index(drop=True)
        tl = eval_ann.label(test)
        for a in ATTRS:
            test[f"ref_{a}"] = tl[a]
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(test_path, index=False)
    if args.prep_only:
        print(f"test reference ready: {test_path} ({len(test)} rows)"); return
    ref = {a: test[f"ref_{a}"].astype(str).to_numpy() for a in ATTRS}
    gem = clean_labels(test, CONTENT)  # MAPLE's Gemini labels of the test set, secondary reference
    X_test = emb(test["text"].astype(str).tolist())

    # --- resume ---
    mpath = out / "metrics.jsonl"
    done = [json.loads(l) for l in mpath.read_text().splitlines()] if mpath.exists() else []
    start = len(done)
    if start > T:
        print("already finished"); return
    if start == 0:
        cand = pd.read_csv(args.pool_csv)
        cand = cand[cand["text"].notna()].reset_index(drop=True)
    else:
        cand = None  # rebuilt below from the last survivors
        sel_df = pd.read_csv(out / f"round_{start - 1}.csv")
        print(f"resuming after round {start - 1}", flush=True)

    def variations(sel_df):
        var = pd.concat([sel_df[METADATA + ["text"]]] * args.L, ignore_index=True)
        prompts = [tok.apply_chat_template(create_review_variation_prompt(r[METADATA].to_dict(), r["text"], []),
                                           tokenize=False, add_generation_prompt=True) for _, r in var.iterrows()]
        outs = llm.generate(prompts, var_sp, use_tqdm=False)
        var["text"] = [o.outputs[0].text for o in outs]
        return pd.concat([sel_df[METADATA + ["text"]], var], ignore_index=True)

    for t in range(start, T + 1):
        t0 = time.time()
        if cand is None:
            cand = variations(sel_df)
        cand = cand[cand["text"].astype(str).str.strip() != ""].reset_index(drop=True)
        # vote
        Xc = emb(cand["text"].astype(str).tolist())
        hist = np.bincount(nearest_pool_index(X_priv, Xc), minlength=len(cand)).astype(float)
        if sigma:
            hist = hist + rng.normal(scale=sigma, size=len(hist))
        # quota selection on loop labels
        lab = loop_ann.label(cand)
        cells = assign_cells(lab, args.blind, kept)
        lam_t = lam_at(t, T, args.lam, args.schedule)
        sel, info = select(hist, cells, p_hat, lam_t, args.K)
        sel_df = cand.iloc[sel].reset_index(drop=True)
        sel_df["vote"] = hist[sel]
        sel_df["cell"] = cells[sel]
        for a in ATTRS:
            sel_df[f"loop_{a}"] = lab[a][sel]
        # evaluate survivors
        ev = eval_ann.label(sel_df)
        for a in ATTRS:
            sel_df[f"eval_{a}"] = ev[a]
        Xs = Xc[sel]
        m = dict(round=t, lam_t=lam_t, n_candidates=len(cand), shortfall=info["shortfall"], moved=info["moved"],
                 mauve=float(compute_mauve(p_features=X_test, q_features=Xs).mauve))
        for a in ATTRS:
            m[f"jsd_{a}"] = jsd(ev[a], ref[a])
        for a in CONTENT:
            m[f"jsd_gemini_{a}"] = jsd(ev[a], gem[a])
        m["jsd_mean"] = float(np.mean([m[f"jsd_{a}"] for a in ATTRS]))
        m["jsd_blind_mean"] = float(np.mean([m[f"jsd_{a}"] for a in args.blind]))
        m["jsd_other_mean"] = float(np.mean([m[f"jsd_{a}"] for a in ATTRS if a not in args.blind]))
        cats = sorted(p_hat)
        ph = np.array([p_hat[c] for c in cats])
        for src, labs in [("loop", lab), ("eval", ev)]:
            lc = assign_cells({a: (labs[a][sel] if src == "loop" else labs[a]) for a in args.blind}, args.blind, kept)
            pc = np.array([np.mean(lc == c) for c in cats])
            m[f"cell_jsd_vs_phat_{src}_labels"] = float(jensenshannon(pc, ph))
        m["seconds"] = round(time.time() - t0, 1)
        sel_df.to_csv(out / f"round_{t}.csv", index=False)
        with open(mpath, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                          if k in ("round", "lam_t", "mauve", "jsd_mean", "jsd_blind_mean", "jsd_other_mean",
                                   "moved", "shortfall", "seconds")}), flush=True)
        cand = None
    (out / "DONE").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
