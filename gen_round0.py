"""Generate MAPLE's round-0 synthetic pool and save it BEFORE the vote.

Why before the vote: MAPLE's PE_iter_0.csv is the pool after 1-NN voting on
private data and top-K selection. Measuring blindness on it would (a) touch
private data and (b) measure a sample already filtered by phi. The pre-vote
pool only uses AIM-privatized metadata + the public in-context examples, so
picking quota attributes on it is post-processing (no extra privacy cost).

This reuses MAPLE's own prompt and RANDOM_API code (biorxiv/MAPLE/MAPLE.py),
with MAPLE's defaults: Qwen2.5-7B-Instruct, 50 in-context examples (10 picked
per prompt by Hamming distance), temperature 1.0, top_p 0.95, max_tokens 1024,
pool size = PE_num_syn * (PE_L_random + 1) = 2000 * 7 = 14,000.

Run inside MAPLE's environment (vllm + private-evolution). Example, two GPUs:
    CUDA_VISIBLE_DEVICES=1 python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --shard 0 --num_shards 2
    CUDA_VISIBLE_DEVICES=2 python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --shard 1 --num_shards 2
    python gen_round0.py --maple_dir $PE_DATA/MAPLE --eps 4.0 --merge --num_shards 2

Output columns: the 9 requested (AIM) metadata values, plus `text`.
`word_count` is the REQUESTED length; it is copied to `word_count_requested`
because run.py recomputes `word_count` from the generated text.

OpenReview (PE_DATASET=openreview): MAPLE's openreview/MAPLE/MAPLE.py round 0. In-context examples = one
validation review per PE.LABEL_ID + one more at random (50), 10 picked per prompt by Hamming distance;
metadata rows sampled per label id from the AIM file ((L + 1) K_g rows for id g, K_g from the private label
counts, pe_data.num_sample_per_id). The pool keeps PE.LABEL_ID (the class-conditional vote needs it) and
the 8 metadata columns of MAPLE's OpenReview prompt.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_data  # noqa: E402
from pe_data import CLASS_CONDITIONAL, LABEL_ID, METADATA  # noqa: E402


def in_context_examples(maple_dir, num_incontext):
    """bioRxiv: the first num_incontext validation rows (MAPLE). OpenReview: one validation review per label id
    plus one more at random (MAPLE openreview/MAPLE/MAPLE.py; uses the global numpy / random state)."""
    df_val = pe_data.load_val(maple_dir)
    if not CLASS_CONDITIONAL:
        idx = list(range(num_incontext))
    else:
        first = list(df_val.groupby(LABEL_ID).sample(n=1).index)
        rest = list(set(df_val.index) - set(first))
        idx = first + [random.choice(rest)]
    return [(df_val.loc[i, METADATA].to_dict(), df_val.loc[i, "text"]) for i in idx]


def round0_metadata(maple_dir, eps, num_syn, L_random):
    """Metadata rows of the round-0 prompts, sampled as MAPLE does (one label id 0 for bioRxiv)."""
    from util import sample_metadata_for_randomapi  # noqa: E402  (MAPLE's util.py)
    df_trn = pe_data.load_aim(maple_dir, eps)
    if not CLASS_CONDITIONAL:
        df_trn[LABEL_ID] = 0
        return sample_metadata_for_randomapi(df_trn[[LABEL_ID] + METADATA], [0], {0: num_syn}, L_random + 1)
    priv_ids = pe_data.load_private(maple_dir)[LABEL_ID]
    k = pe_data.num_sample_per_id(priv_ids, num_syn)
    return sample_metadata_for_randomapi(df_trn[[LABEL_ID] + METADATA], list(set(priv_ids)), k, L_random + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maple_dir", required=True, help="root of the MAPLE git clone")
    p.add_argument("--eps", default="4.0", help="AIM budget of the metadata file: 1.0, 2.0, 4.0, or inf")
    p.add_argument("--out", default=None, help="default: $PE_DATA/round0/eps<eps>")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--num_incontext", type=int, default=50)
    p.add_argument("--n_sample", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--num_syn", type=int, default=2000)
    p.add_argument("--L_random", type=int, default=6)
    p.add_argument("--rng_seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="only merge finished shards into pool.csv")
    args = p.parse_args()

    out = Path(args.out or Path(os.environ.get("PE_DATA", ".")) / "round0" / f"eps{args.eps}")
    out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = [pd.read_csv(out / f"pool_shard{i}of{args.num_shards}.csv") for i in range(args.num_shards)]
        pool = pd.concat(parts, ignore_index=True)
        pool.to_csv(out / "pool.csv", index=False)
        empty = pool["text"].isna() | (pool["text"].astype(str).str.strip() == "")
        print(f"merged {len(pool)} rows -> {out / 'pool.csv'}  (empty texts: {int(empty.sum())})")
        return

    pe_data.check_data_root()
    sys.path.insert(0, str(pe_data.maple_code(args.maple_dir)))
    from vllm import SamplingParams  # noqa: E402
    from MAPLE import API  # noqa: E402  (MAPLE's own prompt + RANDOM_API)
    from util import set_seed  # noqa: E402

    set_seed(args.rng_seed)

    # Public in-context examples from the validation set, as in MAPLE.
    examples = in_context_examples(args.maple_dir, args.num_incontext)

    # AIM metadata, sampled exactly as MAPLE's round 0 does (same seed -> same rows in every shard).
    meta = round0_metadata(args.maple_dir, args.eps, args.num_syn, args.L_random)
    meta["pool_index"] = np.arange(len(meta))

    shard = meta.iloc[args.shard::args.num_shards].copy().reset_index(drop=True)
    print(f"eps={args.eps}: pool {len(meta)} rows, this shard {args.shard}/{args.num_shards}: {len(shard)} rows")

    # No per-request seed (as in MAPLE): identical metadata rows give identical prompts,
    # and a fixed seed would then give identical texts.
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    api = API(args.model_id, sp, examples, args.n_sample, METADATA)
    shard = api.RANDOM_API(shard)

    shard["word_count_requested"] = shard["word_count"]
    path = out / f"pool_shard{args.shard}of{args.num_shards}.csv"
    (shard if CLASS_CONDITIONAL else shard.drop(columns=[LABEL_ID])).to_csv(path, index=False)
    print(f"saved {len(shard)} rows -> {path}")


if __name__ == "__main__":
    main()
