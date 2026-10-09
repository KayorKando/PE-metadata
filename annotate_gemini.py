"""Annotate a CSV of texts with Gemini, using MAPLE's own annotation prompt.

Same prompt (copied at run time from MAPLE/biorxiv/utility_eval/extract_metadata.py),
model (gemini-2.5-flash-lite) and temperature (1.0) as the labels that ship with
MAPLE's private data, so pool labels and private labels come from one annotator.
That removes the circularity of probe labels (a probe is built on phi, the same
embedding the vote uses).

Differences from MAPLE's script: parallel requests, retries with backoff, and a
resumable JSONL cache, so an interrupted run continues where it stopped.

The API key is read from the GEMINI_API_KEY environment variable (not a flag,
so it stays out of the shell history and the process list).

Example:
    export GEMINI_API_KEY=...            # e.g. in ~/.bashrc, chmod 600
    python annotate_gemini.py --csv $PE_DATA/round0/eps4.0/pool.csv \
        --maple_dir $PE_DATA/MAPLE --out $PE_DATA/round0/eps4.0/pool_gemini.csv
    # quick check first: add --limit 20

Output: the input CSV plus map_<attribute> columns (MAPLE's naming), and
<out>.jsonl with one line per row (the cache).
"""
import argparse
import ast
import json
import os
import random
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ATTRS = ["primary_research_area", "model_organism", "experimental_approach", "dominant_data_type",
         "research_focus_scale", "disease_mention", "sample_size", "research_goal"]


def load_maple_prompt(maple_dir):
    """Take annotate_features_prompt from MAPLE's script without importing it
    (importing would need the deprecated google.generativeai package)."""
    path = Path(maple_dir) / "biorxiv" / "utility_eval" / "extract_metadata.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "annotate_features_prompt")
    ns = {"textwrap": textwrap}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["annotate_features_prompt"]


def make_caller(model, temperature, max_output_tokens):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    safety = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in
              ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT",
               "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT"]]
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens,
                                      safety_settings=safety)

    def call(prompt):
        return client.models.generate_content(model=model, contents=prompt, config=cfg).text
    return call


def parse(text):
    t = (text or "").strip().replace("```json", "").replace("```", "").strip()
    return json.loads(t)


def annotate_one(idx, text, prompt_fn, call, retries=6):
    last = None
    for attempt in range(retries):
        try:
            labels = parse(call(prompt_fn(text=text)))
            return {"idx": idx, "labels": {a: labels.get(a) for a in ATTRS}}
        except Exception as e:  # rate limit, server error, bad JSON
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 2 ** attempt) + random.random())
    return {"idx": idx, "error": last}


def main(call=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--maple_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_output_tokens", type=int, default=1024)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (for a quick check)")
    p.add_argument("--retry_errors", action="store_true", help="retry rows that failed in an earlier run")
    args = p.parse_args()

    if call is None:
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("set GEMINI_API_KEY first")
        call = make_caller(args.model, args.temperature, args.max_output_tokens)
    prompt_fn = load_maple_prompt(args.maple_dir)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_suffix(out.suffix + ".jsonl")

    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["idx"]] = r
    todo = [i for i in range(len(df))
            if i not in done or (args.retry_errors and "error" in done[i])]
    print(f"{len(df)} rows, {len(done)} cached, {len(todo)} to annotate with {args.model}", flush=True)

    t0 = time.time()
    with open(cache, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(annotate_one, i, str(df["text"].iloc[i]), prompt_fn, call) for i in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done[r["idx"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 200 == 0 or k == len(futs):
                rate = k / max(time.time() - t0, 1e-9)
                print(f"{k}/{len(futs)}  {rate:.1f}/s  eta {(len(futs) - k) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    labels = [done.get(i, {}).get("labels") or {} for i in range(len(df))]
    res = pd.concat([df.reset_index(drop=True), pd.DataFrame(labels, columns=ATTRS).add_prefix("map_")], axis=1)
    res.to_csv(out, index=False)
    n_err = sum("error" in done.get(i, {"error": 1}) for i in range(len(df)))
    print(f"saved {out}  ({len(df) - n_err} ok, {n_err} failed; rerun with --retry_errors to retry failures)")


if __name__ == "__main__":
    main()
