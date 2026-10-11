"""Re-annotating, step 2: label texts with a local Qwen (vLLM) as the annotator.

Uses MAPLE's own annotation prompt (the one behind the Gemini labels of the
private data) and constrains the output to the MAPLE schema with vLLM guided
JSON decoding, so every label is a valid option. Output format matches
annotate_gemini.py: the input CSV plus map_<attribute> columns.

Runs in the MAPLE env (vLLM). Example:
    CUDA_VISIBLE_DEVICES=1 python reannotate/annotate_qwen.py --csv $PE_DATA/round0/eps4.0/pool.csv \
        --maple_dir $PE_DATA/MAPLE --out $PE_DATA/reannotate/pool_qwen.csv
    # a sample of private texts, to measure Qwen-vs-Gemini agreement on real text:
    CUDA_VISIBLE_DEVICES=2 python reannotate/annotate_qwen.py --csv $PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv \
        --maple_dir $PE_DATA/MAPLE --sample 3000 --out $PE_DATA/reannotate/private_qwen.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from annotate_gemini import ATTRS, load_maple_prompt  # noqa: E402
from blindness.schema import BIORXIV_SCHEMA  # noqa: E402

SCHEMA = {"type": "object",
          "properties": {a: {"type": "string", "enum": BIORXIV_SCHEMA[a]} for a in ATTRS},
          "required": ATTRS, "additionalProperties": False}


def guided_params(enable):
    if not enable:
        return {}
    try:  # vLLM 0.10.x
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(json=SCHEMA)}
    except ImportError:
        try:  # newer vLLM
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(json=SCHEMA)}
        except ImportError:
            print("warning: guided decoding not available, falling back to free-form JSON", flush=True)
            return {}


def parse(text):
    t = (text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        d = json.loads(t)
        return {a: d.get(a) for a in ATTRS}
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--maple_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, stable labels")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--sample", type=int, default=0, help="random sample of N rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_guided", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["text"].notna()].reset_index(drop=True)
    df["src_index"] = df.index  # row in the text-filtered input, for joining back
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    prompt_fn = load_maple_prompt(args.maple_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_fn(text=str(t))}],
                                       tokenize=False, add_generation_prompt=True) for t in df["text"]]
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                        **guided_params(not args.no_guided))
    llm = LLM(model=args.model, max_model_len=args.max_model_len)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    secs = time.time() - t0
    labels = [parse(o.outputs[0].text) for o in outs]

    res = pd.concat([df, pd.DataFrame(labels, columns=ATTRS).add_prefix("map_")], axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    n_bad = int(res[[f"map_{a}" for a in ATTRS]].isna().all(axis=1).sum())
    meta = {**vars(args), "n_rows": len(res), "n_unparsed": n_bad, "generate_seconds": round(secs, 1),
            "rows_per_second": round(len(res) / max(secs, 1e-9), 2)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
