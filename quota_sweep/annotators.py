"""Annotator module (which one the method will use is still open).

An annotator turns texts into schema labels for the content attributes.
word_count is never annotated: it is always measured from the text (free, exact).

    annotator.label(df) -> {attr: np.array of str}     (schema-cleaned, len(df))

Available:
  inherited   labels = the requested AIM metadata carried by each row (MAPLE's default;
              variations inherit their parent's metadata). No model call.
  qwen        local Qwen via vLLM, MAPLE's annotation prompt, guided JSON over the schema,
              greedy. Treated as the ground-truth annotator for now. Shares the generator's
              vLLM engine when given one (no second model in memory); caches by text, so
              survivors are not re-annotated.
  classifier  PENDING: a small per-attribute classifier trained in reannotate/ (to be chosen
              from the re-annotating results). Raises until it is plugged in.

To add one: subclass Annotator, implement _label_content(texts, df), register in make_annotator.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import BIORXIV_SCHEMA, clean_labels, word_count_bin  # noqa: E402

CONTENT = list(BIORXIV_SCHEMA)
ATTRS = CONTENT + ["word_count"]


class Annotator:
    name = "base"

    def label(self, df):
        texts = df["text"].astype(str).tolist()
        out = self._label_content(texts, df)
        out["word_count"] = word_count_bin(texts)
        return out

    def _label_content(self, texts, df):
        raise NotImplementedError


class InheritedAnnotator(Annotator):
    name = "inherited"

    def _label_content(self, texts, df):
        return clean_labels(df, CONTENT)


class QwenAnnotator(Annotator):
    name = "qwen"

    def __init__(self, maple_dir, llm=None, model="Qwen/Qwen2.5-7B-Instruct", max_tokens=512):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from annotate_gemini import load_maple_prompt
        from reannotate.annotate_qwen import guided_params
        self.prompt_fn = load_maple_prompt(maple_dir)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = llm if llm is not None else LLM(model=model, max_model_len=8192)
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, **guided_params(True))
        self.cache = {}

    def _label_content(self, texts, df):
        from reannotate.annotate_qwen import parse
        todo = sorted(set(t for t in texts if t not in self.cache))
        if todo:
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": self.prompt_fn(text=t)}],
                                                    tokenize=False, add_generation_prompt=True) for t in todo]
            for t, o in zip(todo, self.llm.generate(prompts, self.sp, use_tqdm=False)):
                self.cache[t] = parse(o.outputs[0].text)
        raw = pd.DataFrame([{a: self.cache[t].get(a) for a in CONTENT} for t in texts])
        return clean_labels(raw, CONTENT)  # unparsed -> schema fallback ("Other" / "Not Specified")

    def save_cache(self, path):
        pd.DataFrame([{"text": t, **v} for t, v in self.cache.items()]).to_csv(path, index=False)

    def load_cache(self, path):
        if Path(path).exists():
            for r in pd.read_csv(path).to_dict("records"):
                t = r.pop("text")
                self.cache[str(t)] = {a: (None if pd.isna(v) else v) for a, v in r.items()}


class ClassifierAnnotator(Annotator):
    name = "classifier"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("classifier annotator is pending: choose a model from the reannotate/ "
                                  "results, then load it here and implement _label_content")


def make_annotator(name, maple_dir=None, llm=None, model=None):
    if name == "inherited":
        return InheritedAnnotator()
    if name == "qwen":
        return QwenAnnotator(maple_dir, llm=llm, **({"model": model} if model else {}))
    if name == "classifier":
        return ClassifierAnnotator()
    raise ValueError(f"unknown annotator {name}")
