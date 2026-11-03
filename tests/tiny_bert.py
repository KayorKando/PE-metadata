"""A tiny randomly initialised BERT + word-level tokenizer saved to a local dir (no download).
hidden_size defaults to 768 so the normsoftmax initial-loss check sees BiomedBERT's dimension."""
from pathlib import Path

WORDS = ("cell gene protein mouse human tumor neuron brain cancer virus plant yeast data model "
         "method study cohort sample review clinical imaging sequence structure network tissue "
         "population disease immune signal growth binding pathway expression mutation the of and a "
         "in we show that is to with for by on").split()


def make_tiny_bert(path, hidden=768, layers=1, heads=4, max_pos=128, seed=0):
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True, model_max_length=max_pos)
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                     intermediate_size=64, max_position_embeddings=max_pos)
    BertModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def random_texts(n, seed=0, lo=3, hi=40):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, rng.integers(lo, hi))) for _ in range(n)]
