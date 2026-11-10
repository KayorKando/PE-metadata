"""Vote embedding, step 1: fine-tune BiomedBERT once on the round-0 pool with Qwen labels,
so that the embedding (heads removed) can replace sentence-t5 as the PE vote embedding.

Model (one shared backbone for all 8 content attributes of BIORXIV_SCHEMA):
  z = mean of the last hidden states over real tokens (as embed_frozen_bert)
  --loss normsoftmax (Zhai & Wu, BMVC 2019):
      x = LN(z) / ||LN(z)||   (LayerNorm without affine; no new FC layer)
      logit_a = x . P~_a^T / sigma,  rows of P_a L2-normalised, no bias, sigma = 0.05 fixed
      vote with x. For unit vectors ||x_i - x_j||^2 = 2 - 2 x_i.x_j, so the L2 vote sees the
      cosine geometry the loss organises; logits are bounded by +-1/sigma = +-20.
  --loss ce (control arm, the "just use a linear classifier" idea):
      logit_a = W_a z + b_a on the un-normalised mean-pool; vote with z.
Backbone: any HF encoder (default BiomedBERT), or a T5 encoder, e.g. --model
sentence-transformers/sentence-t5-base (the "ft-t5" arm: fine-tune MAPLE's own vote embedding, so the
vote and MAUVE start from the same space). For a sentence-transformers repo, its Dense layer after the
mean-pool (sentence-t5: 768 -> 768, identity activation) is kept, so at init z equals sentence-t5's
pre-normalisation embedding; the Dense weights are trained with the backbone and saved in heads.pt.
Both arms: loss = sum over attributes of the mean cross-entropy over rows with a label
(missing label -> that attribute's term skipped for that row). Same everything else.

Training: stage 1 = heads only (backbone frozen), 1 epoch, lr 1e-3 (random proxies barely move
at 2e-5). Stage 2 = all parameters, backbone lr 2e-5, heads lr 10x, AdamW wd 0.01, 10% warmup,
3 epochs, batch 32, max_len 512, clip 1.0, bf16; best epoch by mean val kappa (nearest proxy /
argmax prediction). Split = re-annotating's group_split (70/15/15 by prompt group), seed 0.
No class weighting. word_count is not a target (left to the quota).

Privacy: trains on the round-0 synthetic pool + Qwen labels only (post-processing of DP output).
Private texts are evaluation only (check_embedding.py) and must not be used to pick settings.

Example:
    python vote_embedding/train_embedder.py --loss normsoftmax \
      --pool_qwen $PE_DATA/reannotate/eps4.0/pool_qwen.csv
    # -> $PE_DATA/vote_embedding/eps4.0/normsoftmax/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from blindness.schema import BIORXIV_SCHEMA  # noqa: E402
from reannotate.train_classifier import (FineTuned, HF_MODELS, REQUESTED_COLS,  # noqa: E402
                                         group_split, labels_from_map)

ATTRS = list(BIORXIV_SCHEMA)          # the 8 content attributes (no word_count)
SIGMA = 0.05
LOSSES = ("normsoftmax", "ce")
MISSING = -1                          # label index for "no label" (or class unseen in train)


# ---------------- model ----------------
def _repo_file(name, rel):
    """Local path of file rel in a local model dir or a (cached) HF repo; raises if it is not there."""
    if Path(name).is_dir():
        f = Path(name) / rel
        if not f.exists():
            raise FileNotFoundError(f)
        return str(f)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(name, rel)


def st_dense(name):
    """The Dense layer of a sentence-transformers model (after the pooling), as nn.Linear; None if the
    model has none (plain HF models, or no modules.json). Only identity activations are supported, and
    the pooling, if declared, must be mean pooling (what SharedEncoder does)."""
    try:
        modules = json.loads(Path(_repo_file(name, "modules.json")).read_text())
    except Exception:
        return None
    for m in modules:
        if m["type"].endswith("models.Pooling"):
            pc = json.loads(Path(_repo_file(name, f"{m['path']}/config.json")).read_text())
            if not pc.get("pooling_mode_mean_tokens", False):
                raise ValueError(f"{name}: only mean pooling is supported, got {pc}")
    dense = [m for m in modules if m["type"].endswith("models.Dense")]
    if not dense:
        return None
    if len(dense) > 1:
        raise ValueError(f"{name}: more than one Dense module")
    d = dense[0]["path"]
    cfg = json.loads(Path(_repo_file(name, f"{d}/config.json")).read_text())
    act = cfg.get("activation_function", "torch.nn.modules.linear.Identity")
    if not act.endswith("Identity"):
        raise ValueError(f"{name}: Dense activation {act} not supported")
    try:
        from safetensors.torch import load_file
        sd = load_file(_repo_file(name, f"{d}/model.safetensors"))
    except Exception:
        sd = torch.load(_repo_file(name, f"{d}/pytorch_model.bin"), map_location="cpu", weights_only=True)
    lin = nn.Linear(cfg["in_features"], cfg["out_features"], bias=cfg.get("bias", True))
    lin.load_state_dict({k.split("linear.", 1)[-1]: v for k, v in sd.items()})
    return lin


class SharedEncoder(nn.Module):
    """Backbone + mean-pool [+ the sentence-transformers Dense, if any]; with normalize=True also
    LN (no affine) + L2 -> unit vector x. T5 checkpoints load as T5EncoderModel (encoder only)."""

    def __init__(self, name, normalize, proj=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel, T5EncoderModel
        cfg = AutoConfig.from_pretrained(name)
        if cfg.model_type in ("t5", "mt5"):
            self.backbone = T5EncoderModel.from_pretrained(name)
            hidden = cfg.d_model
        else:
            self.backbone = AutoModel.from_pretrained(name)
            hidden = cfg.hidden_size
        # proj: None -> look for a sentence-transformers Dense; False -> none; nn.Linear -> use it
        self.proj = st_dense(name) if proj is None else (proj or None)
        self.dim = self.proj.out_features if self.proj is not None else hidden
        self.normalize = normalize
        self.ln = nn.LayerNorm(self.dim, elementwise_affine=False) if normalize else None

    def forward(self, **enc):
        h = self.backbone(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.proj is not None:
            z = F.linear(z, self.proj.weight.float(), None if self.proj.bias is None else self.proj.bias.float())
        if self.normalize:
            z = F.normalize(self.ln(z), dim=-1)
        return z

    embed = forward  # the vote embedding: x (normsoftmax) or z (ce)


class ProxyHeads(nn.Module):
    """One proxy matrix per attribute; logits = x . normalize(P_a)^T / sigma, no bias."""

    def __init__(self, dim, n_classes, sigma=SIGMA):
        super().__init__()
        self.sigma = sigma
        self.attrs = list(n_classes)  # ParameterDict sorts its keys: keep the schema order explicitly
        self.P = nn.ParameterDict({a: nn.Parameter(torch.randn(k, dim)) for a, k in n_classes.items()})

    def proxies(self, a):
        return F.normalize(self.P[a], dim=-1)

    def forward(self, x):
        return {a: x @ self.proxies(a).T / self.sigma for a in self.attrs}


class LinearHeads(nn.Module):
    """Control arm: plain linear heads with bias on the un-normalised mean-pool."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.attrs = list(n_classes)
        self.W = nn.ModuleDict({a: nn.Linear(dim, k) for a, k in n_classes.items()})

    def forward(self, z):
        return {a: self.W[a](z) for a in self.attrs}


def make_model(name, loss, n_classes, sigma=SIGMA, proj=None):
    enc = SharedEncoder(name, normalize=(loss == "normsoftmax"), proj=proj)
    heads = ProxyHeads(enc.dim, n_classes, sigma) if loss == "normsoftmax" else LinearHeads(enc.dim, n_classes)
    return enc, heads


def multitask_loss(logits, Y, mask):
    """logits: {attr: [B, k_a]} in the column order of Y; Y: [B, A] long class indices;
    mask: [B, A] bool (False = no label).
    Sum over attributes of the mean CE over labelled rows; an attribute with no labelled row
    in the batch contributes nothing (and gets no gradient)."""
    total = 0.0
    for j, a in enumerate(logits):
        ok = mask[:, j]
        if ok.any():
            total = total + F.cross_entropy(logits[a][ok].float(), Y[ok, j])
    return total


# ---------------- data ----------------
def encode_labels(lab, rows, classes):
    """lab: {attr: labels, None = missing}; -> Y [n, A] with MISSING for no label / unseen class."""
    Y = np.full((len(rows), len(classes)), MISSING, np.int64)
    for j, (a, cs) in enumerate(classes.items()):
        cid = {c: i for i, c in enumerate(cs)}
        Y[:, j] = [cid.get(v, MISSING) if v is not None else MISSING for v in lab[a][rows]]
    return Y


def run_sorted(fn, tk, ids):
    """fn(enc) -> tensor [b, ...] over length-sorted batches; returns rows in the original order."""
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = [None] * len(ids)
    for s, enc in zip(range(0, len(order), tk.batch), tk._batches(ids, order)):
        res = fn(enc).float().cpu()
        for r, i in zip(res, order[s:s + tk.batch]):
            out[i] = r
    return torch.stack(out) if out else torch.zeros(0)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")


@torch.no_grad()
def predict_ids(enc_model, heads, tk, ids):
    """-> {attr: predicted class index} (nearest proxy for normsoftmax, argmax for ce)."""
    enc_model.eval(); heads.eval()
    with autocast(tk.device):
        L = run_sorted(lambda e: torch.cat(list(heads(enc_model(**e)).values()), -1), tk, ids)
    out, s = {}, 0
    for a, k in heads_sizes(heads).items():
        out[a] = L[:, s:s + k].argmax(-1).numpy()
        s += k
    return out


def heads_sizes(heads):
    if isinstance(heads, ProxyHeads):
        return {a: heads.P[a].shape[0] for a in heads.attrs}
    return {a: heads.W[a].out_features for a in heads.attrs}


def val_kappas(enc_model, heads, tk, ids, Y, classes):
    pr = predict_ids(enc_model, heads, tk, ids)
    out = {}
    for j, a in enumerate(classes):
        ok = Y[:, j] != MISSING
        out[a] = float(cohen_kappa_score(Y[ok, j], pr[a][ok])) if ok.any() else float("nan")
    return out


# ---------------- training ----------------
def train(enc_model, heads, tk, ids_tr, Y_tr, ids_va, Y_va, classes, seed=0, stage1_epochs=1,
          stage1_lr=1e-3, epochs=3, lr=2e-5, head_lr_mult=10.0, wd=0.01, log=print):
    """Stage 1 (heads only) -> stage 2 (all params); keeps the stage-2 epoch with best mean val kappa."""
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    dev, B = tk.device, tk.batch
    enc_model.to(dev); heads.to(dev)
    rng = np.random.default_rng(seed)
    Yt = torch.tensor(Y_tr)
    hist = {"stage1": [], "stage2": [], "first_loss": None,
            "sum_ln_k": float(sum(np.log(len(c)) for c in classes.values()))}

    def epoch(params, opt, sched, frozen):
        enc_model.train(not frozen); heads.train()
        perm = rng.permutation(len(ids_tr))
        losses = []
        for s, enc in zip(range(0, len(perm), B), tk._batches(ids_tr, perm)):
            yb = Yt[perm[s:s + B]].to(dev)
            with autocast(dev):
                if frozen:
                    with torch.no_grad():
                        x = enc_model(**enc)
                else:
                    x = enc_model(**enc)
                loss = multitask_loss(heads(x), yb.clamp(min=0), yb != MISSING)
            if not torch.is_tensor(loss):
                continue
            if hist["first_loss"] is None:
                hist["first_loss"] = float(loss.detach())
                log(f"first-batch loss {hist['first_loss']:.2f} (sum_a ln k_a = {hist['sum_ln_k']:.2f})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            if sched is not None:
                sched.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses)) if losses else float("nan")

    # stage 1: heads only
    for p in enc_model.parameters():
        p.requires_grad_(False)
    hp = list(heads.parameters())
    opt = torch.optim.AdamW(hp, lr=stage1_lr, weight_decay=wd)
    for ep in range(stage1_epochs):
        tl = epoch(hp, opt, None, frozen=True)
        vk = val_kappas(enc_model, heads, tk, ids_va, Y_va, classes)
        hist["stage1"].append({"epoch": ep, "train_loss": tl, "val_kappa": vk, "val_mean_kappa": _mean(vk)})
        log(f"stage1 ep{ep}: train loss {tl:.3f}, val mean kappa {_mean(vk):.3f}")

    # stage 2: everything, heads at 10x lr
    for p in enc_model.parameters():
        p.requires_grad_(True)
    bp = [p for p in enc_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bp, "lr": lr}, {"params": hp, "lr": lr * head_lr_mult}], weight_decay=wd)
    steps = epochs * int(np.ceil(len(ids_tr) / B))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = None
    for ep in range(epochs):
        tl = epoch(bp + hp, opt, sched, frozen=False)
        vk = val_kappas(enc_model, heads, tk, ids_va, Y_va, classes)
        mk = _mean(vk)
        hist["stage2"].append({"epoch": ep, "train_loss": tl, "val_kappa": vk, "val_mean_kappa": mk})
        log(f"stage2 ep{ep}: train loss {tl:.3f}, val mean kappa {mk:.3f}")
        if best is None or mk > best[0]:
            best = (mk, ep, _cpu_state(enc_model), _cpu_state(heads))
    if best is not None:
        enc_model.load_state_dict(best[2]); heads.load_state_dict(best[3])
        hist["best_epoch"], hist["best_val_mean_kappa"] = best[1], best[0]
    return hist


def _mean(d):
    v = [x for x in d.values() if x == x]
    return float(np.mean(v)) if v else float("nan")


def _cpu_state(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


# ---------------- save / load ----------------
def save(path, enc_model, heads, tok, classes, config, hist=None):
    path = Path(path)
    (path / "backbone").mkdir(parents=True, exist_ok=True)
    enc_model.backbone.save_pretrained(path / "backbone")
    tok.save_pretrained(path / "backbone")
    ck = {"heads": _cpu_state(heads), "classes": classes}
    if enc_model.proj is not None:  # sentence-transformers Dense (trained with the backbone)
        ck["proj"] = _cpu_state(enc_model.proj)
        ck["proj_shape"] = [enc_model.proj.in_features, enc_model.proj.out_features, enc_model.proj.bias is not None]
    torch.save(ck, path / "heads.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2))
    if hist is not None:
        (path / "train_log.json").write_text(json.dumps(hist, indent=2))


def load(path, device="cpu"):
    """-> (encoder, heads, classes, config), in eval mode on device."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    ck = torch.load(path / "heads.pt", map_location="cpu", weights_only=False)
    classes = ck["classes"]
    proj = False
    if "proj" in ck:
        i, o, b = ck["proj_shape"]
        proj = nn.Linear(i, o, bias=b)
        proj.load_state_dict(ck["proj"])
    enc_model, heads = make_model(str(path / "backbone"), cfg["loss"], {a: len(c) for a, c in classes.items()},
                                  cfg.get("sigma", SIGMA), proj=proj)
    heads.load_state_dict(ck["heads"])
    return enc_model.to(device).eval(), heads.to(device).eval(), classes, cfg


def share_over(tok, texts, max_len):
    """Share of texts longer than max_len tokens (these are truncated)."""
    n = [len(x) for x in tok(list(texts), truncation=False, verbose=False)["input_ids"]]
    return float(np.mean(np.array(n) > max_len)) if n else 0.0


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_qwen", required=True, help="annotate_qwen.py output on the round-0 pool")
    p.add_argument("--loss", default="normsoftmax", choices=LOSSES)
    p.add_argument("--model", default=HF_MODELS["biomedbert"],
                   help="HF backbone name or local path; sentence-transformers/sentence-t5-base for the ft-t5 arm")
    p.add_argument("--out", default=None, help="default $PE_DATA/vote_embedding/eps<eps>/<loss>")
    p.add_argument("--eps", default="4.0")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage1_epochs", type=int, default=1)
    p.add_argument("--stage1_lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr_mult", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    args = p.parse_args()
    out = Path(args.out or Path(os.environ.get("PE_DATA", ".")) / "vote_embedding" / f"eps{args.eps}" / args.loss)

    t0 = time.time()
    pool = pd.read_csv(args.pool_qwen)
    lab = labels_from_map(pool)
    groups = pool[[c for c in REQUESTED_COLS if c in pool]].astype(str).agg("|".join, axis=1).to_numpy()
    tr, va, te = group_split(groups, args.seed)
    # classes absent from the training split get no proxy (as FineTuned)
    classes = {a: sorted(set(v for v in lab[a][tr] if v is not None)) for a in ATTRS}
    Y_tr, Y_va = encode_labels(lab, tr, classes), encode_labels(lab, va, classes)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = FineTuned(args.model, args.seed, args.epochs, args.lr, args.batch_size, args.max_len, dev)
    texts = pool["text"].astype(str).tolist()
    ids = tk.encode(texts)
    over = share_over(tk.tok, texts, args.max_len)
    print(f"{len(texts)} pool texts (train {len(tr)}, val {len(va)}, test {len(te)}); "
          f"{100 * over:.1f}% over {args.max_len} tokens (truncated); classes per attribute "
          + str({a: len(c) for a, c in classes.items()}), flush=True)

    enc_model, heads = make_model(args.model, args.loss, {a: len(c) for a, c in classes.items()}, args.sigma)
    hist = train(enc_model, heads, tk, [ids[i] for i in tr], Y_tr, [ids[i] for i in va], Y_va, classes,
                 seed=args.seed, stage1_epochs=args.stage1_epochs, stage1_lr=args.stage1_lr, epochs=args.epochs,
                 lr=args.lr, head_lr_mult=args.head_lr_mult, log=lambda s: print(s, flush=True))
    hist.update(share_over_max_len=over, train_seconds=round(time.time() - t0, 1),
                n_train=len(tr), n_val=len(va), n_test=len(te))
    config = {"loss": args.loss, "sigma": args.sigma, "max_len": args.max_len, "base_model": args.model,
              "attributes": ATTRS, "seed": args.seed, "split": "group_split(70/15/15)",
              "pool_qwen": str(args.pool_qwen), **{k: getattr(args, k) for k in
                                                   ("stage1_epochs", "stage1_lr", "epochs", "lr", "head_lr_mult",
                                                    "batch_size")}}
    save(out, enc_model, heads, tk.tok, classes, config, hist)
    np.savez(out / "split.npz", train=tr, val=va, test=te)
    print(f"saved to {out} (best stage-2 epoch {hist.get('best_epoch')}, "
          f"val mean kappa {hist.get('best_val_mean_kappa', float('nan')):.3f}, {hist['train_seconds']}s)")


if __name__ == "__main__":
    main()
