"""Step 1 of the fine-tuned vote embedding (vote_embedding/train_embedder.py), on a tiny local BERT:
normsoftmax initial loss ~ sum_a ln k_a; proxies and x have norm 1; missing labels give no
gradient; save/load gives identical embeddings; both arms train end to end."""
import numpy as np
import pytest
import torch

from blindness.schema import BIORXIV_SCHEMA
from reannotate.train_classifier import FineTuned
from tests.tiny_bert import make_tiny_bert, random_texts
from vote_embedding.train_embedder import (MISSING, ProxyHeads, encode_labels, load, make_model, multitask_loss,
                                           predict_ids, save, train)

K = {a: len(c) for a, c in BIORXIV_SCHEMA.items()}  # 8 attributes, 99 classes


@pytest.fixture(scope="module")
def bert(tmp_path_factory):
    return make_tiny_bert(tmp_path_factory.mktemp("tiny_bert"))


def tk_for(name, batch=16):
    return FineTuned(name, 0, 1, 2e-5, batch, 64, "cpu")


def batch(tk, texts):
    ids = tk.encode(texts)
    return next(tk._batches(ids, np.arange(len(ids))))


def test_schema_has_99_classes():
    assert list(K.values()) == [28, 17, 9, 12, 9, 10, 6, 8] and sum(K.values()) == 99
    assert abs(sum(np.log(k) for k in K.values()) - 19.2) < 0.05


def test_initial_loss_and_unit_norms(bert):
    """At init the cosines x.p are ~N(0, 1/d), so logits ~N(0, s^2) with s = 1/(sigma sqrt d) = 0.72
    for d = 768: E[loss] ~ sum_a ln k_a + ~s^2/2 per attribute ~ 19.2 + 2. Without the LN + L2
    (or with the LN misplaced) the logits are ~sqrt(d)/sigma times larger and the loss explodes."""
    tk = tk_for(bert, batch=64)
    torch.manual_seed(0)
    enc, heads = make_model(bert, "normsoftmax", K)
    enc.eval()
    b = batch(tk, random_texts(64))
    with torch.no_grad():
        x = enc(**b)
        assert torch.allclose(x.norm(dim=-1), torch.ones(len(x)), atol=1e-5)
        for a in K:
            assert torch.allclose(heads.proxies(a).norm(dim=-1), torch.ones(K[a]), atol=1e-5)
        rng = np.random.default_rng(0)
        Y = torch.tensor(np.stack([rng.integers(0, K[a], len(x)) for a in K], 1))
        loss = float(multitask_loss(heads(x), Y, torch.ones_like(Y, dtype=torch.bool)))
    ln_k = sum(np.log(k) for k in K.values())
    print(f"initial loss {loss:.2f}, sum ln k {ln_k:.2f}")
    assert ln_k - 0.5 < loss < ln_k + 3.5
    lg = heads(x)
    assert all(v.abs().max() <= 1 / heads.sigma + 1e-4 for v in lg.values())  # bounded logits


def test_missing_labels_give_no_gradient():
    torch.manual_seed(0)
    k = {"a": 5, "b": 3}
    heads = ProxyHeads(16, k)
    x = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1).requires_grad_(True)
    Y = torch.tensor([[1, 0], [2, 1], [0, 2], [4, 0]])
    mask = torch.tensor([[True, True], [True, False], [True, False], [False, False]])

    def grads(Y):
        x.grad = None; heads.zero_grad()
        multitask_loss(heads(x), Y, mask).backward()
        return x.grad.clone(), heads.P["a"].grad.clone(), heads.P["b"].grad.clone()

    gx, ga, gb = grads(Y)
    assert torch.all(gx[3] == 0)                       # row 3 has no label at all
    Y2 = Y.clone(); Y2[1, 1] = 2; Y2[3] = torch.tensor([0, 1])  # change only masked entries
    gx2, ga2, gb2 = grads(Y2)
    assert torch.equal(gx, gx2) and torch.equal(ga, ga2) and torch.equal(gb, gb2)
    # attribute with no label in the batch: no gradient on its proxies
    mask_b_off = mask.clone(); mask_b_off[:, 1] = False
    heads.zero_grad(); x.grad = None
    multitask_loss(heads(x), Y, mask_b_off).backward()
    assert heads.P["b"].grad is None or torch.all(heads.P["b"].grad == 0)


def test_encode_labels_missing_and_unseen():
    lab = {"a": np.array(["x", None, "z", "y"], dtype=object)}
    Y = encode_labels(lab, np.arange(4), {"a": ["x", "y"]})   # "z" unseen in train -> no proxy
    assert Y[:, 0].tolist() == [0, MISSING, MISSING, 1]


def toy_data(n, seed):
    """Labels that depend on the words, so a few steps can learn something."""
    texts = random_texts(n, seed)
    lab = {a: np.array([BIORXIV_SCHEMA[a][(len(t.split()) + j * ("cell" in t)) % 3] for t in texts], dtype=object)
           for j, a in enumerate(K)}
    lab["sample_size"][::5] = None                       # some missing labels
    return texts, lab


@pytest.mark.parametrize("loss", ["normsoftmax", "ce"])
def test_train_end_to_end_and_save_load(bert, tmp_path, loss):
    texts, lab = toy_data(48, 0)
    tr, va = np.arange(32), np.arange(32, 48)
    classes = {a: sorted(set(v for v in lab[a][tr] if v is not None)) for a in K}
    tk = tk_for(bert)
    ids = tk.encode(texts)
    torch.manual_seed(0)
    enc, heads = make_model(bert, loss, {a: len(c) for a, c in classes.items()})
    hist = train(enc, heads, tk, [ids[i] for i in tr], encode_labels(lab, tr, classes),
                 [ids[i] for i in va], encode_labels(lab, va, classes), classes, epochs=2, log=lambda s: None)
    assert len(hist["stage1"]) == 1 and len(hist["stage2"]) == 2 and hist["best_epoch"] in (0, 1)
    assert np.isfinite(hist["first_loss"])
    if loss == "normsoftmax":
        assert abs(hist["first_loss"] - hist["sum_ln_k"]) < 0.5 * hist["sum_ln_k"]

    save(tmp_path / loss, enc, heads, tk.tok, classes, {"loss": loss, "sigma": 0.05, "max_len": 64}, hist)
    enc2, heads2, classes2, cfg = load(tmp_path / loss)
    assert classes2 == classes and cfg["loss"] == loss
    enc.eval()
    b = batch(tk, texts[:16])
    with torch.no_grad():
        assert torch.equal(enc(**b), enc2(**b))
    p1, p2 = predict_ids(enc, heads, tk, ids), predict_ids(enc2, heads2, tk, ids)
    assert all(np.array_equal(p1[a], p2[a]) for a in K)


def test_heads_keep_schema_order():
    """nn.ParameterDict sorts its keys; logits must follow the schema order (= columns of Y)."""
    k = {"zeta": 2, "alpha": 3}
    heads = ProxyHeads(8, k)
    assert list(heads(torch.zeros(1, 8))) == ["zeta", "alpha"]


# ---------------- Step 2: vote_embedding/embedder.py ----------------
@pytest.fixture(scope="module")
def trained(bert, tmp_path_factory):
    """A briefly trained normsoftmax model saved to disk."""
    texts, lab = toy_data(48, 1)
    tr, va = np.arange(32), np.arange(32, 48)
    classes = {a: sorted(set(v for v in lab[a][tr] if v is not None)) for a in K}
    tk = tk_for(bert)
    ids = tk.encode(texts)
    torch.manual_seed(0)
    enc, heads = make_model(bert, "normsoftmax", {a: len(c) for a, c in classes.items()})
    hist = train(enc, heads, tk, [ids[i] for i in tr], encode_labels(lab, tr, classes),
                 [ids[i] for i in va], encode_labels(lab, va, classes), classes, epochs=1, log=lambda s: None)
    path = tmp_path_factory.mktemp("ft") / "normsoftmax"
    save(path, enc, heads, tk.tok, classes, {"loss": "normsoftmax", "sigma": 0.05, "max_len": 64}, hist)
    return path, texts


def test_ftembedder_matches_model_and_keeps_order(trained):
    from vote_embedding.embedder import FTEmbedder
    path, texts = trained
    emb = FTEmbedder(path, batch=8, device="cpu", log_overflow=False)
    X = emb(texts)
    assert X.dtype == np.float32 and X.shape == (len(texts), 768)
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    single = np.concatenate([emb([t]) for t in texts[:6]])        # length-sorted batching restores order
    assert np.allclose(single, X[:6], atol=1e-5)
    enc, heads, classes, _ = load(path)
    p = predict_ids(enc, heads, emb.tk, emb.tk.encode(texts))
    lab = emb.predict(texts)
    for a in K:
        assert np.array_equal(lab[a], np.array(classes[a])[p[a]])    # nearest proxy
        # nearest proxy = largest cosine = smallest L2 to the unit proxy
        P = heads.proxies(a).detach().numpy()
        assert np.array_equal(p[a], ((X[:, None] - P[None]) ** 2).sum(-1).argmin(1))
    assert emb([]).shape == (0, 768)


def test_ftembedder_cache_round_trip(trained, tmp_path, monkeypatch):
    from vote_embedding.embedder import FTEmbedder
    path, texts = trained
    emb = FTEmbedder(path, batch=8, device="cpu", cache_dir=tmp_path, log_overflow=False)
    X1 = emb(texts)
    files = list(tmp_path.glob(f"emb_{emb.name}_*.npy"))
    assert len(files) == 1 and emb.name.startswith("ft-normsoftmax-")

    def boom(*a, **k):
        raise AssertionError("model called despite cache")
    monkeypatch.setattr(emb, "encode", boom)
    assert np.array_equal(emb(texts), X1)                            # served from disk
    with pytest.raises(AssertionError):
        emb(texts[:-1])                                              # other texts -> other key
    monkeypatch.undo()
    emb2 = FTEmbedder(path, batch=8, device="cpu", cache_dir=tmp_path, log_overflow=False)
    assert emb2.name == emb.name and np.array_equal(emb2(texts), X1)  # same weights -> same key


def test_ftembedder_logs_overflow(trained, capsys):
    from vote_embedding.embedder import FTEmbedder
    path, _ = trained
    emb = FTEmbedder(path, batch=8, device="cpu")
    emb(["cell " * 100, "cell gene"])                                # max_len 64: one of two over
    assert "50.0% of 2 texts over 64 tokens" in capsys.readouterr().out
