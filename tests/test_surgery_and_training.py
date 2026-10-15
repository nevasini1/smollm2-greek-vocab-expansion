"""Embedding surgery, split tied embedding, freezing, loss and scoring (tiny random model, no download)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vocabx.evaluation.scoring import ScoreItem, score_items
from vocabx.modeling.init import INITIALIZERS, InitContext
from vocabx.modeling.surgery import (
    SplitTiedEmbedding,
    TiedLMHead,
    assert_tied,
    expand_vocabulary,
    install_split_embedding,
    merge_split_embedding,
)
from vocabx.training.trainer import lm_loss_sum


class FakeBundle:
    def __init__(self, n_base, n_new):
        self.n_base, self.n_new = n_base, n_new
        self.new_tokens = [f"tok{i}" for i in range(n_new)]
        self.parents = [(i % n_base, (i + 1) % n_base) for i in range(n_new)]
        self.meta = {}

    @property
    def vocab_size(self):
        return self.n_base + self.n_new


def _expanded(tiny_model_factory, n_new=20, method="fvt_mean"):
    m = tiny_model_factory(300)
    E0 = m.get_input_embeddings().weight.detach().clone()
    b = FakeBundle(300, n_new)
    decomps = [[i % 300, (3 * i + 1) % 300, (7 * i + 2) % 300] for i in range(n_new)]
    info = expand_vocabulary(m, b, method, {}, decompositions=decomps)
    return m, E0, decomps, info


def test_expand_keeps_old_rows_and_ties(tiny_model_factory):
    m, E0, decomps, info = _expanded(tiny_model_factory)
    W = m.get_input_embeddings().weight
    assert W.shape[0] == 320
    assert torch.equal(W[:300], E0)
    assert_tied(m)
    for j, d in enumerate(decomps):
        assert torch.allclose(W[300 + j], E0[d].mean(0), atol=1e-6)
    assert m.config.vocab_size == 320


def test_old_token_logits_unchanged_up_to_softmax_mass(tiny_model_factory):
    base = tiny_model_factory(300)
    m, _, _, _ = _expanded(tiny_model_factory)
    x = torch.randint(0, 300, (2, 12))
    with torch.no_grad():
        a = base(x).logits
        b = m(x).logits
    assert torch.allclose(a, b[..., :300], atol=1e-5)


@pytest.mark.parametrize("method", INITIALIZERS.names())
def test_every_initializer_runs(tiny_model_factory, method):
    if method == "script_mean":
        params = {"script_ids": [1, 2, 3]}
    else:
        params = {}
    m = tiny_model_factory(300)
    b = FakeBundle(300, 10)
    decomps = [[i, i + 1] for i in range(10)]
    expand_vocabulary(m, b, method, params, decompositions=decomps)
    assert torch.isfinite(m.get_input_embeddings().weight).all()


def test_mean_inits_stay_in_hull_random_does_not(tiny_model_factory):
    """In-hull init can never be the strict argmax (Mundra et al. Thm. 2)."""
    for method, expect_in_hull in (("fvt_mean", True), ("global_mean", True), ("random_normal", False)):
        m, _, _, _ = _expanded(tiny_model_factory, method=method)
        W = m.get_input_embeddings().weight.detach()
        h = torch.randn(2000, W.shape[1]) * 10
        logits = h @ W.T
        best_new = logits[:, 300:].max(1).values
        best_old = logits[:, :300].max(1).values
        if expect_in_hull:
            assert (best_new <= best_old + 1e-4).all()


def test_split_embedding_matches_and_merges_back(tiny_model_factory):
    m, _, _, _ = _expanded(tiny_model_factory)
    x = torch.randint(0, 320, (2, 10))
    with torch.no_grad():
        ref = m(x).logits
    split = install_split_embedding(m, 300)
    assert isinstance(m.get_output_embeddings(), TiedLMHead)
    assert_tied(m)
    with torch.no_grad():
        got = m(x).logits
    assert torch.allclose(ref, got, atol=1e-5)
    names = [n for n, _ in m.named_parameters()]
    assert sum("embed_tokens" in n for n in names) == 2  # old + new, no duplicate head param
    merge_split_embedding(m)
    assert_tied(m)
    with torch.no_grad():
        back = m(x).logits
    assert torch.allclose(ref, back, atol=1e-5)


def test_frozen_old_rows_do_not_move_under_adamw(tiny_model_factory):
    m, _, _, _ = _expanded(tiny_model_factory)
    split = install_split_embedding(m, 300)
    old0, new0 = split.old.detach().clone(), split.new.detach().clone()
    body = [p for n, p in m.named_parameters() if p is not split.old and p is not split.new]
    opt = torch.optim.AdamW([{"params": [split.old], "weight_decay": 0.1}, {"params": [split.new], "weight_decay": 0.0},
                             {"params": body, "weight_decay": 0.1}], lr=1e-2)
    split.old.requires_grad_(False)
    for p in body:
        p.requires_grad_(False)
    x = torch.randint(0, 320, (2, 16))
    for _ in range(3):
        loss = lm_loss_sum(m, x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert torch.equal(split.old, old0)  # no update AND no weight decay
    assert not torch.equal(split.new, new0)


def test_chunked_loss_equals_full(tiny_model_factory):
    m = tiny_model_factory(300)
    install_split_embedding(m, 300)
    x = torch.randint(0, 300, (3, 33))
    full = lm_loss_sum(m, x[:, :-1], x[:, 1:], 0)
    chunked = lm_loss_sum(m, x[:, :-1], x[:, 1:], 16)
    assert torch.allclose(full, chunked, rtol=1e-5)
    ref = F.cross_entropy(m(x[:, :-1]).logits.reshape(-1, 300), x[:, 1:].reshape(-1), reduction="sum")
    assert torch.allclose(full, ref, rtol=1e-5)


def test_score_items_matches_manual_nll_and_mass(tiny_model_factory):
    m, _, _, _ = _expanded(tiny_model_factory)
    seqs = [[0, 5, 17, 301, 42], [0, 7, 7, 7], [0, 310, 2, 3, 4, 5, 6]]
    items = [ScoreItem(s, 1, i) for i, s in enumerate(seqs)]
    res = score_items(m, items, "cpu", None, batch_tokens=12, n_base=300)
    for s, r in zip(seqs, res):
        with torch.no_grad():
            lp = torch.log_softmax(m(torch.tensor([s])).logits[0].float(), -1)
        manual = -sum(lp[t - 1, s[t]].item() for t in range(1, len(s)))
        assert math.isclose(r.nll, manual, rel_tol=1e-5)
        mass = sum(lp[t - 1, 300:].exp().sum().item() for t in range(1, len(s)))
        assert math.isclose(r.mass_new, mass, rel_tol=1e-4)
        assert r.n_scored == len(s) - 1


@pytest.mark.network
def test_cached_greedy_generation_matches_uncached(tiny_model_factory, smollm_tokenizer):
    from vocabx.evaluation.generation import greedy_continue

    m = tiny_model_factory(49152)
    install_split_embedding(m, 49152)
    prompt = "Η Ελλάδα είναι"
    g = greedy_continue(m, smollm_tokenizer, prompt, "cpu", max_bytes=10_000, max_tokens=12)
    ids = [0] + smollm_tokenizer.encode(prompt, add_special_tokens=False).ids
    ref = []
    with torch.no_grad():
        for _ in range(12):
            nxt = int(m(torch.tensor([ids + ref])).logits[0, -1].argmax())
            if nxt == 0:
                break
            ref.append(nxt)
    assert g["new_tokens"] == len(ref)
    assert g["continuation"] == smollm_tokenizer.decode(ref)


def test_attention_patch_only_on_pre_ampere(monkeypatch):
    import transformers.integrations.sdpa_attention as sa

    from vocabx.modeling.surgery import configure_attention

    original = sa.use_gqa_in_sdpa
    try:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
        assert configure_attention("cuda")["gqa_in_sdpa"] == "default" and sa.use_gqa_in_sdpa is original
        assert configure_attention("cpu")["gqa_in_sdpa"] == "default"
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))  # T4
        info = configure_attention("cuda")
        assert "disabled" in info["gqa_in_sdpa"]
        k = torch.zeros(1, 3, 4, 64)
        assert sa.use_gqa_in_sdpa(None, k, k) is False
    finally:
        sa.use_gqa_in_sdpa = original
