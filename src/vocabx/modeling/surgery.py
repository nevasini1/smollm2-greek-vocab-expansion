"""Model loading, vocabulary surgery, and the split tied embedding used for training.

Surgery (CPU, fp32 - resizing on MPS crashes and bf16 averaging adds noise):
  1. load SmolLM2-135M in fp32;
  2. compute new rows with a registered initializer;
  3. build a new tied embedding [E_old; E_new] by hand (no ``resize_token_embeddings``:
     its default ``mean_resizing`` makes all 8k rows identical copies of the mean);
  4. assert old rows are bit-identical and that input/output weights are still tied.

Training uses ``SplitTiedEmbedding``: ``old`` (49,152 x d) and ``new`` (n_new x d) are
separate ``nn.Parameter``s, concatenated for both the input lookup and the output
logits. That gives exact row freezing (requires_grad=False -> no Adam update, no weight
decay) and per-group learning rates; scaling gradients cannot do this because Adam is
invariant to gradient scale. Exp1 uses the same module with ``n_new = 0`` so both arms
run the same code path. ``merge_split_embedding`` turns it back into a standard HF model
for saving/evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import LOG
from .init import InitContext, run_initializer


def configure_attention(device: str) -> Dict[str, Any]:
    """Keep SmolLM2's grouped-query attention off PyTorch's slow fp32 'math' kernel on pre-Ampere GPUs.

    Without an attention mask, transformers calls SDPA with un-expanded K/V and
    ``enable_gqa=True``. On a T4 (sm75) no fused kernel accepts that (flash/cuDNN need
    sm80+; memory-efficient attention takes GQA only in recent torch), so SDPA falls back
    to the math kernel, which materialises fp32 B x H x T x T scores in every layer
    (~2x activation memory) and is slower. Expanding K/V in transformers (``repeat_kv``)
    lets the memory-efficient kernel run instead. Both arms use the same path.
    """
    import torch

    info: Dict[str, Any] = {"gqa_in_sdpa": "default"}
    if device != "cuda" or torch.cuda.get_device_capability()[0] >= 8:
        return info
    try:
        import transformers.integrations.sdpa_attention as sa

        if hasattr(sa, "use_gqa_in_sdpa"):
            sa.use_gqa_in_sdpa = lambda *a, **k: False
            info["gqa_in_sdpa"] = "disabled (repeat_kv; pre-Ampere GPU)"
    except Exception as e:  # older transformers always expand K/V
        info["gqa_in_sdpa"] = f"unpatched ({type(e).__name__})"
    LOG.info("attention: %s", info["gqa_in_sdpa"])
    return info


def load_base_model(name: str, revision: str = "main", attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = {"revision": revision, "attn_implementation": attn_implementation}
    try:  # transformers >= 4.56 uses `dtype`, older versions `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, **kwargs)
    model = model.float()
    if not getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("vocabx assumes tied input/output embeddings (SmolLM2 has them)")
    return model


def assert_tied(model) -> None:
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if isinstance(inp, SplitTiedEmbedding):
        assert isinstance(out, TiedLMHead) and out.embedding is inp, "split head not tied to split embedding"
        return
    assert out.weight is inp.weight or out.weight.data_ptr() == inp.weight.data_ptr(), "embeddings are not tied"


def expand_vocabulary(
    model,
    bundle,
    init_method: str,
    init_params: Dict[str, Any],
    seed: int = 0,
    decompositions: Optional[list] = None,
) -> Dict[str, Any]:
    """Append ``bundle.n_new`` rows to the tied embedding, initialised by ``init_method``.

    ``decompositions[i]`` = original-tokenizer ids of new token i's bytes; computed with
    the base BPE model when not given.
    """
    from ..tokenization.base import load_base_tokenizer, token_has_greek_bytes

    emb = model.get_input_embeddings()
    n_old, d = emb.weight.shape
    if bundle.n_base != n_old:
        raise ValueError(f"tokenizer base size {bundle.n_base} != model vocab {n_old}")
    if bundle.n_new == 0:
        return {"n_new": 0}
    E_old = emb.weight.detach().float().cpu().clone()
    base_name = bundle.meta.get("base_model", "HuggingFaceTB/SmolLM2-135M")
    if decompositions is None:
        base_model = load_base_tokenizer(base_name).model
        decompositions = [[t.id for t in base_model.tokenize(tok)] for tok in bundle.new_tokens]
    decomps = decompositions
    params = dict(init_params)
    if init_method == "script_mean" and "script_ids" not in params:
        vocab = load_base_tokenizer(base_name).get_vocab()
        params["script_ids"] = [i for s, i in vocab.items() if i < n_old and token_has_greek_bytes(s)]
    ctx = InitContext(
        E_old=E_old,
        new_tokens=bundle.new_tokens,
        decompositions=decomps,
        parents=bundle.parents,
        initializer_range=float(getattr(model.config, "initializer_range", 0.02)),
        seed=seed,
    )
    new_rows = run_initializer(init_method, ctx, params)

    W = torch.cat([E_old, new_rows], 0)
    new_emb = nn.Embedding(W.shape[0], d)
    with torch.no_grad():
        new_emb.weight.copy_(W)
    model.set_input_embeddings(new_emb)
    head = nn.Linear(d, W.shape[0], bias=False)
    head.weight = new_emb.weight
    model.set_output_embeddings(head)
    model.config.vocab_size = W.shape[0]
    assert_tied(model)
    assert torch.equal(model.get_input_embeddings().weight[:n_old].detach(), E_old), "old rows changed"

    norms_old = E_old.norm(dim=1)
    norms_new = new_rows.norm(dim=1)
    info = {
        "n_new": bundle.n_new,
        "init": init_method,
        "mean_pieces_per_new_token": float(sum(map(len, decomps)) / max(len(decomps), 1)),
        "old_row_norm_mean": float(norms_old.mean()),
        "new_row_norm_mean": float(norms_new.mean()),
        "mu_norm": float(E_old.mean(0).norm()),
    }
    LOG.info("expanded vocab %d -> %d with %s (new-row norm %.2f vs old %.2f)", n_old, W.shape[0], init_method,
             info["new_row_norm_mean"], info["old_row_norm_mean"])
    return info


class SplitTiedEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, n_old: int):
        super().__init__()
        self.n_old = n_old
        self.num_embeddings, self.embedding_dim = weight.shape
        self.old = nn.Parameter(weight[:n_old].detach().clone())
        n_new = weight.shape[0] - n_old
        self.new = nn.Parameter(weight[n_old:].detach().clone()) if n_new > 0 else None

    @property
    def n_new(self) -> int:
        return 0 if self.new is None else self.new.shape[0]

    def full_weight(self) -> torch.Tensor:
        return self.old if self.new is None else torch.cat([self.old, self.new], 0)

    @property
    def weight(self) -> torch.Tensor:  # read-only view for code that inspects .weight
        return self.full_weight()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.full_weight())


class TiedLMHead(nn.Module):
    """Output projection sharing the split embedding (not registered as a submodule)."""

    def __init__(self, embedding: SplitTiedEmbedding):
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)

    @property
    def embedding(self) -> SplitTiedEmbedding:
        return self._embedding

    @property
    def weight(self) -> torch.Tensor:
        return self._embedding.full_weight()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self._embedding.full_weight())


def install_split_embedding(model, n_old: int) -> SplitTiedEmbedding:
    emb = model.get_input_embeddings()
    split = SplitTiedEmbedding(emb.weight.data, n_old)
    model.set_input_embeddings(split)
    model.set_output_embeddings(TiedLMHead(split))
    return split


def merge_split_embedding(model) -> None:
    """Replace the split modules with a standard tied nn.Embedding / nn.Linear (in place)."""
    split = model.get_input_embeddings()
    if not isinstance(split, SplitTiedEmbedding):
        return
    W = split.full_weight().detach()
    emb = nn.Embedding(W.shape[0], W.shape[1]).to(W.device)
    with torch.no_grad():
        emb.weight.copy_(W)
    model.set_input_embeddings(emb)
    head = nn.Linear(W.shape[1], W.shape[0], bias=False).to(W.device)
    head.weight = emb.weight
    model.set_output_embeddings(head)
    assert_tied(model)


def build_model(cfg, bundle, device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Base model -> (optional) vocabulary expansion -> split embedding -> device."""
    configure_attention(device)
    model = load_base_model(cfg.model.name, cfg.model.revision, cfg.model.attn_implementation)
    n_old = model.get_input_embeddings().weight.shape[0]
    info = expand_vocabulary(model, bundle, cfg.init.method, cfg.init.params, seed=cfg.seed)
    install_split_embedding(model, n_old)
    model.to(device)
    model.config.use_cache = False
    return model, info


def save_model(model, bundle, out_dir: str) -> None:
    """Save a standard HF checkpoint (+ tokenizer) without disturbing a training model."""
    import copy
    import os

    os.makedirs(out_dir, exist_ok=True)
    m = copy.deepcopy(model).to("cpu")
    merge_split_embedding(m)
    m.config.use_cache = True
    m.save_pretrained(out_dir)
    bundle.save_hf(out_dir)
