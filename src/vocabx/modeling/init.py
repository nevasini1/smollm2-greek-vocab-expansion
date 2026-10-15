"""Embedding initializers for new vocabulary rows (registry).

SmolLM2-135M ties input and output embeddings, so one row is both how the model
*reads* a new token and how strongly it *predicts* it. That makes the output side
decide what is safe:

* Inits inside the convex hull of existing rows (means, FVT, merge-parent, alpha-mixes)
  can never become the argmax for an English context (Mundra et al. 2024, Thm. 2),
  and Hewitt (2021) bounds the probability mass a mean-like row can steal.
* Random rows (and norm-rescaled means) leave the hull; with tying they move
  10-60% of English probability mass onto Greek tokens at step 0 (measured).

Every initializer has the signature ``fn(ctx: InitContext, **params) -> Tensor[n_new, d]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..registry import Registry

INITIALIZERS = Registry("embedding initializer")


@dataclass
class InitContext:
    E_old: torch.Tensor  # [n_old, d] fp32, the original (tied) embedding matrix
    new_tokens: List[str]  # byte-level strings; id = n_old + i
    decompositions: List[List[int]]  # base-tokenizer ids of each new token's bytes
    parents: Optional[List[Tuple[int, int]]]  # merge parents in the extended id space
    initializer_range: float = 0.02
    seed: int = 0

    @property
    def n_old(self) -> int:
        return self.E_old.shape[0]

    @property
    def mu(self) -> torch.Tensor:
        return self.E_old.mean(0)


def _fvt(ctx: InitContext) -> torch.Tensor:
    rows = []
    for dec in ctx.decompositions:
        if not dec:
            rows.append(ctx.mu)
        else:
            rows.append(ctx.E_old[torch.as_tensor(dec, dtype=torch.long)].mean(0))
    return torch.stack(rows) if rows else ctx.E_old.new_zeros(0, ctx.E_old.shape[1])


@INITIALIZERS.register("fvt_mean")
def init_fvt_mean(ctx: InitContext) -> torch.Tensor:
    """Mean of the original-tokenizer pieces (FVT; Gee et al. 2022). Used by Meltemi/Krikri/Liquid."""
    return _fvt(ctx)


@INITIALIZERS.register("merge_parent_mean")
def init_merge_parent_mean(ctx: InitContext) -> torch.Tensor:
    """Recursive mean of the two BPE merge parents (falls back to FVT without parents)."""
    if ctx.parents is None:
        return _fvt(ctx)
    n_old = ctx.n_old
    out = ctx.E_old.new_zeros(len(ctx.new_tokens), ctx.E_old.shape[1])

    def row(i: int) -> torch.Tensor:
        return ctx.E_old[i] if i < n_old else out[i - n_old]

    for j, (a, b) in enumerate(ctx.parents):
        out[j] = 0.5 * (row(a) + row(b))  # parents always precede children
    return out


@INITIALIZERS.register("alpha_shrink")
def init_alpha_shrink(ctx: InitContext, alpha: float = 0.75, base: str = "fvt_mean") -> torch.Tensor:
    """alpha * base + (1 - alpha) * mu: a hull-preserving temperature on new-token logits."""
    rows = INITIALIZERS.get(base)(ctx)
    return alpha * rows + (1.0 - alpha) * ctx.mu


@INITIALIZERS.register("global_mean")
def init_global_mean(ctx: InitContext) -> torch.Tensor:
    """All rows = mean embedding (what HF's mean_resizing effectively does)."""
    return ctx.mu.expand(len(ctx.new_tokens), -1).clone()


@INITIALIZERS.register("script_mean")
def init_script_mean(ctx: InitContext, script_ids: Optional[List[int]] = None) -> torch.Tensor:
    """All rows = mean of existing Greek-containing rows (Downey et al. 2023 style)."""
    if script_ids is None:
        raise ValueError("script_mean needs script_ids (filled in by the model builder)")
    m = ctx.E_old[torch.as_tensor(script_ids)].mean(0)
    return m.expand(len(ctx.new_tokens), -1).clone()


@INITIALIZERS.register("random_normal")
def init_random_normal(ctx: InitContext, std: Optional[float] = None) -> torch.Tensor:
    """N(0, std^2); default std = config.initializer_range (HF's mean_resizing=False)."""
    g = torch.Generator().manual_seed(ctx.seed)
    s = ctx.initializer_range if std is None else std
    return torch.randn(len(ctx.new_tokens), ctx.E_old.shape[1], generator=g) * s


@INITIALIZERS.register("first_subtoken")
def init_first_subtoken(ctx: InitContext) -> torch.Tensor:
    return torch.stack([ctx.E_old[d[0]] if d else ctx.mu for d in ctx.decompositions])


@INITIALIZERS.register("last_subtoken")
def init_last_subtoken(ctx: InitContext) -> torch.Tensor:
    return torch.stack([ctx.E_old[d[-1]] if d else ctx.mu for d in ctx.decompositions])


def run_initializer(name: str, ctx: InitContext, params: Dict[str, Any]) -> torch.Tensor:
    rows = INITIALIZERS.get(name)(ctx, **params)
    if rows.shape != (len(ctx.new_tokens), ctx.E_old.shape[1]):
        raise ValueError(f"initializer {name} returned {tuple(rows.shape)}")
    if not torch.isfinite(rows).all():
        raise ValueError(f"initializer {name} produced non-finite values")
    return rows.float()
