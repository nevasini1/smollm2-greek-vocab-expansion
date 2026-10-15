"""Compute accounting: FLOPs per token for each arm, and a running budget meter.

The brief asks for "same tokens, same steps, same compute". Tokens and steps are
matched exactly by construction; compute cannot be (Exp2's 57,152-row tied head costs
~3% more per token), so it is measured and reported instead of silently assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

import numpy as np


def matmul_params(hf_config, vocab_size: int) -> Dict[str, int]:
    d = hf_config.hidden_size
    L = hf_config.num_hidden_layers
    h = hf_config.num_attention_heads
    kv = getattr(hf_config, "num_key_value_heads", h)
    hd = getattr(hf_config, "head_dim", None) or d // h
    ff = hf_config.intermediate_size
    per_layer = d * h * hd * 2 + d * kv * hd * 2 + 3 * d * ff  # q,o + k,v + gate,up,down
    return {"body": L * per_layer, "head": d * vocab_size, "layers": L, "d": d}


def flops_per_token(hf_config, vocab_size: int, seq_len: int, body_trainable: bool = True) -> Dict[str, float]:
    """Kaplan-style estimate: fwd = 2N + 2*L*ctx*d (attention), bwd = 2x fwd.

    With the body frozen (embedding-only stage), weight gradients of body matmuls are
    skipped but activation gradients still flow through every layer.
    """
    p = matmul_params(hf_config, vocab_size)
    attn = 2 * p["layers"] * seq_len * p["d"]
    fwd = 2 * (p["body"] + p["head"]) + attn
    if body_trainable:
        train = 3 * fwd
    else:
        train = fwd + fwd + 2 * p["head"]  # forward + activation grads + head/embedding weight grad
    return {"forward": float(fwd), "train": float(train), "head_share_fwd": 2 * p["head"] / fwd}


@dataclass
class BudgetMeter:
    steps: int = 0
    tokens: int = 0
    tokens_by_lang: Dict[str, int] = field(default_factory=lambda: {"el": 0, "en": 0})
    bytes_by_lang: Dict[str, int] = field(default_factory=lambda: {"el": 0, "en": 0})
    flops: float = 0.0
    skipped_steps: int = 0
    train_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "BudgetMeter":
        return BudgetMeter(**d)


def token_byte_lengths(tokenizer, vocab_size: int) -> np.ndarray:
    """UTF-8 byte length of every token id (special tokens count 0)."""
    from ..tokenization.base import token_to_bytes

    out = np.zeros(vocab_size, dtype=np.int64)
    special = {t.content for t in tokenizer.get_added_tokens_decoder().values() if t.special}
    for tok, i in tokenizer.get_vocab(with_added_tokens=True).items():
        if i >= vocab_size or tok in special:
            continue
        try:
            out[i] = len(token_to_bytes(tok))
        except KeyError:  # an AddedToken stored as plain text
            out[i] = len(tok.encode("utf-8"))
    return out
