"""Byte-budgeted greedy generation probes.

Held-out NLL can hide generative collapse (Liquid AI 2026 saw MCQ scores stay flat
while generation degenerated), so each model continues the same prompts. The budget
is in BYTES, not tokens: Exp2 packs ~2.8x more Greek per token, and a token budget
would hand it more text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import torch

from ..data.text import script_ratio
from .scoring import backbone

_WORD = re.compile(r"\w+", re.UNICODE)


def _repetition_stats(text: str) -> Dict[str, float]:
    words = _WORD.findall(text.lower())
    if len(words) < 4:  # too short to judge: excluded from the means (counted separately)
        return {"distinct_2": float("nan"), "rep_4gram": float("nan")}
    bi = list(zip(words, words[1:]))
    four = list(zip(words, words[1:], words[2:], words[3:]))
    return {
        "distinct_2": len(set(bi)) / len(bi),
        "rep_4gram": 1.0 - len(set(four)) / len(four),
    }


@torch.no_grad()
def greedy_continue(model, tokenizer, prompt: str, device: str, max_bytes: int, max_tokens: int, amp_dtype=None, eos_id: int = 0) -> Dict[str, Any]:
    """Greedy decoding with a KV cache (works with the split tied embedding as well)."""
    model.eval()
    ids = [eos_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    start = len(ids)
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype) if amp_dtype is not None else _nullctx()
    past = None
    x = torch.as_tensor([ids], device=device)
    n_bytes = 0
    for _ in range(max_tokens):
        with ctx:
            out = model(input_ids=x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].float().argmax(-1))
        if nxt == eos_id:
            break
        ids.append(nxt)
        n_bytes += len(tokenizer.decode([nxt]).encode("utf-8"))  # cheap upper-bound check
        if n_bytes >= max_bytes and len(tokenizer.decode(ids[start:]).encode("utf-8")) >= max_bytes:
            break
        x = torch.as_tensor([[nxt]], device=device)
    gen = ids[start:]
    out = tokenizer.decode(gen)
    out_b = out.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")
    n_in_budget = len(gen)  # tokens whose bytes fit in the budget (the last one may straddle it)
    while n_in_budget > 0 and len(tokenizer.decode(gen[:n_in_budget]).encode("utf-8")) > max_bytes:
        n_in_budget -= 1
    return {"prompt": prompt, "continuation": out_b, "new_tokens": len(gen), "tokens_in_budget": n_in_budget,
            "bytes_in_budget": len(tokenizer.decode(gen[:n_in_budget]).encode("utf-8"))}


def _nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_generation(model, tokenizer, prompts: Dict[str, List[str]], device: str, max_bytes: int = 300, amp_dtype=None) -> Dict[str, Any]:
    samples = []
    summary: Dict[str, Any] = {}
    for lang, plist in prompts.items():
        stats = []
        for p in plist:
            g = greedy_continue(model, tokenizer, p, device, max_bytes, max_tokens=max_bytes, amp_dtype=amp_dtype)
            script = "greek" if lang == "el" else "latin"
            has_letters = any(ch.isalpha() for ch in g["continuation"])
            g.update(language=lang, script_ratio=script_ratio(g["continuation"], script) if has_letters else float("nan"),
                     **_repetition_stats(g["continuation"]))
            g["bytes_per_token"] = g["bytes_in_budget"] / g["tokens_in_budget"] if g["tokens_in_budget"] else float("nan")
            samples.append(g)
            stats.append(g)
        summary[lang] = {key: _nanmean([s[key] for s in stats]) for key in ("script_ratio", "distinct_2", "rep_4gram", "bytes_per_token")}
        summary[lang]["short_or_empty"] = sum(1 for s in stats if s["distinct_2"] != s["distinct_2"])
        summary[lang]["n"] = len(stats)
    return {"summary": summary, "samples": samples}


def prompts_from_chunks(chunks: List[Dict[str, Any]], n: int, max_chars: int = 120) -> List[str]:
    out = []
    for c in chunks:
        first = c["text"].split("\n")[0]
        m = re.search(r"^(.{40,%d}?[.;!?])\s" % max_chars, first)
        p = m.group(1) if m else first[:max_chars].rsplit(" ", 1)[0]
        if len(p) >= 30:
            out.append(p)
        if len(out) >= n:
            break
    return out
