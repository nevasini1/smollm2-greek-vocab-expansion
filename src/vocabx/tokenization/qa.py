"""Tokenizer QA harness: the invariants and efficiency numbers reported in the writeup.

Hard invariants (raise when ``strict=True``):
  * every original vocab entry keeps its id; original merges are an unchanged prefix;
  * texts without Greek characters tokenize to identical ids (English, code, other
    scripts, random non-Greek unicode);
  * ``decode(encode(x)) == x`` on every eval chunk, for both tokenizers.

Reported diagnostics:
  * self-tokenization test (Land & Bartolo 2024): each new token that decodes to
    valid UTF-8 must re-encode to exactly itself;
  * fertility (tokens/word), bytes/token, chars/token, proportion of continued words;
  * new-token utilisation on held-out Greek; English docs that contain Greek letters
    (these legitimately change and are reported, not hidden);
  * for continued BPE, the vocab-size curve from merge prefixes (k = 1k..all).
"""

from __future__ import annotations

import bisect
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tokenizers import Tokenizer

from ..data.text import has_greek
from ..utils import LOG
from .base import is_valid_utf8_token, merges_format, token_to_bytes


def compression_stats(tok: Tokenizer, texts: Sequence[str]) -> Dict[str, float]:
    enc = tok.encode_batch(list(texts), add_special_tokens=False)
    n_tok = sum(len(e.ids) for e in enc)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_chars = sum(len(t) for t in texts)
    words = [w for t in texts for w in t.split()]
    # proportion of continued words, measured IN CONTEXT: count tokens overlapping each
    # whitespace word's span in the full-text encoding (a bare word lacks its 'Ġ' form)
    n_words = n_split = 0
    for t, e in zip(texts, enc):
        starts = [a for a, _ in e.offsets]
        ends = [b for _, b in e.offsets]
        for m in re.finditer(r"\S+", t):
            s0, e0 = m.span()
            n = bisect.bisect_left(starts, e0) - bisect.bisect_right(ends, s0)
            n_words += 1
            n_split += n > 1
    split_words = n_split / max(n_words, 1)
    return {
        "tokens": n_tok,
        "bytes_per_token": n_bytes / max(n_tok, 1),
        "chars_per_token": n_chars / max(n_tok, 1),
        "tokens_per_word": n_tok / max(len(words), 1),
        "pct_continued_words": 100.0 * split_words,
    }


def _random_non_greek(n: int, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    out = []
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F), (0x4E00, 0x4FFF), (0x1F300, 0x1F64F)]
    for _ in range(n):
        s = []
        for _ in range(rng.randint(5, 120)):
            lo, hi = rng.choice(ranges)
            s.append(chr(rng.randint(lo, hi)))
        out.append("".join(s))
    return out


def run_tokenizer_qa(
    base: Tokenizer,
    ext: Tokenizer,
    n_base: int,
    new_tokens: List[str],
    greek_texts: Sequence[str],
    english_texts: Sequence[str],
    strict: bool = True,
    k_curve: Optional[List[int]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"n_base": n_base, "n_new": len(new_tokens)}
    problems: List[str] = []

    # 1. original vocab + merges untouched
    bv, ev = base.get_vocab(with_added_tokens=True), ext.get_vocab(with_added_tokens=True)
    moved = [t for t, i in bv.items() if ev.get(t) != i]
    report["original_ids_changed"] = len(moved)
    if moved:
        problems.append(f"{len(moved)} original vocab ids changed")
    bj, ej = json.loads(base.to_str()), json.loads(ext.to_str())
    bm, em = bj["model"].get("merges", []), ej["model"].get("merges", [])
    report["original_merges_prefix_intact"] = em[: len(bm)] == bm
    report["merges_format"] = merges_format(ej)
    if not report["original_merges_prefix_intact"]:
        problems.append("original merges are not an unchanged prefix")

    # 2. English / non-Greek invariance
    eng_nogreek = [t for t in english_texts if not has_greek(t)]
    eng_greek = [t for t in english_texts if has_greek(t)]
    synthetic = _random_non_greek(500) + [
        "def f(x):\n    return {'a': [1, 2, 3]}  # 2024-10-14\n",
        "naïve café µ ohm 3.14159 \t tabs   spaces",
    ]
    diff_nogreek = sum(base.encode(t).ids != ext.encode(t).ids for t in eng_nogreek)
    diff_synth = sum(base.encode(t).ids != ext.encode(t).ids for t in synthetic)
    tok_base_g = sum(len(base.encode(t).ids) for t in eng_greek)
    tok_ext_g = sum(len(ext.encode(t).ids) for t in eng_greek)
    report["english_invariance"] = {
        "docs_without_greek": len(eng_nogreek),
        "changed_without_greek": diff_nogreek,
        "synthetic_non_greek": len(synthetic),
        "changed_synthetic": diff_synth,
        "docs_with_greek_letters": len(eng_greek),
        "token_change_in_docs_with_greek_pct": 100.0 * (tok_ext_g - tok_base_g) / max(tok_base_g, 1),
    }
    if diff_nogreek or diff_synth:
        problems.append(f"non-Greek text changed tokenization ({diff_nogreek} eval docs, {diff_synth} synthetic)")

    # 3. round trip
    rt = {}
    for name, tk in (("base", base), ("extended", ext)):
        bad = [t for t in list(greek_texts) + list(english_texts)
               if tk.decode(tk.encode(t, add_special_tokens=False).ids, skip_special_tokens=False) != t]
        rt[name] = len(bad)
        if bad:
            problems.append(f"{name}: decode(encode(x)) != x for {len(bad)} texts (e.g. {bad[0][:60]!r})")
    report["roundtrip_failures"] = rt

    # 4. self-tokenization test for new tokens
    stt_fail, partial = [], 0
    for i, t in enumerate(new_tokens):
        if not is_valid_utf8_token(t):
            partial += 1
            continue
        s = token_to_bytes(t).decode("utf-8")
        if ext.encode(s, add_special_tokens=False).ids != [n_base + i]:
            stt_fail.append(s)
    report["self_tokenization"] = {"failures": len(stt_fail), "partial_utf8_tokens": partial, "examples": stt_fail[:10]}

    # 5. compression + utilisation
    report["greek"] = {"base": compression_stats(base, greek_texts), "extended": compression_stats(ext, greek_texts)}
    report["english"] = {"base": compression_stats(base, english_texts), "extended": compression_stats(ext, english_texts)}
    report["greek_token_reduction_x"] = report["greek"]["base"]["tokens"] / max(report["greek"]["extended"]["tokens"], 1)
    if new_tokens:
        used = np.zeros(len(new_tokens), dtype=np.int64)
        for e in ext.encode_batch(list(greek_texts), add_special_tokens=False):
            ids = np.asarray(e.ids)
            ids = ids[ids >= n_base] - n_base
            np.add.at(used, ids, 1)
        report["new_token_usage_heldout"] = {
            "used": int((used > 0).sum()),
            "unused": int((used == 0).sum()),
            "pct_used": 100.0 * float((used > 0).mean()),
            "share_of_greek_tokens_that_are_new": float(used.sum()) / max(report["greek"]["extended"]["tokens"], 1),
        }
        report["examples_new_tokens"] = [token_to_bytes(t).decode("utf-8", "replace") for t in new_tokens[:40]]

    # 6. vocab-size curve from merge prefixes (only meaningful for merge-based extenders)
    if k_curve and new_tokens and len(em) - len(bm) == len(new_tokens):
        curve = {}
        for k in sorted(set(min(k, len(new_tokens)) for k in k_curve)):
            tk = prefix_tokenizer(base, new_tokens, em[len(bm) : len(bm) + k], k)
            curve[k] = compression_stats(tk, greek_texts)
        report["vocab_size_curve"] = curve

    report["problems"] = problems
    if problems:
        msg = "tokenizer QA failed: " + "; ".join(problems)
        if strict:
            raise AssertionError(msg)
        LOG.warning(msg)
    return report


def prefix_tokenizer(base: Tokenizer, new_tokens: List[str], merges_prefix: list, k: int) -> Tokenizer:
    tj = json.loads(base.to_str())
    vocab = tj["model"]["vocab"]
    nxt = max(vocab.values()) + 1
    for t in new_tokens[:k]:
        vocab[t] = nxt
        nxt += 1
    tj["model"]["merges"] = tj["model"]["merges"] + list(merges_prefix)
    return Tokenizer.from_str(json.dumps(tj))


def new_token_exposure(counts: np.ndarray) -> Dict[str, Any]:
    """Summary of how often each new token occurs in the training stream."""
    if counts.size == 0:
        return {}
    return {
        "median": float(np.median(counts)),
        "p10": float(np.percentile(counts, 10)),
        "min": int(counts.min()),
        "never_seen": int((counts == 0).sum()),
        "seen_lt_10": int((counts < 10).sum()),
        "seen_lt_100": int((counts < 100).sum()),
    }
