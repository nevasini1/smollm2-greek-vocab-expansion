"""Intrinsic (language-modeling) evaluation on fixed held-out chunks.

Primary metric: bits per byte, BPB = sum NLL (nats) / (ln 2 * UTF-8 bytes of the text).
It is tokenizer-agnostic (Paloma; The Pile; lm-eval ``bits_per_byte``), so the
original-tokenizer model, Exp1 and Exp2 are scored on exactly the same bytes.

Protocol (removes the context-length confound): every chunk is at most
``eval.max_chars`` characters, cut at line/sentence boundaries, and fits in
``seq_len - 1`` *base*-tokenizer tokens, so it is scored as ONE window, prefixed with
``<|endoftext|>``, by every model. A better tokenizer therefore cannot see more text
than the original one.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..data.text import word_count
from ..utils import LOG, read_jsonl
from .scoring import LN2, ScoreItem, score_items


def load_eval_chunks(cfg, name: str, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
    from ..data.prepare import eval_set_path

    path = eval_set_path(cfg, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"eval set {name!r} not prepared ({path}); add it to the data config")
    rows = list(read_jsonl(path))
    return rows[:max_chunks] if max_chunks else rows


def score_chunks(
    model,
    tokenizer,
    chunks: List[Dict[str, Any]],
    device: str,
    amp_dtype=None,
    seq_len: int = 1024,
    batch_tokens: int = 16384,
    n_base: Optional[int] = None,
    eos_id: int = 0,
) -> List[Dict[str, Any]]:
    items: List[ScoreItem] = []
    owners: List[int] = []
    overflow = 0
    encs = tokenizer.encode_batch([c["text"] for c in chunks], add_special_tokens=False)
    for ci, enc in enumerate(encs):
        ids = list(enc.ids)
        if len(ids) + 1 <= seq_len:
            items.append(ScoreItem([eos_id] + ids, 1, ci))
            owners.append(ci)
        else:  # never happens for base/continued-BPE tokenizers; kept for odd extenders
            overflow += 1
            step = seq_len - 1
            for s in range(0, len(ids), step):
                items.append(ScoreItem([eos_id] + ids[s : s + step], 1, ci))
                owners.append(ci)
    if overflow:
        LOG.warning("%d chunks exceeded seq_len and were scored in disjoint windows", overflow)
    res = score_items(model, items, device, amp_dtype, batch_tokens, n_base)
    agg: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for it, r in zip(items, res):
        a = agg[it.key]
        a["nll"] += r.nll
        a["n_tokens"] += r.n_scored
        a["mass_new"] += r.mass_new
        a["nll_old"] += r.nll_old
        a["n_old_targets"] += r.n_old_targets
    records = []
    for ci, c in enumerate(chunks):
        a = agg[ci]
        records.append(
            {
                "set": c["set"],
                "language": c["language"],
                "doc_id": c["doc_id"],
                "chunk": c["chunk"],
                "n_bytes": len(c["text"].encode("utf-8")),
                "n_chars": len(c["text"]),
                "n_words": word_count(c["text"]),
                "n_tokens": int(a["n_tokens"]),
                "nll": a["nll"],
                "mass_new": a["mass_new"],
                "nll_old": a["nll_old"],
                "n_old_targets": int(a["n_old_targets"]),
                "has_new_vocab": n_base is not None,
            }
        )
    return records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    import math

    by_set: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_set[r["set"]].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for s, rs in by_set.items():
        nll = sum(r["nll"] for r in rs)
        nb = sum(r["n_bytes"] for r in rs)
        nc = sum(r["n_chars"] for r in rs)
        nw = sum(r["n_words"] for r in rs)
        nt = sum(r["n_tokens"] for r in rs)
        n_old = sum(r["n_old_targets"] for r in rs)
        d = {
            "language": rs[0]["language"],
            "chunks": len(rs),
            "docs": len({r["doc_id"] for r in rs}),
            "bpb": nll / (LN2 * nb),
            "bpc": nll / (LN2 * nc),
            "word_ppl": math.exp(min(nll / max(nw, 1), 50)),
            "token_ppl_within_model": math.exp(min(nll / max(nt, 1), 50)),
            "tokens_per_word": nt / max(nw, 1),
            "bytes_per_token": nb / max(nt, 1),
            "mean_mass_new": sum(r["mass_new"] for r in rs) / max(nt, 1),
        }
        if all(r.get("has_new_vocab") for r in rs):
            d["bpb_old_vocab"] = sum(r["nll_old"] for r in rs) / (LN2 * nb)
            d["frac_new_id_targets"] = 1.0 - n_old / max(nt, 1)
        out[s] = d
    return out
