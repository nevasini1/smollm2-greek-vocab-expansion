"""Continued BPE: learn new merges *on top of* the base tokenizer's own segmentation.

Why this and not "train a Greek BPE and append its merges" (Purason et al., EACL
Findings 2026; Liquid AI in-place expansion, 2026):

* The base tokenizer applies all of its 48,900 merges first. A separately trained
  Greek BPE assumes a different starting segmentation, so 6-17% of its tokens are
  never produced (e.g. SmolLM2 already merges ' ' with the UTF-8 lead byte, 'ĠÎ', which
  pre-empts word-initial Greek merges). Learning merges over the base segmentation
  makes every new token reachable by construction.
* New merges are appended *after* all original merges (lowest priority) and each one
  must (a) create a brand-new token string and (b) contain a Greek-unique byte
  pattern. Then any text without Greek characters tokenizes to byte-identical ids:
  original merges always fire first, a new token never appears in an original merge,
  and no new merge can fire on a word without Greek bytes.
* The first k learned merges are themselves a valid k-token extension, so one run
  gives the whole vocab-size curve (used for the 2k/4k/8k/16k analysis).
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from tokenizers import Tokenizer

from ..utils import LOG
from .base import is_strict_greek_token, token_has_greek_bytes

_GREEK_PIECE_MARKERS = ("Î", "Ï")  # byte-level chars for 0xCE / 0xCF


def _piece_may_have_greek(piece: str) -> bool:
    if "Î" in piece or "Ï" in piece:
        return True
    # 0xE1 0xBC-0xBF (Greek Extended) -> 'á' followed by '¼' '½' '¾' '¿'
    i = piece.find("á")
    while i != -1:
        if i + 1 < len(piece) and piece[i + 1] in "¼½¾¿":
            return True
        i = piece.find("á", i + 1)
    return False


@dataclass
class ContinuedBPEResult:
    merges: List[Tuple[str, str]]  # in learned order
    new_tokens: List[str]  # merged strings, ids = n_base + i
    parents: List[Tuple[int, int]]  # (left_id, right_id) in the extended id space
    merge_counts: List[int]  # pair frequency at the time of merging
    word_types: int
    words_total: int


def count_pretokens(base: Tokenizer, texts: Iterable[str]) -> Counter:
    """Count byte-level pre-tokens (Digits + ByteLevel regex) that contain Greek bytes."""
    pre = base.pre_tokenizer
    counts: Counter = Counter()
    for text in texts:
        for piece, _ in pre.pre_tokenize_str(text):
            if _piece_may_have_greek(piece):
                counts[piece] += 1
    return counts


def learn_continued_bpe(
    base: Tokenizer,
    pretoken_counts: Counter,
    num_merges: int,
    min_frequency: int = 2,
    accept: Optional[Callable[[str], bool]] = None,
    min_word_count: int = 1,
    log_every: int = 1000,
) -> ContinuedBPEResult:
    """Greedy BPE over the base segmentation with incremental pair statistics."""
    accept = accept or is_strict_greek_token
    vocab: Dict[str, int] = base.get_vocab(with_added_tokens=True)
    id2str: Dict[int, str] = {i: s for s, i in vocab.items()}
    next_id = max(id2str) + 1
    model = base.model

    words: List[List[int]] = []
    counts: List[int] = []
    for piece, c in pretoken_counts.items():
        if c < min_word_count:
            continue
        ids = [t.id for t in model.tokenize(piece)]
        if len(ids) > 1:
            words.append(ids)
            counts.append(c)
    LOG.info("continued BPE: %d Greek word types (%d occurrences) after base segmentation",
             len(words), sum(counts))

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    where: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, w in enumerate(words):
        c = counts[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += c
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    banned: set = set()
    merges: List[Tuple[str, str]] = []
    new_tokens: List[str] = []
    parents: List[Tuple[int, int]] = []
    merge_counts: List[int] = []

    while len(merges) < num_merges and heap:
        negc, pair = heapq.heappop(heap)
        cur = pair_counts.get(pair, 0)
        if cur != -negc:  # stale heap entry
            if cur > 0:
                heapq.heappush(heap, (-cur, pair))
            continue
        if cur < min_frequency:
            LOG.warning("continued BPE stopped at %d merges: best pair frequency %d < %d", len(merges), cur, min_frequency)
            break
        if pair in banned:
            continue
        a, b = pair
        merged = id2str[a] + id2str[b]
        if merged in vocab or not accept(merged):
            banned.add(pair)
            continue
        new_id = next_id
        next_id += 1
        vocab[merged] = new_id
        id2str[new_id] = merged
        merges.append((id2str[a], id2str[b]))
        new_tokens.append(merged)
        parents.append((a, b))
        merge_counts.append(cur)

        changed: set = set()
        for wi in list(where[pair]):
            w = words[wi]
            c = counts[wi]
            nw: List[int] = []
            i = 0
            n = len(w)
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            old_pairs = list(zip(w, w[1:]))
            new_pairs = list(zip(nw, nw[1:]))
            for p in old_pairs:
                pair_counts[p] -= c
                changed.add(p)
            for p in new_pairs:
                pair_counts[p] += c
                changed.add(p)
            old_set, new_set = set(old_pairs), set(new_pairs)
            for p in old_set - new_set:
                where[p].discard(wi)
            for p in new_set - old_set:
                where[p].add(wi)
            words[wi] = nw
        for p in changed:
            cnt = pair_counts.get(p, 0)
            if cnt <= 0:
                pair_counts.pop(p, None)
                where.pop(p, None)
            elif p not in banned:
                heapq.heappush(heap, (-cnt, p))
        if log_every and len(merges) % log_every == 0:
            LOG.info("  %d merges (last: %r, freq %d)", len(merges), merged, cur)

    if len(merges) < num_merges:
        LOG.warning("only %d/%d merges learned (corpus too small?)", len(merges), num_merges)
    return ContinuedBPEResult(merges, new_tokens, parents, merge_counts, len(words), sum(counts))


def greek_contains_filter(token: str) -> bool:
    """Looser filter: any token with a Greek-unique byte pattern (still English-safe)."""
    return token_has_greek_bytes(token)
