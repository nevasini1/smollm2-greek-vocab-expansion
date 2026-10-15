"""Text normalization, script detection and tokenizer-independent chunking.

Every corpus (training pools *and* eval sets, for every model) goes through
``normalize_text`` exactly once, in the data pipeline, never inside a tokenizer:

* Unicode NFC. Not NFD: under the GPT-2 regex, combining accents are not letters, so
  NFD splits Greek words at every accent. Not NFKC: it rewrites English text
  (µ->μ, ligatures, superscripts) and would break the "English tokenization is
  unchanged" guarantee.
* Control characters are dropped (except ``\t``/``\n``): SmolLM2's byte-level vocab
  lacks 21 byte values (0x04, 0x06, 0x13, 0x14, 0x16, 0x1D, 0xC0, 0xC1, 0xF1, 0xF2,
  0xF5-0xFF) and silently drops them, which would make ``decode(encode(x)) != x`` and
  corrupt bytes-per-byte accounting. Code points >= U+40000 (UTF-8 lead bytes
  0xF1+) are dropped for the same reason.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List, Optional, Tuple

GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
_NEWLINES = re.compile(r"\r\n?")
_SENT_END = re.compile(r"(?<=[.!?;·…])\s+")


def is_greek_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF


def has_greek(text: str) -> bool:
    return GREEK_RE.search(text) is not None


def normalize_text(text: str) -> str:
    text = _NEWLINES.sub("\n", text)
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in "\n\t":
            out.append(ch)
            continue
        cp = ord(ch)
        if cp >= 0x40000:
            continue
        if unicodedata.category(ch) == "Cc":
            continue
        out.append(ch)
    return "".join(out).strip()


def script_ratio(text: str, script: str) -> float:
    """Share of letters belonging to ``script`` ('greek' or 'latin')."""
    letters = 0
    hits = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if script == "greek":
            hits += is_greek_char(ch)
        elif script == "latin":
            hits += ch.isascii() or "LATIN" in unicodedata.name(ch, "")
        else:
            raise ValueError(script)
    return hits / letters if letters else 0.0


LANG_SCRIPT = {"el": "greek", "en": "latin"}


# ----------------------------------------------------------------------------- chunking


def _unit_spans(text: str, max_chars: int) -> List[Tuple[int, int]]:
    """Line spans; long lines are split at sentence ends, then at whitespace."""
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[^\n]+", text):
        s, e = m.span()
        if e - s <= max_chars:
            spans.append((s, e))
            continue
        last = s
        for sm in _SENT_END.finditer(text, s, e):
            spans.append((last, sm.start()))
            last = sm.end()
        spans.append((last, e))
    out: List[Tuple[int, int]] = []
    for s, e in spans:
        while e - s > max_chars:
            cut = text.rfind(" ", s + 1, s + max_chars)
            if cut <= s:
                cut = s + max_chars
            out.append((s, cut))
            s = cut
        if e > s:
            out.append((s, e))
    return out


def chunk_text(
    text: str,
    max_chars: int,
    max_chunks: int,
    fits: Optional[Callable[[str], bool]] = None,
    min_chars: int = 50,
) -> List[str]:
    """Split a document into up to ``max_chunks`` chunks of at most ``max_chars``.

    Chunks follow line/sentence boundaries and are cut from the original text, so
    every model scores exactly the same bytes. ``fits`` (e.g. "base tokenizer gives
    <= seq_len-1 tokens") is enforced by recursive halving.
    """
    chunks: List[str] = []

    def emit(s: int, e: int) -> None:
        t = text[s:e].strip()
        if len(t) < min_chars:
            return
        if fits is not None and not fits(t):
            mid = text.rfind(" ", s + 1, (s + e) // 2 + 1)
            if mid <= s:
                mid = (s + e) // 2
            emit(s, mid)
            emit(mid, e)
            return
        chunks.append(t)

    cs: Optional[int] = None
    ce = 0
    for s, e in _unit_spans(text, max_chars):
        if cs is None:
            cs, ce = s, e
        elif e - cs <= max_chars:
            ce = e
        else:
            emit(cs, ce)
            cs, ce = s, e
        if len(chunks) >= max_chunks:
            break
    if cs is not None and len(chunks) < max_chunks:
        emit(cs, ce)
    return chunks[:max_chunks]


def word_count(text: str) -> int:
    return len(text.split())
