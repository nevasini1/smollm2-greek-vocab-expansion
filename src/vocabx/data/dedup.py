"""Train/eval decontamination: exact paragraph hashing + MinHash-LSH near-duplicates.

Direction follows Paloma: eval documents are fixed first, then any *training*
document that shares a long paragraph with, or is a near-duplicate of, any held-out
document is removed from the training pool. Web crawls mirror Wikipedia, so this runs
across sources of the same language, not only within a source.
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from ..utils import text_hash

_WORD = re.compile(r"\w+", re.UNICODE)
_PRIME = np.uint64((1 << 31) - 1)


def paragraph_hashes(text: str, min_chars: int) -> Set[str]:
    out = set()
    for p in text.split("\n"):
        p = " ".join(p.split()).lower()
        if len(p) >= min_chars:
            out.add(text_hash(p))
    return out


class MinHasher:
    """Word 5-gram MinHash with LSH banding.

    FineWeb's 14 bands x 8 rows flags a Jaccard-0.7 pair with only ~56% probability. For
    decontamination recall matters more than speed, so we use 20 bands x 5 rows:
    P(candidate | J=0.7) = 1 - (1 - 0.7^5)^20 ~ 0.97. Candidates are then confirmed by the
    estimated Jaccard >= threshold, so the extra candidates do not cause false drops.
    """

    def __init__(self, num_perm: int = 100, bands: int = 20, ngram: int = 5, seed: int = 1):
        assert num_perm % bands == 0
        self.num_perm, self.bands, self.rows, self.ngram = num_perm, bands, num_perm // bands, ngram
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, (1 << 31) - 1, size=num_perm).astype(np.uint64)
        self.b = rng.randint(0, (1 << 31) - 1, size=num_perm).astype(np.uint64)

    def signature(self, text: str) -> np.ndarray | None:
        words = _WORD.findall(text.lower())
        if len(words) < self.ngram:
            return None
        shingles = {" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)}
        h = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in shingles), dtype=np.uint64, count=len(shingles))
        # (a*h + b) mod p for all permutations; a,h < 2^32 so the product fits in uint64.
        vals = (np.outer(self.a, h) + self.b[:, None]) % _PRIME
        return vals.min(axis=1)

    def band_keys(self, sig: np.ndarray) -> List[Tuple[int, bytes]]:
        return [(b, sig[b * self.rows : (b + 1) * self.rows].tobytes()) for b in range(self.bands)]


class DecontaminationIndex:
    """Index of held-out documents; query training documents against it."""

    def __init__(self, paragraph_min_chars: int = 60, use_minhash: bool = True, threshold: float = 0.7):
        self.paragraph_min_chars = paragraph_min_chars
        self.use_minhash = use_minhash
        self.threshold = threshold
        self.paragraphs: Set[str] = set()
        self.hasher = MinHasher()
        self.buckets: Dict[Tuple[int, bytes], List[int]] = {}
        self.sigs: List[np.ndarray] = []

    def add(self, text: str) -> None:
        self.paragraphs |= paragraph_hashes(text, self.paragraph_min_chars)
        if self.use_minhash:
            sig = self.hasher.signature(text)
            if sig is None:
                return
            idx = len(self.sigs)
            self.sigs.append(sig)
            for key in self.hasher.band_keys(sig):
                self.buckets.setdefault(key, []).append(idx)

    def contaminated(self, text: str) -> str | None:
        """Return the reason a training doc must be dropped, or None."""
        if self.paragraphs and not self.paragraphs.isdisjoint(paragraph_hashes(text, self.paragraph_min_chars)):
            return "paragraph"
        if self.use_minhash and self.sigs:
            sig = self.hasher.signature(text)
            if sig is None:
                return None
            cands = set()
            for key in self.hasher.band_keys(sig):
                cands.update(self.buckets.get(key, ()))
            for c in cands:
                if float(np.mean(self.sigs[c] == sig)) >= self.threshold:
                    return "minhash"
        return None
