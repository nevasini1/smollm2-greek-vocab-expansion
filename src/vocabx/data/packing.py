"""Pre-tokenized corpora and deterministic, resumable token streams.

Each (tokenizer, data artifact) pair is tokenized once into
``<workdir>/packed/<tokenizer_key>/<data_key>/{el,en}.{tokens,offsets,bytes}.npy``.

A ``TokenStream`` concatenates documents (each prefixed with ``<|endoftext|>``, id 0)
in an order fixed by the run seed and cuts it into blocks of ``seq_len + 1`` tokens.
Both arms of a comparison use the same seed, so they see the same documents in the
same order - only the segmentation differs. When a stream is exhausted it starts a
new epoch with a fresh permutation, and the epoch count is reported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..utils import LOG, read_json, read_jsonl, write_json


@dataclass
class PackedCorpus:
    tokens: np.ndarray  # flat token ids of all docs (no separators)
    offsets: np.ndarray  # [n_docs + 1]
    doc_bytes: np.ndarray  # [n_docs] UTF-8 bytes of each doc
    language: str

    @property
    def n_docs(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_bytes(self) -> int:
        return int(self.doc_bytes.sum())

    @property
    def bytes_per_token(self) -> float:
        return self.n_bytes / max(self.n_tokens, 1)


def _dtype_for(vocab_size: int):
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32


def pack_jsonl(tokenizer, jsonl_path: str, out_prefix: str, vocab_size: int, language: str, batch: int = 512) -> PackedCorpus:
    dtype = _dtype_for(vocab_size)
    chunks: List[np.ndarray] = []
    lengths: List[int] = []
    nbytes: List[int] = []
    buf: List[str] = []

    def flush() -> None:
        for enc in tokenizer.encode_batch(buf, add_special_tokens=False):
            ids = np.asarray(enc.ids, dtype=dtype)
            chunks.append(ids)
            lengths.append(len(ids))
        buf.clear()

    for row in read_jsonl(jsonl_path):
        buf.append(row["text"])
        nbytes.append(len(row["text"].encode("utf-8")))
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    np.save(out_prefix + ".tokens.npy", tokens)
    np.save(out_prefix + ".offsets.npy", offsets)
    np.save(out_prefix + ".bytes.npy", np.asarray(nbytes, dtype=np.int64))
    corpus = PackedCorpus(tokens, offsets, np.asarray(nbytes, dtype=np.int64), language)
    LOG.info("packed %s: %d docs, %d tokens, %.2f bytes/token", os.path.basename(jsonl_path), corpus.n_docs,
             corpus.n_tokens, corpus.bytes_per_token)
    return corpus


def load_packed(prefix: str, language: str) -> PackedCorpus:
    return PackedCorpus(
        np.load(prefix + ".tokens.npy", mmap_mode="r"),
        np.load(prefix + ".offsets.npy"),
        np.load(prefix + ".bytes.npy"),
        language,
    )


def ensure_packed(cfg, bundle, languages=("el", "en")) -> Dict[str, PackedCorpus]:
    from .prepare import data_dir

    ddir = data_dir(cfg)
    out = os.path.join(cfg.workdir, "packed", cfg.key("tokenizer"), cfg.key("data"))
    os.makedirs(out, exist_ok=True)
    result: Dict[str, PackedCorpus] = {}
    for lang in languages:
        src = os.path.join(ddir, f"train_{lang}.jsonl.gz")
        if not os.path.exists(src):
            continue
        prefix = os.path.join(out, lang)
        if os.path.exists(prefix + ".bytes.npy"):
            result[lang] = load_packed(prefix, lang)
        else:
            result[lang] = pack_jsonl(bundle.tokenizer, src, prefix, bundle.vocab_size, lang)
            result[lang] = load_packed(prefix, lang)
        write_json(os.path.join(out, f"{lang}.stats.json"), {
            "docs": result[lang].n_docs, "tokens": result[lang].n_tokens, "bytes": result[lang].n_bytes,
            "bytes_per_token": result[lang].bytes_per_token,
        })
    return result


class TokenStream:
    """Deterministic, resumable stream of ``seq_len + 1``-token blocks."""

    def __init__(self, corpus: PackedCorpus, seq_len: int, seed: int, eos_id: int = 0):
        self.corpus = corpus
        self.seq_len = seq_len
        self.seed = seed
        self.eos_id = eos_id
        self.epoch = 0
        self.block = 0
        self._stream: Optional[np.ndarray] = None
        self._build()

    def _build(self) -> None:
        c = self.corpus
        if c.n_docs == 0:
            raise ValueError(f"empty {c.language} corpus")
        order = np.random.RandomState(self.seed * 1000 + self.epoch).permutation(c.n_docs)
        lengths = (c.offsets[1:] - c.offsets[:-1])[order] + 1
        stream = np.empty(int(lengths.sum()), dtype=c.tokens.dtype)
        pos = 0
        for d in order:
            s, e = c.offsets[d], c.offsets[d + 1]
            stream[pos] = self.eos_id
            stream[pos + 1 : pos + 1 + (e - s)] = c.tokens[s:e]
            pos += 1 + (e - s)
        self._stream = stream
        self.blocks_per_epoch = (len(stream) - 1) // self.seq_len

    def next_block(self) -> np.ndarray:
        if self.block >= self.blocks_per_epoch:
            self.epoch += 1
            self.block = 0
            LOG.warning("%s stream starts epoch %d (data repetition)", self.corpus.language, self.epoch + 1)
            self._build()
        s = self.block * self.seq_len
        self.block += 1
        return self._stream[s : s + self.seq_len + 1]

    @property
    def epochs_consumed(self) -> float:
        return self.epoch + self.block / max(self.blocks_per_epoch, 1)

    def state(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "block": self.block}

    def load_state(self, st: Dict[str, int]) -> None:
        if st["epoch"] != self.epoch:
            self.epoch = st["epoch"]
            self._build()
        self.block = st["block"]


class Mixture:
    """Exact, deterministic interleaving: row r is English iff floor((r+1)p) > floor(rp)."""

    def __init__(self, streams: Dict[str, TokenStream], english_ratio: float):
        if english_ratio > 0 and "en" not in streams:
            raise ValueError("english_ratio > 0 but no English training pool was prepared")
        self.streams = streams
        self.p = english_ratio
        self.row = 0

    def next(self) -> Tuple[str, np.ndarray]:
        r, p = self.row, self.p
        self.row += 1
        lang = "en" if p > 0 and int((r + 1) * p) > int(r * p) else "el"
        return lang, self.streams[lang].next_block()

    def state(self) -> Dict:
        return {"row": self.row, "streams": {k: s.state() for k, s in self.streams.items()}}

    def load_state(self, st: Dict) -> None:
        self.row = st["row"]
        for k, s in st["streams"].items():
            self.streams[k].load_state(s)
