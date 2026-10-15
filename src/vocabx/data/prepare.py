"""Build the cached data artifact: cleaned training pools + fixed, decontaminated eval sets.

Layout (``<workdir>/data/<data_key>/``)::

    train_el.jsonl.gz   training pool, Greek   (docs, already shuffled with data_seed)
    train_en.jsonl.gz   training pool, English (replay stream; may be empty)
    eval/<set>.jsonl    held-out chunks, one row per chunk (identical for all models)
    manifest.json       counts, bytes, decontamination stats, file hashes

Order of operations matters for validity:
 1. held-out docs are drawn FIRST (official test split, or the head of the shuffled
    train stream) and frozen;
 2. the training pool is streamed afterwards and every candidate doc is checked
    against an index of ALL held-out docs of the same language (paragraph hashes +
    MinHash), across sources;
 3. the tokenizer extension is later learned only from the training pool.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config import Config, SourceSpec
from ..utils import LOG, human_bytes, read_json, sha256_file, text_hash, write_json, write_jsonl
from .dedup import DecontaminationIndex
from .sources import DATA_SOURCES, Doc, make_source
from .text import LANG_SCRIPT, chunk_text, normalize_text, script_ratio


def data_dir(cfg: Config) -> str:
    return os.path.join(cfg.workdir, "data", cfg.key("data"))


def eval_set_path(cfg: Config, name: str) -> str:
    return os.path.join(data_dir(cfg), "eval", name.replace(":", "__") + ".jsonl")


class _Cleaner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.seen: set = set()
        self.stats: Dict[str, int] = {}

    def __call__(self, doc: Doc, lang: str) -> Optional[Doc]:
        text = normalize_text(doc.text)
        reason = None
        if len(text) < self.cfg.data.min_chars:
            reason = "too_short"
        elif script_ratio(text, LANG_SCRIPT[lang]) < self.cfg.data.min_script_ratio:
            reason = "wrong_script"
        else:
            h = text_hash(text)
            if h in self.seen:
                reason = "exact_dup"
            else:
                self.seen.add(h)
        if reason:
            self.stats[reason] = self.stats.get(reason, 0) + 1
            return None
        return Doc(doc.id, text, doc.meta)


def _take(it: Iterator[Doc], n: int, clean: _Cleaner, lang: str) -> List[Doc]:
    out: List[Doc] = []
    while len(out) < n:
        try:
            d = next(it)
        except StopIteration:
            LOG.warning("stream exhausted after %d/%d held-out docs", len(out), n)
            break
        c = clean(d, lang)
        if c is not None:
            out.append(c)
    return out


def prepare_data(cfg: Config, base_tokenizer=None, force: bool = False) -> str:
    """Create (or reuse) the data artifact and return its directory."""
    out_dir = data_dir(cfg)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        LOG.info("data artifact exists: %s", out_dir)
        return out_dir
    os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)

    if base_tokenizer is None:
        from ..tokenization.base import load_base_tokenizer

        base_tokenizer = load_base_tokenizer(cfg.model.name, cfg.model.revision)
    max_tokens = cfg.train.seq_len - 1  # +1 for the <|endoftext|> prefix

    def fits(t: str) -> bool:
        return len(base_tokenizer.encode(t, add_special_tokens=False).ids) <= max_tokens

    clean = _Cleaner(cfg)
    specs: List[Tuple[str, SourceSpec]] = [("el", s) for s in cfg.data.greek] + [("en", s) for s in cfg.data.english]

    # ---- 1. held-out docs first --------------------------------------------------
    heldout: Dict[str, List[Doc]] = {}  # set name -> docs
    set_meta: Dict[str, Dict[str, Any]] = {}
    train_streams: Dict[str, Iterator[Doc]] = {}
    for lang, spec in specs:
        src = make_source(spec.source, spec.params)
        if src.language != lang:
            raise ValueError(f"source {spec.source} is '{src.language}' but listed under '{lang}'")
        seed = cfg.data.data_seed
        need_train_stream = spec.train_mb > 0 or spec.dev_docs > 0 or (spec.test_docs > 0 and not src.has_official_test)
        stream = src.iter_docs("train", seed, cfg.data.shuffle_buffer) if need_train_stream and not src.eval_only else None
        if spec.test_docs > 0:
            if src.has_official_test:
                docs = _take(src.iter_docs("test", seed, cfg.data.shuffle_buffer), spec.test_docs, clean, lang)
            else:
                docs = _take(stream, spec.test_docs, clean, lang)
            heldout[spec.source] = docs
            set_meta[spec.source] = {"language": lang, "source": spec.source, "role": "test"}
        if spec.dev_docs > 0:
            if stream is None:
                raise ValueError(f"{spec.source}: dev docs need a train stream")
            heldout[f"{spec.source}:dev"] = _take(stream, spec.dev_docs, clean, lang)
            set_meta[f"{spec.source}:dev"] = {"language": lang, "source": spec.source, "role": "dev"}
        if spec.train_mb > 0:
            if stream is None:
                raise ValueError(f"{spec.source} is eval-only; set train_mb: 0")
            train_streams[spec.source] = stream

    # ---- 2. decontamination index per language ------------------------------------
    index: Dict[str, DecontaminationIndex] = {}
    for name, docs in heldout.items():
        lang = set_meta[name]["language"]
        idx = index.setdefault(
            lang,
            DecontaminationIndex(cfg.data.dedup_paragraph_min_chars, cfg.data.minhash, cfg.data.minhash_threshold),
        )
        for d in docs:
            idx.add(d.text)

    # ---- 3. training pools ----------------------------------------------------------
    pools: Dict[str, List[Dict[str, Any]]] = {"el": [], "en": []}
    decontam: Dict[str, Dict[str, int]] = {}
    for lang, spec in specs:
        if spec.source not in train_streams:
            continue
        target = int(spec.train_mb * 1e6)
        got = 0
        dropped: Dict[str, int] = {}
        it = train_streams[spec.source]
        idx = index.get(lang)
        while got < target:
            try:
                d = next(it)
            except StopIteration:
                LOG.warning("%s exhausted at %s of %s", spec.source, human_bytes(got), human_bytes(target))
                break
            c = clean(d, lang)
            if c is None:
                continue
            why = idx.contaminated(c.text) if idx else None
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            pools[lang].append({"id": c.id, "source": spec.source, "text": c.text})
            got += len(c.text.encode("utf-8"))
        decontam[spec.source] = dropped
        LOG.info("%s: %s of training text (%d docs); dropped vs held-out: %s", spec.source, human_bytes(got),
                 sum(1 for r in pools[lang] if r["source"] == spec.source), dropped)

    files: Dict[str, Any] = {}
    for lang, rows in pools.items():
        random.Random(cfg.data.data_seed).shuffle(rows)  # mix sources; tokenizer uses a prefix
        path = os.path.join(out_dir, f"train_{lang}.jsonl.gz")
        write_jsonl(path, rows)
        files[f"train_{lang}"] = {
            "path": os.path.basename(path),
            "docs": len(rows),
            "bytes": sum(len(r["text"].encode("utf-8")) for r in rows),
        }

    # ---- 4. chunk held-out docs into fixed eval sets ---------------------------------
    for name, docs in heldout.items():
        meta = set_meta[name]
        rows = []
        for d in docs:
            for ci, chunk in enumerate(chunk_text(d.text, cfg.eval.max_chars, cfg.eval.max_chunks_per_doc, fits)):
                rows.append({"set": name, "language": meta["language"], "doc_id": d.id, "chunk": ci, "text": chunk})
        path = eval_set_path(cfg, name)
        write_jsonl(path, rows)
        meta.update(
            docs=len({r["doc_id"] for r in rows}),
            chunks=len(rows),
            bytes=sum(len(r["text"].encode("utf-8")) for r in rows),
            file=os.path.relpath(path, out_dir),
            sha256=sha256_file(path),
        )
        LOG.info("eval set %-24s %4d docs %5d chunks %s", name, meta["docs"], meta["chunks"], human_bytes(meta["bytes"]))

    write_json(
        manifest_path,
        {
            "data_key": cfg.key("data"),
            "data_config": cfg.to_dict()["data"],
            "max_chars": cfg.eval.max_chars,
            "max_tokens_per_chunk": max_tokens,
            "train": files,
            "eval_sets": set_meta,
            "cleaning_drops": clean.stats,
            "decontamination_drops": decontam,
        },
    )
    LOG.info("data artifact ready: %s", out_dir)
    return out_dir


def load_manifest(cfg: Config) -> Dict[str, Any]:
    return read_json(os.path.join(data_dir(cfg), "manifest.json"))
