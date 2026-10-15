"""Tokenizer extenders (registry) and the on-disk tokenizer bundle.

An extender maps (base tokenizer, Greek training texts, config) -> ``Extension``:
the modified ``tokenizer.json`` plus the list of new tokens (ids ``n_base + i``) and,
when known, each new token's merge parents. Registered extenders:

* ``none``          - Exp1: original tokenizer.
* ``continued_bpe`` - Exp2 default: merges learned over SmolLM2's own segmentation.
* ``naive_merge``   - ablation: train a separate Greek BPE, append its novel merges.
* ``added_tokens``  - negative control: frequent Greek words via ``add_tokens``.

All extenders keep every original id and merge untouched (append-only).
"""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tokenizers import AddedToken, Tokenizer

from ..config import Config, TokenizerConfig
from ..registry import Registry
from ..utils import LOG, read_json, write_json
from .base import (
    bytes_to_token,
    is_strict_greek_token,
    load_base_tokenizer,
    merges_format,
    token_has_greek_bytes,
)
from .continued_bpe import count_pretokens, greek_contains_filter, learn_continued_bpe

EXTENDERS = Registry("tokenizer extender")

TextsFn = Callable[[], Iterable[str]]


@dataclass
class Extension:
    tokenizer_json: Dict[str, Any]
    new_tokens: List[str] = field(default_factory=list)  # byte-level strings
    parents: Optional[List[Tuple[int, int]]] = None
    info: Dict[str, Any] = field(default_factory=dict)


def _accept_fn(params: Dict[str, Any]):
    mode = params.get("script_filter", "strict")
    if mode == "strict":
        return is_strict_greek_token
    if mode == "contains":
        return greek_contains_filter
    raise ValueError(f"script_filter must be strict|contains, got {mode!r}")


def append_merges(base_json: Dict[str, Any], merges: List[Tuple[str, str]], new_tokens: List[str]) -> Dict[str, Any]:
    """Append vocab entries + merges, keeping the file's existing merges format."""
    tj = copy.deepcopy(base_json)
    vocab: Dict[str, int] = tj["model"]["vocab"]
    added_ids = [t["id"] for t in tj.get("added_tokens", [])]
    next_id = max(max(vocab.values()), max(added_ids, default=-1)) + 1
    for tok in new_tokens:
        if tok in vocab:
            raise ValueError(f"token {tok!r} already in vocab")
        vocab[tok] = next_id
        next_id += 1
    fmt = merges_format(tj)
    for a, b in merges:
        if " " in a or " " in b:
            raise ValueError("byte-level tokens never contain a literal space")
        tj["model"]["merges"].append(f"{a} {b}" if fmt == "string" else [a, b])
    return tj


@EXTENDERS.register("none")
def extend_none(base: Tokenizer, texts: TextsFn, cfg: TokenizerConfig) -> Extension:
    return Extension(json.loads(base.to_str()))


@EXTENDERS.register("continued_bpe")
def extend_continued_bpe(base: Tokenizer, texts: TextsFn, cfg: TokenizerConfig) -> Extension:
    counts = count_pretokens(base, texts())
    res = learn_continued_bpe(
        base,
        counts,
        num_merges=cfg.num_new_tokens,
        min_frequency=cfg.min_frequency,
        accept=_accept_fn(cfg.params),
        min_word_count=int(cfg.params.get("min_word_count", 1)),
    )
    tj = append_merges(json.loads(base.to_str()), res.merges, res.new_tokens)
    return Extension(
        tj,
        res.new_tokens,
        res.parents,
        {"word_types": res.word_types, "word_occurrences": res.words_total, "merge_counts": res.merge_counts},
    )


@EXTENDERS.register("naive_merge")
def extend_naive_merge(base: Tokenizer, texts: TextsFn, cfg: TokenizerConfig) -> Extension:
    """Train a stand-alone Greek byte-level BPE, then append its novel merges in rank order."""
    from tokenizers import models, pre_tokenizers, trainers

    base_json = json.loads(base.to_str())
    accept = _accept_fn(cfg.params)
    side = Tokenizer(models.BPE())
    side.pre_tokenizer = base.pre_tokenizer  # same Digits + ByteLevel(GPT-2 regex)
    vocab_size = int(cfg.params.get("side_vocab_size", 256 + 4 * cfg.num_new_tokens))
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=cfg.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    side.train_from_iterator(texts(), trainer)
    side_merges = json.loads(side.to_str())["model"]["merges"]
    vocab = set(base.get_vocab(with_added_tokens=True))
    merges: List[Tuple[str, str]] = []
    new_tokens: List[str] = []
    skipped = Counter()
    for m in side_merges:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        s = a + b
        if s in vocab:
            skipped["exists"] += 1
            continue
        if a not in vocab or b not in vocab:
            skipped["missing_parent"] += 1
            continue
        if not accept(s):
            skipped["script"] += 1
            continue
        merges.append((a, b))
        new_tokens.append(s)
        vocab.add(s)
        if len(new_tokens) >= cfg.num_new_tokens:
            break
    if len(new_tokens) < cfg.num_new_tokens:
        LOG.warning("naive_merge produced %d/%d tokens; raise params.side_vocab_size", len(new_tokens), cfg.num_new_tokens)
    ids = base.get_vocab(with_added_tokens=True)
    n_base = max(ids.values()) + 1
    for i, t in enumerate(new_tokens):
        ids[t] = n_base + i
    parents = [(ids[a], ids[b]) for a, b in merges]
    return Extension(append_merges(base_json, merges, new_tokens), new_tokens, parents, {"skipped": dict(skipped)})


@EXTENDERS.register("added_tokens")
def extend_added_tokens(base: Tokenizer, texts: TextsFn, cfg: TokenizerConfig) -> Extension:
    """Negative control: the most frequent multi-token Greek words as AddedTokens."""
    counts: Counter = Counter()
    for text in texts():
        for w in text.split():
            w = w.strip(".,;:!?«»\"'()[]·–—-")
            if w and any("Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in w):
                counts[w] += 1
    vocab = base.get_vocab(with_added_tokens=True)
    words = []
    for w, _ in counts.most_common():
        tok = bytes_to_token(w.encode("utf-8"))
        if tok in vocab or len(base.encode(w, add_special_tokens=False).ids) < 2:
            continue
        words.append(w)
        if len(words) >= cfg.num_new_tokens:
            break
    tk = Tokenizer.from_str(base.to_str())
    tk.add_tokens([AddedToken(w, single_word=bool(cfg.params.get("single_word", False)), normalized=False) for w in words])
    return Extension(json.loads(tk.to_str()), [bytes_to_token(w.encode("utf-8")) for w in words], None, {"kind": "added_tokens"})


# ----------------------------------------------------------------------------- bundle


@dataclass
class TokenizerBundle:
    tokenizer: Tokenizer
    n_base: int
    new_tokens: List[str]
    parents: Optional[List[Tuple[int, int]]]
    path: str
    meta: Dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return self.n_base + len(self.new_tokens)

    @property
    def n_new(self) -> int:
        return len(self.new_tokens)

    @staticmethod
    def from_model_dir(path: str, base_model: str, revision: str = "main") -> "TokenizerBundle":
        """Rebuild a bundle from a saved model dir (tokenizer.json only): new tokens = ids >= base size."""
        if os.path.exists(os.path.join(path, "meta.json")):
            return TokenizerBundle.load(path)
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        base = load_base_tokenizer(base_model, revision)
        n_base = base.get_vocab_size(with_added_tokens=True)
        bv = base.get_vocab(with_added_tokens=True)
        ev = tok.get_vocab(with_added_tokens=True)
        if any(ev.get(t) != i for t, i in bv.items()):
            raise ValueError(f"{path}/tokenizer.json does not extend {base_model}'s vocabulary")
        new = [t for t, i in sorted(ev.items(), key=lambda kv: kv[1]) if i >= n_base]
        meta = {"base_model": base_model, "base_revision": revision, "n_base": n_base, "new_tokens": new,
                "parents": None, "extender": "unknown (rebuilt from model dir)"}
        return TokenizerBundle(tok, n_base, new, None, path, meta)

    @staticmethod
    def load(path: str) -> "TokenizerBundle":
        meta = read_json(os.path.join(path, "meta.json"))
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        parents = [tuple(p) for p in meta["parents"]] if meta.get("parents") else None
        return TokenizerBundle(tok, meta["n_base"], meta["new_tokens"], parents, path, meta)

    def save_hf(self, out_dir: str) -> None:
        """Write HF-loadable tokenizer files + this bundle's meta.json next to a model.

        With meta.json present, ``TokenizerBundle.load(model_dir)`` reconstructs exactly the
        tokenizer the model was trained with (used by ``vocabx eval``).
        """
        save_hf_tokenizer(self.tokenizer, out_dir)
        write_json(os.path.join(out_dir, "meta.json"), self.meta)


def save_hf_tokenizer(tok: Tokenizer, out_dir: str) -> None:
    from transformers import PreTrainedTokenizerFast

    specials = [t.content for t in tok.get_added_tokens_decoder().values() if t.special]
    eos = "<|endoftext|>" if "<|endoftext|>" in specials else (specials[0] if specials else None)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(tok.to_str()),
        bos_token=eos,
        eos_token=eos,
        unk_token=eos,
        additional_special_tokens=[s for s in specials if s != eos],
    )
    hf.save_pretrained(out_dir)


def tokenizer_dir(cfg: Config) -> str:
    return os.path.join(cfg.workdir, "tokenizers", cfg.key("tokenizer"))


def build_tokenizer(cfg: Config, force: bool = False) -> TokenizerBundle:
    """Create (or reuse) the tokenizer bundle for this config."""
    from ..data.prepare import data_dir, prepare_data
    from ..utils import read_jsonl

    out = tokenizer_dir(cfg)
    if os.path.exists(os.path.join(out, "meta.json")) and not force:
        b = TokenizerBundle.load(out)
        if b.meta.get("base_model") != cfg.model.name or b.meta.get("base_revision", "main") != cfg.model.revision:
            raise ValueError(f"cached tokenizer {out} was built for {b.meta.get('base_model')}@{b.meta.get('base_revision')}, "
                             f"not {cfg.model.name}@{cfg.model.revision}; delete it or use another workdir")
        return b
    base = load_base_tokenizer(cfg.model.name, cfg.model.revision)
    n_base = base.get_vocab_size(with_added_tokens=True)

    if cfg.tokenizer.extender == "none":
        ext = Extension(json.loads(base.to_str()))
    else:
        prepare_data(cfg, base)
        train_path = os.path.join(data_dir(cfg), "train_el.jsonl.gz")
        budget = int(cfg.tokenizer.train_mb * 1e6)

        def texts() -> Iterable[str]:
            used = 0
            for row in read_jsonl(train_path):
                if used >= budget:
                    break
                used += len(row["text"].encode("utf-8"))
                yield row["text"]

        LOG.info("building tokenizer: extender=%s, %d new tokens, %.0f MB of Greek training text",
                 cfg.tokenizer.extender, cfg.tokenizer.num_new_tokens, cfg.tokenizer.train_mb)
        ext = EXTENDERS.get(cfg.tokenizer.extender)(base, texts, cfg.tokenizer)

    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_str(json.dumps(ext.tokenizer_json))
    tok.save(os.path.join(out, "tokenizer.json"))
    meta = {
        "extender": cfg.tokenizer.extender,
        "tokenizer_config": cfg.to_dict()["tokenizer"],
        "base_model": cfg.model.name,
        "base_revision": cfg.model.revision,
        "n_base": n_base,
        "n_new": len(ext.new_tokens),
        "new_tokens": ext.new_tokens,
        "parents": ext.parents,
        "info": ext.info,
    }
    write_json(os.path.join(out, "meta.json"), meta)
    bundle = TokenizerBundle.load(out)
    if bundle.tokenizer.get_vocab_size(with_added_tokens=True) != bundle.vocab_size:
        raise AssertionError(
            f"vocab size mismatch: tokenizer {bundle.tokenizer.get_vocab_size(with_added_tokens=True)} vs bundle {bundle.vocab_size}"
        )
    LOG.info("tokenizer bundle saved to %s (vocab %d = %d + %d)", out, bundle.vocab_size, n_base, bundle.n_new)
    return bundle
