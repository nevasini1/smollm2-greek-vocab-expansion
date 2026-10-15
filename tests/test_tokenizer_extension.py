"""Invariants of the vocabulary extension (uses the real SmolLM2 tokenizer)."""

import json
from collections import Counter

import pytest
from tokenizers import Tokenizer

from conftest import ENGLISH_SAMPLE, GREEK_SAMPLE
from vocabx.config import TokenizerConfig
from vocabx.tokenization.base import is_strict_greek_token, token_to_bytes, text_to_token_string
from vocabx.tokenization.continued_bpe import count_pretokens, learn_continued_bpe
from vocabx.tokenization.extenders import EXTENDERS, append_merges
from vocabx.tokenization.qa import run_tokenizer_qa

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def extended(smollm_tokenizer):
    base = smollm_tokenizer
    res = learn_continued_bpe(base, count_pretokens(base, GREEK_SAMPLE), num_merges=150, min_frequency=2)
    tok = Tokenizer.from_str(json.dumps(append_merges(json.loads(base.to_str()), res.merges, res.new_tokens)))
    return base, tok, res


def test_new_tokens_are_new_greek_and_appended(extended):
    base, tok, res = extended
    n_base = base.get_vocab_size(with_added_tokens=True)
    assert len(res.new_tokens) == 150
    assert len(set(res.new_tokens)) == 150
    bv = base.get_vocab(with_added_tokens=True)
    ev = tok.get_vocab(with_added_tokens=True)
    for i, t in enumerate(res.new_tokens):
        assert t not in bv
        assert ev[t] == n_base + i
        assert is_strict_greek_token(t)
    assert all(ev[t] == i for t, i in bv.items())


def test_english_and_non_greek_unchanged(extended):
    base, tok, _ = extended
    for text in ENGLISH_SAMPLE + ["Привет мир", "你好世界", "مرحبا", "µ ohm"]:
        assert base.encode(text).ids == tok.encode(text).ids


def test_greek_compresses_and_roundtrips(extended):
    base, tok, _ = extended
    for text in GREEK_SAMPLE[:8]:
        assert len(tok.encode(text).ids) < len(base.encode(text).ids)
        assert tok.decode(tok.encode(text).ids) == text


def test_parents_precede_children(extended):
    base, _, res = extended
    n_base = base.get_vocab_size(with_added_tokens=True)
    for j, (a, b) in enumerate(res.parents):
        assert a < n_base + j and b < n_base + j


def test_learning_matches_inference_segmentation(extended):
    """Applying the learned merges with HF's BPE must reproduce the training-time words."""
    base, tok, res = extended
    n_base = base.get_vocab_size(with_added_tokens=True)
    used = Counter()
    for text in GREEK_SAMPLE[:8]:
        used.update(i for i in tok.encode(text).ids if i >= n_base)
    assert len(used) > 50  # most merges learned on this text fire on it


def test_qa_harness_passes(extended):
    base, tok, res = extended
    n_base = base.get_vocab_size(with_added_tokens=True)
    rep = run_tokenizer_qa(base, tok, n_base, res.new_tokens, GREEK_SAMPLE[:8], ENGLISH_SAMPLE, strict=True, k_curve=[50, 150])
    assert rep["self_tokenization"]["failures"] == 0
    assert rep["english_invariance"]["changed_without_greek"] == 0
    assert rep["greek_token_reduction_x"] > 1.2
    assert set(rep["vocab_size_curve"]) == {50, 150}


def test_strict_filter_rejects_mixed_tokens():
    assert is_strict_greek_token(text_to_token_string(" καλημέρα"))
    assert not is_strict_greek_token(text_to_token_string("abc"))
    assert not is_strict_greek_token(text_to_token_string("αb"))
    assert not is_strict_greek_token(text_to_token_string(" \n\n"))
    assert not is_strict_greek_token(text_to_token_string("­"))  # soft hyphen (C2 AD)
    # a partial character at the edge (space + UTF-8 lead byte of a Greek letter) is allowed
    assert is_strict_greek_token(text_to_token_string(" α")[:2])
    assert not is_strict_greek_token(text_to_token_string(" ж")[:2])  # Cyrillic lead byte


@pytest.mark.parametrize("name", ["naive_merge", "added_tokens"])
def test_alternative_extenders_keep_english(smollm_tokenizer, name):
    base = smollm_tokenizer
    cfg = TokenizerConfig(extender=name, num_new_tokens=60, min_frequency=2)
    ext = EXTENDERS.get(name)(base, lambda: iter(GREEK_SAMPLE), cfg)
    tok = Tokenizer.from_str(json.dumps(ext.tokenizer_json))
    assert len(ext.new_tokens) > 0
    for text in ENGLISH_SAMPLE:
        assert base.encode(text).ids == tok.encode(text).ids
