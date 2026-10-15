"""Base tokenizer loading and GPT-2 byte-level helpers."""

from __future__ import annotations

import functools
import re
from typing import Dict, List

from tokenizers import Tokenizer


@functools.lru_cache(maxsize=4)
def load_base_tokenizer(model_name: str, revision: str = "main") -> Tokenizer:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "tokenizer.json", revision=revision)
    return Tokenizer.from_file(path)


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map used by ByteLevel BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


@functools.lru_cache(maxsize=1)
def unicode_to_bytes() -> Dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    """Byte-level token string (e.g. 'ĠÎ±') -> raw bytes (b' \\xce\\xb1')."""
    u2b = unicode_to_bytes()
    return bytes(u2b[c] for c in token)


def bytes_to_token(b: bytes) -> str:
    b2u = bytes_to_unicode()
    return "".join(b2u[x] for x in b)


def text_to_token_string(text: str) -> str:
    return bytes_to_token(text.encode("utf-8"))


# Greek-unique UTF-8 byte patterns. 0xCE/0xCF only ever lead U+0380-U+03FF (Greek and
# Coptic); 0xE1 followed by 0xBC-0xBF only ever leads U+1F00-U+1FFF (Greek Extended).
# 0xCD is NOT included: it also leads the combining diacritics U+0340-U+036F that every
# script uses.
_GREEK_BYTES = re.compile(rb"[\xce\xcf]|\xe1[\xbc-\xbf]")


def has_greek_bytes(b: bytes) -> bool:
    return _GREEK_BYTES.search(b) is not None


def token_has_greek_bytes(token: str) -> bool:
    return has_greek_bytes(token_to_bytes(token))


def _is_greek_cp(cp: int) -> bool:
    return 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF


def is_strict_greek_token(token: str) -> bool:
    """True if the token's bytes are Greek-only.

    Allowed: an optional leading space, complete Greek characters, and *partial*
    UTF-8 sequences at either edge (byte-level BPE can split a character between
    tokens). Any complete non-Greek character (Latin, digit, punctuation, whitespace
    run) disqualifies the token, and the token must contain a Greek-unique byte
    pattern so it can never fire on non-Greek text.
    """
    b = token_to_bytes(token)
    if not has_greek_bytes(b):
        return False
    if b.startswith(b" "):
        b = b[1:]
    i, n = 0, len(b)
    # leading continuation bytes = tail of a character started in a previous token
    while i < n and 0x80 <= b[i] <= 0xBF:
        i += 1
    while i < n:
        lead = b[i]
        if lead < 0x80:
            return False  # ASCII (letter, digit, space, punctuation)
        if 0xC0 <= lead <= 0xDF:
            size = 2
        elif 0xE0 <= lead <= 0xEF:
            size = 3
        elif 0xF0 <= lead <= 0xF7:
            size = 4
        else:
            return False
        if i + size > n:  # partial character at the right edge: must be a Greek lead
            return has_greek_bytes(b[i:]) or b[i:] in (b"\xe1",)
        try:
            ch = b[i : i + size].decode("utf-8")
        except UnicodeDecodeError:
            return False
        if not _is_greek_cp(ord(ch)):
            return False
        i += size
    return True


def is_valid_utf8_token(token: str) -> bool:
    try:
        token_to_bytes(token).decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def merges_format(tokenizer_json: Dict) -> str:
    merges: List = tokenizer_json["model"]["merges"]
    if not merges:
        return "pairs"
    return "string" if isinstance(merges[0], str) else "pairs"
