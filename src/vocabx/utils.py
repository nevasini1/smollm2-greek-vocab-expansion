"""Small shared helpers: logging, seeding, devices, JSONL I/O, hashing, environment capture."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

LOG = logging.getLogger("vocabx")


def setup_logging(level: int = logging.INFO, logfile: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- device


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(precision: str, device: str):
    """Return (autocast_dtype or None, use_grad_scaler).

    Master weights are always fp32. On a T4 (no native bf16) we use fp16 autocast
    plus a GradScaler; ``torch.cuda.is_bf16_supported()`` reports True on Turing via
    emulation, so we never rely on it.
    """
    import torch

    if precision == "auto":
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            precision = "bf16" if major >= 8 else "fp16"
        else:
            precision = "fp32"
    if precision == "fp32":
        return None, False
    if precision == "fp16":
        return torch.float16, device == "cuda"
    if precision == "bf16":
        return torch.bfloat16, False
    raise ValueError(f"unknown precision {precision!r}")


# ----------------------------------------------------------------------------- I/O


def _open(path: str, mode: str, gz: Optional[bool] = None):
    if gz is None:
        gz = path.endswith(".gz")
        if mode == "r" and gz:  # tolerate plain-text files that carry a .gz name
            with open(path, "rb") as fb:
                gz = fb.read(2) == b"\x1f\x8b"
    if gz:
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    tmp = path + ".tmp"
    with _open(tmp, "w", gz=path.endswith(".gz")) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with _open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


# ----------------------------------------------------------------------------- misc


@contextmanager
def timed(label: str):
    t0 = time.time()
    yield
    LOG.info("%s took %.1fs", label, time.time() - t0)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "tokenizers", "datasets", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:  # pragma: no cover
        pass
    return info


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"
