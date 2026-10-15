"""Typed, strict, inheritable experiment configuration.

* YAML files may say ``inherits: other.yaml`` (path relative to the file); the child
  is deep-merged over the parent.
* ``--set a.b.c=value`` overrides are applied last (value parsed as YAML).
* Unknown keys are an error, so a typo can never silently fall back to a default.
* ``Config.key(section)`` hashes the parts of the config that determine a cached
  artifact (prepared data, tokenizer, packed corpora), so artifacts are shared
  between runs exactly when they should be.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# Bump when cleaning/dedup/chunking code changes in a way that alters the data artifact,
# so stale caches are never silently reused.
DATA_PIPELINE_VERSION = 2

# --------------------------------------------------------------------------- schema


@dataclass
class ModelConfig:
    name: str = "HuggingFaceTB/SmolLM2-135M"
    revision: str = "main"
    attn_implementation: str = "sdpa"


@dataclass
class SourceSpec:
    """One data source and how much of it to use."""

    source: str
    train_mb: float = 0.0  # MB of cleaned text for the training pool (0 = eval-only source)
    dev_docs: int = 0  # held-out docs for learning curves / LR selection
    test_docs: int = 0  # held-out docs for the final evaluation
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    greek: List[SourceSpec] = field(default_factory=list)
    english: List[SourceSpec] = field(default_factory=list)
    min_chars: int = 200
    min_script_ratio: float = 0.6  # share of letters in the expected script
    shuffle_buffer: int = 10_000
    data_seed: int = 1234
    dedup_paragraph_min_chars: int = 60
    minhash: bool = True
    minhash_threshold: float = 0.7


@dataclass
class TokenizerConfig:
    extender: str = "none"  # see vocabx.tokenization.extenders.EXTENDERS
    num_new_tokens: int = 8000
    train_mb: float = 60.0  # MB of the Greek *training* pool used to learn the extension
    min_frequency: int = 2
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitConfig:
    method: str = "fvt_mean"  # see vocabx.modeling.init.INITIALIZERS
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    lr: float = 3e-4  # peak LR of the shared body schedule
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.01  # only on 2-D body matrices
    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    name: str = "wsd"  # wsd | cosine | constant
    warmup_frac: float = 0.075
    decay_frac: float = 0.2
    min_lr_ratio: float = 0.0


@dataclass
class StageSpec:
    """A training stage. Groups: body, norms, emb_old, emb_new."""

    name: str
    until_frac: float  # stage ends at this fraction of max_steps
    trainable: List[str] = field(default_factory=lambda: ["body", "norms", "emb_old", "emb_new"])


@dataclass
class TrainConfig:
    seq_len: int = 1024
    micro_batch_size: int = 4
    grad_accum: int = 8
    max_steps: int = 600
    english_ratio: float = 0.0  # share of sequences drawn from the English replay stream
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    group_lr_mult: Dict[str, float] = field(default_factory=lambda: {"emb_new": 10.0})
    stages: List[StageSpec] = field(default_factory=list)  # empty = one stage, all trainable
    budget_mode: str = "tokens"  # tokens | greek_bytes (stop once a reference run's Greek bytes are reached)
    budget_reference: Optional[str] = None  # run name whose Greek bytes define the budget (greek_bytes mode)
    allow_repeat: bool = False  # refuse to start if the Greek stream would need > 1 epoch (repetition confound)
    precision: str = "auto"  # auto | fp16 | bf16 | fp32
    grad_scaler_init: float = 1024.0
    loss_chunk_tokens: int = 4096  # chunked fp32 cross-entropy (0 = materialise all logits)
    compile: bool = False
    eval_every: int = 50
    log_every: int = 10
    checkpoint_every_min: float = 5.0
    wallclock_limit_min: float = 30.0  # warn (never stop early: that would break budget matching)


@dataclass
class EvalConfig:
    sets: List[str] = field(default_factory=list)  # final-eval sets (source names)
    dev_sets: List[str] = field(default_factory=list)  # periodic-eval sets ("<source>:dev")
    max_chars: int = 1000  # chunk size; checked to fit seq_len-1 base-tokenizer tokens
    max_chunks_per_doc: int = 4
    dev_max_chunks: int = 300
    batch_tokens: int = 16384
    downstream: List[str] = field(default_factory=list)
    downstream_n: int = 500
    item_seed: int = 0  # fixed item subsample for cloze tasks (independent of the data-order seed)
    generation: bool = True
    generation_prompts: int = 16
    generation_bytes: int = 300
    bootstrap_samples: int = 10_000
    full_eval_at_step0: bool = True


@dataclass
class Config:
    run_name: str = "unnamed"
    workdir: str = "artifacts"
    seed: int = 0  # data-order + init seed; arms sharing a seed see the same document order
    plugins: List[str] = field(default_factory=list)
    notes: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---------------------------------------------------------------- cache keys
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def key(self, what: str) -> str:
        d = self.to_dict()
        if what == "data":
            payload = {
                "pipeline_version": DATA_PIPELINE_VERSION,
                "model": d["model"]["name"],
                "revision": d["model"]["revision"],
                "data": d["data"],
                "max_chars": d["eval"]["max_chars"],
                "max_chunks_per_doc": d["eval"]["max_chunks_per_doc"],
                "seq_len": d["train"]["seq_len"],
            }
        elif what == "tokenizer":
            if self.tokenizer.extender == "none":
                return "base-" + _short_hash({"model": d["model"]["name"], "revision": d["model"]["revision"]})
            payload = {"data": self.key("data"), "tokenizer": d["tokenizer"]}
        else:
            raise ValueError(what)
        return _short_hash(payload)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.workdir, "runs", self.run_name)


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- loading


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e-3 / 3e0 (no decimal point) as floats."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text_or_stream):
    return yaml.load(text_or_stream, Loader=_Loader)


def _load_yaml_with_inheritance(path: str, _stack: tuple = ()) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if path in _stack:  # only the current chain counts, so diamonds (A->B->D, A->C->D) are fine
        raise ValueError(f"Config inheritance cycle at {path}")
    with open(path) as f:
        raw = yaml_load(f) or {}
    here = os.path.dirname(path)
    if "plugins" in raw:  # file plugins are relative to the YAML that declares them
        raw["plugins"] = [
            os.path.normpath(os.path.join(here, p)) if (p.endswith(".py") and not os.path.isabs(p)) else p
            for p in raw["plugins"] or []
        ]
    parents = raw.pop("inherits", None)
    if parents is None:
        return raw
    if isinstance(parents, str):
        parents = [parents]
    merged: Dict[str, Any] = {}
    for p in parents:
        ppath = p if os.path.isabs(p) else os.path.join(here, p)
        merged = _deep_merge(merged, _load_yaml_with_inheritance(ppath, _stack + (path,)))
    if "plugins" in merged and "plugins" in raw:  # plugins accumulate along the chain
        raw["plugins"] = list(dict.fromkeys(merged["plugins"] + raw["plugins"]))
    return _deep_merge(merged, raw)


def _apply_override(d: Dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"Override must look like key.sub=value, got {dotted!r}")
    key, value = dotted.split("=", 1)
    parts = key.strip().split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml_load(value)


def _build(cls: Any, data: Any, where: str) -> Any:
    """Strictly convert nested dicts into dataclasses (unknown keys raise)."""
    if dataclasses.is_dataclass(cls):
        if not isinstance(data, dict):
            raise TypeError(f"{where}: expected a mapping for {cls.__name__}, got {type(data).__name__}")
        hints = typing.get_type_hints(cls)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise KeyError(f"{where}: unknown key(s) {sorted(unknown)}; valid: {sorted(names)}")
        kwargs = {k: _build(hints[k], v, f"{where}.{k}") for k, v in data.items()}
        return cls(**kwargs)
    origin = typing.get_origin(cls)
    args = typing.get_args(cls)
    if origin in (list, List):
        return [_build(args[0], x, f"{where}[{i}]") for i, x in enumerate(data or [])]
    if origin in (dict, Dict):
        return {k: _coerce_number(v) for k, v in (data or {}).items()}
    if origin is typing.Union:  # Optional[X]
        if data is None:
            return None
        inner = [a for a in args if a is not type(None)][0]
        return _build(inner, data, where)
    if cls is float and isinstance(data, (int, str)):
        return float(data)
    if cls is int and isinstance(data, str):
        return int(float(data))
    if cls is int and isinstance(data, float) and data.is_integer():
        return int(data)
    return data


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce_number(v: Any) -> Any:
    if isinstance(v, str) and _NUM.match(v.strip()):
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    return v


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    raw = _load_yaml_with_inheritance(path) if path else {}
    for o in overrides or []:
        _apply_override(raw, o)
    return _build(Config, raw, "config")


def dump_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
