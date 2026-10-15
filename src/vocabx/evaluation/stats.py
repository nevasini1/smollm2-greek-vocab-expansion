"""Paired, document-clustered bootstrap for BPB and accuracy contrasts.

Chunks of one document are correlated, so we resample DOCUMENTS (clusters), with the
same resampling weights for every model -> paired comparisons. BPB of a resample is a
ratio of sums (micro-average within a set); a language's macro-BPB is the mean over
its sets, each set resampled independently. Any contrast (Exp2 - Exp1, the extra
English cost, an exchange rate) is then a function of the per-model bootstrap
arrays, so its CI comes for free.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..utils import LOG
from .scoring import LN2


def _doc_arrays(records: Sequence[Dict], set_name: str) -> Dict[str, Tuple[float, int]]:
    agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    for r in records:
        if r["set"] == set_name:
            a = agg[r["doc_id"]]
            a[0] += r["nll"]
            a[1] += r["n_bytes"]
    return {k: (v[0], v[1]) for k, v in agg.items()}


def bootstrap_bpb(
    models: Dict[str, Sequence[Dict]],
    sets: Sequence[str],
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {model: {set: point+bootstrap BPB array of shape [n_boot + 1]}} (index 0 = point estimate)."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
    for s in sets:
        tables = {m: _doc_arrays(recs, s) for m, recs in models.items()}
        tables = {m: t for m, t in tables.items() if t}  # models that were evaluated on this set
        missing = [m for m in models if m not in tables]
        if missing:
            LOG.warning("set %s missing for %s; their macro BPB excludes it", s, missing)
        if not tables:
            continue
        docs = sorted(set.intersection(*[set(t) for t in tables.values()]))
        if not docs:
            LOG.warning("set %s: no documents shared by all models; skipped", s)
            continue
        ref = next(iter(tables.values()))
        for m, t in tables.items():  # pairing is only valid on identical text
            bad = [d for d in docs if t[d][1] != ref[d][1]]
            if bad:
                raise ValueError(f"set {s}: model {m} scored different bytes on {len(bad)} docs (different eval data?)")
        n = len(docs)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
        W = np.vstack([np.ones((1, n)), W])
        for m, t in tables.items():
            nll = np.array([t[d][0] for d in docs])
            nb = np.array([t[d][1] for d in docs], dtype=np.float64)
            # all models share bytes (same text), so this is a paired resample
            out[m][s] = (W @ nll) / (W @ nb) / LN2
    return out


def macro(boot: Dict[str, np.ndarray], sets: Sequence[str]) -> np.ndarray:
    arrs = [boot[s] for s in sets if s in boot]
    return np.mean(arrs, axis=0) if arrs else np.array([np.nan])


def ci(arr: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
    """(point, lo, hi) from a [n_boot+1] array whose index 0 is the point estimate."""
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2 or not np.isfinite(arr[1:]).any():
        p = float(arr[0]) if arr.size else float("nan")
        return p, float("nan"), float("nan")
    lo, hi = np.nanpercentile(arr[1:], [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(arr[0]), float(lo), float(hi)


def bootstrap_accuracy(models: Dict[str, Sequence[Dict]], metric: str = "acc_norm", n_boot: int = 10_000, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = sorted(set.intersection(*[{r["id"] for r in recs} for recs in models.values()]))
    n = len(ids)
    if n == 0:
        return {}
    W = np.vstack([np.ones((1, n)), rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)]).astype(np.float64)
    out = {}
    for m, recs in models.items():
        lut = {r["id"]: r[metric] for r in recs}
        v = np.array([lut[i] for i in ids])
        out[m] = (W @ v) / n
    return out
