"""Throughput calibration: how many matched steps fit in the wall-clock limit on THIS GPU.

Run it on the SLOWER arm (Exp2: larger vocab) and use the suggested ``max_steps`` for
both arms. Also records which SDPA kernel works (on a T4, flash attention is
unavailable; GQA + memory-efficient attention depends on the torch version).
"""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Dict

import torch

from ..utils import LOG, environment_info, write_json
from .trainer import Trainer, lm_loss_sum


def _sdpa_probe(model, device: str, amp_dtype) -> Dict[str, Any]:
    if device != "cuda":
        return {"checked": False}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover
        return {"checked": False}
    out: Dict[str, Any] = {"checked": True}
    x = torch.zeros((1, 256), dtype=torch.long, device=device)
    # same call pattern as training (no attention mask, grad enabled), restricted to one kernel
    for name, be in (("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel([be]), torch.autocast("cuda", dtype=amp_dtype or torch.float16):
                lm_loss_sum(model, x[:, :-1], x[:, 1:]).backward()
            model.zero_grad(set_to_none=True)
            out[name] = True
        except Exception as e:
            out[name] = f"unavailable: {type(e).__name__}"
    out["training_uses_fused_kernel"] = out.get("efficient") is True or out.get("flash") is True
    if not out["training_uses_fused_kernel"]:
        LOG.error("training attention falls back to the MATH kernel (slow, ~2x activation memory): %s", out)
    return out


def calibrate(cfg, steps: int = 20, target_minutes: float = 27.0) -> Dict[str, Any]:
    tr = Trainer(cfg, for_calibration=True)
    if tr.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        m = tr.train_step(s)
        times.append(m["step_seconds"])
        LOG.info("calibration step %d: %.2fs (%.0f tok/s)", s + 1, m["step_seconds"], m["tokens_per_s"])
    warm = times[min(3, len(times) - 1):]
    sec_per_step = statistics.median(warm)
    tok_s = tr.tokens_per_step / sec_per_step
    suggested = int(math.floor(target_minutes * 60 / sec_per_step))
    res = {
        "run_name": cfg.run_name,
        "environment": environment_info(),
        "device": tr.device,
        "amp_dtype": str(tr.amp_dtype),
        "tokens_per_step": tr.tokens_per_step,
        "seconds_per_step": sec_per_step,
        "tokens_per_second": tok_s,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if tr.device == "cuda" else None,
        "target_minutes": target_minutes,
        "suggested_max_steps": suggested,
        "suggested_tokens_M": suggested * tr.tokens_per_step / 1e6,
        "greek_bytes_per_token": tr.packed["el"].bytes_per_token,
        "greek_MB_needed_at_suggested": suggested * tr.tokens_per_step * (1 - cfg.train.english_ratio) * tr.packed["el"].bytes_per_token / 1e6,
        "greek_MB_available": tr.packed["el"].n_bytes / 1e6,
        "sdpa": _sdpa_probe(tr.model, tr.device, tr.amp_dtype),
    }
    # the Greek pool must cover the run without repetition (else only this arm repeats data)
    per_step_bytes = tr.tokens_per_step * (1 - cfg.train.english_ratio) * tr.packed["el"].bytes_per_token
    res["pool_cap_steps"] = int(tr.packed["el"].n_bytes // max(per_step_bytes, 1))
    res["recommended_max_steps"] = min(suggested, res["pool_cap_steps"])
    path = os.path.join(cfg.workdir, "calibration", f"{cfg.run_name}.json")
    write_json(path, res)
    LOG.info("calibration: %.0f tok/s, %.2f s/step -> %d steps fit %.0f min; pool allows %d; recommended max_steps=%d",
             tok_s, sec_per_step, suggested, target_minutes, res["pool_cap_steps"], res["recommended_max_steps"])
    return res


def persist_budget(workdir: str, res: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Write <workdir>/budget.json ONCE; every later session/run reuses the same max_steps."""
    from ..utils import read_json

    path = os.path.join(workdir, "budget.json")
    if os.path.exists(path) and not force:
        old = read_json(path)
        LOG.info("budget already fixed at max_steps=%d (%s); not overwriting (use --force)", old["max_steps"], path)
        return old
    budget = {"max_steps": res["recommended_max_steps"], "from_calibration": res}
    write_json(path, budget)
    LOG.info("budget fixed: max_steps=%d -> %s", budget["max_steps"], path)
    return budget
