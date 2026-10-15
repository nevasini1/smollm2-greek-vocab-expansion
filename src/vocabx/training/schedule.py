"""Learning-rate schedules (multipliers on a group's peak LR).

Default is WSD (warmup-stable-decay), mirroring SmolLM2's own pretraining schedule:
linear warmup, constant, linear decay to ``min_lr_ratio`` over the final
``decay_frac`` of steps. Decay is anchored to *absolute* steps so every parameter
group - including groups unfrozen mid-run - decays over exactly the same steps in
both experiments. A group that becomes trainable at ``start_step`` re-warms from there.
"""

from __future__ import annotations

import math

from ..config import ScheduleConfig


def lr_multiplier(step: int, max_steps: int, sc: ScheduleConfig, start_step: int = 0) -> float:
    warmup = max(1, round(sc.warmup_frac * max_steps))
    if step < start_step:
        return 0.0
    w = min(1.0, (step - start_step + 1) / warmup)
    if sc.name == "constant":
        return w
    if sc.name == "wsd":
        decay_steps = max(1, round(sc.decay_frac * max_steps))
        decay_start = max_steps - decay_steps
        if step < decay_start:
            return w
        # first decay step keeps the peak, the last one is at 1/decay_steps (never a wasted 0-LR step)
        frac = min(1.0, (step - decay_start) / decay_steps)
        return w * (1.0 - frac * (1.0 - sc.min_lr_ratio))
    if sc.name == "cosine":
        progress = min(1.0, max(0.0, (step - start_step) / max(1, max_steps - start_step)))
        return w * (sc.min_lr_ratio + (1 - sc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown schedule {sc.name!r}")
