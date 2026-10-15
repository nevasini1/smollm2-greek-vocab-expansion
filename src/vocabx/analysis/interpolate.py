"""WiSE-FT interpolation frontiers (eval-only, no extra training).

For a finished run, evaluate theta(a) = (1 - a) * theta_start + a * theta_final for several a.
theta_start is the model the run *started* from: the original model for Exp1, the expanded
model at step 0 (surgery + init) for Exp2. Each a gives one (Greek BPB, English BPB) point,
so every run yields a frontier instead of a single endpoint. Comparing the two arms'
frontiers shows "how much English is given up per unit of Greek gained" without depending
on where along the (shared) LR schedule the 30-minute run happened to stop.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import torch

from ..evaluation.intrinsic import load_eval_chunks, score_chunks, summarize_records
from ..modeling.surgery import build_model, load_base_model, merge_split_embedding
from ..tokenization.extenders import TokenizerBundle
from ..utils import LOG, pick_device, resolve_precision, write_json, write_jsonl


def interpolate_run(cfg, alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0), sets: List[str] | None = None) -> Dict[str, Any]:
    device = pick_device()
    amp, _ = resolve_precision(cfg.train.precision, device)
    final_dir = os.path.join(cfg.run_dir, "final")
    if not os.path.exists(final_dir):
        raise FileNotFoundError(f"{final_dir} missing; train the run first")
    # the tokenizer the run actually trained with (saved next to the model), not one rebuilt from config
    bundle = TokenizerBundle.from_model_dir(final_dir, cfg.model.name, cfg.model.revision)
    start, _ = build_model(cfg, bundle, "cpu")
    merge_split_embedding(start)
    end = load_base_model(final_dir, attn_implementation=cfg.model.attn_implementation)
    # clone: on CPU .to(device) is a no-op, so state_dict tensors would alias the live params
    s0 = {k: v.detach().clone() for k, v in start.state_dict().items()}
    s1 = {k: v.detach().clone() for k, v in end.state_dict().items()}
    if set(s0) != set(s1):
        raise ValueError("start/final parameter names differ")
    sets = sets or cfg.eval.sets
    chunks = {s: load_eval_chunks(cfg, s) for s in sets}
    model = end.to(device)
    out: Dict[str, Any] = {"run": cfg.run_name, "points": []}
    n_base = bundle.n_base if bundle.n_new else None
    for a in alphas:
        with torch.no_grad():
            model.load_state_dict({k: ((1 - a) * s0[k].float() + a * s1[k].float()).to(s1[k].dtype) for k in s1})
            model.tie_weights()
        recs = []
        for s, ch in chunks.items():
            recs.extend(score_chunks(model, bundle.tokenizer, ch, device, amp, cfg.train.seq_len, cfg.eval.batch_tokens, n_base))
        write_jsonl(os.path.join(cfg.run_dir, "eval", f"records_interp{a:.2f}.jsonl.gz"), recs)
        summ = summarize_records(recs)
        point = {"alpha": a}
        for lang in ("el", "en"):
            v = [d["bpb"] for d in summ.values() if d["language"] == lang]
            point[f"{lang}_bpb"] = sum(v) / len(v) if v else None
        out["points"].append(point)
        LOG.info("[interp %s] alpha=%.2f  EL %.4f  EN %.4f", cfg.run_name, a, point["el_bpb"] or float("nan"), point["en_bpb"] or float("nan"))
    write_json(os.path.join(cfg.run_dir, "eval", "interpolation.json"), out)
    return out


def plot_frontiers(workdir: str, runs: Sequence[str], out_path: str) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        p = os.path.join(workdir, "runs", r, "eval", "interpolation.json")
        if not os.path.exists(p):
            continue
        pts = json.load(open(p))["points"]
        ax.plot([q["el_bpb"] for q in pts], [q["en_bpb"] for q in pts], marker="o", label=r)
        for q in pts:
            ax.annotate(f"{q['alpha']:.2g}", (q["el_bpb"], q["en_bpb"]), fontsize=7)
    ax.set_xlabel("Greek BPB (macro)")
    ax.set_ylabel("English BPB (macro)")
    ax.set_title("Weight-interpolation frontiers (labels = alpha)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
