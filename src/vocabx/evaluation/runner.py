"""Full evaluation of one model state: intrinsic BPB, diagnostics, downstream, generation."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from ..utils import LOG, write_json, write_jsonl
from .downstream import run_task
from .generation import prompts_from_chunks, run_generation
from .intrinsic import load_eval_chunks, score_chunks, summarize_records


def run_full_eval(cfg, model, bundle, device: str, amp_dtype, out_dir: str, tag: str) -> Dict[str, Any]:
    t0 = time.time()
    tok = bundle.tokenizer
    n_base = bundle.n_base if bundle.n_new else None
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, Any] = {"tag": tag, "run": cfg.run_name, "vocab_size": bundle.vocab_size}

    # 1. intrinsic
    records: List[Dict[str, Any]] = []
    chunks_by_set: Dict[str, List[Dict[str, Any]]] = {}
    for s in cfg.eval.sets:
        chunks = load_eval_chunks(cfg, s)
        chunks_by_set[s] = chunks
        recs = score_chunks(model, tok, chunks, device, amp_dtype, cfg.train.seq_len, cfg.eval.batch_tokens, n_base)
        records.extend(recs)
    write_jsonl(os.path.join(out_dir, f"records_{tag}.jsonl.gz"), records)
    result["intrinsic"] = summarize_records(records)
    for s, d in result["intrinsic"].items():
        LOG.info("[%s] %-22s BPB %.4f  tok/word %.2f  mass_new %.4f", tag, s, d["bpb"], d["tokens_per_word"], d["mean_mass_new"])

    # 2. downstream cloze tasks
    if cfg.eval.downstream:
        result["downstream"] = {}
        drecs: List[Dict[str, Any]] = []
        for name in cfg.eval.downstream:
            try:
                out = run_task(name, model, tok, device, amp_dtype, cfg.eval.downstream_n, cfg.eval.item_seed, cfg.train.seq_len, cfg.eval.batch_tokens)
            except Exception as e:  # a flaky dataset download must not kill a 30-min run
                LOG.warning("downstream task %s failed: %s", name, e)
                continue
            result["downstream"][name] = out["summary"]
            drecs.extend(out["records"])
            LOG.info("[%s] %-14s acc_norm %.3f (chance %.2f)  gold_bpb %.3f", tag, name, out["summary"]["acc_norm"],
                     out["summary"]["chance"], out["summary"]["gold_bpb"])
        write_jsonl(os.path.join(out_dir, f"downstream_{tag}.jsonl.gz"), drecs)

    # 3. generation probes
    if cfg.eval.generation and cfg.eval.generation_prompts > 0:
        prompts: Dict[str, List[str]] = {}
        for s, chunks in chunks_by_set.items():
            lang = chunks[0]["language"] if chunks else None
            if lang and lang not in prompts:
                prompts[lang] = prompts_from_chunks(chunks, cfg.eval.generation_prompts)
        gen = run_generation(model, tok, prompts, device, cfg.eval.generation_bytes, amp_dtype)
        result["generation"] = gen["summary"]
        write_jsonl(os.path.join(out_dir, f"generations_{tag}.jsonl"), gen["samples"])

    result["eval_seconds"] = time.time() - t0
    write_json(os.path.join(out_dir, f"eval_{tag}.json"), result)
    return result
