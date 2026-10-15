"""Training loop shared by every arm (Exp1, Exp2, ablations).

Fairness by construction:
* same seed -> same document order; same micro-batch, sequence length, accumulation and
  step count -> identical token and step budgets (asserted in the report);
* the same parameter groups and schedule for every shared parameter; Exp2 only adds
  the ``emb_new`` group (its own LR multiplier, no weight decay);
* FLOPs, bytes seen per language, GradScaler-skipped steps and wall-clock are metered
  and written to ``summary.json`` so remaining asymmetries are reported, not hidden.

Robustness on Colab: checkpoints (model, optimizer, scaler, RNG, stream cursors) every
few minutes with exact resume; final model saved in standard HF format.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import Config, StageSpec, dump_config
from ..data.packing import Mixture, TokenStream, ensure_packed
from ..data.prepare import prepare_data
from ..evaluation.intrinsic import load_eval_chunks, score_chunks, summarize_records
from ..evaluation.runner import run_full_eval
from ..evaluation.scoring import backbone
from ..modeling.surgery import SplitTiedEmbedding, assert_tied, build_model, save_model
from ..registry import import_plugins
from ..tokenization.extenders import build_tokenizer
from ..tokenization.qa import new_token_exposure
from ..utils import (
    LOG,
    append_jsonl,
    environment_info,
    pick_device,
    read_json,
    read_jsonl,
    write_jsonl,
    resolve_precision,
    seed_everything,
    setup_logging,
    write_json,
)
from .budget import BudgetMeter, flops_per_token, token_byte_lengths
from .schedule import lr_multiplier

GROUPS = ("body", "norms", "emb_old", "emb_new")
LN2 = math.log(2.0)


def _ce(h: torch.Tensor, W: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(h, W).float(), t, reduction="sum")


def lm_loss_sum(model, x: torch.Tensor, y: torch.Tensor, chunk_tokens: int = 0) -> torch.Tensor:
    """Summed next-token cross-entropy (nats), fp32 softmax, optionally chunked + recomputed."""
    hidden = backbone(model)(input_ids=x).last_hidden_state
    W = model.get_output_embeddings().weight
    h = hidden.reshape(-1, hidden.shape[-1])
    t = y.reshape(-1)
    if chunk_tokens <= 0 or h.shape[0] <= chunk_tokens:
        return _ce(h, W, t)
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk_tokens):
        total = total + checkpoint(_ce, h[s : s + chunk_tokens], W, t[s : s + chunk_tokens], use_reentrant=False)
    return total


class Trainer:
    def __init__(self, cfg: Config, device: Optional[str] = None, for_calibration: bool = False):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        import_plugins(cfg.plugins)
        self.device = device or pick_device()
        self.amp_dtype, use_scaler = resolve_precision(cfg.train.precision, self.device)
        seed_everything(cfg.seed)

        self.bundle = build_tokenizer(cfg)
        prepare_data(cfg)
        self.model, self.surgery = build_model(cfg, self.bundle, self.device)
        self.split: SplitTiedEmbedding = self.model.get_input_embeddings()
        self.n_base = self.bundle.n_base
        self.n_new = self.bundle.n_new
        tc = cfg.train
        self.tokens_per_step = tc.micro_batch_size * tc.grad_accum * tc.seq_len

        # ---- data streams
        langs = ["el"] + (["en"] if tc.english_ratio > 0 else [])
        self.packed = ensure_packed(cfg, self.bundle, langs)
        self.streams = {l: TokenStream(self.packed[l], tc.seq_len, seed=cfg.seed) for l in langs}
        self.mixture = Mixture(self.streams, tc.english_ratio)
        self.tok_bytes = token_byte_lengths(self.bundle.tokenizer, self.bundle.vocab_size)
        self.max_steps = self._resolve_max_steps()

        # ---- parameter groups (identical across arms; emb_new is empty for Exp1)
        groups: Dict[str, List[torch.nn.Parameter]] = {g: [] for g in GROUPS}
        for name, p in self.model.named_parameters():
            if p is self.split.old:
                groups["emb_old"].append(p)
            elif self.split.new is not None and p is self.split.new:
                groups["emb_new"].append(p)
            elif p.ndim >= 2:
                groups["body"].append(p)
            else:
                groups["norms"].append(p)
        self.groups = groups
        oc = tc.optim
        opt_groups = [
            {"params": ps, "name": g, "weight_decay": oc.weight_decay if g == "body" else 0.0, "lr": 0.0}
            for g, ps in groups.items()
            if ps
        ]
        kwargs: Dict[str, Any] = {"betas": tuple(oc.betas), "eps": oc.eps}
        if self.device == "cuda":
            kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(opt_groups, **kwargs)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=tc.grad_scaler_init, enabled=use_scaler)

        # ---- stages
        self.stages: List[StageSpec] = tc.stages or [StageSpec(name="all", until_frac=1.0)]
        for st in self.stages:
            bad = set(st.trainable) - set(GROUPS)
            if bad:
                raise ValueError(f"stage {st.name}: unknown groups {bad}; valid: {GROUPS}")
        self.group_start = {g: self._first_trainable_step(g) for g in GROUPS}

        # ---- accounting
        self.meter = BudgetMeter()
        self.new_counts = np.zeros(self.n_new, dtype=np.int64)
        self.fpt_full = flops_per_token(self.model.config, self.bundle.vocab_size, tc.seq_len, True)
        self.fpt_frozen = flops_per_token(self.model.config, self.bundle.vocab_size, tc.seq_len, False)
        self.dev_sets = {s: load_eval_chunks(cfg, s, cfg.eval.dev_max_chunks) for s in cfg.eval.dev_sets}
        self.theta0 = None if for_calibration else self._snapshot()
        if tc.compile:
            try:
                self.model.model = torch.compile(self.model.model)
            except Exception as e:  # pragma: no cover
                LOG.warning("torch.compile failed (%s); continuing without it", e)

    # ------------------------------------------------------------------ helpers
    def budget_signature(self) -> Dict[str, Any]:
        """Everything that defines this run's budget and optimisation; must not change on resume."""
        return budget_signature(self.cfg, self.max_steps)

    def _resolve_max_steps(self) -> int:
        tc = self.cfg.train
        if tc.budget_mode == "tokens":
            return tc.max_steps
        if tc.budget_mode == "greek_bytes":
            if not tc.budget_reference:
                raise ValueError("budget_mode=greek_bytes needs train.budget_reference (a finished run name)")
            ref = read_json(os.path.join(self.cfg.workdir, "runs", tc.budget_reference, "summary.json"))
            ref_bytes = ref["budget"]["bytes_by_lang"]["el"]
            per_step = self.tokens_per_step * (1 - tc.english_ratio) * self.packed["el"].bytes_per_token
            steps = max(1, round(ref_bytes / per_step))
            LOG.info("greek_bytes budget: %d reference bytes -> %d steps", ref_bytes, steps)
            return steps
        raise ValueError(f"unknown budget_mode {tc.budget_mode!r}")

    def _stage_at(self, step: int) -> StageSpec:
        for st in self.stages:
            if step < round(st.until_frac * self.max_steps):
                return st
        return self.stages[-1]

    def _first_trainable_step(self, group: str) -> int:
        for step in range(self.max_steps):
            if group in self._stage_at(step).trainable:
                return step
        return self.max_steps

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        return {g: torch.cat([p.detach().float().flatten().cpu() for p in ps]) for g, ps in self.groups.items() if ps}

    def _drift(self) -> Dict[str, float]:
        if self.theta0 is None:
            return {}
        out = {}
        for g, ps in self.groups.items():
            if not ps:
                continue
            cur = torch.cat([p.detach().float().flatten().cpu() for p in ps])
            out[g] = float((cur - self.theta0[g]).norm() / (self.theta0[g].norm() + 1e-12))
        return out

    def _set_trainable(self, stage: StageSpec) -> None:
        for g, ps in self.groups.items():
            for p in ps:
                p.requires_grad_(g in stage.trainable)

    def _set_lrs(self, step: int) -> None:
        tc = self.cfg.train
        for pg in self.optimizer.param_groups:
            g = pg["name"]
            mult = lr_multiplier(step, self.max_steps, tc.schedule, start_step=self.group_start[g])
            pg["lr"] = tc.optim.lr * tc.group_lr_mult.get(g, 1.0) * mult

    # ------------------------------------------------------------------ steps
    def train_step(self, step: int) -> Dict[str, Any]:
        tc = self.cfg.train
        stage = self._stage_at(step)
        self._set_trainable(stage)
        self._set_lrs(step)
        self.model.train()
        t0 = time.time()
        loss_total = 0.0
        tgt_bytes = 0
        for _ in range(tc.grad_accum):
            rows = [self.mixture.next() for _ in range(tc.micro_batch_size)]
            blocks = np.stack([b for _, b in rows]).astype(np.int64)
            x = torch.from_numpy(blocks[:, :-1]).to(self.device, non_blocking=True)
            y = torch.from_numpy(blocks[:, 1:]).to(self.device, non_blocking=True)
            if self.amp_dtype is not None:
                with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                    loss_sum = lm_loss_sum(self.model, x, y, tc.loss_chunk_tokens)
            else:
                loss_sum = lm_loss_sum(self.model, x, y, tc.loss_chunk_tokens)
            self.scaler.scale(loss_sum / self.tokens_per_step).backward()
            loss_total += float(loss_sum.detach())
            for lang, b in rows:
                inp = b[:-1]
                self.meter.tokens_by_lang[lang] += len(inp)
                self.meter.bytes_by_lang[lang] += int(self.tok_bytes[inp].sum())
                tgt_bytes += int(self.tok_bytes[b[1:]].sum())
                if self.n_new and lang == "el":
                    new = inp[inp >= self.n_base].astype(np.int64) - self.n_base
                    if new.size:
                        self.new_counts += np.bincount(new, minlength=self.n_new)
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.model.parameters() if p.grad is not None]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tc.optim.grad_clip)) if params else 0.0
        scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.is_enabled() and self.scaler.get_scale() < scale_before
        self.optimizer.zero_grad(set_to_none=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        body_trainable = "body" in stage.trainable
        fpt = self.fpt_full["train"] if body_trainable else self.fpt_frozen["train"]
        self.meter.steps = step + 1
        self.meter.tokens += self.tokens_per_step
        self.meter.flops += fpt * self.tokens_per_step
        self.meter.skipped_steps += int(skipped)
        self.meter.train_seconds += dt
        return {
            "step": step + 1,
            "stage": stage.name,
            "loss_per_token": loss_total / self.tokens_per_step,
            "train_bpb": loss_total / (LN2 * max(tgt_bytes, 1)),
            "lr_body": next((pg["lr"] for pg in self.optimizer.param_groups if pg["name"] == "body"), 0.0),
            "lr_new_rows": next((pg["lr"] for pg in self.optimizer.param_groups if pg["name"] == "emb_new"), None),
            "grad_norm": gnorm,
            "scale": self.scaler.get_scale() if self.scaler.is_enabled() else None,
            "skipped": bool(skipped),
            "tokens_per_s": self.tokens_per_step / dt,
            "step_seconds": dt,
        }

    def dev_eval(self, step: int) -> Dict[str, Any]:
        t0 = time.time()
        res = {}
        for name, chunks in self.dev_sets.items():
            recs = score_chunks(self.model, self.bundle.tokenizer, chunks, self.device, self.amp_dtype, self.cfg.train.seq_len,
                                self.cfg.eval.batch_tokens, self.n_base if self.n_new else None)
            res[name] = summarize_records(recs)[name]
        row = {
            "step": step,
            "tokens": self.meter.tokens,
            "bytes_el": self.meter.bytes_by_lang["el"],
            "bytes_en": self.meter.bytes_by_lang["en"],
            "flops": self.meter.flops,
            "train_seconds": self.meter.train_seconds,
            "sets": res,
            "drift": self._drift(),
        }
        if self.split.new is not None:
            row["new_row_norm"] = float(self.split.new.detach().norm(dim=1).mean())
            row["old_row_norm"] = float(self.split.old.detach().norm(dim=1).mean())
        append_jsonl(os.path.join(self.run_dir, "dev_evals.jsonl"), row)
        msg = "  ".join(f"{k}={v['bpb']:.4f}" for k, v in res.items())
        LOG.info("dev eval @%d (%.0fs): %s", step, time.time() - t0, msg)
        return row

    # ------------------------------------------------------------------ checkpointing
    def _ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "checkpoint.pt")

    def save_checkpoint(self) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "meter": self.meter.to_dict(),
            "mixture": self.mixture.state(),
            "new_counts": self.new_counts,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
            "max_steps": self.max_steps,
            "budget_signature": self.budget_signature(),
        }
        tmp = self._ckpt_path() + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self._ckpt_path())
        LOG.info("checkpoint saved at step %d", self.meter.steps)

    def load_checkpoint(self) -> bool:
        path = self._ckpt_path()
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        _assert_same_budget(state.get("budget_signature") or {"max_steps": state["max_steps"]}, self.budget_signature(),
                            f"checkpoint {path}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.meter = BudgetMeter.from_dict(state["meter"])
        self.mixture.load_state(state["mixture"])
        self.new_counts = state["new_counts"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        np.random.set_state(state["rng"]["numpy"])
        LOG.info("resumed from step %d", self.meter.steps)
        return True

    # ------------------------------------------------------------------ main
    def run(self, resume: bool = True, final_eval: bool = True, keep_checkpoint: bool = False) -> Dict[str, Any]:
        cfg, tc = self.cfg, self.cfg.train
        summary_path = os.path.join(self.run_dir, "summary.json")
        if resume:
            done = _check_finished(cfg)
            if done is not None:
                return done
        dump_config(cfg, os.path.join(self.run_dir, "config.yaml"))
        resumed = resume and self.load_checkpoint()
        LOG.info("run %s: %d steps x %d tokens = %.1fM tokens; device=%s amp=%s; vocab %d (+%d)",
                 cfg.run_name, self.max_steps, self.tokens_per_step, self.max_steps * self.tokens_per_step / 1e6,
                 self.device, self.amp_dtype, self.bundle.vocab_size, self.n_new)
        self._check_data_budget()
        evals_dir = os.path.join(self.run_dir, "eval")
        if not resumed:
            for f in ("dev_evals.jsonl", "train_log.jsonl"):
                if os.path.exists(os.path.join(self.run_dir, f)):
                    os.remove(os.path.join(self.run_dir, f))
            self.dev_eval(0)  # step 0: the original model (Exp1) or the post-surgery model (Exp2)
            if self.n_new and cfg.eval.full_eval_at_step0 and cfg.eval.sets:
                run_full_eval(cfg, self.model, self.bundle, self.device, self.amp_dtype, evals_dir, "step0")
        else:  # drop log rows written after the checkpoint we resumed from (they will be redone)
            for f in ("dev_evals.jsonl", "train_log.jsonl"):
                path = os.path.join(self.run_dir, f)
                if os.path.exists(path):
                    keep = [r for r in read_jsonl(path) if r["step"] <= self.meter.steps]
                    write_jsonl(path, keep)
        last_ckpt = time.time()
        wall_warned = False
        for step in range(self.meter.steps, self.max_steps):
            m = self.train_step(step)
            if not math.isfinite(m["loss_per_token"]):
                raise FloatingPointError(f"non-finite loss at step {m['step']}")
            if m["step"] % tc.log_every == 0 or m["step"] in (1, self.max_steps):
                append_jsonl(os.path.join(self.run_dir, "train_log.jsonl"), m)
                LOG.info("step %4d/%d [%s] loss/tok %.4f train_bpb %.4f lr %.2e%s gn %.2f %s%.0f tok/s",
                         m["step"], self.max_steps, m["stage"], m["loss_per_token"], m["train_bpb"], m["lr_body"],
                         f" (new rows {m['lr_new_rows']:.2e})" if m["lr_new_rows"] is not None else "",
                         m["grad_norm"], f"scale {m['scale']:.0f} " if m["scale"] else "", m["tokens_per_s"])
            if m["step"] % tc.eval_every == 0 or m["step"] == self.max_steps:
                self.dev_eval(m["step"])
            if not wall_warned and self.meter.train_seconds > tc.wallclock_limit_min * 60:
                LOG.warning("training time exceeded %.0f min; finishing the matched step budget anyway", tc.wallclock_limit_min)
                wall_warned = True
            if time.time() - last_ckpt > tc.checkpoint_every_min * 60 and m["step"] < self.max_steps:
                self.save_checkpoint()
                last_ckpt = time.time()

        final_dir = os.path.join(self.run_dir, "final")
        if self.max_steps > 0:
            save_model(self.model, self.bundle, final_dir)
        summary = self._summary()
        if final_eval and cfg.eval.sets:
            ev = run_full_eval(cfg, self.model, self.bundle, self.device, self.amp_dtype, evals_dir, "final")
            summary["final_eval"] = {"intrinsic": ev.get("intrinsic"), "downstream": ev.get("downstream"),
                                     "generation": ev.get("generation")}
        write_json(summary_path, summary)
        if not keep_checkpoint and os.path.exists(self._ckpt_path()):
            os.remove(self._ckpt_path())
        LOG.info("run %s done: %.1f min of training", cfg.run_name, self.meter.train_seconds / 60)
        return summary

    def _check_data_budget(self) -> None:
        tc = self.cfg.train
        total_rows = self.max_steps * tc.grad_accum * tc.micro_batch_size
        for lang, s in self.streams.items():
            share = tc.english_ratio if lang == "en" else 1 - tc.english_ratio
            need = total_rows * share
            epochs = need / max(s.blocks_per_epoch, 1)
            level = LOG.warning if epochs > 1 else LOG.info
            level("%s stream: %.0f blocks needed, %d per epoch -> %.2f epochs", lang, need, s.blocks_per_epoch, epochs)
            if lang == "el" and epochs > 1 and not tc.allow_repeat:
                raise ValueError(
                    f"the Greek stream would need {epochs:.2f} epochs ({need:.0f} blocks, {s.blocks_per_epoch}/epoch): "
                    "only this arm would repeat data. Enlarge data.greek train_mb, lower train.max_steps, "
                    "or set train.allow_repeat=true to accept (and report) repetition."
                )

    def _summary(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "run_name": cfg.run_name,
            "status": "done",
            "config": cfg.to_dict(),
            "environment": environment_info(),
            "device": self.device,
            "amp_dtype": str(self.amp_dtype),
            "tokenizer": {
                "key": cfg.key("tokenizer"),
                "extender": cfg.tokenizer.extender,
                "vocab_size": self.bundle.vocab_size,
                "n_new": self.n_new,
            },
            "surgery": self.surgery,
            "max_steps": self.max_steps,
            "tokens_per_step": self.tokens_per_step,
            "budget_signature": self.budget_signature(),
            "budget": self.meter.to_dict(),
            "flops_per_token": {"full": self.fpt_full, "frozen_body": self.fpt_frozen},
            "epochs": {l: s.epochs_consumed for l, s in self.streams.items()},
            "packed_bytes_per_token": {l: c.bytes_per_token for l, c in self.packed.items()},
            "new_token_exposure": new_token_exposure(self.new_counts),
        }


def budget_signature(cfg: Config, max_steps: int) -> Dict[str, Any]:
    tc = cfg.train
    d = cfg.to_dict()
    return {
        "max_steps": max_steps,
        "tokens_per_step": tc.micro_batch_size * tc.grad_accum * tc.seq_len,
        "seq_len": tc.seq_len,
        "english_ratio": tc.english_ratio,
        "optim": d["train"]["optim"],
        "schedule": d["train"]["schedule"],
        "group_lr_mult": dict(tc.group_lr_mult),
        "stages": d["train"]["stages"],
        "seed": cfg.seed,
        "data_key": cfg.key("data"),
        "tokenizer_key": cfg.key("tokenizer"),
        "init": d["init"],
    }


def _check_finished(cfg: Config) -> Optional[Dict[str, Any]]:
    """Return the summary of an already finished run (after checking its budget), else None.

    Runs before any heavy setup, so re-running a plan skips finished runs in seconds.
    """
    path = os.path.join(cfg.run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    done = read_json(path)
    if done.get("max_steps", 0) == 0 and cfg.train.max_steps == 0:
        return done
    # greek_bytes runs derive max_steps from their reference; compare with what they used
    ms = cfg.train.max_steps if cfg.train.budget_mode == "tokens" else done.get("max_steps")
    old = done.get("budget_signature") or {"max_steps": done.get("max_steps"), "tokens_per_step": done.get("tokens_per_step")}
    # keys that only identify cached artifacts may legitimately change between code versions
    new = {k: v for k, v in budget_signature(cfg, ms).items() if k not in ("data_key", "tokenizer_key")}
    _assert_same_budget({k: v for k, v in old.items() if k not in ("data_key", "tokenizer_key")}, new,
                        f"finished run {cfg.run_name}")
    LOG.info("run %s already finished with the same budget; skipping (delete %s to rerun)", cfg.run_name, path)
    return done


def _assert_same_budget(old: Dict[str, Any], new: Dict[str, Any], what: str) -> None:
    diff = {k: (old[k], new.get(k)) for k in old if k in new and old[k] != new[k]}
    if diff:
        raise ValueError(
            f"{what} was made with a different budget/optimisation {diff}. Matched budgets would break: "
            "reuse the original settings (e.g. the persisted max_steps), or delete the run directory to retrain."
        )


def run_training(cfg: Config, resume: bool = True, final_eval: bool = True) -> Dict[str, Any]:
    setup_logging(logfile=os.path.join(cfg.run_dir, "train.log"))
    if resume:
        done = _check_finished(cfg)
        if done is not None:
            return done
    if cfg.train.max_steps == 0 and cfg.train.budget_mode == "tokens":
        return evaluate_untrained(cfg)
    return Trainer(cfg).run(resume=resume, final_eval=final_eval)


def evaluate_untrained(cfg: Config) -> Dict[str, Any]:
    """Eval-only run (the off-the-shelf baseline, or an expanded model at step 0)."""
    import_plugins(cfg.plugins)
    os.makedirs(cfg.run_dir, exist_ok=True)
    summary_path = os.path.join(cfg.run_dir, "summary.json")
    if os.path.exists(summary_path):
        return read_json(summary_path)
    dump_config(cfg, os.path.join(cfg.run_dir, "config.yaml"))
    device = pick_device()
    amp_dtype, _ = resolve_precision(cfg.train.precision, device)
    bundle = build_tokenizer(cfg)
    prepare_data(cfg)
    model, surgery = build_model(cfg, bundle, device)
    ev = run_full_eval(cfg, model, bundle, device, amp_dtype, os.path.join(cfg.run_dir, "eval"), "final")
    summary = {
        "run_name": cfg.run_name,
        "status": "done",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "tokenizer": {"key": cfg.key("tokenizer"), "extender": cfg.tokenizer.extender,
                      "vocab_size": bundle.vocab_size, "n_new": bundle.n_new},
        "surgery": surgery,
        "max_steps": 0,
        "tokens_per_step": 0,
        "budget": BudgetMeter().to_dict(),
        "final_eval": {"intrinsic": ev.get("intrinsic"), "downstream": ev.get("downstream"), "generation": ev.get("generation")},
    }
    write_json(summary_path, summary)
    return summary
