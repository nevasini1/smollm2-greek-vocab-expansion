"""Command line entry point: ``python -m vocabx <command> --config <yaml> [--set key=value ...]``.

Commands
  prepare     build the cached data artifact (training pools + fixed eval sets)
  tokenizer   build the tokenizer bundle and run the tokenizer QA harness
  calibrate   measure throughput on this GPU and suggest a matched max_steps
  train       train (or, with max_steps=0, just evaluate) one run; resumable
  eval        re-evaluate a finished run's saved model (e.g. after adding eval sets)
  plan        run a list of configs in order (skips finished runs), then report
  report      results table + plots for a set of finished runs
  registries  list every registered plugin (sources, extenders, inits, tasks)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import yaml

from .config import load_config, yaml_load
from .registry import import_plugins
from .utils import LOG, setup_logging


def load_with_budget(path: str, sets: List[str], entry_sets: Optional[List[str]] = None):
    """Load a config; trained runs inherit max_steps from <workdir>/budget.json unless set explicitly.

    budget.json is written once by `vocabx calibrate`, so every run in every later session uses
    the same step budget (eval-only configs with max_steps=0 and greek_bytes runs are untouched).
    """
    from .utils import read_json

    entry_sets = list(entry_sets or [])
    cfg = load_config(path, list(sets) + entry_sets)
    explicit = any(s.split("=", 1)[0].strip() == "train.max_steps" for s in list(sets) + entry_sets)
    bpath = os.path.join(cfg.workdir, "budget.json")
    if not explicit and os.path.exists(bpath) and cfg.train.max_steps > 0 and cfg.train.budget_mode == "tokens":
        n = int(read_json(bpath)["max_steps"])
        if n != cfg.train.max_steps:
            LOG.info("%s: max_steps %d -> %d from %s", cfg.run_name, cfg.train.max_steps, n, bpath)
        cfg = load_config(path, list(sets) + [f"train.max_steps={n}"] + entry_sets)
        sets = list(sets) + [f"train.max_steps={n}"]
    lr_explicit = any(s.split("=", 1)[0].strip() == "train.optim.lr" for s in list(sets) + entry_sets)
    lpath = os.path.join(cfg.workdir, "lr_choice.json")
    if not lr_explicit and os.path.exists(lpath) and cfg.train.max_steps > 0:
        lr = float(read_json(lpath)["lr"])
        if lr != cfg.train.optim.lr:
            LOG.info("%s: peak LR %.2e -> %.2e from %s (LR-probe decision)", cfg.run_name, cfg.train.optim.lr, lr, lpath)
        cfg = load_config(path, list(sets) + [f"train.optim.lr={lr}"] + entry_sets)
    import_plugins(cfg.plugins)  # every command sees user plugins (sources, extenders, inits, tasks)
    return cfg


def _cfg(args):
    return load_with_budget(args.config, args.set)


def cmd_prepare(args) -> None:
    from .data.prepare import prepare_data

    prepare_data(_cfg(args), force=args.force)


def cmd_tokenizer(args) -> None:
    from .data.prepare import load_manifest, prepare_data
    from .evaluation.intrinsic import load_eval_chunks
    from .tokenization.base import load_base_tokenizer
    from .tokenization.extenders import build_tokenizer
    from .tokenization.qa import run_tokenizer_qa
    from .utils import write_json

    cfg = _cfg(args)
    bundle = build_tokenizer(cfg, force=args.force)
    prepare_data(cfg)
    man = load_manifest(cfg)
    greek, english = [], []
    for name, meta in man["eval_sets"].items():
        if meta["role"] != "test":
            continue
        texts = [c["text"] for c in load_eval_chunks(cfg, name)]
        (greek if meta["language"] == "el" else english).extend(texts)
    base = load_base_tokenizer(cfg.model.name, cfg.model.revision)
    report = run_tokenizer_qa(base, bundle.tokenizer, bundle.n_base, bundle.new_tokens, greek, english,
                              strict=not args.no_strict, k_curve=[1000, 2000, 4000, 8000, 12000, 16000])
    write_json(os.path.join(bundle.path, "qa.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("examples_new_tokens",)}, indent=2, ensure_ascii=False))


def cmd_calibrate(args) -> None:
    from .training.calibrate import calibrate, persist_budget

    setup_logging()
    cfg = _cfg(args)
    budget_path = os.path.join(cfg.workdir, "budget.json")
    if os.path.exists(budget_path) and not args.force:
        LOG.info("%s exists; the matched budget is already fixed (use --force to re-calibrate)", budget_path)
        return
    res = calibrate(cfg, steps=args.steps, target_minutes=args.minutes)
    persist_budget(cfg.workdir, res, force=args.force)


def cmd_train(args) -> None:
    from .training.trainer import run_training

    run_training(_cfg(args), resume=not args.no_resume, final_eval=not args.no_eval)


def cmd_eval(args) -> None:
    from .evaluation.runner import run_full_eval
    from .modeling.surgery import load_base_model
    from .tokenization.extenders import build_tokenizer
    from .utils import pick_device, resolve_precision

    from .tokenization.extenders import TokenizerBundle

    cfg = _cfg(args)
    setup_logging(logfile=os.path.join(cfg.run_dir, "eval.log"))
    final = os.path.join(cfg.run_dir, "final")
    if not os.path.exists(final):
        raise SystemExit(f"{final} not found; train the run first (or use `train` with max_steps=0 for baselines)")
    device = pick_device()
    amp, _ = resolve_precision(cfg.train.precision, device)
    # the tokenizer saved WITH the model, never one rebuilt from the (possibly edited) config
    bundle = TokenizerBundle.from_model_dir(final, cfg.model.name, cfg.model.revision)
    model = load_base_model(final, attn_implementation=cfg.model.attn_implementation).to(device)
    if model.config.vocab_size != bundle.vocab_size:
        raise SystemExit(f"model vocab {model.config.vocab_size} != saved tokenizer vocab {bundle.vocab_size}")
    run_full_eval(cfg, model, bundle, device, amp, os.path.join(cfg.run_dir, "eval"), args.tag)


def cmd_plan(args) -> None:
    from .analysis.report import build_report
    from .training.trainer import run_training

    with open(args.plan) as f:
        plan = yaml_load(f)
    base_dir = os.path.dirname(os.path.abspath(args.plan))
    global_set: List[str] = list(plan.get("global_set", [])) + list(args.set or [])
    names = []
    workdirs = set()
    for entry in plan["runs"]:
        cpath = entry["config"]
        if not os.path.isabs(cpath):
            cpath = os.path.join(base_dir, cpath)  # plan entries are relative to the plan file
        cfg = load_with_budget(cpath, global_set, list(entry.get("set", [])))
        names.append(cfg.run_name)
        workdirs.add(cfg.workdir)
        if args.only and cfg.run_name not in args.only:
            continue
        LOG.info("=== plan: %s (%s)", cfg.run_name, cpath)
        run_training(cfg, resume=True, final_eval=True)
    rep = plan.get("report")
    if rep:
        if len(workdirs) != 1:
            raise SystemExit(f"plan runs use different workdirs {workdirs}; cannot build one report")
        wd = workdirs.pop()
        build_report(wd, rep.get("runs", names), rep["baseline"], rep.get("reference"),
                     out_dir=os.path.join(wd, "results", rep.get("name", "main")), title=rep.get("title", "Results"),
                     pairs=rep.get("pairs"))


def cmd_report(args) -> None:
    from .analysis.report import build_report

    setup_logging()
    res = build_report(args.workdir, args.runs, args.baseline, args.reference, args.out, n_boot=args.n_boot)
    print(res["markdown"])


def cmd_interpolate(args) -> None:
    from .analysis.interpolate import interpolate_run, plot_frontiers

    cfg = _cfg(args)
    setup_logging(logfile=os.path.join(cfg.run_dir, "interpolate.log"))
    interpolate_run(cfg, alphas=args.alphas, sets=args.sets)
    if args.plot_runs:
        plot_frontiers(cfg.workdir, args.plot_runs, os.path.join(cfg.workdir, "results", "interpolation_frontiers.png"))


def cmd_curves(args) -> None:
    """Dev-set deltas per run + the pre-registered LR-probe decision rule.

    Every run is measured from the ORIGINAL model's dev BPB (the step-0 row of a run without
    new rows), not from its own step 0: an expanded model's step 0 is the post-surgery state,
    and "recovering from surgery" must not count as Greek gain when choosing the shared LR.
    """
    import re

    from .utils import read_jsonl, write_json

    evs = {}
    for run in args.runs:
        path = os.path.join(args.workdir, "runs", run, "dev_evals.jsonl")
        if os.path.exists(path):
            evs[run] = list(read_jsonl(path))
        else:
            print(f"{run}: no dev_evals.jsonl")
    anchor = next((ev[0] for ev in evs.values() if "new_row_norm" not in ev[0]), None)
    if anchor is None:
        print("WARNING: no run without new rows -> falling back to each run's own step 0 (biased for expanded runs)")

    def lang_mean(row, lang):
        v = [d["bpb"] for d in row["sets"].values() if d["language"] == lang]
        return sum(v) / len(v) if v else float("nan")

    rows = []
    for run, ev in evs.items():
        first, last = (anchor or ev[0]), ev[-1]
        d_el = lang_mean(last, "el") - lang_mean(first, "el")
        d_en = lang_mean(last, "en") - lang_mean(first, "en")
        m = re.search(r"lr([0-9.e+-]+)$", run)
        rows.append((run, m.group(1) if m else "?", d_el, d_en))
        ratio = d_en / -d_el if d_el < 0 else float("inf")
        print(f"{run:28s} steps {last['step']:4d}  dEL {d_el:+.4f}  dEN {d_en:+.4f}  EN-loss/EL-gain {ratio:.3f}")
    by_lr: Dict[str, list] = {}
    for _, lr, d_el, d_en in rows:
        by_lr.setdefault(lr, []).append((d_el, d_en))
    ok = []
    for lr, v in by_lr.items():
        if lr == "?":
            continue
        gel = -sum(x for x, _ in v) / len(v)
        ren = sum(y for _, y in v) / len(v)
        if gel > 0 and ren <= gel / 3:
            ok.append(float(lr))
    probed = sorted(float(lr) for lr in by_lr if lr != "?")
    choice = max(ok) if ok else (probed[0] if probed else None)
    rule = "largest LR whose English-dev loss <= 1/3 of its Greek-dev gain (both vs the original model, averaged over arms)"
    print(f"decision rule ({rule}): lr = {choice}" + ("" if ok else "  [no LR passed; smallest probed LR]"))
    if choice is not None:
        write_json(os.path.join(args.workdir, "lr_choice.json"), {"lr": choice, "rule": rule, "passed": sorted(ok),
                                                                   "deltas": rows})


def cmd_registries(args) -> None:
    from . import load_all_plugins

    load_all_plugins()
    from .data.sources import DATA_SOURCES
    from .evaluation.downstream import TASKS
    from .modeling.init import INITIALIZERS
    from .tokenization.extenders import EXTENDERS

    for reg in (DATA_SOURCES, EXTENDERS, INITIALIZERS, TASKS):
        print(f"{reg.kind}: {', '.join(reg.names())}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="vocabx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, config=True):
        sp = sub.add_parser(name)
        if config:
            sp.add_argument("--config", required=True)
            sp.add_argument("--set", action="append", default=[], help="override, e.g. train.max_steps=300")
        sp.set_defaults(fn=fn)
        return sp

    add("prepare", cmd_prepare).add_argument("--force", action="store_true")
    sp = add("tokenizer", cmd_tokenizer)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-strict", action="store_true", help="report QA failures instead of raising")
    sp = add("calibrate", cmd_calibrate)
    sp.add_argument("--steps", type=int, default=20)
    sp.add_argument("--minutes", type=float, default=27.0)
    sp.add_argument("--force", action="store_true", help="re-calibrate and overwrite budget.json")
    sp = add("train", cmd_train)
    sp.add_argument("--no-resume", action="store_true")
    sp.add_argument("--no-eval", action="store_true")
    add("eval", cmd_eval).add_argument("--tag", default="final")
    sp = sub.add_parser("plan")
    sp.add_argument("plan")
    sp.add_argument("--set", action="append", default=[])
    sp.add_argument("--only", nargs="*", help="run only these run_names (still reports all)")
    sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser("report")
    sp.add_argument("--workdir", default="artifacts")
    sp.add_argument("--runs", nargs="+", required=True)
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--reference", default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--n-boot", type=int, default=10_000)
    sp.set_defaults(fn=cmd_report)
    sp = add("interpolate", cmd_interpolate)
    sp.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    sp.add_argument("--plot-runs", nargs="*", default=None, help="runs to include in the frontier plot")
    sp.add_argument("--sets", nargs="*", default=None, help="eval sets to score (default: eval.sets; dev sets are faster)")
    sp = sub.add_parser("curves", help="dev-set deltas per run + LR-probe decision rule")
    sp.add_argument("--workdir", default="artifacts")
    sp.add_argument("--runs", nargs="+", required=True)
    sp.set_defaults(fn=cmd_curves)
    add("registries", cmd_registries, config=False)

    args = p.parse_args(argv)
    setup_logging()
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
