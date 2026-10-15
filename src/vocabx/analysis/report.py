"""Results tables + plots from finished runs.

Rows: the off-the-shelf baseline, every trained run, and (for vocabulary-extended runs)
their step-0 state (surgery only, no training) - which separates the cost of the
vocabulary swap from the effect of training.

Headline quantities (all from a paired, document-clustered bootstrap):
  dEL = BPB_el(model) - BPB_el(base)          (negative = Greek gain)
  dEN = BPB_en(model) - BPB_en(base)          (positive = English lost)
  extra English cost of X vs reference R = BPB_en(X) - BPB_en(R)
  extra Greek gain   of X vs reference R = BPB_el(R) - BPB_el(X)
  exchange rate = extra English cost / extra Greek gain (bits of English per bit of Greek)
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..evaluation.stats import bootstrap_accuracy, bootstrap_bpb, ci, macro
from ..utils import LOG, read_json, read_jsonl, write_json


def _load_run(workdir: str, run: str) -> Dict[str, Any]:
    rd = os.path.join(workdir, "runs", run)
    s = read_json(os.path.join(rd, "summary.json"))
    out = {"summary": s, "records": {}, "downstream": {}, "dev": []}
    for tag in ("final", "step0"):
        p = os.path.join(rd, "eval", f"records_{tag}.jsonl.gz")
        if os.path.exists(p):
            out["records"][tag] = list(read_jsonl(p))
        p = os.path.join(rd, "eval", f"downstream_{tag}.jsonl.gz")
        if os.path.exists(p):
            out["downstream"][tag] = list(read_jsonl(p))
        p = os.path.join(rd, "eval", f"eval_{tag}.json")
        if os.path.exists(p):
            out.setdefault("evals", {})[tag] = read_json(p)
    p = os.path.join(rd, "dev_evals.jsonl")
    if os.path.exists(p):
        out["dev"] = list(read_jsonl(p))
    return out


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(d, dict):
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def _signature_diff(opt_sig: Dict[str, str]) -> str:
    flat = {r: _flatten(json.loads(v)) for r, v in opt_sig.items()}
    keys = sorted({k for f in flat.values() for k in f})
    parts = []
    for k in keys:
        vals = {r: f.get(k) for r, f in flat.items()}
        if len({json.dumps(v) for v in vals.values()}) > 1:
            parts.append(f"`{k}`: " + ", ".join(f"{r}={v}" for r, v in vals.items()))
    return "; ".join(parts)


def exchange_cell(extra_en: np.ndarray, extra_el: np.ndarray) -> str:
    """Bits of English given up per bit of Greek gained, only when the Greek gain is clearly > 0."""
    g = ci(extra_el)
    e = ci(extra_en)
    if np.isfinite(g[1]) and g[1] > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ci(extra_en / extra_el)
        return f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
    if g[0] <= 0 and e[0] <= 0:
        return "n/a: worse EL, better EN"
    if g[0] <= 0:
        return "n/a: worse EL and worse EN"
    return "n/a: EL gain CI includes 0"


def _fmt(p: float, lo: float, hi: float, digits: int = 4, signed: bool = False) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def build_report(
    workdir: str,
    runs: Sequence[str],
    baseline: str,
    reference: Optional[str] = None,
    out_dir: Optional[str] = None,
    n_boot: int = 10_000,
    title: str = "Results",
    pairs: Optional[Sequence[Sequence[str]]] = None,
    pool_pairs: bool = False,
) -> Dict[str, Any]:
    out_dir = out_dir or os.path.join(workdir, "results")
    os.makedirs(out_dir, exist_ok=True)
    missing = [r for r in runs if not os.path.exists(os.path.join(workdir, "runs", r, "summary.json"))]
    if missing:
        LOG.warning("skipping unfinished runs in report: %s", missing)
    loaded = {r: _load_run(workdir, r) for r in runs if r not in missing}
    if reference in missing:
        reference = None

    # ---- models = run finals + step0 states
    models: Dict[str, List[Dict]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for r, L in loaded.items():
        s = L["summary"]
        if "final" in L["records"]:
            models[r] = L["records"]["final"]
            meta[r] = s
        if "step0" in L["records"]:
            models[f"{r}@step0"] = L["records"]["step0"]
            meta[f"{r}@step0"] = {**s, "max_steps": 0, "budget": {"tokens": 0, "steps": 0, "bytes_by_lang": {"el": 0, "en": 0}, "flops": 0, "train_seconds": 0}}
    if baseline not in models:
        raise ValueError(f"baseline run {baseline!r} has no final records")
    sets_by_lang: Dict[str, List[str]] = {}
    for rec in models[baseline]:
        sets_by_lang.setdefault(rec["language"], [])
        if rec["set"] not in sets_by_lang[rec["language"]]:
            sets_by_lang[rec["language"]].append(rec["set"])
    all_sets = [s for v in sets_by_lang.values() for s in v]
    boot = bootstrap_bpb(models, all_sets, n_boot=n_boot)
    # language macros only over sets EVERY model was scored on, so contrasts compare like with like
    common = {lang: [s_ for s_ in sets if all(s_ in boot[m] for m in models)] for lang, sets in sets_by_lang.items()}
    dropped_sets = sorted({s_ for lang, sets in sets_by_lang.items() for s_ in sets if s_ not in common[lang]})
    if dropped_sets:
        LOG.warning("sets left out of the macro BPB (not scored for every model): %s", dropped_sets)
    mac = {m: {lang: macro(boot[m], sets) for lang, sets in common.items() if sets} for m in models}

    # ---- budget sanity: token-budget runs must match tokens and steps (greek_bytes ablations are
    # deliberately shorter and are excluded from the check)
    trained = [r for r in runs if meta.get(r, {}).get("max_steps", 0) > 0]
    token_runs = [r for r in trained
                  if (meta[r].get("config", {}).get("train", {}).get("budget_mode", "tokens")) == "tokens"]
    budget_check = {r: (meta[r]["budget"]["tokens"], meta[r]["budget"]["steps"]) for r in token_runs}
    matched = len(set(budget_check.values())) <= 1
    # optimisation must match too (peak LR, schedule, sequence length, tokens/step), incl. greek_bytes runs
    opt_sig = {}
    for r in trained:
        sig = meta[r].get("budget_signature") or {}
        if sig:
            opt_sig[r] = json.dumps({k: sig.get(k) for k in ("optim", "schedule", "seq_len", "tokens_per_step")}, sort_keys=True)
    opt_matched = len(set(opt_sig.values())) <= 1
    if not opt_matched:
        groups: Dict[str, List[str]] = {}
        for r, v in opt_sig.items():
            groups.setdefault(v, []).append(r)
        LOG.error("optimisation settings differ across trained runs: %s", list(groups.values()))
    matched = matched and opt_matched
    if not matched:
        LOG.error("budgets/optimisation differ across trained runs: %s -- extra-cost columns are NOT valid", budget_check)

    # ---- main table
    lines = [f"# {title}", ""]
    lines.append(f"Baseline: `{baseline}`; reference for extra-cost columns: `{reference}`. "
                 f"95% CIs from a paired bootstrap over documents ({n_boot} resamples). BPB = bits per UTF-8 byte (lower is better).")
    lines.append("")
    lines.append(f"Token/step budgets and optimisation (peak LR, schedule, seq len) matched across trained runs: "
                 f"**{'yes' if matched else 'NO'}** {budget_check}")
    if not opt_matched:
        lines.append("")
        lines.append("> **Optimisation differs between runs** (fields that are not identical): " + _signature_diff(opt_sig))
    if dropped_sets:
        lines.append("")
        lines.append(f"> Sets excluded from the macro BPB (not scored for every model): {dropped_sets}")
    if not matched:
        lines.append("")
        lines.append("> **WARNING: budgets differ, so the extra-cost / exchange columns compare unequal training. Do not report them.**")
    lines.append("")
    hdr = ["model", "vocab", "steps", "tokens (M)", "Greek MB seen", "Greek epochs", "train PFLOPs", "train min",
           "EL BPB", "ΔEL vs base", "EN BPB", "ΔEN vs base"]
    if reference:
        hdr += ["extra EN cost vs ref", "extra EL gain vs ref", "EN-per-EL exchange"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    table: Dict[str, Dict[str, Any]] = {}
    for m in models:
        s = meta[m]
        b = s.get("budget", {})
        el, en = mac[m].get("el"), mac[m].get("en")
        bel, ben = mac[baseline].get("el"), mac[baseline].get("en")
        row = {
            "vocab": s.get("tokenizer", {}).get("vocab_size"),
            "steps": b.get("steps", 0),
            "tokens_M": b.get("tokens", 0) / 1e6,
            "greek_MB": b.get("bytes_by_lang", {}).get("el", 0) / 1e6,
            "greek_epochs": (s.get("epochs") or {}).get("el", 0.0) if not m.endswith("@step0") else 0.0,
            "pflops": b.get("flops", 0) / 1e15,
            "train_min": b.get("train_seconds", 0) / 60,
        }
        cells = [f"`{m}`", str(row["vocab"]), str(row["steps"]), f"{row['tokens_M']:.1f}", f"{row['greek_MB']:.1f}",
                 f"{row['greek_epochs']:.2f}", f"{row['pflops']:.2f}", f"{row['train_min']:.1f}"]
        for lang, arr, barr in (("el", el, bel), ("en", en, ben)):
            if arr is None:
                cells += ["-", "-"]
                continue
            p, lo, hi = ci(arr)
            d = ci(arr - barr)
            row[f"{lang}_bpb"] = (p, lo, hi)
            row[f"d{lang}"] = d
            cells += [_fmt(p, lo, hi), _fmt(*d, signed=True)]
        if reference:
            if reference in models and m != reference and el is not None and en is not None:
                extra_en = mac[m]["en"] - mac[reference]["en"]
                extra_el = mac[reference]["el"] - mac[m]["el"]
                row["extra_en_cost"] = ci(extra_en)
                row["extra_el_gain"] = ci(extra_el)
                row["exchange"] = exchange_cell(extra_en, extra_el)
                cells += [_fmt(*row["extra_en_cost"], signed=True), _fmt(*row["extra_el_gain"], signed=True),
                          row["exchange"]]
            else:
                cells += ["-", "-", "-"]
        table[m] = row
        lines.append("| " + " | ".join(cells) + " |")

    # ---- explicit paired contrasts (CRN seed pairs, replay pair, ...)
    pair_rows = []
    for x, r in pairs or []:
        if x not in models or r not in models:
            LOG.warning("pair (%s, %s) skipped: missing records", x, r)
            continue
        ee = mac[x]["en"] - mac[r]["en"]
        eg = mac[r]["el"] - mac[x]["el"]
        pair_rows.append((x, r, ee, eg))
    if pair_rows:
        lines += ["", "## Paired contrasts", "",
                  "| model | vs | extra EN cost (BPB) | extra EL gain (BPB) | EN-per-EL exchange |", "|---|---|---|---|---|"]
        for x, r, ee, eg in pair_rows:
            lines.append(f"| `{x}` | `{r}` | {_fmt(*ci(ee), signed=True)} | {_fmt(*ci(eg), signed=True)} | {exchange_cell(ee, eg)} |")
        if pool_pairs and len(pair_rows) > 1:  # only meaningful for seed replicates of one contrast
            ee_m = np.mean([p[2] for p in pair_rows], axis=0)
            eg_m = np.mean([p[3] for p in pair_rows], axis=0)
            pts_ee = [p[2][0] for p in pair_rows]
            lines.append(f"| **mean of {len(pair_rows)} pairs** | | {_fmt(*ci(ee_m), signed=True)} (pair SD {np.std(pts_ee, ddof=1):.4f}) "
                         f"| {_fmt(*ci(eg_m), signed=True)} | {exchange_cell(ee_m, eg_m)} |")
            lines.append("")
            lines.append("Pair-mean CIs cover evaluation noise only; the pair SD across seeds estimates training noise.")
        lines.append("")
        lines.append("Each pair's CI is a paired document bootstrap (evaluation noise only, not training noise).")

    # ---- per-set table
    lines += ["", "## Per eval set (BPB, point estimates)", ""]
    lines.append("| model | " + " | ".join(all_sets) + " |")
    lines.append("|---|" + "---|" * len(all_sets))
    for m in models:
        lines.append(f"| `{m}` | " + " | ".join(f"{boot[m][s][0]:.4f}" if s in boot[m] else "-" for s in all_sets) + " |")

    # ---- Exp2 mechanism: English softmax mass on new tokens + renormalised BPB
    lines += ["", "## English probability mass on new (Greek) tokens", "",
              "`mass_new` = mean probability the model assigns to the new token ids at English positions; "
              "`BPB (old-vocab renorm)` removes that leakage, isolating representational forgetting.", ""]
    lines.append("| model | set | BPB | BPB (old-vocab renorm) | mass_new |")
    lines.append("|---|---|---|---|---|")
    for m, L in ((m, loaded.get(m.split("@")[0])) for m in models):
        tag = "step0" if m.endswith("@step0") else "final"
        ev = (L or {}).get("evals", {}).get(tag, {}).get("intrinsic", {})
        for s in sets_by_lang.get("en", []):
            d = ev.get(s)
            if d and d.get("mean_mass_new", 0) > 0:
                lines.append(f"| `{m}` | {s} | {d['bpb']:.4f} | {d.get('bpb_old_vocab', float('nan')):.4f} | {d['mean_mass_new']:.5f} |")

    # ---- tokenizer efficiency
    lines += ["", "## Tokenizer efficiency on held-out text", "", "| model | set | tokens/word | bytes/token |", "|---|---|---|---|"]
    for m, L in ((m, loaded.get(m.split("@")[0])) for m in models if not m.endswith("@step0")):
        ev = (L or {}).get("evals", {}).get("final", {}).get("intrinsic", {})
        for s in all_sets:
            if s in ev:
                lines.append(f"| `{m}` | {s} | {ev[s]['tokens_per_word']:.2f} | {ev[s]['bytes_per_token']:.2f} |")

    # ---- downstream
    ds_models = {m: L["downstream"].get("final") for m, L in loaded.items() if L["downstream"].get("final")}
    if ds_models:
        tasks = sorted({r["task"] for recs in ds_models.values() for r in recs})
        lines += ["", "## Zero-shot cloze tasks (acc_norm, 95% CI; gold-answer BPB)", ""]
        lines.append("| model | " + " | ".join(tasks) + " |")
        lines.append("|---|" + "---|" * len(tasks))
        for m, recs in ds_models.items():
            cells = []
            for t in tasks:
                tr = [r for r in recs if r["task"] == t]
                if not tr:
                    cells.append("-")
                    continue
                acc = bootstrap_accuracy({m: tr}, "acc_norm", n_boot=2000)[m]
                p, lo, hi = ci(acc)
                gb = np.mean([r["gold_bpb"] for r in tr])
                cells.append(f"{p:.3f} [{lo:.3f},{hi:.3f}] / {gb:.3f}")
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")

    # ---- generation
    gen_rows = []
    for m, L in loaded.items():
        g = L.get("evals", {}).get("final", {}).get("generation")
        if g:
            for lang, d in g.items():
                gen_rows.append(f"| `{m}` | {lang} | {d['script_ratio']:.3f} | {d['distinct_2']:.3f} | {d['rep_4gram']:.3f} | {d['bytes_per_token']:.2f} |")
    if gen_rows:
        lines += ["", "## Greedy generation probes (byte-budgeted)", "",
                  "| model | lang | script ratio | distinct-2 | repeated 4-grams | bytes/token |", "|---|---|---|---|---|---|"] + gen_rows

    # ---- budget / exposure details
    lines += ["", "## Budget and exposure", ""]
    for r in trained:
        s = meta[r]
        lines.append(f"- `{r}`: {s['budget']['steps']} steps, {s['budget']['tokens']/1e6:.1f}M tokens, "
                     f"Greek bytes {s['budget']['bytes_by_lang']['el']/1e6:.1f}MB, English bytes {s['budget']['bytes_by_lang']['en']/1e6:.1f}MB, "
                     f"{s['budget']['flops']/1e15:.2f} PFLOPs, {s['budget']['train_seconds']/60:.1f} min, "
                     f"GradScaler skipped {s['budget']['skipped_steps']} steps, new-token exposure {s.get('new_token_exposure')}")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    write_json(os.path.join(out_dir, "results.json"), {"table": table, "budget_matched": matched, "sets": sets_by_lang})
    try:
        make_plots(loaded, out_dir)
    except Exception as e:  # plotting must never hide the table
        LOG.warning("plotting failed: %s", e)
    LOG.info("report written to %s", out_dir)
    return {"markdown": md, "table": table}


def make_plots(loaded: Dict[str, Dict[str, Any]], out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {r: L for r, L in loaded.items() if L["dev"]}
    if not runs:
        return
    any_dev = next(iter(runs.values()))["dev"][0]["sets"]
    el_sets = [s for s, d in any_dev.items() if d["language"] == "el"]
    en_sets = [s for s, d in any_dev.items() if d["language"] == "en"]

    def lang_bpb(row, sets):
        v = [row["sets"][s]["bpb"] for s in sets if s in row["sets"]]
        return float(np.mean(v)) if v else np.nan

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for r, L in runs.items():
        dev = L["dev"]
        steps = [d["step"] for d in dev]
        gb = [d["bytes_el"] / 1e6 for d in dev]
        fl = [d["flops"] / 1e15 for d in dev]
        el = [lang_bpb(d, el_sets) for d in dev]
        en = [lang_bpb(d, en_sets) for d in dev]
        axes[0].plot(steps, el, marker=".", label=r)
        axes[1].plot(gb, el, marker=".", label=r)
        axes[2].plot(fl, el, marker=".", label=r)
        axes[3].plot(steps, en, marker=".", label=r)
    # the original model's dev BPB = step-0 row of a run without new rows (Exp2 starts post-surgery)
    anchor = next((L["dev"][0] for L in runs.values() if "new_row_norm" not in L["dev"][0]), None)
    for i, (ax, xl, yl) in enumerate(zip(axes, ["optimizer step (matched budget)", "Greek MB seen", "training PFLOPs", "optimizer step"],
                                         ["Greek dev BPB", "Greek dev BPB", "Greek dev BPB", "English dev BPB"])):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        if anchor is not None:
            ax.axhline(lang_bpb(anchor, en_sets if i == 3 else el_sets), color="gray", ls="--", lw=1, label="original model")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for r, L in runs.items():
        el = [lang_bpb(d, el_sets) for d in L["dev"]]
        en = [lang_bpb(d, en_sets) for d in L["dev"]]
        ax.plot(el, en, marker="o", ms=3, label=r)
        ax.annotate("start", (el[0], en[0]), fontsize=7)
    anchor = next((L["dev"][0] for L in runs.values() if "new_row_norm" not in L["dev"][0]), None)
    if anchor is not None:
        ax.plot([lang_bpb(anchor, el_sets)], [lang_bpb(anchor, en_sets)], "k*", ms=12, label="original model")
    ax.set_xlabel("Greek dev BPB (lower = better Greek)")
    ax.set_ylabel("English dev BPB (lower = more English retained)")
    ax.set_title("Trade-off trajectories during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tradeoff_trajectories.png"), dpi=130)
    plt.close(fig)

    if en_sets and any("mean_mass_new" in L["dev"][0]["sets"].get(en_sets[0], {}) for L in runs.values()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for r, L in runs.items():
            ys = [np.mean([d["sets"][s].get("mean_mass_new", 0.0) for s in en_sets]) for d in L["dev"]]
            if max(ys) > 0:
                ax.plot([d["step"] for d in L["dev"]], ys, marker=".", label=r)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("mean prob. mass on new tokens (English)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "english_mass_on_new_tokens.png"), dpi=130)
        plt.close(fig)
