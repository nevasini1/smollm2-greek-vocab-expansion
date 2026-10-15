"""Zero-shot cloze tasks (registry), scored tokenizer-agnostically.

At 135M parameters, letter-choice (MCF) formats sit at chance, so every task uses the
cloze formulation and reports: ``acc`` (raw log-likelihood), ``acc_norm`` (LL divided
by the continuation's character length - lm-eval's tokenizer-agnostic
normalisation), and ``gold_bpb`` (bits per byte of the correct answer, a
continuous, higher-SNR signal; Heineman et al. 2025).

Greek and English versions of HellaSwag and ARC-Easy are built from the SAME ILSP
items (Greek fields vs. the ``*_orig``/``*_en`` source fields), so Greek-vs-English
comparisons are on parallel content.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..registry import Registry
from ..utils import LOG
from .scoring import LN2, ScoreItem, score_items

TASKS = Registry("downstream task")


@dataclass
class ClozeItem:
    id: str
    context: str
    choices: List[str]
    gold: int


@dataclass
class TaskSpec:
    language: str
    loader: Callable[[int, int], List[ClozeItem]]
    chance: float


def _subsample(n_total: int, n: int, seed: int) -> List[int]:
    if n <= 0 or n >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), n))


def _hs_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hellaswag(lang: str):
    def load(n: int, seed: int) -> List[ClozeItem]:
        from datasets import load_dataset

        ds = load_dataset("ilsp/hellaswag_greek", split="validation")
        sfx = "" if lang == "el" else "_orig"
        out = []
        for i in _subsample(len(ds), n, seed):
            r = ds[i]
            ctx = r["ctx_a" + sfx] + " " + (r["ctx_b" + sfx] or "").capitalize()
            endings = r["endings" + sfx]
            if isinstance(endings, str):
                import ast

                endings = ast.literal_eval(endings)
            out.append(
                ClozeItem(
                    f"hellaswag:{r['ind']}",
                    _hs_pre(r["activity_label" + sfx] + ": " + ctx),
                    [" " + _hs_pre(e) for e in endings],
                    int(r["label"]),
                )
            )
        return out

    return load


def _arc_easy(lang: str):
    q_word, a_word = ("Ερώτηση", "Απάντηση") if lang == "el" else ("Question", "Answer")

    def load(n: int, seed: int) -> List[ClozeItem]:
        from datasets import load_dataset

        ds = load_dataset("ilsp/arc_greek", "ARC-Easy", split="test")
        q_f, c_f = ("question", "choices") if lang == "el" else ("question_en", "choices_en")
        out = []
        for i in _subsample(len(ds), n, seed):
            r = ds[i]
            labels = list(r[c_f]["label"])
            if r["answerKey"] not in labels:
                continue
            out.append(
                ClozeItem(
                    f"arc_easy:{r['id']}",
                    f"{q_word}: {r[q_f]}\n{a_word}:",
                    [" " + t for t in r[c_f]["text"]],
                    labels.index(r["answerKey"]),
                )
            )
        return out

    return load


def _piqa(n: int, seed: int) -> List[ClozeItem]:
    from datasets import load_dataset

    ds = load_dataset("baber/piqa", split="validation")
    out = []
    for i in _subsample(len(ds), n, seed):
        r = ds[i]
        out.append(ClozeItem(f"piqa:{i}", f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
    return out


TASKS.register("hellaswag_el")(TaskSpec("el", _hellaswag("el"), 0.25))
TASKS.register("hellaswag_en")(TaskSpec("en", _hellaswag("en"), 0.25))
TASKS.register("arc_easy_el")(TaskSpec("el", _arc_easy("el"), 0.25))
TASKS.register("arc_easy_en")(TaskSpec("en", _arc_easy("en"), 0.25))
TASKS.register("piqa_en")(TaskSpec("en", _piqa, 0.5))


def run_task(
    name: str,
    model,
    tokenizer,
    device: str,
    amp_dtype=None,
    n: int = 500,
    seed: int = 0,
    seq_len: int = 1024,
    batch_tokens: int = 16384,
    eos_id: int = 0,
    items: Optional[List[ClozeItem]] = None,
) -> Dict[str, Any]:
    spec: TaskSpec = TASKS.get(name)
    items = items if items is not None else spec.loader(n, seed)
    seqs: List[ScoreItem] = []
    for ii, it in enumerate(items):
        ctx_ids = tokenizer.encode(it.context, add_special_tokens=False).ids
        for ci, ch in enumerate(it.choices):
            full = tokenizer.encode(it.context + ch, add_special_tokens=False).ids
            if full[: len(ctx_ids)] == ctx_ids:
                cont = full[len(ctx_ids) :]
            else:
                cont = tokenizer.encode(ch, add_special_tokens=False).ids
            ids = [eos_id] + ctx_ids + cont
            score_from = 1 + len(ctx_ids)
            if len(ids) > seq_len:  # keep the continuation, trim context from the left
                cut = len(ids) - seq_len
                ids = ids[cut:]
                score_from = max(1, score_from - cut)
            seqs.append(ScoreItem(ids, score_from, (ii, ci)))
    res = score_items(model, seqs, device, amp_dtype, batch_tokens)
    ll: Dict[tuple, float] = {s.key: -r.nll for s, r in zip(seqs, res)}
    rows = []
    for ii, it in enumerate(items):
        lls = [ll[(ii, ci)] for ci in range(len(it.choices))]
        norm = [l / max(len(c), 1) for l, c in zip(lls, it.choices)]
        gold_bytes = len(it.choices[it.gold].encode("utf-8"))
        rows.append(
            {
                "task": name,
                "id": it.id,
                "acc": float(max(range(len(lls)), key=lls.__getitem__) == it.gold),
                "acc_norm": float(max(range(len(norm)), key=norm.__getitem__) == it.gold),
                "gold_bpb": -lls[it.gold] / (LN2 * gold_bytes),
            }
        )
    k = max(len(rows), 1)
    summary = {
        "task": name,
        "language": spec.language,
        "n": len(rows),
        "chance": spec.chance,
        "acc": sum(r["acc"] for r in rows) / k,
        "acc_norm": sum(r["acc_norm"] for r in rows) / k,
        "gold_bpb": sum(r["gold_bpb"] for r in rows) / k,
    }
    summary["acc_norm_se"] = math.sqrt(summary["acc_norm"] * (1 - summary["acc_norm"]) / k)
    return {"summary": summary, "records": rows}
