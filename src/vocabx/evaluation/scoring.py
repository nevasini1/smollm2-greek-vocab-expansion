"""Batched log-likelihood scoring shared by intrinsic and downstream evaluation.

Every scored sequence is ``ids`` with predictions counted for ``ids[score_from:]``.
Logits are produced in row slices and reduced in fp32, so a 57k-vocab softmax never
materialises for a whole batch. When ``n_base`` is given (extended vocab), each
position also yields:

* ``mass_new``  - probability mass the model puts on the *new* (Greek) token ids;
* ``nll_old``   - NLL renormalised over the original ``n_base`` ids (a target that is
  itself a new id keeps its full NLL; their share is reported). For English text,
  nll - nll_old isolates the cost of the new rows competing in the tied softmax from
  genuine representational forgetting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

LN2 = math.log(2.0)


@dataclass
class ScoreItem:
    ids: List[int]
    score_from: int = 1
    key: Any = None


@dataclass
class ScoreResult:
    nll: float = 0.0
    n_scored: int = 0
    mass_new: float = 0.0
    nll_old: float = 0.0
    n_old_targets: int = 0
    token_nlls: Optional[List[float]] = None


def backbone(model):
    if hasattr(model, "get_decoder"):
        try:
            dec = model.get_decoder()
            if dec is not None:
                return dec
        except Exception:
            pass
    return model.model


def _batches(order: List[int], items: Sequence[ScoreItem], batch_tokens: int):
    batch: List[int] = []
    max_len = 0
    for i in order:
        L = len(items[i].ids)
        if batch and max(max_len, L) * (len(batch) + 1) > batch_tokens:
            yield batch
            batch, max_len = [], 0
        batch.append(i)
        max_len = max(max_len, L)
    if batch:
        yield batch


@torch.no_grad()
def score_items(
    model,
    items: Sequence[ScoreItem],
    device: str,
    amp_dtype: Optional[torch.dtype] = None,
    batch_tokens: int = 16384,
    n_base: Optional[int] = None,
    slice_rows: int = 2048,
    keep_token_nlls: bool = False,
    pad_id: int = 0,
) -> List[ScoreResult]:
    was_training = model.training
    model.eval()
    results = [ScoreResult() for _ in items]
    order = sorted(range(len(items)), key=lambda i: -len(items[i].ids))
    body = backbone(model)
    head = model.get_output_embeddings()
    for batch in _batches(order, items, batch_tokens):
        T = max(len(items[i].ids) for i in batch)
        inp = torch.full((len(batch), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        rows_b, rows_t, tgts = [], [], []
        for bi, i in enumerate(batch):
            ids = items[i].ids
            inp[bi, : len(ids)] = torch.as_tensor(ids)
            att[bi, : len(ids)] = 1
            for t in range(max(items[i].score_from, 1), len(ids)):
                rows_b.append(bi)
                rows_t.append(t - 1)
                tgts.append(ids[t])
        if not tgts:
            continue
        inp, att = inp.to(device), att.to(device)
        ctx = torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype) if amp_dtype else _null()
        with ctx:
            hidden = body(input_ids=inp, attention_mask=att).last_hidden_state
        rb = torch.as_tensor(rows_b, device=device)
        rt = torch.as_tensor(rows_t, device=device)
        tg = torch.as_tensor(tgts, device=device)
        h = hidden[rb, rt]
        W = head.weight
        nll_all, mass_all, nllold_all = [], [], []
        for s in range(0, h.shape[0], slice_rows):
            with ctx:
                logits = torch.nn.functional.linear(h[s : s + slice_rows], W)
            logits = logits.float()
            lse = torch.logsumexp(logits, dim=-1)
            tgt = tg[s : s + slice_rows]
            tl = logits.gather(1, tgt[:, None]).squeeze(1)
            nll_all.append(lse - tl)
            if n_base is not None and logits.shape[1] > n_base:
                lse_new = torch.logsumexp(logits[:, n_base:], dim=-1)
                lse_old = torch.logsumexp(logits[:, :n_base], dim=-1)
                mass_all.append(torch.exp(lse_new - lse))
                # renormalised over the original ids; a (rare) new-id target keeps its full NLL
                nllold_all.append(torch.where(tgt < n_base, lse_old - tl, lse - tl))
        nll = torch.cat(nll_all).cpu().double()  # move first: MPS has no float64
        mass = torch.cat(mass_all).cpu().double() if mass_all else None
        nllold = torch.cat(nllold_all).cpu().double() if nllold_all else None
        tg_cpu = tg.cpu()
        pos = 0
        for bi, i in enumerate(batch):
            n = len(items[i].ids) - max(items[i].score_from, 1)
            if n <= 0:
                continue
            r = results[i]
            seg = nll[pos : pos + n]
            r.nll = float(seg.sum())
            r.n_scored = n
            if keep_token_nlls:
                r.token_nlls = seg.tolist()
            if mass is not None:
                r.mass_new = float(mass[pos : pos + n].sum())
                r.nll_old = float(nllold[pos : pos + n].sum())
                r.n_old_targets = int((tg_cpu[pos : pos + n] < n_base).sum())
            pos += n
    if was_training:
        model.train()
    return results


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
