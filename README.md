# vocabx: vocabulary expansion vs. fine-tuning for SmolLM2-135M → Greek

This repo compares two ways of adapting [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
to Greek, with **matched tokens and steps**, **measured compute**, and **tokenizer-agnostic evaluation**:

- **Exp1:** continue training with the original 49,152-token tokenizer.
- **Exp2:** add **8,000 Greek tokens** by *continued BPE* (new merges learned on top of SmolLM2's
  own merges), initialise them as the **mean of their original sub-token embeddings**, then train
  on the same budget.

Every component is a plug-in behind a registry and every experiment is a YAML file, so new
tokenizer extenders, embedding initialisers, training schedules, data sources and evals can be
added without touching the training loop.

## Results (free Colab T4, matched budget: 453 steps / 14.8M tokens, ≤30 min per arm)

| bits per byte (lower = better), macro over 3 test sets | Greek | English |
|---|---|---|
| original SmolLM2-135M | 1.285 | 0.964 |
| **Exp1** fine-tune, original tokenizer | 0.874 | 1.117 |
| **Exp2** +8,000 Greek tokens | **0.751** | 1.201 |
| Exp2 stopped at Exp1's Greek bytes (supplementary: 159 steps, 35% of the compute) | 0.865 | **1.073** |

- **Matched tokens:** Exp2 gains an extra 0.122 BPB of Greek for an extra 0.084 BPB of English.
- **Matched Greek text:** Exp2 matches Exp1's Greek with about a third of the compute and forgets 0.044 BPB less English.
- **Writeup (1–2 pages):** [`docs/writeup.md`](docs/writeup.md). Raw outputs:
  [`results/t4_run/`](results/t4_run/).

## Documentation

- Why each choice was made, with evidence: [`docs/design.md`](docs/design.md)
- Full literature/tooling sweep: [`docs/research_report.md`](docs/research_report.md)
- The 1–2 page writeup with results: [`docs/writeup.md`](docs/writeup.md)

## Quick start

### Colab (free T4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nevasini1/smollm2-greek-vocab-expansion/blob/main/notebooks/colab_run_all.ipynb)

Open [`notebooks/colab_run_all.ipynb`](notebooks/colab_run_all.ipynb) (or the badge), choose a T4 runtime, and run the
cells top to bottom. Artifacts persist on Google Drive and every stage resumes after a disconnect.

### Command line

```bash
pip install -e .
```

```bash
python -m vocabx prepare   --config configs/exp2_extend.yaml   # data pools + fixed eval sets (cached)
```

```bash
python -m vocabx tokenizer --config configs/exp2_extend.yaml   # +8k tokens and the QA harness
```

```bash
python -m vocabx calibrate --config configs/exp2_extend.yaml   # fixes the matched budget ONCE -> artifacts/budget.json
```

```bash
python -m vocabx plan configs/plans/main.yaml   # base + Exp1 + Exp2 + report (every run reads budget.json)
```

**The matched budget is fixed once.** `calibrate` measures the slower arm (Exp2) and writes
`artifacts/budget.json` with max_steps = min(steps that fit 27 min, steps the Greek pool covers
without repetition). Every later trained run reads that file, in this session or any later one.
- A finished or checkpointed run that was trained with a different budget is refused, not
  silently mixed in.
- If you run the LR probe (`plans/lr_probe.yaml` + `vocabx curves`), its pre-registered choice is
  saved to `artifacts/lr_choice.json` and applied to all later runs in the same way.
- An explicit `--set train.max_steps=...` or `--set train.optim.lr=...` always wins.

Results land in `artifacts/results/main/`:
- `results.md`: tables with bootstrap CIs
- `learning_curves.png`
- `tradeoff_trajectories.png`
- `english_mass_on_new_tokens.png`

Laptop smoke test (tiny data, 6 steps, every code path, CPU/MPS):

```bash
python -m vocabx plan configs/plans/smoke.yaml
```

Tests (`-m "not network"` skips the ones that download the SmolLM2 tokenizer):

```bash
python -m pytest
```

## What happens in a run

```
prepare ─► held-out docs drawn FIRST (official test split / head of shuffled stream)
           └► training pool streamed, cleaned (NFC, control chars, script filter, exact dedup),
              decontaminated vs ALL held-out docs (paragraph hash + MinHash), shuffled
tokenizer ─► continued BPE on the Greek *training* pool → append 8k merges/ids
             └► QA: original ids untouched · non-Greek text tokenizes identically · round trip
                · every new token reachable · compression + 1k–16k vocab-size curve
train ─► build model (fp32) → vocabulary surgery (FVT init, tie + old-row asserts)
         → split tied embedding [old | new] → step-0 eval → WSD training with metered
           tokens/bytes/FLOPs, dev curves every 50 steps, 5-min checkpoints → final eval
report ─► paired, document-clustered bootstrap → tables + plots; asserts matched budgets
```

## Default protocol

The rationale for each item is in [`docs/design.md`](docs/design.md).

| | Exp1 | Exp2 |
|---|---|---|
| tokenizer | original (49,152) | + 8,000 continued-BPE Greek tokens (57,152 = 893×64) |
| new-row init | – | mean of original sub-token embeddings (FVT), tied |
| data | 150 MB Greek pool (Wikipedia + FineWeb-2), same docs, same order | same |
| budget | 600 steps × 32,768 tokens (19.7M), seq 1024 | identical (asserted) |
| optimizer | AdamW (0.9, 0.95), wd 0.01 on body matrices, clip 1.0 | identical + new rows at 10× LR, wd 0 |
| schedule | WSD, peak 3e-4, 7.5% warmup, last 20% linear decay | identical (same absolute steps) |
| precision | fp32 master, fp16 autocast + GradScaler on T4 | identical |
| eval | BPB on 3 Greek + 3 English held-out sets, single-window chunks; cloze tasks; generation | identical + step-0 eval + new-token diagnostics |

## Repository map

```
configs/
  base.yaml                 shared settings (every experiment inherits it)
  base_model.yaml           off-the-shelf baseline (eval only)
  exp1_finetune.yaml        Experiment 1
  exp2_extend.yaml          Experiment 2
  ablations/*.yaml          one file per ablation (3-6 lines each)
  plans/*.yaml              main / ablations / seeds / lr_probe / smoke run lists
src/vocabx/
  config.py                 typed, strict, inheritable YAML config + cache keys
  registry.py               the plug-in Registry + external plugin loading
  data/                     sources (registry), cleaning, chunking, decontamination, packing/streams
  tokenization/             continued BPE, extenders (registry), byte-level helpers, QA harness
  modeling/                 initializers (registry), surgery, split tied embedding
  training/                 trainer, WSD schedule, FLOPs/budget meter, calibration
  evaluation/               batched scorer, BPB, cloze tasks (registry), generation, bootstrap
  analysis/report.py        tables + plots
notebooks/colab_run_all.ipynb
tests/                      invariants (English identity, reachability, freezing, BPB math, ...)
```

## Extending it

Each extension is a registry entry plus a config line. Modules listed under `plugins:` in a
config are imported first, so extensions can live outside this package.

**A new embedding initializer** (e.g. a learned or model-aware init):
```python
# my_plugins.py
from vocabx.modeling.init import INITIALIZERS, InitContext

@INITIALIZERS.register("fvt_norm_matched")
def fvt_norm_matched(ctx: InitContext, scale: float = 0.8):
    rows = INITIALIZERS.get("fvt_mean")(ctx)
    target = ctx.E_old.norm(dim=1).mean() * scale
    return rows / rows.norm(dim=1, keepdim=True) * target
```
```yaml
# configs/ablations/exp2_init_normmatched.yaml
inherits: ../exp2_extend.yaml
run_name: exp2_init_normmatched
plugins: [my_plugins.py]
init: {method: fvt_norm_matched, params: {scale: 0.8}}
```

**A new tokenizer extender:** register `fn(base_tokenizer, texts_fn, tokenizer_cfg) -> Extension`
in `EXTENDERS`. Return the modified `tokenizer.json`, the new tokens, and optionally merge parents.
The QA harness, surgery and evaluation pick it up automatically.

**A new training schedule:** no code needed for staged freezing or per-group LRs.
```yaml
train:
  stages:
    - {name: warmup, until_frac: 0.1, trainable: [emb_new]}
    - {name: full,   until_frac: 1.0, trainable: [body, norms, emb_old, emb_new]}
  group_lr_mult: {emb_new: 10.0}
```

**A new corpus or eval set:** subclass `HFSource` and register it in `DATA_SOURCES`, then list it
under `data.greek` or `data.english` (with `test_docs`/`dev_docs`) and under `eval.sets`.

**A new downstream task:** register a `TaskSpec(language, loader, chance)` in `TASKS`, where
`loader(n, seed)` returns `ClozeItem`s.

**Re-evaluating old runs with new evals:** `python -m vocabx eval --config <run config>` re-scores
the saved `final/` model. It uses the tokenizer saved *with* that model, never one rebuilt from an
edited config. For the baseline, and for Exp2's step-0 state (`--untrained`), it rebuilds the
untrained model and re-scores it. Per-document records allow paired comparison with every earlier run.

**More analysis commands:**
- `vocabx interpolate`: WiSE-FT frontiers θ(α) between a run's start and end weights.
- `vocabx curves`: dev-set deltas and the LR-probe decision.
- A plan's `report.pairs` block: explicit paired contrasts (CRN seed pairs, the replay pair) with bootstrap CIs.

Show everything that is registered:

```bash
python -m vocabx registries
```

## Reproducibility

- **Cache keys.** Each cached artifact (data, tokenizer, packed corpus) is keyed by a hash of
  the config fields that determine it. Arms that should share data do so by construction.
- **Per-run records.** Every run stores:
  - its resolved `config.yaml`
  - environment versions
  - `train_log.jsonl`
  - `dev_evals.jsonl`
  - per-document eval records
  - `summary.json` (budget: tokens, bytes per language, FLOPs, skipped steps, wall-clock, epochs, new-token exposure)
- **Eval set identity.** Eval sets are frozen files with SHA-256 hashes in the data manifest.
- **Determinism.** Seeds control the document order and init. Arms with the same `seed` form
  common-random-number pairs.
