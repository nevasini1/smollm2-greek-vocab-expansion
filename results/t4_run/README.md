# Colab T4 run, 2026-10-15

Raw outputs of the study, as written by `vocabx` to the run's `workdir` (model weights and
checkpoints omitted). The writeup is [`docs/writeup.md`](../../docs/writeup.md).

**The executed Colab notebook, with every cell's output (logs, QA report, budget, results tables and
plots), is [`colab_run_all.executed.ipynb`](colab_run_all.executed.ipynb).** Its first cell lists how it
differs from the notebook in the repo today and what was cleaned before publishing.

## Provenance

| | |
|---|---|
| code | `main` at 90f6ba9 (cloned by `notebooks/colab_run_all.ipynb`) |
| hardware | free Colab, Tesla T4 (sm 7.5), 15 GB |
| software | Python 3.13, torch 2.11.0+cu130, transformers 5.18.0, tokenizers 0.23.2, datasets 4.8.5 |
| storage | Colab VM disk (`/content/vocabx_store/artifacts`); the Drive mount timed out, so Drive was not used |
| throughput | 9,182 tokens/s on Exp2 (3.57 s per 32,768-token step), peak 9.4 GB |
| matched budget | `budget.json`: 453 steps × 32,768 tokens = 14.8M tokens per arm (27-min fit; the Greek pool allowed 750) |
| data | 150 MB Greek training text (Wikipedia el + FineWeb-2 ell_Grek), 30 MB English (replay pool, unused in these runs); 325 Greek and 33 English training docs removed by decontamination |

The notebook was run cell by cell in this order:

1. prepare
2. tokenizer and QA
3. calibrate
4. `plans/main.yaml` (base, Exp1, Exp2)
5. `interpolate` on both arms, using the dev sets
6. `plans/ablations.yaml --only exp2_bytes_matched`, with
   `--set 'eval.downstream=[]' --set eval.generation=false --set eval.full_eval_at_step0=false --set train.eval_every=100`
   (so that run has no cloze or generation numbers)

### Deviations from the notebook as it was at 90f6ba9

The notebook and CLI have since been updated to follow the same steps (the notebook's `results/main` plots
now use the newer plotting code). At the time:

- **Setup cell.** The notebook's setup cell always mounted Drive, and the mount timed out. A substitute
  cell was typed in Colab: the same clone and `pip install -e .`, with `DRIVE_ROOT = "/content/vocabx_store"`
  and no mount. The notebook now has `USE_DRIVE` (`True`, `"auto"` to fall back to the VM disk, or `False`).
- **`prepare` did not exit.** It logged `data artifact ready` (the manifest is written last), but the process
  stayed alive, so the cell was interrupted by hand. PR #2 (`os._exit(0)` after each CLI command, on `main`
  since cd617f8) fixes this.
- **Ablation cell.** The cell was replaced with the step-6 command above. It is now the notebook's default §7 cell.
- **Export.** Two zips were made from Colab's terminal (after §6 and after §7), leaving out model weights
  (`runs/*/final/`) and checkpoints, and downloaded. The notebook's new §8 makes one such zip with Python and
  also builds `results/combined/`; for this run both were done outside the notebook.
- **Plots.** `results/main` and `results/ablations` were drawn on Colab with the plotting code at 90f6ba9
  (learning curves as a 1×4 strip). `results/combined/` and `results/interpolation_frontiers.png` were
  regenerated locally with the plotting code now on `main` (2×2 grid, shorter frontier title), using the
  commands below; notebook §8 builds the same combined report on Colab.

## Layout

```
colab_run_all.executed.ipynb         the notebook as run on Colab, with all outputs
budget.json, calibration/            matched step budget and the measurement behind it
data/<key>/manifest.json             data sizes, cleaning and decontamination counts, eval-set hashes
tokenizers/ad1bfd17b231/             the +8,000-token tokenizer (tokenizer.json, meta.json, qa.json)
runs/<run>/summary.json              budget (tokens, steps, bytes, FLOPs, minutes), config, environment
runs/<run>/dev_evals.jsonl           learning curves
runs/<run>/train_log.jsonl           loss, LR, grad norm, GradScaler scale
runs/<run>/eval/records_*.jsonl.gz   per-chunk NLL/bytes records (re-bootstrappable)
runs/<run>/eval/interpolation.json   weight-interpolation frontier
results/main, results/ablations      reports generated on Colab (90f6ba9 plotting code)
results/combined                     report over all four models + paired contrasts
results/interpolation_frontiers.png
```

## Regenerating the combined report and plots

From the repo root, after `pip install -e .`:

```bash
python -c "from vocabx.analysis.report import build_report; build_report('results/t4_run', ['base','exp1_finetune','exp2_extend','exp2_bytes_matched'], 'base', 'exp1_finetune', out_dir='results/t4_run/results/combined', title='SmolLM2-135M -> Greek on a Colab T4: baseline, Exp1, Exp2, and Exp2 matched on Greek bytes', pairs=[['exp2_extend','exp1_finetune'],['exp2_bytes_matched','exp1_finetune']])"
```

```bash
python -c "from vocabx.analysis.interpolate import plot_frontiers; plot_frontiers('results/t4_run', ['exp1_finetune','exp2_extend'], 'results/t4_run/results/interpolation_frontiers.png')"
```
