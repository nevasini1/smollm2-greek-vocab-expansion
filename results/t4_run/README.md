# Colab T4 run, 2026-10-15

Raw outputs of the study, as written by `vocabx` to the run's `workdir` (model weights and
checkpoints omitted). The writeup is [`docs/writeup.md`](../../docs/writeup.md).

## Provenance

| | |
|---|---|
| code | `main` at 90f6ba9 (cloned by `notebooks/colab_run_all.ipynb`) |
| hardware | free Colab, Tesla T4 (sm 7.5), 15 GB |
| software | Python 3.13, torch 2.11.0+cu130, transformers 5.18.0, tokenizers 0.23.2, datasets 4.8.5 |
| storage | Colab VM disk (`/content/vocabx_store/artifacts`); Drive mount was skipped |
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
   `--set 'eval.downstream=[]' eval.generation=false eval.full_eval_at_step0=false train.eval_every=100`
   (so that run has no cloze or generation numbers)

## Layout

```
budget.json, calibration/            matched step budget and the measurement behind it
data/<key>/manifest.json             data sizes, cleaning and decontamination counts, eval-set hashes
tokenizers/ad1bfd17b231/             the +8,000-token tokenizer (tokenizer.json, meta.json, qa.json)
runs/<run>/summary.json              budget (tokens, steps, bytes, FLOPs, minutes), config, environment
runs/<run>/dev_evals.jsonl           learning curves
runs/<run>/train_log.jsonl           loss, LR, grad norm, GradScaler scale
runs/<run>/eval/records_*.jsonl.gz   per-chunk NLL/bytes records (re-bootstrappable)
runs/<run>/eval/interpolation.json   weight-interpolation frontier
results/main, results/ablations      reports generated on Colab
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
