# SmolLM2-135M -> Greek: fine-tune (Exp1) vs. vocabulary extension (Exp2)

Baseline: `base`; reference for extra-cost columns: `exp1_finetune`. 95% CIs from a paired bootstrap over documents (10000 resamples). BPB = bits per UTF-8 byte (lower is better).

Token/step budgets and optimisation (peak LR, schedule, seq len) matched across trained runs: **yes** {'exp1_finetune': (14843904, 453), 'exp2_extend': (14843904, 453)}

| model | vocab | steps | tokens (M) | Greek MB seen | Greek epochs | train PFLOPs | train min | EL BPB | ΔEL vs base | EN BPB | ΔEN vs base | extra EN cost vs ref | extra EL gain vs ref | EN-per-EL exchange |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` | 49152 | 0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0 | 1.2851 [1.2790, 1.2914] | +0.0000 [+0.0000, +0.0000] | 0.9643 [0.9568, 0.9720] | +0.0000 [+0.0000, +0.0000] | -0.1524 [-0.1539, -0.1509] | -0.4115 [-0.4154, -0.4078] | n/a: worse EL, better EN |
| `exp1_finetune` | 49152 | 453 | 14.8 | 31.8 | 0.21 | 13.55 | 28.1 | 0.8736 [0.8680, 0.8794] | -0.4115 [-0.4154, -0.4078] | 1.1167 [1.1085, 1.1249] | +0.1524 [+0.1509, +0.1539] | - | - | - |
| `exp2_extend` | 57152 | 453 | 14.8 | 90.5 | 0.60 | 13.96 | 29.0 | 0.7512 [0.7449, 0.7575] | -0.5339 [-0.5388, -0.5291] | 1.2012 [1.1928, 1.2098] | +0.2369 [+0.2346, +0.2392] | +0.0845 [+0.0827, +0.0863] | +0.1224 [+0.1206, +0.1242] | 0.690 [0.672, 0.709] |
| `exp2_extend@step0` | 57152 | 0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0 | 2.0337 [2.0239, 2.0437] | +0.7486 [+0.7420, +0.7554] | 0.9646 [0.9570, 0.9722] | +0.0002 [+0.0002, +0.0003] | -0.1522 [-0.1536, -0.1507] | -1.1601 [-1.1663, -1.1539] | n/a: worse EL, better EN |

## Per eval set (BPB, point estimates)

| model | wikipedia_el | fineweb2_el | belebele_el | fineweb_edu_en | bbc_news_2025_en | belebele_en |
|---|---|---|---|---|---|---|
| `base` | 1.2997 | 1.2820 | 1.2735 | 0.9178 | 1.0486 | 0.9266 |
| `exp1_finetune` | 0.8618 | 0.9081 | 0.8508 | 1.0595 | 1.2049 | 1.0857 |
| `exp2_extend` | 0.7328 | 0.7945 | 0.7262 | 1.1308 | 1.2945 | 1.1783 |
| `exp2_extend@step0` | 2.0993 | 2.0533 | 1.9485 | 0.9181 | 1.0488 | 0.9268 |

## English probability mass on new (Greek) tokens

`mass_new` = mean probability the model assigns to the new token ids at English positions; `BPB (old-vocab renorm)` removes that leakage, isolating representational forgetting.

| model | set | BPB | BPB (old-vocab renorm) | mass_new |
|---|---|---|---|---|
| `exp2_extend` | fineweb_edu_en | 1.1308 | 1.1091 | 0.05369 |
| `exp2_extend` | bbc_news_2025_en | 1.2945 | 1.2662 | 0.06490 |
| `exp2_extend` | belebele_en | 1.1783 | 1.1475 | 0.07420 |
| `exp2_extend@step0` | fineweb_edu_en | 0.9181 | 0.9178 | 0.00074 |
| `exp2_extend@step0` | bbc_news_2025_en | 1.0488 | 1.0486 | 0.00056 |
| `exp2_extend@step0` | belebele_en | 0.9268 | 0.9266 | 0.00081 |

## Tokenizer efficiency on held-out text

| model | set | tokens/word | bytes/token |
|---|---|---|---|
| `base` | wikipedia_el | 5.67 | 2.12 |
| `base` | fineweb2_el | 5.48 | 2.16 |
| `base` | belebele_el | 5.51 | 2.19 |
| `base` | fineweb_edu_en | 1.32 | 4.64 |
| `base` | bbc_news_2025_en | 1.31 | 4.48 |
| `base` | belebele_en | 1.27 | 4.79 |
| `exp1_finetune` | wikipedia_el | 5.67 | 2.12 |
| `exp1_finetune` | fineweb2_el | 5.48 | 2.16 |
| `exp1_finetune` | belebele_el | 5.51 | 2.19 |
| `exp1_finetune` | fineweb_edu_en | 1.32 | 4.64 |
| `exp1_finetune` | bbc_news_2025_en | 1.31 | 4.48 |
| `exp1_finetune` | belebele_en | 1.27 | 4.79 |
| `exp2_extend` | wikipedia_el | 2.05 | 5.87 |
| `exp2_extend` | fineweb2_el | 1.85 | 6.42 |
| `exp2_extend` | belebele_el | 1.74 | 6.92 |
| `exp2_extend` | fineweb_edu_en | 1.32 | 4.64 |
| `exp2_extend` | bbc_news_2025_en | 1.31 | 4.48 |
| `exp2_extend` | belebele_en | 1.27 | 4.79 |

## Zero-shot cloze tasks (acc_norm, 95% CI; gold-answer BPB)

| model | arc_easy_el | arc_easy_en | hellaswag_el | hellaswag_en | piqa_en |
|---|---|---|---|---|---|
| `base` | 0.254 [0.218,0.292] / 1.654 | 0.558 [0.516,0.602] / 1.157 | 0.293 [0.255,0.331] / 1.174 | 0.409 [0.367,0.453] / 0.899 | 0.674 [0.632,0.714] / 1.182 |
| `exp1_finetune` | 0.240 [0.202,0.276] / 1.257 | 0.520 [0.476,0.562] / 1.265 | 0.323 [0.285,0.361] / 0.846 | 0.403 [0.361,0.445] / 1.035 | 0.652 [0.612,0.694] / 1.336 |
| `exp2_extend` | 0.268 [0.230,0.306] / 1.210 | 0.484 [0.442,0.528] / 1.483 | 0.287 [0.246,0.329] / 0.732 | 0.381 [0.339,0.425] / 1.100 | 0.638 [0.598,0.680] / 1.437 |

## Greedy generation probes (byte-budgeted)

| model | lang | script ratio | distinct-2 | repeated 4-grams | bytes/token |
|---|---|---|---|---|---|
| `base` | el | 1.000 | 0.330 | 0.616 | 1.96 |
| `base` | en | 1.000 | 0.531 | 0.380 | 4.32 |
| `exp1_finetune` | el | 1.000 | 0.554 | 0.378 | 2.00 |
| `exp1_finetune` | en | 0.605 | 0.579 | 0.324 | 3.17 |
| `exp2_extend` | el | 1.000 | 0.799 | 0.113 | 5.83 |
| `exp2_extend` | en | 0.407 | 0.602 | 0.292 | 5.32 |

## Budget and exposure

- `exp1_finetune`: 453 steps, 14.8M tokens, Greek bytes 31.8MB, English bytes 0.0MB, 13.55 PFLOPs, 28.1 min, GradScaler skipped 0 steps, new-token exposure {}
- `exp2_extend`: 453 steps, 14.8M tokens, Greek bytes 90.5MB, English bytes 0.0MB, 13.96 PFLOPs, 29.0 min, GradScaler skipped 0 steps, new-token exposure {'median': 538.0, 'p10': 265.0, 'min': 0, 'never_seen': 3, 'seen_lt_10': 24, 'seen_lt_100': 251}
