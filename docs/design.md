# Design: vocabulary expansion vs. plain fine-tuning for SmolLM2-135M → Greek

This document records every design decision, why it was made, and the evidence behind it.
The evidence comes from a literature/tooling sweep that was adversarially fact-checked (see
`docs/research_report.md`) and from measurements on the real SmolLM2 tokenizer and model.
Numbers marked *(measured)* were reproduced locally.

---

## 1. The problem in one table

| | SmolLM2 tokenizer on **English** | SmolLM2 tokenizer on **Greek** |
|---|---|---|
| bytes / token | 3.8–4.9 (slice-dependent; digits are split) | **2.12** |
| tokens / word | 1.2–1.6 | **5.6–5.7** |
| vocab entries containing the script | ~all | **60 of 49,152** |
| words split into >1 token | – | 99.6% |

*(measured)* Greek is tokenized almost character by character, and every capital except Δ is
split into two UTF-8 *byte* tokens. So Greek costs ~3.5× more tokens per word than English. The
model's context window holds ~3× less Greek text, and every Greek forward pass is ~3× more
expensive per word.

What the two experiments trade off:
- **Exp1** (original tokenizer) keeps the model's input/output space unchanged. All its budget
  goes into learning Greek through pieces the model already has.
- **Exp2** (+8,000 Greek tokens) buys a ~2.8× shorter Greek sequence. It has to pay for that by
  learning 8,000 new rows from scratch inside the same 30-minute budget. Those rows also sit in
  a **tied** softmax that English predictions share.

## 2. Principles that make the comparison valid

1. **Same tokens, same steps, measured compute.** Both arms use the same micro-batch, sequence
   length, accumulation and step count, so tokens and steps match exactly. The report asserts it.
   Compute cannot be equal: Exp2's 57,152-row tied head costs ~3% more FLOPs per token. We
   therefore *meter* FLOPs, wall-clock and GradScaler-skipped steps and report them, rather than
   pretend they match.
2. **Same data, same order.** Both arms draw from the same cleaned document pool, in the same
   seed-determined order. Only the segmentation differs.
3. **Identical optimisation for every shared parameter.** Same optimizer, betas, weight decay,
   clip, peak LR, and WSD schedule anchored to absolute steps. Exp2 only adds one parameter
   group (the 8k new rows).
4. **Tokenizer-agnostic evaluation on identical bytes.** The metric is bits per byte, and every
   eval chunk fits in a single window of the *original* tokenizer. Section 6 explains how this
   removes the context-length confound.
5. **Held-out data is drawn first and protected.** Training docs are decontaminated against
   held-out docs across sources. The tokenizer is learned only from training data.
6. **What Exp2 sees on top of Exp1 is stated, not hidden.** At matched tokens, Exp2 reads ~2.8×
   more Greek bytes. That is the efficiency gain being bought, so we report it and also run a
   byte-matched ablation.

## 3. Data

| role | source | why |
|---|---|---|
| Greek train (75 MB) + held-out docs | `wikimedia/wikipedia` 20231101.el | encyclopedic Greek; CC-BY-SA; ungated |
| Greek train (75 MB) + **official test split** | `HuggingFaceFW/fineweb-2` ell_Grek | diverse web Greek; ODC-By; ungated; ships a designated test split |
| Greek parallel eval | `facebook/belebele` ell_Grek (488 FLORES passages) | out-of-domain; parallel with English, so cross-language cost is commensurate |
| English replay pool + "seen-distribution" eval | `HuggingFaceFW/fineweb-edu` sample-10BT | SmolLM2's own pretraining distribution, so it measures forgetting of what the model knew |
| English post-cutoff eval | `RealTimeData/bbc_news_alltime` 2025-01…06 | published after SmolLM2's release, so it cannot be contaminated |
| English parallel eval | Belebele eng_Latn | same passages as the Greek set |

OSCAR-2301 is suspended and CulturaX is gated. A `culturax_el` source is registered for users
with a Hugging Face token.

**Cleaning (identical for every model and every split).** Steps:
- Unicode NFC.
- Drop control characters and code points ≥ U+40000.
- Minimum 200 characters.
- ≥ 60% of letters in the expected script.
- Exact-duplicate removal.

Do not use NFD: under the GPT-2 regex, combining accents are not letters, so Greek words would
split at every accent. Do not use NFKC: it rewrites English (µ→μ, ligatures). Control bytes
are stripped because SmolLM2's vocab lacks 21 byte values and silently drops them, which
would break `decode(encode(x)) == x` and the byte accounting. *(measured)*

**Splits and decontamination.** For each source, the first N held-out docs are taken (the
official split for FineWeb-2), plus dev docs for learning curves. Only then is the training
pool streamed. Every candidate training doc is checked against *all* held-out docs of the same
language, across sources, because web crawls mirror Wikipedia. It is dropped on a shared ≥60-character
paragraph or a MinHash near-duplicate (word 5-grams, Jaccard ≥ 0.7). FineWeb's 14×8 banding would
catch a 0.7-similar pair only ~56% of the time, so we use 20 bands × 5 rows (~97%) and confirm each
candidate by its estimated Jaccard. This is Paloma's direction: drop training docs, keep the eval set fixed.

**Byte budget arithmetic.** The default budget is 600 steps × 32,768 tokens = 19.7M tokens.
- Exp1 at 2.12 bytes/token reads ~42 MB of Greek.
- Exp2 at ~5.9 bytes/token reads ~116 MB.
- A 100 MB pool would make Exp2 repeat data (≈1.2 epochs) while Exp1 does not.
- We therefore pull **150 MB**, so both arms stay at ≤1 epoch.
- On the free Colab T4, `calibrate` fixed 453 steps (14.8M tokens). Exp1 then read 31.8 MB (0.21 epochs)
  and Exp2 read 90.5 MB (0.60 epochs) (`results/t4_run/`).

`vocabx calibrate` prints the MB needed for the suggested step count, and the trainer warns if
a stream wraps. (Repetition up to ~4 epochs is nearly harmless, Muennighoff et al. 2023, but it
would still be a confound between arms.)

## 4. Experiment 1

Continued pretraining of all parameters with the original tokenizer.

| setting | value | reason |
|---|---|---|
| precision | fp32 master weights, fp16 autocast + GradScaler (T4 has no bf16) | `is_bf16_supported()` is True on Turing only via emulation, so it is never trusted |
| seq len / batch | 1024 × 4 × 8 accum = 32,768 tokens/step | fits a T4 without checkpointing; shorter sequences give more updates in a small budget |
| optimizer | AdamW (0.9, 0.95), eps 1e-8, wd 0.01 on 2-D body matrices only, clip 1.0 | mirrors SmolLM2's pretraining config |
| peak LR | **3e-4**, shared by both arms | ~1/10 of SmolLM2's pretraining peak (3e-3), the usual re-warm level. The evidence spans 1e-4 to 5e-4 (PT/30 to PT/6), and SmolLM2's heavy over-training argues for the low end, because re-warming an annealed model forgets. The LR probe (dev data, both arms, pre-registered rule) can move it |
| schedule | WSD: 7.5% warmup (45 steps at 600; 34 at the T4's calibrated 453, slightly under the ≥~40 guideline, kept at 7.5% so both arms share one schedule), constant, linear decay over the last 20% | SmolLM2's own schedule; decay is anchored to absolute steps so both arms anneal identically; ≥~40 warmup steps tames the re-warming stability gap |
| steps | 600 by default; fixed once by `calibrate` on Exp2 (the slower arm) → `budget.json` | must fit ≤30 min on the slower arm *and* stay ≤1 epoch of the Greek pool; persisted so a Colab reconnect cannot change it |
| attention | SDPA; on pre-Ampere GPUs K/V are expanded before SDPA | otherwise grouped-query attention without a mask falls back to the fp32 *math* kernel on a T4 (~2× activation memory, slower); `calibrate` fails loudly if training still uses it |

## 5. Experiment 2: how the vocabulary is modified

### 5.1 Which tokens, and how they enter the tokenizer: continued BPE

We learn **new merges on top of SmolLM2's own segmentation of Greek words** (Purason et al.,
EACL Findings 2026; the same construction as Liquid AI's 2026 in-place expansion):

1. Pre-tokenize the Greek *training* pool with SmolLM2's own pre-tokenizer (Digits +
   ByteLevel/GPT-2 regex). Keep the pre-tokens that contain Greek bytes, with their counts.
2. Segment each with the original 48,900 merges. This is exactly where SmolLM2 stops.
3. Run greedy BPE on those segmentations, with incremental pair counts and a lazy max-heap.
   Accept the most frequent pair only if its concatenation (a) is not already a vocab entry
   and (b) is pure Greek: an optional leading space, complete Greek characters, and possibly a
   partial UTF-8 character at an edge, always containing a Greek-unique byte pattern
   (0xCE/0xCF, or 0xE1 + 0xBC–0xBF).
4. Stop at 8,000 merges. Append the merges after all original merges (lowest priority) and the
   tokens as ids 49,152…57,151. 57,152 = 893 × 64, so the vocabulary is tensor-core aligned
   with no padding rows.

**Why this guarantees English is untouched.** HF BPE always applies the lowest-ranked available
merge first, so every original merge fires before any new one. A new token never appears in an
original merge. And a new merge can only fire inside a word that contains Greek bytes.
Therefore any text without Greek characters maps to **identical token ids**. The QA harness
checks this on all English eval text, code, and 500 random strings from other scripts. English
documents that *contain* Greek letters (α-helix, etymologies) legitimately change; the harness
counts them.

**Why not the usual recipes.** *(measured on SmolLM2)*
- Training a separate Greek BPE and appending its merges leaves 6–17% of the new tokens
  unreachable. SmolLM2 has already merged the space with the UTF-8 lead byte ('ĠÎ'), which
  pre-empts word-initial Greek merges. Those dead rows still receive softmax gradients.
- `tokenizer.add_tokens` matches before pre-tokenization. It splits words greedily (και+ν+ού+…),
  turns spaces into separate tokens, and compresses worst (3.5 vs 5.9 bytes/token).
- Both are kept as registered ablations (`naive_merge`, `added_tokens`).

**Measured on 12 MB of Greek Wikipedia** (8,000 merges learned in 31 s, pure Python):

| | Greek bytes/token | Greek tokens/word |
|---|---|---|
| original | 2.12 | 5.74 |
| +8k continued BPE | 6.05 | 2.01 |

On held-out text: 0 self-tokenization failures, 7,419 of 8,000 new tokens used, round-trip exact.

A useful property is that the first k merges are themselves a valid k-token extension. So
one run yields a compression curve over merge prefixes (1k/2k/4k/8k for the default +8k tokenizer,
up to 16k for the k16000 ablation). Other `k` values are also registered ablations.

### 5.2 How the new rows are initialised: FVT mean, inside the convex hull

SmolLM2 **ties** input and output embeddings, so one row is both how a new token is *read* and
how strongly it is *predicted*. That makes the output side decide what is safe:

| init (step 0, no training) | Greek BPB | English BPB |
|---|---|---|
| original model + tokenizer | 1.260 | 0.870 |
| HF default (`mean_resizing`: every row ≈ mean) | 3.556 | 0.870 |
| **FVT: mean of the original pieces** | **2.032** | **0.872** |
| first sub-token copy | 2.128 | 0.899 |
| random N(0, 0.0417) (HF `mean_resizing=False`) | 3.92 | 0.993 |

*(measured)* Any row inside the convex hull of existing rows can never become the argmax in an
English context (Mundra et al. 2024, Thm. 2), and Hewitt (2021) bounds the mass a mean-like row
can steal. Random rows leave the hull and immediately pull English probability onto Greek
tokens. FVT is also what Meltemi (Greek), Krikri (Greek), SambaLingo and Liquid use.

Surgery is done on CPU in fp32, by hand rather than with `resize_token_embeddings`. Afterwards
the code asserts that old rows are bit-identical and the weights are still tied.

Alternatives are one-line ablations in the `INITIALIZERS` registry:
- `alpha_shrink`: α·FVT + (1−α)·μ, a hull-preserving temperature on new-token logits.
- `merge_parent_mean`
- `global_mean` and `random_normal`, as controls.

Step-0 loss is a diagnostic, not the decision rule: "Beyond Initialization Loss" (NVIDIA 2026)
shows that init-time loss can mis-rank inits after even ~50 training steps.

### 5.3 How the new rows are trained

Training uses a **split tied embedding**: `old` (49,152 × 576) and `new` (8,000 × 576) are
separate parameters, concatenated for both the input lookup and the output logits. This gives
two properties:
- Exact freezing and per-group LRs. Gradient scaling cannot do this, because Adam is
  scale-invariant.
- A frozen row receives neither updates nor weight decay. A gradient-zeroing hook would still
  let AdamW decay the row.

Exp1 runs the identical module with zero new rows, so both arms use the same code path.

- **Default:** a single stage. New rows get their own group at **10× the body LR** (3e-3,
  SmolLM2's pretraining peak, so never hotter than the base model was trained) and no weight
  decay. Body, norms and old rows follow *exactly* Exp1's schedule, so the only differences
  between arms are the tokenizer, the init and this one extra group.
- **Ablation `exp2_staged`:** the Meltemi/Krikri/Liquid recipe. For the first 10% of steps,
  train only the new rows; then train everything, with the body re-warming and decaying on the
  same absolute steps. Liquid AI found that training *old* tied rows in that stage collapses
  generation, so old rows are never in the warm-up.

This is not the default because, at a fixed 600-step budget, it removes 10% of the body's
updates. The literature (Chinese-LLaMA, SambaLingo, Swallow) is split on whether staging helps
at all. The ablation answers the question for this budget.

## 6. Evaluation

**Primary metric: bits per byte (BPB).** BPB = Σ NLL / (ln 2 · UTF-8 bytes) on the exact same
text for every model (Paloma; The Pile; lm-eval). Per-token perplexity is invalid across
tokenizers and appears only as a within-model diagnostic.

**Context control.** Held-out docs are cut at line/sentence boundaries into chunks of ≤1,000
characters. Each chunk is verified to fit in 1,023 *original*-tokenizer tokens and is scored as
one window with an `<|endoftext|>` prefix. Two consequences:
- Every model scores identical bytes with identical available context.
- Exp2 cannot win just because a 1,024-token window holds 2.8× more Greek.

Bootstrap resamples whole documents (≤4 chunks per doc).

**Sets.**
- Greek: Wikipedia (in-domain), FineWeb-2 test (web), Belebele (out-of-domain, parallel).
- English: FineWeb-Edu (seen distribution), BBC 2025 (post-cutoff), Belebele (parallel).
- Macro BPB per language = mean over its sets.

**Exp2-specific diagnostics.** Computed in the same pass:
- **mass_new**: probability mass on the new Greek ids at English positions.
- **BPB renormalised over the original 49,152 ids**: separates (a) new tied rows competing in
  the softmax from (b) genuine forgetting in shared weights.
- Per-group parameter drift ‖θ−θ₀‖/‖θ₀‖.
- New/old row norms.
- New-token exposure histogram in the training stream (Magikarp-style under-training check).

**Secondary tasks.**
- Zero-shot **cloze** tasks with char-normalised accuracy and gold-answer BPB: HellaSwag and
  ARC-Easy in Greek *and* English built from the *same* ILSP items, plus PIQA. MCF/letter
  formats sit at chance at 135M (Heineman et al. 2025).
- Byte-budgeted greedy generation probes: script ratio, distinct-2, repeated 4-grams. Generation
  can collapse while log-likelihood looks fine (Liquid 2026).

**States evaluated.**
- the base model;
- Exp2 at **step 0** (surgery only), which separates the swap cost from training;
- dev learning curves every 50 steps, plotted against steps, Greek bytes seen and FLOPs;
- the final models.

**Statistics.** Paired, document-clustered bootstrap (10k resamples, shared weights across
models). Headline quantities:
- ΔEL, ΔEN vs base;
- **extra English cost** = BPB_en(Exp2) − BPB_en(Exp1);
- **extra Greek gain** = BPB_el(Exp1) − BPB_el(Exp2);
- their ratio (bits of English given up per bit of Greek gained).

Training noise is not covered by the document bootstrap. `plans/seeds.yaml` adds two
common-random-number seed pairs. The report's `pairs` section gives each pair's paired CI plus the
across-seed SD of the extra English cost. An effect counts as "no extra cost" only if its CI sits
inside a pre-declared margin (0.005 BPB). With 3 seed pairs the minimum detectable effect is
~0.006–0.007 BPB (gap analysis in the research report), so smaller differences are reported as
"below MDE", not as "no difference".

The exchange rate is printed only when the extra Greek gain's CI is entirely above 0. Otherwise the
table names the quadrant (e.g. "worse EL, better EN"), because a ratio of two signed contrasts is
ambiguous.

**Weight-interpolation frontiers** (`vocabx interpolate`) evaluate θ(α) = (1−α)θ_start + αθ_final,
where θ_start is the expanded step-0 model for Exp2. Comparing the arms' frontiers shows English
given up per unit of Greek gained without depending on where the shared LR schedule ended.

## 7. Expected outcome (priors from the literature)

- **Exp2 should win on efficiency.** It needs ~2.8× fewer Greek tokens per document, so it gets
  more Greek per context window and is faster to generate.
- **Greek quality at a matched 20M tokens could go either way.** Exp2 starts ~0.8 BPB behind
  (step 0) and must learn 8k rows quickly. Vocabulary extension typically pays off only after
  hundreds of millions to billions of tokens (Tejaswi 2024; LLaMA-Beyond-English; Dagan 2024).
  It also sees 2.8× more Greek text, which pushes the other way.
- **English.** Both arms forget, because a re-warmed, over-trained 135M model is trained on 100%
  Greek. Exp2 may lose a little extra through softmax leakage onto the new tied rows. The
  renormalised-BPB diagnostic quantifies exactly that part.

Any of these outcomes is informative, and the writeup reads it through the diagnostics above.

## 8. Ablations (ranked by insight per GPU-hour; all are config files)

1. `exp2_bytes_matched`: Exp2 stops at Exp1's Greek bytes. Separates "better tokenizer" from
   "more text".
2. `exp1_replay20` + `exp2_replay20`: 20% English in both arms. Does Exp2's extra English cost
   survive best practice?
3. `exp2_staged`: new-rows-only warm-up inside the budget.
4. `exp2_init_alpha075`: logit temperature on new rows.
5. `exp2_k2000` / `k4000` / `k16000`: fewer, better-trained tokens vs. more compression.
6. `exp2_newlr1x`, `exp2_init_merge`, `exp2_naive_merge`, controls `exp2_init_random`,
   `exp2_init_globalmean`.
7. `plans/seeds.yaml`: training-noise floor.
8. `plans/lr_probe.yaml`: LR choice by a pre-registered rule on dev data, for both arms.

## 9. Risk register

| risk | guard |
|---|---|
| eval text leaks into training or the tokenizer | held-out first; cross-source paragraph + MinHash decontamination; tokenizer learned from the train pool only |
| English "forgetting" is really a tokenization change | English-identity invariant asserted on all eval text; Greek-containing English docs counted separately |
| Exp2 wins only by fitting more text per window | single-window chunks sized for the original tokenizer |
| Exp2 repeats data, Exp1 does not | 150 MB pool; epochs reported; warning on wrap |
| budgets silently differ | report asserts equal tokens/steps; FLOPs, skipped steps and wall-clock logged |
| fp16 overflow on a bf16-trained model | fp32 master weights and residual stream; GradScaler with logged skips; fp32 softmax/CE |
| HF resize gives identical mean rows / unties weights | manual surgery; asserts on old rows and tying |
| frozen rows still decay under AdamW | split parameters + `requires_grad=False` (no grad means no decay) |
| a single 30-min run is noise | document bootstrap + seed plan + pre-declared margin |
| Colab disconnects | checkpoints every 5 min (model, optimizer, scaler, RNG, stream cursors) on Drive; exact resume; logs truncated to the checkpoint step |
| budget drifts between sessions | `budget.json` written once; a finished/checkpointed run with a different budget signature (steps, tokens/step, LR, schedule, stages, init, seed) is refused |
| re-evaluation uses the wrong tokenizer | `vocabx eval` loads the tokenizer saved with the model (meta.json), never one rebuilt from config |
| LR choice favours one arm | shared LR, fixed a priori; probe uses dev data, both arms, and a pre-registered rule |

## 10. Considered and not used (and why)

These remain good candidates for future plug-ins.

| method | why not here |
|---|---|
| ZeTT hypernetworks | no SmolLM2 hypernetwork; meta-training costs days of TPU/H100 time |
| FOCUS / WECHSEL / CLP / OFA | need fastText, bilingual dictionaries or a donor model; no consistent gain over subtoken means for expansion with little data (Yamaguchi 2024; Tejaswi 2024) |
| OMP transplant, TokAlign, MATT | built for whole-vocabulary replacement or need a donor; small models suffer |
| Token Distillation, VocADT | promising model-aware inits, but with tied embeddings they need norm/hull control, and their steps must be charged to Exp2's budget. Natural next plug-ins |
| LoRA | "learns less and forgets less" (Biderman 2024); worse for language adaptation (Tejaswi 2024); full FT is cheap at 135M |
| untied head for Exp2 | changes the architecture and adds 33M parameters vs Exp1, confounding the comparison |

## References

- Purason et al., *Teaching Old Tokenizers New Words* (EACL Findings 2026). https://arxiv.org/abs/2512.03989; toolkit https://github.com/taidopurason/tokenizer-extension
- Liquid AI, *In-Place Tokenizer Expansion* (2026). https://arxiv.org/abs/2607.15232
- Meltemi (Greek Mistral-7B). https://arxiv.org/abs/2407.20743; Llama-Krikri. https://arxiv.org/abs/2505.13772
- Tejaswi et al., *Exploring Design Choices for Building Language-Specific LLMs*. https://arxiv.org/abs/2406.14670
- Yamaguchi et al., *How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?* https://arxiv.org/abs/2406.11477
- Mundra et al., *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for LMs*. https://arxiv.org/abs/2407.05841
- Hewitt, *Initializing New Word Embeddings for Pretrained Language Models* (2021). https://nlp.stanford.edu/~johnhew/vocab-expansion.html
- Land & Bartolo, *Fishing for Magikarp* (under-trained tokens). https://arxiv.org/abs/2405.05417
- Joshi et al., *Beyond Initialization Loss* (2026). https://arxiv.org/abs/2608.03494
- Magnusson et al., *Paloma*. https://arxiv.org/abs/2312.10523
- Heineman et al., *Signal and Noise*. https://arxiv.org/abs/2508.13144
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train LLMs*. https://arxiv.org/abs/2403.08763
- Gupta et al., *Continual Pre-Training of LLMs: How to (re)warm your model?* https://arxiv.org/abs/2308.04014
- Muennighoff et al., *Scaling Data-Constrained Language Models*. https://arxiv.org/abs/2305.16264
- SmolLM2 paper. https://arxiv.org/abs/2502.02737; pretraining config https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml
- EEVE. https://arxiv.org/abs/2402.14714; Swallow. https://arxiv.org/abs/2404.17790; SambaLingo. https://arxiv.org/abs/2404.05829
- Zhao et al., *LLaMA Beyond English*. https://arxiv.org/abs/2401.01055; Dagan et al., *Getting the most out of your tokenizer*. https://arxiv.org/abs/2402.01035
- Biderman et al., *LoRA Learns Less and Forgets Less*. https://arxiv.org/abs/2405.09673
- FOCUS https://arxiv.org/abs/2305.14481; WECHSEL https://arxiv.org/abs/2112.06598; ZeTT https://arxiv.org/abs/2405.07883; VocADT https://arxiv.org/abs/2410.09644; Token Distillation https://arxiv.org/abs/2505.20133
