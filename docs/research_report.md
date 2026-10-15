# Vocabulary Expansion on SmolLM2-135M: Research Report

This report supports a take-home study that adapts `HuggingFaceTB/SmolLM2-135M` to Greek two ways at matched token and step budgets on a free Colab T4:

- **Exp1:** continued pretraining (CPT) with the original 49,152-token tokenizer.
- **Exp2:** add about 8,000 Greek tokens, then train.

The two arms are compared on Greek held-out LM quality and English retention.

**How the evidence was built.**
- Seven research dimensions were swept independently.
- An adversarial verifier then re-read the primary sources for each and, where possible, reproduced local measurements.
- A completeness critic commissioned gap-fill studies.

**Conventions.**
- Where a verifier marked a claim *nuanced* or *refuted*, this report uses the corrected statement.
- Where measurements disagree because they were taken on different samples, the range is given.
- *Local* numbers were measured on the real SmolLM2 tokenizer or model during the sweep and have no URL.
- *Gap-fill* results had no separate verifier and are labelled as such.

The design under test is `docs/design.md`. Each section says where the evidence supports it and where it does not.

---

## 1. Executive summary

1. **SmolLM2 tokenizes Greek almost character by character and knows almost no Greek.**
   - Only 60 of 49,152 vocabulary entries contain a Greek letter. Every capital except Δ costs two byte tokens [1].
   - Greek runs at 2.1–2.2 bytes/token and 5.4–5.8 tokens/word.
   - On parallel Belebele passages, Greek needs 4.74× more tokens than English (local, reproduced exactly) [2].
   - On FLORES Greek the base model scores 1.299 BPB, worse than a Greek word-unigram model at 1.209 (gap-fill) [3].
2. **Continued BPE with Greek-only merges appended last is the right extender.**
   - It gives 0 unreachable tokens and 2.6–2.9× Greek compression (5.6–6.2 bytes/token).
   - Text without Greek characters keeps byte-identical ids *by construction*, but only if the Greek filter tests the **concatenated** merge bytes.
   - Unconstrained, it learned 11–40 non-Greek merges that retokenized 10–30% of English documents.
   - Against a fair naive baseline its compression edge is about 5–6%. Downstream, after CPT, Purason et al. saw no substantial difference [4].
3. **Same tokens is not same text or same compute.**
   - At matched tokens Exp2 reads 2.4–2.9× more Greek bytes.
   - Its 57,152-row tied head costs +3.1% FLOPs/token at sequence length 1024 (0.913 → 0.941 GFLOP).
4. **With tied embeddings, initialization is an output-side problem.**
   - Any init inside the convex hull of the old rows gives 0% new-token argmax on English. This follows from a theorem [5].
   - Out-of-hull inits move 5–61% of English probability mass onto Greek rows at step 0.
   - FVT (mean of the original pieces) costs ≤0.006 nats/token of English and starts 0.77–0.86 BPB behind the base model on Greek.
5. **Training drift dwarfs init effects.** In a 20-step local micro-run at LR 4e-4:
   - Greek improved from 1.94 to 1.47 BPB.
   - English worsened by 0.14 nats/token, more than 10× FVT's whole step-0 cost.
   - English mass leaking onto new rows grew from 0.10% to 0.56%.
   - Init-time loss is also an unreliable way to select an init [6].
6. **Never train the old tied rows in an embedding-only or high-LR phase.** In Liquid AI's tied 8B MoE, training the full tied matrix instead of only the new rows dropped HumanEval+ from 43.9 to 1.8, while MMLU stayed flat [7].
7. **"Everyone warms up the new rows first" was refuted.**
   - Only Meltemi, Krikri, EEVE (stages 1–3) and Liquid train new rows alone first.
   - The only test below 1B parameters found freezing the body unnecessary once the init is good [8].
   - The design's single-stage default with a staged ablation is defensible.
8. **The learning-rate (LR) anchors span 5×; overtraining argues for the low end.**
   - Rules scaled to SmolLM2's 3e-3 pretraining peak give 1e-4, 3e-4 and 5e-4 [9][10].
   - SmolLM2 saw about 15.6k tokens per parameter, and overtrained models forget more at a fixed LR [11].
   - The design's 3e-4, plus a pre-registered probe on both arms, is a reasonable hedge.
9. **Replay of English is the strongest guard against forgetting, and every study counts it inside the budget.**
   - At 405M (English→German), 1% replay cut Pile loss from 3.56 to 2.83, and 25% replay to 2.33 [12].
   - Greek production recipes used 19–23% English.
   - This supports the design's 0% main pair plus a 20% companion pair.
10. **Evaluation must fix both the bytes and the context.**
    - 57–62% of Greek documents exceed one 1,024-token base window.
    - Chunks of ≤1,000 characters scored as single windows remove the confound.
    - Letter-format benchmarks are uninformative at 135M. Gold-answer BPB lifts the signal-to-noise ratio (SNR) from 10.0 to 31.5 [13].
11. **Seed noise, not eval size, limits what can be detected.**
    - A document-only bootstrap CI covered the truth only 38–79% of the time once seed noise reached 0.002 BPB.
    - With 3 seed pairs the minimum detectable effect (MDE) is about 0.006–0.007 BPB. The forecast extra English cost of Exp2 is +0.01 BPB (gap-fill).
    - The design's 0.005 BPB equivalence margin is below the achievable MDE.
12. **Colab's stock attention path is slow on a T4.**
    - Colab ships torch 2.11 [14]. On a T4, its grouped-query SDPA call falls back to the fp32 MATH kernel, about doubling activation memory (10.0 vs 5.0 GB at 4×1024).
    - 600 steps need about 12.1k tok/s on Exp2, which is feasible only with K/V pre-expansion, as the design does.
    - fp16 headroom is 3.2×, set by a measured 20.4k activation.

---

## 2. Problem framing

### 2.1 Why Greek is expensive for SmolLM2

SmolLM2 uses a GPT-2-style byte-level BPE with no normalizer, a `Digits` + `ByteLevel` pre-tokenizer, 48,900 merges, `ignore_merges=false` and no byte fallback [1]. Greek letters take two UTF-8 bytes, and the base merges fused the *space* with the lead byte (`ĠÎ`, `ĠÏ`). Word-initial Greek is therefore split mid-character.

Measured locally:
- **Vocabulary coverage.** 60 Greek-containing entries. ξ and ψ are missing. All capitals except Δ, all accented capitals, ϊ and ϋ take two tokens, and so do 13 space-prefixed lowercase letters, including the articles ` η` and ` ο`. ` της` takes three.
- **Fragmentation.** 5.4–5.8 tokens/word and 2.11–2.19 bytes/token across Wikipedia, FineWeb-2 and Belebele. 99.6–99.8% of words are split, and 13.7% of tokens are partial UTF-8 fragments.
- **English baseline.** 3.8–4.8 bytes/token, depending on the slice: digits are tokenized one per token, and digit density varies. Measure it on the actual eval set rather than hard-coding it.
- **Context.** A 1,024-token window holds about 2.2 KB of Greek under the original tokenizer and about 6.1 KB under the extended one.
- **Per-token perplexity flatters Greek.** It is 6.7 for Greek vs 12.4 for English, while bits per byte are 1.27 vs 0.87.

### 2.2 What Exp1 and Exp2 trade off

| | Exp1 (original tokenizer) | Exp2 (+8k Greek tokens) |
|---|---|---|
| Model | unchanged, 134.5M params | +4.6M params (+3.4%); tied head +16.3% |
| Greek units | bytes and characters the model already has | about 2.8× fewer tokens; 8k rows learned from a mean init |
| Step-0 Greek | base model | 0.77–0.86 BPB worse until the new rows are learned |
| Step-0 English | identical | identical ids; ≤0.006 nats of softmax competition (FVT) |
| Greek text per token budget | 1× | 2.4–2.9× |
| Inference | slow on Greek | about 2.8× cheaper per Greek byte |

Prior work says expansion pays off only after hundreds of millions to tens of billions of tokens [15][16][17]. Those models already knew the target language, though. SmolLM2's Greek is below unigram level, so the argument that the old segmentation carries knowledge is weak here (gap-fill).

### 2.3 Budget-matching subtleties

- **Bytes.** Matched tokens give Exp2 2.4–2.9× more Greek text: 2.78 on a wiki+web held-out set, 2.85 in the design's 12 MB measurement, and 2.43 for a weaker merge-append extension. 600 × 32,768 = 19.7M tokens needs about 116 MB for Exp2, hence the design's 150 MB pool.
- **The budget definition can flip the sign.** For Llama-2 Arabic, extension was about 2% better at equal tokens (loss 0.6539 vs 0.6668). The original tokenizer won at equal bytes (0.6371 vs 0.6539) [18]. The task's rule is equal tokens, and the writeup should say so.
- **FLOPs.** The tied head is 21.1% of linear FLOPs at 49,152 rows and 23.7% at 57,152, so Exp2 costs +3.1% per token (2.2–3.0% once attention is counted). A frozen-body stage A is 22.6–28.6% cheaper per token. Meter FLOPs and wall-clock rather than forcing them equal.
- **Steps.** GradScaler can skip different numbers of steps in the two arms. Log the skips per arm.
- **Repetition.** "Up to 4 epochs ≈ unique data" comes from from-scratch scaling runs [19], not short CPT. It is moot at ≤1 epoch, which the design guarantees.

---

## 3. Tokenizer extension methods

All rows are +8,000 tokens on SmolLM2 Greek, measured on held-out text. The base tokenizer gives 2.1–2.2 bytes/token.

| Method | How | Reachability / unused | English invariance | Bytes/token | Cost | Verdict |
|---|---|---|---|---|---|---|
| **Continued BPE, Greek-only, appended** [4][7] | Segment Greek pre-tokens with the original merges and keep merging. Accept only brand-new tokens whose concatenated bytes contain a Greek-only pattern. Append as ids 49152–57151 | 0 unreachable (2–3 partial-UTF-8 fragments). 0.5–7.3% unused on held-out. About 30 "scaffold" tokens never emitted; about 200 seen fewer than 10 times | Provable: original merges always rank first, and new tokens never appear in them | 5.6–6.2 (2.6–2.9×) | Seconds to 2 min CPU | **Default (design)** |
| Same, unconstrained | No script filter | same | **Broken**: whitespace, `»,`, ` Hotels`, kana merges retokenized 10–30% of English | ≈0.1% better | same | Reject |
| Naive: append a separate Greek BPE's merges | Train a Greek BPE; append its new tokens and merges | 5.3–16.7% unreachable, depending on implementation; mostly word-initial (`ĠÏ` fires first). 8–25% unused | Holds if Greek-only | 5.07–5.51 | Seconds | Ablation |
| Naive "regen" (Purason's baseline) | Regenerate merges per added token | 4.3% unreachable, 8.9% unused | Holds if Greek-only | 5.63 | Seconds | Fair control |
| Naive + `ignore_merges=true` | Look up the whole pre-token first | 0–0.2% unreachable, 13% unused | Not guaranteed (995/995 English docs identical locally) | 5.59–5.80 | One flag | Strong control |
| `add_tokens` | Added tokens matched greedily before pre-tokenization | No BPE semantics; partial-UTF-8 tokens can't be added | Holds on Greek-free text | 3.5–3.6 with whole words (spaces become tokens; `καινούργιο` → `και+ν+ού+ρ+γιο`) | Trivial | Negative control |
| SentencePiece piece merge (Chinese-LLaMA, Meltemi) | Append pieces | Score 0 means **highest** priority [20] | Reorders shared pieces (Meltemi English fertility 1.49 → 1.44) | n/a | Format conversion | Background |
| Leaf pruning / Picky BPE [21] | Learn 9–10k merges, prune leaves back to 8k | Keeps 0 unreachable; fewer junk tokens | Holds | ≈ same | Minutes | Optional |
| Fixed-size reallocation [22] | Evict low-value ids, insert Greek reachably | Guided insertion fixes the 16% naive unreachability | English changes ≤0.05% | Ukrainian fertility 2.71 → 1.80 | Low | Future arm |
| Replacement / transplant | New tokenizer plus predicted rows | n/a | Breaks English ids | n/a | Swaps need ~50B+ tokens [16] | Out of scope |
| SuperBPE [23] | Merges across whitespace | n/a | New pre-tokenizer | up to 33% fewer tokens (from scratch) | Moderate | Future |

**Corrections from the verifiers.**
- **Smaller fair gain.** The first-reported +12.8% compression gain was measured against a weak naive baseline. Against regen-naive it is about +5.6%, and +2.7–6% against naive with `ignore_merges=true`.
- **Purason's averages.** Across 70 languages: naive 3.6–10.5% unreachable vs 0.0%, and +2.7–9.6% compression. For Greek specifically only +1.3–1.9% (Llama-3) and +0.0–0.4% (Qwen-2.5). After about 7.2B CPT tokens on 1B/3B models, downstream results were not substantially different [4].
- **Filter loophole.** Checking each half for "Greek fragments" admits other scripts: lead byte 0xE1 is shared with Georgian, Hangul and Vietnamese, and a soft-hyphen merge (C2 AD) also slipped through. Russian and Georgian text then changed.
  - The design avoids this: it filters the merged token for 0xCE/0xCF or 0xE1+0xBC–BF, and its QA runs 500 strings in other scripts.
  - The verifier's pattern also allows CD B0–BF (U+0370–037F). Omitting it can only drop merges, never break invariance.
- **Toolkit.** The Apache-2.0 toolkit (pushed 2026-08-22) has no script filter for byte-level BPE. Unconstrained, it learned 29 non-Greek tokens that changed 32/200 English documents [24]. Use it as a cross-check only.
- **No knee at 8k.** Bytes/token is 4.17–4.19 (1k), 4.70–4.72 (2k), 5.30–5.33 (4k), 5.93–5.96 (8k), 6.30–6.34 (12k) and 6.54–6.60 (16k). The compression ratio grows about log-linearly, roughly +0.25× per doubling up to 32k. 8k is justified because the task fixes it. This mildly contradicts the design's "diminishing returns past ~8k".
- **Rényi efficiency** falls as tokens are added (0.635 → 0.502) and does not track quality [25]. Report it but don't select on it.

**For this task.** Keep continued BPE with the concatenated-byte filter. Register regen-naive and naive+`ignore_merges` as fair controls. Present the gain as compression and no dead rows, not as guaranteed LM quality.

---

## 4. Embedding initialization methods

SmolLM2 ties input and output embeddings, so each new row is both an input vector and an output classifier. Two theoretical results set the safety bounds:
- **Hewitt:** global-mean rows bound the KL change by log(1+k/n), here ≤0.151 nats/token [26].
- **Mundra:** a row inside the convex hull of the old rows can never be the strict argmax [5].

The step-0 numbers below come from four independent local measurements on different eval sets. Greek BPB is under the extended tokenizer; the original model scores 1.20–1.34.

| Method | Resources | Step-0, tied (Greek BPB / English cost) | Reported gains | Cost | Verdict at 135M, tied |
|---|---|---|---|---|---|
| HF `mean_resizing=True` (N(μ, 1e-9·Σ); identical rows in bf16) | none | 3.19–3.73 / ≈0 | Safe but uninformative | 0 | Control |
| Random N(0, 0.0417) | none | 3.74–4.07 / +0.32–0.60 nats, 8–14% mass | Hindi FLORES: Random 17.82 vs Mean 25.24 [15] | 0 | Negative control |
| Zero init / random at std 0.135 | none | 4.4–6.1 / +0.18 to +5.5 nats, up to 61% mass | — | 0 | Avoid |
| FVT rescaled to old-row norm | none | 3.2–4.5 / +1.0–1.5 nats, 24–35% mass | Calibrating both sides raised init loss to 17.6 [6] | 0 | **Never with tied rows** |
| **FVT: mean of original pieces** [27] | none | 2.03–2.15 / ≤0.006 nats | Used by Meltemi, Krikri, Swallow, SambaLingo, Liquid; Mean ≈ FOCUS [15] | <1 s | **Default (design)** |
| Merge-parent mean [28] | merge tree | 2.03–2.05 / ≈ FVT | Merge ≈ Mean for Greek | 0 | Equivalent ablation |
| Script-wise mean [29] | none | 1.95–1.96 / ≈0.0004–0.002 | Hindi: poor after 50 steps (4.74 vs 2.81) | 0 | Cautionary control |
| α-shrink, α·FVT + (1−α)·μ | none | α = 0.75: 1.92–1.93 / 0.0008 | Own proposal; not validated in training | 0 | Cheap ablation |
| First/last-subtoken copy [30] | none | 2.13–2.24 / 0.02–0.14 nats | EEVE (untied) | 0 | Avoid: duplicates logits |
| CW2V, VocADT [31] | training data | Inside the hull by construction | VocADT Greek MT: 0.922 vs 0.845 | GPU hours, counts against budget | Future |
| FOCUS / WECHSEL / OFA / CLP [32][8] | fastText, dictionaries, donor model | not tested | FOCUS below Mean on Greek [28] | CPU min to impossible | Not used |
| ZeTT [33] | per-model hypernetwork | n/a | Best zero-shot where one exists | Days; none for SmolLM2 | Infeasible |
| OMP / TokAlign [34][35] | donor model / GloVe | 1:1 copies duplicate logits | OMP ≈ MeanEmbed on MMLU (0.3640 vs 0.3641) | Minutes to hours | Replacement tools |
| Token Distillation / Tokens2Words / FragMend [36][37][38] | the model's own hidden states | Input-side only; needs norm/hull control with tying | +3.8 over Mean (biomedical); about +20 for non-Latin scripts (30B) | Minutes at 135M (est.), counts against budget | Best future upgrade |

**What the verifiers changed.**
- **FVT's English cost was mostly an artifact.** About 90% of the first-reported 0.0127 nats came from 34 non-Greek tokens in that prototype. One token, ` (,`, leaked 0.84% of English mass on its own. With Greek-only tokens FVT costs 0.0013–0.0058 nats.
- **The α-shrink's advantage is calibration, not ranking.** Its English edge is negligible. Its Greek edge comes entirely from old-token targets, i.e. it re-balances total new-vs-old probability mass.
- **FVT rows carry no ranking among new tokens at step 0.** Their NLL on new-token targets (9.52 nats) is worse than spreading the same mass uniformly (9.10). Step-0 BPB cannot pick an init.
- **The hull bounds the argmax, not the mass.** Copy inits are in-hull yet leaked 2.1–8.7% of English mass, and they tied the top old token at 15% of positions. Their leakage comes from duplicating *frequent* pieces (`\n`, `.`, ` π`), not from 1,721 tokens sharing ` Î`.
- **Beyond Initialization Loss** [6], in context:
  - Setting: untied Nemotron-3-Nano-30B, 25.6k Hindi tokens.
  - The ">6× fewer steps" result needs an asymmetric input/output init, which is impossible with tied rows.
  - Plain mean vs mean-of-all: init loss 7.50 vs 13.35; step-50 validation 2.81 vs 5.53.
  - Its "50-step probe" is about 0.42B tokens, more than this whole run.
  - The ranking inversions were among close configs only.
- **Token Distillation** failed on tied Llama-3.2-3B *base*: "very large norm" rows, and generation stuck on one token. Adding next-token loss fixed it [36].
- **Convergence claims, corrected.**
  - Mundra's "inits converge within <4 checkpoints" is for RoBERTa. On LLaMA2-7B, CW2V and OFA stayed better than Mean after 2.5B tokens.
  - SambaLingo shows a single checkpoint: inits converge by 10B tokens, and mean converges faster [39].

**For this task.**
- **Default.** FVT, computed in fp32, written into the split tied matrix.
  - Decompose partial-UTF-8 tokens at the byte-symbol level; decode→encode yields U+FFFD.
  - Overwrite the rows *after* `resize_token_embeddings`, which re-ties weights and fills mean rows [40].
  - Never rescale norms.
- **Ablations.** α-shrink, global mean, random and merge-parent.
- **Step-0 metrics.** Use them as a gate, not as a decision rule. Every cited convergence result is at 0.4B–10B+ tokens, so at about 20M tokens init stays first-order.

---

## 5. Training recipes and forgetting mitigation

### 5.1 Learning rate and re-warming

The released SmolLM2-135M ends at LR 0: its WSD schedule decays linearly to 0 over the last 20% [41]. It then went through an undocumented 8k long-context stage. The evidence is a "SmolLM2-135M-8k" model-card column and rope_theta 1e5 [42][43]. That stage's LR is unpublished and no optimizer state is released. Any continuation must therefore warm up from 0 with a fresh AdamW; only the peak is a choice.

| Anchor | Source | Peak LR for SmolLM2 (pretraining peak 3e-3) |
|---|---|---|
| Fine-tuning at peak/30 | Béthune (41M–1.27B) [9] | 1e-4 |
| Peak/15, shaded down for overtraining | forgetting-control gap-fill | 2e-4 |
| Re-warm to 10% of peak ("common practice"; ALLaM's 3e-5 is 10% of Llama-2's peak) | [10][44] | **3e-4 (design)** |
| Square-root batch scaling | training-recipe sweep | 5e-4 |
| 0.5–2× peak | [12][45] | 1.5e-3 to 6e-3 (meant for 50–300B tokens) |

**Evidence.**
- **Re-warming alone forgets.** Re-warming on the Pile itself raised Pile loss by about 0.1 at 3e-4 and 0.2 at 6e-4 [12].
- **Peak LR trades adaptation for forgetting** [12][45]. Gupta's "warmup length doesn't matter" was measured after 50B tokens on a weak shift. In about 600 steps the early "stability gap" is a large share of training [46].
- **Warmup length.** With β2 = 0.95, Adam's second-moment memory is about 20 steps. Warm up for at least 40 steps; the design's 30 is slightly short by this rule.
- **Overtraining.** At a fixed LR, forgetting grows with pretraining tokens [11].
- **Sailor misquote, corrected.** Sailor's 120M fit is English loss = 0.11x² − 1.04x + 5.54, with x = log(EN share) − log(LR). LR 1e-4 is where *target-language* gains flatten, not an English threshold [47].

**Verdict.**
- 3e-4 is defensible as the 10% convention. Overtraining, and a local micro-run that lost +0.14 nats of English in 20 steps at 4e-4, push toward 2e-4.
- The design's probe ({1e-4, 3e-4, 1e-3} on dev data, both arms) has a pre-registered rule: take the largest LR whose English loss rises by ≤1/3 of its Greek gain. That matches the gap-fill's proposed rule.
- Running the main pair at a second LR is the cheapest test of whether the conclusion depends on LR.

### 5.2 Schedule (WSD)

- **Cooldown matches cosine.** A constant LR plus a ≤20% cooldown matches cosine at 33M–360M and was validated at 1B/8B. A 1-sqrt cooldown slightly beats linear [48].
- **Fit.** SmolLM2 itself used a 20% linear decay. The design's WSD, with decay anchored to absolute steps, fits.
- **Comparability.** Checkpoints taken before the cooldown are not comparable to final ones.
- **Checkpoint caveat.** Pre-decay checkpoints with optimizer state exist [49][50], but they are 2k-context models with rope 1e4, i.e. a different baseline. Use them only as an extension with their own baseline row.

### 5.3 Staged warm-up: the evidence

The training-recipe sweep claimed new-rows-only warm-up is "the pattern" across Meltemi, Krikri, Tejaswi, Chinese-LLaMA, ReTok and EEVE. **The verifier refuted this.**

| Recipe | What stage 1 actually trains | Notes |
|---|---|---|
| Meltemi [51] | New embedding + LM-head rows; 2,500 × 1.5M tokens at 2.5e-4 | Full model afterwards at 2.5e-5 (10× lower) |
| Krikri [52] | Only the new rows; 5,000 × 2M tokens at 1e-3 | CPT peak 3e-4, a 3.3× ratio, not 10× |
| Liquid (tied) [7] | Only new rows, copied rows frozen | All rows trainable → HumanEval+ 43.9 → 1.8 |
| EEVE [30] | Stages 1–3 new rows; stage 4 unfreezes **all** output rows ("critical") | Untied; one LR throughout |
| Tejaswi [15] | **Full** embedding and un-embedding, 10M tokens at 1e-3 | "Small gains" |
| Chinese-LLaMA [53] | All embeddings | Two-stage dropped as "less efficient" |
| SambaLingo, Swallow [39][54] | No embedding-only stage | — |
| Sailor 7B [47] | Embedding warm-up | Did not rescue expansion |
| Yamaguchi, ~5M tokens [28] | 2-stage vs joint | No consistent gain; worst case −3 on Greek MT |
| WECHSEL, GPT-2 117M [8] | Body frozen for 10% of steps | Needed with random init, unnecessary with a good init |
| de Vries & Nissim, GPT-2 [55] | Embeddings first | Random-init joint training: perplexity 193/219 vs 42/40 staged |

**Corrected statement.**
- New-rows-only warm-up followed by full CPT is *one* common recipe.
- For tied embeddings, Liquid's ablation shows copied rows must stay frozen in any such stage.
- Below 1B parameters, the only evidence (WECHSEL) says freezing is unnecessary once the init is good.

**Design vs research.**
- The training-recipe and Greek prior-art sweeps wanted stage A as the default; both gap-fills wanted it off.
- The design follows the gap-fills (`exp2_staged` is an ablation). The small-scale evidence supports that choice.
- Caveat: stage A removes 10% of body updates, which *mechanically* reduces forgetting. Attribute any English gain from it accordingly.

### 5.4 New-row LR and the split tied embedding

- **The design's choice.** New rows get their own parameter group at 10× the body LR (3e-3, which equals SmolLM2's pretraining peak), with weight decay 0.
- **Why 10× (gap-fill, unverified).** Embedding entries have std 0.135. Over 600 steps, Adam can move a coordinate at most about 0.16 at 3e-4 but about 1.6 at 3e-3. So at 1× the new rows barely leave their FVT init.
- **Precedents.** Meltemi used 10×, Krikri 3.3×, Tejaswi about 10–300×. Bori-2, the only public SmolLM2-135M extension (+8,981 Korean tokens), used 5× but reports no results [56].
- **Risk.** Fast tied rows can raise English leakage, and rare rows get constant push-down gradients. `exp2_newlr1x` and a row-norm monitor cover this.
- **Implementation.** Adam ignores gradient scaling, and gradient-zeroing hooks still let decoupled weight decay and stale moments move "frozen" rows [57]. The design's split `[old | new]` parameter is the correct implementation. PEFT's `trainable_token_indices` is an alternative [58].

### 5.5 English replay

| Study | Setting | Finding |
|---|---|---|
| Ibrahim [12] | 405M; 300B Pile → 200B German | Pile loss (2.17 before CPT): 3.56 / 2.83 / 2.57 / 2.46 / 2.33 / 2.24 at 0 / 1 / 5 / 10 / 25 / 50% replay. German 1.11 → 1.16 at 25% |
| Zheng [59] | 1.4B; English → Chinese | English 2.40 → 3.68 without replay, "even more severe in smaller models". 5–30% replay prevents it |
| Gogoulou [60] | 126M–1.3B | 126M English 3.45 → 4.43 after Icelandic. Forgetting "does not change noticeably" with size (contradicts Zheng) |
| Béthune [9] | ≤1.27B, weak shift | About 1% injection prevents forgetting |
| Elhady [61] | 7B | English in the first 10% of steps ≈ English throughout. The damage shows in in-context learning, not target perplexity |
| Zhao [17] | LLaMA, Chinese-only CPT | English perplexity 14.69 → 198.84 after about 0.5B tokens |

- **Practice.** Real recipes use 10–50% English: Swallow 10%, Latxa about 18% (corrected from 9%) [62], Meltemi 19.3%, Krikri 22.9%, SambaLingo 25%, Typhoon 50% [63].
- **Accounting.** Every study counts replay *inside* the budget: replay tokens replace target tokens.

**Design vs research.**
- The Greek prior-art sweep called 15–25% replay in both arms a must for the main pair. The forgetting-control gap-fill recommends 0% for the headline pair plus a 20% companion pair inside the budget. The design does the latter, which is the cleaner reading of the spec.
- **Replay is not orthogonal to the tokenizer.** With tied rows, English contexts push down the new Greek logits, so replay may shrink Exp2's leakage more than Exp1's forgetting. Report the extra English cost under both replay conditions, and flag a sign flip if it happens.
- **Scale gap.** All replay evidence comes from runs of 10⁹–10¹¹ tokens. At 2×10⁷ tokens the stability gap may dominate, so replay does not replace LR control.
- **Replay source.** Take replay text from SmolLM2's own distribution [64], disjoint from every English eval set.

### 5.6 Other mitigations

- **LoRA vs full fine-tuning.** LoRA learns less and forgets less: HumanEval 0.263 for full fine-tuning vs 0.224 at rank 256 [65]. Tejaswi's "much worse" verdict on LoRA is a one-line aside. Yamaguchi's top+bottom-2-layer training ("2×2 LS") beat LoRA at about 5M tokens. Full fine-tuning is right at 135M; LoRA and 2×2 LS are plug-ins.
- **Weight interpolation.** Forgetting tracks the size of the weight change [66][67]. Interpolation frontiers cost only evaluation (the design's `vocabx interpolate`).
- **Smaller extras.**
  - An English-only re-warm control isolates optimizer disturbance from language shift.
  - A training-time weight EMA (7B evidence only) and ElChat [68] are future plug-ins.

### 5.7 fp16 on a T4

The T4 has no bf16: 65 TFLOPS fp16 tensor-core, 8.1 TFLOPS fp32, 16 GB, 70 W [69]. SmolLM2 ships in bf16.

- **Recipe.** fp32 master weights and Adam states, fp16 autocast, GradScaler (init_scale ~1024), fp32 logits and cross-entropy.
- **Two traps.** `torch.cuda.is_bf16_supported()` returns True through emulation [70], and transformers v5 loads bf16 unless told otherwise. The design handles both.
- **Overflow risk** [71].
  - The largest activation is about 20.4k, from layer 11's `down_proj` (layer 28: about 15.9k). That is 3.2× below fp16's maximum of 65,504.
  - Everything else is <2.7k.
  - fp16-autocast loss matched fp32 to three decimals.
  - The risk is growth during training. Log `amax` for those two modules plus GradScaler skips per arm, and optionally run the two modules in fp32 (+2–3% step time).
- **Correction.** Pure-bf16 "RMSNorm updates flushed to zero" is really absorption of updates smaller than |W|/128 [72]. SmolLM2's largest RMSNorm weight is 3.34, so it barely applies.

---

## 6. Evaluation methodology

### 6.1 Metric

- **Use bits per byte.** Per-token perplexity is not comparable across vocabularies. Use BPB = ΣNLL / (ln 2 · UTF-8 bytes), as Paloma, the Pile, HELM and lm-eval do [73][74].
- **Count bytes on the original NFC text.**
  - SmolLM2 silently drops 21 byte values: 0x04, 0x06, 0x13, 0x14, 0x16, 0x1D, 0xC0, 0xC1, 0xF1, 0xF2 and 0xF5–0xFF.
  - Exclude special-token strings. The prototype counted `<|endoftext|>` as 13 bytes, giving base Greek 1.266 instead of 1.260 BPB (corrected in `design.md`).
- **Greek and English BPB are not commensurate.** Greek is 1.74–1.82 bytes/character vs 1.00 for English [75][76]. Use relative changes or bits per parallel passage for exchange rates.

### 6.2 Context confound and chunk protocol

- **Size of the confound.** 57–62% of Greek Wikipedia/FineWeb-2 test documents exceed 1,024 base tokens, and 28–39% exceed 2,048. 75–79% of Greek eval tokens sit in documents over 2,048 tokens.
- **Token windows bias the comparison.** A fixed token window gives Exp2 about 2.8× the byte context.
  - lm-eval's `loglikelihood_rolling` gives only the first window the prefix; later windows condition on **one** previous token [77].
  - Windowing alone moves perplexity by about 15% (19.44 vs 16.44) [78].
- **The design's fix.**
  - Split documents into chunks of ≤1,000 characters.
  - Assert each fits in ≤1,023 *original* tokens, and score it as a single window with an `<|endoftext|>` prefix.
  - Both models then see identical bytes with identical context.
- **Never concatenate documents.** Paloma's 92.23±17.33 vs 42.57±0.29 comes from an early checkpoint; the lasting lesson is variance [73].
- **Report Exp2's long-context advantage separately.** Use a fixed-*byte* sliding window (Compute-Optimal Tokenization uses 8,192 bytes) [79] and loss-vs-offset curves.
- **Set lm-eval's `max_length` explicitly.** Otherwise it resolves to 8192 [80].

### 6.3 Split hygiene and decontamination

- **Split first.** Split documents before tokenizer training and subsampling, and learn merges only on training text.
- **Decontaminate the training side.** The design drops matching *training* documents, which is Paloma's direction. It uses a paragraph match plus MinHash, across sources, because web crawls mirror Wikipedia.
- **MinHash is probabilistic.** FineWeb's 5-gram, 14×8-band setting catches only about 56% of pairs at Jaccard 0.70, 77% at 0.75 and 98.8% at 0.85 [81]. The design's "≥0.7" threshold is therefore soft, and the paragraph matcher matters. Use a Greek-aware word splitter for the n-grams. Train–test overlap affected >4% of validation sets in standard datasets [82].
- **Normalize with NFC everywhere.** NFD splits Greek at accents (8 vs 23 tokens on one sentence with the extended tokenizer). NFKC rewrites µ and U+0384 [83]. NFC maps the Ohm sign to Ω, and 64 of 400 long English Wikipedia articles contained Greek. Count those documents separately, as the design does.

### 6.4 Eval sets

| Role | Set | Notes |
|---|---|---|
| Greek, in-domain | Wikipedia 20231101.el held-out [84] | The only Greek snapshot |
| Greek, web | FineWeb-2 ell_Grek test split, 32,936 docs [85] | The card says "should not be trained on" |
| Greek, OOD parallel | Belebele ell_Grek, 488 passages [2] | Parallel with English |
| English, seen | FineWeb-Edu | A confirmed SmolLM2 training source |
| English, post-cutoff | BBC news 2025-01…06 [86] | Should be the **primary** retention set |
| English, parallel | Belebele eng_Latn | From Wikibooks/Wikinews/Wikivoyage, so likely seen; weakens the parallel exchange unit |
| External anchor (proposed) | FLORES-200 with the Goldfish protocol [87][3] | Tokenizer-independent reference rows (gap-fill) |

FineWeb-Edu's 2025 crawls are not guaranteed clean: crawl date is not content date [88]. Eval-set noise is large. On FLORES the unpaired standard error of aggregate BPB is 0.0069 for English (1,012 sentences) and 0.0058 for Greek (gap-fill), so pair the arms on identical documents.

### 6.5 Benchmarks usable at 135M

- **Cloze format.** Use cloze with character-normalized accuracy (`acc_norm` = argmax ll/len(chars)) [89]. Also record gold-answer BPB and the log-prob margin.
- **Avoid letter formats.** lm-eval's `belebele_ell_Grek` and `global_mmlu_el` use A/B/C/D letters [90]. lighteval has a cloze Belebele, but its `LogProbTokenNorm` depends on the tokenizer [91].
- **Signal and Noise** [13], corrected: BPB lifts the 30-task SNR from 10.0 to 31.5 and decision accuracy from 77.0% to 83.7%. Its "chance at ≤1B" refers to its own small models in letter format, not to cloze. Its +2.4% gain comes from averaging *scores* of final checkpoints.
- **Greek sets.**
  - ARC-Easy: `ilsp/arc_greek`, 2,376 test items, with `question_en` [92].
  - HellaSwag: `ilsp/hellaswag_greek`, 10,024 validation items, with `*_orig` fields [93].
  - Belebele in cloze form: 900 items.
  - MultiBLiMP ell: 1,096 pairs [94].
  - Global-PIQA ell (100 items) is too small.
- **English baselines** [42]. HellaSwag 42.1, ARC 43.9, PIQA 68.4, OBQA 34.6. Winogrande 51.3, TriviaQA 4.1 and GSM8K 1.4 are at floor. Re-run the base model in your own harness.
- **Interpretation.** Greek sets are machine-translated, and the standard error is about 1–2 points. Task-selection methods: [95][96].
- **Generation probes.** Add byte-budgeted generation probes, scored for script ratio, repetition and language ID [97]. Liquid saw generation collapse while log-likelihood scores held.

### 6.6 Exp2-specific diagnostics

- **Step-0 evaluation.** Separates the cost of the vocabulary swap from training effects.
- **English leakage.** Track the probability mass on new ids at English positions, plus BPB renormalized over the original 49,152 ids. This splits softmax competition from genuine forgetting.
- **Row health.**
  - New/old row norms (old rows: mean 3.18, p5 2.34, p95 4.19), drift from init, and an exposure histogram.
  - Expected exposure: a median of about 100–115 occurrences per 3M Greek tokens, with about 175–210 rows seen fewer than 10 times.
  - Magikarp's usual reference bytes (0xF5–0xFF) do not exist in SmolLM2, so use frequency counts and drift instead [98].
- **Canonical vs marginal likelihood.**
  - BPB scores only the canonical segmentation. A freshly extended model still puts mass on old character paths, so Exp2's early BPB is pessimistic.
  - Trained models show gaps of 0.9–1.9% out of domain [99] or <0.5% (Paloma). For a fresh extension the gap is **unquantified**: that gap study never completed.
  - Add an old-path BPB and a two-path log-sum-exp bound [100][101].

### 6.7 Statistics: seeds versus bootstrap

**Seed noise is unmeasured for this setting.** No CPT paper reports training-seed variance of held-out loss.
- From-scratch priors: 124M at 0.0019 nats [102]; 150M at a median 0.0021 nats, about 0.0007–0.003 BPB, and roughly 3× larger early in training [103].
- English after Greek CPT is out of distribution and may vary more [104].
- Working prior: 0.001–0.003 BPB per arm (gap-fill).

**MDE** at 80% power, α = 0.05, 600 documents, σ_run = 0.002 BPB, common-random-number (CRN) correlation ρ = 0.5:

| Seeds per arm | Variance from | MDE (BPB) |
|---|---|---|
| 1 | external prior | 0.0057–0.0066 |
| 2 | the runs themselves (t multiplier 14.1) | 0.020 |
| 3 | the runs themselves (multiplier 5.4) | 0.0063–0.0071 |
| 2 / 3 | known σ | 0.0041–0.0052 / 0.0034–0.0047 |

- **Without pairing** (ρ = 0), MDEs are about 1.4× larger. At σ_run = 0.005, the 3-seed MDE is about 0.016.
- **More documents barely help.** Doubling to 1,200 documents shrinks the MDE by <10% once σ_run ≥ 0.002.

**CI coverage** (simulated, nominal 95%):
- document bootstrap only: 38–79%;
- Multi-Bootstrap with 2–3 seeds [105]: 68–90%;
- variance components with t(S−1): 89–95%;
- with a correct prior: 96–97%.

**Recommended procedure.**
- SE² = SE_doc² + v_seed/S, with v_seed = max(observed, pooled across configs, prior).
- Resample documents, not chunks: clustered SEs run 1.1–3.05× the naive ones [106].
- Claim "no extra cost" only if the 90% CI sits inside a pre-declared margin.

**Design vs research.**
- The design's 3 seed pairs match the minimum. Its document bootstrap needs the added seed term.
- Its 0.005 margin is below the 3-pair MDE, so expect "inconclusive below MDE" rather than "no extra cost".
- Allocate seeds adaptively. If the English contrast is ≥0.03 BPB after two pairs, spend the remaining runs on ablations.

### 6.8 Reporting the trade-off

Report all of:
- endpoint deltas at matched tokens;
- (Greek BPB, English BPB) trajectories from step 0, plotted against steps, Greek bytes and FLOPs [79];
- English cost at matched Greek BPB;
- an exchange rate, only when the Greek gain's CI excludes 0;
- interpolation frontiers.

The design does all of this. The gap-fill adds dense early logging (every 25 steps up to step 200) and a **break-even step**, where Exp2 crosses Exp1 on Greek BPB.

---

## 7. Greek-specific prior art and comparable projects

### 7.1 Greek models

| Project | Vocabulary change | Init and schedule | Data | Results |
|---|---|---|---|---|
| **Meltemi-7B** (ILSP 2024) [51] | Mistral 32,000 → 61,362 (+29,362) | Subtoken mean for embedding and LM head (the paper overrides slides that say random). Stage 1: new rows, 2,500 × 1.5M at 2.5e-4. Stage 2: full, 24,800 × 4.5M ≈ 111.6B tokens at 2.5e-5 | 54.5B-token corpus (≈2 passes); 19.3% English, 1.2% parallel | Greek fertility 6.80 → 1.52; English 1.49 → 1.44. Greek avg 33.20 → 52.89. **English 60.41 → 54.43** |
| **Llama-Krikri-8B** (ILSP 2025) [52] | +20,992, allocated by domain: general 15k, legal 4k, science 1k, literature 500, Ancient 492 | Subtoken mean. New rows only, 5,000 × 2M at 1e-3; CPT peak 3e-4 | 110B tokens, 22.9% English, curriculum, then annealing with 189M synthetic QA tokens | Greek fertility 2.73 → 1.65; English unchanged at 1.33. Greek 48.7 → 59.5. English **66.2 → 62.2 after CPT**, 67.0 only after annealing |
| GREEK-BERT [107] | 35k SentencePiece | Lowercased, accents stripped | — | Unusable for generation |

**The 2026 landscape.**
- There is no newer ILSP base LLM [108].
- 2026 work is mostly benchmarks and fine-tunes: GreekMMLU (21,805 native questions) [109] and Maistros-8B, a LoRA distillation [110].
- `apertus-8b-greek-cpt` exists, but its details are unverified [111].
- No SmolLM2-135M expansion study has been published. Bori-2 (Korean) gives a configuration but no numbers [56].

**Greek Unicode facts.**
- **What NFC does.** It maps oxia to tonos (U+1F71 → U+03AC), U+037E to `;`, and U+0387 to `·`.
- **NFKC is unsafe.** It rewrites U+0384, the ordinal mark in Α΄, and changed about 35% of el-wiki documents. U+0384 appears in 21–80% of el-wiki documents depending on the slice.
- **All-caps words** are 2–2.8× more frequent in web text than in Wikipedia. The first report said 9×; the gap comes from definition and sample.
- **Polytonic** characters are <0.1% of letters.
- **Casing.** `casefold()` turns ς into σ, so keep case.

### 7.2 Comparable projects

| Project | Change | Key numbers | Lesson |
|---|---|---|---|
| Swallow [54] | Llama-2 +11,176 (16k/32k/48k were auxiliary tokenizer sizes) | 56.2% fewer Japanese tokens. XL-Sum −5 to −15%. English −5.0/−4.8/−2.3 at 7B/13B/70B (LR-confounded) | Roughly neutral except generation |
| SambaLingo [39] | +25k | Inits converge by 10B tokens | 25% English |
| EEVE [30] | +8,960 | Phi-2 BoolQ 0.834 → 0.740; SOLAR +2.4 | Small models lose more English |
| Cerebras Arabic [18] | +32k | Equal tokens +2% better, equal bytes worse. First-step loss: random 16.31 vs mean 5.47 | The budget definition decides the sign |
| Csaki [112] | **Replaced** 5k tokens | Hungarian MC 50.8 → 54.9; BLEU 8.8 → 6.1 | Replacement, not extension |
| LLaMA-Beyond-English [17] | Chinese | At 0.5B tokens: original vocabulary 2.399 vs extended 2.279 (GPT-4-judged, after SFT) | "Might not be a suitable choice … tens of billions" |
| Sailor [47] | Mistral +15k | Thai QA still 10% worse after 20B tokens | Abandoned expansion |
| Yamaguchi [28][113] | 50–5k tokens, ~5M target tokens, 7–9B, LoRA | CPT-only beat Align in 10/20 cases (Align won 4). English −5.3 to −18.7 for Align vs ≤2 for CPT, measured on Burmese/Sinhala/Telugu, not Greek | Its perplexity "2.9 vs 5.9" compares different vocabularies and is invalid; 8k is outside its tested range |
| Tejaswi [15] | 1k/10k/50k (main runs 50k) | Hindi +10k: 46.69 vs 45.62. Recovers after ~200M tokens; Gemma-7B never does | Low priority for new tokens is fine |
| SAVA [114] | Vocabulary **substitution** | Plain CPT 67.56 vs 65.62–66.99 at 2k steps | Indirect evidence only |
| Purason, 1B [4] | Continued BPE +16k | 5.3B extended tokens ≈ 7.2B default tokens on the same text | Parity at matched *text* |

### 7.3 Small-scale analogues (gap-fill, not verified)

- **WECHSEL, GPT-2 117M** [8]. French perplexity at 25k steps: 23.47 with WECHSEL init vs 67.97 with random init and a frozen body. A good init matters more as data shrinks: 78.3 vs 281.5 at 16 MiB.
- **FOCUS, XLM-R 278M** [32]. Extension scored slightly *below* original-vocabulary training (XNLI 75.2 vs 75.5).
- **BLOOM+1, 560M** [115]. Needs about 100M tokens to adapt, so even Exp1 is low-data here.
- **Goldfish, Greek from scratch** [116][117]. FLORES second-half BPB: 0.986 (5 MB), 0.915 (10 MB, about 24.9M tokens), 0.668 (100 MB). SmolLM2 scores 1.299.

**Forecast** (extrapolated, with Δ = Exp2 − Exp1):
- **Greek BPB.** Median −0.04, 80% interval [−0.15, +0.08], P(Exp2 better) ≈ 0.6–0.65.
  - This holds only with the 10× new-row LR. At 1×, Exp2 probably trails.
  - Break-even is expected around step 100–300.
- **English BPB.** Median +0.01, 80% interval [−0.005, +0.035], i.e. 10–30% of Exp1's own forgetting (guessed at +0.03 to +0.12).

---

## 8. 2025–2026 frontier: tools and libraries

| Method / tool | What it does | Maintenance and license | Fit |
|---|---|---|---|
| **Continued BPE** (Purason et al., EACL Findings 2026, pp. 6492–6516) [4][118][24] | Learns merges on the model's own segmentation; leaf pruning; FVT resize | Apache-2.0; pushed 2026-08-22; byte-level BPE supported; **no script filter** | Default construction; use the toolkit as a cross-check |
| **Liquid In-Place Tokenizer Expansion** (Jul 2026) [7] | Same construction, then new-rows-only stage 1 and full CPT, on a tied model | Paper only | Strongest tied evidence. It **reassigns ids** (63,151 of 64,400 surface forms survive), so it is not an append. English decoding ~9% slower per character. 8.3B MoE, 1T tokens; mechanism not isolated |
| **Beyond Initialization Loss** (NVIDIA) [6] | Compares 20+ inits; shows init loss is unreliable | Paper | Supports FVT. Its best recipe needs an untied head |
| **Token Distillation** (ICLR 2026) [36] | Matches hidden states to learn new input rows | Code released | Best next init; needs norm/hull control; charge its steps to Exp2 |
| **VocADT** (ICLR 2025) [31] | Adapter mixes old rows; model frozen | GitHub | Greek benefits most; separate in/out adapters clash with tying |
| **ZeTT / tokenkit** [33][119][120] | Hypernetwork transfer; ALM distillation | tokenkit is JAX, Python ≤3.10, ZeTT via FVT only. Hypernetworks exist only for XLM-R, Mistral-7B, TinyLlama, Llama-3-8B | Not plug-and-play |
| **OMP / mergekit-tokensurgeon** [34][121] | Sparse donor-anchor transplant; also has `subword` (= FVT) and `--magikarp` options | mergekit (LGPL-3.0 per the sweep) | Needs a donor that contains our tokens. Ties MeanEmbed on MMLU. GSM8K −74 to −79% when number tokenization differs. "Small models suffer" is unsupported: Tucano 2's full swap hurt 0.6–4B alike [122] |
| **TokAlign** [35] | GloVe alignment, 1:1 copy | — | Replacement tool; duplicates logits |
| **MATT** [123] | Attention-distillation warm-up | No code linked; NeurIPS 2026 acceptance **unverified** | 0.6B model recovers 36.30 vs 43.74 in 3h38m on an H100. Too heavy |
| **Tokens2Words / FragMend** [37][38] | Initializes rows from hidden states | FragMend's Procrustes step unverified | Assumes the model already knows the words; SmolLM2 may not for Greek |
| **Writing-system-level adaptation** [22] | Fixed-size, BPE-guided insertion | Non-archival | A parameter-matched arm |
| **SuperBPE** [23]; **BLT / Bolmo** [124][125] | Superword or byte-level models | Bolmo is arXiv 2512.15586; its Nature claim is unverified | Out of scope |

**Verdict.** The 2025–2026 work converges on the design's choices: continued BPE, mean-of-pieces init, and new rows only in any embedding stage. The heavy tools target vocabulary *replacement* and do not fit a 30-minute T4 budget.

---

## 9. Engineering facts

### 9.1 SmolLM2-135M as inspected [126][1]

- **Architecture.** LlamaForCausalLM: hidden size 576, 30 layers, 9 query / 3 KV heads (head dim 64), MLP 1536, rope_theta 1e5, 8,192 positions, tied embeddings, 134,515,008 parameters (embeddings are 21%). There is no `lm_head` tensor.
- **The nanotron YAML is not the released model.** The YAML uses rope 1e4 and 2,048 positions, so load the HF config.
- **Embedding statistics.** Entry std 0.135, mean row norm 3.18, mean-vector norm 2.18 (anisotropic).
- **Tokenizer layout.** 17 special tokens, 235 byte symbols and 48,900 merges stored as `"a b"` strings. `<|endoftext|>` (id 0) is BOS, EOS and unk. There is no pad token and no automatic BOS/EOS.
- **Vocabulary size.** 57,152 = 893 × 64, so no padding is needed; padding gave nanoGPT about 25% on an A100 [127]. 57,152 < 65,536, so token ids fit uint16 memmaps.
- **Merge format.** tokenizers ≥0.20 writes merges as pairs, and a file mixing both formats fails to load [128]. Pin versions.

### 9.2 Data access

| Source | Access and license | Size | Role |
|---|---|---|---|
| Wikipedia 20231101.el [84] | Ungated; CC-BY-SA-3.0/GFDL | 226,834 articles, 637 MB | Train + held-out |
| FineWeb-2 ell_Grek [85] | Ungated; ODC-By | 47.4M docs, 222 GB; test split 32,936 docs | Train + official test |
| CulturaX el [129] | Gated form; follows mC4/OSCAR terms | 51.4M docs, 43.1B tokens | Optional |
| OSCAR-2301 [130] | Access suspended | — | Unusable |
| HPLT 2.0 [131] / MADLAD-400 [132] | Ungated; cc0 (card metadata) / ODC-By | 248 GB / 127 GB | Alternatives |
| FineWeb-Edu [88]; BBC news [86]; Belebele [2] | Ungated | — | Replay; English eval; parallel eval |

Sampled Greek text was already NFC: 0 of 700 documents changed.

### 9.3 T4 throughput: estimates, not measurements

No measured SmolLM2-135M training throughput on a T4 has been published. Anchors:
- nanoGPT-124M on one T4 with compile ran at 18.0–19.2k tok/s, with iteration time rising 6% by step 100 from throttling [133].
- Measured T4 GEMM throughput: 41.6 TFLOPS fp16, 7.2 TFLOPS fp32 [134].
- SmolLM2 needs 0.913 (Exp1) / 0.941 (Exp2) GFLOP per token.

Estimates from the compute gap-fill:

| Path | Throughput | Tokens in 27 min (V49k / V57k) | Steps of 32,768 |
|---|---|---|---|
| (A) stock eager, MATH attention | 6–8.5k tok/s | 9.7–13.8M / 9.3–13.2M | 285–420 |
| (B) + K/V pre-expansion | 9.5–14k tok/s | 15.4–22.7M / 14.8–21.8M | 450–690 |
| (C) B + `torch.compile` | 11–16k tok/s | 16.2–23.5M / 15.5–22.6M | — |

- **Fit.** 600 steps need about 12.1k tok/s on Exp2, so only path B or C fits.
- **Calibration.** Calibrate on Exp2 for ≥40 steps, take the median of the later half, and multiply by about 0.92. The repo's 20-step calibration overestimates sustained speed.
- **Memory.** About 2.2–2.5 GB of fp32 states, plus 5.0 GB (fused) or 10.0 GB (MATH) of activations at 4×1024. 4×1024 fits; 8×1024 is tight. Logits peak at about 2.3 GB.

### 9.4 SDPA and grouped-query attention (GQA) on Turing

- **Verifier correction.** "A T4 always falls back to MATH" is false in general: torch **2.14** lets the memory-efficient kernel run GQA on sm75 [135][136].
- **What Colab ships.** On 2026-10-03 the runtime had torch **2.11.0**, transformers 5.18.0, tokenizers 0.23.2 and triton 3.6.0, without flash-attn or liger [14]. Kaggle's image derives from it [137].
- **Why MATH is chosen.**
  - In torch 2.8–2.13, the memory-efficient kernel rejects unequal head counts [138], and flash/cuDNN need sm80.
  - transformers passes `enable_gqa=True` when no mask is given [139].
  - So the repo's mask-free packed call selects **MATH**: fp32 attention and a materialized score matrix.
- **Independent confirmation.** A Kaggle T4 measurement found that pre-expanded K/V select the EFFICIENT kernel, and generation ran 1.62–1.84× faster [140].
- **The design's fix.** It expands K/V on pre-Ampere GPUs and fails `calibrate` loudly on MATH, which is correct on every torch version.
- **Two more traps.**
  - bf16 inputs on sm75 always go to MATH.
  - Under `torch.compile` with torch < 2.14, the attention mask is materialized [141]. Use one compile setting for all arms.
- **Optional.** `flash-attention-turing` exists [142].

### 9.5 Compile and fused cross-entropy on sm75

- **No tensor cores in Triton.** Since Triton 3.3, `tl.dot` below sm80 runs on FMA units [143].
- **Limited compile gains.** Inductor fuses pointwise ops and leaves matmuls on cuBLAS. max-autotune needs ≥68 SMs (the T4 has 40). Compiling takes about 2 min and changes fp16 numerics. Enable it only if it gives ≥15% net gain.
- **Liger.** Its fused linear cross-entropy crashed on sm75 in 0.8.1 and was fixed in 0.8.2 [144]. It isn't needed at micro-batch 4.

### 9.6 Quotas

- **Free Colab.** Sessions last at most 12 h; limits are unpublished [145]. Users report 15–30 GPU-h/week.
- **Kaggle.** About 30 GPU-h/week, with T4×2. Its P100 is broken with current torch [146].
- **Paid.** Colab pay-as-you-go: $9.99 per 100 CU, with a T4 at 1.19–1.76 CU/h [147]. Modal: T4 $0.59/h [148].
- **The full plan** is about 12.5 T4-hours: 3–5 days on free Colab, or about 6.5 h on Kaggle T4×2 with one arm per GPU.
- **Priority cut** if quota runs short:
  - main pair → seeds → bytes-matched, random-init and the replay pair → k-sweep and staged → the rest.
- **Version drift.** Colab bumped transformers 9 times in 4 months. Pin and log versions.

### 9.7 Apple MPS

All three precisions run on MPS at 0.5–1.1k tok/s, which is enough for smoke tests only. `resize_token_embeddings` crashed on MPS (">2³² bytes"), so do the embedding surgery on CPU, as the design does.

---

## 10. Pitfalls checklist

- [ ] Per-token perplexity, token-normalized accuracy, or lighteval `LogProbTokenNorm` compared across tokenizers.
- [ ] Token-length eval windows, `loglikelihood_rolling` on long documents, truncation, or lm-eval's implicit `max_length` = 8192.
- [ ] Bytes counted from decoded tokens or special-token strings. 21 byte values are silently dropped.
- [ ] A half-wise Greek filter (lets Georgian, Hangul and soft-hyphen merges through), or unconstrained continued BPE (whitespace and Latin merges retokenize 10–30% of English).
- [ ] Merges learned on eval text, or on Wikipedia only (yields boilerplate tokens).
- [ ] NFD or NFKC anywhere, or `casefold()`.
- [ ] Assuming English documents contain no Greek.
- [ ] New merges prepended or given high priority.
- [ ] HF `mean_resizing` rows, random init, norm-rescaled FVT or copy init in a tied head.
- [ ] FVT computed via decode→encode for partial-UTF-8 tokens.
- [ ] No assertion that `lm_head.weight is embed_tokens.weight` after resizing.
- [ ] Per-row LR by gradient scaling. "Frozen" rows still decayed by AdamW.
- [ ] Old tied rows trained in any embedding-only or high-LR phase.
- [ ] LR tuned on Exp1 alone or on test data.
- [ ] Wall-clock treated as the budget. GradScaler skips ignored. Compile, precision or GPU type mixed across arms.
- [ ] Trusting `is_bf16_supported()`, or loading SmolLM2 in bf16, on a T4.
- [ ] Stock GQA SDPA on torch < 2.14. A 20-step cold calibration.
- [ ] The 20.4k activation left unmonitored under fp16.
- [ ] Replay added on top of the budget, unequal across arms, or drawn from the English eval distribution.
- [ ] Reading de Vries' random-init disaster as evidence against joint training from a good init.
- [ ] Choosing an init from step-0 metrics.
- [ ] A document-only bootstrap CI used for post-training contrasts. Chunks resampled instead of documents. Two-seed variance tested with z instead of t(1) = 12.7.
- [ ] Per-arm best-checkpoint selection on eval data.
- [ ] Letter-format benchmarks at 135M. Card numbers compared against your own harness.
- [ ] `max_new_tokens` generation budgets (use bytes). Un-cooled WSD checkpoints compared with final ones.
- [ ] Citing Yamaguchi's 2.9 vs 5.9, Meltemi's "54.5B trained", Latxa's "9% English", or Krikri's "English +0.8 from CPT". All are corrected above.

---

## 11. Ideas to plug in later, ranked by expected insight per GPU-hour

**Near-zero GPU:**
1. **Old-path vs canonical BPB** at step 0 and at every checkpoint. Shows how much of the 0.77–0.86 BPB deficit is a scoring artifact. CPU minutes.
2. **Goldfish-protocol FLORES anchor rows**: unigram, bigram, Goldfish 5 MB–1 GB and XGLM. Tokenizer-independent context. About 10 CPU-minutes per model.
3. **Interpolation frontiers and checkpoint-jitter noise proxies** for every run. Evaluation only.
4. **Tokenizer-level K sweep, leaf pruning and exposure histograms**, plus the fair naive controls. Seconds.

**Cheap runs:**
5. **Bytes-matched Exp2** (about 11 min). Separates better units from more text; Purason predicts parity.
6. **New-row LR sweep {1×, 3×, 10×}** with break-even curves. Tests the Adam-displacement argument.
7. **20% replay pair, later a 1% arm.** Does Exp2's extra English cost survive best practice?
8. **A second shared LR (2e-4 or 5e-4)** for the main pair.
9. **`exp2_staged`.** Its English effect is partly mechanical.
10. **K = 4000, then 2000.** Expected within ±0.02 BPB of 8k.
11. **α-shrink init and an English-only re-warm control.**

**Heavier:**
12. **Token Distillation with a hull constraint**, with its steps charged to the budget.
13. **Fixed-size reallocation arm** at 49,152 rows.
14. **Untied head** (about +33M params).
15. **SmolLM2-360M replication; pre-decay checkpoint branch; BPE-dropout; EMA; LoRA / 2×2 LS; OMP from a Greek donor; FragMend.**

---

## 12. Open questions that remain unverified

1. **T4 throughput.** Sustained tok/s on a Colab T4, with and without K/V expansion and compile. Only estimates exist (6–16k).
2. **fp16 headroom.** Does the 20.4k activation grow during Greek CPT, especially with new rows at 3e-3?
3. **Canonical vs marginal likelihood.** How large is the gap for a fresh extension, and how fast does it close?
4. **Seed noise.** The data-order-only seed noise of a 600-step fp16 CPT, and the CRN correlation between arms that read different text.
5. **Break-even.** Does Exp2 cross Exp1 on Greek within 600 steps? Nothing at ≤500M parameters has tested it.
6. **8k vs 4k.** Does 8k beat 4k at about 20M tokens, or do the ~200 rarely seen rows erase the gain?
7. **Where Exp2's extra English cost comes from.** Is it leakage, larger early body updates, or tied-row drift? Do replay or staging remove it?
8. **α-shrink after training.** Does its step-0 edge survive training? The init sweep was aborted because the machine was swapping.
9. **Sailor's law at this scale.** Does its LR/English-share law (fitted at 167 tokens/param) hold at about 15.6k tokens/param?
10. **SmolLM2's long-context stage.** What LR and token budget did the undocumented 8k stage use?
11. **Unverified details.** Krikri's stage-A configuration; MATT's NeurIPS acceptance; Bolmo's Nature claim; `apertus-8b-greek-cpt`; Kaggle's T4×2 quota rate; which FLORES split Goldfish used (about ±2–3% on the anchors); mergekit's license.

---

## References

1. SmolLM2-135M tokenizer.json. https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json
2. Belebele dataset. https://huggingface.co/datasets/facebook/belebele
3. Goldfish FLORES perplexities (data file). https://raw.githubusercontent.com/tylerachang/goldfish/main/data/flores_perplexities.tsv
4. Purason et al., Teaching Old Tokenizers New Words (EACL Findings 2026). https://arxiv.org/html/2512.03989v2
5. Mundra et al., Vocabulary Expansion and Initialization Approaches (CoNLL 2024). https://arxiv.org/abs/2407.05841
6. Joshi et al. (NVIDIA), Beyond Initialization Loss (2026). https://arxiv.org/abs/2608.03494
7. Liquid AI, In-Place Tokenizer Expansion for Pre-trained LLMs (2026). https://arxiv.org/abs/2607.15232
8. Minixhofer et al., WECHSEL (2022). https://arxiv.org/abs/2112.06598
9. Béthune et al., Scaling Laws for Forgetting with Pretraining Data Injection (ICML 2025). https://arxiv.org/abs/2502.06042
10. Wang et al., continual pre-training dynamics (2025). https://arxiv.org/pdf/2505.07796
11. Springer et al., Overtrained Language Models Are Harder to Fine-Tune (ICML 2025). https://arxiv.org/abs/2503.19206
12. Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train LLMs (2024). https://arxiv.org/abs/2403.08763
13. Heineman et al., Signal and Noise (NeurIPS 2025). https://arxiv.org/html/2508.13144v1
14. Colab backend-info GPU pip freeze. https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.gpu.txt
15. Tejaswi, Gupta & Choi, Exploring Design Choices for Building Language-Specific LLMs (2024). https://arxiv.org/abs/2406.14670
16. Dagan et al., Getting the most out of your tokenizer (ICML 2024). https://arxiv.org/abs/2402.01035
17. Zhao et al., LLaMA Beyond English (2024). https://arxiv.org/abs/2401.01055
18. Cerebras/G42, bilingual adaptation of Llama 2 to Arabic (2024). https://arxiv.org/html/2407.12869v2
19. Muennighoff et al., Scaling Data-Constrained Language Models (NeurIPS 2023). https://arxiv.org/abs/2305.16264
20. Chinese-LLaMA merge_tokenizers.py. https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/scripts/merge_tokenizer/merge_tokenizers.py
21. Chizhov et al., Picky BPE (2024). https://arxiv.org/pdf/2409.04599
22. Writing-System-Level Tokenizer Adaptation for Byte-Level BPE (2026). https://arxiv.org/abs/2608.00582
23. Liu et al., SuperBPE (2025). https://arxiv.org/abs/2503.13423v2
24. tokenizer-extension toolkit (Apache-2.0). https://github.com/taidopurason/tokenizer-extension
25. Cognetta et al., Two Counterexamples to Tokenization and the Noiseless Channel (2024). https://arxiv.org/abs/2402.14614
26. Hewitt, Initializing New Word Embeddings for Pretrained LMs (2021). https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
27. Gee et al., Fast Vocabulary Transfer (EMNLP Industry 2022). https://aclanthology.org/2022.emnlp-industry.41/
28. Yamaguchi, Villavicencio & Aletras, vocabulary expansion with little target data. https://arxiv.org/abs/2406.11477
29. Downey et al., Embedding structure matters (2023). https://arxiv.org/abs/2309.04679
30. EEVE (2024). https://arxiv.org/abs/2402.14714
31. Han et al., VocADT (ICLR 2025). https://arxiv.org/abs/2410.09644
32. Dobler & de Melo, FOCUS (2023). https://arxiv.org/abs/2305.14481
33. Minixhofer et al., Zero-Shot Tokenizer Transfer (NeurIPS 2024). https://arxiv.org/abs/2405.07883
34. Goddard & Fernandes Neto, OMP tokenizer transplantation (2025). https://arxiv.org/abs/2506.06607
35. Li et al., TokAlign (ACL 2025). https://arxiv.org/html/2506.03523v1
36. Dobler, Elliott & de Melo, Token Distillation (ICLR 2026). https://arxiv.org/abs/2505.20133
37. Kaplan et al., From Tokens to Words (ICLR 2025). https://arxiv.org/abs/2410.05864
38. FragMend, Defragmenting Language Models (2026). https://arxiv.org/abs/2604.16656
39. SambaLingo (2024). https://arxiv.org/html/2404.05829v2
40. transformers modeling_utils.py. https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
41. SmolLM2-135M nanotron pretraining config. https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml
42. SmolLM2-135M model card. https://huggingface.co/HuggingFaceTB/SmolLM2-135M
43. SmolLM2-135M-Instruct config (8k long-context name). https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/c369d07e8e4c5b825466d13a28b6d8800656c343/config.json
44. ALLaM (2024). https://arxiv.org/html/2407.15390
45. Gupta et al., Continual Pre-Training of LLMs: How to (re)warm your model? (2023). https://arxiv.org/abs/2308.04014
46. Guo et al., stability gap in continual pretraining (2024). https://arxiv.org/abs/2406.14833
47. Sailor (2024). https://arxiv.org/pdf/2404.03608
48. Hägele et al. (2024), constant LR with cooldown (WSD). https://arxiv.org/html/2405.18392v3
49. SmolLM2 nanotron checkpoints (pre-decay/final). https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt
50. SmolLM2-135M intermediate checkpoints. https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints
51. Meltemi, ILSP (2024). https://arxiv.org/abs/2407.20743
52. Llama-Krikri, ILSP (2025). https://arxiv.org/abs/2505.13772
53. Chinese-LLaMA (2023). https://arxiv.org/abs/2304.08177
54. Fujii et al., Swallow (2024). https://arxiv.org/abs/2404.17790
55. de Vries & Nissim, As Good as New (2021). https://arxiv.org/abs/2012.05628
56. Bori-2-135M-Base model card. https://huggingface.co/brandonbaek/Bori-2-135M-Base
57. PyTorch AdamW docs. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
58. PEFT trainable tokens docs. https://huggingface.co/docs/peft/package_reference/trainable_tokens
59. Zheng et al., EMNLP 2024 (English replay in CPT). https://aclanthology.org/2024.emnlp-main.441.pdf
60. Gogoulou et al., Continual Learning Under Language Shift (2023). https://arxiv.org/abs/2311.01200
61. Elhady, Agirre & Artetxe, critical early period of forgetting (ACL 2025). https://arxiv.org/abs/2506.00288
62. Latxa (2024). https://arxiv.org/pdf/2403.20266
63. Typhoon (2023). https://arxiv.org/html/2312.13951v1
64. SmolLM corpus. https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
65. Biderman et al., LoRA Learns Less and Forgets Less (2024). https://arxiv.org/html/2405.09673v2
66. Alexandrov et al., Branch-and-Merge (2024). https://arxiv.org/abs/2407.08699
67. WiSE-FT, robust fine-tuning by weight interpolation. https://arxiv.org/pdf/2109.01903
68. ElChat (2024). https://arxiv.org/abs/2412.11704
69. NVIDIA T4 datasheet. https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf
70. torch.cuda source (is_bf16_supported). https://docs.pytorch.org/docs/2.5/_modules/torch/cuda.html
71. Sun et al., Massive Activations in LLMs (2024). https://arxiv.org/abs/2402.17762
72. Dobler & de Melo, language adaptation on a tight budget (2024). https://arxiv.org/abs/2408.15793
73. Magnusson et al., Paloma (2023). https://arxiv.org/html/2312.10523v1
74. Biderman et al., Lessons from the Trenches (lm-eval, 2024). https://arxiv.org/abs/2405.14782
75. Yang, Wilcox & Arnett, Apples to Apples? (EMNLP 2026). https://arxiv.org/abs/2608.25089
76. Arnett, Chang & Bergen, byte premium (2024). https://arxiv.org/abs/2403.00686
77. lm-evaluation-harness utils.py (rolling windows). https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/utils.py
78. HF transformers perplexity guide. https://huggingface.co/docs/transformers/perplexity
79. Limisiewicz et al., Compute Optimal Tokenization (2026). https://arxiv.org/html/2605.01188
80. lm-evaluation-harness models/utils.py (max_length resolution). https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/models/utils.py
81. FineWeb (2024). https://arxiv.org/abs/2406.17557
82. Lee et al. (2021), train-test overlap and deduplication. https://arxiv.org/pdf/2107.06499
83. Unicode TR15 normalization forms. https://www.unicode.org/reports/tr15/
84. wikimedia/wikipedia dataset. https://huggingface.co/datasets/wikimedia/wikipedia
85. FineWeb-2 dataset. https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
86. RealTimeData/bbc_news_alltime dataset. https://huggingface.co/datasets/RealTimeData/bbc_news_alltime
87. FLORES-200. https://github.com/facebookresearch/flores/tree/main/flores200
88. FineWeb-Edu (HF API). https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu
89. lm-evaluation-harness api/task.py. https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/api/task.py
90. lm-evaluation-harness Belebele template. https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/belebele/_default_template_yaml
91. lighteval multilingual Belebele task. https://raw.githubusercontent.com/huggingface/lighteval/main/src/lighteval/tasks/multilingual/tasks/belebele.py
92. ilsp/arc_greek. https://huggingface.co/datasets/ilsp/arc_greek
93. ilsp/hellaswag_greek. https://huggingface.co/datasets/ilsp/hellaswag_greek
94. MultiBLiMP. https://huggingface.co/datasets/jumelet/multiblimp
95. Gu et al., OLMES (2024). https://arxiv.org/abs/2406.08446
96. HF FineTasks blog. https://huggingfacefw-blogpost-fine-tasks.hf.space/
97. Language Confusion Benchmark (2024). https://arxiv.org/html/2406.20052v3
98. Land & Bartolo, Fishing for Magikarp (EMNLP 2024). https://arxiv.org/abs/2405.05417
99. Cao & Rimell, marginal likelihood over tokenisations (EMNLP 2021). https://aclanthology.org/2021.emnlp-main.161/
100. Phan et al., exact byte-level probabilities (2024). https://arxiv.org/abs/2410.09303
101. Vieira et al., From LMs over tokens to LMs over characters (2024). https://arxiv.org/abs/2412.03719
102. modded-nanogpt records. https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-01-04_SoftCap
103. DataDecide perplexity results. https://huggingface.co/datasets/allenai/DataDecide-ppl-results
104. McCoy et al. (2019), seed variance in vs out of distribution. https://arxiv.org/abs/1911.02969
105. Sellam et al., The MultiBERTs (ICLR 2022). https://arxiv.org/abs/2106.16163
106. Miller (2024), clustered standard errors for LM evals. https://arxiv.org/html/2411.00640
107. GREEK-BERT. https://ar5iv.labs.arxiv.org/html/2008.12014
108. ILSP on Hugging Face. https://huggingface.co/ilsp
109. GreekMMLU (2026). https://arxiv.org/abs/2602.05150
110. IMISLab Maistros-8B-Instruct. https://huggingface.co/IMISLab/Maistros-8B-Instruct
111. glossAPI apertus-8b-greek-cpt. https://huggingface.co/glossAPI/apertus-8b-greek-cpt
112. Csaki et al., Efficiently Adapting Pretrained LMs to New Languages (2023). https://arxiv.org/abs/2311.05741
113. Yamaguchi et al., Computational Linguistics 52(1), 2026. https://aclanthology.org/2026.cl-1.9/
114. SAVA (NAACL Findings 2025). https://arxiv.org/html/2504.17025v1
115. Yong et al., BLOOM+1 (2023). https://arxiv.org/pdf/2212.09535
116. Chang et al., Goldfish (2024). https://arxiv.org/abs/2408.10441
117. Goldfish ell_grek_10mb model card. https://huggingface.co/goldfish-models/ell_grek_10mb
118. Purason et al., ACL Anthology entry. https://aclanthology.org/2026.findings-eacl.341/
119. tokenkit (Minixhofer). https://github.com/bminixhofer/tokenkit
120. zett hypernetwork repository. https://github.com/bminixhofer/zett
121. mergekit tokensurgeon docs. https://github.com/arcee-ai/mergekit/blob/main/docs/tokensurgeon.md
122. Tucano 2 (2026). https://arxiv.org/pdf/2603.03543
123. Haltiuk & Smywiński-Pohl, MATT (2025). https://arxiv.org/abs/2510.21954
124. Byte Latent Transformer (2024). https://arxiv.org/abs/2412.09871v1
125. Bolmo (2025). https://arxiv.org/abs/2512.15586
126. SmolLM2-135M config.json. https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
127. Karpathy on vocabulary padding (via Simon Willison). https://simonwillison.net/2023/Feb/4/andrej-karpathy
128. HF tokenizers v0.20.0 release notes. https://github.com/huggingface/tokenizers/releases/tag/v0.20.0
129. CulturaX dataset. https://huggingface.co/datasets/uonlp/CulturaX
130. OSCAR-2301 (HF API). https://huggingface.co/api/datasets/oscar-corpus/OSCAR-2301
131. HPLT 2.0 cleaned (HF API). https://huggingface.co/api/datasets/HPLT/HPLT2.0_cleaned
132. MADLAD-400 (HF API). https://huggingface.co/api/datasets/allenai/MADLAD-400
133. nanoGPT on an Azure T4 (16x.engineer). https://16x.engineer/2023/12/29/nanoGPT-azure-T4-ubuntu-guide.html
134. Jia et al. (2019), T4 microbenchmarks. https://arxiv.org/pdf/1903.07486
135. PyTorch 2.14 scaled_dot_product_attention docs. https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
136. PyTorch v2.14.0 sdp_utils.cpp. https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
137. Kaggle docker-python Dockerfile. https://raw.githubusercontent.com/Kaggle/docker-python/main/Dockerfile.tmpl
138. PyTorch v2.11.0 sdp_utils.cpp. https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
139. transformers v5.18.0 sdpa_attention.py. https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/src/transformers/integrations/sdpa_attention.py
140. transformers PR #45776 (Kaggle T4 measurement). https://github.com/huggingface/transformers/pull/45776#issuecomment-5230620009
141. transformers v5.18.0 masking_utils.py. https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/src/transformers/masking_utils.py
142. flash-attention-turing. https://github.com/ssiu/flash-attention-turing
143. Triton issue #5066. https://github.com/triton-lang/triton/issues/5066
144. Liger-Kernel PR #1316. https://github.com/linkedin/Liger-Kernel/pull/1316
145. Google Colab FAQ. https://research.google.com/colaboratory/faq.html
146. Kaggle docker-python issue #1546. https://github.com/Kaggle/docker-python/issues/1546
147. McCormick, Colab GPUs features and pricing. https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/
148. Modal pricing. https://modal.com/pricing
