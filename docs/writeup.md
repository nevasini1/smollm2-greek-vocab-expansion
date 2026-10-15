# Teaching SmolLM2-135M Greek: new vocabulary vs. plain fine-tuning

## Summary
**Question.** With the same training budget, is it better to add Greek tokens to the vocabulary
(Exp2) or to keep training with the original tokenizer (Exp1)?

> **Terms used below.**
> - **BPB (bits per byte)** measures how well a model predicts held-out text. Lower is better. Unlike
>   perplexity, it can be compared across tokenizers.
> - **Exp1:** fine-tune with the original 49,152-token vocabulary.
> - **Exp2:** add 8,000 Greek tokens, then fine-tune on the same budget.
> - **Exp2-early:** an extra, shorter Exp2 run (159 steps, with its own full learning-rate schedule)
>   that reads the same 31.8 MB of Greek as Exp1. It is outside the brief's matched budget and only
>   helps explain the results (`exp2_bytes_matched` in plots).

**Answer.**
- **Exp2 is clearly better at Greek but forgets more English.** Compared with Exp1, it is 0.122 BPB
  better on Greek and 0.084 BPB worse on English: it gives up 0.69 BPB of English per 1 BPB of extra
  Greek.
- **Most of the extra forgetting is because Exp2 gets further in the same budget, not because its
  trade-off is worse.** Each Exp2 token holds about 2.8× more Greek text, so in the same 453 steps it
  reads 90.5 MB of Greek against Exp1's 31.8 MB. At the *same* Greek quality, Exp2 had lost *less*
  English than Exp1 (§5).

## 1. Setup
- **Model.** HuggingFaceTB/SmolLM2-135M; the unmodified model is the baseline.
- **Same budget.** 453 steps of 32,768 tokens (14.8M tokens) each, under 30 min on a free Colab T4 (28.1
  and 29.0 min), set by timing the slower Exp2. Exp2 needs 3% more compute for its larger output layer.
- **Same data and recipe.** One 150 MB pool of cleaned Greek (half Wikipedia, half FineWeb-2), read in
  the same order, with no document seen twice. It exceeds the brief's ~100 MB so that Exp2, which reads
  2.8× more text per token (90.5 MB in total), stays well under one pass (0.60 epochs; Exp1 0.21). All
  weights trained with AdamW, peak learning rate 3e-4, warmup–stable–decay schedule. Only difference:
  Exp2's new embedding rows learn 10× faster, with no weight decay.

## 2. How the vocabulary is changed
The new tokens come from **continuing SmolLM2's own BPE**:
1. 60 MB of Greek training text is tokenized with SmolLM2's existing tokenizer.
2. New merges are learned on top. A merge is kept only if it creates a new all-Greek token, e.g. the
   piece `ικ` or the words ` και` ("and") and ` της` ("of the").
3. The 8,000 merges are appended after the original ones, as token ids 49,152–57,151.

Nothing is removed or renumbered, and the tokenizer really produces every new token (appending a
separately trained Greek tokenizer would leave 5–17% of its tokens unusable here).

- **English without Greek letters tokenizes exactly as before**, since new merges only match Greek.
  Checked on all 3,761 such English test passages and 502 random non-Greek strings.
- **Greek needs 2.9× fewer tokens:** 1.93 instead of 5.57 per word on held-out text.
- **New tokens start safely.** SmolLM2 uses one matrix to read and to predict tokens (*tied
  embeddings*), so a new row also sets how often its token is predicted. Each starts as the average of
  the original tokens it replaces. Before training, English is unchanged (+0.0002 BPB); Greek is 0.75
  BPB worse, since the new tokens are untrained, but Exp2 passes the original model by step 50.

## 3. How evaluation works
All models are scored on the same six test sets, set aside *before* the training data was built.
Training documents sharing a paragraph with a test document, or nearly duplicating one (MinHash),
were removed (325).

| test set | language | why it is included | size |
|---|---|---|---|
| Wikipedia | Greek | same domain as training | 500 docs, 2.0 MB |
| FineWeb-2 (official test split) | Greek | same domain as training | 500 docs, 1.7 MB |
| Belebele | Greek | translated passages, not from the training sources | 486 docs, 0.5 MB |
| FineWeb-Edu | English | SmolLM2's own training data | 500 docs, 1.2 MB |
| BBC News, Jan–Jun 2025 | English | written after the model was released | 500 docs, 1.4 MB |
| Belebele | English | the same passages as the Greek set | 483 docs, 0.2 MB |

- **Fair to both tokenizers.** Documents are cut into passages of at most 1,000 characters, each
  fitting one context window even with the original tokenizer, so Exp2 gets no extra context.
- **Uncertainty.** 95% intervals from a paired bootstrap over documents: evaluation noise only.
- **Learning curves** (Figs. 1–2) use separate dev sets, so their values differ somewhat from Table 1.
- **Extra checks:** leakage (§5), multiple-choice tasks (HellaSwag and ARC-Easy in both languages, PIQA)
  and short generations.

## 4. Results
*Table 1. Test-set BPB, averaged over each language's 3 sets (change from the original model). Lower is
better.*

| model | training | Greek text read | Greek BPB | English BPB |
|---|---|---|---|---|
| original SmolLM2-135M | none | – | 1.285 | 0.964 |
| **Exp1:** fine-tune | 453 steps, 13.6 PFLOPs | 31.8 MB | **0.874** (−0.412) | **1.117** (+0.152) |
| **Exp2:** +8,000 Greek tokens | 453 steps, 14.0 PFLOPs | 90.5 MB | **0.751** (−0.534) | **1.201** (+0.237) |
| Exp2-early (extra run) | 159 steps, 4.9 PFLOPs | 31.8 MB | 0.865 (−0.420) | 1.073 (+0.109) |

*Table 2. Each Exp2 run against Exp1 (BPB, 95% interval).*

| comparison | Greek gained vs Exp1 | extra English lost vs Exp1 |
|---|---|---|
| **Exp2, same budget** | **+0.122** [0.121, 0.124] | **+0.084** [0.083, 0.086] |
| Exp2-early, same Greek text | +0.009 [0.007, 0.011] | −0.044 [−0.045, −0.043] (less lost) |

**The brief's question, answered:** with Exp1 as the reference, Exp2 gives up an extra 0.084 BPB of
English for an extra 0.122 BPB of Greek. It is better on Greek and worse on English on all six test
sets. Full tables: [`results.md`](../results/t4_run/results/combined/results.md).

![learning curves](../results/t4_run/results/combined/learning_curves.png)
*Figure 1. Dev-set BPB during training. Blue = Exp1, orange = Exp2, green = Exp2-early, dashed = original
model. Greek against steps (top left), Greek text read (top right) and compute (bottom left); English
against steps (bottom right).*

<img src="../results/t4_run/results/combined/tradeoff_trajectories.png" width="440" alt="trade-off trajectories">

*Figure 2. Greek vs English dev BPB during training; lower-left is better. Paths move left from "start"
as Greek improves. Each Exp2 path's long first segment is a straight line to its first measurement.*

## 5. What the trade-off means
**Exp2 forgets more because it gets further.** Per MB of Greek read, both learn at almost the same rate
once Exp2's new tokens catch up (Fig. 1, top right). Exp2-early reads Exp1's 31.8 MB in 159 steps
instead of 453, matches Exp1's Greek (0.865 vs 0.874) with about a third of the compute, and loses 0.044
BPB *less* English. Since it also takes fewer steps, these runs cannot say whether forgetting follows
the number of updates or the amount of text.

**At equal Greek quality, Exp2 has lost less English** (Fig. 2). Below about 1.1 Greek BPB, where both
have measured points, Exp2's path is lower: at Exp1's final Greek level it had lost about 0.04 BPB less
English, and at Exp1's final English level it was about 0.04 BPB better on Greek. The 0.69 ratio looks
high only because it counts the *extra* Greek beyond Exp1, which costs more than the first gains; from
the original model the ratio is 0.37 for Exp1 and 0.44 for Exp2. (Per character rather than per byte,
the 0.69 is about 0.38, because a Greek letter takes 2 bytes.)

**What the extra English loss is made of.**
- **About a third is leakage: Exp2 predicts its new Greek tokens inside English text.** Through the tied
  embeddings, the fast-learning new rows compete with English words: by the end, 5–7% of the
  probability at English positions goes to them (0.07% before training). Excluding them recovers
  0.022–0.031 of the 0.084 BPB.
- **The rest** coincides with about 14% larger changes to the transformer layers; its cause is not
  isolated.

**Other checks: mostly consistent, with two exceptions.**
- **Generation** (16 greedy continuations per language): on English prompts both models drift into
  Greek (59% of letters for Exp2, 40% for Exp1, on average); Exp2 switches script mid-sentence in 8 of
  16 (Exp1: 2), as leakage predicts.
- **Multiple-choice tasks** (500 questions each) are mostly noise. English accuracy falls with
  forgetting (ARC-Easy: original 0.56, Exp1 0.52, Exp2 0.48). *Exception:* on Greek HellaSwag, near
  chance for all, Exp2 is slightly worse than Exp1 (0.287 vs 0.323; interval of the difference −0.070
  to −0.004), although it assigns the right answer higher probability.
- *Exception:* **blending** each run's final weights with the original's gives a second trade-off curve
  ([plot](../results/t4_run/results/interpolation_frontiers.png)). On it Exp2 wins only below about 0.90
  Greek BPB, likely because a blend of Exp2 carries half-trained new rows.

## 6. Learnings
- **How tokens are added and initialised matters.** Continuing the model's own BPE leaves no unusable
  tokens and keeps English tokenization intact; averaging old embeddings kept English unchanged at step 0.
  How *many* tokens to add was not tested.
- **Extension paid off almost immediately:** Exp2 overtook Exp1 by step 50 (1.6M tokens), far sooner
  than published work suggests, likely because SmolLM2 splits 98% of Greek words into pieces.
- **Same tokens ≠ same text.** Against steps or compute Exp2 learns Greek far faster; against Greek text
  read, both learn at about the same rate (Fig. 1).

## 7. Limitations and next steps
- **One training run per experiment** (seed repeats planned, not run), so seed noise was not measured
  here. Published figures for models this size trained from scratch are 0.001–0.003 BPB, and continued
  training may vary more. The main effects (0.044–0.122) are far larger, but the 0.009 Greek gap between
  Exp2-early and Exp1 is too small to call real: read it as a tie.
- **One learning rate**, set in advance, plus Exp2's 10× rate for new rows, which may drive leakage.
- **No English in the training mix**, since the brief says to continue training "on the Greek corpus",
  so both models forget a lot of English.
- **Next** (configs in `configs/ablations/`): new rows at the normal rate, 20% English in the mix,
  training new rows alone first, and 2k/4k/16k new tokens.

*AI tools.* Claude Code was used for the literature review ([`research_report.md`](research_report.md)),
implementation, and two adversarial code reviews that fixed 20+ bugs before any GPU time. Design
rationale: [`design.md`](design.md). Code: `src/vocabx/`; notebook: `notebooks/`.
