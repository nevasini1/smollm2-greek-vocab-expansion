# Vocabulary expansion on SmolLM2-135M → Greek: writeup

> Fill the `⟨…⟩` placeholders from `artifacts/results/main/results.md` after running
> `notebooks/colab_run_all.ipynb`. Everything else describes the setup as implemented.
> Full rationale: `docs/design.md`.

## 1. What was done

SmolLM2's tokenizer encodes Greek almost character by character:
- 60 of 49,152 vocabulary entries contain Greek;
- 5.7 tokens per word vs ~1.4 for English;
- 2.1 bytes per token.

I compared two adaptations under one budget of ⟨600⟩ optimizer steps × 32,768 tokens
(⟨19.7M⟩ tokens; ≤30 min on a T4). The two arms share the same document order, optimizer,
schedule and shared-parameter learning rates.

- **Exp1:** continued pretraining with the original tokenizer.
- **Exp2:** +8,000 Greek tokens, initialised and trained as described in §2.
- **Baseline:** the off-the-shelf model, evaluated on everything.
- **Exp2 at step 0:** surgery only, no training. This separates the cost of the swap from the
  effect of training.

**Data.**
- Training: 150 MB of Greek, Wikipedia and FineWeb-2 in equal parts.
- Evaluation: held-out docs drawn *before* the training pool. Training docs sharing a paragraph
  with, or MinHash-near-duplicating, any held-out doc are removed across sources.
- Text: everything is NFC-normalised once in the pipeline, with control bytes the vocab cannot
  represent stripped.

## 2. How the existing vocabulary is modified

**Continued BPE, append-only.**
1. Greek training text is segmented with SmolLM2's own pre-tokenizer and its 48,900 merges.
2. New merges are learned *on top of that segmentation*. A merge is accepted only if it creates
   a new string made of Greek characters.
3. The 8,000 merges are appended after the original ones (lowest priority), as ids
   49,152–57,151. Nothing is removed or renumbered.

**Consequences, verified by the QA harness.**
- Every original id and merge is unchanged.
- Every new token is reachable: 0 self-tokenization failures. By contrast, appending a separately
  trained Greek BPE leaves 6–17% dead tokens, because SmolLM2 already merges the space with the
  UTF-8 lead byte.
- Any text without Greek characters, including all English eval text and code, tokenizes to
  **identical ids**. So English differences between the arms come from the weights, not the
  tokenizer.

**Compression on held-out Greek.** ⟨2.12 → 5.9⟩ bytes/token and ⟨5.7 → 2.0⟩ tokens/word. Over the
1k…16k vocab-size curve, returns diminish steadily with no sharp knee; 8k is the brief's target.

**Embeddings.** SmolLM2 ties input and output embeddings, so a new row is also a new output
logit. Each new row is set to the mean of its original sub-token embeddings (FVT). This keeps it
inside the convex hull of existing rows, so it cannot dominate an English prediction at step 0.
English BPB at step 0 is ⟨0.872 vs 0.870⟩ (Greek ⟨2.03 vs 1.26⟩). Random rows (0.99) and the HF default (identical
mean rows; Greek 3.57 BPB) are worse.

**Training.** New rows are a separate tied parameter at 10× the body LR (3e-3, SmolLM2's own
pretraining peak) with no weight decay. All shared parameters follow Exp1's schedule exactly.

## 3. Evaluation design

- **Metric:** bits per byte on identical text. Per-token perplexity is not comparable across
  tokenizers.
- **Chunking:** chunks of ≤1,000 characters, each fitting one 1,024-token window of the
  *original* tokenizer, so Exp2 cannot win by seeing more text per window.
- **Greek sets:** Wikipedia (in-domain), the FineWeb-2 official test split (web), Belebele/FLORES
  (out-of-domain, parallel).
- **English sets:** FineWeb-Edu (SmolLM2's own distribution), BBC News 2025 (after SmolLM2's
  release), Belebele (parallel).
- **Statistics:** paired bootstrap over documents.
- **Secondary:** Greek and English cloze tasks built from the same items, and byte-budgeted
  generation probes.
- **Exp2 diagnostic:** probability mass on new tokens at English positions, plus English BPB
  renormalised over the original vocabulary. Together these split Exp2's English cost into
  softmax leakage and true forgetting.

## 4. Results

⟨paste the main table from results.md⟩

| model | EL BPB ↓ | ΔEL vs base | EN BPB ↓ | ΔEN vs base | Greek MB seen | PFLOPs |
|---|---|---|---|---|---|---|
| base | ⟨⟩ | – | ⟨⟩ | – | 0 | 0 |
| Exp2 @ step 0 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | 0 | 0 |
| Exp1 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨~42⟩ | ⟨⟩ |
| Exp2 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨~116⟩ | ⟨+3%⟩ |

Exp2 vs Exp1:
- extra English cost = ⟨Δ [CI]⟩ BPB
- extra Greek gain = ⟨Δ [CI]⟩ BPB
- exchange rate = ⟨⟩ bits of English per bit of Greek

## 5. Reading of the trade-off

⟨Use these questions to read the results:⟩
- Did Exp2 recover its step-0 Greek deficit within the budget? Compare the learning curves
  plotted against steps and against Greek bytes seen.
- How much of Exp2's English change is softmax leakage (`mass_new`; renormalised BPB) versus
  forgetting shared with Exp1?
- Is Exp2's Greek advantage, if any, explained by the 2.8× more Greek text it read? See the
  byte-matched ablation.
- Efficiency: ⟨2.8×⟩ fewer Greek tokens per document, which means more context and faster
  generation at inference.

## 6. Learnings

- How tokens are added matters more than how many. Continued BPE gives the most compression
  per added token, no dead rows, and a *provable* English invariance.
- With tied embeddings, initialisation is an output-calibration problem as much as an input one.
- "Same tokens" is not "same text": the comparison must be read on bytes as well as steps.
- ⟨anything the results contradict⟩
