# Data, teacher, and the distillation-dataset vetting

## Pretraining data mix (~200B tokens, WSD)

**Stable phase (~160B tok):** anchor on **Nemotron-CC-HQ** (empirically #1 dataset at 130M scale, +5.6 MMLU
vs DCLM) + **DCLM-baseline** (commonsense) + **FineWeb-Edu** (knowledge). Mix ≈ **78% web / 15% code
(Stack-Edu) / 7% math (FineMath4+, MegaMath)**.

**Decay phase (~40B tok):** shift to quality — ≈ **60% web / 22% code / 15% math (+OpenMathReasoning) / 3%
Cosmopedia-v2** + a slice of instruction data.

| dataset | role | license notes |
|---|---|---|
| **Nemotron-CC / -CC-HQ** | primary web anchor | NVIDIA Open Data License (not CC-BY); fine to train on, track for weight release |
| **DCLM-baseline** | commonsense web | permissive |
| **FineWeb-Edu** | educational/knowledge web | ODC-BY |
| **Stack-Edu** | code | permissive (check subsets) |
| **FineMath / MegaMath / OpenMathReasoning** | math | permissive; verify per-set |
| **Cosmopedia-v2** | synthetic textbooks (decay) | permissive |
| **SmolTalk** (+ filtered) | SFT | permissive |

## The teacher — CLOSE, not frontier (the key decision)

For **offline top-k logit KD**, the optimal teacher is **Qwen3-1.7B or Llama-3.2-1B** — a strong but *close*
model, **not** a 30B+ frontier model.

- A 130M student is ~7–13% of a 1–2B teacher — right at the empirical effectiveness floor (**student should
  be ≥ ~10% of the teacher**). A frontier teacher widens the capacity gap and **helps less**.
- The teacher must share the student's **tokenizer** (pick the teacher's 49k BPE first) so full soft-label
  KD works without a lossy vocab-projection hack.
- Method: precompute the converged teacher's **top-k logits** (top-0.95, k≈50) over the corpus once; train
  the student with **KL loss, α≈0.9**, under WSD.

**This directly refutes the "use the best models" instinct** — for a 100M student, GPT-5.5/Claude-scale
teachers are the *wrong* choice.

## Distillation-dataset vetting (why the "frontier trace" datasets are mostly wrong)

We surveyed ~14 HuggingFace "frontier distillation" datasets (GPT-5.5 / Fable-5 / GLM / Claude traces). Two
problems recur, and both matter more than the model name in the title:

1. **Trace length backfires at 100M.** Even the *legit* reasoning sets have enormous traces — e.g.
   `Jackrong/GLM-5.1-Reasoning-1M` averages 4.5k tokens (math subset **28k**, P95 64k). A 130M model
   physically can't represent these; feeding them *degrades* it (capacity-gap backfire). Usable only if
   heavily filtered to short, clean traces.
2. **Provenance / fit.** Most are (a) **agentic tool-use** traces (`Fable-5-traces`: 81% Bash/Edit calls;
   `GLM-5.2-Agent`) — off-target, a 130M model can't do agentic tool-use; or (b) **fake/synthetic**
   (`claude_mythos_distilled_25k` self-admits "Not real Claude outputs"; `GPT-5.5-Thinking-Max` frames
   itself as "god-level recursive seed AI") — unusable; or (c) **unverified scraped mixes** (the
   `Sonnet-Opus-...-mega` aggregate — "raw API scraping," unverifiable).

**The one worth filtering:** `clzoro/Claude-Distills` (real Claude Sonnet/Opus 4.6, reasoning+instruction,
deduped, MIT) — usable *as a SFT-stage ingredient after filtering to short traces*, with an Anthropic-ToS
caveat since we ship weights.

### The 10-second rubric for any distillation dataset

1. **Real?** verifiable teacher + reputable uploader (not "god-level seed" / "mythos" / synthetic).
2. **Fit?** reasoning/instruction — **not** agentic tool-use.
3. **Short enough?** traces a 130M can represent (the make-or-break filter).
4. **Clean license?** Apache/MIT/CC-BY and ToS-compatible (we ship weights).
5. **Relevant domain?** general / math reasoning, not DevOps / roleplay.

Most community "frontier distill" datasets fail #1, #2, or #3.

## Bottom line

The decisive distillation lever is **offline logit-KD from a close ~1–2B teacher during pretraining**, not
SFT on frontier traces. Frontier-trace SFT is at best a **secondary, heavily-filtered** post-train ingredient
(`Claude-Distills`, short-filtered). **Data quality (Nemotron-CC-HQ + the decay-phase mix) is a co-equal
lever** — possibly the dominant one at 130M, where KD's marginal benefit is unproven.
