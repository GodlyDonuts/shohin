# Data, teacher, and the reasoning-distillation recipe

> ⤴ **Superseded by [MASTER_PLAN.md](MASTER_PLAN.md) §4 (tokenizer) + §6 (data plan)** (the plan of record).
> Kept as background; the master plan adds the Reasoning-Gym procedural corpus, concrete token counts, and the
> verifier.

## Pretrain mix — reasoning-tilted (~100–200B tok, WSD)

A 130M model has no tokens to waste on trivia. The pretrain buys a **language floor + a reasoning substrate**,
tilted far harder toward math/code than a generalist would.

**Stable phase:** language floor from filtered web (DCLM-baseline / FineWeb-Edu / Nemotron-CC-HQ) blended with
a **heavy math+code diet from the start** (OpenMathReasoning problems, FineMath4+ / MegaMath, Stack-Edu). Target
mix ≈ **~45–55% web / ~25–30% code / ~20–25% math** — math and code weighted far above a generalist's ~15/7.

**Decay phase:** shift almost entirely to quality reasoning — math/code/logic + short-CoT traces + a slice of
instruction data. Linear LR decay to ~0.

## Tokenizer — compact, reasoning-tuned (a decisive early call)

**Decision: a compact ~32k BPE — English + code + math, with single-digit number tokenization — NOT the
teacher's 128–151k vocab.** *(Judgment call; veto if you disagree — it changes the arch.)*

- **Why:** our decisive lever is *trace* distillation (SFT on the teacher's generated **text**), which does
  **not** require a shared tokenizer — so we're free to drop the giant vocab. At d_model ≈ 576, a 151k tied
  embedding ≈ **87M params (~2/3 of a 130M budget)**; a 32k embedding ≈ **18M**. Every param not spent on an
  embedding table is spent on **reasoning depth**. This is a concrete edge over MobileLLM-R1, which carried the
  128k Llama vocab and paid for it in layers.
- **Single-digit numbers** materially help arithmetic (a known math-reasoning trick).
- **Trade-offs we accept:** (a) no vocab-level logit-KD — fine, logit-KD is demoted; (b) no same-tokenizer
  reference model; (c) slightly longer sequences on number-heavy text — worth it for arithmetic.

## The teacher — CLOSE and reasoning-strong (Apache/MIT)

| teacher | params | why | license | vocab |
|---|---|---|---|---|
| **Qwen3-1.7B** | 1.7B | strong math/code + thinking mode; GSM8K 75.4 / MATH 43.5 base | Apache 2.0 | 151,669 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1.5B | best long-CoT math traces (MATH ~84) — **compress before use** | MIT | ~151,936 |
| Qwen3-0.6B | 0.6B | optional cheaper trace source | Apache 2.0 | 151,669 |

A 130M student is ~7–13% of a 1–2B teacher — right at the empirical effectiveness floor (**student ≥ ~10% of
teacher**). A frontier (30B+) teacher *widens the capacity gap and helps less* — distillation scaling laws show
an over-capable teacher can *degrade* a tiny student (U-shaped regime). Because we **trace-distill**, the
tokenizer mismatch between our compact vocab and the teacher's is a non-issue.

## The decisive lever — short-CoT reasoning distillation + rejection sampling

1. **Generate** verifiable reasoning traces with the teacher over math/code/logic problem banks (NuminaMath,
   GSM8K/MATH train, code tasks, logic templates).
2. **Rejection-sample: keep only correct traces** (answer-checked / unit-tested). Cheap, huge quality lever;
   universal in MobileLLM-R1 / Phi-4-mini / OpenMathReasoning.
3. **Keep them SHORT:** enforce a tight token budget and **compress** long R1-style traces (Compress-Distill
   style) so a 130M model can actually represent them. **Measure trace token lengths locally — don't assume.**
4. **Mix short:long ≈ 4:1** ("Mix Distillation", 2502.12143) — short-heavy beats either alone.
5. **Curriculum:** easy→hard, with gradually growing trace length (MobileLLM-R1's proven move).
6. **SFT** the student on this corpus with standard next-token loss (no logit matching needed).

## RL (RLVR / GRPO) — optional polish, not the engine

Run *only* as an honest A/B on top of a strong SFT'd base, on verifiable rewards (answer-checkable math/code).
**Expect little or no gain at 130M** — Meta: SFT 74.0 vs RL 57.0 at 950M; verified RLVR floor ~0.5B; reward
sparsity worsens as params shrink (a 130M base rarely samples a correct trace to reward). If flat, drop it —
the story is distillation, not RL.

## Distillation-dataset vetting — the 10-second rubric

1. **Real?** verifiable teacher, reputable uploader (not "god-level seed" / "mythos" / synthetic).
2. **Verifiable domain?** math/code/logic with checkable answers — **not** agentic tool-use or roleplay.
3. **Short enough / compressible?** the make-or-break filter at 130M.
4. **Correct?** rejection-sample; never train on unverified traces.
5. **Clean license?** Apache/MIT/CC-BY and ToS-compatible (we ship weights).

Ready-made trace sets (OpenR1-Math-220k, OpenThoughts-114k, OpenMathReasoning, OpenCodeReasoning-2) are all
**long-CoT** — usable only **compressed + rejection-filtered**. Expect to **generate our own short-CoT traces**
as the primary source; published avg-trace-lengths are unreliable, so measure before feeding.

## Bottom line

The win is **short, correct, verifiable reasoning traces from a close ~1–2B teacher**, on a reasoning-tilted
base with a compact vocab. Data quality (short-CoT + rejection sampling) is the **dominant** lever; RL is a
maybe; logit-KD and long-CoT are out.
