# Strategy — two tracks, and why

The Psi mission split cleanly into two complementary tracks. This repo is Track B.

## Track A — the custom stack (Psi, the crown jewel)

A fully custom C++/CUDA/Metal SLM stack (own autograd, own GEMM kernels, **no PyTorch**), whose north star is
**capability-per-BIT** and whose signature move is **ternary (~1.58-bit) weights on a hand-written kernel**.
Its crown jewel is the **sub-1M TinyStories record** (smallest model to clear the TinyStories coherence bar).
This is the *novelty / craft* track — nobody else hand-writes a ternary GEMM and trains a from-scratch model to
a capability bar. Compute-light, laptop-native, differentiated.

Result so far (the crown jewel campaign): a sweep of 115K–1.2M models on the custom CUDA backend, graded on an
LLM rubric — e.g. `femto` (115K) grammatical-but-drifts, `nano` (215K) close-on-grammar; the record candidates
(small 354K / mid 574K) landing the smallest-that-clears result.

## Track B — this repo (Shohin, the reasoning specialist, PyTorch)

The **best sub-200M reasoning model of 2026** — verifiable reasoning (**math / code / logic**), English-only —
built pragmatically in PyTorch to win an uncrowded benchmark race. Uses the levers the custom stack can't
deliver quickly: **short-CoT reasoning distillation + rejection sampling** from a close ~1–2B teacher, a
reasoning-tilted data diet, bf16 / multi-GPU throughput. See [PLAN.md](PLAN.md).

## Why separate — and the hybrid endgame

Forcing the model onto the custom stack would cost ~8–14 person-weeks of infra (bf16/tensor-core GEMM,
multi-GPU DP, trace-generation pipeline) *before* a competitive model trains, and fp32-no-tensor-cores means
5–15× less throughput. So: **PyTorch for capability here, custom stack for craft there.** The hybrid payoff:
train + win the reasoner in PyTorch, then **ternary-QAT the winner on the Psi C++ kernel** — a world-class-per-
bit small reasoner no framework ships. That aims the from-scratch investment at its actual edge
(inference-time capability-per-bit), orthogonal to the benchmark race.

## The pivot on this repo (recorded honestly)

Shohin started as a **commonsense generalist** (beat SmolLM2-135M on HellaSwag/PIQA/etc.). We reversed it to a
**verifiable-reasoning specialist** after two realizations:

1. **Reasoning is the right axis for a tiny model.** Knowledge is storage-capped (~2 bits/param); *reasoning is
   an algorithm* and far less capacity-bound. Specializing where a 100M model can actually improve beats
   spreading thin across axes physics has already capped.
2. **The niche is open.** Sub-200M reasoning is a one-horse race (MobileLLM-R1-140M); sub-200M commonsense is a
   crowd.

The **cost** of the pivot, stated plainly: we trade commonsense breadth for reasoning depth (MobileLLM-R1's
commonsense *trailed* SmolLM2). Commonsense drops to "no catastrophic regression." We don't get both at 130M.

## Broader research findings that shaped this (from grounding the plan, mid-2026)

- **Short CoT > long CoT for tiny students** (learnability gap, 2502.12143) — the central lever; a 130M model
  can't represent long R1 traces anyway, so short-CoT is the recommended regime, not a compromise.
- **RL is *not* the reasoning engine ≤200M** — SFT/distillation beats RL at 950M (74.0 vs 57.0); verified RLVR
  floor ~0.5B. RL is optional polish. *(Reverses our initial RL-first hunch.)*
- **Compact vocab > teacher's big vocab** — because we trace-distill (no shared tokenizer needed), a ~32k
  English+math+code vocab frees ~60M params for reasoning instead of an embedding table. A concrete edge over
  MobileLLM-R1's 128k vocab.
- **MoE — no** (trades stored bits for FLOPs; dense optimal <500M). **Knowledge — no** (physics). **Tool-use —
  a product feature, not a reasoning claim** (never in the graded number).
- **Looped / latent reasoning** — the most interesting novelty bet (depth without params), unproven ≤200M; a
  stretch prototype, not plan-of-record.

## Immediate next steps (see [README.md](README.md))

1. Pin the **reasoning** eval harness (GSM8K, MATH-500, HumanEval, MBPP, a logic set); re-run MobileLLM-R1-140M.
2. Reasoning-tilted ~130M base (~100–150B tok, compact vocab) → a good reasoning substrate + working infra.
3. The decisive A/B: **short-CoT vs long-CoT distillation** from the ~1–2B teacher; attribute the lift.
4. In parallel: secure 8× H100 (PI email for `highgpu`, or cloud) — see [COMPUTE.md](COMPUTE.md); prototype
   looped-reasoning (stretch).
