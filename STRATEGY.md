# Strategy — two tracks, and why

The Psi mission split cleanly into two complementary tracks. This repo is Track B.

## Track A — the custom stack (Psi, the crown jewel)

A fully custom C++/CUDA/Metal SLM stack (own autograd, own GEMM kernels, **no PyTorch**), whose north star
is **capability-per-BIT** and whose signature move is **ternary (~1.58-bit) weights on a hand-written
kernel**. Its crown jewel is the **sub-1M TinyStories record** (smallest model to clear the TinyStories
coherence bar). This is the *novelty / craft* track — nobody else hand-writes a ternary GEMM and trains a
from-scratch model to a capability bar. Compute-light, laptop-native, differentiated.

Result so far (the crown jewel campaign): a sweep of 115K–1.2M models on the custom CUDA backend, graded on
an LLM rubric. e.g. `femto` (115K) grammatical-but-drifts (6/3/3/2), `nano` (215K) close-on-grammar
(7/5/4/4); the record candidates (small 354K / mid 574K) landing the smallest-that-clears result.

## Track B — this repo (raw capability, PyTorch)

The **most powerful ~130M model of 2026**, built pragmatically in PyTorch to actually win a benchmark race.
Uses every modern lever (Nemotron-scale data, offline logit-KD from a close teacher, WSD, light post-train)
— exactly the levers the custom stack can't deliver quickly. See [PLAN.md](PLAN.md).

## Why separate — and the hybrid endgame

Forcing the 100M model onto the custom stack would cost ~8–14 person-weeks of infra (bf16/tensor-core GEMM,
multi-GPU DP, KD pipeline) *before* a competitive model trains, and fp32-no-tensor-cores means 5–15× less
throughput — forfeiting the overtraining lever that produces the win. So: **PyTorch for capability here,
custom stack for craft there.**

**The hybrid payoff:** train + win the 130M model in PyTorch, then **ternary-QAT the winner on the Psi C++
kernel** — a world-class-per-bit 130M model no framework ships. That aims the from-scratch investment at its
actual edge (inference-time capability-per-bit), which is orthogonal to the benchmark race.

## Broader research findings that shaped this (from the strategy pass)

- **MoE — no.** It trades stored *bits* for FLOPs — the inverse of capability-per-bit. Dense is optimal
  below ~500M params (Abnar et al., ICML 2025). Wrong lever for either track.
- **Reasoning — a data lever, not architecture, and NOT at 100M for multiple-choice.** Genuine reasoning
  emergence is ~1.6B; raw frontier CoT traces backfire below ~1.5B. Multiple-choice benchmarks don't benefit
  from reasoning at any scale. (A narrow *math* reasoner is the only small-scale reasoning play, and it
  trades away commonsense — declined here.)
- **Distillation = offline data/logit, no live teacher needed at train time** — fits any framework. The
  winning form here is logit-KD from a **close** ~1–2B teacher, not SFT on frontier traces.
- **"Crush SmolLM" is real on commonsense/instruction, capacity-capped on knowledge/math** — physics
  (~2 bits/param). Grade only on the winnable axes.

## Immediate next steps (see [README.md](README.md))

1. Pin the eval harness; re-run SmolLM2-135M on it.
2. No-KD ~130M baseline (~100–150B tok) → within a couple points of SmolLM2.
3. Add offline logit-KD from the ~1–2B teacher; attribute the lift (the one thing literature doesn't prove
   below 330M).
4. In parallel: secure 8× H100 (PI email for `highgpu`, or cloud) — see [COMPUTE.md](COMPUTE.md).
