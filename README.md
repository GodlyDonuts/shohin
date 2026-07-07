# Shohin — the best sub-200M reasoning model of 2026

**Goal:** build a ~130M-param dense model that is the **best small model at verifiable reasoning** —
grade-school & competition **math, code, and logic** — decisively beating **MobileLLM-R1-140M** (the current
≤200M reasoning SoTA), using 2026 methods: **short-CoT reasoning distillation + rejection sampling** from a
close ~1–2B reasoning teacher, on a reasoning-tilted data diet. **English-only.** Trained in **PyTorch**.

> **🟢 LIVE RUN — operators start at [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md).** Pretraining is currently
> running on a single GPU (UCF Newton) under an autonomous custody loop. The runbook is the operational
> plan of record: live job state, the custody loop, the 60k feedback+extend transition, cluster access,
> and the failure/recovery playbook. Read it before touching the run.

This repo is the "raw capability" track. Its sibling — the **Psi** custom C++/CUDA stack — is the
"capability-per-bit craft" track (from-scratch, no-PyTorch, eventually-ternary; the TinyStories record is its
crown jewel). **Deliberately separate:** PyTorch here for capability, the custom stack there for novelty. The
hybrid endgame: win here, then **ternary-QAT the winner on the Psi kernel**.

> **📋 Execution plan of record: [MASTER_PLAN.md](MASTER_PLAN.md)** — the full 72h / 8×H100 build (Muon
> speedrun stack, Reasoning-Gym procedural corpus, verifier + best-of-N, tiered win conditions). This README is
> the thesis/identity front-door; where specific numbers differ (token budget, target tiers), **the master
> plan wins.**

---

## The thesis (why a 100M model should reason, not memorize)

**Knowledge is storage; reasoning is an algorithm.** LMs store ~2 bits/param (Allen-Zhu 2404.05405), so
~130M holds ~30 MB of facts — MMLU/TriviaQA are *physically* capped and we don't chase them. But **reasoning
is a learned procedure, not a lookup table.** A 130M model can't *store* the world, yet it can *execute*
multi-step arithmetic, symbolic manipulation, and deduction. **Verifiable reasoning** (checkable answers) is
the one axis where a tiny model can genuinely be pushed — so it's the axis we specialize on.

## The opening (why now)

The sub-200M reasoning niche is **essentially a one-horse race.** As of mid-2026 the only serious, documented
≤200M reasoner is **MobileLLM-R1-140M** (Meta, Sept 2025) — and nothing verified has beaten or even joined it
since. Its scores are *low in absolute terms* (GSM8K 16.3, MATH-500 4.6, HumanEval 15.9, MBPP 5.4) — it wins
by being ~9× SmolLM2, not by being good. That is the opportunity: an uncrowded lane, a beatable bar, and
obvious headroom (Meta did **not** optimize for the small-model learnability gap, short-CoT, or vocab budget).

## The one-paragraph plan

Train a **~130M dense** reasoner (deep-and-thin, GQA, SwiGLU/RMSNorm/RoPE, tied embeddings, a **compact ~32k
English+code+math tokenizer with single-digit numbers**) on a **reasoning-tilted pretrain** (web for a
language floor + heavy math/code), then the decisive phase: **short-CoT reasoning distillation** —
rejection-sampled, correct-only, short/curriculum-ordered traces from a **close ~1–2B reasoning teacher**
(Qwen3-1.7B / DeepSeek-R1-Distill-1.5B) — plus a reasoning SFT. **RL (GRPO) is optional polish, not the
engine.** Grade only on the **verifiable axes**; keep commonsense at no-catastrophic-regression. De-risk with
the single A/B that matters: **short-CoT vs long-CoT distillation at 140M.**

## Two findings that shape everything (both from grounding the plan)

1. **Short CoT > long CoT for tiny students.** The "small-model learnability gap" (2502.12143): models ≤3B
   learn *better* from short, simple reasoning than from long teacher traces; long-CoT distillation
   *underperforms*. A 130M model physically can't represent 10k-token R1 traces — so short-CoT isn't a
   compromise, it's the recommended regime. **This is our central lever.**
2. **RL is not the reasoning engine at this scale.** Meta's own 950M comparison: SFT/distillation **74.0**
   GSM8K vs RL **57.0**. The verified RLVR floor is ~0.5B, and even there SFT usually wins; a 130M base rarely
   samples a correct trace to reward. **RL demotes to an optional A/B — gains come from distillation + data.**
   *(This reverses the RL-first hunch we started with.)*

## The bar to beat (grounded, mid-2026)

| benchmark | MobileLLM-R1-140M | Shohin target | verdict |
|---|---:|---:|---|
| GSM8K | 16.3 | 22–30 | **headline win** |
| MATH-500 | 4.6 | 10–15 | win (low base → most room) |
| HumanEval (pass@1) | 15.9 | 18–22 | win |
| MBPP (pass@1) | 5.4 | 10–15 | win |
| logic / deduction (BBH / ProntoQA / Reasoning-Gym) | *unreported* | establish & lead | **cleanest uncontested SoTA** |
| commonsense suite (HellaSwag, PIQA, …) | — | no catastrophic regression | report only |
| MMLU / TriviaQA | — | don't chase | capacity-capped |

**Stretch (paper-worthy, not plan-of-record):** GSM8K >40 or MATH-500 >25 at 130M would be a genuine research
result. See [TARGETS.md](TARGETS.md).

## Docs

| file | what |
|---|---|
| **[MASTER_PLAN.md](MASTER_PLAN.md)** | **the plan of record** — 72h/8×H100 schedule, budget math, arch, tokenizer, Muon config, data mix, verifier, RLVR, eval, pre-window program, risks, release |
| [STRATEGY.md](STRATEGY.md) | two-track split; why reasoning-specialist; the pivot history — *context the master plan doesn't repeat* |
| [COMPUTE.md](COMPUTE.md) | hardware reality (the 8×H100-access blocker) + FLOP budget — *context the master plan doesn't repeat* |
| [PLAN.md](PLAN.md) · [DATA.md](DATA.md) · [TARGETS.md](TARGETS.md) | *(background, subsumed by the master plan)* the earlier grounded recipe / data / targets |
| [build-plan.html](build-plan.html) | visual one-page version (⚠ stale — regenerate against the master plan) |

## First moves (de-risk before the full spend)

1. **Pin a reasoning eval harness** (GSM8K, MATH-500, HumanEval, MBPP, a logic set; + commonsense/MMLU for
   no-regression). Re-run **MobileLLM-R1-140M ourselves** — that's the scoreboard.
2. **Reasoning-tilted base** (~130M, compact vocab, ~100–150B tok, web + heavy math/code). Prove the substrate.
3. **The decisive A/B — short-CoT vs long-CoT distillation** from the ~1–2B teacher on identical bases. Answers
   the single biggest question (does short-CoT distillation clear MobileLLM-R1 at 140M?) cheaply.

## The single biggest uncertainty

**The capacity wall.** 140M→600M is 16→60 on GSM8K — capacity is the binding constraint and we sit on the low
side. Our headroom is real but bounded: beating MobileLLM-R1 by optimizing for short-CoT + a compact vocab
(more params for reasoning, fewer for embeddings) is plausible; a *category* jump is not. If short-CoT
distillation can't clear 140M's bar, the honest fallback is "the best open, reproducible sub-200M reasoning
recipe" — still a real contribution.
