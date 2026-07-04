# Targets & eval — what "best sub-200M reasoner" means, in numbers

> ⤴ **Superseded by [MASTER_PLAN.md](MASTER_PLAN.md) §0 (tiered win conditions) + §8 (eval protocol).** Kept as
> background; the master plan's T1 floor (GSM8K ≥15%) comfortably clears the ~22–30 range framed here, and
> T2/T3 add the verifier + synthetic-data levers.

**The bar:** **MobileLLM-R1-140M** (Meta, arXiv 2509.24945, Sept 2025) — the current, essentially *only*,
documented ≤200M reasoning model. Nothing verified has beaten it since. Grade on a **fixed verifiable-reasoning
suite re-run on our own harness** against MobileLLM-R1-140M. (Quoted aggregates across papers use different
protocols and are **not comparable** — always re-run.)

## Pre-registered targets — verifiable reasoning is the scoreboard

| benchmark | MobileLLM-R1-140M | our target | Δ | verdict |
|---|---:|---:|---:|---|
| **GSM8K** (8-shot / 0-shot CoT) | 16.3 | **22–30** | +6–14 | **win (headline)** |
| **MATH-500** | 4.6 | **10–15** | +5–10 | **win** (lowest base → most room) |
| **HumanEval** (pass@1) | 15.9 | **18–22** | +2–6 | win |
| **MBPP** (pass@1) | 5.4 | **10–15** | +5–10 | win |
| **logic / deduction** (BBH, ProntoQA, or Reasoning-Gym subset) | *not reported* | **establish & lead** | — | **cleanest uncontested SoTA** |
| commonsense-suite mean (HellaSwag/PIQA/ARC/WinoGrande/OBQA/CSQA) | — | **no catastrophic regression** | — | report only |
| MMLU (cloze) / TriviaQA | — | don't chase | — | **capacity-capped** |

## Winnable vs capacity-capped

**Winnable (reasoning distillation + data quality move these):** GSM8K, MATH, HumanEval/MBPP, logic/deduction.

**Capacity-capped (physics, ~2 bits/param — report, never chase):** MMLU, TriviaQA, closed-book knowledge,
anything knowledge-bound.

**No-regression (spend nothing chasing, but don't torch):** the commonsense suite. Specializing for reasoning
costs commonsense breadth (MobileLLM-R1's commonsense *trailed* SmolLM2) — that trade is accepted, but a
*catastrophic* collapse would signal we over-cooked the mix.

## The honest asterisks

- **The capacity wall is real.** 140M→600M = 16→60 GSM8K. At 130M, GSM8K in the 20s is a *good* outcome; >40
  would be paper-worthy and is a **stretch, not plan-of-record.** MATH-500 (base 4.6) and the *unreported*
  logic axis have the most headroom — the bar is lowest or absent.
- **Beating MobileLLM-R1 is achievable, not guaranteed.** Meta used long-CoT SFT and a 128k vocab; we bet on
  **short-CoT + a compact vocab** (more reasoning params). Real, *bounded* headroom — not a solved recipe.
- **No tool-use in graded runs.** A calculator/Python interpreter raises end-task accuracy but makes a
  "reasoning" claim ambiguous. Tool-use is a product feature, reported separately, never in the SoTA number.
- **Contamination discipline.** Decontaminate train against GSM8K/MATH/HumanEval/MBPP test sets and **report
  the check.** A contaminated "win" is worthless — and math/code test leakage is common.

## Eval harness (milestone 0)

Stand up a fixed harness (lm-evaluation-harness + a sandboxed code executor for HumanEval/MBPP) with: GSM8K,
MATH-500, HumanEval, MBPP, a logic/deduction set — plus MMLU + the commonsense suite for no-regression
tracking. **Re-run MobileLLM-R1-140M on it ourselves** and record the numbers; that is the scoreboard for
every subsequent run. Same few-shot settings, same scoring, same decode config, every time.
