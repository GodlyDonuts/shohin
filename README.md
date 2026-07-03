# psi-100m — the most powerful ~100M model of 2026

**Goal:** build a ~130M-param dense language model that **clearly beats SmolLM2-135M** on the axes a model
that size can actually win — commonsense, language modeling, instruction-following — using 2026 methods
(better data + distillation from a *close* teacher). Trained in **PyTorch**.

This repo is the "raw capability" track. Its sibling — the **Psi** custom C++/CUDA stack — is the
"capability-per-bit craft" track (a from-scratch, no-PyTorch, eventually-ternary SLM; the TinyStories
record is its crown jewel). **The two are deliberately separate:** PyTorch here for raw capability; the
custom stack there for novelty. The hybrid endgame: train + win here, then **ternary-QAT the winner on the
Psi kernel** as the differentiated deliverable.

---

## The one-paragraph plan

Train a **~130M dense** model (deep-and-thin, GQA, SwiGLU/RMSNorm/RoPE, tied embeddings, the *teacher's*
49k tokenizer) on **~200B tokens** with a **WSD** schedule, running **offline top-k logit distillation from
a close ~1–2B teacher** (Qwen3-1.7B / Llama-3.2-1B — **not** a frontier model) throughout, then a light SFT
(+ optional DPO). Grade only on the **winnable axes**. De-risk with a **no-KD baseline first**, then add KD
and attribute the lift.

## The honest verdict (read this first)

"A 2026 100M model crushes a 2024 100M model" is **TRUE on commonsense / instruction-following / data-
efficiency, and FALSE on knowledge / math.** That's physics, not recipe: LMs store ~2 bits/param, so ~130M
holds ~30 MB of facts. **MMLU / TriviaQA / GSM8K stay capped no matter the teacher** — report them for
no-regression, never chase them.

## Two counterintuitive findings that shape everything

1. **Use a CLOSE teacher, not a frontier one.** For logit-KD a 130M student wants a ~1–2B teacher
   (student ≥ ~10% of teacher). A 30B+ frontier teacher (GPT-5.5/Claude) *widens the capacity gap and helps
   less*. "Use the best model" is backwards here. See [DATA.md](DATA.md).
2. **No reasoning / thinking-mode.** Multiple-choice benchmarks (our whole suite) don't benefit from CoT
   reasoning at any scale — it's a format finding. Building a thinking-mode would *hurt* the graded axes.

## Docs

| file | what |
|---|---|
| [PLAN.md](PLAN.md) | the full stage-by-stage build recipe (arch, data, distillation, post-train, milestones, risks) |
| [TARGETS.md](TARGETS.md) | the exact SmolLM2-135M numbers to beat + winnable vs capacity-capped axes |
| [DATA.md](DATA.md) | data recipe, the close-teacher decision, and the distillation-dataset vetting rubric |
| [COMPUTE.md](COMPUTE.md) | compute options (Newton constraints, cloud), FLOP budget, wall-clock |
| [STRATEGY.md](STRATEGY.md) | why PyTorch here + custom-stack there; the broader Psi strategy context |
| [build-plan.html](build-plan.html) | the visual one-page version of PLAN.md |

## First moves (de-risk before the full spend)

1. **Pin an eval harness** (HellaSwag, ARC-e/c, PIQA, WinoGrande, OBQA, CSQA) and re-run SmolLM2-135M on it
   *ourselves* — that's the scoreboard.
2. **No-KD baseline:** ~130M, ~100–150B tokens, WSD, on the data mix. Land within a couple points of
   SmolLM2. Proves data + arch + infra.
3. **Add KD, attribute the lift.** Identical config + offline logit-KD from the ~1–2B teacher. This answers
   the single biggest open question (does logit-KD help below 330M?) as a cheap A/B.

## The single biggest uncertainty

**Does pretraining logit-KD actually help a 130M student?** Every clean "beat SmolLM2" in the literature is
a *larger* model; KD was only tested down to 330M, and smaller students benefit less. Our headline win bets
on KD paying off ~2.5× below the smallest tested scale. Milestone 3 exists to answer it cheaply. If the
answer is no, the honest fallback is a data-quality + post-train win of a few points — real, but not a
"crush."
