---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- math
- code
- shohin
pretty_name: Shohin training data
---

# Shohin — training data

Data for **Shohin**, a ~135M-parameter **verifiable-reasoning** language model (math / code / logic,
English-only). Every corpus here is quality-controlled: verified, concise, and decontaminated against a fixed
reasoning eval suite.

## Contents

| path | what | source / license |
|---|---|---|
| `tokenizer/shohin-tok-32k.json` | 32k BPE — single-digit numbers, byte-fallback, reserved `<think>`/`<code>` tokens | ours |
| `reasoning_gym/rg_train.jsonl` | ~560k **verified** (question, answer) items across 28 families | Reasoning-Gym (Apache-2.0) |
| `reasoning_gym/rg_traces_train.jsonl` | ~100k **verified execution-trace** documents (`<think>` worked steps `</think>`) | ours (generated + verifier-checked) |
| `reasoning_gym/rg_eval.jsonl` | held-out eval — families **and** seeds never seen in train | Reasoning-Gym |
| `sft/openmath2_concise*.clean.jsonl` | concise (≤400 tok), decontaminated math solutions | OpenMathInstruct-2 (CC-BY-4.0) |
| `decontam/evalgrams.pkl` | 13-gram set over GSM8K / GSM8K-Platinum / MATH-500 / HumanEval / MBPP | derived |

## Quality controls

- **Decontamination** — every corpus is 13-gram-checked against the eval suite. OpenMathInstruct-2 measured
  ~0.7% contaminated → filtered out; Reasoning-Gym is decontaminated-by-construction (0.000%).
- **Verification** — Reasoning-Gym items and every execution trace are checked by the task verifier; only
  correct items are kept (rejection sampling).
- **Concise-CoT** — SFT solutions filtered to ≤400 tokens under the Shohin tokenizer (short reasoning beats
  long for small students).

## Not yet included

The decontaminated pretrain shards (FineMath-4+, OpenWebMath, code — zstd `uint16`) are still building and will
be added.

## Attribution & eval integrity

Built on Reasoning-Gym (open-thought) and OpenMathInstruct-2 (NVIDIA). The **raw eval test sets are
deliberately not re-hosted here** — download them from their original sources (GSM8K, MATH-500, HumanEval,
MBPP, GSM8K-Platinum) to keep evaluation clean. Only the derived decontamination gram-set is included.
