# Raw 300k Checkpoint and Interaction Result

**Status:** immutable pretraining milestone and descriptive transcript result.
This is not an SFT checkpoint, benchmark promotion, reasoning claim, or evidence
that the model has a reliable hidden-thought mechanism.

## Checkpoint custody

Two-H100 flagship job `686732` completed cleanly on `evc34` at exactly 300,000
steps after 153,869 seconds. The final log line before completion was:

```text
step 299990 loss 1.6554 gnorm 0.11 lr 0.0005 281,959 tok/s
[done] 300000 steps in 153869s
```

The log contains 134 printed skip lines. The final four consecutive gnorm skips
at steps 299546--299549 recovered, and the run subsequently completed every
remaining step. The log is 498,688 bytes with MD5
`28b9a13d596f48c39ee1139c004a17e8` and SHA-256
`f359671e256fea784c063747a9d76641384dad8762e4bfae5bf6177fa308669e`.

The trainer's terminal artifact is intentionally model-only. It contains no
optimizer state and must use a fresh-optimizer rewarmup if resumed. Newton
preserves it as both `ckpt_0300000.pt` and
`best_step300000.model.pt`; the numbered and best names are hard links to the
same read-only preserved inode, not links to the writable `ckpt_final.pt`.

| Property | Value |
|---|---|
| Step | 300,000 |
| Parameters | 125.1M trained parameters |
| Global tokens/update | 524,288 |
| Nominal tokens through 300k | 157,286,400,000 |
| Mounted manifest corpus capacity | 57,826,022,271 tokens |
| Artifact bytes | 500,448,522 |
| MD5 | `60de77c31b449060ff0417d8db16d3b0` |
| SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |

The local Mac copy at `train/flagship_out/ckpt_0300000.pt` is mode 0444 and
matches Newton by byte count, MD5, and SHA-256.

Nominal update tokens are not unique corpus tokens. The run necessarily
replayed the mounted corpus; the two quantities must not be conflated.

## Fixed direct interaction

Raw 300k was run through the existing deterministic seven-case, five-turn
`manual_capability_probe_v1` protocol on Apple MPS with `max_new=128`. The
protocol asks for an initial answer, independent review, use of a verified
intermediate fact, compact-state emission, and compact-state reuse.

Evidence bindings:

| Artifact | SHA-256 |
|---|---|
| `artifacts/eval_history/manual_capability_raw300k_20260715_mps.json` | `b9bd46937838c143355f7bedd3ea7395e3c9809c7f278f98ebf0737124bc229e` |
| exact probe source bytes used | `0bb7a41074145e0a5bd34af37402eb74459b46f5e432eef574aa0f3b44934e86` |

### Strict scores

| Checkpoint | Initial | Review | Verified fact | State reuse |
|---|---:|---:|---:|---:|
| raw 200k | 1/7 | 0/7 | 1/7 | 0/7 |
| raw 260k | 1/7 | 0/7 | 1/7 | 0/7 |
| **raw 300k** | **1/7** | **0/7** | **1/7** | **0/7** |

Raw 300k therefore shows no strict aggregate improvement on this fixed
protocol. It still cannot reliably follow answer-only formatting, correct a
prior response, emit a valid compact state, or reuse one.

### Descriptive transcript change

One fixed case improved qualitatively despite failing the strict parser. For
the sequential state update, raw 300k generated:

```text
14 + 9 = 23
23 * 3 = 69
69 - 20 = 49
49 is the integer.
```

The strict score is false because the answer protocol requires the final
integer at the start of the response and the model continued generating after
49. This differs from raw 200k, which answered 41, and raw 260k, which omitted
the multiply. However, raw 190k had already generated the same correct
23 -> 69 -> 49 trajectory before entering a loop. The mode therefore
disappeared and reappeared rather than improving monotonically. It is evidence
of one fragile visible multistep behavior on a fixed probe, not evidence of
broad or latent reasoning or a 260k-to-300k capability transition.

The remaining cases stay structurally weak:

- multiplication still asserts `29 times 16 = 496`;
- base conversion does not apply positional weights correctly;
- sort/deduplicate repeats the input and drifts into code-like boilerplate;
- string insertion degenerates into repeated formatting;
- the logic answer is correct but review/state reuse fail;
- Python generation does not produce a valid answer-only function.

No `<think>`-style hidden reasoning was recovered by this protocol. The raw
pretraining checkpoint mostly emits continuation-style explanations and
templates rather than controlled deliberation.

## Decision

1. Preserve 300k as a durable raw-pretraining anchor.
2. Do not claim that 300k is broadly smarter than 260k from this probe.
3. Preserve the sequential-state transcript as a qualitative lead for future
   fresh confirmation, not as a tuned score.
4. Do not launch the planned 600k language-balanced continuation until the 25B
   FineWeb replacement and both language-source approval records exist and
   pass their hash-bound gates.
5. Continue the R12 mechanism program independently; raw next-token pretraining
   alone has not produced reliable state transport or self-correction.
