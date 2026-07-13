# Latent Reasoning and Memory Research

## Status

This document records a research program, not a capability claim. Shohin does
not yet demonstrate reliable broad reasoning, visible working, or semantic
context retention. Every proposed mechanism remains isolated from the flagship
pretraining writer until it beats a matched non-latent control on held-out
transfer and survives direct interaction.

## Literature Boundary

The first continuous-latent pilot is not claimed as a new architecture:

- [Coconut](https://arxiv.org/abs/2412.06769) feeds a final hidden state back
  as a continuous thought input. Shohin's `latent_rollout.py` is a constrained,
  answer-only control in that family.
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) passes
  dedicated memory tokens between segments.
- [Compressive Transformer](https://arxiv.org/abs/1911.05507) compresses old
  activations into a secondary memory.
- [Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841)
  studies recurrent transformer memory for very long contexts.

The research opportunity is therefore not to rename a latent token mechanism.
It is to establish a small-model, verifier-backed protocol showing whether a
fixed continuous packet can retain and update useful information after the
source text is unavailable, while preserving semantic transfer under strict
resource constraints.

## Current Evidence

The first matched 180k pilots use the same raw checkpoint, data hash,
`24,000` selected examples, `1,500` updates, seed, optimizer, batch size, and
answer-only target. The only intended difference is the progressive number of
continuous feedback tokens: `L=0` control versus `L<=4` pilot.

The original full-OOD test simultaneously changed wording, names, value range,
and depth. It is deliberately retained as a hard end gate, but cannot diagnose
which transfer failed. The factorized v2 read-only evaluation has `896` rows:

| Regime | What changes from training | Rows |
| --- | --- | ---: |
| `fit_iid` | nothing except examples | 256 |
| `depth_ood` | composition depth 5/6/8 | 192 |
| `language_ood` | domain, labels, and event wording | 256 |
| `full_ood` | language, value range, and depth | 192 |

The matched `L=0` control completed the v2 diagnostic with these exact results:

| Regime | Exact accuracy |
| --- | ---: |
| `fit_iid` | 129/256 = 50.39% |
| `depth_ood` | 27/192 = 14.06% |
| `language_ood` | 34/256 = 13.28% |
| `full_ood` | 0/192 = 0.00% |

This proves that the low answer-only loss was not a parser failure: the model
learned part of the in-template program. It also proves that this narrow
curriculum is not sufficient for semantic or compositional reasoning.

The paired continuous-feedback checkpoint was evaluated with the same v2
suite at `L=0`, `L=2`, and `L=4`:

| Model / decode steps | Fit IID | Depth OOD | Language OOD | Full OOD |
| --- | ---: | ---: | ---: | ---: |
| matched control, `L=0` | 50.39% | 14.06% | 13.28% | 0.00% |
| feedback pilot, `L=0` | 41.41% | 6.77% | 10.94% | 0.00% |
| feedback pilot, `L=2` | 44.92% | 9.38% | 11.72% | 0.00% |
| feedback pilot, `L=4` | 46.48% | 12.50% | 11.72% | 0.00% |

The feedback pilot loses to the matched control in every regime. Its original
full-OOD report is also 0/600 at every evaluated decode depth `L=0/1/2/4/8`.
The simple final-hidden-state feedback route is therefore rejected for
reasoning and context-scaling promotion. It remains a documented negative
control, not a future flagship component.

## Next Mechanism: Source-Dropping Memory Packet

The next architecture must be materially different from the current feedback
loop. The current loop retains the full source prompt at every latent step, so
it offers extra computation but **not context compression**.

The proposed isolated experiment is a **source-dropping memory packet**:

1. Split source evidence into bounded token chunks.
2. Prepend a fixed bank of `M` continuous memory slots and append `M` learned
   write slots to each source chunk.
3. Read the write-slot hidden states as the next memory packet and recursively
   carry only that packet to the next chunk.
4. After the final chunk, delete every source token. Decode the answer from
   the `M` continuous slots plus a fresh query only.
5. Train end-to-end on the answer, with no serialized `think`, `state`,
   capsule, external execution, answer injection, or selection oracle.

This is a fixed-capacity read/write recurrent memory experiment. It is not
claimed to be novel in isolation. Its project-specific contribution would be a
small-model, source-removal protocol with explicit semantic, compression, and
behavioral gates.

## Required Falsification Gates

No source-dropping packet may be called useful unless all of these are met:

1. **Fit gate:** a fresh exact-prompt-disjoint in-distribution split proves the
   model learned the task rather than an answer prior.
2. **Source-removal gate:** decoder input contains only memory slots and query;
   an assertion records that no source token IDs or KV cache are present.
3. **Memory ablation:** `M=0`, detached-state, and shuffled-state controls
   under the same update/data budget establish that the packet carries causal
   information.
4. **Length gate:** held-out chunk counts exceed training counts while source
   tokens stay unavailable after their chunk is written.
5. **Semantic gate:** labels, domains, event language, and values change
   independently. One aggregate full-OOD number is never enough.
6. **Behavior gate:** direct transcripts must show correct intermediate
   information use and final answer; a format token or final answer alone is
   insufficient.
7. **Generalization boundary:** passing a synthetic memory task is still not a
   broad reasoning promotion. Public and human-style held-out evaluations
   remain separate requirements.

## Immediate Decision Rule

- The completed feedback pilot did not materially exceed the matched control
  on any regime, so it is rejected as a useful latent-computation path.
- Only the source-dropping packet can test the constrained-context objective;
  its source-removal and memory-ablation gates are mandatory before training.

## Source-Packet Pilot Rule

The current source-packet run is deliberately a `24,000`-example, `6,000`
update M0/M1 screen, not a result claim. M0 has zero memory slots and M1 has
eight; both start from the same raw-180k checkpoint and use the same examples,
seed, optimizer schedule, and exact answer target. A packet receives more
training only if its held-out normal condition satisfies all of the following:

1. It beats **each** of M0, zeroed packet, and shuffled packet by at least 15
   percentage points on the fit-IID regime.
2. It beats those same controls by at least 5 percentage points on the combined
   length-OOD and language-OOD regimes; a fit-only result is not context scale.
3. It has a positive normal-packet margin on at least three chunk counts and
   two distinct query kinds; one favorable slice is not context scaling.
4. Read-only transcript inspection shows the source-free decoder uses the
   actual retained value rather than a fixed answer prior.

The screen is intentionally permitted to reject the answer-only packet. Its
short mechanics canary already demonstrates that low loss and formatted answers
are insufficient evidence.

## If Answer-Only Memory Fails: Certified Latent Ledger

The next bounded mechanism is a **Certified Latent Ledger (CLL)**, not a hidden
external program. The writer still receives source chunks only once and the
decoder still receives only continuous slots plus a fresh natural-language
query. The change is the training signal:

1. After each written chunk, sample several solver-verified readback queries
   about the current record (both individual values and a relation such as a
   sum or difference). The model must answer from the packet alone.
2. Reuse the *same* final packet for multiple fresh queries. This makes a
   packet useful only if it retains information rather than associating one
   query template with one answer distribution.
3. Train counterfactual source pairs that share a prefix and query but differ
   by one final event. A correct packet must separate the two consequences;
   the evaluator checks this exact pairwise distinction.
4. Keep all supervision token-level and verifier-backed. There is no controller
   that executes, repairs, selects, injects, or serializes the ledger at
   inference time.
5. Evaluate normal, zero, shuffled, slot-drop, longer-length, paraphrased, and
   counterfactual conditions separately. CLL is rejected if normal packets do
   not cause the measured advantage.

This is a stronger route to constrained-context scaling because it trains the
continuous state as an information-bearing interface throughout a sequence,
while preserving the hard source-removal boundary. It still would establish
narrow retained-information reasoning first, not broad intelligence.
