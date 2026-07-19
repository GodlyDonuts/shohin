# R12 S6 Contextual Affine Law Induction: Development Result

**Date:** 2026-07-19  
**Decision:** `reject_s6_contextual_affine_law_development`  
**Confirmation:** forbidden; no confirmation board was generated or read

## Question

Can a small card-conditioned transformer infer an operation law absent from
training, execute it recurrently, and preserve exact state without receiving
the hidden slope or intercept?

For prime modulus `m`, each law is

```text
d(x) = a*x + b (mod m), a != 0
```

The treatment receives only `(m, 0->y0, 1->y1, x)`. Two witnesses uniquely
identify the law; one witness leaves exactly `m-1` laws possible. Training uses
atomic destination supervision on training laws only. Development uses
disjoint laws in recurrent programs of depth three through eight.

## Frozen evidence

- Source/prereg commits: `93418b0`, `c09024b`, `48e6182`
- Board seed: `4930377975126057597`
- Training seed: `412095620685111169`
- Atomic training rows: 961
- Primary development rows: 2,048
- Modulus-13 scale diagnostics: 512
- Train/development law overlap: zero
- Treatment parameters: 4,753,677
- Complete system parameters: 138,448,546
- Favorable law-ID control parameters: 4,780,301
- Primary development SHA-256:
  `8fd78f761207e8446562c75e1816d1a0821d90ecd64fcb6b837f7a92fe808047`
- CPU mechanics SHA-256:
  `a31a232c83a53d0b7aff87b4a495abd6740d98589059325951e2e4688e2bded6`

## Custody

Submission `693291` failed before Python initialization because the frozen
64-bit training seed was assigned directly to CPython's unsigned 32-bit
`PYTHONHASHSEED`. It created an empty output directory and did not initialize a
model, read the board, or produce a statistic. Launcher-only commit `676af2c`
derives the interpreter seed modulo `2^32` while retaining the full frozen seed
for model and data RNGs. The distinct `retry1` output is the only scientific
run.

Job `693293` completed once on H100 `evc25` in 3m42s with exit `0:0`.
Treatment and favorable control each fit 961/961 atomic training rows. The
evaluator read development once and confirmation zero times.

## Scores

| Arm / intervention | Exact state | Answer |
|---|---:|---:|
| Host theorem/executor | 100.000% | 100.000% |
| Treatment | **8.154%** | **30.908%** |
| Deranged two-witness card | 1.270% | 24.121% |
| One-witness ablation | 1.123% | 25.195% |
| State reset between events | 2.832% | 26.953% |
| Favorable law-ID memorizer, OOV law | 0.684% | 24.609% |
| Unseen modulus 13 diagnostic | 0.781% | 27.539% |

Held-out atomic destination accuracy is **78/318 = 24.528%** despite exact
training fit. Recurrent treatment state accuracy decays with depth:

| Depth | Exact state |
|---:|---:|
| 3 | 15.497% |
| 4 | 10.850% |
| 5 | 7.331% |
| 6 | 5.263% |
| 7 | 4.985% |
| 8 | 4.985% |

Nonce-name recoding is bit-identical, as expected because names do not enter
the law unit.

## Gate outcome

Passed:

- atomic-only training contract;
- treatment and favorable-control training fit at least 99%;
- one development access and zero confirmation access;
- parameter caps;
- nonce-name invariance.

Failed:

- held-out atomic destination at least 95%;
- exact state and answer at least 95%;
- every depth at least 92%;
- host parity;
- all required causal-drop margins;
- favorable law-ID control trailing by at least 40 points.

## Interpretation

This is not an optimization failure: both arms fit every training cell with
final losses below `2e-5`. It is an algorithmic-generalization failure. The
treatment's +6.88-point state advantage over card derangement and +7.03-point
advantage over one-witness input show that both demonstrations carry causal
signal. But a generic categorical transformer represents that signal as a
weak interpolating lookup surface rather than the identified affine law.

The failure rules out widening, extra epochs, or post-score tuning of this arm
as the next scientific move. The surviving hypothesis is architectural:
compile contextual demonstrations into a compositional group action whose
reuse is enforced by representation structure, then compare that mechanism to
this frozen transformer and matched structure-breaking controls on a wholly
fresh board.

## Artifact hashes

- Checkpoint SHA-256:
  `a440e8677f2006235b76f7fa50dcdfb3541e9667c5f609d6d319062bd85af6d6`
- Evaluation SHA-256:
  `1cfd88a86bd8ad2de2c263af29b989f61e787dfc44db60f929040d6d7a87aa5b`
- Assessment SHA-256:
  `e9f0f6a1354fd8a8bf950d814757f737775bce115ebe984af5e70ddcd0ad718c`

The checkpoint and reports are mirrored locally under
`train/s6_contextual_affine_law_4930377975126057597_412095620685111169_retry1/`
and at the matching Newton path. The model checkpoint is not a promoted
reasoning artifact.
