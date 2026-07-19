# R12 S7 Learned Cayley Law Compiler: Development Result

**Date:** 2026-07-19
**Decision:** `qualify_s7_learned_cayley_for_fresh_confirmation`
**Job:** `693344`, H100 `evc25`, 2m25s, exit `0:0`
**Confirmation access:** zero

## Frozen setup

- Source commit: `b9a9414`
- Board commit: `6d3fd42`
- Board seed: `4905719171551557987`
- Training seed: `1314309421681697406`
- Treatment: 218 parameters / 133,695,087 complete-system parameters
- Treatment supervision: 23 true successor cells and three zero anchors
- Structure-breaking supervision: 23 `S^2` successor cells and the same anchors
- Favorable ordinary transformer: 4,753,677 parameters, 984 train-law atomic cells
- Development: 150 held-out atomic cells and 2,048 recurrent programs over 16
  never-read laws, balanced at depths three through eight

All three arms pass their frozen training-fit gates: true and false generators
fit 23/23 successor cells plus 3/3 zero anchors; the ordinary transformer fits
984/984 atomic train cells.

## Development scores

| Arm / intervention | Held-out atomic | Exact recurrent state | Answer |
|---|---:|---:|---:|
| Host theorem/executor | n/a | 2,048/2,048 = 100.000% | 100.000% |
| **Learned Cayley treatment** | **150/150 = 100.000%** | **2,048/2,048 = 100.000%** | **100.000%** |
| Favorable ordinary transformer | 34/150 = 22.667% | 52/2,048 = 2.539% | 27.295% |
| Learned `S^2` false generator | n/a | 19/2,048 = 0.928% | 25.684% |
| Deranged two-witness cards | n/a | 27/2,048 = 1.318% | 23.584% |
| One-witness unit default | n/a | 29/2,048 = 1.416% | 25.439% |
| State reset between events | n/a | 63/2,048 = 3.076% | 29.248% |

Treatment exact state is 100% at every individual depth from three through
eight. Recoding every operation nonce leaves every predicted state bit-identical.
All 19 immutable development gates pass.

## Interpretation

S7 is a real architectural generalization result inside its stated boundary.
The generic transformer repeats S6's failure despite more parameters and 984
atomic examples. The 218-parameter treatment sees no train-law cards beyond
successor/zero anchors, yet exactly compiles 16 unseen operation laws under
fresh hidden coordinates. The wrong-cycle, card, witness, and recurrence
controls rule out identity lookup, law-ID memorization, answer recreation, and
state-independent execution.

The gain comes from forcing computation through a learned generator basis. It
does not establish unrestricted native reasoning. Cyclic topology, exact
equality, bounded nested replay, event invocation, and pop-insert remain
architectural/structural. Natural-language law grounding, arbitrary algebra,
learned active-step/halt, and open-ended planning are not tested.

## Artifact custody

- Checkpoint SHA-256:
  `c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607`
- Evaluation SHA-256:
  `e02ed2d3111f8a483a96910286dcd682f9b4ee0a867910450becd8782224688f`
- Assessment SHA-256:
  `2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4`

The checkpoint and reports are mirrored locally and on Newton. They are frozen.
No refit, threshold change, alternate board, or development rescore is allowed.
An unchanged-weight confirmation-only evaluator may read the already sealed
2,048-row confirmation board exactly once after its code and gates are committed.
