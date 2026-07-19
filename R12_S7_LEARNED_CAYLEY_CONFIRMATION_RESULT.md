# R12 S7 Learned Cayley Law Compiler: Confirmation Result

**Date:** 2026-07-19
**Decision:** `confirm_s7_learned_cayley_contextual_law_compilation`
**Job:** `693346`, H100 `evc25`, 15s, exit `0:0`
**Accesses:** one development / one confirmation; both now closed

## Frozen confirmation

- Source commit: `b9a9414`
- Board commit: `6d3fd42`
- Confirmation-code commit: `f2e1527`
- Board seed: `4905719171551557987`
- Training seed: `1314309421681697406`
- Frozen checkpoint SHA-256:
  `c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607`
- Sealed confirmation SHA-256:
  `c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5`
- Confirmation: 2,048 recurrent programs over 18 laws disjoint from all
  training and development laws, balanced at depths three through eight
- Treatment: 218 learned parameters; 133,695,087 parameters in the complete
  promoted system

No weight, mechanism, threshold, board, or evaluation change occurred between
development qualification and confirmation.

## Confirmation scores

| Arm / intervention | Exact recurrent state | Answer |
|---|---:|---:|
| Host theorem/executor | 2,048/2,048 = 100.000% | 100.000% |
| **Learned Cayley treatment** | **2,048/2,048 = 100.000%** | **100.000%** |
| Favorable ordinary transformer | 32/2,048 = 1.562% | 27.148% |
| Learned `S^2` false generator | 18/2,048 = 0.879% | 23.779% |
| Deranged two-witness cards | 15/2,048 = 0.732% | 23.340% |
| One-witness unit completion | 45/2,048 = 2.197% | 25.439% |
| State reset between events | 30/2,048 = 1.465% | 26.953% |

Treatment exact state is 100% independently at every depth from three through
eight. Operation-nonce recoding leaves all predicted states bit-identical. All
18 immutable confirmation gates pass.

## What is confirmed

S7 learns a compact model-owned representation of the cyclic successor law
from 23 successor labels and three zero anchors. It then uses the same learned
generator to infer and execute previously unseen affine operation cards from
two demonstrations, compose those operations recurrently, and preserve exact
state through depth eight. It receives no recurrent, answer, development-law,
or confirmation-law supervision.

The exact-fit ordinary transformer fails on the same unseen laws, while wrong
topology, broken cards, insufficient evidence, and state-reset controls all
collapse. S7 is therefore promoted as the strongest confirmed bounded native
reasoning component: learned symbolic dynamics plus contextual law induction
and exact recurrent reuse.

## What is not confirmed

The result does not establish unrestricted native reasoning. Cyclic topology,
equality, bounded nested replay, event invocation, pop-insert state mutation,
and the loop limits are architectural. Natural-language grounding, arbitrary
algebra discovery, model-owned active-step selection, learned halt, open-ended
planning, and transfer into the frozen Shohin language model remain open.

The next phase is integration, not a wider repeat of this board: ground the
confirmed generator/compiler through S4/S5's model-owned parser and controller,
then test fresh natural-language operations and learned termination under the
same causal and sealed-confirmation discipline.

## Artifact custody

- Confirmation evaluation SHA-256:
  `4b4a539565fbf821c075f6ec4b16d34aa30e130f08f33caa28ca7c4f41f4360d`
- Confirmation assessment SHA-256:
  `ceda83124e27efb80a188797c379ff3a429b4bb1db22bc272a006811e1181511`
- Development evaluation SHA-256:
  `e02ed2d3111f8a483a96910286dcd682f9b4ee0a867910450becd8782224688f`
- Development assessment SHA-256:
  `2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4`

Checkpoint, board, evaluations, assessments, and promotion manifest are
mirrored locally and on Newton. The S7 development and confirmation boards are
closed permanently; there is no rescore, refit, threshold repair, or second
confirmation board.
