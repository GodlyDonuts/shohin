# R12 ER-CST Witness Equality Bus v1.1 Confirmation Preregistration

**Status:** evaluator implemented before source freeze or confirmation access.

## Development authorization

Sole qualification job `694567` completed on `evc48` in 16m34s. The pilot and
independent assessor both return `authorize_one_sealed_confirmation`; all 14
scientific gates and eight development-assessor gates pass. Treatment reaches
2,038/2,048 = 99.512% exact packets and joints, 2,040/2,048 = 99.609% exact
recurrent states, and 2,048/2,048 exact answers. Its minimum depth joint is
96.875% and minimum renderer joint is 99.414%. Family-deranged/equality-ablated
packet accuracy is 0.098%/0%, joint accuracy is 0.098%/0%, and state accuracy is
17.822%/15.479%.

Exact authorization artifacts:

| Artifact | SHA-256 |
|---|---|
| Scientific source | `87d53b53462d8d15660663238fd33886c010efb7` |
| Board report | `22cb355e58e9f3b8125a57c60c7aafb7aadead4406abc426da368e3b3b2cff75` |
| Trained checkpoint | `917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7` |
| Development evidence | `1a7504eb9b08d7d123e89705360f2eb37a861f5cd75b3ebc73570c8e904327fb` |
| Development report | `d295f8f67f32916386e04674fc782a0982b9b1b55f7b82aa1eaab6f59bb1ae35` |
| Development assessment | `29e4349225ed9523ec3b8096cd2cd16ef1b55c727797421a1ac0b39c042f11b2` |
| Development ledger | `5b6e233b3cc9d3cf49a32525ca11f6c6f846005486df67a252e9ca4ec36b4db3` |
| Sealed confirmation registration | `6593bb17690fc72e5392b953af75f8686a92e799bcd600307affcb7fc0080c4d` |

Training seed is `2262748995832026278`. Complete/trainable parameter counts are
192,726,827/12,021,276, leaving 7,273,173 below the absolute 200M ceiling.

## Frozen confirmation procedure

1. Run from a clean exact evaluator commit. Require the scientific commit above
   to remain an ancestor and every scientific runtime path to remain byte-exact.
2. Hash-verify the board, checkpoint, development evidence/report/assessment,
   and sole development ledger. Require the exact authorization decision and
   custody `1/0`.
3. Write an immutable authorization artifact before opening confirmation. Then
   create one `O_EXCL`, mode-`0444` confirmation ledger before hashing or parsing
   the sealed confirmation bytes.
4. Perform no fitting, gradients, search, repair, retry, threshold change, or
   checkpoint selection.
5. Reconstruct all three frozen arms from the exact checkpoint. Compile all
   2,048 sealed rows once, delete the source at the existing architecture
   boundary, execute the hard categorical packets, and emit raw evidence.
6. Invoke a separate assessor that recomputes every metric, gate, parameter
   certificate, artifact hash, authorization binding, and custody fact.

## Frozen gates

Confirmation retains all development thresholds: treatment packet/state/answer/
joint at least 90%; minimum renderer joint at least 85%; minimum depth joint at
least 80%; all packet fields at least 95%; all pointers, including all 18
witness pointers, at least 90%; treatment packet and joint advantages over both
controls at least 50 percentage points; control packets at most 35%; control
states at most 40%; exact finite motor/reader certificates; unchanged confirmed
parent; and complete system below 200M. The development-only custody gate is
replaced by final custody exactly `1/1`. The independent assessor must also pass
artifact-hash, parameter, authorization, metric-recomputation, gate-vector, and
custody checks.

All gates passing yields `confirm_er_cst_witness_equality_v1_1`. Any failure
yields rejection. The sealed board is read once and never rescored.

## Claim boundary

A pass confirms bounded inference of fresh episodic `S_3` operation meanings
from before/after witness equality, followed by source-deleted categorical
composition, internal halt, and late-query readout across split-disjoint
renderers and names. It does not establish unrestricted language grounding,
arbitrary operation induction, arithmetic, planning, or broad general reasoning.
