# R12 S9.1 Alpha-Closed Structured Compiler CPU Result

**Decision:** admit S9.1 mechanics for one fresh neural development board

**Falsifier report:** `artifacts/r12/s9_1_alpha_closed_cpu_falsifier.json`

**Report SHA-256:**
`a43824595c513226f52f54a629bad5d52f0d7f3c2a67e672103d0a16284dc563`

## Result

The source-frozen CPU falsifier evaluated all 2,048 rows of the already closed
S9 development board. This was a mechanics test, not a new S9 score read.

| Arm | Exact or valid rows |
|---|---:|
| Oracle structured graph | 2,048/2,048 exact |
| Operation-recoded oracle | 2,048/2,048 exact |
| Old greedy after one required child is below `none` | 0/2,048 exact |
| Structured decoder under the same intervention | 2,048/2,048 exact |
| Uniform logits | 0/2,048 valid |
| Shuffled relation roles | 0/2,048 exact |
| Deliberately wrong high-margin child | 0/2,048 exact |
| Recoded sources within width-four proposal cap | 2,048/2,048 |

Every one of the 17,437 card/event regions left multiple syntax-valid local
choices. The minimum candidate count was 55, the median 83, and the maximum
123. Therefore the structured decoder is not obtaining the gold child from a
unique grammar slot.

## Interpretation

The result proves four bounded facts:

1. the structured assignment is representationally sufficient on every closed
   source;
2. it directly repairs the measured all-or-nothing missing-child failure;
3. it does not invent anchors from uniform scores; and
4. it does not semantically repair a wrong model preference.

It does not establish learned alpha closure or generalization. Those claims
require the sole fresh-development evaluation with equal-budget treatment,
no-class, and shuffled-label arms. Confirmation remains sealed unless every
development gate passes.
