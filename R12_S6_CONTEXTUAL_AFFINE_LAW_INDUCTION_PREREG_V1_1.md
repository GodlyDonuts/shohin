# R12 S6.1 Contextual Affine Law Induction Split Repair

**Status:** frozen after the scoreless v1 CPU-gate failure and before neural
implementation, board generation, fit, model access, or score.

The original S6 preregistration's raw SHA-256 bucket split is rejected because
the exhaustive falsifier found that modulus-5 training laws never place value
`1` in the second law-card coordinate. No model, board, fit, development row,
confirmation row, or score exists. Architecture, capability, theorem, controls,
thresholds, and claim boundary remain unchanged.

V1.1 changes only law-split admission. Begin with the original hash buckets.
For each admitted modulus, inspect in order:

1. first card coordinate values in ascending order;
2. second card coordinate values in ascending order; and
3. destination values in ascending order.

If a value is absent from training, move the lexicographically first law that
supplies it from confirmation to training; if confirmation has no movable law,
use development. Never move the last law in a held-out split. Repeat until all
three coordinate sets equal `range(m)`. Record every move in the CPU report.

This is a pre-model identifiability repair, not result tuning: it prevents an
unseen-law score from being confounded by an unseen categorical coordinate. The
repaired train, development, and confirmation law sets remain pairwise disjoint.
Failure of the unchanged falsifier after this repair closes S6.1.
