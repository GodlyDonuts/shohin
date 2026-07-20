# R12 S9.2 Global Anchor Closure CPU Result

**Decision:** admit mechanics for one fresh-board neural development experiment

**Seed:** `7509220561492772015`

**Closed mechanics rows:** 2,048 S9 development rows; no neural checkpoint or
sealed confirmation row read

**Report SHA-256:**
`91d653c7e2a131ad7e21319dd72a52dee95c00520fbd59771fe5f7a08fe52e24`

## Result

All 17 frozen mechanics gates pass.

| Mechanic | Result |
|---|---:|
| Oracle exact graph | 2,048/2,048 |
| Operation-recoded oracle exact graph | 2,048/2,048 |
| One weak required root breaks local selection | 2,048/2,048 |
| Global assignment recovers that weak root | 2,048/2,048 |
| One extra positive root breaks local cardinality | 2,048/2,048 |
| Global assignment ignores that extra root | 2,048/2,048 |
| Uniform-logit abstention | 2,048/2,048 |
| Flat-positive syntax control exact | 0/2,048 |
| Shuffled root-score control exact | 0/2,048 |
| Wrong high-margin root selected | 2,048/2,048 |
| Wrong high-margin root exact | 0/2,048 |
| Wrong high-margin count followed | 2,048/2,048 |
| Wrong high-margin count exact | 0/2,048 |
| Metadata/target poisoning leaves selection identical | 2,048/2,048 |

Every row has at least 632 distinct complete syntax-valid assignments under the
measured one-slot lower bound; median is 986 and maximum is 1,459. Thus finite
grammar and cardinality do not uniquely reveal the graph. The independent
reduced exhaustive solver agrees with interval Viterbi on 10,000/10,000 cases
with zero score error.

Instrumentation records zero compiler and executor calls during root
optimization. Full decode calls the compiler exactly once and never calls the
executor. Deliberately wrong higher model scores are followed into rejection or
wrong output rather than repaired from semantics.

The coordinate-free hard-negative orbit loss is exactly zero under identical
score multisets, positive when one competitor changes, and has finite nonzero
gradients on both views.

## Boundary

This admits only the mechanics and anti-leakage boundary. It does not show that
learned logits choose the right global assignment, improve S9.1, remain alpha
closed, or generalize beyond the templated graph language. Those are frozen
fresh-development gates. Confirmation remains sealed.
