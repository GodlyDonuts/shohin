# R12 S9.1 Alpha-Closed Structured Compiler Development Result

**Decision:** reject S9.1 for confirmation; retain as the strongest bounded
fresh-development compiler/reasoner baseline

**Sole valid job:** Newton `693793`, `evc47`, completed `0:0` in 41m23s

**Score access:** development `1`, confirmation `0`

## Artifact custody

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `0c04039821fdb130da9b6aaf3d303c7652768ed769db3f3671a7792e78d4c8b8` |
| Evaluation | `e0d77a32cbab9276e0cbc048a08f698594eca1fa4d98ff912c56ef33dbfcfa5c` |
| Assessment | `727c913db8d8fc5765dc6f39074c4e4d2a09fcec6b5f0df942ad621feda873c6` |

Newton and local copies match all three hashes. Scoreless job `693789` was
canceled on `evc28` after CUDA initialization hung before model/data access; it
wrote no artifact and is not a scientific run.

## Primary result

| Arm | Exact graph | Exact state | Exact answer |
|---|---:|---:|---:|
| Gold graph | 2,048/2,048 | 2,048/2,048 | 2,048/2,048 |
| **S9.1 treatment** | **2,025/2,048 = 98.877%** | **98.877%** | **98.877%** |
| S9.1 unconstrained decode | 2,023/2,048 = 98.779% | not promoted | not promoted |
| Equal-budget no-class | 1,766/2,048 = 86.230% | 86.230% | 86.230% |
| Shuffled relations | 0/2,048 | not promoted | not promoted |
| Lexical-source-free | 0/2,048 | not promoted | not promoted |
| Uniform logits | 0/2,048 | not promoted | not promoted |

Treatment beats its matched no-class arm by **12.646 percentage points** exact
graph and closed S9 by **4.102 points**. Every depth from three through eight is
at least 97.947% exact state. The complete system remains 134,580,264
parameters.

## Failure decomposition

All 2,025 valid treatment graphs are exact and all execute to exact state and
answer. Span precision is 100%, recall 98.813%, and F1 99.403%. The same 2,025
rows are exact span/class rows. The remaining 23 rows do not produce a valid
graph; there are no valid-but-wrong treatment graphs.

Structured child assignment contributes two exact graphs over the frozen
unconstrained decoder (2,025 versus 2,023). Therefore the measured missing-child
repair is real but no longer the main residual. The next bottleneck is complete
root anchor/cardinality selection: roster, state, card, or event anchors must
all be present before a graph exists.

## Alpha-closure result

Class-ID and relation-storage reindexing are exact on 2,025/2,025 valid graphs.
Operation recoding gives:

- 2,024/2,025 originally valid rows with a valid recoded graph;
- 2,024/2,024 identical recurrent states and answers; but
- 2,022/2,024 bit-identical canonical graphs.

This is a large improvement over S9's 18 invalidated recodes, but it misses the
preregistered all-valid and exact-canonical-graph requirements by one and two
rows respectively. State/answer equality cannot substitute for the stronger
graph gate because accidental semantic equivalence is possible.

## Causal controls

Treatment state accuracy collapses from 98.877% to:

- 8.838% with reversed links;
- 0.977% with deranged cards;
- 3.857% with one witness;
- 2.832% with state reset; and
- 3.760% with early nil.

The no-class gap, zero shuffled/source-free/uniform exactness, exact conditional
execution, and causal collapses jointly establish that the bounded computation
is model-grounded rather than host-solved. They do not establish free-form
general reasoning.

## Training accounting

Each treatment/control arm used exactly 24,000 unique source episodes, 48,000
charged original-plus-recoded views, batch 64, 750 updates, and 128 sampled
negative candidates per view. Treatment finished at 100% sampled candidate and
positive accuracy with supervised loss `8.585e-06` and orbit loss `4.883e-04`.
The shuffled arm retained only 15.708% positive accuracy.

## Decision and next theory

Twenty-seven of 30 frozen gates pass. The three failures are operation-recode
eligibility, canonical graph identity, and the consequent all-gates decision.
Confirmation remains sealed and this board must never be rescored.

The admissible S9.2 hypothesis is **global anchor closure**, not more arithmetic
or a wider transformer: choose roster/state/card/event anchor sets jointly from
model logits under only the existing finite grammar, and strengthen alpha
equivariance across both positive anchors and their hard negative competitors.
It requires a new theorem/falsifier, equal-budget controls, and a fresh board.
No threshold may be relaxed.
