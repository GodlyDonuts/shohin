# R12 SD-CST Complete Physical Fresh-Board v1.2 Result

**Decision:** closed evaluator non-result. Never rescore this board and never
open its sealed confirmation.

**Claim boundary:** the preserved artifacts contain a strong source-free
diagnostic signal for bounded fresh renderer/name compilation. They do not form
an authorizing development score because the frozen report and independent
assessment did not complete. They do not establish broad native reasoning.

## Frozen contract

| Item | Value |
|---|---:|
| Scientific source | `fab094f6e32f1e928551f1830509b8c83fbd759e` |
| Board seed | `4196082084031177718` |
| Training seed | `5923413289392567580` |
| Slurm job / node | `694355` / `evc50` |
| Training rows | 48,000 = 12,000 latent families x four renderers |
| Development rows | 2,048 = 512 latent families x four unseen renderers |
| Sealed confirmation rows | 2,048, unopened |
| Updates | 3,000 treatment + 3,000 family-deranged control |
| Complete deployed parameters | 192,129,179 |
| Fresh-trainable parameters | 12,152,855 across 102 tensor names |
| Strict-200M headroom | 7,870,821 |

The source, board, parent checkpoints, execution core, H100, and bf16 gates all
passed. Both arms completed. The checkpoint and gate config were written before
the immutable development ledger. Development was opened exactly once and the
compiler produced source-free hard packets, pointer-range evidence, and outputs
from the separate categorical executor.

The pilot then raised during report assembly. `fit_arm` stores renderer metrics
under `fit["train_metrics"]`; `_minimum_fit_packet` incorrectly iterated every
top-level fit value and attempted to index integer metadata as a renderer
record. No development report or independent assessment was written. This is
an evaluator defect after score access, so the board is spent even though the
saved evidence is complete.

## Training fit

| Arm | Exact packets |
|---|---:|
| Treatment | 48,000/48,000 = 100% |
| Family-deranged labels | 1,223/48,000 = 2.548% |

Treatment is 12,000/12,000 exact on each of the four training renderers. The
control uses one nonidentity three-cycle of entity roles per latent family and
shares all source bytes, initialization, renderer views, minibatch order,
optimizer, updates, trainable names, and parameter count.

## Source-free development diagnostic

The following was recomputed after closure using only the immutable hard-packet,
pointer-range, and executor artifacts. The development JSONL was not reopened.

| Metric | Treatment | Family-deranged labels |
|---|---:|---:|
| Exact complete packet | **2,048/2,048 = 100%** | 0/2,048 = 0% |
| Initial state field | **100%** | 0% |
| Event kind field | **100%** | 100% |
| Event identity field | **100%** | 0% |
| Event amount field | **100%** | 100% |
| Late query field | **100%** | 100% |
| All-nine line pointers | **100%** | 100% |
| All-three binding pointers | **100%** | 0% |
| All-three initial-occurrence pointers | **100%** | 100% |
| All active-event entity pointers | **100%** | 100% |
| Exact final state | **2,048/2,048 = 100%** | 131/2,048 = 6.396% |
| Exact answer | **2,048/2,048 = 100%** | 504/2,048 = 24.609% |
| Exact state and answer | **2,048/2,048 = 100%** | 131/2,048 = 6.396% |

Every unseen renderer composition is independently 512/512 exact treatment
packets and 512/512 exact treatment state/answer joints.

## Causal controls

| Source-blind executor arm | Exact state | Exact answer |
|---|---:|---:|
| Uniform packet | 33.203% | 33.398% |
| Shuffled packet | 33.203% | 33.203% |
| Reset | 57.422% | 63.086% |
| Freeze | 18.164% | 35.156% |
| Post-HALT perturbation | 100% | 100% |
| Force alive after HALT | 28.711% | 43.359% |
| Query rotation | 100% state | 0% answer |
| Initial-state rotation | 68.164% | 76.172% |
| Event-kind flip | 55.078% | 60.742% |
| Event-identity rotation | 56.445% | 61.133% |
| Event-amount flip | 84.570% | 90.820% |

Post-HALT perturbation is bit-identical across final state, answer, state
trajectory, and alive trajectory. Program/query source poisoning is bit-
identical for treatment and control before separate execution.

## Artifact custody

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `2c9ce2beb6e0ee320c773cf1c17c1c1d07323c51eee136a3d8e639010b2bf47f` |
| Development evidence | `672e2b95600d057703243321103051818daaac8c421de8561e4a2de36cce30d7` |
| Executor outputs | `12327103c1ed40e543bf7eaac932ace52b13a1186a11a433d50de29c449524ab` |
| Hard packets | `3c73c113060a24dec87ffa8428895aa939dd5e9eb293ee6d9822c0d03ab5a80b` |
| Gate config | `dc1a9f5850e290888f81ea9e2c9fe37e08742e60dd084af7f21b24debe2a4b6b` |
| Development ledger | `0f0d2c4d9537b5f6d423b8246e44051c3fc637a971cd1e8f59ef73d465bb5ad4` |
| Slurm log | `4fe53ef1361154ba655eff3c2cacdf1d6a49debbcbd0e978db9fd795fc805687` |

Local mirrors and Newton originals match. Development/confirmation custody is
exactly `1/0`.

## Next admissible test

V1.3 changes only report aggregation and schema/protocol identity. It keeps the
same architecture, parameters, data counts, renderer split, arms, optimizer,
updates, thresholds, controls, source-deletion boundary, and claim boundary.
It adds realistic nested-fit regression coverage and a complete synthetic
source-free assessor acceptance test. It requires a new source commit, board
seed, sealed board, committed board receipt, and training seed.
