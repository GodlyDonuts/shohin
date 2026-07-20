# R12 SD-CST Complete Physical Fresh-Board v1.3 Development Result

**Decision:** `authorize_one_sealed_confirmation`.

**Status:** complete pilot and independent assessment; confirmation remains
unopened pending a separately committed evaluator.

## Exact run contract

| Item | Value |
|---|---:|
| Scientific source | `eed66757c47e126b6566ee269bc73b0c0cef4fab` |
| Board seed | `8920874392524997882` |
| Training seed | `8446904969546017898` |
| Slurm job / node / elapsed | `694383` / `evc23` / 11m48s |
| Training rows | 48,000 = 12,000 families x four even-parity renderers |
| Development rows | 2,048 = 512 families x four odd-parity renderers |
| Updates | 3,000 treatment + 3,000 family-deranged control |
| Complete deployed parameters | 192,129,179 |
| Fresh-trainable parameters | 12,152,855 across 102 tensor names |
| Strict-200M headroom | 7,870,821 |

Every source, board, parent-checkpoint, execution-core, H100, bf16, output-
absence, and pre-access gate passed. The checkpoint and immutable gate config
were written after both endpoints and before the atomic development ledger.
Program/query sources were poisoned after packet sealing. A separate process
received only 25 categorical program bytes plus one query byte and the frozen
execution core.

## Training fit

Treatment is 48,000/48,000 exact packets, including 12,000/12,000 on each
training renderer. The family-deranged control changes only three entity-role
labels per latent family while sharing bytes, initialization, updates, and
compute. It remains far below treatment.

## Independent development metrics

| Metric | Treatment | Family-deranged labels |
|---|---:|---:|
| Exact complete packet | **2,048/2,048 = 100%** | 0/2,048 = 0% |
| Initial state | **100%** | 0% |
| Event kind | **100%** | 100% |
| Event identity | **100%** | 0% |
| Event amount | **100%** | 100% |
| Late query | **100%** | 100% |
| All-nine line pointers | **100%** | 100% |
| All-three binding pointers | **100%** | 0% |
| All-three initial-occurrence pointers | **100%** | 100% |
| All active-event entity pointers | **100%** | 100% |
| Exact final state | **2,048/2,048 = 100%** | 148/2,048 = 7.227% |
| Exact answer | **2,048/2,048 = 100%** | 467/2,048 = 22.803% |
| Exact state and answer | **2,048/2,048 = 100%** | 148/2,048 = 7.227% |

Each unseen renderer composition is independently 512/512 exact treatment
packets and joints. This exactly repeats the perfect diagnostic signal from the
spent v1.2 board under a new source, board, seed, completed report, and completed
independent assessment.

## Causal controls

| Source-blind executor arm | Exact state | Exact answer |
|---|---:|---:|
| Uniform packet | 33.203% | 33.398% |
| Shuffled packet | 33.203% | 33.203% |
| Reset | 50.977% | 59.375% |
| Freeze | 13.672% | 28.711% |
| Post-HALT perturbation | 100% | 100% |
| Force alive after HALT | 33.203% | 46.289% |
| Query rotation | 100% state | 0% answer |
| Initial-state rotation | 70.312% | 78.320% |
| Event-kind flip | 53.125% | 59.375% |
| Event-identity rotation | 53.516% | 59.180% |
| Event-amount flip | 81.641% | 88.086% |

Post-HALT perturbation is bit-identical across final state, answer, state
trajectory, and alive trajectory. All 18 pilot core gates and all 18
independently recomputed core gates pass. All four assessor gates pass: artifact
hashes, exact parameter certificate, independent metric recomputation, and gate-
vector equality.

## Artifact hashes and custody

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8` |
| Gate config | `ab466c339b77d4193cbdbc383a2c9a28bd4ce6afcf9579f3a714b301e8d9a990` |
| Hard packets | `00a45016f81287361e9f0bcdd7869e386bcf792b68d990f230cae98ef44f44bf` |
| Pointer evidence | `80f477af17d624b66fd3d9f6d5f7a7de7efd62943f63271034505ce32dbc77d6` |
| Executor outputs | `4bcfc44183c3eb610302c21f472c6f5da11fd5d7abddd4b97dc196b97af8eaae` |
| Development report | `7dc048cc9ad16e1e326c7e4180fb06539428a4518c4d79440a92b794754b6bc2` |
| Independent assessment | `1c5fad49a6eba6c2d76420945166e78b807d002f947a616c542c8a85ba35e497` |
| Development ledger | `15a9edd09f008084c0533672e57af8da96935da3dc1ca86edbaa4200e2f499e0` |
| Slurm log | `585f3eadf2800775fcab381d159fbcf98286e281ec0946a7dd6d6b8bd78c50a3` |

Local mirrors hash-match Newton. Development/confirmation custody is exactly
`1/0`; confirmation remains mode `0600`.

## Claim boundary

This establishes a clean development pass for bounded fresh renderer/name
compilation into a source-deleted categorical executor. It does not establish
unconstrained language grounding, arbitrary programs, learned arithmetic,
self-directed planning, or broad general reasoning. Only the separately frozen
one-read confirmation can promote this bounded mechanism.
