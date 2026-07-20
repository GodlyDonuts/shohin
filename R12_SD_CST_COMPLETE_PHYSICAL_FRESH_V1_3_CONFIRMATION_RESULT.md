# R12 SD-CST Complete Physical Fresh v1.3 Confirmation Result

**Decision:** `confirm_complete_physical_fresh_v1_3`.

**Status:** independently confirmed on the sole sealed-board read. The retained
checkpoint is a bounded compiler/executor baseline, not evidence of broad general
reasoning.

## Exact run contract

| Item | Value |
|---|---:|
| Scientific source | `eed66757c47e126b6566ee269bc73b0c0cef4fab` |
| Confirmation evaluator source | `94b26058bfa9d43089ce02277b3cdaeb9a1d6594` |
| Board seed | `8920874392524997882` |
| Training seed | `8446904969546017898` |
| Slurm job / node / elapsed | `694451` / `evc46` / 33s |
| Training rows | 48,000 = 12,000 families x four even-parity renderers |
| Development rows | 2,048 = 512 families x four odd-parity renderers |
| Sealed-confirmation rows | 2,048 = 512 new families x four odd-parity renderers |
| Fitting | None during confirmation |
| Complete deployed parameters | 192,129,179 |
| Fresh-trained parameters | 12,152,855 across 102 tensor names |
| Strict-200M headroom | 7,870,821 |

The evaluator hash-bound the exact development authorization, checkpoint, gate
config, board, and sole development ledger before opening the confirmation split.
It then wrote an immutable `O_EXCL` confirmation ledger, compiled every sealed row
once, poisoned the source, executed only the 25 categorical program bytes plus one
query byte in a separate process, and invoked an independent assessor.

## Sealed-confirmation metrics

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
| Exact final state | **2,048/2,048 = 100%** | 157/2,048 = 7.666% |
| Exact answer | **2,048/2,048 = 100%** | 515/2,048 = 25.146% |
| Exact state and answer | **2,048/2,048 = 100%** | 157/2,048 = 7.666% |

Treatment is 512/512 exact packets and joints on each of the four unseen renderer
compositions. It is also 100% joint at every frozen depth from one through six.
All 19 scientific gates pass. The independent assessor recomputes the metrics and
gate vector and passes all four artifact, parameter, metric, and gate-vector checks.

## Causal controls

| Source-blind executor arm | Exact state | Exact answer |
|---|---:|---:|
| Uniform packet | 33.203% | 33.398% |
| Shuffled packet | 33.203% | 33.203% |
| Reset | 49.609% | 57.812% |
| Freeze | 15.625% | 34.375% |
| Post-HALT perturbation | 100% | 100% |
| Force alive after HALT | 31.250% | 45.117% |
| Query rotation | 100% state | 0% answer |
| Initial-state rotation | 73.242% | 78.516% |
| Event-kind flip | 50.977% | 55.664% |
| Event-identity rotation | 54.688% | 58.203% |
| Event-amount flip | 83.789% | 90.625% |

Post-HALT perturbation remains bit-identical. The source-poison check is also
bit-identical for treatment and control. These interventions establish that the
categorical packet fields, recurrent update, HALT, and query readout are causally
used rather than reconstructed from surviving source text.

## Artifact hashes and custody

| Artifact | SHA-256 |
|---|---|
| Trained compiler checkpoint | `a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8` |
| Confirmation authorization | `b6add75e1d6596f0f32054fca212bf12e127c27c9e8239ac13be88b6497adcc7` |
| Hard packets | `3f9189595ed1500f054f03e673e7d023ac595debe3801d5865c477250d03d994` |
| Pointer evidence | `99342e9e16cbbc1ec20b62372711407fa63b6cfb098bd0ba6963b55e711e73e9` |
| Executor outputs | `676b82947bfa04bbf3f9d98240807f091014f6cba9d2f3c5e13b35572d768107` |
| Confirmation report | `2857f94f0816053ed7ac9610ef8eab5e3749360cb53cb1d0e6e8ef707dc8ded7` |
| Independent assessment | `4629a745f6eed2e388eb6e1f78b29dff346ee6939e21275ae6ff1d66719d3cb9` |
| Confirmation ledger | `b9bf805f3e9e821da4479828ca47cd058db3e471badbbdfb43cf2df04ea7842e` |
| Slurm log | `a1ce866afefa63795e9453fb52cd93cd5fc76f573be26b75a8a55ea07d12eb9b` |

Development/confirmation custody is exactly `1/1`. Local mirrors match Newton,
and a separate local assessor replay is byte-identical to the Newton assessment.
The checkpoint, report, and assessment are retained read-only at:

`/lustre/fs1/home/sa305415/shohin_promoted/sd_cst_complete_physical_fresh_v1_3`

## Claim boundary

This confirms fresh finite renderer/name compilation into a model-owned categorical
program and source-deleted recurrent executor. The model transfers across new opaque
names and four unseen compositions of known rendering factors, preserves explicit
state across one-to-six operations, halts internally, and reads the requested entity.

It does **not** establish unconstrained language grounding, arbitrary program
induction, learned arithmetic, open-domain planning, self-directed search, or general
native reasoning. Those require distinct fresh-board experiments rather than expanding
this result's claim.
