# R12 ER-TT v1 Development Result

**Protocol:** `R12-ER-TT-v1`

**Decision:** reject ER-TT v1 and permanently leave its confirmation split
unopened.

## Custody

- Scientific source: `3bd8a329ab94f3426be61ae87db92a14a285f2f7`
- Board seed: `1209366536012979338`
- Training seed: `4773363983426630371`
- Sole score-bearing job: `694758` on H100 `evc40`
- Runtime: 22m59s, exit code zero
- Development/confirmation accesses: `1/0`
- Complete/trainable/headroom parameters:
  `192,740,854 / 12,037,293 / 7,259,146`

The immutable checkpoint was written before the exclusive development ledger.
The independent assessor reproduced the rejection. Confirmation was never
opened and this board must never be rescored.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `compiler.pt` | `0c12b2eb68411f63c97cea136d80c8e44a0c0923cc3afa1cbf38fc88de6ffad3` |
| `development_evidence.pt` | `5504dc273c1c9abfd4dae333f80473248fc36d83f13aafb5844d1ec8d99a48b0` |
| `development_report.json` | `2b935215037908e0fa5154e5348fecc23f4bc42e83082a0a1fd33cc5b8cf0113` |
| `development_assessment.json` | `3628e2d071f980841795f6c9a71cc18986be5eed935e27fb7a7aef50513d977a` |
| Development ledger | `000aa5012563fb8251542c60b5dcfcb83e6e4445ba091649f0ecc830eb80e8b7` |

Newton and local copies of all four run artifacts hash-match. A local replay of
the independent assessor is byte-identical to
`development_assessment.json`.

## Frozen scores

| Arm | Packet | State | Answer | Joint |
|---|---:|---:|---:|---:|
| Treatment | 2/2,048 (0.098%) | 315/2,048 (15.381%) | 669/2,048 (32.666%) | 2/2,048 (0.098%) |
| Family-deranged | 0/2,048 | 252/2,048 (12.305%) | 470/2,048 (22.949%) | 0/2,048 |
| Equality-ablated | 0/2,048 | 241/2,048 (11.768%) | 579/2,048 (28.271%) | 0/2,048 |

Treatment cardinality, binding pointers, initial pointers, line pointers,
initial state, active-rule mask, halt, query pointer, and query are all
2,048/2,048 exact. The failure is concentrated in episodic relation extraction
and its downstream event use:

| Treatment field | Exact rows |
|---|---:|
| Relation tensor | 2/2,048 (0.098%) |
| Witness pointers, whole packet | 12/2,048 (0.586%) |
| Events, whole packet | 1,235/2,048 (60.303%) |

All 2,048 scored families contain a non-bijective relation. Non-bijective
joint is therefore the same 2/2,048 result.

## Granular diagnosis

Independent recomputation from the frozen raw evidence gives:

| Quantity | Treatment | Family-deranged | Equality-ablated |
|---|---:|---:|---:|
| Active relation cells | 10,092/27,628 (36.528%) | 7,456/27,628 (26.987%) | 7,975/27,628 (28.866%) |
| Complete active rules | 193/6,140 (3.143%) | 103/6,140 (1.678%) | 70/6,140 (1.140%) |
| Active event cells | 21,621/24,576 (87.976%) | 22,983/24,576 (93.518%) | 21,364/24,576 (86.930%) |
| Active witness occurrences | 40,679/55,256 (73.619%) | 36,305/55,256 (65.703%) | 40,346/55,256 (73.017%) |

Treatment before-witness localization is 26,322/27,628 = 95.273%, while
after-witness localization is only 14,357/27,628 = 51.965%. Relation-cell
accuracy falls monotonically with cardinality: 49.652%, 39.659%, 36.198%, and
28.144% at `N=3,4,5,6`.

The training losses make this an architecture/optimization failure rather than
an exact-packet artifact. By epoch two, cardinality, rule-active, halt, query,
and event losses are effectively zero, but treatment relation-row loss remains
`1.495433`, approximately the chance cross-entropy of the mixed-cardinality
task. Witness-pointer loss remains `0.768608`. The system learned ordinary
layout and control fields but did not learn a reliable variable relation bus.

## Causal and invariance evidence

The two exact treatment packets pass the eligible relation, cardinality, reset,
and query interventions, but two rows are insufficient for the preregistered
effectiveness gates. Source invariance exposes severe content-dependent
routing:

| Source transform | Fully exact rows |
|---|---:|
| Rule-storage reindex | 1,997/2,048 |
| Physical-record reindex | 1,840/2,048 |
| Witness alpha rename | 6/2,048 |
| Opcode alpha rename | 32/2,048 |
| Post-HALT suffix | 1,512/2,048 |
| Post-seal source poison | 2,048/2,048 |

Witness alpha rename preserves exact relation tensors on only 8/2,048 rows;
opcode alpha rename preserves them on 93/2,048. This rejects the intended
alpha-equivariant semantic-binding claim.

## Failure localization

ER-TT v1 successfully removes the learned motor and finite `S_3` card table,
but its soft byte-pointer/equality path entangles two jobs:

1. **where:** locate each occurrence from syntax and record position; and
2. **what:** preserve the selected symbol identity so equality can define the
   relation and event binding.

The same content-sensitive token memory performs both. It finds most
before-side occurrences, loses nearly half of after-side occurrences, mixes
symbol fingerprints under uncertain soft pointers, and memorizes name/opcode
surface. More data or a larger relation head would not address that causal
failure.

## Admitted successor hypothesis

A fresh-board successor may test a **dual-stream symbol transport bus**:

- an alpha-invariant structural stream performs line/slot/span routing from
  token classes, delimiters, and order, without raw symbol identity;
- an identity stream pools the entire selected whitespace-delimited symbol,
  retaining raw bytes only after routing;
- relation rows are produced only by shared before/after identity equality;
- event-to-rule binding uses the same identity-equality bus rather than whole-
  record similarity;
- straight-through or constrained span selection prevents soft mixtures from
  corrupting identity fingerprints;
- treatment, family-deranged, equality-ablated, source-free, and alpha-recoded
  controls remain matched; and
- the complete deployed system remains strictly below 200M parameters.

This is a repair hypothesis, not a reasoning result. It requires CPU
falsification, exact parameter accounting, fresh source/board/training seeds,
and a new one-read development contract. ER-TT v1 itself is permanently closed.
