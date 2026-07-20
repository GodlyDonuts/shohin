# R12 Episodic Relation Tensor Transport: CPU Result

**Protocol:** `R12-ER-TT-v1-cpu`

**Status:** CPU and parameter-free tensor mechanics admitted. Neural compilation,
fresh-board qualification, and reasoning claims remain unopened.

## Frozen provenance

| Field | Value |
|---|---|
| Source commit | `0bf6d91b86c90a8772a8f109803131f729e1602f` |
| Mechanics seed | `2718` |
| Trials | `10,000` |
| Mechanics source SHA-256 | `27c04b35f26ec63da0649e9e973db32e7f740d69b7c164bbf3405752f712e8bd` |
| Tensor motor SHA-256 | `af9916b355d4a5d1c0aa126420f21383ef879a47e1ea4e5c889358a20c98840f` |
| Report SHA-256 | `28e5acc29e34c723bf35d58f49634535ac5c5d23cf841a8385e7955a09804cce` |
| Episode registration SHA-256 | `da8d4c3bf3e17b9f50114f486e9387d6f1afa4662bf810c62182329f87f05d7c` |

The durable report is
`artifacts/r12/er_relation_tensor_cpu_2718.json`. It was created with exclusive
file creation after the source commit and was not regenerated in place.

## Results

Cardinality was exactly balanced: 2,500 episodes each at `N=3,4,5,6`.

| Measure | Exact | Rate |
|---|---:|---:|
| Witness relation inference | 10,000/10,000 | 100% |
| Program execution and trajectory | 10,000/10,000 | 100% |
| Relation composition | 10,000/10,000 | 100% |
| Witness alpha rename | 10,000/10,000 | 100% |
| Opcode alpha rename | 10,000/10,000 | 100% |
| Card-storage reindex | 10,000/10,000 | 100% |
| Cardinality padding | 10,000/10,000 | 100% |
| Post-HALT suffix | 10,000/10,000 | 100% |
| Source-deleted packet | 10,000/10,000 | 100% |
| Episodes containing a non-bijective rule | 9,936/10,000 | 99.36% |
| Family-deranged final-state exact | 753/10,000 | 7.53% |
| Equality-ablated final-state exact | 417/10,000 | 4.17% |

All thirteen preregistered mechanics gates pass. The zero-parameter tensor motor
also passes focused torch tests for variable cardinality, hard relation
selection, recurrent composition, and persistent pre-apply HALT.

## Interpretation

This establishes that a variable-cardinality relation representation can be
inferred exactly from determining witnesses and composed without an enumerated
`S_3` class table or a learned recurrent transition table. It also shows that
the intended controls materially change terminal state on this distribution.

It does **not** establish that Shohin can compile the relation rows from source,
that a neural system generalizes to fresh relation families, or that ER-TT is a
general reasoning mechanism. The only admitted next action is implementation
and exact parameter audit of the smallest neural compiler extension below the
strict 200M complete-system ceiling. No neural board or seed may be drawn before
that adapter and its tests are committed.
