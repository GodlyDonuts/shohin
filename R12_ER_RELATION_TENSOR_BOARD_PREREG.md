# R12 ER-TT Fresh Board Preregistration

**Protocol:** `R12-ER-TT-v1-board`

**Status:** seedless builder qualified in memory. No scientific seed, board
directory, training seed, GPU run, or scored access exists.

## Board contract

| Split | Families | Views | Rows |
|---|---:|---:|---:|
| Train | 12,000 | 4 | 48,000 |
| Development | 512 | 4 | 2,048 |
| Sealed confirmation | 512 | 4 | 2,048 |

Every family fixes one semantic problem while its four rows independently
shuffle eighteen physical records and render the four source-field factors.
Training uses one renderer-parity coset and both scored splits use its disjoint
complement. Exact prompt, word-13-gram, compact name, semantic-family, and family
ID overlap must be zero between every split pair.

Cardinality `N=3..6` is exact-balanced. Active rule count 2/3/4 and depth 1–12
are balanced to within one family per scored split, which is the arithmetic
minimum because 512 is not divisible by 3 or 12. Every family contains at least
one non-bijective relation. Atomic relation rows may recur; complete relation
sets, programs, initial states, queries, symbols, and renderer compositions form
the split-disjoint semantic signature.

## Public training fields

Training rows contain only:

- the shuffled program bytes and late-query bytes;
- physical and semantic line roles;
- cardinality and active rule count;
- binding, initial-entity, active before/after witness, and query spans;
- active relation rows inferred from witnesses;
- event references, HALT, and query position; and
- renderer/family metadata.

No training row contains terminal state, answer, recurrent trajectory, or any
development/confirmation oracle. An independent public-byte parser infers
relations by equality and executes each row without consulting compiler targets.
It must agree with generation on every row before bytes can be written.

## Seedless full-scale dry result

Fixture seed `104729` is a non-scientific source test and is permanently barred
from score-bearing use. Its in-memory 48,000/2,048/2,048 build passes all 15
gates:

- 13,024 rows at each cardinality;
- rule-count rows 17,368 / 17,368 / 17,360;
- depth rows differ by at most eight, exactly the two scored-split remainder;
- 13,024/13,024 semantic families contain a non-bijective rule;
- maximum program/line lengths are 610/96 bytes, below inherited 640/144 limits;
- all pairwise name, semantic-family, exact-prompt, and word-13-gram overlap is zero;
- train/scored renderer sets are disjoint;
- family-deranged state exactness is 1,064/13,024 = 8.170%; and
- equality-ablated state exactness is 1,026/13,024 = 7.878%.

## Admission and custody

The exact builder, renderer/parser, tests, this preregistration, and parent
adapter contract must be committed and pushed before a scientific board seed is
derived. The first production build must be independently reproduced
byte-for-byte. `confirmation.jsonl` must be mode `0600`; access must remain
`0/0`. Any collision, parser mismatch, overlap, size violation, or failed gate
retires that seed before training.

Passing this board build admits only score-bearing training/evaluator source
implementation. It is not evidence that the neural compiler succeeds.
