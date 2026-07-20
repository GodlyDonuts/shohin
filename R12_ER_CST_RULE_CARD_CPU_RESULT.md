# R12 ER-CST Rule-Card CPU Result

**Decision:** `admit_er_cst_rule_card_neural_implementation`.

**Status:** finite mechanics pass only. No neural architecture, board, seed, H100
run, development score, or confirmation access exists.

## Frozen contract

| Item | Value |
|---|---:|
| Exact source commit | `5a03824d2adcaa11633c6b7fd77cebe73afbd99e` |
| CPU seed | 1,729 |
| Registered episodes | 10,000 |
| Rule family | all six three-position permutations in `S_3` |
| Opaque cards per episode | 3 |
| Program depth | 1--12 |

Each card is inferred from one determining before/after witness containing three
distinct fresh symbols. The inferred card maps output positions to input positions.
Programs then invoke three fresh opaque opcode names and stop at a sampled HALT.

## Results

| Gate | Exact / total |
|---|---:|
| Determining-witness inference | **10,000/10,000** |
| Program execution and full trajectory | **10,000/10,000** |
| Witness-symbol alpha rename | **10,000/10,000** |
| Opcode alpha rename | **10,000/10,000** |
| Card-storage reindex | **10,000/10,000** |
| Post-HALT suffix invariance | **10,000/10,000** |
| Family card derangement final-state exact | 1,508/10,000 = **15.08%** |

All seven frozen CPU gates pass. Malformed witnesses, repeated-symbol ambiguous
witnesses, invalid cards, unknown opcodes, and invalid HALT positions are rejected
rather than repaired.

## Hashes

| Artifact | SHA-256 |
|---|---|
| Durable report | `90c5e6fe90acb5185b1dd1ff41ca5f6774a283c587244d31a02ebccc5d8d055f` |
| Episode registration | `a3802185b26b356c9cf6472fd0d2d71cf6674e1299834b396a8e7e0e6bd27a93` |

## Interpretation

The finite rule-card representation is well-defined, uniquely identifiable from the
named witness family, exactly compositional, alpha-invariant, storage-order invariant,
and HALT-persistent. Card derangement produces the expected near-chance collapse.

This admits implementation of a parameter-audited neural compiler and tied rule-card
motor under the remaining 7,870,820-parameter budget. It is not neural evidence and
does not establish episodic rule induction, natural-language grounding, planning, or
general reasoning.
