# R12 ER-TT v1 Score-Bearing Preregistration

**Protocol:** `R12-ER-TT-v1-score`

**Status:** locally qualified before scientific source freeze and training-seed
draw. The fixed production board remains at development/confirmation access
`0/0`. No GPU fit or neural score exists.

## Fixed inputs

- Board source: `bd77c0fafdbc527688ba57aedd74ccdfbe2ed1cf`
- Board report SHA-256:
  `64ea4c0e19ea029102af240d44242c830d7b014e49a59af09a836b2d3efb6010`
- Train/development/confirmation rows: `48,000 / 2,048 / 2,048`
- Train/development/confirmation SHA-256 prefixes:
  `1982aeb2 / 59be0c40 / cac2515b`
- Confirmed ER-CST witness parent checkpoint SHA-256:
  `917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7`
- Confirmed parent assessment SHA-256:
  `4a0fb47233d86887bb46aa853560bf81d319840610d62abe4f1dfaa899671310`

The deployed system has 192,740,854 complete parameters, 12,037,293 trainable
parameters, no learned motor or reader, and 7,259,146 parameters of headroom
below the absolute 200M ceiling.

## Equal-Budget Arms

All three arms reconstruct the confirmed parent independently, begin from the
same trainable-state digest, use the same family order, 48,000 rows, two epochs,
3,000 optimizer updates, batch 32, AdamW settings, and renderer-consistency
weight.

1. **Treatment:** true witness equality and relation-row labels.
2. **Family derangement:** unchanged source bytes with relation labels supplied
   by another family in the same `(cardinality, rule_count)` bucket.
3. **Equality ablation:** unchanged relation labels and byte offsets, but every
   after-witness name is replaced by a deterministic fresh fixed-width name.
   Before witnesses, declarations, opcodes, events, queries, and all non-after
   bytes are unchanged.

Training supervises only compiler fields: semantic-line pointers, active
declaration/witness/query pointers, cardinality, active-rule mask, initial and
relation rows, event references, HALT, query, and cross-renderer consistency.
Final state, answer, recurrent trajectory, and every scored oracle are absent
from training.

## Source-Deleted Evaluation

The compiler is converted to a hard finite packet before execution. Program
bytes, token memory, residuals, pointer logits, and the model module are not
inputs to the recurrent motor. Invalid live references fail state and answer
scoring; they are not repaired. The motor applies the selected relation tensor
with persistent pre-apply HALT. The answer is a parameter-free terminal-state
row selection.

Raw evidence contains every predicted/target semantic field, active pointer
range, final state, answer, validity flag, cardinality, depth, renderer, and
non-bijective grouping. A separate assessor recomputes every metric and all
four interventions with an independent list executor.

## Frozen Causal Controls

Source-level recompilation tests:

- rule-record storage reindex;
- complete physical-record reindex;
- consistent witness alpha rename;
- consistent opcode alpha rename; and
- post-HALT suffix replacement.

Packet-level tests:

- source poisoning after packet seal;
- active relation-row derangement;
- cardinality-mask corruption with deterministic removed-index repair;
- initial-state reset; and
- query swap.

The first four source transformations must preserve the entire semantic packet,
state, and answer on every row. Post-HALT replacement must preserve state and
answer on every row. Every packet intervention must be exact on all otherwise
exact packets and must change every semantically sensitive packet.

## Immutable Development Gates

The sole development read advances only if:

1. treatment packet, state, answer, and joint are each at least 90%;
2. cardinality, initial rows, relation rows, rule activity, events, HALT, and
   query are each at least 95%;
3. every active line/binding/initial/witness/query pointer field is at least
   95%;
4. minimum joint by cardinality, depth, and unseen renderer is at least 80%;
5. non-bijective-family joint is at least 85%;
6. treatment packet and joint exceed both controls by at least 50 percentage
   points;
7. both controls remain at or below 35% packet and joint;
8. every source invariance is exact on all 2,048 rows;
9. all packet interventions are exact and effective on their sensitive rows;
10. the parent remains byte-identical outside the declared trainable set;
11. parameters are exactly 192,740,854 and below 200M; and
12. custody is exactly one development access and zero confirmation accesses.

Passing authorizes one separately frozen confirmation evaluator with unchanged
weights and thresholds. Failure permanently closes ER-TT v1; the board will not
be repaired or rescored and confirmation will remain sealed.

## Local Qualification

Twenty-five focused tests pass across mechanics, adapter, board, training,
interventions, and independent assessment. Ruff, byte compilation, shell
syntax, and diff checks pass. A real confirmed-parent reconstruction and
production-family backward pass are finite; all 110 declared trainable tensors
receive gradient, including every new cardinality, rule-activity, query, role,
and coordinate-witness component. Exact parameter accounting is unchanged.

## Claim Boundary

A pass would establish bounded fresh episodic compilation and recurrent
composition of arbitrary total finite copy relations over cardinalities three
through six under source deletion. It would not establish unrestricted
language grounding, unbounded algorithms, arithmetic, branching, planning, or
broad general reasoning.
