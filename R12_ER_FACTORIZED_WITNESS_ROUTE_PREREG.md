# R12 ER-TT Factorized Witness Route Preregistration

## Status

Pre-freeze, train-only architectural falsifier. It may read only the existing
ER-TT `train.jsonl`. Development and confirmation are forbidden.

## Closed predecessors

Marginal-route v1.1 is rejected by one frozen gate at 89.925% complete witness
pointers, despite 90.9375% packet/joint/relation, 97.0625% state, 98.5375%
answer, 100% alpha invariance, and 100% oracle-route transport. All 806 failed
rows contain exactly one adjacent occurrence error.

The occurrence-addressed embedding repair is rejected more strongly at 59.500%
witness pointers and 60.9125% packet/joint/relation. Its read-only scale audit,
report SHA `d958cc0507fe85a489a3b85368f52ed67cfda6caf9fc5efc8d686216f28f6934`,
shows:

- zero ordinal information gives 0% witness rows;
- zero count information gives 0.4875% witness rows;
- endpoint scale 1.0 gives 59.500% witness / 60.9125% joint; and
- ordinal scale 1.5 gives 70.425% witness / 69.3875% joint.

The failure is not excess positional magnitude. Count and ordinal are both
necessary, but adding their embeddings inside structural query/key memory
entangles them and changes every route geometry. That architecture is closed.

## Hypothesis

Keep the v1.1 structural query/key dot products numerically unchanged. Add one
zero-initialized residual bias only to witness-route logits. The bias is indexed
by three model-visible, alpha-invariant discrete coordinates:

1. opaque candidate count in the physical rule record;
2. semantic witness role among six before and six after roles; and
3. opaque candidate ordinal within that physical record.

The table has shape `14 x 12 x 14`; twelve bounded role gates bring the total
to 2,364 learned scalars. Table values pass through `tanh`, are centered across
valid candidates, and are multiplied by `4*tanh(gate_role)`. The table starts
small random while every gate starts at exact zero, making route logits exactly
equal to v1.1 at initialization without blocking the first gate gradient. It
receives ordinary source pointer supervision. It cannot read symbol identity, target
relations, recurrent state, answer, executor output, development, or
confirmation. Raw symbol bytes remain confined to exact marginal equality after
routing. Declaration, initial, opcode, event, line, query, motor, and reader
paths receive no address residual.

This is a distinct factorized residual bus, not a scale patch or reuse of failed
weights. Zero initialization must make every route logit exactly equal to the
v1.1 architecture before fitting.

## Architecture and budget

- Parent: reconstructed confirmed witness-equality lineage; failed canary
  checkpoints are forbidden as initialization.
- New parameters: `14 x 12 x 14 + 12 = 2,364`.
- Expected complete system: `185,534,660` parameters.
- Expected trainable parameters: `11,131,868`.
- Headroom below 200M: `14,465,340`.
- Learned motor/reader parameters: zero/zero.

## Data, optimization, and gates

Use a new post-commit seed over the existing 48,000-row ER-TT training split.
Fit 10,000 families and probe 2,000 disjoint families, four views each. Four
same-seed arms receive identical confirmed-parent common initialization, rows,
family order, two epochs, 2,500 updates, 32 rows/update, AdamW LR `2e-4`,
100-step warmup, cosine decay, and no outcome supervision:

1. factorized treatment;
2. same-parameter baseline with the residual disabled;
3. structural-only route with the content dot product removed; and
4. shuffled-address control that rotates candidate ordinal by physical record
   position while preserving count, parameter count, and compute.

All unchanged gates must pass:

- packet/state/answer/joint each at least 85%;
- relation and complete witness-pointer rows each at least 90%;
- events and HALT each at least 95%;
- minimum cardinality-specific joint at least 75%;
- every hard output exactly invariant on 8,000/8,000 alpha recodes;
- oracle-route initial/relation/event/joint exactly 8,000/8,000;
- exact parameter certificate below 200M;
- unchanged confirmed parent; and
- custody exactly train-only/development/confirmation `1/0/0`.

Attribution also requires treatment witness rows to exceed the same-run
baseline and shuffled-address control by at least 0.5 percentage points. The
structural-only score is descriptive: a high score restricts the claim to a
finite syntax route rather than content-grounded language compilation.

Failure closes the factorized route. Passing authorizes only a separately
committed fresh-board development test. It is not natural-language, broad, or
unrestricted reasoning evidence.
