# R12 ER-TT Opcode-Coupled Route Train-Only Preregistration

**Status:** pre-source-freeze; train-only qualification; no development or
confirmation access is permitted.

## 1. Hypothesis

Ordinal-route v1.2 scores retained witness candidates but does not positively
identify the excluded candidate as the opcode. The fresh-board rejection shows
the resulting endpoint-exclusion degeneracy exactly on opcode-middle rules.

V1.3 changes one factor. For each candidate exclusion, the route score is the
sum of its ordered witness scores plus the existing model-owned rule-opcode
score on the complementary candidate. No renderer metadata, target outcome,
host parser, executor feedback, retry, or repair enters the compiler.

## 2. Equal-budget arms

Both arms reconstruct the same qualified v1.2 trainable state and fit the same
10,000 fresh-training families in the same order for two epochs / 2,500
updates.

1. `opcode_coupled`: complement opcode weight 1.
2. `legacy_uncoupled`: complement opcode weight 0, exactly recovering the v1.2
   unscored-exclusion lattice.

Every other parameter, optimizer, loss, batch, source row, and update is
identical. The complete system remains 185,532,296 parameters; the executor
motor and reader remain parameter-free.

## 3. Train-only transfer probe

Two thousand disjoint families from the already-public fresh training split
are withheld from fitting. They are scored mechanically, never with a stored
outcome oracle, in two views:

- the four original training renderer compositions;
- four deterministic opcode-relocation twins rendered from the same training
  semantics with the complementary renderer coset.

The second view moves the opcode between first and middle positions and flips
event/query correlations without using development or confirmation bytes. New
distractors and storage orders are deterministically generated from the
post-commit canary seed. All rows are reparsed and independently executed before
model evaluation.

## 4. Frozen gates

The canary advances only if all gates pass:

1. coupled canonical and relocation packet/state/answer/joint, relation rows,
   and complete witness pointers are each at least 99%;
2. minimum per-cardinality and per-renderer joint is at least 99% in both
   views;
3. all semantic packet fields are invariant across all eight views for every
   one of 2,000 families;
4. complete alpha recoding and distractor-only rotation are exact on every
   relocation row;
5. source-free identity collapse is at most 10% joint;
6. coupled relocation joint exceeds legacy relocation joint by at least 20
   percentage points and the legacy arm is at most 80%;
7. both arms share byte-identical initialization, preserve all excluded parent
   state, and execute exactly 2,500 updates;
8. the exact parameter certificate remains below 200M with zero learned
   motor/reader; and
9. development and confirmation reads remain zero.

Failure closes v1.3 before any new board. Passing authorizes only a separately
frozen fresh-board source and seed. It does not authorize reuse of the closed
v1 board or opening its confirmation.

## 5. Claim boundary

A pass would establish that coupling model-owned opcode grounding to the
complement witness route repairs renderer-position transfer on consumed
training semantics. It would not establish fresh semantic transfer, natural
language grounding, arithmetic, planning, or general reasoning.
