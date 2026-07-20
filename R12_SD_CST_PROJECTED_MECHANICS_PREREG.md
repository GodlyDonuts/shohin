# R12 SD-CST Projected Source-Deletion Mechanics Gate

**Status:** preregistered training-only causal gate; no scored access authorized

## Fixed input

The dedicated projected binding pilot is a training-only pass:

- source commit `9bd2e04ea93406eb50a6fd112cd844892b72a7c4`;
- pilot seed `6715972906370623241`;
- compiler checkpoint SHA-256
  `f347d1aea90dd3c60f7500167c7c22884451b365880259698306c6fce8ab10f3`;
- report SHA-256
  `5d6be14798af3a75781898c6405e956fe9eb040e861ee63e669e7b87e7fa6f32`;
- 8,000/8,000 held-out consumed-training whole tapes, initial states,
  identities, kinds, amounts, and queries;
- declaration, initial-occurrence, and event-occurrence all-slot pointers at
  7,999/8,000, 8,000/8,000, and 7,998/8,000; and
- all 14 frozen pilot gates passed.

The complete system is 146,057,595 parameters. This mechanics gate preserves
the pilot's stricter sub-150M comparison contract even though future systems now
have a hard global ceiling strictly below 200M.

The 8,000 rows come from the already consumed training split and were observed
during the pilot. They are mechanics data, not an independent score.

## Source-deletion contract

For each batch, the compiler process must:

1. compile the program alone;
2. convert its output immediately to a CPU `uint8` `HardProgramTape` containing
   exactly 25 categorical elements per row;
3. poison and destroy program IDs, masks, logits, pointers, memory, and the full
   compiler output;
4. only then disclose and compile the separate late query to one CPU `uint8`;
5. poison and destroy query IDs, masks, memory, logits, and output; and
6. serialize only the 26 categorical bytes.

A separate process imports no projected compiler, board, parser, row-ID,
target, or source module. It may read only the 26-byte packet bundle and the
independently fitted motor/reader weights. It executes the same production
`rollout_hard_categorical` function used by `SDCSTSystem.rollout_hard`.

The motor receives only the complete 72 non-STOP state/action cells plus six
STOP cells and must fit 78/78 at lr `0.003` in at most 1,000 updates. The reader
receives only the 18 state/query cells and must fit 18/18 at lr `0.005` in at
most 500 updates. Neither receives an episode, trajectory, final state, or
answer from the 8,000 programs.

## Compiler controls

The frozen compiler is run without further training on:

- canonical rows;
- deterministic equal-width global alpha renaming;
- one valid pre-STOP event-name counterfactual per row;
- a declaration-role swap with the corresponding abstract-role oracle; and
- semantics-preserving reversal of all eight physically stored event lines.

Canonical whole tape and every late-query byte must be 100%. Alpha rename and
line relocation must preserve at least 99% whole tapes. Event-name and
declaration-role counterfactuals must reach at least 95% of their changed whole
tapes. The projected pilot's untrained prefit whole-tape rate must remain at
most 1%, preventing a capacity-only interpretation.

## Source-blind execution controls

Every arm below must match an independently implemented categorical oracle on
final state, answer, their joint result, all eight intermediate states, and all
eight alive bits for every row:

- canonical packet;
- all five compiler controls;
- initial-state rotation;
- first-active event-kind flip;
- first-active identity rotation;
- first-active amount flip;
- late-query rotation;
- categorical state swap after step zero;
- reset-to-initial before every active transition;
- frozen state;
- post-STOP operand perturbation;
- forced-alive execution of that post-STOP suffix;
- operand suffix swap from slot four; and
- whole-program packet shuffle while keeping recipient queries.

The motor must be called exactly eight times. Every observed STOP-position
bucket must be 100% exact. Normal and post-STOP-perturbation arms must halt;
post-STOP perturbation must be exactly invariant. Query rotation must change
every answer while preserving state.

To reject vacuous controls, each initial/kind/identity/amount intervention must
create at least 1,024 changed state-or-answer oracles, the state swap at least
512 changed states, reset/freeze/force-alive at least 20% changed states, and
suffix swap at least 10%. The shuffled packet may retain at most 25% original
states and 45% original answers.

## Parameter and custody gates

- Exact input hashes above and strict checkpoint loading are mandatory.
- Complete parameters must remain below both 150M for this comparison and the
  global 200M ceiling.
- Development and confirmation access remain `0/0`.
- The packet and executor-output hashes are recorded.
- Any failure closes or revises mechanics before a fresh board.

## Claim boundary

A pass establishes a hard, source-deleted, causally intervenable finite-state
execution path driven by a learned compiler on consumed training mechanics. It
does not establish fresh-distribution generalization, natural-language breadth,
or broad native reasoning. Only a new post-commit board with matched controls
can make the next claim.
