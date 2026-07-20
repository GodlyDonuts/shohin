# R12 ER-CST Fresh Neural Qualification Preregistration

**Protocol:** `R12-ER-CST-v1.2`
**Status:** pre-source-freeze; no training seed drawn; development and confirmation unopened
**Absolute complete-system ceiling:** fewer than 200,000,000 parameters

## Question

Can Shohin infer three fresh, problem-local operations from determining
before/after witnesses, bind later opaque opcode uses to those inferred rule
cards, delete the source, and recurrently compose the cards through an internal
HALT before answering a separately rendered late query?

This is a bounded episodic semantic-binding test over the six permutations of
three symbols (`S_3`). A pass is evidence for fresh rule inference plus
source-deleted recurrent reuse. It is not evidence for arbitrary algebra,
unrestricted natural-language reasoning, planning, or general intelligence.

## Frozen Inputs

- Confirmed parent: SD-CST Complete Physical Fresh v1.3, checkpoint SHA-256
  `a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8`.
- Independent parent assessment SHA-256
  `4629a745f6eed2e388eb6e1f78b29dff346ee6939e21275ae6ff1d66719d3cb9`.
- Addressed board source commit
  `9cf9d043d0e86a30d18c6d5e3b838c80ec054d7c`.
- Board seed `8277659525319823840`; board report SHA-256
  `589b203fb4fec3c55b1b4d77efaf78d127ce127fc98633279c69b305dad2e704`.
- Train/development/sealed-confirmation rows: 48,000 / 2,048 / 2,048.
- Train/development/confirmation SHA-256 values begin
  `b5cb2f14` / `5cd0395f` / `7404b247`; current custody is `0/0`.

The training split contains compiler fields only. It contains no final state,
answer, trajectory, recurrent target, development target, or confirmation
target. Every family has four renderer views. Train and scored renderer
compositions, opaque names, prompts, families, and word 13-grams are disjoint.

## Architecture And Budget

The confirmed 125,081,664-parameter Shohin trunk is frozen. The inherited and
ER-CST compiler has 67,336,999 parameters. A 2,438-parameter tied motor applies
one selected `S_3` card at every recurrent step. An independent 835-parameter
reader maps only final categorical state and late-query position to one of
three answer roles.

| Component | Parameters |
|---|---:|
| Frozen Shohin trunk | 125,081,664 |
| Complete compiler | 67,336,999 |
| Tied rule-card motor | 2,438 |
| Independent final-state reader | 835 |
| **Complete system** | **192,421,936** |
| Trainable compiler + motor | 11,716,385 |
| Headroom below 200M | 7,578,064 |

The motor and reader receive only complete finite-domain table supervision:
36 state/card transitions and 18 state/query answers. They receive no program,
trajectory, development, or confirmation examples. Both tables must be exact
before compiler fitting continues. Their learned weights, not a host operation
table, execute the scored recurrence.

## Three Equal-Budget Arms

All arms start from byte-identical compiler, motor, and reader initialization,
use the same family minibatch order, and receive exactly 48,000 rows for two
epochs (3,000 optimizer updates).

1. **Treatment:** true determining witnesses and true rule-card labels.
2. **Family-deranged:** source is unchanged, but a family-stable random
   nonidentity rotation assigns witness-card labels to the wrong storage slots.
   Renderer views retain the same false mapping.
3. **Equality-ablated:** all six witness symbols in each before/after record are
   replaced independently while preserving every byte width and target. This
   retains layout and surface positions but removes the equality relation that
   determines the permutation.

Neither negative arm receives less optimization, fewer rows, a weaker model,
or a different renderer-consistency objective.

## Optimization

- Compiler: AdamW, learning rate `2e-4`, betas `(0.9, 0.95)`, weight decay
  `0.01`, 100-update warmup, cosine decay to zero, global gradient clip `1.0`.
- Batch: eight complete semantic families = 32 renderer rows per update.
- Epochs/updates: two / 3,000.
- Renderer consistency weight: `1.0` over initial state, rule cards, event-card
  references, HALT, and late-query distributions.
- Motor: AdamW, 1,000 updates, learning rate `0.003`, zero weight decay.
- Reader: AdamW, 500 updates, learning rate `0.005`, zero weight decay.
- Arithmetic: bf16 autocast on one H100-class GPU; losses and categorical
  decisions are accumulated/evaluated in float32 where implemented.

Supervision covers thirteen semantic-line pointers, declaration binding and
initial pointers, the late-query pointer, initial state, three inferred rule
cards, nine event references, nine HALT flags, and late-query position. Event
card identity is ignored only at the explicit HALT slot. Post-HALT source
records remain parser targets but cannot alter recurrent state after HALT.

## Custody And Scoring

The source commit is frozen and pushed before the training seed is drawn. The
job must verify exact source bytes, parent hashes, board hashes, confirmation
mode `0600`, bf16 CUDA, and the parameter certificate. It loads only
`train.jsonl` during fitting.

After all three arms finish, one read-only checkpoint containing every fitted
compiler/motor/reader state is atomically written and hash-bound with
development/confirmation access `0/0`. Only then may an `O_EXCL`, mode-`0444`
development ledger be created. The 2,048 development rows are opened once.
The sealed confirmation is never read by the pilot or assessor.

An independent assessor recomputes exact packet, pointer, state, answer, joint,
depth, and renderer metrics from raw categorical evidence and verifies all
artifact hashes, finite motor/reader tables, parameter counts, and custody.

## Frozen Development Gates

- Treatment exact packet, state, answer, and joint: each at least 90% overall.
- Treatment joint: at least 85% for every unseen renderer composition and at
  least 80% at every depth one through eight.
- Every packet field: at least 95% overall.
- Every pointer family: at least 90% overall.
- Treatment packet and joint advantage: at least 50 percentage points over
  each equal-budget negative arm.
- Each negative arm: packet at most 35% and state at most 40%.
- All 36 motor and 18 reader cells exact for every arm.
- Confirmed parent digest unchanged for every arm.
- Complete system strictly below 200M parameters.
- Development/confirmation custody exactly `1/0` after scoring.
- Pilot metrics and gate vector must match independent recomputation exactly.

Only if every scientific and assessor gate passes may a separate committed
one-read confirmation evaluator be implemented. Thresholds will not be relaxed,
the board will not be rescored, and a failed development board will remain
closed.
