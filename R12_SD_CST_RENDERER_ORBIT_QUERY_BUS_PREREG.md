# R12 SD-CST Renderer-Orbit Query Bus Preregistration

**Status:** v1 closed as an unscored numerical-loss failure; optimization-only
v1.1 implemented and locally audited before its source freeze or seed

**Parent:** rejected projected fresh v2; retained exact source-deleted executor
and v2 treatment checkpoint

**Claim class:** renderer/query identifiability on consumed training rows only;
no reasoning score

## 1. Evidence that fixes the target

Projected fresh v2 is not execution-limited. Every one of its 672 exact fresh
packets executes exactly. State is 89.193% overall and 100% on seven of eight
variants, but exact packets are 29.167%, answers are 33.116%, and the late query
is exactly 33.333%. The held-out paraphrase renderer has 0/288 exact packets and
39/288 exact states.

The board source exposes the identifiability error. All 48,000 training rows use
one direct program renderer and one training-only query frame. The only program
paraphrase appears in development, and development also changes the query marker
and verb phrase wholesale. V2 trains only the projected exact-surface binding
parameters; its parent source encoder, line/kind/amount heads, and direct
three-class query compiler remain frozen. Additional executor capacity cannot
repair these interfaces.

## 2. Hypothesis

Renderer transfer should be trained as a finite group-action problem rather
than left to incidental OOD generalization. The treatment receives multiple
surface views of the same latent program and is penalized when its categorical
packet logits differ across views. Complete renderer combinations are held out,
while every declaration, event, and query atom appears in both train and
holdout. This makes the pilot a test of compositional renderer recombination,
not unseen-word semantics.

The query path receives a stricter intervention. Contextual memory may select
the byte span expressing the requested ordinal, but the final three-class motor
receives only a weighted sum of raw, position-free byte embeddings. It cannot
read the query template, absolute position, split marker, or contextual memory
after selection. Pointer and ordinal classification are separately supervised.

## 3. Architecture and exact resource boundary

`RendererOrbitGroundedCompiler` loads the exact 20,955,890-parameter projected
parent and v2 treatment binding state. Every inherited parameter remains frozen.
It adds:

- a 257-entry, 512-wide byte embedding;
- a 640-entry, 512-wide position embedding;
- eight pre-norm 512-wide, eight-head transformer layers with 2,048-wide MLPs;
- a 512-to-384 residual projection and scalar tanh gate initialized to zero;
- a contextual ordinal pointer; and
- a position-free raw-byte value projection, normalization, and three-class
  ordinal motor.

Exact accounting from live module construction is:

| Component | Parameters |
|---|---:|
| nominal Shohin base | 125,081,664 |
| complete renderer-orbit compiler | 47,621,366 |
| motor | 19,206 |
| reader | 835 |
| **complete deployed system** | **172,723,071** |
| strict-200M headroom | 27,276,929 |
| trainable renderer/query front end | 26,665,476 |

The complete system is strictly below 200,000,000. This pilot does not consume
the remaining headroom. Historical sub-150M experiment contracts remain closed
and unchanged.

## 4. Renderer orbit

Renderer surface is the binary product of three factors:

1. direct bindings versus reverse registry declaration;
2. event/move/by versus action/send/for clauses; and
3. position-question versus slot-report query.

The four even-parity combinations are fit views. The four odd-parity
combinations are held-out views. The sets contain no common complete renderer,
but each set contains both values of every factor. Renderer names exist only in
metadata and never in model text.

The training-only pilot SHA-orders the already-consumed 48,000 v2 training
semantics, uses the first 12,000 latent programs for fit and the next 2,000 for
held-out evaluation, and renders four views per latent program. No development,
confirmation, oracle answer, final state, or recurrent trajectory can be read.

## 5. Training contract

- initialization: exact byte parent SHA `e5f87a1d...` plus exact v2 treatment
  checkpoint SHA `1d338651...`;
- trainable parameters: the exact 110 renderer-orbit/ordinal parameter names;
- frozen parameters: all inherited source, binding, packet, motor, reader, and
  Shohin parameters;
- two epochs over 12,000 semantic families;
- family batch size eight, four renderer views per family;
- AdamW, lr `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`;
- 100-update warmup, cosine decay, gradient clip `1.0`;
- supervised losses for initial, kind, identity, amount, query, all program
  address classes, and query ordinal address; and
- Jensen-Shannon categorical consistency over initial/kind/identity/amount/query
  logits across the four views of each semantic family, weight `1.0`.

Only compiler fields and byte spans are labels. No execution-derived target is
available.

### V1 numerical closure and v1.1 repair

Exact v1 source commit `a16a555cae21dca845689f8ddc119b1d8f9a0f91` and seed
`7492631734612190994` reached one training-only epoch in job `694059`. The
inherited uniform span cross-entropy multiplied zero target mass by masked
`-inf` log probabilities under bf16. Event-address loss and therefore total
loss were infinite. The run was canceled before epoch two and before creating
an output directory, checkpoint, or report. It had no route to development or
confirmation and cannot be interpreted as a mechanism result.

V1.1 changes only the mathematically equivalent loss arithmetic: logits are
promoted to float32, target entries are selected with `torch.where`, and active
target spans must be nonempty and finite. A regression containing explicit
`-inf` masked logits verifies finite value and gradients. Architecture, data,
partition, labels, objective weights, optimizer, update count, gates, and claim
boundary are unchanged. V1.1 requires a new source commit and post-commit seed.

## 6. Training-only gates

Every gate must pass on every one of the four odd-parity renderer combinations:

1. initial state at least 95%;
2. complete kind at least 95%;
3. complete active identity at least 90%;
4. complete active amount at least 95%;
5. late query at least 99%;
6. query ordinal pointer at least 99%;
7. complete packet at least 80%;
8. exact complete-system parameter count below 200M; and
9. development and confirmation access exactly zero.

Failure rejects or revises the front end without a fresh board. Passing permits
only full fresh-board preregistration with equal-budget controls; it is not a
reasoning result.

## 7. Required fresh-board controls after a pilot pass

A future score-bearing board must be generated after a new source commit and
must include:

- treatment with correct same-semantics renderer families;
- equal-parameter/equal-view arm with renderer consistency weight zero;
- equal-parameter wrong-family arm whose consistency pairs different latent
  programs;
- direct-only equal-update arm;
- row-shuffled compiler supervision;
- query-context-only arm with the selected raw-byte value deleted;
- query-value-only oracle-pointer diagnostic, quarantined from attribution;
- the exact retained source-deleted executor and all v2 packet interventions;
- renderer-orbit holdout, name, prompt, sequence, and 13-gram audits; and
- one development read, with sealed confirmation opened only after every frozen
  development gate passes.

The primary fresh claim requires treatment to beat both the no-consistency arm
and wrong-family arm, not merely to fit a larger synthetic corpus.

## 8. Honest boundary

The mechanism is a project-specific combination of established transformer,
attention, contrastive-consistency, and discrete packet components. No prior-art
novelty claim is made. A pilot pass would establish only that a constrained
sub-200M system can learn recombinable renderer coordinates and a template-
blocked query bus on this finite language. A confirmed fresh-board pass would
still not establish unseen-word semantics, arbitrary natural language,
self-generated plans, or general reasoning.
