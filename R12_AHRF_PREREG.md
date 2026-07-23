# R12 Autocatalytic Hysteretic Relation Field Preregistration

**Status:** architecture and matched controls frozen at source commit
`c67d945`; no score-bearing AHRF result exists.

Pre-artifact launch `b4dcbf0` exhausted local MPS memory in a dense
parent-by-child membrane expansion before an optimizer update or output write.
Source `4fc5a11` replaces it with an exact gather after adding a fail-closed
one-child-per-typed-role validator. The full-geometry MPS canary completed
without OOM; this is a systems repair, not a score.

Pre-artifact launch `84eec59` was then stopped before any output or optimizer
update after a structural audit proved that its recurrent field propagated
only same-cell child state. It therefore had no path that could combine
relation cells `(i,k)` and `(k,j)` to update `(i,j)`, making exact relational
composition unrepresentable regardless of optimization. Source `c088261`
adds a learned object-equivariant dynamic triadic contraction over the two
typed argument-role membranes. The corrected full-geometry MPS canary wrote a
checkpoint and report with 291,666 added parameters and 125,373,330 total
parameters. Its one-update scores are not evidence and will not be reused.

The first launch after preregistration commit `26cb046` was stopped and its
ephemeral optimizer state destroyed before it wrote an output directory. An
independent structural audit found two more decisive faults: no runtime path
could transpose a live child relation for a converse node, and clamping exact
hard events before binary cross entropy killed the intended straight-through
terminal gradient. The same audit showed that the 16-step safety cap was below
the conservative propagation envelope of the proposed board.

Source `0dccab5` adds a generic direct/transposed channel for every typed child
role, uses an MSE terminal surrogate only during exact hard-event updates,
adds learned/false/zero triad modes with identical parameter counts, records
the transitive source set before and after training, and derives a fail-closed
minimum safety horizon from expression depth and fixed-point updates. Source
`1dd7a5d` additionally keeps the generic control's card encoder active through
object-marginal features and removes all primitive-classifier-head layers from
the warm start. A full-geometry 64-step MPS canary completed in 56 seconds with
316,824 standalone parameters and a hypothetical integrated total of
125,398,488. Its one-update scores are systems evidence only and will not be
reused.

A real-board batch-four canary then reached the local MPS 9.07 GiB memory cap
before any output write. Batch two completed the full 80-train/50-development
geometry without OOM or source drift. Source `c67d945` therefore freezes 2,000
field updates and 400 halt updates at batch two, preserving the intended 4,000
and 800 sampled examples respectively. Unsafe MPS watermark overrides are
forbidden.

## Question

Can a standalone architecture that fits within a sub-200M Shohin integration
budget infer fresh episode-local relation laws, propagate their consequences
through a host-compiled, source-deleted recursive graph, preserve facts as
write-once state, and decide when to halt without a host executor, operation
labels, an execution schedule, or a host convergence test?

This is a test of a bounded synthetic relational reasoning mechanism. A pass is
not evidence of unrestricted language reasoning, and AHRF is not yet connected
to the protected Shohin trunk.

## Architecture

The treatment is the Autocatalytic Hysteretic Relation Field (AHRF):

1. an object-equivariant opaque witness-card encoder with row, column,
   transpose, and triadic pair messages;
2. a node-equivariant recurrent graph field;
3. two typed operation-argument edge roles and one distinct equation-feedback
   edge role;
4. a learned object-equivariant dynamic triadic membrane contraction that
   combines `(i,k)` state from the first argument role with `(k,j)` state from
   the second argument role to drive `(i,j)`;
5. direct and generically transposed fact/membrane channels for every typed
   child role, allowing an opaque card to select relation orientation without
   a named converse primitive;
6. exact write-once fact and evidence latches with straight-through gradients;
7. continuous membrane state;
8. a learned event-triggered absorbing halt latch; and
9. a fixed maximum recurrence used only as a safety bound.

The runtime score path receives structural node kinds, graph links, equation
feedback links, root masks, constants, opaque witness cards, and object masks.
It receives no primitive identity, operation name, target relation, answer,
trajectory, iteration count, schedule, host-executor output, or convergence
flag. Every active node must reach a root, every opaque card must be used, and
padding/unused card arguments must be exactly zero.

Default `hidden_dim=64`, `card_rounds=2`, `max_steps=64` has 316,824 standalone
parameters and leaves 74,601,512 parameters under a hypothetical 200M
flagship-integration cap. This is budget accounting, not evidence that the
trunk and reasoner are integrated. The exact receipt is checkpointed and
independently testable.

## Initialization

The card encoder may warm-start only from a hash-bound CWEB treatment trained
after this source freeze with:

- width 64;
- two learned triadic rounds;
- no statistics-only architecture;
- no false- or zero-triad control; and
- exact source/config/checkpoint receipts.

Only the pair input, pair rounds, and witness encoder are copied. No primitive
classifier layer or output, target relation, host program, or halt parameter is
transferred.

## Board

The first pilot uses the hardened source-deleted Bekić board:

- factorial train orbits;
- held-out in-range, motif, scale, depth, and joint development cells;
- P, P-prime, equivalent P, constant-only rewire, and compose-only reversal;
- split-disjoint individual depth-two and depth-three motif receipts;
- held-out motif absent from every training arm;
- canonical recursive pressure requiring both variables to change, at least
  two convergence updates, and at least three total variable-change events;
- a fail-closed propagation envelope `max(depth * (updates + 1))`, with every
  admitted board required to fit within the frozen 64-step safety bound;
- fresh opaque slot IDs, node IDs, card order, witness order, object order, and
  varied input densities; and
- exact independent simultaneous and nested set oracles used only outside the
  model score path.

The clean source commit precedes the board/training seed. Development is
score-bearing for this pilot; no confirmation claim is authorized.

## Optimization

The frozen local pilot budget is batch two, 2,000 field updates, and 400 halt
updates. Every matched learned control receives the same sampled-example,
update, and recurrence budget.

1. Fit terminal root relations from final-state supervision only.
2. Use continuous monotone write events for the first 90% of field updates.
3. Use exact straight-through write events for the final 10%.
   Binary cross entropy is used for the continuous phase and MSE for the exact
   hard phase so incorrect hard bits retain a finite nonzero gradient.
4. Keep halt absorption disabled during field fitting.
5. Freeze the field and fit only the halt head from whether the model's own
   hard per-step root facts exactly equal the training target.
6. Evaluate with exact write events and learned halt enabled.

No intermediate node target, primitive label, host execution trace, or fixed
halt deadline may supervise the field.

## Frozen Controls

Every promoted run requires same-board, same-budget controls:

1. **no feedback:** remove equation-root-to-variable feedback links;
2. **no hysteresis:** replace write-once fact latches with ordinary overwrites;
3. **shuffled cards:** permute opaque cards between slots while preserving
   arity and all graph statistics;
4. **generic recurrence:** parameter/FLOP-matched recurrent field without
   aligned card-conditioned pair messages; the same card encoder remains
   trainable and supplies only object-marginal card features;
5. **false triad:** same-parameter, same-cost contraction using the wrong
   object-equivariant `A-transpose times B` alignment;
6. **zero triad:** evaluation ablation only; it is not a matched learned
   control because its triad parameters receive zero gradient; and
7. **fixed deadline:** disable learned halt and read only at the safety bound.

Identity-delay twins are a mechanics falsifier: terminal semantic facts must
remain equal while learned halt latency increases by the inserted relay depth.

## Gates

All must pass on at least five independent seeds before promotion:

- at least 99% exact terminal packets on train and development;
- at least 99% exact in every development cell and score-bearing arm;
- at least 99% learned halt with at most 1% safety exhaustion;
- 100% object, node-storage, card-order, and witness-order equivariance;
- 100% P/equivalent-P terminal agreement;
- treatment exceeds each nontrivial control by at least 20 percentage points
  in exact development packets;
- no source/target/covert-state validation failure;
- exact complete-system parameter receipt below 200M; and
- independent replay from the frozen source/checkpoint hashes.

Failure rejects the treatment on this board. Passing authorizes only a fresh,
preregistered transfer board spanning at least Horn closure, dataflow analysis,
and one non-relational family. No synthetic Bekić result, by itself, may be
called genuine general reasoning.

## Existing Evidence Boundary

The contextual witness-equivariant binder is a useful warm-start component but
not a decisive law-learning result: its exploratory shifted score is 99.61%,
while a statistics-only control reaches roughly 97.8%. The current card
ontology is therefore strongly marginal-solvable. AHRF must win through
recursive terminal execution and model-owned halt, not by citing the binder
classification score.
