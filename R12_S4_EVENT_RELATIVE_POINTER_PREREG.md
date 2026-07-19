# R12 S4 Event-Relative Pointer Preregistration

## Status

Frozen source and gate specification after S4 v1/v1.1 rejection and before v2 training, fresh-board
generation, production seed selection, or v2 score access.

## Causal diagnosis

S4 v1 learns exact event count on 2,048/2,048 public-development sources and exact execution on all
1,932 structurally valid tapes. Gold intro/query boundaries raise exact programs to 97.217%. The
shared role head fails by fragmenting variable-width roster spans and by giving every event the same
unconditioned entity/literal score. The zero-fit width decoder confirms that global role maxima do
not contain enough boundary information.

## Treatment

Freeze the entire v1 treatment parser, including its base model, memory encoder, event-count role
head, and semantic heads. Add only:

- three roster start and three roster end pointer heads;
- one query start and one query end pointer head;
- event-conditioned entity start/end and literal start/end pointer query/key projections.

For each event, the query is the mean frozen memory at its direction span. It scores every source
token as an argument start or end. The same tied projections serve every event and therefore admit
arbitrary event count. Training uses gold direction spans only to define the supervised query;
inference uses model-discovered direction anchors in source order. The pointer heads receive no
depth, operation index, answer, final state, or gold event count.

## Controls

1. **Frozen v1 parser:** the already scored favorable shared-role baseline.
2. **Shuffled pointer supervision:** identical frozen v1 initialization, architecture, parameters,
   examples, updates, and optimizer; all pointer targets are permuted within source.
3. **Gold tape sanity:** locked S3 execution of exact source events.

No joint v1 fine-tuning, extra epoch, width change, decoder sweep, or result selection is allowed.

## Fresh-board rule

The old S4 development board is closed. After this preregistration, generator, model, trainer,
evaluator, assessor, tests, and jobs are committed, draw one random seed and generate a new 2,048-row
development board. Its names, exact prompts, word 13-grams, and factor signatures must be disjoint
from the full S4 v1 train/development corpus and all supplied public compiler/executor boards. V2 may
read that board once per frozen arm. No post-score repair or rescore is admissible.

## Frozen gates

- exact model-owned event count at least 98% overall and 95% at every depth;
- exact program at least 95% overall and 90% at each depth 5--8;
- exact locked-S3 state and answer at least 95% overall and 90% at depth eight;
- exact initial roster at least 95% overall;
- shuffled exact programs at most 40%;
- gold tape state/answer at least 99%;
- strict total parameters below 150,000,000;
- development access exactly one and confirmation access zero.

A pass authorizes one separately frozen confirmation board. It does not establish unseen action
semantics, planning, free-form reasoning, benchmark improvement, or novelty.
