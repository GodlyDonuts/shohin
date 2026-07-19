# R12 RGDE Recurrent-Depth Confirmation Preregistration

**Status:** source frozen before production seed selection

**Claim class:** confirmation of a source-deleted recurrent execution component
at depths three through eight. Operation count and halt remain externally
scheduled. A pass is not autonomous language reasoning or learned halting.

## Question

RGDE v1.1 learned one shared update cell only from independent atomic examples
and composed it twice at 99.707% answer accuracy. The fresh question is:

> Does that exact frozen cell preserve model-owned permutation state when it is
> reused three to eight times on unseen entity names, unseen language-factor
> combinations, and semantically matched long-program twins?

No fit, calibration, retry, seed sweep, or checkpoint selection is permitted.

## Commit-before-seed board

The production seed has no source-code default and must be generated only after
this preregistration, generator, evaluator, tests, assessor, and Slurm job are
committed and pushed. The board contains 512 semantic quartets / 2,048 rows,
balanced across depths 3--8. Every row has:

- three fresh paired nonce names absent from all public factorized data;
- a long program and exact terminal state agreed by two CPU executors;
- canonical/paraphrase surfaces with identical semantics;
- an operation-order twin and entity-binding twin with the same normalized word
  bag but a different answer at the fixed query;
- only known language atoms, but factor combinations absent from public train
  and development;
- zero exact-prompt, word-13-gram, entity-name, and factor-combination overlap
  against public train/compositional/lexical-OOD data.

The old factorized confirmation is neither an input nor an audit source. The
generator rejects any public path whose filename contains `confirmation`.

## Packet-stream boundary

Each long program is rendered as a stream of ordinary two-operation source
cards. The already qualified compiler processes one card at a time. Its
set-valued lexical/contextual packet is gathered and both source memories are
deleted. The frozen tied executor then applies the packet's active operations
to one persistent `3 x 3` state. An odd final card contains an ignored filler
operation; the host supplies only the declared active count. The final card's
model-owned query packet consumes the final state.

Host code supplies card order, active operation count, and `halt_after=depth`.
It does not decode operation direction, entity, amount, query, state,
transition, or answer. This isolates recurrence depth; halting is explicitly
out of scope.

## Immutable system

- tied atomic executor file SHA-256:
  `adb6323202f6d25280f3a1cfd34a5b88fbc876331643726e38db389ead746b74`;
- tied state SHA-256:
  `d31fd3e6150dd352cd0eea5063f960393e9017ac3ab729b5c536fd1f1c432184`;
- raw-300k base, ordinary compiler, tokenizer, packet mode, widths, and every
  parameter are unchanged from the passed v1.1 development run;
- total system parameters remain 135,180,829; trainable parameters are zero.

Frozen source SHA-256 values are generator `790279d9...`, generator tests
`51852281...`, evaluator `6e6a8211...`, assessor `125eef0f...`, and Slurm job
`1b311def...`. Seventeen focused CPU tests plus Ruff, `py_compile`, shell
syntax, and `git diff --check` pass before production seed selection.

## One-shot evaluations

One Slurm job serially writes four immutable outputs:

1. predicted packet, no intervention;
2. gold packet diagnostic rescore;
3. globally deranged operation stream within the same depth;
4. globally deranged query within the same depth.

Every intervention changes all 2,048 semantic keys and swaps only the declared
bounded field. Gold roles/kinds never enter the primary result.

## Frozen gates

The depth mechanism passes only if all are true:

1. every board CPU/data/custody gate passes with zero old-confirmation access;
2. depths 3--4 each reach at least 95% answers, 95% exact final state, and 90%
   all-transition exactness;
3. depths 5--6 each reach at least 90% answers, 90% final state, and 80%
   all-transition exactness;
4. depths 7--8 each reach at least 85% answers, 85% final state, and 70%
   all-transition exactness;
5. every surface reaches at least 85% answers and 400/512 quartets have all
   four answers correct;
6. overall predicted answers/final state are within three points of gold;
7. entity-match and amount accuracy are each at least 98%;
8. operation derangement is at most 40% answers/final state and loses at least
   50 points on each relative to primary;
9. query derangement is at most 5% answers, loses at least 90 points, and
   changes exact final state by at most one point;
10. one job completes `0:0`; every output binds board/report/evaluator/base/
    compiler/executor/tokenizer, records 2,048 effective interventions where
    requested, and records zero old-confirmation access.

Failure at shallow depth rejects packet-stream confirmation. A monotonic depth
collapse localizes recurrent-state drift. Gold-only success localizes compiler
transfer. A clean pass establishes reusable bounded state updates to depth
eight, but still does not establish free-form planning, learned stopping, or
natural-language chain-of-thought.
