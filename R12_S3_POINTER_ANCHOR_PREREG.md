# R12 S3 Structural Pointer-Anchor Preregistration

**Status:** frozen before public score.

## Hypothesis

The rejected lexical confirmation missed two of 11,248 direction decisions
because its 0.5 soft-pointer-mass threshold sometimes declined to use an exact
known direction atom. The repair is threshold-free: take the frozen compiler's
global operation-kind pointer argmax and use a training-lexicon class exactly
when that token lies inside one unambiguous exact pattern. If no pattern contains
the anchor, or opposite classes overlap at the anchor, retain the neural kind
head. No weight is fitted and no confirmation board is read.

This is a structural interface repair, not a claim that unseen direction
semantics, schedules, halt, or open-language planning are solved.

## Frozen Inputs

- immutable 300k base and ordinary source compiler;
- frozen equivariant v1.1 executor state plus closure-complete S3 action;
- the exact 12-pattern training-only lexicon already qualified in v1.3;
- public compositional, lexical-OOD, and depth-3--8 development boards only.

The historical mass decoder remains unchanged and is not rescored. This arm
writes a new output directory and records `fit_updates=0` and
`confirmation_access=0`.

## Gates

The existing v1.3 public gates remain frozen without relaxation:

- compositional mean answer/state/transitions >=95%, every surface answer >=94%;
- lexical-OOD lexicon coverage <=5% and answer accuracy >=75%;
- depth lexicon coverage, direction, and amount each >=99.5%;
- depth mean answer/state/transitions >=90% / 88% / 80%, depth-eight answer >=85%;
- depth ordered answer/state/transitions each >=98%;
- depth gold answer >=98.5%, with exact 100% state and transition chains; and
- every output has zero fit and zero confirmation access.

A pass authorizes one wholly new seed-after-commit confirmation with the same
strict causal and exact-gold philosophy. A failure closes this repair. The
rejected confirmation board remains sealed and cannot be used for diagnosis,
threshold selection, or scoring.
