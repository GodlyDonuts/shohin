# R12 S4 Pointer-Anchored Event Tape Repair

## Status

Frozen zero-fit public-development repair after S4 v1 treatment evaluation and before any repaired
score. No model weight, corpus row, optimizer, update count, seed, threshold, or confirmation input
changes.

## Failure diagnosis

S4 v1 predicts exact event count on 2,048/2,048 held-out rows and is fully correct on every valid
tape, but strict decoding invalidates 116 rows: 66 initial-roster cardinality errors, 47 event-role
component-cardinality errors, and three entity-identity errors. Gold initial/query boundaries lift
exact programs from 94.336% to 97.217%, every depth at least 96.471%. Shuffled supervision remains
zero. The remaining miss is hard argmax span fragmentation, not count or event semantics.

## Sole repair

Build a structural lexicon from the admitted training split only:

- exact known direction token patterns and class;
- exact amount and query-literal token patterns and value;
- the set of training entity-span token widths.

At inference:

1. Each of the three schema-fixed initial-role global pointer anchors expands to the highest-scoring
   training-width window that contains it.
2. An event exists only when an exact direction pattern contains a token whose model argmax role is
   `event.kind`. These anchored patterns, ordered by source position, define event count.
3. Inside each adjacent anchored-event interval, exact occurrences of the three model-predicted
   initial token sequences compete under `event.entity` role score; exact known literals compete
   under `event.literal` score.
4. The query-role global anchor expands only to an exact training query-literal pattern.
5. Any missing, overlapping, ambiguous, or duplicate structural selection is invalid. No gold depth,
   count, span, entity, event, state, or answer enters inference.

This is a deterministic structured decoder over model logits, equivalent to lexicon-constrained
semantic parsing. It is not a new reasoning primitive.

## Frozen gates

The original S4 gates remain unchanged: at least 98% exact count overall and 95% every depth; at
least 95% exact programs overall and 90% every held-out depth; at least 95% answers overall and 90%
at depth eight; gold-count rescue below two points; shuffled exact programs at most 40%; locked S3
gold sanity; total parameters below 150M; zero confirmation access.

V1.1 may run once on the same public development rows after source, lexicon builder, evaluator, and
this repair are committed. A pass authorizes only a separately frozen fresh confirmation protocol.
