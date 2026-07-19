# R12 S4 Self-Delimiting Event Tape Preregistration

## Status

Frozen theory and interface specification before corpus generation, neural fit, or score access.
This lane follows the confirmed pointer-anchor S3 v1.4 executor and must not read any sealed S3
confirmation board.

## 1. Capability theorem

For a source containing an initial three-entity roster, a finite ordered sequence of complete
movement clauses, and one terminal position query, define an **event tape** as the ordered list of
complete triples `(direction, entity, amount)` recovered from the source. If a source-only parser
recovers every triple and the query, the frozen S3 v1.4 transition table returns the exact terminal
state and answer for any finite tape length. No external operation count is required: the event
sequence terminates after the final complete recovered triple.

The bounded empirical claim is therefore:

> A frozen-Shohin token parser can recover a variable-length event tape from one unpadded source,
> and the already confirmed S3 executor can consume that model-owned tape through held-out depths.

This is not a claim of planning, free-form language reasoning, unseen action semantics, or a novel
reasoning primitive.

## 2. Axiomatic primitive

An event is valid iff the parser emits one contiguous `kind` span, one contiguous `entity` span,
and one contiguous `literal` span whose source intervals belong to the same movement clause. Events
are ordered by source position. The tape halts when no later complete event exists. The query is a
separate terminal span.

No `active_operations`, chunk index, fixed operation slot, filler operation, depth label, or host
slice may enter inference. Source deletion occurs after the event tape and query packet are built.

## 3. Equivalence dossier

The event parser is a conventional bidirectional token tagger plus deterministic span grouping. It
is equivalent in representational class to semantic-role labeling followed by a finite-state
transducer. The S3 consumer is the already confirmed exact categorical register. This lane tests an
autonomous interface boundary; it does not claim a new computational primitive.

Favorable controls:

1. **Gold event tape:** exact spans and semantics; ceiling for the frozen S3 consumer.
2. **Fixed eight-slot parser:** favorable externally sized parser with one slot per maximum depth.
3. **Host-count parser:** token predictions grouped using gold operation count; isolates counting.
4. **Shuffled role supervision:** equal architecture and budget with source/role association broken.

## 4. Exact collapse test

If the model-owned variable-length parser does not beat shuffled supervision, or if host-count
grouping materially rescues it, the autonomous schedule claim collapses. If gold events fail, the
failure is downstream of parsing and S3 v1.4 must not be blamed without a new causal audit.

The architecture novelty claim is rejected in advance because sequence tagging and finite-state
grouping are established methods. Only the bounded removal of the source-external schedule oracle
can pass.

## 5. Prior-art boundary

Known semantic-role labeling, token classification, pointer parsing, monotone alignment, finite-
state transduction, and exact symbolic execution are controls or implementation tools. No result may
be described as a new reasoning architecture solely because these components are connected.

## 6. Finite CPU falsifiers

Before a neural fit:

1. Audit the old chunked board and prove that the padding label conflicts with a legitimate event
   under the same equivariant semantic signature.
2. Build unpadded whole-source rows at depths 1--8 and prove exact dual-executor agreement.
3. Recover every event with a gold-span finite-state parser and prove that event count equals depth.
4. Prove that deleting or duplicating one recovered event changes the semantic program and that
   matched order/binding twins remain behaviorally separated.
5. Prove the longest tokenized source fits Shohin's 2,048-token context.

## 7. Frozen controls and development gates

Train only on depths 1--4. Evaluate without refitting on depths 3--8, reporting depths 5--8
separately. Development advances only if:

- model-owned event count is exact on at least 98% overall and 95% at every depth;
- exact event programs are at least 95% overall and 90% at every held-out depth;
- frozen-S3 answers are at least 95% overall and 90% at depth eight;
- host-count grouping improves exact programs by less than two points;
- shuffled supervision is at most 40% exact programs;
- gold events retain at least 99% exact state and answer;
- total parameters remain strictly below 150,000,000;
- no sealed confirmation bytes are read.

These are public-development gates. They authorize at most a separately frozen, freshly seeded
confirmation protocol.

## 8. Score-blind confirmation rule

No confirmation corpus exists at preregistration. If every development gate passes, source,
generator, evaluator, assessor, and job bytes must be committed before drawing one production seed.
That seed may be used once. A failure is sealed; thresholds cannot be relaxed and the board cannot
be rerun.

## Existing-board no-go

The old depth board renders every chunk with exactly two normal-looking operations and stores the
true count in `active_operations`. Odd final chunks pad with `(left, initial_entity_0, 1)`, while
the same semantic operation can be a legitimate second update. Therefore a semantic equivariant
halt classifier cannot recover the hidden label exactly from that board. Training a halt head on it
would at best exploit incidental source identities or metadata. S4 replaces the corpus rather than
fitting that invalid target.
