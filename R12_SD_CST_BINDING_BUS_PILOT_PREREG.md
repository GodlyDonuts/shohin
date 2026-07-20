# R12 SD-CST Content-Addressable Binding Bus Training Pilot

**Status:** frozen training-only mechanism admission; development and sealed
confirmation access are forbidden

## 1. Measured diagnosis

The byte-addressed SD-CST compiler pilot on job `693969` solved source
localization and local event extraction but did not solve variable binding. On
the deterministic 8,000-row held-out partition of the already-consumed training
split it reached:

- all nine line pointers: 8,000/8,000;
- raw event kind with exactly one raw STOP: 8,000/8,000;
- event amount: 8,000/8,000;
- late query: 8,000/8,000;
- event identity: 5/8,000 = 0.0625%;
- initial state: 1,338/8,000 = 16.725%; and
- raw whole tape: 0/8,000.

Report SHA-256 is
`0bd0b6bbbc68f904ce9fdc06e35e5484114a66db85a3a20af528a9c05e86766e`.
The result rejects more generic depth or another grammar decoder. The remaining
failure is exact cross-occurrence binding: map arbitrary instance-local entity
names in the declaration, initial ordering, and event mentions onto the same
three anonymous roles.

## 2. Falsifiable mechanism

The pilot adds a **content-addressable binding bus** to the admitted
byte-addressed compiler:

1. three model-owned declaration pointers select the three binding-name spans;
2. three model-owned initial-order pointers select the three name occurrences
   in initial order;
3. eight model-owned event pointers select each event's entity occurrence;
4. selected spans are converted into shared, position-free fingerprints by
   pooling trainable UTF-8 byte-bigram embeddings over adjacent byte pairs for
   which both bytes are selected;
5. cosine similarity between each occurrence fingerprint and each declaration
   fingerprint emits anonymous role logits; and
6. the six-way initial-state logit for a permutation is the sum of its three
   occurrence-to-role match logits.

The last selected byte cannot import its following delimiter: a bigram is
weighted only when both adjacent positions are selected. The same fingerprint
function and projection are reused at every declaration, initial, and event
occurrence. No name dictionary, tokenizer vocabulary identity, row metadata,
gold role, state, answer, executor output, recurrent trace, repair, retry, or
search is available at inference.

This is an explicit structural equality prior. It tests exact repeated surface
binding only; it is not a claim of aliases, pronouns, coreference, or semantic
entity resolution.

## 3. Frozen architecture and size

The parent byte compiler remains six 384-wide/eight-head source layers, nine
line-address slots, two slot-mixing layers, and unchanged kind/amount/query
heads. The binding bus adds:

- 14 learned pointer queries: three declaration, three initial, eight event;
- a 65,537-entry by 96-wide byte-bigram embedding;
- one shared 96-by-96 bias-free fingerprint projection; and
- one learned positive similarity scale.

The compiler has **20,513,138** parameters. Complete accounting is:

| Component | Parameters |
|---|---:|
| Frozen Shohin trunk | 125,081,664 |
| Binding-bus compiler | 20,513,138 |
| Retained exact motor | 19,206 |
| Retained reader | 835 |
| **Complete system** | **145,614,843** |
| **Headroom below 150M** | **4,385,157** |

The training-only pilot instantiates the compiler but does not load or alter the
protected Shohin checkpoint. User authority permits future systems strictly
below 200M; this pilot intentionally keeps the earlier 150M comparison gate so
its result isolates the binding mechanism rather than a capacity increase.

## 4. Frozen data and optimization

- Input is only `train.jsonl` from the permanently consumed SD-CST v1.1 board.
- The board receipt and train SHA are verified before parsing.
- Rows are sorted by `sha256(row_id)`; first 40,000 fit, remaining 8,000
  training-held-out.
- Development and confirmation paths are never accepted by the program.
- Four epochs, batch 64, AdamW, lr `3e-4`, betas `(0.9, 0.95)`, weight decay
  `0.01`, 100-update warmup, cosine decay, and gradient clip `1.0` are fixed.
- Losses are initial state, raw kind, active identity, active amount, late
  query, and four uniform exact-span address losses: line, declaration,
  initial occurrence, and event occurrence.
- Kind weights are `[1, 1, 4]`.
- The seed is drawn only after this source and the exact job script are
  committed and pushed.

Span targets are allowed only because this is a compiler-mechanism pilot on
consumed training data. A later score-bearing experiment must use model-owned
pointers without target spans at inference and a fresh post-commit board.

## 5. Immutable pilot gates

All gates are evaluated on the 8,000-row training-held-out partition. The
mechanism advances only if every gate passes:

1. all line pointers exact on at least 90%;
2. all declaration pointers exact on at least 90%;
3. all initial-occurrence pointers exact on at least 90%;
4. all active event-occurrence pointers exact on at least 90%;
5. initial state exact on at least 80%;
6. raw event kind exact on at least 90%;
7. active event identity exact on at least 80%;
8. active amount exact on at least 90%;
9. late query exact on at least 98%;
10. raw whole tape exact on at least 60%;
11. every row emits exactly one raw STOP without constrained repair;
12. complete system remains strictly below 150,000,000 parameters; and
13. development/confirmation access remains `0/0`.

No constrained one-STOP score can promote the mechanism. Failure revises or
rejects this binding bus without opening a scored split.

## 6. Required evidence after a pilot pass

A pass is not a native-reasoning result. Before any scored board, a separate
post-result source freeze must add:

- shuffled role-label and declaration-swap controls;
- no-address and address-permutation controls;
- identical-name positive and same-length/different-name hard negatives;
- source-position relocation and delimiter-change invariance;
- binding-fingerprint swaps that predictably swap only the addressed entity;
- a source-deletion check that prevents the motor/reader from receiving bytes,
  pointers, or contextual residuals;
- end-to-end integration with the retained tied motor, STOP gate, and reader;
  and
- a fresh board with split-disjoint names and no development/confirmation
  access before all evaluator, assessor, threshold, and custody bytes freeze.

Only the future integrated experiment may test bounded native reasoning.
