# R12 Factorized-Language Complete-Compiler Board Preregistration

**Status:** development closed; primary gates pass; islands attribution fails;
confirmation sealed

## 1. Question

The first compiler overfit renderer coordinates. Bidirectional role supervision improved binding but
disconnected operation semantics. Separate parameter islands restored composition and reached 59.8%
answers / 42.8% exact programs on one unseen paraphrase grammar, but failed another unseen lexical
grammar. The next question is:

> Does broad factorized language supervision teach a complete source-pointer compiler to compose
> known lexical and syntactic atoms in unseen combinations, and what remains when the operation
> lexemes themselves are unseen?

This is a data-identifiability experiment over the admitted v1.3 architecture, not a new reasoning
primitive.

## 2. Renderer factorization

Do not define a renderer as one monolithic template. Generate each source by independently sampling:

1. intro frame;
2. list separator and conjunction style;
3. operation ordinal vocabulary;
4. operation clause frame;
5. entity/kind/literal argument order;
6. left/right lexical pair;
7. distractor frame and insertion location;
8. query frame;
9. harmless punctuation/case variation.

Every atomic factor used by the primary compositional development stratum must appear in training,
but its exact cross-product combination must not. A separate lexical-OOD stratum uses direction
lexemes absent from training. Do not merge these two scores.

## 3. Fresh splits

- train: 96,000 rows / 24,000 semantic quartets;
- development-compositional: 2,048 rows / 512 quartets, unseen factor combinations with known atoms;
- development-lexical-OOD: 2,048 rows / 512 quartets, unseen direction lexemes and combinations;
- confirmation: 8,192 rows / 2,048 quartets, sealed;
- disjoint nonce-name pools across all splits;
- zero exact and word-13-gram overlap;
- token-bag-matched order and binding twins in every quartet;
- all ten model-owned source targets present and nonempty;
- two independent CPU executors agree on every row.

The generator and tests must be committed before any confirmation seed exists. Development seeds are
also generated only after source commit. The v1.1 confirmation seed and bytes remain permanently
unread for this lane.

## 4. Arms

Run equal examples, optimizer updates, seed families, and inference inputs:

1. v1.1 free-slot pointer compiler;
2. v1.2 bidirectional structured compiler;
3. v1.3 structural/semantic parameter islands;
4. favorable ordinary bidirectional sequence-tagger/pointer control with matched parameter budget;
5. lexical oracle: operation class supplied, all source roles model-owned;
6. structure oracle: target role masks supplied, selected values and operation classes model-owned;
7. full source-span oracle ceiling;
8. shuffled role-label negative control.

Oracle information and host computation must be reported explicitly and may establish ceilings only.

## 5. Development gates for v1.3

### Primary compositional stratum

- >=85% answer accuracy;
- >=75% semantic-program exact;
- >=65% full ten-binding pointer exact;
- >=95% operation-kind accuracy;
- >=80% initial-state joint exact;
- >=192/512 canonical+paraphrase both exact;
- >=96/512 all-four exact.

### Lexical-OOD diagnostic

- report every metric without a promotion floor;
- compare against the frozen base's lexical-oracle gap;
- no lexical-OOD result may be pooled into the primary score.

### Attribution

The islands arm must exceed the favorable ordinary parser by at least 5 percentage points in
semantic-program exact or be treated as an implementation choice rather than a mechanism win.

## 6. Decisions

- CPU board failure: repair or reject before neural code.
- Primary development failure: reject the curriculum/architecture pair; do not open confirmation.
- Primary pass without attribution: retain as a conventional compiler baseline only.
- Primary and attribution pass: freeze arm identity and run confirmation once.
- No outcome authorizes executor/halt integration until complete compilation survives confirmation.

## 7. Frozen outcome

Parameter islands and the favorable ordinary tagger both reach 100% exact
semantic programs, full pointers, answers, and quartet consistency on the
2,048-row compositional development split. The islands advantage is therefore
0.0 points rather than the required +5. Free slots reach 98.242% exact programs,
structured parsing reaches 100%, and shuffled-label islands reach 0.146%.

The absolute gate passes, the attribution gate fails, and the decision is to
retain a conventional compiler baseline while leaving confirmation sealed.
See `R12_REFERENTIAL_LITERAL_POINTER_COMPILER_FACTORIZED_DEVELOPMENT_RESULT.md`.
