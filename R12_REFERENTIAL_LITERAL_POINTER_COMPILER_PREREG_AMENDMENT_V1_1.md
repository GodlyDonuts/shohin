# R12 Referential Literal-Pointer Compiler Preregistration Amendment v1.1

**Status:** **FROZEN BEFORE ANY NEURAL FIT, MODEL SCORE, OR CONFIRMATION
OPENING.** This amendment supersedes only the pointer-output inventory in
`R12_REFERENTIAL_LITERAL_POINTER_COMPILER_PREREG.md`. All thresholds, controls,
stage boundaries, and claim restrictions remain unchanged.

## Defect found

The v1 preregistration required two operation codons and one query pointer but
did not explicitly require the compiler to bind the initial ordered entity
state. A host parser could therefore provide the executor's initial order while
the model predicted only updates. That would repeat the R4 boundary defect in a
different field.

The already frozen corpus rows contain exact tokenizer spans for
`intro.entity0`, `intro.entity1`, and `intro.entity2`. No row regeneration or
semantic change is required. The v1 acquisition ledger undercounted pointer
labels because it did not charge these existing spans.

## Corrected complete interface

Every neural arm must now own ten source bindings per row:

```text
initial state:
  intro.entity0 pointer
  intro.entity1 pointer
  intro.entity2 pointer

operation 0:
  kind grounding pointer/class
  entity pointer
  literal pointer

operation 1:
  kind grounding pointer/class
  entity pointer
  literal pointer

late query:
  query-position pointer
```

The compiler output is therefore

```text
[(initial entity pointer) x 3,
 (kind, entity pointer, literal pointer) x 2,
 query pointer,
 STOP]
```

At inference, no structured initial order, entity name, operation, literal,
query, answer, renderer, or target span may be provided. The host may only
dereference model-selected source spans and apply the independently frozen
machine semantics. A wrong initial pointer makes the full program wrong.

## Amended gates

Before fit:

- all 102,144 corpus rows must contain nonempty tokenizer spans for all ten
  targets;
- the acquisition ledger must count `1,021,440` pointer labels, not `715,008`;
- regenerated v1.1 JSONL files must be byte-identical to the frozen v1 files,
  proving this amendment changes only auditing and model obligations;
- the three initial-pointer exact accuracies and joint initial-order exactness
  must be reported separately on development and confirmation;
- full exact-program accuracy requires all three initial pointers, both
  operation kinds, both entity pointers, both literal pointers, query pointer,
  and STOP.

This repair narrows the claim and makes the pilot harder. It does not authorize
executor or HALT integration and does not create a reasoning result.
