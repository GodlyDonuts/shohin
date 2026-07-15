# R12 Certified Language Bridge Boundary

**Status:** broad bridge rejected. A synthetic residual-state result may cross
to language only on machine-checkable deterministic semantic systems whose
certificate map reflects every claimed future distinction.

## 1. Why the current corpus is insufficient

Flattened reasoning rows retain a question, trace, answer, and family, but not
the semantic transition states needed to prove future equivalence or derive
distinguishing continuations. Answer verification for OpenMath and bounded unit
tests for code likewise do not prove residual equivalence.

Therefore a synthetic WGRQ pass cannot be used to relabel the existing SFT mix
as causal-state supervision. Provenance-rich sources must be re-extracted into
certified bounded systems first.

## 2. Admissible semantic systems

An admitted domain is a bounded deterministic system

```
M = (S, E, Q, A, delta, output).
```

Examples are deliberately narrow:

- exact register programs, finite-field programs, or proof-assistant states;
- total loop-free bit-vector programs or bounded programs with exhaustive/SMT
  equivalence, not merely passing tests;
- finite Horn/Datalog worlds or finite model sets with canonical closure and
  exact entailment.

For each world, the builder must produce:

1. distinct histories reaching one certified residual state;
2. a lexically matched mutation in a different residual class;
3. a shortest certificate-backed distinguishing continuation/query;
4. independently worded surfaces that deterministically round-trip to the same
   typed AST or proof object.

Teacher proposals are allowed only before deterministic verification. Canonical
states, class IDs, witnesses, and answers remain training-side metadata and are
not emitted as textual packet targets.

## 3. Bridge non-reflection theorem

Let `phi` map language histories into a certified synthetic state. If there are
histories `h,g` and a claimed continuation/query `(c,q)` such that

```
phi(h) = phi(g)
```

but the correct language answers after `(c,q)` differ, then any source-deleted
system whose committed packets are interchangeable whenever `phi` agrees must
produce the same output distribution on both histories. On a balanced pair
with distinct correct answers, exact accuracy is at most `1/2`.

If the two packet-conditioned output distributions differ by at most total
variation `epsilon`, balanced exact accuracy is at most `(1+epsilon)/2`.

Thus one future-distinguishable collision kills the bridge claim regardless of
perfect synthetic performance. The certificate map must be a future-reflecting
transition homomorphism over the entire declared continuation/query family.

## 4. Hard source barrier

A confirmation claim requires process-level deletion:

1. a writer exports only a fixed-size committed packet and exits;
2. source IDs, embeddings, residuals, KV cache, paths, and RNG state die with
   that process;
3. the continuation/query is sampled only afterward;
4. a fresh reader receives fixed weights, tokenizer, packet, continuation, and
   query with an empty cache;
5. the reader image has no verifier, solver, retrieval path, source mount, or
   feedback from scoring.

Training-time non-access is weaker and must be described separately.

## 5. Acquisition ledger

Count mutually exclusive channels:

- `T`: every generative-teacher request, including rejected attempts;
- `V`: certificate checks that only validate a supplied object;
- `O`: every label-producing transition, readout, equivalence, witness, or
  target-dependent search query.

Returned bits, query descriptions, adaptive rounds, external search compute,
source tokens, and target tokens are also recorded. A solver that discovers a
witness is an oracle, not merely a verifier. Shared acquisition cost is charged
to every arm; candidate-only adaptive calls disqualify a matched comparison.

## 6. Decision

Reject a broad synthetic-to-language bridge and any use of the current flat SFT
rows as equivalence evidence. Retain a future certificate-bearing language
board as a separate phase only after a synthetic source-deleted protocol passes
its own controls. One failed domain or one non-reflecting collision rejects the
language claim.
