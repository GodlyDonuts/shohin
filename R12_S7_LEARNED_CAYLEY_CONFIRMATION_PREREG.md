# R12 S7 Learned Cayley Law Compiler: Confirmation Preregistration

**Status:** freeze after development qualification and before confirmation access
**Mechanism changes:** none
**Weight changes:** none
**Board changes:** none

## Bound artifacts

- Frozen checkpoint SHA-256:
  `c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607`
- Development assessment SHA-256:
  `2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4`
- Required development decision:
  `qualify_s7_learned_cayley_for_fresh_confirmation`
- Sealed confirmation SHA-256:
  `c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5`
- Confirmation rows: 2,048
- Confirmation laws: 18, disjoint from the 16 development and all train laws
- Development accesses before run: one
- Confirmation accesses before run: zero

## One-read protocol

The confirmation job loads the frozen checkpoint without training and evaluates
the same arms on `confirmation.sealed.jsonl`:

1. host theorem/executor;
2. learned Cayley treatment;
3. frozen favorable ordinary transformer;
4. frozen learned `S^2` generator;
5. deranged law cards;
6. one-witness unit completion;
7. state reset;
8. nonce-operation recoding.

There is no confirmation atomic file, so confirmation repeats the recurrent and
causal gates but not the development-only held-out atomic gate.

## Immutable confirmation gates

All must pass:

- treatment exact state at least 98%;
- treatment answers at least 98%;
- every depth exact state at least 96%;
- treatment within one point of host state and answer;
- treatment exceeds ordinary transformer by at least 40 state points;
- treatment exceeds `S^2` generator by at least 60 state points;
- deranged cards drop state by at least 60 points;
- one-witness unit completion drops state by at least 40 points;
- state reset drops state by at least 20 points;
- operation-nonce recoding is bit-identical;
- checkpoint, board, development assessment, training contract, and parameter
  hashes/counts match;
- development accesses equal one and confirmation accesses equal one after the
  sole read;
- complete system remains below 150M.

Passing records `confirm_s7_learned_cayley_contextual_law_compilation` and
promotes S7 as the strongest bounded unseen-law component. Failure records
`reject_s7_learned_cayley_confirmation`; no repair, second confirmation board,
rescore, or refit is permitted.
