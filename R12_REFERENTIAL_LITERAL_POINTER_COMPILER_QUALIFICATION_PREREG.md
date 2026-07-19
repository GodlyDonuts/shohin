# R12 Conventional Complete-Compiler One-Shot Qualification Preregistration

**Status:** design frozen before qualification seed; no qualification data or
score exists

## Question

The factorized development matrix showed that broad language coverage, not
parameter islands, is sufficient for exact known-atom source compilation. The
current sealed confirmation cannot be opened because the islands attribution
gate failed. This board asks a narrower operational question without touching
those bytes:

> Does the already-selected favorable ordinary parser reproduce exact complete
> compilation once on a fresh, untouched known-atom board?

## Frozen arm

- base: raw Shohin 300k, SHA-256
  `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`;
- adapter: ordinary token tagger from job `693049`, SHA-256
  `747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991`;
- adapter parameters: 8,607,886; total parameters: 133,689,550;
- no training, tuning, seed sweep, oracle, retry, or alternative arm selection;
- one evaluation only.

## Board

- 2,048 semantic quartets / 8,192 rows;
- all language atoms occur in the public factorized training corpus;
- every complete factor combination is absent from public train,
  compositional-development, and lexical-OOD data;
- all entity and neutral-anchor names are absent from those public splits;
- zero exact-prompt and word-13-gram overlap against all public factorized data;
- canonical/paraphrase/order-twin/binding-twin and token-bag gates are unchanged;
- all ten source spans and both independent CPU executors must pass;
- the generator has no sealed-confirmation input or path.

The generator and this preregistration must be committed before the
qualification seed is chosen. The report may expose counts and hashes before
evaluation, but no row answer or model score may be read manually.

## Gates

- answer accuracy >=99%;
- semantic-program exact >=99%;
- full ten-pointer exact >=99%;
- operation-kind accuracy >=99.9%;
- initial-state joint exact >=99%;
- at least 2,000/2,048 all-four exact quartets;
- zero confirmation access and every board structural gate passes.

Pass qualifies this exact conventional parser only as Stage-A infrastructure
for a separately preregistered source-deleted executor/consumer development
experiment. Failure rejects integration and requires diagnosis on this
qualification board. Neither outcome establishes execution, halting,
autonomous rollout, native reasoning, or architectural novelty.
