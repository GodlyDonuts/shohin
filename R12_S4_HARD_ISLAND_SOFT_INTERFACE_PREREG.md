# R12 S4 Hard-Island / Soft-Interface Preregistration

## Status

**Frozen before production seed selection, board generation, model evaluation, or score access.**

## Measured motivation

On fresh v4, monotone regions plus soft roster/query are causal but diffuse regional argument
softmax reduces exact programs to 70.46%. On the same board frozen v1 reaches 95.70% when only its
intro/query boundaries are replaced by gold, showing that hard event-role islands are the stronger
argument representation. V5 combines these independently measured components without fitting.

## Frozen mechanism

1. Load raw Shohin 300k, frozen S4 v1 treatment parser, and locked exact S3.
2. Discover kind anchors and monotone midpoint regions exactly as frozen v4.
3. Within each region, enumerate complete contiguous argmax islands for frozen `event.entity` and
   `event.literal` roles.
4. If more than one island exists, choose the island with maximum summed target-role logit minus
   best-other-role logit. Ties prefer the longer then earlier island. If none exists, fail closed.
5. Convert the complete entity island to a uniform vocabulary histogram and match it by frozen
   cosine scale 20 against three full-sequence soft roster histograms.
6. Mean frozen amount logits across the complete literal island. Recover query through the frozen
   full-sequence soft query interface. Execute only with locked S3.

No gold depth/span/identity/literal/query/answer, learned lexical table, fitted weight, fallback, or
score-derived threshold exists. New trainable parameters are exactly zero.

## Controls and custody

- identical-board strict frozen-v1 baseline;
- roster carrier rotation `(1,2,0)`;
- cyclic event-region assignment `i+1 mod depth`;
- locked S3 gold sanity;
- one production seed only after source commit;
- 2,048 rows / 512 groups, depths 3--8;
- zero exact, word-13-gram, nonce/name, factor, and roster-token-multiset overlap against source
  train/development and all closed v2--v4 fresh boards;
- one serial evaluation; no repair/rescore after development access; confirmation inaccessible.

## Frozen gates

Count >=98% overall and >=95% each depth; program >=95% overall and >=90% at depths 5--8; state
and answer >=95%; query >=98%; roster >=95%; program >= frozen v1 plus one point; roster and region
derangements each <=40% programs; S3 sanity; zero new trainable parameters; total <150M;
development access one; confirmation access zero. All gates must pass before one new confirmation.

Passing is bounded known-atom parsing evidence, not unseen semantics, planning, learned halt,
free-form reasoning, public benchmark improvement, novelty, or model promotion.
