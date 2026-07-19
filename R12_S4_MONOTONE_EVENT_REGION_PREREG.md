# R12 S4 Monotone Event-Region Decoder Preregistration

## Status

**Closed negative on 2026-07-19.** The frozen assessor records
`reject_s4_v4_fresh_development`: 70.46% exact programs versus 93.21% for frozen v1. Both roster
and event-region derangements fall to zero programs, so locality is causal but the diffuse regional
softmax is insufficiently precise. Confirmation was never generated or read. Full evidence is in
`R12_S4_MONOTONE_EVENT_REGION_RESULT.md`.

Source was frozen at commit `0c8aa8c` before production seed selection, board generation, model
evaluation, or score access.

Production seed `3847103809226516730` generated exactly 2,048 rows / 512 matched groups after the
source freeze. All declared overlap and mechanics gates pass. Data, report, and read-only safe
archive SHA-256 values are respectively
`3c06f58c4ade457ac5017be41afbd97fd3c23a90200430af1318af7e5a988f19`,
`27b795d1eedbba65d3697d50ab8eaec5175616e4743a573c5ca5419c21aea0f4`, and
`7443f4962f19c5a3740b85ccf0a2d38bfa52adbf5cc1ab8a08df8e69fd8bddf2`.

First submission `693175` failed closed during shell preflight, before model or development-board
access, because the job named a nonexistent frozen-v1 parser directory. The only repair replaces
that literal with the preserved `train/s4_event_tape_treatment_2026071904/parser.pt`; mechanism,
board, controls, evaluator, assessor, and gates are unchanged.

## Measured motivation

S4 v3 recovered event count (100%), roster carriers (99.46%), and query (100%), and its identity
channel was causal. It failed exact programs (9.33%) because a learned global event query had to
discover local syntax and lexical identity simultaneously. Exactness decayed to zero by depth eight.
No additional parser fit is justified until locality itself is isolated.

## Hypothesis

Model-discovered kind anchors already provide an ordered event clock. If consecutive anchors
partition the source at the midpoint of their intervening gaps, frozen v1 `event.entity` and
`event.literal` evidence should identify each anchor's arguments inside its own local region. A
vocabulary-aligned soft set then resolves the local entity against the three frozen roster sets.

This is a deterministic monotone structured decoder, not a claimed new reasoning primitive. It
adds zero weights and receives no gold depth, event boundary, entity label, literal label, query,
answer, source template, or development-derived threshold.

## Frozen mechanism

1. Load raw Shohin 300k, the frozen S4 v1 treatment parser, and locked S3 executor.
2. Admit a lexical kind occurrence only when at least one of its tokens receives the frozen v1
   `event.kind` argmax role. Sort admitted anchors by source position.
3. Place a boundary between adjacent anchors at `floor((previous_end + current_start) / 2)`. The
   first region begins at token zero and the last ends at sequence length.
4. Within each region independently, softmax frozen `event.entity` role logits. Scatter the weights
   into an exact vocabulary histogram.
5. Build three roster histograms by softmaxing frozen `intro.entity0..2` role logits over the full
   valid sequence. Select identity by cosine similarity scaled by the already frozen factor 20.
6. Within the same event region, softmax frozen `event.literal` role logits and average frozen
   amount logits; take argmax plus one.
7. Recover query by the frozen v3 full-sequence soft `query.position` weighting and frozen query
   head. Execute the resulting program only with locked exact S3 semantics.

There are no trainable tensors. Total parameters must remain the frozen v1 total and below 150M.

## Causal controls

- **Frozen v1 baseline:** strict autonomous v1 on the identical fresh board.
- **Roster derangement:** rotate the three roster carriers `(1,2,0)` while holding all model
  outputs and event regions fixed.
- **Event-region derangement:** cyclically assign region `i+1 mod depth` to kind anchor `i` while
  holding kinds, roster carriers, and all model outputs fixed.
- **Gold S3 sanity:** symbolic gold programs must execute consistently; gold programs are never
  supplied to the treatment decoder.

## Fresh-board custody

- Exactly one production seed will be sampled only after this source and protocol are committed.
- Exactly 2,048 rows / 512 matched groups, depths 3--8.
- Zero exact prompt, word-13-gram, nonce/name, factor, and roster-token-multiset overlap against all
  supplied public sources, including the closed v2 and v3 boards.
- One treatment evaluation and one identical-board frozen-v1 baseline. No repair, rescore, or
  threshold change after development access.
- Confirmation remains inaccessible unless every gate passes.

## Frozen qualification gates

All gates must pass:

1. event count >=98% overall and >=95% at every depth;
2. exact program >=95% overall and >=90% at depths 5--8;
3. exact state >=95%, answer >=95%, query >=98%, roster recovery >=95%;
4. exact program >= frozen v1 plus one percentage point;
5. roster-deranged and event-region-deranged exact programs each <=40%;
6. locked S3 gold sanity passes;
7. zero new trainable parameters, total system <150M;
8. development access exactly one and confirmation access zero.

Failure closes this exact decoder on the board. A pass permits one newly generated confirmation;
it does not itself establish unseen semantics, planning, free-form reasoning, benchmark improvement,
or novelty.
