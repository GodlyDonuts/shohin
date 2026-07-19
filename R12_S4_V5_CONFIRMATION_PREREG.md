# R12 S4 v5 Confirmation Preregistration

## Status

**Protocol frozen at commit `6731743` before confirmation seed selection, board generation, or
confirmation access. The sole confirmation board is admitted and frozen; no model has accessed
it.**

Seed `14809014609581254328` generated 2,048 rows / 512 groups with every declared gate passing.
Confirmation data, report, and read-only archive SHA-256 values are respectively
`b16534b3c41d21737370f0eb852cb6c53d75e81d661d6d9592927709551a08cf`,
`ce0e2671c8dcd07b5b789798da0325b94066087ce4222082267919f57afdb261`, and
`af94e855ba81905c3fed18ef8f4764e16574afc2bfd29a4f13247f9af5df337f`.

Development assessor SHA `41a2dd2eab37c1976803d49e36f1a4ae35b62e8568ebefc3159c118383ab2eb5`
qualified the unchanged hard-island/soft-interface mechanism at 96.92% exact programs. This
protocol adds confirmation-only board, access-accounting, and assessor plumbing. It must import the
frozen `s4_hard_island_soft_interface.py`; no decoder, parser, model, lexicon, selector, tie-break,
threshold, fallback, S3 semantics, or gate changes are allowed.

Exactly one 2,048-row / 512-group board will be sampled after this source commit. It must pass all
fresh mechanics and have zero exact, word-13-gram, nonce/name, factor, and roster-token-multiset
overlap against source train/development and every closed or qualified v2--v5 development board.
The inherited row split label remains `s4_event_tape_development` for frozen parser compatibility;
the report and authoritative artifact role are explicitly confirmation. `artifacts.development` is
an exact receipt alias to the same bytes solely so the unchanged frozen-v1 baseline evaluator can
read them.

One serial job runs strict frozen v1, unchanged v5, roster rotation, event-region rotation, and the
confirmation assessor. Gates are identical to development: program >=95% overall, >=90% at depths
5--8, and >=v1+1 point; state/answer >=95%; count/query/roster gates; both interventions <=40%;
zero trainable v5 parameters; total <150M. Development access must be zero and confirmation access
exactly one. No repair, rescore, second board, or reuse is permitted after the confirmation read.

A pass confirms only bounded known-atom structured parsing/execution. It does not establish unseen
semantics, open-ended planning, learned halt, free-form reasoning, benchmark gains, or novelty.
