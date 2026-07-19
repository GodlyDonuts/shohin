# R12 S4 Event-Relative Pointer Result

## Decision

**Reject S4 v2 on fresh development. Do not score confirmation and do not repair or rescore on the
closed board.**

The treatment preserves model-owned event count but replaces the strong S4 v1 token-role parser
with independently decoded absolute start/end coordinates. Those coordinates fit the admitted
training corpus, then fail compositionally on fresh names and longer tapes. This is a clean negative
result, not an ambiguous threshold miss.

## Custody

- Source/preregistration commit before the fresh seed: `fceddee`.
- Fresh-board freeze commit: `c2e070d`.
- Production development seed: `3662806511482505284`.
- Fresh board: 2,048 rows in 512 matched groups, depths 3--8.
- Development data SHA-256:
  `bed2e261484e7cece2f6f4eb748504f8f5a30aecdd1c7ec15718b43579ef2219`.
- Development report SHA-256:
  `352d8215e0bc984fc0d30bf18b2f1290c25b45c349115c2a66fa87810580c1c6`.
- Safe archive SHA-256:
  `61d3e0ec75b260184cf80c8e7bfae3311232c1586d7c4375c510a80c18dd490d`.
- Exact prompt, word-13-gram, name, nonce, and factor overlap: zero against the old S4
  train/development corpus and supplied public compiler/executor boards.
- Confirmation access: zero.
- Treatment train/eval jobs: `693162` / `693165`.
- Shuffled-label train/eval jobs: `693163` / `693166`.
- Both arms use 48,000 examples, 750 updates, one epoch, the same frozen S4 v1 parser, and the same
  optimizer/schedule. Only pointer labels differ.

## Parameter and optimization check

The treatment initializes all 71 non-base S4 v1 tensors and freezes them. All 16 trainable tensors
belong to the new pointer modules.

| Quantity | Count |
|---|---:|
| Raw Shohin base | 125,081,664 |
| Complete adapter including frozen v1 | 9,790,999 |
| New trainable pointer parameters | 1,182,728 |
| Total system | 134,872,663 |

The treatment completed in 307.63 seconds with final logged loss 0.11427 and adapter SHA-256
`4db4bf5f393aec69f35a5b7de83f6e240797c9fbb517b22233366b0a9486d7d8`. The shuffled arm completed
in 308.18 seconds with final logged loss 4.67416 and adapter SHA-256
`fbd28342e71d389b60347f71cdee4b49fca254493708a042d12456a36b76af6d`.

## Fresh-board result

| Arm | Event count | Initial roster | Exact program | Exact state | Correct answer |
|---|---:|---:|---:|---:|---:|
| Frozen S4 v1 baseline | **2048/2048 (100%)** | 1914/2048 (93.46%) | **1914/2048 (93.46%)** | **1914/2048 (93.46%)** | **1914/2048 (93.46%)** |
| S4 v2 event-relative pointers | 2044/2048 (99.80%) | 382/2048 (18.65%) | 254/2048 (12.40%) | 296/2048 (14.45%) | 318/2048 (15.53%) |
| Shuffled pointer labels | 1386/2048 reported count exact | 0/2048 | 0/2048 | 0/2048 | 0/2048 |

The shuffled count number is not a separate count-head measurement. Early pointer failures return a
zero predicted count in the v2 decoder, so that aggregate is partially confounded by decode failure.
It does not affect the causal conclusion: shuffled pointer supervision produces zero valid programs,
while the treatment is far below both the frozen v1 baseline and every frozen advancement gate.

Treatment exact-program accuracy by depth is 51.74%, 18.90%, 2.94%, 0.29%, 0%, and 0% at depths
3--8. By contrast, frozen v1 remains 92.35--95.00% across those depths. The treatment's 1,794
non-exact rows contain:

- 1,179 crossed or invalid event argument boundaries (`event_pointer`);
- 477 event entity spans that fail to equal one roster span (`event_identity`);
- four invalid intro boundaries;
- 134 structurally valid but semantically wrong programs.

Treatment evaluation SHA-256 is
`12005bd33248d7467036fb462a2e535866db29522bea2632dff0b2e24c7f58fe`; shuffled evaluation SHA-256
is `409aa8b18ad8efd077c0ebd6ad3ccc14f053341fc9459c5082bced928c44d2d4`; assessment SHA-256 is
`1c7af1ceb19ae5b0fceaa49fba5f111b6425002c1e40a75fbfe8b43e83367275`. The frozen assessor records
`reject_s4_v2_fresh_development`.

## Interpretation

The result falsifies the proposed repair. A direction-anchor query followed by independent absolute
start and end argmax is not a compositional identity representation. It can reduce supervised
coordinate loss without learning the invariant relation "this event name is the same lexical object
as roster item i." Variable BPE width makes two independent extrema especially brittle, and each
additional event supplies another opportunity for a crossed or renderer-specific boundary.

The surviving evidence is stronger than before:

1. S4 v1's event-count signal generalizes perfectly to a wholly fresh board.
2. S4 v1's shared token-role representation also generalizes strongly: 93.46% exact programs,
   including 92.35% at depth seven and 93.82% at depth eight.
3. Exact execution is already solved by locked S3 once a correct program exists.
4. The open interface is lexical identity transport, not event counting or recurrence.
5. The next lawful mechanism must represent identity without selecting two absolute coordinates.

The most direct next test is a set-valued identity carrier: aggregate a soft token set for each
roster item, aggregate an event-conditioned soft token set for each event entity, and classify by
set similarity. It must keep S4 v1 frozen, train only the carrier, use a shuffled-identity control,
and evaluate once on a newly generated board after source freeze.

## Claim boundary

This result concerns fresh-development parsing of known operation atoms only. It establishes no
confirmation, unseen action semantics, planning, learned halt, free-form reasoning, public benchmark
gain, novelty, or Shohin promotion.
