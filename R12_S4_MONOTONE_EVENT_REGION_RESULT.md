# R12 S4 Monotone Event-Region Decoder Result

## Decision

**Reject S4 v4 on fresh development. Do not generate confirmation or rescore the closed board.**

Monotone locality is strongly causal, but replacing frozen v1's hard event-role islands with a
diffuse softmax over each complete region loses too much argument precision. Preserve the region
partition and soft roster/query interfaces; restore hard model-owned event islands.

## Custody

- Source/preregistration freeze: commit `0c8aa8c`.
- Board freeze: commit `1cf83e7`.
- Production seed: `3847103809226516730`.
- Board: 2,048 rows / 512 matched groups, depths 3--8, maximum 341 tokens.
- Data SHA-256: `3c06f58c4ade457ac5017be41afbd97fd3c23a90200430af1318af7e5a988f19`.
- Report SHA-256: `27b795d1eedbba65d3697d50ab8eaec5175616e4743a573c5ca5419c21aea0f4`.
- Safe archive SHA-256: `7443f4962f19c5a3740b85ccf0a2d38bfa52adbf5cc1ab8a08df8e69fd8bddf2`.
- Exact prompt, word-13-gram, nonce/name, factor, and roster-token-multiset overlap: zero against
  source train/development and both closed fresh boards.
- Confirmation access: zero.

Submission `693175` failed shell preflight before model or board access because of a nonexistent
parser path. Commit `cb3be54` changes only that literal to the preserved frozen-v1 parser.
Replacement `693176` completed the baseline, treatment, both interventions, and assessor serially
in 62 seconds on `evc24`.

## Fresh-board result

| Arm | Count | Roster | Query | Exact program | Exact state | Correct answer |
|---|---:|---:|---:|---:|---:|---:|
| Frozen S4 v1 | **100%** | 93.21% strict | 93.21% strict | **1909/2048 = 93.21%** | 93.21% | 93.21% |
| S4 v4 local soft regions | **100%** | **2006/2048 = 97.95%** | **100%** | 1443/2048 = 70.46% | 83.15% | 87.06% |
| V4 + roster rotation | 100% | unchanged | 100% | **0/2048** | 2.44% | 9.67% |
| V4 + event-region rotation | 100% | unchanged | 100% | **0/2048** | 14.99% | 29.25% |

Treatment exact programs by depth are 85.76%, 73.84%, 70.59%, 65.88%, 67.65%, and 58.82% at
depths 3--8. Surface accuracies stay tightly grouped from 69.34% to 71.29%, so no single renderer
explains the failure. Both interventions eliminate every exact program. Locality and roster identity
are therefore causal, but the regional soft distributions have a per-event precision error that
compounds with chain length.

The same frozen-v1 baseline reaches 1926/2048 = 94.04% exact programs with host count and
1960/2048 = 95.70% when only intro/query boundaries are supplied. Its strict failure inventory is
83 event-component-cardinality, 51 intro-cardinality, and five entity-identity cases. This shows why
v4 regressed: it solved roster/query interfaces but discarded the much sharper hard event islands
already present in v1.

Baseline, treatment, assessment, and job-log SHA-256 values are respectively
`df4b0ddbc20c77efd5d39bb0f2fe3286245d1c68d38bec9c642d49f346daa44e`,
`47604341285fb63a4fb2abce3c6db7859fa227eaef796858387c2d3ad5145985`,
`63f9d5db3e2db70acaf646dfeda5e6ff503b890b10abd002d76c8f00b0c68b08`, and
`d539d220347dbdceb0b42f2350b9e09d2617c68ae867a8b256135bd0c5de14c0`. The frozen assessor records
`reject_s4_v4_fresh_development`.

## Next constraint

A bounded v5 may retain each predicted kind region but select complete contiguous frozen
`event.entity` and `event.literal` argmax islands inside that region. Entity islands become uniform
vocabulary carriers matched to the soft roster; literal islands use their mean frozen amount logits;
the soft query remains. If a region contains duplicate islands, select by summed frozen role margin,
not a learned or score-tuned threshold. Require fresh roster and region derangements.

## Claim boundary

This is fresh-development evidence over known operation atoms. It establishes a causal local
decomposition, not confirmation, unseen semantics, planning, learned halt, free-form reasoning,
public benchmark improvement, novelty, or model promotion.
