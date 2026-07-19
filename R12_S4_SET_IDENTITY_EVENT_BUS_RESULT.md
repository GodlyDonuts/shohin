# R12 S4 Set-Identity Event Bus Result

## Decision

**Reject S4 v3 on fresh development. Do not generate confirmation and do not repair or rescore on
the closed board.**

The set-valued roster carrier is strongly validated, but the learned global event-conditioned token
membership does not pair each operation anchor with its arguments reliably enough to compose. Keep
the roster/query carrier as a component; reject the event-attention bus as the S4 parser.

## Custody

- Initial source freeze commit: `2ac31a5`.
- Public-audit repair commit before production board: `3019ba8`.
- Board freeze commit before model access: `ab52072`.
- Sole production seed: `11437896185638727043`.
- Retired before board/model access: `14970823073944690832`, `939143060519850990`, and
  `15848092346808854751`.
- Board: 2,048 rows / 512 matched groups, depths 3--8, maximum 344 tokens.
- Data SHA-256: `b49ddbbfad3da04181d6ec5401f8412b2953185e5e91e344208c8b6b0c5ba1e8`.
- Report SHA-256: `808b0e0287e53576ffb234a5ea855943552ef3e60b2d3d20847b79f7254d692c`.
- Safe archive SHA-256: `28302861b383fbdc8e5056e25bbd98b188487e87b241d25d2ef5ac82cebd43ae`.
- Exact prompt, word-13-gram, nonce/name, factor, and roster-token-multiset overlap: zero against
  every supplied public source.
- Confirmation access: zero.

The first treatment/shuffled jobs `693167/693168` failed before their first update because a
per-row mask was paired with batch-padded logits. They wrote no parser artifact and had no
development access. Commit `c6d9f00` fixes only that shape slice and adds a mixed-length real
backward check. Corrected treatment `693170` and shuffled `693171` then each completed exactly one
epoch / 750 updates. One-shot evaluations are `693172/693173`; frozen assessor is `693174`.

## Parameter and training receipt

| Quantity | Count |
|---|---:|
| Raw Shohin base | 125,081,664 |
| Complete adapter including frozen v1 | 9,198,095 |
| New trainable set-membership maps | 589,824 |
| Total system | 134,279,759 |

Both arms load exactly 71 frozen v1 tensors; only four 384x384 tensors train. Treatment completes in
273.31 seconds with adapter SHA-256
`ff718f6c83fb1ed3c369ad0ae55b30e35d3539d3ed743faebe9fc23ac2fb6a92`; shuffled completes in
275.38 seconds with adapter SHA-256
`29849ae8dc8b21102e8311c69440629010d5e8ec7639108fcca473ee5543b3a5`.

## Fresh-board result

| Arm | Count | Roster recovery | Query | Exact program | Exact state | Correct answer |
|---|---:|---:|---:|---:|---:|---:|
| Frozen S4 v1 | **100%** | 93.41% strict | 93.41% strict | **1913/2048 = 93.41%** | **93.41%** | **93.41%** |
| S4 v3 set bus | **2048/2048 = 100%** | **2037/2048 = 99.46%** | **2048/2048 = 100%** | 191/2048 = 9.33% | 685/2048 = 33.45% | 949/2048 = 46.34% |
| Shuffled membership | 100% | 99.46% | 100% | 2/2048 = 0.10% | 18.99% | 34.47% |
| Treatment + roster derangement | 100% | unchanged | 100% | **0/2048** | 11.82% | 27.73% |

Treatment exact programs by depth are 35.76%, 15.70%, 2.94%, 0.88%, 0.29%, and 0% at depths
3--8. This chain-length decay is consistent with a partially correct atomic pairing probability
being multiplied across events; it is not a failure of event count, roster recovery, query
classification, or locked S3 execution.

The treatment beats shuffled supervision by 9.23 points in exact programs, 14.46 points in exact
state, and 11.87 points in answers. Cyclically deranging only the three roster carriers removes all
191 exact programs and reduces state/answer strongly. Therefore the set identity channel is causal,
not an unused diagnostic. It is simply too inaccurate at event-to-argument alignment.

Baseline, treatment, shuffled, and assessment report SHA-256 values are respectively
`c2236edc9da3ee68e8bb1a7e96a33194cfcff44bd7b642e8787c143a03b04bca`,
`3677c08c3e5402d61c8d40159c1d92a205d65df1f8913087c54a2e20767b98ce`,
`f96b5164eec7694e04a25ca07465c48ae7ebcc10614397b56615be617610c1fc`, and
`1b6cb30e5a75fd0e3315ccb369d0131aaa381c208a4c8a8e6627851510511b71`. The assessor records
`reject_s4_v3_fresh_development`.

## Interpretation and next constraint

The representation theorem survived only at the roster interface. A vocabulary-aligned weighted
token set transports same-name identity across occurrence and BPE width. The failure comes from
asking a learned global query/key map to discover which event-local entity/literal belongs to each
kind anchor. That map must solve syntactic segmentation and identity at once; one-epoch train loss
separates from shuffled, but fresh exactness decays to zero with depth.

The next lawful repair must not retrain lexical identity or another absolute/global pointer. It
should preserve:

1. frozen v1's exact model-owned kind-anchor count;
2. v3's 99.46% soft roster recovery and 100% query recovery;
3. frozen v1's much stronger event-role evidence;
4. locked S3 execution.

A bounded candidate is a zero-fit monotone event-region decoder. Consecutive model-discovered kind
anchors partition the source into ordered event regions; frozen entity/literal role evidence is
normalized only inside its event region, then the resulting entity set is matched to the soft roster
carrier. This removes learned global pairing while adding no gold depth, boundary label, lexical
table, threshold, or new parameter. It must be preregistered and scored once on a new board with
event-region and roster derangements.

## Claim boundary

This is a fresh-development result over known operation atoms. It is causal evidence for a bounded
set-valued lexical identity channel, not confirmation, unseen semantics, planning, learned halt,
free-form reasoning, public benchmark improvement, novelty, or model promotion.
