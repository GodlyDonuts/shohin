# R12 SD-CST Physical-Record Write-Bus Result

**Decision:** `retain_physical_record_bus_reject_one_to_one_attribution`

**Claim boundary:** consumed-training compiler mechanics only. This is a
conventional structured parser baseline, not native reasoning, a novelty claim,
or evidence from development/confirmation.

## 1. Immutable contract

- scientific source commit:
  `5c9a2855a202692996e6e4100c927e9d8842bf48`;
- execution source commit:
  `4eeb86f4c4d37e84af78f88d175a2ce7955f3a5c` (documentation-only receipts
  after the scientific freeze);
- raw seed beacon: `18183044536483492966`;
- signed-safe seed: `8959672499628717158`;
- consumed train SHA-256:
  `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`;
- parent checkpoint SHA-256:
  `4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d`;
- fit-source SHA-256:
  `d53e4e805037e56183971bcbd835ce836006c7cd1a5c174e63f2fae877ee4610`;
- heldout-source SHA-256:
  `0efd0993f4ec3d832e8b1d8b5daa053b966af65ece5cededa1211157c1f18719`;
- shared new-parameter initialization SHA-256:
  `c6b6086a57ef52c238756f3aa6b30979732c40d8f10e3786491e855428d5d50e`;
- sole Slurm job: `694136`, H100 `evc37`, completed cleanly in 14m18s;
- scored accesses: development `0`, confirmation `0`.

Slurm test-only admitted the job on `evc37`. Runtime preflight verified the
clean committed source, input hashes, bf16 H100 allocation, and exact parameter
certificate before constructing the optimizer or output.

## 2. Exact system

| Quantity | Count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler, including frozen parent | 65,831,689 |
| new trainable record-bus parameters | 11,106,830 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **190,933,394** |
| **strict-200M headroom** | **9,066,606** |

The complete 56,000-view pre-run renderer scan found exactly nine physical
records in every source. Maximum payload length was 132 bytes and maximum
compiler record length, retaining newline, was 133 bytes under the frozen
144-byte window.

Both arms reconstructed the same frozen parent and shared byte-identical values
for every one of the 88 `record_*` parameter names. Parent state SHA-256 remained
`9b3b34bd13df31b477d66bba4dfc489cb3cdf717e3fe0d8f76e140e22bacf150`
before and after both arms.

## 3. Matched result

Each arm trained for two epochs / 3,000 updates on 48,000 rendered fit rows and
was evaluated on 8,000 rendered held-out rows. The four fit renderers and four
held-out renderers are disjoint parity combinations over 12,000 and 2,000
disjoint latent programs.

| Arm | Fit exact packets | Heldout exact packets | Minimum heldout line pointer | Minimum heldout event pointer |
|---|---:|---:|---:|---:|
| constrained Sinkhorn/greedy one-to-one | 48,000/48,000 | 8,000/8,000 | 100% | 100% |
| independent assignment | 48,000/48,000 | 8,000/8,000 | 100% | 100% |

For both arms, minimum held-out-renderer initial, query, declaration pointer,
initial-occurrence pointer, all-nine line pointer, active event pointer, kind,
identity, amount, whole tape, and complete packet are all exactly **100%**.
Both arms already reached every listed metric at 100% after epoch one.

The constrained arm used 384.335 seconds; the independent arm used 361.111
seconds. Its endpoint losses are also effectively identical. Therefore no speed,
quality, or sample-efficiency advantage is attributable to Sinkhorn or greedy
one-to-one assignment under this board.

## 4. Gate accounting

Seventeen of twenty frozen gates pass. Every absolute compilation,
preservation, parameter, matched-count, and access gate passes. Exactly the
three frozen differential gates fail:

1. constrained packet does not beat independent by five points: 100% vs 100%;
2. constrained line pointer does not beat independent by five points: 100% vs
   100%; and
3. constrained event pointer does not beat independent by five points: 100% vs
   100%.

The preregistered decision is therefore
`retain_physical_record_bus_reject_one_to_one_attribution`. It is not permissible
to report all gates as passed or to attribute the result to doubly stochastic
assignment.

## 5. What changed scientifically

The failed joint global-query parent ended with held-out per-slot line/event-
address/kind/amount/identity of only
42.029%/25.466%/55.731%/68.000%/50.325% and zero complete packets. Without
changing the parent, executor, data volume, or optimization schedule, explicit
delimiter-bounded record encoding plus local field extraction reaches 100%
complete packets on fit and held-out renderer combinations.

This establishes a strong conventional mechanism result:

- the old failure was not lack of parameter capacity or dead optimization;
- physical locality removes the multiplicative global-address bottleneck;
- a shared local entity motor plus the frozen matcher is sufficient;
- unseen parity combinations of the finite renderer factors transfer exactly;
  and
- semantic-slot exclusivity emerges without an explicit one-to-one constraint.

The causal credit belongs to the **physical-record/local-field factorization as
a package**, not to one-to-one assignment. A stricter component attribution
would require a separate matched no-locality architecture and is not necessary
before fresh-board qualification because both current arms are conventional
baselines.

## 6. What remains unproved

The pilot reuses already-consumed training rows and an explicit finite grammar:
newline delimiters, exactly nine records, eight event slots, categorical field
heads, and a frozen known executor. It does not establish:

- transfer to fresh names, renderer families, or source distributions;
- language parsing without record delimiters/cardinality;
- self-selected operations, schedules, or programs;
- learned halt beyond the existing fixed categorical tape;
- source-deleted recurrent state transport on a fresh board;
- broad natural-language reasoning; or
- benchmark improvement in the 125M language model.

No existing development or sealed confirmation may be opened. The retained
record bus is eligible only for a separately committed fresh-board
qualification with new source bytes, controls, thresholds, and access ledger.

## 7. Preserved artifacts

- local checkpoint:
  `train/sd_cst_physical_record_bus_pilot_8959672499628717158/compiler.pt`;
- local report:
  `train/sd_cst_physical_record_bus_pilot_8959672499628717158/report.json`;
- committed report copy:
  `artifacts/r12/sd_cst_physical_record_bus_pilot_8959672499628717158.report.json`;
- Newton output:
  `/lustre/fs1/home/sa305415/shohin_sd_cst_physical_record_bus_pilot_8959672499628717158/`;
- checkpoint SHA-256:
  `89ab7d7417918e72da60028e6d5936908a3ee29c0981f5fdac9dc385c3099419`;
- report SHA-256:
  `9c768fa8b9fd00ba259d36dcf3cf13a39ff6b455d6f59e8f9ee8d6c59ebcf2a4`.

Local and Newton hashes match exactly.
