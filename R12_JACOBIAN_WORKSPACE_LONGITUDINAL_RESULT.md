# R12 Raw-300k Future-Jacobian Workspace Longitudinal Result

**Decision:** **FAIL.** Raw 300k has a reproducible averaged future-causal
transport map, but the frozen semantic workspace gate remains zero in the
decisive held-out region. No J-lens coordinate swap, workspace promotion,
reasoning claim, or training intervention is authorized.

## 1. Custody and execution

The preregistration was committed and pushed as part of `4338c49` before any
raw-300k Jacobian output existed. Scientific Python, prompt seeds, source
layers, target layer, evaluation board, layer-selection rule, and thresholds
are unchanged from raw 200k.

Two infrastructure-only corrections were required:

- jobs `690014/690015` exited `124` in the shared Lustre Python import
  preflight and wrote no matrix;
- jobs `690020/690021` staged the existing hash-bound node-local Torch runtime
  but exited before science because that minimal runtime lacked `tokenizers`.

Commits `e2ba68b` and `735050a` stage the immutable node-local runtime plus the
11 MiB hash-bound `tokenizers` package. They do not alter model code,
Jacobian math, prompts, labels, board, or gates. The valid chain is:

| Job | Role | State | Node | Elapsed |
|---|---|---|---|---:|
| `690028` | lens seed `20260714` | completed | `evc48` | 2m28s |
| `690030` | lens seed `20260715` | completed | `evc41` | 51s |
| `690031` | frozen 896-case readout | completed | `evc40` | 49s |

The first valid job includes cold filesystem latency; prompt-level exact
Jacobian passes were about 2.6--4.4 seconds once loaded. Every model parameter
remained frozen.

## 2. Immutable artifacts

| Artifact | SHA-256 |
|---|---|
| `jacobian_workspace_raw300k_p8_v1.pt` | `dd687d232d41b970816245c80503a4002d9d23ea71b3721ecdc2e764249e9f6a` |
| `jacobian_workspace_raw300k_p8_v2.pt` | `17388eaf20971ac777771dc7563ef8d12f7a7d4d14a64af2f0c6f7adaba3e358` |
| `jacobian_readout_raw300k_p16_v1.json` | `305186c3e660ec16127fc964b325e3f129471f7f6f6946c8a25677be0f7d39ef` |
| job `690028` log | `0e3978c7c26551193695ace60403419da591946551eff1d7a34896769b01ec75` |
| job `690030` log | `0470a8c7adf84c22a7e9fe4965d646b7bfe782ec3a146a8f4c9ff1c05ba4946f` |
| job `690031` log | `ce42c21c0efe9471e258376c84606e2a6eb646670211ca9a33ffc0f9a961e088` |

All three scientific artifacts and logs are mode `0444` on Newton. The three
scientific artifacts are mirrored locally with matching SHA-256 values.

## 3. Frozen decision

The original three gates are:

| Gate | Raw 300k |
|---|---|
| disjoint lens prompt samples | pass |
| every within-300k matrix cosine at least 0.90 | pass |
| selected future MRR at least 1.25x immediate | pass |
| selected future top-10 gain at least 10 points | **fail** |

The frozen selected layer moves from 13 at raw 200k to 25 at raw 300k. On the
decisive 2,304 language/full targets:

| Checkpoint | Selected layer | Readout | MRR | Median rank | Top-10 | Top-100 |
|---|---:|---|---:|---:|---:|---:|
| raw 200k | 13 | immediate | 0.0001535 | 9,961 | 0% | 0% |
| raw 200k | 13 | future | 0.0002588 | 6,123 | 0% | 0% |
| raw 300k | 25 | immediate | 0.0001024 | 14,856 | 0% | 0% |
| raw 300k | 25 | future | 0.0004949 | 2,817 | **0%** | **0%** |

The selected raw-300k MRR ratio is about 4.83x, but every target remains below
rank 100. This is a rank-tail effect, not a usable semantic workspace. At the
same layer 25, raw-200k future MRR was 0.0005352 with median rank 2,442, so the
raw-300k selected result is not even monotonic under a fixed-layer comparison.

## 4. Geometry

The two independent raw-300k matrices are highly reproducible. Whole-matrix
cosine rises from 0.9546 at layer 5 to 0.9991 at layer 28; top-16 right-subspace
overlap spans 0.8619--0.9960.

Paired same-prompt raw-200k versus raw-300k matrices change materially:
whole-matrix cosine spans 0.6992--0.8233. Their top-16 right-subspace overlap is
still 0.7014--0.9074, indicating a persistent broad causal subspace whose exact
linear map evolved during continued pretraining. Neither fact provides the
missing held-out semantic readout.

## 5. Interpretation

This result rejects one precise hypothesis:

> By 300k, raw next-token pretraining has produced a vocabulary-aligned,
> averaged-future-Jacobian workspace that exposes the operation and query
> concepts required by the frozen referential task.

It does not prove that every distributed state is absent. The diagnostic is
restricted to the paper's vocabulary-aligned averaged Jacobian object and the
frozen operation/query concept targets. A task-defined causal subspace could
exist without naming those concepts as single tokens, and a trainable workspace
could still be installed. Either successor must be preregistered and must use
bidirectional donor swaps, complement ablation, random/norm-matched controls,
held-out consumers, and source-deleted reuse. Reading or linearly decoding a
state is insufficient.

The July 2026 workspace study explicitly says that it does not know how the
workspace scales to smaller models or when it emerges during pretraining. This
longitudinal result supplies Shohin's answer at 125.1M parameters through raw
300k: the exact tested semantic workspace has not emerged.
