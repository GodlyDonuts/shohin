# R12 Cursor-Conditioned Token-Tape Result

**Decision:** `external_cursor_token_tape_no_go`

**Claim boundary:** At raw step 260k, pre-final token states are not
renderer-invariantly readable by the frozen single-query attention plus linear
decoder family under its specified optimization recipe. This does not close
all token-tape or external-cursor mechanisms: post-final, multi-query,
nonlinear, and order-level access were not tested. It does not authorize a
reasoning, internal-cursor, retrieval, compositionality, actuation, or novelty
claim.

## Custody

- implementation commit:
  `2778d7999ce3539866c2df80d6a5dc4f975361af`;
- raw 260k base SHA-256:
  `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`;
- confirmation-free development view SHA-256:
  `24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9`;
- independent view-audit SHA-256:
  `33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25`;
- runtime SHA-256:
  `af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c`;
- Newton job `689976`, completed on `evc44` in 87 seconds with exit code 0;
- preserved Slurm log: `logs/r12_cursor_token_tape_689976.out`, 964 bytes,
  mode `0444`, SHA-256
  `246976f5177e9e790ebf3e3208e9ad7aafb57c298a550e13b77bf379f0a58b42`;
- immutable result:
  `artifacts/r12/cursor_token_tape_dev_v1.json`, 9,815,027 bytes, mode
  `0444`, SHA-256
  `7065401b13fd83b8a5b514be9a9b2a8cd5158af39abfa5464df43b368bd825e1`.

The H100 process received only the 5,760 train and 960 development cells. It
received no source canary, source audit, tokenizer path, or confirmation row.
The job reconstructed all source from the frozen commit, staged every input
node-locally, re-hashed it before load, removed ambient Python and preload
variables, and recorded the actual Python, PyTorch, dependency, and model
module paths. The result preserves preprocessing vectors, probe states, query
norms, attention summaries, and every development prediction.

## Scores

| Arm | Train cells | Dev cells | Dev non-DONE | Exact groups | Min non-DONE cursor | Min renderer |
|---|---:|---:|---:|---:|---:|---:|
| shared deep seed 0 | 99.60% | 41.25% | 26.56% | 0/192 | 25.00% | 40.00% |
| shared deep seed 1 | 99.36% | 50.42% | 38.02% | 2/192 | 28.65% | 48.75% |
| shared deep seed 2 | 99.86% | 48.33% | 35.42% | 1/192 | 26.56% | 43.96% |
| cursor-specific deep seed 0 | 99.97% | 56.98% | 46.22% | 11/192 | 29.69% | 50.83% |
| cursor-specific deep seed 1 | 99.46% | 57.81% | 47.27% | 10/192 | 29.69% | 56.46% |
| cursor-specific deep seed 2 | 99.57% | 57.50% | 46.88% | 3/192 | 21.88% | 57.08% |
| mean joint | 95.87% | 48.65% | 35.81% | 6/192 | 26.04% | 47.08% |
| embedding-only shared | 43.33% | 40.00% | 25.00% | 0/192 | 25.00% | 40.00% |
| position-only shared | 40.00% | 40.00% | 25.00% | 0/192 | 25.00% | 40.00% |
| source-deranged shared | 51.58% | 38.96% | 23.70% | 0/192 | 22.92% | 38.75% |
| raw deep shared | 58.07% | 42.50% | 28.13% | 0/192 | 25.00% | 41.88% |
| token-RMS deep shared | 96.30% | 37.19% | 33.72% | 0/192 | 25.00% | 34.38% |
| source only | 20.00% | 20.00% | 12.50% | 0/192 | 11.46% | 20.00% |
| cursor only | 40.00% | 40.00% | 25.00% | 0/192 | 25.00% | 40.00% |

The shared deep family fits train in all three seeds, so this is not an
optimization-inconclusive result. Its median development accuracy is 48.33%,
only 8.33 percentage points above the best matched control and below the frozen
10-point margin. No shared or cursor-specific replicate approaches the 95%
cell, renderer, and per-cursor gates or the 90% exact-group gate. Source-only
and cursor-only stay at their nominal ceilings, while source derangement falls
below the cursor shortcut.

The source-only ceiling is not a valid leakage detector here. A
cursor-independent unique prediction necessarily matches exactly one of five
cells because every source contains each target once. Its 20% result is
therefore mathematically automatic. Cursor-only and the parameter-matched
embedding, position, and source-deranged shared arms remain informative, but
the cursor-specific deep family lacks cursor-specific matched controls.

## Mechanistic diagnosis

The cursor-specific family's descriptive 56.98--57.81% score is concentrated
in the first instruction and deterministic DONE: cursor-zero
accuracy is 72.40--81.25%, DONE is 100%, but cursor one is 36.46--37.50%, cursor
two is 21.88--29.69%, and cursor three is 44.79--47.92%. Only 3--11 of 192
sources recover the full four-operation order in any seed. Because the shared
family misses its frozen ten-point matched-control margin and the
cursor-specific family has no parameter-matched controls of its own, this lift
is descriptive; it is not an authorized deep-representation finding.

The observed asymmetry is consistent with partial first-clause access and poor
later-clause routing, but the present experiment cannot distinguish that story
from unmeasured cursor-specific lexical or positional shortcuts. The 99% train
fit and poor renderer-held-out development performance do reject a stable
factorization through this exact probe family.

Raw and per-token-RMS controls reach only 58.07% and 96.30% train accuracy, so
neither satisfies the 99% train-fit condition. They are optimization-
inconclusive and do not resolve whether the standardized-arm behavior depends
on train-derived feature scaling.

## Decision and next boundary

Under the frozen preregistration, this pre-final single-query probe family
stops. Do not tune it on the exposed development result or present its lift as
a reasoning mechanism. Development includes all 24 permutations, so it cannot
support unseen-permutation extrapolation in any case.

A branch-wide representation conclusion would require a newly preregistered
fresh-data diagnostic that includes post-final tapes, a 24-way order readout,
cursor-specific embedding/position/deranged controls, and train-only renderer
holdouts for optimization selection. Gold-span layerwise paired-transposition
readout is the smallest diagnostic that can separate operation erasure from
failed routing. It cannot reuse this development split as confirmation.

Independently, any capability mechanism must create and causally update its own
sequential state rather than assume an oracle cursor. Before architecture code
or H100 training, the R12 invention charter still requires an explicit
state-transport theorem, an equivalence dossier, an exact collapse test, and a
finite synthetic falsifier with held-out permutations and fresh renderers.
