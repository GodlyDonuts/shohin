# R12 DRS Causal Cycle Result

**Status:** canonical r3 diagnostic completed; mechanically valid; residual-only
autonomous-cycle hypothesis rejected.

## Frozen execution

- Slurm job: `691847` on `evc33`
- Accounting: `COMPLETED 0:0`, elapsed `00:32:18`
- Report: `artifacts/evals/drs_causal_cycle_post_drs_r3.json`
- Report SHA-256:
  `0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a`
- Report mode: `0444` on Newton and the local mirror
- Audit identity: `drs_causal_cycle_post_drs_v3`
- Cases: 50, with ten from each frozen regime
- Mechanical validity: pass
- Cached identity token mismatches: 0
- Teacher-forced identity failures: 0

The checkpoint, heldout set, tokenizer, and five scientific-source hashes in
the report match the r3 preregistration. The job executed from its verified
private snapshot under canonical CUDA BF16 mode.

## Locked endpoints

| Endpoint | Result | Frozen decision |
|---|---:|---|
| Baseline first-state exactness | 38/50 = 76% | diagnostic |
| Counterfactual residual first-state exactness | 14/50 = 28% | write/serialization fail |
| Same-target residual first-state exactness | 31/50 = 62% | no native rescue signal |
| Direct two-token ceiling first-state exactness | 50/50 = 100% | pass |
| Paired next-call active-digit switch | 40/50 = 80% | insufficient for consumer pass |
| Integrated residual-authored two-call cycle | 9/50 = 18% | fail |
| Irrelevant-transplant argmax invariance | 49/50 = 98% | pass |
| Irrelevant-sham token equality | 49/50 = 98% | pass |

The counterfactual residual arm was below the preregistered aggregate threshold
and below the per-regime floor in `fit_w4` (10%) and `value_ood_w6` (20%). The
two-token ceiling was 100% in every regime.

Same-target residual replacement rescued 5/12 baseline failures (41.7%) but
reduced overall first-state exactness by 14 percentage points. It therefore
fails both native-residual rescue criteria.

The paired active-digit switch cleared its aggregate threshold, but the full
consumer gate failed because teacher-forced carry accuracy was only 60% for the
base state and 50% for the counterfactual state. Teacher-forced digit accuracy
was substantially stronger at 90% and 88%, respectively. Several per-regime
carry and digit rates also missed the frozen 50% floor.

## Diagnosis

The post-DRS model contains a causally active late digit-bearing residual, but
that fact is not an autonomous reasoning cycle. At the tested layer and
interface, the residual does not reliably author the required digit/carry text,
transport the intended state, and support the following transition. Directly
forcing the two target tokens removes the first-state failure completely, so
non-field serialization is not the bottleneck. The carry interface and its
next-step consumption remain materially weaker than the digit path.

This closes the decode-only interpretation of the residual workspace. It does
not show that a learned low-dimensional state bus is impossible. Any successor
must explicitly train and score the state-to-token actuator, carry update, and
unpatched multi-step consumption rather than treating a linearly decodable
residual as sufficient.

## Authorized next work

1. Keep the terminal-carry/width factorial curriculum as the data
   identifiability test; the old DRS corpus cannot distinguish the intended
   transition rule from the terminal-carry-zero alternative.
2. Admit a compact carry/cursor packet architecture only behind matched-token,
   matched-update, and ordinary-SFT controls.
3. Require autonomous two-step and width/value-OOD tests with no oracle residual
   or token injection before any reasoning claim.
4. Reject any architecture that grows the packet into a result tape, depends on
   generated-token KV state, or wins only through extra supervision or compute.

The report is a localization result, not a capability improvement and not a
reasoning claim.
