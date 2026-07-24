# Quotient-Fisher Causal Retraction Protocol

**Status:** v2 preregistration draft; no fresh result has been consumed.

## Motivation and prior boundary

The consumed ACSO v2 audit used four row-normalized Euclidean logit-gradient
updates with step `0.1` and recovered `0/672` faults at all three margins.
Post-hoc analysis on those consumed worlds found:

- the Euclidean cycle-zero direction was adverse on all `2,016` margin/fault
  cases;
- raising the same cotangent with the rowwise categorical Fisher
  pseudoinverse made the cycle-zero direction favorable on all `2,016`;
- four quotient-Fisher cycles with step `1.0` recovered all `2,016`; but
- four Euclidean cycles with the same step `1.0` also recovered all `2,016`
  and reached a slightly lower final innovation.

Therefore the consumed evidence does **not** attribute recovery to Fisher
geometry. It localizes the original failure primarily to the frozen `0.1`
step bound. This protocol asks whether quotient geometry has any fresh-world
or harder-margin advantage over the equal-step Euclidean control.

## Frozen hypothesis

For a categorical row with probabilities `p = softmax(z)` and a zero-sum logit
cotangent `g`, define the quotient-Fisher raised direction

`h = g / p - mean(g / p)`.

Normalize each row by `max(abs(h))` and update

`z_next = z - step * h_normalized`.

The treatment uses exact depth-three causal closure, four cycles, and fixed
step `1.0`. It adds zero parameters and uses no runtime autograd.

## Parameter boundary

The maximum HSC lane already adds `70,411,445` reasoning-specific parameters
to the protected `125,081,664`-parameter Shohin trunk, for `195,493,109`
complete parameters. QFCR changes categorical update geometry and adds no
parameters, so it does not leave an unused 70M allocation: HSC already spends
that allocation on source compilation and behavioral-code prediction.

Only after a separate model-predicted-signature protocol may up to `3,995,137`
parameters be reused for a permutation-invariant damping/trust controller.
That would produce `199,488,246` complete parameters and leave `511,754`
below the strict 200M ceiling. Such a controller may predict only global or
row-invariant trust scalars; it may not emit a learned coordinatewise update
direction without a new architecture and matched control.

## Fresh board

Use a two-phase public freeze:

0. The immutable experiment anchor is the already-public pre-QFCR commit
   `b93641619e25a8302cd46e96cfb6f20d5e657537`.
1. Commit and push the reviewed seven-file source freeze as the single direct
   successor of that anchor on public `origin/main`.
2. Create a second commit that changes only the fixed authorization file
   `artifacts/r12/qfcr_fresh_oracle_authorization.json`. It must name the
   source-freeze parent commit, exact NIST Beacon chain/pulse indices, and a
   minimum six-hour delay after the authorization commit's public GitHub
   PushEvent. It may not select or replace the trusted certificate.
3. Consume exactly that future pulse and its immediate predecessor. Verify
   both RSA signatures and output hashes independently under the reviewed
   source-pinned NIST certificate digest, then verify the current pulse's
   previous-output link and precommitment reveal.

Derive the board seed from the authorization commit and verified NIST output
under domain `efc-qfcr-fresh-oracle-v2`. The auditor must verify a clean
worktree, public `origin/main` equality, the authorization Git blob, and that
the authorization commit differs from its named parent only by the fixed
authorization file before board generation. It must also find distinct public
GitHub PushEvents for both the source-freeze parent and authorization child,
plus the immutable anchor, bind each exact repository, branch, before/head
SHA, and canonical event hash. Git parent relations and exact diffs prove each
is one commit because the public Events API omits commit-count fields. The
source commit must not
contain an authorization file and must change exactly the seven frozen paths.
The source and authorization commits must be the first two direct public
`main` successors of the unique anchor PushEvent. Any intervening, retired,
reset, sibling, repeated, deleted-authorization, or relabeled-source attempt
fails closed. If the anchor event is absent from the bounded public event
window, the audit fails rather than treating missing history as uniqueness.
The GitHub server `Date` on the consumed event response must be at least six
hours after the authorization PushEvent, matching GitHub's documented
worst-case Events API latency and the pulse delay.

Official execution must not import outcome code from the mutable working tree.
Pipe the committed bootstrap blob to isolated Python using the command frozen
in `pipeline/run_episode_functor_quotient_fisher_retraction_frozen.py`. That
stdlib-only bootstrap verifies clean public `origin/main`, creates a detached
worktree at the exact public Git object, and starts a second isolated Python
process there. The audit source closure includes the bootstrap even though the
child does not import it. Direct invocation of the auditor is development-only
and cannot publish an official result.

Before entropy consumption, the detached child fixes Torch intra-op and
inter-op threads to one and enables deterministic algorithms. The
authorization precommits the canonical SHA-256 of the expected runtime
receipt before the pulse. Generate it only through the source commit's
committed-blob bootstrap `--environment-output` mode, which uses the same
detached `python -I` child as the later audit. That receipt records and
post-outcome rechecks the
resolved Python executable and SHA-256, Python runtime identity, the complete
non-cache Python standard-library tree, platform/machine/processor, Torch
version/default dtype, the complete non-cache Torch package tree, every Torch
native extension/library hash, native dependency link maps and resolvable
dependency hashes, the complete filtered CPU model/cache/feature receipt, the
actual native images loaded into the process, OpenSSL
executable/version/SHA-256,
deterministic/thread settings, and all environment variables whose names can
select Torch, CPU dispatch, BLAS, CUDA/MPS, or thread behavior. Any
authorization mismatch, changed tree, or missing receipt fails before board
generation. This is an environment-bound CPU result, not a cross-platform
bitwise claim.

Generate exactly the existing `96/48/32/24`
train/mechanics/development/confirmation latent-world counts. The generator
emits multiple renderer sources per latent world; group them by `world_id`,
verify their machine/split/family identities agree, and enumerate faults once
per unique world. No generated world may be rejected or replaced after
outcomes are computed.

For every world, enumerate every deep transition fault satisfying:

1. the wrong and correct destinations have identical immediate answers under
   both observers; and
2. their exact depth-three base signatures differ.

The audit aborts before outcomes unless:

- all 200 worlds are present;
- at least 50 worlds contain a deep fault;
- at least 300 unique deep faults exist; and
- all four splits contribute at least one eligible world.

All eligible faults are scored. No subsampling is allowed.

## Margins and arms

The frozen margins are `0.05`, `0.10`, `0.20`, `0.40`, and `0.80`. Each
machine begins with exactly one wrong, untied transition row and otherwise
correct categorical winners.

Run four equal-cycle, equal-radius cycles for:

1. **QF causal treatment:** quotient-Fisher direction, step `1.0`, full
   depth-three closure.
2. **Euclidean equal-step control:** ordinary row-normalized logit cotangent,
   step `1.0`, full depth-three closure.
3. **QF small-step control:** quotient-Fisher direction, step `0.1`, full
   depth-three closure.
4. **QF one-step control:** quotient-Fisher direction, step `1.0`, but only
   base words through length one and derivative words at length zero.

Every arm receives the same initial logits, exact target tensors, cycle count,
maximum row update, hardening rule, and evaluator. Fisher raising has modest
additional arithmetic, so no equal-FLOP claim is made. The Euclidean control
is the decisive attribution control.

## Recoding and evidence

Repeat every arm under one deterministic, nonidentity permutation of states,
actions, observers, and answers per world. Record, for every cycle and case:

- base, derivative, and total innovation;
- exact-machine recovery;
- intended-row recovery;
- categorical ties; and
- correct-minus-wrong logit gap.

The report must bind:

- source commit and the complete pre-outcome runtime project-module closure,
  rechecked after outcomes with no newly loaded project module;
- public GitHub PushEvent, fixed authorization blob, verified NIST pulse,
  previous-link, certificate, and derived-seed receipts;
- generator seed and exact split counts;
- canonical board manifest;
- ordered fault inventory;
- exact evidence identity set;
- arm configuration;
- recoding receipt;
- output reservation; and
- final payload hash.

Recoded hardened transition and observer tensors must be inverted into the
original categorical coordinates and compared exactly at every cycle; summary
booleans are insufficient. Any missing, duplicate, nonfinite, changed, or
extra evidence fails closed.

## Decisions

### Mechanics pass

QF causal mechanics pass only if, at every margin and represented world:

- treatment exact-machine and intended-row recovery are 100%;
- treatment has zero ties and zero per-case monotonicity violations;
- recoded treatment decisions match exactly;
- maximum recoding innovation delta is at most `1e-5`; and
- all custody, source, count, and evidence bindings pass.

### Geometry attribution

Fisher geometry is attributed only if the mechanics pass and:

- treatment is never worse than equal-step Euclidean exact recovery at any
  margin;
- treatment beats equal-step Euclidean by at least 5 percentage points at one
  or more frozen stress margins (`0.40`, `0.80`); and
- treatment reaches exact recovery at least one cycle earlier in median at one
  or more margins where both arms finish at 100%.

If treatment and equal-step Euclidean tie, the decision is
`step_scale_sufficient_qfcr_not_attributed`. QFCR must not be promoted as a
reasoning primitive.

### Failure

- A mechanics failure kills this fixed QFCR rule.
- A mechanics pass without attribution retains QFCR only as a valid intrinsic
  control and closes the architecture claim.
- A geometry-attribution pass authorizes a separate model-predicted-signature
  protocol. It does not authorize HSC integration, fitting, native reasoning,
  or pretraining.

Gauss-Newton, learned coordinatewise directions, adaptive cycle counts, line
search, altered margins, or altered thresholds are outside v2 and require a
new named protocol.
