# R12 Counterfactual Conjugate Commit hypothesis

**Status:** theory only. No implementation or H100 launch is authorized before
the preregistered carry-only motor is scored. This document records a possible
successor if writer repair alone leaves a measured reader/cycle bottleneck.

## 1. Empirical premise

The post-DRS evidence localizes a transaction failure rather than a missing
local arithmetic rule:

- first transition after DRS SFT: 497/500;
- native frozen-cycle first state: 38/50;
- all 12 native failures first diverge at serialized carry;
- target result digit under teacher forcing: 50/50;
- final-layer carry linear-probe test accuracy: 694/800 = 86.75%;
- paired next-call active-digit switch: 40/50;
- autonomous integrated two-call cycle: 9/50.

For a fixed-width local transition, the operands, operation, cursor, and result
prefix are already serialized. The only persistent arithmetic state is one bit:

\[
F_x(c_p) = (r_p, c_{p+1}).
\]

The open failure is whether one semantic bit survives the complete map

\[
q_{in}(E(A(h_p))) = q_{out}(h_p),
\]

where `h_p` is the late residual, `A` emits the carry token, `E` embeds that
token on the next call, and `q_in/q_out` recover its semantic value.

## 2. Minimality and the C3-1 hypothesis

For each local symbol `x=(op,a_p,b_p)`, decimal execution is a two-state Mealy
transducer:

\[
F_x(c_p)=(d_p,c_{p+1}).
\]

Carry zero and one are behaviorally distinguishable by a one-symbol
continuation, while one bit realizes every local transition. The minimal
arithmetic quotient therefore has exactly two states. The cursor and immutable
tape remain visibly serialized; C3-1 is not allowed to add a hidden result
tape, retry loop, solver, or extra transformer pass.

Conditional on a successful frozen writer, learn one nonzero consumer direction
`u in R^576` at one fixed result-digit site. Define

\[
P_u={uu^T\over u^Tu},\qquad
K_u(h,c)=(I-P_u)h+(2c-1)u.
\]

This is a hard one-bit clamp rather than an additive hint:

\[
u^T K_u(h,c)=(2c-1)u^Tu.
\]

The carry-flip map

\[
G_u=I-2P_u
\]

is an exact involution and conjugates the committed token labels:

\[
G_u^2=I,\qquad K_u(h,1-c)=G_uK_u(h,c).
\]

The complete transaction is therefore

\[
h_p^{write}\to W_8\to token(c_{p+1})\to E(c_{p+1})
\to K_u\to h_{p+1}^{consume}.
\]

Training must still establish all 400 `(op,a,b,c)` local cells. Flip
equivariance alone is not evidence of arithmetic execution.

For a tied optimization geometry, freeze a writer-side carry axis `v_o` from
fit-only data and parameterize the consumer chart as a Householder transport of
that axis. A single Householder reflection can map `v_o` to any consumer axis,
so this tied arm and a freely learned `u` have the same hypothesis class. Any
advantage is optimization geometry, not expressivity.

## 3. Equivalence boundary

C3 is not yet a new computational primitive. A transformer can represent it
already, and its state machine is computationally equivalent to a two-state
finite transducer. The writer alone is an output adapter. The only potentially
useful contribution is the tied output/input group action, hard quotient
projection, and counterfactual commutation objective. It changes optimization
geometry, not expressivity.

The frozen rank-8 writer has 4,634 parameters. C3-1 adds exactly 576, for 5,210
total. With `(u^Tu)^-1` precomputed, the consumer costs one dot product, one
scalar projection, and one vector update: 1,153 multiplies and 1,152
adds/subtracts. It retains exactly one bit and adds no token, KV slot, hidden
persistent state, or sequential step.

## 4. Required matched arms

Only after a carry-writer result localizes a remaining reader failure:

1. frozen base;
2. equal-parameter additive consumer `h+(2c-1)u` with ordinary paired-row CE;
3. untied hard clamp with freely parameterized `u`;
4. tied C3-1 Householder transport from the frozen writer axis;
5. C3-1 with counterfactual pairs shuffled inside identical nuisance strata;
6. C3-1 without the `(I-P_u)` erase term.

All learned arms must share base, data, initialization scale, optimizer update
count, forward positions, and decoding. Primary endpoints are the frozen
50-case cycle, autonomous episodes, unseen widths, counterfactual selectivity,
and direct transcripts. Teacher-forced carry is secondary.

## 5. Discriminating predictions

1. Writer-only repair improves carry serialization; a true tied transaction
   gives additional integrated-cycle and full-episode gain.
2. Literal carry-token flips and `G_u` interventions agree on the next active
   digit. Donor carry interchange changes only the source case's local
   transition; double reflection restores baseline. Random orthogonal and
   irrelevant-result shams do neither.
3. Removing hard projection may preserve one-step behavior but loses accuracy
   as chain length grows.
4. A genuine one-bit mechanism transfers to widths 8 and 10. Fit-width-only
   gain falsifies the state claim.
5. All 400 local cells must pass. A persistent low cycle score after local
   success proves another cursor, tape, or control map is missing.

## 6. Kill criteria

Reject C3-1 mechanism support if any condition holds:

- the frozen writer is below 99% carry accuracy on a fresh board;
- local `(digit,next-carry)` exactness is below 99% overall or below 98% in any
  width/style/operation/carry stratum;
- the frozen cycle is below 45/50, fresh two-call exactness is below 90%, or
  autonomous full-trace exactness is below 90% separately at widths 4/6/8/10;
- tied C3-1 is less than +10 points over both additive and untied controls on
  cycles or less than +15 points on width-8/10 full traces on any frozen seed;
- shuffled pairing retains at least 50% of treatment gain;
- carry-flip interventions are below 95% selective;
- any non-DWS router fire or gate-off logit change occurs;
- removing hard projection changes long-chain accuracy by less than 2 points.

At 95% per-step reliability, a ten-step trace succeeds only `0.95^10 = 59.87%`
under independent errors; width-10 reliability of 90% requires at least
98.952% per step. Local accuracy alone is never the primary claim.

If the additive or untied adapter matches C3-1, reject the coupling claim and
retain the simpler interface repair. If every local gate passes but full traces
fail, reject one-bit carry as sufficient and localize the remaining
cursor/tape/control state instead of enlarging the packet speculatively.
