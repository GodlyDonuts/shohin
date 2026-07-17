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

## 2. Hypothesis

Counterfactual Conjugate Commit (C3) represents carry as a two-element group
action shared by the output and input interfaces, rather than as unrelated
features learned at each boundary.

Let `s(c)=2c-1`, and learn unit directions `v_o,v_i in R^576`. The output-side
projection and carry-logit actuator are

\[
Pi_o(h)=h-v_o(v_o^T h)+rho STsign(v_o^T h)v_o,
\]

\[
Delta(l_1-l_0)=2 alpha v_o^T Pi_o(h).
\]

At one fixed consumer layer, the next-call carry token applies

\[
h^{(k)} <- h^{(k)} + beta s(c)v_i.
\]

Carry-flip involutions

\[
G_o(h)=h-2(v_o^T h)v_o,\qquad
G_i(h)=h-2(v_i^T h)v_i
\]

are trained to commute with the external token swap `S`:

\[
A(G_o h)=S(A(h)),\qquad E(S(c)) \approx G_i E(c).
\]

A paired transition loss then requires counterfactual `c=0` and `c=1` inputs
to produce their respective exact local transitions.

## 3. Equivalence boundary

C3 is not yet a new computational primitive. A transformer can represent it
already, and its state machine is computationally equivalent to a two-state
finite transducer. The writer alone is an output adapter. The only potentially
useful contribution is the tied output/input group action, hard quotient
projection, and counterfactual commutation objective. It changes optimization
geometry, not expressivity.

The smallest form adds 1,152 direction parameters, retains exactly one bit,
uses no extra context token or transformer pass, and costs roughly `4*d`
multiply-adds per transition.

## 4. Required matched arms

Only after a carry-writer result localizes a remaining reader failure:

1. frozen base;
2. equal-parameter conventional carry writer;
3. untied writer plus reader;
4. tied C3 treatment;
5. C3 with counterfactual pairs shuffled inside identical nuisance strata;
6. C3 without hard projection.

All learned arms must share base, data, initialization scale, optimizer update
count, forward positions, and decoding. Primary endpoints are the frozen
50-case cycle, autonomous episodes, unseen widths, counterfactual selectivity,
and direct transcripts. Teacher-forced carry is secondary.

## 5. Discriminating predictions

1. Writer-only repair improves carry serialization; a true tied transaction
   gives additional integrated-cycle and full-episode gain.
2. `G_o` selectively swaps carry logits, while `G_i` changes the next active
   digit according to arithmetic. Random orthogonal directions do neither.
3. Removing hard projection preserves one-step behavior but loses accuracy as
   chain length grows.
4. A genuine one-bit mechanism transfers to widths 8 and 10. Fit-width-only
   gain falsifies the state claim.
5. Once writer and reader each exceed 95%, integrated success should approach
   their conditional composition. A persistent low cycle score proves another
   update/control map is missing.

## 6. Kill criteria

Reject C3 if any condition holds:

- less than +10 percentage points over the untied arm on the frozen cycle or
  less than +15 points on full episodes;
- carry reaches 95% but the frozen cycle remains below 25/50;
- shuffled pairing retains at least 80% of treatment gain;
- unseen-width gain is below 10 points despite strong fit-width gain;
- carry-flip interventions are not selective;
- any non-DWS router fire or gate-off logit change occurs;
- removing hard projection changes long-chain accuracy by less than 2 points.

If the untied adapter matches C3, reject the novelty claim and retain the
simpler interface repair.
