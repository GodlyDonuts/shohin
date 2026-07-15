# R12 WGRQ CPU Preregistration: Delayed-Witness Edge-Parity Ring

**Status:** **CLOSED 2026-07-15 before any fit.** The frozen acquisition was
generated and independently replay-audited, but the adversarial implementation
audit found protocol-breaking defects described in Section 10. No Stage-A fit,
Shohin checkpoint fit, H100 job, language-transfer claim, or change to the
protected flagship is authorized from this version.

**Claim class:** empirical neural optimization under information-identical
frozen oracle transcripts. WGRQ is not a new state object, algorithm,
oracle-complexity result, recurrent primitive, or general-reasoning mechanism.

## 1. Exact family

For even `n >= 4`, the delayed-witness edge-parity ring `DWEPR_n` has physical
state `x in GF(2)^n`, initial state `0^n`, and two reversible events:

```
(R x)_i = x_(i+1 mod n)
F(x)_0 = x_0 xor 1, with every other coordinate unchanged.
```

The only query is `READ`, with output

```
O(x) = x_0 xor x_1.
```

There is no coordinate-selecting query. A future continuation must rotate a
latent difference to the fixed sensor.

### Residual theorem

For two states, let `d=x xor y`. Shared flips cancel from their difference and
shared rotations only rotate it. Therefore after a continuation containing
`r` rotations,

```
O(T_w(x)) xor O(T_w(y)) = d_r xor d_(r+1).
```

The states are future-equivalent exactly when

```
y=x  or  y=x xor 1^n.
```

Hence there are `2^(n-1)` residual classes. The canonical quotient is the edge
vector `e_i=x_i xor x_(i+1)`, whose even parity leaves exactly `n-1`
independent bits. `R` rotates the edge vector, `F` toggles edges `e_(n-1)` and
`e_0`, and `READ` returns `e_0`.

Every exact source-deleted packet therefore needs at least `n-1` history-
dependent bits. The canonical edge representation attains the bound.

### Delayed witnesses

For inequivalent states, let `b=e(x) xor e(y)`. Their shortest distinguishing
continuation is

```
R^k, where k=min{i : b_i=1}.
```

Because a nonzero even-parity `b` cannot have only its final bit set, the
maximum shortest-witness depth is `n-2`, and the bound is attained. `n=3` is
the smallest physical system with a nonempty worst-case witness; even scales
are used in the board for token-count controls.

## 2. Absolute symbolic gates

Before any fit, exhaustive enumeration must verify at `n=3` and `n=6`:

- physical transitions and reversibility;
- `x~y` iff all determining continuations `R^0...R^(n-2)` agree;
- exactly `2^(n-1)` quotient classes of size two;
- representative-independent quotient transitions;
- every shortest-witness depth and the tight `n-2` case;
- no collision or over-splitting in the serialized canonical code.

The minimum check count is

```
(n-1) * 2^(2n) + 3 * 2^n.
```

Any mismatch rejects the board before training.

Cancellation controls include `F F R^n` (identity),
`F R F R^(n-1)` (same counts, nonidentity), `F R` versus `R F`, the
observationally null global-complement word `G=(F R)^n`, and the equal-count
identity `(F F)^(n/2) R^n`. The generator must balance labels within declared
length, event-count, endpoint, and gadget strata wherever mathematically
possible and report the unavoidable parity obstruction separately.

## 3. Frozen acquisition

Training scales are `n in {4,6,8}` with two source-length bands, at most `2n`
and `8n`. There are exactly 3,072 episodes in each of the six scale/length
cells, or 18,432 episodes total.

Each episode contains four source histories:

- two distinct histories from one residual class;
- two histories from different residual classes;
- the non-equivalent pair is stratified over shortest-witness depths
  `0...n-2`;
- histories and pair roles are generated before model initialization.

Every history receives eight frozen continuation/read probes. The bank includes
all `n-1` determining rotations, adds the redundant final rotation, and repeats
the bank deterministically only when needed to reach eight. Thus every episode
contains exactly 32 one-bit ordinary oracle answers. Equivalence labels and the
first-distinguishing-witness mask are deterministic functions of those public
answers and add no oracle channel.

All arms receive byte-identical histories, probes, answers, equivalence labels,
witness masks, order, and batching. Training acquisition is exactly 589,824
ordinary one-bit answer calls. No model-dependent mining, target-dependent
rejection, reseeding, seed search, equivalence oracle, counterexample oracle,
or hidden state ID is allowed.

Generation uses

```
SHA256(seed || 0x00 || ASCII(domain) || uint64_be(counter))
```

with rejection sampling only for unbiased finite-bank selection. The generator,
auditor, transcript, report, and hashes are frozen before fitting.

## 4. Matched learner

Every neural arm uses exactly:

- a 15-bit packet, with only the first `n-1` bits active;
- a public 15-bit scale mask;
- a two-bit event code;
- tied transition MLP `32 -> 64 -> 15`;
- readout MLP `30 -> 64 -> 1`;
- 5,136 trainable fp32 scalars;
- straight-through hard bitpacking after every transition;
- no source tokens, cache, per-step parameter, oracle handle, or external
  execution in the committed packet or reader.

For hard bitpacking, probabilities are `sigmoid(logits)`, the forward packet is
`1[p>=0.5]`, and the backward value is the standard straight-through estimator.
At evaluation, exactly 15 bits are serialized; masked bits must be zero.

All arms use AdamW with learning rate `3e-4`, betas `(0.9,0.95)`, epsilon
`1e-8`, matrix decay `0.01`, gradient clip `1.0`, batch 64, four epochs,
exactly 1,152 updates, 64 warmup updates, then fixed cosine decay. There is no
dropout, early stopping, checkpoint selection, score-dependent scheduling, or
seed replacement.

## 5. Loss arms

All loss tensors are computed eagerly in every arm. Only frozen coefficients
differ.

Let `A` be answer BCE over every frozen probe. Let `E` be behavioral
Jensen-Shannon divergence between the equivalent pair over every shared probe.
Let `S_short` be a unit-margin separation hinge on the non-equivalent pair at
its first distinguishing probe. `S_uniform` uses a deterministic uniform probe
from the same bank. `R=(E+S)/2`. `R_sham` applies the same computation after a
deterministic wrong-partner permutation within scale, length, event-count,
endpoint, and answer-signature strata. Let `C` be the common mean
`p*(1-p)` bit-commitment penalty.

Primary arms:

```
WGRQ-shortest:       0.75 A + 0.25 R_short   + 0.01 C
active-answer-only:  1.00 A                  + 0.01 C
uniform-witness:     0.75 A + 0.25 R_uniform + 0.01 C
relation-sham:       0.75 A + 0.25 R_sham    + 0.01 C
```

A fifth favorable capacity control receives direct canonical edge-bit targets
but uses the identical model, optimizer, updates, and data. It is a privileged
ceiling, not an information-matched denominator. Exact symbolic partition
refinement is the non-neural ceiling.

The allowed positive claim concerns optimization only: the relational
objective may bias the same finite recurrent program toward the observable
quotient. It cannot claim new target information or a better oracle rate.

## 6. Process-level deletion and confirmation

Confirmation is generated only after every final checkpoint hash is frozen.
It has three untouched strata with 1,024 committed-history episodes each:

- length OOD: `n=8`, source length up to `64n`;
- scale OOD: `n=16`, source length up to `8n`;
- full OOD: `n=16`, source length up to `64n`, including witness depth `n-2`.

Each episode contains four histories and 32 continuation/read branches, exactly
128 ordinary one-bit answers. Total confirmation acquisition is 393,216 calls.

The writer receives one source history, serializes exactly 15 bits, and exits.
A fresh reader process receives only fixed weights, public scale mask, the
15-bit packet, one continuation, and fixed `READ`. It clones the byte-identical
packet for all 32 branches. Source events, source IDs, activations, RNG state,
cache, paths, simulator, verifier, and cross-branch memory must be absent.
The original packet must remain byte-identical after every branch.

`episode_exact=1` only when all normal reads, equivalent-history interchanges,
non-equivalent donor reads at selected witnesses, process-deletion checks,
masked-bit checks, and packet-reuse checks pass. Individual probes are never
independent scoring units.

## 7. Seeds and decision rule

The paired initialization/order seeds are frozen:

```
17011, 27103, 38119, 49201, 50311, 61403,
72503, 83609, 94709, 105019, 116027, 127031
```

All five neural arms run all 12 seeds: 60 fits. No failed seed is replaced.

Use 20,000 deterministic two-way paired bootstrap replicates. Resample seed IDs
and committed-history episode IDs while retaining every arm, history, probe,
and intervention in its cluster. Define

```
G = min over the three OOD strata of:
    WGRQ_episode_exact - 0.95
    WGRQ_episode_exact - AAO_episode_exact - 0.05
    WGRQ_episode_exact - uniform_episode_exact - 0.05
    WGRQ_episode_exact - sham_episode_exact - 0.05
```

GO requires all symbolic gates, privileged-edge ceiling accuracy at least
0.99 in every stratum, the simultaneous one-sided 95% bootstrap lower bound of
`G` strictly above zero, and at least 10 of 12 paired seeds beating active
answer-only by five points on full OOD.

Any oracle mismatch, transcript difference, source/cache leak, resource
mismatch, symbolic error, missing seed, masked-bit violation, failed stratum,
or near miss closes this version. It cannot trigger threshold, seed, board,
loss-weight, or hyperparameter changes.

## 8. Prior-art and allowed claim

The residual partition is Moore-machine minimization. Distinguishing
continuations are active automata-learning suffixes/homing experiments. The
committed state is a predictive state. Behavioral swaps are interchange/
bisimulation-style supervision. These boundaries forbid every primitive,
algorithm, oracle, and general-intelligence novelty claim.

The maximum claim after GO is:

> On a frozen delayed-observation reversible-ring family, shortest-witness
> relational loss improves exact source-deleted length and scale extrapolation
> for a minimal-bit tiny recurrent learner over information-identical neural
> controls.

No language bridge follows. `R12_CERTIFIED_LANGUAGE_BRIDGE_BOUNDARY.md` remains
a separate prerequisite.

## 9. Disjoint implementation namespace

Only these new paths are authorized for Stage A:

```
pipeline/wgrq_residual_oracle.py
pipeline/generate_wgrq_falsifier_v1.py
pipeline/audit_wgrq_falsifier_v1.py
pipeline/score_wgrq_falsifier_v1.py
pipeline/test_wgrq_residual_oracle.py
pipeline/test_generate_wgrq_falsifier_v1.py
pipeline/test_audit_wgrq_falsifier_v1.py
pipeline/test_score_wgrq_falsifier_v1.py
train/wgrq_state_machine.py
train/train_wgrq_cpu.py
train/eval_wgrq_cpu.py
train/test_wgrq_state_machine.py
train/test_train_wgrq_cpu.py
train/test_eval_wgrq_cpu.py
```

Any implementation need discovered outside this namespace requires a new
preregistration revision before editing.

## 10. Post-freeze execution and closure

Stokes job `739105` generated exactly 18,432 episodes and 589,824 ordinary
one-bit answer calls. Job `739106` independently replayed every history and
answer and passed the symbolic/data-admission audit. The immutable artifacts
are:

```
artifact                                      bytes       SHA-256
train.jsonl                              113675439       ae2849db5d57fda36e2e2fd634ce6e1d0f11eaed7fefe8d9ce722f016f28295a
ordinary_calls.jsonl                     188866874       251d85432d845c31ce64da1adae132fa8df8f6a63b5db744654b519f2413c9e8
generation_report.json                       23417       12c1e54f23b27f3a97a86857b723fec3573f5d558b7528e1615c55746899befb
audit_report.json                               6773       8f5fac80e0c50bdc807287599f8468194431f3612d6d79a1331f51a073fa2dd4
```

The acquisition is valid, but this version cannot fit or score a claim:

1. The relation-sham implementation rotates a whole sorted batch rather than
   deranging partners within each frozen stratum. On the exact board this
   creates 13,045 equivalent-relation and 13,905 non-equivalent-relation
   stratum mismatches. Thousands of declared strata are singletons, so the
   preregistered sham is not realizable on this acquisition.
2. The trainer expects obsolete audit fields and can accept the generator
   report instead of the independent audit, violating the admission barrier.
3. The scorer trusts supplied protocol booleans and `episode_exact` rows after
   hashing arbitrary checkpoint bytes. Its own positive test uses arbitrary
   text checkpoints and hand-authored evaluation rows, so the final decision
   can pass vacuously.
4. The independent auditor proves internal bundle consistency but does not
   itself require the generator's hard-coded frozen transcript, ledger, and
   report hashes.

Per the locked decision rule, these are version-closing mismatches rather than
post-score implementation details. No one of the 60 fits was launched. A
future version would require a new board with constructively non-singleton sham
strata, strict independent-audit binding, checkpoint/evaluation seals, and
end-to-end adversarial negative tests frozen before acquisition.
