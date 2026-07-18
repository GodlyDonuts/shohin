# R12 Source-Firewalled Separating Query Basis Theory

**Status:** theory and falsifier schema only. No implementation, data build,
fit, score, cluster copy, CPU publication, accelerator job, capability claim,
or novelty claim is authorized by this file.

**Decision in one sentence:** supervising one query-oblivious state against a
separating family of future questions can certify that the state has not aliased
the tested causal quotient, but it is predictive-state representation learning,
not a new reasoning primitive; the only surviving experimental conjecture is
that source-firewalled multi-query supervision may allocate source-processing
compute more efficiently and make Shohin's existing post-DRS state easier to
read, update, and reuse.

## 1. Empirical trigger and claim boundary

The immutable post-DRS probe establishes a narrow but strong fact. On the same
40 matched carry directions used by the raw-200k control, layer-29 residual
swaps move the DRS carry-token margin toward the source on `40/40` directions
with mean delta-logodds `+3.147395`; raw-200k is `20/40` with mean `+0.014188`.
DRS result-digit swaps are `40/40` toward-source at every tested layer 17--29.
The evidence spans fit width four/six, value-OOD width four/six, and width-OOD
width eight.

This proves neither autonomous state update nor reasoning. The prefixes are
teacher-forced, the intervention replaces a full residual vector, and each
measurement concerns one microstep. Autonomous digit-motor results further show
that a high-accuracy teacher-forced reader can fail to improve the closed loop.
The unresolved failure is therefore not simply absence of a local signal. It is
the joint problem of:

1. reading a task-relevant state rather than exploiting a global logit bias;
2. committing the read state before output corruption;
3. updating the state under a new event;
4. consuming the updated state on later steps; and
5. preserving unrelated language and direct-answer behavior.

The candidate in this document is a training and measurement protocol for those
properties. It is not claimed to be a new state ontology, architecture, or
computational class.

## 2. Formal object

Let `H` be a finite set of admissible histories, `C` a set of future
continuations, `Q` a set of late queries, and `A` a finite answer set extended
with inadmissibility symbol `bottom`. The task relation is deterministic here:

```text
R(h, c, q) in A union {bottom}.
```

Histories have the usual residual equivalence

```text
h == g  iff  R(h,c,q) = R(g,c,q) for every (c,q) in C x Q.
```

Write `S(h)=[h]` for the causal state. An event `e` induces the residual
derivative

```text
U_e(S(h)) = S(he),
```

and a late query observes

```text
O_(c,q)(S(h)) = R(h,c,q).
```

An exact realization remains an ordinary Moore/Mealy residual transducer up to
a change of coordinates. Nothing below changes that no-go theorem.

### 2.1 Separating query basis

A finite family

```text
B = {(c_1,q_1), ..., (c_k,q_k)}
```

is separating on a subset `S_0` of reachable causal states when

```text
s != t  implies  there exists b in B with O_b(s) != O_b(t).
```

Its response signature is

```text
Phi_B(s) = (O_b(s))_(b in B).
```

The basis need not be minimal or unique. Calling it a basis asserts only that
its signatures separate the frozen state set. It does not assert linearity,
independence, canonical coordinates, or a direct-product factorization. The
counterexamples in `R12_QUERY_KERNEL_FACTORIZATION_NO_GO.md` remain binding.

### 2.2 Source-firewalled realization

An encoder commits to a query-oblivious state before learning which basis query
will be asked:

```text
z_h = E(h).
```

After commitment, the original history bytes, source-token KV, generated result
tape, cached answers, and any host-computed semantic state are inaccessible.
Only `z_h`, the late continuation `c`, and query `q` may reach the consumer:

```text
D(z_h,c,q).
```

For recurrent use, the updater may receive only the prior committed state and
the current event:

```text
z_(he) = T(z_h,e).
```

Re-encoding the full history, reading previous generated answers, parsing an
emitted state, host arithmetic, retrieval, verifier correction, retries, or
source-visible attention after commitment changes the resource model and is an
ineligible treatment.

## 3. What a separating loss can certify

### Theorem 1: exact non-aliasing

Let `B` separate `S_0`. Suppose one deterministic encoder `E` and consumers
`D_b` satisfy

```text
D_b(E(h)) = O_b(S(h))
```

for every reachable `S(h) in S_0` and every `b in B`. Then

```text
E(h) = E(g)  implies  S(h) = S(g)
```

on `S_0`.

**Proof.** If `E(h)=E(g)`, every deterministic consumer receives the same input,
so every basis response is equal. Hence `Phi_B(S(h))=Phi_B(S(g))`. Separation
implies `S(h)=S(g)`. QED.

This is an injectivity certificate only on the frozen state set and query
family. It gives no minimality result. The representation may contain the
entire source, an arbitrary lookup key, or irrelevant information unless the
firewall and resource ledger exclude those paths.

### Theorem 2: approximate geometric separation

Let each true query response be represented by a distribution `P_b(.|s)`. Let
`B` be `gamma`-separating in total variation:

```text
for every s != t, some b has TV(P_b(.|s), P_b(.|t)) >= gamma.
```

Assume every learned decoder `Q_b(.|z)` is `L`-Lipschitz from state norm to
total variation and has uniform error

```text
TV(Q_b(.|E(s)), P_b(.|s)) <= eta
```

for all `s,b`, with `eta < gamma/2`. Then every distinct pair satisfies

```text
||E(s)-E(t)|| >= (gamma - 2 eta) / L.
```

**Proof.** For the separating `b`, the triangle inequality gives

```text
gamma
 <= TV(P_b(.|s), P_b(.|t))
 <= eta + TV(Q_b(.|E(s)),Q_b(.|E(t))) + eta
 <= 2 eta + L ||E(s)-E(t)||.
```

Rearrange. QED.

The bound is diagnostic, not a reasoner theorem. A highly non-Lipschitz decoder
or a weak empirical estimate of uniform error makes it vacuous.

### Theorem 3: representation sufficiency does not imply update correctness

For any injective `E` on a finite state set with at least two states, there is
an updater `T_bad` that is wrong on every non-self transition while every
immediate basis decoder remains exact.

**Construction.** Let `T_bad(z,e)=z` for every input. Immediate basis queries at
the encoded histories are unchanged and remain exact, but every event that
changes causal state is mapped incorrectly. QED.

Therefore a query-basis fit cannot promote autonomous reasoning. Update and
later consumption require separate interventions and closed-loop tests.

### Corollary: query-specific readers do not prove a shared state

If each query is allowed its own encoder `E_b(h)`, perfect answers do not imply
that any one state separates the quotient. Query identity must be hidden until
after a single committed `z_h` is frozen, and all consumers must be proven to
read those exact committed bytes.

## 4. Equivalence and prior-art dossier

The central object is established machinery:

- [Predictive Representations of State](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
  represents state by predictions of tests and proves that a linear PSR need
  not exceed the minimal POMDP state count. A separating future-query signature
  is a deterministic finite PSR / observable quotient.
- Bisimulation and latent-model learning, including
  [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html), train latent states
  to preserve reward/transition behavior. Adding an explicit updater and future
  observations is state-representation learning under the same boundary.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) trains a
  context representation to predict future latent content. Negative sampling or
  a contrastive query signature would be a CPC-family auxiliary loss, not a new
  primitive.
- Multi-task future prediction, successor features, auxiliary state losses,
  knowledge distillation, and ordinary supervised sufficient-statistic learning
  all cover nearby training objectives.
- A learned updater over the committed state is an ordinary RNN/Mealy
  transducer. Fixed-length execution can be unrolled into a feed-forward
  circuit. Any contribution must be resource-relative rather than ontological.
- The 2026
  [global-workspace study](https://transformer-circuits.pub/2026/workspace/index.html)
  provides evidence that some verbalizable residual vectors are broadcast and
  flexibly consumed. It motivates the intervention surface, but does not make a
  future-query auxiliary objective new or prove that Shohin can update the
  relevant state.

**Novelty decision:** rejected as a new reasoning primitive. The only reopenable
claim is a bounded training/oracle-allocation and causal-measurement protocol
under an explicit resource vector.

## 5. Surviving resource conjecture

Let `K` separating queries supervise one history prefix. A naive direct-SFT
baseline that presents the source separately for every query processes roughly
`K` copies of the source. A source-firewalled shared-state protocol can encode
the source once and apply `K` small consumers:

```text
naive source work       Theta(K * F_source)
shared-state work       Theta(F_source + K * F_consumer).
```

When `F_consumer << F_source`, this is a real source-processing advantage. It is
not automatically a training-FLOP, example, or oracle-call advantage:

- a packed multi-answer transformer may reuse source KV and erase the gain;
- one oracle that returns a complete state label may dominate both methods;
- constructing a separating basis may require more oracle calls than direct
  answers;
- the shared encoder may need greater width or optimization effort;
- an ordinary recurrent transducer may exploit exactly the same sharing.

The mandatory resource vector is

```text
(parameters,
 retained bits and bytes,
 numeric precision,
 source bytes read before and after commitment,
 source encodings,
 training examples,
 oracle calls and oracle output bits,
 training FLOPs,
 inference FLOPs,
 model calls,
 sequential depth,
 external memory,
 external execution,
 generated-token/KV bytes visible to update,
 wall time and accelerator allocation).
```

### Conjecture SQB-1

On a task whose reachable quotient has a compact separating query family,
source-firewalled multi-query supervision can reach a fixed autonomous
transition error with fewer source encodings than independently prompted direct
SFT, while matching an ordinary recurrent transducer in all other resources.

This is intentionally weak. If packed direct SFT or the favorable recurrent
control matches the source-encoding count and score, the resource claim is
rejected. A behavioral gain without matched resources is package-level only.

## 6. Exact collapse tests before neural code

Any executable successor must first pass a CPU-only symbolic suite.

### 6.1 Residual-table reconstruction

For a finite deterministic board, enumerate all reachable states, events,
continuations, and queries. Compute the exact residual table, minimize it by
partition refinement, and verify:

1. the declared basis separates every frozen quotient state;
2. every proper subset claimed nonseparating has an explicit collision witness;
3. appending the same event preserves declared equivalences;
4. every claimed distinction has a concrete continuation-query witness;
5. inadmissibility `bottom` is included in the signature.

### 6.2 Planted leakage negatives

The suite must prove that the validator rejects all of the following even when
they score perfectly:

- source-visible consumer after commitment;
- query revealed before or during state encoding;
- one encoder or cached answer per query;
- full-history re-encoding on every step;
- generated result tape or generated-token KV feeding the updater;
- host arithmetic, parser repair, verifier correction, retry, or search;
- answer table indexed by episode, seed, width, or prompt hash;
- source bytes hidden in padding, dtype slack, filenames, environment, timing,
  RNG state, allocator state, or external files;
- a nonseparating query family reported as separating;
- a stale identity updater that fits only immediate queries;
- score-dependent basis, seed, threshold, or confirmation selection.

### 6.3 Matched constructive controls

The same finite board must instantiate:

1. the minimal exact residual transducer;
2. a parameter/state-favorable ordinary RNN/Mealy realization;
3. direct SFT with independent prompts;
4. packed multi-answer direct SFT with source/KV reuse;
5. neutral and shuffled future-query objectives;
6. a source-visible upper bound;
7. a query-before-state leakage upper bound;
8. a favorable constant-logit calibration null at every typed readout;
9. a favorable nuisance-only calibration null that may use preregistered
   operation, width, cursor, and terminal-position metadata but no committed
   hidden state; and
10. retrieval and full-state-label controls with their resources disclosed.

If the proposed package reduces to any control with equal behavior and a
constant/polylogarithmic resource-vector overhead, the corresponding novelty or
resource claim is rejected.

## 7. Smallest Shohin falsifier schema

This section is not executable. A later byte-bound preregistration must freeze
every row, source, hash, seed, threshold, denominator, and consumer before any
fit.

### 7.1 Frozen parent and task

- Parent: immutable post-DRS checkpoint, not the raw 300k base.
- Backbone: frozen for the first falsifier.
- Aperture: one predeclared late residual/register location.
- Family: digitwise add/subtract transitions with exact protocol-derived
  oracles.
- Fit regimes: widths four and six.
- Development: disjoint width-four/six values and transition depths.
- Confirmation: fresh value-OOD widths four/six and width-OOD eight/ten.
- State is committed before a random hidden-independent query index is opened.
- After commitment, original prefix/source history is unavailable. A recurrent
  update receives only the previous committed state and the current typed event.

The exact query family must be derived from the minimized finite residual table,
not hand-picked after seeing model scores. It should include continuation-query
witnesses that separate carry/borrow, terminality, and every other retained
state distinction required by the declared task. Local digit labels alone are
not a separating basis for multi-step execution.

### 7.2 Arms

All arms use frozen train/development/confirmation rows and receive favorable
matching where exact matching is impossible.

| Arm | Purpose |
|---|---|
| `base` | Frozen DRS without new training. |
| `sqb` | Source-firewalled shared state, separating queries, and recurrent update. |
| `neutral` | Same query counts/tokens/compute, labels independent of causal state. |
| `shuffled` | Same label marginals and budgets, state-query associations permuted within frozen strata. |
| `direct_independent` | Separate direct prompt/answer examples for each query. |
| `direct_packed` | Favorable packed multi-answer SFT with shared source/KV. |
| `rnn_mealy` | Favorable ordinary recurrent transducer with the same state bytes, events, heads, losses, and compute. |
| `source_visible` | Upper bound whose consumer may reread source; ineligible for a reasoning claim. |
| `constant_bias` | Best grammar-gated state-independent logit calibration at every typed readout. |
| `nuisance_calibration` | Favorable grammar-gated calibration using only operation/width/cursor/terminal metadata, never the committed residual. Width extrapolation is fit on training widths before width-OOD reveal. |
| `stale_update` | Correct reader with identity/frozen updater, proving immediate query fit is insufficient. |

Treatment and `rnn_mealy` must share ordered data, semantic targets, loss
coefficients, optimizer, update count, precision, state allocation, consumer
capacity, source calls, and inference depth. If that leaves no treatment
variable, SQB is only the training package and no architecture attribution is
allowed.

### 7.3 Required interventions

1. **Query concealment:** changing the unopened future query cannot alter the
   committed state bytes.
2. **State swap:** swapping committed states between matched histories must
   redirect all basis answers toward the source state.
3. **Event swap:** holding state fixed and swapping the next event must redirect
   the successor state and later answers according to the exact derivative.
4. **Updater ablation:** zeroing or staling the updater must damage two-plus-step
   execution while leaving immediate reader accuracy largely intact.
5. **Consumer ablation:** ablating the reader must damage answers without
   changing the committed state identity.
6. **Source deletion:** the exact same scores must survive physical removal of
   all pre-commit source bytes available to the consumer/updater.
7. **Generated-output deletion:** removing emitted tokens and generated-token KV
   from the causal path must not change state transitions.
8. **Counterfactual continuation:** the same committed state must support fresh
   unseen separating continuations without re-encoding the prefix.

## 8. Noncompensatory decision gates

A later executable protocol may tighten these floors, but may not weaken them
after any score is observed.

### A. Custody and resource gates

- Every source, row set, implementation, runner, runtime closure, checkpoint,
  report, and receipt hash matches before and after execution.
- The protected base remains immutable and every candidate output is isolated.
- Total unique parameters remain strictly below `150,000,000`; with base
  `125,081,664`, additions must not exceed `24,918,335`.
- Post-commit source bytes, generated-output bytes, host semantic operations,
  retries, retrieval calls, and external execution visible to the causal path
  are exactly zero.
- The complete resource vector is machine-readable and replayed independently.

### B. Basis and representation gates

- The exact residual table proves that the frozen basis separates every scored
  quotient state; one counterexample is failure.
- Query identity is hidden-independent until after commitment; changing it
  changes zero committed-state bytes.
- Confirmation basis accuracy is at least `99%` on fit widths and at least
  `95%` on both value-OOD and width-OOD states.
- Every matched state-swap direction follows the source signature on every
  required query family; aggregate means cannot compensate for a failed family.

### C. Calibration and feature-use gates

- `sqb` must beat the favorable `constant_bias` arm by at least `10.0`
  percentage points on each of positive carry, negative carry, value-OOD, and
  width-OOD exact readout, with positive difference on every seed.
- `sqb` must also beat `nuisance_calibration` by at least `10.0` points on every
  confirmation stratum. Within-stratum label balance, not aggregate balance,
  is mandatory. A metadata-only tie rejects feature-dependent state reading.
- Gate-off logits are bitwise identical across `base`, `sqb`, and
  both calibration controls.
- `sqb` must beat shuffled and neutral controls on every confirmation stratum.

### D. Update and autonomous gates

- One-step successor-state signature exactness is at least `95%` on every
  confirmation stratum.
- Two-, four-, and eight-step exactness are reported separately. Every depth
  must beat `stale_update`; no shorter-depth gain can compensate for an
  eight-step failure.
- Width-eight and width-ten complete transition traces each reach at least
  `50%` exact and beat base by at least `15.0` points on every seed.
- `sqb` beats the favorable `rnn_mealy` arm by at least `10.0` points on the
  mean of value-OOD and width-OOD complete-trace exactness to support an
  SQB-specific package claim. If they tie, retain the simpler ordinary
  transducer and reject SQB-specific advantage.
- Autonomous execution uses no teacher-forced state/query answers and no
  post-failure repair.

### E. Preservation and interaction gates

- Existing frozen language, primitive, and direct-answer preservation boards
  do not regress beyond their preregistered confidence bounds.
- Fresh direct interaction includes unseen arithmetic transitions, state
  perturbations, requests for review, and compact-state reuse. Transcripts are
  read manually before any summary claim.
- A visible trace or correct final answer without an exact internal transition
  record cannot substitute for the autonomous state gates.

## 9. Kill conditions

Stop before H100 work if any of these holds:

1. the basis is nonseparating or was chosen after score access;
2. a packed direct or ordinary recurrent control preserves the resource vector
   and subsumes the claimed advantage;
3. source, query, output, host, retry, or retrieval leakage is nonzero;
4. constant or nuisance-only calibration matches the learned reader;
5. immediate query fit is high but event-update or later-consumption
   interventions fail;
6. width/value-OOD autonomous execution does not improve on every required
   stratum;
7. the parameter ledger reaches or exceeds `150,000,000`;
8. any custody identity, receipt, runtime closure, or confirmation secrecy gate
   fails.

No rename, wider head, extra recurrence, larger fit, or new seed may rescue a
failed exact gate without a new preregistration and independent review.

## 10. Final claim boundary

Passing Theorems 1--2 empirically would establish only that one committed state
is sufficient to answer the frozen separating query family without tested
aliasing. Passing the interventions and autonomous gates would establish a
bounded learned state-update/consumption result for the declared arithmetic
transducer. It would still not establish a new computational primitive,
consciousness, arbitrary natural-language reasoning, broad intelligence, or an
asymptotic separation.

The strongest admissible positive is:

> Under a frozen source-firewalled protocol and complete resource ledger,
> multi-query supervision produced a causally read, updated, and reused state
> on the declared held-out digitwise family, and outperformed the named matched
> controls by the preregistered margins.

Until those gates pass, the current conclusion remains narrower: DRS contains
a causally actionable local residual, while autonomous transport, update, and
reuse remain unsolved.
