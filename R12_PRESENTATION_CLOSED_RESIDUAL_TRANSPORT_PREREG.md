# R12 Presentation-Closed Residual Transport Preregistration

**Status:** **NO-GO BEFORE IMPLEMENTATION.** Independent theorem and prior-art
reviews reject PCRT as a reasoning mechanism. No learner, confirmation board,
Shohin adapter, H100 job, production data, promotion, or reasoning claim is
authorized. The historical falsifier design is preserved below only so the
rejected proposal and its failure mode remain auditable.

**Claim boundary:** Presentation-Closed Residual Transport (PCRT) is a proposed
Shohin training protocol over an ordinary source-sealed recurrent transducer.
Recurrence, predictive/residual state, group presentations, robust optimization,
and relation consistency are known machinery. Until a complete prior-art audit
shows otherwise, PCRT is not a new primitive and is not claimed to be a
world-first method. Primary-source review found substantial overlap for every
ingredient and direct overlap for the Coxeter-relation/longer-word core. The
resource-matched conjunction is at most an optimization-control protocol.

## 0. Final decision and theorem audit

PCRT is closed for four independent reasons.

1. The universal generator premise
   `A(U(s,tau_i)) = T_i A(s)` supplies the complete recursive answer algorithm.
   Theorem 1 is a valid induction, but it assumes the capability PCRT was meant
   to discover. Supplying `T_i` is behavioral gold-successor supervision in
   the complete all-query answer chart.
2. Exact generator intertwining plus the identity anchor already implies every
   group relation. Presentation loss adds no exact feasible-set constraint; it
   can only reweight finite-sample optimization errors.
3. The historical approximate theorem omitted anchor error. The valid bound is

   ```text
   error(w) <= min(1, epsilon_anchor + |w| * epsilon_generator).
   ```

   A uniform observer has zero generator and relation defects while remaining
   wrong by `1 - 1/m`, which is a direct counterexample to the old statement.
4. A frozen finite continuation set is not proven to separate every learned
   reachable state. A depth-triggered learner can agree on every exposed word
   and collapse immediately afterward.

The proposed CPU board is also non-executable as written: the robust aggregate,
learner architecture, precision, optimizer, seeds, anchor count, source-erasure
test, raw artifact schemas, and independent score release were not frozen. Its
all-even `m=2` confirmation lengths cannot provide a balanced parity gate, and
the relation sham can accidentally preserve involution identities.

Primary prior art establishes the relevant boundaries:

- [predictive-state representations](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
  and [predictive-state decoders](https://papers.nips.cc/paper_files/paper/2017/hash/61b4a64be663682e8cb037d9719ad8cd-Abstract.html)
  supervise recurrent state through future observables;
- [AIDN](https://arxiv.org/abs/2012.01141) trains neural generator maps to
  satisfy defining relations;
- [MatrixNet](https://openreview.net/forum?id=b8jwgZrAXG) regularizes
  symmetric-group Coxeter relations and evaluates longer words;
- [group DRO](https://openreview.net/forum?id=ryxGuJrFvS) supplies worst-group
  objectives;
- [equivariant networks](https://proceedings.mlr.press/v48/cohenc16.html) and
  [homomorphism autoencoders](https://arxiv.org/abs/2207.12067) impose
  intertwining;
- [interchange intervention training](https://proceedings.mlr.press/v162/geiger22a.html)
  supplies causal donor tests.

No reviewed primary paper was found that combines every PCRT control in one
experiment, but that scoped absence does not rescue a primitive or reasoning
claim. At most, a future project could compare robust generator-only against
generator-plus-relation training as an optimization regularizer. That is not
worth a Shohin CPU or H100 run under the R12 invention charter.

## 1. Why the previous fork objective is closed

`R12_FORKED_STATE_TRANSPORT_PREREG.md` is a theorem-level no-go. Averaging
`K` continuation-query losses from one prefix has the same population risk,
expected gradient, and minimizers as ordinary single-future supervision.
Prefix reuse is common-subexpression elimination. A replacement must change
the objective, not merely group examples.

PCRT changes two things:

1. it minimizes the **worst observable residual defect**, not the mean defect;
2. it enforces the defining relations of the event action on states reached by
   different words.

Both channels are explicit training oracles and appear in the resource ledger.

## 2. Finite capability family

For scale `m`, events are adjacent transpositions

```text
tau_i = (i, i+1),  i in {0, ..., m-2}.
```

A word `w=e_1...e_L` acts as

```text
pi_w = e_L compose ... compose e_1.
```

After the source is sealed, late query `q` asks for `pi_w(q)`. The event action
has the Coxeter presentation

```text
tau_i^2 = I
tau_i tau_j = tau_j tau_i                  when |i-j| > 1
tau_i tau_(i+1) tau_i = tau_(i+1) tau_i tau_(i+1).
```

The board includes repeated generators, arbitrary multiplicity, equivalent
words induced by every relation, non-equivalent order twins, and lengths far
beyond training. A unique-action semantic-successor table is not used.

The causal quotient has `m!` states and requires at least
`ceil(log2(m!))` dynamic bits. Any fixed-state asymptotic claim is prohibited.

## 3. Source-sealed state and observable residual chart

The inference interface is

```text
s_0       = InitialState(m)
s_(t+1)   = U(s_t, e_t)
p_s(q)    = O(s, q) in Delta({0,...,m-1}).
```

Each event is consumed once. After consumption, the state updater and observer
receive no source replay, source cursor, source token, source-containing KV
cache, retrieval handle, external executor result, or gold state.

Define the all-query residual chart

```text
A(s) = (p_s(0), ..., p_s(m-1)).
```

For a generator `tau_i`, let `T_i` permute the answer alphabet by swapping
labels `i` and `i+1`. `T_i` is known from the semantic event contract and is
available only to the training loss and evaluator, never to inference.

## 4. Presentation-closed objective

### 4.1 Grounded anchors

The identity state is grounded by

```text
L_anchor = max_q CE(p_s0(q), q).
```

Additional absolute answer anchors may be supplied on a frozen minority of
short words. Their count and information content are charged. They cannot
include confirmation words or internal state labels.

### 4.2 Generator-intertwining defect

For every sampled reachable state `s` and admissible generator `tau_i`, define

```text
D_gen(s,i) = max_q TV(p_(U(s,tau_i))(q), T_i p_s(q)).
```

This compares future behavior, not latent coordinates. It is self-consistency
under the known semantic action. A constant observer fails the identity anchor.

### 4.3 Presentation defect

For relation words `u=v` in the frozen presentation and sampled reachable
state `s`, define

```text
D_rel(s,u,v) = max_(c,q)
  TV(p_(U_c(U_u(s)))(q), p_(U_c(U_v(s)))(q)).
```

The continuation set contains the empty word, every generator, and frozen
longer separating continuations. Taking observable residual distance avoids
penalizing harmless latent changes of coordinates.

### 4.4 Non-additive robust aggregation

PCRT minimizes an epigraph approximation to

```text
L_PCRT = L_anchor
       + lambda_gen * max_(s,i) D_gen(s,i)
       + lambda_rel * max_(s,u=v) D_rel(s,u,v)
       + lambda_abs * max_(grounded w,q) CE(p_(s_w)(q), pi_w(q)).
```

The implementation may use a frozen smooth maximum or top-tail CVaR only if
its temperature/tail fraction is committed before development scores. Mean
aggregation is a matched control. Resampling does not turn a mean into PCRT.

## 5. Exact all-length theorem

### Theorem 1: local intertwining implies all-length correctness

Assume deterministic one-hot observations on every reachable state and:

```text
p_s0(q) = one_hot(q)                                      for every q,
p_(U(s,tau_i))(q) = T_i p_s(q)                            for every reachable s,i,q.
```

Then for every finite word `w` and query `q`,

```text
p_(s_w)(q) = one_hot(pi_w(q)).
```

**Proof.** The empty word follows from the identity anchor. Suppose the claim
holds for `w` and append `tau_i`. The intertwining identity gives

```text
p_(s_(w tau_i))(q) = T_i p_(s_w)(q)
                    = T_i one_hot(pi_w(q))
                    = one_hot(pi_(w tau_i)(q)).
```

Induction proves the claim for every finite length. QED.

This theorem identifies a sufficient training contract. It does not say a
finite sampled learner satisfies the universal premise.

### Theorem 2: exact presentation closure factors through the group

If `D_rel(s,u,v)=0` for every reachable state, every defining relation, and a
separating continuation-query family, then observably equivalent words in the
Coxeter presentation induce the same residual behavior. Thus the observable
update action factors through `S_m` rather than the free word monoid.

**Proof sketch.** Every equality derivable from the presentation is a finite
sequence of relation substitutions inside word contexts. Closure under the
frozen separating continuations makes each substitution behavior-preserving;
transitivity completes the derivation. QED.

The theorem is observable, not latent: distinct latent vectors may represent
the same residual state.

### Rejected Theorem 3: approximate error law

The historical statement claimed that if answer-label permutations are TV
isometries and every reachable generator defect is at most `epsilon`, then
after length `L`,

```text
max_q TV(p_(s_w)(q), one_hot(pi_w(q))) <= L * epsilon.
```

This is false without an identity-anchor error term. A uniform observer has
zero generator defect because every `T_i` preserves it, but remains far from a
one-hot answer. If identity-anchor error is at most `epsilon_0`, the repaired
bound is

```text
max_q TV(p_(s_w)(q), one_hot(pi_w(q)))
  <= min(1, epsilon_0 + L * epsilon).
```

The repaired inequality follows by the triangle inequality and TV isometry.
It still assumes a supremum over every reachable state; a sampled smooth
maximum or CVaR does not certify that premise.

This correction further weakens PCRT: small sampled local error is not an
all-length certificate, and the exact universal premise already encodes the
answer algorithm.

## 6. What is and is not different

The inference mechanism collapses exactly to a known recurrent transducer. On
a finite board it is a finite-state machine; at bounded length it can be
unrolled. The all-query chart is a predictive/residual-state representation.
The presentation loss uses known algebraic relations, and the worst-case loss
is robust optimization.

The proposed empirical delta is narrower:

> At the same recurrent architecture, dynamic state, trainable parameters,
> precision, grounded labels, semantic event calls, optimizer updates, and a
> control-favorable compute budget, does worst-residual plus relation-closure
> training reach exact unseen-length transport where mean answer supervision,
> mean consistency, and relationless controls fail?

If prior work already uses this exact conjunction and causal gate, PCRT loses
even that method-level novelty and remains only a replication/control.

## 7. Oracle and resource accounting

The CPU test grants semantic event identity and the corresponding answer-label
action `T_i`. These are task-relation oracles. The ledger includes:

```text
event semantic bits supplied
T_i applications in training
relation identities supplied
continuation-query witness calls
absolute labels supplied
source bytes retained after sealing
trainable and fixed parameters
dynamic state elements and precision
transition calls
observer calls
optimizer updates
training and inference MACs
sequential depth
external memory and execution.
```

Passing only establishes state-update learnability after semantic compilation
has already been solved. It cannot cross
`R12_CERTIFIED_LANGUAGE_BRIDGE_BOUNDARY.md`.

## 8. Frozen CPU falsifier design

One shared implementation supports `m_max=12`. The source-sealed state has
enough measured precision-bits to exceed `log2(12!)`; no result may hide state
in Python objects, dataloader metadata, or model-global mutable storage.

### Partitions

| Partition | Scales | Lengths | Access |
|---|---|---|---|
| fit | `m in {5,8,12}` | `0..8` | optimizer |
| development | `m in {5,8,12}` | `10,12` | correctness repair only |
| confirmation | `m in {5,8,12}` | `16,24,32,64` | one score-blind release |

Scale generalization is not claimed because every scale is represented in fit.
The claim is unseen-length and unseen-word generalization under one uniform
parameterization. A later scale extrapolation requires a separate architecture
and fresh board.

Every partition contains all-query terminals, repeated events, generator-count
balanced random words, relation-equivalent pairs, non-equivalent order twins,
shared continuations, and a balanced `m=2` parity board. No exact word,
relation rewrite, or normalized 13-event window crosses partitions.

### Arms

1. **PCRT:** robust generator-intertwining plus robust presentation closure.
2. **Mean-answer recurrent:** same architecture, labels, updates, and favorable
   full-source recomputation; mean CE only.
3. **Mean-consistency recurrent:** same generator/relation examples with mean
   rather than max/CVaR aggregation.
4. **Robust relationless recurrent:** robust generator defect, no presentation
   relations.
5. **Relation sham:** identical robust graph with relation right-hand sides
   deranged inside matched `(m,length,relation-type)` cells.
6. **Source-visible sequence control:** may reread the complete source and
   receives a favorable parameter/compute budget.
7. **Exact permutation oracle:** evaluator and board-solvability ceiling.
8. **Hard local-swap control:** exact event routing and swap, charged as an
   external symbolic executor.

Known recurrent controls receive at least as many transition calls, observer
calls, absolute labels, optimizer updates, parameters, and MACs as PCRT. If
budgets cannot match exactly, the control receives the larger budget.

## 9. Frozen causal and relation tests

1. exact all-query answer groups by scale and length;
2. per-generator worst-cell error, not only mean accuracy;
3. involution, commutation, and braid closure from every sampled state;
4. equivalent-word state transplant followed by fresh continuations;
5. non-equivalent donor transplant with certified distinguishing queries;
6. source erasure after every consumed event;
7. zero/reset state and matched shuffled-donor interventions;
8. free rollout to length 64 without teacher forcing or state repair;
9. parity accuracy while length doubles;
10. exact resource and oracle ledger replay.

## 10. Frozen decision gates

Every gate must pass in all three committed seeds.

### Contract gates

- exact oracle scores 100% on every cell;
- zero confirmation access or cross-partition overlap before release;
- no nonfinite state, hidden source handle, evaluator fallback, or unscored row;
- exact replay of data, code, initialization, parameters, and resource hashes;
- source bytes reachable after sealing equal zero.

### PCRT capability gates

- fit all-query exact groups at least 99.9%;
- confirmation per-query accuracy at least 99.5%;
- confirmation exact-all-query groups at least 95% at each length through 64;
- each presentation relation at least 99.9% observably closed;
- equivalent-transplant invariance at least 99%;
- non-equivalent donor-following at least 95%;
- `m=2` parity at least 99% at every confirmation length;
- median length-64 exact-group score exceeds the best non-oracle matched
  recurrent control by at least 10 percentage points;
- PCRT wins that comparison in every seed.

### Automatic rejection

- any seed fails to fit;
- a mean or relationless matched control comes within 10 points;
- relation closure is high but donor state does not control answers;
- PCRT needs source replay, state labels, confirmation tuning, or external
  execution;
- success disappears when repeated events or free length-64 rollout is used;
- only a favorable seed, width, checkpoint, smooth-max temperature, or board
  passes after score inspection.

If all recurrent arms pass, state transport is learnable but the PCRT delta is
rejected as unnecessary. If only the hard swap passes, the learned update law
is rejected at this budget.

## 11. Authority sequence

1. adversarial theorem review;
2. primary-source prior-art boundary;
3. exact symbolic collapse and resource audit;
4. committed CPU implementation and unit tests;
5. frozen development execution;
6. independent score-blind confirmation generation and one release;
7. only after a full pass, a separately preregistered tiny Shohin canary with
   fresh data and matched controls.

No step in this sequence authorizes changing the flagship or the base GPT
forward path. A CPU failure closes PCRT at the tested resource budget.
