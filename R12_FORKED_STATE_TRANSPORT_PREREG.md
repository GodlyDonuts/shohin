# R12 Forked State Transport Preregistration

**Status:** **NO-GO BEFORE IMPLEMENTATION.** Independent theorem audit showed
that the additive fork objective below has exactly the same population risk,
expected gradient, and minimizers as ordinary single-future supervision. No
CPU learner, Shohin fit, H100 job, production-data build, confirmation score,
architecture promotion, reasoning claim, or novelty claim is authorized.

**Frozen claim class:** a learnability hypothesis about a known recurrent
transducer trained with shared-prefix counterfactual obligations. The recurrent
state, event update, late-query observer, finite-state realization, and
predictive-state interpretation are not claimed as new primitives.

## 0. Decision and exact collapse

Let `z=(c,q,y)` be one continuation-query-answer obligation sampled
conditionally on prefix `h`, and let

```text
g_theta(h,z) = loss(O_theta(Fold_theta(c, initial=s_h),q), y).
```

The proposed normalized `K`-fork objective is

```text
L_K(theta) = (1/K) * sum_(k=1)^K g_theta(h,z_k),
z_k iid from P(. | h).
```

Linearity of expectation gives the **fork-risk collapse theorem**:

```text
E[L_K(theta)] = E_(h,z)[g_theta(h,z)] = E[L_1(theta)].
```

Under ordinary regularity conditions the expected gradients are also equal.
Materializing `s_h` once is common-subexpression elimination for grouped
ordinary examples; it is not a new learning signal.

Forking does not generically reduce total gradient variance. If
`m(h)=E[g' | h]`, then

```text
Cov(mean fork gradient)
  = Cov_h(m(h)) + (1/K) E_h[Cov(g' | h)].
```

By contrast, `K` independent prefix-obligation examples divide both terms by
`K`. Fork grouping trades prefix diversity for lower conditional continuation
variance and may save compute, but neither effect establishes systematic
transport.

There is also a finite-horizon nonidentifiability obstruction. For any exposed
horizon `H`, constant-dimensional recurrences can agree on every continuation
through `H` and disagree at `H+1`. One witness is `U(s,a)=s+1` with threshold
observers at `H+1/2` and `H+3/2`. No number of forks whose support ends at `H`
distinguishes them.

Therefore the learnability conjecture and implementation authority in the
historical proposal below are rejected. A non-additive worst-witness loss or
an explicit algebraic closure objective would be a different proposal and must
receive its own theorem, prior-art boundary, resource ledger, and preregistration.

## 1. The failure this test isolates

Shohin's recent controls can often learn a local transition or recover a weak
first-clause feature, but fail exact composition, later-position transport, and
unseen depth. The R12 token-tape result rejects only its frozen pre-final,
single-query attention plus linear-decoder family. It does not show that order
information is absent from every layer or that an internally updated state is
unlearnable.

This experiment asks one narrower question:

> At a fixed recurrent architecture, state width, parameter budget, optimizer
> budget, event budget, and answer-loss budget, does reusing one sealed prefix
> state across independently sampled continuation-query forks make the causal
> update law easier to learn than ordinary answer-only supervision?

The experiment does not test semantic parsing, natural-language transfer,
proof discovery, arithmetic skill, or general intelligence. Events are
provided through an oracle semantic boundary and that external information is
charged explicitly.

## 2. Capability object

For scale `m`, let events be the adjacent transpositions

```text
tau_i = (i, i+1),  i in {0, ..., m-2}.
```

For a word `w = e_1 ... e_L`, let

```text
pi_w = e_L compose ... compose e_1.
```

After the event word is sealed, a late query `q in {0, ..., m-1}` asks for
`pi_w(q)`. Histories with the same permutation are causally equivalent and
histories with different permutations have a distinguishing late query. The
exact causal quotient therefore has `m!` states and needs at least
`ceil(log2(m!))` dynamic bits.

The test includes repeated generators, involution cancellations, distant
commutations, braid-equivalent words, and non-equivalent order twins. A method
that relies on each operation occurring once is ineligible.

## 3. Axiomatic interface

The candidate interface is defined without neural-module vocabulary:

```text
z_t       = E(e_t)
s_0       = s_empty(m)
s_(t+1)   = U(s_t, z_t)
answer    = O(s_L, q)
```

The source is **sealed** after each event is consumed. While constructing or
using `s_L`, the mechanism receives no source token, source index, cursor,
source replay, retrieval key, KV cache containing the source, or external
executor result. The observer receives only the final state, scale, and late
query.

The CPU falsifier grants an oracle event encoder: it supplies the semantic
adjacent-transposition identity rather than asking the network to infer it from
language. The ledger therefore includes `L` oracle event calls and
`L * ceil(log2(m-1))` semantic source bits. Passing cannot authorize a language
claim; it can only keep state transport alive as a separate mechanism target.

## 4. Forked residual supervision

For a sampled prefix `h`, compute its state once:

```text
s_h = Fold(h).
```

Sample `K >= 2` obligations independently. Obligation `k` contains a
continuation `c_k` and a late query `q_k`. Reuse the same prefix state:

```text
s_hc_k = Fold(c_k, initial=s_h)
y_k    = O(s_hc_k, q_k)
L_fork = sum_k CE(y_k, R(h c_k, q_k)).
```

Gradients from every obligation meet at the same materialized prefix state.
No branch may recompute, copy from source tokens, or receive a branch-specific
prefix representation. The state is not supervised to equal a hand-authored
permutation. Only future behavior is supervised.

The proposed delta is **fork-consistent residual training**, not recurrence.
The hypothesis is that counterfactual obligations penalize prefix encodings
that are sufficient for one sampled answer but not stable under other futures.

## 5. Finite separation theorem

Let a finite deterministic board have reachable causal states `S`. Let `W` be
a finite set of continuation-query witnesses such that for every distinct
`s,t in S`, some `w in W` has `R(s,w) != R(t,w)`. Define the residual signature

```text
Psi(s) = (R(s,w))_(w in W).
```

### Theorem 1: witness-complete signatures are injective

`Psi` is injective on `S`.

**Proof.** If `Psi(s)=Psi(t)`, every witness in `W` gives the same answer. The
separation property says this is impossible for distinct `s,t`. Therefore
`s=t`. QED.

### Corollary 1.1: exact fork obligations can certify a finite quotient

Suppose a deterministic learned state and observer answer every witness in a
separating `W` exactly for every reachable prefix, and the same state is reused
for those obligations. Then two learned prefix states that are extensionally
equal under all observers cannot merge two distinct causal states on the
finite board.

This is a certificate theorem, not a learning theorem. Sampling a few forks,
fitting a finite training board, or obtaining low average loss does not imply
witness completeness. A finite model can still memorize every exposed prefix.

## 6. Rejected learnability conjecture

Fix the source-sealed recurrent architecture, state width, initialization
distribution, train examples, transition-call budget, optimizer updates,
answer-loss terms, trainable parameters, precision, and random seeds.

**Rejected conjecture FST-L.** On the frozen adjacent-transposition family, forked
residual supervision has higher exact unseen-length and unseen-scale causal
transport than the best matched answer-only recurrent control, because it
identifies more of the finite residual signature per materialized prefix.

The conjecture does not follow from the stated loss. It may show an
implementation-specific optimization effect under a frozen presentation, but
that would require comparing the exact same `(h,c,q,y)` multiset and complete
resource vector against shuffled grouping. It would not establish a reasoning
mechanism or systematic length generalization, so the planned CPU experiment
is not worth running.

The empirical claim requires all three:

1. exact answers to every late query after unseen event lengths;
2. equivalent-word state interchange with unchanged continuation behavior;
3. non-equivalent-state transplant effects that follow the donor state rather
   than the recipient source.

No asymptotic theorem or general reasoning claim follows from a finite pass.

## 7. Exact equivalence and collapse dossier

The computational mechanism collapses to established machinery:

- finite exact state plus event updates is a deterministic recurrent
  transducer and residual machine;
- a GRU/LSTM realization is tied recurrence;
- a source-conditioned transition table is a fast-weight or hypernetwork
  realization;
- retaining the source and rereading it is retrieval/source replay;
- fixed maximum length can be unrolled into a feed-forward circuit;
- exact permutation vectors are an oracle symbolic state;
- future-answer signatures are predictive-state representations.

Forked residual supervision is a multi-future training protocol over this
known interface. A positive result may support only a resource-matched
learnability claim. It is rejected as a distinct primitive even if it wins.

The resource vector frozen for comparisons is:

```text
(trainable parameters, dynamic state bits, precision, source bytes retained,
 oracle calls, training examples, answer-loss terms, transition calls,
 optimizer updates, training FLOPs, inference FLOPs, sequential depth,
 external memory, external execution).
```

Any favorable control may use the same recurrent implementation and full
budget. Extensional finite unrolling alone does not reject the learnability
hypothesis unless the reduction preserves this vector within constant or
polylogarithmic overhead.

## 8. Frozen CPU board

One implementation must support `m_max=12`; scale is an explicit input. The
score-blind generator freezes three disjoint partitions before fitting:

| Partition | Scales | Event lengths | Purpose |
|---|---|---|---|
| fit | `m in {5,8}` | `1..8` | optimizer data only |
| development | `m in {5,8,12}` | `10,12` | implementation diagnosis only |
| confirmation | `m in {8,12}` | `16,24,32` | one release after all hashes freeze |

Every partition contains:

- uniform random words with balanced generator counts;
- repeated-generator words;
- involution, distant-commutation, and braid-equivalent pairs;
- non-equivalent order twins with stored distinguishing queries;
- shared continuations appended to equivalent and non-equivalent prefixes;
- all late queries for each scored terminal state;
- balanced `m=2` parity as a separate depth-doubling sanity board.

No exact prompt, event word, equivalent rewrite, or normalized 13-event window
may cross partitions. Confirmation generation uses a committed seed that is
unavailable to the trainer and development scorer.

## 9. Arms and matched budgets

The minimum neural family uses one 64-wide state, one shared event encoder, one
shared state updater, and one late-query observer. The implementation must
publish exact parameter and MAC counts before fitting.

1. **FST treatment:** one materialized prefix state reused across `K=4`
   independent continuation-query obligations.
2. **Answer-only recurrent:** same network; four ordinary complete examples
   chosen so transition calls and CE terms match treatment.
3. **Recomputed-fork recurrent:** same obligations, but each branch recomputes
   the prefix state independently. This has favorable extra compute and tests
   whether shared-prefix gradient intersection matters.
4. **Reset-state sham:** same graph and labels, but reset the state at the fork.
5. **Label-shuffled sham:** same graph and marginals, with fork obligations
   deranged within `(m,length,query)` cells.
6. **Commutative pool:** parameter-favorable sum/mean event pool plus observer.
7. **Exact-state oracle:** exact permutation state plus the learned observer;
   establishes dataset/evaluator solvability.
8. **Source-visible control:** a favorable sequence model may reread the whole
   event source and is charged for retained source bytes and attention compute.

The treatment, answer-only, recomputed-fork, reset, and shuffled arms must have
identical trainable parameter counts, initialization hashes, optimizer updates,
answer-loss terms, and semantic transition-call counts. If exact matching is
impossible, the control receives the larger budget and the discrepancy is
reported before scores are read.

## 10. Causal tests

For each scored example, preserve the internal state bytes needed for the
following frozen interventions:

1. **Equivalent transplant:** replace a prefix state with one from a different
   word realizing the same permutation, then append the same continuation and
   query.
2. **Separating transplant:** replace it with a state from a different
   permutation and use a stored distinguishing continuation-query witness.
3. **Donor-following test:** the intervened answer must match the donor causal
   state, not the recipient source.
4. **Zero/reset state:** removes history while preserving continuation/query.
5. **Shuffled donor:** deranges state within matched scale/length cells.
6. **Source erasure:** after the state is formed, erase every source tensor and
   verify bytewise that the observer has no source handle.

An arm cannot pass by answer accuracy alone.

## 11. Historical decision gates (void)

The following gates record what the rejected experiment would have used. They
are void and authorize no execution. All percentages would have been exact
count ratios with every gate passing in all three seeds unless explicitly
described as a median comparison.

### Contract gates

- zero confirmation access before release;
- zero cross-partition exact or 13-event-window overlap;
- exact parameter/compute/state/source ledger for every arm;
- exact-state oracle at least 99.9% on every board cell;
- source erasure proves no post-seal source tensor or handle is reachable;
- no NaN, nonfinite state, evaluator fallback, or unscored row.

### Fit and capability gates

- treatment fit all-query accuracy at least 99.5%;
- treatment confirmation answer accuracy at least 98.33% per query;
- treatment confirmation exact-all-queries groups at least 90%;
- equivalent transplant invariance at least 99%;
- separating donor-following accuracy at least 95%;
- length-32 and `m=12` exact-all-queries each at least 85%;
- median treatment exact-all-queries exceeds the best non-oracle matched
  recurrent control by at least 10 percentage points;
- treatment wins that comparison in every seed.

The 98.33% per-query floor is chosen so a four-edge unique-action diagnostic
would have a 90% union-bound floor. It is retained here as a demanding local
accuracy gate, not as a proof that dependent errors obey the union bound.

### Automatic no-go conditions

- any treatment seed fails to fit;
- exact oracle fails;
- treatment uses source replay, hidden answers, state labels, or an executor;
- treatment fails unseen scale or unseen length despite passing fit;
- a matched recurrent control meets the same exact gates within 10 points;
- causal transplants do not follow the donor state;
- the result depends on selecting a favorable seed, width, board, or checkpoint
  after reading confirmation.

If every matched recurrent arm succeeds, the capability is learnable but the
forked-training delta is rejected as unnecessary. If only the source-visible
control succeeds, source-sealed transport is rejected for this budget.

## 12. Optional semantic-successor diagnostic

For a separate board where each action appears exactly once, an action identity
can key its semantic successor:

```text
head = first_action
sigma(action_i) = action_(i+1)
sigma(final_action) = DONE.
```

This representation is conjugate to an ordinal cursor and collapses to a hard
pointer, content-addressed attention, a fast-weight table, or a finite unrolled
lookup. It is not the main mechanism and cannot handle repeated identical
actions without adding occurrence identity, which restores an ordinary
position pointer. It may be used only as a diagnostic for whether semantic
addressing is easier to learn than ordinal addressing; it cannot rescue an FST
failure or support a novelty claim.

## 13. Release and authority

Independent adversarial review has rejected the objective. Implementation may
not begin. The implementation, generator, tests, fit manifest, scorer, and
confirmation board described here must not be created.

No hypothetical CPU pass under this additive objective would authorize a
Shohin canary. Any replacement still may not modify the base GPT forward path,
change the flagship, train on confirmation, or claim language reasoning, and a
language-facing experiment would additionally require a future-reflecting
certificate map under `R12_CERTIFIED_LANGUAGE_BRIDGE_BOUNDARY.md`.
