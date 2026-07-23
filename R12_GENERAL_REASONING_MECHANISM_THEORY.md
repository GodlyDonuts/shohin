# Endogenous Congruence Completion Reactor

**Status:** mathematical architecture theory; no implementation, training, or
capability claim
**Short name:** ECCR
**Protected base:** `train/flagship_out/ckpt_0300000.pt`
**Protected-base parameters:** 125,081,664
**Protected-base SHA-256:**
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
**Complete-system ceiling:** strictly below 200,000,000 unique parameters
**Date:** 2026-07-23

## 1. Conclusion First

The smallest capability missing from the retained Shohin mechanisms is:

> **model-owned induction and refinement of the causal congruence on which an
> episode's operations, compositions, and late observations are well-defined.**

S7 learns a law after the cyclic quotient has been chosen. QERARM selects and
composes operations after relation registers and their global algebra have
been chosen. AHRF propagates facts after binary-relation incidence has been
chosen. N-TCRR mutates graphs after constructors, types, rewrite sides, and
occurrence paths have been chosen. ABCR adds query direction, reversible
hypotheses, and conflict, but still receives an anonymous rule hypergraph whose
semantic roles have already been separated.

These are not minor defects. Each retained mechanism computes **inside a
preselected ontology**. Cross-family reasoning requires the model to decide
which physical distinctions are causal, which are aliases, which operations
act on the same state, and which composed paths must agree. That decision must
survive source deletion and must itself be causally consumed by later
computation.

ECCR makes that missing decision a hard recurrent object. It constructs:

1. an episode-local quotient from physical records to causal state classes;
2. anonymous generator actions that descend to that quotient;
3. a congruence over composed generator paths;
4. distinction certificates that prevent destructive over-merging;
5. a private state functor that applies the induced generators;
6. model-owned split, merge, rewrite, branch, commit, and halt transactions;
   and
7. a late-query reader whose observations factor through the same quotient.

This is not asserted to be sufficient for unrestricted intelligence. It is the
smallest common extension that can, in principle, turn the retained
family-specific reasoners into one source-deleted compositional mechanism.

## 2. Evidence Constraint

This proposal preserves the following hard conclusions from the existing
history:

| Retained result | What it establishes | What it assumes |
|---|---|---|
| S7 | Contextual unseen-law induction and exact recurrent reuse can work | A cyclic Cayley topology, exact equality, and bounded invocation |
| QERARM | A neural controller can own relation-algebra phase, work registers, and halt | A fixed global operation bank and fixed relation-register ontology |
| AHRF | A local neural field can, in principle, own closure and halt without a host executor | A monotone binary-relation fact ontology |
| TCRR mechanics | Deletion, branching, occurrence-specific mutation, cycles, and normal-form sets have an auditable transaction semantics | Typed constructors, rewrite cards, and graph ontology are supplied |
| ABCR theory | Query demand, reversible tentative state, and conflict backflow are plausible missing control variables | Premise/conclusion incidence and proof-state roles are supplied |
| Fresh physical compilers | Renderer-factor transfer and source deletion are possible on bounded grammars | The packet schema and task ontology remain fixed |

The common gap is not another state buffer. It is the absence of an endogenous
criterion for saying when two records or states are "the same for every future
computation" and when a superficially similar pair must remain distinct.

## 3. Formal Task Object

### 3.1 Episode presentation

A bounded episode is presented as

```text
E = (X, G, W, P, Omega, Tau)
```

where:

- `X` is a finite set of physical records, atomic state factors, or reified
  incidence records;
- `G` is a finite set of anonymous episode-local generators;
- `W_g` is a set or relation of witnessed transitions for generator `g`;
- `P` is a finite set of typed generator paths;
- `Omega` is a set of anonymous observation or late-query ports; and
- `Tau` contains only type and incidence structure.

The presentation may originate in language, a graph, a table, or another
renderer. Physical storage order, surface names, generator names, and query
names have no global meaning.

`X` is not a table of complete reachable world configurations. Higher-arity
records are reified into anonymous nodes and typed incidence edges. The live
task state is a finite relational structure over the resulting causal atoms,
so a bounded packet can represent exponentially many global configurations
without allocating one class per configuration.

The neural source compiler may emit candidate physical records, incidence,
and witness bundles. It may not emit a family label, canonical state ID,
global opcode, target quotient, execution schedule, normal form, answer, or
halt time.

### 3.2 Causal congruence

Let `~` be an equivalence relation on `X`. It is a **causal congruence** when
both conditions hold.

**Observation preservation**

```text
x ~ y  =>  omega(x) = omega(y) for every admissible omega in Omega.
```

**Generator compatibility**

```text
x ~ y  =>  delta_g(x) ~ delta_g(y) for every admissible g in G.
```

For nondeterministic systems, the second condition uses the powerset lifting:
successor sets must agree after quotienting. For probabilistic systems it is
the corresponding lumpability condition.

The desired quotient is the coarsest causal congruence supported by the
episode. It removes renderer and storage distinctions while preserving every
distinction that can affect a future operation or late observation.

### 3.3 Matrix descent invariant

In categorical form, let `F_E` be the model-emitted map from the physical
presentation to the private causal representation. Every learned generator
must make this square commute:

```text
F_E after T_g = A_g after F_E.                       (0)
```

This is a naturality/descent condition, not a request that the two
representations share coordinates.

Let:

- `C in {0,1}^{N x M}` be a hard row-one-hot assignment of physical records to
  causal classes;
- `T_g in {0,1}^{N x N}` be the physical witnessed-transition relation; and
- `A_g in {0,1}^{M x M}` be the induced private generator relation.

Using row-state convention, generator `g` is well-defined on the quotient
exactly when:

```text
T_g C = C A_g.                                      (1)
```

Equation (1) is the central executable invariant. It says that "act, then
forget physical detail" equals "forget physical detail, then act." A model
that merely finds addresses, stores a latent vector, or memorizes answers need
not satisfy this square.

Equation (1) is the finite unary-relation slice of (0). Higher-arity state is
represented by reified records and incidence, and the same square is required
for the complete emitted graph transaction.

For a query port with physical observation vector `o_q`, observational
sufficiency requires a private reader `r_q` such that:

```text
o_q = C r_q.                                        (2)
```

The actual late query may select `q` only after terminal state commitment, but
the private object must already be sufficient for every admissible query port.

### 3.4 Path congruence

Let `F(G)` be the free typed category generated by the anonymous episode graph.
A path is a typed word

```text
p = g_k ... g_2 g_1.
```

The model maintains an episode-local relation `equiv` over compatible paths.
It must be:

1. reflexive, symmetric, and transitive;
2. endpoint typed; and
3. closed under context:

```text
p equiv q  =>  a p b equiv a q b                  (3)
```

whenever both composites are typed.

The induced actions must respect path congruence:

```text
p equiv q  =>  A_p = A_q on every reachable class. (4)
```

where `A_p = A_{g_1} ... A_{g_k}` under row-state convention. The episode
therefore defines a finite presentation

```text
C_E = F(G) / equiv.
```

S7 is the one-object cyclic special case. QERARM is a fixed relation-algebra
special case. Bounded term rewriting is a path category whose arrows are
rewrite transactions. ECCR's new burden is to infer the quotient and path
congruence rather than receive either as the task ontology.

### 3.5 Separation invariant

Descent alone admits the useless quotient that merges everything. ECCR must
therefore certify both merge and split decisions.

A **merge certificate** for two classes is a finite relation `B` containing
their pair such that every pair in `B`:

1. has identical observations at every admissible query port; and
2. has generator successors that remain paired in `B`.

This is a bounded coinductive bisimulation witness. A merge may not be
justified by the model's failure to find a counterexample.

A **distinction certificate** is an admissible continuation path `w` and
observation port `q` such that:

```text
(e_i A_w) r_q != (e_j A_w) r_q.                     (5)
```

No certificate is needed for two physical records assigned to the same class.
Every two distinct private classes require at least one certificate within the
bounded continuation horizon.

Equation (5) is a bounded Myhill-Nerode condition. It prevents partition
collapse and gives every split a causal interpretation.

### 3.6 Reindex naturality

Let `P`, `S`, and `U` independently permute physical records, anonymous
generators, and query ports. There must exist only a private class relabeling
`Pi`, not a change in semantics, such that:

```text
C(P E)              = P C(E) Pi^T
A_{S(g)}(P E)       = Pi A_g(E) Pi^T
R_{U(q)}(P E)       = Pi R_q(E)
Eq_{S(p),S(p')}(P E)= Eq_{p,p'}(E).                 (6)
```

The exact left/right convention is frozen in implementation, but the
commuting content of (6) is not negotiable. Renderer recoding may change the
physical presentation. It may not change the private computation after
alignment.

### 3.7 Non-bijective presentation naturality

Generalization cannot stop at permutations. Let `H : E -> E'` be a typed
presentation morphism that may split one physical record into aliases, merge
bisimilar aliases, or reify a higher-arity relation into extra incidence
nodes. Let `H_bar` be the induced private map. The model must satisfy:

```text
H C'              = C H_bar
A_g H_bar         = H_bar A'_{H(g)}
R_q               = H_bar R'_{H(q)}.                (7)
```

Equation (7) is scored on independently generated split, merged, and reified
presentations. It is the stronger cross-presentation invariant. A model that
only memorizes storage permutations can pass (6) and still fail (7).

## 4. ECCR State

For batch `b`, hypothesis lane `k`, physical record `i`, private class `c`,
generator `g`, and path slot `p`, the hard recurrent state is:

```text
C[b,i,c]       physical-record to causal-class assignment
A[b,g,c,c']    anonymous generator action on private classes
R[b,q,c,a]     late-query observation map
Path[b,p,d]    bounded generator word
Eq[b,p,p']     path-congruence relation
Bisim[b,c,c']  positive merge/bisimulation certificate
Z1[b,k,r,c]    unary private state registers
Z2[b,k,r,c,c'] binary private relation registers
ZG[b,k,s,...]  reified private graph records and incidence
Cert[b,c,c']   distinction-certificate pointer
Ob[b,p]        unresolved descent/equation/conflict obligation
Alive[b,k]     absorbing execution state
```

Continuous record, generator, path, and obligation features support learning
but are not the scientific state claim. Promotion depends on the hard tensors
above and their interventions.

All class, record, generator, path, query-port, and lane indices are freshly
reindexed per episode. The architecture has no family embedding.

## 5. Model-Owned Transactions

One shared equivariant controller emits exactly one complete transaction per
tick:

```text
SPLIT(class, discriminator, child assignments)
MERGE(class_a, class_b, quotient remap)
INSTALL(generator, source_class, target_class)
ASSERT_EQ(path_a, path_b)
RETRACT_EQ(path_a, path_b)
APPLY(lane, generator)
FORK(lane, successor_a, successor_b)
COMMIT(lane)
HALT
ABSTAIN
```

The transaction includes every affected pointer and replacement tensor. A
fixed rule-blind committer may enforce only:

- tensor shape and pointer range;
- one-hotness and declared type compatibility;
- storage and probability conservation;
- branch and path-bank capacity; and
- exact installation of the transaction the model emitted.

The committer may not:

- compute a partition refinement;
- decide whether two records are equivalent;
- identify or apply a semantic operation;
- pattern-match a rewrite card;
- choose or join a critical pair;
- produce a distinction path;
- select an agenda item;
- repair a transaction;
- test an answer;
- detect convergence; or
- decide halt.

Invalid transactions remain in the denominator.

## 6. Recurrent Mechanism

### 6.1 Basal evidence and apical obligation

Physical witnesses propose local descent constraints. Late-query declaration
ports, path equations, and unresolved counterfactuals create obligations. The
controller receives both:

```text
support = witnessed transition/equality incidence
demand  = unresolved descent, observation, and path-congruence residuals
```

Their interaction is useful only because it selects a concrete transaction.
No biological analogy is part of the claim.

### 6.2 Conflict-directed refinement

For two records currently in one class, disagreement in an observation or in
the quotient destination of a witnessed generator creates a split residual:

```text
v_split(i,j) =
    observation_disagreement(i,j)
  + sum_g quotient_successor_disagreement(g,i,j).
```

For two separate classes, agreement under every compiled generator and query
port creates a merge proposal. A merge is not admissible unless the model
emits a positive `Bisim` certificate closed under those generators and
observations. Failure to emit a distinction certificate is never sufficient.

This is the only role retained from active-inference or conflict-backflow
language: prediction error must name the partition edge that caused it.
Undirected global "surprise" is not an executable mechanism.

### 6.3 Congruence completion

The model compares typed path pairs, predicts local equations, and proposes
critical overlaps. When two equivalent reductions disagree, the controller
must do one of three things:

1. refine the causal partition;
2. revise a generator action; or
3. install a joining path equation.

The host never chooses among them. A path equation becomes causal only if
deleting or transplanting it changes the predicted composed state.

### 6.4 Execution

For the unary/set-valued slice, once the active obligations for a lane are
resolved, state transition is:

```text
Z_next = Z A_g
```

over the Boolean, categorical, or probabilistic semiring declared by the
packet type. Binary relations and reified graph state use the corresponding
model-emitted typed transaction; the committer only installs it. Branching
uses exchangeable lanes and set-valued terminal collection. The tensor
contraction is generic architecture, not a family-specific semantic switch.
All answer-relevant semantics reside in the model-emitted `C`, `A`, `Eq`,
`R`, and transaction tensors.

### 6.5 Halt

The halt head reads:

- unresolved descent residual;
- unresolved observation residual;
- unjoined critical-pair obligations;
- unresolved distinction challenges;
- branch coverage;
- state velocity; and
- private terminal evidence.

`HALT` is valid only when emitted by the model. A fixed tick limit is a
fail-closed safety bound. Fixed-deadline readout is a control.

### 6.6 Late query

Before source deletion, the compiler stores anonymous query-key carriers and
their private observation ports, not the eventual query selection. After
terminal commitment, a separately encoded late query binds by nominal
equality to one query-key carrier and selects `R_q`.

Changing the late query may change the answer but must not change `C`, `A`,
`Eq`, the execution trajectory, or terminal `Z`.

## 7. Why ECCR Is Not An External Executor

The host does not know or choose:

- the causal classes;
- generator meanings;
- path equations;
- distinction certificates;
- operation sequence;
- branch agenda;
- terminal state;
- late-query answer; or
- halt time.

The only fixed operations are index-safe tensor installation, typed tensor
contraction, and finite-capacity storage. Those are architectural dataflow in
the same sense that attention, convolution, and matrix multiplication are
architectural dataflow. A semantic host executor would inspect a rule and
compute its consequence. ECCR forbids that.

The CPU oracle may create offline labels and assess a sealed transcript after
evaluation. It may not exist in the model process, return feedback, select a
candidate, or repair a failed transaction.

## 8. Why ECCR Is Not A Latent Scratchpad

A free latent scratchpad can store arbitrary vectors without committing to
what they mean. ECCR's score-bearing state has externally falsifiable algebraic
obligations:

1. hard quotient assignments must induce a valid equivalence relation, their
   normalized quotient projector must be idempotent, and both must be
   permutation natural;
2. every generator must satisfy descent equation (1);
3. every query map must satisfy observation factorization (2);
4. every path equation must satisfy contextual closure (3);
5. equivalent paths must act identically as in (4);
6. every merge must have a closed bisimulation certificate and distinct
   classes must have certificates as in (5);
7. all representation changes must satisfy naturality equation (6); and
8. non-bijective equivalent presentations must satisfy equation (7); and
9. interventions on `C`, `A`, `Eq`, and `Z` must produce different, predicted
   causal effects.

The mechanism is rejected if a norm-matched continuous-state control, a
decoder probe, or a generic recurrence reproduces the outcome without these
objects.

## 9. Representable Task Class

Define `FCR(N,G,D,K,B)` as the class of episodes satisfying:

1. at most `N` physical records and at most `K` causal quotient classes;
2. at most `G` anonymous generators;
3. generator and equation paths of length at most `D`;
4. branch width at most `B`;
5. a finite characteristic witness set that determines the quotient,
   generators, and query observations up to reindexing;
6. a finite causal congruence with finite merge certificates;
7. every required terminal outcome is reachable within the recurrence bound;
8. every pair of inequivalent classes has a distinction certificate within
   the continuation bound; and
9. complete task-state transitions factor through local generator morphisms on
   a finite relational structure; no reachable-global-state lookup table is
   supplied.

The class allows deterministic, finitely branching, and finite fixed-point
tasks. It does not require a common renderer, vocabulary, state cardinality,
or family label.

### 9.1 Representation theorem

**Proposition.** Given an exact ECCR packet with at least `K` private classes,
enough path and branch capacity, and exact model transactions, ECCR can
represent every task in `FCR(N,G,D,K,B)` exactly.

**Proof sketch.**

1. Let `~` be the episode's causal congruence. Assign one private atom slot to
   each equivalence class and set `C` to the quotient map. Encode each
   higher-arity physical record by an anonymous private record plus typed
   incidence to those atoms.
2. Generator compatibility and the merge certificates guarantee that each
   physical generator induces a well-defined relation `A_g` on quotient
   classes, so equation (1) holds.
3. Observation preservation guarantees a private reader `R_q`, so equation
   (2) holds for every late query.
4. Map each typed generator word to the corresponding product of `A_g`
   tensors. Quotient by the least contextual congruence generated by the
   episode equations. Equations (3) and (4) make the result independent of the
   selected path representative.
5. Extend each local generator to the finite relational state through the
   packet's typed incidence. By induction on path length, recurrent tensor
   application reaches the same quotient relational structure as the
   represented episode after every step. No full global transition table is
   used.
6. For nondeterministic rewriting, apply the powerset lifting and allocate one
   exchangeable lane per live branch. Induction on branch depth gives the exact
   reachable outcome set within capacity.
7. Distinction certificates ensure that no two behaviorally different states
   are forced into one class.
8. Since every terminal lies within the recurrence bound, the model can
   represent a valid halt or abstention observation for each terminal class.

This is an expressivity result, not a learnability result. The falsifier below
tests whether the neural system discovers the representation from finite
witnesses and transfers it.

### 9.2 Retained mechanisms as special cases

| Mechanism | ECCR embedding |
|---|---|
| S7 | One object, one learned cyclic generator, path equations from the cycle |
| QERARM | Relation-register states with anonymous algebra generators |
| AHRF | Monotone powerset state with closure generators and absorbing facts |
| TCRR | Term-graph states with rewrite generators and powerset branch lifting |
| ABCR | Obligation-driven transaction selection and conflict-directed splits |

ECCR does not replace their useful local motors. It supplies the missing
model-owned quotient and presentation on which a retained or learned motor can
be reused without a family-specific ontology.

### 9.3 Explicit limits

The proposition does not cover:

- unbounded tapes or recursion beyond configured storage;
- exact real arithmetic without a finite representation;
- episodes whose evidence does not identify a unique causal congruence;
- tasks whose required distinction word exceeds the bound;
- branching wider than the lane bank;
- source language the compiler cannot ground into physical records; or
- unrestricted theorem discovery.

Failure outside these limits is not evidence against ECCR. Failure inside them
under the frozen board is.

## 10. Decisive Falsifier: Congruence-Collision Orbits

The first experiment must not begin with broad natural language. It must first
test the missing capability directly.

### 10.1 Orbit construction

Each abstract episode generates six matched presentations:

1. **base:** one physical presentation;
2. **alpha:** complete record, generator, path, and query reindexing;
3. **split refinement:** one causal state is replaced by two bisimilar physical
   copies;
4. **merged presentation:** two observationally identical physical aliases are
   represented once;
5. **minimal noncongruent twin:** one continuation or late observation is
   changed so the same apparent aliases must be split; and
6. **path twin:** one pair is equation-equivalent in one episode and
   noncommuting or outcome-distinct in its matched twin.

All six presentations match:

- number and type of physical records;
- generator and query-port marginals;
- witness count;
- path-length histogram;
- in/out degree histogram;
- renderer length;
- answer distribution; and
- local one-step label distribution.

The base, alpha, split, and merged presentations have the same terminal
behavior. The minimal noncongruent and path twins require a different
quotient or composed result. A surface lookup, identity quotient, merge-all
quotient, and bag-of-witnesses model cannot satisfy both sides.

### 10.2 Family rotation

Optimization families:

1. cyclic contextual laws;
2. finite relation fixed points;
3. Horn/dataflow closure; and
4. algebraic/list rewriting.

Completely held-out families:

1. typed stack reduction; and
2. finite delete-effect planning.

Every local transition motif appears in optimization. Held-out families change
the state topology, composition pattern, branching, and terminal observation.
No family ID or family-specific head exists.

### 10.3 Split axes

| Partition | Physical records | Path depth | Composition | Families |
|---|---:|---:|---|---|
| Train local | 6-12 | 1-3 | primitive and two-step | four optimization families |
| Train autonomous | 8-16 | 2-6 | shallow mixed programs | four optimization families |
| Development composition | 12-24 | 7-12 | unseen equations and critical pairs | optimization families |
| Development scale | 25-32 | 4-12 | known motifs, unseen width | optimization families |
| Development family | 8-24 | 3-12 | unseen family topology | two held-out families |
| Confirmation | 16-32 | 8-16 | all axes jointly | all six families |

Canonical graph, quotient, path-equation, depth-2/depth-3 motif, and
presentation-orbit hashes are split-disjoint. Confirmation does not exist
until source, controls, thresholds, seeds, and independent assessment are
frozen.

### 10.4 Source deletion

The score-bearing evaluator contains only:

- immutable model weights;
- model runtime and rule-blind committer;
- source-deleted physical packets;
- a late-query packet revealed after terminal commitment; and
- hash-bound receipts.

Source text, compiler residuals, KV state, board generator, oracle,
partition labels, transitions, schedules, expected outcomes, and family IDs
are absent. Raw model transactions are sealed before an independent assessor
opens one oracle artifact.

## 11. Matched Arms And Causal Controls

### 11.1 Learned arms

1. **ECCR treatment:** full quotient, descent, path congruence, distinction,
   recurrent execution, and halt.
2. **Identity-quotient control:** every physical record remains distinct; all
   unused quotient/refinement parameters are reallocated to its recurrent
   processor.
3. **Merge-only control:** may merge but cannot split after a conflict;
   favorable to monotone AHRF-like abstraction.
4. **No-descent control:** predicts the same quotient and generators, but
   generator weights are not tied through equation (1).
5. **No-path-congruence control:** retains `C` and `A` but removes `Eq` and
   critical-pair obligations.
6. **Fixed-presentation TCRR control:** receives the same physical packet and
   compute, but uses one fixed term-graph ontology.
7. **Generic recurrent control:** identical hard-state capacity, trainable
   parameters, sequential ticks, and measured FLOPs, with unconstrained
   exchangeable recurrent slots.
8. **Family-specialized control:** separate processors per optimization
   family with the same aggregate parameters; held-out families use the
   averaged or routed processor without fine-tuning.

The generic recurrent control is qualified only if it reaches at least 99%
train joint and 95% in-distribution development joint.

### 11.2 Interventions

Required interventions are:

- record, class, generator, path, lane, and query-port reindexing;
- source/residual/KV poisoning after packet seal;
- quotient transplant with generators held fixed;
- generator transplant with quotient held fixed;
- equation deletion and equation transplant;
- correct split versus false split;
- correct merge versus false merge;
- distinction-certificate shuffle;
- equivalent-path substitution;
- noncommuting path reversal;
- private-state reset and donor-state transplant;
- fixed deadline, forced alive, early halt, and post-halt suffix mutation;
- late-query rotation with terminal-state invariance; and
- family-label probe and family-head ablation.

Each intervention has a preregistered predicted direction. "Score changed" is
not enough.

### 11.3 Oracle ceilings

The following are diagnostics and never reasoning claims:

1. gold quotient plus learned generators and reactor;
2. learned quotient plus gold generators;
3. gold private packet plus model-owned reactor;
4. gold complete private trajectory; and
5. independent CPU execution.

## 12. Optimization Contract For A Future Test

No training is authorized by this theory. A future preregistration should use:

```text
L = 1.00 L_partition
  + 2.00 L_descent
  + 1.00 L_observation_factor
  + 1.00 L_path_congruence
  + 0.50 L_merge_certificate
  + 0.50 L_distinction
  + 1.00 L_transaction
  + 1.00 L_terminal
  + 0.50 L_branch_coverage
  + 0.25 L_halt
  + 0.10 L_naturality
  + 10.0 L_invalid_soft
```

Training phases:

1. physical record and equality mechanics;
2. one-generator quotient descent;
3. two-path equation and distinction twins;
4. autonomous split/merge/install/apply transactions;
5. hard recurrent composition and branch coverage;
6. model-owned halt and late query;
7. cross-family mixed optimization; and
8. five frozen confirmation seeds.

Teacher forcing decays to zero. The final polish uses only hard recurrent state.
No single privileged path may supervise a set-valued outcome. Evaluation is
hard from the first tick.

## 13. Parameter And Resource Ledger

### 13.1 Parameter ceilings

These are component ceilings, not an instantiated count:

| Component | Added parameter ceiling |
|---|---:|
| Shohin physical-record and source adapters | 7,500,000 |
| Congruence signature, split, and merge module | 9,000,000 |
| Anonymous generator and equation compiler | 7,500,000 |
| Typed transaction and critical-pair reactor | 10,000,000 |
| Private state functor and branch bank | 6,000,000 |
| Obligation controller and halt | 3,000,000 |
| Late-query reader and serializer | 3,000,000 |
| Integration contingency | 2,000,000 |
| **Total added ceiling** | **48,000,000** |
| Protected Shohin trunk | **125,081,664** |
| **Complete-system ceiling for ECCR-1** | **173,081,664** |
| **Headroom below 199,999,999** | **26,918,335** |

All tied modules count once by parameter identity. An exact deduplicated
instantiated receipt is mandatory before any board seed.

### 13.2 Frozen geometry ceiling

| Resource | ECCR-1 ceiling |
|---|---:|
| Physical records `N` | 32 |
| Private causal classes `M` | 32 |
| Anonymous generators `G` | 16 |
| Query ports | 16 |
| Path slots | 96 |
| Path depth | 16 |
| Hypothesis/branch lanes | 8 |
| Unary private registers | 8 per lane |
| Binary private registers | 4 per lane |
| Reified graph slots | 32 per lane |
| Recurrent safety ticks | 128 |
| Continuous hidden width | 384 |
| Hard recurrent-state budget | 256,000 categorical bits per episode |
| Continuous recurrent-state budget | 16 MiB bf16 per episode |

The categorical and continuous state ceilings include every live lane and
register but exclude immutable source-packet storage. Exact runtime memory and
FLOPs must be measured, not inferred from analytic MAC counts.

### 13.3 Full resource vector

The first score-bearing preregistration must report:

- unique parameters and optimizer state;
- hard categorical bits and continuous bytes retained per tick;
- source bytes before seal and private-packet bytes after seal;
- training examples and oracle-generated labels;
- optimizer updates and training FLOPs;
- inference FLOPs and wall time per tick;
- sequential recurrence depth;
- branch and path-bank capacity;
- invalid/capped/halted denominator counts; and
- external semantic work, which must be zero during evaluation.

The treatment and generic recurrent control receive the same examples, update
count, hard-state capacity, tick count, and measured-compute envelope.

## 14. Promotion And Rejection Criteria

### 14.1 Mechanics gate

Before neural work:

- two independent CPU implementations agree on every exhaustive small case;
- quotient descent, observation factorization, path contextual closure,
  bisimulation certificates, and distinction certificates are exact;
- every orbit and noncongruent twin has the intended result;
- source deletion and custody tests pass;
- all tensor reindexings are exact; and
- no board, oracle, or family label is reachable from the model process.

### 14.2 Neural promotion gate

All five seeds must individually achieve:

- at least 99.5% exact unseen one-generator quotient successors;
- at least 95% exact quotient, generator packet, terminal state, and halt on
  canonical development;
- at least 90% exact joint in every unseen-composition, scale, renderer, and
  held-out-family cell;
- at least 95% exact on all-axes-at-once development;
- at least 99% learned halt or valid abstention, with at most 1% safety
  exhaustion;
- 100% hard descent, observation-factor, path-congruence, type, storage, and
  conservation validity;
- 100% record/class/generator/path/query/lane reindex invariance;
- at least 99% equivalent-presentation invariance;
- at least 99% minimal-noncongruent and noncommuting-twin separation;
- at least 99% correct quotient, generator, equation, state, and late-query
  intervention responses;
- at least 20 percentage points over every qualified matched learned control;
  and
- a paired 95% lower confidence bound above a 10-point treatment advantage.

Only after these gates may one sealed confirmation be created and read once.

### 14.3 Precise rejection rule

ECCR-1 is rejected without threshold repair if any of the following occurs:

1. canonical development is below 80% exact joint for any seed;
2. any held-out family is below 60% exact joint for any seed;
3. any hard descent, path-congruence, conservation, or custody violation
   occurs;
4. equivalent-presentation invariance is below 95%;
5. noncongruent-twin separation is below 90%;
6. the treatment is less than 10 points above a qualified generic recurrent,
   identity-quotient, or fixed-presentation control;
7. quotient or equation interventions are causally inert;
8. family identity is required by the executor or a family-specific head is
   needed;
9. the late query changes pre-query execution;
10. source poisoning after seal changes the terminal state;
11. the mechanism requires host matching, host completion, verifier feedback,
    answer selection, retry, or repair; or
12. the complete deduplicated system reaches 200,000,000 parameters.

A failure localizes one of three interfaces:

- quotient induction (`C`);
- generator/equation induction (`A`, `Eq`); or
- autonomous control (`Z`, obligations, halt).

It does not authorize wider reruns on the same board.

## 15. Novelty Audit

### 15.1 Known components

The following ingredients are established and are not claimed as new:

- Myhill-Nerode equivalence and automata minimization by partition refinement;
- bisimulation, behavioral equivalence, and transition-system lumpability;
- congruence closure and Knuth-Bendix-style critical-pair completion;
- free categories and quotienting a presentation by path equations;
- predictive and causal state representations;
- equivariant graph neural networks and exchangeable object slots;
- discrete neural algorithmic reasoning;
- generalist shared neural processors across predefined algorithms;
- source-deleted private memory and recurrent categorical state; and
- support/demand, competition, conflict, and active-inference motifs.

Relevant primary or technical sources include:

- Berstel, Boasson, Carton, and Fagnot,
  [Minimization of Automata](https://arxiv.org/abs/1010.5318);
- Deifel, Milius, Schroder, and Wissmann,
  [Generic Partition Refinement and Weighted Tree Automata](https://arxiv.org/abs/1811.08850);
- Hansen-Estruch et al.,
  [Bisimulation Makes Analogies in Goal-Conditioned Reinforcement Learning](https://proceedings.mlr.press/v162/hansen-estruch22a.html);
- Ibarz et al.,
  [A Generalist Neural Algorithmic Learner](https://proceedings.mlr.press/v198/ibarz22a.html);
- Rodionov and Prokhorenkova,
  [Discrete Neural Algorithmic Reasoning](https://proceedings.mlr.press/v267/rodionov25a.html);
  and
- Knuth and Bendix, *Simple Word Problems in Universal Algebras* (1970).

### 15.2 Proposed new combination

The possibly new contribution is the conjunction of:

1. a **model-emitted hard causal quotient**, not a latent metric alone;
2. **episode-local anonymous generators** required to descend through it;
3. **model-owned path-congruence completion** under typed composition;
4. **explicit distinction certificates** preventing quotient collapse;
5. **source deletion before autonomous execution**;
6. **late-query observational sufficiency** over the same quotient;
7. **one family-blind transaction machine** spanning closure, fixed points,
   rewriting, and planning;
8. **causal transplantation of quotient, generator, equation, and state as
   separately identifiable objects**; and
9. **matched controls and one-read custody** that distinguish a useful
   structural prior from a finite atlas or external executor.

This document makes no literature-priority claim. Each ingredient has close
precedents. The novelty hypothesis is that their exact combination makes
endogenous ontology formation the score-bearing recurrent operation rather
than a preprocessing objective or a host algorithm. A broader literature
review and independent equivalence audit are mandatory before publication.

### 15.3 Equivalence hazards

ECCR must be demoted if analysis shows it is equivalent, under the measured
resource vector, to any of:

- a family-routed finite-state atlas;
- ordinary partition refinement executed by the host;
- a fixed universal VM whose full program is supplied in the packet;
- TCRR with only renamed tensor fields;
- ABCR with an untested partition probe;
- generic recurrence plus auxiliary consistency losses;
- retrieval over presentation hashes;
- a verifier-driven search loop; or
- latent scratch space decoded only after the answer is known.

The treatment earns a distinct mechanism claim only when its hard quotient
and path congruence are necessary, intervene correctly, transfer across whole
held-out families, and beat the favorable controls.

## 16. Claim Ladder

1. **Theory only:** this document.
2. **Finite mechanics:** independent quotient/path oracles and matched twins.
3. **Neural quotient induction:** exact unseen causal classes and descent.
4. **Within-family composition:** autonomous hard recurrence and halt.
5. **Cross-family systematic reasoning:** held-out families and all-axis
   transfer with one processor.
6. **Shohin source integration:** neural physical records, actual source/KV
   deletion, and late-query binding.
7. **Natural-language reasoning:** post-training, direct interaction, public
   benchmark gain, and preservation controls.

Only rung 5 supports a bounded cross-family systematic-reasoning claim. Only
rung 7 may change Shohin's natural-language general-reasoning claim.

## 17. Final Decision

**Admit ECCR as a theory and CPU-falsifier candidate only.**

The main architectural conclusion is:

> Do not add another executor over a fixed packet ontology. Make the model
> construct the causal quotient that defines the ontology, force every learned
> generator and observation to descend through it, and require composed-path
> congruence plus distinction certificates before halt.

If this mechanism fails the congruence-collision board, the project should
reject endogenous quotient completion as the missing capability rather than
hide the failure behind more width, more recurrence, or broader SFT.
