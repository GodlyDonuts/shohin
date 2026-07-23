# Endogenous Sheaf-Join Workspace Preregistration

**Short name:** ESJW

**Status:** theory preregistration only. No ESJW implementation, corpus, fit,
score, checkpoint, GPU job, or capability claim exists at this freeze.

**Date frozen:** 2026-07-23

**Protected base:** `train/flagship_out/ckpt_0300000.pt`

**Protected base parameters:** 125,081,664

**Protected base SHA-256:**
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`

**Complete-system limit:** strictly less than 200,000,000 learned parameters.

## 1. Decision Being Tested

Shohin often identifies a locally legal operation, local relation, or local
state, but loses the computation when several such decisions must share
variables, bindings, order, and termination. ESJW tests one specific causal
hypothesis:

> Shohin's local-competence/composition gap is partly caused by committing
> locally plausible states before proving that they are restrictions of one
> coherent global computation. Preserving explicit local hypothesis sets and
> joining only boundary-compatible hypotheses should transfer composition more
> reliably than averaging local vectors or independently decoding fields.

The proposed mechanism has four inseparable parts:

1. each locally solvable clause produces a finite **fiber of hypotheses** rather
   than one committed vector;
2. each physical overlap has a model-produced **restriction code** describing
   how a local hypothesis appears on the shared boundary;
3. signed gluing residuals persist as an internal **dual defect state**; and
4. unresolved or ambiguous overlaps trigger a model-owned **natural join** that
   creates a larger chart whose hypotheses explicitly retain parent
   correlations.

The mechanism is source-deleted and differentiable. No host process applies a
task rule, selects a schedule, executes arithmetic, searches for an answer,
repairs a join, or retries a failed decode.

A positive first experiment would establish a bounded architecture-native
composition mechanism. It would not establish unrestricted language
understanding, open-domain planning, theorem proving, or genuine general
reasoning.

## 2. Evidence That Fixes The Target

The proposal is constrained by the following frozen observations rather than
by analogy:

- Source-scheduled continuation reached 115/256 = 44.92%, while autonomous
  whole-problem generation reached 9/256 = 3.52%.
- DRS produced the correct first local state on 497/500 core episodes but only
  275/500 complete final answers.
- A real post-DRS digit workspace exists, including roughly +31 causal
  log-odds under digit swaps, but readable state did not create a reliable
  autonomous state-update cycle.
- The factorized N-TCRR motor reached 0/96 exact train and 0/32 exact
  development transactions. It often selected plausible train rule, path, and
  binding factors, but its independently decoded graph delta left only 2/96
  valid train commits and 0/32 valid development commits. Held-out path
  localization was 0/32.
- The eight-round pairwise ECCR inducer reached 45/64 = 70.3125% exact fresh
  development quotients. A by-construction equivalence decoder made all 64/64
  outputs valid equivalence relations but reached only 44/64 = 68.75% exact.
  Its errors were 59 false collisions and zero false splits; noncommuting
  contexts reached 3/16 and minimal noncongruence reached 2/8.
- AHRF supplies a learned recurrent relation field with write-once facts and
  model-owned halt, but no score-bearing AHRF report existed at this
  preregistration freeze. Its fixed relation-field ontology therefore cannot
  be cited as positive evidence.
- COFC correctly separates physical occurrence from nominal identity and
  proposes coherent source parsing, but its current role is compiler-side.
  ESJW instead tests source-deleted semantic composition after local evidence
  has been compiled.

These results reject three easy stories:

1. pairwise validity alone is the missing mechanism;
2. one larger recurrent vector will necessarily preserve correlations; and
3. independent local heads become a coherent transaction merely by receiving
   more capacity.

## 3. Why This Is Mathematically More Than A Metaphor

ESJW uses the literal gluing object from a finite sheaf-like system.

Let `C` be a finite set of active charts. Each chart `c` has a finite local
hypothesis set

```text
H_c = {1, ..., A_c},       1 <= A_c <= A_max.
```

For every overlap edge `e=(c,d)`, let `B_e` be a finite anonymous boundary
alphabet. A restriction map sends a local hypothesis to its boundary value:

```text
rho_(e<-c): H_c -> B_e
rho_(e<-d): H_d -> B_e.
```

A family of local hypotheses `(a_c)_(c in C)` is a **global section** exactly
when

```text
rho_(e<-c)(a_c) = rho_(e<-d)(a_d)
```

for every overlap edge `e=(c,d)`.

The hard compatibility matrix is

```text
J_e[a,b] =
    1  if rho_(e<-c)(a) = rho_(e<-d)(b)
    0  otherwise.
```

For adjacent charts, the natural join is

```text
H_(c join_e d) = {(a,b) in H_c x H_d : J_e[a,b] = 1}.
```

### 3.1 Exact join lemma

Assume exact restriction maps and no hypothesis pruning. Replacing charts
`c,d` by, or augmenting them with, their natural join preserves the set of
global sections bijectively.

**Proof.** Every old global section contains one compatible pair `(a_c,a_d)`,
which is an element of the join. Appending that pair produces a section of the
augmented cover. Conversely, projecting a joined hypothesis `(a,b)` to its two
parents recovers compatible parent hypotheses, and all other overlap
conditions are inherited. The two maps are inverse. Repeating the argument
proves the result for any sequence of exact joins. QED.

### 3.2 Consequence

If repeated joins span a connected cover, no correct hypothesis is pruned, and
the episode has exactly one global section, the terminal answer is determined
by local evidence plus gluing. The architecture need not invent semantic
information that was absent from every chart.

This is also the limit of the theorem. It says nothing when:

- the compiler emits the wrong local hypotheses;
- a learned restriction map assigns the wrong boundary code;
- top-`L` pruning removes the correct compatible pair;
- the supplied local evidence permits multiple future-distinguishable global
  sections; or
- all local charts encode the same wrong semantic law.

ESJW does not solve underdetermination. A result that depends on hidden answer
information or an oracle restriction map is invalid.

## 4. Exact Architecture

### 4.1 Fixed geometry

The first implementation is bounded by:

| Quantity | Frozen maximum |
|---|---:|
| Base charts | 24 |
| Total charts after joins | 32 |
| Local hypotheses per chart, `A_max` | 64 |
| Boundary values per overlap, `B_max` | 16 |
| Active overlap edges | 64 |
| Defect-reconciliation rounds per epoch | 8 |
| Join epochs | 4 |
| Hard joins per epoch | 1 |
| Total recurrent safety rounds | 32 |
| Hypothesis feature width | 256 |
| Chart feature width | 256 |
| Learned parameter ceiling | 42,000,000 |

Any episode exceeding a geometry limit is rejected before training or scoring.
No truncation is allowed.

### 4.2 Compiler output

The compiler receives source records during the compilation phase and emits
only:

```text
chart_active             [C]
hypothesis_active        [C,A]
hypothesis_features      [C,A,D]
base_hypothesis_logits   [C,A]
edge_active              [E]
edge_chart_left          [E,C]
edge_chart_right         [E,C]
left_restriction_logits  [E,B,A]
right_restriction_logits [E,B,A]
query_chart_weights      [C]
serializer_state         [D]
```

All chart, hypothesis, and boundary identities are anonymous and freshly
permuted per episode. The score path receives no target answer, global
solution, class label, schedule, trajectory, allowed-pair matrix, oracle
restriction, task-family ID, renderer ID, or assessor product.

Restriction probabilities are column-normalized:

```text
R_(e<-c)[:,a] = softmax(left_restriction_logits[e,:,a])
R_(e<-d)[:,b] = softmax(right_restriction_logits[e,:,b]).
```

The soft compatibility used for training is

```text
J_e[a,b] = sum_k R_(e<-c)[k,a] R_(e<-d)[k,b].
```

At hard evaluation, each restriction column is thresholded by one `argmax`,
and `J_e` is exact equality of the two resulting boundary codes. There is no
clustering, closure, search, repair, or threshold selection after seeing a
score.

### 4.3 Local beliefs and signed gluing residue

At recurrent round `t`, each chart carries logits `l_c^t` and a belief over
its active hypotheses:

```text
p_c^t = masked_softmax(l_c^t / tau_t).
```

Each side of an overlap induces a boundary belief:

```text
mu_(e<-c)^t = R_(e<-c) p_c^t
mu_(e<-d)^t = R_(e<-d) p_d^t.
```

The signed coboundary residue is

```text
r_e^t = mu_(e<-c)^t - mu_(e<-d)^t.
```

The dual defect state is persistent:

```text
lambda_e^(t+1) = kappa_t lambda_e^t + rho_t r_e^t,
```

where `0 <= kappa_t <= 1` and `rho_t > 0` are learned bounded scalars shared
across every episode and chart identity.

The scalar compatible mass is

```text
chi_e^t = sum_(a,b) p_c^t[a] J_e[a,b] p_d^t[b].
```

The unresolved support defect is

```text
delta_e^t = -log(epsilon + chi_e^t).
```

The signed residue detects marginal disagreement. The support defect prevents
uniform but weakly compatible beliefs from appearing solved merely because
their boundary marginals coincide.

### 4.4 Tied reconciliation update

For `e=(c,d)`, the compatibility message from `d` to hypothesis `a` of `c` is

```text
m_(d->c)^t[a] =
    log(epsilon + sum_b J_e[a,b] p_d^t[b]).
```

The primal correction associated with the signed dual is

```text
g_(e->c)^t[a] =
    sum_k R_(e<-c)[k,a] (lambda_e^(t+1)[k] + beta_t r_e^t[k]).
```

Orientation reverses the sign for the right chart. One shared, positive-gated
cell updates every chart:

```text
u_c^t =
    l_c^0
    + alpha_t sum_(e incident c) m_(neighbor->c)^t
    - gamma_t sum_(e incident c) sign(c,e) g_(e->c)^t

l_c^(t+1) =
    (1 - eta_t) l_c^t
    + eta_t u_c^t
    + F_theta(h_c, p_c^t, aggregate_m_c^t, aggregate_r_c^t).
```

`alpha_t, beta_t, gamma_t, eta_t` are bounded shared scalars.
`F_theta` is one tied chart-equivariant residual MLP. It cannot read source
tokens, the answer, the task family, or absolute chart indices.

This update is differentiable. The complete recurrent state is

```text
Z_t = (l_t, p_t, lambda_t, active charts, active edges,
       join history, halt latch).
```

No state is stored in an external database or host process.

### 4.5 Defect-triggered chart join

After eight reconciliation rounds, every eligible edge receives one join
score:

```text
s_e =
    G_theta(delta_e, ||r_e||_1, ||lambda_e||_1,
            H(p_c), H(p_d), H(p_query), chart_features).
```

Training uses a differentiable sparse distribution over edges. Hard evaluation
selects one edge with one `argmax`. The maximum must be unique. An exact tie
between eligible edges is a safety failure and makes the episode wrong; it is
not broken by storage order or an absolute chart identity. There is no retry.

For the selected edge, all `A_c A_d <= 4096` pairs are scored internally:

```text
q_e[a,b] =
    l_c[a] + l_d[b] + log(epsilon + J_e[a,b])
    + K_theta(h_c[a], h_d[b], lambda_e, r_e).
```

The treatment creates at most `L=64` composite hypotheses. Training uses a
continuous top-`L` relaxation; hard evaluation selects the top `L` once and
marks every incompatible pair invalid. If the correct pair is pruned, the
episode is wrong. If no compatible pair survives, the episode is wrong.

For a retained pair `(a,b)`, the new chart state is

```text
h_(c join d)[a,b] =
    LayerNorm(h_c[a] + h_d[b]
              + J_theta(h_c[a], h_d[b], boundary_summary_e)).
```

Its base energy is the parent energy sum plus learned compatibility energy.
Restrictions to all external neighbors are inherited from the relevant parent
hypothesis. Restrictions to both parents are exact projection codes derived
from the selected parent hypothesis indices. These projections are generic
tensor plumbing inside the architecture; they do not contain a task rule.

The parent charts remain active. The joined chart is redundant evidence whose
hard section set must equal the parent natural join. This choice permits a
direct artifact audit of the exact-join lemma and prevents a faulty join from
silently destroying its provenance.

### 4.6 Model-owned halt and reader

The halt head receives only:

```text
max_e ||r_e||_1
max_e delta_e
max_e ||lambda_e||_1
max_c ||p_c^(t+1) - p_c^t||_1
query entropy
query top-1 margin
remaining chart capacity
internal chart summaries.
```

It does not receive a target, oracle convergence flag, expected step count, or
fixed answer deadline. Its hard event is thresholded once and latched:

```text
halt_(t+1) = halt_t OR ST[halt_logit_t >= 0].
```

The 32-round maximum is a safety bound. A safety-exhausted episode is wrong.

The reader consumes only the joined query-chart belief, retained anonymous
serializer state, and a learned output projection into Shohin's vocabulary.
The host may decode token IDs but may not map an internal class to the correct
answer.

## 5. Source-Deletion Boundary

The evaluated process has two irreversible phases.

### Phase A: compile

Shohin and the ESJW compiler receive the source. Public clause separators may
define local attention windows, but no host component identifies variables,
relations, compatible hypotheses, a global graph solution, or a schedule.
Segment-window bytes and their compute are counted as architectural prior.

The compiler emits the tensor bundle in Section 4.2. The bundle is sealed and
hashed before reasoning begins.

### Phase B: reason

Before the first ESJW recurrence:

- source token IDs are destroyed;
- source embeddings and Shohin residuals are destroyed;
- attention KV state is destroyed;
- compiler scratch buffers and global pooled states are destroyed;
- file handles to source rows are closed; and
- the reasoning process is restricted to the sealed tensor bundle and its own
  recurrent state.

The generator, oracle, local relation evaluator, target global section, answer,
training labels, and assessor are absent from the reasoning process and its
allowlisted filesystem.

Required deletion interventions:

1. poisoning source memory after the seal must leave all ESJW outputs
   bit-identical;
2. replacing every deleted tensor by independent noise must leave all outputs
   bit-identical;
3. removing the sealed tensor bundle must collapse performance;
4. shuffling restriction codes inside the sealed bundle must causally alter
   joins and answers; and
5. replacing only the query chart must alter the answer without altering
   source-deleted non-query chart sections.

Any post-seal source access voids the run.

## 6. Parameter And Runtime Budget

The first implementation receives the following hard ceilings:

| Component | Maximum learned parameters |
|---|---:|
| Segment-local chart compiler and hypothesis encoder | 18,000,000 |
| Restriction, local-energy, and compatibility heads | 8,000,000 |
| Tied reconciliation and join cells | 12,000,000 |
| Halt, query reader, and serializer | 4,000,000 |
| **Total added** | **42,000,000** |
| Protected Shohin base | 125,081,664 |
| **Maximum complete system** | **167,081,664** |
| Remaining headroom below 200M | 32,918,336 |

Every realized model must publish an exact tensor-by-tensor parameter ledger.
Unused budget cannot be transferred after a development score without a new
preregistration.

At maximum geometry, the explicitly retained bf16 state must report:

- chart and hypothesis features;
- local logits and beliefs;
- both restriction tensors;
- vector dual defects;
- join candidates and parent provenance;
- halt state; and
- all temporary top-`L` and compatibility tensors.

Inference sequential depth is at most four epochs of eight reconciliation
rounds plus four join decisions. Runtime must publish measured FLOPs, peak
accelerator memory, bytes retained after deletion, and synchronization points.

No external memory, search queue, symbolic solver, repair loop, oracle call, or
unreported test-time sampling is allowed.

## 7. Minimal CPU Mechanics Gate

No GPU source freeze is authorized until an implementation and an independent
reference satisfy every gate below.

1. **Exact join preservation.** Exhaustively enumerate connected covers with
   up to five charts, up to four hypotheses per chart, and up to three boundary
   values over a frozen finite fixture family. For every admissible join order,
   projected joined sections equal brute-force global sections exactly.
2. **Unique local-ambiguous solution.** Construct fixtures where every chart
   has at least two legal hypotheses but the cover has exactly one global
   section. ESJW must recover it without an answer channel.
3. **Frustrated cycle.** A binary odd cycle with pairwise incompatible parity
   must not halt with a valid section. Pairwise marginal agreement alone is
   insufficient; support defect and joins must expose the empty global join.
4. **Noncommuting order twin.** Two covers with identical local-hypothesis and
   boundary-code multisets but reversed noncommuting composition must produce
   different terminal sections.
5. **Cover split/merge naturality.** Splitting one chart into two charts joined
   by an identity boundary, or merging them back, preserves global sections and
   terminal readout exactly.
6. **Permutation equivariance.** Chart, edge, hypothesis, boundary, variable,
   and storage reindexing commute exactly with hard outputs.
7. **Recoding invariance.** Injective recoding of anonymous source values and
   boundary names leaves semantic output unchanged after inverse mapping.
8. **Top-`L` fail-closed behavior.** A fixture with more than `L` compatible
   pairs exercises the frozen one-shot top-`L` selector. If a required pair is
   ranked `L+1`, the fixture is counted wrong and is never retried, expanded,
   or repaired.
9. **Source deletion.** Post-seal source poison and replacement are
   bit-identical; pre-seal source mutations that change a required local
   relation change the compiled bundle.
10. **No single-chart answer channel.** In answer twins, every individual
    chart tensor and every chart-wise marginal is identical while the correct
    global answer differs. Only cross-chart gluing separates the twins.
11. **Gradient reachability.** A terminal loss has finite nonzero gradients to
    every required local-energy, restriction, reconciliation, join, halt, and
    reader parameter group on a nondegenerate fixture.
12. **Resource receipt.** Parameter count, recurrent-state bytes, temporary
    bytes, join enumeration count, and measured CPU operations remain within
    the frozen bounds.

The independent reference may enumerate sections only during mechanics tests
and offline assessment. It may not be imported by training or evaluation.

## 8. Frozen GPU Experiment

The first accelerator campaign is one staged experiment. Stage B is launched
only if Stage A passes without threshold changes.

### 8.1 Board

The board is a finite local-global composition task, not a claim of open
language reasoning.

Each episode has:

- 8 to 20 anonymous variables;
- a domain of size 2 to 16;
- 10 to 24 binary or ternary local clauses;
- 8 to 24 source charts;
- at least two legal hypotheses in every base chart;
- exactly one global section;
- a query at graph distance at least five from every explicit anchor;
- no single chart or chart-wise marginal that determines the answer;
- fresh variable names, relation-card names, chart order, hypothesis order,
  boundary codes, and renderers; and
- order, binding, cover-split, cover-merge, and source-recoding twins.

Local relations are specified by opaque in-episode witnesses. Global operation
names and family labels never appear in model input.

Training contains affine finite relations, finite permutation actions, and
Boolean/Horn local relations. Development includes:

1. unseen compositions of those relations;
2. depths 9 through 16 after training depths 2 through 8;
3. treewidth three and four after training treewidth at most two;
4. noncommuting cycles;
5. unseen renderer compositions;
6. complete held-out typed-stack/dataflow relation families; and
7. cover split/merge presentations of the same semantic problem.

The held-out families must reuse the same anonymous tensor contract but may not
share exact local relation tables, normalized clause windows, graph motifs,
global sections, answer twins, or renderer templates with training.

Frozen sizes:

| Partition | Episodes |
|---|---:|
| Local-clause calibration train | 64,000 |
| Autonomous composition train | 96,000 |
| In-range development | 8,192 |
| Depth/treewidth development | 8,192 |
| Held-out-family development | 8,192 |
| Sealed confirmation | 16,384 |

Exact prompts, normalized word 13-grams, semantic graph hashes, local relation
hashes, global-section hashes, and answer-twin hashes must be split-disjoint.
Source code and all thresholds are committed before data seeds. Confirmation is
generated and sealed only after source freeze and is accessed once.

### 8.2 Stage A: source-deleted packet qualification

Stage A bypasses natural-language ambiguity by supplying anonymous local chart
records, opaque local witnesses, physical overlap incidence, and a query port.
It does not supply restriction codes, compatibility matrices, a global
section, answer, schedule, or trajectory.

The question is whether learned fibers, learned restrictions, dual defects, and
joins solve the composition problem. Passing Stage A does not authorize a
language claim.

### 8.3 Stage B: Shohin compiler integration

Stage B replaces the chart packet encoder with frozen Shohin plus the
segment-local ESJW compiler over rendered source text. The base checkpoint is
not modified. Only the preregistered ESJW parameters train.

Public clause separators may define local windows. Variable identity,
relation semantics, overlap restrictions, hypothesis energies, join order,
halt, and answer remain model-owned.

The Phase A/Phase B deletion boundary in Section 5 is mandatory. No score from
a source-retained run can substitute for the source-deleted score.

### 8.4 Optimization

Each learned arm receives the same examples, optimizer updates, maximum
recurrent rounds, join opportunities, and token budget.

Frozen maximum:

```text
local calibration updates:       20,000
source-deleted composition:      80,000
joint low-rate polish:           20,000
total optimizer updates:        120,000
maximum joins per episode:            4
maximum recurrent rounds:            32
independent seeds:                     5
```

Stage A may supervise local hypothesis legality and boundary restriction codes
on training rows. Autonomous composition training additionally uses terminal
section, answer, halt, and invariance losses. No intermediate join order,
oracle global section prefix, privileged trajectory, or host-selected cluster
may supervise the treatment.

Stage B uses the same local and terminal objectives, but all local evidence is
compiled from source text. Confirmation never participates in optimization,
thresholding, model selection, or early stopping.

## 9. Mandatory Matched Controls

Every control receives the same train rows, active tensors, maximum parameter
count, maximum inference FLOPs, recurrent state bytes, and optimizer
opportunities unless explicitly identified as a ceiling.

1. **No join:** full primal-dual reconciliation, but composite charts are
   disabled.
2. **No dual:** compatibility messages and joins remain, but
   `lambda_e=0` at every round.
3. **Ordinary loopy BP:** same local energies, restrictions, rounds, and
   hidden width; standard tied compatibility messages replace persistent
   defects and joins.
4. **Generic recurrent GNN:** parameter/FLOP-matched chart-edge recurrence
   over pooled vectors, with no explicit hypothesis fibers.
5. **Global chart transformer:** favorable same-parameter global attention over
   every chart and hypothesis, with the same source-deletion boundary and
   reader.
6. **Random join:** same number and cost of joins, with a seeded random eligible
   edge instead of the defect selector.
7. **Graph-only min-fill join:** a favorable fixed junction schedule selected
   from public incidence alone, with identical join cells and top-`L` budget.
8. **Shuffled restrictions:** preserve chart logits, degree, boundary
   cardinality, and compute while permuting restriction columns between
   episodes.
9. **Fused-vector control:** a joined chart receives one pooled parent vector
   rather than explicit compatible parent pairs.
10. **Source-retained ceiling:** the reasoner retains source residuals at every
    round. It is reported but cannot satisfy native/source-deleted gates.
11. **Oracle restriction ceiling:** exact offline boundary codes replace learned
    codes. It localizes a compiler failure but cannot satisfy the treatment
    claim.
12. **Direct Shohin baseline:** identical reader and training examples without
    the ESJW sidecar.

If an arm cannot use the full parameter ceiling naturally, it receives
trainable inert parameter banks whose optimizer and memory costs are counted
but whose values cannot enter its forward pass. Compute matching is measured,
not inferred from parameter count.

## 10. Causal Interventions

The following predictions are frozen:

| Intervention | Predicted effect |
|---|---|
| Consistent chart/hypothesis/boundary reindexing | Semantic answer unchanged |
| Swap two boundary codes, preserve chart energies | Affected joins and answers change |
| Shuffle local energies, preserve restrictions | Local sections and answer collapse |
| Zero dual state after each round | Loopy/deep cells degrade more than trees |
| Disable joins | High-treewidth and noncommuting cells degrade most |
| Randomize join selector, preserve join count | Same compute, lower exact sections |
| Replace a correct composite chart with an incompatible pair | Query belief and answer degrade |
| Split or merge the cover semantically | Answer unchanged |
| Poison source after sealing | Bit-identical output |
| Reset beliefs every round | Depth transfer collapses |
| Transplant a correct joined chart between matched twins | Only downstream dependent query changes |
| Derange query chart after reasoning | Terminal non-query sections unchanged; answer changes |

Interventions occur once. Invalid results remain wrong; no search or repair is
allowed.

## 11. Metrics And Advancement Gates

All rates count malformed, invalid, pruned, non-halted, and safety-exhausted
episodes as wrong.

### 11.1 Mechanics and local competence

- 100% CPU mechanics gates;
- at least 99.5% local hypothesis-set accuracy;
- at least 99.5% hard boundary-restriction accuracy;
- at least 99.9% correct-pair survival through every hard top-`L` join;
- 100% chart/hypothesis/boundary permutation equivariance; and
- 100% source-poison invariance after sealing.

### 11.2 Stage A composition

Across all five seeds:

- at least 95% exact global sections and answers in-range;
- at least 90% exact in every depth, treewidth, noncommuting, renderer, and
  cover-refinement cell;
- at least 85% exact in each fully held-out task family;
- at least 99% learned halt;
- at most 1% safety exhaustion;
- at least 99% paired split/merge answer consistency;
- treatment at least 15 percentage points above no-join, random-join,
  fused-vector, loopy-BP, and generic-GNN controls on the combined hard cells;
- treatment at least 10 points above the global chart transformer on
  depth/treewidth extrapolation, or a paired 95% lower confidence bound above
  zero if the global control already exceeds 90%; and
- a paired 95% lower confidence bound above a 10-point treatment advantage over
  every nontrivial learned control except the explicitly favorable global
  transformer.

The min-fill control determines the selector claim. If it matches ESJW, exact
joins may survive as a mechanism, but defect-triggered adaptive refinement does
not.

### 11.3 Stage B language integration

Across all five seeds:

- at least 90% exact in-range source-deleted answers;
- at least 85% exact on unseen renderers and depths;
- at least 75% exact in each held-out family;
- at least 99% learned halt with at most 1% safety exhaustion;
- at least 99% source recoding and cover split/merge consistency;
- treatment at least 15 points above direct Shohin, no-join, random-join,
  shuffled-restriction, and generic-GNN controls; and
- oracle restrictions improve treatment by no more than 10 points.

A larger oracle-restriction gap localizes the failure to compilation and blocks
a composition claim.

### 11.4 Confirmation

One frozen architecture and threshold set is selected from development before
confirmation access. All five seeds run unchanged. Confirmation must meet every
Stage B absolute gate and the aggregate attribution gates. No failed seed may
be dropped.

## 12. Hard Kill Criteria

The mechanism is rejected without scale-up if any condition below occurs.

1. Exact CPU join preservation, frustrated-cycle rejection, or cover
   naturality fails.
2. Stage A train exactness is below 95% after the full budget.
3. Stage A canonical development is below 80%.
4. Any held-out family is below 60%.
5. Treatment-control separation is below 10 points on the combined hard cells.
6. Ordinary loopy BP or the generic GNN matches treatment within a paired
   95% interval while using the same resource vector.
7. Fused-vector control matches explicit hypothesis joins. This would show that
   retained correlations were not causal.
8. Random join matches the defect selector. This would reject defect
   localization.
9. Graph-only min-fill equals or beats treatment on every hard cell. This would
   reject adaptive refinement, though not necessarily the fixed join
   architecture.
10. Oracle restriction codes fail to exceed 90% Stage A exactness. The local
    hypothesis/fusion substrate is then inadequate.
11. Oracle restriction codes exceed learned codes by more than 25 points after
    Stage B. The dominant bottleneck remains compilation, so ESJW is not the
    promoted answer.
12. A single chart, pooled compiler residual, serializer state, or padding field
    predicts the answer above the frozen leakage ceiling.
13. Post-seal source poison changes any output bit.
14. Any target, answer, family, renderer, schedule, trajectory, global section,
    oracle compatibility, or assessor value enters model input or recurrent
    state.
15. Any hard join retries, backtracks, asks an assessor, or repairs a pruned
    pair.
16. Any realized complete system reaches 200,000,000 parameters.
17. Performance depends on development-selected recurrence depth, top-`L`,
    temperature, halt threshold, or join count not frozen here.
18. Five-seed confirmation fails any absolute or causal gate.

Blindly increasing width, data, update count, top-`L`, or join depth after a
kill is forbidden. A repair requires a new hypothesis and preregistration.

## 13. Collapse, Prior-Art, And Novelty Boundary

The terms "sheaf," "defect," and "join" are not evidence.

- Finite sheaf gluing, natural joins, constraint satisfaction, junction trees,
  belief propagation, dual decomposition, message-passing neural networks,
  adaptive graph coarsening, and differentiable top-`k` selection all have
  substantial prior art.
- On a tree with exact fixed restrictions and no joins, ESJW may reduce to
  ordinary message passing.
- With an oracle cover and exhaustive unpruned joins, ESJW may reduce to a
  conventional junction-tree or variable-elimination computation.
- Persistent disagreement is not biological proof of cortical predictive
  coding. Adaptive refinement is not evidence that the brain performs mesh
  refinement or renormalization.
- Exact joins do not manufacture missing semantic information. Common-mode
  wrong local laws and genuinely ambiguous global sections remain fatal.

The project-novel, falsifiable combination is narrower:

> Shohin compiles source into anonymous local hypothesis fibers and learned
> restriction codes; source is irreversibly deleted; a tied internal
> primal-dual state measures gluing defects; unresolved defects trigger
> correlation-preserving natural joins; and a model-owned halt/reader consumes
> the resulting section.

This is genuinely distinct from the current Shohin mechanisms:

- ECCR commits one quotient relation and loses minimal noncongruence through
  false collisions; ESJW preserves competing local hypotheses until they are
  globally joined.
- N-TCRR independently decodes a whole graph transaction; ESJW creates one
  compatible composite hypothesis before terminal commitment.
- AHRF diffuses and latches facts in a fixed relation field; ESJW changes its
  effective chart scope when unresolved correlations demand it.
- COFC jointly parses source occurrences; ESJW operates after source deletion
  and tests semantic global-section formation.

If the favorable controls match it, the mechanism is rejected as renamed
message passing or junction-tree inference. No publication-level novelty claim
is authorized by this preregistration.

## 14. Claim Ladder

| Evidence | Maximum authorized claim |
|---|---|
| CPU mechanics pass | ESJW tensor mechanics are coherent |
| Stage A passes | Bounded source-deleted local hypotheses can compose by learned joins |
| Stage B development passes | Shohin can compile rendered clauses into an ESJW substrate |
| Five-seed confirmation passes | Confirmed bounded architecture-native compositional reasoning |
| Transfer to new open task families and ordinary language | Requires a new preregistration |
| Genuine general reasoning | Not authorized by this experiment alone |

## 15. Custody And Publication

Before any score-bearing run:

1. implementation and tests are committed;
2. CPU mechanics report and source hashes are published;
3. source, board generator, split seeds, renderer sets, thresholds, and control
   identities are frozen;
4. confirmation generation is isolated from training and development;
5. the protected checkpoint hash is reverified;
6. every output path is isolated from flagship and other reasoning runs; and
7. the training/evaluation process allowlist is recorded.

Every report must include:

- exact source and checkpoint hashes before and after;
- all model and optimizer parameters;
- all data counts and split-isolation receipts;
- complete per-cell and per-seed metrics;
- every hard join and prune count;
- learned-halt and safety-exhaustion traces;
- source-deletion receipts;
- measured parameter, memory, FLOP, and sequential-depth ledgers;
- raw terminal transcripts or section traces for a frozen sample; and
- all control and intervention results.

The protected flagship checkpoint must remain byte-identical.

## 16. Preregistered Decision

`AUTHORIZE_CPU_MECHANICS_ONLY`.

ESJW earns a GPU source freeze only if the exact finite gluing object, support
defect, top-`L` failure behavior, source-deletion boundary, and resource ledger
pass independently. It earns a bounded composition claim only if explicit
hypothesis joins beat matched message passing, vector fusion, global attention,
and fixed/random join schedules under the frozen multi-family gates.

The reason to test ESJW is not that nature uses sheaves. It is that Shohin's
measured failures repeatedly show locally plausible pieces that are never
forced to be one computation. ESJW turns that diagnosis into a hard
architectural invariant with an equally hard rejection path.
