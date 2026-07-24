# Evidence-Aligned Frontier Plan: Causal Object-File Compilation

**Status:** revised after marginal-route v1.1, the rejected
occurrence-addressed canary, its completed read-only scale audit, and external
review of the Causal Object-File Compiler (COFC) proposal. This is a research
plan, not authorization for a neural run. The existing factorized witness-route
preregistration remains the immediate, cheapest train-only falsifier.

Exact factorized-canary source commit
`4643d1a51defe53397f9bed481051621d85c0b11` is frozen and pushed. No
post-commit seed, H100 job, or new probe read exists yet.

## 1. Objective

Shohin does not currently need a stronger recurrent executor. S5 and S7 show
that learned laws can execute, and the ER-TT motor executes every valid emitted
relation packet exactly after source deletion. The unresolved problem is the
compiler:

> Map renderer-varying language into exact physical occurrence and relation
> packets when names repeat, cardinality varies, and relations may be
> many-to-one.

The key distinction is between:

- **occurrence identity:** this physical mention at this source location;
- **nominal identity:** the entity or opcode denoted by the mention; and
- **causal state:** the anonymous state transformed by the compiled relation
  program.

The current evidence says these objects must interact, but must not be fused
into one unconstrained vector.

## 2. Closed Evidence

### 2.1 Marginal-route v1.1 localizes the residual

The closed train-only canary reaches:

| Metric | Result |
|---|---:|
| Packet / joint / relation rows | 7,275/8,000 = **90.9375%** |
| Recurrent state | 7,765/8,000 = **97.0625%** |
| Answer | 7,883/8,000 = **98.5375%** |
| Complete witness pointers | 7,194/8,000 = **89.925%** |
| Individual witness occurrences | 214,722/215,528 = **99.626%** |
| Alpha-invariant hard outputs | 8,000/8,000 |
| Oracle-route identity transport | 8,000/8,000 |

All 806 witness-failed rows contain exactly one wrong occurrence, usually an
adjacent duplicate in a late after-witness role. This is unusually clean
evidence for a remaining physical-occurrence routing error rather than a
relation-execution or nominal-equality failure.

### 2.2 Vector-level address/content fusion is rejected

The addressed canary inserted learned candidate-count and occurrence-ordinal
embeddings into the query/key memory. It regressed to:

| Metric | Result |
|---|---:|
| Packet / joint / relation rows | **60.9125%** |
| Recurrent state | **80.1375%** |
| Answer | **90.2250%** |
| Complete witness pointers | **59.5000%** |
| Minimum-cardinality joint | **43.606%** |
| Alpha invariance / oracle transport | **100% / 100%** |

The learned ordinal norm reached 8.64, count norm 2.02, and adjacent ordinal
rows 6/7 reached cosine 0.878. Errors expanded from one occurrence to as many
as six per row and worsened with cardinality. The architecture is closed.

### 2.3 The read-only scale audit is complete

Job `694932` inspected the immutable rejected checkpoint without optimization
or scored-split access. Its report SHA-256 is
`d958cc0507fe85a489a3b85368f52ed67cfda6caf9fc5efc8d686216f28f6934`.

| Audit arm | Witness rows | Packet/joint | State | Answer |
|---|---:|---:|---:|---:|
| ordinal 0, count 1 | 0.0000% | 0.2000% | 21.9250% | 40.2750% |
| ordinal 1, count 0 | 0.4875% | 3.2750% | 33.7875% | 58.5875% |
| ordinal 0.75, count 1 | 34.2250% | 40.7125% | 71.0875% | 85.7250% |
| ordinal 1, count 1 | 59.5000% | 60.9125% | 80.1375% | 90.2250% |
| ordinal 1.5, count 1 | 70.4250% | 69.3875% | 80.5625% | 87.4250% |

Increasing ordinal scale improves witness routing but still remains far below
the marginal parent and begins to damage events at scale 1.5. Removing either
address component collapses the route. The failure is therefore not explained
by a simple excessive-ordinal-norm story. Count and ordinal carry useful joint
information, but injecting them into shared vectors corrupts other geometry.

The audit is diagnostic only. None of its scales is eligible for selection or
promotion.

## 3. Immediate Experiment: Factorized Witness Address Residual

A local preregistration now defines the smallest distinct test. It leaves the
successful v1.1 structural logits numerically unchanged and adds a
zero-initialized witness-only table:

```text
bias[candidate_count, semantic_witness_role, candidate_ordinal]
```

The table has `14 x 12 x 14 = 2,352` learned scalars plus twelve bounded role
gates, for 2,364 new parameters. Table values are centered over valid
candidates and bounded by `tanh`; zero-initialized gates make the complete route
exactly equal to v1.1 at initialization while preserving first-step gate
gradients. It cannot read symbol bytes, relation targets, state, answer,
executor output, development, or confirmation. Expected complete system size
is 185,534,660 parameters, leaving 14,465,340 below 200M.

This is the right immediate falsifier because it tests whether structural
address is sufficient when isolated at the decision logit. It does **not** by
itself establish object files, joint inference, renderer-invariant grounding,
or cardinality extrapolation.

If it passes, the narrow conclusion is:

> A source-local count/role/ordinal coordinate repairs the current bounded
> witness grammar without disturbing the nominal equality bus.

The updated score path includes same-seed, same-parameter baseline,
structural-only, and shuffled-address arms. Attribution requires treatment
witness rows to beat baseline and shuffled address by at least 0.5 percentage
points. A high structural-only score restricts the claim to finite syntax
routing.

## 4. Frontier Architecture: Causal Object-File Compiler

The strongest new architectural hypothesis is to make physical occurrences
first-class objects rather than features attached to identity vectors.

### 4.1 Occurrence ledger: which mention?

For each model-detected opaque candidate, allocate an anonymous token `o_i`
and a source-local address:

```text
a_i = (record, segment, left_rank, right_rank, candidate_count,
       boundary_signature)
```

No raw symbol bytes enter `a_i`. Two mentions with identical bytes retain
different tokens and addresses. Candidate detection must remain model-owned or
a frozen architectural transform over public source bytes; gold spans are not
available at inference.

The object-file interpretation is well motivated but is not itself evidence of
success. Cognitive object files are temporary episodic representations linking
successive states of an object, and database provenance tags input tuples so
their origins survive later operations. Those are useful design analogies, not
proofs about Shohin ([object files](https://pubmed.ncbi.nlm.nih.gov/1582172/),
[provenance semirings](https://web.cs.ucdavis.edu/~green/papers/pods07.pdf)).

### 4.2 Nominal ledger: what identity?

Separately compute the existing exact whole-symbol fingerprint `f_i` and its
nominal class `e_i`:

```text
o_i != o_j may coexist with e_i == e_j
```

This is essential for ER-TT. A non-bijective after-witness may contain several
distinct physical occurrences of the same nominal entity. Equality must create
the relation semantics without erasing provenance.

The June 2026 Dual-State Slot Attention preprint reports an analogous objective
conflict when appearance and persistent identity share one slot, but it is
recent convergent evidence from video, not validation of COFC
([arXiv:2606.12601](https://arxiv.org/abs/2606.12601)).

### 4.3 Causal ledger: what state is transformed?

The compiled initial state and relation tensors continue to use anonymous
entity indices. The zero-parameter ER-TT motor remains unchanged:

```text
S_next = R_t @ S_t
```

After the packet is sealed, source-facing occurrence representations are
deleted. Only the compiled packet and explicitly counted retained state may
reach execution.

## 5. Joint Decoding, With A Grammar Firewall

Independent softmax pointers can choose mutually inconsistent local maxima.
For an ordered witness segment with candidates `c_1...c_M` and `N` roles, COFC
instead scores a complete monotone path:

```text
pi = (j_1, ..., j_N),  j_1 < ... < j_N

Score(pi) = sum_k unary(k, j_k)
          + sum_k transition(j_(k-1), j_k)
          + start(j_1) + end(j_N)
```

Forward-backward can train over legal paths; Viterbi can emit one hard path.
This imports the computational idea of globally scoring an alignment rather
than committing to unrelated local matches
([Needleman-Wunsch](https://pubmed.ncbi.nlm.nih.gov/5420325/)).

The constraint applies **separately inside the ordered before and after
segments**. It does not impose a one-to-one relation between nominal entities,
rules, or records. Distinct after-occurrence tokens may share one nominal class,
preserving arbitrary many-to-one relations and legal self-maps.

### 5.1 Important current-board limitation

The present ER-TT renderer writes every active rule as:

```text
opcode  before_1 ... before_N  separator  after_1 ... after_N
```

There are no opaque distractors inside either witness segment. Once the segment
boundary and cardinality are known, a monotone path nearly reduces to selecting
all `N` candidates in order. A COFC pass on this board could therefore be a
bounded grammar parser, not evidence that joint multi-hypothesis inference
solves a general occurrence-binding problem.

Before a neural COFC run, a CPU legality audit must prove both:

1. the decoder preserves every valid non-bijective ER-TT packet; and
2. an expanded source-local grammar with distractors contains cases where
   independent top-one pointers fail but a uniquely best complete path wins.

### 5.2 Allowed and forbidden factors

A small factor graph may connect declaration, witness, opcode, event,
cardinality, activity, and HALT decisions. Factor graphs are appropriate when a
global score decomposes into local functions
([Kschischang, Frey, and Loeliger](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf)).

Allowed factors must be frozen source-local syntax or type constraints, such as:

- within-segment monotonicity and candidate non-reuse;
- before/after cardinality agreement;
- equality between a model-selected event opcode and compiled rule opcode; and
- agreement between declaration and initial nominal classes.

Forbidden factors include:

- executor state, trajectory, answer, or outcome;
- a gold graph, relation tensor, or target span;
- retry after execution or answer inspection;
- global relation bijectivity; and
- a host validator that repairs an invalid packet.

Use a fixed number of inference rounds. An invalid or noncommitted row remains
wrong. Internal proofreading is only an analogy to repeated discriminative
stages; it must not become an uncounted external repair loop
([Hopfield 1974](https://www.pnas.org/doi/10.1073/pnas.71.10.4135)).

## 6. Tied Reader And Intrinsic Addressing

For a later cardinality-extrapolation board, replace position-specific query
tables with one tied transition:

```text
q_(k+1) = G(q_k, a_(j_k), record_state)
```

The cell emits the next role or STOP, so learned parameter count does not grow
with `N`. A compositional two-sided position code may replace finite ordinal
lookup tables, but it must be tested as a separate intervention. Multi-period
grid-cell coding motivates compositional position representations; it does not
establish renderer invariance for text
([Fiete, Burak, and Brookings](https://www.jneurosci.org/content/28/27/6858)).

Do not bundle tied recurrence, intrinsic codes, factor messages, object
directories, and new renderer supervision into one first canary.

## 7. Late Query: Current Board Versus Future Board

The external proposal assumes a late query containing a nominal referent that
can be matched against an object directory. Current ER-TT does not have that
interface. Its late query is a position numeral (`Q1...Q6` or `ASK 1...6`), and
the marginal and addressed canaries already route it exactly on 8,000/8,000
rows.

Therefore:

- do not modify the current ER-TT query path;
- do not credit an object directory for solving the current witness residual;
- on a future referential-query board, retain a counted read-only directory of
  `(object_id, nominal_signature)` pairs and test query-by-object-file; and
- include directory shuffle, late-query swap, source deletion, and post-seal
  poison interventions.

The directory is explicit retained state. Its bytes, compute, and any learned
matcher parameters must be counted.

## 8. Experimental Sequence

### Stage 0: finish the factorized residual canary

Use the existing `R12_ER_FACTORIZED_WITNESS_ROUTE_PREREG.md` contract only
after source freeze and seed custody. A pass authorizes a fresh test of that
mechanism, not COFC. A failure closes count/role/ordinal residual lookup on the
current route.

### Stage 1: CPU COFC falsifier

Before training, exhaustively test:

- cardinalities `N=2...16`;
- duplicate nominal runs of length one through five;
- zero through eight source-local distractors;
- left, right, and bidirectional witness renderings;
- inserted punctuation and variable token widths;
- physical-record permutations and source alpha renaming;
- arbitrary total/non-bijective relations and self-maps; and
- cases where gold is local rank two or three but the complete path is uniquely
  optimal.

Also quantify how often the current ER-TT grammar admits more than one legal
path. If it does not, the current board cannot identify a joint-inference gain.

### Stage 2: matched train-only mechanism canary

Only if Stage 1 establishes a nontrivial joint problem, freeze same-run arms:

1. retained marginal route;
2. 2,364-parameter factorized address treatment;
3. COFC joint alignment with separate occurrence/nominal ledgers;
4. COFC with independent decoding but identical unary logits;
5. COFC with occurrence and nominal state fused;
6. COFC without cross-record factors; and
7. structural-only and shuffled-address controls.

Use identical rows, renderer views, updates, optimizer opportunity, source-local
targets, and decoder budget. No graph, state, answer, or outcome supervision is
allowed.

### Stage 3: fresh in-range board

First test `N=3...6` on fresh names, renderer compositions, non-bijective rules,
self-maps, distractors, and anonymous object-ID permutations. This isolates
renderer and occurrence transfer without changing cardinality range.

### Stage 4: separate cardinality-extrapolation board

Only after Stage 3 passes should a newly preregistered board train on `N=3...6`
and score `N=7...9`. This requires new shape limits, source grammar, evaluator,
resource accounting, and gates. Bundling unseen cardinality with the first COFC
test would make a failure uninterpretable.

## 9. Required Interventions

| Intervention | Must change | Must remain unchanged |
|---|---|---|
| Swap two occurrence addresses, preserve bytes | role pointers and affected relation | nominal equality |
| Swap two nominal signatures, preserve addresses | equality-derived relation/event binding | structural paths |
| Remove joint decoding, retain unary logits | duplicate/distractor stress | unambiguous rows |
| Alpha-rename consistently | nothing semantic | occurrence-address structure |
| Reindex physical records | storage coordinates only | relation/state/answer |
| Poison source after packet sealing | nothing | all emitted and executed outputs |
| Derange relation tensors | state and answer | occurrence evidence |
| Reset recurrent state each step | state and answer collapse | compiled packet |
| Shuffle future object directory | future referential answer | terminal recurrent state |

## 10. Advancement Gates

Exact thresholds belong in a preregistration, not this review. A COFC fresh
board should nevertheless require:

- complete witness and relation rows materially above the matched independent
  decoder, not merely above historical scores;
- packet, state, answer, and joint gates with per-cardinality and per-renderer
  minima;
- exactly 100% alpha invariance, object-ID permutation equivariance, and
  record-storage reindex invariance;
- exactly 100% source deletion and post-seal poison invariance;
- every valid emitted packet executing exactly;
- structural-only, shuffled-address, independent-decoder, and fused-ledger
  controls below treatment;
- invalid/noncommitted packets counted as wrong;
- immutable checkpoint-before-access custody; and
- complete deployed system strictly below 200M parameters.

The external proposal's suggested 97% complete-pointer and 95% joint floors are
reasonable design targets, but they are not frozen gates and may not be selected
after viewing a result.

## 11. Resource Boundary

The retained marginal parent has 185,532,296 complete parameters. The current
factorized treatment is expected at 185,534,660. The external COFC estimate of
2--3M additional parameters, or roughly 188M total, is plausible but unaudited.

Any implementation must count:

- exact learned parameters and optimizer state;
- dynamic-programming and factor-message FLOPs;
- temporary path/factor memory;
- number of sequential inference rounds;
- paired-renderer target bits, if used; and
- retained object-directory bytes.

No architecture may rely on an uncounted host parser, solver, retry loop, or
external memory.

## 12. Decision

The proposal's strongest contribution is ontological, not biological:

> Preserve separate physical-occurrence and nominal-identity objects, then
> make compiler decisions over coherent complete parses rather than unrelated
> pointers.

That is the best long-range direction currently available. It directly matches
the observed adjacent-duplicate failures and explains why vector-level address
fusion was destructive.

The immediate sequence remains disciplined:

1. run the already prepared 2,364-parameter factorized canary under its frozen
   train-only contract;
2. build the CPU legality/nontriviality falsifier for joint alignment;
3. admit COFC only as a matched distinct mechanism, with segment-local
   monotonicity and no global bijection;
4. keep the current numeral query path unchanged; and
5. defer `N=7...9` and referential object-directory queries to separately
   identifiable boards.

Success would establish a bounded causal compiler factor. It would not yet
establish open-domain grounding, alias resolution, natural-language object
permanence, unbounded planning, or general reasoning.
