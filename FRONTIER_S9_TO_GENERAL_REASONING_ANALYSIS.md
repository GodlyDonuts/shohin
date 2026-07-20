# Frontier S9-to-General-Reasoning Evidence Review

**Decision:** adopt the central compiler-machine thesis and the orbit/structured
assignment repair; reject unsupported S9 failure diagnoses and defer the larger
architecture bundle until S9.1 confirmation

**Reviewed sources:**

- `FRONTIER_NOMINAL_GRAPH_REWRITE_MACHINE_PLAN.md`, exact attachment SHA-256
  `e39f9787d620484a428e8cb4e7717a8537f94782133eeba9246082fdddd60e45`
- `FRONTIER_S9_ARCHITECTURE_PROPOSAL.md`, exact attachment SHA-256
  `d4d2d622ce9a2ab10126bc98d8f7be1f40efe9a398fcb735454fe88c924b6378`

The source files are preserved verbatim. Literature citations inside them are
leads, not locally reproduced evidence and are not required for the decision
below.

## Executive verdict

The strongest shared thesis is consistent with Shohin's evidence:

> Shohin should become a renaming-invariant compiler for a small model-owned
> reasoning computer, rather than a language model trained to emit longer
> reasoning-shaped prose.

S7 and S9 make this more than an analogy. S7 confirms exact learned cyclic-law
compilation and recurrent reuse. S9 development demonstrates a causal
occurrence-class binding effect: 94.775% exact graphs versus 46.387% for the
equal-budget no-class arm and 0% for shuffled relations. The valid S9 graph
then owns event order, nil halt, recurrent state updates, and query
consumption. The remaining S9.1 problem is robustness at the compiler boundary,
not a need for a larger language decoder or a new arithmetic runtime.

The proposals become speculative after that point. First-class rule cards,
learned primitive programs, an obligation graph, source-deleted realization,
and learned alias partitions are coherent future stages, but none has passed a
Shohin theorem or finite falsifier. They must not be bundled into S9.1.

## Corrections required before implementation

### 1. The S9 rename failure is not candidate-width failure

The frontier plan attributes the 18 operation-renaming failures to token-width
bounded candidates disappearing under BPE recoding. The frozen board evidence
does not support that diagnosis. Across all 2,048 development sources:

| Span width | Original count | Recoded count |
|---:|---:|---:|
| 1 | 4,562 | 4,564 |
| 2 | 101,355 | 101,341 |
| 3 | 8,877 | 8,889 |
| >4 | 0 | 0 |

The maximum original and recoded gold width is three under the frozen
width-four candidate cap. The evaluator also compiled every recoded row before
scoring. The 18 failures therefore arise from learned span/role selection or
relation assembly after tokenization changes, not from absence of a legal
candidate.

Byte-aligned spans remain a plausible future representation, especially for
open-vocabulary aliases, but they are not the measured S9.1 repair. Enumerating
every byte span up to 24 or 32 bytes would also enlarge candidate count by
orders of magnitude unless preceded by a learned boundary lattice. It needs a
separate resource and representability theorem.

### 2. The current class aggregator is already permutation invariant

S9 uses a mean over all candidate-span representations sharing the same exact
surface. Mean and sum are both permutation invariant. Replacing mean with sum
does not create alpha equivariance; it changes multiplicity scaling. A count
feature may be tested only if occurrence multiplicity is independently
randomized so it cannot become a renderer shortcut.

### 3. Anonymous class IDs already exist downstream

After S9 selects source spans, exact surface equality creates anonymous
per-example class IDs. Lexical residuals still help the compiler locate and
type mentions, but the S8/S7 runtime receives the quotient graph rather than
the operation string. An additional anonymous-atom stream is an ablation, not
a prerequisite. Removing lexical content too early could erase the context
needed to classify a mention.

### 4. A structured decoder must not solve the task by grammar

Model-logit-only constrained assignment is the highest-confidence repair for
S9's all-or-nothing graph invalidity. It is admissible only if the report shows
that syntax leaves many valid assignments and that uniform, source-free, and
shuffled logits remain near chance. The decoder may enforce arity, type,
non-overlap, reachability, and one nil-terminated chain. It may not consult the
executor, final state, answer, gold depth, semantic solver, or retries.

### 5. Turing completeness is not evidence of learned reasoning

A stack, branch, heap, or host ALU can make an interpreter universal without
making Shohin a general reasoner. Semantic primitive transitions must be
learned on determining atomic boards, transferred to held-out compositions,
and causally consumed after source deletion. Host arithmetic remains an upper
bound, not a promoted treatment.

## Admitted S9.1 contract

S9.1 should change only two scientific interfaces on a fresh board.

### A. Renaming-orbit equivariance

Every training episode receives independently sampled source-level operation
renamings. The compiler is trained on original and recoded sources with:

1. ordinary span/relation supervision in both views;
2. aligned participation and role logits after the known source-coordinate
   map;
3. class-canonical graph consistency after anonymous class reindexing.

The same number of optimizer updates and source families must be used for the
no-class control. Orbit examples replace part of the fixed training budget;
they do not silently increase it.

### B. Globally structured model-logit assignment

The model emits unary span-role scores and, if needed, bounded pairwise link
scores. A deterministic decoder chooses the maximum-score graph under only the
frozen S8 grammar and typing constraints. Required controls are:

- unconstrained S9 decoding;
- structured treatment;
- structured no-class message;
- structured shuffled labels;
- structured uniform logits;
- structured source-free logits;
- oracle logits;
- per-source count or lower bound for grammar-valid assignments.

The output still enters the unchanged S8 graph validator and S7 executor.

### Frozen success requirements

- retain every S9 absolute, causal, attribution, parameter, and access gate;
- at least 95% all-row exact class membership;
- at least 90% valid graphs and 85% exact graphs;
- at least 80% state and 85% answers;
- at least +5 points exact graph over the equal-budget no-class arm;
- zero or preregistered near-chance exact graphs for shuffled, uniform, and
  source-free controls;
- 100% recoding eligibility for every originally valid graph;
- bit-identical canonical graph, state, and answer under renaming;
- sealed confirmation opened once only after every development gate passes.

No S9 threshold may be relaxed and the closed S9 board may not be rescored.

## Ordered route after S9.1

1. **S10 nominal quotient:** predict alias/coreference partitions with exact
   equality as a prior, explicit abstention, transitivity, and similar-but-
   distinct negatives. Pairwise threshold plus transitive closure is not enough
   because one false bridge can merge two classes; use a partition-level score
   or correlation-clustering objective with bounded exact decoding.
2. **S11 first-class rule cards:** select a typed hypothesis family, emit its
   determining fingerprint and discrete microprogram, and preserve multiple
   candidates when demonstrations are underdetermined.
3. **S12 agenda graph:** model-owned obligation creation, rule binding,
   transactional updates, branch choice, retirement, and halt. The host may
   reject malformed edits but may not schedule or repair semantics.
4. **S13 source-deleted integration:** compiler source is deleted; only the
   anonymous graph, rule cards, agenda, and recurrent state remain available.
5. **S14 causal realization:** the language decoder sees terminal state and
   query/output bindings, not the original semantic source. Donor-state swaps,
   graph zeroing, and node-order permutations must control its answer.

The strongest new theoretical direction is therefore not a larger scratchpad.
It is a sequence of explicit quotients and machines:

```text
surface occurrences
  -> nominal identity partition
  -> typed relation/program graph
  -> learned primitive state machine
  -> model-owned agenda
  -> terminal state-conditioned language
```

Only the first occurrence quotient has strong development evidence today.
S9.1 must confirm that foundation before the project spends capacity on the
later machine.
