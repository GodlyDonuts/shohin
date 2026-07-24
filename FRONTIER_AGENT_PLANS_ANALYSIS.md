# Causal Object-File Compiler: Evidence Review

**Status:** external COFC proposal reviewed against the live ER-TT grammar, all
closed canaries, the completed scale audit, and the locally prepared factorized
witness-route preregistration. No new neural run is authorized by this review.

**Reviewed source:** `FRONTIER_AGENT_PLANS.md`

**Workspace SHA-256:**
`8903788809f987372ba23d1cff77e8e86641a174e9cc8b02828c740ab2a63271`

The reviewed plan contains 433 lines, 2,550 words, and 18,451 bytes. It is an
evidence-aligned rewrite of the external proposal, not a verbatim copy.

## 1. Executive Verdict

COFC is the strongest long-range architecture proposed so far because it
separates two variables that the failures show must not be collapsed:

- a unique token for each physical occurrence; and
- a nominal equivalence class for what that occurrence denotes.

Its second important move is joint decoding of a coherent witness parse rather
than independent pointer decisions. That directly targets marginal-route
v1.1's 806 one-occurrence failures.

The supplied plan was not ready to run verbatim. It contained one stale stage,
one interface mismatch, and two large attribution risks. The revised decision
is:

> `RETAIN_COFC_AS_LEADING_SUCCESSOR; RUN_CURRENT_FACTORIZED_FALSIFIER_FIRST;
> REQUIRE_CPU_NONTRIVIALITY_AND_MATCHED_CONTROLS_BEFORE_NEURAL_COFC`.

## 2. What The New Evidence Changes

### 2.1 The addressed experiment is already closed

The proposal's Stage A says to run the ordinal/count addressed canary. Job
`694928` has already done so and is rejected:

- witness rows: 59.500%;
- packet/joint/relation: 60.9125%;
- state: 80.1375%;
- answer: 90.225%; and
- minimum-cardinality joint: 43.606%.

It cannot be presented as a pending experiment.

### 2.2 The scale audit rejects a simple magnitude diagnosis

The no-optimizer audit is also complete. Zero ordinal yields 0% witness rows;
zero count yields 0.4875%; scale 1.5 improves witness rows to 70.425% but remains
far below the 89.925% marginal parent and degrades events to 94.9625%.

This is stronger than the previous "ordinal norm became too large" story.
Address variables contain useful information, but the rejected model needs both
and entangles them with shared query/key geometry. A separately routed scalar
address bus remains scientifically distinct; post-hoc rescaling does not.

### 2.3 The immediate factorized test is already concrete

`R12_ER_FACTORIZED_WITNESS_ROUTE_PREREG.md` defines a witness-only
`14 x 12 x 14` residual table plus twelve bounded, zero-initialized role gates,
for 2,364 parameters. Centering and bounded output prevent an unbounded
cardinality shift; zero gates preserve the v1.1 logits exactly at
initialization. The route cannot read identity bytes or outcomes.

This is cheaper and more attributable than jumping immediately to a 2--3M
parameter COFC. The updated score path includes same-seed, same-parameter
baseline, structural-only, and shuffled-address arms, with a frozen +0.5-point
treatment advantage requirement. A pass would show that the current grammar
needs a source-local count/role/ordinal coordinate. It would not validate object
files or joint inference; a high structural-only score would instead identify
finite syntax routing.

Exact source commit `4643d1a51defe53397f9bed481051621d85c0b11` is now frozen
and pushed before any post-commit seed. No H100 job or new probe score exists.

## 3. Strongest Parts Of COFC

### 3.1 Occurrence and identity are correctly separated

ER-TT deliberately allows the after-witness list to repeat the same nominal
symbol. The correct representation therefore requires:

```text
physical occurrence A != physical occurrence B
nominal identity(A) may equal nominal identity(B)
```

This cleanly explains why a wrong duplicate occurrence remains a pointer error
even when the equality-derived relation is semantically close or identical.

The interdisciplinary references support the abstraction, not the result.
Kahneman, Treisman, and Gibbs describe object files as temporary episodic
representations linking successive states of one object
([PubMed](https://pubmed.ncbi.nlm.nih.gov/1582172/)). Green,
Karvounarakis, and Tannen explicitly tag input tuples so provenance survives
relational operations
([PODS paper](https://web.cs.ucdavis.edu/~green/papers/pods07.pdf)). The
June 2026 Dual-State Slot Attention preprint reports slot swapping when
appearance and identity share one state and improves its video task by
separating them ([arXiv](https://arxiv.org/abs/2606.12601)). None of these
papers demonstrates that COFC will solve ER-TT.

### 3.2 Joint paths match the observed error geometry

Every marginal-route failure has exactly one wrong occurrence, commonly a
neighbor that locally outranks the correct late after-witness. A complete path
can recover a locally second-ranked candidate when the aggregate path is best.
This is the useful computational import from global sequence alignment
([Needleman-Wunsch](https://pubmed.ncbi.nlm.nih.gov/5420325/)).

### 3.3 The executor remains untouched

COFC correctly leaves the source-deleted relation motor alone. That respects
the strongest causal fact in the project: valid packets execute exactly, so
compiler work should not be disguised as more recurrence.

### 3.4 The proposal exposes good interventions

Address swaps, nominal-signature swaps, joint-decoder ablation, object-ID
permutation, source poison, and directory shuffle manipulate distinct causal
objects. These are much more diagnostic than another undifferentiated capacity
increase.

## 4. Corrections Required Before Implementation

### 4.1 Current ER-TT monotonicity is almost deterministic

The renderer emits each active rule as an opcode, exactly `N` before symbols,
a separator, and exactly `N` after symbols. There are no opaque distractors
inside a witness segment. Once a system finds the boundary and cardinality,
the legal monotone path is effectively the ordered list itself.

That makes monotone alignment legal, but weakens its scientific value on the
current board. A success could be fixed-grammar parsing rather than resolution
of ambiguous associations. The CPU falsifier must measure the number of legal
paths and introduce a separately preregistered distractor grammar before COFC
can test its claimed mechanism.

### 4.2 One-to-one is local to occurrence positions

Candidate non-reuse is valid within one ordered before or after path because
physical source positions are distinct. It is invalid as a global relation
constraint. ER-TT relations are intentionally total and non-bijective, and
self-maps are legal.

Thus:

- separate monotone paths per segment are admissible;
- repeated nominal classes across selected after occurrences are admissible;
- global Sinkhorn/Hungarian relation assignment remains forbidden; and
- factors may never force nominal bijectivity.

### 4.3 The proposed query directory does not match this board

The external text assumes that the late query contains a symbol referent whose
signature can be matched against an object directory. Current ER-TT queries are
position numerals: `Q1...Q6` or `ASK 1...6`. Both recent canaries already score
query routing at 8,000/8,000.

An object directory is a good future test for referential or alias-bearing
queries, but it is irrelevant to the current witness residual. Adding it now
would change an interface that is already exact and confound attribution.

### 4.4 Factor messages can become host repair

Monotonicity, segment membership, cardinality agreement, and equality between
model-selected source objects are legitimate source-local factors. Graph
validity, executor consistency, final answers, retries, or target relations are
not. The factor schedule must be fixed; invalid outputs stay wrong.

The factor-graph literature shows how global functions can be decomposed into
local factors and processed by message passing
([primary paper](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf)).
It does not determine which Shohin factors are scientifically permissible.

### 4.5 Cardinality extrapolation is a separate experiment

Training on `N=3...6` and immediately scoring `N=7...9` simultaneously changes
tensor shape, grammar length, candidate count, query vocabulary, state size,
and evaluator limits. That is a valuable eventual test, but not the first COFC
board. First establish fresh in-range renderer/occurrence transfer; then freeze
a separate extrapolation board.

### 4.6 The parameter estimate is provisional

The proposal's 2--3M estimate is plausible under the 14.47M remaining headroom,
but no implementation or optimizer-state audit exists. Dynamic programming,
factor rounds, temporary state, object-directory bytes, and sequential compute
must be reported even when they add no parameters.

## 5. Disposition Of Proposed Components

| Component | Disposition | Reason |
|---|---|---|
| Separate occurrence and nominal ledgers | **Retain as core** | Directly represents the observed aliasing distinction |
| Segment-local joint alignment | **Retain conditionally** | Matches rank-two errors; first prove a nontrivial legal-path problem |
| Small source-local factor graph | **Retain with firewall** | Can enforce coherent parses without execution only if factors are frozen and local |
| Tied recurrent role reader | **Defer one stage** | Useful for extrapolation but should not be bundled with first joint test |
| Intrinsic two-sided address code | **Defer one stage** | Better extrapolation hypothesis than finite lookup, but a separate intervention |
| Object-directory late query | **Future board only** | Current ER-TT query is positional and already exact |
| Fixed proofreading rounds | **Optional later ablation** | Must remain internal, fixed-cost, and unable to inspect outcomes |
| `N=7...9` development | **Separate board** | Otherwise mixes mechanism, shape, and extrapolation failures |
| Global one-to-one assignment | **Reject** | Contradicts legal non-bijective relations |
| Strong biological equivalence claims | **Reject** | References motivate abstractions but do not validate Shohin |

## 6. Best Experimental Sequence

1. Freeze and run the existing 2,364-parameter factorized train-only canary.
2. Build a CPU COFC legality/nontriviality suite before any neural COFC source.
3. Quantify whether the current grammar has multiple legal paths; if not, add a
   new distractor grammar rather than claiming joint inference on a deterministic
   parse.
4. Compare independent marginals, the factorized table, joint COFC, independent
   COFC with identical unary logits, fused-ledger COFC, and structural-only/
   shuffled controls in one matched train-only experiment.
5. If the matched mechanism gates pass, generate a fresh `N=3...6` board with
   unseen renderers, distractors, object-ID permutations, and non-bijective
   rules.
6. Only then freeze a distinct `N=7...9` extrapolation and referential-query
   program.

## 7. What Different Outcomes Would Mean

- **Factorized table passes:** current ER-TT needed a local structural
  coordinate; do not call that general object-file compilation.
- **Factorized table fails, joint COFC passes:** independent role decisions were
  the causal bottleneck.
- **Joint COFC helps only with distractors:** current board was too deterministic
  to identify the mechanism, but the expanded board supports it.
- **Joint alignment passes while unseen renderers fail:** unary landmark or
  segment discovery remains the bottleneck.
- **Compilation passes while a future referential query fails:** the retained
  nominal directory or query matcher is inadequate.
- **Only grammar-heavy factors pass:** the result is bounded parser engineering;
  the next experiment must weaken or vary the grammar.
- **Source-retained control wins after source-deleted COFC fails:** the compiled
  object state omits query-relevant information.

## 8. Final Decision

The smart model found a genuinely better conceptual architecture. “Surrogate
keys plus coherent joint parsing” is a more faithful response to the evidence
than adding richer positional vectors to independent bilinear heads.

The evidence-aligned version is narrower than the submitted text:

- the addressed canary and scale audit are finished, not pending;
- the 2,364-parameter residual route remains the immediate falsifier;
- monotone alignment must first be shown nontrivial;
- one-to-one constraints stop at physical positions inside a segment;
- current positional queries remain unchanged; and
- cardinality extrapolation and referential object directories receive their
  own later boards.

With those corrections, COFC is the leading successor architecture if the
factorized lookup cannot close the witness gate—or the leading harder-board
test if that lookup succeeds too easily.
