# R12 CTAA Neural Falsifier

## Status

**Revision 2 draft architecture and custody contract. Not source-frozen.** No production
board seed, training seed, H100 job, development access, confirmation access,
or reasoning claim is authorized by this document.

An independent adversarial review rejected revision 1 for source freeze. Its
confidence gate was unattainable at the stated stratum sizes; the generator did
not instantiate the claimed board; semantic, renderer, and lexical shifts were
confounded; class and query were coupled; long programs had repeated-opcode
shortcuts; deletion depth ignored the actual state and answer; and the late
query existed on disk before execution. Revision 2 changes the experiment
rather than relaxing those findings.

The preceding CPU audit establishes only coherent finite mechanics. This
experiment asks whether a closure-aligned recurrent parameterization learns
fresh episodic copy actions and reuses them over causally long programs more
reliably than a favorable generic recurrence with identical parameter count,
state, and effectively identical FLOPs.

The strongest possible positive claim is bounded: episodic compilation and
source-deleted recurrent execution of three-position copy actions. The causal
quotient contains only 27 maps, about 4.76 bits. Passing cannot establish broad
natural-language reasoning, arithmetic, planning, theorem proving, or general
program induction.

## Immutable Base

| Object | Frozen identity |
|---|---|
| Raw Shohin checkpoint | `train/flagship_out/ckpt_0300000.pt` |
| Raw checkpoint SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| Base unique parameters | 125,081,664 |
| Tokenizer | `artifacts/tokenizer/tokenizer.json` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Vocabulary / PAD | 32,768 / token ID 1 |
| Qualified memory initialization | ordinary compiler from job `693049` |
| Qualified compiler SHA-256 | `747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991` |

The checkpoint must load strictly into the unmodified `GPT` in
`train/model.py`, with zero missing or unexpected keys. `model.py`, tokenizer,
`n_loop`, embeddings, cache semantics, and trunk blocks may not change. The
base stays frozen throughout the falsifier.

## Architecture

### Source compiler

The compiler accepts only right-padded token IDs and derives validity as
`ids != 1`. It receives no spans, parser metadata, row metadata, entity IDs,
targets, executor output, answer, retry result, or verifier signal.

One frozen Shohin pass captures:

- causal residuals after block 19, `h19 [B,L,576]`;
- normalized final residuals after block 29, `h29 [B,L,576]`.

The layer-19 `LayerNorm(576)`, `Linear(576,384)`, and five-layer 384-wide
encoder are initialized from the hash-qualified ordinary compiler. A separate
`LayerNorm(576) -> Linear(576,384)` reads layer 29. Its projection starts at
zero; scored qualification requires nonzero causal dependence on both paths.
The fused memory is:

```text
Encoder(W19(LN19(h19)) + W29(LN29(h29)))
```

A two-layer, eight-head, width-384, FF-1024 generic slot decoder emits:

```text
cards        [B,4,3,3]   categorical logits
binding      [B,4,4]     local-opcode to physical-card logits
initial      [B,3,3]     categorical logits
opcode_tape  [B,41,5]    four declaration-local opcodes plus STOP
```

Card slots are source-visible addresses `W1..W4`; physical rule lines are
reordered by the exact `S4` balance design. A determining before/after witness
uses three distinct entities, so each card is identifiable. The compiler emits
an independent opcode-to-card permutation. The local opcode tape contains
exactly one model-compiled STOP followed by valid adversarial suffix events;
the physical execution schedule is derived from binding plus tape.

### A4 binding-completion neural slice

The binding-identification diagnostic is a separate source-only qualification,
not a scored recurrent result. Every semantic compiler family is expanded into
the complete 24-member declaration-order orbit. Cards, opaque names, renderer,
initial state, physical schedule, query, depth, and program class remain fixed
inside an orbit. The 12 even permutations form optimization data and the 12 odd
permutations remain sealed. Every local opcode/card cell therefore appears
`3/3/3/3` in each half, and train/confirmation token-length histograms must
match within every orbit.

Four opcode slots and four physical-card slots are decoded independently from
the same source memory. No opcode may self-attend to another opcode, and no
card may self-attend to another card. One globally connected structured
qualifier is trained on A4 only, then the compiler freezes and the qualifier is
discarded for the comparison. A single detached eight-slot A4 cache is hashed.
Every fresh readout consumes those byte-identical tensors in the same order:

- **bi-equivariant treatment:** one shared pair scorer receives
  `[opcode_i, card_j, zero_context]` for each of the 16 opcode/card pairs;
- **favorable global structured control:** the exact same pair scorer receives
  `[opcode_i, card_j, all_eight_slots]`, permitting arbitrary parity shortcuts;
- **atomic lookup negative:** an eight-slot `-> 24` classifier, reported only as a
  deliberately support-starved lookup baseline because odd classes have no
  positive A4 examples.

Treatment and favorable control both emit `[B,4,4]`, receive identical
four-cell cross-entropy, use the same `3840 -> 156 -> 1` network for the same
16 calls, have exactly `599,353` readout parameters, and require exactly
`9,587,136` dense analytic MACs. The treatment is equivariant to all 24 opcode
permutations crossed with all 24 card-slot permutations by construction. The
complete decisive system is `138,589,297` parameters, leaving `11,410,702`
below the strict cap. Profiler receipts remain mandatory; analytic equality
does not establish equal kernel-launch or wall-clock cost.

Both decisive arms must fit all 12 A4 bindings to the preregistered threshold
in all five seeds before confirmation source is opened. Every seed artifact
durably freezes the compiler, readouts, probes, exact A4 cache, ordered row
commitments, and train metrics with zero confirmation access. A separate
source-free freeze process validates every admitted hyperparameter, immutable
input hash, fit gate, arm/probe lattice, cache commitment, and access counter,
then writes one exact five-seed hash manifest. A source-only predictor requires
that manifest, accepts no oracle path, opens the odd source once, and commits
raw logits plus odd slot caches. A separate assessor accepts no source path,
validates all five seed artifacts again, atomically claims the globally unique
oracle-access ledger, opens one oracle blob once, and scores without tuning.
Every source/oracle row shares one opaque unique row ID; ordered ID equality is
mandatory. Report projected and raw binding exactness, raw assignment
validity, projection rescue, NLL, parity confusion, and every individual
permutation.

Four disposable single-slot probes test whether any one slot contains the full
binding. The compositional chimera is now source-free: every one of its four
slots comes from a distinct A4/even donor in the same family, while a
transposition derives an odd target. No odd representation or label enters
this diagnostic. After the assessment freezes, a separate disposable all-S4
job consumes the committed labels without reopening the oracle and must show
that each readout can optimize all 24 classes. A source-free finalizer applies
the admission's immutable five-seed thresholds to treatment accuracy, raw
assignment validity, matched-control advantage, single-slot leakage, A4-derived
odd chimeras, complete 60-byte packet exactness, opcode excitation, binding
counterfactual effect, all-S4 capacity, and a counterbalanced measured
forward/backward/optimizer resource receipt. Failure of any mandatory gate
invalidates binding attribution. Passing this slice establishes only
declaration-binding completion; it does not establish source-deleted memory,
multi-step execution, or reasoning.

The admission binds the exact Git commit and a canonical SHA-256 over every
tracked protocol source and direct execution dependency. Production entry
points reject untracked or dirty protocol files and reject outputs outside the
single absolute custody directory. Cross-stage tensor artifacts are hashed
and deserialized from one `O_NOFOLLOW` file-descriptor read under restricted
`weights_only=True`; executable pickle loading is forbidden. Artifact and
ledger writes use exclusive creation plus directory `fsync`. Decision
thresholds are source-canonical: all five seeds pass, A4 fit is 99%, held-out
factorized exactness is 75%, matched-control advantage is ten points,
single-slot leakage is at most 10%, A4-derived odd chimera exactness is 75%,
and measured resource gap is at most 5%.

These local mechanics are not sufficient custody. Source freeze still requires
an independently owned, secret post-freeze challenge; separate source/oracle
OS identities or mount namespaces; a hermetic launcher with sanitized imports
and dependency hashes; externally signed append-only stage lineage; and actual
source-deleted packet execution through the frozen core. Owner-readable `0400`
files and a public deterministic challenge seed are explicitly insufficient.

### Hard deletion boundary

Compilation materializes exactly:

```text
cards        uint8[4,3]   12 bytes
binding      uint8[4]      4 bytes
initial      uint8[3]      3 bytes
opcode_tape  uint8[41]    41 bytes
total                       60 bytes per row
```

The scored runtime must serialize these bytes, terminate the compiler process,
destroy source IDs, validity masks, residuals, logits, caches, and lexical
memories, then start an executor process that receives only the packet and the
core checkpoint. The host resolves one local opcode to one physical card at a
time; the core sees only that current card and state. It never receives
future events, source, query, target, or verifier state.

Only after a read-only execution receipt commits is a separate query source
materialized and disclosed. A separate
Shohin pass and the same memory/decoder parameters materialize one canonical
query-position byte. The query never conditions the recurrent state. Raw
opaque-name realization is outside this falsifier.

Scored custody uses three keyed files per partition. `*_program.jsonl` contains
only family ID and program source. `*_query.jsonl` contains only family ID and
query source and remains sealed until the execution receipt commits.
`*_oracle.jsonl` contains categorical packet labels, prefix states, terminal
state, answer, and grouping metadata and is opened only by the independent
assessor after raw predictions commit. No process receives two stages merely by
ignoring extra JSON keys.

### Recurrent treatment

Categorical action/state inputs are row-one-hot `L,R in {0,1}^{3x3}`. CTAA
receives only composition-aligned products:

```text
Phi_T(L,R)[i,j,k] = L[i,j] * R[j,k]     # 27 features
h = ReLU(Linear(27,2912)(Phi_T))
logits = Linear(2912,9)(h)              # reshape [3,3]
```

The same 107,753 parameters implement action-on-state and action-on-action.
After every call, exact categorical argmax is materialized before reuse. The
executor maintains two hard routes:

```text
state route:      s_t = F(a_t, s_(t-1))
composition route c_t = F(a_t, c_(t-1)); s'_t = F(c_t, s_0)
```

STOP latches both routes. Neither route receives continuous persistent state.

### Favorable generic control

The Full Outer-Product Recurrent Control (OPRC) receives every pairwise input
interaction, including every CTAA interaction:

```text
Phi_G(L,R)[p,q] = vec(L)[p] * vec(R)[q] # 81 features
h = ReLU(Linear(81,1184)(Phi_G))
logits = Linear(1184,9)(h)              # reshape [3,3]
```

Its 81-feature representation separates all 729 finite `(L,R)` pairs and its
1,184 hidden units exceed that count. It must fit an independently generated
arbitrary 729-cell transition table to 100% before it is accepted as a
control. Failure invalidates the experiment rather than helping CTAA.

## Exact Resource Ledger

Measured against the real raw-300k checkpoint:

| Component | Parameters |
|---|---:|
| Shohin trunk | 125,081,664 |
| Shared compiler addition | 12,800,527 |
| CTAA or OPRC core | 107,753 |
| **Complete system** | **137,989,944** |
| **Headroom below 149,999,999** | **12,010,055** |

CTAA and OPRC core parameters are exactly equal. Their analytic one-transition
costs are 215,530 and 215,584 FLOPs, a 54-FLOP or 0.0251% difference. Final
admission requires measured profiler receipts for forward, backward, optimizer,
curriculum selection, compiler training, and inference at active depths
1/16/32/39. The packet has 41 event slots and requires both STOP and a poison
suffix, so 39 is the maximal executable depth; the former depth-64 profiling
requirement was geometrically impossible.
Train FLOPs must match within 5%; active and trainable parameters within 0.1%;
committed state and packet bytes exactly.

## Fixed Semantic Split

An action `abc` maps `(s0,s1,s2)` to `(s[a],s[b],s[c])`. The representation is
three categorical pointers, never a 27-way class. Every pointer value occurs
exactly three times at every coordinate in every split. Rank distribution is
1/6/2 in every split.

| Split | Actions |
|---|---|
| Train | `000 011 012 101 120 122 202 211 220` |
| Development | `002 010 021 100 110 112 201 221 222` |
| Confirmation | `001 020 022 102 111 121 200 210 212` |

No scored action may enter training atomic labels, closure outputs, excitation
allocation, continuation bases, query bases, or tuning.
Only the 35 train-action pairs whose composition also remains in train may
receive closure supervision.

The OPRC representation-capacity preflight is the sole exception: before any
board seed exists, disposable weights must fit an arbitrary label assigned to
each of the complete 729 finite tuple pairs. Those weights, optimizer state,
labels, and seed are destroyed and cannot initialize or select any training
arm. This establishes control capacity, not task exposure.

## Renderer And Names

Six binary renderer factors cover declaration, witness, initial state,
schedule, STOP/suffix, and query grammar. For factor bits `x0..x5`:

```text
p1 = x0 xor x1 xor x2 xor x3
p2 = x2 xor x3 xor x4 xor x5
```

Train uses syndrome `00`, development `01`, confirmation `10`, and `11` is
reserved. Each coset has 16 compositions and matched low-order factor
marginals. Name pools are split-neutral in surface form, fixed-width,
tokenizer-admitted, component-disjoint, cryptographically assigned after
semantic generation, and independent of actions.

Semantic, renderer, and lexical novelty are independent axes. Development and
confirmation each contain all eight `2 x 2 x 2` factorial cells: every axis is
either in-distribution train or the partition's held-out value. Single-shift,
pair-shift, and triple-shift results are reported separately. Shared fixed
grammar is an explicit whitelist; admission requires zero non-whitelisted
token 13-gram overlap and matched token-length distributions, not the
impossible claim that controlled grammar itself never repeats. Maximum length
is 2,048 tokens.

## Board Sizes And Supervision

Repeated optimization contexts are not independent samples. Atomic and
two-action domains are finite and exhaustively enumerated, so their primary
result is exact finite-set accuracy, not a confidence interval over duplicated
rows. Views, twins, interventions, and renderings of one long semantic family
remain one statistical cluster.

| Component | Optimization exposures / unique cases |
|---|---:|
| Train atomic: `9 actions x 27 states x 64 contexts` | 15,552 / 243 |
| Train two-action: `35 closed pairs x 27 states x 64 contexts` | 60,480 / 945 |
| Compiler schedules: `4,096 x depths 1..8` | 32,768 |
| **Total optimization exposures** | **108,800** |
| Exact atomic audit per semantic axis | 243 |
| Exact two-action audit per semantic axis | 2,187 |
| Long per split: `8 cells x 3 classes x 2 depths x 576` | 27,648 |
| Triple-shift primary interventions: `3 x 3,456` | 10,368 |
| Targeted equivalent/prefix/STOP diagnostics: `3 x 6 x 144` | 2,592 |
| **Long scored total per split** | **40,608** |

Compiler cards, initial state, schedule, STOP, and late query may be directly
supervised. Recurrent execution labels are allowed only for atomic and
two-action rows. No depth above two receives trajectory state, terminal state,
answer, repair, query-conditioned state, or verifier supervision.

Every class-depth-factorial cell has 576 unique semantic families. It contains
exactly two examples in every one of the 288 renderer by
query-position/initial-permutation cells, eliminating the former parity
confound between separate 16- and 18-way marginal cycles. Initial-state symbols
are always distinct, preventing action effects from disappearing because equal
symbols were sampled.

The three long classes are:

1. `stable_rank_two`: varied programs whose composite remains rank two;
2. `implicit_final_collapse`: no rank-one card, with rank one created only by
   the final active composition;
3. `explicit_final_collapse`: a rank-one card is the final active event.

Every program uses at least three card slots, has maximum opcode run three,
normalized event entropy at least 0.75, and map-deletion depth at least one
quarter of raw depth. STOP follows the final active event and a valid poison
suffix follows STOP. The evaluator reports map-, terminal-state-, and
answer-deletion depth separately, along with shortest equivalent word length.
Because the monoid has only 27 maps, raw length is never represented as
intrinsic final-answer complexity. Primary sequential evidence is exactness of
the complete unseen prefix-state trace under the source-blind event streamer.

Required clustered counterfactuals include order twins,
equivalent-composite twins, prefix twins, renderer/name recodings, and
source-poison variants. Canonical packet hashes reject duplicate semantic
families before rendering.

## Training Arms

Within each paired seed, every arm shares one frozen compiler checkpoint and
the same compiled hard packets. Core initialization, optimizer family,
precision, batch policy, tuning-trial count, query reader, and update budget
are matched. Across five paired seeds the compiler is independently initialized
and trained from the same qualified memory initialization.

1. CTAA with atomic and closure supervision.
2. Parameter/state/FLOP-matched OPRC with the identical atomic and closure
   supervision.
3. CTAA without closure supervision, padded with charged atomic calls to match
   the transition-call budget.
4. CTAA with a seed-paired permutation of closure labels.

Every primary arm receives the same finite cases, exposure counts, update
count, optimizer family, and four differentiable transition calls per
two-action example. Any dummy or repeated call used for compute matching is
recorded and charged. Compiler weights freeze before core fitting; core weights
freeze before any scored packet is executed. Scored source is compiled once
before arm identity is attached, so core results cannot affect compilation.

The evaluator is physically staged and oracle-blind. A program compiler accepts
only `family_id` plus `program_source` and commits raw card, binding,
initial-state, and local-opcode tape bytes. The resolved physical schedule is
derived deterministically and is never a separately predicted packet field. A
separate sealer derives packet validity from exact STOP
geometry; invalid rows remain in every denominator and never reach the
executor. A fresh process receives only valid fixed packets and one frozen core.
Only after its read-only execution receipt exists may another process open the
sealed query source and materialize query bytes. An oracle-blind committer then
records every source row, including missing downstream stages. The assessor has
no source input and spends the partition access before opening oracle-only
labels. Assessment retains family-level outcomes needed for paired clustered
statistics. The source implementation now places all mandatory interventions
below in a versioned 29-operation runtime plan over 864 anchors (25,056
attempts per seed), including independently replayed card-only, binding-only,
and compensated three-cycle controls. Capability-time resource/intervention
receipts, independent dual rescoring, unmocked Linux custody, and the stronger
binding-identification boards remain incomplete; therefore this paragraph is
not source-freeze or seed authorization.

Five paired master training seeds are required. Each derives initialization,
batching, compiler, core, and curriculum seeds through tagged SHA-256. Report
every seed, equal-seed means, exact family counts, finite-domain exact scores,
one-sided cluster-bootstrap bounds for long families, simultaneous Holm
correction across preregistered marginal strata, and a 100,000-draw paired
hierarchical bootstrap over seeds and semantic families. A confidence bound is
never computed by treating repeated renderings or steps from one family as
independent.

## Mandatory Interventions

- zero, batch-rotate, and donor-transplant `h19` and `h29` independently with
  identical right-padding masks;
- source deletion and post-seal source poison;
- entity, witness, and opcode alpha recoding;
- renderer substitution and physical rule-line shuffle;
- card-storage reindex with binding reindex and byte-identical local tape;
- card-only, binding-only, and compensated non-involutive opcode relabeling;
- `A4`-only binding training with odd-permutation confirmation, plus
  adjacent-transposition rebinding paths through the exact `S4` diameter six;
- witness corruption and paired shuffled law;
- schedule order twin and future masking;
- STOP relocation and post-STOP poison;
- midpoint donor-state transplant and action-card transplant;
- late-query swap and query isolation;
- packet transplant across source texts;
- state-route versus composed-route agreement.

The treatment may not inspect candidate validity, executor results, terminal
state, answer, control result, or retry feedback while compiling or executing.

## Advancement Gates

Every one of five seeds must pass:

- fresh cards, independent binding, local opcode tape, derived schedule,
  initial state, and STOP each at least 99%;
- exact finite atomic transition accuracy 100% on all 243 cases in each
  semantic axis and exact two-action route accuracy 100% on all 2,187 cases;
- long-program prefix-state accuracy is reported for each action, rank,
  renderer, and step-quartile marginal separately; no cross-product stratum is
  implied;
- depth-16 exact-chain lower bound at least 95%;
- depth-32 exact-chain lower bound at least 90%;
- donor-state and donor-action following lower bound at least 95%;
- state-route/composed-route agreement at least 99.9%;
- alpha, source-poison, future, query, and post-STOP invariance exactly 100% on
  the frozen intervention board;
- CTAA depth-16 accuracy exceeds every favorable neural control by at least ten
  points in every seed, with paired 95% lower bound also at least ten points;
- closure CTAA exceeds no-closure CTAA by at least five points;
- shuffled-closure CTAA remains at least ten points below intact-closure CTAA.

If OPRC comes within three points of CTAA, closure-specific attribution is
rejected even if both systems are strong. Any parameter, state, FLOP, custody,
source-deletion, or control-capacity failure invalidates the capability result.

## Custody

Before a board seed exists, all of the following must be complete and committed:

1. complete typed board builder and independent CPU oracle;
2. fixed-width tokenizer-admitted name allocator;
3. exact family/twin/intervention generator;
4. compiler/executor/query orchestrator that materializes query bytes only
   after an immutable execution receipt exists;
5. training/evaluation/assessment code;
6. arbitrary-table OPRC capacity preflight;
7. profiler and parameter/state receipts;
8. mutation tests for metadata, outcome, source, query, future, and verifier
   leakage;
9. clean source commit and independent adversarial review.

After that commit, draw one public board seed, build once, independently rebuild
byte-identically, hash and seal confirmation, and commit the admission receipt
before drawing five training seeds. Each seed freezes one checkpoint before the
sole development read. Confirmation opens once only if every frozen development
gate passes. No threshold may be relaxed and no board rescored.
