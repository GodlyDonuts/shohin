# R12 Conflict-Driven Residual Localization

**Status:** THEORY AND EQUIVALENCE AUDIT. No Shohin fit, H100 job, production
data build, confirmation score, architecture promotion, reasoning claim, or
primitive-novelty claim is authorized by this document.

**Working name:** Conflict-Driven Residual Localization (CDRL).

**Claim class:** a bounded **training / sample-allocation protocol** over a
known residual transducer family. It does not propose a new state ontology,
workspace, recurrence primitive, or late-query compressor. It borrows its
organizing metaphor from conflict-driven clause learning (CDCL) in SAT, not
from contemporary LLM architecture papers.

**Relation to live work:** complementary to Addressed Categorical Workspace
(ACW) / CGBR. ACW asks whether a hard addressed packet can learn durable
source-deleted transport under collision refinement. CDRL asks whether, at a
fixed updater class, supervising **minimal residual-preserving event cores**
improves depth-OOD exactness relative to matched non-structural curricula.
CDRL does not compete with Track S custody and must not divert ACW pilot
bytes, seeds, or confirmation protocol.

---

## 0. Why this object, and why now

Shohin's evidence chain shows a recurring shape:

- local one-step competence can appear (DRS first transitions 497/500; R4
  pointer binding large matched gains; source-scheduled atomic executor
  footholds);
- multi-step composition, width-OOD transport, and source-deleted reuse then
  collapse;
- averaging multiple futures from one prefix is not a new learning signal
  (`R12_FORKED_STATE_TRANSPORT_PREREG.md`);
- group-presentation / relation losses do not identify generators
  (`R12_PRESENTATION_CLOSED_RESIDUAL_TRANSPORT_PREREG.md`,
  `R12_AXIOMATIC_PRESENTATION_NO_GO.md`);
- exact residual realizations are Moore transducers up to conjugacy
  (`R12_REASONING_INVENTION_CHARTER.md`);
- the remaining admissible target is a training or oracle-allocation protocol
  with a resource-vector advantage
  (`R12_STRUCTURED_RESIDUAL_RESOURCE_LAW.md` §7).

Contemporary LLM doctrine answers composition failure with more tokens, longer
CoT, latent loops, or RL. Those levers are already matched controls here and
have not established causal transport at 125M. CDRL instead imports a method
from automated reasoning that was built exactly for **localizing blame inside
long failing traces**: conflict analysis.

In CDCL, a wrong assignment is not reweighted as a soft loss on the whole
formula. The solver extracts a small conflict clause, learns it, and backjumps.
CDRL asks for the residual analogue: given a long event history, extract a
short subsequence that preserves the residual class, and allocate supervision
to that core.

---

## 1. Capability theorem (exact core existence)

### 1.1 Residual family

Fix finite event alphabet `A`, query set `Q`, and answer set `Y`. A history is
a word `h in A*`. The residual behavior is

```text
rho_h(c, q) = R(h c, q) in Y union {bottom}
```

with causal equivalence `h ==_R h'` iff `rho_h = rho_h'`. Write `[h]` for the
class and `N = |{[h]}|` for the number of reachable residual states at the
scales under test.

### 1.2 Residual-preserving cores

A subsequence `h' ≼ h` (order-preserving, not merely a subset) is
**residual-preserving** when `[h'] = [h]`. It is a **core** when no proper
subsequence of `h'` is residual-preserving for `h`.

**Theorem A (core existence and length).** Every history has at least one
core. Every core has length at most the length of a shortest representative of
`[h]`. In particular, if the residual monoid admits representatives of length
`≤ w([h])`, then every core of `h` has length `≤ w([h])`, even when `|h|` is
arbitrarily large.

**Proof.** The set of residual-preserving subsequences of `h` is nonempty
(`h` itself) and finite. Any length-minimal element is a core. A shortest
global representative of `[h]` is residual-preserving for every history in the
class after deleting only residual-neutral material; any core is at most that
short.

**Theorem B (distractor deletion).** If `h = u e v`, and `[u v] = [u e v]`,
then event `e` is residual-neutral in that context and is absent from every
core of `h`. Consequently, uniform full-history supervision can spend gradient
on events that do not affect any future answer.

### 1.3 What is not claimed

Theorem A is classical residual-monoid hygiene, not a neural invention. It does
not beat the `ceil(log2 N)` retained-bit lower bound, does not identify hidden
coordinates under conjugacy
(`R12_HIDDEN_COORDINATE_IDENTIFIABILITY_NO_GO.md`), and does not give
polynomial active identification for arbitrary compact hypothesis classes
(`R12_ACTIVE_VERIFIER_QUERY_NO_GO.md` §4).

The only admissible empirical claim is resource-bounded optimization:

> **Conjecture C (core-allocation learnability).** Fix updater class `H`,
> parameter budget `p`, label budget `L`, update budget `U`, and precision.
> Let `D_full` be iid full-history terminal supervision. Let `D_core` replace
> each training history by one lexicographically-first minimum-length core
> under a frozen public residual oracle, keeping the same late queries and
> answers. Let `D_rand` replace each history by a random subsequence of the
> same length as that core, and `D_hard` keep full histories but upsample the
> highest-loss quartile. Then on a preregistered depth-OOD exact-transport
> board, the core-trained member of `H` exceeds each of `D_full`, `D_rand`,
> and `D_hard` by a locked margin at equal `(p,L,U)`.

Conjecture C is falsifiable and may be false. It is not a reasoning claim.

---

## 2. Axiomatic primitive (oracle protocol, not a module)

CDRL is defined without neural vocabulary.

1. **Public residual oracle `O*`.** On synthetic boards, `O*` evaluates
   `R(h,q)` by the exact task algebra already used for ACW/DRS generators. It
   is target-coupled and must appear in the oracle-call ledger.
2. **Core extractor `K`.** Given `h`, return the lexicographically-first
   subsequence among all minimum-length residual-preserving subsequences.
   Deterministic tie-break is part of the protocol identity.
3. **Allocation map.** Training set `D_core = {(K(h), q, O*(h,q))}`.
4. **Updater class `H`.** Any fixed class admitted by a sibling experiment
   (ACW packet updater, dense categorical recurrence, GRU, etc.). CDRL does
   not enlarge `H`.
5. **Evaluation.** Source-deleted late-query exactness on held-out depths,
   plus equivalent-history invariance and non-equivalent separation, with a
   complete resource vector.

No learned workspace, attention slot, or latent scratchpad is part of the
primitive. If a neural fit later uses ACW's packet as `H`, that packet remains
ACW's object; CDRL only changed the label allocation.

---

## 3. Equivalence dossier

Mandatory resource vector:
`(parameters, retained bits, precision, source bytes, training examples,
oracle calls, training FLOPs, inference FLOPs, sequential depth, external
memory, external execution)`.

| Candidate reduction | Preserves vector? | Verdict |
|---|---|---|
| Ordinary SFT on full histories | Same `H`, fewer effective distractors in CDRL | **Control**, not collapse |
| Fork-averaged multi-future loss | Different objective; fork collapses to mean CE | Distinct; fork already NO-GO |
| PCRT worst-witness + Coxeter relations | CDRL has no group presentation loss | Distinct; PCRT already NO-GO |
| CGBR packet-collision injection | CGBR splits on learned packet equality; CDRL projects histories by true residual equality before learning | Related control; must be matched |
| Hard-example mining by loss | No structural subsequence; `D_hard` is mandatory control | Control |
| Random length-matched subsequences | `D_rand` is mandatory control | Control |
| Active verifier / L* / CEGIS | CDRL freezes oracle transcripts; does not claim query-complexity invention | Boundary respected |
| External step executor / reject-retry decode | Inference protocol, not CDRL | Separate diagnostic; see §7 |
| Self-authenticating coded state | No in-state certificate | Distinct; coding NO-GO stands |
| MDL / shortest program selection | Core length is residual-representative length, not Kolmogorov complexity over programs | Distinct; MDL NO-GO stands |

**Exact collapse test (symbolic).** On any commutative event monoid where every
event is residual-essential with equal length, `K(h)=h` always, so
`D_core=D_full` and Conjecture C is vacuous. CDRL can win only on families
with residual-neutral distractors or compressible representatives. The finite
falsifier must include both a compressible family and a non-compressible
negative control where cores equal full histories.

**Resource-preserving unrolling.** Finite unrolling of any learned updater in
`H` remains in `H`'s comparator class. CDRL does not claim separation from
static circuits; it claims a sample-allocation advantage inside one `H`.

---

## 4. Prior-art boundary

Searched after the object was defined:

- CDCL conflict analysis and clause learning (SAT): metaphor and blame
  localization; not residual monoids over language-model updaters.
- Automata residual / Nerode congruence and shortest representatives: Theorem A.
- Grammatical inference state merging (RPNI, EDSM): partition refinement on
  observed tails; CDRL does not merge states, it projects training words.
- Coresets and prototype selection: related sample reduction; mandatory
  controls when adapted to sequences.
- CGBR / counterexample-guided synthesis: sibling project method; matched
  control, not identity.
- Group DRO / hard mining: loss-based, not residual-structural.
- AIDN / MatrixNet relation losses: rejected here as PCRT ingredients.

**Delta.** CDRL is the conjunction of (i) Nerode-core projection as the only
change to a frozen updater class, (ii) locked matched controls
`D_full` / `D_rand` / `D_hard` / CGBR-style collision sets, (iii) source-deleted
depth-OOD exact transport as the only promotion metric. No reviewed primary
source was found that states this conjunction as a tiny-LM residual-transport
protocol. Scoped absence does not license a world-first or primitive claim.

---

## 5. Finite falsifier (CPU only; gate 6 prerequisite)

### 5.1 Compressible positive family (Heisenberg mod `M`)

State `(x,y,z) in (Z/MZ)^3`. Events:

```text
A: (x,y,z) -> (x+1, y, z)
B: (x,y,z) -> (x, y+1, z+x)
C: (x,y,z) -> (x, y, z+1)
```

Late queries read any single coordinate. Residuals have size `M^3`. Words with
many cancelling distractors (e.g., inserts of `A` followed later by an inverse
only when `M`-arithmetic provides neutral pairs, or pure `C` padding when
equivalent representatives exist under fixed `(x,y)` commitments) admit cores
shorter than raw histories. Practical board construction uses explicit
**padding events** `P` with `U_P = Id`, which are residual-neutral by
definition and must be stripped by every correct core.

### 5.2 Non-compressible negative control

Free-word residual: late queries may read any event by index, so the residual
class of a history is the history itself. Cores equal full histories.
Conjecture C must not show a positive margin here; a spurious win rejects the
extractor or the evaluator. (Register-overwrite families are compressible and
are not this negative control.)

### 5.3 Frozen mechanics gates (no neural fit)

1. Core extractor is deterministic; byte-identical across two processes.
2. Every core is residual-preserving under exhaustive query replay.
3. No core contains a padding event.
4. On the negative control, core length equals history length for every sample.
5. Oracle-call ledger counts every `R` evaluation during extraction.
6. Length distribution of `D_core` vs `D_rand` is identical by construction.

Only after these CPU gates pass may a separately preregistered neural
optimization board be proposed. That board is not authorized here.

---

## 6. Matched controls for any future neural board

If and only if a future preregistration reopens a neural test, arms are:

| Arm | Allocation | Notes |
|---|---|---|
| `full` | raw histories | Ordinary CE |
| `core` | `K(h)` | Treatment |
| `rand` | random subsequence of `|K(h)|` | Length-matched sham |
| `hard` | full histories, loss upweight | Non-structural hard mining |
| `cgbr` | collision-injected set at equal labels | Project sibling method |
| `short_native` | iid native short histories with same length law as cores | Distribution control |

All arms share `H`, `p`, `L`, `U`, seed, and precision. Promotion requires
pre-registered depth-OOD margins over **every** control, not over `full` alone.

---

## 7. Rejected sibling: locally verified microstep reject-retry

A tempting inference fix for DRS compounding is: check each emitted microstate
with a public transition checker and resample on failure. Exact analysis:

- A **complete** local checker for a deterministic step computes that step. Using
  it to accept/reject proposals is external single-step execution plus a
  proposal distribution test. It is an SSC-class diagnostic, not internalized
  reasoning.
- A checker that only sees **model-authored** prior state cannot stop
  compounding: it certifies consistency with an already-wrong rail.
- A checker that sees **solver** prior state is external state transport.

Therefore reject-retry decoding is **not** part of CDRL and is not authorized
as an R12 invention. It may remain a counted diagnostic of proposal rank, akin
to oracle@k.

---

## 8. Decision

1. **Admit Theorem A/B** as accounting lemmas for residual-neutral distractors.
2. **Reject CDRL as a primitive, workspace, or reasoning mechanism.**
3. **Conjecture C: CLOSED NEGATIVE** on frozen board `R12-CDRL-NEURAL-v1`
   (Newton job `691750`, decision SHA-256
   `ad94ac15ca17eaa2c5381aa0a3f94fc60a49dbbf2a528552a1212b3ecf1cabdb`).
   Core-only allocation loses to full/hard by ~78pp median depth-OOD exactness
   when evaluation restores distractors. See
   `R12_CDRL_NEURAL_OPTIMIZATION_RESULT.md`.
4. **CPU mechanics suite: PASS** on the Heisenberg padding family and the
   free-word negative control (`pipeline/cdrl_conflict_cores.py`; report
   SHA-256 `82f74581db7259c29298bb9734c6e49cbb40d727f3215b34eb8a75fdbcde1d9c`).
5. **Do not authorize** Shohin fits, ACW weight changes, confirmation seeds,
   Track C work, threshold retunes, or any claim that core training alone
   produces autonomous reasoning.
6. A mixture `core∪full` successor would need a new preregistration; CGBR/ACW
   remains the durable state-transport claim.

This keeps the project's invention bar intact while opening a SAT-inspired
sample-allocation axis that the residual-ontology chain has not yet tested.
