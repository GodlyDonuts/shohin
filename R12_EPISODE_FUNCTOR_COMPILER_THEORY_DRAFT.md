# R12 EPISODE Functor Compiler — Architecture-Theory Draft

**Status:** theory and architecture redesign; supersedes the implementation-readiness claim of `R12_EPISODE_ORBIT_CAUSAL_FIRST_FIT_PREREG_DRAFT.md`. No source freeze, board seed, training seed, development read, GPU job, or capability claim is authorized by this document.

**Disposition of frozen source `80dc07a`:** retain as a favorable baseline and custody reference. It is no longer a constraint on the candidate architecture.

**Protected base:** Shohin step 300,000, 125,081,664 parameters, SHA-256 `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`.

**Global complete-system limit:** strictly below 200,000,000 unique parameters.

**Continuation-pretraining hold:** unchanged. This work does not start, prepare, queue, or modify continuation pretraining.

---

## 1. Correction

The previous OCSI draft made two different errors.

First, it treated an explanatory factorization

```text
world -> state -> binding -> operator -> query -> answer
```

as though the frozen implementation already exposed those factors as independently trainable objects. The reported source audit shows that it does not: the committed object is only a four-slot workspace; binding and operator choices arise during query execution; assessment overrides are non-differentiable; and the detached seal blocks compiler gradients. Therefore the proposed transplant losses were not defined on the actual frozen interface.

Second, and more importantly, the existing two-query EPISODE board does not identify a reusable world model even under an ideal architecture. It identifies only a query-blind sufficient statistic for two answers.

The correct response is not to rename hidden tensors or to add a donor override to the old module. It is to change both the mathematical target and the architecture so that:

1. the committed object is an explicit episode-local transition system;
2. opaque action names and action dynamics are separately manipulable;
3. grammar parsing is separated from state transition;
4. ordered composition occurs in a source-deleted execution core; and
5. the late-query family is large enough that an answer-pair cache cannot satisfy the protocol.

---

## 2. Decisive No-Go: The Finite-Query Cache Theorem

Let `X` be the set of world prefixes, let `Q={q_1,...,q_k}` be the complete set of late queries that can ever be asked for one compiled world, and let

```text
f : X x Q -> Y
```

be the target answer function.

### Theorem 1 — finite-query cache

There exists a query-blind compiler and late reader

```text
C(x) = (f(x,q_1), ..., f(x,q_k))
R(C(x), q_i) = C(x)[i]
```

that are exact on every row and use at most

```text
k * ceil(log2 |Y|)
```

answer-bearing bits, ignoring fixed indexing overhead.

### Consequence for current EPISODE

For each binding variant, current EPISODE exposes only two late query orders. A compiler can therefore store the two correct answers and a reader can select one after the query appears. Such a system is:

- query-blind during compilation;
- source-deleted after commitment;
- exact under compile-once / ask-both evaluation;
- invariant to post-seal source poison; and
- responsive to query swap.

It need not infer an action algebra, a physical world state, or a reusable transition law.

The six-case orbit does not repair this by itself because the three binding variants are separately visible prefixes. Each prefix can carry its own two-answer code. Cross-case training can improve optimization, but unless the architecture and query family force a common executable structure, high accuracy still does not identify reasoning.

**Decision:** the current 1,920-packet board remains a valuable action-sensitivity and anti-bagging diagnostic. It is not sufficient as the sole advancement board for reusable causal state.

---

## 3. Mathematical Capability Object

An episode defines an unknown finite controlled system

```text
M_e = (S_e, A_e, delta_e, Q_e, O_e)
```

where:

- `S_e` is the episode's causal-state set;
- `A_e` is a set of opaque episode-local action names;
- `delta_(e,a) : S_e -> S_e` is the action induced by name `a`;
- `Q_e` is a family of late observation queries; and
- `O_(e,q) : S_e -> Y` is the answer observer selected by query `q`.

For a word `w=a_1...a_L`, use row-time order

```text
delta_w = delta_(a_L) o ... o delta_(a_1).
```

The answer to a late program/query pair `(w,q)` from initial state `s_0` is

```text
y = O_q(delta_w(s_0)).
```

### 3.1 Causal residual quotient

Two histories or states are equivalent exactly when no admissible future can distinguish them:

```text
s ~ t
iff
for every continuation word w and every late query q,
O_q(delta_w(s)) = O_q(delta_w(t)).
```

The desired committed state is a representation of `S_e / ~`, not a named human ontology.

Every action descends to an endomorphism of this quotient:

```text
T_a([s]) = [delta_a(s)].
```

Every late query descends to an observer:

```text
R_q([s]) = O_q(s).
```

Ordered reasoning is composition in the transition monoid generated by `{T_a}`.

### 3.2 Gauge freedom

Any faithful internal chart is acceptable. If `G` is an invertible change of internal coordinates, then

```text
z'      = G z
T'_a    = G T_a G^-1
R'_q    = R_q G^-1
```

has identical behavior. The protocol must therefore avoid requiring arbitrary latent tensor equality across independently compiled episodes. It should score commuting behavior, interventions, and source-deleted execution. A categorical implementation reduces the admissible gauge to a permutation of anonymous latent states, which is easier to audit.

### 3.3 Faithfulness criterion

Let `phi` map causal quotient states into the compiled representation. If

```text
phi(delta_a(s)) = T_a phi(s)
R_q(phi(s))     = O_q(s)
```

for every reachable `s`, action `a`, and a separating family of continuation-query probes, then `phi` is injective on the causal quotient. If two quotient states mapped to one representation, all future probes would agree, contradicting separation.

This is the mathematical criterion the neural experiment should approximate. It does not require gold names for states, bindings, or operators.

---

## 4. Candidate Architecture: Episodic Functor Compiler

The candidate is an **Episodic Functor Compiler (EFC)**. A transformer remains the perceptual front end, but it no longer performs the entire reasoning computation through ordinary token residual flow.

The source compiler constructs an episode-local representation of the free action category in a finite causal state space. The late query is parsed into a path. A separate source-deleted core composes the corresponding morphisms.

```text
raw source tokens
    -> perceptual transformer
    -> fixed-shape episodic machine
         initial causal state
         unordered action records: (opaque key, transition map)
         unordered observer records: (opaque key, readout map)
    -> source deletion

late query tokens
    -> query parser using only retained opaque keys
    -> action path + observer selection
    -> explicit ordered transition composition
    -> answer
```

The source transformer is therefore a **compiler of a machine**, not the machine's implicit runtime.

### 4.1 Hard episodic machine

The first implementation should use an anonymous categorical Moore machine because it is exact, finite, auditable, and sufficient for the EPISODE family.

With maximum causal-state capacity `K`, maximum action records `M`, and maximum observer records `P`, the committed object is:

```text
initial_state        : categorical index in {0,...,K-1}
causal_state_active  : bit[K]

action_active        : bit[M]
action_key           : fixed-precision vector[M, d_key]
action_next          : categorical destination[M, K]

observer_active      : bit[P]
observer_key         : fixed-precision vector[P, d_key]
observer_answer      : categorical answer[P, K]

schema_version, masks, and seal receipt
```

`action_next[i,s]` is the destination of anonymous causal state `s` under action record `i`. It is the hard form of a row-one-hot transition matrix. `observer_answer[j,s]` is the answer emitted by observer record `j` at state `s`.

The machine contains no source token IDs, source positions, residuals, KV cache, query, answer for a particular future program, target label, trajectory, family ID, or assessor data.

### 4.2 Why categorical transition tables are not a cheat

A finite exact causal state is extensionally a finite transducer. The project should stop trying to claim otherwise. The scientific question is whether a small neural compiler can infer the correct episode-specific transducer from raw tokens and whether the explicit transition structure gives a resource and generalization advantage over a favorable generic recurrent model.

The categorical machine is acceptable only because:

- its state names are anonymous and episode-local;
- its action names are opaque and episode-local;
- the source compiler must construct it from raw tokens;
- the query appears after source deletion;
- ordered composition is performed by the compiled transitions;
- the query family is too large to reduce to the exposed answer pair; and
- the machine is independently intervenable at state, key, transition, and observer boundaries.

### 4.3 Separate action key and action transition

Each action record is a pair

```text
(opaque key k_i, transition map T_i).
```

The key answers **which query token denotes this action**. The transition answers **what the action does to causal state**.

This separation makes the two central causal interventions well-defined:

- **binding intervention:** permute or transplant keys while holding transitions fixed;
- **operator intervention:** permute or transplant transitions while holding keys fixed.

No fourth physical operator is applied to grammar tokens. Grammar tokens are processed only by the query parser. Only query spans that bind to an active action key enter the action path.

### 4.4 Source compiler

A provisional compiler uses the lower Shohin trunk as a perceptual encoder, followed by a new slot/set module with three types of anonymous records:

1. physical-state / transition-witness slots;
2. action records; and
3. observer/object records.

No gold slot identity is present at inference. The source compiler may use competitive attention, shared set-equivariant updates, and a fixed number of refinement rounds. It emits soft versions of the hard machine during training.

Two implementation paths are admissible:

**Direct machine hypernetwork**

The compiler directly emits soft initial-state, transition, and observer tables from raw source residuals. This is simplest and should be the first favorable baseline.

**Differentiable system-identification compiler**

The compiler first emits soft latent before/action/after witness assignments, then solves for one shared transition map per action record using a differentiable constrained least-squares or message-passing layer. This more strongly enforces reuse of one action law across all of its source witnesses.

The second path is the main treatment only if a CPU falsifier proves that the source contains enough witness structure to identify the required reachable machine without oracle segmentation.

### 4.5 Query parser

The query parser is a small separate transformer or recurrent pointer network. It receives only:

- late query tokens;
- retained `action_key` records;
- retained `observer_key` records; and
- fixed schema/masks.

It does **not** receive source tokens, transition tables as token context, the current answer, or assessor feedback.

It emits:

```text
action_path : categorical indices [0..L_max-1] plus STOP
observer     : one categorical observer index
```

The parser may compare query token features to retained opaque keys. It may not apply a transition to every query token. Grammar is converted into a path; the path is then executed.

### 4.6 Source-deleted executor

For hard state `s_0` and parsed action path `i_1...i_L`:

```text
s_(t+1) = action_next[i_t, s_t]
y       = observer_answer[j, s_L].
```

The same transition table is reused at every occurrence and every program depth. There are no timestep-specific parameters, host semantic branches, query-conditioned source replay, verifier repairs, or answer search.

A soft training analogue uses state distributions and row-stochastic transition matrices. The hard evaluator uses categorical indices only.

### 4.7 Attached training and detached scoring

Training and scoring require different graph boundaries but the same numerical function.

- **Training:** the soft episodic machine remains attached so gradients reach the compiler.
- **Scoring:** the hard fixed-shape machine is serialized, detached, process-sealed, and executed in a source-free process.
- **Shadow equivalence:** at frozen checkpoints, an attached hard forward and serialized detached forward must produce identical machine bytes and predictions under deterministic evaluation.

There is no proposal to backpropagate through `detach().clone()`. A detached scoring path is a custody boundary, not an optimization phase.

---

## 5. The Board Must Change: From Two Queries To A Separating Challenge Family

The current six-case orbit should be retained as a low-depth diagnostic. The advancement board must expose a much larger late-query family.

For each physical world, the source defines the same opaque action witnesses and initial world. After the source-derived machine is sealed, a challenge process samples:

```text
query = (action word w, observer q, renderer r)
```

where action words include repeated actions and varying lengths.

### 5.1 Minimum query family

For three active actions, the query universe should include:

- all words through a small exhaustive depth;
- random words at longer depths;
- repeated-action words;
- noncommuting order twins with identical action bags;
- equivalent words where the physical system has an independently verified relation;
- every observer/object query, not one selected query; and
- renderer and opaque-name recodings.

The challenge seed is generated or opened only after the compiler state seal exists. The compiler knows the query distribution but not the realized query.

### 5.2 Train/development separation

A provisional split should separate at least:

- source worlds and operator families;
- action-binding permutations;
- query words;
- maximum program depth;
- action multiplicity patterns;
- observer/rendering combinations; and
- composition motifs.

Training may use words up to depth `D_train`; development must include unseen words and depths up to at least `2*D_train`, subject to a finite mechanics proof that the causal state capacity is sufficient.

### 5.3 Resource separation

For maximum query depth `L`, three actions, and `P` observers, the raw answer table has

```text
P * sum_(ell=0)^L 3^ell
```

entries. The committed machine has approximately

```text
log2 K
+ M*K*log2 K
+ P*K*log2 |Y|
+ retained key bits
```

hard semantic bits.

The protocol must report both numbers. The machine need not be the theoretically smallest representation, but the advancement board should make the exposed answer-table strategy substantially more expensive than the committed algebraic machine.

This is a resource distinction, not a proof of a new computational primitive.

### 5.4 Official development custody

Mechanics and identifiability checks use training data plus a dedicated non-score mechanics split. Official development targets and intervention expectations remain unopened until architecture source, board source, thresholds, controls, state schema, parameter receipt, and all model seeds are frozen.

---

## 6. Training Objective Without Gold Latent Objects

Let the compiler produce a soft machine `M_hat(x)`. Let the query parser produce a soft path and observer selection. Let `Exec` be the differentiable categorical executor.

For sampled post-compile challenges `(w,q)`:

```text
L_behavior = CE(Exec(M_hat(x), w, q), y(x,w,q)).
```

Compile once and evaluate many independently sampled challenges from the same attached machine. There is one behavior loss, not duplicate diagonal and query-reuse losses over the same predictions.

### 6.1 Minimax residual pressure

Average loss allows a machine to sacrifice rare but separating futures. For each world, sample a fixed-size challenge panel `W_x` and use a frozen smooth worst-case objective:

```text
L_residual = tau * log sum_(u in W_x) exp(L_u / tau).
```

The temperature and panel generation are frozen before training. This does not prove complete residual identification, but it changes the objective from average answer fit toward worst-observed future sufficiency.

### 6.2 Orbit equivariance through behavior

Under a known action-name permutation `g`, do not force raw transition-table equality across independently compiled gauges. Instead require functional transport:

```text
Answer(x, w, q)
=
Answer(g.x, g.w, g.q)
```

for a fixed sampled panel of action words and observers. Internal tables may differ by anonymous-state permutation.

### 6.3 Source-witness consistency

If a source contains before/action/after evidence, the model-owned witness extractor and emitted transition table must agree on the selected latent states. This can be imposed without gold state IDs:

```text
p_after ~= p_before * T_action.
```

The witness extractor is part of the model and receives no gold segment or action class at evaluation. A shuffled-witness arm receives the identical architecture and loss with witness associations reassigned.

### 6.4 Hardening

Soft initial state, action binding, transition rows, observer rows, and query path are gradually hardened under a frozen entropy schedule. Final training uses a straight-through hard machine in the forward pass while retaining attached gradients.

### 6.5 No latent labels

The score-bearing treatment receives no:

- causal-state ID;
- action binding ID;
- transition table;
- observer table;
- query program parse;
- execution trajectory;
- terminal state;
- answer repair; or
- verifier feedback.

The only target-coupled signal is the answer to sampled late challenges, plus source-local self-consistency that can be computed from the raw source and model-selected witnesses.

---

## 7. Favorable Controls

Every learned control receives the same source worlds, late challenges, updates, precision, state-byte ceiling, query calls, and measured compute envelope.

1. **Frozen `80dc07a` workspace:** existing four-slot bind-select architecture, trained with its lawful attached interface.
2. **Generic recurrent machine:** same hard-state capacity and sequential depth, but no explicit action transition table or key/operator separation.
3. **Direct machine hypernetwork:** explicit categorical machine emitted directly, without witness-tied system identification.
4. **Fused action record:** key and transition are one inseparable vector; independent binding/operator interventions are unavailable by construction.
5. **Commutative action pool:** same action evidence and compute, but query actions are aggregated without order.
6. **Untied-depth executor:** equal or greater parameters, but separate transition parameters per program position.
7. **Answer-cache control:** fixed query-indexed storage under the same committed-byte limit; expected to pass the old two-query slice and fail the expanded challenge family.
8. **Shuffled-witness control:** same architecture and gradients with source transition associations permuted.
9. **Source-retained upper bound:** diagnostic only; never deployable.
10. **Oracle-machine ceiling:** receives the independently computed hard machine; tests query parsing, execution, and late reading only.

The treatment is scientifically interesting only if it beats the generic recurrent and direct-machine controls on unseen compositions while retaining the intervention signatures of an explicit compiled machine.

---

## 8. Required Interventions

The explicit state schema permits clean interventions that were undefined in the frozen architecture.

| Intervention | Changed field | Required effect |
|---|---|---|
| Binding/key permutation | `action_key` only | query action names follow new transitions |
| Operator permutation | `action_next` only | same names now execute donor dynamics |
| Compensated key+operator permutation | both, consistently | semantics unchanged |
| Initial-state transplant | `initial_state` only | all late challenges follow donor state |
| One transition-row transplant | one `action_next[i,s]` | only continuations reaching that row change |
| Observer-key permutation | `observer_key` only | query referents select different observers |
| Observer-map transplant | `observer_answer` only | terminal state unchanged, answer follows donor observer |
| Equivalent-word substitution | query path only | terminal state and answer unchanged |
| Noncommuting order reversal | query path only | terminal state changes on registered twins |
| Source poison after seal | unavailable source | zero machine/prediction change |
| Query before seal | custody violation | process abort |
| State reset each step | executor state | composition collapses |
| Transition-table shuffle | dynamics | accuracy falls to frozen ceiling |

Intervention targets are generated by an independent oracle after raw model outputs are sealed. The model process never receives the changed target.

---

## 9. CPU Falsifiers Before Neural Source Freeze

A neural implementation is unauthorized until all of the following pass.

### 9.1 Old-board cache falsifier

Implement the two-answer cache from Theorem 1 and show that it passes every old query-blind/compile-once/query-swap gate that does not explicitly intervene on a real machine object. This formally records why the old board cannot carry the new claim.

### 9.2 Causal-quotient and capacity audit

For every generated small world:

- independently enumerate reachable physical states;
- compute the exact future-indistinguishability quotient;
- compute its transition monoid and observer maps;
- verify the chosen `K` covers the quotient;
- verify the query family separates every pair of quotient states; and
- reject worlds whose semantics are not identifiable from the admitted source evidence.

### 9.3 Machine execution audit

Exhaustively compare two independent implementations of:

- hard action composition;
- STOP/path termination;
- observer readout;
- key and operator interventions;
- state transplants;
- equivalent and noncommuting words; and
- source deletion.

### 9.4 Query-parser nontriviality

Show that action-bag, length, renderer, observer-frequency, and grammar-only controls remain below frozen ceilings. Include repeated action names and action words whose correct answer changes under order but not bag.

### 9.5 State-schema audit

The sealed object must be fixed shape and fixed byte length. Every semantic field, precision, active mask, and maximum cardinality is counted. No variable-length hidden channel, diagnostic tensor, or source-derived metadata may cross the boundary.

---

## 10. Provisional Architecture Budget

The first implementation should target at most 16,000,000 new unique parameters:

| Component | Provisional ceiling |
|---|---:|
| source slot/set compiler | 9,000,000 |
| machine heads / optional system-identification layer | 3,000,000 |
| late query parser | 2,500,000 |
| readout / integration / contingency | 1,500,000 |
| **Added ceiling** | **16,000,000** |
| Protected base | **125,081,664** |
| **Provisional complete ceiling** | **141,081,664** |

This is a design ceiling, not an instantiated receipt.

A first hard schema with `K=32`, `M=4` including one inactive padding record, and a small observer bank can fit in well below 16 KiB excluding the seal receipt and fixed vector keys. Exact bytes and the answer-table comparison must be frozen after implementation and before board generation.

If mechanics require `K>32`, the board and state budget must be revised before any score-bearing seed. A post-score capacity increase is forbidden.

---

## 11. Advancement Gates

Exact thresholds require an admitted board, but the following gate classes are mandatory.

### 11.1 Behavioral transfer

Every seed must pass:

- high clean exactness on unseen worlds, action bindings, and renderers;
- high exactness on action words absent from training;
- high exactness at unseen program lengths;
- minimum accuracy for every action multiplicity, observer, operator family, and composition motif; and
- all registered noncommuting order twins.

### 11.2 Machine causality

Every seed must pass:

- binding/key interventions in the independently predicted direction;
- operator interventions in the independently predicted direction;
- compensated key/operator invariance;
- initial-state and local transition-row transplants;
- observer-key and observer-map separation;
- state-reset collapse;
- transition-table shuffle collapse; and
- source-poison and late-query custody invariance.

### 11.3 Reuse and challenge timing

- one machine compilation per world;
- multiple challenge queries executed from the identical sealed machine;
- challenge seed or query bytes unavailable before seal;
- no recompilation after query disclosure;
- fixed machine bytes across all late queries; and
- query parser and executor run without source files, source descriptors, residuals, or KV state.

### 11.4 Controls

- treatment must beat the generic recurrent, direct-machine, commutative, untied-depth, and answer-cache controls by frozen margins;
- the generic recurrent control must be qualified by strong training and in-distribution fit;
- shuffled witnesses must not preserve treatment performance; and
- all seeds pass individually; mean performance cannot rescue one failed seed.

A pass supports only a bounded claim that Shohin compiled and executed an episode-local causal machine from raw tokens under a large post-seal query family. It does not establish unrestricted language reasoning.

---

## 12. Experimental Sequence

1. Mark the previous OCSI draft `NO-GO AS WRITTEN`; retain its useful custody and orbit-control ideas.
2. Implement the old-board two-answer cache and add it to the permanent control ledger.
3. Specify the enlarged continuation-query family and prove separation on exhaustive small worlds.
4. Implement the categorical machine runtime and intervention suite independently of the neural compiler.
5. Implement the three-process compiler/query/assessor custody boundary around the explicit machine object.
6. Implement and compare the direct-machine hypernetwork and the witness-tied system-identification compiler.
7. Instantiate the exact parameter and state-byte receipt below 200M.
8. Freeze source before generating a fresh board. Gate Zero uses training plus a dedicated mechanics split, never official development.
9. Train every matched arm from fresh recorded seeds with attached hard-machine gradients.
10. Verify attached-versus-serialized numerical equivalence at fixed checkpoints.
11. Seal all checkpoints and consume one development access.
12. Only a full all-seed pass may authorize one separately generated sealed confirmation.

---

## 13. Research Claim

The architecture is deliberately controversial in one respect: it stops asking an ordinary transformer residual stream to spontaneously become a world model, parser, action binder, executor, and answer channel at once.

It instead imposes a sharp computational contract:

> The transformer must compile raw evidence into an episode-local representation of states, actions, and observations. Once the source disappears, reasoning is ordered composition inside that compiled representation.

This is not a claim that finite-state machines are new. They are not. The possible breakthrough is a learnability and systems result: a sub-200M language model may become substantially more systematic when it is forced to emit and causally use an episode-specific algebra rather than burying every stage in one tokenwise residual computation.

The candidate should be killed if the explicit machine does not beat a qualified generic recurrent control, if the enlarged query family remains cacheable under the counted state budget, if the source evidence does not identify the reachable causal quotient, or if the key/transition/state interventions are inert.
