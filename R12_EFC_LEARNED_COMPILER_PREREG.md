# R12 Learned EPISODE Functor Compiler Preregistration

**Status:** architecture and mechanics preregistration draft. The train-only
implementation is connected read-only to the protected Shohin checkpoint, but
neural fitting, GPU work, official development generation, reasoning
promotion, and continuation pretraining remain unauthorized.

**Pretraining hold:** absolute. This experiment does not start, prepare,
modify, queue, or resume Shohin pretraining.

## 1. Bounded Hypothesis

Shohin can use a learned perceptual compiler to turn raw episode evidence into
one fixed-size anonymous finite machine. After the source and compiler state
are deleted, a separately learned query parser can bind post-seal opaque keys
and a fixed generic executor can reuse the machine for ordered compositions
that were not presented during compilation.

A pass supports only:

> From raw, uniquely identifying source evidence, Shohin compiled an unseen
> finite world into a sealed episode-local machine and reused it to execute
> unseen post-seal action compositions.

It does not establish unrestricted language reasoning, theorem proving, or
general intelligence.

## 2. Exact Deployed Object

The score-bearing object is the existing 1,536-byte C/Rust EFC wire:

- maximum 16 anonymous states;
- maximum 8 action records;
- maximum 8 observer records;
- exact copied `uint64` state, action, and observer keys;
- one categorical transition destination per action/state cell;
- one `uint64` answer per observer/state cell;
- prefix-canonical active masks, zero padding, version header, and SHA-256;
- no learned or behaviorally active initial state in the machine; deployed byte
  56 is a legacy validated field fixed to zero and counted in the 1,536 bytes;
- the post-seal query supplies its opaque start-state key;
- no source token IDs, positions, residuals, KV state, targets, trajectories,
  challenge coordinates, family IDs, renderer IDs, or assessor data.

The attached soft runtime may carry gradients during optimization. Scoring
uses only a detached hard machine and a query that arrives after that machine
has been sealed. The production system API accepts only `(sealed_machine,
late_query)`; it cannot receive source bytes and late-query bytes in one call.
The late query resolves exact opaque bytes directly against the hard copied-key
table. It never receives a soft source assignment tensor. The attached global
straight-through key transport and detached hard assignment must select the
same one-to-one key/axis permutation.

The old `WireProtocolSpec` requires an identity observer and therefore rejects
`K=8, |Y|=4`. It remains frozen for the old custody rehearsal. The learned
board uses a separate dimension validator without that identity-observer
assumption; the C/Rust binary layout is unchanged.

## 3. Two Scientific Stages

### EFC-C: compiler qualification

EFC-C may use gold key-span, witness-assignment, transition-cell, and
observer-cell supervision on training data. Public mechanics is evaluation
only for every learned arm. Its purpose
is to establish that the architecture can parse, copy, canonicalize, harden,
serialize, and execute the machine. EFC-C cannot be promoted as reasoning.

EFC-C labels are built only in
`pipeline/episode_functor_qualification_supervisor.py`. The candidate receives
the exact `CandidateSource(source: bytes)` projection through
`pipeline/episode_functor_qualification_boundary.py`; source SHA-256 is the
only join key. The candidate batch has no label, family, split, renderer,
machine, transition, observer, target, or execution field. The supervisor
object is rejected by the compiler forward API and is not serialized with a
sealed machine.

The frozen EFC-C objective is implemented in
`train/episode_functor_qualification_loss.py`. It joins labels only after the
candidate forward by exact source SHA-256 and reports whole-row key, record,
role, and answer exactness; every transition/observer cell; the one hidden
cell per relation; and exact complete machines. The optimizer boundary in
`train/episode_functor_qualification_trainer.py` updates only the source
compiler, requires a cryptographically verified protected trunk by default,
fails on nonfinite gradients or parameters, and materializes an exact
optimizer-state byte receipt. It does not generate data, open a split, write an
artifact, or launch a job.

### EFC-I: identification treatment

EFC-I receives no causal-state IDs, action IDs, transition table, observer
table, query parse, execution trajectory, terminal state, answer repair, or
verifier feedback. Its target-coupled signal is:

1. answers to independently sampled post-compile challenges; and
2. source-local witness consistency computed from raw evidence and
   model-selected witnesses.

EFC-I compiles once and answers many challenges from the same attached
machine. Final scoring hardens, serializes, deletes source/compiler state, and
opens a later challenge.

## 4. Primary Identifiable Source

Complete JSON, line-event, and cycle-program sources remain oracle/mechanics
ceilings. They expose the machine almost directly and cannot carry a reasoning
claim.

The primary source uses `K=8`, `M=3`, `P=2`, and four answer symbols:

- each action is a permutation of eight states;
- exactly seven of eight action cells are exposed;
- the missing source cell and destination are uniquely implied by permutation
  totality;
- each observer is balanced, with each answer occurring exactly twice;
- exactly seven of eight observer cells are exposed;
- the missing observer answer is uniquely implied by the balance constraint;
- no family or renderer identifier is present.

For every admitted source `x`, two separately written solvers must establish:

```text
V(x) = {machines satisfying all public source constraints}
|V(x)| = 1
```

The more general admissibility condition is behavioral uniqueness:

```text
for every M1, M2 in V(x) and every legal challenge c:
    Answer(M1, c) = Answer(M2, c)
```

Any behaviorally distinct pair is an immediate no-go.

## 5. Renderer Metagrammar

Renderer transfer is compositional, not arbitrary-language generalization.
Source and query grammars each factor into a frozen binary cube:

1. record framing;
2. field organization; and
3. numeric/lexical codec.

Training sees every primitive factor but only four even-parity combinations.
Development and confirmation use held-out combinations. Any lexical codec
whose meaning is not mechanically derivable includes an in-source legend.
No renderer ID crosses the candidate boundary.

Semantically equivalent renderer variants remain inside one split. Worlds are
split by a canonical form invariant to state, action, observer, and answer
renaming.

## 6. World Families And Splits

The intended full board uses algebraically distinct eight-state action
families with matched source geometry:

- random transitive subgroups of `S8`;
- affine actions on `F_2^3`;
- dihedral actions on eight vertices;
- regular `D4` actions;
- regular quaternion `Q8` actions; and
- cube-rotation actions.

Every admitted action triple is noncommuting, every state is reachable, and
the observer/action continuation family separates the exact causal quotient.

Provisional full scale:

| Split | Worlds | Family scope | Query depth |
|---|---:|---|---:|
| train | 24,576 | F0-F2 | 0-4 |
| public mechanics | 384 | adversarial microfixtures | exhaustive 0-4 |
| development | 1,792 | F0-F3 | 5-8 |
| confirmation | 3,072 | F0-F5 | 9-12 |

A small deterministic pilot may validate mechanics only. Pilot scores do not
authorize promotion or threshold changes.

## 7. Challenge Family And Resource Bound

Training exhausts all words through depth four. Development includes all
depth-five words and fixed target-independent samples at depths six through
eight. Confirmation samples depths nine through twelve. Every word is queried
from all eight starts through both observers.

At confirmation scale, a raw answer table requires approximately 4,096 bytes
for the sampled panel and 3,188,644 bytes for the complete depth-zero-through-
twelve support. The complete deployed machine budget is exactly 1,536 bytes.
Every persistent source-dependent bit, including copied keys and metadata, is
counted. A cache control receives exactly the same byte ceiling.

Panels include repeated actions, alternations, all-action words, same-bag order
twins, equivalent words, and changed-order twins. Syntactic panel selection
cannot inspect world answers.

## 8. Candidate Architecture

The first implementation is a proof-carrying transport witness compiler:

1. a generic bounded numeral-span copier exposes exact key bits without
   assigning semantic roles;
2. hash-bound frozen Shohin residuals from blocks 9, 19, and 29 plus a
   byte/record encoder produce source memory;
3. set-equivariant role slots select distinct state/action/observer keys;
4. witness slots select transition and observation records;
5. one global one-to-one key transport causally controls both copied keys and
   transition/observer axes;
6. a zero-parameter Birkhoff/balanced-transport layer projects witness
   evidence into one lawful anonymous soft machine;
7. a separate query parser resolves post-seal start/action/observer keys
   directly against the sealed hard key table;
8. the fixed categorical executor performs ordered composition; and
9. a matched no-host completion arm replaces the public-law projector with a
   shared permutation-equivariant relational network. It can aggregate row,
   column, and global evidence, but it is not forced to emit a permutation or
   balanced observer and must learn completion from train-only labels.

The compiler never receives a late query. The query parser never receives
source tokens, transition tables as token context, targets, current answers,
soft source assignments, or assessor feedback.

The implementation ceiling is 200,000,000 total parameters including immutable
Shohin. The protected checkpoint was loaded read-only under SHA-256
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`.
The instantiated receipts are:

| Component | Solver arm | No-host arm |
|---|---:|---:|
| Frozen Shohin | 125,081,664 | 125,081,664 |
| Source compiler | 3,595,792 | 3,821,202 |
| Learned completion submodule | 0 | 225,410 |
| Late-query parser | 728,993 | 728,993 |
| Added trainable total | 4,324,785 | 4,550,195 |
| Complete connected system | 129,406,449 | 129,631,859 |
| Headroom under 200M | 70,593,551 | 70,368,141 |

The minimal no-host arm is an attribution probe, not a presumption that
4.55 million added parameters are sufficient. The remaining capacity is
available for mechanisms that preserve the same source/query boundary and
causal controls. Exact constructor counts for two preregisterable escalation
lanes are:

| Capacity lane | Architecture | Added parameters | Complete system | Remaining headroom |
|---|---|---:|---:|---:|
| Wide | 384-wide 8+4-layer compiler; 512-wide 8-round completer; 256-wide 4-layer query parser | 35,625,267 | 160,706,931 | 39,293,069 |
| Maximum prereg candidate | 512-wide 8+4-layer compiler; 640-wide 8-round completer; 320-wide 4-layer query parser | 60,552,883 | 185,634,547 | 14,365,453 |

These immutable profiles and constructor-checked receipts are implemented in
`train/episode_functor_capacity_lanes.py`. They are architecture receipts, not
fitted systems or capability claims.
Scale is admitted only after matched controls identify undercapacity rather
than a broken interface. A learned executor, recurrent relational compiler,
typed memory, or changes to normally fixed transformer components must be
introduced as separately named arms so that their causal contribution remains
measurable; they must never be smuggled into an existing result after scoring.

### 8.1 Structural escalation arms

The capacity ceiling is a mechanism budget, not a width target. Two staged
treatments are preregistered for implementation only after the current
mechanics and custody gates close.

**Hankel-shift causal code (HSC).** Replace the generic relational completer in
the maximum lane with a behavioral code. For every anonymous state `s`, learn
a depth-three predictive signature

```text
Sigma(s)[w, q, y] = P(observer q returns y after action word w from s)
```

for all 40 words through depth three, two observers, and four answers. A
separate derivative branch predicts the left-shifted signature for every
action. Transition `a(s)=t` is decoded by matching the derivative code of
`(a,s)` to the base code of `t`; the empty-word coordinate supplies observer
readout. The shift disagreement is an explicit syndrome that can localize an
inconsistent proposed transition. State identity is therefore tied to
distinguishable futures rather than a learned coordinate label.

HSC remains inside the existing maximum receipt: no more than 60,552,883
added parameters, 185,634,547 total, and 14,365,453 headroom. The transient
signature has 10,240 logits per source, but only the ordinary 1,536-byte hard
machine persists after sealing. It does not hardcode permutation or observer
balance. Required controls are the parameter-matched current completer, a
direct transition hypernetwork, random word/incidence correspondence,
commutative word bags, depth-zero signatures, shuffled signature labels, and
an oracle-signature ceiling. Kill HSC if it fails exact recoding equivariance,
does not improve exact-machine accuracy over the strongest matched control,
or loses its gain when the true left-shift incidence is randomized.

**Sealed predictive sheaf compiler (SPSC).** This is a later architecture
treatment for the failure mode where a one-shot local parse cannot revise a
globally inconsistent machine. It adds source-only rank-96 adapters to the
normally frozen Shohin blocks, typed factor memory for opaque roles and
machine cells, bidirectional residual/factor bridges, and three recurrent
bind/propose/predict/revise cycles. A target-independent closure bank executes
all words through depth three on each provisional machine and returns only
composition contradictions to the compiler. The final seal and late-query
path remain unchanged and source-deleted.

The current estimate is at most 197,035,539 total parameters:
29,306,880 source adapters, 9,338,880 residual/factor bridges, 26,257,920
reused predictive blocks, at most 2.5M typed heads, and the existing minimal
system. Exact constructor accounting is mandatory before this estimate can
become a receipt. Because SPSC adapts the parent computation, it cannot use
the current `connected` frozen-trunk claim. It requires a separately named
`adapted_base` treatment receipt proving the protected checkpoint tensors are
unchanged while accounting for every adapter, routing decision, update, byte,
and FLOP.

SPSC controls must include an isomorphic open-loop arm whose contradiction
signals cannot affect machine logits, a scrambled-composition arm, an
adapter-only one-shot compiler, and the maximum HSC/current-completer arms.
Kill SPSC if successive revision cycles do not monotonically improve
held-out exact-machine accuracy, inference-time feedback ablation costs less
than five points, or the full treatment fails to beat both open-loop and
scrambled controls by a preregistered paired margin.

Neither proposal is claimed to be literature-novel. Predictive-state
representations, behavioral Hankel matrices, error-correcting codes,
predictive coding, factor graphs, recurrence, and adapters all have
precedents. The falsifiable contribution is their use in a recoding-equivariant
source-only compiler that is irreversibly reduced to a fixed source-deleted
machine. Advancement depends on causal results, not terminology.

The standalone compiler receipt remains explicitly
`integration_status=not_connected`; only the checkpoint-backed wrapper may
report `integration_status=connected`. A same-sized GPT passed through the
public trunk constructor also remains `not_connected`. Connected status is not
granted by a replayable Python sentinel: receipt generation re-hashes the
checkpoint file; compares its configuration and every checkpoint tensor with
the frozen in-memory parent; compares nonpersistent runtime buffers and the
exact module graph against a fresh `model.py` construction; rejects hooks,
compiled-call overrides, instance method replacement, and changed runtime
attributes; and compares the executing Python code manifest with a fresh load
of the bound source and fixed clean-runtime manifests. The manifests bind
function code, defaults, keyword defaults, recursive function-valued closures,
annotations, function attributes, referenced globals and builtins, selected
external inference/dispatch methods, model properties, ordered module
topology, trunk execution/configuration, transport dispatch, and published
feature width. The protected SHA-256, source semantics, runtime state, and
parameter count must all match. Parameter, RoPE-buffer, hook, topology,
class-method, property, method-default, transport, builtin, and
referenced-callable mutations each invalidate verification. This is a
reproducible Python execution receipt, not malicious-host, native-kernel, or
hardware attestation; official fitting still requires process/runtime custody.

Shohin's 2,048-token context is not silently exceeded. The longest current
source renderer is 2,420 Shohin tokens and 440/888 sources exceed the parent
context. Frozen residual extraction therefore uses deterministic disconnected
contiguous windows of at most 2,048 tokens, resetting attention and RoPE at
each window; it is only a local frozen perceptual feature source. The trainable
byte compiler carries global source context. Every one of the 888 pilot sources
has exact, nonoverlapping tokenizer-offset coverage of every source byte.

### Solver-augmented claim boundary

The Birkhoff/balanced-transport projector knows the public
`K=8, M=3, P=2, Y=4` permutation and observer-balance laws. It is therefore an
explicit solver-augmented architecture arm. A success may support learned
source parsing, binding, machine construction, and post-seal composition, but
cannot by itself establish that Shohin learned permutation/balance completion.
A matched learned-completion/no-host-projector arm is mandatory before any
claim about learned deductive completion. General reasoning promotion further
requires transfer to different law families rather than only unseen worlds
under this fixed public law.

The no-host arm in
`train/episode_functor_learned_completion.py` is the matched attribution
treatment. Its shared cell updater is equivariant to state recoding and has no
state, action, observer, or answer coordinate embeddings. A zeroed instance
emits tied soft tables whose coordinate-first argmax would be invalid, proving
that permutation and balance laws are not silently imposed. Such exact
categorical ties are rejected before straight-through or detached hardening
because no deterministic one-hot tie break can be recoding-equivariant. Under
unique maxima, hard state, action, observer, and answer recodings are exact.
Its key transport remains the same one-to-one copied-key mechanism as the
solver arm and is causally separate from relation completion.

## 9. Optimization

EFC-C may use key pointer, witness assignment, and machine-field losses.
EFC-I uses:

```text
L = L_behavior
  + 0.25 * L_smooth_worst_world
  + 0.50 * L_renderer_orbit
  + 0.50 * L_source_witness_consistency
  + 0.25 * L_intervention
  + 0.10 * L_hardening
```

The worst-world term is a frozen-temperature log-sum-exp over a fixed
per-world challenge panel. Hardening uses a frozen schedule and ends with a
straight-through hard forward. No post-score schedule changes are allowed.

## 10. Matched Controls

Every learned arm receives the same worlds, challenge labels, update budget,
precision, parameter ceiling, compiler time, challenge calls, and persistent
byte ceiling:

1. frozen four-slot workspace;
2. qualified generic recurrent machine;
3. direct-machine hypernetwork;
4. fused key/operator records;
5. commutative action pool;
6. untied-depth executor;
7. exact-byte answer cache;
8. shuffled-witness treatment;
9. source-retained diagnostic ceiling;
10. oracle source normalizer;
11. oracle query parser; and
12. oracle machine.

The generic recurrent control is qualified only after at least 98% train and
95% in-distribution exactness.

The resource vector is reported in full:

```text
examples, target bits, source bytes, oracle calls, updates, parameters,
optimizer bytes, compiler FLOPs/time, persistent bytes, executor FLOPs/query
```

`pipeline/episode_functor_resource_receipt.py` enforces this schema,
distinguishes forecast from measured values, binds board/source/config hashes,
and fails closed on unknown fields or inconsistent bounds. No arm-specific
receipt is frozen yet because update count, optimizer precision, and measured
compiler cost remain unset. This is an explicit fit no-go, not permission to
fill those values after seeing a score.

## 11. Causal Interventions

Required interventions include:

- key-only action permutation;
- transition-only action permutation;
- compensated key/transition permutation;
- state conjugation;
- start-state transplant;
- one transition-cell transplant with locality accounting;
- observer-key permutation;
- observer-map transplant;
- equivalent-word substitution;
- noncommuting order reversal;
- source poison after seal;
- state reset each step; and
- transition-table shuffle.

Intervention targets are independently generated only after machine and
prediction seals. No changed target enters the candidate process.

## 12. Frozen Advancement Gates

- independent oracle compiler, parser, C runtime, Rust runtime, and assessor:
  exactly 100%;
- candidate transition and observer cells: at least 99.5%;
- exact semantic machines: at least 95% overall and 14/16 in every factorial
  cell;
- end-to-end exactness: at least 98% per family/renderer/depth aggregate and
  at least 95% in every cell;
- exact complete word blocks: at least 90%;
- opaque recoding and renderer transport: at least 99%;
- compensated gauge interventions and source-poison invariance: exactly 100%;
- other eligible interventions: at least 99%;
- treatment exceeds the strongest qualified recurrent control by ten points,
  with paired 99% confidence lower bound above five points;
- five optimizer seeds pass individually;
- confirmation repeats over three domain-separated board seeds.

No averaging rescues a failed seed or cell.

## 13. Immediate Kill Conditions

Stop before neural fitting for:

- source ambiguity or a behaviorally distinct version-space pair;
- visible split, family, or renderer IDs;
- unequal information presented as renderer transfer;
- semantic-orbit overlap across splits;
- query-dependent world admission;
- challenge bytes available before machine seal;
- uncounted parser or key state;
- source access after sealing;
- machine mutation or recompilation between queries;
- nonzero wire padding or attached/detached byte drift;
- oracle ceiling below 100%;
- cache control above 30%;
- unqualified strongest control;
- any family-specific executor branch; or
- a failed frozen seed/cell gate.

## 14. Authorization Sequence

1. Implement and test the metagrammar, unique-completion source, two
   independent version-space solvers, canonical split form, and resource audit.
2. Qualify the exact soft/hard machine boundary and deployed-wire adapter.
3. Implement EFC-C and matched compiler controls on train/public mechanics
   data only.
4. Implement EFC-I without gold machine fields.
5. Freeze source, schemas, thresholds, seeds, parameter/resource receipts, and
   custody launchers.
6. Generate development worlds from a future public beacon.
7. Compile once, publish machine roots, then open a later challenge beacon.
8. Permit one development read.
9. Only an all-seed pass authorizes unchanged confirmation evaluation.

No step authorizes Shohin continuation pretraining. Only the user may lift the
pretraining hold.
