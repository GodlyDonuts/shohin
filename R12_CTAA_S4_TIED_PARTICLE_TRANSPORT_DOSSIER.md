# R12 CTAA S4-Tied Particle Transport Development Dossier

## Status

**Component mechanics pass after adversarial repair. Neural source is not
complete, preregistered, frozen, or authorized.**

This document records retrospective development evidence and drafts the
requirements for a possible successor to static CTAA binding completion. It is
not itself a preregistration. It does not authorize a production board seed,
training seed, development read, confirmation read, GPU job, or
native-reasoning claim.

The project-wide authority is now **strictly below 200,000,000 unique
parameters**. Closed 150M experiments remain closed under their original
contracts. This successor may use the larger ceiling only after a smaller
mechanism-matched pilot establishes a causal advantage.

## 1. Motivation From Existing Evidence

Shohin's strongest repeated pattern is:

1. local fields and operations can be learned;
2. source-visible pointers and binding can become nearly exact;
3. fixed or host-side executors can compose an exact packet;
4. autonomous composition, binding transport, and consumption remain the
   failure boundary.

The prior ER dual-stream route reached 76.074% exact fresh packets, 85.400%
state, and 90.527% answer while its fixed tensor executor was exact conditional
on an exact packet. That is valuable compiler evidence, but not native
reasoning. ACW accumulated strong deterministic-custody machinery but no
scored capability result. The causal carry recovery lane is an unrun
preregistration. None should be merged wholesale into CTAA.

The reusable facts are narrower:

- occurrence coordinates and nominal identity must remain distinct;
- local opcode coordinates and physical action cards require a causal binding;
- the source must be destroyed before execution and late query;
- execution must occur inside the model forward path, not in a verifier;
- a useful architecture must preserve composition order, not merely the
  multiset of observed cues.

Static A4-to-odd binding completion tests the second point. It cannot establish
dynamic rebinding because one binding is frozen for an entire program.

## 2. Hypothesis

A tiny workspace can preserve composition order without a textual scratchpad
if its hidden state is a probability distribution over a **non-abelian group**
and rebinding cues update that state by group convolution.

For CTAA the relevant group is `S4`, the 24 permutations mapping four local
opcodes to four physical action cards. Let

`p_t(g)` be the workspace probability assigned to binding `g in S4`.

The source compiler emits pair logits `L[i,j]` for opcode `i` and card `j`.
The initial binding state is

`p_0(g) = softmax_g sum_i L[i, g(i)]`.

For cue `c`, a learned kernel `K_c(delta)` updates the state:

`p_(t+1)(g) = sum_h p_t(h) K_c(h^-1 g)`.

This is right convolution in the group algebra of `S4`. Every binding particle
then executes the same learned CTAA transition core. A late query reads the
posterior-weighted final categorical state. The source bytes, trunk residuals,
and source KV are unavailable after `p_0`, action cards, initial state, opcode
tape, and cue tape are committed.

The historical working name was **Non-Abelian Holonomy Workspace (NAHW)**.
Because the complete finite state is ordinary operator recurrence, the
scientifically accurate name is **S4-Tied Particle Transport (S4-TPT)**.
Two cue sequences with the same cue multiset can end at different binding
states because their ordered group products differ; this is useful structure,
not a new reasoning primitive.

## 3. Controls And No-Go Boundary

The decisive control replaces `S4` convolution with circular convolution over
the abelian group `Z24`:

`q_(t+1)(j) = sum_i q_t(i) K_c((j - i) mod 24)`.

Treatment and this mechanistic ablation have:

- 24 particles;
- the same initial pairwise binding readout;
- one learned 24-value kernel per cue;
- identical trainable parameter count;
- identical 24 x 24 transport MACs per cue;
- the same optimizer, examples, update count, and late reader.

The difference is only the multiplication table. This makes `Z24` a precise
mechanistic ablation, **not** the decisive favorable control: it is
architecturally forbidden from retaining noncommuting order.

The decisive favorable control assigns one unconstrained learned `24 x 24`
row-stochastic transition matrix to every cue. It uses the same 24-particle
state and the same 576 transport MACs per cue, but 3,456 transport parameters
instead of 144. Every positive `S4` convolution kernel embeds exactly into
this dense control by tying matrix entries with the `h^-1 g` index. Its
hypothesis class therefore contains the treatment. NAHW must beat this stronger
control on held-out compositions; beating `Z24` alone is insufficient.

### Finite separation theorem

For any two `Z24` kernels `A` and `B`,

`(q * A) * B = (q * B) * A`

because circular convolution is commutative. Therefore the control cannot
distinguish cue orders `AB` and `BA` when the cue multiset is fixed.

For `S4`, choose two noncommuting transpositions `a` and `b`. Delta kernels at
those elements yield final particles `ab` and `ba`, which differ. Thus NAHW
can represent an order-dependent binding distinction that the equal-resource
abelian control cannot represent for any parameter values.

This is a resource-preserving separation from the abelian ablation, not from a
generic recurrence. `R12_HOLONOMY_STATE_NO_GO.md` already proves that complete
finite holonomy state reduces to ordinary operator recurrence or PSR/OOM
machinery. NAHW is therefore rejected as an R12 invention or fundamentally new
reasoning primitive. The surviving empirical hypothesis is narrower:
non-abelian parameter tying may improve sample efficiency and systematic
composition relative to the stronger dense operator control.

## 4. Implemented CPU Mechanics

Implemented source:

- `train/ctaa_s4_particle_transport.py`
- `train/test_ctaa_s4_particle_transport.py`
- `pipeline/ctaa_s4_transport_mechanics.py`
- `pipeline/test_ctaa_s4_transport_mechanics.py`
- `pipeline/ctaa_s4_transport_development.py`
- `pipeline/test_ctaa_s4_transport_development.py`

The repaired deterministic component audit passes:

| Audit | Result |
|---|---:|
| `S4` elements | 24 |
| Inverse checks | 24/24 |
| Independent composition-oracle checks | 576/576 |
| Associativity checks | 13,824/13,824 |
| Ordered transposition-cue pairs | 36 |
| Noncommuting treatment pairs | 24 |
| `Z24` order collapses | 36/36 |
| Opcode/card coordinate round trips | 13,824/13,824 |
| Transport-equivariance checks with conjugated cues | 82,944/82,944 |
| Interleaved binding/state/action/opcode cases | 69,984/69,984 |
| One-step state plus binding checks | 139,968/139,968 |
| One-step probability-mass checks | 69,984/69,984 |
| CTAA action maps admitted | 27/27 |
| Post-STOP, mixed-mass, and gradient gates | 4/4 |
| Focused tests | 22/22 |

Deterministic report payload SHA-256:

`6152538ad3118d254da296ebcb978a5f40b8798885eb22a84392a35f45a6fd93`

The report decision is
`record_component_mechanics_only_no_neural_authorization`.

Adversarial review invalidated the original “complete coordinate
equivariance” claim: invertible particle reindexing alone did not prove that
transport commuted with it. Under opcode reindexing, each right-acting cue must
be conjugated by the opcode permutation. The repaired audit exhausts all
`24 x 24 x 24 x 6 = 82,944` binding/opcode/card/generator cases. The fixed
multiplication tables are now non-persistent buffers, so loading matched
learned weights cannot silently overwrite `Z24` with `S4`. Empty cue sequences
are accepted.

The component now also includes a differentiable interleaved event executor.
It carries a joint distribution over 24 bindings and all 27 categorical
three-register states. Cue events transport binding mass, action events update
physical state conditional on the current binding, STOP latches the joint
state, and a late categorical query reads one register. This fixes the earlier
all-cues-before-all-actions error. It still receives hard particle, card,
event, and query tensors; it is not a byte-source compiler or a source-deleted
Shohin system.

### Retrospective source-free transition-law canary

The second CPU gate gives every matched-data arm exactly six supervised
transitions: one from the identity particle for each transposition cue. Each
arm fits all six examples, then composes every unseen cue word at depths two,
three, and four. The dense data-rich ceiling instead receives all `24 x 6 =
144` one-step transitions. It is not a matched arm; it proves that the dense
control has sufficient capacity and can be optimized when its untied rows are
identified.

Across five fixed seeds:

| Arm | Supervised transitions | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| `S4` tied treatment | 6 | 36/36 | 216/216 | 1,296/1,296 |
| `Z24` abelian ablation | 6 | 8/36 | 9/216 | 47/1,296 |
| Dense favorable control | 6 | 0--3/36 | 36/216 | 0/1,296 |
| Dense data-rich ceiling | 144 | 36/36 | 216/216 | 1,296/1,296 |

Every row is identical across seeds except the dense six-example depth-two
range shown above. All arms fit their own supervision exactly. The treatment's
100% result follows from learning six cue kernels while the frozen `S4`
multiplication table ties all unobserved source rows. The dense control contains
that solution but the six labels do not identify its other 23 rows. This is a
taut but valid hardcoded-prior sample-efficiency signature. The canary was
implemented before this dossier was committed and is retrospective development
evidence, not a preregistered advancement gate. It is not evidence that Shohin
representations, language grounding, source deletion, late query, or autonomous
reasoning work.

The canary decision is
`record_retrospective_parameter_tying_signature_only`. It uses an independent
composition oracle for all 576 pair products. Deterministic report payload
SHA-256:

`2f07fbd9e7b5a656b24a397f50e17cd2f80a937b0926036d0cfc337f6741d3c4`

## 5. Parameter And Compute Ledger

| Component | Parameters |
|---|---:|
| Frozen Shohin + qualified CTAA compiler + transition core | 137,989,944 |
| Shared bi-equivariant pair readout | 599,353 |
| Six learned 24-value cue kernels | 144 |
| **Complete NAHW pilot** | **138,589,441** |
| Dense favorable-control transport | 3,456 |
| **Complete dense favorable control** | **138,592,753** |
| Strict ceiling | 199,999,999 |
| **Headroom** | **61,410,558** |

Each cue uses 576 matrix-vector transport MACs in all arms, and the pair
readout uses 9,587,136 analytic dense MACs. This is not compute parity: the
group arms normalize one 24-value kernel while the dense arm normalizes 24
rows. Measured forward/backward/runtime/memory receipts are mandatory before
any comparison. The complete-system totals above add the workspace ledger to
the last verified CTAA base; they are provisional arithmetic, not an
instantiated deduplicated-model receipt.

The unused parameter budget is deliberate. If NAHW fails against the matched
control, widening it is not an admitted repair. If the transport mechanism
passes but source compilation is the localized bottleneck, up to roughly 61M
parameters may be allocated to a renderer-invariant object-file compiler under
a separately frozen factorial.

## 6. Neural Board Blockers And Draft Design

**No neural board is authorized.** The implemented component starts from hard
particle probabilities, card tensors, event kinds/values, and a late query. It
does not yet implement the required byte-source-to-private-object-file path.
The earlier proposal to freeze the dense control after six labels is retired:
it deliberately left 23 transition rows unidentified and made the treatment
advantage tautological.

Before source freeze, one unified module must implement:

1. byte source through the exact frozen Shohin trunk;
2. model-owned physical cards, initial binding belief, initial state, cue
   evidence, interleaved event tape, and STOP;
3. cue grounding to a soft 24-element kernel without a hard group ID in the
   committed packet;
4. irreversible source-token, source-residual, and source-KV destruction;
5. interleaved cue and action execution over the private 24 x 27 joint state;
6. query materialization only after execution and source deletion;
7. a model-owned late-query reader;
8. no host parser repair, state update, schedule execution, retry, arithmetic,
   or generated-token feedback.

The source must define cue semantics through opaque, renderer-factorial
witnesses rather than globally exposing transposition IDs. Random opcode
reindexing must conjugate cue semantics, and random particle relabeling must
transform the multiplication table and scorer consistently.

### Draft board

Each program contains four opaque action cards, an initial binding/state,
source-visible cue witnesses, and one interleaved cue/action/STOP event stream.
The query is absent until the private execution commits. Matched twins share
cards, initial state/binding, cue and opcode multisets, renderer, token-length
histogram, and query; only a noncommuting cue order differs. Commuting twins
must remain invariant.

- Train: lengths 0--4, all initial particles and cue types, factorial
  renderer/name/opcode/card coordinates.
- Development: disjoint family roots and lengths 5--6.
- Confirmation: independently generated renderers/names, lengths 7--8, and
  independently chosen particle relabelings.

No exact source, family root, renderer, ordered word, or cue-witness wording
may cross a split.

## 7. Draft Arms

1. **S4-TPT treatment:** group-tied transport inside the unified source-deleted
   model.
2. **Equally informed dense favorable control:** unconstrained `24 x 24`
   cue operators receiving the same cue features, every training example, and
   every Stage-B gradient. It contains the treatment and may use more
   parameters.
3. **Abelian mechanistic ablation:** `Z24` transport with the same cue compiler
   and reader; it localizes noncommutative order only.
4. **Wrong-law control:** a fixed randomly relabeled or incompatible
   multiplication table, with the relabeling hidden from the target oracle.
5. **State-reset and state-transplant controls:** remove or swap the private
   binding/state joint during execution.
6. **Source-retained upper bound:** never deployable.
7. **Oracle-object-file ceiling:** tests only interleaved execution and late
   read; never enters a reasoning claim.

The treatment versus equally informed dense control is decisive. The
retrospective six-label canary is not a scored arm or threshold.

## 8. Requirements Before A Real Preregistration

The following must be executable and independently reviewed before a board or
training seed exists:

1. one unified byte-to-object-to-deleted-source-to-interleaved-execution-to-
   late-query forward path;
2. no hard group ID, target binding, resolved schedule, or answer in the
   committed inference packet;
3. exact transport covariance under opcode/card/particle reindexing and cue
   conjugation;
4. empty-cue, interleaved-cue/action, post-STOP suffix, midpoint state
   transplant, reset, and source-poison tests;
5. treatment, dense, abelian, and wrong-law arms receive identical examples,
   updates, optimizer settings, and all end-to-end gradients;
6. an instantiated unique-parameter ledger below 200M and measured
   forward/backward/optimizer/runtime/memory receipts;
7. an independently implemented multiplication/target oracle and raw scorer;
8. source commit before random board/training seeds, immutable split custody,
   one-read development/confirmation ledgers, and external adversarial review;
9. five-seed thresholds frozen before any scored bytes exist, including
   binding/state/late-answer exactness, treatment advantage over dense,
   noncommuting twins, commuting invariance, recoding, transplantation, and
   source deletion;
10. confirmation on unseen renderer/name/particle coordinates, not merely
    longer walks over the same visible automaton.

A future passing synthetic result would establish only bounded,
architecture-native structured computation. It would not establish general
reasoning.

## 9. Collapse And Kill Conditions

Reject S4-TPT before GPU use if:

- cue order is available to a host scheduler after source deletion;
- hard group IDs or target bindings enter the inference packet;
- the late query is visible during source compilation;
- any arm receives fewer examples, updates, optimizer steps, or usable
  gradients without that disadvantage being explicitly favorable to the
  control;
- noncommuting twins can be solved from a single local cue or token-length
  artifact;
- the particle state is not causally necessary under reset/transplantation;
- only a final answer motor, rather than binding and state trajectories,
  improves;
- a generic recurrent control matches the result under equal resources;
- unmocked source/KV deletion cannot be demonstrated.

## 10. Claim Boundary

The strongest possible future claim, if a separately committed protocol later
passes, is:

> Under a synthetic source-deleted late-query protocol, a 24-particle
> non-abelian parameter tying improved held-out order-sensitive rebinding and
> recurrent categorical execution over a stronger dense 24-state operator
> recurrence under a source-deleted late-query protocol.

That is architecture-native structured computation and a sample-efficiency
result. It is not a fundamentally new reasoning primitive, open-domain
reasoning, mathematical reasoning, language understanding, or evidence of a
world-first mechanism.
