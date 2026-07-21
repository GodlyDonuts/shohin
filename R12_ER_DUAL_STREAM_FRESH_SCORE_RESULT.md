# R12 ER-TT Ordinal-Route Fresh Score Result

**Decision:** `reject_er_dual_stream_fresh_v1`

**Custody:** one development read, zero confirmation reads. The confirmation
split remains sealed and this board may never be rescored.

## 1. Frozen provenance

- Scientific source: `e0d4d57ac2160480946f0006acbef4ed7fa5d382`
- Qualified train-only checkpoint:
  `99be7b89e0b7dfe35f745abf1320c6640ad61f2fb62624b288fb8f9502cd97e7`
- Board source: `627b6c3d97a885017041aacb5971874680e1b289`
- Board report:
  `6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f`
- Board seed: `6249020340651282430`
- Training seed: `5499768532556522119`
- Sole score job: Newton `695002`, one H100 `evc47`, 47m34s, exit zero
- Complete/trainable/headroom parameters:
  `185,532,296 / 11,129,504 / 14,467,704`
- Outcome supervision: zero
- Learned executor motor/reader parameters: zero/zero

Every arm began from the same byte-identical qualified state and received
48,000 source-only rows, 12,000 four-view families, two epochs, and 3,000
updates. The independent assessor reconstructed every metric and the recurrent
execution from immutable raw tensors.

## 2. Primary result

| Arm | Packet / relation rows | State | Answer | Joint | Witness pointers |
|---|---:|---:|---:|---:|---:|
| Treatment | 1,558/2,048 = 76.074% | 85.400% | 90.527% | 76.074% | 50.000% |
| Family deranged | 1,201/2,048 = 58.643% | 77.246% | 86.719% | 58.643% | 49.023% |
| Equality ablated | 1,199/2,048 = 58.545% | 68.408% | 76.953% | 58.545% | 56.396% |
| Source-free identity collapse | 0/2,048 = 0% | 12.939% | 25.537% | 0% | 50.000% |

The treatment advantage is only 17.432 percentage points over family
derangement and 17.529 points over equality ablation, not the frozen 50-point
minimum. Both matched controls exceed the frozen 35% ceiling. The source-free
control is zero joint, so symbol identity is causally necessary, but the fitted
controls expose a strong structural shortcut.

All declaration fields, initial rows, cardinality, rule activity, events,
HALT, late query, line pointers, binding pointers, initial pointers, and query
pointers are 2,048/2,048 exact. Every packet intervention executes exactly on
all 1,558 eligible packets and changes every sensitive result. Therefore the
source-deleted recurrent motor, state transport, halt persistence, and answer
consumer are not the observed bottleneck.

## 3. Localized compiler failure

The aggregate rejection reduces to one misspecified routing factor.

### 3.1 Renderer split

Both opcode-first witness renderers are 512/512 packet, state, answer, and
joint exact. Both opcode-middle witness renderers are 267/512 = 52.148% joint,
with 70.5%-71.1% state and 80.9%-81.3% answer. Declaration, event, and query
renderer factors do not explain the split.

### 3.2 Cardinality split

| Cardinality | Packet / joint | State | Answer |
|---:|---:|---:|---:|
| 3 | 266/512 = 51.953% | 69.531% | 78.906% |
| 4 | 268/512 = 52.344% | 72.070% | 83.203% |
| 5 | 512/512 = 100% | 100% | 100% |
| 6 | 512/512 = 100% | 100% | 100% |

Depth one through twelve stays near the same 75%-79% packet band. The failure
is not recurrent-depth accumulation.

### 3.3 Immutable pointer audit

For every opcode-first rule, every before and after witness occurrence is
selected exactly. For opcode-middle rules:

- the first after-witness slot selects the opcode on every active `N=4..6`
  rule;
- at `N=4`, the next slot selects after-witness zero;
- at `N=5` and `N=6`, the remaining hard marginal maxima are exact even though
  the first after slot remains on the opcode;
- at `N=3`, the lattice alternates between two endpoint-exclusion modes: one
  shifts the before half toward the opcode and the other shifts the after half
  away from it.

This is the exact signature of the frozen lattice. It scores a path only by
the selected witness candidates. The excluded candidate is never positively
scored as the opcode. Its inherited docstring also assumes an opcode followed
by the two witness lists. On the fresh `before : opcode : after` grammar, the
path posterior therefore collapses toward excluding an endpoint instead of the
middle opcode. Marginal identity still contains enough correct path mass to
solve every `N=5..6` relation, explaining why semantic rows can be exact while
the hard pointer row is not.

Treatment epoch-two relation and witness losses remain 0.1772 and 0.2085, so
the issue is visible in fit as well as transfer. It is not a scoring artifact.

## 4. Robustness and controls

- Complete alpha recoding: 1,694/2,048 = 82.715%
- Distractor-only rotation: 1,806/2,048 = 88.184%
- Rule-record reindex: 2,035/2,048 = 99.365%
- Full physical-record reindex: 2,018/2,048 = 98.535%

Alpha and distractor failures occur at nearly the same rates on packet-correct
and packet-incorrect rows. They are secondary evidence that the route posterior
is unstable, not a failure of exact whole-symbol equality. Physical reindexing
is nearly exact, further excluding storage order as the main defect.

## 5. Artifact custody

- Compiler checkpoint SHA-256:
  `01cbaa8c9de2c59ce75ff0eb95b6414e1cbb4c2636ca59fca03ee59d07ef3106`
- Raw evidence SHA-256:
  `ad60a5d008c29fc6531edaec78d68d9150315426d31501f327190f4dc65586dc`
- Primary report SHA-256:
  `5e54a559926a41a8c70a0d2bf5034c6f5d3f84428e23db11ec4c8646a34222d1`
- Independent assessment SHA-256:
  `ddd607f326bf1b8cab6a5a67989ba5d5908a3323f4efba2b6c22d1184cf2e483`
- Development ledger SHA-256:
  `ddbe2b037f56b787d9e764000efb08e13d3f7351ad2a647a4c96198afc01cee0`

The checkpoint, raw evidence, report, and assessment are mirrored locally with
hashes matching Newton. Confirmation remains mode `0600` and unopened.

## 6. Frontier-plan interpretation and next hypothesis

The result supports the strongest conclusion in
`FRONTIER_AGENT_PLANS_ANALYSIS.md`: keep a pointer-grounded compiler,
phase-separated parameter islands, a source-deleted recurrent executor, and a
separately testable halt interface. The executor and halt interfaces pass their
causal tests here. Adding Hopfield memory, VQ, ACT, an SSM, RL, or more generic
parameters would not address the measured error.

The only admitted repair is a **complement-coupled opcode/witness lattice**.
For each possible excluded candidate, its path score must include both:

1. the sum of the ordered witness-slot scores on the retained candidates; and
2. the existing model-owned rule-opcode score on the excluded candidate.

This makes the compiler explain all `2N+1` candidates: `2N` as witnesses and
one as opcode. It uses no renderer metadata, target outcome, host parser,
executor feedback, retry, or repair. The existing rule-opcode query is already
causally trained through event-to-rule identity; the new coupling removes the
unscored-exclusion degeneracy rather than adding a new ontology.

Before any new board, the repair must pass a train-only family-disjoint probe
with every witness grammar, cardinality, and renderer factor decorrelated. It
must report coherent path-MAP pointers in addition to independent marginal
argmax pointers, and it must retain matched deranged/equality/source-free
controls. Only a frozen pass may authorize one new fresh board and seed.

## 7. Claim boundary

This negative result does not establish native or general reasoning. It does
establish that, conditional on an exact episodic relation packet, the deployed
parameter-free recurrent executor composes arbitrary non-bijective relations,
halts, and reads the queried state exactly on the fresh board. The remaining
measured failure is compiler grounding under a held-out grammar composition.
