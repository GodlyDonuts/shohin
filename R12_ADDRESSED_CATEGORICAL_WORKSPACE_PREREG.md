# R12 Addressed Categorical Workspace Preregistration

**Status:** REVISION 3 PREREGISTRATION; CPU SYMBOLIC/NEURAL FALSIFIER ONLY.
No Shohin fit, H100 job, capability claim, or novelty claim is authorized until
the committed PCPT v3 evidence receives a final independent GO and the CPU gate
below passes unchanged. A CPU pass establishes only the frozen empirical
conjunction; any novelty claim requires a separate documented prior-art review.

**Working name:** Addressed Categorical Workspace (ACW), trained with
Counterexample-Guided Broadcast Refinement (CGBR).

## 1. Decision being tested

Shohin's raw 300k transformer does not reliably preserve or update a compact
state, and existing SFT/trace/recurrence variants have not established causal
transport. The next test therefore changes the architecture itself. It adds a
small, explicit, recurrent state channel whose contents and writes can be
intervened on exactly.

The claim under test is narrow:

> On structured systems whose true transition changes one latent register per
> event, a hard addressed categorical workspace can learn an approximately
> causally sufficient source-deleted predictive state on at least 90% of
> depth-64 held-out histories from rank-limited terminal supervision, with a
> measurable advantage over parameter-, label-, and update-count-matched
> recurrent controls.

This is a learnability and dynamic-sparsity hypothesis. It is not an
expressivity claim: dense recurrence and finite transformers can realize the
same bounded functions. It is not a claim that categorical memory, recurrence,
active counterexamples, predictive state, or workspace routing is new.

## 2. Two capability tracks

The experiment keeps memory and control separate.

### Track S: scheduled state transport

The environment provides the event and destination address. The learned system
must encode the source, update a compact packet, delete all source/KV access,
and answer a late query from the packet. Passing Track S establishes learned
durable state only. It cannot establish reasoning because an external schedule
still chooses the operation and address.

### Track C: autonomous packet controller

Only after Track S passes may a controller receive the packet, current
observation, and goal; select an operator and read/write address; invoke the
tied updater; and select HALT or CONTINUE. No externally supplied execution
schedule is available. A reasoning claim requires Track C plus fresh direct
interaction. Track C is not authorized by this document.

## 3. Mathematical object

Let the packet be

```text
p = (p_1, ..., p_d) in [K]^d
```

with `K=17`. A scheduled event is `(e, a)`, where `a in [d]` is the one
register allowed to change. The architecture must implement

```text
r = U_theta(onehot(p), onehot(c_theta(e)), onehot(a)) in [K]
p'_a = r
p'_j = p_j for every j != a.
```

The unchanged registers are copied byte-identically. The updater cannot write
them. The source writer and event coder each emit `d x K` logits and use a
straight-through hard categorical choice during training. All evaluation,
intervention, and publication paths use literal integer symbols.

A late reader receives only `(p, q)`. It may not receive source tokens, source
hidden states, prior attention K/V, event history, a replay buffer, or a hidden
continuous state. Process instrumentation must account for every dynamic byte.

### Minimal-packet theorem used as a gate

Let the event/query relation be totalized with an explicit inadmissibility
answer. Histories `h` and `g` are residual-equivalent exactly when every common
finite continuation `c` and late query `q` has the same answer. For a closed
deterministic relation with `N` reachable residual classes, assume a
deterministic history-to-packet encoder, exact compositional packet updates, and
exact observations for every totalized continuation/query after source
deletion. Packet equality then cannot merge distinct residual classes.
Therefore `K^d >= N`. If `K^d = N`, the reachable packet is a bijection with
the causal quotient and every packet update is conjugate to the residual
derivative.

Proof: if two distinct residual classes had the same packet, deterministic
packet updates and reads would give equal answers for every common future,
contradicting their separation. Equality of finite cardinalities then gives the
bijection; composing through it gives the conjugate update.

This is the standard minimal-state argument in packet coordinates, not a new
state ontology.

### CGBR finite bound

When two histories collide in a hard packet but differ in residual behavior,
an oracle returns one separating continuation/query. That witness is applied to
the entire collision block and retained permanently. A complete separator
oracle requires at most `N-1` strict partition refinements to split `N`
residual classes. This is not an optimization-convergence or label bound: one
refinement can add many labels. For witnesses `w` with answer alphabets `A_w`,
the information lower bound is

```text
sum_w log2(|A_w|) >= log2(N).
```

For one fixed alphabet `A`, it implies at least
`ceil(log_|A|(N))` scalar outcomes in the best case. Every oracle call, witness
byte, and added answer label is charged.

The affine `F_17^d` board needs exactly `d` independent linear witnesses. Its
coordinate basis is already optimal, so success there is a correctness control,
not evidence for a new method. The learnability hypothesis is tested only when
the distinguishing basis and answer recoding are hidden.

## 4. Architecture frozen for a Shohin sidecar

The base remains byte-identical and frozen at 125,081,664 parameters.

```text
source hidden [576] -> source projector -> 4 x 17 logits -> hard packet
event hidden  [576] -> event projector  -> 4 x 17 logits -> hard event code
[packet 68, event 68, address 4] -> MLP 140 -> 64 -> 17 replacement logits
packet one-hot [68] -> bias-free bridge 68 -> 64 -> q_delta
```

Parameter ledger:

| Component | Parameters |
|---|---:|
| source projector `576 -> 68` with bias | 39,236 |
| event projector `576 -> 68` with bias | 39,236 |
| updater `140 -> 64 -> 17` with biases | 10,129 |
| packet bridge `68 -> 64`, no bias | 4,352 |
| **Total sidecar** | **92,953** |

The sidecar is 0.07431% of the frozen base. The packet's semantic capacity is
`log2(17^4) = 16.35` bits and its ideal packed width is 20 bits or three bytes.
Actual persistent evaluation storage is four `uint8` values (32 bits/four
bytes) or four `int64` values (256 bits/32 bytes), depending on the frozen
runtime. Training uses a 68-element BF16 straight-through one-hot (1,088
bits/136 bytes) plus a separate 68-element BF16 transient logit tensor of the
same size. Every arm reports semantic capacity, actual persistent bytes,
transient bytes, and dtype separately. During a Shohin test, the bridge adds
`q_delta` only to head zero of the final transformer block. Query tokens and the
frozen language decoder remain available; source and event K/V do not.

The two projectors are deliberately separate and counted. Tying them is a
smaller ablation, not the treatment. Any parser, schedule, cache, or external
executor must be listed in the resource ledger. Track S may receive the
destination address; Track C may not.

## 5. CPU falsifier domains

### A. Exact affine control

Use `F_17^d` for `d in {2, 3}` with events

```text
x_i <- alpha*x_i + beta*x_j + gamma mod 17.
```

Exhaustively test widths `d-1`, `d`, and `d+1`. Width `d-1` must exhibit a
certified collision. The literal coordinate packet at width `d` must pass every
state, event, query, donor swap, and output recoding. Failure rejects the board
and evaluator before neural training.

### B. Hidden-basis sparse systems

Generate a closed family from the same sparse latent transitions, but hide the
state basis behind a seeded `GL(3, F_17)` recoding, render source and event IDs as
fixed opaque features, and independently permute every answer alphabet. The
destination schedule is visible only in Track S. The learner receives no
packet, state, intermediate, basis, or update-target labels.

Each source receives exactly two distinct terminal scalar consumers before
CGBR. This rank-thinning prevents one example from directly spelling out the
whole state. The frozen generator contract is:

| Field | Frozen value |
|---|---|
| field / latent dimension | `F_17`, `d=3` |
| non-scored curriculum-pilot data/optimizer seed | `2026071600` |
| development seeds | `2026071601`, `2026071602`, `2026071603` |
| uniform-query control seed | `2026071604` |
| confirmation entropy | three domain-separated seeds from one exact future NIST Beacon 2.0 pulse; the target pulse, complete code/checkpoint/selection identity, KDF, and failure policy must be committed and publicly timestamped before the pulse exists |
| source/event feature dimensions | 96 / 96, IEEE float32 |
| source rendering | latent state times seeded `GL(3,F_17)` matrix, then 51-dimensional coordinate one-hot times seeded normalized Gaussian `51 x 96` projection |
| event rendering | typed one-hot `(dst,src,alpha,beta,gamma)` times seeded normalized Gaussian projection to 96 |
| event bank | 48 events, exactly 16 per destination, sampled without replacement then ID-sorted |
| public query bank | 24 seeded nonzero affine covectors; first three full rank over `F_17` |
| post-freeze query bank | 8 seeded nonzero affine covectors, guaranteed coefficient-disjoint from all 24 public covectors |
| answer recoding | one independently seeded uniform permutation of 17 labels per query |
| public train histories | 4,096, source/endpoint in train split, accepted-depth quotas balanced on `0..8` (counts differ by at most one) |
| public oracle histories | the same 4,096 public train histories, ID-sorted |
| adaptation histories | 1,024, source/endpoint in adaptation split |
| evaluation histories | 2,048 at each exact depth `8,16,32,64,65`, source/endpoint in evaluation split |
| state split | SHA-256 of `seed || canonical state`, buckets 0-69 train, 70-84 adaptation, 85-99 evaluation; intermediate visits unrestricted and counted by bucket |
| optimizer | AdamW, LR `0.003`, weight decay `0.0001`, batch 256, float32 |
| optimizer RNG | development/pilot use their public domain seed; confirmation uses a separately domain-separated digest of the post-pulse seed commitment |
| direct-state diagnostic | same ACW and schedule; final/source/every-active-transition packet supervision with `answer_CE + 4.0 * mean_wrong_register_MSE` |
| refinement | 200-update initial warmup, 12 x 200 refinement/filler updates, then 800 final updates; 3,400 total |
| new-reader adaptation | writer/updater frozen; 500 updates, LR `0.003`, batch 256 |
| maximum labels | 57,344 = 4,096 histories x 14 scalar labels |
| maximum witness selections | 512 per round / 6,144 total |
| maximum oracle candidate evaluations | 512 groups x 24 queries x 12 rounds = 147,456 |

The static Keychain commitments in revision 1 are **retired**. They prove only
that a value matches a digest, not that project operators lacked the value while
developing the experiment. The generator now rejects every confirmation
identity with `disabled_pending_future_nist_beacon_v2`; the old preimages cannot
produce a canonical dataset through either the CLI or the public generation
function. No scored confirmation run is authorized until the future-pulse
opener below is implemented, reviewed, committed, and replay-tested.

The replacement confirmation protocol is commit-then-reveal with public future
entropy:

1. After all development checkpoints and the exact arm-selection record freeze,
   write one hash-bound authorization containing their complete file and
   metadata bindings, every scientific/runtime path, the exact KDF, and one
   canonical NIST Beacon chain/index/timestamp at least 48 hours in the future.
2. Require the authorization commit to equal `origin/main`, not merely be its
   ancestor. Publish a deterministic witness in a public transparency log before
   the target pulse. The witness must bind the repository, exact commit,
   authorization SHA-256, pulse URI, and target timestamp.
3. After the target time, fetch only that committed URI. Verify the deployed
   Beacon 2.0 cipher-suite-0 serialization, certificate identifier, RSA-4096 /
   SHA-512 signature, recomputed `outputValue`, exact chain/index/timestamp,
   previous-output link, and prior precommitment reveal. `pipeline/acw_nist_beacon.py`
   and its archived pulse fixture are the minimum verifier regression.
4. Derive three 32-byte seeds from the public `outputValue`, authorization hash,
   exact commit, pulse URI, and domain index. Never seed from
   `localRandomValue`. Serialize the authorization hash, pulse payload hash, and
   seed commitment in every identity; do not accept a caller-supplied label.
5. Generate and score all three domains. Network failure retries the same URI;
   invalid cryptographic evidence aborts and publishes a failure receipt; a
   valid poor result is final; code changes require a new future authorization.

This protocol still trusts NIST's beacon operation, RSA/SHA-512, the public
timestamp service, and the frozen local verifier. It does not claim that NIST
itself lacked the beacon's internal randomness. Those residual assumptions must
be reported with the result.

Train on event depths `0..8`; test exact depths `8`, `16`, `32`, `64`, and the
one-step-beyond horizon `65`. Freeze the writer/updater before the confirmer
opens final new consumers, continuations, and answer recodings. Only a new
reader may train on those frozen packets.

Splits constrain source and endpoint states only. Intermediate trajectory states
may cross buckets and therefore are not claimed as state-disjoint; each artifact
reports train/adaptation/evaluation bucket visits at every position. The
post-freeze reader task contains exactly eight new seeded affine covectors with
eight new independent 17-label permutations. Its 1,024 adaptation histories
provide all eight labels (8,192 records). Each 2,048-history evaluation depth
provides all eight labels (16,384 records per depth). The writer, event coder,
updater, and packet bridge remain frozen for all 500 reader updates.

### C. Process roles and CGBR algorithm

The **generator/oracle** sees latent truth for public development data. It
serializes public features, event addresses, query IDs, recoded answers, and
immutable history IDs. A separate, non-scored curriculum pilot with seed
`2026071600` is the only model whose hard packets are shown to the oracle. It
produces one frozen CGBR curriculum before any scored arm starts. The
**trainer** runs on Stokes from a hash-bound bundle containing only public
records, the frozen curriculum, model/trainer code, and the allowed arm ID. It
does not contain confirmation preimages. The **confirmer** remains sealed until
arm selection and model freeze; it permits no collision queries, architecture
changes, checkpoint selection, or CGBR.

The pilot implementation and canonical configuration must be committed and
pushed before execution. One canonical command owns the complete run: it starts
from absent canonical paths, generates and byte-replays the public pilot domain,
launches two distinct measured child processes with the exact frozen
hyperparameters, and requires byte-identical schedules and reports. The freezer
holds both children alive on inherited parent-release pipes after atomic output,
reconciles their real PID/PPID identities with the live Slurm allocation, then
independently reruns all 3,400 updates from the registered data while both
children remain live. It compares the tensor-state hash, loss transcript,
schedules, report, and regenerated arrays before releasing the children and
requiring zero exits. Every later canonical report load repeats this
recomputation; copied or locally rehashed reports therefore have no standing.
Execution receipts bind positive wall time and peak RSS, process/host/runtime
identity, allocated CPU count, numeric Slurm job ID, and a hash-bound live
`scontrol show job` snapshot. Exact working scientific files must equal their
`HEAD` blobs, and `HEAD` must equal `origin/main` before, during, and after the
run. Any divergence blocks the lane; additional identical replays may audit but
may not select among schedules. After each 200-update curriculum-pilot round, the oracle receives only hard
packet tuples and history IDs for the public oracle pool. It groups in ascending packet order,
keeps groups containing multiple residual states, and sorts them by decreasing
number of residual classes then packet tuple then minimum history ID. For at
most 512 groups, it scans only queries unused by every member, in ascending ID,
and selects the first query maximizing distinct answers. If the common-unused
intersection is empty, the group is recorded as witness-exhausted and receives
no selected witness. If every common-unused query has only one distinct answer,
the group is recorded as query-bank-unresolved and likewise receives no selected
witness. Every history receives exactly one new unused query that
round: the separating query for its selected collision group or its seeded
uniform unused filler query otherwise. Records are deduplicated and serialized
in `(history_id, query_id)` order. This fixes multiplicity and prevents
collision-block size from becoming a label-count side channel. Zero eligible
cross-residual collisions stops witness selection, not training: deterministic
filler-only rounds continue through round 12 so every history has exactly 14
labels and the primary endpoint always uses 57,344 labels. Round zero contains
the two initial labels per history (8,192 total); rounds 1 through 12 contain
exactly one new label per history (4,096 each).
Every candidate `(collision group, query)` inspected during the scan is charged
as an oracle candidate evaluation, whether selected or not.

The PID/Slurm record is operational provenance, not cryptographic remote
attestation. It trusts the committed parent process, Stokes kernel, Slurm
controller, and filesystem during execution. The result's durable numerical
standing comes from mandatory fresh deterministic recomputation by every
canonical consumer, not from treating a historical receipt as a signature.

### Cross-runtime byte portability amendment

The first otherwise-complete public execution (`740053`, generator v2) is not a
canonical pilot result. Independent replay on macOS reproduced all 59
integer/state/query arrays exactly but found one-ULP differences in all eight
float32 feature arrays. The random projection matrices were byte-identical;
BLAS-dependent reduction order in one-hot matrix multiplication caused the
drift. Same-runtime replay is insufficient for this protocol because canonical
consumers must regenerate the dataset on an independent runtime.

Generator v3 therefore constructs every event feature by adding its five
selected projection rows in fixed semantic order, and every source feature by
adding its three selected rows in coordinate order. Each addition is an
explicit float32 elementwise operation; no BLAS reduction is permitted. Golden
SHA-256 tests bind the complete 48-event bank and the source rendering of all
`17^3 = 4,913` states. Any v2 dataset or report is diagnostic-only and cannot be
anchored, consumed by a scored arm, or cited as learned evidence. Before the v3
pilot runs, independently generated full Mac and Stokes manifests and all 67
registered arrays must be byte-identical.

The resulting ordered `(history_id, query_id)` curriculum is frozen and replayed
identically to ACW, dense categorical, addressed continuous, GRU, packet-token,
answer-motor, and source-retained scored arms. No scored arm has an arm-native
collision oracle. Uniform-query ACW instead receives a separately committed
seeded-uniform query curriculum with identical history IDs, per-history
multiplicity, round boundaries, and 57,344 final labels; differing query choice
is the controlled treatment. Direct-state ACW receives the frozen CGBR
curriculum plus its declared state auxiliary labels and is diagnostic only.

## 6. Arms and resource matching

All primary scored architecture arms use identical source/event features,
histories, frozen CGBR labels, optimizer evaluations, seeds, and stopping rule.
Uniform-query and direct-state ACW differ only in the declared curriculum or
auxiliary labels above and are not included in the identical-label comparison.

1. **ACW treatment:** hard `K^d` packet and one-register write.
2. **Dense categorical recurrence:** same hard symbols and parameter budget,
   but every event may rewrite all registers.
3. **Addressed continuous single-write:** same supplied address and exact copy
   mask, three float32 registers, one scalar replacement, parameter matched.
4. **Continuous GRU:** favorable 39-float state and parameter-matched update.
5. **Packet-token transformer:** three recurrent packet tokens, one 24-wide,
   four-head block with FFN width 128.
6. **Uniform-query ACW:** same architecture and final label count, but no
   collision-conditioned witness selection.
7. **Direct-state ACW:** favorable diagnostic with packet/state supervision.
   It must pass; it cannot support the main claim.
8. **Compiled sparse-register realization:** literal coordinate packet and
   affine update, with every external arithmetic operation charged. This is the
   known compilation and must pass; it cannot support neural learnability.
9. **Answer motor:** equal/favorable parameters trained only to reproduce the
   current consumers. It diagnoses answer-specific shortcuts.
10. **Source-retained reader:** diagnostic upper bound with source/KV access;
    source `96 -> 128`, `GRUCell(99,128)`, retained-source readout
    `[state128,source96,query16] -> 256 -> 17`, 166,801 parameters.
   It is not a valid source-deleted comparator.

For the CPU domain, the shared reader is query embedding `24 x 16`, followed by
`48 -> 64 -> 17`. Exact frozen core widths and trainable parameters are:

| Arm | Widths | Parameters | Persistent state |
|---|---|---:|---|
| ACW / uniform / direct-state | categorical updater hidden 80 | 26,008 | 3 `uint8` eval symbols = 3 bytes; 51 float32 train one-hot = 204 bytes |
| dense categorical | updater hidden 64 | 26,250 | same as ACW |
| addressed continuous | replacement MLP hidden 272 | 26,008 | 3 float32 = 12 bytes |
| GRU | hidden 39 | 26,036 | 39 float32 = 156 bytes |
| packet-token transformer | width 24, heads 4, FFN 128, one block | 25,872 | 3 categorical symbols = 3 bytes; transient token state reported separately |
| answer motor | commutative source/event summary `208 -> 113 -> 17` | 25,939 | source plus one 96-float event mean; no recurrent state |

The compiled sparse realization and source-retained diagnostic are not
parameter-matched claims. Every valid control receives the same supplied
destination address, features, batch schedule, optimizer evaluations,
label/oracle cap, and stop rule. No hyperparameter search is permitted in the
canonical run.

Parameter differences above 5%, retained-bit differences, actual/transient
bytes, mixed precision, extra source bytes, oracle calls, and train/inference
FLOPs are reported rather than hidden. "Matched" means parameters, labels,
optimizer updates, inputs, and schedules are matched; more-compute controls
remain eligible and favorable rather than being excluded. FLOPs and wall time
are measured for every arm and cannot be used post hoc to remove the strongest
control.

The parameter formulas are part of the contract. All MLPs use SiLU and include
biases except the named bridges:

- shared reader: `24*16 + (32+16)*64 + 64 + 64*17 + 17 = 4,625`;
- ACW: two `96 -> 51` projectors, `105 -> 80 -> 17` updater,
  bias-free `51 -> 32` bridge, plus reader = 26,008;
- dense categorical: the same projectors/bridge/reader and
  `105 -> 64 -> 51` updater = 26,250;
- addressed continuous: `96 -> 3` source, `96 -> 51` hard event code,
  `57 -> 272 -> 1` addressed replacement, bias-free `3 -> 32` bridge,
  plus reader = 26,008;
- GRU: `96 -> 39` source, `GRUCell(99,39)`, bias-free `39 -> 32`
  bridge, plus reader = 26,036;
- packet-token: two `96 -> 51` projectors, shared `17 -> 24` token
  projection with bias,
  three 24-wide address embeddings, one four-head 24-wide
  transformer block with FFN 128, `24 -> 17` requantizer, bias-free
  `51 -> 32` bridge, plus reader = 25,872;
- answer motor: query embedding `24*16`, then commutative
  `[source96, mean_event96, query16] -> 113 -> 17` = 25,939.

The packet-token control re-quantizes all three registers to literal 17-way
symbols after every event. Its persistent state is therefore three `uint8`
symbols; its seven 24-wide float32 packet/event/address tokens consume 672
transient bytes before attention intermediates, which are measured at runtime.

For every learned arm, the resource artifact contains a complete training-step
and inference-batch record at batch 256: active event count, wall time, process
peak RSS, PyTorch operator-reported FLOPs, largest runtime operator allocation,
largest self-operator allocation, and total positive operator allocations.
AdamW is included in the training measurement. Unsupported profiler operations
are explicitly uncounted rather than imputed, so an operator-reported FLOP total
is never represented as an exact hardware FLOP count. The packet transformer's
runtime allocation record is the preregistered transient-attention measurement.
The exact compiled sparse control separately reports event/query arithmetic,
table bytes, and persistent bytes while replaying every source state and event
ID; reading stored final state as its prediction is forbidden. It is not a
learned arm, and fresh wall time is excluded from the score artifact so the
required deterministic evaluator replay remains byte-identical.

## 7. Causal interventions

The frozen evaluator reports two separate metrics. **Scalar accuracy** is the
fraction of individual `(history, query)` answers correct; its balanced chance
is `1/17`. **State exactness** is the fraction of histories for which all 24
public queries are correct; its independent-uniform reference is `(1/17)^24`,
although empirical shuffled controls are decisive. Seen-depth metrics are
reported separately at depth 8; no depths are pooled.

The frozen evaluator performs all of the following:

- replace the packet with a donor packet while holding query and source ID
  fixed; answers must follow the donor residual state;
- shuffle packets within a batch; performance must fall to chance;
- hold packet fixed while changing source bytes; answers must not follow the
  inaccessible source;
- append equivalent and non-equivalent event words; equivalent packets must be
  query-equivalent and non-equivalent packets must admit a separator;
- train a new reader after packet freeze on unseen consumers and a new output
  recoding;
- evaluate exact depths 8, 16, 32, 64, and 65;
- verify unchanged registers are byte-identical after every addressed update;
- rerun from the same seed and require byte-identical score artifacts.

The donor map is the one-position cyclic roll of the ID-sorted depth-64
evaluation histories. Event-word evaluation uses the first 256 ID-sorted
depth-64 histories and the first lexicographic distinct two-event words that
produce an equal endpoint, plus the first lexicographic pair with unequal
endpoints. The post-freeze reader uses an eight-entry, 16-wide query embedding
and `[state,query16] -> 64 -> 17`, AdamW with the frozen optimizer settings,
seed `2026071699`, and exactly 500 updates. These choices cannot change after a
checkpoint is read.

## 8. Frozen pass and kill criteria

The direct-state diagnostic must first reach 99% scalar accuracy and 95% state
exactness. If it does not, the implementation or optimization setup is invalid.

Across all three development seeds and at least two of three unopened
confirmation seeds, ACW must satisfy all of these. The reported result is the
median across the three confirmation seeds, with every seed shown:

1. at least 99% scalar accuracy and 95% state exactness after source deletion
   at depth 8;
2. at least 99% scalar / 92% state exactness at depth 32, at least 98% scalar /
   90% state exactness at depth 64, and at least 97% scalar / 85% state
   exactness at depth 65;
3. at least 99% scalar donor-following accuracy and shuffled-packet scalar
   accuracy no more than two percentage points above `1/17`;
4. at least 98% scalar accuracy and 90% eight-query state exactness for readers
   trained after packet freeze on unseen consumers and output recodings;
5. zero illegal multi-register writes;
6. the **primary comparative endpoint**, median confirmation depth-64 state
   exactness at exactly 57,344 scalar labels, is at least 90% and at least ten
   absolute points above the strongest valid equal-label architecture control;
7. the all-three-development / two-of-three-confirmation rule above and all
   resource ledgers complete.

Any source/KV path, hidden packet supervision in the treatment, post-score seed
or threshold change, confirmation leak, illegal write, missing control, or
resource-ledger omission invalidates the run. If a matched control ties ACW
within three points at equal resources and labels, the claimed resource
advantage is rejected even if ACW itself works. No Shohin sidecar fit follows a
CPU no-go.

Label efficiency is secondary and cannot rescue a failed primary endpoint. It
is reported at the frozen cumulative checkpoints
`8,192 + 4,096*r` labels for `r in 0..12`. A CGBR efficiency statement is
allowed only if both CGBR and uniform-query ACW cross 90% depth-64 state
exactness; the ratio uses the first frozen checkpoint crossing that threshold.
Candidate oracle evaluations and witness selections are reported separately
and are never treated as zero-cost labels.

The trainer serializes a hash-bound model state immediately after each of the
first 12 curriculum rounds at those exact cumulative label counts. The
`r=12` / 57,344-label state is serialized after its 200 round updates and the
frozen 800 final refinement updates, at 3,400 total updates, so it is the same
model as the primary endpoint. The frozen
evaluator rehashes and scores every state at depth 64. A separate committed
adjudicator rejects missing/duplicate seeds, arms, reports, resource fields, or
label checkpoints; enforces the direct-state gate, all-three-development and
two-of-three-confirmation rules, every per-depth/causal/new-reader threshold,
the confirmation median, and the strongest-control margin; and writes one
immutable hash-bound decision. Historical Git blobs and the files actually
executing must both match the checkpoint scientific identity, with all listed
scientific paths clean.

## 9. Prior-art and equivalence boundary

The causal state is equivalent up to coordinates to a minimal residual machine,
predictive-state representation, or deterministic automaton. Hard symbols are
vector quantization. The updater is a recurrent state machine. External memory,
neural status registers, modular recurrent mechanisms, recurrent-memory
transformers, block-recurrent transformers, and workspace routing are known
families. Collision-guided refinement is adjacent to active automata learning,
counterexample-guided synthesis, and distinguishing-sequence construction.

Accordingly, no component or primitive is called world-first. A CPU pass does
not authorize novelty language. The only empirical contribution left open is
the measured conjunction: rank-limited terminal
supervision plus commit-then-challenge collision refinement plus hard
single-write source-deleted state, with a demonstrated label/compute advantage
over favorable controls. A sparse-register compilation with the same resource
vector rejects even that narrow claim.

## 10. Shohin admission after a CPU pass

The smallest H100 fit, if authorized, freezes the immutable 300k base and trains
only the 92,953-parameter sidecar on frozen, execution-verified transition and
reuse data. The data split and every researcher-written evaluation prompt are
committed first. The fit output is isolated from all flagship paths.

Promotion requires source-deleted state update, late-query recoding, donor
intervention, ordinary-language preservation, and full fresh multi-turn
transcripts authored and judged after the checkpoint freezes. Fit loss,
synthetic exactness, visible `<think>` tags, or benchmark movement alone cannot
promote it.

If Track S succeeds, a separate Track C preregistration will add an operator,
address, and HALT/CONTINUE controller around the same packet. Until that
controller chooses and verifies its own computation, the result is durable
learned memory, not reasoning.
