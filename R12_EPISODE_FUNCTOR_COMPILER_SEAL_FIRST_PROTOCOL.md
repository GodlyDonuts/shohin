# R12 EPISODE Functor Compiler Seal-First Protocol

**Status:** preregistration draft for CPU falsification. This document does not
authorize a neural fit, source freeze, GPU run, capability claim, or
continuation pretraining.

**Date:** 2026-07-23

## Decision

The Episodic Functor Compiler (EFC) remains an admissible architecture
hypothesis, but neither the supplied theory draft nor the existing EPISODE
corpus is sufficient for an advancement claim.

Two corrections are mandatory:

1. the committed machine must retain opaque state keys when a late query names
   its start state; and
2. official challenge coordinates and bytes must be generated from entropy
   that does not exist until after the machine artifact is externally sealed.

The current corpus remains a consumed diagnostic. It uses one deterministic
PRNG trajectory for world mechanics, opaque keys, demonstration order, and
hidden query coordinates. Candidate-world acceptance also inspects sampled
query outcomes. Splitting complete packets into separate files afterward
establishes file custody, not temporal independence.

## Capability Contract

The candidate system must perform three model-owned operations:

```text
raw world evidence
    -> neural compiler
    -> fixed-width anonymous categorical machine

late query
    -> key/path/observer parser
    -> fixed generic ordered executor
    -> answer
```

The compiler may learn. The sealed executor may use only fixed tensor gathers,
categorical transitions, masks, and observer readout. It may not read source
tokens, call an oracle, branch on task semantics, search, retry, repair, or
receive assessor feedback.

The result, if all gates pass, is bounded episode-local machine compilation and
ordered execution. It is not unrestricted language reasoning.

## Frozen Protocol Fields

Before official entropy exists, publish a canonical `protocol.json` that fixes:

- protocol, generator, oracle, renderer, compiler, executor, and assessor
  source hashes;
- candidate checkpoint hashes;
- machine schema version and exact serialized byte length `B`;
- field precision, padding, masks, and every persistent source-dependent byte;
- state count `K`, action count `M`, observer count `P`, and answer alphabet;
- opaque state/action/observer key contracts;
- whether depth zero is admissible;
- training and scoring depth supports;
- challenge quotas and duplicate policy;
- controls, parameter/byte/compute limits, and kill thresholds;
- attached-hard versus serialized-detached numerical tolerance; and
- the strict complete-system parameter ceiling.

No field may change after publication.

## Phase 1: Model And World Seal

All candidate weights must be sealed before official worlds exist.

World entropy is derived from a preannounced future public randomness beacon:

```text
S_world = SHA256(
  "EFC/world/v1" ||
  protocol_sha ||
  model_root ||
  beacon_round ||
  beacon_value
)
```

World generation uses domain-separated cryptographic streams for:

- transition operators;
- state nonces;
- action nonces;
- observer nonces and maps;
- demonstration order; and
- renderer choices.

No mutable RNG stream may be shared with future challenge generation.

The world admissibility predicate may inspect only world mechanics: transition
completeness, noncommutativity, causal-quotient size, observer separation, and
fixed capacity. It must never sample or inspect a query, answer, query depth,
candidate prediction, or future challenge behavior.

Official worlds must come from an independently implemented held-out generator,
or from a post-seal random lottery between two independently implemented
equivalent generators. Seal the public world evidence as `world_root`; latent
mechanics remain assessor-only.

## Phase 2: Machine Seal

For each world, start a fresh compiler process that may read only:

- the sealed candidate checkpoint;
- one world evidence row; and
- the frozen protocol.

It emits exactly `B` bytes as `machine.bin`. Variable-length state, source
excerpts, persistent KV caches, undeclared diagnostics, and sidecar memory are
forbidden.

Each machine receipt binds:

```text
protocol_sha
model_sha
world_sha
compiler_sha
machine_sha
machine_bytes
policy_sha
schema_version
```

Atomically publish `machine_root`, terminate every compiler, close inherited
descriptors, and make world evidence unreachable from the executor. The
challenge beacon must occur after externally timestamped publication of
`machine_root`.

## Phase 3: Post-Seal Challenge

Challenge entropy is independently derived:

```text
S_challenge = SHA256(
  "EFC/challenge/v1" ||
  protocol_sha ||
  model_root ||
  world_root ||
  machine_root ||
  beacon_value
)
```

The coordinate generator receives only `S_challenge`, a world ordinal, and
frozen constants. It must not receive world bytes, machine bytes, world seeds,
generator candidate indices, model weights, predictions, or targets.

It first commits abstract coordinates:

```text
(start_index, action_index_word, observer_index, renderer_index)
```

Only after the coordinate root is published may an assessor-only process map
indices to opaque keys. No rejection or reroll may depend on an answer or model
behavior.

For the current `K=8`, `M=3` mechanics:

```text
nonempty depth 1..6 support = 8,736 queries
depth 0..6 support          = 8,744 queries
```

The protocol must choose one convention and use it everywhere. The current
query grammar requires nonempty words, so its correct receipt is 8,736 queries
and 26,208 three-bit answers.

That depth-six table is only 3,276 bytes. A 16 KiB machine ceiling therefore
does not exclude an exhaustive depth-six answer table. The official score
support must be materially larger. With depth zero through twelve:

```text
8 * sum_(d=0)^12 3^d = 6,377,288 queries
three-bit answer table     = 19,131,864 bits
packed answer bytes        = 2,391,483 bytes
```

A frozen 16 KiB source-dependent machine is then about 146 times too small for
the exhaustive answer table, before query-routing metadata.

For a sparse `k`-entry cache over the current 8,736-query support, even granting
canonical query indexing, a favorable lower bound is:

```text
ceil(log2(choose(8736, k))) + 3k bits.
```

The 261-bit minimum explicit-machine receipt permits at most 20 such entries;
the 276-bit conservative receipt permits at most 21. Because every current
action is a permutation, a uniformly sampled all-start challenge is exactly
output-balanced. A default-answer cache with those exceptions is therefore
bounded near 12.73% accuracy.

Byte accounting alone cannot distinguish a machine from every possible
compressed evaluator. A decoder that stores three generators and composes them
is functionally a transition machine. The defensible distinction is operational:
query-oblivious prefix state, next-action closure, repeated use of identical
sealed bytes, and the predicted intervention behavior.

One provisional per-world panel is:

- 968 exhaustive queries at depths zero through four;
- 64 uniformly sampled unique queries at each depth five through twelve,
  totaling 512;
- 64 noncommuting order-twin pairs, totaling 128; and
- 24 repeated-action depth-twelve probes.

Total: 1,632 queries per world. Exact quotas, inclusion of depth zero, and
duplicate handling must be frozen before the protocol seal.

## Phase 4: Execution And Opening

The executor receives only:

- one sealed `machine.bin`;
- the committed query panel; and
- the frozen protocol.

It may not access world evidence, latent mechanics, targets, oracle code,
compiler code paths, or mutable machine state across candidate comparisons.

Hash the machine before and after every panel. Seal predictions before target
opening. Two independently implemented assessor oracles must agree on every
target.

## Independent Runtime Gate

The current two Python audits are separately written but are not independent
enough for a source-freeze gate. Both sort opaque keys, use row-major integer
transition tables, and execute the same left-to-right update loop.

Before a neural fit, require:

1. a standalone C runtime using explicit key lookup and a flat
   `next[action*K + state]` representation;
2. a standalone Rust runtime using one-hot state and boolean-relation image;
3. no shared Shohin imports, parser, serializer, JSON helper, transition
   layout, or expected-answer implementation;
4. input restricted to sealed fixed-width machine, query, and intervention
   files; and
5. a third assessor using direct relation composition or exhaustive
   enumeration.

Two-runtime agreement is not sufficient without deliberate format mutation
tests and the independent assessor.

### Current CPU implementation

The first fixed-width runtime gate is implemented:

- `tools/episode_functor_runtime_c.c` uses independent linear typed-key lookup
  and flat `next[action*K + state]` updates;
- `tools/episode_functor_runtime_rust.rs` independently parses the wire format
  and executes one-hot state bitsets through Boolean-relation images;
- both consume only `machine.bin` and `queries.bin`;
- both emit no-replace fixed records bound to the exact machine and query
  payload SHA-256 values; and
- strict C11/Rust compilation plus mutation/cross-runtime tests pass 47/47.

The provisional deployed wire record is exactly 1,536 machine bytes with
capacities for 16 states, eight actions, and eight observers. The current
active schema uses masks and zero padding inside that fixed record. This is a
CPU mechanics gate, not a frozen neural-machine budget or capability result.

## Causal Intervention Gates

Every gate is evaluated by both sealed runtimes and the third assessor:

- action-key permutation with transition slots fixed;
- transition-slot permutation with keys fixed;
- compensated key/transition permutation invariance;
- start-state or query-start transplant, named according to the actual schema;
- every transition-row transplant against every alternate destination;
- non-hitting prefix invariance and first-divergence locality;
- distinct equivalent words with identical full state transformations;
- noncommuting order reversals with matched action bags;
- state-coordinate gauge conjugacy;
- observer-key and observer-map interventions;
- source truncate, rewrite, rename, unlink, and poison after machine seal; and
- repeated challenge seeds against exactly the same machine hash with zero
  compiler invocations.

## Quotient Gate

The current identity observer makes all eight states distinct at depth zero.
That is a useful transition-compilation board, but a trivial quotient board.

A quotient-induction claim requires noninjective observers with:

- states indistinguishable at the empty continuation;
- future action words that separate some initially merged states;
- exact agreement between independent partition refinement and exhaustive
  future-behavior enumeration;
- observer and transition interventions that change the quotient in predicted
  ways; and
- held-out quotient structures, not only held-out opaque names.

If several physical start keys collapse into one quotient class, the artifact
must retain an explicit key-to-class map. Retaining only one key per quotient
state is insufficient to bind every late query referent.

The consumed CPU-only quotient fixture now contains twelve machines: six train
and six development, covering quotient sizes three through eight. Each has
eight physical keys, three actions, two nonconstant noninjective observers,
future-only separators, equivalent-word and noncommuting witnesses, explicit
key-to-class maps, gauge conjugacy, and isolated action/observer
interventions. Partition refinement, pair-product reachability, and exhaustive
future behavior through depth seven agree on every machine; train/development
structural-signature overlap is zero. Fixture SHA-256:
`349b8f4c4a163afd5eab288727c7bd59a1e52fb6782d25071820fac1151c678d`.
All thirteen mechanics gates pass. This fixture is exploratory and consumed,
not an official sealed board.

## Mandatory Controls

- frozen four-slot workspace control;
- state/parameter/compute-matched generic recurrent control;
- direct-machine hypernetwork;
- fused key/operator record;
- commutative action-pool control;
- untied-depth executor;
- exact-byte-matched answer-cache control;
- shuffled source witnesses;
- source-retained upper bound; and
- oracle-machine ceiling.

The EFC treatment must beat qualified generic recurrence and direct-machine
emission on unseen worlds, generators, renderers, lengths, multiplicities, and
composition motifs while retaining the predicted causal intervention
signature.

## Kill Tests

Any one of the following rejects the run:

1. query sampling or query-dependent world rejection before machine seal;
2. any challenge artifact or seed predating `machine_root`;
3. changing challenge entropy changes world or machine bytes;
4. a second challenge seed invokes the compiler or changes the machine hash;
5. the executor can read source evidence after sealing;
6. collapse on the independently implemented held-out generator;
7. incorrect covariance under nonce recoding or demonstration permutation;
8. undeclared source-dependent bytes or an exhaustive answer table fitting
   within the declared machine budget;
9. a byte-matched answer-cache control covering unseen post-seal coordinates;
10. disagreement between independent assessors;
11. host arithmetic, semantic branches, search, retry, repair, or oracle calls
    inside the deployed executor;
12. mismatch between attached-hard and serialized-detached execution;
13. malformed machine inputs accepted by either runtime; or
14. any attempt to treat CPU oracle mechanics as a neural reasoning result.

## Authorized Next Work

The following CPU mechanics are complete:

1. fixed-width deployed wire format and mutation tests;
2. independent C and Rust runtimes plus Python relation assessor;
3. a two-beacon CPU rehearsal with no-replace artifacts;
4. exact current-board semantic-bit receipts;
5. nontrivial-observer quotient mechanics; and
6. source-poison and causal-intervention falsifiers on consumed fixtures.

Only these remaining actions are authorized:

1. unify the two-beacon rehearsal with the deployed 1,536-byte C/Rust wire
   format;
2. implement a genuinely independent held-out world generator;
3. freeze byte accounting for every source-dependent parser/compiler field;
4. prove raw evidence alone identifies the required machine without oracle
   segmentation;
5. subject the unified protocol to a new hostile audit; and
6. only after that audit, write a neural compiler preregistration for user
   review.

No neural compiler, GPU job, new official board, source freeze, reasoning
promotion, or continuation pretraining is authorized by this document.
