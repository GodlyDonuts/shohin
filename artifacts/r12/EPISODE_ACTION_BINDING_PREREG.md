# EPISODE action-binding mechanics preregistration

Date: 2026-07-23

Status: frozen before implementation

## 1. Purpose

MCTFR is closed as a learned mechanism because true-target and shuffled-target
training both produce the same perfect hard quotient. The next board must make
model-owned selection necessary. It must be impossible to pass by propagating
every available transition, activating every operator, or treating the episode
as an unlabeled union graph.

The proposed mechanism is **EPISODE**: Endogenous Predictive Intervention-State
and Operator Discovery Engine. A future neural implementation will compile raw
tokens into competing episode-specific state/action hypotheses, refute
inconsistent hypotheses using their worst contradictory witness, select one
hypothesis once, delete the source, and execute the query only through the
selected binding and ordered operators.

This first CPU board tests the identifiability and anti-shortcut mechanics. It
does not test Shohin, language, or a learned neural mechanism.

## 2. Frozen finite system

Each base system has:

- eight anonymous states;
- three bijective, generally noncommuting physical operators;
- three fresh opaque action tokens;
- eight fresh opaque state tokens;
- a complete set of demonstration transitions;
- one query start state;
- a query word of depth one through six.

Only ordinary integer token IDs and padding are model-visible. Demonstration,
transition, query, and separator markers are ordinary source tokens. Physical
operator identities, latent state identities, action bindings, renderer
provenance, cyclic-orbit membership, target state, and assessor products remain
offline.

The first board uses complete transition demonstrations intentionally. It
isolates episode-local binding and ordered composition before adding sparse
system identification. Passing it is necessary but not sufficient for the full
EPISODE proposal.

## 3. Cyclic action-binding triples

For each base system, construct three variants indexed by
`r in {0, 1, 2}`. Visible action token `a_j` denotes physical operator
`T_(j+r mod 3)` in variant `r`.

The query start state and visible query action word are identical across the
three variants. Generation accepts a query only when the three cyclic bindings
produce three distinct terminal states.

Because every bijective operator is demonstrated once from every state:

- each visible action token occurs equally often in every variant;
- each state token occurs equally often as a demonstration source and target;
- the complete raw-token histogram is identical across the triple;
- erasing action-token identity from demonstrations yields the same ordered
  transition witnesses;
- the unlabeled union of transition graphs is identical;
- query length, start state, query token sequence, padding, and renderer are
  identical;
- the three correct target tokens are distinct.

Any deterministic action-agnostic or all-actions-union method must emit the same
answer on all three variants and therefore has exact accuracy at most `1/3` on
every complete triple.

## 4. Ordered noncommuting twins

For qualifying systems, construct matched queries `ab` and `ba` from the same
start state whenever they have different terminal states. Demonstrations,
query-action histograms, and all non-query tokens are identical. A
query-order-bagging method must emit the same answer for both and is capped at
`1/2`.

## 5. Independent offline oracles

Two independent CPU procedures must agree on every scored episode:

1. **Visible-table executor:** parse the raw demonstration stream into a typed
   `(state token, action token) -> state token` table and execute the ordered
   query.
2. **Binding enumerator:** enumerate all permutations from visible action
   tokens to the assessor's three physical operators that are consistent with
   every demonstration, then execute the query under every surviving binding.

An episode is identifiable only when all surviving bindings yield one terminal
state token. An intentionally underidentified episode must return `ABSTAIN`.
No ambiguous episode may be scored as an ordinary answer.

## 6. Source-deletion contract

A future model may compile and retain only:

- anonymous state slots;
- episode-specific action slots and binding probabilities;
- candidate transition operators;
- the model-proposed ordered query queue;
- competing-hypothesis weights;
- serializer state and a halt/abstain latch.

Before execution it must destroy:

- input token IDs and embeddings;
- transformer residuals and KV cache;
- compiler scratch activations not listed above;
- physical operator and latent state identities;
- renderer, orbit, family, target, and oracle metadata;
- all source-file access.

Post-deletion source poisoning must leave output bit-identical. Scrambling the
selected action binding must sharply reduce exactness. Scrambling discarded
hypotheses must not.

## 7. Mandatory controls

- action-erased union graph;
- every action active simultaneously;
- fixed global action meanings;
- shuffled final targets;
- shuffled demonstration action labels;
- query-order bagging;
- demonstration-order bagging;
- query-only prediction with demonstrations masked;
- frozen random compiler plus trained reader;
- parameter-matched ordinary transformer;
- host-supplied visible transition table as an explicitly labeled ceiling.

## 8. Mechanics-board gates

The CPU generator is qualified only if:

- two independent oracles agree on every generated episode;
- every cyclic triple has three distinct targets;
- every cyclic triple has an identical raw-token histogram;
- every cyclic triple has an identical action-erased transition stream;
- the action-agnostic and all-actions controls are at most `1/3`;
- every order twin has identical query-action histograms and different targets;
- the order-bagging control is at most `1/2`;
- state/action nonce renaming preserves oracle answers under the induced map;
- irrelevant demonstration permutation and padding preserve answers;
- malformed, duplicated, conflicting, truncated, and out-of-domain streams
  fail closed;
- model-visible payloads contain no offline key, physical operator identity,
  latent state identity, binding, target, or orbit membership.

## 9. Neural promotion gates

A neural pilot may be promoted only if, on one unopened manifest:

- exact cyclic-triple accuracy is at least 90%;
- both members are solved in at least 85% of order twins;
- treatment exceeds the action-agnostic control by at least 45 points;
- unseen action and state nonce recoding exactness is at least 90%;
- depth-five/six composition exactness is at least 80%;
- selected-binding scrambling costs at least 40 points;
- discarded-hypothesis scrambling costs at most two points;
- shuffled-target training remains at or below 40%;
- source deletion and tamper evidence pass;
- three fresh model/data seeds reproduce the result before Shohin integration.

Passing establishes bounded raw-token episode-specific binding and ordered
composition. It does not establish unrestricted language reasoning.

## 10. Parameter and continuation boundary

The complete model remains below 200 million parameters. A provisional maximum
budget is:

| Component | Added parameters |
|---|---:|
| token/event adapters | 18M |
| competitive state/action compiler | 16M |
| operator hypernetwork | 18M |
| hypothesis/refutation controller | 9M |
| executor/halt/serializer | 8M |
| **added ceiling** | **69M** |
| protected Shohin base | **125,081,664** |
| **complete ceiling** | **194,081,664** |

The protected base remains
`train/flagship_out/ckpt_0300000.pt`, SHA-256
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`.

No trillion-token continuation is authorized merely because this CPU board
passes. The mechanism must first survive neural controls, integrate into the
residual path, and learn under a mixed next-token plus state-transition,
operator-binding, counterfactual, and composition objective. Continuation data
must contain observable before/action/after structure such as code traces,
mathematical transformations, edits, plans, and counterfactual narratives.
Ordinary next-token loss alone is not assumed to train the mechanism.

## 11. Pre-implementation hostile-audit amendment

The following requirements were added before any neural model or corpus score.
They strengthen rather than relax the gates above.

1. **Six-case clusters are indivisible.** Every statistical family contains
   three cyclic action bindings for one query word and the same three bindings
   for a reordered query with the identical action-token bag. The world prefix
   is byte-identical between the two query orders for each binding. Promotion
   scores complete six-case clusters, not six independent examples.
2. **Late query custody is mandatory.** The world is compiled and hash-committed
   before the query suffix is materialized. Query-conditioned compilation of
   the world is prohibited. Multiple queries over one committed world must not
   change its serialized state.
3. **No host-built particles.** The treatment forward receives token IDs and a
   conventional attention mask only. The host may not supply spans,
   demonstration indices, state/action roles, candidate systems, particles,
   chart overlaps, query queues, identifiability flags, renderer IDs, family
   IDs, or latent cardinality.
4. **Surface controls are explicit.** Token bags, token-position histograms,
   lengths, masks, template features, action-erased streams, union graphs,
   path counts, endpoint majorities, sinks, and stationary-distribution
   controls must remain at or below the exact one-third symmetry ceiling.
5. **Particle custody is model-owned.** Particles start exchangeably. Random
   particle reindexing must reindex internal state equivariantly. No
   target-based matching or best-particle evaluator may affect autonomous
   scoring.
6. **Execution is one-shot.** One hard hypothesis selection, one ordered
   rollout, and one answer are scored. Invalid, abstaining, timed-out, and
   malformed cases remain in the denominator; there is no retry, beam rescue,
   repair, set-valued credit, or best-seed selection.
7. **Deletion is process-level for the decisive gate.** A fresh execution
   process receives only the allowlisted serialized workspace. It cannot read
   source tokens, embeddings, residuals, KV cache, compiler scratch,
   environment metadata, or source files.
8. **Fresh semantics, not fresh spelling.** Confirmation holds out complete
   physical operator triples and state conjugacy families in addition to
   renderers, nonces, and composition words.

The initial implementation is authorized only for deterministic CPU mechanics,
two independent oracles, symmetry controls, fail-closed parsing, world/query
commitment, and complete-cluster audits. Neural training remains unauthorized
until those receipts are frozen.
