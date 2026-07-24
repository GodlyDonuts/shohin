# R12 EPISODE Functor Compiler - Codex Analysis

**Date:** 2026-07-23
**Input theory SHA-256:** `e3c7420fd7aef36834cee79af58afe681359e1cbf5ca35a1ad855d14bfcabd36`
**Frontier handoff text SHA-256:** `a83536547b121d000cd8c28d9ce4beb059a661f49a294eb6492d7db0e61e3531`
**Disposition:** admit the architecture redesign as a research hypothesis; reject the
old-board two-answer-cache no-go as proven in its present form.

This analysis does not authorize neural fitting, a GPU job, development access,
reasoning promotion, or continuation pretraining.

## Executive Judgment

The draft identifies the right architectural direction:

1. compile raw episodic evidence into an explicit anonymous machine;
2. keep action keys separate from action transitions;
3. reveal challenges only after the machine is sealed;
4. parse query grammar separately from machine execution;
5. execute ordered composition in a fixed generic tensor runtime; and
6. test causal interventions on the committed machine fields.

That is materially stronger than asking one residual stream to simultaneously
discover state, bind opaque actions, execute a program, and emit an answer.

The draft also correctly warns that a finite set of known future queries can be
cached. However, its direct application to the current EPISODE custody protocol
is incomplete. The external author explicitly did not inspect the local source.
The actual compiler receives only `development_worlds.jsonl`. The late query
contains a start-state nonce, an action word, multiplicities, order, and depth;
none of those challenge fields is present in the sealed world input.

The theorem

```text
C(x) = (f(x,q_1), ..., f(x,q_k))
```

requires the compiler to know the complete query coordinates `q_i`, or to derive
them from `x`. Merely observing that the frozen dataset realizes two queries per
world does not prove that a source-only compiler can construct those two cache
entries.

## Direct Audit Of The Current Board

The frozen development custody bundle contains:

| Quantity | Value |
|---|---:|
| Sealed worlds | 192 |
| Late queries | 384 |
| Realized late queries per world | 2 |
| Query depth 5 rows | 192 |
| Query depth 6 rows | 192 |
| State nonces per world | 8 |
| Action nonces per world | 3 |
| Full nonempty depth-1-through-6 support | 8,736 |

For each world, the two realized queries share one hidden start state and one
action multiset but use different orders. If all starts and all action words at
the realized depth were admissible, the query support would be:

```text
depth 5: 8 * 3^5 = 1,944
depth 6: 8 * 3^6 = 5,832
```

Therefore the current source-only compiler is not lawfully handed a two-entry
query index.

This does not fully rescue the current advancement board. The source world and
the two challenges are generated from one deterministic PRNG trajectory before
the state is sealed. Each exact world occurs once. A sufficiently pathological
model could exploit generator correlation or fixed-manifest identity rather
than learn reusable dynamics. More importantly, the protocol does not
demonstrate reuse across a large independently sampled post-seal challenge
family. The current board remains a useful binding/order diagnostic, but it is
not the strongest possible identification test for a reusable causal machine.

A source audit sharpens this concern: candidate worlds are accepted or rejected
after inspecting sampled hidden-query outcomes. The later custody split
therefore proves that compiler, executor, and assessor read different files; it
does not prove that challenge information did not influence world selection.
The old corpus is rejected for advancement and retained only as a consumed
diagnostic.

## What Is Accepted

### 1. Explicit anonymous machine state

The proposed categorical Moore machine is a legitimate architectural object:

```text
initial_state
action_key
action_next
observer_key
observer_answer
active masks
```

Anonymous state indices respect gauge freedom. Behavioral equivalence,
interventions, and conjugacy-consistent transformations should be scored
instead of raw latent equality.

### 2. Key/transition separation

Binding and dynamics become independently manipulable. This makes key-only,
transition-only, and compensated key-plus-transition interventions exact and
auditable.

### 3. Attached training and detached scoring

The soft or straight-through hard machine may remain attached during
optimization. Scoring must serialize a fixed-shape hard machine and execute it
without source tokens, residuals, KV state, targets, or assessor feedback.

### 4. Independent post-seal challenges

This is the most important protocol correction. The challenge seed and bytes
must not exist until after the machine artifact is sealed and the compiler
process has exited. Multiple challenges must reuse identical machine bytes.

### 5. Causal quotient and resource accounting

Small-world mechanics should independently enumerate reachable states, compute
future-indistinguishability classes, prove that `K` is sufficient, and verify
that the challenge family separates every quotient class. The committed-machine
bit budget must be compared with explicit query-indexed cache budgets.

## Required Corrections

### The old-board cache falsifier must be scoped honestly

An oracle cache built using query identities or labels before sealing proves
only that an intentionally weakened protocol is underidentified. It does not
show that the current source-deleted compiler can lawfully form the cache.

A valid source-only cache falsifier must do one of the following:

1. derive the two exact future query identities from world bytes alone;
2. prove that those identities are deterministic functions of the world under
   the admitted generator; or
3. store answers for the full admissible query support within the frozen state
   budget.

If none holds, the claimed two-answer counterexample is rejected.

### The executor must be architecture-native

The runtime may use fixed tensor gather, categorical transition, masking, and
readout operations. It may not contain task semantics, a host transition
oracle, verifier repair, answer search, or query-specific branches. A Python
reference executor is acceptable only as an independent mechanics oracle, not
as the deployed reasoning mechanism.

### Explicit machines do not automatically imply reasoning

A direct hypernetwork can emit a table that overfits source identity. Promotion
requires unseen worlds, bindings, renderers, word lengths, multiplicities, and
composition motifs, plus intervention signatures and qualified matched
controls.

### Query support must dominate the cache budget

The new board must freeze exact values for `K`, `M`, `P`, answer alphabet,
retained key precision, and maximum challenge depth. It must then show that a
query-indexed cache under the same committed-byte ceiling cannot cover the
challenge support.

For the current nonempty depth-one-through-six grammar, the exhaustive
three-bit answer table is only 3,276 packed bytes. It therefore fits inside a
hypothetical 16 KiB machine budget. Depth six is adequate for mechanics but not
for a byte-capacity exclusion. One corrected proposal scores through depth
twelve, where the depth-zero-through-twelve universe contains 6,377,288
queries and needs 2,391,483 packed answer bytes.

### Current start-state binding must be represented

The current late query supplies an opaque start-state nonce. The proposed hard
schema retains an `initial_state`, action keys, and observer keys, but no
state-key records. It therefore cannot bind the current query start after
source deletion. A revision must either retain fixed-shape opaque
`state_key[K,d_key]` records or change the board so the source fixes the initial
state and the late query never names one. These are different capability
contracts and must be chosen before source freeze.

## Authorized Next Work

Only CPU mechanics and protocol falsifiers are authorized:

1. implement a conditional cache audit that distinguishes realized queries
   from source-derivable admissible queries;
2. split source generation from challenge generation with an independent
   post-seal seed;
3. enumerate exact causal quotients and transition monoids on small worlds;
4. implement two independent hard-machine runtimes and intervention suites;
5. freeze exact machine and answer-cache bit accounting; and
6. test whether raw source evidence identifies every required transition
   without oracle segmentation.

The two current Python auditors do not yet satisfy item 4. They are separately
written but share sorted-key canonicalization, integer state slots, row-major
tables, and left-to-right updates. The replacement gate requires sealed-artifact
C and Rust runtimes with different state/transition representations and a third
independent assessor.

No neural source freeze or GPU run is justified until those gates pass.

The implemented CPU audits now execute the categorical machine exactly on all
1,920 frozen packets across 960 committed worlds. They count 8,736 nonempty
late queries per world, 26,208 answer-table bits, and eight exact quotient
classes. A lawful canonical world-only two-entry cache covers 0/384 development
queries, while a deliberately leaky cache given hidden query identities and
assessor answers reaches 384/384. Those measurements strengthen the custody
correction without constituting neural evidence.

## Current Decision

The existing 907,269-parameter causal workspace remains a control and custody
reference. Its pending neural pilot is not launched.

The Episodic Functor Compiler is admitted as the next architecture hypothesis,
subject to CPU falsification. It is not yet a reasoning mechanism, and it does
not alter the continuation-pretraining hold.

## Process-Custody Implementation Update

The theory's machine and phase boundary have now been translated into a
standalone deterministic candidate, a later independent assessor, and three
source languages:

1. canonical shuffled transition/observation events;
2. strict line records; and
3. canonical permutation-cycle programs plus answer-labelled observer
   partitions.

All three compile to one fixed 1,536-byte deployed machine in local mechanics
tests. This demonstrates that the proposed machine has a precise executable
contract; it does not demonstrate learned identification.

A hostile audit rejected the first custody draft because a snapshot-supplied
certificate could self-authorize, Git commit time did not prove publication
before the pulse, the parent trusted an assessor verdict, the network probe
accepted unrelated socket errors, role/input paths had replacement windows,
and Linux had no unmocked replay. The revised candidate now:

- requires a fixed Git-tracked authorization blob;
- pins the exact NIST certificate DER SHA-512 and exact future pulse index;
- requires a matching public GitHub PushEvent for the exact published head;
- copies and hash-binds each role source inside the sandbox;
- recomputes complete machine equality in the parent;
- accepts only policy-specific network-denial outcomes;
- rejects symlinks, nonregular closure entries, oversized inputs, and
  30-second timeouts; and
- keeps Linux explicitly unadmitted pending an unmocked Bubblewrap plus
  Landlock replay.

The remaining scientific barrier is unchanged: deterministic compilation is
the mechanics ceiling. A neural EFC must infer a correct anonymous machine
from raw source evidence on unseen worlds and source languages, beat
capacity-matched cache and generic recurrence controls, and survive
post-seal challenges and key/transition interventions. No neural freeze or
pretraining is authorized by the process work.

## Externally Timed Process-Custody Result

The first externally timed attempt is retired as void because a relative CLI
authorization path failed before candidate launch. The repaired source passed
the complete `195/195` regression and was frozen at source root
`b6c8ae3ec00fcc4f98dc696491acb5e31de798f26583024ab8292da8dac01042`.

Authorization commit `aff33d48c670be24fb69d2cf89e1010ab27c29eb`
was publicly witnessed by GitHub PushEvent `15682040576` at
`2026-07-24T04:48:14Z`. Exact precommitted NIST chain-2 pulse `1,873,055`
arrived at `04:55:00Z`, 406 seconds later. Both consecutive pulse signatures,
certificate pin, output hash, chain link, and precommitment reveal verified.

The unmocked macOS run passed. Three fresh candidate processes received,
respectively, canonical JSON events, strict line events, and a cycle/program
source. All three produced the same exact 1,536-byte machine, SHA-256
`2c1503db5ba41ce10d8dfcfebad7e22e858d3f6a5d905c8662f5a89b7b260a13`.
Each later assessor agreed, and the parent independently recomputed all three
complete byte comparisons. The default-deny blindness probe blocked secret
read/list, repository read, outside write, and network while permitting only
the declared input and local output. Independent post-run verification
recomputed all 27 pre-report file hashes and the complete receipt chain.

Final report SHA-256 is
`f2c4cd16a246f5d5da7116512601d4705ba21c14e23c582cdd95d50265450c04`;
the full boundary is in `R12_EFC_PROCESS_CUSTODY_RESULT.md`.

This closes the macOS process-custody mechanics gate only. Linux remains
unadmitted because Newton still fails at DNS. More importantly, the candidate
was deterministic CPU code. No learned compiler, neural generalization,
architecture-native autonomous execution, or Shohin reasoning result exists.
The next scientific gate is a separately frozen learned source-to-machine
compiler with unseen-world/renderer tests and matched cache/recurrence
controls. Continuation pretraining remains forbidden.

## Identifiable Learned-Architecture Update

The theory has now been translated into a train-only neural architecture
candidate rather than only deterministic compiler mechanics:

- an identifiable `K=8, M=3, P=2, |Y|=4` board with one hidden cell per
  action/observer relation and singleton version spaces under two independent
  solvers;
- six algebraically distinct action families and an eight-cell renderer
  metagrammar;
- a proof-carrying witness compiler whose one global key transport controls
  both copied opaque keys and transition/observer axes;
- a zero-parameter Birkhoff/balanced-transport projector that emits only
  lawful action permutations and balanced observers;
- a matched trainable, permutation-equivariant relational completer that is
  not law constrained and can therefore attribute missing-cell completion to
  learned computation rather than host projection;
- a separately learned raw-byte late-query grammar parser;
- an exact 1,536-byte source-deleted machine executed by independent C and
  Rust runtimes; and
- a candidate collation boundary that accepts only
  `CandidateSource.source`, rejecting family/split/renderer/target-bearing
  objects; and
- a separate train-only supervisor whose sole join key is source SHA-256 and
  whose labels cannot enter the compiler forward object.

Two hostile P0 findings were accepted and corrected. The production neural API
no longer accepts source and query together. Compilation and hard sealing occur
first; only then may raw query bytes be passed, and they resolve directly
against the sealed `HardFunctorKeys`. The old source-derived soft-assignment
bridge is absent from the production path. The train-time straight-through key
transport now uses the same global one-to-one assignment as hard sealing, with
an adversarial regression proving attached slot argmax and sealed assignment
cannot diverge.

Shohin is also connected rather than merely counted. The protected 300k
checkpoint was loaded read-only under SHA-256
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`.
The first verification design used an importable sentinel and was rejected by
hostile audit as replayable. It has been removed. Receipt generation now
re-hashes the checkpoint, compares its configuration and tensors against the
in-memory parent, compares nonpersistent buffers and the exact module graph
against a fresh `model.py` construction, rejects runtime hooks/overrides, and
compares the executing code manifest with a fresh load of the bound source.
The fixed clean-runtime manifests bind function code, defaults, keyword
defaults, recursive function-valued closures, annotations, function
attributes, referenced globals and builtins, selected external inference and
container dispatch, local transport behavior, class properties, ordered module
topology, selected residual blocks, and published feature width. The actual
protected checkpoint passes; parameter, RoPE-buffer, hook, topology,
class-method, property, method-default, transport, builtin, and
referenced-callable mutations each invalidate verification. This is a Python
execution receipt rather than malicious-host, native-kernel, or hardware
attestation.
Frozen residuals from blocks 9, 19, and 29 feed both source and query
perception through exact tokenizer-to-byte offsets. The current 888-source
pilot has exact nonoverlapping offset coverage; the longest source is 2,420
tokens and 440/888 sources exceed the parent context. The frozen trunk uses
disconnected deterministic 2,048-token windows that reset attention and RoPE,
so it supplies local perceptual features only; the trainable byte encoder
carries global context.

The exact connected receipts are:

| Component | Solver arm | No-host arm |
|---|---:|---:|
| Frozen Shohin | 125,081,664 | 125,081,664 |
| Source compiler | 3,595,792 | 3,821,202 |
| Query parser | 728,993 | 728,993 |
| Added trainable parameters | 4,324,785 | 4,550,195 |
| Complete instantiated system | 129,406,449 | 129,631,859 |
| Headroom under 200M | 70,593,551 | 70,368,141 |

The minimal no-host arm is an attribution probe rather than a final capacity
claim. Named constructor-verified profiles now reserve a 35,625,267-parameter
wide treatment at 160,706,931 total parameters and a 60,552,883-parameter
maximum treatment at 185,634,547 total parameters. The maximum profile uses a
512-wide 8+4-layer compiler, a 640-wide eight-round relational completer, and
a 320-wide four-layer query parser, leaving 14,365,453 parameters below the
200M ceiling. Two structural alternatives are now staged rather than treating
this budget as width alone. The first is a Hankel-shift causal code whose
state representation is its finite future-behavior signature and whose action
table is decoded by left-shift agreement. It fits inside the existing maximum
receipt. The second is a recurrent sealed predictive compiler that allows
provisional machine contradictions to revise source perception; it may reach
approximately 197.0M total parameters and therefore requires a distinct
adapted-base receipt and stronger controls. Both remain unimplemented
research treatments with explicit open-loop, scrambled-incidence, direct
hypernetwork, adapter-only, and oracle controls. They are not capability or
novelty claims.
These profiles make it possible to diagnose undercapacity without
changing a treatment after scoring. A learned executor, recurrent memory, or
changes to normally fixed transformer components remain allowed only as
separately named causal arms with matched controls.

This is still **not a reasoning result**. The public-law projector performs
permutation/balance completion as an architectural solver. It may support a
claim about learned parsing, binding, machine induction, and post-seal
composition, but not a claim that Shohin learned the missing-cell deduction.
The no-host learned-completion arm now exists and its forward path is tested,
including source-to-machine integration, state-recoding equivariance,
non-forced invalid-table behavior, and attached gradients. It has not been
fit. A hostile audit found that coordinate-first argmax broke hard equivariance
on exact ties. Hard ties now fail closed before straight-through or detached
hardening, while unique-max hard state/action/observer/answer recodings pass
exactly. Transfer to different law families remains mandatory for a stronger
claim. Candidate/supervisor process custody, complete resource accounting,
controls, source freeze, and neural qualification remain unfinished.
Therefore no neural fit, development read, reasoning promotion, GPU job, or
continuation pretraining is authorized.

The train-only compiler qualification path is now executable at the loss and
optimizer-step level. Labels remain in a separately typed supervisor and join
only after the source-only forward by source SHA-256. Exact metrics isolate the
five hidden completion cells per rendered source from exposed-cell copying.
The trainer updates only the source compiler, requires the verified protected
trunk by default, and rejects nonfinite gradients. A canonical resource-receipt
schema also exists. There is still no custody launcher, frozen arm-specific
resource receipt, neural fit, score, or development access.
