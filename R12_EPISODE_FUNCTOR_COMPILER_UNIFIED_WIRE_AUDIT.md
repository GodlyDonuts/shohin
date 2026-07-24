# R12 EFC Unified Deployed-Wire Audit

**Date:** 2026-07-23

**Decision:** CPU contract pass; neural compiler preregistration **NO-GO**

**Pretraining:** prohibited by the standing user hold

## Question

Can the corrected two-beacon Episodic Functor Compiler protocol use the exact
1,536-byte deployed machine wire, delete source before a later query exists,
execute the same sealed bytes in independent C and Rust runtimes, and match a
third assessor without silently restoring the old answer-cache or shared-RNG
defects?

For a consumed synthetic rehearsal, yes. For a neural reasoning claim or
official confirmation board, no.

## Implemented Contract

The new implementation is:

- `pipeline/episode_functor_independent_world.py`
- `pipeline/episode_functor_source_renderers.py`
- `pipeline/episode_functor_wire_protocol.py`
- `pipeline/run_episode_functor_wire_rehearsal.py`
- `pipeline/episode_functor_multiworld_custody.py`
- `pipeline/run_episode_functor_multiworld_rehearsal.py`
- `pipeline/test_episode_functor_independent_world.py`
- `pipeline/test_episode_functor_source_renderers.py`
- `pipeline/test_episode_functor_wire_protocol.py`
- `pipeline/test_episode_functor_multiworld_custody.py`

The independent world generator imports no original EPISODE generator,
protocol, serializer, or runtime. It uses a separate counter-mode SHA-256
stream, Fisher-Yates permutation construction, mechanics-only candidate
admission, and exact commitments for every mechanics, key, observation-order,
demonstration-order, and renderer stream actually consumed.

Public evidence is `efc-raw-world-evidence-v2`. It contains shuffled typed
transition and observation events only. It does not provide state, action, or
observer key inventories, positionally aligned observer rows, latent state
indices, query fields, or targets for a future query. The deterministic CPU
compiler must infer the three opaque key classes, prove exact cardinalities,
reconstruct every transition/observation cell, and fail closed on duplicates,
omissions, unknown keys, or schema taint.

The machine is exactly 1,536 little-endian bytes:

- 16 state slots, eight action slots, and eight observer slots;
- active counts, masks, fixed zero padding, and one initial-state slot;
- opaque uint64 state/action/observer keys;
- flat action-by-state uint8 transition cells;
- observer-by-state uint64 answer cells; and
- SHA-256 over bytes 0 through 1,503 in bytes 1,504 through 1,535.

Every machine byte has a frozen receipt. For the consumed five-state,
three-action, two-observer cell:

| Receipt | Bytes |
|---|---:|
| Total accounted | 1,536 |
| World-source direct/derived, including final hash | 207 |
| Unaccounted | 0 |

The protocol freezes source and executable hashes before the world beacon.
The world root commits both public evidence and the sealed assessor latent
hash. The machine hash is retained in memory and rechecked before source
deletion and every challenge. Protocol files are rechecked before every phase.
Events form a SHA-256 chain. Runtime outputs are staged, compared, and only
then atomically published with no replacement.

## Temporal Rehearsal

The consumed runner uses fixed synthetic beacons and therefore proves call
order, not external unpredictability:

1. commit protocol and attested runtime executable hashes;
2. consume synthetic world beacon;
3. generate query-free world evidence;
4. compile exactly one machine;
5. poison and delete public source;
6. consume a strictly later synthetic challenge beacon;
7. commit abstract coordinates;
8. render one canonical deployed query wire;
9. execute independently in C and Rust;
10. seal byte-identical transcripts;
11. open third-assessor relation-composition answers.

Two challenge beacons reuse the exact same machine and compile count remains
one. The deployed wire has one renderer. The earlier two-renderer abstract
axis was removed because both values serialized to the same binary query and
created hidden duplicates.

## Consumed Result

Artifact:
`artifacts/r12/episode_functor_wire_rehearsal_v1_20260723`

Final report SHA-256:
`7a141efbbccdbd8328e2b38708061d9b2a4f3be8a43c3617fe4e8a413e4fd35c`

| Field | Result |
|---|---:|
| Machine bytes | 1,536 |
| Machine SHA-256 | `e4eafa3cfd205c377515cf7619f5ae723a591b5a2ed2eeab3888477bcf652000` |
| Protocol root | `f03e4630a798701eac942fcb72d3b7729f5d6e83e4943555d22e9e8501d78b08` |
| World root | `0165ba88592f50f074009baf766d75e3e4d0f714637febeb8ff9257dfded16a0` |
| Machine root | `d7a359da5494259b0feea2b017b94fec3bd933a36eb2c7b376b8ffd69de30d08` |
| Challenge panels | 2 |
| Queries per panel | 100 |
| Duplicate deployed queries | 0 |
| Compiler invocations | 1 |
| C/Rust transcript disagreements | 0 |
| Third-assessor disagreements | 0 |
| Empty-observation classes | 4 of 5 |
| Future-behavior classes | 5 of 5 |
| Query fields inspected during world admission | 0 |
| Event-chain tip | `08f83b6fdbc1a46d1cfd596da2cf1cea6f7c8a9297829a10c8e9ccf19f6b2d6d` |

The observers are noninjective at depth zero: five physical states collapse to
four empty-observation classes. Future action words separate all five states.
This kills the old identity-observer shortcut without making the CPU fixture
ambiguous.

## Source-Renderer And Multiworld Extension

The compiler now accepts two strict byte-distinct source serializations:
canonical raw-event JSON and `EFC-RAW-LINES-V1` line records. Both normalize to
the same typed transition/observation events and compile to the exact same
1,536-byte machine. Six malformed line mutations fail closed. These are two
source serializations, not two natural-language families; the result proves
parser/serialization invariance only.

A separate consumed multiworld rehearsal freezes eight train and four
development worlds, seals a placeholder candidate root, and only then
materializes four confirmation worlds from a different, strictly later
synthetic beacon. Every split alternates both source serializations.

World structures are rejected across all splits when their exact 25-byte
canonical forms collide. Canonicalization exhaustively quotients
physical-state, action, and observer permutations plus independent binary
observer-output recodings. SHA-256 is only the receipt for that exact form, not
the collision decision. All sixteen worlds are structurally distinct, have
nontrivial empty-observation partitions, and separate all five physical states
under future behavior.

Every published payload is revalidated before a later phase can proceed.
Persisted source bytes must recompile byte-for-byte to the persisted machine,
and the source-renderer receipt must match JSON versus line syntax. This is
durable in-process tamper detection, not process isolation.

Artifact:
`artifacts/r12/episode_functor_multiworld_rehearsal_v2_20260723`

Final report SHA-256:
`d391640620de8f80c5925ac9c0a67879483909c08b60dc3f9d424f51002cbe6b`

| Field | Result |
|---|---:|
| Train/development/confirmation worlds | 8 / 4 / 4 |
| Source serializations in every split | 2 |
| Cross-split structural collisions | 0 |
| Protocol root | `8268a36372d6dc68a3e5b2b538a1201cba16b095e8d3f651d832d2475ea67f64` |
| Open-splits root | `76f5690a606220716b54b8d2e2ae684acfe63545eb2a1feb0e52ea57ca04b9ce` |
| Confirmation manifest root | `e29bde2a07905e963a810445a3b5a5497f550748e5362d78f92845fdbb49a246` |
| Event-chain tip | `44c91d61b210c9c10ce60b7ac1ac296361897cb340f88548ecfdd1d713e1db3e` |
| Assessor secrets inaccessible to candidate | no |
| Filesystem/process isolation established | no |

The complete relevant regression command reports:

```text
245 passed, 1 skipped in 24.82s
```

The skip is the inherited platform-dependent strict runtime check.

## Hostile Audit Findings

The first audit correctly rejected four issues in the initial uncommitted
implementation:

1. arbitrary binaries could be passed under the names `c` and `rust`;
2. assessor latent mechanics were not bound into the world root;
3. renderer-distinct abstract coordinates collapsed to duplicate deployed
   queries; and
4. evidence exposed explicit key inventories and positional observer rows.

All four were repaired before this result. Runtime binaries are now attested,
latent bytes are sealed, the deployed protocol has one renderer and exact
duplicate rejection, and raw v2 event evidence must be structurally inferred.
The stream receipt was also corrected to commit actual accepted-candidate
streams rather than nominal unused labels.

Two audit findings remain intentionally open:

### External temporal unpredictability

The consumed beacons are fixed test values. A local class can prove phase order
but cannot prove that a future challenge was unknowable before machine seal.
An official run needs an externally sourced, independently recorded future
beacon whose provenance and retrieval time are outside the candidate process.

The multiworld phase guards narrow this gap but do not close it: the candidate
root in the consumed rehearsal is a placeholder and the later beacon remains a
caller-supplied string. A legitimate confirmation opening needs an externally
witnessed candidate publication followed by cryptographic verification of a
preannounced future beacon round.

### Process custody

The consumed artifact intentionally keeps assessor `latent.json` and deployed
`machine.bin` files beside each world for audit. No candidate process was run,
and no filesystem namespace or file-descriptor allowlist prevented a
hypothetical candidate from reading those files. The phase API and hash checks
therefore prove deterministic ordering and tamper detection only. They do not
prove candidate blindness or assessor isolation.

### Source-language transfer

Canonical JSON and strict line records exercise independent byte parsers but
encode the same typed fields directly. They do not test source-language
identification, paraphrase robustness, declaration binding, or transfer to an
unseen grammar. Those remain neural evaluation obligations.

### Neural identification

The CPU compiler now proves that raw typed events uniquely identify the
machine. It does not prove that Shohin can learn that compilation from source
tokens, preserve it under renderer/nonce recoding, or beat matched recurrent
and direct-machine controls on unseen worlds. The assessor latent remains in a
separate assessor lane by design; an official run still needs process-level
sandboxing so candidate/compiler/executor processes cannot access it.

## Decision Boundary

This result establishes:

- exact deployed-wire serialization;
- query-free independent consumed world generation;
- structural identification from shuffled typed raw events;
- source deletion before challenge execution;
- one-machine/multiple-later-query reuse;
- independent C/Rust execution;
- third-assessor agreement;
- nontrivial empty-observation structure; and
- complete byte, source, binary, and event receipts.

It does **not** establish:

- external beacon unpredictability;
- an official unopened multiworld confirmation split;
- natural-language or cross-renderer compilation;
- a learned neural compiler;
- superiority to mandatory controls;
- Shohin-native reasoning; or
- permission to continue pretraining.

## Remaining Gate

Before writing a neural preregistration:

1. freeze an external future-beacon provider and provenance verifier;
2. replace the consumed placeholder candidate with an externally witnessed
   candidate root before the frozen future round;
3. isolate candidate, compiler, executor, and assessor processes with explicit
   file-descriptor and filesystem allowlists;
4. add genuinely different source-language families beyond the two typed
   serializations;
5. freeze mandatory matched controls and complete-system parameter accounting;
6. have an independent auditor reproduce every root and receipt; and
7. submit the resulting neural compiler preregistration for user review.

## Historical v1 artifact boundary

The original unified-wire consumed artifact and its published roots are
historical evidence bound to commit `91934e7` and the v1 wire protocol. The
current parser uses the v2 protocol root and binds
`pipeline/episode_functor_source_renderers.py`. Current source must not be
claimed to reproduce the historical v1 protocol root unless commit `91934e7`
is checked out. The new v2 multiworld artifact is reported separately above.

Until then, EFC remains an admissible architecture family with a qualified CPU
contract, not a reasoning result.
