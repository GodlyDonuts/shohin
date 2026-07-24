# R12 EFC Multiworld Custody Result

**Date:** 2026-07-23

**Decision:** deterministic CPU phase rehearsal pass; filesystem/process
custody **NO-GO**; neural preregistration **NO-GO**

**Pretraining:** prohibited by the standing user hold

## Question

Can the EFC source-to-machine contract be exercised deterministically across
structurally disjoint train, development, and post-candidate confirmation
worlds, with two byte-distinct serializations, exact content roots,
fail-closed phase guards, and revalidation of every persisted payload?

For a consumed deterministic rehearsal, yes. For an official unopened
confirmation, learned compiler, or reasoning claim, no.

## Frozen Mechanics

The implementation is:

- `pipeline/episode_functor_source_renderers.py`
- `pipeline/episode_functor_multiworld_custody.py`
- `pipeline/run_episode_functor_multiworld_rehearsal.py`
- `pipeline/test_episode_functor_source_renderers.py`
- `pipeline/test_episode_functor_multiworld_custody.py`

The source boundary accepts two strict, byte-distinct encodings of the same
typed raw events:

1. canonical `efc-raw-world-evidence-v2` JSON; and
2. `EFC-RAW-LINES-V1` canonical line records.

Both decoders reject noncanonical encodings and normalize to the same
transition/observation event object. Both compile to the exact same 1,536-byte
machine. This proves serialization invariance only. It does not establish
natural-language, paraphrase, or novel-grammar transfer.

The multiworld protocol freezes:

- eight train worlds;
- four development worlds;
- a candidate root after both open splits;
- four confirmation worlds only after a strictly later beacon; and
- both source encodings in every split.

Every world receipt binds the exact child-beacon derivation, generation
attempt, accepted stream commitments, world-seed commitment, evidence bytes,
assessor latent bytes, machine bytes, source renderer, and structural
form. The canonical accepted-stream and admissibility payloads are
retained beside every world so their manifest hashes can be independently
recomputed rather than trusted as opaque digest claims. Before each phase
transition and after final publication, the rehearsal process revalidates the
protocol, event chain, manifests, roots, every world payload hash, the exact
machine-internal payload hash, renderer identity from source syntax, source to
machine recompilation, and the exact canonical structural form.

## Structural Isolation

The exact canonical form is a 25-byte value that canonicalizes over all:

- `5!` physical-state gauges;
- `3!` action permutations;
- `2!` observer permutations; and
- independent binary recodings of both observer outputs.

Generation compares these exact 25-byte values and rejects a candidate if its
canonical form appeared in any earlier split. SHA-256 is retained only as a
portable receipt for that form, not as the collision decision. The consumed
artifact has zero train/development,
train/confirmation, and development/confirmation structural overlap.

All sixteen worlds have a nontrivial empty-observation partition and five
future-behavior classes over five physical states. The board therefore
requires future action behavior to separate at least one initially merged
state.

## Temporal Boundary

Within one deterministic Python process, the phase API prevents confirmation
materialization until:

1. train and development manifests are frozen;
2. their combined root is published;
3. a syntactically valid candidate SHA-256 root is sealed; and
4. a different, strictly later beacon is supplied.

The confirmation child seed binds the candidate root and open-splits root.
Changing either the candidate or later beacon changes the confirmation
manifest root. Repeated phases, protocol mutation, same-round beacons, and
repeated beacon values fail closed.

The consumed runner uses fixed caller-supplied strings. It proves phase
ordering and content addressing, not that the later value was externally
unpredictable. The placeholder candidate root is not a model. Therefore the
confirmation split is consumed rehearsal data, not official confirmation.

The rehearsal also does **not** establish process custody. Assessor
`latent.json` files and deployed `machine.bin` files are intentionally retained
under the same artifact root for audit. No candidate process was launched, no
filesystem namespace or descriptor allowlist hid those files from a candidate,
and the runner cannot prove that a hypothetical candidate lacked access to
them. The report therefore records both
`filesystem_process_isolation_established=false` and
`assessor_secrets_candidate_inaccessible=false`.

## Consumed Artifact

Artifact:
`artifacts/r12/episode_functor_multiworld_rehearsal_v2_20260723`

Final report SHA-256:
`d391640620de8f80c5925ac9c0a67879483909c08b60dc3f9d424f51002cbe6b`

| Field | Result |
|---|---:|
| Train worlds | 8 |
| Development worlds | 4 |
| Confirmation worlds | 4 |
| Source encodings per split | 2 |
| Cross-split structural collisions | 0 |
| Worlds with nontrivial empty partition | 16/16 |
| Worlds with full future separation | 16/16 |
| Protocol root | `8268a36372d6dc68a3e5b2b538a1201cba16b095e8d3f651d832d2475ea67f64` |
| Open-splits root | `76f5690a606220716b54b8d2e2ae684acfe63545eb2a1feb0e52ea57ca04b9ce` |
| Confirmation manifest root | `e29bde2a07905e963a810445a3b5a5497f550748e5362d78f92845fdbb49a246` |
| Event-chain tip | `44c91d61b210c9c10ce60b7ac1ac296361897cb340f88548ecfdd1d713e1db3e` |
| External unpredictability established | no |
| Candidate process executed | no |
| Assessor secrets inaccessible to candidate | no |
| Filesystem/process isolation established | no |
| Neural preregistration authorized | no |
| Official confirmation | no |
| Pretraining authorized | no |

The expanded EFC and protected-workspace regression reports:

```text
245 passed, 1 skipped in 24.82s
```

The skip is the inherited platform-dependent strict runtime check.

## Scientific Decision

This milestone establishes:

- deterministic multiworld phase-order rehearsal;
- exact source-serialization invariance;
- exact small-cell structural split isolation;
- candidate/open-root binding of later confirmation generation;
- mechanics-only world admission; and
- complete receipts and in-process revalidation for all consumed worlds.

It does not establish:

- external time or future-beacon provenance;
- independent cryptographic verification of a beacon signature;
- process-level assessor isolation;
- candidate blindness to assessor latent or machine files;
- transfer across genuinely different source languages;
- a learned neural compiler;
- superiority to recurrent/direct-machine controls;
- Shohin-native reasoning; or
- permission to continue pretraining.

The next legitimate custody step is an externally witnessed candidate seal,
filesystem/process isolation of candidate versus assessor, and then a
cryptographically verified future public beacon. The next scientific step,
after those custody gates and a genuinely different source-language family,
is a small neural compiler preregistration with mandatory renderer, structure,
recurrence, direct-machine, answer-cache, and oracle controls. Until then the
EFC architecture family remains open, but neural preregistration remains
**NO-GO**.
