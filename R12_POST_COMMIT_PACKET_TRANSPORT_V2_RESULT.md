# R12 Post-Commit Packet Transport V2 Result

**Decision:** NO CANONICAL RESULT YET. Two exhaustive precommit engineering
canaries passed byte-identically, but they have no scientific standing because
the implementation was not committed before execution. This document will be
replaced only after a clean post-commit canonical run and replay.

## 1. Precommit canary custody

The superseded canary is retained locally as
`artifacts/r12/post_commit_packet_transport_v2_precommit_canary.json` so the
failed custody sequence remains auditable. Its former canonical-path hashes
were:

```text
file SHA-256:    908a6ff7039360dad72a7e571ecb6c25797709b8ac55e247d8decfdfbc895b42
payload SHA-256: f8e740faa413f7aa511b8dc3642dd78469b8d8452226235cd4cf6a65ca011ed0
recorded code:   efde07cf7a6179adb147a9f448086d219b9a496e685c9135cbf4c9321a20fa18
```

These values are engineering evidence only. They must not be cited as a v2
pass.

## 2. Independent review findings

Two precommit reviews returned **REVISE BEFORE COMMIT** because the canary or
first repair:

1. was not bound to the reviewed implementation;
2. streamed packet bytes in memory instead of invoking file-to-file updaters;
3. compared parsed symbols rather than canonical bytes;
4. skipped all events in the stale-packet control instead of exactly one;
5. gave a special horizon reader an explicit depth field;
6. sampled derangements rather than uniform nonidentity permutations; and
7. asserted phase order without a measured manifest-to-challenge binding;
8. left packet history visible in shared updater/reader directories;
9. loaded the canonical challenge seed into the same executable as every role;
10. did not make a second complete run a prerequisite for `pass`;
11. allowed incomplete top-level and gate schemas through report verification;
    and
12. derived cell and process expectations from observed results rather than a
    frozen five-public/15-decisive layout;
13. retained only the first run's subprocess evidence after replay;
14. still allowed evidence-empty reports with fabricated true gates; and
15. defined the canonical challenge seed before the phase-one commitment.

The scientific harness now fails closed on committed source identity, moves
packet state only through immutable files, uses the canonical reader interface
for the bounded-horizon control, performs canonical byte-row scoring, and
records a read-only phase-two manifest bound to the phase-one manifest, uses a
physically separate seed-free role executable, gives every updater and reader
a one-packet invocation directory, gates the exact frozen cell/process layout,
requires the exact report schema, and performs two full byte-identical core
runs before `pass` can become true. The final artifact embeds the entire second
core report, the verifier independently replays result layouts, scores, role
cardinalities, manifest hashes, and post-commit seed derivation, and no numeric
canonical seed exists before the phase-one manifest is frozen.

## 3. Canonical result fields

The following remain intentionally blank until execution from the clean
scientific commit:

```text
scientific commit: PENDING
artifact mode:     PENDING
artifact bytes:    PENDING
file SHA-256:      PENDING
payload SHA-256:   PENDING
canonical replay:  PENDING
independent audit: PENDING
```

No neural fit, Shohin adapter, SFT, H100 job, workspace claim, or reasoning
claim is authorized by a precommit canary.
