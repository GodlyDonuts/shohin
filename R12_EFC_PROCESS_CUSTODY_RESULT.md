# R12 EFC Process-Custody Result

**Date:** 2026-07-24  
**Protocol:** `efc-signed-process-rehearsal-v1`  
**Decision:** **PASS on macOS CPU custody mechanics; Linux and neural claims
remain unadmitted.**

This result does not authorize a neural fit, a GPU job, continuation
pretraining, or a Shohin reasoning claim.

## Question

Can one publicly frozen deterministic compiler receive only public evidence,
compile the exact 1,536-byte EFC machine in a default-deny process, exit, and
only then be assessed against a secret expected machine that was never visible
to the candidate?

For the exact macOS protocol below, the answer is **yes**.

## Retired Attempt

The first externally timed attempt used authorization commit `c10d28f` and
NIST chain-2 pulse `1,873,040`. It rejected before candidate launch because a
relative CLI authorization path was not normalized before Git-blob lookup.
No artifact root, candidate, assessor, or report was published. That attempt
is void and its pulse is retired.

The repair:

- resolves the fixed authorization path once before validation;
- repository-bounds every Git-blob path;
- includes a direct relative-path regression; and
- passes the complete frozen EFC suite, `195/195`, plus Ruff and
  `py_compile`.

The repair source was committed and published as `e594575`.

## External Timing And Source Freeze

| Receipt | Value |
|---|---|
| Frozen source root | `b6c8ae3ec00fcc4f98dc696491acb5e31de798f26583024ab8292da8dac01042` |
| Authorization commit | `aff33d48c670be24fb69d2cf89e1010ab27c29eb` |
| Authorization SHA-256 | `71fde2c583b96285f7478c10d617d64df10954471688ade3405c07b41eb9f4f7` |
| GitHub PushEvent ID | `15682040576` |
| GitHub PushEvent time | `2026-07-24T04:48:14Z` |
| NIST chain / pulse | `2 / 1,873,055` |
| NIST pulse time | `2026-07-24T04:55:00.000Z` |
| Publication-to-pulse separation | `406 seconds` |
| Pinned NIST certificate DER SHA-512 | `528943a555f5f8ca54423be6dfb95925a35c7b552046420e7d7cd072058a14d6536ad3a8e9754b6582f164a90b0cd86a65d659f5426a2659a947595d1c816c8c` |
| Snapshot payload SHA-256 | `62e261405b8fae1a158e34cf7de7b79cb140abcc2785281f57e5ae3f401501f8` |
| Snapshot file SHA-256 | `78bb7319a93e6c0d81f889a96de6b26b46abac3ed17c0796eeafcd8263e68d7f` |

The runner independently fetched the certificate from the official NIST
HTTPS endpoint and verified both the target and immediately preceding RSA
signatures, target output hash, previous-output link, previous
precommitment reveal, exact chain/index/timestamp, certificate identifier,
authorization pin, published origin head, frozen Git blobs, and the matching
public GitHub event.

## Candidate And Assessor Result

One independently generated admissible world was rendered in three distinct
source languages:

| Source language | Evidence bytes | Evidence SHA-256 | Exact machine |
|---|---:|---|---:|
| Canonical JSON events | 2,444 | `65a8792bdf300e4460c8b6dcb570c6f8188b90d77198b29b2d8ac08d931f5668` | yes |
| Strict line events | 1,194 | `f8b23896c4d693c122bb6791d4125ed43a13df9ece6c24b72d27660bffec6cef` | yes |
| Cycle/program language | 858 | `f12e2f8169ffb9bdf8b94fcd5db93cb5d3fff0c76557335b944173a865b76373` | yes |

Each source was handled by a fresh candidate process whose initial directory
contained only `evidence.bin`. Each candidate exited with code zero and
produced only `machine.bin` and `candidate_receipt.json`. The expected machine
was not in any candidate directory.

Only after each candidate exited did a fresh assessor process receive the
candidate and expected machines. All three assessors exited with code zero,
and the parent independently recomputed complete byte equality rather than
trusting assessor booleans.

| Aggregate | Value |
|---|---:|
| Candidate processes | 3 |
| Exact candidate/expected matches | 3/3 |
| Source-language machine agreement | 3/3 |
| Machine bytes | 1,536 |
| Machine SHA-256 | `2c1503db5ba41ce10d8dfcfebad7e22e858d3f6a5d905c8662f5a89b7b260a13` |
| Candidate/assessor order | candidate exit before assessor start |

## Blindness And Artifact Gates

The unmocked macOS `/usr/bin/sandbox-exec` probe under the same frozen
default-deny policy passed every gate:

- allowed input read;
- local output write;
- secret-file read blocked;
- secret-directory listing blocked;
- repository content read blocked;
- outside write blocked; and
- network socket blocked with a policy-specific denial.

All candidate and assessor roles ran with sandbox enforcement, exact
hash-bound role source, `-I -S`, closed input/output file sets, zero stderr,
zero stdout, and zero exit status. The world admission checks passed
transition completeness, bijection, noncommutativity, observer shape,
nontrivial empty-observation partition, and full future separation while
inspecting zero query fields.

The atomic artifact contains 27 pre-report regular files, no symlinks, no
undeclared temporary root, and a complete content map:

| Artifact receipt | SHA-256 |
|---|---|
| Pre-report file-map root | `c9ba244ca67daa1658c6be435d4ff895276dc4e1cb94694b4adf03710d00aea0` |
| Final report | `f2c4cd16a246f5d5da7116512601d4705ba21c14e23c582cdd95d50265450c04` |

The local post-run verifier independently recomputed all 27 file hashes,
file-map root, authorization and snapshot hashes, frozen source root, target
and predecessor NIST verification, origin head, 406-second publication
separation, role exits, sandbox flags, candidate blindness, assessor order,
and all three complete machine equalities.

An independent hostile post-run audit found no P0, P1, or P2 loopholes for the
narrow macOS CPU custody-mechanics claim and returned `PASS / FREEZE`. It
independently extracted all 13 frozen source blobs from authorization commit
`aff33d48c670be24fb69d2cf89e1010ab27c29eb`, replayed the generator and all
three compilers, verified all twelve persisted machine copies and their
internal self-hashes, and repeated the NIST and artifact verification.

Local artifact:

`artifacts/r12/episode_functor_process_custody_1873055`

## Claim Boundary

This pass supports only:

> One publicly frozen deterministic compiler transformed three source
> languages into one exact finite machine while blinded from a later
> assessor secret under an externally delayed world seed and an unmocked
> macOS default-deny process boundary.

It does **not** establish:

- a learned source-to-machine compiler;
- Shohin ownership of the compiled machine;
- natural-language grounding;
- unseen-world or unseen-task neural generalization;
- architecture-native autonomous execution;
- broad or general reasoning;
- Linux process custody; or
- any reason to resume continuation pretraining.

The local process and timing receipts are also not hardware attestation. They
assume the host administrator and operating-system trust base are not
malicious, and they do not claim crash-durable publication.

## Next Gate

1. Replay the role boundary unmocked on Linux with root-owned Bubblewrap,
   network-namespace isolation, and Landlock when Newton is reachable.
2. Freeze a separate learned compiler preregistration with unseen worlds,
   held-out renderers, post-seal challenges, source deletion, exact machine
   scoring, capacity-matched cache controls, generic-recurrence controls, and
   key/transition causal interventions.
3. Run any learned first fit in isolation from the protected step-300k
   checkpoint. The user hold on continuation pretraining remains absolute.
