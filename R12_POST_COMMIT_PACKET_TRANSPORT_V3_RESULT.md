# R12 Post-Commit Packet Transport V3 Result

**Decision:** PROTOCOL PASS FOR EXACT PACKET TRANSPORT; NO LEARNED-REASONING
CLAIM. Scientific commit
`36906818f17aa4f03b9f5622dbcb65110ae95abf` completed the canonical v3
command. The independently implemented verifier reconstructed the complete
evidence twice, required the committed scientific sources through Git, and
alone published the read-only artifact and receipt.

Canonical command:

```bash
python3 pipeline/post_commit_packet_transport_falsifier.py run \
  --out artifacts/r12/post_commit_packet_transport_v3.json
```

Canonical result:

```text
protocol:                      R12-PCPT-F17x4-v3
status:                        evidence_reconstructed_and_published
scientific commit:             36906818f17aa4f03b9f5622dbcb65110ae95abf
gates:                         35/35 PASS
sandbox probes:                27/27 PASS
roles per core:                337
role counts per core:          writer 5 / updater 227 / reader 64 /
                               oracle 40 / raw-reader 1
public cells:                  5
decisive cells:                15
public state / motor:          83,521/83,521 / 83,521/83,521
decisive state / motor:        83,521/83,521 / 4,913/83,521
horizon depths 1,2,4,8:        83,521/83,521 each
horizon depth 9:               4,913/83,521
first core payload SHA-256:    fd212c4a648356dece649ff49d32a2a685c2d00ff2a25cc9cb321dd0e0e3102b
second core payload SHA-256:   fd212c4a648356dece649ff49d32a2a685c2d00ff2a25cc9cb321dd0e0e3102b
artifact payload SHA-256:      b64a5ee3498e378d7f574fcb032ec34c6671914eafef4b1d0e616ef6565d2dbd
Git verification:              required_and_passed
```

Published files:

| File | Bytes | Mode | MD5 | File SHA-256 |
|---|---:|---:|---|---|
| `artifacts/r12/post_commit_packet_transport_v3.json` | 2,273,380 | `0444` | `a84f126ad934edf0b86ae30b8f2c4813` | `6f3846d6a58bca7d61e753fe1297c9f7090c29ef44f7585716a206dca3bba685` |
| `artifacts/r12/post_commit_packet_transport_v3.receipt.json` | 1,232 | `0444` | `fd99cb4edd4be05ea6f0a41619801895` | `55de668bdf4f98b04ad0ce7d20b4e4b06baa53332f2e66b004076fe2b61d63f2` |

The receipt payload SHA-256 is
`41eba8d49b8c06e0f93a0b84d9fbc54f293052660c56bf7f61fa3d3deb638a01`.
The independently committed verifier source SHA-256 is
`0eacabce52cf8bbe14ca5b73120ad37cc227ac814c12cc73ea66d3f8e2c23478`.
A fresh post-publication invocation of `verify-publication`, with Git
verification required, also exited successfully.

## Final Independent Audit

A fresh read-only adversarial audit of committed publication commit `61c627f`
returned **GO**. It verified `HEAD == origin/main`, checked every reviewed file
against its Git blob, reran all 38 focused tests, reran `verify-publication`
with Git verification required, reconstructed the role counts and score
arithmetic, and confirmed the artifact/receipt hashes and local `0444` modes.
The auditor also forged a score in both replay cores and recomputed every
affected self-hash; independent semantic reconstruction rejected that
self-consistent forgery.

The audit records two non-blocking limitations. Git stores the blobs as `100644`,
so a fresh clone does not preserve the local publication mode, and neither
historical execution nor entropy provenance is cryptographically attested.
Both are outside the frozen claim. Final authorization is GO for a separately
preregistered learned architecture experiment and NO-GO for describing PCPT v3
as learned reasoning, novelty evidence, or tamper-proof attestation.

## What Passed

The protocol demonstrates an exact finite mechanism that writes a four-symbol
packet, updates it through isolated one-event processes, deletes privileged
source access, and answers late queries from the terminal packet. The packet
transport remains exact while a deliberately insufficient motor control and a
depth-eight transport cap fail at the preregistered boundaries. The parent
cannot publish canonical evidence; the independent implementation reconstructs
all role byte streams, affine chains, scores, challenge binding, sandbox
evidence, and deterministic replay before publication.

## Claim Boundary

This is a pass for **exact symbolic packet transport and evidence custody**.
It does not show that a neural network can learn the packet, choose its own
operations, halt, generalize in language, or reason. It is not a claim of a
novel memory primitive, historical tamper-proof attestation, or SoTA
capability. The finite affine board has a known sufficient coordinate basis and
therefore serves as a mechanism/control gate, not as novelty evidence.

The pass authorizes only a separately preregistered learned architecture lane.
That lane must distinguish durable state learning from autonomous control,
include equal-budget favorable controls, delete source/KV access, test unseen
queries and recodings, and require fresh researcher-written interactions. It
may not alter the immutable 300k checkpoint or claim reasoning from fit loss.

## Defect History

Three precommit adversarial rounds closed default-allow filesystem escapes,
nonce timing and caller-control defects, self-published receipts, incomplete
independent reconstruction, and claim-boundary drift. The first commit-bound
run from `5903fbf` then failed closed before publication because a relative
artifact path was passed to a verifier launched from a fresh cwd. Commit
`3690681` resolves publication paths before process launch and adds defect-21
regression coverage. The complete suite remained 38/38 passing before the
successful canonical run.
