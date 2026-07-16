# R12 Post-Commit Packet Transport V3 Preregistration

**Status:** frozen repair specification only. V2 is a post-commit NO-GO. No
neural fit, Shohin adapter, SFT, H100 job, workspace claim, or reasoning claim
is authorized unless v3 passes a clean post-commit run and an independent
adversarial audit.

## 1. Frozen question

Can an explicit four-cell packet be transported through source-free one-event
processes and answer late linear consumers after fresh output recoding, while a
rank-two motor packet fails at exactly chance on hidden coordinates, under an
OS-enforced interface and independently recomputed evidence?

The algebra, five public cells, 15 decisive cells, depths, state/motor packet
definitions, exact score thresholds, stale/shuffled/horizon controls, and
337-role cardinality per core are unchanged from v2. V3 changes custody only.
The v2 numerical result may not tune any cell, threshold, or control.

## 2. Claim-blocking v2 defects

V3 exists only to close these frozen defects:

1. a self-hash-consistent attacker could alter invocation evidence and make
   `verify_report` accept the forgery;
2. child processes could traverse the shared temporary parent and repository;
3. the challenge was deterministic from committed bytes before phase one;
4. the completed artifact/result hash had not been committed before audit.

One failure to close any item is a v3 NO-GO.

## 3. OS-enforced role boundary

Every writer, updater, reader, raw reader, and oracle must execute through
`/usr/bin/sandbox-exec`; absence or failure of that launcher is fatal.

The generated profile must:

- start from `(deny default)` and deny network access, so no finite path
  blacklist can be bypassed by another writable mount or directory;
- import neither `system.sb` nor `dyld-support.sb`, because both grant broad
  inherited reads; inline only the fixed non-filesystem dyld bootstrap
  operations and explicitly allow content reads from the root-owned Command
  Line Tools Python runtime, one exact committed role file, one fresh
  invocation directory, and the exact `/`, `/dev/urandom`, and `/dev/null`
  bootstrap objects required to launch that runtime;
- allow writes only below that invocation directory;
- use exact metadata grants only for path traversal to those allowlisted
  objects, never a metadata subtree over the run root or repository;
- execute the root-owned Command Line Tools Python binary rather than a
  user-writable framework, Homebrew, virtualenv, or repository interpreter;
- re-allow read access to exactly the committed seed-free role executable;
- re-allow only the invocation directory and its declared packet files;
- give stdin/stdout-only roles a fresh empty invocation directory;
- remove `PYTHONPATH`, set `HOME` and `TMPDIR` to the invocation directory,
  disable user site packages and bytecode writes, launch Python with `-I -S`,
  and use an allowlisted environment; and
- record a path-normalized policy hash in every invocation.

Before phase one, an executed sentinel probe under the identical policy must
prove that the child can read its allowed file but cannot read a parent-run
sentinel, cannot read `AGENT_RUNBOOK.md`, cannot read or write `/tmp`, cannot
write `/Applications`, cannot open a network socket, and cannot write outside
its invocation directory. Parent-run and repository directory enumeration must
also fail. Separate sentinels must also prove that writable
`/Library/Caches`, `/private/var/db/DiagnosticsReporter`,
`/Library/OSAnalytics/Diagnostics`, its data-volume firmlink,
`/Library/Logs/DiagnosticReports`, and
`/Library/Application Support/Blackmagic Design` content is neither readable
nor writable. `/etc/passwd` content and metadata and `/System` directory
listing must also fail, specifically detecting inherited baseline-profile
grants. The report must bind that probe. Directory listings remain necessary
but are no longer treated as sufficient isolation.

## 4. Post-phase unpredictable challenge

The canonical parent harness must sample the primary 256-bit nonce with
`secrets.token_bytes(32)` only
after both phase-one writers exit and the phase-one manifest is fsynced and
mode 0444. The nonce is then fsynced to a mode-0444 commitment file. Challenge
seeds are derived from:

```text
SHA256(domain || phase_one_manifest_sha256 || challenge_nonce)
```

Here `domain` and the lowercase manifest digest are ASCII bytes, the nonce is
the raw 32-byte value, and the deterministic challenge generator is seeded by
the unsigned big-endian integer represented by the digest's first eight bytes.

No nonce, seed, event, consumer, codebook, or challenge path may be visible to
the writer. Each updater sees exactly one current event and its one input
packet, but no source, prior/future event history, consumer, output recoding,
nonce, seed, codebook, or challenge path. The artifact records the nonce and
commitment hash so an auditor can replay the exact challenge.

The artifact proves commitment, ordering consistency, and deterministic seed
derivation, but it cannot cryptographically prove that the recorded bytes came
from an honest entropy source. The final external reviewer must witness the
canonical CLI path; no `external nonce` or entropy-attestation claim is
permitted.

The second complete core reuses the first core's committed nonce solely for
byte-identical replay. Its writers still run before receiving any challenge
input and remain OS-confined from the nonce and phase-two files.

## 5. Independent verifier

`pipeline/audit_post_commit_packet_transport_v3.py` must not import the
orchestrator or role executable. It must independently implement canonical
JSON, F_17 algebra, packet serialization, challenge parsing, transport, oracle,
reader, collision, horizon, stale, shuffled, and score calculations.

For each core it must reconstruct the exact expected invocation sequence and
validate every role record, including command/arguments, stdin/stdout hashes,
file bytes/modes/hashes, exit code, directory inventories, sandbox launcher,
normalized policy hash, and scientific source identity. It must reject
self-hash-consistent mutations of any evidence field and independently verify
both complete cores and final payload hashes.

This is evidence reconstruction under an honestly executed committed local
harness, not cryptographic proof of historical execution against a malicious
machine owner. The independent verifier can prove exact consistency and replay
all committed finite calculations; a later external reviewer witnesses the
canonical run and inspects the source. V3 must not claim hardware-rooted or
tamper-proof attestation.

Regression tests must mutate at least exit codes, extra visible files, packet
hashes, output permutations, missing records, sandbox fields, nonce binding,
both-core self-hashes, and publisher receipts. Every mutation must fail.

## 6. Canonical execution and custody

The scientific commit must include this preregistration, orchestrator, seed-free
role executable, independent verifier, and both test suites. The canonical path
is:

```text
artifacts/r12/post_commit_packet_transport_v3.json
```

The command must fail closed if any scientific path differs from `HEAD`. It
must build two full byte-identical cores before writing, run both the internal
verifier, and hand a read-only candidate to the independent verifier process.
Only that independent process may create the canonical artifact. It must first
reconstruct all evidence with Git verification enabled, then atomically create,
fsync, and chmod the canonical artifact 0444, and finally create a separate
read-only receipt at
`artifacts/r12/post_commit_packet_transport_v3.receipt.json`. The receipt must
bind the final artifact path, bytes, mode, MD5, file SHA-256, payload SHA-256,
scientific commit, verifier source SHA-256, committed verifier-path SHA-256,
and Git-verification mode. The parent process must contain no canonical writer
and must not accept a caller-supplied dictionary as proof of verification.

The completed result document must record commit, bytes, mode, MD5, file
SHA-256, payload SHA-256, exact scores, both invocation ledgers, and independent
audit verdict. The artifact and result record must be committed before the
final claim audit.

## 7. Pass boundary

A v3 pass validates only the exact symbolic packet algebra and reproducible
process-isolation behavior under the committed harness. It does not establish
tamper-proof historical attestation, learning, reasoning, language ability, or
novelty. Only after an independent audit passes may a separate learned
rank-thinned recurrent packet-lane preregistration be written.
