# R12 EFC Process-Custody Preregistration

**Status:** CPU custody/mechanics candidate pending public source freeze and a
strictly later verified NIST Beacon pulse. This document does not authorize a
neural fit, a GPU job, a reasoning claim, or continuation pretraining.

**Protected base:** Shohin step 300,000 remains immutable at SHA-256
`211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`.
No code in this gate loads or writes that checkpoint.

## Question

Can one frozen candidate compiler receive only public world evidence, compile
the exact 1,536-byte EFC machine in a default-deny process, exit, and only then
be assessed against a secret expected machine that was never visible to the
candidate?

This is a custody question. Exact compilation by the deterministic CPU
candidate is not evidence that a neural model learned to reason.

## Consumed Attempt Ledger

Exact chain-2 pulse `1,873,040` at `2026-07-24T04:40:00.000Z` was consumed
under authorization commit `c10d28f`. Snapshot payload SHA-256 was
`776799d658e0469431605463b7ed6f76984a653140af24e56483093aae7636d8`.
The run rejected before candidate launch because the relative CLI spelling of
the fixed authorization path reached Git-blob verification without absolute
normalization. No output root or report was published. This attempt is void,
the old authorization is retired, and its pulse may not be reused. The source
fix resolves and repository-bounds the path before both fixed-path and Git
blob checks; a relative-path Git regression is mandatory before the next
source freeze.

## Frozen Source Languages

The same independently generated finite world is rendered three ways:

1. canonical JSON transition and observation events;
2. strict line-oriented transition and observation records; and
3. a closed cycle/program language whose actions are canonical disjoint-cycle
   programs and whose observers are answer-labelled state partitions.

The third language is structurally different from the first two. It declares
complete algebraic programs rather than serializing event rows. All three must
compile to byte-identical deployed machines. Malformed, noncanonical, partial,
duplicate, out-of-range, and renderer-forged inputs fail closed.

## Process Roles

### Candidate compiler

The candidate role is a standalone standard-library program. Its fresh
invocation directory initially contains exactly:

```text
evidence.bin
```

It may create exactly:

```text
machine.bin
candidate_receipt.json
```

It cannot import the Shohin repository. The expected machine, latent world,
beacon snapshot, checkpoint, assessor code, and other source-language
renderings are not present in its invocation directory.

### Assessor

The assessor starts only after the candidate process has exited. Its fresh
invocation directory initially contains exactly:

```text
candidate_machine.bin
expected_machine.bin
```

It validates the complete fixed-width machine header, masks, dimensions,
padding, and self-hash before reporting exact byte equality. A valid but
different machine must produce a clean negative assessment rather than a
parser failure.

### Blindness probe

An unmocked role under the same policy must demonstrate:

- allowed input read succeeds;
- local write succeeds;
- secret-file read is blocked;
- secret-directory listing is blocked;
- repository read is blocked;
- outside write is blocked; and
- network socket use is blocked.

On macOS the frozen v1 policy uses `/usr/bin/sandbox-exec`, a default-deny
Seatbelt profile, an exact root-owned Python runtime, one exact role script,
and one writable invocation directory. Repository directories receive
metadata traversal only; no repository subtree content grant exists.

Linux is not admitted by inference from the macOS result. A Linux claim
requires a separate unmocked Bubblewrap network-namespace plus Landlock replay
on Newton. The Linux launcher must use root-owned, non-group/world-writable
Bubblewrap and Python executables, copy the Git-frozen role and Landlock
sources into the isolated root, run Python with `-I -S`, and bind the hashes of
those exact copies into every role receipt.

## Source Freeze And Future Pulse

The exact source file hashes are reduced to one `candidate_source_root` in
`artifacts/r12/efc_process_custody_authorization_v1.json`. The authorization
and source must be committed and published at `origin` before beacon
consumption.

The authorization freezes one exact future chain-2 pulse index and the exact
lowercase SHA-512 digest of the NIST certificate DER before publication. The
only admitted pulse rule is:

```text
exact precommitted chain-2 pulse index, whose timestamp is at least 60 seconds
after GitHub's public PushEvent timestamp for the exact authorization commit
```

The runner verifies:

- canonical snapshot encoding;
- both consecutive NIST RSA pulse signatures;
- the certificate identifier, its equality to the precommitted DER digest,
  and equality with the certificate independently fetched from NIST's
  official HTTPS certificate endpoint;
- output hash;
- previous-output link;
- previous-pulse precommitment reveal;
- frozen chain and pulse identity;
- candidate source-root equality;
- clean local worktree;
- local `HEAD` equality with the published branch at `origin`; and
- byte equality between every frozen local source, its `HEAD` blob, and the
  exact copied sandbox role;
- byte equality between the fixed authorization path and its `HEAD` blob;
- a public GitHub PushEvent whose branch and head equal the published
  authorization commit; and
- the minimum public-PushEvent-to-pulse delay.

The GitHub event is an externally hosted, server-stamped HTTPS publication
receipt, not a cryptographic RFC 3161 timestamp. The report preserves its
event ID, `created_at`, reference, and HTTP response date. Absence, lag, or
ambiguity fails closed.

The pulse output and frozen candidate source root seed one independently
generated world. World admission inspects mechanics only and receives no
challenge, candidate prediction, or answer.

## Acceptance Gates

All gates must pass in one atomic report:

1. three candidate processes start and exit cleanly;
2. every candidate starts with exactly one public evidence file;
3. no candidate input receipt names the expected machine;
4. all three candidate outputs are valid 1,536-byte machines;
5. the parent independently recomputes each complete byte comparison and
   agrees with the later assessor's hashes, verdict, and exit code;
6. all three machine byte strings are identical;
7. every assessor starts after its candidate exits;
8. the blindness probe passes every check;
9. the independent world passes all frozen mechanics-admission checks;
10. the fixed authorization blob is present at published `HEAD`, its pinned
    certificate matches, and the NIST pulse satisfies the frozen future rule;
11. the report accounts for every file present before final-report
    publication.

Any missing output, undeclared file, symlinked input, overlapping root,
noncanonical receipt, stale source hash, unpublished commit, old pulse,
sandbox failure, unexpected certificate, absent public push event, machine
mismatch, assessor disagreement, nonregular artifact entry, oversized input,
timeout, or source-language disagreement rejects the run.

## Honest Claim Boundary

A pass supports only:

> One publicly frozen deterministic compiler transformed three source
> languages into one exact finite machine while blinded from a later assessor
> secret under an externally delayed world seed.

It does not show that Shohin compiled the machine, that any learned parameters
generalize, that natural-language reasoning emerged, or that continuation
pretraining should start. Neural preregistration remains **NO-GO** until this
custody mechanism is replayed cross-host and a qualified learned compiler plus
matched controls is frozen.
