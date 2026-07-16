# R12 Post-Commit Packet Transport V2 Preregistration

**Status:** IMPLEMENTED; CANONICAL RUN PENDING A CLEAN SCIENTIFIC COMMIT. Two
precommit engineering canaries passed but have no scientific standing because
their implementation was not yet bound to a commit. The canonical artifact
path remains `artifacts/r12/post_commit_packet_transport_v2.json`. No neural
fit, Shohin adapter, SFT, H100 job, workspace claim, or reasoning claim is
authorized until a post-commit canonical run and independent claim audit both
close.

## 1. Question

Can an exact harness enforce the mechanism absent from PCIF v1: a packet is
written before a challenge exists, updated one event at a time after the source
is unavailable, and consumed by a fresh reader that must itself emit a
late-recoded answer?

This remains a symbolic positive/control test. It does not ask whether a neural
network can learn the packet.

## 2. Exact system

Retain the v1 system over `F_17^4`, its public subspace
`W = span(e_0,e_1)`, invertible affine events, consumers, public and decisive
challenge construction, depths `1,2,4,8,9`, and exhaustive 83,521-source
space.

The state writer emits `(x_0,x_1,x_2,x_3)`. The favorable rank-two motor writer
emits `(x_0,x_1,0,0)`. Both packets have the same serialized schema and four
field elements. The motor remains an information-poor negative control, not an
information-matched neural comparator.

For both arms, the updater applies the supplied single affine event directly
to the current four-field packet. Public updates preserve exact public answers.
Decisive challenges expose the missing hidden coordinates of the motor arm.

## 3. Four-role process contract

The canonical runner must execute four explicit subprocess roles from a
precommitted source tree. Their executable is physically separate from the
orchestrator and contains no canonical/alternate challenge seed or import of
the v1 challenge generator:

1. **Writer:** consumes source vectors through standard input and emits packet
   JSONL. It receives no challenge seed, event, consumer, or codebook. It exits
   before challenge generation.
2. **Stateless updater:** receives exactly one packet file and one affine event,
   emits the next packet file, and exits. It receives no source, source path,
   prior/future event, consumer, codebook, oracle output, or scorer feedback.
3. **Fresh reader:** receives only the terminal packet, one consumer, and one
   post-commit 17-symbol permutation. It emits the recoded symbol. It receives
   no source or event history and may not emit an unrecoded answer for the
   scorer to transform.
4. **Oracle:** separately receives source vectors and the complete challenge
   and emits recoded truth. Its output is unavailable to writer, updater, and
   reader roles.

The orchestrator may compare reader and oracle files byte-for-byte. It may not
repair, recode, or reinterpret reader outputs.

Every subprocess invocation, role argument list, exit code, input hash, output
hash, and source-tree hash must be included in the result. Environments are
allowlisted and contain no project secrets.

Every updater invocation directory contains exactly `packet_in.jsonl` before
launch and exactly `packet_in.jsonl` plus its immutable `packet_out.jsonl`
after exit. Every reader invocation directory contains exactly one immutable
`terminal_packet.jsonl`. A previous or future packet may not coexist in either
role's working directory.

## 4. Two-phase custody

Before challenges are generated:

1. write all state and motor packet streams;
2. validate packet schemas and exact row counts;
3. fsync and hash both streams;
4. terminate both writers;
5. freeze an append-only phase-one manifest containing packet and code hashes.

Only then derive the challenge seed from the immutable phase-one manifest hash
and a precommitted domain separator, then generate affine events, consumers,
and independent output permutations. No numeric canonical seed exists before
the manifest freeze. Changing only the phase-two domain must leave phase-one
packet bytes and hashes identical.

For each challenge, copy the committed packet stream into an isolated working
path. Invoke one fresh updater process per event. The reader sees only the
terminal path. Intermediate packet hashes form the transport transcript.

## 5. Frozen scientific cells

Use the same five public and 15 decisive cells as v1. Each cell covers all
83,521 sources. Generate a separate nonidentity uniform permutation per cell
after phase-one commitment.

Required scores:

1. public state and motor arms: 83,521/83,521 reader-emitted recoded symbols;
2. decisive state arm: 83,521/83,521 at every depth;
3. decisive motor arm: exactly 4,913/83,521 at every depth;
4. incremental state packets equal direct exact affine application for every
   source and challenge;
5. every decisive motor cell has a packet-collision witness with distinct
   reader-required symbols.

## 6. Executed decoys

Every decoy must run through the same role interface and write a scored row:

1. **source pointer:** packet with any pointer/source field is rejected;
2. **query-visible writer:** writer invocation with consumer/codebook arguments
   is rejected by its role parser;
3. **event-history updater:** updater invocation with more than one event or a
   history argument is rejected;
4. **source-visible updater/reader:** any source argument is rejected;
5. **stale packet:** skip one nonidentity event; at least one decisive cell must
   fall below exact state accuracy;
6. **shuffled packet:** a deterministic packet-row permutation must fail exact
   state accuracy;
7. **horizon reader:** the canonical reader interface is fed a packet from a
   transport capped at eight events. It must remain exact for cells through
   depth 8 and fall below exact at depth 9; the reader receives no depth field;
8. **scorer-side recoding:** an unrecoded reader output file must fail schema or
   byte comparison rather than being transformed by the scorer.

## 7. Frozen gates

The result passes only if all role/process, phase-order, exact-score,
direct-versus-incremental, collision, decoy, deterministic replay, and
seed-independence gates pass. The canonical command must build two complete
core reports and require byte identity before it may set the replay gate or
write an artifact. The verifier requires the exact frozen top-level and gate
schemas. One failure closes v2. Thresholds may not be relaxed after any result
exists.

## 8. Authorized files

Only these new scientific objects are authorized:

```text
pipeline/post_commit_packet_transport_falsifier.py
pipeline/post_commit_packet_transport_roles.py
pipeline/test_post_commit_packet_transport_falsifier.py
artifacts/r12/post_commit_packet_transport_v2.json
R12_POST_COMMIT_PACKET_TRANSPORT_V2_RESULT.md
```

The implementation may import pure algebra and packet-schema helpers from the
committed v1 module. It may not import a solver into a future neural reader or
reinterpret symbolic correctness as learned reasoning.

## 9. Successor boundary

A complete v2 pass authorizes only writing, not running, a separate tiny neural
preregistration. That document must match utilized packet entropy, parameters,
examples, scalar-label count and rank, optimizer updates, FLOPs, and search
budget across PCFT and same-information controls. It must freeze one uniform
variable-size model evaluated on unseen states, renderings, events,
compositions, state dimensions, and depths.

No Shohin or H100 experiment is authorized by a v2 symbolic pass.
