# RSP-C2: Source-Deleted Residual Packet Control

**Status:** frozen pre-seed salvage contract on 2026-07-15. No production C2
seed has been derived, no beacon pulse has been fetched for C2, and no C2
board, training row, fit, transcript, or score exists.

RSP-C2 is a clean replacement for the closed RSP-C1 experiment. It tests a
bounded arithmetic controller. It is not evidence of general reasoning,
latent reasoning, universal context compression, or a new computational
primitive.

## 1. Conditional authorization

C2 remains unauthorized until the source-scheduled prerequisite has completed
and two independently implemented scorers agree on every integer count,
probability, integrity gate, and exact two-sided McNemar test. Both must report
all locked gates passing and `advance_to_internalization=true`.

Authorization requires one immutable, read-only prerequisite receipt with
exact schema `source_scheduled_reasoning_confirmation_pass_v1` and these exact
fields:

```text
schema
confirmation_contract_sha256
confirmation_result_sha256
primary_scorer_sha256
independent_scorer_sha256
primary_score_receipt_sha256
independent_score_receipt_sha256
advance_to_internalization
all_locked_gates_pass
independent_recomputation_complete
result_immutable
scorers_agree
```

All five booleans must be true. Scorer implementation hashes and score-receipt
hashes must differ. Missing evidence, disagreement, malformed custody, or a
near miss closes C2 without requesting a beacon or generating a seed.

## 2. Absolute C1 quarantine

C2 may not consume, import as a production dependency, copy, transform,
filter, perturb, or regenerate from any C1 concrete artifact. This prohibition
includes C1:

- board rows, sources, packets, trajectories, answers, hashes, and seeds;
- treatment or sham data, manifests, audits, caches, and token receipts;
- checkpoints, transcripts, scores, fit seeds, intervention assignments, and
  case order;
- production `v1` generator, auditor, evaluator, scorer, tests, or job files.

No C1 hash may be a C2 input or an exclusion oracle. C2 generation must not
query C1 cases to search for a disjoint or favorable replacement. The bounded
compiler/updater hypothesis may be independently restated, but C2 production
software must live in separately versioned paths and must not execute C1 code.

## 3. Pre-seed implementation freeze

After prerequisite authorization, but before any production beacon request,
one C2 freeze commit must contain and bind:

- this preregistration and the C1 closure;
- the offline seed-derivation implementation and tests;
- exact C2 production generator, independent auditor, evaluator, two
  independent scorers, job wrappers, runtime/container identity, and hash
  binder;
- exact board geometry, renderer inventory, value ranges, operation grammar,
  held-out structures, case ordering, uniqueness rules, and abort behavior;
- exact training counts, matching rules, packing implementation, fit
  hyperparameters, evaluation prompts, parsing, interventions, metrics, gates,
  and resource ledger;
- toy-only tests that never call the production generator or use a production
  seed.

The full commit OID must be pushed to the designated remote. An independent
observer writes an external server-observed push receipt using exact schema
`rsp_c2_freeze_push_receipt_v1`:

```json
{
  "branch": "BRANCH",
  "commit_oid": "FULL_LOWERCASE_GIT_OID",
  "observed_at": "YYYY-MM-DDTHH:MM:SS.mmmZ",
  "observer_id": "OBSERVER_ID",
  "observer_implementation_sha256": "LOWERCASE_SHA256",
  "pushed_at": "YYYY-MM-DDTHH:MM:SS.mmmZ",
  "ref": "refs/heads/BRANCH",
  "remote_ref_evidence_sha256": "LOWERCASE_SHA256",
  "remote_ref_oid": "FULL_LOWERCASE_GIT_OID",
  "remote_url": "https://github.com/OWNER/REPOSITORY.git",
  "schema": "rsp_c2_freeze_push_receipt_v1"
}
```

The actual file is compact canonical JSON with keys sorted, no insignificant
whitespace, one trailing newline, no duplicate keys, no extra fields, and no
write bits. `ref` must equal `refs/heads/{branch}`; `commit_oid` and the
server-observed `remote_ref_oid` must be identical full OIDs. `remote_url` must
be a canonical HTTPS GitHub `.git` URL. `observed_at` must be no earlier than
`pushed_at` and no more than 300,000 milliseconds later. The observer identity,
observer implementation hash, and immutable remote-ref evidence hash are
mandatory. Git author or committer timestamps and command-line timestamp or
hash overrides are not admissible substitutes.

No production seed derivation may begin until one hour after that externally
observed push time. Replacing, amending, force-pushing, or superseding the
freeze commit after its push receipt exists closes C2. A new attempt requires a
new version and preregistration before another future pulse is knowable.

## 4. Toy-only pre-seed testing

Before seed revelation, tests may use only small synthetic fixtures with
obviously nonproduction timestamps, fake hashes, and fake beacon records. They
may test determinism, parsers, invariants, failure paths, and custody mechanics.

Forbidden pre-seed actions include:

- invoking any C2 production-board or production-data generator;
- substituting a candidate seed into production generation code;
- computing a production board hash or inspecting candidate production rows;
- probing the model on candidate C2 cases;
- fetching a NIST pulse for C2 before the freeze and one-hour delay are fixed.

## 5. Future beacon and deterministic seeds

Seed derivation is performed only by
`pipeline/derive_residual_packet_c2_seeds.py`. That program is offline and must
not contain HTTP, socket, or beacon-fetching code. Its only production input
interface is:

```text
--freeze-push-receipt READ_ONLY_CANONICAL_JSON
--prerequisite-pass-receipt READ_ONLY_CANONICAL_JSON
--beacon-json READ_ONLY_RAW_NIST_JSON
--beacon-verification-receipt READ_ONLY_CANONICAL_JSON
--out EXCLUSIVE_NEW_PATH
```

There is no free-form commit, ref, pushed-at timestamp, or receipt-hash
argument. Parsed receipt fields are authoritative.

Let:

```text
F = full pushed C2 freeze commit OID
T = externally observed pushedAt(F) + 3,600,000 milliseconds
R = SHA-256 of the immutable freeze push receipt
P = raw SHA-256 of the independently passing prerequisite receipt
Q = confirmation_result_sha256 embedded in that receipt
J = raw SHA-256 of the selected NIST beacon response
V = raw SHA-256 of a dual-verifier beacon-validation receipt
```

After time `T`, two independent validators retrieve and validate the first NIST
Randomness Beacon 2.0 pulse at or after `T`. They must agree on the raw pulse
hash, signed chain and certificate validation, chain index, pulse index,
timestamp, 60,000-millisecond period, status code, and 512-bit `outputValue`.
The raw pulse is written read-only. Their compact canonical validation receipt
uses exact schema `rsp_c2_nist_beacon_verification_v1` and exact fields:

```json
{
  "beacon_raw_sha256": "LOWERCASE_SHA256",
  "certificate_id": "NIST_CERTIFICATE_ID",
  "certificate_sha256": "LOWERCASE_SHA256",
  "chain_index": 1,
  "output_value_sha256": "LOWERCASE_SHA256",
  "period_ms": 60000,
  "pulse_index": 1,
  "schema": "rsp_c2_nist_beacon_verification_v1",
  "signature_value_sha256": "LOWERCASE_SHA256",
  "status_code": 0,
  "time_stamp": "YYYY-MM-DDTHH:MM:SS.mmmZ",
  "time_stamp_ms": 1,
  "validators": [
    {
      "certificate_valid": true,
      "chain_valid": true,
      "evidence_sha256": "LOWERCASE_SHA256",
      "implementation_sha256": "LOWERCASE_SHA256",
      "raw_beacon_sha256": "LOWERCASE_SHA256",
      "signature_valid": true,
      "validated_at": "YYYY-MM-DDTHH:MM:SS.mmmZ",
      "validator_id": "VALIDATOR_A"
    },
    {
      "certificate_valid": true,
      "chain_valid": true,
      "evidence_sha256": "DIFFERENT_LOWERCASE_SHA256",
      "implementation_sha256": "DIFFERENT_LOWERCASE_SHA256",
      "raw_beacon_sha256": "LOWERCASE_SHA256",
      "signature_valid": true,
      "validated_at": "YYYY-MM-DDTHH:MM:SS.mmmZ",
      "validator_id": "VALIDATOR_B"
    }
  ],
  "validators_agree": true
}
```

The receipt has no missing or extra fields. Each validator record has exactly
the shown fields. Validator identifiers, implementation hashes, and evidence
hashes must be pairwise distinct. Both records must bind the exact raw beacon
hash and independently assert successful signature, certificate, and chain
validation with nonempty evidence hashes. Bare booleans or a hash-only receipt
are inadmissible. The receipt must exactly match the raw pulse's raw hash,
certificate identifier, chain and pulse indices, timestamp in string and
millisecond forms, period, status, output-value hash, and signature-value hash.

The selected pulse must satisfy:

```text
0 <= pulse.timeStamp_ms - T < 60000
pulse.period == 60000
pulse.statusCode == 0
```

There is no fallback beacon, alternate pulse, manual seed, modulo search, or
second draw. If the selected pulse cannot be authenticated, C2 waits. Using any
other pulse closes C2.

The exact base preimage is the following byte concatenation. `field(x)` means
ASCII `x` followed by one NUL byte. Integer fields use unsigned base-10 ASCII;
the beacon output uses lowercase hexadecimal. `U`, `H`, `S`, and `O` are the
parsed freeze receipt's `remote_url`, `ref`, canonical `pushed_at`, and canonical
`observed_at`.

```text
SHA256(
  b"SHOHIN-RSP-C2-SEED-BASE-v2\0" ||
  field(F) ||
  field(U) ||
  field(H) ||
  field(S) ||
  field(O) ||
  field(R) ||
  field(P) ||
  field(Q) ||
  field(J) ||
  field(pulse.chainIndex) ||
  field(pulse.pulseIndex) ||
  field(pulse.timeStamp_ms) ||
  field(lowercase(pulse.outputValue)) ||
  field(V)
)
```

Call that 32-byte digest `B`. For each exact ASCII label below, derive:

```text
seed_digest(label) = SHA256(
  b"SHOHIN-RSP-C2-SEED-v2\0" || B || b"\0" || label
)
seed_integer(label) = unsigned_big_endian_integer(seed_digest(label))
```

The only labels are:

```text
board
training
observation
sham
fit-a
fit-b
```

The derivation writes one exclusive, canonical, read-only
`rsp_c2_seed_receipt_v2`. It records the computed raw SHA-256 of every receipt
and the raw beacon, all parsed freeze identities, all exact beacon bindings, and
both validator evidence records. Existing output, mutable or changing inputs,
duplicate JSON keys, noncanonical receipts, malformed hashes, inconsistent
branch/ref/OID/timestamps, a nonpassing prerequisite, nonindependent scorers or
beacon validators, mismatched beacon evidence, or a pulse outside the first
eligible slot aborts before any seed receipt exists.

## 6. Frozen experiment geometry

C2 has exactly 256 evaluation cases, 64 in each fixed stratum order:

1. renderer OOD;
2. value OOD;
3. operation-order OOD;
4. length OOD.

The separately versioned generator frozen in `F` must fully enumerate the
ranges, renderers, held-out operation bigrams, uniqueness rules, and positive-
intermediate constraints. It must deterministically consume only the C2
`board` seed and must not read C1 artifacts or model outputs.

Training has exactly 4,096 semantic programs: 1,024 length two, 2,048 length
three, and 1,024 length four. Each program contributes one compiler row and one
updater row per instruction. Updater observations are independently sampled,
mathematically wrong for the first operation, and contain no correct arithmetic
transition or source final answer.

Treatment maps each source to its exact packet. Sham uses a deterministic,
stratified derangement. Updater rows are byte-identical. Treatment and sham
must have identical prompt order, per-row encoded length, response-token
multiset, supervision-mask geometry, packed sequence count, discarded-token
count, packed forward positions, and supervised target-token count.

## 7. One-shot materialization

After the seed receipt exists, production proceeds through one sealed,
noninteractive workflow:

1. Snapshot and hash every frozen input and runtime dependency.
2. Invoke the production board generator once with the `board` seed.
3. Write the board through exclusive creation, fsync it, set it read-only, and
   record artifact and canonical-row hashes without printing row content.
4. Run the independent board audit once.
5. Mechanically replace only predeclared board-hash placeholders in downstream
   consumers using the precommitted binder. Any other diff closes C2.
6. Generate treatment, sham, and manifest once using only the `training`,
   `observation`, and `sham` seeds.
7. Run the independent data audit once and freeze all artifacts read-only.

Any production generator invocation counts, including one that exits after
opening or partially writing an output. Preflight must therefore finish on toy
fixtures. A generation failure, overwrite attempt, second invocation, manual
row inspection, seed substitution, or code change closes C2. No output may be
silently deleted and regenerated.

## 8. Fits, runtime, and provenance

Each treatment/sham pair starts from the same immutable raw checkpoint and uses
the same derived fit seed and record order. Paired seeds are `fit-a` and
`fit-b`. The exact optimizer, learning rates, warmup, clipping, packing, batch
size, epochs, and completion mask are frozen in `F`; there is no early stopping
or checkpoint selection.

Every fit executes from the same hash-pinned runtime image and immutable input
snapshot. A fit receipt binds arm, fit seed, initialization, data, tokenizer,
trainer, imported local modules, runtime image, hardware, exact hyperparameters,
executed update count, consumed target tokens, checkpoint hash, and post-run
input rehashes. Treatment and sham checkpoints cannot be admitted by filename
or free-form manifest text.

## 9. Evaluation and independent scoring

Evaluation is greedy, source-deleted after compilation, fresh-cache per call,
and retry-free. The interpreter may parse exact packet grammar, select the first
operation, transport strings and integers, and render a fixed atomic prompt. It
may not calculate, retain the source, repair, search, rank, inspect gold data,
or provide verifier feedback.

Raw transcripts contain prompts, responses, token counts, call order, stop
reasons, and immutable input hashes, but no trusted correctness fields. Output
is exclusive, complete, atomically finalized, hashed, and read-only.

Two independently implemented scorers with different source hashes must:

- independently validate the board, manifests, fit receipts, transcript
  completeness, case pairing, call counts, and all artifact hashes;
- independently parse packets and executor output;
- replay every mathematical program and trajectory;
- independently recompute exact two-sided McNemar tests and all aggregates;
- agree on every integer count and every probability to `1e-12`.

They may not share semantic scoring helpers. Disagreement closes C2 without
choosing a preferred scorer.

## 10. Locked gates

Both treatment fit seeds must independently satisfy all gates:

```text
raw external scheduler gold answers       >= 128 / 256
oracle-packet exact closed loops           >= 230 / 256
initial compilation                        >= 224 / 256
conditional packet-update accuracy         >= 95%
strict source-deleted closed loops          >= 192 / 256
per-stratum compilation                     >= 52 / 64
per-stratum strict closed loop              >= 40 / 64
final external-trajectory mismatches        <= 8 / 256
treatment - sham compilation                >= 30 percentage points
treatment - sham strict closed loop         >= 25 percentage points
packet-swap follows swapped packet          >= 60 / 64
```

Treatment must beat sham in every stratum. Each paired fit seed requires exact
two-sided McNemar `p < 0.01` separately for compilation and strict closed-loop
success. Both independent scorers must agree. There is no threshold, parser,
family, seed, or checkpoint adjustment after any production output is read.

## 11. Allowed claim

Passing every custody and capability gate permits only:

> On one preregistered bounded arithmetic distribution, a learned compiler and
> updater controlled an immutable atomic executor through counted source-
> deleted recurrence better than a causally matched sham.

It does not establish autonomous general reasoning, latent recurrence, or
constant-size universal memory. Failure or custody invalidation is reported
without replacement analysis under the C2 name.
