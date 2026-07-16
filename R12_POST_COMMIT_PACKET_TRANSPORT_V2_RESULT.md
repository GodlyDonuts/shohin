# R12 Post-Commit Packet Transport V2 Result

**Decision:** POST-COMMIT CANONICAL NO-GO. The run from scientific commit
`a9c1f53` passed its internal 33 gates and exact algebra, but an independent
adversarial audit found that the verifier accepts forged invocation evidence,
the role processes are not OS-confined from parent paths, and the deterministic
manifest-derived challenge is predictable before phase one. The artifact is
preserved byte-for-byte under a rejected filename and has no standing as a v2
protocol pass. No neural fit is authorized.

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
16. reread temporary phase-one packet paths while assembling the seed-
    independence gate after the temporary directory had already been removed.

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

The defect-16 repair compares the replayed writer payload hashes directly to
the immutable phase-one manifest hashes. A cleanup regression proves the gate
still works after the packet paths no longer exist and rejects a mutated replay
payload. The corrected local suite passes 13/13 tests. Because the scientific
implementation and test changed, another clean commit is required before the
next canonical attempt.

## 3. Rejected post-commit artifact

```text
scientific commit: a9c1f53c275927ef7955d9578b3f1e493a460426
preserved path:    artifacts/r12/post_commit_packet_transport_v2_postcommit_no_go_a9c1f53.json
artifact mode:     0444
artifact bytes:    892632
file MD5:          f7502dc0824bff212d4ffe1ebfa4e161
file SHA-256:      4f63123028fe33717981d026ca4accc854943400a502e56698ff040145a2ab0d
payload SHA-256:   9fa2a1fef402f4751a115f4597ff113f764b0654e19fc608bf93a64d2a2f3c59
embedded replay:   byte-identical, 337 role records per core
internal verifier: PASS, 33/33 asserted gates
independent audit: NO-GO
```

The five public cells are state/motor 83,521/83,521. Every one of the 15
decisive cells is state 83,521/83,521 versus rank-two motor 4,913/83,521. The
bounded horizon arm is exact through depth eight and falls to 4,913/83,521 at
depth nine. These counts validate the inspected algebra only.

## 4. Independent audit NO-GO

The audit adds four claim-blocking defects:

17. `verify_report` can accept an evidence-forged report after the attacker
    mutates invocation exit codes, directory listings, hashes, or permutations
    and consistently recomputes the self-hashes and embedded replay;
18. fresh role directories are process-local conventions, not OS filesystem
    isolation, because a role can traverse parent directories containing
    phase-one packets and challenges;
19. the challenge seed is deterministic from committed source and the
    deterministic phase-one manifest, so it supplies ordering but no post-
    phase-one unpredictability against a precomputed finite table; and
20. the completed artifact/result custody had not yet been anchored in Git.

Defect 20 is closed by this hash-bound rejection record; it does not rescue the
protocol. A v3-quality repair must use an OS sandbox that denies every role
access outside its executable and declared files, a parent-generated random
nonce committed only after phase one, and a separately implemented verifier that
recomputes every transport chain, score, witness, and invocation record rather
than trusting self-attested booleans. The corrected artifact hash and completed
result record must then be committed and independently audited.

No neural fit, Shohin adapter, SFT, H100 job, workspace claim, or reasoning
claim is authorized by this rejected result.
