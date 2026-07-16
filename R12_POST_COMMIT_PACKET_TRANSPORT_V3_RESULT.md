# R12 Post-Commit Packet Transport V3 Result

**Decision:** PRECOMMIT IMPLEMENTATION GO; NO CANONICAL RESULT YET. Three
adversarial review rounds found claim-blocking defects before publication. The
first found an open `/tmp` escape, incomplete runtime binding, helper bypass
paths, and an overbroad historical-custody claim. The second demonstrated that
the expanded finite blacklist still allowed writes to four user-writable
`/Library` paths and that the parent could forge the verifier receipt accepted
by its own canonical writer. The third found a byte-level claim-boundary drift
between the parent and independent reconstruction plus stale nonce terminology.
All are repaired and covered by regressions. No canonical v3 artifact exists.

The current repair replaces the blacklist with default-denied filesystem and
network access, a root-owned protected Python runtime, exact role/cwd grants,
and executed probes against every demonstrated escape. It also removes the
parent canonical writer: the separately implemented verifier must reconstruct
the evidence, publish the artifact itself, and emit a hash-bound receipt. The
current-source precommit integration then completed two full byte-identical
cores and an independent reconstruction:

```text
unit/adversarial tests:     38/38 PASS
roles per core:             337
total confined role calls:  674
sandbox probes:              27/27 PASS
public cells:                 5
decisive cells:              15
first core payload SHA-256:  f545653cef9bdcdc9e384b59168a3be3b1ad0fd9e3a47ecf337361af588a8be2
second core payload SHA-256: f545653cef9bdcdc9e384b59168a3be3b1ad0fd9e3a47ecf337361af588a8be2
final payload SHA-256:       aca7e38833c7dfb342e171d6fe05525fb0764f3e6a768110f977e887c9ccee04
independent reconstruction: PASS
```

This run used a synthetic all-zero commit identity over the exact current
source hashes. It is an integration gate, not a post-commit scientific result,
and it wrote no canonical output. Freeze the scientific paths in Git, execute
the canonical CLI from that commit, require the separately implemented verifier
to publish both a mode-0444 artifact and bound receipt, and audit those outputs
before this status can become a protocol pass.

V3 preserves the v2 algebra and freezes only the custody repairs in
`R12_POST_COMMIT_PACKET_TRANSPORT_V3_PREREG.md`. No neural fit, Shohin adapter,
SFT, H100 job, workspace claim, or reasoning claim is authorized.
