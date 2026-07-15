# R12 Residual Packet C2 Reproducibility Audit Result

**Status:** closed before beacon, seed, board, packing, fit, or evaluation.

## Verdict

The C2 toy generator and auditor are **not admissible** for a production freeze.
This is independent of the earlier prerequisite failure that already closed C2.
No production seed or generator was invoked.

Audited untracked snapshot hashes:

- generator: `8f428df0ee5b775721985a166fbd58a014b680b24ce81d31780a345460e50a7e`
- auditor: `1df476ecc9d5a1e7d2a64d8dd406e1f43332c7df58d708f1d93f07d9e03e04e4`

## Blocking findings

1. One semantic beacon pulse admits multiple valid seeds because raw JSON bytes
   and post-pulse validator metadata enter the entropy. Pretty-printing,
   validator ordering, or `validated_at` changes produced different accepted
   commitments for the same parsed pulse. This is a seed-grinding channel.
2. The required freeze surface is incomplete. Production evaluator, scorers,
   jobs, runtime identity, provenance binder, packing, fit, evaluation, gates,
   and resource ledger are absent.
3. Prerequisite, freeze, and validator evidence are caller-supplied hashes and
   booleans rather than replayed evidence.
4. Seed consumers do not independently replay the raw beacon, signature,
   certificate, freeze receipt, validator evidence, or prerequisite results.
5. The claimed freeze commit is not proven to contain the executed bytes.
6. Runtime replay depends on CPython, `random.Random`, `tokenizers`, and absolute
   paths without an immutable runtime or conformance vectors.
7. Local `O_EXCL` and mode `0444` do not prove one-shot execution or preserve
   failures in an external append-only ledger.
8. The second implementation is not clean-room: tests prove only absence of an
   import, not independent control flow or independently generated golden
   vectors.

Targeted tests passed `42/42`, and 32 toy differential replays agreed. Those
facts establish internal consistency under the tested local runtime only; they
do not repair the custody failures above.

## Decision

C2 remains closed. The untracked toy files are retained only as quarantined
mechanics and must not be committed as a production-ready protocol, used to
request a beacon, derive a seed, generate a board, or authorize a fit. Reopening
would require a new canonical seed scheme, complete immutable freeze surface,
content-addressed externally timestamped bundle, git-tree binder, runtime
digest or runtime-independent primitives, append-only attempt ledger, and a
genuinely independent implementation with adversarial golden vectors.
