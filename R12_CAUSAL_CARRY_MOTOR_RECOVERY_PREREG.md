# R12 Dual-Provenance Carry-Motor Recovery Preregistration

**Status: CPU/H100 execution NO-GO.** Local CPU tests and static review are the
only authorized actions. No recovery plan job, fit, development evaluation,
confirmation generation, or confirmation evaluation is authorized until a
fresh independent hostile reviewer returns exact-byte `GO` for this document
and the other three recovery files. The review must be published as the
immutable receipt required by the recovery executable. This document is not a
capability claim.

## 1. Purpose and frozen upstream lineage

The sole purpose of this protocol is to recover the already preregistered carry
motor fit from a mechanical Python/JSON representation defect without claiming
that new executor code produced the upstream plan or feature tensors.

The immutable upstream identities are:

- source commit:
  `a0c258e6709766c643cf127a429a7d6ef4a4211b`
- source-manifest SHA-256:
  `9ae61e1a3e8f672a71a01edc16e6a5f1f8f3c69f49afd5e97f41c6cde15350a9`
- canonical plan SHA-256:
  `1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d`
- confirmation-commitment SHA-256:
  `1ee32e4e2e8f9eb56026b7b8de1fdff207e9fd3694e0ae354f103d58ebb820da`
- fit-row SHA-256:
  `6517b1ff3aa557e449a2eef9c5540c3d5f8699482d933d5c320b606adb4a0f1b`
- canonical board SHA-256:
  `d6282610ba845b23ebe849efe574233bf657a50aea0a7edb901e9e1d95b24391`

The eight immutable feature-shard receipts, in shard-index order, are:

1. `4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96`
2. `4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c`
3. `e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43`
4. `ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f`
5. `ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144`
6. `f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83`
7. `009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db`
8. `8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4`

The upstream root
`artifacts/carry_motor/canonical_a0c258e6709766c643cf127a429a7d6ef4a4211b`
is read-only evidence. Recovery must require its root and shard directories to
remain mode `0555`, its plan and shard files mode `0444` and one-link, and its
fit, development, and confirmation directories empty mode `0700`. Recovery
never writes, renames, links, copies, chmods, or publishes inside that root.

## 2. Observed failure and exact normalization proof

Job `692563` successfully replayed all eight shards and completed the frozen
2,000 treatment plus 2,000 shuffled updates. It then failed before publication
because the generated in-memory board contained integer histogram keys while
the JSON-loaded plan contained string keys. The prepublication fit directory
remained empty.

An independent reconstruction from the exact tokenizer and episode bytes
generated 65,536 rows with the frozen row digest. A recursive type-sensitive
comparison found exactly these two differences:

```json
[
  {
    "generated_key_type": "int",
    "generated_keys": [97, 99, 103, 105],
    "path": "board.prompt_length_histogram",
    "sealed_key_type": "str",
    "sealed_keys": ["97", "99", "103", "105"]
  },
  {
    "generated_key_type": "int",
    "generated_keys": [114, 116, 120, 122],
    "path": "board.token_length_histogram",
    "sealed_key_type": "str",
    "sealed_keys": ["114", "116", "120", "122"]
  }
]
```

The ledger SHA-256 is
`b43cb4a6fbfab97c659e8658f63185ae8b3dc1d8cce34089958d3b09df0593b6`.
All non-histogram fields, histogram counts, row order, labels, and values are
type-strict equal. Strict finite JSON serialization followed by
duplicate-key-rejecting parsing produces the exact sealed plan board and the
canonical board digest above.

The sole allowed transformation is
`strict_json_round_trip_of_complete_generated_fit_board`. It has zero permitted
semantic changes and zero additional transformations. A count change, extra
key, non-histogram difference, bool/int or int/float alias, duplicate JSON key,
or a third type difference fails closed.

## 3. Dual provenance

The recovery lineage has two noninterchangeable source identities:

1. **Upstream protocol source.** The exact `a0c258e` source contract recorded by
   the sealed plan and every shard. This identity owns the board, features,
   labels, controls, fit mathematics, confirmation commitment, and frozen
   scientific semantics.
2. **Recovery executor source.** A later reviewed Git commit containing exactly
   this preregistration, `train/causal_carry_motor_recovery.py`, its tests, and
   its Slurm wrapper. This identity owns only binding, strict board
   normalization, recovery validation, and v9 publication.

Runtime requires `HEAD` to equal the recovery commit, a clean checkout, exact
working bytes equal to `git show`, and an exact manifest over those four files.
The recovery commit must have the full `a0c258e` commit above as its sole direct
parent. `git diff --name-status --no-renames` between those commits must be
exactly four additions: this preregistration, the recovery executor, its test
file, and its wrapper. A modified baseline file, fifth file, rename, merge,
grandchild, extra commit, untracked file, or module shadow fails closed. Every
loaded recovery, upstream, and model module must resolve to its exact reviewed
path. Both wrapper and executor compare every non-`.git` filesystem leaf against
`git ls-files`; ignored files are not trusted as clean, so an ignored
`sitecustomize.py` or package shadow also fails before executor import. Each of
the four recovery sources must be Git mode `100644` and a one-link,
non-symlink mode-`0644` regular checkout file; hard-link aliases fail. Every
imported upstream scientific dependency must still equal its bytes
in `a0c258e`. Passing the old commit for modified code, `PYTHONPATH`
substitution, monkeypatching, dirty checkout execution, or relabelling an old
shard as recovery-produced fails.

## 4. Independent review gate

Before the recovery plan exists, a fresh independent hostile reviewer publishes one
mode-`0444`, one-link `hostile_review.json` in the exact mode-`0555` directory
`artifacts/carry_motor/recovery_reviews/review_${RECOVERY_COMMIT}`. It has only:

- audit `causal_carry_motor_recovery_hostile_review_v2`;
- decision exactly `GO`;
- the complete recovery executor source contract;
- the complete pinned executor runtime contract;
- the upstream plan SHA-256;
- the complete normalization contract and sole allowed transformation; and
- the frozen review claim boundary.

The receipt SHA-256 is supplied separately. A missing, writable, linked,
aliased, post-source, wrong-commit, `NO-GO`, or expanded receipt fails before
recovery planning or CPU/H100 execution.

The runtime contract fixes the launcher to
`/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python` and Git to the regular,
non-symlink `/usr/bin/git`. It records the resolved interpreter identity and
SHA-256, Python version, ABI, exact startup flags and `sys.path`, Torch and
Tokenizers versions plus entrypoint identities and SHA-256 values, exact module
paths, and the reviewed deterministic environment. Caller override of the
Python launcher is impossible. `PYTHONPATH` is exactly the reviewed `train`
directory; user-site and bytecode writes are disabled; hash seeding is fixed;
thread counts and CUBLAS workspace are fixed; Python startup injection,
`LD_PRELOAD`, and Torch deserialization override variables are forbidden. The
same runtime is reconstructed and compared type-strictly before publication.

## 5. Immutable recovery plan

The exact recovery root is derived, not selected:

```text
artifacts/carry_motor/recoveries/
  upstream_${UPSTREAM_PLAN_SHA256}_executor_${RECOVERY_COMMIT}/
```

It must not exist before publication. The planner validates and safely loads all
eight upstream shards, independently regenerates rows, normalizes the board,
recomputes the shuffled control, batch schedule, initial motor state, sentinel
identities, and merged feature receipts, and then publishes one immutable
`causal_carry_motor_recovery_plan_v2` document. Its root is mode `0555`; the plan
is one-link mode `0444`; and fit, development, and confirmation directories are
empty mode `0700`.

The recovery plan binds:

- both source contracts and the hostile-review receipt;
- upstream plan, commitment, generator, source, frozen-input, and all eight
  shard identities;
- the complete normalization proof;
- exact checkpoint step, dimensions, token IDs, board, row order, control,
  2,000-update schedule, batch 512, rank 8, learning rate 0.003, weight decay
  0.0001, seed, and initial state;
- the upstream merged feature and teacher-metric hashes;
- exact new recovery output paths; and
- explicit safe deserialization behavior.

It also binds a complete upstream custody snapshot, not only content receipts. The
snapshot covers the canonical root, plan, all eight shard directories and
files, the empty fit/development/confirmation directories, and the confirmation
commitment directory and file. Each entry records its exact lexical path,
kind, device, inode, mode, link count, owner, group, size, mtime, ctime, closed
world children, and file SHA-256 where applicable. The complete snapshot is
reconstructed and compared type-strictly immediately before and immediately
after artifact publication and again around final directory sealing. A same-byte
inode replacement, mode change, new child in an empty directory, shard mutation,
or directory substitution is fatal.

Any caller-selected alias, output under the old canonical root, changed budget,
changed shard receipt, changed source, or extra transformation fails closed.

## 6. Safe deserialization

Checkpoint and shard tensor files are bound by exact lexical path, no-symlink
open descriptor, inode/stat identity, and SHA-256 before deserialization.
`torch.load` is called explicitly with `weights_only=True` inside a safe-global
scope containing only `torch.torch_version.TorchVersion`, which is required by
the already sealed runtime metadata. There is no unrestricted-pickle fallback.
Both `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` and
`TORCH_FORCE_WEIGHTS_ONLY_LOAD` are forbidden ambient overrides.

## 7. Fit and publication

The reviewed H100 wrapper requires one visible `NVIDIA H100 PCIe`, four CPUs,
`Requeue=0`, restart count zero, an exact clean recovery checkout, exact spooled
wrapper bytes, the sealed hostile-review receipt, and the sealed recovery plan.
The fit exposes no mutable optimization flags. It uses the plan's frozen values.

The executor replays and validates the eight upstream shards, fits treatment and
shuffled arms from the same initial state and schedule, recomputes all retained
teacher evidence and diagnostics, and passes the complete upstream v8 payload
validator in memory. Before that legacy validator, a recovery-owned exhaustive
validator checks every legacy payload field with exact Python types. It rejects
`bool`/`int`, `int`/`float`, mapping-subclass, state-container, tensor-subclass,
fit-report, diagnostic, and nested evidence aliases; requires finite float loss
and accuracy fields; recomputes the expected teacher evidence; and asserts that
its field-coverage set equals the complete frozen legacy schema. It never
publishes a v8 object. The sole output is
`causal_carry_motor_fit_v9_recovery`, with top-level recovery status, both source
domains, upstream plan and shard receipts, normalization proof, deserialization
contract, and a headerless scientific fit payload. A top-level `canonical`
field or v8 audit is forbidden.

Publication is recovery-owned and descriptor-bound. With the exact mode-`0700`
fit directory open by descriptor and proven empty, the executor creates the
final leaf `motor.pt` directly using `O_CREAT|O_EXCL|O_NOFOLLOW` at mode `0600`.
It serializes to that same descriptor, flushes and fsyncs, verifies the linked
name is the same one-link inode and the directory has no second child, hashes
through the descriptor, then chmods and fsyncs the same inode to `0444`. There
is no staging path, rename, hard link, upstream atomic helper, or replace
operation. It safe-loads and fully revalidates the published v9 object before
descriptor-sealing the one-file fit directory to mode `0555`. The upstream
custody snapshot must remain exactly unchanged around both operations.

The fit directory has exactly four accepted states: empty mode `0700`; one
mode-`0600`, one-link `motor.pt` in mode `0700` after interruption during direct
serialization; one mode-`0444`, one-link `motor.pt` in mode `0700` after a crash
immediately after publication but before directory sealing; or the same sole
artifact in sealed mode `0555`. The exact executor removes only the closed-world
mode-`0600` interrupted leaf and fsyncs the directory before retry. It may seal
the mode-`0444` crash-recoverable state only after safe-loading and fully
validating the v9 bundle and re-verifying the runtime, recovery plan, hostile
review, upstream plan, confirmation commitment, frozen inputs, every shard
binding, and full upstream custody snapshot. Any second link, staging child,
other mode, child, filename, or substituted directory fails closed.

## 8. Threat model

The fail-closed boundary assumes an attacker or accidental operator may supply
an aliased path, dirty checkout, wrong commit, merge or grandchild commit,
additional committed or untracked file, shadow module, alternate interpreter,
unsafe environment, modified review receipt, same-byte inode substitution,
linked or renamed artifact, partial serialization, crash after publication,
type-aliased Python payload, changed budget, changed board, changed shard,
additional normalization, old-root output, or replacement confirmation
generator. The executor must detect these before making or sealing a claim.

The protocol does not claim protection against a compromised kernel, root user,
storage firmware, Git or Python binary whose bytes change after their final
descriptor check, malicious CUDA hardware, SHA-256 collision, or a dishonest
independent reviewer who deliberately signs the exact bad source/runtime. Those
are explicit trust roots. Network availability is irrelevant because execution
uses no network source. Recovery code has no authority to regenerate or inspect
the confirmation secret and no authority to reinterpret a fit as capability.

## 9. Downstream boundary

This commit designs fit publication only. Development and confirmation recovery
must receive separate preregistration and hostile review before implementation
or execution. Any future confirmation path must keep the exact `a0c258e`
confirmation generator source contract from the pre-fit commitment and record a
separate recovery evaluator source contract. Recovery executor code may not
substitute itself as the secret-derived board generator.

No fit result, teacher-forced accuracy, development score, confirmation score,
mechanism conclusion, autonomous capability, or reasoning claim is established
by this preregistration.

## 10. Required CPU gates before review

The exact recovery source must pass:

- normalization success with exactly two frozen key-type differences;
- rejection of non-histogram, count, extra-key, duplicate-key, and scalar-type
  rewrites;
- path alias, symlink, receipt, shard, source, and executor substitution tests;
- sole-parent/four-addition history, extra-file, grandchild, and shadow-module
  rejection tests;
- pinned-interpreter, package-entrypoint, startup-flag, and environment tests;
- complete upstream custody snapshot tests, including all empty directories,
  modes, and same-byte inode replacement;
- frozen-budget, old-root output, and extra-transformation rejection tests;
- confirmation-generator substitution rejection;
- explicit weights-only/TorchVersion loading and ambient-override rejection;
- immutable closed-world plan publication tests;
- direct no-replace publication, interruption cleanup, immediate-post-publish
  crash recovery, and no-staging/no-two-link tests;
- exhaustive legacy payload scalar/container type-alias rejection tests;
- v9-only dual-provenance schema tests;
- warning-clean CPU Pytest, Ruff, Python compilation, `bash -n`, and whitespace
  checks.

Passing these local CPU gates does not change CPU/H100 execution status. Only a
fresh independent exact hostile-review receipt changes the recovery fit from
NO-GO to eligible.
