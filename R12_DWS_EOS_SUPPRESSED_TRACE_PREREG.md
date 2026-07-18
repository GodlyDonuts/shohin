# R12 DWS EOS-Suppressed Trace Field Screen Preregistration

**Protocol:** `R12-DWS-EOS-SUPPRESSED-FIELD-SCREEN-DEV-v4`

**Status:** CPU mechanics GO; Linux publication, Newton deployment, H100
execution, and full-state recurrence NO-GO.
This document authorizes only the local 100-episode development screen in
`train/eval_dws_eos_suppressed_trace.py`. It does not authorize promotion,
hidden confirmation, training, checkpoint changes, a verifier-backed capability
claim, or an H100 claim. Linux/Lustre qualification and a fresh hostile review
of the exact four uncommitted file bytes remain mandatory before commit or any
qualification. Any later commit must preserve the reviewed hashes exactly. All
commit, Linux, Stokes, Newton, publication, and H100 gates remain NO-GO for this
untracked repair.

**Execution custody status:** blocked until these exact repaired bytes have been
reviewed and committed. After that commit, a separate external read-only
runtime-source manifest and externally signed run authorization must bind the
exact commit, source bytes, allocation, inputs, and output custody as specified
below. This revision does not authorize or report a GPU launch, a Newton run, or
an accepted publication artifact.

## 1. Question and prior observations

The hypothesis is deliberately broad:

> Suppressing a model-selected EOS can expose partial recurrent use of generated
> DWS history under an external decode clock.

It is **not** a newline-specific hypothesis. Fixed token overrides supply an
external halt veto and clock. They are not model-authored halting, autonomous
state control, or autonomous reasoning.

The following independent local development observations were supplied before
this preregistration and are motivation only. They are not results produced by
the frozen evaluator:

- At the first three observed EOS boundaries, generic EOS masking selected
  tokenizer ID `211`, which decodes to one LF.
- For `428 - 181`, ordinary decoding had exact prefix 1; LF `211` and semicolon
  `39` had prefix 2; space `233` had prefix 1.
- For `181868 + 116989`, ordinary decoding had exact prefix 1; LF, semicolon,
  and space had prefix 2.
- LF continuations could emit labels such as `Check:`; semicolon could produce
  text shaped like `;z=0;dws...`; space could emit a foreign token before a new
  DWS line. These observations veto a uniquely causal LF interpretation.
- In generated-history counterfactuals, changing only the emitted carry often
  failed to change the next state even when the active result digit should
  change. Changing the already-written result digit propagated into the next
  tape. Active operand perturbations changed the next digit imperfectly.
- On an independent 16-case local board, nominal next-state exactness was
  `5/16`. Carry-flip full-target exactness was `7/16`, but output changed only
  `3/16`. In the true-carry-one stratum, nominal exactness was `0/8` while
  carry-flip exactness was `6/8`, consistent with a default-carry-zero policy
  producing spurious counterfactual accuracy. Written-result output changed in
  `7/16`, while its full counterfactual target was exact in `1/16`.
- On eight further carry interventions, full-history `S0 + generated S1`
  continuation had true exact `3/8`, counterfactual exact `4/8`, output switch
  `1/8`, and paired causal exact `1/8`. A fresh exact core prompt containing
  only canonicalized `S1` had true exact `8/8`, counterfactual exact `7/8`,
  output switch `8/8`, and paired causal exact `7/8` on the same model. That
  contrast jointly removes prior context, resets positions, and canonicalizes
  the state surface; it does not identify which component caused recovery.
- A non-preregistered 12-case cross-width replication, balanced two per
  `(width 4/6/8, add/sub)`, found full-history paired carry exact `2/12`
  (`0/4`, `2/4`, `0/4` by width) versus compound fresh-state paired exact
  `10/12` (`3/4`, `4/4`, `3/4`). Fresh-state true exact was `12/12`,
  counterfactual exact `10/12`, and output switch `12/12`.
- A no-weight cache-pruning probe was negative. Keeping only latest generated-
  state KV became malformed after the first interval; immutable-prefix plus
  latest-state KV also failed; dropping only stale `S0` keys while retaining
  the original prefix, suffix, and latest state stayed near full-history behavior
  and did not reproduce fresh-prompt isolation. Cached suffix/state values were
  already contextualized by `S0`, so post-hoc slicing could not undo that source.

Consequently, full-target exactness and paired target-switch response are
separate, non-substitutable endpoints. Nominal trace prefixes cannot establish
field use. Failed carry target-switch is a noncompensatory carry-use veto.

## 2. Frozen inputs

| Input | SHA-256 |
|---|---|
| `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt` | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| `artifacts/shohin-tok-32k.json` | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| `artifacts/evals/digitwise_recurrent_v2_heldout.jsonl` | `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646` |

The exact checkpoint-step JSON value is the string `"sft_ep1"`. The evaluator
accepts no checkpoint, tokenizer, heldout, case-count, prompt-style, arm,
budget, device, or precision override. Each input must be a regular,
non-symlink, single-link file. The checkpoint is hashed before loading from the
same open descriptor and hashed again after loading. The small inputs are
parsed only from their verified bytes.

### Runtime-source seal

The execution wrapper is
`train/jobs/eval_dws_eos_suppressed_trace.sbatch`. It accepts execution only
from an absolute normalized `SOURCE_ROOT` after the submitter sets
`SOURCE_ROOT_REVIEWED=YES`, and only when `SOURCE_COMMIT` is the exact lowercase
40-hex checked-out `HEAD`. The source tree must be clean under
`git status --porcelain=v1 -z --untracked-files=all`.

An externally stored regular, non-symlink, single-link manifest with no write
permission bits must use schema
`r12_dws_eos_suppressed_runtime_sources_v7`. Its separate externally supplied
SHA-256 is mandatory. The manifest has exactly these top-level keys:
`schema`, `source_root`, `source_commit`, `files`, `runtime`. Its `files` object
has exactly the following lexicographically ordered closure and one lowercase
SHA-256 for each. The complete manifest must be canonical sorted compact JSON
with one final LF:

```text
R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md
train/eval_dws_eos_suppressed_trace.py
train/jobs/eval_dws_eos_suppressed_trace.sbatch
train/model.py
train/test_eval_dws_eos_suppressed_trace.py
```

The test and preregistration sources are sealed conservatively even though the
evaluator does not import them. Adding any other repository-local runtime
import requires a reviewed protocol revision and an expanded manifest closure.

The `runtime` object freezes schema `r12_dws_eos_runtime_identity_v4`;
normalized absolute Python, Git, `scontrol`, `sacct`, and `nvidia-smi` paths,
executable hashes and version output; complete installed PyTorch and tokenizers
distribution closures; the isolated Python startup flags, search-path inode
surface, loaded module origins, and every consumed regular source, bytecode, or
extension component; every absolute file mapping already present in Linux
`/proc/self/maps` bound to the mapping record's device and inode; and the CUDA,
precision, math-SDPA, cuBLAS, and
deterministic-algorithm backend contract. Cryptography is not a runtime package
dependency for authorization: Ed25519 public-key derivation, signing, and
verification are implemented in the reviewed evaluator source. Point decoding
requires a 32-byte RFC 8032 canonical field encoding (`y < 2^255 - 19`), rejects
the set sign bit when the recovered `x` is zero, requires exact canonical
re-encoding, membership in the prime-order subgroup, and an explicit
non-identity point. This rejects canonical and alternate identity encodings,
noncanonical field encodings, identity public keys, identity `R`, and every
small-order trivial-forgery input. Signature scalar decoding separately requires
an exact 32-byte little-endian integer strictly below the Ed25519 subgroup order.

The wrapper replaces Bash with the descriptor-held interpreter using `exec`
and starts Python with exactly `-I -S -B`. `site`, `sitecustomize`,
`usercustomize`, `.pth` processing, Python startup/path environment variables,
and preauthorization site-package search paths are forbidden. Every already
consumed stdlib source, existing `.pyc`, native extension, built-in/frozen
module origin, and Python search-path inode or verified archive is included in
the startup closure before run authorization is accepted. The exact closure is
recomputed immediately before and immediately after production-signature
verification. A lazy import or path/component change at either boundary makes
the authorization ineffective; no key request or namespace mutation has yet
occurred. PyTorch and
tokenizers paths are not activated by `site`; their complete externally sealed
RECORD closures are rehashed first, their verified distribution roots are then
appended explicitly, and the distributions are rehashed after import.

Each package closure is reconstructed from one strict CSV `RECORD`. The exact
`importlib.metadata.files` set must equal the exact `RECORD` path set. Every
listed file, including transitive Python, native extension, console-script,
metadata, and `RECORD` files, is bound by normalized absolute path, original
canonical RECORD-relative path, SHA-256, byte count, device, inode, UID, mode,
and link count. Leading `../` components are permitted only in canonical wheel
form and only when their normalized target remains under the running
interpreter's normalized `sys.prefix`; an absolute path, mid-path `..`, symlink,
hardlink, group/other-writable file, missing file, duplicate, manifest mismatch,
or declared RECORD hash/size mismatch is fatal. The sorted full file manifest,
distribution root, installation root, name, and version have one canonical
closure digest. The closure is observed before third-party package import and
rehashed after import; the imported `__init__.py` paths and versions must match.

Before external authorization is accepted on Linux, all absolute file mappings
already loaded into the process are independently hashed with the same
regular-file/inode/mode rules. The parsed hexadecimal device major/minor and
decimal inode from each `/proc/self/maps` row must equal the reopened
pathname's actual device and inode. A deleted mapping, path replacement, zero
inode, or one pathname associated with multiple mapped identities is fatal.
The same map-record binding is required for later loaded-runtime snapshots.
These records form the canonical native-library closure.
The source-contained Ed25519 verifier validates the production signature before
PyTorch or tokenizers is imported, so unsealed third-party package code cannot
verify the authority that permits its own import. Non-Linux execution records
an explicit development-only absence of `/proc/self/maps` qualification.

For every listed path, wrapper and evaluator require current SHA-256 equality
to the manifest and byte-for-byte equality to
`git show SOURCE_COMMIT:<path>`. The wrapper opens and retains descriptors for
the reviewed Python, Git, `scontrol`, `sacct`, `nvidia-smi`, external manifest,
and actual Slurm-spooled `$0`. It executes those binaries through Linux
`/proc/self/fd`, rechecks each held descriptor against its reviewed pathname,
and requires the running interpreter inode to equal the held Python inode. The
spooled wrapper bytes and the exact parsed Slurm `Command` file bytes must both
equal the manifest's wrapper hash.

Every Git invocation uses the held executable with a constructed environment,
not the inherited Git environment. System/global configuration, replacement
objects, optional locks, pagers, prompts, and system attributes are disabled;
`core.fsmonitor=false`, `core.hooksPath=/dev/null`, an empty external diff, and
a null attributes file override repository configuration. Thus `git status`,
`show`, `cat-file`, and `rev-parse` cannot execute an unsealed fsmonitor or hook
from local, global, system, or injected `GIT_CONFIG_*` configuration.

After source verification, the wrapper copies the exact evaluator and
`train/model.py` bytes into separate anonymous sealed memfds. Generation and
independent verification execute the evaluator memfd under the descriptor-held
interpreter with `-I -S -B`; model construction compiles only the sealed model
memfd bytes. Pathname replacement cannot change those executed source bytes.
The wrapper holds the external manifest and output-directory descriptors for
the complete transaction and rechecks source, commit, executable, manifest,
spooled-wrapper, Slurm, and output-directory identities before acceptance.

The accepted output path and its deterministic
`.r12-acceptance-commit.json` sidecar must be absolute, normalized, outside
`SOURCE_ROOT`, and absent. Their pre-existing parent must be a normalized,
current-UID-owned mode-`0700` directory. The wrapper opens it with
`O_DIRECTORY|O_NOFOLLOW`,
records device and inode, and uses only relative operations through that held
dirfd. Before inspecting or deleting any protocol residue, the wrapper takes a
nonblocking exclusive advisory `flock` on that exact held directory inode and
binds the lease to its PID, Slurm job ID, accepted name, production-authorization
nonce, and directory device/inode. A live cooperating publisher therefore makes
startup fail before cleanup. Every publication phase rechecks that the pathname
still names the held inode. Thus execution is ineligible while any repaired owned file is
uncommitted or while the external seal has not been created from that exact
commit.

### External production authorization

The wrapper cannot root its own acceptance authority. The production Ed25519
public anchor is source-pinned as key ID
`r12-production-authority-2026-07-v1`, raw public key
`3805039655eef59153ba2b148551df2376d7c2cfa550ee5f6386745d4d0ed857`,
raw-key SHA-256
`b47cb1db3d3ef97ad5cf9f80405e979f4be58f5ea1d570f092825f272bd978dc`,
and canonical external public-key-file SHA-256
`09b3f77db76b1277c79f82fa28920fff1d749925231de1ba1f85e73e2ae7e642`.
Only the public key is in the reviewed repository. No corresponding production
private key may be present in source, the runtime manifest, wrapper memory, or
any repository pathname. Provisioning and custody of the matching external
private key are outside this protocol and remain a prerequisite for execution.

Before scientific decoding, the wrapper requires canonical schema
`r12_dws_eos_external_run_authorization_v5` signed by that production key. It
binds a positive authorization sequence, random nonce, issue/not-before/expiry
times, exact source commit, external manifest path and hash, canonical output
path and parent device/inode/UID/mode, accepted filename, all checkpoint/
tokenizer/heldout/prereg paths and hashes, ordered-board hash, the complete live
`scontrol`/`sacct`/cgroup/GPU allocation record, and one delegated publication
key whose exact scopes are the domain-separated post-publication marker and
durable post-fsync acceptance receipt. It also embeds one exact replayable
`r12_dws_eos_linux_qualification_authorization_receipt_v1` object produced by
the reviewed version-6 Linux/Lustre qualification. That object contains the
complete structured qualification result and the accepted qualification
broker request/report/marker/receipt publication, with both the inner marker/
receipt signatures and an outer mechanics-only signature replayed before the
production signature is accepted. The qualification evaluator SHA-256 must
equal the sealed evaluator SHA-256 in the report acceptance context. The
production authorization signature, not the mechanics-only key, binds that
receipt into accepted report state. It also binds a sorted exact
`stale_cleanup_entries` list. Every such entry identifies one protocol-shaped
candidate, temp, or quarantine under the exact accepted name by name, device,
inode, UID, mode, link count, size, and SHA-256. Candidate and temp names may
encode a prior canonical job ID and nonce, and terminal quarantine names may
likewise come from a prior run, but only an exact entry in the fresh signed
authorization grants mutation or recovery authority. No unsigned pattern match
grants deletion authority.
Expiry is enforced before generation and again at both wrapper final-check
stages. Output no-overwrite, exact job identity, sequence, nonce, and expiry
make an authorization non-portable to another run or output.

The delegated publication private key is not accepted from a pathname and is
never inherited by Bash. The reviewed Slurm directives require
`--export=NONE`, and the job starts the reviewed wrapper with Bash privileged
startup mode (`#!/bin/bash -p`), so inherited user environment, `BASH_ENV`,
exported functions, and normal startup files cannot run. Before opening any
held source descriptor or executing Python, Bash rejects every nonempty
`LD_PRELOAD`, `LD_AUDIT`, `LD_LIBRARY_PATH`, `LD_DEBUG`, `LD_PROFILE`,
`LD_USE_LOAD_BIAS`, `GLIBC_TUNABLES`, and listed `DYLD_*` variable, then unsets
all shell, Python, and loader startup variables. The external scheduler and
controller remain trust roots for enforcing the exact `--export=NONE` launch;
an already loader-compromised shell cannot repair itself from inside the job.
Bash inherits only the connected message-boundary broker socket, never opens
the production authority or authorization path, opens only the reviewed source,
manifest, and executable descriptors, and uses `exec` to replace itself with
the held Python interpreter under `-I -S -B`; no dumpable Bash parent survives.

The Python process sets and verifies `PR_SET_DUMPABLE=0` before opening either
external authorization file or inspecting the inherited broker socket. It first
verifies exact source, manifest, sanitized Git state, and startup/package/native
closure from held reviewed descriptors. It then opens and retains the production
public key and signed authorization with `O_NOFOLLOW`, requires exact
device/inode/mode/link/size/mtime and bytes through every authorization use, and
verifies the canonical signature, full live Slurm/output identity,
source/manifest/prereg cross-binding, and signed stale-cleanup set. Only after
those checks may it validate the broker descriptor and send canonical broker
schema `r12_delegated_key_broker_request_v1`. The request binds its PID, exact
Python executable and startup closure, authorization/manifest/evaluator/wrapper
hashes, and the authorized delegated public key and private-key hash.
Qualification requests canonicalize `sys.executable` once to its strict
resolved absolute non-symlink path and hash the bytes at that same path; parent
and child therefore bind one identical executable record even when the parent
was invoked through a Python symlink.

The external broker must independently withhold authority from every Bash or
unattested requester. Before responding, it must verify the peer PID survived
the required exec boundary, `/proc/PID/exe` and held descriptor identities,
exact `-I -S -B` command/startup state, non-dumpability, and every request hash
against the independently reviewed authorization. It may then send exactly one
anonymous 32-byte, fully sealed, zero-link memfd by `SCM_RIGHTS` with canonical
response schema `r12_delegated_key_broker_response_v1`. The wrapper rejects
extra descriptors, stream transports, truncation, wrong response binding,
wrong bytes, wrong public key/hash, wrong seal mask, or an inheritable received
descriptor. The broker socket contains no key bytes and is closed immediately
after the transfer. The signed production authorization binds both the key's
public key and private-key SHA-256 plus the exact two permitted publication
schemas. A separate test-only key uses scope
`test_only_no_production_authority`; production validation rejects that scope.
The external broker/controller and signer are operational trust roots, not
scientific evidence. Their real integration is not exercised by this repair.

This is not a complete immutable executed-runtime seal. Generator and verifier
record their loaded-file mapping source, every observed loaded file hash and
inode, shared-object closure, libc and loader objects, CUDA-family objects,
CUDA runtime, driver API version, device UUID, `/proc/driver/nvidia/version`,
the required absence of all forbidden loader environment, including
`LD_LIBRARY_PATH`, `LD_PRELOAD`, and `DYLD_INSERT_LIBRARIES`. Dynamic-loader and
CUDA library resolution still occurs from system configuration at process
construction, before loaded-object snapshots can be taken. These observations
narrow the custody claim to the scheduler-enforced empty exported environment,
held executables, and measured loaded-object snapshots; they do not prove that
every runtime byte was immutable from process start to exit.
The preauthorization native mapping closure is a point-in-time snapshot; later
dynamic loads are observed again but are not made immutable retroactively. The
package RECORDs and externally signed manifest bind installed bytes, but do not
independently prove the upstream package supply chain. The source-contained
Ed25519 implementation also requires fresh specialist review before production
authorization.

### Frozen Slurm allocation

The only eligible wrapper allocation is partition `normal`, one node, one task,
four CPUs for that task, node memory `64G` (`68,719,476,736` bytes), time limit
`08:00:00` (`28,800` seconds), no requeue, and GRES
`gpu:nvidia_h100_pcie:1`. The wrapper directives freeze those values.

The wrapper independently parses one exact `scontrol show job -o JOB_ID` record
and one pipe-delimited base-job record from pinned `sacct`. It requires exact
agreement on job ID/name/state, partition, node list, CPU allocation, requested
memory, time, requested and allocated TRES, plus `scontrol` user, command, host,
node/task/CPU counts, `MinMemoryNode`, `Requeue`, `Gres`, and `TresPerNode`.
Requested and allocated TRES must each contain exactly one generic GPU and one
typed `nvidia_h100_pcie` GPU, plus `cpu=4`, `mem=64G`, and `node=1`; any
additional GPU TRES is fatal. Non-GPU accounting TRES such as billing are
retained. `Gres` is exactly `gpu:nvidia_h100_pcie:1`, and `TresPerNode` is
exactly `gres/gpu:nvidia_h100_pcie=1`. Raw `scontrol`, `sacct`, and cluster
configuration hashes are bound into the authorization and marker.

Node identity has one exact single-node representation. `scontrol BatchHost`,
`scontrol NodeList`, pipe-delimited base-job `sacct NodeList`, and the current
`socket.gethostname()` value must be byte-identical lowercase canonical names
matching `[a-z0-9][a-z0-9-]{0,62}`. Missing or empty fields, FQDNs, brackets,
commas, ranges, ambiguous lists, case normalization, disagreement, or any
multi-node allocation fail closed; no nodelist expansion or substring matching
is accepted.

Visible CUDA hardware must map one-to-one to that scheduler allocation. The
wrapper parses the current process's cgroup-v1 `devices` controller, requires an
exact `job_<JOB_ID>` path component, records the complete canonical
`devices.list`, and requires the concrete major/minor for exactly one
`/dev/nvidiaN`. The same single-row pinned `nvidia-smi` query binds display
index, `minor_number`, UUID, PCI bus ID, name, and MIG state. The device node is
derived from `minor_number`, never from display index, and its character-device
minor must equal that observation. GPU-major wildcards or another allowed
concrete physical-GPU minor are fatal. Concrete rules on the physical-GPU major
are enumerated from `devices.list`, not inferred by walking only existing
`/dev/nvidiaN` nodes; an extra permitted minor is rejected even when its device
node is absent. Observed named NVIDIA control-device major/minors are separated
from physical-GPU permissions only after each named identity is checked against
the allocated physical-GPU character-device identity. An exact alias is fatal
for every named control device, and a named control identity cannot hide a
second concrete physical-GPU permission. Duplicate named identities,
noncanonical control identities, and an unclassified concrete minor all fail
closed.
Cgroup-v2 device-BPF state is not inferred: without separately reviewable exact
controller evidence, it fails closed. Pinned `nvidia-smi` must report exactly
one physical `NVIDIA H100 PCIe`, one UUID and PCI bus address, MIG disabled, and
no MIG devices. The PCI sysfs vendor must be `0x10de` and class must be display/
compute class `0x03...`; BDF, vendor, device, class, and their canonical hash are
recorded. `CUDA_VISIBLE_DEVICES` and `SLURM_JOB_GPUS` must each contain exactly
one selector, resolve through the same physical index/UUID inventory, and match
the CUDA runtime UUID observed by both generator and verifier. Slurm controller,
accounting, configured device cgroup, sysfs, and NVIDIA driver inventory are
explicit infrastructure trust roots. Environment strings or device-name
substrings alone cannot satisfy this contract.

## 3. Frozen score-blind board

Selection uses only frozen episode metadata, never model output or a score.
For each cell in this order:

1. regimes `fit_w4`, `fit_w6`, `value_ood_w4`, `value_ood_w6`,
   `width_ood_w8`;
2. operations `add`, `sub`;
3. rank by ascending
   `SHA256(b"R12-DWS-EOS-SUPPRESSED-FIELD-SCREEN-DEV-v1\0" + id_ascii)`,
   then by case ID;
4. take the first 10.

This freezes 10 cases in each of 10 cells, 100 total. Widths 4 and 6 are the
primary partition (`n=80`); width 8 is extrapolation (`n=20`). The ordered case
IDs, joined with LF and a final LF, have SHA-256
`c83796c32fdc69efd99bff579103b0a6e2be9812cbc94b91a061bcbb24a1ad7b`.
Changing row order in the source file cannot change selection. Selection is
completed before any model output exists.

An exact evaluator-owned 12-case replication subset is selected only from that
frozen 100. For widths 4, 6, and 8, then operations add and sub, rank by
`SHA256(selection_domain + b"replication\0" + id_ascii)` and take two. Its
ordered IDs are:

```text
fit_w4-00258
value_ood_w4-00217
fit_w4-00261
fit_w4-00196
fit_w6-00122
value_ood_w6-00028
value_ood_w6-00280
value_ood_w6-00067
width_ood_w8-00120
width_ood_w8-00176
width_ood_w8-00180
width_ood_w8-00103
```

Those IDs with LF separators and a final LF have SHA-256
`1dc75ec7995e61a85f7bec9ae1fa62aa1adaf71bd46172e880aea901482396b9`.
This subset is a prospective replication of the supplied 12-case pattern, not
a reuse or claim-bearing rescore of the earlier observations.

## 4. Exact prompt bytes

Every episode starts at its frozen canonical `p=0,c=0,z=0` DWS state. The sole
prompt is the exact ASCII concatenation below, with no leading or trailing byte
beyond the shown `Answer:` suffix:

```text
Microstate update. Digits in a, b, and r are least-significant first. Use the digit at p with c, write only r[p], then advance p by one.
State: {initial_state}
Return exactly one dws state line.
Answer:
```

The final code-block newline is Markdown presentation only and is not part of
the prompt. Frozen byte commitments are:

| Component | Bytes | SHA-256 |
|---|---:|---|
| prefix through `State: ` | 144 | `875d5f9e27adefcd06c06be7e68f177f3fc6e7d5865e4c14003782871b8a96a1` |
| suffix beginning LF | 43 | `60307c5d8511691fe05a0d9c346714c304503a28da707c598a7533b41b5967f6` |
| prefix + literal `{initial_state}` + suffix | 202 | `90713da16e103d71e8ff70806a355bacf89815d637f670ffd04c62cc1c0d5814` |

Each report records exact prompt text, byte count and SHA-256, the complete
ordered prompt token-ID array, token count, and canonical little-endian
token-ID SHA-256. Every primary and secondary decode independently records its
complete prompt token-ID array and hash. Replay reconstructs each initial,
full-history, destroyed-history, intervention, and fresh-core prompt and
requires exact ordered ID equality, not merely equal count or a self-rehashed
reported array.

## 5. Primary decode arms

All arms are greedy, batch one, KV-cached, and capped at exactly 768 new tokens.
The initial prompt is prefilled once per arm; every later model call receives
only the selected previous token and the uninterrupted cache. There is no
sampling, retry, online parser, arithmetic, solver, verifier, semantic stop,
answer stop, state schedule, state repair, or state injection in a primary arm.

| Arm | Rule |
|---|---|
| `ordinary_eos_stop` | Stop only when raw greedy selects EOS ID 0, or at the 768-token cap. |
| `eos_masked_argmax` | Set only EOS logit to `-inf` at every position and select argmax; emit exactly 768 tokens. |
| `eos_to_lf_211` | When raw argmax is EOS, feed fixed ID 211 (LF); otherwise feed raw argmax; emit exactly 768 tokens. |
| `eos_to_space_233` | Same rule with fixed ID 233 (space). |
| `eos_to_semicolon_39` | Same rule with fixed ID 39 (`;`). |
| `eos_to_nonformat_x_100` | Same rule with fixed ID 100 (`x`). |

The fixed replacement IDs must decode to exactly the one-character text shown
and that text must encode back to exactly the same single ID. The generic mask
must leave every non-EOS logit bit-identical. Every raw model-selected EOS event
records generated index, absolute token position, EOS logit, best non-EOS ID,
text and logit, actual replacement ID, text and raw logit, and whether an
override occurred. Every arm reports generated-token count, exact EOS positions,
mask positions, override positions, and override count.

The five fixed-budget arms share the same generated-token budget and model
decode-step count. No wall-clock, FLOP, memory-traffic, prompt-length, or
hardware compute-matching claim is made for any sham or replacement arm.

### Deterministic CUDA contract

The wrapper exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`PYTHONDONTWRITEBYTECODE=1`, unsets Python startup/path variables, and invokes
the absolute sealed interpreter with `-I -S -B`, so `site`, customization
hooks, user startup, and repo-local import caches are excluded. The evaluator requires exactly one visible CUDA
device with BF16 support and the exact scheduler/cgroup-bound
`CUDA_VISIBLE_DEVICES` selector.
Both CUDA name sources must equal the complete string `NVIDIA H100 PCIe`, compute
capability must equal `9.0`, a device UUID is mandatory, and reported physical
memory must lie in the frozen full-device band from 75 through 85 GiB. Its UUID
must equal the `nvidia-smi`, Slurm-selector, cgroup-device, and signed
authorization UUID. This
rejects A100, H100 SXM, non-H100 BF16 devices, and reduced-memory MIG views. It calls
`torch.use_deterministic_algorithms(True, warn_only=False)`, seeds CPU and all
visible CUDA RNGs with 0, sets cuDNN deterministic mode, disables cuDNN
benchmarking, disables both CUDA-matmul and cuDNN TF32, and sets float32 matmul
precision to `highest`. Any unavailable control is fatal.

All model attention is run inside an exclusive
`torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` context. Math SDPA is the
single pinned backend; Flash, memory-efficient, and cuDNN SDPA must all report
disabled inside that context. Before model execution, a fixed CUDA BF16 SDPA
probe is run twice and must be bitwise equal. Math SDPA supports the planned
H100 BF16 execution path, but selecting it is a determinism choice, not a
throughput or H100 performance claim. The report records Python, PyTorch, CUDA
runtime, `CUDA_VISIBLE_DEVICES`, visible-device count, exact device name,
capability, total memory, UUID, every determinism flag, the backend identity,
and probe result. Generator and independent-verifier observations must agree on
all device fields. This document records no device run or result.

## 6. Offline first-boundary field screen

Only after all primary decodes finish may the evaluator decode text, parse DWS
lines, or construct an oracle. The ordinary response must stop at exactly one
model-selected EOS and its pre-EOS content must be exactly one canonical,
token-round-tripping, nonterminal `p=1` DWS line. Missing, malformed, labeled,
duplicated, or noncanonical content makes every field endpoint for that episode
a failure.

Field-screen availability is not trusted from the reported field object. Full
replay derives it independently from the ordinary arm's exact mode, stop rule,
EOS event, generated IDs, decoded bytes, canonical one-line grammar, `p=1,z=0`
state, and tokenizer round trip. A missing field screen is accepted only when
that derivation returns the exact frozen unavailable reason. If the ordinary
decode proves an eligible boundary, missing field work is a schema failure.

At that frozen first boundary, the evaluator prepares five offline history
branches:

1. `intact`: the exact model-emitted state;
2. `carry_flip`: change only `c` to `1-c`;
3. `written_result_r0_flip`: change only `r[0]` by minus one modulo 10;
4. `active_operand_digit_perturbation`: at position `p`, try tapes `a` then
   `b`, and deltas plus one then minus one modulo 10; select the first canonical
   state whose next active result digit differs;
5. `equal_token_length_destroyed_history`: replace every emitted-history token
   with non-format ID 100, preserving exact history token count.

The five clocks are the generic non-EOS runner-up and fixed IDs 211, 233,
39, and 100. For every clock and history branch, a fresh prefix contains the
exact initial prompt tokens, that branch's history tokens, and the boundary
clock token. It then receives one independent, uninterrupted, fixed 768-token
cached continuation under the same clock rule. Parsing and scoring happen only
after all such calls finish. This secondary causal screen deliberately injects
offline-edited generated history; it is not an autonomous trace arm and cannot
be counted in nominal prefix scores.

Only destroyed history is guaranteed equal-token-length. Digit substitutions
may change tokenizer length and their exact counts are reported. No branch is
claimed compute-matched.

### Compound fresh latest-state re-encoding

A separate external context renders a fresh exact core prompt with the latest
model-emitted `S1` branch in the sole `State:` slot. It jointly removes `S0` and
prior generated-history tokens, resets all token positions, and canonicalizes
the latest-state surface. For intact, carry-flip, written-result-flip, and
active-operand branches, it uses ordinary EOS stopping with the same 768-token
cap. It never uses the equal-length destroyed-state prefix.

This is one compound fresh-latest-state re-encoding/canonicalization package.
It is neither an autonomous trace arm nor a position-, surface-, and
clock-matched stale-context contrast. A recovery is attributable only to the
compound package and cannot identify any specific context-selection or
architectural mechanism.

### Finite negative control: post-hoc KV slicing

The supplied no-weight cache-pruning observation is frozen as a descriptive
negative control with three variants: latest-state KV only, immutable prefix
plus latest-state KV, and stale-`S0` key deletion while retaining contextualized
suffix/latest tensors. This evaluator does not rerun those variants because the
completed finite probe already falsified post-hoc slicing as a sufficient test.
It records them as
`posthoc_kv_slicing_negative_control.status=descriptive_supplied_development_negative_not_reexecuted`.

No zero-weight cache-surgery mechanism is promised. A later mechanism test must
apply attention masking or segment isolation while representations are
constructed/trained, or add a genuinely position-, surface-, and clock-matched
contrast. Fresh re-encoding alone does not supply that identification.

## 7. Frozen offline scoring

The scorer independently reconstructs the complete decimal oracle from the
initial DWS state after generation. It also verifies the frozen heldout
`expected_states` and `expected_answer` against that reconstruction.
Full report validation reopens the exact hash-verified heldout bytes, reruns the
frozen selector, and binds each report index to that selected row's exact ID,
split, operation, width, initial state, and selection digest. Every oracle record
also contains the exact selected-row SHA-256, expected-state-list SHA-256, case
ID, and expected answer. Missing, substituted, reordered, or internally
self-consistent but non-heldout case/oracle content is a failure; no heldout
verification field is hardcoded.

For each primary arm, every line containing `dws:` is inspected. A DWS line is
valid only when the whole line is the exact canonical serialization and passes
state invariants. Malformed candidates, repeated states, and any DWS candidate
after the first valid `z=1` state are recorded separately. The scorer reports:

- first-state exactness;
- longest exact state prefix;
- full exact trace through the first `z=1` state;
- exact terminal state and exact final `r,c` tape;
- terminal tape-derived answer and any explicit `answer=<integer>` line;
- all-line canonicality, transition grammar, response grammar;
- all EOS positions, logits, replacements, generated-token counts, and
  serialization diagnostics.

For each clock, `intact` reports both nominal second-state exactness against the
episode oracle and adjacent-transition exactness against `T(emitted state)`.
These are distinct.

Each carry, written-result, and operand intervention reports:

- full counterfactual target-state exactness;
- whether the whole output changed;
- endpoint-level intact target, counterfactual target, intact observation, and
  counterfactual observation;
- whether the target changed, whether output changed, each endpoint's target
  exactness, and paired target-switch exactness.

Paired target-switch exactness requires a changed target, a changed observed
endpoint, exact intact endpoint response, and exact counterfactual endpoint
response. Carry uses the next active result digit as its noncompensatory switch
endpoint and also reports next carry. Written-result uses retained `r[0]`.
Operand uses the next active result digit and perturbed operand digit.

Destroyed history reports whether intact adjacent transition was exact,
whether destroyed history still matched that target, whether output changed,
and paired loss (`intact exact && destroyed not exact`). Full-target exactness
never rescues target-switch, and target-switch never rescues full-target
exactness.

The aggregate denominator for each carry clock is all 100 planned episodes;
missing or malformed episodes count as failures. Unless paired carry
target-switch is exact in all 100, that clock receives
`carry_target_switch_noncompensatory_veto=true`. Any clock veto makes the global
carry-use veto true. This strict veto cannot be overridden by nominal prefix,
terminal, serialization, full-target, written-result, operand, or destroyed-
history scores.

The compound fresh re-encoding package separately reports intact adjacent
exactness, carry full-target exactness, carry output switch, carry paired switch,
written-result and operand endpoints, and per-case paired carry recovery versus
the full-history LF clock. Results are split at widths 4, 6, and 8 on both the
100-case board and frozen 12-case subset. The descriptive compound screen pass
requires every check below:

| Gate | Overall minimum | Every-width minimum |
|---|---:|---:|
| compound-fresh intact adjacent exact | 0.90 | 0.80 |
| compound-fresh carry full-target exact | 0.75 | 0.65 |
| compound-fresh carry output switch | 0.90 | 0.80 |
| compound-fresh carry paired switch | 0.75 | 0.70 |
| paired-switch rate gain over full-history LF | 0.40 | 0.30 |

Passing these gates emits only
`compound_fresh_reencoding_screen_pass=true`. Because the contrast changes
context, positions, surface form, and stopping clock together, this flag is not
a mechanism signal. It does not clear the full-history carry veto, change
full-state recurrence from NO-GO, or authorize promotion.

## 8. Output and custody

The report schema is exactly
`r12_dws_eos_suppressed_field_screen_dev_v4` under protocol
`R12-DWS-EOS-SUPPRESSED-FIELD-SCREEN-DEV-v4`.

The exact top-level JSON keys are:

```text
schema, protocol, development_only, claim_boundary, frozen_contract,
execution, aggregate, cases, adjudication, wrapper_acceptance,
generator_attestation
```

`wrapper_acceptance` uses exact schema
`r12_dws_eos_wrapper_acceptance_v8`. The canonical report package has exactly
`schema, report` under `r12_dws_eos_canonical_report_bundle_v5`; it has no
completion field. The separately published marker uses
`r12_dws_eos_post_publication_marker_v5` and is not acceptance. Durable
acceptance requires the distinct
`r12_dws_eos_durable_post_fsync_acceptance_receipt_v1` record with status
`wrapper_durable_post_fsync_acceptance_complete`.

Every case has exactly:

```text
case_id, split, operation, width, selection_sha256, initial_state, prompt,
oracle, primary_arms, field_screen
```

Every primary arm has exactly:

```text
mode, decode_prompt_token_ids, decode_prompt_token_ids_sha256,
decode_prompt_token_count, generated_token_ids, generated_token_count,
content_token_count, generated_token_ids_sha256, stop_reason, fixed_budget,
eos_mask_applied_positions, model_selected_eos_positions, override_positions,
override_count, eos_events, response_text, response_sha256, trace_score
```

The validator rejects extra or missing keys at these levels, wrong arm sets,
wrong case order/count, or a changed adjudication. It mechanically replays each
decode's mode and arm identity, exact prompt-token array/hash/count, generated and content-token
counts, generated-token hash, mask positions, model-selected EOS positions,
absolute EOS positions, replacement identity, override count, and ordinary-stop
versus fixed-budget rule. Fixed-budget arms must contain exactly 768 non-EOS
generated tokens; generic masking must list every generated position and no
other arm may report a mask. Ordinary EOS may appear only as the final token of
an EOS-stopped decode. Full validation reloads the hash-verified frozen
tokenizer, re-tokenizes each initial and branch prompt, re-decodes every content
token list, and verifies all reported runner-up and replacement token text.

Before semantic checks, every accepted report, context, authorization,
attestation, verifier record, package, marker, and receipt must be a recursively plain
JSON tree. Dict/list subclasses, bool-as-int, int-as-float, float-as-int, custom
numeric subclasses, NaN, infinity, and overflow-produced nonfinite values are
rejected. Counters and token IDs require exact built-in integers; logits and
rates require exact finite built-in floats where specified. Recomputed trace,
intervention, aggregate, and adjudication objects use type-strict recursive
equality so Python numeric aliases cannot pass by `1 == 1.0`.

The validator independently rebuilds each oracle and primary trace score. For
every available field screen it reconstructs the intact, carry, written-result,
active-operand, and destroyed-history branch contracts from the emitted state;
reconstructs each branch target; reparses each first continuation state; and
recomputes full-target, whole-output-switch, endpoint target-switch,
history-destruction, compound-fresh-re-encoding, aggregate, and adjudication
fields. Both the fresh-re-encoding record and its detail must state
`external_reencoding=true` and the exact fresh-core-prompt mode. No reported
full-target value can substitute for a failed paired response during replay.

### Crash-atomic publication

The report is serialized as canonical UTF-8 JSON with sorted keys, compact
separators, one final LF, duplicate-key rejection on readback, and no non-finite
JSON numbers. The wrapper creates distinct random generator and verifier
Ed25519 keys in anonymous sealed memfds. The acceptance context records their
public keys, private-key hashes, and exact seal mask. Only the generator receives
the generator descriptor, and only the independent verifier receives the
verifier descriptor. Marker authority instead comes from the externally signed
run authorization and its bound, post-exec broker-delivered sealed delegated
marker descriptor. No wrapper-created or report-embedded key can root
acceptance. Bash never holds any authority-bearing private key. Before asking
the external broker for that key, the exec-replaced Linux Python process sets
itself non-dumpable with `prctl(PR_SET_DUMPABLE, 0)` and verifies
`PR_GET_DUMPABLE == 0`, preventing same-UID processes from inspecting its
non-inherited descriptors or memory through proc/ptrace. Production root
private material is never available to the wrapper. No private-key bytes are
written into the report or marker.

Generation receives no output-directory descriptor. It writes the complete
report only to a wrapper-owned anonymous mode-`0600` descriptor and signs a
canonical attestation that binds the report-body hash, sealed evaluator/model
hashes, wrapper and manifest hashes, parsed Slurm job ID, and nonce. The wrapper
validates that signature and the full report before publishing a hidden private
candidate. Candidate publication uses an exclusive random same-directory temp,
write/fsync/readback/full replay, mode `0400`, and Linux
`renameat2(..., RENAME_NOREPLACE)` through the held output dirfd. It then fsyncs
the parent and verifies exact candidate inode, mode, link count, size, bytes,
and full schema through the descriptor retained from creation. Before candidate
rename, while the private temp pathname still names that creation-held inode,
the publisher also opens and validates a non-inheritable `O_RDONLY` descriptor
for the verifier. The wrapper never reopens the published candidate pathname.
Before candidate consumption, the wrapper parses the exact creation-held bytes,
requires equality to the in-memory report and read-only verifier binding, and
revalidates the path, device, inode, mode, link count, size, mtime, and bytes. A
move-then-error removes no pathname. Exact rollback and successful candidate consumption instead use
`RENAME_NOREPLACE` to move the held inode to a deterministic hidden rollback
quarantine, fsync the parent, and revalidate the quarantine pathname against the
same held descriptor and bytes. The wrapper retains that descriptor through
canonical publication and performs a final quarantine hold before success. An
inode substitution or same-inode byte mutation is preserved and is fatal.

The independent verifier receives only a read-only descriptor for that
mode-`0400`, single-link candidate, not the output dirfd. It reruns the exact
heldout selection and semantic replay, checks the signed generator attestation,
rechecks source/manifest/executable/Slurm custody, rehashes every frozen input at
its exact frozen path, records an independent loaded-runtime snapshot, and signs
one canonical verifier record with its distinct Ed25519 key. That signature
binds candidate hash and inode, report-body and generator-attestation hashes,
validation-time frozen input paths/hashes, and validation-time runtime
observation. The wrapper checks the signed record against the still-open
candidate descriptor; generation cannot self-rehash or fabricate it.

Before canonical publication the wrapper renames the now-verified private
candidate with `RENAME_NOREPLACE` to a deterministic retained rollback
quarantine and fsyncs the held parent; it never pathname-unlinks the candidate.
It revalidates that quarantine against the exact descriptor retained from the
candidate temp's creation, and keeps that descriptor open through canonical
publication and the final quarantine hold. It then creates a new exclusive
random temp in the same held directory and writes a canonical mode-`0444`
report package. That package contains only its schema and the unchanged signed
report. It contains no acceptance receipt, completion status, or wrapper
acceptance signature. It is fully parsed and replayed, fsynced, and atomically
renamed to the absent canonical name with `RENAME_NOREPLACE`.

Canonical rename is not acceptance. The wrapper fsyncs the parent; rechecks that
the output pathname still names the held current-UID mode-`0700` directory
inode; reads and fully validates the final canonical inode; rechecks source,
external-manifest inode and bytes, held executables, actual wrapper, the complete
parsed Slurm allocation and cluster config; and requires final mode `0444`, link
count 1, exact size and bytes. Only after all those post-publication checks
succeed does the wrapper exclusively create the deterministic durable-receipt
name as an empty inode, change it to mode `0444`, fsync that inode, and fsync the
parent. The wrapper retains its original writable `O_SYNC` descriptor and fails
closed if synchronous-open semantics are unavailable. This durable empty
slot is intentionally invalid evidence; replay rejects it until valid signed
receipt bytes are later written and fsynced through that retained descriptor.
The wrapper then generates a fresh 256-bit marker nonce and creates a separate
exclusive commit-marker temp.

The commit-marker builder consumes the report descriptor retained from its
exclusive temp creation through canonical rename. It requires the held dirfd
pathname, full descriptor policy identity, and exact bytes to remain equal
before it can use the externally authorized delegated marker key. The signed
marker binds the
production authorization hash, authority key ID/fingerprint, authorization
sequence, delegated public key, canonical report hash,
full final inode/mode/link-count/size record, its own future final inode, both
signed attestation hashes, complete parsed Slurm allocation, wrapper/manifest/
runtime hashes, frozen input observation, marker nonce and timestamp, and five
explicit report-publication checks. It is read back and signature-validated,
changed to mode `0444`, and atomically renamed without replacement to the
deterministic sidecar name. The marker status is only
`wrapper_post_publication_marker_complete`; the marker never claims that its own
rename is parent-durable and never constitutes acceptance.
The marker signature is accepted only after independently verifying the
production signature over the embedded run authorization and its exact marker
scope delegation.
Self-rehashing cannot edit any signed field or replace the authority root.

Immediately after marker rename, the deterministic failure-injection boundary
`before_commit_marker_parent_fsync` runs before any parent fsync. The wrapper
then fsyncs the parent, reads and fully validates the marker through its retained
descriptor, semantically replays report plus marker from both retained
descriptors, and repeats all final source,
authorization, Slurm/cgroup/GPU, runtime, and output-parent checks. It next
requires both canonical pathnames still to name the exact transaction-owned
device/inode/mode/link/size/mtime and bytes, fsyncs those same held descriptors,
and fsyncs the held parent. These are fresh durability operations, not an
assertion that an earlier fsync happened.

Only after those operations return does the wrapper build the externally
authorized durable acceptance receipt. The production-signed run authorization
must explicitly delegate the receipt schema to the same sealed publication key.
The signed receipt binds report, marker, and receipt names; full report and
marker inode/mode/link-count/size records; its own preflushed inode; report and
marker hashes; output parent; authorization hash, sequence, authority identity,
and delegated key; a fresh nonce and timestamp; and the just-completed retained-
descriptor revalidation, file-fsync, parent-fsync, and final-wrapper checks. The
wrapper writes the bytes
only through the retained `O_SYNC` descriptor for the already parent-fsynced
empty slot. Each completed write is synchronous; an interrupted partial prefix
is invalid JSON, while a completed final write is already durable. The wrapper
then fsyncs that receipt inode again and reads it back. There is no receipt rename after
the durability checks, so there is no second rename-before-parent-fsync
acceptance window.

The wrapper then holds report, marker, and receipt descriptors concurrently,
fully replays the same three-artifact parser/validator twice, and requires every
canonical path plus descriptor to retain the transaction-owned device, inode,
mode, link count, size, mtime, and exact bytes. Restart replay likewise opens
the external manifest, running evaluator source, report, marker, and receipt
once, retains all descriptors through semantic replay, and performs a final
exact-byte hold before returning. A symlink, hardlink, rename, same-inode byte
mutation, or replacement of any artifact is fatal. Wrapper success is printed
only after this final three-artifact validation returns.

Normal failure never pathname-unlinks a transaction-derived report, marker,
receipt, candidate, or temp. It moves each still-exact held entry with
`RENAME_NOREPLACE` to a deterministic rollback quarantine, fsyncs the parent,
and revalidates full policy identity and exact bytes there. A swap before rename
moves the foreign entry into quarantine, preserves both inodes, and makes
rollback fail closed. Startup cleanup occurs only after the
production signature, authority identity, live Slurm allocation, output inode,
delegated key identity, source commit, manifest path/hash, and prereg hash have
all cross-bound, and after the exact lease binding validates. All canonical names must
be absent, the complete observed protocol-residue set must exactly equal the
externally signed cleanup set, and every entry is fully read and identity-checked
before mutation. Each active authorized entry is moved with
`RENAME_NOREPLACE` to a deterministic protocol quarantine name derived from the
signed entry, and the held parent is fsynced immediately after that rename. The
entry is re-read and must retain the signed device, inode, bytes, mode, link
count, and size. Startup stale cleanup never calls pathname unlink on either an
active entry or quarantine. This is intentional: POSIX offers no atomic
"unlink this pathname only if it still names inode X" operation against a
hostile same-UID namespace mutator. A replacement after rename is therefore
preserved at quarantine and causes failure rather than risking deletion of a
foreign inode.

A hard death after either rollback or stale-cleanup quarantine rename leaves an
inert deterministic quarantine. A later authorization may list that exact
quarantine name and full inode/hash record, including after a new job or nonce.
Cleanup moves a signed rollback quarantine into terminal stale-cleanup
quarantine; it recognizes and revalidates an already terminal quarantine without
moving or unlinking it, then permits the publication transaction to continue.
Physical deletion of terminal quarantines is outside
this wrapper and requires separate offline custody. An omitted, additional,
live-owner, unrelated, or differently scoped residue is never mutated. An empty
signed list authorizes no cleanup. The directory lease is advisory and protects
only cooperating publishers; hostile same-UID denial of service remains
possible and real Lustre `flock` behavior is still unqualified. A pre-existing
canonical name is never replaced. A hard process death after
canonical report fsync can leave a report without a marker. A hard death at the
injected pre-parent-fsync boundary can leave a visible report, valid marker, and
durable empty receipt slot. A hard death after marker parent fsync but during
receipt writes can leave an empty or invalid partial receipt. Every such state is an
uncommitted partial publication, and normal read-only replay rejects it. Only a
fully signed, inode-bound receipt whose bytes were written after the fresh
report, marker, and parent fsync sequence authorizes replay. Protocol-shaped
residue from a different run/stage identity requires a new externally signed
recovery authorization; unrelated names are untouched.

No mutating durability-recovery mode is implemented. Any future idempotent
recovery mode must require separate external authorization, reopen and fsync the
canonical report and marker, fsync the parent, and only then publish a new
durable acceptance receipt. It must not convert a marker claim into evidence or
self-attest that any fsync occurred before recovery began.

Acceptance is independently replayable after the Slurm job ends only from the
triple of canonical report package, valid post-publication marker, and valid
durable post-fsync acceptance receipt, using
the source-pinned production public key, its externally signed run
authorization, the delegated marker public key, the generator/verifier public
keys, the externally frozen
runtime-source manifest, the exact frozen tokenizer/heldout/checkpoint paths,
and all three inode bindings. The report/marker pair is never accepted. Replay does
not require a live Slurm job and cannot silently replace parsed Slurm fields with
environment strings. A test-scope authorization, marker, or receipt is rejected by
production replay. The evaluator's generation CLI has no output-dir or
accepted-output descriptor and no canonical publication path; wrapper authority
is the held dirfd, externally authorized sealed delegated marker descriptor,
and independently signed verifier record.

The execution record embeds the external manifest path, hash, device, inode,
UID, mode, and size; exact source root and commit; all five source hashes;
clean-status and `git show` equality assertions; exact runtime identity; sealed
evaluator/model memfd records; and isolated Python startup mode.
`wrapper_acceptance` binds the report to complete parsed `scontrol`, `sacct`,
cluster configuration, devices-cgroup, `/dev` major/minor, `nvidia-smi`, PCI
sysfs, CUDA selector, and Slurm selector observations. It includes exact job,
user, command, host, partition, node/task/CPU, memory, time, requeue, typed TRES,
GPU UUID/BDF/full-MIG state, raw hashes, and one-to-one normalized selector
mapping. Slurm environment name/cluster strings and substring matching are not
evidence. The context also binds the output-directory inode, wrapper, manifest
and runtime hashes, nonce, candidate/final names, external authority-key file,
signed authorization file/hash/body, delegated marker key record, generator/
verifier public keys, and sealed generator hashes.

The external-controller ceiling is deliberately **descriptive and unexecuted**
in this implementation: a future controller could feed every oracle state into
a fresh canonical one-step prompt. Its absence is explicit in the frozen output
contract and cannot be silently represented as an autonomous comparator.

The maximum frozen workload is 100 ordinary primary calls, 500 fixed-budget
primary calls (`384,000` generated tokens), up to 2,500 fixed-budget
full-history field calls (`1,920,000` generated tokens), and up to 400
ordinary-stop compound-fresh calls capped at 768 tokens. Missing or malformed
first states reduce secondary calls but remain endpoint failures. No throughput
or hardware-equivalence claim follows from these accounting bounds.

### Required Linux/Lustre qualification

Command `linux-smoke` is an explicit non-scientific qualification surface. It
requires Linux, an exact reviewed evaluator SHA-256, and a current-UID-owned
mode-`0700` root on a mount parsed exactly as `lustre`. It performs no tokenizer,
checkpoint, CUDA, model, or DWS decode. It exercises real
`renameat2(RENAME_NOREPLACE)` success and collision, sealed memfd readback,
`prctl` non-dumpability, file and directory fsync, injected failures immediately
before and after directory fsync with exact held-inode rollback quarantine, random crash-temp
cleanup, symlink and hardlink rejection, held-dirfd pathname substitution, and
Lustre rename/fsync/reopen durability. It also acquires the production publisher
lease helper, forks a separate concurrent process that opens the same directory
and must lose a real nonblocking `flock`, validates an ephemeral
mechanics-only Ed25519 authorization over an exact stale inode, executes the
production stale-quarantine helper, substitutes a foreign inode after the
quarantine rename, and requires that foreign inode to survive. These are
reported as separate structured evidence records only after each observation
returns. The ephemeral mechanics signature has no production or test authority.
Success requires exactly 16 ordered checks: publisher lease acquisition,
concurrent `flock` rejection, production `SCM_RIGHTS` receiver traversal,
signed stale quarantine, foreign-inode preservation, no-replace rename,
directory-fsync rollback, three receipt crash/replay outcomes, held-evaluator
pathname substitution before the first child, symlink rejection, hardlink
rejection, random crash-temp cleanup, held-directory substitution rejection,
and Lustre rename/fsync/reopen.

The qualification first exercises the production canonical broker-request
builder and production connected message-boundary `SCM_RIGHTS` receiver. Each
crash child repeats that same transfer, obtains only a deterministic
mechanics-only signing key, and binds the exact broker request hash and public
key into distinct qualification report, marker, and receipt schemas. Those
artifacts carry valid Ed25519 signatures under the transferred ephemeral key,
but the explicit
`mechanics_only_ephemeral_broker_key_no_production_or_test_authority` boundary
cannot satisfy production or test-authority replay.

The same qualification runs three deterministic cases in separate child
processes using fixed hard-exit codes, then waits for each child and independently
opens and replays its private scenario directory. Every child acquires the
production publisher lease and invokes the same
`publish_accepted_bundle_exclusive` transaction as production with inert
qualification-only parsers and builders. Thus qualification traverses the same
exclusive temp creation, no-replace report and marker renames, retained report/
marker descriptors, parent and file fsyncs, empty mode-`0444` `O_SYNC` receipt
slot, signed receipt construction, complete synchronous write, exact rollback,
and final shared three-descriptor replay code. The first child exits after marker
creation but before its publication-parent fsync, leaving an empty receipt. The
second writes exactly a nonempty proper receipt prefix through the retained
`O_SYNC` descriptor and exits. The final child exits after the shared complete
`O_SYNC` write helper returns but before its redundant explicit receipt fsync.

Before any qualification child is spawned, the parent reads the exact reviewed
evaluator into one anonymous memfd, changes it to mode `0400`, applies and
verifies the complete Linux seal mask, and records its SHA-256, byte count,
descriptor kind, and seals. Each child inherits only that held descriptor. A
minimal isolated `-I -S -B -c` bootstrap reads the descriptor with `pread`, requires the
exact byte count and SHA-256, compiles those bytes, and executes them as the
worker; the mutable evaluator pathname is not the executed image. The worker
again verifies the same held descriptor and embeds that exact source record in
the version-3 qualification report and receipt. Parent replay requires exact
source-record equality. Production `linux-smoke` first creates a private
mode-`0444` exact-byte evaluator probe in its Lustre transaction. Its
preregistered `first_child_before_spawn` callback renames the original probe
inode to retained evidence, fsyncs the parent, creates and fsyncs a hostile
replacement at the probe pathname, and records both inode/hash identities. The
first child must still execute the held reviewed bytes; absence of the callback
or its retained/substituted evidence is fatal. The non-Linux unit-test
fallback uses an unlinked held file descriptor and is explicitly development
only; it is not Linux/Lustre qualification evidence.

Fresh parent replay must reject the pre-parent-fsync empty slot and the partial
receipt, and must accept only the canonical complete receipt whose inode and
hash bindings state all required report-file, marker-file, receipt-slot,
retained-descriptor, and parent fsync steps. Report and marker are independently replayed in
all three cases so an unrelated failure cannot count as receipt rejection. The
qualification requires receipt sizes to progress from zero to a strict partial
prefix to a complete payload. It cleans each scenario plus its private random
transaction directory and fsyncs each parent before success.

This mode has not been run by this revision. Its schema is
`r12_dws_eos_linux_publication_qualification_v6`, and success status states only
that the listed Linux/Lustre mechanics were exercised for exact-byte review. It
does not infer that an unlisted production, authority, scientific, CUDA, or H100
path ran. A successful Stokes/Linux run is
mechanical qualification evidence only; it does not authorize production,
validate the external signer/controller integration, exercise H100 decoding, or
produce a scientific result. The output additionally contains one canonical
`r12_dws_eos_linux_qualification_authorization_receipt_v1` object. Its exact
keys are `schema, qualification_result, accepted_publication, signature_hex`;
`accepted_publication` has exactly `broker_request, report, marker, receipt`.
Replay verifies all inner hashes, schemas, inode records, durability booleans,
broker identity, and Ed25519 signatures, then verifies the outer signature over
the complete result and accepted publication. The mechanics-only key grants no
authority: a separately reviewed production signer must embed this exact object
in a fresh version-5 authorization, whose signed body is then embedded in the
scientific report and hash-bound by the final marker and durable receipt. Linux
publication and H100 remain NO-GO until the exact committed bytes pass that run
and a fresh review accepts and binds its output.

### Local repair verification

On 2026-07-18, the unrepaired suite first reproduced the independent symlinked-
Python result exactly: 2 failed and 85 passed. After this repair, the isolated
macOS development gate passed all 92 warning-strict tests under the same
symlinked `python3`, with third-party pytest plugin autoload and
pytest cache disabled. The seven-test focused defect gate also passed. Ruff
formatting was applied only to the evaluator and its test;
`ruff check` then passed both files. Isolated
no-bytecode in-memory compilation under `python3 -I -S -B -W error` passed both
Python files, and `bash -n` passed the wrapper. ASCII and whitespace checks also
passed all four owned files. The tests cover the real message-boundary
`SCM_RIGHTS` broker transfer and socket-identity race, strict canonical point and
scalar Ed25519 bounds, every canonical small-order encoding, disabled
site/`.pth` customization, loader/startup neutralization order, explicit
verified-package activation order, held descriptor mode/link/same-inode byte
mutation, both preauthorization and later mapped-device/inode pathname
substitution, Git environment/config/fsmonitor suppression, a real forked
concurrent publisher `flock`, signed exact stale quarantine, prior-job/nonce
quarantine discovery, a rollback validation-to-rename pathname swap that
preserves both inodes, foreign substitution after quarantine rename, actual hard
process death and signed quarantine recovery, report/marker/receipt same-inode
mutation through publication and restart final holds, candidate creation-FD
custody across pathname substitution, canonical Python identity under symlinked
invocation, held-evaluator substitution wired into production `linux-smoke`,
replayable qualification-receipt authorization/report binding, rename-only
retained candidate evidence, shared production broker/publisher qualification
mechanics, and exact Slurm/GPU cross-binding. They do not substitute for a reviewed production
broker/controller, Linux/Lustre, Stokes, Newton, H100, or independent exact-byte
review.

## 9. Claim boundary

The only allowed conclusion is a development characterization of token-clocked
serialization and field response under externally overridden EOS. Full-state
recurrence remains NO-GO. Fixed token overrides are external resources. The
screen cannot establish autonomous halting, autonomous recurrence, robust carry
use, arithmetic reasoning, broad transfer, hidden-set performance, promotion
readiness, a new computational primitive, or a stale-context mechanism. A
compound-fresh recovery only characterizes that joint re-encoding package; it
does not nominate a component without a matched future contrast.
Post-hoc KV slicing remains the finite negative control and is not promised as
a zero-weight mechanism. The present repaired working tree contains no sealed
execution artifact: no launch is permitted until exact commit review and the
external manifest, production authorization, signer/controller integration,
and Linux/Lustre qualification are complete. No job or artifact publication was
performed and no result is claimed by this revision. CPU mechanics are GO only
for local development validation. Linux publication, Newton deployment, and
H100 launch remain NO-GO pending fresh review of the repaired committed bytes
and a real Stokes/Linux qualification run.

## 10. Required next actions

1. Obtain a fresh independent hostile review of the exact four-file hashes,
   before commit, qualification, cluster access, publication, or scientific
   execution. Concentrate on strict Ed25519 canonicality, full RECORD and native
   mapping closure, authorization/broker ordering, retained descriptor policy,
   signed cleanup/lease/prior-quarantine recovery, the shared qualification
   publisher path, and single-node cross-binding.
2. Only after review, commit exactly those reviewed hashes and verify that the
   commit preserves them. Then generate the external version-7 runtime manifest
   from independently observed bytes.
3. Run the same 92-test warning-strict suite on the target Linux runtime, then
   run `linux-smoke` on a real Lustre directory and preserve its complete
   version-6 output and exact nested authorization receipt. Confirm real
   `flock`, sealed memfd, first-child evaluator-path substitution,
   `/proc/self/maps`, hostname/Slurm, `O_SYNC`, no-replace rename, fsync,
   crash/replay, exact final receipt verification, and foreign-inode behavior
   under that filesystem and kernel.
4. Only after independent replay accepts that exact qualification output may
   the external signer create a version-5 production authorization embedding
   its exact `authorization_receipt`. Any stale cleanup list must be enumerated
   and signed out of process from exact inode/content evidence; never infer
   mutation authority from a filename.
5. Keep publication, Newton deployment, and H100 execution blocked unless every
   preceding gate passes on one exact committed source/runtime identity. A
   failure requires a new reviewed revision, not an exception or relaxed field.
