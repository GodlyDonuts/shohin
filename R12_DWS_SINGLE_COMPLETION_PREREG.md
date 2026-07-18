# R12 DWS Single-Completion and Commit-Reencode Preregistration

**Protocol:** `R12-DWS-SINGLE-COMPLETION-DEV-v1`

**Status:** hostile-review repair;
COMMIT/CPU MECHANICS/PUBLICATION/LINUX/STOKES/NEWTON/H100 NO-GO pending fresh
independent review. This document authorizes no production corpus, model fit,
accelerator job, checkpoint promotion, confirmation-board access, or capability
claim.

**Owned implementation surface:**

```text
R12_DWS_SINGLE_COMPLETION_PREREG.md
pipeline/generate_dws_single_completion_v1.py
pipeline/test_generate_dws_single_completion_v1.py
```

The implementation may read, but may not modify, the frozen DWS solver,
existing one-step row builder, tokenizer, parent-checkpoint receipt, optimizer
implementations, and held-out cross-width source. Every such byte dependency is
bound from outside the generated bundle.

## 1. Smallest untried experiment

Existing DRS and factorial training decomposes each episode into separate
one-transition, digit-readout, and final-answer rows. Existing recurrent
evaluators call the model once per microstate and then issue a separate final
call. Repository and artifact inspection found no canonical training arm whose
single target contains every successive DWS state, the answer, and supervised
EOS in one logical completion.

This protocol asks two narrow questions:

1. At equal source identities, supervised token budget, active context, dense
   positions, optimizer updates, and multiline exposure, does canonical
   full-trace supervision improve generated-state composition over decomposed
   one-step and discontinuous multiline controls?
2. At equal data, parameters, tokens, and re-encode forwards, does the complete
   model-triggered commit/re-encode package improve composition over ordinary
   full-history attention?

The scored mechanism is a **model-triggered external commit/re-encode runtime**.
It is not autonomous base-model reasoning, an ordinary uninterrupted-KV decode,
or a no-host-schedule claim. A positive context contrast supports only the
complete commit-reencode package. It does not identify stale-source retirement,
masking, re-encoding, extra forward depth, or cache replacement as the cause.
Component attribution awaits separate SCERT extra-depth/no-retirement,
mask-only, contaminated-replay, and fresh-host-prompt controls.

## 2. Frozen development evidence

These observations motivated the factorial. They are non-preregistered
development evidence and grant no promotion authority.

### 2.1 Generated-history intervention diagnostic

On 16 cases:

| Endpoint | Result |
|---|---:|
| nominal second-state exact | `5/16` |
| carry-flip target exact | `7/16` |
| carry-flip paired output target-switch | `3/16` |
| true-carry-one nominal wrong but matching flipped target | `6/8` |
| written-result flip changed output | `7/16` |
| written-result flip full-target exact | `1/16` |

Counterfactual exactness is confounded by default-carry-zero behavior. A causal
success requires both nominal and counterfactual target exactness and a paired
output switch in the target direction. Carry target-switch is a veto. These
results do not already justify full-trace training.

### 2.2 Fresh-latest-state diagnostic and cross-width replication

For eight carry interventions, continuing from full history `S0+generated-S1`
was paired-causal exact on `1/8`; a fresh prompt containing only the same `S1`
was exact on `7/8`.

The exact frozen 12-case replication contains two cases in every
`width {4,6,8} x operation {add,sub}` cell.

```text
case-list SHA-256:
1dc75ec7995e61a85f7bec9ae1fa62aa1adaf71bd46172e880aea901482396b9

source-board SHA-256:
89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646
```

| Mode | Overall | width 4 | width 6 | width 8 |
|---|---:|---:|---:|---:|
| full-history `S0+S1` paired carry exact | `2/12` | `0/4` | `2/4` | `0/4` |
| fresh latest-state prompt paired carry exact | `10/12` | `3/4` | `4/4` | `3/4` |

Fresh-latest-state nominal exact was `12/12`, counterfactual exact was `10/12`,
and output switch was `12/12`. This supports testing context retirement but does
not isolate a stale-source-specific mechanism. The preregistered contrast is
the complete commit-reencode package.

### 2.3 Finite no-weight cache-pruning negative control

Post-hoc pruning on a checkpoint trained under full causal attention failed:

- keeping only latest generated-state KV became malformed after one interval;
- keeping immutable prefix plus latest-state KV failed;
- dropping only stale `S0` keys while retaining contextualized prefix, suffix,
  and latest state remained approximately full-history on two cases.

Those retained representations were already contextualized by `S0`. Post-hoc
KV slicing is a finite negative control, not a sufficient test and not a
promised zero-weight solution. A valid treatment must isolate or re-encode
state during forward construction and training.

## 3. Frozen inputs and source authority

Production generation requires externally supplied values for all of:

- tokenizer locator path and SHA-256; the frozen tokenizer SHA-256 is
  `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`;
- parent-checkpoint SHA-256;
- cross-width source locator path and SHA-256;
- aggregate source-binding SHA-256 over every code/document dependency;
- aggregate runtime-binding SHA-256 over the executed reviewed modules, Python
  runtime, and tokenizer runtime described below;
- mode, generation seed, per-cell counts, and lane length;
- external sealed-root path, SHA-256, and byte count for verification.

The generated manifest has no authority to redefine these values. Rehashing a
changed plan, source receipt, parent checkpoint, authorization, promotion bit,
board, schedule, or pack into a new manifest must still fail independent
replay against the externally expected constants and current source bytes.

Tokenizer and cross-width paths are locators, not identity. The exact sealed
policy is `sha256_authenticated_regular_file_relocation_allowed`: each locator
must open as a regular non-symlink file under no-follow semantics, and the
externally supplied SHA-256 is the content identity. The path string is not
committed. Build and verification may therefore use different regular-file
locations only when the authenticated bytes are identical. The staging identity
commits the policy, and the sealed root carries a type-strict
`inputs.location_contract`; independent replay requires that exact contract.

The generator refuses to start if either `digitwise_protocol` or
`pipeline.generate_digitwise_recurrent_v1` was already loaded. It imports each
module only after fixing its one allowed source path. It performs a stable
no-follow source read, compiles and executes those exact source bytes directly
without accepting a bytecode-cache payload, rereads the source after execution,
and requires the module's resolved `__file__` and import-spec origin to equal
that path. Before
generation and before independent replay, the runtime receipt binds and
cross-checks each executed module's resolved path, byte count, and SHA-256
against the externally reviewed source-binding bytes. Replaced module objects
or bound callables fail closed. Runtime-binding schema v6 additionally captures
every Python function defined by either reviewed module, including helpers
reached transitively by `rows_from_episode`. Each live function must retain its
strict function type, exact function object, exact code object, defaults,
keyword defaults, closure, and closure-cell identities. The receipt commits
deterministic hashes of canonical recursive code-state bytes and callable state.
Generator aliases
and the row builder's imported protocol aliases must still name those same
authenticated implementations. Consumption does not call those mutable source
exports directly: the generator creates exact-code clones whose private globals
replace every reviewed helper with another exact-code clone. The row builder's
dynamic `digitwise_protocol` import is routed through a private frozen import
view, so a transient source-module alias replacement cannot enter construction
or replay. Those clone-global and builtins mappings have one exact sealed
`dict` subtype, reject assignment, deletion, update, and in-place merge, and are
revalidated by exact key set and value identity at every runtime snapshot.

An append-only Python audit hook protects every reviewed source function, every
consumed clone, and the frozen import callable. Any attempted assignment to
`__code__`, `__defaults__`, or `__kwdefaults__` raises `ContractError` before
the mutation occurs. Hook installation is self-tested during import, the hook
protects its own implementation, and the runtime receipt commits the blocked
attributes, audit event, and protected-callable count. A swap-consume-restore
attack therefore cannot cross a snapshot window: the initial swap itself is
rejected and there is nothing to restore.

The generator's own live production functions are separately recreated after
all imports with one private frozen builtins mapping. The mapping contains the
exact key order and object identity of every builtins entry, rejects ordinary
mutation methods, and remains the `__builtins__` mapping for every recreated
module function and every production class method reached through
`FrozenTokenizer` or the pinned-directory wrapper. Every later-defined runtime
boundary function is created under that same mapping. Runtime validation
requires the public `builtins` module and the private mapping to retain the same
exact entries and requires each generator function to retain that private
builtins identity. Thus a public builtin replacement after validation cannot be
consumed by construction, recovery, publication, or replay and is rejected at
the next runtime boundary. Callable code receipts encode every code field and
nested code constant recursively from de-adapted `co_code` bytes instead of
marshaling a live code object; CPython quickening and string interning therefore
cannot make an honest receipt change during its own first validation call.

The generator is valid only when imported or directly executed under the
reviewed executable with all three flags `-I -S -B`. It requires and receipts
every field of `sys.flags`,
including isolated mode, ignored Python environment configuration, disabled
`site` and user-site startup, safe-path mode, and disabled bytecode writes.
`site`, `sitecustomize`, and `usercustomize` must remain absent. Thus the current
directory, `PYTHONPATH`, `PYTHONSTARTUP`, user site, `.pth` files, and repository
or site customization cannot configure executable startup. After that check,
the generator explicitly appends only the resolved `sysconfig` `purelib`
directory so the externally bound native tokenizer can be imported; this does
not execute `site` or any `.pth` file.

The same externally supplied runtime-binding SHA-256 commits to Python's
presented and resolved executable paths, executable byte count and SHA-256,
implementation, complete version string and version tuple, build, compiler,
API version, and complete isolated-startup receipt. It binds the executing
generator module object, current byte count and SHA-256, and cross-checks those
bytes against the source binding. Imported execution requires resolved
`__file__` and import-spec origin to be identical. CPython direct-script
execution has `__main__.__spec__ is None`; that exact state is captured and
allowed only for `__main__`, while resolved `__file__` must still equal the one
reviewed generator path. Any later change to `__name__`, `__file__`, or
`__spec__` fails closed. It
also binds every directly imported runtime module by exact module object and,
for file-backed modules/extensions, resolved path/bytes; frozen and built-in
modules are bound through the executable bytes. This inventory includes
`struct`/`_struct`, `hashlib`/`_hashlib`, `sysconfig`, filesystem modules, JSON,
regex, ctypes, and the remaining direct standard-library dependencies. Native
backends reached by those wrappers, including `posix`, `_json`, `_sre`, `_stat`,
and `_ctypes`, are included explicitly; built-in backends are bound through the
executable receipt and extension backends by their own resolved bytes.

Runtime-binding schema v6 also receipts every executed `json.encoder`,
`json.decoder`, and `json.scanner` module and every loaded
`importlib.metadata._adapters`, `_collections`, `_functools`, `_itertools`,
`_meta`, and `_text` submodule. Directly consumed `json.dumps`, `json.loads`,
JSON encoder/decoder methods and native accelerators, `importlib.import_module`,
`importlib.util` loaders, and `importlib.metadata` distribution/version exports
are captured by exact object identity and rechecked. Python implementations are
covered by the append-only code/default mutation guard. Serialization and
distribution checks call captured exports only; replacing a live export or an
executed submodule fails the next runtime boundary without consuming the
replacement.

The exact non-filesystem mutable-export inventory is 44 names. Control and
bootstrap consume `argparse.ArgumentParser`, `collections.namedtuple`,
`ctypes.PyDLL`, `importlib.import_module`,
`importlib.util`, `importlib.util.spec_from_file_location`,
`importlib.util.module_from_spec`,
`importlib.metadata.Distribution`,
`importlib.metadata.PackageNotFoundError`,
`importlib.metadata.distribution`, and `importlib.metadata.version`.
Serialization consumes `json.dumps`, `json.loads`, `json.JSONDecodeError`,
`json.JSONDecoder`, `json.JSONEncoder`, `json._default_decoder`,
`json._default_encoder`, `json.decoder.JSONDecodeError`,
`json.decoder.JSONDecoder`, `json.decoder.c_scanstring`,
`json.decoder.py_scanstring`, `json.decoder.scanstring`,
`json.encoder.JSONEncoder`, `json.encoder._make_iterencode`,
`json.encoder.c_encode_basestring`,
`json.encoder.c_encode_basestring_ascii`, `json.encoder.c_make_encoder`,
`json.encoder.encode_basestring`, `json.encoder.encode_basestring_ascii`,
`json.scanner.c_make_scanner`, `json.scanner.make_scanner`,
`json.scanner.py_make_scanner`, `_json.encode_basestring`,
`_json.encode_basestring_ascii`, `_json.make_encoder`, `_json.make_scanner`,
and `_json.scanstring`. Numeric, regex, startup, and runtime identity consume
`math.isfinite`, `re.compile`, `sysconfig.get_path`,
`platform.python_build`, `platform.python_compiler`, and
`platform.python_implementation`. Each name is captured by exact object
identity; Python implementations include code/default/closure identity and
mutation protection. Bootstrap-only calls dispatch through the captured alias.
The three `platform` values are frozen from those captured calls and are not
looked up again while constructing a receipt. Runtime validation checks every
live public export before accepting the frozen values.

The four callable class aliases reached directly by production code are
`collections.Counter`, `collections.defaultdict`, `collections.deque`, and
`pathlib.Path`. Their exact generator-global identities are receipted. All
generator globals that exist when bootstrap closes, not only names matching a
prefix, reject module assignment and deletion. The bound `sys.modules` and
`sys.path` container identities are also checked, and internal lookup uses
those bound containers. The production descriptors of `FrozenTokenizer` and
the pinned-directory wrapper are sealed against ordinary class assignment or
deletion after bootstrap. This inventory distinguishes protected aliases from
module bytes; a module-byte receipt alone is not treated as protection for a
mutable export.

The filesystem trust path captures all 22 Python-callable exports it consumes from
`os`/`posix`, `os.path`/`posixpath`, `stat`/`_stat`, `fcntl`, and `_ctypes`, plus
the 11 open, lock, separator, and collision constants that select their exact
behavior. The runtime receipt records each callable's strict type, Python code
and defaults when present, native implementation owner otherwise, and the
corresponding executable/module-byte binding. Construction, recovery, staging,
durability, readback, cleanup, and verification call only those captured
implementations rather than performing a fresh mutable module-attribute
lookup. The same append-only mutation guard protects Python-implemented
filesystem callables and all Python RNG methods against transient code/default
replacement. Atomic publication likewise uses one captured and typed
`renameatx_np`/`renameat2` process-image symbol and its frozen no-replace flag;
the symbol object, argument/result types, platform, null `errcheck` state, and
implementation binding are receipted and rechecked. Descriptor-relative
`unlinkat` binds the same mutable `_FuncPtr.errcheck` state. Both native
callables must start with `errcheck is None`, retain that exact null state at
every runtime snapshot, and recheck it after argument preparation immediately
before native dispatch. An audit hook therefore cannot install a callback in a
post-snapshot window, consume it during rename or cleanup, and remove it before
the next snapshot. The executing generator module is converted to a
sealed module type after bootstrap. Assignment or deletion of captured runtime
callables/constants, frozen mappings, reviewed-function aliases, protected
filesystem dispatch functions, or their mutation guards raises `ContractError`
before the binding changes. The seal methods and their exact live code/defaults
are audit-protected, receipted, and rechecked. Durability dispatch additionally
captures `open`, `write`, `fsync`, `fchmod`, and `close` in protected function
defaults, so changing either the public `os.fsync` export or a module binding
cannot redirect a consumed synchronization call.
Ordinary access to the executing module's `__dict__` returns a read-only mapping
view, so normal direct dictionary assignment cannot bypass the module seal.

The 22 filesystem exports are exactly `os.close`, `os.fchmod`, `os.fsencode`,
`os.fspath`, `os.fstat`, `os.fsync`, `os.geteuid`, `os.listdir`, `os.mkdir`,
`os.open`, `os.path.abspath`, `os.read`, `os.replace`, `os.stat`,
`os.strerror`, `os.write`, `fcntl.flock`, `ctypes.get_errno`, `stat.S_IMODE`,
`stat.S_ISDIR`, `stat.S_ISLNK`, and `stat.S_ISREG`. Cleanup does not dispatch
through mutable `os.unlink` or `os.rmdir`: it binds libc `unlinkat` from the
same `ctypes.PyDLL` process-image handle, receipts its type and signature, and
uses flag `0` for files and `AT_REMOVEDIR` for directories. `PyDLL` retains the
GIL across the native rename/unlink calls, so a later Python audit hook cannot
enter between the immediate null-`errcheck` check and the syscall.

Packing uses captured `struct.pack` and `struct.unpack` objects whose live
Python/native exports must remain identical. Generation calls the captured
`_pack_payload` function object, and the receipt commits its canonical recursive
code-state SHA-256. Hashing likewise uses the captured native-backed `hashlib.sha256`
object. The receipt separately commits to the Python `random` module and native
`_random` extension paths/bytes, exact bound `Random` class/native base, and
every constructor/state/sampling method used by generation. It commits to the
`tokenizers` distribution/module version, exact native `Tokenizer` class and
`from_str`, `get_vocab_size`, `token_to_id`, `encode`, and `decode`
descriptors, the exact native `Encoding` class and its `ids` descriptor, and
the presented/resolved paths, byte counts, and
SHA-256 values of both `tokenizers/__init__.py` and the loaded native
`tokenizers` extension. Both tokenizer modules must be absent before initial
binding. Generation uses only captured tokenizer, RNG, hashing, packing,
reviewed-function, filesystem, and native-publication objects. The complete
runtime snapshot is checked before construction, after construction, before
staging publication, immediately before atomic no-replace rename, immediately
after the rename and parent-directory fsync, before replay, and after final
replay. A substitution after an earlier check therefore either cannot intercept
the consumed operation, is rejected before mutation, or is detected before a
verified receipt can return. Native rename and cleanup additionally enforce
their bound null `errcheck` state at the immediate call boundary; cleanup uses
the captured primitives and preserves the existing recovery rules.

Random construction dispatches directly through the captured `__new__` and
`__init__` descriptors, and generation dispatches `getstate`, `getrandbits`,
`randrange`, and `shuffle` through their captured descriptors rather than
performing instance attribute lookup. The receipt and validation inventory also
includes the transitively required `seed`, `randint`, `choice`, `random`, and
`_randbelow` descriptors, for 11 exact `random.Random` method bindings.

Generation seed, training per-cell count, development per-cell count, and lane
length must each have exact Python type `int` before either test or production
validation proceeds. Boolean and floating-point values are rejected even when
ordinary Python equality would compare them equal to a frozen integer.

## 4. Source episodes and development commitments

### 4.1 Training episodes

Training contains exactly 2,048 unique width-four episodes:

```text
2 operations x 8 intermediate carry/borrow patterns x 128 = 2,048

canonical episode-list SHA-256:
1dd913b12d2ffb2201530997102ef50a1e2d581fe7595c4e9ad5ae8c9fe3f009
```

The pattern is `c` after positions one, two, and three. Every three-bit pattern
occurs 128 times for addition and 128 times for subtraction, making each
intermediate position exactly balanced between `c=0` and `c=1` within each
operation. Every episode begins at `p=0,c=0,z=0` and ends at `p=4,c=0,z=1`.
Terminal `c=1` is initially excluded to isolate recurrence from the known
terminal-support hole.

Each initial state, four successors, and answer must replay exactly under
`train/digitwise_protocol.py`. Operation, operand pair, and semantic episode
signature are unique.
The isolated source suite reconstructs all 2,048 episodes, requires that exact
digest and stratum inventory, and invokes the consumed frozen
`rows_from_episode` implementation for all episodes, yielding exactly 18,432
reviewed source rows before the single-completion selector is applied.

### 4.2 Generated width-four development board

The separately generated development commitment has exactly 256 episodes:

```text
2 operations x 8 intermediate carry/borrow patterns x 16 = 256
```

No development row is selected from training. Each row freezes the complete
nominal trace and three generated-history branches at `p=1`: nominal,
carry-flip, and already-written `r[0]` flip. Every branch freezes:

- canonical prefix state;
- full-history prefix;
- fresh-latest-state prompt;
- all successor states;
- expected answer and terminal carry;
- full target response and supervised EOS token.

### 4.3 Frozen cross-width replication board

The generator extracts only the 12 frozen source IDs and requires exact solver
replay, exact case-list/source hashes, exact `2` rows per
`(width, operation)` cell, and uniqueness of IDs, initial states, prompts,
episode signatures, and intervention prefix/target tuples. It freezes the same
complete nominal, carry-flip, and written-result-flip fields listed above.

Training, generated development, and cross-width inventories must be disjoint
over all of:

- left/right operands, nominal answers, and every counterfactual target answer;
- canonical states, full target strings, branch prefixes, and branch targets;
- exact base, full-history, and fresh-latest-state prompts;
- episode and branch semantic signatures.

This enforces zero training overlap for operands, answers, all counterfactual
target scalars/strings, exact prompts, and semantic signatures. The development
and cross-width boards are commitments, not confirmation boards. No hidden
confirmation board or public benchmark prompt is created or used.

## 5. Three matched data arms

All arms use the same ordered 2,048 source episode identities, operation and
carry/borrow cells, supervised target-token count, active token IDs, attention
surface, state-epoch surface, dense positions, updates, and optimizer.

### 5.1 Canonical full trace

One prompt contains one canonical initial state. Its target is exactly:

```text
{successor p=1}
{successor p=2}
{successor p=3}
{successor p=4,z=1}
answer={integer}<EOS>
```

The target has four canonical state lines, one answer line, and one supervised
EOS. EOS is a token target and is not embedded in the JSON response string.

### 5.2 Decomposed one-step control

The same episode is decomposed into the existing four one-step rows plus one
final row. They occupy five independent batch lanes, so no row can attend to a
prior row. Transition targets supervise their LF terminator but no intermediate
EOS; the final row supervises EOS. Concatenating the five supervised targets is
token-ID identical to the canonical full-trace target, including the one final
EOS.

### 5.3 Full-length multiline discontinuity sham

The sham has the same one-prompt, four-state-line, answer, and EOS surface. For
each line position, donors are permuted within matched operation,
carry/borrow-pattern, and tokenizer-length strata. Every donated line is a
valid reachable state at that position, but every adjacent transition is
invalid. A fifth matched donor supplies the answer. The donor answer must differ
from both the treatment answer and terminal sham-line answer. No donor or source
may repeat within a sham trace.

The sham preserves line count, per-line operation/width/carry, answer-token
length, and the global supervised-token multiset. If a bijective no-leak donor
assignment does not exist, generation fails closed. It is scored only as a
multiline-format and token-surface control.

Every donor bijection is computed by deterministic iterative Hopcroft-Karp
search. Source and candidate traversal use the frozen hash order; no recursive
augmenting-path call is permitted. The production path must complete a
2,048-source adversarial chain whose augmenting depth exceeds 1,000 under the
retired implementation.

## 6. Exact token, attention, loss, and compute matching

Each source episode produces one logical pack with seven independent 768-token
batch lanes:

| Lane | Frozen block |
|---:|---|
| 0 | canonical full trace |
| 1 | multiline discontinuity sham |
| 2-5 | decomposed transitions 0-3 |
| 6 | decomposed final row |

Every arm physically carries all seven blocks with byte-identical token IDs,
attention masks, state-epoch IDs, and dense positions. Only the loss mask
changes:

| Arm | Supervised lanes | Loss-masked active filler lanes |
|---|---|---|
| canonical full trace | `{0}` | `{1,2,3,4,5,6}` |
| multiline sham | `{1}` | `{0,2,3,4,5,6}` |
| decomposed one-step | `{2,3,4,5,6}` | `{0,1}` |

The filler carries the exact matched token IDs and remains attention-active so
active context counts match exactly. Each lane is a separate batch sequence;
the attention graph is block diagonal with no cross-lane edge. Consequently a
loss-masked filler block cannot affect a supervised lane. Right padding uses
the frozen EOS ID with attention and loss both zero. Unpadding and
variable-length kernels are forbidden.

Dense positions are explicit normative data, not a trainer default. Every
lane serializes exactly `position_ids = [0, 1, ..., lane_length - 1]`, including
attention-masked right padding; positions do not restart at the active/padding
boundary. At the frozen lane length this is `[0, ..., 767]`. A pack is the
concatenation, in this exact order, of all-lane little-endian `uint32` token
IDs, `uint8` attention masks, `uint8` loss masks, little-endian `uint16` epoch
IDs, and little-endian `uint16` position IDs. Thus each physical lane position
occupies exactly 10 serialized bytes. Lane length must be at most 65,536.

Verification independently unpacks every pack and requires every serialized
position vector to equal `range(lane_length)`. It recomputes and receipts, per
lane and per arm, token IDs, active-token counts, attention counts, state
epochs, loss counts, supervised IDs, and the position-vector SHA-256. Canonical and decomposed
supervised IDs must be exactly equal; sham IDs must have the same count and
multiset. Arm totals must also match.

Fields such as `treatment_answer`, donor IDs, and training-group labels are
audit metadata in JSON rows. The current packer constructs binary lanes only
from the explicitly selected prompt/response token surfaces; those metadata
keys are not serialized. No trainer consumption implementation is present in
this repair. A future separately reviewed trainer must prove from raw pack
decoding and model inputs that metadata, especially `treatment_answer`, can
never enter token IDs, labels, attention, positions, or any auxiliary feature.

Each arm has 2,048 logical packs, seven physical lanes per pack, two logical
packs per optimizer update, and exactly 1,024 updates.

## 7. Six-cell data-by-context factorial

The three data arms cross two context policies, producing six separately
trained cells per seed. A context policy applies during teacher-forced training
and scored decode, not only as an evaluation-time patch.

### 7.1 Full-history replay-discard control

Training uses ordinary full-history causal attention. At each supervised LF it
executes the same latest-epoch re-encode forward as the treatment, discards that
fresh representation, and continues from full history. Decode performs the
same action after each model-emitted LF. This matches four training-time
re-encode forwards per logical pack while retaining full-history context.

### 7.2 Commit-reencode isolation treatment

LF must tokenize to exactly one frozen token. During teacher forcing, every
supervised LF commits the target-token epoch since the previous LF. During
decode, every model-emitted LF commits the generated-token epoch. The external
runtime compares token IDs to the frozen LF ID, tracks the latest token span,
re-encodes that span under the fixed DWS prefix/suffix, replaces stale KV with
the fresh latest-state representation, and continues.

The runtime does not parse DWS fields, validate syntax, compute arithmetic,
repair output, select a schedule, inject a gold state, inspect an answer, or
retry. A malformed line still commits. Delimiter positions are defined by model
token IDs, not an unreported host parser. Replay counts are recorded because
generated histories may diverge.

Both policies have equal parameter counts, data, active and supervised tokens,
updates, and training-time re-encode calls. A same-weight decode-policy
ablation is diagnostic only; it cannot replace the equal-budget six-cell
training comparison.

## 8. Frozen optimizer and three-seed schedules

The complete optimizer implementation is bound by source hash:

```text
train/muon.py
863e79aaaaebb681382f0c88078390b5683ab39be79ac7df60f26d1c04b21762

train/sft.py
9caa62b38a36addda9eb667b72f74dedb7165062f98bef9e1bfe49102af71921
```

Frozen optimizer and numerical contract:

- trainable 2D parameters except names containing `tok` or `head` use Muon;
  all other trainable parameters use AdamW;
- Muon: LR `0.001`, momentum `0.95`, Nesterov enabled, five
  Newton-Schulz steps, coefficients `[3.4445,-4.7750,2.0315]`, normalization
  epsilon `1e-7`, weight decay `0`;
- AdamW: LR `0.0002`, betas `[0.9,0.95]`, epsilon `1e-8`, weight decay `0`,
  `amsgrad=False`, `foreach=False`, `fused=False`, `capturable=False`, and
  `maximize=False`;
- one global L2 clip over all trainable gradients before both optimizer steps,
  maximum norm `1.0`;
- 50-update linear warmup followed by cosine decay to scale `0.1` at update
  1,024, with no early stopping;
- cross entropy with ignore index `-1`, reduced as supervised-token loss sum
  divided by the exact supervised-token count in the two-pack update;
- fp32 parameters, bf16 autocast, fp32 loss/reduction and gradient
  accumulation, TF32 disabled;
- two packs per update, one accumulation step, and `drop_last=False`.

Paired training seeds are exactly `2026071811`, `2026071812`, and
`2026071813`. `seed_schedules.json` freezes, for each seed:

- Python, NumPy, Torch CPU, and Torch CUDA initialization values;
- hash of initial Python RNG state;
- the first four 64-bit RNG draws and the post-probe state hash;
- the complete 2,048-episode pack order and its SHA-256;
- the same order hash for all six cells.

All three startup probes and all three schedule hashes must be genuinely
distinct. Within each seed, the schedule and initial parameter/RNG contract are
paired across all six cells. A repeated or collapsed schedule is fatal even if
the bundle is rehashed.

## 9. Decode, primary decisions, and multiplicity

Primary scoring is one greedy logical completion from the initial prompt under
the cell's frozen runtime. It uses no online semantic parser, arithmetic,
verifier, answer stop, repair, retry, or gold intermediate. It stops only on
model EOS or a fixed 768-token cap. Parsing and solver scoring occur after
generation.

Primary exactness requires all four ordered states, the answer, and
model-emitted EOS. Secondary endpoints are first-state exactness, longest exact
prefix, answer exactness, and EOS rate.

The five primary contrasts are:

1. full trace versus decomposed under full history;
2. full trace versus sham under full history;
3. full trace versus decomposed under commit-reencode;
4. full trace versus sham under commit-reencode;
5. commit-reencode package versus full history for full-trace training.

For every contrast, every seed must show the preregistered direction, at least
a 10-point paired effect, and two-sided exact McNemar `p < 0.01`. Directional
success is required on `3/3` seeds. A failed seed or noncompensatory gate cannot
be rescued by pooling or favorable seed selection.

Each seed report must include paired cell counts, exact McNemar result, effect,
10,000 source-episode cluster-bootstrap replicates, and a 95% interval. The
seed-level report gives all three effects, minimum, median, and range. Pooled
results are descriptive only and cluster first by seed and then source episode.
Partial scores may not be inspected before all six cells are immutable.

Multiplicity is frozen as five contrasts times three seeds, or 15 tests. Apply
Holm-Bonferroni at familywise alpha `0.05`, ordered by exact two-sided McNemar
`p` with contrast then seed as the tie break. All 15 adjusted decisions must
pass in the preregistered direction. EOS, first-state, causal target-switch, and
width-separated SCERT gates are noncompensatory and outside pooling.

Additional per-seed gates are model-emitted EOS at least 90% and first-state
exactness no more than five points below the decomposed control.

## 10. Secondary paired causal diagnostics

After primary decode, secondary calls may prefill the exact frozen `p=1`
nominal, carry-flipped, or written-result-flipped prefix. This host prefix
intervention is a causal diagnostic and is never counted as autonomous
reasoning.

A paired target-switch passes only when all four conditions hold:

1. nominal output exactly equals the nominal target;
2. counterfactual output exactly equals the counterfactual target;
3. nominal and counterfactual targets differ;
4. outputs differ in the target direction.

Counterfactual exactness alone is invalid. Carry paired target-switch is a
promotion veto. On the 256-row board it must reach at least 50%, beat both
matched controls by at least 10 points, and reach at least 40% separately for
nominal `c=0` and nominal `c=1`.

On the frozen 12-case cross-width board, the complete package must satisfy the
following exact integer/rational gates. No binary floating-point threshold is
authoritative:

| Metric | Overall gate | Each four-case width gate |
|---|---:|---:|
| paired carry target-switch exact-success rate | at least `9/12` | at least `3/4` |
| counterfactual full-target exactness rate | at least `9/12` | at least `3/4` |
| output-switch rate | at least `11/12` | at least `4/4` |
| paired exactness-rate improvement over matched full history | at least `2/5` | at least `1/2` |

No evaluator implementation is present in this repair. A future separately
reviewed evaluator must evaluate every rational threshold by integer cross
multiplication using the documented numerator and denominator; conversion to
binary floating point is forbidden. Metadata stores the first three rows as
integer `minimum_successes` and `cases`, and the last row as integer
`minimum_rate.numerator`, `minimum_rate.denominator`, and `cases`.

Any failed width cell vetoes the cross-width result. Written-result
target-switch is corroborative and cannot compensate for failed carry
target-switch.

## 11. Sealed publication and independent verification

The publication contains exactly one externally receipted commitment root and
one closed artifact directory:

```text
<publication>/sealed_manifest.json
<publication>/bundle/<frozen artifacts>
```

`sealed_manifest.json` is the one-file commitment root, but it cannot
authenticate itself. `verify_bundle` requires its exact path plus a SHA-256 and
byte-count receipt supplied outside the publication. It also requires every
external frozen input listed in Section 3. A path whose final component
contains `.partial` is unconditionally rejected as a publication, even if it
contains a byte-complete root and bundle.

The verifier does not trust manifest receipts, audit booleans, plan booleans,
or generated commitments as evidence. From raw published bytes, current frozen
sources, and external constants, it independently regenerates every training
episode, arm row, sham assignment, development/cross-width branch, pack byte,
pack receipt, seed schedule, plan, audit, and root byte-for-byte. In doing so it
reconstructs solver replay, balance, uniqueness, overlap inventories, sham
discontinuity, tokenization, per-lane active/attention/loss counts, supervised
IDs, serialized dense positions, schedules, optimizer contract, runtime
bindings, and authorization. It then rereads and revalidates source,
cross-width, executed-module, Python, and tokenizer-runtime bytes.

Every manifest and JSONL object is decoded as ASCII with duplicate-key and
literal `NaN`/`Infinity` rejection. The verifier then recursively rejects every
decoded non-finite float, including exponent overflow such as `1e999` inside
nested arrays or objects. Recovery identity comparison is recursively
type-strict for every dictionary, list, and scalar. In particular, JSON integer
`128` and float `128.0`, and Boolean/integer analogues, are different identities
even though ordinary Python equality would consider them equal.

Durable publication is fail closed:

- every file is created with exclusive no-follow semantics, content-fsynced,
  changed to exact mode `0444`, fsynced again to include mode metadata
  durability, and required to have link count `1`; every regular-file and
  directory chmod is followed immediately by descriptor-relative `fstat`, with
  unchanged device/inode, expected type/link count, and exact resulting mode
  required before the path can progress. The original exclusive-create
  descriptor remains open through both fsyncs, chmod validation, directory-entry
  device/inode comparison, and exact-byte readback. Readback may use a readable
  descriptor only while the original descriptor is still held, and the two
  descriptors plus the directory entry must identify the same inode before the
  original descriptor is closed;
- bundle and publication directories are exact mode `0555`;
- every publication-parent ancestor is walked with no-follow directory opens,
  and every resulting descriptor remains held for the invocation; each
  ancestor path/device/inode and the final parent identity are committed.
  Existing ancestor symlinks and any later path retarget fail closed;
- staging is the deterministic destination-filesystem entry
  `.<publication-name>.partial` beneath that pinned parent and is held under a
  nonblocking exclusive advisory lock while its creating process is alive;
- before artifact writes, staging contains a fsynced `0444`, link-count-one
  ownership marker binding the exact target path and all external generation
  commitments; the final sealed root carries the same identity and atomically
  replaces that marker only after the closed bundle is sealed;
- staging lock/recovery/validation, file creation, manifest replacement,
  cleanup, fsync, publication readback, and publication all use operations
  relative to the same pinned parent/stage directory descriptors;
- immediately before publication, the source pathname's device/inode must equal
  the already locked staging descriptor's device/inode. Publication uses an
  operating-system descriptor-relative atomic no-overwrite rename, never a
  check-then-replace fallback, and the destination pathname must equal that
  same locked device/inode immediately afterward. A renamed-aside locked stage
  plus a substituted source pathname is preserved and fails before publication;
- every file, the bundle directory, publication directory, and parent
  directory are fsynced at the required pre/post-rename boundaries;
- every durability call dispatches through the captured authenticated
  `posix.fsync` implementation. Replacing the mutable `os.fsync` export cannot
  intercept content, post-`chmod`, directory, or post-rename synchronization;
- final readback opens and retains the publication directory, bundle directory,
  sealed root, and every artifact descriptor before semantic reconstruction.
  Initial bytes are read only while same-inode held descriptors exist. After
  independent replay and final source/runtime revalidation, the verifier rereads
  every payload through those original descriptors, requires exact payload and
  sealed inode metadata stability, rechecks both closed inventories and all
  descriptor-relative pathname identities, and only then returns
  `verified=True`. The final boundary never reopens a publication pathname as
  authority;
- no invocation overwrites a destination. Unsupported atomic primitives,
  symlinks, hard links, extra files, wrong modes, changed sources, and partial
  trees fail closed;
- ordinary pre-rename exceptions invoke bounded cleanup only after validating its
  marker/root identity and bounded tree inventory. Cleanup opens each validated
  member once, retains every file and directory descriptor through chmod and
  native descriptor-relative `unlinkat`, compares the pathname to that held
  descriptor immediately before removal, and requires the held file's link
  count to become zero afterward. There is no validate-close-reopen sequence and
  no Python `os.unlink`/`os.rmdir` audit callback between comparison and syscall.
  Any identity instability aborts cleanup and preserves the remaining tree for
  manual inspection;
- pre-rename process death releases the advisory lock. A subsequent invocation
  for the same exact target and commitments deterministically validates and
  removes the stale protocol-owned staging tree before rebuilding. A foreign
  identity, live lock, unexpected member, symlink, hard link, wrong owner, or
  invalid mode is never removed automatically;
- process death or an ordinary Python exception after destination rename but
  before the caller receives the external success receipt leaves the complete
  destination and no staging pathname. The failing invocation never recursively
  deletes a successfully renamed destination. A restart first obtains a
  nonblocking exclusive lock on that exact
  destination. A live lock fails closed. After lock acquisition, the restart
  requires the completed root's type-strict staging identity, closed inventory,
  modes, and link counts, then performs the same independent byte-for-byte
  replay while holding and finally revalidating every publication descriptor
  and payload before returning publication-receipt schema v2 with disposition
  `recovered_existing_exact_publication`. A previously receipted exact
  publication is intentionally handled by the same idempotent replay because
  no unauthenticated sidecar is allowed to claim whether the prior caller
  received its in-memory receipt. Foreign destinations and unrelated partial
  trees are never deleted by this recovery path;
- the guarantee intentionally begins after the ownership marker is completely
  written and fsynced. Death between directory creation and marker durability,
  or a torn/unreadable marker, leaves a non-authoritative `.partial` path and
  fails closed for manual inspection rather than deleting an unproven tree.

The substitution threat model for these guarantees is same-process Python code
entering at audited operations plus concurrent generators that honor the
advisory lock. The native final rename/unlink calls retain the GIL and do not
emit a Python pathname-mutation audit window. This is not a sandbox against a
separate malicious process running as the same UID that ignores the advisory
lock and issues filesystem syscalls concurrently, nor against direct
`type.__setattr__`, arbitrary `ctypes` memory mutation beyond the bound
`_FuncPtr.errcheck` property, debugger, kernel, storage-device, or power-loss
attacks outside the reviewed Python/POSIX durability model. Such interference
can force failure or leave state for manual inspection; it is not claimed to be
prevented as a host security boundary.

Demonstrated hostile probes must cover an actual direct CLI commitments/build/
verify cycle, hash-authenticated tokenizer/replication relocation, external-root
SHA/path substitution,
self-rehashed plan authorization, parent checkpoint, source binding, promotion,
collapsed generated and cross-width boards, repeated seed schedules, pack-byte
tampering, hard-link aliasing, byte-identical artifact-inode replacement after
semantic replay, byte-identical publication-directory replacement in the same
window, restart recovery attacked by a byte-identical artifact replacement and
then retried after restoring the held inode, atomic no-overwrite contention in
which a foreign destination is created at the final pre-rename hook and
survives byte-for-byte, injected
atomic-publication failure, subprocess `os._exit` before rename with
deterministic recovery, subprocess `os._exit` after destination rename with
verified restart recovery, an ordinary injected post-rename exception with the
same retained-destination restart replay, two real concurrent publication processes with
exactly one coherent winner and no deletion of a foreign partial tree,
foreign partial identity preservation, partial-tree verification rejection,
preloaded reviewed-module rejection, serialized-position tampering, and wrong
source/runtime/tokenizer commitments. Runtime probes additionally substitute
both tokenizer exports, all consumed tokenizer/encoding descriptors,
`random.Random` and a consumed `Random` method descriptor, `struct.pack`,
`_pack_payload`, live `json.dumps`/`json.loads`, an internal JSON encoder export,
the executed JSON module entry, `importlib.metadata.version`,
`collections.namedtuple`, `math.isfinite`, `re.compile`,
`sysconfig.get_path`, all three consumed `platform.python_*` functions, all four
direct callable aliases, the
live `rows_from_episode.__code__`, and `os.fsync`. Deterministic phase probes
are installed as one-shot audit hooks and therefore enter the exact production
call graph without replacing a dispatch function. They attempt a reviewed-code
swap after the final pre-construction validation, would
consume the malicious body, and would restore the original code afterward. The
audit guard must reject the swap itself, the malicious body must not execute,
the original code must remain installed, and no staging path may appear. A
second phase probe tries the same swap/consume/restore pattern against a helper
inside the consumed row builder's private globals; the sealed mapping must
reject the initial assignment. Separate probes replace public `os.fsync` and
the captured `_BOUND_OS_FSYNC` binding inside the atomic-publication callback
after the immediately preceding runtime check. Both must intercept zero fsync
calls, fail before returning verified, and leave neither publication nor
partial staging.
Another phase probe replaces a public mutable builtin while construction uses
the private frozen builtin. It must intercept zero calls, fail the next runtime
boundary, and leave no publication or staging tree.
Native-call probes install mutable `errcheck` callbacks from one-shot audit
hooks immediately before atomic rename and during held-member cleanup. Each
callback would be consumed and remove itself on the old path. The immediate
native-call checks must instead reject both installations before callback use;
the rename probe leaves no publication or staging tree, and the cleanup probe
leaves the held member linked for explicit test cleanup. Deleting each test
callback restores the bound null state and the next runtime receipt must match.
Filesystem probes also retarget an already-pinned ancestor, substitute a
locked staging pathname before rename, replace a just-created sealed pathname
during descriptor chmod, and replace a validated cleanup member during its
descriptor chmod. Both file-level substitutions must preserve the foreign path
and the original held inode rather than trusting a reopen or unlinking the
replacement. Recovery probes preserve a foreign
marker whose float field compares numerically equal to the expected integer.
Strict-JSON probes include nested overflow values such as `1e999`.

The canonical warning-strict test invocation is exactly:

```bash
/usr/bin/env -i PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
  -I -S -B -c 'import sys,sysconfig; sys.path.append(sysconfig.get_path("purelib")); import pytest; raise SystemExit(pytest.main(["-q","-W","error","-p","no:cacheprovider","pipeline/test_generate_dws_single_completion_v1.py"]))'
```

The canonical file collects exactly `88` tests after these source repairs.
Third-party pytest plugin autoload is therefore disabled before pytest import.
Operational subprocess probes use the same absolute executable and
`-I -S -B` flags with an empty environment; they do not inherit the parent
shell or pytest environment.

## 12. Authorization boundary

A passing test-local bundle proves only that a CPU experiment contract is
mechanically matched, externally bound, and independently replayable. The
current review status remains
COMMIT/CPU MECHANICS/PUBLICATION/LINUX/STOKES/NEWTON/H100 NO-GO. Exact trainer
consumption and exact scoring are intentionally absent. This repair does not
authorize a commit, durable publication, training, production data, a
Stokes/Newton/Linux/H100 job, checkpoint promotion, a confirmation-board claim,
or a reasoning claim. Separately reviewed trainer, evaluator, runtime,
publication, and accelerator preregistrations are required before any model
experiment. Fresh independent rereview is required.
