#!/bin/bash -p
#SBATCH --job-name=shohin-r12-dws-eos-dev
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/shohin-r12-dws-eos-dev-%j.out
#SBATCH --open-mode=truncate
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?submit with sbatch; direct execution is forbidden}"
: "${SOURCE_ROOT:?set the absolute reviewed clean checkout path}"
: "${SOURCE_ROOT_REVIEWED:?set SOURCE_ROOT_REVIEWED=YES after commit review}"
: "${SOURCE_COMMIT:?set the reviewed exact 40-hex commit}"
: "${SOURCE_MANIFEST:?set the absolute external runtime-source manifest path}"
: "${SOURCE_MANIFEST_SHA256:?set the externally frozen manifest SHA-256}"
: "${OUT:?set an absolute absent accepted-bundle path outside SOURCE_ROOT}"
: "${PYTHON_BIN:?set the absolute pinned Python interpreter path}"
: "${GIT_BIN:?set the absolute pinned Git executable path}"
: "${SCONTROL_BIN:?set the absolute pinned scontrol executable path}"
: "${SACCT_BIN:?set the absolute pinned sacct executable path}"
: "${NVIDIA_SMI_BIN:?set the absolute pinned nvidia-smi executable path}"
: "${PRODUCTION_AUTHORITY_PUBLIC_KEY:?set the absolute external production public-key path}"
: "${RUN_AUTHORIZATION:?set the absolute pre-run signed authorization path}"
: "${RUN_AUTHORIZATION_SHA256:?set the frozen signed-authorization SHA-256}"
: "${DELEGATED_MARKER_KEY_BROKER_FD:?controller must inject an attesting Unix broker socket FD}"

if [[ "$SOURCE_ROOT_REVIEWED" != "YES" ]]; then
  echo "SOURCE_ROOT_REVIEWED must equal YES" >&2
  exit 64
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be exactly 40 lowercase hex characters" >&2
  exit 64
fi
if [[ ! "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "SOURCE_MANIFEST_SHA256 must be exactly 64 lowercase hex characters" >&2
  exit 64
fi
if [[ ! "$RUN_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "RUN_AUTHORIZATION_SHA256 must be exactly 64 lowercase hex characters" >&2
  exit 64
fi
if [[ ! "$DELEGATED_MARKER_KEY_BROKER_FD" =~ ^[1-9][0-9]*$ ]] || \
  (( DELEGATED_MARKER_KEY_BROKER_FD < 15 )); then
  echo "DELEGATED_MARKER_KEY_BROKER_FD must be an inherited descriptor >=15" >&2
  exit 64
fi
for environment_name in LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH LD_DEBUG LD_PROFILE \
  LD_USE_LOAD_BIAS GLIBC_TUNABLES DYLD_INSERT_LIBRARIES DYLD_LIBRARY_PATH \
  DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FALLBACK_FRAMEWORK_PATH; do
  if [[ -n "${!environment_name:-}" ]]; then
    echo "dynamic-loader environment is forbidden: $environment_name" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT" "$SOURCE_MANIFEST" "$OUT" "$PYTHON_BIN" \
  "$GIT_BIN" "$SCONTROL_BIN" "$0"; do
  if [[ "$path" != /* ]]; then
    echo "all custody paths must be absolute: $path" >&2
    exit 64
  fi
done
for path in "$SACCT_BIN" "$NVIDIA_SMI_BIN" "$PRODUCTION_AUTHORITY_PUBLIC_KEY" \
  "$RUN_AUTHORIZATION"; do
  if [[ "$path" != /* ]]; then
    echo "all authorization/runtime paths must be absolute: $path" >&2
    exit 64
  fi
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export R12_REVIEWED_SOURCE_ROOT="$SOURCE_ROOT"
unset BASH_ENV ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE \
  PYTHONINSPECT PYTHONWARNINGS LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH LD_DEBUG \
  LD_PROFILE LD_USE_LOAD_BIAS GLIBC_TUNABLES DYLD_INSERT_LIBRARIES \
  DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH \
  DYLD_FALLBACK_FRAMEWORK_PATH

# Fixed inherited descriptors prevent executable/manifest pathname swaps after checks.
exec 5<"$0"
exec 6<"$SOURCE_MANIFEST"
exec 7<"$SCONTROL_BIN"
exec 8<"$PYTHON_BIN"
exec 9<"$GIT_BIN"
exec 10<"$SACCT_BIN"
exec 11<"$NVIDIA_SMI_BIN"

exec "/proc/self/fd/8" -I -S -B - \
  "$SOURCE_ROOT" "$SOURCE_COMMIT" "$SOURCE_MANIFEST" \
  "$SOURCE_MANIFEST_SHA256" "$OUT" "$PYTHON_BIN" "$GIT_BIN" \
  "$SCONTROL_BIN" "$SACCT_BIN" "$NVIDIA_SMI_BIN" "$SLURM_JOB_ID" "$0" \
  "$PRODUCTION_AUTHORITY_PUBLIC_KEY" "$RUN_AUTHORIZATION" \
  "$RUN_AUTHORIZATION_SHA256" "$DELEGATED_MARKER_KEY_BROKER_FD" \
  5 6 7 8 9 10 11 <<'PY_ORCHESTRATE'
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import types
from pathlib import Path

SOURCE_FILES = (
    "R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md",
    "train/eval_dws_eos_suppressed_trace.py",
    "train/jobs/eval_dws_eos_suppressed_trace.sbatch",
    "train/model.py",
    "train/test_eval_dws_eos_suppressed_trace.py",
)
MANIFEST_SCHEMA = "r12_dws_eos_suppressed_runtime_sources_v7"


def fail(message):
    raise RuntimeError(f"wrapper custody failure: {message}")


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def stable(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
    )


def read_held_path(path, descriptor, *, require_single_link=True):
    before = path.lstat()
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or identity(before) != identity(opened):
        fail(f"held descriptor differs from pathname: {path}")
    if require_single_link and opened.st_nlink != 1:
        fail(f"held file is not single-link: {path}")
    payload = os.pread(descriptor, opened.st_size + 1, 0)
    if (
        len(payload) != opened.st_size
        or identity(os.fstat(descriptor)) != identity(opened)
        or identity(path.lstat()) != identity(opened)
    ):
        fail(f"held file changed while reading: {path}")
    return payload, opened


def open_held_path(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        fail(f"held input is not a regular single-link file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload, opened = read_held_path(path, descriptor)
        return descriptor, payload, opened
    except BaseException:
        os.close(descriptor)
        raise


def read_regular(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        fail(f"runtime source is not a regular single-link file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if identity(opened) != identity(before):
            fail(f"runtime source changed while opening: {path}")
        payload = os.pread(descriptor, opened.st_size + 1, 0)
        if len(payload) != opened.st_size or identity(os.fstat(descriptor)) != identity(opened):
            fail(f"runtime source changed while reading: {path}")
    finally:
        os.close(descriptor)
    if identity(path.lstat()) != identity(before):
        fail(f"runtime source pathname changed: {path}")
    return payload


def run_executable(descriptor, *arguments, cwd=None, environment=None):
    process = subprocess.run(
        [f"/proc/self/fd/{descriptor}", *arguments],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(descriptor,),
        env=environment,
    )
    if process.returncode:
        fail(process.stderr.decode("utf-8", errors="replace").strip())
    return process.stdout


SEALED_GIT_ENVIRONMENT = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "LC_ALL": "C",
    "LANG": "C",
    "PATH": "",
}


def git(*arguments):
    return run_executable(
        git_fd,
        "--no-pager",
        "--no-optional-locks",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "diff.external=",
        "-c",
        "core.attributesFile=/dev/null",
        "-C",
        str(source_root),
        *arguments,
        environment=SEALED_GIT_ENVIRONMENT,
    )


def make_sealed_memfd(name, payload):
    required = (
        fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_WRITE
    )
    descriptor = os.memfd_create(name, os.MFD_ALLOW_SEALING)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            fail(f"sealed memfd write failed: {name}")
        view = view[written:]
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required)
    if (
        fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != required
        or os.pread(descriptor, len(payload) + 1, 0) != payload
    ):
        fail(f"sealed memfd readback failed: {name}")
    return descriptor


def run_evaluator(arguments, pass_descriptors, *, capture_stdout):
    process = subprocess.run(
        [
            f"/proc/self/fd/{python_fd}",
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{evaluator_fd}",
            *arguments,
        ],
        check=False,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=None,
        pass_fds=tuple(
            sorted({python_fd, evaluator_fd, *pass_descriptors})
        ),
        env=child_environment,
    )
    if process.returncode:
        fail(f"sealed evaluator process failed with status {process.returncode}")
    return process.stdout


source_root = Path(sys.argv[1])
source_commit = sys.argv[2]
manifest_path = Path(sys.argv[3])
manifest_sha256 = sys.argv[4]
accepted_path = Path(sys.argv[5])
python_path = Path(sys.argv[6])
git_path = Path(sys.argv[7])
scontrol_path = Path(sys.argv[8])
sacct_path = Path(sys.argv[9])
nvidia_smi_path = Path(sys.argv[10])
job_id = sys.argv[11]
running_wrapper_path = Path(sys.argv[12])
authority_key_path = Path(sys.argv[13])
authorization_path = Path(sys.argv[14])
authorization_sha256 = sys.argv[15]
delegated_marker_key_broker_fd = int(sys.argv[16])
(
    wrapper_fd,
    manifest_fd,
    scontrol_fd,
    python_fd,
    git_fd,
    sacct_fd,
    nvidia_smi_fd,
) = map(int, sys.argv[17:24])

libc = ctypes.CDLL(None, use_errno=True)
PR_SET_DUMPABLE = 4
PR_GET_DUMPABLE = 3
if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 or libc.prctl(
    PR_GET_DUMPABLE, 0, 0, 0, 0
) != 0:
    fail("wrapper could not disable same-UID ptrace/proc descriptor inspection")
for label, path in (
    ("SOURCE_ROOT", source_root),
    ("SOURCE_MANIFEST", manifest_path),
    ("PYTHON_BIN", python_path),
    ("GIT_BIN", git_path),
    ("SCONTROL_BIN", scontrol_path),
    ("SACCT_BIN", sacct_path),
    ("NVIDIA_SMI_BIN", nvidia_smi_path),
):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        fail(f"{label} is not normalized absolute or contains a symlink")
if (
    not accepted_path.is_absolute()
    or accepted_path.parent.resolve(strict=True) != accepted_path.parent
    or accepted_path != accepted_path.parent / accepted_path.name
):
    fail("OUT parent is not normalized absolute or contains a symlink")
if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
    fail("SOURCE_COMMIT is not exact lowercase 40-hex")
if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
    fail("manifest SHA-256 is not exact lowercase 64-hex")
if not re.fullmatch(r"[0-9a-f]{64}", authorization_sha256):
    fail("authorization SHA-256 is not exact lowercase 64-hex")
if not re.fullmatch(r"[1-9][0-9]*", job_id):
    fail("SLURM_JOB_ID is not a positive decimal selector")
try:
    manifest_path.relative_to(source_root)
except ValueError:
    pass
else:
    fail("SOURCE_MANIFEST must be outside SOURCE_ROOT")
try:
    accepted_path.relative_to(source_root)
except ValueError:
    pass
else:
    fail("OUT must be outside SOURCE_ROOT")
if accepted_path.name in {"", ".", ".."}:
    fail("OUT must name one plain file")


def verify_all_sources():
    manifest_bytes, manifest_info = read_held_path(manifest_path, manifest_fd)
    if manifest_info.st_mode & 0o222:
        fail("external SOURCE_MANIFEST has a write permission bit")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        fail("external SOURCE_MANIFEST hash differs")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=strict_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"SOURCE_MANIFEST is not strict UTF-8 JSON: {error}")
    if stable(manifest) != manifest_bytes:
        fail("SOURCE_MANIFEST is not canonical stable JSON")
    if (
        set(manifest) != {"schema", "source_root", "source_commit", "files", "runtime"}
        or manifest["schema"] != MANIFEST_SCHEMA
        or manifest["source_root"] != str(source_root)
        or manifest["source_commit"] != source_commit
        or not isinstance(manifest["files"], dict)
        or tuple(manifest["files"]) != SOURCE_FILES
    ):
        fail("SOURCE_MANIFEST identity or source closure differs")
    for path, descriptor in (
        (python_path, python_fd),
        (git_path, git_fd),
        (scontrol_path, scontrol_fd),
        (sacct_path, sacct_fd),
        (nvidia_smi_path, nvidia_smi_fd),
    ):
        payload, _ = read_held_path(path, descriptor)
        runtime_key = {
            python_path: "python",
            git_path: "git",
            scontrol_path: "scontrol",
            sacct_path: "sacct",
            nvidia_smi_path: "nvidia_smi",
        }[path]
        if hashlib.sha256(payload).hexdigest() != manifest["runtime"][runtime_key]["sha256"]:
            fail(f"held executable hash differs: {path}")
    if git("rev-parse", "--show-toplevel").rstrip(b"\n") != os.fsencode(source_root):
        fail("SOURCE_ROOT is not the exact Git top level")
    if git("rev-parse", "HEAD").decode("ascii").strip() != source_commit:
        fail("SOURCE_COMMIT is not checked-out HEAD")
    if git("cat-file", "-t", source_commit) != b"commit\n":
        fail("SOURCE_COMMIT is not a commit object")
    if git("status", "--porcelain=v1", "-z", "--untracked-files=all"):
        fail("SOURCE_ROOT is not clean")
    verified = {}
    for relative_path, expected_sha256 in manifest["files"].items():
        current = read_regular(source_root / relative_path)
        if hashlib.sha256(current).hexdigest() != expected_sha256:
            fail(f"current runtime source hash differs: {relative_path}")
        if current != git("show", f"{source_commit}:{relative_path}"):
            fail(f"runtime source differs from git show: {relative_path}")
        verified[relative_path] = current
    wrapper_bytes, _ = read_held_path(
        running_wrapper_path, wrapper_fd, require_single_link=False
    )
    if hashlib.sha256(wrapper_bytes).hexdigest() != manifest["files"][
        "train/jobs/eval_dws_eos_suppressed_trace.sbatch"
    ]:
        fail("actual Slurm-spooled wrapper bytes differ from external seal")
    return manifest, manifest_bytes, manifest_info, verified


manifest, manifest_bytes, manifest_info, verified = verify_all_sources()
evaluator_fd = make_sealed_memfd(
    "r12-evaluator", verified["train/eval_dws_eos_suppressed_trace.py"]
)
model_fd = make_sealed_memfd("r12-model", verified["train/model.py"])

module = types.ModuleType("_r12_wrapper_verified_evaluator")
module.__file__ = "sealed-memfd:train/eval_dws_eos_suppressed_trace.py"
sys.modules[module.__name__] = module
exec(
    compile(
        verified["train/eval_dws_eos_suppressed_trace.py"],
        module.__file__,
        "exec",
    ),
    module.__dict__,
)
observed_runtime = module.observe_runtime_identity(
    python_path,
    git_path,
    scontrol_path,
    sacct_path,
    nvidia_smi_path,
    python_descriptor=python_fd,
    git_descriptor=git_fd,
    scontrol_descriptor=scontrol_fd,
    sacct_descriptor=sacct_fd,
    nvidia_smi_descriptor=nvidia_smi_fd,
    expected_packages=manifest["runtime"]["packages"],
    require_sealed_startup=True,
)
if observed_runtime != manifest["runtime"]:
    fail("runtime/package/backend identity differs from SOURCE_MANIFEST")

for label, path in (
    ("PRODUCTION_AUTHORITY_PUBLIC_KEY", authority_key_path),
    ("RUN_AUTHORIZATION", authorization_path),
):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        fail(f"{label} is not normalized absolute or contains a symlink")
authority_key_fd, authority_key_bytes, authority_key_info = open_held_path(
    authority_key_path
)
authorization_fd, authorization_bytes, authorization_info = open_held_path(
    authorization_path
)
for label, info in (
    ("production authority public key", authority_key_info),
    ("run authorization", authorization_info),
):
    if info.st_mode & 0o222:
        fail(f"{label} has a write permission bit")
if (
    authority_key_bytes
    != (module.PRODUCTION_AUTHORITY_PUBLIC_KEY_HEX + "\n").encode("ascii")
    or hashlib.sha256(authority_key_bytes).hexdigest()
    != module.PRODUCTION_AUTHORITY_FILE_SHA256
):
    fail("external production authority public key differs from source pin")
if hashlib.sha256(authorization_bytes).hexdigest() != authorization_sha256:
    fail("signed run authorization hash differs")
try:
    authorization = json.loads(
        authorization_bytes.decode("utf-8"), object_pairs_hook=strict_object
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    fail(f"signed run authorization is not strict UTF-8 JSON: {error}")
if stable(authorization) != authorization_bytes:
    fail("signed run authorization is not canonical stable JSON")
for path in (authority_key_path, authorization_path):
    try:
        path.relative_to(source_root)
    except ValueError:
        pass
    else:
        fail(f"external authorization material must be outside SOURCE_ROOT: {path}")
delegated_marker_key_fd = None
delegated_marker_key_bytes = None
delegated_marker_info = None


def verify_external_authorization_files():
    observed_key, observed_key_info = read_held_path(
        authority_key_path, authority_key_fd
    )
    observed_auth, observed_auth_info = read_held_path(
        authorization_path, authorization_fd
    )
    if (
        observed_key != authority_key_bytes
        or identity(observed_key_info) != identity(authority_key_info)
        or observed_auth != authorization_bytes
        or identity(observed_auth_info) != identity(authorization_info)
    ):
        fail("external authorization descriptor custody changed")
    if delegated_marker_key_fd is not None:
        observed_marker_info = os.fstat(delegated_marker_key_fd)
        observed_marker_key = os.pread(delegated_marker_key_fd, 33, 0)
        if (
            delegated_marker_info is None
            or observed_marker_key != delegated_marker_key_bytes
            or identity(observed_marker_info) != identity(delegated_marker_info)
            or fcntl.fcntl(delegated_marker_key_fd, fcntl.F_GET_SEALS) != 0x000F
        ):
            fail("delegated marker key descriptor custody changed")


output_fd, output_record = module.open_owned_output_directory(accepted_path.parent)
publisher_lease = None
try:
    slurm_identity = module.observe_slurm_identity(
        str(scontrol_path),
        str(sacct_path),
        str(nvidia_smi_path),
        job_id,
        scontrol_descriptor=scontrol_fd,
        sacct_descriptor=sacct_fd,
        nvidia_smi_descriptor=nvidia_smi_fd,
    )
    wrapper_sha256 = manifest["files"][
        "train/jobs/eval_dws_eos_suppressed_trace.sbatch"
    ]
    if slurm_identity["command_sha256"] != wrapper_sha256:
        fail("parsed Slurm Command bytes differ from reviewed wrapper")
    nonce = authorization["authorization_nonce"]
    expected_delegated_marker_key = (
        module.delegated_signing_key_record_from_authorization(authorization)
    )
    if module._preauthorization_python_runtime_identity(
        require_sealed_startup=True
    ) != observed_runtime["python_startup"]:
        fail("Python startup closure changed before authorization verification")
    module.validate_run_authorization(
        authorization,
        expected_slurm=slurm_identity,
        expected_output_directory=output_record,
        expected_accepted_name=accepted_path.name,
        expected_source_manifest_sha256=manifest_sha256,
        expected_delegated_marker_key=expected_delegated_marker_key,
        require_current=True,
    )
    if module._preauthorization_python_runtime_identity(
        require_sealed_startup=True
    ) != observed_runtime["python_startup"]:
        fail("Python startup closure changed during authorization verification")
    if (
        authorization["source_commit"] != source_commit
        or authorization["source_manifest_path"] != str(manifest_path)
        or authorization["frozen_inputs"]["prereg_sha256"]
        != manifest["files"]["R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md"]
    ):
        fail("run authorization source/manifest/prereg binding differs")
    verify_external_authorization_files()

    if delegated_marker_key_broker_fd < 15 or not stat.S_ISSOCK(
        os.fstat(delegated_marker_key_broker_fd).st_mode
    ):
        fail("delegated marker key broker is not an inherited socket descriptor")

    publisher_lease = module.acquire_publisher_lease(
        output_fd,
        output_record,
        accepted_path.name,
        job_id,
        nonce,
    )
    commit_marker_name = module.acceptance_commit_marker_name(accepted_path.name)
    durable_receipt_name = module.durable_acceptance_receipt_name(accepted_path.name)
    for output_name in (
        accepted_path.name,
        commit_marker_name,
        durable_receipt_name,
    ):
        if module._entry_stat_or_none(output_fd, output_name) is not None:
            raise FileExistsError(f"refusing to overwrite accepted output: {output_name}")
    module.cleanup_stale_publication_entries(
        output_fd, accepted_path.name, authorization, publisher_lease
    )

    broker_request = module.build_delegated_key_broker_request(
        authorization_sha256=authorization_sha256,
        source_manifest_sha256=manifest_sha256,
        evaluator_sha256=manifest["files"][
            "train/eval_dws_eos_suppressed_trace.py"
        ],
        wrapper_sha256=wrapper_sha256,
        runtime_identity=observed_runtime,
        expected_signing_key=expected_delegated_marker_key,
    )
    delegated_marker_key_fd, delegated_marker_key_bytes = (
        module.receive_delegated_signing_key_from_broker(
            delegated_marker_key_broker_fd,
            broker_request,
            expected_delegated_marker_key,
        )
    )
    os.close(delegated_marker_key_broker_fd)
    delegated_marker_key_broker_fd = -1
    delegated_marker_info = os.fstat(delegated_marker_key_fd)
    verify_external_authorization_files()

    generator_key_fd = make_sealed_memfd("r12-generator-key", os.urandom(32))
    verifier_key_fd = make_sealed_memfd("r12-verifier-key", os.urandom(32))
    generator_key = os.pread(generator_key_fd, 32, 0)
    verifier_key = os.pread(verifier_key_fd, 32, 0)
    if len({generator_key, verifier_key, delegated_marker_key_bytes}) != 3:
        fail("generator, verifier, and delegated marker signing keys collided")
    candidate_name = (
        f".{accepted_path.name}.r12-candidate-{job_id}-{nonce}"
    )
    context = {
        "schema": module.WRAPPER_ACCEPTANCE_SCHEMA,
        "publication_state": module.PRIVATE_CANDIDATE_STATE,
        "slurm_identity": slurm_identity,
        "wrapper_sha256": wrapper_sha256,
        "source_manifest_sha256": manifest_sha256,
        "runtime_identity_sha256": hashlib.sha256(stable(observed_runtime)).hexdigest(),
        "nonce": nonce,
        "output_directory": output_record,
        "candidate_name": candidate_name,
        "accepted_name": accepted_path.name,
        "production_authority_key_file": module.external_file_record(
            authority_key_path, authority_key_bytes, authority_key_info
        ),
        "run_authorization_file": module.external_file_record(
            authorization_path, authorization_bytes, authorization_info
        ),
        "run_authorization_sha256": authorization_sha256,
        "run_authorization": authorization,
        "generator_signing_key": module.signing_key_record(generator_key),
        "verifier_signing_key": module.signing_key_record(verifier_key),
        "delegated_marker_signing_key": module.signing_key_record(
            delegated_marker_key_bytes
        ),
        "sealed_generator": {
            "evaluator_sha256": manifest["files"][
                "train/eval_dws_eos_suppressed_trace.py"
            ],
            "model_sha256": manifest["files"]["train/model.py"],
        },
    }
    module._validate_wrapper_acceptance(context, context)
    if context["delegated_marker_signing_key"] != expected_delegated_marker_key:
        fail("brokered delegated marker key differs from signed authorization")
    module.activate_pinned_runtime_packages(observed_runtime)
    context_bytes = stable(context)
    context_fd = make_sealed_memfd("r12-acceptance-context", context_bytes)
    context_sha256 = hashlib.sha256(context_bytes).hexdigest()
    report_fd = os.memfd_create("r12-generator-output", os.MFD_ALLOW_SEALING)
    os.fchmod(report_fd, 0o600)

    child_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "R12_REVIEWED_SOURCE_ROOT": str(source_root),
    }
    for environment_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if environment_name in os.environ:
            child_environment[environment_name] = os.environ[environment_name]
    common_arguments = [
        "--acceptance-context-fd",
        str(context_fd),
        "--acceptance-context-sha256",
        context_sha256,
        "--source-root",
        str(source_root),
        "--source-commit",
        source_commit,
        "--runtime-source-manifest",
        str(manifest_path),
        "--runtime-source-manifest-fd",
        str(manifest_fd),
        "--runtime-source-manifest-sha256",
        manifest_sha256,
        "--python-bin",
        str(python_path),
        "--git-bin",
        str(git_path),
        "--scontrol-bin",
        str(scontrol_path),
        "--sacct-bin",
        str(sacct_path),
        "--nvidia-smi-bin",
        str(nvidia_smi_path),
        "--python-source-fd",
        str(python_fd),
        "--git-source-fd",
        str(git_fd),
        "--scontrol-source-fd",
        str(scontrol_fd),
        "--sacct-source-fd",
        str(sacct_fd),
        "--nvidia-smi-source-fd",
        str(nvidia_smi_fd),
        "--evaluator-source-fd",
        str(evaluator_fd),
        "--development-only",
    ]
    run_evaluator(
        [
            "generate-sealed-report",
            *common_arguments,
            "--model-source-fd",
            str(model_fd),
            "--generator-signing-key-fd",
            str(generator_key_fd),
            "--report-output-fd",
            str(report_fd),
        ],
        {
            context_fd,
            manifest_fd,
            python_fd,
            git_fd,
            scontrol_fd,
            sacct_fd,
            nvidia_smi_fd,
            evaluator_fd,
            model_fd,
            generator_key_fd,
            report_fd,
        },
        capture_stdout=False,
    )
    os.fchmod(report_fd, 0o400)
    os.fsync(report_fd)
    fcntl.fcntl(report_fd, fcntl.F_ADD_SEALS, module._required_memfd_seals())
    if fcntl.fcntl(report_fd, fcntl.F_GET_SEALS) != module._required_memfd_seals():
        fail("generator report descriptor could not be sealed after generation")
    manifest2, manifest_bytes2, manifest_info2, _ = verify_all_sources()
    if manifest2 != manifest or manifest_bytes2 != manifest_bytes or identity(manifest_info2) != identity(manifest_info):
        fail("source or manifest custody changed after generation")
    report_payload = os.pread(report_fd, os.fstat(report_fd).st_size + 1, 0)
    report = json.loads(report_payload.decode("utf-8"), object_pairs_hook=strict_object)
    if stable(report) != report_payload:
        fail("generator report descriptor is not canonical stable JSON")
    module.validate_report_schema(report, context, live_custody=False)
    module.verify_live_slurm_context(
        context,
        report["execution"]["runtime_source_manifest"],
        scontrol_descriptor=scontrol_fd,
        sacct_descriptor=sacct_fd,
        nvidia_smi_descriptor=nvidia_smi_fd,
    )
    (
        candidate_sha256,
        candidate_fd,
        candidate_verifier_fd,
        candidate_creation_info,
    ) = (
        module.publish_private_candidate_exclusive(
            output_fd, candidate_name, report, context
        )
    )
    try:
        verifier_output = run_evaluator(
            [
                "verify-private-candidate",
                *common_arguments,
                "--candidate-fd",
                str(candidate_verifier_fd),
                "--verifier-signing-key-fd",
                str(verifier_key_fd),
            ],
            {
                context_fd,
                manifest_fd,
                python_fd,
                git_fd,
                scontrol_fd,
                sacct_fd,
                nvidia_smi_fd,
                evaluator_fd,
                candidate_verifier_fd,
                verifier_key_fd,
            },
            capture_stdout=True,
        )
        verifier_receipt = json.loads(
            verifier_output.decode("utf-8"), object_pairs_hook=strict_object
        )
        if stable(verifier_receipt) != verifier_output:
            fail("independent verifier output is not one canonical JSON record")
        module._validate_independent_verifier_receipt(verifier_receipt, report)
        candidate_payload, candidate_info = module.read_private_candidate_descriptor(
            candidate_fd
        )
        candidate_report = module._parse_json_object_bytes(
            candidate_payload, "wrapper-held private candidate"
        )
        if (
            hashlib.sha256(candidate_payload).hexdigest() != candidate_sha256
            or candidate_payload != module.stable_json_bytes(report)
            or candidate_report != report
            or module._file_identity(candidate_info)
            != module._file_identity(candidate_creation_info)
            or verifier_receipt["candidate_inode"]
            != {
                "device": candidate_info.st_dev,
                "inode": candidate_info.st_ino,
                "uid": candidate_info.st_uid,
                "mode": stat.S_IMODE(candidate_info.st_mode),
                "nlink": candidate_info.st_nlink,
                "size": candidate_info.st_size,
            }
        ):
            fail("independent verifier receipt is not descriptor-bound to candidate")
        module._read_held_directory_entry_descriptor(
            output_fd,
            candidate_name,
            candidate_fd,
            "wrapper private candidate consumption",
            expected_info=candidate_info,
            expected_payload=candidate_payload,
        )
        candidate_retired, candidate_quarantine_name = (
            module._quarantine_held_entry_if_exact(
                output_fd,
                accepted_path.name,
                nonce,
                candidate_name,
                candidate_fd,
                candidate_info,
                candidate_payload,
                "wrapper private candidate consumption",
            )
        )
        if not candidate_retired or candidate_quarantine_name is None:
            fail("private candidate consumption quarantine was not exact")

        def final_verification(stage):
            if stage not in {
                "after_final_readback",
                "after_final_canonical_path_validation",
            }:
                return
            manifest3, manifest_bytes3, manifest_info3, _ = verify_all_sources()
            if (
                manifest3 != manifest
                or manifest_bytes3 != manifest_bytes
                or identity(manifest_info3) != identity(manifest_info)
            ):
                fail("source or manifest custody changed before acceptance completion")
            if module.observe_slurm_identity(
                str(scontrol_path),
                str(sacct_path),
                str(nvidia_smi_path),
                job_id,
                scontrol_descriptor=scontrol_fd,
                sacct_descriptor=sacct_fd,
                nvidia_smi_descriptor=nvidia_smi_fd,
            ) != slurm_identity:
                fail("parsed live Slurm identity changed before acceptance completion")
            module.revalidate_runtime_identity(
                observed_runtime,
                python_path,
                git_path,
                scontrol_path,
                sacct_path,
                nvidia_smi_path,
                python_descriptor=python_fd,
                git_descriptor=git_fd,
                scontrol_descriptor=scontrol_fd,
                sacct_descriptor=sacct_fd,
                nvidia_smi_descriptor=nvidia_smi_fd,
            )
            verify_external_authorization_files()
            module.validate_run_authorization(
                authorization,
                expected_slurm=slurm_identity,
                expected_output_directory=output_record,
                expected_accepted_name=accepted_path.name,
                expected_source_manifest_sha256=manifest_sha256,
                expected_delegated_marker_key=context[
                    "delegated_marker_signing_key"
                ],
                require_current=True,
            )
            module.validate_output_directory_fd(
                output_fd, output_record, require_path_identity=True
            )

        accepted_payload, commit_payload, receipt_payload, durable_receipt = (
            module.publish_accepted_bundle_exclusive(
                output_fd,
                candidate_report,
                verifier_receipt,
                manifest_bytes,
                manifest_info,
                delegated_marker_key_bytes,
                failure_injector=final_verification,
            )
        )
        module._read_held_directory_entry_descriptor(
            output_fd,
            candidate_quarantine_name,
            candidate_fd,
            "wrapper private candidate final quarantine hold",
            expected_info=candidate_info,
            expected_payload=candidate_payload,
        )
    finally:
        os.close(candidate_verifier_fd)
        os.close(candidate_fd)
    print(
        json.dumps(
            {
                "accepted_bundle_sha256": hashlib.sha256(accepted_payload).hexdigest(),
                "accepted_output": str(accepted_path),
                "commit_marker": str(accepted_path.parent / commit_marker_name),
                "commit_marker_sha256": hashlib.sha256(commit_payload).hexdigest(),
                "durable_acceptance_receipt": str(
                    accepted_path.parent / durable_receipt_name
                ),
                "durable_acceptance_receipt_sha256": hashlib.sha256(
                    receipt_payload
                ).hexdigest(),
                "final_inode": durable_receipt["final_inode"],
                "report_sha256": durable_receipt["report_sha256"],
                "slurm_identity": slurm_identity,
                "status": "wrapper_durable_post_fsync_acceptance_complete",
                "wrapper_sha256": wrapper_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
finally:
    if publisher_lease is not None:
        module.release_publisher_lease(output_fd, publisher_lease)
    if delegated_marker_key_fd is not None:
        os.close(delegated_marker_key_fd)
    if delegated_marker_key_broker_fd >= 0:
        os.close(delegated_marker_key_broker_fd)
    os.close(authority_key_fd)
    os.close(authorization_fd)
    os.close(output_fd)
PY_ORCHESTRATE
