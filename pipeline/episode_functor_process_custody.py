#!/usr/bin/env python3
"""Process-separated custody for one EFC candidate compilation.

The candidate and assessor execute in fresh default-deny macOS sandboxes. The
candidate sees only public evidence. A later assessor sees the sealed candidate
machine and an independently generated expected machine. This module is a
custody mechanism, not a neural compiler or a reasoning result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile
from typing import Mapping, Sequence


PROCESS_CUSTODY_SCHEMA = "efc-process-custody-v1"
SANDBOX_POLICY_NAME = "efc-default-deny-role-cwd-v1"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
PROTECTED_RUNTIME_ROOT = Path("/Library/Developer/CommandLineTools")
ROLE_PYTHON = PROTECTED_RUNTIME_ROOT / (
    "Library/Frameworks/Python3.framework/Versions/3.9/Resources/"
    "Python.app/Contents/MacOS/Python"
)
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROLE = ROOT / "pipeline" / "episode_functor_candidate_role.py"
ASSESSOR_ROLE = ROOT / "pipeline" / "episode_functor_assessor_role.py"
PROBE_ROLE = ROOT / "pipeline" / "episode_functor_sandbox_probe_role.py"
LANDLOCK_LAUNCHER = ROOT / "train" / "landlock_stage_exec.py"
LINUX_BWRAP = Path("/usr/bin/bwrap")
LINUX_ROLE_PYTHON = Path("/usr/bin/python3")


class ProcessCustodyError(RuntimeError):
    """The filesystem, sandbox, role process, or receipt violated custody."""


@dataclass(frozen=True)
class RoleRun:
    role: str
    command: tuple[str, ...]
    exit_code: int
    input_files: tuple[dict[str, object], ...]
    output_files: tuple[dict[str, object], ...]
    stderr_sha256: str
    stdout_sha256: str
    sandbox_enforced: bool
    sandbox_launcher_source_sha256: str | None
    sandbox_policy_name: str
    sandbox_profile_sha256: str
    role_source_sha256: str
    cwd_regular_files_before: tuple[str, ...]
    cwd_regular_files_after: tuple[str, ...]

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessCustodyReport:
    schema: str
    evidence_sha256: str
    expected_machine_sha256: str
    candidate_machine_sha256: str
    exact_machine_match: bool
    candidate_run: RoleRun
    assessor_run: RoleRun
    candidate_root_files: tuple[str, ...]
    assessor_root_files: tuple[str, ...]
    candidate_never_received_expected_machine: bool
    assessor_started_after_candidate_exit: bool
    source_tree_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _network_namespace_identity() -> str:
    metadata = os.stat("/proc/self/ns/net")
    return f"{metadata.st_dev}:{metadata.st_ino}"


def _regular_files(directory: Path) -> tuple[str, ...]:
    names: list[str] = []
    for path in directory.iterdir():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ProcessCustodyError(
                f"invocation contains a nonregular entry: {path.name}"
            )
        names.append(path.name)
    return tuple(sorted(names))


def _real_path(path: Path) -> str:
    return os.path.realpath(path)


def _validate_plain_file(path: Path, label: str) -> Path:
    descriptor = _open_plain_file(path, label)
    try:
        resolved = Path(_real_path(path))
        if not resolved.is_absolute():
            raise ProcessCustodyError(f"{label} did not resolve absolutely")
        return resolved
    finally:
        os.close(descriptor)


def _open_plain_file(path: Path, label: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ProcessCustodyError(
                f"{label} must be a nonsymlink regular file"
            ) from exc
        raise ProcessCustodyError(f"{label} is unavailable") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ProcessCustodyError(f"{label} must be a nonsymlink regular file")
    return descriptor


def _read_plain_file(path: Path, label: str, *, maximum: int | None = None) -> bytes:
    descriptor = _open_plain_file(path, label)
    try:
        metadata = os.fstat(descriptor)
        if maximum is not None and metadata.st_size > maximum:
            raise ProcessCustodyError(f"{label} exceeds its byte bound")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        if len(payload) != metadata.st_size:
            raise ProcessCustodyError(f"{label} changed size during read")
        return payload
    finally:
        os.close(descriptor)


def _validate_empty_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProcessCustodyError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ProcessCustodyError(f"{label} must be a nonsymlink directory")
    resolved = Path(_real_path(path))
    if any(resolved.iterdir()):
        raise ProcessCustodyError(f"{label} must be empty before custody starts")
    return resolved


def _validate_root_owned_executable(path: Path, label: str) -> Path:
    try:
        link_metadata = path.lstat()
    except OSError as exc:
        raise ProcessCustodyError(f"{label} is unavailable") from exc
    if (
        link_metadata.st_uid != 0
        or link_metadata.st_mode & 0o022
        or not (
            stat.S_ISREG(link_metadata.st_mode) or stat.S_ISLNK(link_metadata.st_mode)
        )
    ):
        raise ProcessCustodyError(
            f"{label} path must be root-owned and not group/world writable"
        )
    path = _validate_plain_file(Path(_real_path(path)), label)
    metadata = path.stat()
    if metadata.st_uid != 0 or metadata.st_mode & 0o022 or not os.access(path, os.X_OK):
        raise ProcessCustodyError(
            f"{label} must be root-owned, executable, and not group/world writable"
        )
    return path


def _immutable_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ProcessCustodyError("immutable copy destination already exists")
    source_descriptor = _open_plain_file(source, "copy source")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_descriptor = os.open(destination, flags, 0o400)
    try:
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise ProcessCustodyError("immutable copy made no progress")
                view = view[written:]
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)


def _file_evidence(directory: Path, filename: str) -> dict[str, object]:
    path = directory / filename
    try:
        payload = _read_plain_file(path, f"role file {filename}")
    except ProcessCustodyError:
        return {"exists": False, "path": filename}
    return {
        "bytes": len(payload),
        "exists": True,
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        "path": filename,
        "sha256": sha256_bytes(payload),
    }


def _run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=environment,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessCustodyError("sandbox role exceeded 30 seconds") from exc


def _load_canonical_receipt(path: Path, label: str) -> dict[str, object]:
    payload = _read_plain_file(path, label, maximum=1024 * 1024)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProcessCustodyError(f"{label} is malformed") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ProcessCustodyError(f"{label} is not canonical JSON")
    return value


def _sandbox_profile(
    role_script: Path,
    role_cwd: Path,
    isolation_root: Path,
) -> tuple[str, str]:
    actual = {
        "<PROTECTED_RUNTIME_ROOT>": _real_path(PROTECTED_RUNTIME_ROOT),
        "<ROLE_PYTHON>": _real_path(ROLE_PYTHON),
        "<ROLE_SCRIPT>": _real_path(role_script),
        "<ROLE_SCRIPT_PARENT>": _real_path(role_script.parent),
        "<REPOSITORY_ROOT>": _real_path(ROOT),
        "<PROJECTS_ROOT>": _real_path(ROOT.parent),
        "<USER_HOME>": _real_path(ROOT.parent.parent),
        "<USERS_ROOT>": "/Users",
        "<SYSTEM_CRYPTEX_OS>": _real_path(Path("/System/Cryptexes/OS")),
        "<ISOLATION_ROOT>": _real_path(isolation_root),
        "<ROLE_CWD>": _real_path(role_cwd),
    }
    metadata_paths = (
        "/",
        "/System",
        "/System/Cryptexes",
        actual["<SYSTEM_CRYPTEX_OS>"],
        "/Library",
        "/Library/Developer",
        actual["<PROTECTED_RUNTIME_ROOT>"],
        actual["<USERS_ROOT>"],
        actual["<USER_HOME>"],
        actual["<PROJECTS_ROOT>"],
        actual["<REPOSITORY_ROOT>"],
        actual["<ROLE_SCRIPT_PARENT>"],
        "/private",
        "/private/tmp",
        actual["<ISOLATION_ROOT>"],
        actual["<ROLE_CWD>"],
    )

    def build(paths: Mapping[str, str], *, normalized: bool) -> str:
        reverse = {path: name for name, path in actual.items()}

        def render(raw: str) -> str:
            resolved = _real_path(Path(raw))
            value = reverse.get(resolved, resolved) if normalized else resolved
            return json.dumps(value)

        lines = [
            "(version 1)",
            "(deny default)",
            "(allow syscall-unix (syscall-number SYS___mac_syscall "
            "SYS_getfsstat SYS_getfsstat64 SYS_map_with_linking_np SYS_open "
            "SYS_openat SYS_fstatat SYS_fstatat64 SYS_dup))",
            "(allow system-fcntl (fcntl-command F_ADDFILESIGS_RETURN "
            "F_CHECK_LV F_GETPATH))",
            '(with-filter (mac-policy-name "Sandbox") '
            "(allow system-mac-syscall (mac-syscall-number 2)))",
            "(deny network*)",
            "(allow sysctl-read)",
            f"(allow process-exec (literal {json.dumps(paths['<ROLE_PYTHON>'])}))",
            '(allow file-read* (literal "/"))',
            '(allow file-read* (literal "/dev/urandom"))',
            '(allow file-read* file-write-data (literal "/dev/null"))',
        ]
        lines.extend(
            f"(allow file-read-metadata (literal {render(path)}))"
            for path in metadata_paths
        )
        lines.extend(
            (
                "(allow file-read* (subpath {}))".format(
                    json.dumps(paths["<PROTECTED_RUNTIME_ROOT>"])
                ),
                "(allow file-read* (literal {}))".format(
                    json.dumps(paths["<ROLE_SCRIPT>"])
                ),
                "(allow file-read* (subpath {}))".format(
                    json.dumps(paths["<ROLE_CWD>"])
                ),
                "(allow file-write* (subpath {}))".format(
                    json.dumps(paths["<ROLE_CWD>"])
                ),
            )
        )
        return "".join(lines)

    placeholders = {name: name for name in actual}
    return build(actual, normalized=False), build(placeholders, normalized=True)


def normalized_sandbox_profile(role_script: Path) -> str:
    root = Path("/private/tmp/efc-normalized-process-root")
    return _sandbox_profile(role_script, root / "invocation", root)[1]


def _run_role_linux(
    *,
    role: str,
    role_script: Path,
    arguments: Sequence[str],
    logical_cwd: Path,
    file_inputs: Sequence[str],
    file_outputs: Sequence[str],
    allow_nonzero: bool,
) -> RoleRun:
    bwrap = _validate_root_owned_executable(LINUX_BWRAP, "bubblewrap")
    role_python = _validate_root_owned_executable(
        LINUX_ROLE_PYTHON, "Linux role Python"
    )
    launcher_source = _validate_plain_file(LANDLOCK_LAUNCHER, "Landlock launcher")
    role_source = _validate_plain_file(role_script, "role script")
    logical_cwd = Path(_real_path(logical_cwd))
    with tempfile.TemporaryDirectory(prefix=f"efc-{role}-") as directory:
        isolation_root = Path(_real_path(Path(directory)))
        isolated_runtime = isolation_root / "runtime"
        isolated_runtime.mkdir()
        launcher = isolated_runtime / "landlock_stage_exec.py"
        isolated_role = isolated_runtime / "role.py"
        _immutable_copy(launcher_source, launcher)
        _immutable_copy(role_source, isolated_role)
        launcher_source_sha256 = sha256_bytes(
            _read_plain_file(launcher, "isolated Landlock launcher")
        )
        role_source_sha256 = sha256_bytes(
            _read_plain_file(isolated_role, "isolated role source")
        )
        isolated_cwd = isolation_root / "invocation"
        isolated_cwd.mkdir()
        for filename in file_inputs:
            _immutable_copy(logical_cwd / filename, isolated_cwd / filename)
        policy_receipt = isolation_root / "landlock_policy.json"
        runtime_roots = [
            role_python.parent.parent,
            role_python,
            isolated_role,
        ]
        runtime_roots.extend(
            path
            for path in (
                Path("/lib"),
                Path("/lib64"),
                Path("/usr"),
                Path("/etc/ld.so.cache"),
                Path("/dev/null"),
                Path("/dev/urandom"),
                Path("/proc/self/ns/net"),
            )
            if path.exists()
        )
        landlock_command = [
            str(role_python),
            "-I",
            "-S",
            str(launcher),
            "--stage",
            role,
        ]
        for path in sorted(
            {Path(_real_path(path)) for path in runtime_roots},
            key=str,
        ):
            landlock_command.extend(("--ro", str(path)))
        landlock_command.extend(
            (
                "--rw",
                str(isolated_cwd),
                "--policy-receipt",
                str(policy_receipt),
                "--",
                str(role_python),
                str(isolated_role),
                *arguments,
            )
        )
        actual_command = [
            str(bwrap),
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind",
            str(isolation_root),
            str(isolation_root),
            "--chdir",
            str(isolated_cwd),
            "--",
            *landlock_command,
        ]
        command = (
            "<BWRAP>",
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind",
            "<ISOLATION_ROOT>",
            "<ISOLATION_ROOT>",
            "--chdir",
            "<ROLE_CWD>",
            "--",
            "<PYTHON>",
            "<LANDLOCK_LAUNCHER>",
            "--stage",
            role,
            "<CANONICAL_ALLOWLIST>",
            "--rw",
            "<ROLE_CWD>",
            "--policy-receipt",
            "<POLICY_RECEIPT>",
            "--",
            "<PYTHON>",
            "<ROLE_SCRIPT>",
            *arguments,
        )
        environment = {
            "HOME": str(isolated_cwd),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SHOHIN_EFC_NETWORK_MODE": "isolated-netns",
            "SHOHIN_EFC_PARENT_NETNS": _network_namespace_identity(),
            "TMPDIR": str(isolated_cwd),
        }
        before = _regular_files(isolated_cwd)
        input_evidence = tuple(
            _file_evidence(isolated_cwd, filename) for filename in file_inputs
        )
        completed = _run_bounded(
            actual_command,
            cwd=isolated_cwd,
            environment=environment,
        )
        after = _regular_files(isolated_cwd)
        output_evidence = tuple(
            _file_evidence(isolated_cwd, filename) for filename in file_outputs
        )
        expected_after = tuple(sorted((*file_inputs, *file_outputs)))
        if after != expected_after:
            raise ProcessCustodyError(
                f"{role} emitted undeclared files on Linux: {after!r}"
            )
        policy_sha256 = sha256_bytes(
            _read_plain_file(
                policy_receipt,
                f"{role} Landlock policy receipt",
                maximum=1024 * 1024,
            )
        )
        if not all(row["exists"] for row in output_evidence):
            raise ProcessCustodyError(f"{role} omitted a declared Linux output")
        if completed.returncode != 0 and not allow_nonzero:
            raise ProcessCustodyError(
                f"{role} failed with exit {completed.returncode}; "
                f"stderr SHA-256 {sha256_bytes(completed.stderr)}"
            )
        for filename in file_outputs:
            _immutable_copy(isolated_cwd / filename, logical_cwd / filename)
        return RoleRun(
            role=role,
            command=command,
            exit_code=completed.returncode,
            input_files=input_evidence,
            output_files=output_evidence,
            stderr_sha256=sha256_bytes(completed.stderr),
            stdout_sha256=sha256_bytes(completed.stdout),
            sandbox_enforced=True,
            sandbox_launcher_source_sha256=launcher_source_sha256,
            sandbox_policy_name="efc-landlock-bwrap-network-v1",
            sandbox_profile_sha256=policy_sha256,
            role_source_sha256=role_source_sha256,
            cwd_regular_files_before=before,
            cwd_regular_files_after=after,
        )


def _run_role(
    *,
    role: str,
    role_script: Path,
    arguments: Sequence[str],
    logical_cwd: Path,
    file_inputs: Sequence[str],
    file_outputs: Sequence[str],
    allow_nonzero: bool = False,
) -> RoleRun:
    if any(Path(name).name != name for name in (*file_inputs, *file_outputs)):
        raise ProcessCustodyError("role declarations must be relative filenames")
    if set(file_inputs) & set(file_outputs):
        raise ProcessCustodyError("role input and output declarations overlap")
    if platform.system() == "Linux":
        return _run_role_linux(
            role=role,
            role_script=role_script,
            arguments=arguments,
            logical_cwd=logical_cwd,
            file_inputs=file_inputs,
            file_outputs=file_outputs,
            allow_nonzero=allow_nonzero,
        )
    if platform.system() != "Darwin":
        raise ProcessCustodyError("process custody is supported only on macOS or Linux")
    sandbox_exec = _validate_root_owned_executable(
        SANDBOX_EXEC, "macOS sandbox launcher"
    )
    role_python = _validate_root_owned_executable(ROLE_PYTHON, "macOS role Python")
    role_source = _validate_plain_file(role_script, "role script")
    logical_cwd = Path(_real_path(logical_cwd))
    with tempfile.TemporaryDirectory(
        prefix=f"efc-{role}-", dir="/private/tmp"
    ) as directory:
        isolation_root = Path(_real_path(Path(directory)))
        isolated_role = isolation_root / "role.py"
        _immutable_copy(role_source, isolated_role)
        role_source_sha256 = sha256_bytes(
            _read_plain_file(isolated_role, "isolated role source")
        )
        isolated_cwd = isolation_root / "invocation"
        isolated_cwd.mkdir()
        for filename in file_inputs:
            _immutable_copy(logical_cwd / filename, isolated_cwd / filename)
        actual_profile, normalized_profile = _sandbox_profile(
            isolated_role,
            isolated_cwd,
            isolation_root,
        )
        policy_sha256 = sha256_bytes(normalized_profile.encode("utf-8"))
        environment = {
            "HOME": str(isolated_cwd),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": f"{ROLE_PYTHON.parent}:/usr/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SHOHIN_EFC_SANDBOX_ENFORCED": "1",
            "SHOHIN_EFC_SANDBOX_POLICY_SHA256": policy_sha256,
            "SHOHIN_EFC_SANDBOX_STAGE": role,
            "SHOHIN_EFC_NETWORK_MODE": "socket-deny",
            "TMPDIR": str(isolated_cwd),
        }
        command = (
            str(sandbox_exec),
            "-p",
            normalized_profile,
            str(role_python),
            "-I",
            "-S",
            "<ROLE_SCRIPT>",
            *arguments,
        )
        actual_command = (
            str(sandbox_exec),
            "-p",
            actual_profile,
            str(role_python),
            "-I",
            "-S",
            str(isolated_role),
            *arguments,
        )
        before = _regular_files(isolated_cwd)
        input_evidence = tuple(
            _file_evidence(isolated_cwd, filename) for filename in file_inputs
        )
        completed = _run_bounded(
            actual_command,
            cwd=isolated_cwd,
            environment=environment,
        )
        after = _regular_files(isolated_cwd)
        output_evidence = tuple(
            _file_evidence(isolated_cwd, filename) for filename in file_outputs
        )
        expected_after = tuple(sorted((*file_inputs, *file_outputs)))
        if after != expected_after:
            raise ProcessCustodyError(f"{role} emitted undeclared files: {after!r}")
        if not all(row["exists"] for row in output_evidence):
            raise ProcessCustodyError(f"{role} omitted a declared output")
        if completed.returncode != 0 and not allow_nonzero:
            raise ProcessCustodyError(
                f"{role} failed with exit {completed.returncode}; "
                f"stderr SHA-256 {sha256_bytes(completed.stderr)}"
            )
        for filename in file_outputs:
            _immutable_copy(isolated_cwd / filename, logical_cwd / filename)
        return RoleRun(
            role=role,
            command=command,
            exit_code=completed.returncode,
            input_files=input_evidence,
            output_files=output_evidence,
            stderr_sha256=sha256_bytes(completed.stderr),
            stdout_sha256=sha256_bytes(completed.stdout),
            sandbox_enforced=True,
            sandbox_launcher_source_sha256=None,
            sandbox_policy_name=SANDBOX_POLICY_NAME,
            sandbox_profile_sha256=policy_sha256,
            role_source_sha256=role_source_sha256,
            cwd_regular_files_before=before,
            cwd_regular_files_after=after,
        )


def run_process_custody(
    *,
    public_evidence: Path,
    expected_machine: Path,
    candidate_root: Path,
    assessor_root: Path,
) -> ProcessCustodyReport:
    """Compile and assess one world in temporally separate sandboxes."""

    public_evidence = _validate_plain_file(public_evidence, "public evidence")
    expected_machine = _validate_plain_file(expected_machine, "expected machine")
    evidence_bytes = _read_plain_file(
        public_evidence,
        "public evidence",
        maximum=1024 * 1024,
    )
    expected_bytes = _read_plain_file(
        expected_machine,
        "expected machine",
        maximum=1_536,
    )
    if len(expected_bytes) != 1_536:
        raise ProcessCustodyError("expected machine is not 1,536 bytes")
    candidate_root = _validate_empty_directory(candidate_root, "candidate root")
    assessor_root = _validate_empty_directory(assessor_root, "assessor root")
    roots = {
        _real_path(public_evidence.parent),
        _real_path(expected_machine.parent),
        _real_path(candidate_root),
        _real_path(assessor_root),
    }
    if len(roots) != 4:
        raise ProcessCustodyError(
            "public, secret, candidate, and assessor roots overlap"
        )

    _immutable_copy(public_evidence, candidate_root / "evidence.bin")
    candidate_run = _run_role(
        role="candidate-compiler",
        role_script=CANDIDATE_ROLE,
        arguments=("evidence.bin", "machine.bin", "candidate_receipt.json"),
        logical_cwd=candidate_root,
        file_inputs=("evidence.bin",),
        file_outputs=("machine.bin", "candidate_receipt.json"),
    )
    candidate_machine = candidate_root / "machine.bin"
    candidate_receipt = _load_canonical_receipt(
        candidate_root / "candidate_receipt.json",
        "candidate receipt",
    )
    if set(candidate_receipt) != {
        "candidate_source_sha256",
        "declared_input_files",
        "declared_output_files",
        "evidence_sha256",
        "machine_bytes",
        "machine_sha256",
        "regular_files_before",
        "sandbox_enforced",
        "sandbox_policy_sha256",
        "sandbox_stage",
        "schema",
    }:
        raise ProcessCustodyError("candidate role receipt fields differ")
    candidate_bytes = _read_plain_file(
        candidate_machine,
        "candidate machine",
        maximum=1_536,
    )
    if (
        candidate_receipt.get("schema") != "efc-candidate-compiler-role-v1"
        or candidate_receipt.get("sandbox_stage") != "candidate-compiler"
        or candidate_receipt.get("sandbox_policy_sha256")
        != candidate_run.sandbox_profile_sha256
        or candidate_receipt.get("candidate_source_sha256")
        != candidate_run.role_source_sha256
        or candidate_receipt.get("evidence_sha256") != sha256_bytes(evidence_bytes)
        or candidate_receipt.get("machine_sha256") != sha256_bytes(candidate_bytes)
        or candidate_receipt.get("machine_bytes") != len(candidate_bytes)
        or candidate_receipt.get("declared_input_files") != ["evidence.bin"]
        or candidate_receipt.get("declared_output_files")
        != ["machine.bin", "candidate_receipt.json"]
        or candidate_receipt.get("regular_files_before") != ["evidence.bin"]
        or candidate_receipt.get("sandbox_enforced") is not True
    ):
        raise ProcessCustodyError("candidate role receipt differs from custody")
    _immutable_copy(candidate_machine, assessor_root / "candidate_machine.bin")
    _immutable_copy(expected_machine, assessor_root / "expected_machine.bin")
    assessor_run = _run_role(
        role="machine-assessor",
        role_script=ASSESSOR_ROLE,
        arguments=(
            "candidate_machine.bin",
            "expected_machine.bin",
            "assessment.json",
        ),
        logical_cwd=assessor_root,
        file_inputs=("candidate_machine.bin", "expected_machine.bin"),
        file_outputs=("assessment.json",),
        allow_nonzero=True,
    )
    assessment = _load_canonical_receipt(
        assessor_root / "assessment.json",
        "assessor receipt",
    )
    parent_exact = candidate_bytes == expected_bytes
    if set(assessment) != {
        "assessor_source_sha256",
        "candidate_machine_sha256",
        "declared_input_files",
        "declared_output_files",
        "exact_machine_match",
        "expected_machine_sha256",
        "regular_files_before",
        "sandbox_enforced",
        "sandbox_policy_sha256",
        "sandbox_stage",
        "schema",
    }:
        raise ProcessCustodyError("assessor role receipt fields differ")
    if (
        assessment.get("schema") != "efc-machine-assessor-role-v1"
        or assessment.get("exact_machine_match") is not parent_exact
        or parent_exact is not (assessor_run.exit_code == 0)
        or assessor_run.exit_code not in {0, 2}
        or assessment.get("sandbox_stage") != "machine-assessor"
        or assessment.get("sandbox_policy_sha256")
        != assessor_run.sandbox_profile_sha256
        or assessment.get("assessor_source_sha256") != assessor_run.role_source_sha256
        or assessment.get("candidate_machine_sha256") != sha256_bytes(candidate_bytes)
        or assessment.get("expected_machine_sha256") != sha256_bytes(expected_bytes)
        or assessment.get("declared_input_files")
        != ["candidate_machine.bin", "expected_machine.bin"]
        or assessment.get("declared_output_files") != ["assessment.json"]
        or assessment.get("regular_files_before")
        != ["candidate_machine.bin", "expected_machine.bin"]
        or assessment.get("sandbox_enforced") is not True
    ):
        raise ProcessCustodyError("assessor result or exit contract differs")
    source_tree_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "episode_functor_assessor_role.py": assessor_run.role_source_sha256,
                "episode_functor_candidate_role.py": (candidate_run.role_source_sha256),
                "episode_functor_process_custody.py": sha256_bytes(
                    _read_plain_file(Path(__file__), "process custody source")
                ),
            }
        )
    )
    return ProcessCustodyReport(
        schema=PROCESS_CUSTODY_SCHEMA,
        evidence_sha256=sha256_bytes(evidence_bytes),
        expected_machine_sha256=sha256_bytes(expected_bytes),
        candidate_machine_sha256=sha256_bytes(candidate_bytes),
        exact_machine_match=parent_exact,
        candidate_run=candidate_run,
        assessor_run=assessor_run,
        candidate_root_files=_regular_files(candidate_root),
        assessor_root_files=_regular_files(assessor_root),
        candidate_never_received_expected_machine=(
            candidate_run.cwd_regular_files_before == ("evidence.bin",)
            and all(
                row["path"] != "expected_machine.bin"
                for row in candidate_run.input_files
            )
        ),
        assessor_started_after_candidate_exit=True,
        source_tree_sha256=source_tree_sha256,
    )


def run_sandbox_blindness_probe(
    *,
    public_input: Path,
    forbidden_secret: Path,
    probe_root: Path,
) -> tuple[dict[str, object], RoleRun]:
    """Exercise the exact role policy against secret, repository, and network."""

    public_input = _validate_plain_file(public_input, "probe public input")
    forbidden_secret = _validate_plain_file(forbidden_secret, "probe forbidden secret")
    probe_root = _validate_empty_directory(probe_root, "probe root")
    if (
        len(
            {
                _real_path(public_input.parent),
                _real_path(forbidden_secret.parent),
                _real_path(probe_root),
            }
        )
        != 3
    ):
        raise ProcessCustodyError("probe public, secret, and output roots overlap")
    if _read_plain_file(public_input, "probe public input", maximum=7) != b"public\n":
        raise ProcessCustodyError("probe public input must contain canonical sentinel")
    _immutable_copy(public_input, probe_root / "allowed.txt")
    forbidden_write = forbidden_secret.parent / "forbidden_probe_write.txt"
    if forbidden_write.exists() or forbidden_write.is_symlink():
        raise ProcessCustodyError("probe outside-write sentinel already exists")
    role_run = _run_role(
        role="blindness-probe",
        role_script=PROBE_ROLE,
        arguments=(
            "allowed.txt",
            str(forbidden_secret),
            str(forbidden_secret.parent),
            str(ROOT / "AGENT_RUNBOOK.md"),
            str(forbidden_write),
            "probe.json",
        ),
        logical_cwd=probe_root,
        file_inputs=("allowed.txt",),
        file_outputs=("probe.json",),
        allow_nonzero=True,
    )
    result = _load_canonical_receipt(probe_root / "probe.json", "probe receipt")
    expected_checks = {
        "allowed_input_read",
        "local_write_allowed",
        "network_socket_blocked",
        "outside_write_blocked",
        "repository_read_blocked",
        "secret_directory_listing_blocked",
        "secret_read_blocked",
    }
    if (
        set(result)
        != {
            "all_gates_pass",
            "allowed_input_read",
            "local_write_allowed",
            "network_socket_blocked",
            "network_test_mode",
            "outside_write_blocked",
            "probe_source_sha256",
            "repository_read_blocked",
            "sandbox_policy_sha256",
            "schema",
            "secret_directory_listing_blocked",
            "secret_read_blocked",
        }
        or result.get("schema") != "efc-process-sandbox-probe-v1"
        or result.get("all_gates_pass") is not True
        or role_run.exit_code != 0
        or result.get("sandbox_policy_sha256") != role_run.sandbox_profile_sha256
        or result.get("probe_source_sha256") != role_run.role_source_sha256
        or result.get("network_test_mode")
        != ("isolated-netns" if platform.system() == "Linux" else "socket-deny")
        or not all(result.get(name) is True for name in expected_checks)
        or forbidden_write.exists()
    ):
        raise ProcessCustodyError("sandbox blindness probe failed")
    return result, role_run


__all__ = [
    "ASSESSOR_ROLE",
    "CANDIDATE_ROLE",
    "LANDLOCK_LAUNCHER",
    "PROBE_ROLE",
    "PROCESS_CUSTODY_SCHEMA",
    "ProcessCustodyError",
    "ProcessCustodyReport",
    "RoleRun",
    "canonical_json_bytes",
    "normalized_sandbox_profile",
    "run_process_custody",
    "run_sandbox_blindness_probe",
]
