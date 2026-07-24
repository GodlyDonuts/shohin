#!/usr/bin/env python3
"""Fail-closed Landlock launcher for isolated Shohin custody stages."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import stat
import sys
from typing import Iterable, Mapping, Sequence


POLICY_SCHEMA = "shohin_landlock_stage_policy_v1"
MIN_SUPPORTED_ABI = 3
MAX_SUPPORTED_ABI = 10

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
PR_SET_DUMPABLE = 4
CLOSE_RANGE_UNSHARE = 1 << 1
CLOSE_RANGE_SYSCALL = 436

ACCESS_FS_EXECUTE = 1 << 0
ACCESS_FS_WRITE_FILE = 1 << 1
ACCESS_FS_READ_FILE = 1 << 2
ACCESS_FS_READ_DIR = 1 << 3
ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_MAKE_CHAR = 1 << 6
ACCESS_FS_MAKE_DIR = 1 << 7
ACCESS_FS_MAKE_REG = 1 << 8
ACCESS_FS_MAKE_SOCK = 1 << 9
ACCESS_FS_MAKE_FIFO = 1 << 10
ACCESS_FS_MAKE_BLOCK = 1 << 11
ACCESS_FS_MAKE_SYM = 1 << 12
ACCESS_FS_REFER = 1 << 13
ACCESS_FS_TRUNCATE = 1 << 14
ACCESS_FS_IOCTL_DEV = 1 << 15
ACCESS_FS_RESOLVE_UNIX = 1 << 16

_RIGHTS = (
    ("execute", ACCESS_FS_EXECUTE, 1),
    ("write_file", ACCESS_FS_WRITE_FILE, 1),
    ("read_file", ACCESS_FS_READ_FILE, 1),
    ("read_dir", ACCESS_FS_READ_DIR, 1),
    ("remove_dir", ACCESS_FS_REMOVE_DIR, 1),
    ("remove_file", ACCESS_FS_REMOVE_FILE, 1),
    ("make_char", ACCESS_FS_MAKE_CHAR, 1),
    ("make_dir", ACCESS_FS_MAKE_DIR, 1),
    ("make_reg", ACCESS_FS_MAKE_REG, 1),
    ("make_sock", ACCESS_FS_MAKE_SOCK, 1),
    ("make_fifo", ACCESS_FS_MAKE_FIFO, 1),
    ("make_block", ACCESS_FS_MAKE_BLOCK, 1),
    ("make_sym", ACCESS_FS_MAKE_SYM, 1),
    ("refer", ACCESS_FS_REFER, 2),
    ("truncate", ACCESS_FS_TRUNCATE, 3),
    ("ioctl_dev", ACCESS_FS_IOCTL_DEV, 5),
    ("resolve_unix", ACCESS_FS_RESOLVE_UNIX, 9),
)

_BASE_FILE_READ = ACCESS_FS_EXECUTE | ACCESS_FS_READ_FILE
_BASE_DIRECTORY_READ = _BASE_FILE_READ | ACCESS_FS_READ_DIR
_BASE_FILE_WRITE = ACCESS_FS_WRITE_FILE
_BASE_DIRECTORY_WRITE = (
    ACCESS_FS_WRITE_FILE
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_CHAR
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SOCK
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_MAKE_BLOCK
    | ACCESS_FS_MAKE_SYM
)


class LandlockStageError(RuntimeError):
    """Base class for launcher and policy failures."""


class LandlockUnsupportedError(LandlockStageError):
    """The host cannot enforce the exact supported Landlock contract."""


class LandlockPolicyError(LandlockStageError):
    """The requested stage or allowlist is invalid."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def validate_abi(abi: int) -> int:
    """Accept only ABI versions whose filesystem rights are understood."""

    if isinstance(abi, bool) or not isinstance(abi, int):
        raise LandlockUnsupportedError("Landlock ABI must be an integer")
    if not MIN_SUPPORTED_ABI <= abi <= MAX_SUPPORTED_ABI:
        raise LandlockUnsupportedError(
            f"Landlock ABI {abi} is outside the supported "
            f"{MIN_SUPPORTED_ABI}..{MAX_SUPPORTED_ABI} range"
        )
    return abi


def handled_access_fs(abi: int) -> int:
    """Return every filesystem right known to be enforceable by this ABI."""

    validate_abi(abi)
    return sum(bit for _, bit, introduced in _RIGHTS if abi >= introduced)


def access_names(mask: int) -> list[str]:
    """Return canonical, bit-ordered names for an access mask."""

    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0:
        raise LandlockPolicyError("access mask must be a nonnegative integer")
    known = sum(bit for _, bit, _ in _RIGHTS)
    if mask & ~known:
        raise LandlockPolicyError("access mask contains an unknown right")
    return [name for name, bit, _ in _RIGHTS if mask & bit]


def validate_stage(stage: str) -> str:
    """Validate a stable stage identity suitable for environment export."""

    if not isinstance(stage, str) or not stage:
        raise LandlockPolicyError("stage must be a nonempty string")
    if len(stage) > 128:
        raise LandlockPolicyError("stage exceeds 128 characters")
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if any(character not in allowed for character in stage):
        raise LandlockPolicyError(
            "stage may contain only ASCII letters, digits, dot, dash, and underscore"
        )
    return stage


def _canonical_path(raw: str | os.PathLike[str], *, cwd: Path) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise LandlockPolicyError("allowlist paths must be strings or path-like")
    value = os.fspath(raw)
    if not value or "\x00" in value:
        raise LandlockPolicyError("allowlist path is empty or contains NUL")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LandlockPolicyError(
            f"allowlist path cannot be canonicalized: {candidate}"
        ) from exc
    if not resolved.is_absolute():
        raise LandlockPolicyError("canonical allowlist path is not absolute")
    return resolved


def _rights_for_mode(mode: str, *, is_directory: bool, abi: int) -> int:
    supported = handled_access_fs(abi)
    if mode == "list_dir":
        if not is_directory:
            raise LandlockPolicyError("--list-dir requires a directory")
        requested = ACCESS_FS_READ_DIR
    elif mode == "ro":
        requested = _BASE_DIRECTORY_READ if is_directory else _BASE_FILE_READ
    elif mode == "rw":
        requested = _BASE_FILE_READ | _BASE_FILE_WRITE
        if is_directory:
            requested |= ACCESS_FS_READ_DIR | _BASE_DIRECTORY_WRITE
        if abi >= 2 and is_directory:
            requested |= ACCESS_FS_REFER
        if abi >= 3:
            requested |= ACCESS_FS_TRUNCATE
        if abi >= 5:
            requested |= ACCESS_FS_IOCTL_DEV
        if abi >= 9:
            requested |= ACCESS_FS_RESOLVE_UNIX
    else:
        raise LandlockPolicyError(f"unknown allowlist mode: {mode}")
    result = requested & supported
    if result == 0:
        raise LandlockPolicyError(f"{mode} produced an empty access mask")
    return result


def canonicalize_policy(
    *,
    stage: str,
    abi: int,
    ro: Iterable[str | os.PathLike[str]] = (),
    rw: Iterable[str | os.PathLike[str]] = (),
    list_dir: Iterable[str | os.PathLike[str]] = (),
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build the exact canonical policy consumed by launcher and verifier."""

    stage = validate_stage(stage)
    abi = validate_abi(abi)
    base = Path.cwd() if cwd is None else Path(cwd).expanduser()
    try:
        base = base.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LandlockPolicyError("policy working directory is invalid") from exc
    if not base.is_dir():
        raise LandlockPolicyError("policy working directory is not a directory")

    effective: dict[str, dict[str, object]] = {}
    sources = (("ro", ro), ("rw", rw), ("list_dir", list_dir))
    for mode, values in sources:
        for raw in values:
            path = _canonical_path(raw, cwd=base)
            try:
                metadata = path.stat()
            except OSError as exc:
                raise LandlockPolicyError(
                    f"allowlist path cannot be inspected: {path}"
                ) from exc
            is_directory = stat.S_ISDIR(metadata.st_mode)
            object_type = "directory" if is_directory else "file"
            rights = _rights_for_mode(mode, is_directory=is_directory, abi=abi)
            key = os.fspath(path)
            existing = effective.get(key)
            if existing is None:
                effective[key] = {
                    "path": key,
                    "object_type": object_type,
                    "st_dev": metadata.st_dev,
                    "st_ino": metadata.st_ino,
                    "allowed_access_fs": rights,
                }
            else:
                if existing["object_type"] != object_type:
                    raise LandlockPolicyError(
                        f"allowlist object type changed during canonicalization: {path}"
                    )
                existing["allowed_access_fs"] = (
                    int(existing["allowed_access_fs"]) | rights
                )

    rules = []
    for path in sorted(effective):
        row = dict(effective[path])
        row["allowed_access_fs_names"] = access_names(int(row["allowed_access_fs"]))
        rules.append(row)
    handled = handled_access_fs(abi)
    return {
        "schema": POLICY_SCHEMA,
        "landlock_abi": abi,
        "stage": stage,
        "handled_access_fs": handled,
        "handled_access_fs_names": access_names(handled),
        "rules": rules,
    }


def canonical_policy_bytes(policy: Mapping[str, object]) -> bytes:
    """Serialize an already canonical policy to its exact verification bytes."""

    required = {
        "schema",
        "landlock_abi",
        "stage",
        "handled_access_fs",
        "handled_access_fs_names",
        "rules",
    }
    if set(policy) != required or policy.get("schema") != POLICY_SCHEMA:
        raise LandlockPolicyError("policy schema or fields differ")
    abi = validate_abi(policy["landlock_abi"])
    stage = validate_stage(policy["stage"])
    if policy["stage"] != stage:
        raise LandlockPolicyError("policy stage differs")
    handled = policy["handled_access_fs"]
    if (
        isinstance(handled, bool)
        or not isinstance(handled, int)
        or handled != handled_access_fs(abi)
    ):
        raise LandlockPolicyError("handled access declaration differs from ABI")
    if policy["handled_access_fs_names"] != access_names(handled) or not isinstance(
        policy["handled_access_fs_names"], list
    ):
        raise LandlockPolicyError("handled access names differ from ABI")
    rules = policy["rules"]
    if not isinstance(rules, list):
        raise LandlockPolicyError("policy rules must be a list")
    previous = ""
    for rule in rules:
        if not isinstance(rule, Mapping) or set(rule) != {
            "path",
            "object_type",
            "st_dev",
            "st_ino",
            "allowed_access_fs",
            "allowed_access_fs_names",
        }:
            raise LandlockPolicyError("policy rule fields differ")
        path = rule["path"]
        if (
            not isinstance(path, str)
            or not path
            or "\x00" in path
            or not Path(path).is_absolute()
            or os.path.normpath(path) != path
        ):
            raise LandlockPolicyError("policy rule path is not absolute")
        if path <= previous:
            raise LandlockPolicyError("policy rules are not uniquely path-sorted")
        previous = path
        if rule["object_type"] not in {"directory", "file"}:
            raise LandlockPolicyError("policy rule object type differs")
        if (
            not isinstance(rule["st_dev"], int)
            or isinstance(rule["st_dev"], bool)
            or rule["st_dev"] < 0
            or not isinstance(rule["st_ino"], int)
            or isinstance(rule["st_ino"], bool)
            or rule["st_ino"] <= 0
        ):
            raise LandlockPolicyError("policy rule object identity differs")
        mask = rule["allowed_access_fs"]
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise LandlockPolicyError("policy rule access is not an integer")
        if mask == 0 or mask & ~handled:
            raise LandlockPolicyError("policy rule contains unsupported access")
        if not isinstance(rule["allowed_access_fs_names"], list) or rule[
            "allowed_access_fs_names"
        ] != access_names(mask):
            raise LandlockPolicyError("policy rule access names differ")
    return json.dumps(
        dict(policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def policy_sha256(policy: Mapping[str, object]) -> str:
    """Hash the exact canonical policy bytes used by stage verification."""

    return hashlib.sha256(canonical_policy_bytes(policy)).hexdigest()


def _syscall_numbers() -> tuple[int, int, int]:
    if platform.system() != "Linux":
        raise LandlockUnsupportedError("Landlock requires Linux")
    machine = platform.machine().lower()
    supported = {
        "aarch64",
        "amd64",
        "arm64",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "x86_64",
    }
    if machine not in supported:
        raise LandlockUnsupportedError(
            f"Landlock syscall numbers are not defined for architecture {machine}"
        )
    return 444, 445, 446


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    library.prctl.restype = ctypes.c_int
    return library


def _checked_result(result: int, operation: str) -> int:
    if result >= 0:
        return result
    error = ctypes.get_errno()
    detail = os.strerror(error) if error else "unknown error"
    raise LandlockStageError(f"{operation} failed: [errno {error}] {detail}")


def query_landlock_abi() -> int:
    """Query and validate the running kernel's Landlock ABI."""

    create_ruleset, _, _ = _syscall_numbers()
    library = _libc()
    ctypes.set_errno(0)
    result = library.syscall(
        ctypes.c_long(create_ruleset),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 0:
        error = ctypes.get_errno()
        if error in {
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EINVAL,
        }:
            raise LandlockUnsupportedError(
                f"Landlock is unavailable: [errno {error}] {os.strerror(error)}"
            )
        _checked_result(result, "Landlock ABI query")
    return validate_abi(int(result))


def enforce_policy(policy: Mapping[str, object]) -> None:
    """Install one filesystem policy on the current single-threaded process."""

    canonical_policy_bytes(policy)
    create_ruleset, add_rule, restrict_self = _syscall_numbers()
    abi = int(policy["landlock_abi"])
    current_abi = query_landlock_abi()
    if current_abi != abi:
        raise LandlockStageError(
            f"Landlock ABI changed from policy {abi} to runtime {current_abi}"
        )
    library = _libc()
    _prepare_inherited_descriptors(library)
    ruleset_attr = _RulesetAttr(handled_access_fs=int(policy["handled_access_fs"]))
    ctypes.set_errno(0)
    ruleset_fd = _checked_result(
        library.syscall(
            ctypes.c_long(create_ruleset),
            ctypes.byref(ruleset_attr),
            ctypes.c_size_t(ctypes.sizeof(ruleset_attr)),
            ctypes.c_uint(0),
        ),
        "landlock_create_ruleset",
    )
    path_fds: list[int] = []
    try:
        flags = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
        for rule in policy["rules"]:
            path_fd = os.open(str(rule["path"]), flags)
            path_fds.append(path_fd)
            _validate_rule_object_identity(rule, path_fd)
            path_attr = _PathBeneathAttr(
                allowed_access=int(rule["allowed_access_fs"]),
                parent_fd=path_fd,
            )
            ctypes.set_errno(0)
            _checked_result(
                library.syscall(
                    ctypes.c_long(add_rule),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(path_attr),
                    ctypes.c_uint(0),
                ),
                f"landlock_add_rule({rule['path']})",
            )

        ctypes.set_errno(0)
        _checked_result(
            library.prctl(
                ctypes.c_int(PR_SET_DUMPABLE),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            ),
            "prctl(PR_SET_DUMPABLE)",
        )
        ctypes.set_errno(0)
        _checked_result(
            library.prctl(
                ctypes.c_int(PR_SET_NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            ),
            "prctl(PR_SET_NO_NEW_PRIVS)",
        )
        ctypes.set_errno(0)
        _checked_result(
            library.syscall(
                ctypes.c_long(restrict_self),
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint(0),
            ),
            "landlock_restrict_self",
        )
    finally:
        for path_fd in path_fds:
            os.close(path_fd)
        os.close(ruleset_fd)


def _prepare_inherited_descriptors(library: ctypes.CDLL) -> None:
    import fcntl

    null_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.dup2(null_fd, 0)
    finally:
        if null_fd != 0:
            os.close(null_fd)
    for descriptor, label in ((1, "stdout"), (2, "stderr")):
        try:
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        except OSError as exc:
            raise LandlockStageError(f"{label} is unavailable") from exc
        if flags & os.O_ACCMODE != os.O_WRONLY:
            raise LandlockStageError(f"{label} must be write-only")
    ctypes.set_errno(0)
    result = library.syscall(
        ctypes.c_long(CLOSE_RANGE_SYSCALL),
        ctypes.c_uint(3),
        ctypes.c_uint(0xFFFFFFFF),
        ctypes.c_uint(CLOSE_RANGE_UNSHARE),
    )
    _checked_result(result, "close_range")


def parse_cli(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher options and require an explicit command separator."""

    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise LandlockPolicyError(
            "command must follow an explicit -- separator"
        ) from exc
    options = list(argv[:separator])
    command = list(argv[separator + 1 :])
    if not command:
        raise LandlockPolicyError("no command follows the -- separator")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--ro", action="append", default=[])
    parser.add_argument("--rw", action="append", default=[])
    parser.add_argument("--list-dir", action="append", default=[])
    parser.add_argument("--policy-receipt", required=True)
    namespace = parser.parse_args(options)
    return namespace, command


def launch(argv: Sequence[str]) -> None:
    """Canonicalize, restrict, export policy identity, and run one Python stage."""

    args, command = parse_cli(argv)
    abi = query_landlock_abi()
    policy = canonicalize_policy(
        stage=args.stage,
        abi=abi,
        ro=args.ro,
        rw=args.rw,
        list_dir=args.list_dir,
    )
    digest = policy_sha256(policy)
    policy_receipt = Path(args.policy_receipt).absolute()
    policy_receipt.parent.mkdir(parents=True, exist_ok=True)
    with policy_receipt.open("xb") as handle:
        handle.write(canonical_policy_bytes(policy))
        handle.flush()
        os.fsync(handle.fileno())
    policy_receipt.chmod(0o444)
    directory_fd = os.open(policy_receipt.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    enforce_policy(policy)
    if (
        len(command) < 2
        or not os.path.samefile(command[0], sys.executable)
        or command[1].startswith("-")
        or not Path(command[1]).is_file()
    ):
        raise LandlockPolicyError(
            "confined command must be this Python interpreter and one script"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "SHOHIN_LANDLOCK_ENFORCED": "1",
            "SHOHIN_LANDLOCK_ABI": str(abi),
            "SHOHIN_LANDLOCK_STAGE": str(policy["stage"]),
            "SHOHIN_LANDLOCK_POLICY_SHA256": digest,
            "SHOHIN_LANDLOCK_POLICY_PATH": str(policy_receipt),
        }
    )
    os.environ.clear()
    os.environ.update(environment)
    sys.argv = command[1:]
    runpy.run_path(command[1], run_name="__main__")


def _validate_rule_object_identity(
    rule: Mapping[str, object],
    descriptor: int,
) -> None:
    metadata = os.fstat(descriptor)
    expected_directory = rule["object_type"] == "directory"
    if (
        metadata.st_dev != rule["st_dev"]
        or metadata.st_ino != rule["st_ino"]
        or stat.S_ISDIR(metadata.st_mode) != expected_directory
    ):
        raise LandlockStageError(
            f"allowlist object changed before enforcement: {rule['path']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        launch(sys.argv[1:] if argv is None else argv)
    except (LandlockStageError, OSError) as exc:
        print(f"landlock-stage-exec: {exc}", file=sys.stderr, flush=True)
        return 125
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
