"""Minimal filesystem and Landlock custody used by HSC worker processes."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


LANDLOCK_RECEIPT_SCHEMA = "shohin_landlock_stage_receipt_v1"
DENIED_PROBE_RECEIPT_SCHEMA = "shohin_landlock_denied_probe_receipt_v1"


class RuntimeCustodyError(ValueError):
    """A runtime process or immutable publication left custody."""


def _process_dumpable() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeCustodyError(
            "Landlock stage verification requires Linux"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "prctl"):
        raise RuntimeCustodyError("prctl is unavailable")
    libc.prctl.restype = ctypes.c_int
    result = libc.prctl(3, 0, 0, 0, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise RuntimeCustodyError(
            f"PR_GET_DUMPABLE failed: {os.strerror(error)}"
        )
    return int(result)


def _read_json_verified(path: Path, expected_sha256: str) -> object:
    with path.open("rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeCustodyError(
            f"{path.name} hash differs from its runtime receipt"
        )
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeCustodyError(
            f"{path.name} is invalid JSON"
        ) from exc


def verify_landlock_stage(
    stage: str,
    deny_probe: Path,
) -> dict[str, object]:
    """Prove confinement and one real denied read inside the worker."""

    if os.environ.get("SHOHIN_LANDLOCK_ENFORCED") != "1":
        raise RuntimeCustodyError(
            "stage is not running under enforced Landlock"
        )
    if os.environ.get("SHOHIN_LANDLOCK_STAGE") != stage:
        raise RuntimeCustodyError("Landlock stage identity differs")
    try:
        abi = int(os.environ["SHOHIN_LANDLOCK_ABI"])
    except (KeyError, ValueError) as exc:
        raise RuntimeCustodyError(
            "Landlock ABI receipt is invalid"
        ) from exc
    if abi <= 0 or _process_dumpable() != 0:
        raise RuntimeCustodyError(
            "Landlock ABI or process dumpability differs"
        )
    policy_sha256 = os.environ.get(
        "SHOHIN_LANDLOCK_POLICY_SHA256",
        "",
    )
    if len(policy_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in policy_sha256
    ):
        raise RuntimeCustodyError(
            "Landlock policy SHA-256 is invalid"
        )
    policy_path_text = os.environ.get("SHOHIN_LANDLOCK_POLICY_PATH")
    if not policy_path_text:
        raise RuntimeCustodyError(
            "Landlock policy receipt path is missing"
        )
    canonical_policy = _read_json_verified(
        Path(policy_path_text),
        policy_sha256,
    )
    if (
        not isinstance(canonical_policy, dict)
        or canonical_policy.get("schema")
        != "shohin_landlock_stage_policy_v1"
        or canonical_policy.get("stage") != stage
        or canonical_policy.get("landlock_abi") != abi
        or not isinstance(canonical_policy.get("rules"), list)
    ):
        raise RuntimeCustodyError(
            "Landlock canonical policy differs"
        )
    try:
        with deny_probe.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise RuntimeCustodyError(
                "forbidden-input probe failed for a reason other "
                "than access denial"
            ) from exc
        denied_errno = exc.errno
    else:
        raise RuntimeCustodyError(
            "Landlock allowed a forbidden-input probe"
        )
    process_id = os.getpid()
    return {
        "schema": LANDLOCK_RECEIPT_SCHEMA,
        "stage": stage,
        "enforced": True,
        "dumpable": False,
        "abi": abi,
        "policy_sha256": policy_sha256,
        "canonical_policy": canonical_policy,
        "process_id": process_id,
        "denied_probe_receipt": {
            "schema": DENIED_PROBE_RECEIPT_SCHEMA,
            "stage": stage,
            "process_id": process_id,
            "operation": "open_read",
            "path": str(deny_probe.absolute()),
            "path_name": deny_probe.name,
            "path_sha256": hashlib.sha256(
                os.fsencode(deny_probe.absolute())
            ).hexdigest(),
            "denied": True,
            "errno": denied_errno,
        },
    }


def write_json_fsync(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    staging.chmod(0o555)
    fsync_directory(staging)
    fsync_directory(staging.parent)
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staging)
    destination = os.fsencode(output)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(source, destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source, -100, destination, 0x00000001)
    else:
        raise OSError(
            errno.ENOTSUP,
            "kernel no-replace directory publication is unavailable",
        )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to replace {output}")
        raise OSError(error, os.strerror(error), str(output))
    output.chmod(0o555)
    fsync_directory(output.parent)


def atomic_bundle_directory(output: Path) -> tuple[Path, Path]:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    lock = output.with_name(f".{output.name}.lock")
    descriptor = os.open(
        lock,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    os.close(descriptor)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.payload.",
            dir=output.parent,
        )
    )
    return staging, lock


def abort_atomic_bundle(staging: Path, lock: Path) -> None:
    if staging.exists():
        staging.chmod(0o755)
        for path in staging.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
    shutil.rmtree(staging, ignore_errors=True)
    lock.unlink(missing_ok=True)


def finish_atomic_bundle(
    staging: Path,
    output: Path,
    lock: Path,
) -> None:
    try:
        _publish_directory_noreplace(staging, output)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    lock.unlink(missing_ok=True)
    fsync_directory(output.absolute().parent)


__all__ = [
    "RuntimeCustodyError",
    "abort_atomic_bundle",
    "atomic_bundle_directory",
    "finish_atomic_bundle",
    "fsync_directory",
    "verify_landlock_stage",
    "write_json_fsync",
]
