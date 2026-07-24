#!/usr/bin/env python3
"""Standalone adversarial probe for the EFC role sandbox profile."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


PROBE_SCHEMA = "efc-process-sandbox-probe-v1"


class ProbeError(RuntimeError):
    """The probe contract or expected confinement result is invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _blocked_by_confinement(operation: object) -> bool:
    try:
        operation()
    except OSError as exc:
        return exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}
    return False


def _write_immutable(filename: str, payload: bytes) -> None:
    path = Path(filename)
    if path.is_absolute() or len(path.parts) != 1:
        raise ProbeError("probe output must be one relative filename")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, 0o400)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sandbox_receipt() -> tuple[str, str]:
    for prefix in ("SHOHIN_EFC_SANDBOX", "SHOHIN_LANDLOCK"):
        if (
            os.environ.get(f"{prefix}_ENFORCED") == "1"
            and os.environ.get(f"{prefix}_STAGE")
            and os.environ.get(f"{prefix}_POLICY_SHA256")
        ):
            return (
                os.environ[f"{prefix}_STAGE"],
                os.environ[f"{prefix}_POLICY_SHA256"],
            )
    raise ProbeError("probe is not inside the frozen sandbox")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 6:
        raise ProbeError(
            "usage: probe ALLOWED SECRET SECRET_DIR REPO OUTSIDE_WRITE OUTPUT"
        )
    (
        allowed_name,
        secret_path,
        secret_directory,
        repository_path,
        outside_write,
        output_name,
    ) = arguments
    sandbox_stage, sandbox_policy_sha256 = _sandbox_receipt()
    if Path(allowed_name).name != allowed_name or sandbox_stage != "blindness-probe":
        raise ProbeError("probe sandbox stage or input differs")

    def seatbelt_network_denied() -> bool:
        handle: socket.socket | None = None
        try:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.settimeout(0.25)
            handle.connect(("192.0.2.1", 9))
        except OSError as exc:
            return exc.errno in {errno.EACCES, errno.EPERM}
        finally:
            if handle is not None:
                handle.close()
        return False

    def network_namespace_isolated() -> bool:
        parent_identity = os.environ.get("SHOHIN_EFC_PARENT_NETNS")
        if not parent_identity:
            return False
        metadata = os.stat("/proc/self/ns/net")
        child_identity = f"{metadata.st_dev}:{metadata.st_ino}"
        if child_identity == parent_identity:
            return False
        handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        handle.settimeout(0.25)
        try:
            handle.connect(("192.0.2.1", 9))
        except OSError as exc:
            return exc.errno in {
                errno.ENETDOWN,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
            }
        finally:
            handle.close()
        return False

    network_mode = os.environ.get("SHOHIN_EFC_NETWORK_MODE")
    if network_mode == "socket-deny":
        network_blocked = seatbelt_network_denied()
    elif network_mode == "isolated-netns":
        network_blocked = network_namespace_isolated()
    else:
        raise ProbeError("probe network mode is absent or unknown")

    checks = {
        "allowed_input_read": Path(allowed_name).read_bytes() == b"public\n",
        "local_write_allowed": False,
        "network_socket_blocked": network_blocked,
        "outside_write_blocked": _blocked_by_confinement(
            lambda: Path(outside_write).write_bytes(b"escape\n")
        ),
        "repository_read_blocked": _blocked_by_confinement(
            lambda: Path(repository_path).read_bytes()
        ),
        "secret_directory_listing_blocked": _blocked_by_confinement(
            lambda: tuple(Path(secret_directory).iterdir())
        ),
        "secret_read_blocked": _blocked_by_confinement(
            lambda: Path(secret_path).read_bytes()
        ),
    }
    local = Path("local_write.txt")
    local.write_bytes(b"local\n")
    checks["local_write_allowed"] = local.read_bytes() == b"local\n"
    local.unlink()
    result = dict(checks)
    result.update(
        {
            "all_gates_pass": all(checks.values()),
            "network_test_mode": network_mode,
            "probe_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "sandbox_policy_sha256": sandbox_policy_sha256,
            "schema": PROBE_SCHEMA,
        }
    )
    _write_immutable(output_name, _canonical_json_bytes(result))
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProbeError) as exc:
        print(f"efc-sandbox-probe: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(125) from exc
