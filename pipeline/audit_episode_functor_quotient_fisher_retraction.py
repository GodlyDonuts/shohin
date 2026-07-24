#!/usr/bin/env python3
"""Fresh-board audit for quotient-Fisher causal retraction."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "train"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from episode_functor_causal_syndrome_observer import (  # noqa: E402
    explicit_causal_adjoint,
)
from episode_functor_quotient_fisher_retraction import (  # noqa: E402
    quotient_fisher_direction,
)
from pipeline.acw_nist_beacon import verify_pulse  # noqa: E402
from pipeline.audit_episode_functor_acso_oracle_recovery import (  # noqa: E402
    Fault,
    MachineTables,
    OutputReservation,
    _fault_logits,
    _hard_evidence,
    _innovation_components,
    _permutations,
    _publish,
    _recode,
    _reserve_output,
    _reservation_valid,
    _row_normalized,
    _tables,
    _targets,
    deep_fault_inventory,
)
from pipeline.audit_episode_functor_identifiable_board import (  # noqa: E402
    DEFAULT_COUNTS,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    generate_pilot_rows,
)
SCHEMA = "efc-qfcr-fresh-oracle/v2"
SEED_DOMAIN = "efc-qfcr-fresh-oracle-v2"
MARGINS = (0.05, 0.10, 0.20, 0.40, 0.80)
CYCLES = 4
MINIMUM_ELIGIBLE_WORLDS = 50
MINIMUM_FAULTS = 300
MONOTONIC_TOLERANCE = 1e-7
RECODING_TOLERANCE = 1e-5
MINIMUM_STRESS_GAP = 0.05
PROTOCOL_FILE = "R12_EFC_QUOTIENT_FISHER_RETRACTION_PROTOCOL.md"
BOOTSTRAP_FILE = (
    "pipeline/run_episode_functor_quotient_fisher_retraction_frozen.py"
)
AUTHORIZATION_FILE = (
    "artifacts/r12/qfcr_fresh_oracle_authorization.json"
)
AUTHORIZATION_SCHEMA = "efc-qfcr-fresh-oracle-authorization/v1"
MINIMUM_PULSE_DELAY_SECONDS = 6 * 60 * 60
GITHUB_EVENTS_URL = "https://api.github.com/repos/GodlyDonuts/shohin/events"
GITHUB_REPOSITORY = "GodlyDonuts/shohin"
SOURCE_ANCHOR_COMMIT = (
    "b93641619e25a8302cd46e96cfb6f20d5e657537"
)
SOURCE_ANCHOR_PARENT_COMMIT = (
    "27d5c4bd00591fbafa3dffe68a4c209bda0e8099"
)
SOURCE_FREEZE_PATHS = (
    PROTOCOL_FILE,
    "pipeline/audit_episode_functor_quotient_fisher_retraction.py",
    "pipeline/run_episode_functor_quotient_fisher_retraction_frozen.py",
    "pipeline/test_audit_episode_functor_quotient_fisher_retraction.py",
    "pipeline/test_run_episode_functor_quotient_fisher_retraction_frozen.py",
    "train/episode_functor_quotient_fisher_retraction.py",
    "train/test_episode_functor_quotient_fisher_retraction.py",
)
TRUSTED_NIST_CERTIFICATE_DER_SHA512 = (
    "528943a555f5f8ca54423be6dfb95925a35c7b552046420e7d7cd072058a14d65"
    "36ad3a8e9754b6582f164a90b0cd86a65d659f5426a2659a947595d1c816c8c"
)


class QFCRAuditError(ValueError):
    """The fresh-board audit contract failed closed."""


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    geometry: str
    routing_mode: str
    step: float


@dataclass(frozen=True, slots=True)
class Entry:
    world_id: str
    split: str
    machine: MachineTables
    fault: Fault


ARMS = (
    Arm("qf_causal", "quotient-fisher", "causal", 1.0),
    Arm("euclidean_equal_step", "euclidean", "causal", 1.0),
    Arm("qf_small_step", "quotient-fisher", "causal", 0.1),
    Arm("qf_one_step", "quotient-fisher", "one-step-control", 1.0),
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _configure_deterministic_runtime() -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise QFCRAuditError(
            "Torch thread runtime was initialized before custody"
        ) from exc
    torch.use_deterministic_algorithms(True)
    fixture = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    fixture = fixture.softmax(-1)
    torch.matmul(fixture, fixture.T)
    torch.einsum("ij,jk->ik", fixture, fixture)
    fixture.argmax(-1)
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise QFCRAuditError(
            "deterministic runtime configuration differs"
        )


def _cpu_identity_receipt() -> dict[str, object]:
    if sys.platform == "darwin":
        output = subprocess.check_output(
            ("sysctl", "-a"),
            text=True,
            stderr=subprocess.STDOUT,
        )
        prefixes = (
            "hw.byteorder:",
            "hw.cache",
            "hw.cpufrequency",
            "hw.cputype:",
            "hw.cpusubtype:",
            "hw.logicalcpu",
            "hw.machine:",
            "hw.memsize:",
            "hw.model:",
            "hw.ncpu:",
            "hw.optional.",
            "hw.packages:",
            "hw.pagesize:",
            "hw.perflevel",
            "hw.physicalcpu",
            "machdep.cpu.",
        )
        selected = sorted(
            line
            for line in output.splitlines()
            if line.startswith(prefixes)
        )
        if not selected:
            raise QFCRAuditError(
                "macOS CPU feature receipt is empty"
            )
        payload = "\n".join(selected) + "\n"
        return {
            "source": "sysctl-filtered-v1",
            "payload": payload,
            "payload_sha256": sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        payload = cpuinfo.read_text(
            encoding="utf-8",
            errors="strict",
        )
        return {
            "source": "proc-cpuinfo-v1",
            "payload": payload,
            "payload_sha256": sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        }
    raise QFCRAuditError(
        "CPU feature receipt is unavailable"
    )


def _loaded_native_image_receipt() -> dict[str, object]:
    paths = set()
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        count = process._dyld_image_count
        count.restype = ctypes.c_uint32
        name = process._dyld_get_image_name
        name.argtypes = (ctypes.c_uint32,)
        name.restype = ctypes.c_char_p
        for index in range(int(count())):
            value = name(index)
            if value:
                paths.add(
                    value.decode("utf-8", errors="strict")
                )
    elif Path("/proc/self/maps").is_file():
        for line in Path("/proc/self/maps").read_text(
            encoding="utf-8",
            errors="strict",
        ).splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[-1].startswith("/"):
                paths.add(fields[-1])
    else:
        raise QFCRAuditError(
            "loaded native image enumeration is unavailable"
        )
    rows = []
    for value in sorted(paths):
        path = Path(value)
        row = {"path": value}
        if path.is_file():
            row["bytes"] = path.stat().st_size
            row["sha256"] = _sha256_file(path)
        else:
            row["bytes"] = None
            row["sha256"] = None
        rows.append(row)
    if not rows:
        raise QFCRAuditError(
            "loaded native image receipt is empty"
        )
    dyld_cache = {}
    if sys.platform == "darwin":
        cache_root = Path(
            "/System/Volumes/Preboot/Cryptexes/OS/"
            "System/Library/dyld"
        )
        cache_files = sorted(
            path
            for path in cache_root.glob(
                f"dyld_shared_cache_{platform.machine()}*"
            )
            if path.is_file()
        )
        if not cache_files:
            raise QFCRAuditError(
                "active dyld shared cache receipt is unavailable"
            )
        dyld_cache = {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in cache_files
        }
    return {
        "image_count": len(rows),
        "manifest_sha256": sha256(
            _canonical_json_bytes(rows)
        ).hexdigest(),
        "images": rows,
        "dyld_shared_cache_files": dyld_cache,
    }


def _tree_receipt(
    root: Path,
    *,
    excluded_top_level: tuple[str, ...] = (),
) -> dict[str, object]:
    rows = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not relative.parts
            or relative.parts[0] in excluded_top_level
            or "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_symlink():
            rows.append(
                {
                    "path": str(relative),
                    "symlink": os.readlink(path),
                }
            )
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": str(relative),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise QFCRAuditError(
            f"runtime tree receipt is empty: {root}"
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256(
            _canonical_json_bytes(rows)
        ).hexdigest(),
    }


def _native_dependency_receipt(
    paths: set[Path],
) -> dict[str, object]:
    otool_name = shutil.which("otool")
    ldd_name = shutil.which("ldd")
    tool = otool_name or ldd_name
    if tool is None:
        raise QFCRAuditError(
            "native dependency inspection tool is unavailable"
        )
    executable = Path(tool).resolve()
    rows = {}
    dependency_files = set()
    for path in sorted(paths):
        output = subprocess.check_output(
            (str(executable), "-L", str(path))
            if otool_name
            else (str(executable), str(path)),
            text=True,
            stderr=subprocess.STDOUT,
        )
        rows[str(path)] = output
        for line in output.splitlines():
            token = line.strip().split(" ", 1)[0]
            candidate = Path(token)
            if candidate.is_absolute() and candidate.is_file():
                dependency_files.add(candidate.resolve())
    return {
        "tool": str(executable),
        "tool_sha256": _sha256_file(executable),
        "link_maps": rows,
        "resolved_dependency_files": {
            str(path): _sha256_file(path)
            for path in sorted(dependency_files)
        },
    }


def _environment_receipt() -> dict[str, object]:
    executable = Path(sys.executable).resolve()
    openssl_name = shutil.which("openssl")
    if (
        not executable.is_file()
        or openssl_name is None
        or not Path(openssl_name).resolve().is_file()
    ):
        raise QFCRAuditError(
            "runtime executable receipt is unavailable"
        )
    openssl = Path(openssl_name).resolve()
    torch_root = Path(torch.__file__).resolve().parent
    native_candidates = {
        Path(torch._C.__file__).resolve(),
        *(
            path.resolve()
            for path in (torch_root / "lib").glob("*")
            if path.is_file()
        ),
    }
    native = {
        str(path): _sha256_file(path)
        for path in sorted(native_candidates)
        if path.is_file()
    }
    if not native:
        raise QFCRAuditError(
            "Torch native runtime receipt is empty"
        )
    openssl_version = subprocess.check_output(
        (str(openssl), "version", "-a"),
        text=True,
        stderr=subprocess.STDOUT,
    )
    numeric_environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if any(
            token in name.upper()
            for token in (
                "ATEN",
                "BLAS",
                "CUDA",
                "DNNL",
                "MKL",
                "MPS",
                "NUMEXPR",
                "OMP",
                "ONEDNN",
                "PYTHONHASHSEED",
                "TORCH",
                "VECLIB",
            )
        )
    }
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve()
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK",
        )
    }
    return {
        "python_executable": str(executable),
        "python_executable_sha256": _sha256_file(executable),
        "python_version": sys.version,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_flags": {
            name: getattr(sys.flags, name)
            for name in (
                "debug",
                "optimize",
                "dont_write_bytecode",
                "no_user_site",
                "no_site",
                "ignore_environment",
                "isolated",
                "hash_randomization",
                "safe_path",
            )
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_identity": _cpu_identity_receipt(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_native_files": native,
        "torch_package_tree": _tree_receipt(torch_root),
        "python_stdlib_tree": _tree_receipt(
            stdlib_root,
            excluded_top_level=("site-packages",),
        ),
        "native_dependencies": _native_dependency_receipt(
            {
                executable,
                openssl,
                *native_candidates,
            }
        ),
        "loaded_native_images": _loaded_native_image_receipt(),
        "openssl_executable": str(openssl),
        "openssl_executable_sha256": _sha256_file(openssl),
        "openssl_version": openssl_version,
        "relevant_environment": relevant_environment,
        "numeric_environment": numeric_environment,
    }


def _runtime_source_paths() -> tuple[str, ...]:
    paths = {PROTOCOL_FILE, BOOTSTRAP_FILE}
    if (ROOT / AUTHORIZATION_FILE).is_file():
        paths.add(AUTHORIZATION_FILE)
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix == ".py" and path.is_file():
            paths.add(str(relative))
    return tuple(sorted(paths))


def _source_receipt(
    paths: tuple[str, ...],
) -> tuple[str, list[dict[str, object]], bool]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
    ).strip()
    rows = []
    matched = True
    for relative in paths:
        local = (ROOT / relative).read_bytes()
        try:
            committed = subprocess.check_output(
                ("git", "show", f"HEAD:{relative}"),
                cwd=ROOT,
            )
        except subprocess.CalledProcessError:
            committed = b""
        row_match = local == committed
        matched = matched and row_match
        rows.append(
            {
                "path": relative,
                "sha256": sha256(local).hexdigest(),
                "head_sha256": sha256(committed).hexdigest(),
                "matches_head": row_match,
            }
        )
    return commit, rows, matched


def _load_canonical_json(
    path: Path,
    *,
    maximum_bytes: int,
) -> tuple[dict[str, object], bytes]:
    path = path.resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > maximum_bytes
    ):
        raise QFCRAuditError("custody JSON path differs")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise QFCRAuditError("custody JSON is malformed") from exc
    if (
        not isinstance(value, dict)
        or payload != _canonical_json_bytes(value) + b"\n"
    ):
        raise QFCRAuditError("custody JSON is not canonical")
    return value, payload


def _github_push_receipts(
    *,
    anchor_head: str,
    source_head: str,
    authorization_head: str,
    branch: str,
) -> dict[str, object]:
    request = urllib.request.Request(
        GITHUB_EVENTS_URL + "?per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "shohin-qfcr-custody/1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise QFCRAuditError(
                f"GitHub Events returned HTTP {response.status}"
            )
        raw = response.read(4 * 1024 * 1024 + 1)
        response_date = response.headers.get("Date")
    if len(raw) > 4 * 1024 * 1024:
        raise QFCRAuditError(
            "GitHub Events response exceeds four MiB"
        )
    try:
        events = json.loads(raw)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise QFCRAuditError(
            "GitHub Events response is malformed"
        ) from exc
    if not isinstance(events, list):
        raise QFCRAuditError(
            "GitHub Events response is not a list"
        )
    target_ref = f"refs/heads/{branch}"
    expected = {
        "anchor": anchor_head,
        "source": source_head,
        "authorization": authorization_head,
    }
    receipts: dict[str, object] = {}
    positions: dict[str, int] = {}
    branch_events = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("type") != "PushEvent"
            or event.get("public") is not True
            or not isinstance(event.get("payload"), dict)
            or not isinstance(event.get("repo"), dict)
            or event["repo"].get("name") != GITHUB_REPOSITORY
            or event["payload"].get("ref") != target_ref
        ):
            continue
        branch_events.append((len(branch_events), event))
    for label, expected_head in expected.items():
        matching = []
        for position, event in branch_events:
            payload = event.get("payload")
            repository = event.get("repo")
            if (
                not isinstance(payload, dict)
                or not isinstance(repository, dict)
                or repository.get("name") != GITHUB_REPOSITORY
                or payload.get("ref") != target_ref
                or payload.get("head") != expected_head
                or event.get("public") is not True
                or not isinstance(event.get("id"), str)
                or not isinstance(event.get("created_at"), str)
            ):
                continue
            matching.append((position, event))
        if len(matching) != 1:
            raise QFCRAuditError(
                f"public GitHub PushEvent is not unique for {label} commit"
            )
        position, event = matching[0]
        payload = event["payload"]
        created_at = str(event["created_at"])
        try:
            parsed = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise QFCRAuditError(
                "GitHub push event timestamp is malformed"
            ) from exc
        if parsed.tzinfo is None:
            raise QFCRAuditError(
                "GitHub push event timestamp lacks timezone"
            )
        receipts[label] = {
            "github_push_event_before": payload.get("before"),
            "github_push_event_created_at": created_at,
            "github_push_event_head": expected_head,
            "github_push_event_id": event["id"],
            "github_push_event_payload_sha256": sha256(
                _canonical_json_bytes(event)
            ).hexdigest(),
            "github_push_event_public": True,
            "github_push_event_ref": target_ref,
            "github_push_event_repository": GITHUB_REPOSITORY,
        }
        positions[label] = position
    matched_payloads = {
        label: next(
            event["payload"]
            for position, event in branch_events
            if position == positions[label]
        )
        for label in expected
    }
    if (
        matched_payloads["anchor"].get("before")
        != SOURCE_ANCHOR_PARENT_COMMIT
        or
        matched_payloads["authorization"].get("before")
        != source_head
        or matched_payloads["source"].get("before")
        != anchor_head
        or positions["authorization"] >= positions["source"]
        or positions["source"] - positions["authorization"] != 1
        or positions["source"] >= positions["anchor"]
        or positions["anchor"] - positions["source"] != 1
    ):
        raise QFCRAuditError(
            "source and authorization are not the first direct public "
            "successors of the immutable anchor"
        )
    anchor_timestamp = datetime.fromisoformat(
        str(
            receipts["anchor"][
                "github_push_event_created_at"
            ]
        ).replace("Z", "+00:00")
    )
    source_timestamp = datetime.fromisoformat(
        str(
            receipts["source"][
                "github_push_event_created_at"
            ]
        ).replace("Z", "+00:00")
    )
    authorization_timestamp = datetime.fromisoformat(
        str(
            receipts["authorization"][
                "github_push_event_created_at"
            ]
        ).replace("Z", "+00:00")
    )
    if (
        source_timestamp < anchor_timestamp
        or authorization_timestamp < source_timestamp
    ):
        raise QFCRAuditError(
            "public source/authorization chronology differs"
        )
    if response_date is None:
        raise QFCRAuditError(
            "GitHub Events response lacks server Date"
        )
    try:
        response_timestamp = parsedate_to_datetime(response_date)
    except (TypeError, ValueError) as exc:
        raise QFCRAuditError(
            "GitHub Events server Date is malformed"
        ) from exc
    if (
        response_timestamp.tzinfo is None
        or (
            response_timestamp - authorization_timestamp
        ).total_seconds()
        < MINIMUM_PULSE_DELAY_SECONDS
    ):
        raise QFCRAuditError(
            "GitHub event history has not reached its maturity gate"
        )
    return {
        "github_events_response_date": response_date,
        "anchor": receipts["anchor"],
        "source": receipts["source"],
        "authorization": receipts["authorization"],
    }


def _authorization_receipt(
    *,
    head: str,
    environment_sha256: str,
    authorization_path: Path,
    beacon_snapshot_path: Path,
) -> tuple[str, dict[str, object]]:
    status = subprocess.check_output(
        ("git", "status", "--porcelain"),
        cwd=ROOT,
        text=True,
    )
    if status:
        raise QFCRAuditError(
            "worktree must be clean before entropy consumption"
        )
    branch = subprocess.check_output(
        ("git", "branch", "--show-current"),
        cwd=ROOT,
        text=True,
    ).strip()
    if branch != "" or not (ROOT / ".git").is_file():
        raise QFCRAuditError(
            "official audit requires a detached Git-object worktree"
        )
    public_branch = "main"
    remote = subprocess.check_output(
        ("git", "ls-remote", "origin", "refs/heads/main"),
        cwd=ROOT,
        text=True,
    ).strip()
    if not remote or remote.split()[0] != head:
        raise QFCRAuditError(
            "authorization HEAD is not public origin/main"
        )
    fixed_authorization = (ROOT / AUTHORIZATION_FILE).resolve()
    if authorization_path.resolve() != fixed_authorization:
        raise QFCRAuditError(
            "authorization path is not the fixed repository path"
        )
    authorization, authorization_payload = _load_canonical_json(
        fixed_authorization,
        maximum_bytes=16 * 1024,
    )
    committed_authorization = subprocess.check_output(
        ("git", "show", f"HEAD:{AUTHORIZATION_FILE}"),
        cwd=ROOT,
    )
    if committed_authorization != authorization_payload:
        raise QFCRAuditError(
            "authorization differs from public HEAD blob"
        )
    parent = subprocess.check_output(
        ("git", "rev-parse", "HEAD^"),
        cwd=ROOT,
        text=True,
    ).strip()
    source_parent = subprocess.check_output(
        ("git", "rev-parse", "HEAD^^"),
        cwd=ROOT,
        text=True,
    ).strip()
    changed = tuple(
        line
        for line in subprocess.check_output(
            (
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "HEAD^",
                "HEAD",
            ),
            cwd=ROOT,
            text=True,
        ).splitlines()
        if line
    )
    source_changed = tuple(
        sorted(
            line
            for line in subprocess.check_output(
                (
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    "HEAD^^",
                    "HEAD^",
                ),
                cwd=ROOT,
                text=True,
            ).splitlines()
            if line
        )
    )
    expected_keys = {
        "schema",
        "source_commit",
        "nist_chain_index",
        "target_pulse_index",
        "minimum_pulse_delay_seconds",
        "expected_environment_sha256",
    }
    if (
        set(authorization) != expected_keys
        or authorization["schema"] != AUTHORIZATION_SCHEMA
        or authorization["source_commit"] != parent
        or source_parent != SOURCE_ANCHOR_COMMIT
        or source_changed != tuple(sorted(SOURCE_FREEZE_PATHS))
        or changed != (AUTHORIZATION_FILE,)
        or subprocess.run(
            (
                "git",
                "cat-file",
                "-e",
                f"HEAD^:{AUTHORIZATION_FILE}",
            ),
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
        or authorization["minimum_pulse_delay_seconds"]
        != MINIMUM_PULSE_DELAY_SECONDS
        or authorization["expected_environment_sha256"]
        != environment_sha256
        or not isinstance(
            authorization["expected_environment_sha256"],
            str,
        )
        or len(
            authorization["expected_environment_sha256"]
        )
        != 64
        or not isinstance(
            authorization["nist_chain_index"], int
        )
        or not isinstance(
            authorization["target_pulse_index"], int
        )
    ):
        raise QFCRAuditError("authorization contract differs")
    github = _github_push_receipts(
        anchor_head=SOURCE_ANCHOR_COMMIT,
        source_head=parent,
        authorization_head=head,
        branch=public_branch,
    )
    snapshot, snapshot_payload = _load_canonical_json(
        beacon_snapshot_path,
        maximum_bytes=4 * 1024 * 1024,
    )
    if set(snapshot) != {
        "certificate_pem",
        "previous_pulse",
        "pulse",
    }:
        raise QFCRAuditError("beacon snapshot schema differs")
    certificate = snapshot["certificate_pem"]
    pulse = snapshot["pulse"]
    previous = snapshot["previous_pulse"]
    if (
        not isinstance(certificate, str)
        or not isinstance(pulse, dict)
        or not isinstance(previous, dict)
    ):
        raise QFCRAuditError("beacon snapshot types differ")
    try:
        previous_receipt = verify_pulse(
            previous,
            certificate.encode("ascii"),
            expected_chain_index=int(
                authorization["nist_chain_index"]
            ),
            expected_pulse_index=int(
                authorization["target_pulse_index"]
            )
            - 1,
        )
        pulse_receipt = verify_pulse(
            pulse,
            certificate.encode("ascii"),
            previous_pulse=previous,
            expected_chain_index=int(
                authorization["nist_chain_index"]
            ),
            expected_pulse_index=int(
                authorization["target_pulse_index"]
            ),
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise QFCRAuditError(
            "NIST pulse verification failed"
        ) from exc
    if (
        previous_receipt["certificate_der_sha512"]
        != TRUSTED_NIST_CERTIFICATE_DER_SHA512
        or
        pulse_receipt["certificate_der_sha512"]
        != TRUSTED_NIST_CERTIFICATE_DER_SHA512
    ):
        raise QFCRAuditError("NIST certificate pin differs")
    pulse_timestamp = datetime.fromisoformat(
        str(pulse_receipt["timestamp"]).replace("Z", "+00:00")
    )
    push_timestamp = datetime.fromisoformat(
        str(
            github["authorization"][
                "github_push_event_created_at"
            ]
        ).replace("Z", "+00:00")
    )
    if (
        pulse_timestamp - push_timestamp
    ).total_seconds() < MINIMUM_PULSE_DELAY_SECONDS:
        raise QFCRAuditError(
            "NIST pulse is not sufficiently after public freeze"
        )
    seed_material = (
        f"{SEED_DOMAIN}\0{head}\0"
        f"{pulse_receipt['output_value']}"
    ).encode("ascii")
    seed = f"{SEED_DOMAIN}:{sha256(seed_material).hexdigest()}"
    return seed, {
        "authorization_sha256": sha256(
            authorization_payload
        ).hexdigest(),
        "authorization": authorization,
        "authorization_commit": head,
        "source_commit": parent,
        "source_anchor_commit": source_parent,
        "source_changed_paths": source_changed,
        "changed_paths": changed,
        "github": github,
        "beacon_snapshot_sha256": sha256(
            snapshot_payload
        ).hexdigest(),
        "previous_pulse": previous_receipt,
        "pulse": pulse_receipt,
        "derived_seed_sha256": sha256(
            seed.encode("ascii")
        ).hexdigest(),
    }


def _fresh_board(
    seed: str,
) -> tuple[
    list[Entry],
    list[dict[str, object]],
    dict[str, int],
    dict[str, int],
]:
    rows = generate_pilot_rows(seed=seed, counts=DEFAULT_COUNTS)
    entries = []
    manifest = []
    eligible_by_split = {split: 0 for split in DEFAULT_COUNTS}
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        machine = _tables(row.machine)
        existing = unique.get(row.world_id)
        if existing is None:
            unique[row.world_id] = {
                "split": row.split,
                "family": row.family,
                "machine": machine,
                "canonical_sha256": row.canonical_sha256,
                "source_count": 1,
            }
            continue
        if (
            existing["split"] != row.split
            or existing["family"] != row.family
            or existing["machine"] != machine
            or existing["canonical_sha256"] != row.canonical_sha256
        ):
            raise QFCRAuditError(
                "renderer variants disagree on latent world"
            )
        existing["source_count"] = int(
            existing["source_count"]
        ) + 1
    observed_split_counts = dict(
        Counter(
            str(item["split"]) for item in unique.values()
        )
    )
    for world_id in sorted(unique):
        item = unique[world_id]
        machine = item["machine"]
        split = str(item["split"])
        faults = deep_fault_inventory(machine)
        eligible_by_split[split] += int(bool(faults))
        manifest.append(
            {
                "world_id": world_id,
                "split": split,
                "family": item["family"],
                "canonical_sha256": item["canonical_sha256"],
                "source_count": item["source_count"],
                "transitions": machine.transitions,
                "observations": machine.observations,
                "fault_count": len(faults),
            }
        )
        entries.extend(
            Entry(
                world_id=world_id,
                split=split,
                machine=machine,
                fault=fault,
            )
            for fault in faults
        )
    return (
        entries,
        manifest,
        eligible_by_split,
        observed_split_counts,
    )


def _batch(
    entries: list[Entry],
    margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    transitions = []
    observers = []
    bases = []
    derivatives = []
    for entry in entries:
        transition, observer = _fault_logits(
            entry.machine,
            entry.fault,
            margin,
        )
        base, derivative = _targets(entry.machine)
        transitions.append(transition)
        observers.append(observer)
        bases.append(base)
        derivatives.append(derivative)
    return tuple(
        torch.cat(values)
        for values in (
            transitions,
            observers,
            bases,
            derivatives,
        )
    )


def _hard_batch(
    transition: torch.Tensor,
    observer: torch.Tensor,
    entries: list[Entry],
) -> tuple[
    list[bool],
    list[bool],
    list[bool],
    list[float],
    list[list[list[int]]],
    list[list[list[int]]],
]:
    if (
        not bool(torch.isfinite(transition).all())
        or not bool(torch.isfinite(observer).all())
    ):
        raise QFCRAuditError("hard evidence is nonfinite")
    exact = []
    recovered = []
    tied = []
    gaps = []
    for index, entry in enumerate(entries):
        item_exact, item_recovered, item_tied = _hard_evidence(
            transition[index : index + 1],
            observer[index : index + 1],
            entry.machine,
            entry.fault,
        )
        correct = entry.machine.transitions[
            entry.fault.action
        ][entry.fault.state]
        gap = (
            transition[
                index,
                entry.fault.action,
                entry.fault.state,
                correct,
            ]
            - transition[
                index,
                entry.fault.action,
                entry.fault.state,
                entry.fault.wrong,
            ]
        )
        exact.append(item_exact)
        recovered.append(item_recovered)
        tied.append(item_tied)
        gaps.append(float(gap))
    return (
        exact,
        recovered,
        tied,
        gaps,
        transition.argmax(-1).tolist(),
        observer.argmax(-1).tolist(),
    )


def _run_arm(
    entries: list[Entry],
    *,
    margin: float,
    arm: Arm,
) -> dict[str, object]:
    transition, observer, base, derivative = _batch(entries, margin)
    cycles = []
    for cycle in range(CYCLES + 1):
        base_value, derivative_value = _innovation_components(
            transition,
            observer,
            base,
            derivative,
            routing_mode=arm.routing_mode,
        )
        (
            exact,
            recovered,
            tied,
            gaps,
            transition_decision,
            observer_decision,
        ) = _hard_batch(
            transition,
            observer,
            entries,
        )
        if (
            not bool(torch.isfinite(base_value).all())
            or not bool(torch.isfinite(derivative_value).all())
            or not all(math.isfinite(gap) for gap in gaps)
        ):
            raise QFCRAuditError("arm evidence is nonfinite")
        cycles.append(
            {
                "cycle": cycle,
                "base_innovation": base_value.tolist(),
                "derivative_innovation": derivative_value.tolist(),
                "exact_machine": exact,
                "fault_row_recovered": recovered,
                "any_tied_row": tied,
                "correct_minus_wrong_gap": gaps,
                "transition_decision": transition_decision,
                "observer_decision": observer_decision,
            }
        )
        if cycle == CYCLES:
            break
        adjoint = explicit_causal_adjoint(
            transition,
            observer,
            base,
            derivative,
            max_depth=3,
            routing_mode=arm.routing_mode,
        )
        if arm.geometry == "euclidean":
            transition_direction = _row_normalized(
                adjoint.transition_logit_adjoint
            )
            observer_direction = _row_normalized(
                adjoint.observer_logit_adjoint
            )
        elif arm.geometry == "quotient-fisher":
            direction = quotient_fisher_direction(
                transition,
                observer,
                adjoint.transition_logit_adjoint,
                adjoint.observer_logit_adjoint,
            )
            transition_direction = direction.transition
            observer_direction = direction.observer
        else:
            raise QFCRAuditError("audit arm geometry differs")
        if (
            not bool(torch.isfinite(transition_direction).all())
            or not bool(torch.isfinite(observer_direction).all())
        ):
            raise QFCRAuditError("arm direction is nonfinite")
        transition = (
            transition - arm.step * transition_direction
        )
        observer = observer - arm.step * observer_direction
        if (
            not bool(torch.isfinite(transition).all())
            or not bool(torch.isfinite(observer).all())
        ):
            raise QFCRAuditError("arm update is nonfinite")
    return {
        "configuration": asdict(arm),
        "cycles": cycles,
    }


def _recoded_entries(entries: list[Entry]) -> list[Entry]:
    result = []
    for entry in entries:
        machine, fault = _recode(
            entry.machine,
            entry.fault,
            _permutations(entry.world_id),
        )
        result.append(
            Entry(
                world_id=entry.world_id,
                split=entry.split,
                machine=machine,
                fault=fault,
            )
        )
    return result


def _first_exact_cycles(
    arm_result: dict[str, object],
) -> list[int]:
    cycles = arm_result["cycles"]
    count = len(cycles[0]["exact_machine"])
    result = []
    for index in range(count):
        result.append(
            next(
                (
                    int(cycle["cycle"])
                    for cycle in cycles
                    if cycle["exact_machine"][index]
                ),
                CYCLES + 1,
            )
        )
    return result


def _inverse_permutation(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for old, new in enumerate(permutation):
        inverse[new] = old
    return tuple(inverse)


def _unrecode_decisions(
    world_id: str,
    transition: list[list[int]],
    observer: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    state, action, observer_key, answer = _permutations(world_id)
    inverse_state = _inverse_permutation(state)
    inverse_answer = _inverse_permutation(answer)
    original_transition = [
        [
            inverse_state[
                transition[action[old_action]][state[old_state]]
            ]
            for old_state in range(len(state))
        ]
        for old_action in range(len(action))
    ]
    original_observer = [
        [
            inverse_answer[
                observer[observer_key[old_observer]][state[old_state]]
            ]
            for old_state in range(len(state))
        ]
        for old_observer in range(len(observer_key))
    ]
    return original_transition, original_observer


def _arm_receipt(
    original: dict[str, object],
    recoded: dict[str, object],
    entries: list[Entry],
) -> dict[str, object]:
    original_cycles = original["cycles"]
    recoded_cycles = recoded["cycles"]
    final = original_cycles[-1]
    count = len(final["exact_machine"])
    monotonic = 0
    recoded_monotonic = 0
    decision_mismatches = 0
    maximum_delta = 0.0
    for index in range(count):
        left_totals = [
            float(cycle["base_innovation"][index])
            + float(cycle["derivative_innovation"][index])
            for cycle in original_cycles
        ]
        right_totals = [
            float(cycle["base_innovation"][index])
            + float(cycle["derivative_innovation"][index])
            for cycle in recoded_cycles
        ]
        monotonic += sum(
            right > left + MONOTONIC_TOLERANCE
            for left, right in zip(left_totals, left_totals[1:])
        )
        recoded_monotonic += sum(
            right > left + MONOTONIC_TOLERANCE
            for left, right in zip(right_totals, right_totals[1:])
        )
        maximum_delta = max(
            maximum_delta,
            max(
                max(
                    abs(
                        float(left[kind][index])
                        - float(right[kind][index])
                    )
                    for kind in (
                        "base_innovation",
                        "derivative_innovation",
                    )
                )
                for left, right in zip(
                    original_cycles,
                    recoded_cycles,
                    strict=True,
                )
            ),
        )
        for left, right in zip(
            original_cycles,
            recoded_cycles,
            strict=True,
        ):
            transition, observer = _unrecode_decisions(
                entries[index].world_id,
                right["transition_decision"][index],
                right["observer_decision"][index],
            )
            decision_mismatches += int(
                left["transition_decision"][index]
                != transition
                or left["observer_decision"][index]
                != observer
                or left["any_tied_row"][index]
                != right["any_tied_row"][index]
            )
    first_cycles = _first_exact_cycles(original)
    represented_worlds = sorted(
        {entry.world_id for entry in entries}
    )
    world_exact_recovery = {}
    for world_id in represented_worlds:
        selected = [
            index
            for index, entry in enumerate(entries)
            if entry.world_id == world_id
        ]
        world_exact_recovery[world_id] = (
            sum(final["exact_machine"][index] for index in selected)
            / len(selected)
        )
    return {
        "case_count": count,
        "exact_recovery": sum(final["exact_machine"]) / count,
        "row_recovery": sum(final["fault_row_recovered"]) / count,
        "final_ties": sum(final["any_tied_row"]),
        "monotonic_violations": monotonic,
        "recoded_monotonic_violations": recoded_monotonic,
        "recoding_decision_mismatches": decision_mismatches,
        "maximum_recoding_innovation_delta": maximum_delta,
        "median_first_exact_cycle": statistics.median(first_cycles),
        "minimum_world_exact_recovery": min(
            world_exact_recovery.values()
        ),
        "world_exact_recovery": world_exact_recovery,
        "valid": bool(
            math.isfinite(maximum_delta)
            and 0.0 <= maximum_delta
            and all(
                math.isfinite(value)
                for cycle in original_cycles + recoded_cycles
                for kind in (
                    "base_innovation",
                    "derivative_innovation",
                    "correct_minus_wrong_gap",
                )
                for value in cycle[kind]
            )
        ),
    }


def _decision(
    margin_receipts: list[dict[str, object]],
    *,
    bindings_pass: bool,
) -> tuple[str, bool, bool]:
    expected_arms = {arm.name for arm in ARMS}
    if (
        len(margin_receipts) != len(MARGINS)
        or {float(row["margin"]) for row in margin_receipts}
        != set(MARGINS)
        or any(
            set(row["arms"]) != expected_arms
            for row in margin_receipts
        )
    ):
        raise QFCRAuditError(
            "decision receipt set differs from frozen configuration"
        )
    treatments = {
        float(row["margin"]): row["arms"]["qf_causal"]
        for row in margin_receipts
    }
    euclidean = {
        float(row["margin"]): row["arms"]["euclidean_equal_step"]
        for row in margin_receipts
    }
    controls_valid = all(
        bool(
            receipt["valid"]
            and receipt["recoding_decision_mismatches"] == 0
            and receipt["maximum_recoding_innovation_delta"]
            <= RECODING_TOLERANCE
        )
        for row in margin_receipts
        for receipt in row["arms"].values()
    )
    mechanics = bool(
        bindings_pass
        and controls_valid
        and all(
            receipt["valid"]
            and
            receipt["exact_recovery"] == 1.0
            and receipt["row_recovery"] == 1.0
            and receipt["final_ties"] == 0
            and receipt["monotonic_violations"] == 0
            and receipt["recoded_monotonic_violations"] == 0
            and receipt["recoding_decision_mismatches"] == 0
            and receipt["maximum_recoding_innovation_delta"]
            <= RECODING_TOLERANCE
            and receipt["minimum_world_exact_recovery"] == 1.0
            for receipt in treatments.values()
        )
    )
    never_worse = all(
        treatments[margin]["exact_recovery"]
        >= euclidean[margin]["exact_recovery"]
        for margin in MARGINS
    )
    stress_gap = max(
        treatments[margin]["exact_recovery"]
        - euclidean[margin]["exact_recovery"]
        for margin in (0.40, 0.80)
    )
    earlier = any(
        treatments[margin]["exact_recovery"] == 1.0
        and euclidean[margin]["exact_recovery"] == 1.0
        and treatments[margin]["median_first_exact_cycle"]
        < euclidean[margin]["median_first_exact_cycle"]
        for margin in MARGINS
    )
    attributed = bool(
        mechanics
        and never_worse
        and stress_gap >= MINIMUM_STRESS_GAP
        and earlier
    )
    if attributed:
        return "qfcr_geometry_attributed", mechanics, attributed
    if mechanics:
        return (
            "step_scale_sufficient_qfcr_not_attributed",
            mechanics,
            attributed,
        )
    return "qfcr_mechanics_no_go", mechanics, attributed


def audit(
    reservation: OutputReservation,
    *,
    authorization_path: Path,
    beacon_snapshot_path: Path,
) -> dict[str, object]:
    if not _reservation_valid(reservation):
        raise QFCRAuditError("audit lacks a valid output reservation")
    _configure_deterministic_runtime()
    environment_receipt = _environment_receipt()
    source_paths = _runtime_source_paths()
    commit, source_rows, source_match = _source_receipt(
        source_paths
    )
    if not source_match:
        raise QFCRAuditError(
            "source binding failed before board generation"
        )
    seed, entropy_receipt = _authorization_receipt(
        head=commit,
        environment_sha256=sha256(
            _canonical_json_bytes(environment_receipt)
        ).hexdigest(),
        authorization_path=authorization_path,
        beacon_snapshot_path=beacon_snapshot_path,
    )
    (
        entries,
        board_manifest,
        eligible_by_split,
        observed_split_counts,
    ) = _fresh_board(seed)
    world_count = len(board_manifest)
    eligible_world_count = sum(
        int(row["fault_count"] > 0) for row in board_manifest
    )
    fault_count = len(entries)
    counts_pass = bool(
        dict(DEFAULT_COUNTS)
        == {
            "confirmation": 24,
            "development": 32,
            "mechanics": 48,
            "train": 96,
        }
        and observed_split_counts == dict(DEFAULT_COUNTS)
        and world_count == 200
        and eligible_world_count >= MINIMUM_ELIGIBLE_WORLDS
        and fault_count >= MINIMUM_FAULTS
        and all(value > 0 for value in eligible_by_split.values())
    )
    if not counts_pass:
        raise QFCRAuditError(
            "source or pre-outcome board binding failed"
        )
    recoded = _recoded_entries(entries)
    fault_manifest = [
        {
            "world_id": entry.world_id,
            "split": entry.split,
            "action": entry.fault.action,
            "state": entry.fault.state,
            "wrong": entry.fault.wrong,
            "correct": entry.machine.transitions[
                entry.fault.action
            ][entry.fault.state],
        }
        for entry in entries
    ]
    fault_manifest_sha = sha256(
        _canonical_json_bytes(fault_manifest)
    ).hexdigest()
    recoding_manifest = []
    for world_id in sorted(
        {entry.world_id for entry in entries}
    ):
        state, action, observer_key, answer = _permutations(world_id)
        recoding_manifest.append(
            {
                "world_id": world_id,
                "state": state,
                "action": action,
                "observer": observer_key,
                "answer": answer,
            }
        )
    recoding_manifest_sha = sha256(
        _canonical_json_bytes(recoding_manifest)
    ).hexdigest()
    evidence = {}
    margin_receipts = []
    evidence_identities = []
    for margin in MARGINS:
        margin_evidence = {}
        arm_receipts = {}
        for arm in ARMS:
            original = _run_arm(
                entries,
                margin=margin,
                arm=arm,
            )
            recoded_result = _run_arm(
                recoded,
                margin=margin,
                arm=arm,
            )
            margin_evidence[arm.name] = {
                "original": original,
                "recoded": recoded_result,
            }
            arm_receipts[arm.name] = _arm_receipt(
                original,
                recoded_result,
                entries,
            )
            evidence_identities.extend(
                f"{entry.world_id}:{entry.fault.action}:"
                f"{entry.fault.state}:{entry.fault.wrong}:"
                f"{margin:.2f}:{arm.name}:{recoding}:{cycle}"
                for entry in entries
                for recoding in ("original", "recoded")
                for cycle in range(CYCLES + 1)
            )
        evidence[f"{margin:.2f}"] = margin_evidence
        margin_receipts.append(
            {
                "margin": margin,
                "arms": arm_receipts,
            }
        )
    if len(evidence_identities) != len(set(evidence_identities)):
        raise QFCRAuditError("evidence identity set is duplicated")
    expected_identity_count = (
        len(entries)
        * len(MARGINS)
        * len(ARMS)
        * 2
        * (CYCLES + 1)
    )
    if len(evidence_identities) != expected_identity_count:
        raise QFCRAuditError("evidence identity count differs")
    evidence_sha = sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    board_sha = sha256(
        _canonical_json_bytes(board_manifest)
    ).hexdigest()
    identity_sha = sha256(
        _canonical_json_bytes(sorted(evidence_identities))
    ).hexdigest()
    post_environment_receipt = _environment_receipt()
    environment_match = (
        post_environment_receipt == environment_receipt
    )
    post_source_paths = _runtime_source_paths()
    (
        post_commit,
        post_source_rows,
        post_source_match,
    ) = _source_receipt(source_paths)
    source_closure_match = bool(
        post_source_paths == source_paths
        and post_commit == commit
        and post_source_match
        and post_source_rows == source_rows
    )
    bindings_pass = bool(
        source_match
        and source_closure_match
        and environment_match
        and counts_pass
        and _reservation_valid(reservation)
    )
    decision, mechanics, attributed = _decision(
        margin_receipts,
        bindings_pass=bindings_pass,
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "decision": decision,
        "mechanics_pass": mechanics,
        "geometry_attributed": attributed,
        "source": {
            "git_commit": commit,
            "files": source_rows,
            "matches_head": source_match,
            "runtime_closure": source_paths,
            "post_runtime_closure": post_source_paths,
            "post_matches_head": post_source_match,
            "closure_match": source_closure_match,
        },
        "external_entropy": entropy_receipt,
        "environment": {
            "pre": environment_receipt,
            "pre_sha256": sha256(
                _canonical_json_bytes(environment_receipt)
            ).hexdigest(),
            "post": post_environment_receipt,
            "post_sha256": sha256(
                _canonical_json_bytes(post_environment_receipt)
            ).hexdigest(),
            "match": environment_match,
        },
        "board": {
            "seed": seed,
            "counts": DEFAULT_COUNTS,
            "observed_split_counts": observed_split_counts,
            "world_count": world_count,
            "eligible_world_count": eligible_world_count,
            "fault_count": fault_count,
            "eligible_worlds_by_split": eligible_by_split,
            "manifest_sha256": board_sha,
            "manifest": board_manifest,
            "fault_manifest_sha256": fault_manifest_sha,
            "fault_manifest": fault_manifest,
            "counts_pass": counts_pass,
        },
        "recoding": {
            "manifest_sha256": recoding_manifest_sha,
            "manifest": recoding_manifest,
        },
        "configuration": {
            "margins": MARGINS,
            "cycles": CYCLES,
            "arms": [asdict(arm) for arm in ARMS],
        },
        "margin_receipts": margin_receipts,
        "evidence_identity_count": len(evidence_identities),
        "expected_evidence_identity_count": expected_identity_count,
        "evidence_identity_sha256": identity_sha,
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
        "bindings_pass": bindings_pass,
        "output_reservation": {
            "path": str(reservation.output),
            "lock_path": str(reservation.lock),
            "lock_sha256": reservation.lock_sha256,
            "device": reservation.device,
            "inode": reservation.inode,
        },
    }
    report["payload_sha256"] = sha256(
        _canonical_json_bytes(report)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--environment-output", type=Path)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / AUTHORIZATION_FILE,
    )
    parser.add_argument(
        "--beacon-snapshot",
        type=Path,
    )
    args = parser.parse_args()
    if args.environment_output is not None:
        if args.output is not None or args.beacon_snapshot is not None:
            raise QFCRAuditError(
                "environment receipt mode cannot run an audit"
            )
        _configure_deterministic_runtime()
        receipt = _environment_receipt()
        payload = {
            "schema": "efc-qfcr-environment-receipt/v1",
            "environment_sha256": sha256(
                _canonical_json_bytes(receipt)
            ).hexdigest(),
            "environment": receipt,
        }
        encoded = _canonical_json_bytes(payload) + b"\n"
        output = args.environment_output.resolve()
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(payload["environment_sha256"])
        return
    if args.output is None or args.beacon_snapshot is None:
        raise QFCRAuditError(
            "official audit requires output and beacon snapshot"
        )
    reservation = _reserve_output(args.output)
    report = audit(
        reservation,
        authorization_path=args.authorization,
        beacon_snapshot_path=args.beacon_snapshot,
    )
    _publish(reservation, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "mechanics_pass": report["mechanics_pass"],
                "geometry_attributed": report[
                    "geometry_attributed"
                ],
                "payload_sha256": report["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
