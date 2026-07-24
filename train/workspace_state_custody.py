"""Strict serialization for source-deleted causal workspace states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import torch

from causal_bind_select_workspace import (
    CausalWorkspaceGPT,
    WorkspaceState,
)
from workspace_checkpoint import (
    ProtectedCheckpointReceipt,
    file_sha256,
    protected_checkpoint_reference,
    runtime_source_manifest,
    state_dict_sha256,
)


COMPILED_STATE_SCHEMA = "shohin_compiled_workspace_states_v1"
COMPILER_SOURCE_RECEIPT_SCHEMA = "shohin_episode_workspace_compiler_source_receipt_v1"
LANDLOCK_RECEIPT_SCHEMA = "shohin_landlock_stage_receipt_v1"
DENIED_PROBE_RECEIPT_SCHEMA = "shohin_landlock_denied_probe_receipt_v1"
COMPILER_STAGE = "compiler"
COMPILER_PRIMARY_PATH = "train/compile_episode_causal_workspace.py"
COMPILER_DENIED_PATH_NAME = "development_queries.jsonl"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPILER_DEPENDENCY_PATHS = (
    "pipeline/episode_workspace_custody.py",
    "train/workspace_state_custody.py",
    "train/causal_bind_select_workspace.py",
    "train/model.py",
    "train/workspace_checkpoint.py",
)
COMPILER_SOURCE_PATHS = (COMPILER_PRIMARY_PATH, *COMPILER_DEPENDENCY_PATHS)
EXPECTED_WORLD_COUNT = 192
EXPECTED_TOKEN_POSITION = 145
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class WorkspaceStateCustodyError(ValueError):
    """A compiled-state shape, source, hash, or custody invariant failed."""


def save_compiled_states(
    path: Path,
    states: Mapping[str, WorkspaceState],
    *,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
    workspace_delta_sha256: str,
    world_source_sha256: str,
    compiler_source_receipt: Mapping[str, object],
) -> str:
    """Atomically serialize only allowlisted workspace slots and receipts."""

    validated_compiler_receipt = _validate_compiler_source_receipt(
        compiler_source_receipt
    )
    if len(states) != EXPECTED_WORLD_COUNT:
        raise WorkspaceStateCustodyError("compiled world cardinality drifted")
    tensors: dict[str, torch.Tensor] = {}
    for world_id, state in sorted(states.items()):
        if len(world_id) != 64:
            raise WorkspaceStateCustodyError("compiled world ID is invalid")
        if state.token_position != EXPECTED_TOKEN_POSITION or not state.sealed:
            raise WorkspaceStateCustodyError("compiled workspace is not sealed at 145")
        slots = state.slots.detach().to(device="cpu", copy=True).contiguous()
        if slots.shape != (
            1,
            model.workspace_config.num_slots,
            model.workspace_config.slot_width,
        ):
            raise WorkspaceStateCustodyError("compiled slot tensor shape drifted")
        if not slots.is_floating_point() or not torch.isfinite(slots).all():
            raise WorkspaceStateCustodyError("compiled slots are invalid")
        tensors[world_id] = slots.squeeze(0)
    tensor_sha256 = state_dict_sha256(tensors)
    payload = {
        "schema": COMPILED_STATE_SCHEMA,
        "protected_base": protected_checkpoint_reference(protected_receipt),
        "workspace_delta_sha256": workspace_delta_sha256,
        "world_source_sha256": world_source_sha256,
        "runtime_source_manifest": runtime_source_manifest(),
        "compiler_source_receipt": validated_compiler_receipt,
        "workspace_config": asdict(model.workspace_config),
        "token_position": EXPECTED_TOKEN_POSITION,
        "world_count": EXPECTED_WORLD_COUNT,
        "state_tensor_sha256": tensor_sha256,
        "states": tensors,
        "source_tokens_serialized": False,
        "query_tokens_seen": False,
        "labels_seen": False,
        "optimizer_state": None,
        "pretraining_started": False,
        "continuation_pretraining_authorized": False,
    }
    if (
        _validate_compiler_source_receipt(validated_compiler_receipt)
        != validated_compiler_receipt
    ):
        raise WorkspaceStateCustodyError(
            "compiler source receipt changed before serialization"
        )
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def load_compiled_states(
    path: Path,
    *,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
    expected_sha256: str,
    expected_workspace_delta_sha256: str,
    expected_world_source_sha256: str,
    expected_compiler_source_sha256: str,
    expected_repository_commit: str,
) -> tuple[dict[str, WorkspaceState], dict[str, object]]:
    """Load and validate a source-free state payload from one verified handle."""

    path = path.absolute()
    with path.open("rb") as handle:
        digest = _handle_sha256(handle)
        if digest != expected_sha256:
            raise WorkspaceStateCustodyError(
                f"compiled-state hash mismatch: {digest}, expected {expected_sha256}"
            )
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != COMPILED_STATE_SCHEMA:
        raise WorkspaceStateCustodyError("compiled-state schema is invalid")
    expected_keys = {
        "schema",
        "protected_base",
        "workspace_delta_sha256",
        "world_source_sha256",
        "runtime_source_manifest",
        "compiler_source_receipt",
        "workspace_config",
        "token_position",
        "world_count",
        "state_tensor_sha256",
        "states",
        "source_tokens_serialized",
        "query_tokens_seen",
        "labels_seen",
        "optimizer_state",
        "pretraining_started",
        "continuation_pretraining_authorized",
    }
    if set(payload) != expected_keys:
        raise WorkspaceStateCustodyError("compiled-state fields differ")
    if payload["protected_base"] != protected_checkpoint_reference(protected_receipt):
        raise WorkspaceStateCustodyError("compiled state references another base")
    if payload["workspace_delta_sha256"] != expected_workspace_delta_sha256:
        raise WorkspaceStateCustodyError("compiled state references another delta")
    if payload["world_source_sha256"] != expected_world_source_sha256:
        raise WorkspaceStateCustodyError("compiled state references another world set")
    if payload["runtime_source_manifest"] != runtime_source_manifest():
        raise WorkspaceStateCustodyError("compiled-state implementation differs")
    compiler_source_receipt = _validate_compiler_source_receipt(
        payload["compiler_source_receipt"],
        expected_primary_sha256=expected_compiler_source_sha256,
        expected_repository_commit=expected_repository_commit,
    )
    if payload["workspace_config"] != asdict(model.workspace_config):
        raise WorkspaceStateCustodyError("compiled-state configuration differs")
    if (
        payload["token_position"] != EXPECTED_TOKEN_POSITION
        or payload["world_count"] != EXPECTED_WORLD_COUNT
    ):
        raise WorkspaceStateCustodyError("compiled-state dimensions drifted")
    if (
        payload["source_tokens_serialized"] is not False
        or payload["query_tokens_seen"] is not False
        or payload["labels_seen"] is not False
        or payload["optimizer_state"] is not None
        or payload["pretraining_started"] is not False
        or payload["continuation_pretraining_authorized"] is not False
    ):
        raise WorkspaceStateCustodyError("compiled-state custody flags are invalid")
    raw_states = payload["states"]
    if not isinstance(raw_states, Mapping) or len(raw_states) != EXPECTED_WORLD_COUNT:
        raise WorkspaceStateCustodyError("compiled state mapping is invalid")
    tensors: dict[str, torch.Tensor] = {}
    states: dict[str, WorkspaceState] = {}
    for world_id, raw in sorted(raw_states.items()):
        if not isinstance(world_id, str) or len(world_id) != 64:
            raise WorkspaceStateCustodyError("compiled world ID is invalid")
        if not isinstance(raw, torch.Tensor):
            raise WorkspaceStateCustodyError("compiled slot value is not a tensor")
        tensor = raw.detach().to(device="cpu", copy=True).contiguous()
        if tensor.shape != (
            model.workspace_config.num_slots,
            model.workspace_config.slot_width,
        ):
            raise WorkspaceStateCustodyError("compiled slot tensor shape drifted")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise WorkspaceStateCustodyError("compiled slot tensor is invalid")
        tensors[world_id] = tensor
        states[world_id] = WorkspaceState(
            slots=tensor.unsqueeze(0),
            token_position=EXPECTED_TOKEN_POSITION,
            sealed=True,
        )
    if state_dict_sha256(tensors) != payload["state_tensor_sha256"]:
        raise WorkspaceStateCustodyError("compiled state tensor digest drifted")
    receipt = {key: value for key, value in payload.items() if key != "states"}
    receipt["compiler_source_receipt"] = compiler_source_receipt
    return states, receipt


def _validate_compiler_source_receipt(
    value: object,
    *,
    expected_primary_sha256: str | None = None,
    expected_repository_commit: str | None = None,
) -> dict[str, object]:
    receipt = _strict_dict(
        value,
        {
            "schema",
            "primary_path",
            "primary_sha256",
            "repository_commit",
            "local_source_manifest",
            "runtime_source_manifest",
            "process_id",
            "landlock_receipt",
        },
        "compiler source receipt",
    )
    if receipt["schema"] != COMPILER_SOURCE_RECEIPT_SCHEMA:
        raise WorkspaceStateCustodyError("compiler source receipt schema is invalid")
    if receipt["primary_path"] != COMPILER_PRIMARY_PATH:
        raise WorkspaceStateCustodyError("compiler primary path is invalid")

    primary_sha256 = _strict_hex(
        receipt["primary_sha256"],
        _SHA256_PATTERN,
        "compiler primary SHA-256",
    )
    repository_commit = _strict_hex(
        receipt["repository_commit"],
        _GIT_COMMIT_PATTERN,
        "compiler repository commit",
    )
    if expected_primary_sha256 is not None:
        expected_primary_sha256 = _strict_hex(
            expected_primary_sha256,
            _SHA256_PATTERN,
            "expected compiler source SHA-256",
        )
        if primary_sha256 != expected_primary_sha256:
            raise WorkspaceStateCustodyError(
                "compiled state references another compiler source"
            )
    if expected_repository_commit is not None:
        expected_repository_commit = _strict_hex(
            expected_repository_commit,
            _GIT_COMMIT_PATTERN,
            "expected repository commit",
        )
        if repository_commit != expected_repository_commit:
            raise WorkspaceStateCustodyError(
                "compiled state references another repository commit"
            )

    local_commit = _local_repository_commit()
    if repository_commit != local_commit:
        raise WorkspaceStateCustodyError(
            "compiler repository commit differs from the local checkout"
        )

    source_manifest = _strict_dict(
        receipt["local_source_manifest"],
        set(COMPILER_SOURCE_PATHS),
        "compiler local source manifest",
    )
    canonical_source_manifest: dict[str, str] = {}
    for relative_path in COMPILER_SOURCE_PATHS:
        claimed = _strict_hex(
            source_manifest[relative_path],
            _SHA256_PATTERN,
            f"compiler dependency SHA-256 for {relative_path}",
        )
        actual = file_sha256(REPOSITORY_ROOT / relative_path)
        if claimed != actual:
            raise WorkspaceStateCustodyError(
                f"compiler dependency hash mismatch for {relative_path}"
            )
        canonical_source_manifest[relative_path] = claimed
    if canonical_source_manifest[COMPILER_PRIMARY_PATH] != primary_sha256:
        raise WorkspaceStateCustodyError(
            "compiler primary SHA-256 differs from its source manifest"
        )

    expected_runtime_manifest = runtime_source_manifest()
    runtime_manifest = _strict_dict(
        receipt["runtime_source_manifest"],
        set(expected_runtime_manifest),
        "compiler runtime source manifest",
    )
    canonical_runtime_manifest: dict[str, str] = {}
    for name, expected in expected_runtime_manifest.items():
        claimed = _strict_hex(
            runtime_manifest[name],
            _SHA256_PATTERN,
            f"compiler runtime source SHA-256 for {name}",
        )
        if claimed != expected:
            raise WorkspaceStateCustodyError(
                f"compiler runtime source hash mismatch for {name}"
            )
        dependency_path = f"train/{name}"
        if canonical_source_manifest.get(dependency_path) != claimed:
            raise WorkspaceStateCustodyError(
                f"compiler source manifests disagree for {dependency_path}"
            )
        canonical_runtime_manifest[name] = claimed

    process_id = _strict_positive_integer(
        receipt["process_id"],
        "compiler process ID",
    )
    landlock_receipt = _validate_landlock_receipt(
        receipt["landlock_receipt"],
        process_id=process_id,
    )
    return {
        "schema": COMPILER_SOURCE_RECEIPT_SCHEMA,
        "primary_path": COMPILER_PRIMARY_PATH,
        "primary_sha256": primary_sha256,
        "repository_commit": repository_commit,
        "local_source_manifest": canonical_source_manifest,
        "runtime_source_manifest": canonical_runtime_manifest,
        "process_id": process_id,
        "landlock_receipt": landlock_receipt,
    }


def _validate_landlock_receipt(
    value: object,
    *,
    process_id: int,
) -> dict[str, object]:
    receipt = _strict_dict(
        value,
        {
            "schema",
            "stage",
            "enforced",
            "dumpable",
            "abi",
            "policy_sha256",
            "canonical_policy",
            "process_id",
            "denied_probe_receipt",
        },
        "Landlock receipt",
    )
    if receipt["schema"] != LANDLOCK_RECEIPT_SCHEMA:
        raise WorkspaceStateCustodyError("Landlock receipt schema is invalid")
    if receipt["stage"] != COMPILER_STAGE:
        raise WorkspaceStateCustodyError("Landlock receipt stage is not compiler")
    if receipt["enforced"] is not True:
        raise WorkspaceStateCustodyError("Landlock was not enforced")
    if receipt["dumpable"] is not False:
        raise WorkspaceStateCustodyError("compiler process was dumpable")
    abi = _strict_positive_integer(receipt["abi"], "Landlock ABI")
    policy_sha256 = _strict_hex(
        receipt["policy_sha256"],
        _SHA256_PATTERN,
        "Landlock policy SHA-256",
    )
    canonical_policy = _validate_canonical_policy(
        receipt["canonical_policy"],
        abi=abi,
        policy_sha256=policy_sha256,
    )
    landlock_process_id = _strict_positive_integer(
        receipt["process_id"],
        "Landlock process ID",
    )
    if landlock_process_id != process_id:
        raise WorkspaceStateCustodyError(
            "Landlock process ID differs from the compiler process"
        )
    denied_probe = _validate_denied_probe_receipt(
        receipt["denied_probe_receipt"],
        process_id=process_id,
    )
    return {
        "schema": LANDLOCK_RECEIPT_SCHEMA,
        "stage": COMPILER_STAGE,
        "enforced": True,
        "dumpable": False,
        "abi": abi,
        "policy_sha256": policy_sha256,
        "canonical_policy": canonical_policy,
        "process_id": process_id,
        "denied_probe_receipt": denied_probe,
    }


def _validate_denied_probe_receipt(
    value: object,
    *,
    process_id: int,
) -> dict[str, object]:
    receipt = _strict_dict(
        value,
        {
            "schema",
            "stage",
            "process_id",
            "operation",
            "path",
            "path_name",
            "path_sha256",
            "denied",
            "errno",
        },
        "Landlock denied-probe receipt",
    )
    if receipt["schema"] != DENIED_PROBE_RECEIPT_SCHEMA:
        raise WorkspaceStateCustodyError(
            "Landlock denied-probe receipt schema is invalid"
        )
    if receipt["stage"] != COMPILER_STAGE:
        raise WorkspaceStateCustodyError("Landlock denied-probe stage is not compiler")
    probe_process_id = _strict_positive_integer(
        receipt["process_id"],
        "Landlock denied-probe process ID",
    )
    if probe_process_id != process_id:
        raise WorkspaceStateCustodyError(
            "Landlock denied-probe process ID differs from the compiler process"
        )
    if receipt["operation"] != "open_read":
        raise WorkspaceStateCustodyError("Landlock denied-probe operation is invalid")
    path = receipt["path"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or receipt["path_name"] != COMPILER_DENIED_PATH_NAME
        or Path(path).name != COMPILER_DENIED_PATH_NAME
    ):
        raise WorkspaceStateCustodyError(
            "Landlock denied-probe path is not the development query source"
        )
    path_sha256 = _strict_hex(
        receipt["path_sha256"],
        _SHA256_PATTERN,
        "Landlock denied-probe path SHA-256",
    )
    if hashlib.sha256(os.fsencode(path)).hexdigest() != path_sha256:
        raise WorkspaceStateCustodyError(
            "Landlock denied-probe path hash differs"
        )
    if receipt["denied"] is not True:
        raise WorkspaceStateCustodyError("Landlock denied probe did not succeed")
    error_number = receipt["errno"]
    if (
        not isinstance(error_number, int)
        or isinstance(error_number, bool)
        or error_number not in {errno.EACCES, errno.EPERM}
    ):
        raise WorkspaceStateCustodyError(
            "Landlock denied-probe errno is not an access denial"
        )
    return {
        "schema": DENIED_PROBE_RECEIPT_SCHEMA,
        "stage": COMPILER_STAGE,
        "process_id": process_id,
        "operation": "open_read",
        "path": path,
        "path_name": COMPILER_DENIED_PATH_NAME,
        "path_sha256": path_sha256,
        "denied": True,
        "errno": error_number,
    }


def _validate_canonical_policy(
    value: object,
    *,
    abi: int,
    policy_sha256: str,
) -> dict[str, object]:
    policy = _strict_dict(
        value,
        {
            "schema",
            "landlock_abi",
            "stage",
            "handled_access_fs",
            "handled_access_fs_names",
            "rules",
        },
        "Landlock canonical policy",
    )
    if (
        policy["schema"] != "shohin_landlock_stage_policy_v1"
        or policy["landlock_abi"] != abi
        or policy["stage"] != COMPILER_STAGE
        or not isinstance(policy["handled_access_fs"], int)
        or isinstance(policy["handled_access_fs"], bool)
        or not isinstance(policy["handled_access_fs_names"], list)
        or not isinstance(policy["rules"], list)
    ):
        raise WorkspaceStateCustodyError("Landlock canonical policy is invalid")
    encoded = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != policy_sha256:
        raise WorkspaceStateCustodyError(
            "Landlock canonical policy hash differs"
        )
    return policy


def _strict_dict(
    value: object,
    expected_keys: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected_keys:
        raise WorkspaceStateCustodyError(f"{label} fields differ")
    return value


def _strict_hex(
    value: object,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise WorkspaceStateCustodyError(f"{label} is invalid")
    return value


def _strict_positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise WorkspaceStateCustodyError(f"{label} must be a positive integer")
    return value


def _local_repository_commit() -> str:
    if os.environ.get("SHOHIN_LANDLOCK_ENFORCED") == "1":
        return _strict_hex(
            os.environ.get("SHOHIN_REPOSITORY_COMMIT"),
            _GIT_COMMIT_PATTERN,
            "sealed repository commit",
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkspaceStateCustodyError(
            "local repository commit is unavailable"
        ) from exc
    return _strict_hex(commit, _GIT_COMMIT_PATTERN, "local repository commit")


def _handle_sha256(handle) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
