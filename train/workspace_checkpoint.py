"""Strict protected-base upgrade and delta custody for the causal workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from causal_bind_select_workspace import (
    PROTECTED_BASE_PARAMETERS,
    CausalBindSelectWorkspace,
    CausalWorkspaceConfig,
    CausalWorkspaceGPT,
    WorkspaceContractError,
    WorkspaceParameterReceipt,
    freeze_protected_base,
)
from model import GPT, GPTConfig


PROTECTED_CHECKPOINT_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
PROTECTED_CHECKPOINT_STEP = 300_000
PROTECTED_CHECKPOINT_BYTES = 500_448_522
PROTECTED_DATA_SEED = 777
PROTECTED_DATA_STREAM_GENERATION = 1
PROTECTED_DATA_STREAM_SEED = 1_000_780
PROTECTED_CONFIG_SHA256 = (
    "8d8e45f3abee8f124b3d8b03ea83169497e03e95f40653d13436d8a54765119a"
)
PROTECTED_BASE_STATE_SHA256 = (
    "321356c4940a7a27f7385ea304557dc5575b6d4d188504e8ce204eb24211abab"
)
PROTECTED_STATE_KEY_SHA256 = (
    "82571f8cf50a436dd5014e12b02606d11ca4333e5cc1961236d3e68a89e642bf"
)
PROTECTED_STATE_KEY_COUNT = 333
PROTECTED_BASE_CONFIG = {
    "vocab_size": 32768,
    "n_layer": 30,
    "n_head": 9,
    "n_kv_head": 3,
    "d_model": 576,
    "d_ff": 1536,
    "seq_len": 2048,
    "rope_theta": 50_000.0,
    "qk_norm": True,
    "tie_embeddings": True,
    "zloss": 0.0001,
    "n_loop": 1,
}
PROTECTED_CHECKPOINT_KEYS = frozenset(
    {
        "model",
        "cfg",
        "step",
        "data_seed",
        "data_stream_generation",
        "data_stream_seed",
    }
)
DELTA_SCHEMA = "shohin_causal_workspace_delta_v1"
WORKSPACE_SOURCE_PATH = Path(__file__).with_name("causal_bind_select_workspace.py")
MODEL_SOURCE_PATH = Path(__file__).with_name("model.py")
CHECKPOINT_SOURCE_PATH = Path(__file__)


def _source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMPORTED_RUNTIME_SOURCE_MANIFEST = {
    "causal_bind_select_workspace.py": _source_file_sha256(WORKSPACE_SOURCE_PATH),
    "model.py": _source_file_sha256(MODEL_SOURCE_PATH),
    "workspace_checkpoint.py": _source_file_sha256(CHECKPOINT_SOURCE_PATH),
}


class WorkspaceCheckpointError(ValueError):
    """A protected-base, delta, or receipt invariant failed."""


@dataclass(frozen=True)
class ProtectedCheckpointReceipt:
    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    step: int
    data_seed: int
    data_stream_generation: int
    data_stream_seed: int
    base_config: dict[str, object]
    config_sha256: str
    base_state_sha256: str
    state_key_sha256: str
    state_key_count: int
    protected_base_parameters: int
    workspace_parameters: int
    complete_system_parameters: int
    remaining_under_cap: int
    strict_state_load: bool
    protected_base_frozen: bool
    optimizer_state_loaded: bool
    pretraining_started: bool


def load_protected_workspace_model(
    checkpoint_path: Path,
    workspace_config: CausalWorkspaceConfig,
) -> tuple[CausalWorkspaceGPT, ProtectedCheckpointReceipt]:
    """Load only the immutable production step-300k trust root."""

    return _load_workspace_model(
        checkpoint_path,
        workspace_config,
        expected_sha256=PROTECTED_CHECKPOINT_SHA256,
        expected_step=PROTECTED_CHECKPOINT_STEP,
        expected_base_parameters=PROTECTED_BASE_PARAMETERS,
        expected_base_state_sha256=PROTECTED_BASE_STATE_SHA256,
    )


def _load_workspace_model_for_test(
    checkpoint_path: Path,
    workspace_config: CausalWorkspaceConfig,
    *,
    expected_sha256: str,
    expected_step: int,
    expected_base_parameters: int,
) -> tuple[CausalWorkspaceGPT, ProtectedCheckpointReceipt]:
    """Test-only loader for synthetic checkpoints with explicit expectations."""

    return _load_workspace_model(
        checkpoint_path,
        workspace_config,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
        expected_base_parameters=expected_base_parameters,
        expected_base_state_sha256=None,
    )


def _load_workspace_model(
    checkpoint_path: Path,
    workspace_config: CausalWorkspaceConfig,
    *,
    expected_sha256: str,
    expected_step: int,
    expected_base_parameters: int,
    expected_base_state_sha256: str | None,
) -> tuple[CausalWorkspaceGPT, ProtectedCheckpointReceipt]:
    """Hash, strictly load, freeze, and wrap one model-only checkpoint."""

    checkpoint_path = checkpoint_path.resolve()
    checkpoint, actual_sha256 = _torch_load_verified(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(checkpoint, dict):
        raise WorkspaceCheckpointError("protected checkpoint is not a dictionary")
    keys = frozenset(checkpoint)
    if keys != PROTECTED_CHECKPOINT_KEYS:
        missing = sorted(PROTECTED_CHECKPOINT_KEYS - keys)
        unexpected = sorted(keys - PROTECTED_CHECKPOINT_KEYS)
        raise WorkspaceCheckpointError(
            f"protected checkpoint keys differ; missing={missing}, unexpected={unexpected}"
        )
    if checkpoint["step"] != expected_step:
        raise WorkspaceCheckpointError(
            f"protected checkpoint step is {checkpoint['step']}, expected {expected_step}"
        )
    config_payload = checkpoint["cfg"]
    if not isinstance(config_payload, dict):
        raise WorkspaceCheckpointError("protected checkpoint cfg is not a dictionary")
    try:
        base_config = GPTConfig(**config_payload)
    except TypeError as exc:
        raise WorkspaceCheckpointError("protected checkpoint cfg is invalid") from exc
    if base_config.n_loop != 1:
        raise WorkspaceCheckpointError("protected base must have n_loop=1")

    model_state = checkpoint["model"]
    if not isinstance(model_state, Mapping) or not model_state:
        raise WorkspaceCheckpointError("protected model state is missing or empty")
    base = GPT(base_config)
    try:
        incompatibility = base.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise WorkspaceCheckpointError(
            "protected model state failed strict loading"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise WorkspaceCheckpointError("strict load returned incompatible keys")
    if base.num_params() != expected_base_parameters:
        raise WorkspaceCheckpointError(
            f"protected base has {base.num_params()} parameters, "
            f"expected {expected_base_parameters}"
        )
    base_state_sha256 = state_dict_sha256(base.state_dict())
    if (
        expected_base_state_sha256 is not None
        and base_state_sha256 != expected_base_state_sha256
    ):
        raise WorkspaceCheckpointError(
            f"protected base tensor hash mismatch: {base_state_sha256}"
        )

    wrapper = CausalWorkspaceGPT(base, workspace_config)
    freeze_protected_base(wrapper)
    parameter_receipt = wrapper.workspace.parameter_receipt(
        protected_base_parameters=expected_base_parameters
    )
    receipt = ProtectedCheckpointReceipt(
        checkpoint_path=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_sha256=actual_sha256,
        step=checkpoint["step"],
        data_seed=checkpoint["data_seed"],
        data_stream_generation=checkpoint["data_stream_generation"],
        data_stream_seed=checkpoint["data_stream_seed"],
        base_config=dict(config_payload),
        config_sha256=json_sha256(config_payload),
        base_state_sha256=base_state_sha256,
        state_key_sha256=json_sha256(sorted(model_state)),
        state_key_count=len(model_state),
        protected_base_parameters=parameter_receipt.protected_base_parameters,
        workspace_parameters=parameter_receipt.workspace_parameters,
        complete_system_parameters=parameter_receipt.complete_system_parameters,
        remaining_under_cap=parameter_receipt.remaining_under_cap,
        strict_state_load=True,
        protected_base_frozen=not any(
            parameter.requires_grad for parameter in wrapper.base.parameters()
        ),
        optimizer_state_loaded=False,
        pretraining_started=False,
    )
    return wrapper, receipt


def workspace_delta_payload(
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> dict[str, object]:
    """Build a production delta bound to the immutable step-300k trust root."""

    _validate_protected_receipt_constants(protected_receipt)
    return _workspace_delta_payload(model, protected_receipt)


def _workspace_delta_payload_for_test(
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> dict[str, object]:
    return _workspace_delta_payload(model, protected_receipt)


def _workspace_delta_payload(
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> dict[str, object]:
    """Build a workspace-only payload that references rather than copies the base."""

    parameter_receipt = _validate_model_bound_to_receipt(model, protected_receipt)
    workspace_state = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in model.workspace.state_dict().items()
    }
    return {
        "schema": DELTA_SCHEMA,
        "protected_base": protected_checkpoint_reference(protected_receipt),
        "runtime_source_manifest": runtime_source_manifest(),
        "workspace_config": asdict(model.workspace_config),
        "parameter_receipt": asdict(parameter_receipt),
        "workspace_state": workspace_state,
        "optimizer_state": None,
        "pretraining_started": False,
    }


def save_workspace_delta(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> str:
    """Atomically publish a production delta without replacing an existing file."""

    _validate_protected_receipt_constants(protected_receipt)
    return _save_workspace_delta(path, model, protected_receipt)


def _save_workspace_delta_for_test(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> str:
    return _save_workspace_delta(path, model, protected_receipt)


def _save_workspace_delta(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> str:
    """Atomically save only workspace tensors and return the file SHA-256."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _workspace_delta_payload(model, protected_receipt)
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
        _publish_noreplace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def load_workspace_delta(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
    *,
    expected_sha256: str,
) -> None:
    """Load a production delta only against the immutable step-300k trust root."""

    _validate_protected_receipt_constants(protected_receipt)
    _load_workspace_delta(
        path,
        model,
        protected_receipt,
        expected_sha256=expected_sha256,
    )


def _load_workspace_delta_for_test(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
    *,
    expected_sha256: str,
) -> None:
    _load_workspace_delta(
        path,
        model,
        protected_receipt,
        expected_sha256=expected_sha256,
    )


def _load_workspace_delta(
    path: Path,
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
    *,
    expected_sha256: str,
) -> None:
    """Strictly load a workspace delta only if its protected reference matches."""

    path = path.resolve()
    _validate_model_bound_to_receipt(model, protected_receipt)
    payload, _ = _torch_load_verified(
        path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(payload, dict) or payload.get("schema") != DELTA_SCHEMA:
        raise WorkspaceCheckpointError("workspace delta schema is invalid")
    if payload.get("optimizer_state") is not None:
        raise WorkspaceCheckpointError(
            "workspace delta must not contain optimizer state"
        )
    if payload.get("pretraining_started") is not False:
        raise WorkspaceCheckpointError("workspace delta pretraining flag is invalid")
    if payload.get("runtime_source_manifest") != runtime_source_manifest():
        raise WorkspaceCheckpointError("workspace delta implementation differs")
    protected_payload = payload.get("protected_base")
    if protected_payload != protected_checkpoint_reference(protected_receipt):
        raise WorkspaceCheckpointError(
            "workspace delta references another protected base"
        )
    if payload.get("workspace_config") != asdict(model.workspace_config):
        raise WorkspaceCheckpointError("workspace delta configuration differs")
    expected_parameters = asdict(
        _validate_model_bound_to_receipt(model, protected_receipt)
    )
    if payload.get("parameter_receipt") != expected_parameters:
        raise WorkspaceCheckpointError("workspace delta parameter receipt differs")
    workspace_state = payload.get("workspace_state")
    if not isinstance(workspace_state, Mapping):
        raise WorkspaceCheckpointError("workspace delta state is missing")
    with torch.random.fork_rng(devices=[]):
        temporary = CausalBindSelectWorkspace(model.workspace_config)
    try:
        incompatibility = temporary.load_state_dict(workspace_state, strict=True)
    except RuntimeError as exc:
        raise WorkspaceCheckpointError("workspace delta failed strict loading") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise WorkspaceCheckpointError("workspace delta returned incompatible keys")
    _validate_workspace_state_tensors(temporary.state_dict())
    target_state = model.workspace.state_dict()
    prepared = {
        name: tensor.detach().to(
            device=target_state[name].device,
            dtype=target_state[name].dtype,
            copy=True,
        )
        for name, tensor in temporary.state_dict().items()
    }
    with torch.no_grad():
        for name, tensor in prepared.items():
            target_state[name].copy_(tensor)


def write_architecture_receipt(
    path: Path,
    receipt: ProtectedCheckpointReceipt,
    workspace_config: CausalWorkspaceConfig,
) -> str:
    """Publish a production architecture receipt without model tensors."""

    _validate_protected_receipt_constants(receipt)
    _validate_architecture_receipt(receipt, workspace_config)
    return _write_architecture_receipt(path, receipt, workspace_config)


def _write_architecture_receipt_for_test(
    path: Path,
    receipt: ProtectedCheckpointReceipt,
    workspace_config: CausalWorkspaceConfig,
) -> str:
    return _write_architecture_receipt(path, receipt, workspace_config)


def _write_architecture_receipt(
    path: Path,
    receipt: ProtectedCheckpointReceipt,
    workspace_config: CausalWorkspaceConfig,
) -> str:
    """Write an fsync'd JSON architecture receipt without model tensors."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "shohin_causal_workspace_architecture_receipt_v1",
        "protected_checkpoint": asdict(receipt),
        "runtime_source_manifest": runtime_source_manifest(),
        "workspace_config": asdict(workspace_config),
        "claim_scope": (
            "architecture compatibility and parameter custody only; "
            "no neural training, reasoning, or pretraining claim"
        ),
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_noreplace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise WorkspaceCheckpointError(f"state entry {name} is not a tensor")
        contiguous = tensor.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = memoryview(contiguous.view(torch.uint8).numpy())
        digest.update(raw.nbytes.to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def runtime_source_manifest() -> dict[str, str]:
    current = {
        "causal_bind_select_workspace.py": file_sha256(WORKSPACE_SOURCE_PATH),
        "model.py": file_sha256(MODEL_SOURCE_PATH),
        "workspace_checkpoint.py": file_sha256(CHECKPOINT_SOURCE_PATH),
    }
    if current != _IMPORTED_RUNTIME_SOURCE_MANIFEST:
        raise WorkspaceCheckpointError(
            "runtime source files changed after process import"
        )
    return dict(_IMPORTED_RUNTIME_SOURCE_MANIFEST)


def protected_checkpoint_reference(
    receipt: ProtectedCheckpointReceipt,
) -> dict[str, object]:
    """Return the portable cryptographic identity used by workspace deltas."""

    payload = asdict(receipt)
    payload.pop("checkpoint_path")
    return payload


def _validate_parameter_receipts(
    parameter_receipt: WorkspaceParameterReceipt,
    protected_receipt: ProtectedCheckpointReceipt,
) -> None:
    expected = (
        protected_receipt.protected_base_parameters,
        protected_receipt.workspace_parameters,
        protected_receipt.complete_system_parameters,
        protected_receipt.remaining_under_cap,
    )
    actual = (
        parameter_receipt.protected_base_parameters,
        parameter_receipt.workspace_parameters,
        parameter_receipt.complete_system_parameters,
        parameter_receipt.remaining_under_cap,
    )
    if actual != expected:
        raise WorkspaceContractError("workspace and protected receipts differ")


def _validate_model_bound_to_receipt(
    model: CausalWorkspaceGPT,
    protected_receipt: ProtectedCheckpointReceipt,
) -> WorkspaceParameterReceipt:
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise WorkspaceCheckpointError("protected base is not frozen")
    if not protected_receipt.protected_base_frozen:
        raise WorkspaceCheckpointError("protected receipt does not freeze the base")
    base_config = asdict(model.base.cfg)
    if base_config != protected_receipt.base_config:
        raise WorkspaceCheckpointError("protected base configuration differs")
    if json_sha256(base_config) != protected_receipt.config_sha256:
        raise WorkspaceCheckpointError("protected base configuration hash differs")
    if (
        state_dict_sha256(model.base.state_dict())
        != protected_receipt.base_state_sha256
    ):
        raise WorkspaceCheckpointError("protected base tensor state differs")
    parameter_receipt = model.workspace.parameter_receipt(
        protected_base_parameters=protected_receipt.protected_base_parameters
    )
    _validate_parameter_receipts(parameter_receipt, protected_receipt)
    return parameter_receipt


def _validate_protected_receipt_constants(
    receipt: ProtectedCheckpointReceipt,
) -> None:
    expected = {
        "checkpoint_bytes": PROTECTED_CHECKPOINT_BYTES,
        "checkpoint_sha256": PROTECTED_CHECKPOINT_SHA256,
        "step": PROTECTED_CHECKPOINT_STEP,
        "data_seed": PROTECTED_DATA_SEED,
        "data_stream_generation": PROTECTED_DATA_STREAM_GENERATION,
        "data_stream_seed": PROTECTED_DATA_STREAM_SEED,
        "base_config": PROTECTED_BASE_CONFIG,
        "config_sha256": PROTECTED_CONFIG_SHA256,
        "base_state_sha256": PROTECTED_BASE_STATE_SHA256,
        "state_key_sha256": PROTECTED_STATE_KEY_SHA256,
        "state_key_count": PROTECTED_STATE_KEY_COUNT,
        "protected_base_parameters": PROTECTED_BASE_PARAMETERS,
        "strict_state_load": True,
        "protected_base_frozen": True,
        "optimizer_state_loaded": False,
        "pretraining_started": False,
    }
    payload = asdict(receipt)
    differing = sorted(
        name for name, value in expected.items() if payload.get(name) != value
    )
    if differing:
        raise WorkspaceCheckpointError(
            "receipt is not the protected step-300k trust root: " + ",".join(differing)
        )


def _validate_architecture_receipt(
    receipt: ProtectedCheckpointReceipt,
    workspace_config: CausalWorkspaceConfig,
) -> None:
    try:
        workspace_config.validate(n_layer=int(PROTECTED_BASE_CONFIG["n_layer"]))
        if workspace_config.d_model != int(PROTECTED_BASE_CONFIG["d_model"]):
            raise WorkspaceContractError(
                "workspace d_model must match the protected base"
            )
        with torch.random.fork_rng(devices=[]):
            workspace = CausalBindSelectWorkspace(workspace_config)
        parameter_receipt = workspace.parameter_receipt(
            protected_base_parameters=PROTECTED_BASE_PARAMETERS
        )
    except WorkspaceContractError as exc:
        raise WorkspaceCheckpointError(
            f"workspace architecture is invalid: {exc}"
        ) from exc
    expected = (
        parameter_receipt.workspace_parameters,
        parameter_receipt.complete_system_parameters,
        parameter_receipt.remaining_under_cap,
    )
    actual = (
        receipt.workspace_parameters,
        receipt.complete_system_parameters,
        receipt.remaining_under_cap,
    )
    if actual != expected:
        raise WorkspaceCheckpointError(
            "architecture receipt parameter ledger differs from workspace config"
        )


def _validate_workspace_state_tensors(
    state: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise WorkspaceCheckpointError(f"workspace state {name} is not a tensor")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise WorkspaceCheckpointError(
                f"workspace state {name} contains nonfinite values"
            )


def _torch_load_verified(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[object, str]:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise WorkspaceCheckpointError(f"file hash mismatch: {actual_sha256}")
        handle.seek(0)
        payload = torch.load(
            handle,
            map_location="cpu",
            weights_only=True,
        )
    return payload, actual_sha256


def _publish_noreplace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
