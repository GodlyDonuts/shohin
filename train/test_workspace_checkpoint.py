from __future__ import annotations

from dataclasses import asdict, replace
import inspect
from pathlib import Path

import pytest
import torch

from causal_bind_select_workspace import (
    PROTECTED_BASE_PARAMETERS,
    CausalBindSelectWorkspace,
    CausalWorkspaceConfig,
    CausalWorkspaceGPT,
    freeze_protected_base,
)
from model import GPT, GPTConfig
from workspace_checkpoint import (
    PROTECTED_BASE_CONFIG,
    PROTECTED_BASE_STATE_SHA256,
    PROTECTED_CHECKPOINT_BYTES,
    PROTECTED_CHECKPOINT_SHA256,
    PROTECTED_CHECKPOINT_STEP,
    PROTECTED_CONFIG_SHA256,
    PROTECTED_DATA_SEED,
    PROTECTED_DATA_STREAM_GENERATION,
    PROTECTED_DATA_STREAM_SEED,
    PROTECTED_STATE_KEY_COUNT,
    PROTECTED_STATE_KEY_SHA256,
    ProtectedCheckpointReceipt,
    WorkspaceCheckpointError,
    _load_workspace_delta_for_test as load_workspace_delta,
    _load_workspace_model_for_test,
    _save_workspace_delta_for_test as save_workspace_delta,
    _workspace_delta_payload_for_test as workspace_delta_payload,
    _write_architecture_receipt_for_test as write_architecture_receipt,
    file_sha256,
    json_sha256,
    load_protected_workspace_model,
    runtime_source_manifest,
    state_dict_sha256,
    workspace_delta_payload as production_workspace_delta_payload,
    write_architecture_receipt as production_write_architecture_receipt,
)


def _small_base() -> GPT:
    torch.manual_seed(2026072360)
    return GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )


def _small_workspace_config() -> CausalWorkspaceConfig:
    return CausalWorkspaceConfig(
        d_model=24,
        slot_width=16,
        num_slots=4,
        num_operators=4,
        operator_rank=4,
        stage_after_block=1,
    )


def _write_small_checkpoint(path: Path) -> tuple[GPT, str]:
    base = _small_base()
    payload = {
        "model": base.state_dict(),
        "cfg": asdict(base.cfg),
        "step": 123,
        "data_seed": 7,
        "data_stream_generation": 1,
        "data_stream_seed": 100,
    }
    torch.save(payload, path)
    return base, file_sha256(path)


def _receipt(model: CausalWorkspaceGPT, base_hash: str) -> ProtectedCheckpointReceipt:
    freeze_protected_base(model)
    parameter = model.workspace.parameter_receipt(
        protected_base_parameters=model.base.num_params()
    )
    return ProtectedCheckpointReceipt(
        checkpoint_path="/tmp/base.pt",
        checkpoint_bytes=1,
        checkpoint_sha256=base_hash,
        step=123,
        data_seed=7,
        data_stream_generation=1,
        data_stream_seed=100,
        base_config=asdict(model.base.cfg),
        config_sha256=json_sha256(asdict(model.base.cfg)),
        base_state_sha256=state_dict_sha256(model.base.state_dict()),
        state_key_sha256="keys",
        state_key_count=len(model.base.state_dict()),
        protected_base_parameters=parameter.protected_base_parameters,
        workspace_parameters=parameter.workspace_parameters,
        complete_system_parameters=parameter.complete_system_parameters,
        remaining_under_cap=parameter.remaining_under_cap,
        strict_state_load=True,
        protected_base_frozen=True,
        optimizer_state_loaded=False,
        pretraining_started=False,
    )


def _frozen_target() -> CausalWorkspaceGPT:
    target = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    freeze_protected_base(target)
    return target


def _protected_receipt(
    workspace_config: CausalWorkspaceConfig | None = None,
) -> ProtectedCheckpointReceipt:
    workspace_config = workspace_config or CausalWorkspaceConfig()
    with torch.random.fork_rng(devices=[]):
        workspace = CausalBindSelectWorkspace(workspace_config)
    parameter = workspace.parameter_receipt(
        protected_base_parameters=PROTECTED_BASE_PARAMETERS
    )
    return ProtectedCheckpointReceipt(
        checkpoint_path="/protected/ckpt_0300000.pt",
        checkpoint_bytes=PROTECTED_CHECKPOINT_BYTES,
        checkpoint_sha256=PROTECTED_CHECKPOINT_SHA256,
        step=PROTECTED_CHECKPOINT_STEP,
        data_seed=PROTECTED_DATA_SEED,
        data_stream_generation=PROTECTED_DATA_STREAM_GENERATION,
        data_stream_seed=PROTECTED_DATA_STREAM_SEED,
        base_config=dict(PROTECTED_BASE_CONFIG),
        config_sha256=PROTECTED_CONFIG_SHA256,
        base_state_sha256=PROTECTED_BASE_STATE_SHA256,
        state_key_sha256=PROTECTED_STATE_KEY_SHA256,
        state_key_count=PROTECTED_STATE_KEY_COUNT,
        protected_base_parameters=parameter.protected_base_parameters,
        workspace_parameters=parameter.workspace_parameters,
        complete_system_parameters=parameter.complete_system_parameters,
        remaining_under_cap=parameter.remaining_under_cap,
        strict_state_load=True,
        protected_base_frozen=True,
        optimizer_state_loaded=False,
        pretraining_started=False,
    )


def test_strict_small_checkpoint_upgrade_preserves_every_base_tensor(tmp_path) -> None:
    checkpoint_path = tmp_path / "base.pt"
    original, digest = _write_small_checkpoint(checkpoint_path)
    model, receipt = _load_workspace_model_for_test(
        checkpoint_path,
        _small_workspace_config(),
        expected_sha256=digest,
        expected_step=123,
        expected_base_parameters=original.num_params(),
    )
    assert receipt.strict_state_load is True
    assert receipt.protected_base_frozen is True
    assert receipt.optimizer_state_loaded is False
    assert receipt.pretraining_started is False
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, model.base.state_dict()[name])


def test_production_loader_exposes_no_overrideable_trust_root(tmp_path) -> None:
    parameters = inspect.signature(load_protected_workspace_model).parameters
    assert tuple(parameters) == ("checkpoint_path", "workspace_config")
    checkpoint_path = tmp_path / "base.pt"
    _write_small_checkpoint(checkpoint_path)
    with pytest.raises(WorkspaceCheckpointError, match="hash mismatch"):
        load_protected_workspace_model(
            checkpoint_path,
            _small_workspace_config(),
        )


def test_production_delta_api_rejects_synthetic_receipt() -> None:
    model = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(model, "a" * 64)
    with pytest.raises(WorkspaceCheckpointError, match="not the protected"):
        production_workspace_delta_payload(model, receipt)


def test_checkpoint_hash_and_top_level_keys_fail_closed(tmp_path) -> None:
    checkpoint_path = tmp_path / "base.pt"
    original, digest = _write_small_checkpoint(checkpoint_path)
    with pytest.raises(WorkspaceCheckpointError, match="hash mismatch"):
        _load_workspace_model_for_test(
            checkpoint_path,
            _small_workspace_config(),
            expected_sha256="0" * 64,
            expected_step=123,
            expected_base_parameters=original.num_params(),
        )

    payload = torch.load(checkpoint_path, weights_only=False)
    payload["unexpected"] = True
    bad_path = tmp_path / "bad.pt"
    torch.save(payload, bad_path)
    with pytest.raises(WorkspaceCheckpointError, match="keys differ"):
        _load_workspace_model_for_test(
            bad_path,
            _small_workspace_config(),
            expected_sha256=file_sha256(bad_path),
            expected_step=123,
            expected_base_parameters=original.num_params(),
        )
    assert digest != file_sha256(bad_path)


def test_missing_base_tensor_fails_strict_loading(tmp_path) -> None:
    checkpoint_path = tmp_path / "base.pt"
    original, _ = _write_small_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=False)
    del payload["model"]["blocks.0.n1.w"]
    bad_path = tmp_path / "missing.pt"
    torch.save(payload, bad_path)
    with pytest.raises(WorkspaceCheckpointError, match="strict loading"):
        _load_workspace_model_for_test(
            bad_path,
            _small_workspace_config(),
            expected_sha256=file_sha256(bad_path),
            expected_step=123,
            expected_base_parameters=original.num_params(),
        )


def test_workspace_delta_contains_no_base_or_optimizer_state() -> None:
    model = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(model, "a" * 64)
    payload = workspace_delta_payload(model, receipt)
    assert payload["optimizer_state"] is None
    assert payload["pretraining_started"] is False
    assert payload["runtime_source_manifest"] == runtime_source_manifest()
    assert set(payload["workspace_state"]) == set(model.workspace.state_dict())
    assert not any(name.startswith("base.") for name in payload["workspace_state"])


def test_workspace_delta_payload_is_a_nonaliasing_snapshot() -> None:
    model = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(model, "a" * 64)
    payload = workspace_delta_payload(model, receipt)
    name = next(iter(model.workspace.state_dict()))
    snapshot = payload["workspace_state"][name].clone()
    with torch.no_grad():
        model.workspace.state_dict()[name].add_(1)
    assert torch.equal(payload["workspace_state"][name], snapshot)


def test_workspace_delta_rejects_mutated_in_memory_base(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "a" * 64)
    with torch.no_grad():
        next(iter(source.base.parameters())).view(-1)[0].add_(1)
    with pytest.raises(WorkspaceCheckpointError, match="tensor state differs"):
        workspace_delta_payload(source, receipt)

    clean = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    clean_receipt = _receipt(clean, "a" * 64)
    path = tmp_path / "workspace.pt"
    digest = save_workspace_delta(path, clean, clean_receipt)
    target = _frozen_target()
    with torch.no_grad():
        next(iter(target.base.parameters())).view(-1)[0].add_(1)
    with pytest.raises(WorkspaceCheckpointError, match="tensor state differs"):
        load_workspace_delta(
            path,
            target,
            clean_receipt,
            expected_sha256=digest,
        )


def test_workspace_delta_round_trip_is_strict_and_atomic(tmp_path) -> None:
    torch.manual_seed(2026072361)
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    delta_path = tmp_path / "workspace.pt"
    digest = save_workspace_delta(delta_path, source, receipt)
    assert digest == file_sha256(delta_path)
    target = _frozen_target()
    load_workspace_delta(
        delta_path,
        target,
        receipt,
        expected_sha256=digest,
    )
    for name, tensor in source.workspace.state_dict().items():
        assert torch.equal(tensor, target.workspace.state_dict()[name])
    with pytest.raises(FileExistsError):
        save_workspace_delta(delta_path, source, receipt)


def test_workspace_delta_validation_preserves_global_rng_state(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    digest = save_workspace_delta(path, source, receipt)
    target = _frozen_target()
    torch.manual_seed(2026072362)
    before = torch.random.get_rng_state().clone()
    load_workspace_delta(
        path,
        target,
        receipt,
        expected_sha256=digest,
    )
    assert torch.equal(torch.random.get_rng_state(), before)


def test_workspace_delta_hash_is_required_before_deserialization(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    target = _frozen_target()
    before = {
        name: tensor.clone() for name, tensor in target.workspace.state_dict().items()
    }
    with pytest.raises(WorkspaceCheckpointError, match="hash mismatch"):
        load_workspace_delta(
            path,
            target,
            receipt,
            expected_sha256="0" * 64,
        )
    for name, tensor in target.workspace.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_rejected_workspace_delta_cannot_partially_mutate_target(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    payload = torch.load(path, weights_only=True)
    state = payload["workspace_state"]
    first, last = tuple(state)[0], tuple(state)[-1]
    state[first] = state[first] + 10
    state[last] = state[last].reshape(-1)[:-1]
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)
    target = _frozen_target()
    before = {
        name: tensor.clone() for name, tensor in target.workspace.state_dict().items()
    }
    with pytest.raises(WorkspaceCheckpointError, match="strict loading"):
        load_workspace_delta(
            malformed,
            target,
            receipt,
            expected_sha256=file_sha256(malformed),
        )
    for name, tensor in target.workspace.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_workspace_delta_rejects_nonfinite_state_without_mutation(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    payload = torch.load(path, weights_only=True)
    name = next(iter(payload["workspace_state"]))
    payload["workspace_state"][name].view(-1)[0] = float("nan")
    nonfinite = tmp_path / "nonfinite.pt"
    torch.save(payload, nonfinite)
    target = _frozen_target()
    before = {
        key: tensor.clone() for key, tensor in target.workspace.state_dict().items()
    }
    with pytest.raises(WorkspaceCheckpointError, match="nonfinite"):
        load_workspace_delta(
            nonfinite,
            target,
            receipt,
            expected_sha256=file_sha256(nonfinite),
        )
    for key, tensor in target.workspace.state_dict().items():
        assert torch.equal(tensor, before[key])


def test_workspace_delta_rejects_another_base_receipt(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    wrong = ProtectedCheckpointReceipt(
        **{**asdict(receipt), "checkpoint_sha256": "c" * 64}
    )
    target = _frozen_target()
    with pytest.raises(WorkspaceCheckpointError, match="another protected base"):
        load_workspace_delta(
            path,
            target,
            wrong,
            expected_sha256=file_sha256(path),
        )


def test_workspace_delta_reference_is_portable_across_local_paths(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    relocated = ProtectedCheckpointReceipt(
        **{**asdict(receipt), "checkpoint_path": "/another/host/base.pt"}
    )
    target = _frozen_target()
    load_workspace_delta(
        path,
        target,
        relocated,
        expected_sha256=file_sha256(path),
    )
    for name, tensor in source.workspace.state_dict().items():
        assert torch.equal(tensor, target.workspace.state_dict()[name])


def test_workspace_delta_rejects_another_implementation(tmp_path) -> None:
    source = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(source, "b" * 64)
    path = tmp_path / "workspace.pt"
    save_workspace_delta(path, source, receipt)
    payload = torch.load(path, weights_only=True)
    payload["runtime_source_manifest"] = {
        **runtime_source_manifest(),
        "model.py": "0" * 64,
    }
    changed = tmp_path / "changed.pt"
    torch.save(payload, changed)
    target = _frozen_target()
    with pytest.raises(WorkspaceCheckpointError, match="implementation differs"):
        load_workspace_delta(
            changed,
            target,
            receipt,
            expected_sha256=file_sha256(changed),
        )


def test_architecture_receipt_is_json_and_claim_bounded(tmp_path) -> None:
    model = CausalWorkspaceGPT(_small_base(), _small_workspace_config())
    receipt = _receipt(model, "d" * 64)
    path = tmp_path / "receipt.json"
    digest = write_architecture_receipt(path, receipt, model.workspace_config)
    assert digest == file_sha256(path)
    text = path.read_text()
    assert '"pretraining_started":false' in text
    assert "no neural training, reasoning, or pretraining claim" in text
    with pytest.raises(FileExistsError):
        write_architecture_receipt(path, receipt, model.workspace_config)


def test_production_architecture_receipt_binds_exact_parameter_ledger(
    tmp_path,
) -> None:
    config = CausalWorkspaceConfig()
    receipt = _protected_receipt(config)
    path = tmp_path / "production.json"
    digest = production_write_architecture_receipt(path, receipt, config)
    assert digest == file_sha256(path)

    forged = ProtectedCheckpointReceipt(
        **{**asdict(receipt), "workspace_parameters": receipt.workspace_parameters + 1}
    )
    with pytest.raises(WorkspaceCheckpointError, match="parameter ledger differs"):
        production_write_architecture_receipt(
            tmp_path / "forged.json",
            forged,
            config,
        )

    with pytest.raises(WorkspaceCheckpointError, match="d_model"):
        production_write_architecture_receipt(
            tmp_path / "wrong_width.json",
            receipt,
            replace(config, d_model=24),
        )

    with pytest.raises(WorkspaceCheckpointError, match="exceeds parameter cap"):
        production_write_architecture_receipt(
            tmp_path / "over_cap.json",
            receipt,
            replace(config, parameter_cap=PROTECTED_BASE_PARAMETERS),
        )
