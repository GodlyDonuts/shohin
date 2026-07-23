from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ctaa_neural_core import ClosureFeatureTransitionCore
from profile_ctaa_resources import (
    OBSERVATION_SCHEMA,
    PROFILE_ARMS,
    PROFILE_DEPTHS,
    PROFILE_PHASES,
    SHARED_BINDING_KEYS,
    _curriculum_selection_factory,
    _observation_digest,
    _restore_state,
    _runtime_profile,
    _state_dict_sha256,
    build_matched_arm_comparisons,
    profile,
    resource_gates_pass,
    validate_resource_receipt,
)


def _digest(character: str) -> str:
    return character * 64


def _expected_bindings() -> dict[str, dict[str, str]]:
    shared = {
        key: _digest(character)
        for key, character in zip(
            SHARED_BINDING_KEYS,
            ("a", "b", "c", "d", "e", "f", "7", "8"),
            strict=True,
        )
    }
    return {
        "closure_feature": {
            **shared,
            "core_checkpoint_sha256": _digest("1"),
            "core_kind": "closure_feature",
            "admission_device": "cuda:0",
        },
        "outer_product_control": {
            **shared,
            "core_checkpoint_sha256": _digest("2"),
            "core_kind": "outer_product_control",
            "admission_device": "cuda:0",
        },
    }


def _matrix() -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    bindings = _expected_bindings()
    observations = []
    sequence = 0
    for arm in PROFILE_ARMS:
        for phase in PROFILE_PHASES:
            for depth in PROFILE_DEPTHS:
                sequence += 1
                value: dict[str, object] = {
                    "schema": OBSERVATION_SCHEMA,
                    "arm": arm,
                    "phase": phase,
                    "active_depth": depth,
                    "device": "cpu" if phase == "curriculum_selection" else "cuda:0",
                    "batch_size": 64,
                    "repeats": 5,
                    "warmup_count": 3,
                    "elapsed_ns": 1_000_000 + sequence,
                    "milliseconds_per_iteration": (1_000_000 + sequence)
                    / 5
                    / 1_000_000,
                    "rows_per_second": 64 * 5 * 1_000_000_000 / (1_000_000 + sequence),
                    "peak_allocated_bytes": (
                        0 if phase == "curriculum_selection" else 10_000 + sequence
                    ),
                    "work_units_per_iteration": depth,
                    "bindings": deepcopy(bindings[arm]),
                }
                value["observation_sha256"] = _observation_digest(value)
                observations.append(value)
    return observations, bindings


def _validate(
    observations: list[dict[str, object]],
    bindings: dict[str, dict[str, str]],
) -> None:
    comparisons = build_matched_arm_comparisons(
        observations,
        expected_bindings=bindings,
    )
    validate_resource_receipt(
        observations=observations,
        comparisons=comparisons,
        expected_bindings=bindings,
    )


def _rehash(value: dict[str, object]) -> None:
    value["observation_sha256"] = _observation_digest(value)


def test_runtime_profile_covers_real_packet_depth_geometry() -> None:
    assert PROFILE_DEPTHS == (1, 16, 32, 39)
    report = _runtime_profile(
        ClosureFeatureTransitionCore(),
        torch.device("cpu"),
        batch_size=1,
        repeats=1,
        depth=39,
    )
    assert report["active_depth"] == 39
    assert report["rows_per_second"] > 0


def test_runtime_profile_rejects_impossible_depth_64() -> None:
    with pytest.raises(ValueError, match="depth"):
        _runtime_profile(
            ClosureFeatureTransitionCore(),
            torch.device("cpu"),
            batch_size=1,
            repeats=1,
            depth=64,
        )


def test_admission_profile_rejects_absent_runtime_before_artifact_loading() -> None:
    missing = Path("/intentionally/missing")
    with pytest.raises(ValueError, match="requires measured runtime"):
        profile(
            base_path=missing,
            qualified_path=missing,
            tokenizer_path=missing,
            compiler_train_path=missing,
            atomic_train_path=missing,
            closure_train_path=missing,
            treatment_core_path=missing,
            control_core_path=missing,
            output_path=missing,
            runtime_device=None,
            batch_size=1,
            repeats=1,
            warmup_count=1,
        )


def test_measurement_subject_can_be_restored_to_bound_checkpoint_state() -> None:
    core = ClosureFeatureTransitionCore()
    baseline = {
        name: value.detach().cpu().clone() for name, value in core.state_dict().items()
    }
    baseline_sha256 = _state_dict_sha256(baseline)
    with torch.no_grad():
        next(core.parameters()).add_(1.0)
    assert _state_dict_sha256(core.state_dict()) != baseline_sha256
    _restore_state(core, baseline)
    assert _state_dict_sha256(core.state_dict()) == baseline_sha256


def test_complete_measured_matrix_and_derived_comparisons_pass() -> None:
    observations, bindings = _matrix()
    _validate(observations, bindings)


def test_curriculum_selection_is_deterministic_and_source_bound(monkeypatch) -> None:
    pools = {
        "atomic": torch.tensor([11, 12, 13]),
        "closure": torch.tensor([21, 22, 23]),
        "compiler_depth_16": torch.tensor([31, 32, 33]),
    }
    sources = {
        "compiler_training_source_sha256": _digest("a"),
        "atomic_training_source_sha256": _digest("b"),
        "closure_training_source_sha256": _digest("c"),
    }
    factory, first_sha256 = _curriculum_selection_factory(
        pools,
        depth=16,
        batch_size=2,
        repeats=3,
        warmup_count=1,
        source_hashes=sources,
    )
    _factory_again, repeated_sha256 = _curriculum_selection_factory(
        pools,
        depth=16,
        batch_size=2,
        repeats=3,
        warmup_count=1,
        source_hashes=sources,
    )
    assert repeated_sha256 == first_sha256

    calls = []
    original_randint = torch.randint

    def counted_randint(*args, **kwargs):
        calls.append((args, kwargs))
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", counted_randint)
    factory()()
    assert len(calls) == 3

    changed = {**pools, "closure": torch.tensor([21, 22, 99])}
    _changed_factory, changed_sha256 = _curriculum_selection_factory(
        changed,
        depth=16,
        batch_size=2,
        repeats=3,
        warmup_count=1,
        source_hashes=sources,
    )
    assert changed_sha256 != first_sha256


@pytest.mark.parametrize("phase", PROFILE_PHASES)
def test_omitted_phase_measurement_fails_closed(phase: str) -> None:
    observations, bindings = _matrix()
    observations = [
        value
        for value in observations
        if not (
            value["arm"] == "closure_feature"
            and value["phase"] == phase
            and value["active_depth"] == 32
        )
    ]
    with pytest.raises(ValueError, match="incomplete"):
        _validate(observations, bindings)


def test_substituted_core_checkpoint_fails_even_with_rehashed_observation() -> None:
    observations, bindings = _matrix()
    target = observations[0]
    target["bindings"]["core_checkpoint_sha256"] = _digest("3")  # type: ignore[index]
    _rehash(target)
    with pytest.raises(ValueError, match="artifact binding"):
        _validate(observations, bindings)


@pytest.mark.parametrize("source_key", SHARED_BINDING_KEYS)
def test_substituted_source_or_checkpoint_binding_fails_closed(source_key: str) -> None:
    observations, bindings = _matrix()
    target = observations[-1]
    target["bindings"][source_key] = _digest("9")  # type: ignore[index]
    _rehash(target)
    with pytest.raises(ValueError, match="artifact binding"):
        _validate(observations, bindings)


def test_substituted_depth_fails_even_with_rehashed_observation() -> None:
    observations, bindings = _matrix()
    observations[0]["active_depth"] = 2
    _rehash(observations[0])
    with pytest.raises(ValueError, match="identity"):
        _validate(observations, bindings)


def test_unmeasured_zero_elapsed_fails_even_with_rehashed_observation() -> None:
    observations, bindings = _matrix()
    observations[0]["elapsed_ns"] = 0
    _rehash(observations[0])
    with pytest.raises(ValueError, match="measurement"):
        _validate(observations, bindings)


def test_forged_matched_arm_link_is_rejected() -> None:
    observations, bindings = _matrix()
    comparisons = build_matched_arm_comparisons(
        observations,
        expected_bindings=bindings,
    )
    comparisons[0]["control_observation_sha256"] = observations[0]["observation_sha256"]
    with pytest.raises(ValueError, match="comparison differs"):
        validate_resource_receipt(
            observations=observations,
            comparisons=comparisons,
            expected_bindings=bindings,
        )


def test_missing_matched_arm_comparison_fails_closed() -> None:
    observations, bindings = _matrix()
    comparisons = build_matched_arm_comparisons(
        observations,
        expected_bindings=bindings,
    )[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        validate_resource_receipt(
            observations=observations,
            comparisons=comparisons,
            expected_bindings=bindings,
        )


def test_omitted_curriculum_selection_fails_closed() -> None:
    observations, bindings = _matrix()
    observations = [
        value
        for value in observations
        if not (
            value["arm"] == "outer_product_control"
            and value["phase"] == "curriculum_selection"
            and value["active_depth"] == 39
        )
    ]
    with pytest.raises(ValueError, match="incomplete"):
        _validate(observations, bindings)


@pytest.mark.parametrize("field", ("milliseconds_per_iteration", "rows_per_second"))
def test_forged_derived_rate_fails_even_when_observation_is_rehashed(
    field: str,
) -> None:
    observations, bindings = _matrix()
    observations[5][field] = float(observations[5][field]) * 1.01
    _rehash(observations[5])
    with pytest.raises(ValueError, match="derived timing"):
        _validate(observations, bindings)


def test_missing_warmup_count_fails_even_when_observation_is_rehashed() -> None:
    observations, bindings = _matrix()
    observations[0].pop("warmup_count")
    _rehash(observations[0])
    with pytest.raises(ValueError, match="schema"):
        _validate(observations, bindings)


def test_cuda_phase_requires_positive_measured_peak_bytes() -> None:
    observations, bindings = _matrix()
    target = next(
        value
        for value in observations
        if value["phase"] == "optimizer_step" and value["active_depth"] == 16
    )
    target["peak_allocated_bytes"] = 0
    _rehash(target)
    with pytest.raises(ValueError, match="CUDA.*memory"):
        _validate(observations, bindings)


def test_resource_pass_is_derived_and_cannot_be_hardcoded_true() -> None:
    observations, bindings = _matrix()
    comparisons = build_matched_arm_comparisons(
        observations,
        expected_bindings=bindings,
    )
    assert resource_gates_pass(
        observations=observations,
        comparisons=comparisons,
        expected_bindings=bindings,
    )
    observations = [
        value for value in observations if value["phase"] != "curriculum_selection"
    ]
    with pytest.raises(ValueError, match="incomplete"):
        resource_gates_pass(
            observations=observations,
            comparisons=comparisons,
            expected_bindings=bindings,
        )
