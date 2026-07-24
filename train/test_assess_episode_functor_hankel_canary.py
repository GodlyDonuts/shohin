from __future__ import annotations

import json
from pathlib import Path

import pytest

from assess_episode_functor_hankel_canary import (
    HankelCanaryAssessmentError,
    PASS_DECISION,
    assess_hankel_canary,
)
from pipeline.episode_functor_runtime_custody import write_json_fsync
from run_episode_functor_hankel_canary import _write_complete_receipt


HASHES = {
    "source_manifest_sha256": "1" * 64,
    "package_sha256": "2" * 64,
    "arm_receipt_sha256": "3" * 64,
    "initialization_receipt_sha256": "4" * 64,
    "canary_receipt_sha256": "5" * 64,
}
TRAINABLE = 64_407_956


def _step(index: int) -> dict[str, object]:
    return {
        "update_index": index,
        "loss": 2.0 / index,
        "gradient_norm": 1.0 / index,
        "trainable_parameters": TRAINABLE,
        "optimizer_state_bytes": 512,
        "exact_metrics": {"rows": 4},
        "training_manifest_sha256": "6" * 64,
        "candidate_input_manifest_sha256": "7" * 64,
    }


def _report() -> dict[str, object]:
    return {
        "schema": "efc-hankel-measurement-canary/v1",
        "decision": "measurement_only_qualification_fit_no_go",
        "claim_scope": "measurement only",
        "process_id": 123,
        "landlock_receipt": {
            "stage": "efc-hsc-canary",
            "enforced": True,
            "dumpable": False,
            "abi": 6,
            "process_id": 123,
        },
        "network_namespace": {
            "environment_receipt": True,
            "interfaces": ["lo"],
            "non_loopback_interfaces": 0,
        },
        "gpu": {
            "device_count": 1,
            "device_index": 0,
            "name": "NVIDIA H100 PCIe",
            "total_memory_bytes": 80_000_000_000,
        },
        **HASHES,
        "source_manifest_rows": 26,
        "initial_trainable_state_sha256": "8" * 64,
        "final_trainable_state_sha256": "9" * 64,
        "unique_rows": 4,
        "cumulative_presentations": 8,
        "unique_source_bytes": 100,
        "cumulative_source_bytes_presented": 200,
        "independent_target_bits_per_presentation": 1_000,
        "supplied_target_bits_per_presentation": 2_000,
        "cumulative_supplied_target_bits": 4_000,
        "updates": 2,
        "elapsed_ns": 20,
        "nanoseconds_per_update": 10,
        "peak_allocated_bytes": 1_000,
        "peak_reserved_bytes": 1_200,
        "optimizer_state_bytes": 512,
        "steps": [_step(1), _step(2)],
        "trainer_phase": "sealed",
        "weights_persisted": False,
        "optimizer_persisted": False,
        "development_visible": False,
        "confirmation_visible": False,
        "fit_authorized": False,
        "pretraining_authorized": False,
    }


def _write_output(tmp_path: Path, report: dict[str, object]) -> Path:
    output = tmp_path / "result"
    output.mkdir()
    write_json_fsync(output / "canary_report.json", report)
    _write_complete_receipt(output)
    return output


def _assess(output: Path) -> dict[str, object]:
    return assess_hankel_canary(
        output,
        expected_source_manifest_sha256=HASHES[
            "source_manifest_sha256"
        ],
        expected_package_sha256=HASHES["package_sha256"],
        expected_arm_receipt_sha256=HASHES["arm_receipt_sha256"],
        expected_initialization_receipt_sha256=HASHES[
            "initialization_receipt_sha256"
        ],
        expected_canary_receipt_sha256=HASHES[
            "canary_receipt_sha256"
        ],
        expected_trainable_parameters=TRAINABLE,
    )


def test_assessment_accepts_exact_measurement_and_authorizes_nothing(
    tmp_path: Path,
) -> None:
    result = _assess(_write_output(tmp_path, _report()))
    assert result["decision"] == PASS_DECISION
    assert result["trainable_parameters"] == TRAINABLE
    assert result["fit_authorized"] is False
    assert result["development_authorized"] is False
    assert result["confirmation_authorized"] is False
    assert result["reasoning_claim_authorized"] is False
    assert result["pretraining_authorized"] is False
    assert len(result["assessment_sha256"]) == 64


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("gpu", "name"), "NVIDIA A100", "runtime custody"),
        (("steps", 0, "gradient_norm"), 0.0, "update 1"),
        (("steps", 1, "training_manifest_sha256"), "a" * 64, "custody changed"),
        (("peak_reserved_bytes",), 999, "resource arithmetic"),
        (("fit_authorized",), True, "claim boundary"),
        (
            ("final_trainable_state_sha256",),
            "8" * 64,
            "claim boundary",
        ),
    ),
)
def test_assessment_rejects_invalid_measurement(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
    match: str,
) -> None:
    report = _report()
    target: object = report
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(HankelCanaryAssessmentError, match=match):
        _assess(_write_output(tmp_path, report))


def test_assessment_rejects_mutated_authenticated_report(
    tmp_path: Path,
) -> None:
    output = _write_output(tmp_path, _report())
    report = _report()
    report["elapsed_ns"] = 22
    (output / "canary_report.json").write_text(
        json.dumps(report),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="completion receipt differs"):
        _assess(output)
