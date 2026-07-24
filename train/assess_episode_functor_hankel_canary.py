#!/usr/bin/env python3
"""Fail-closed closeout for one authenticated HSC measurement canary.

The assessment verifies launch/resource mechanics only. A pass cannot
authorize a qualification fit, scored access, reasoning claim, or pretraining.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from run_episode_functor_hankel_canary import (
    CANARY_COMPLETE_SCHEMA,
    CANARY_REPORT_SCHEMA,
    verify_complete_output,
)


ASSESSMENT_SCHEMA = "efc-hankel-measurement-assessment/v1"
PASS_DECISION = "measurement_canary_pass_fit_still_no_go"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPORT_KEYS = {
    "schema",
    "decision",
    "claim_scope",
    "process_id",
    "landlock_receipt",
    "network_namespace",
    "gpu",
    "source_manifest_sha256",
    "source_manifest_rows",
    "package_sha256",
    "arm_receipt_sha256",
    "initialization_receipt_sha256",
    "canary_receipt_sha256",
    "initial_trainable_state_sha256",
    "final_trainable_state_sha256",
    "unique_rows",
    "cumulative_presentations",
    "unique_source_bytes",
    "cumulative_source_bytes_presented",
    "independent_target_bits_per_presentation",
    "supplied_target_bits_per_presentation",
    "cumulative_supplied_target_bits",
    "updates",
    "elapsed_ns",
    "nanoseconds_per_update",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "optimizer_state_bytes",
    "steps",
    "trainer_phase",
    "weights_persisted",
    "optimizer_persisted",
    "development_visible",
    "confirmation_visible",
    "fit_authorized",
    "pretraining_authorized",
}
_STEP_KEYS = {
    "update_index",
    "loss",
    "gradient_norm",
    "trainable_parameters",
    "optimizer_state_bytes",
    "exact_metrics",
    "training_manifest_sha256",
    "candidate_input_manifest_sha256",
}


class HankelCanaryAssessmentError(ValueError):
    """The canary output cannot support a measurement closeout."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HankelCanaryAssessmentError(f"{label} is not canonical SHA-256")
    return value


def _read_report(output: Path) -> tuple[dict[str, object], str]:
    complete = verify_complete_output(output)
    if complete["schema"] != CANARY_COMPLETE_SCHEMA:
        raise HankelCanaryAssessmentError("completion schema differs")
    report_sha256 = complete["files_sha256"][0]["sha256"]
    try:
        report = json.loads(
            (output.resolve(strict=True) / "canary_report.json").read_text(
                encoding="ascii"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HankelCanaryAssessmentError("canary report is invalid") from exc
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise HankelCanaryAssessmentError("canary report closure differs")
    return report, report_sha256


def _validate_runtime(report: dict[str, object]) -> None:
    landlock = report["landlock_receipt"]
    network = report["network_namespace"]
    gpu = report["gpu"]
    process_id = report["process_id"]
    if (
        not isinstance(process_id, int)
        or isinstance(process_id, bool)
        or process_id < 1
        or not isinstance(landlock, dict)
        or landlock.get("stage") != "efc-hsc-canary"
        or landlock.get("enforced") is not True
        or landlock.get("dumpable") is not False
        or landlock.get("process_id") != process_id
        or not isinstance(landlock.get("abi"), int)
        or int(landlock["abi"]) < 1
        or not isinstance(network, dict)
        or network.get("environment_receipt") is not True
        or network.get("non_loopback_interfaces") != 0
        or not isinstance(network.get("interfaces"), list)
        or any(name != "lo" for name in network["interfaces"])
        or not isinstance(gpu, dict)
        or gpu.get("device_count") != 1
        or gpu.get("device_index") != 0
        or "H100" not in str(gpu.get("name", "")).upper()
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or int(gpu["total_memory_bytes"]) < 1
    ):
        raise HankelCanaryAssessmentError("runtime custody or H100 receipt differs")


def _validate_steps(
    report: dict[str, object],
    *,
    expected_trainable_parameters: int,
) -> tuple[float, float]:
    steps = report["steps"]
    if not isinstance(steps, list) or len(steps) != 2:
        raise HankelCanaryAssessmentError("canary step count differs")
    losses: list[float] = []
    gradients: list[float] = []
    manifests: set[str] = set()
    candidates: set[str] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or set(step) != _STEP_KEYS:
            raise HankelCanaryAssessmentError("canary step closure differs")
        loss = step["loss"]
        gradient = step["gradient_norm"]
        if (
            step["update_index"] != index
            or step["trainable_parameters"] != expected_trainable_parameters
            or not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or not isinstance(gradient, (int, float))
            or isinstance(gradient, bool)
            or not math.isfinite(float(gradient))
            or float(gradient) <= 0.0
            or not isinstance(step["optimizer_state_bytes"], int)
            or step["optimizer_state_bytes"] < 1
            or not isinstance(step["exact_metrics"], dict)
        ):
            raise HankelCanaryAssessmentError(
                f"canary update {index} measurement differs"
            )
        losses.append(float(loss))
        gradients.append(float(gradient))
        manifests.add(
            _require_sha256(
                step["training_manifest_sha256"],
                "training manifest",
            )
        )
        candidates.add(
            _require_sha256(
                step["candidate_input_manifest_sha256"],
                "candidate manifest",
            )
        )
    if len(manifests) != 1 or len(candidates) != 1:
        raise HankelCanaryAssessmentError("step custody changed between updates")
    return max(losses), max(gradients)


def assess_hankel_canary(
    output: Path,
    *,
    expected_source_manifest_sha256: str,
    expected_package_sha256: str,
    expected_arm_receipt_sha256: str,
    expected_initialization_receipt_sha256: str,
    expected_canary_receipt_sha256: str,
    expected_trainable_parameters: int,
) -> dict[str, object]:
    """Validate one exact two-update result and return a nonauthorizing receipt."""

    expected_hashes = {
        "source_manifest_sha256": _require_sha256(
            expected_source_manifest_sha256,
            "expected source manifest",
        ),
        "package_sha256": _require_sha256(
            expected_package_sha256,
            "expected package",
        ),
        "arm_receipt_sha256": _require_sha256(
            expected_arm_receipt_sha256,
            "expected arm receipt",
        ),
        "initialization_receipt_sha256": _require_sha256(
            expected_initialization_receipt_sha256,
            "expected initialization receipt",
        ),
        "canary_receipt_sha256": _require_sha256(
            expected_canary_receipt_sha256,
            "expected canary receipt",
        ),
    }
    if (
        not isinstance(expected_trainable_parameters, int)
        or isinstance(expected_trainable_parameters, bool)
        or expected_trainable_parameters < 1
    ):
        raise HankelCanaryAssessmentError(
            "expected trainable parameter count differs"
        )
    report, report_sha256 = _read_report(output)
    if (
        report["schema"] != CANARY_REPORT_SCHEMA
        or report["decision"] != "measurement_only_qualification_fit_no_go"
        or any(report[name] != value for name, value in expected_hashes.items())
        or report["source_manifest_rows"] != 26
        or report["unique_rows"] != 4
        or report["cumulative_presentations"] != 8
        or report["updates"] != 2
        or report["trainer_phase"] != "sealed"
        or report["weights_persisted"] is not False
        or report["optimizer_persisted"] is not False
        or report["development_visible"] is not False
        or report["confirmation_visible"] is not False
        or report["fit_authorized"] is not False
        or report["pretraining_authorized"] is not False
        or report["initial_trainable_state_sha256"]
        == report["final_trainable_state_sha256"]
    ):
        raise HankelCanaryAssessmentError(
            "canary authorization or claim boundary differs"
        )
    for name in (
        "initial_trainable_state_sha256",
        "final_trainable_state_sha256",
    ):
        _require_sha256(report[name], name)
    _validate_runtime(report)
    maximum_loss, maximum_gradient_norm = _validate_steps(
        report,
        expected_trainable_parameters=expected_trainable_parameters,
    )
    positive_ints = (
        "unique_source_bytes",
        "cumulative_source_bytes_presented",
        "independent_target_bits_per_presentation",
        "supplied_target_bits_per_presentation",
        "cumulative_supplied_target_bits",
        "elapsed_ns",
        "nanoseconds_per_update",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "optimizer_state_bytes",
    )
    if any(
        not isinstance(report[name], int)
        or isinstance(report[name], bool)
        or report[name] < 1
        for name in positive_ints
    ):
        raise HankelCanaryAssessmentError("canary resource measurement differs")
    if (
        report["cumulative_source_bytes_presented"]
        != report["unique_source_bytes"] * 2
        or report["cumulative_supplied_target_bits"]
        != report["supplied_target_bits_per_presentation"] * 2
        or report["nanoseconds_per_update"] != report["elapsed_ns"] // 2
        or report["peak_reserved_bytes"] < report["peak_allocated_bytes"]
        or report["optimizer_state_bytes"]
        != report["steps"][-1]["optimizer_state_bytes"]
    ):
        raise HankelCanaryAssessmentError("canary resource arithmetic differs")
    assessment = {
        "schema": ASSESSMENT_SCHEMA,
        "decision": PASS_DECISION,
        "canary_report_sha256": report_sha256,
        "source_manifest_sha256": report["source_manifest_sha256"],
        "package_sha256": report["package_sha256"],
        "arm_receipt_sha256": report["arm_receipt_sha256"],
        "initialization_receipt_sha256": (
            report["initialization_receipt_sha256"]
        ),
        "canary_receipt_sha256": report["canary_receipt_sha256"],
        "trainable_parameters": expected_trainable_parameters,
        "updates": 2,
        "elapsed_ns": report["elapsed_ns"],
        "peak_allocated_bytes": report["peak_allocated_bytes"],
        "peak_reserved_bytes": report["peak_reserved_bytes"],
        "optimizer_state_bytes": report["optimizer_state_bytes"],
        "maximum_loss_hex": maximum_loss.hex(),
        "maximum_gradient_norm_hex": maximum_gradient_norm.hex(),
        "fit_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "reasoning_claim_authorized": False,
        "pretraining_authorized": False,
    }
    assessment["assessment_sha256"] = sha256(
        _canonical_json_bytes(assessment)
    ).hexdigest()
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--expected-arm-receipt-sha256", required=True)
    parser.add_argument("--expected-initialization-receipt-sha256", required=True)
    parser.add_argument("--expected-canary-receipt-sha256", required=True)
    parser.add_argument("--expected-trainable-parameters", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            assess_hankel_canary(
                args.output,
                expected_source_manifest_sha256=(
                    args.expected_source_manifest_sha256
                ),
                expected_package_sha256=args.expected_package_sha256,
                expected_arm_receipt_sha256=(
                    args.expected_arm_receipt_sha256
                ),
                expected_initialization_receipt_sha256=(
                    args.expected_initialization_receipt_sha256
                ),
                expected_canary_receipt_sha256=(
                    args.expected_canary_receipt_sha256
                ),
                expected_trainable_parameters=(
                    args.expected_trainable_parameters
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
