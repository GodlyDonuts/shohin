#!/usr/bin/env python3
"""Disposable all-S4 capacity gate after the primary assessment is committed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from assess_ctaa_binding_completion import SCHEMA as ASSESSMENT_SCHEMA
from ctaa_binding_completion import (
    FactorizedBindingReadout,
    GlobalStructuredBindingReadout,
    WholePermutationReadout,
    factorized_loss,
    whole_loss,
)
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from predict_ctaa_binding_completion import SCHEMA as PREDICTION_SCHEMA
from train_ctaa_binding_completion import (
    evaluate_readout,
    fixed_schedule,
    load_state,
    safe_torch_load,
    sha256_file,
    tagged_seed,
    train_readout,
    validate_frozen_seed,
    write_once,
)


SCHEMA = "r12_ctaa_a4_binding_completion_capacity_v1"


def run_capacity(
    *,
    assessment_path: Path,
    prediction_path: Path,
    admission_path: Path,
    output: Path,
    device_name: str,
) -> dict[str, object]:
    admission = load_admission(admission_path)
    updates = int(admission["capacity_updates"])
    batch_size = int(admission["batch_size"])
    learning_rate = float(admission["learning_rate"])
    minimum_exact = float(admission["minimum_train_exact"])
    require_admitted_protocol_source(admission)
    require_admitted_artifact_path(
        output,
        admission,
        "capacity_artifact_name",
    )
    require_admitted_artifact_path(
        assessment_path,
        admission,
        "assessment_artifact_name",
    )
    require_admitted_artifact_path(
        prediction_path,
        admission,
        "prediction_artifact_name",
    )
    assessment, assessment_sha256 = safe_torch_load(assessment_path)
    prediction, prediction_sha256 = safe_torch_load(prediction_path)
    admission_sha256 = sha256_file(admission_path)
    if (
        not isinstance(assessment, dict)
        or assessment.get("schema") != ASSESSMENT_SCHEMA
        or assessment.get("confirmation_oracle_access") != 1
        or assessment.get("prediction_sha256") != prediction_sha256
        or assessment.get("admission_sha256") != admission_sha256
    ):
        raise ValueError("CTAA completion capacity assessment differs")
    if (
        not isinstance(prediction, dict)
        or prediction.get("schema") != PREDICTION_SCHEMA
        or prediction.get("confirmation_oracle_access") != 0
        or prediction.get("admission_sha256") != admission_sha256
    ):
        raise ValueError("CTAA completion capacity prediction differs")
    oracle_bindings = assessment.get("oracle_bindings")
    if not isinstance(oracle_bindings, torch.Tensor):
        raise ValueError("CTAA completion capacity labels differ")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA completion capacity requires available CUDA")
    arm_specs = {
        "factorized": (FactorizedBindingReadout, factorized_loss),
        "global_structured": (GlobalStructuredBindingReadout, factorized_loss),
        "whole": (WholePermutationReadout, whole_loss),
    }
    results = []
    for predicted in prediction["seed_predictions"]:
        frozen_path = Path(str(predicted["frozen_seed_path"]))
        seed = int(predicted["seed"])
        if (
            frozen_path.resolve().parent
            != Path(str(admission["custody_root"]))
        ):
            raise ValueError("CTAA completion capacity seed custody differs")
        frozen, _ = safe_torch_load(
            frozen_path,
            expected_sha256=str(predicted["frozen_seed_sha256"]),
        )
        validate_frozen_seed(
            frozen,
            admission=admission,
            admission_sha256=admission_sha256,
            expected_seed=seed,
        )
        slots = torch.cat(
            (
                frozen["train_slot_cache"].float(),
                predicted["confirmation_slot_cache"].float(),
            )
        )
        bindings = torch.cat(
            (
                frozen["train_bindings"].long(),
                oracle_bindings.long(),
            )
        )
        schedule = fixed_schedule(
            slots.shape[0],
            updates,
            batch_size,
            tagged_seed(seed, "post-assessment-all-s4-capacity"),
        )
        arms = {}
        for arm, (factory, loss_function) in arm_specs.items():
            torch.manual_seed(tagged_seed(seed, f"{arm}-all-s4-capacity"))
            state, last = train_readout(
                factory(),
                slots,
                bindings,
                schedule,
                loss_function=loss_function,
                learning_rate=learning_rate,
                device=device,
            )
            model = factory()
            load_state(model, state)
            metrics = evaluate_readout(
                model,
                slots,
                bindings,
                arm=arm,
                batch_size=batch_size,
                device=device,
            )
            arms[arm] = {
                "last": last,
                "metrics": metrics,
                "fit_qualified": (
                    metrics["projected_binding_exact"] >= minimum_exact
                ),
            }
        results.append(
            {
                "seed": seed,
                "arms": arms,
                "all_arms_qualified": all(
                    value["fit_qualified"] for value in arms.values()
                ),
            }
        )
    all_pass = len(results) == 5 and all(
        result["all_arms_qualified"] for result in results
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": "disposable_capacity_only",
        "assessment_sha256": assessment_sha256,
        "prediction_sha256": prediction_sha256,
        "admission_sha256": admission_sha256,
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "minimum_exact": minimum_exact,
        "seed_results": results,
        "all_s4_capacity_gate": all_pass,
        "additional_confirmation_oracle_access": 0,
        "valid_for_binding_attribution": False,
        "remaining_gate": "measured_resource_and_end_to_end_packet_causality",
    }
    digest = write_once(output, payload)
    return {
        "capacity_sha256": digest,
        "all_s4_capacity_gate": all_pass,
        "additional_confirmation_oracle_access": 0,
        "valid_for_binding_attribution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_capacity(
        assessment_path=args.assessment,
        prediction_path=args.predictions,
        admission_path=args.admission,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
