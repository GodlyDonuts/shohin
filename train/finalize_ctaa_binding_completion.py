#!/usr/bin/env python3
"""Finalize CTAA binding attribution from already committed artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from assess_ctaa_binding_completion import (
    SCHEMA as ASSESSMENT_SCHEMA,
    packet_metrics,
)
from capacity_ctaa_binding_completion import SCHEMA as CAPACITY_SCHEMA
from ctaa_binding_completion import readout_resource_receipt
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from freeze_ctaa_binding_completion_seeds import write_json_once
from profile_ctaa_binding_completion_resources import (
    MEASURED_UPDATES,
    SCHEMA as RESOURCE_SCHEMA,
    WARMUP_UPDATES,
    percentile,
)
from predict_ctaa_binding_completion import (
    PREDICTION_KEYS,
    SCHEMA as PREDICTION_SCHEMA,
    load_seed_freeze,
)
from train_ctaa_binding_completion import (
    fixed_schedule,
    metrics_from_logits,
    safe_torch_load,
    sha256_file,
    tagged_seed,
    tensor_sha256,
    validate_frozen_seed,
)


SCHEMA = "r12_ctaa_a4_binding_completion_decision_v1"


def evaluate_seed(
    row: dict[str, object],
    capacity_row: dict[str, object],
    admission: dict[str, object],
) -> dict[str, object]:
    seed = int(row["seed"])
    confirmation = row["confirmation_metrics"]
    packet = row["program_packet_metrics"]["factorized"]
    probes = row["single_slot_probe_metrics"]
    chimera = row["two_slot_chimera_metrics"]["factorized"]
    factorized = confirmation["factorized"]
    global_structured = confirmation["global_structured"]
    factorized_exact = float(factorized["projected_binding_exact"])
    advantage = factorized_exact - float(
        global_structured["projected_binding_exact"]
    )
    maximum_probe = max(
        float(value["projected_binding_exact"])
        for value in probes.values()
    )
    gates = {
        "projected_confirmation": (
            factorized_exact
            >= float(admission["minimum_confirmation_factorized_exact"])
        ),
        "raw_confirmation": (
            float(factorized["raw_binding_exact"])
            >= float(admission["minimum_confirmation_factorized_exact"])
        ),
        "raw_assignment_valid": (
            float(factorized["raw_assignment_valid"])
            >= float(admission["minimum_confirmation_factorized_exact"])
        ),
        "matched_control_advantage": (
            advantage >= float(admission["minimum_factorized_advantage"])
        ),
        "single_slot_leakage": (
            maximum_probe <= float(admission["maximum_single_slot_exact"])
        ),
        "a4_derived_odd_chimera": (
            float(chimera["projected_binding_exact"])
            >= float(admission["minimum_chimera_exact"])
        ),
        "packet_program_exact": (
            float(packet["program_exact"])
            >= float(admission["minimum_confirmation_factorized_exact"])
        ),
        "packet_persistent_excitation": (
            float(packet["opcode_persistent_excitation"]) == 1.0
        ),
        "packet_binding_counterfactual": (
            float(packet["binding_counterfactual_effect"]) == 1.0
        ),
        "all_s4_capacity": (
            capacity_row.get("all_arms_qualified") is True
        ),
    }
    return {
        "seed": seed,
        "factorized_confirmation_exact": factorized_exact,
        "factorized_advantage": advantage,
        "maximum_single_slot_exact": maximum_probe,
        "gates": gates,
        "seed_pass": all(gates.values()),
    }


def finalize(
    *,
    admission_path: Path,
    prediction_path: Path,
    assessment_path: Path,
    capacity_path: Path,
    resource_path: Path,
    output: Path,
) -> dict[str, object]:
    admission = load_admission(admission_path)
    require_admitted_protocol_source(admission)
    for path, key in (
        (assessment_path, "assessment_artifact_name"),
        (prediction_path, "prediction_artifact_name"),
        (capacity_path, "capacity_artifact_name"),
        (resource_path, "resource_artifact_name"),
        (output, "decision_artifact_name"),
    ):
        require_admitted_artifact_path(path, admission, key)
    assessment, assessment_sha256 = safe_torch_load(assessment_path)
    prediction, prediction_sha256 = safe_torch_load(prediction_path)
    capacity, capacity_sha256 = safe_torch_load(capacity_path)
    resource_encoded = resource_path.read_bytes()
    resource = json.loads(resource_encoded)
    resource_sha256 = sha256_file(resource_path)
    admission_sha256 = sha256_file(admission_path)
    if (
        assessment.get("schema") != ASSESSMENT_SCHEMA
        or assessment.get("admission_sha256") != admission_sha256
        or assessment.get("confirmation_oracle_access") != 1
        or assessment.get("prediction_sha256") != prediction_sha256
    ):
        raise ValueError("CTAA completion decision assessment differs")
    if (
        set(prediction) != PREDICTION_KEYS
        or prediction.get("schema") != PREDICTION_SCHEMA
        or prediction.get("admission_sha256") != admission_sha256
        or prediction.get("confirmation_oracle_access") != 0
    ):
        raise ValueError("CTAA completion decision prediction differs")
    if (
        capacity.get("schema") != CAPACITY_SCHEMA
        or capacity.get("admission_sha256") != admission_sha256
        or capacity.get("assessment_sha256") != assessment_sha256
        or capacity.get("prediction_sha256") != prediction_sha256
        or capacity.get("additional_confirmation_oracle_access") != 0
    ):
        raise ValueError("CTAA completion decision capacity differs")
    if (
        not isinstance(resource, dict)
        or resource.get("schema") != RESOURCE_SCHEMA
        or resource.get("admission_sha256") != admission_sha256
        or resource.get("code_commit") != admission["code_commit"]
        or resource.get("protocol_source_sha256")
        != admission["protocol_source_sha256"]
    ):
        raise ValueError("CTAA completion decision resource receipt differs")
    if assessment.get("ordered_row_ids") != prediction.get("ordered_row_ids"):
        raise ValueError("CTAA completion decision row identities differ")
    seed_freeze_path = (
        Path(str(admission["custody_root"]))
        / str(admission["seed_freeze_manifest_name"])
    )
    freeze_records, seed_freeze_sha256 = load_seed_freeze(
        seed_freeze_path,
        admission_sha256=admission_sha256,
    )
    if seed_freeze_sha256 != prediction.get("seed_freeze_sha256"):
        raise ValueError("CTAA completion decision seed freeze differs")
    if (
        resource.get("seed_freeze_sha256") != seed_freeze_sha256
        or resource.get("frozen_seed_sha256")
        != prediction["seed_predictions"][0]["frozen_seed_sha256"]
        or resource.get("seed") != admission["seeds"][0]
    ):
        raise ValueError("CTAA completion decision resource seed differs")
    frozen_seeds = []
    for predicted, record, seed in zip(
        prediction["seed_predictions"],
        freeze_records,
        admission["seeds"],
        strict=True,
    ):
        frozen_path = Path(str(predicted["frozen_seed_path"]))
        frozen, _ = safe_torch_load(
            frozen_path,
            expected_sha256=str(record["artifact_sha256"]),
        )
        validate_frozen_seed(
            frozen,
            admission=admission,
            admission_sha256=admission_sha256,
            expected_seed=int(seed),
        )
        frozen_seeds.append(frozen)
    first_frozen = frozen_seeds[0]
    resource_schedule = fixed_schedule(
        first_frozen["train_slot_cache"].shape[0],
        WARMUP_UPDATES + MEASURED_UPDATES,
        int(admission["batch_size"]),
        tagged_seed(
            int(admission["seeds"][0]),
            "measured-resource-batches",
        ),
    )
    if (
        resource.get("slots_sha256")
        != tensor_sha256(first_frozen["train_slot_cache"].float())
        or resource.get("bindings_sha256")
        != tensor_sha256(first_frozen["train_bindings"].long())
        or resource.get("schedule_sha256")
        != tensor_sha256(resource_schedule)
    ):
        raise ValueError("CTAA completion decision resource inputs differ")
    ledger_path = (
        Path(str(admission["custody_root"]))
        / str(admission["oracle_access_ledger_name"])
    )
    ledger_encoded = ledger_path.read_bytes()
    ledger_sha256 = sha256_file(ledger_path)
    ledger = json.loads(ledger_encoded)
    if (
        ledger_sha256 != assessment.get("oracle_access_ledger_sha256")
        or not isinstance(ledger, dict)
        or ledger.get("access_number") != 1
        or ledger.get("admission_sha256") != admission_sha256
        or ledger.get("prediction_sha256") != prediction_sha256
        or ledger.get("oracle_sha256")
        != assessment.get("confirmation_oracle_sha256")
        or ledger.get("assessment_output")
        != str(assessment_path.resolve())
    ):
        raise ValueError("CTAA completion decision oracle ledger differs")
    arms = resource.get("arms")
    analytic = readout_resource_receipt()
    if (
        not isinstance(arms, dict)
        or set(arms) != {"factorized", "global_structured"}
        or resource.get("warmup_updates") != WARMUP_UPDATES
        or resource.get("measured_updates") != MEASURED_UPDATES
        or resource.get("counterbalanced_order") is not True
        or float(resource.get("analytic_relative_mac_gap", -1.0))
        != float(analytic["relative_mac_gap"])
    ):
        raise ValueError("CTAA completion decision resource lattice differs")
    medians = {}
    memory_peaks = {}
    for arm in ("factorized", "global_structured"):
        receipt = arms[arm]
        values = receipt.get("update_ms")
        peaks = receipt.get("peak_memory_bytes")
        if (
            not isinstance(values, list)
            or len(values) != MEASURED_UPDATES
            or any(
                type(value) is not float or not 0.0 < value < 1_000_000.0
                for value in values
            )
            or not isinstance(peaks, list)
            or len(peaks) != MEASURED_UPDATES
            or any(type(value) is not int or value <= 0 for value in peaks)
            or receipt.get("parameters")
            != analytic[f"{arm}_parameters"]
            or receipt.get("analytic_macs_per_row")
            != analytic[f"{arm}_macs"]
            or receipt.get("measured_updates") != MEASURED_UPDATES
        ):
            raise ValueError("CTAA completion decision resource arm differs")
        median = statistics.median(values)
        if (
            abs(float(receipt["median_update_ms"]) - median) > 1e-12
            or abs(
                float(receipt["p95_update_ms"])
                - percentile(values, 0.95)
            )
            > 1e-12
        ):
            raise ValueError("CTAA completion decision resource summary differs")
        medians[arm] = median
        memory_peaks[arm] = max(peaks)
    relative_runtime_gap = abs(
        medians["factorized"] - medians["global_structured"]
    ) / max(medians.values())
    relative_peak_memory_gap = abs(
        memory_peaks["factorized"]
        - memory_peaks["global_structured"]
    ) / max(memory_peaks.values())
    resource_gate_recomputed = (
        analytic["factorized_parameters"]
        == analytic["global_structured_parameters"]
        and float(analytic["relative_mac_gap"])
        <= float(admission["maximum_resource_relative_gap"])
        and relative_runtime_gap
        <= float(admission["maximum_resource_relative_gap"])
        and relative_peak_memory_gap
        <= float(admission["maximum_resource_relative_gap"])
    )
    if (
        abs(
            float(resource["measured_relative_runtime_gap"])
            - relative_runtime_gap
        )
        > 1e-12
        or abs(
            float(resource["measured_relative_peak_memory_gap"])
            - relative_peak_memory_gap
        )
        > 1e-12
        or resource["resource_gate"] is not resource_gate_recomputed
    ):
        raise ValueError("CTAA completion decision resource gate differs")
    capacity_by_seed = {
        int(row["seed"]): row
        for row in capacity["seed_results"]
    }
    assessment_rows = assessment.get("seed_results")
    oracle_rows = assessment.get("oracle_rows")
    oracle_bindings = assessment.get("oracle_bindings")
    if (
        not isinstance(assessment_rows, list)
        or [int(row["seed"]) for row in assessment_rows]
        != list(admission["seeds"])
        or set(capacity_by_seed) != set(admission["seeds"])
        or len(capacity["seed_results"]) != 5
        or not isinstance(oracle_rows, list)
        or not isinstance(oracle_bindings, torch.Tensor)
    ):
        raise ValueError("CTAA completion decision seed lattice differs")
    recomputed_rows = []
    for assessed, predicted, frozen in zip(
        assessment_rows,
        prediction["seed_predictions"],
        frozen_seeds,
        strict=True,
    ):
        confirmation = {
            arm: metrics_from_logits(
                logits,
                oracle_bindings,
                arm=arm,
            )
            for arm, logits in predicted["arm_logits"].items()
        }
        programs = {
            arm: packet_metrics(
                predicted["common_program_logits"],
                logits,
                oracle_rows,
                arm=arm,
            )
            for arm, logits in predicted["arm_logits"].items()
        }
        probes = {
            label: metrics_from_logits(
                logits,
                oracle_bindings,
                arm="global_structured",
            )
            for label, logits in predicted[
                "single_slot_probe_logits"
            ].items()
        }
        if (
            confirmation != assessed["confirmation_metrics"]
            or programs != assessed["program_packet_metrics"]
            or probes != assessed["single_slot_probe_metrics"]
            or assessed["two_slot_chimera_metrics"]
            != frozen["training"]["a4_derived_odd_chimera_metrics"]
        ):
            raise ValueError("CTAA completion decision metric replay differs")
        recomputed_rows.append(
            {
                **assessed,
                "confirmation_metrics": confirmation,
                "program_packet_metrics": programs,
                "single_slot_probe_metrics": probes,
            }
        )
    seed_results = []
    for row in recomputed_rows:
        seed = int(row["seed"])
        seed_results.append(
            evaluate_seed(
                row,
                capacity_by_seed[seed],
                admission,
            )
        )
    passing_seeds = sum(row["seed_pass"] for row in seed_results)
    resource_gate = resource_gate_recomputed
    capacity_gate = capacity.get("all_s4_capacity_gate") is True
    valid = (
        len(seed_results) == 5
        and passing_seeds >= int(admission["minimum_seed_passes"])
        and resource_gate
        and capacity_gate
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "binding_completion_attribution_only_not_recurrent_reasoning"
        ),
        "admission_sha256": admission_sha256,
        "assessment_sha256": assessment_sha256,
        "prediction_sha256": prediction_sha256,
        "seed_freeze_sha256": seed_freeze_sha256,
        "oracle_access_ledger_sha256": ledger_sha256,
        "capacity_sha256": capacity_sha256,
        "resource_sha256": resource_sha256,
        "preregistered_thresholds": {
            key: admission[key]
            for key in (
                "minimum_confirmation_factorized_exact",
                "minimum_factorized_advantage",
                "maximum_single_slot_exact",
                "minimum_chimera_exact",
                "minimum_seed_passes",
                "maximum_resource_relative_gap",
            )
        },
        "seed_results": seed_results,
        "passing_seeds": passing_seeds,
        "resource_gate": resource_gate,
        "all_s4_capacity_gate": capacity_gate,
        "confirmation_source_access": 0,
        "confirmation_oracle_access": 0,
        "valid_for_binding_attribution": valid,
        "valid_for_recurrent_reasoning": False,
    }
    digest = write_json_once(output, payload)
    return {
        "decision_sha256": digest,
        "passing_seeds": passing_seeds,
        "valid_for_binding_attribution": valid,
        "valid_for_recurrent_reasoning": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(
        admission_path=args.admission,
        prediction_path=args.predictions,
        assessment_path=args.assessment,
        capacity_path=args.capacity,
        resource_path=args.resources,
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
