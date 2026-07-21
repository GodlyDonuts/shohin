#!/usr/bin/env python3
"""Independently verify the structured-route train-only qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from pilot_er_dual_stream_opcode_canary import (
    EVIDENCE_SCHEMA,
    FITTED_ARMS,
    REPORT_SCHEMA,
    SCHEMA,
    compute_gates,
)
from pilot_er_dual_stream_relation_adapter import EXPECTED_PARAMETERS
from pilot_er_relation_tensor import atomic_json_save
from pilot_sd_cst_byte_addressed import sha256_file


ASSESSMENT_SCHEMA = "r12_er_dual_stream_structured_route_assessment_v1_4"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"structured-route {label} is not a mapping")
    return value


def verify_path_evidence(evidence: Mapping[str, object]) -> dict[str, object]:
    checked = 0
    map_exact = 0
    complement_exact = 0
    probability_exact = 0
    probability_softmax_exact = 0
    row_exact_by_arm_mode: dict[str, int] = {}
    for arm_name, arm_raw in _mapping(evidence["arms"], "arms").items():
        arm = _mapping(arm_raw, f"arm {arm_name}")
        modes = _mapping(arm["modes"], f"arm {arm_name} modes")
        for mode_name in ("s0_qstruct", "s1_qstruct"):
            mode = _mapping(modes[mode_name], f"mode {mode_name}")
            relocated = _mapping(mode["relocated"], "relocated evidence")
            scores = relocated["path_scores"].float()
            probabilities = relocated["path_probability"].float()
            predicted_cardinality = relocated["pred_cardinality"].long()
            target_cardinality = relocated["target_cardinality"].long()
            target_rule_count = relocated["target_rule_count"].long()
            map_exclusion = relocated["map_exclusion"].long()
            target_exclusion = relocated["target_exclusion"].long()
            candidates = relocated["candidate_positions"].long()
            witness = relocated["pred_witness_pointer"].long()
            opcode = relocated["rule_opcode_pointer"].long()
            row_route_exact = torch.ones(scores.shape[0], dtype=torch.bool)
            if scores.shape != (8_000, 4, 4, 13):
                raise ValueError("structured-route path score shape differs")
            if probabilities.shape != scores.shape:
                raise ValueError("structured-route path probability shape differs")
            if predicted_cardinality.shape != target_cardinality.shape:
                raise ValueError("structured-route cardinality evidence differs")
            for row in range(scores.shape[0]):
                cardinality = int(predicted_cardinality[row])
                if not 3 <= cardinality <= 6:
                    continue
                cardinality_index = cardinality - 3
                for rule in range(int(target_rule_count[row])):
                    checked += 1
                    if int(target_cardinality[row]) != cardinality:
                        row_route_exact[row] = False
                        continue
                    selected = int(map_exclusion[row, rule])
                    target = int(target_exclusion[row, rule])
                    if selected < 0 or target < 0:
                        row_route_exact[row] = False
                        continue
                    path = scores[row, rule, cardinality_index, : 2 * cardinality + 1]
                    stored_probability = probabilities[
                        row,
                        rule,
                        cardinality_index,
                        : 2 * cardinality + 1,
                    ]
                    map_exact += int(int(path.argmax()) == selected)
                    row_route_exact[row] &= selected == target
                    probability_exact += int(
                        torch.isclose(
                            stored_probability.sum(),
                            torch.tensor(1.0),
                            atol=2e-3,
                        )
                    )
                    probability_softmax_exact += int(
                        torch.allclose(
                            stored_probability,
                            path.softmax(-1),
                            atol=2e-3,
                            rtol=2e-3,
                        )
                    )
                    candidate = candidates[row, rule]
                    candidate = candidate[candidate.ge(0)]
                    slots = tuple(range(cardinality)) + tuple(
                        range(6, 6 + cardinality)
                    )
                    selected_witness = witness[row, rule, list(slots)]
                    partition = torch.cat(
                        (selected_witness, opcode[row, rule, None])
                    ).sort().values
                    complement_exact += int(
                        candidate.numel() == 2 * cardinality + 1
                        and torch.equal(partition, candidate.sort().values)
                    )
            row_exact_by_arm_mode[f"{arm_name}:{mode_name}"] = int(
                row_route_exact.sum()
            )
    return {
        "active_routes_checked": checked,
        "map_argmax_exact": map_exact,
        "coherent_complement_exact": complement_exact,
        "conditional_probability_normalized": probability_exact,
        "conditional_probability_matches_softmax": probability_softmax_exact,
        "row_exact_by_arm_mode": row_exact_by_arm_mode,
        "all_map_argmax_exact": checked > 0 and map_exact == checked,
        "all_coherent_complements_exact": checked > 0 and complement_exact == checked,
        "all_conditional_probabilities_normalized": checked > 0
        and probability_exact == checked,
        "all_conditional_probabilities_match_softmax": checked > 0
        and probability_softmax_exact == checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing structured-route assessment: {args.out}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    evidence = torch.load(args.evidence, map_location="cpu", weights_only=False)
    report = json.loads(args.report.read_text())
    checkpoint = _mapping(checkpoint, "checkpoint")
    evidence = _mapping(evidence, "evidence")
    report = _mapping(report, "report")

    report_arms = _mapping(report["arms"], "report arms")
    shared_initialization = (
        len(
            {
                str(_mapping(value, "report arm")["initial_state_sha256"])
                for value in report_arms.values()
            }
        )
        == 1
    )
    recomputed_gates, recomputed_diagnosis = compute_gates(
        report_arms,
        parameters=_mapping(report["parameters"], "parameters"),
        shared_initialization=shared_initialization,
    )
    path = verify_path_evidence(evidence)
    route_metric_exact = all(
        int(
            _mapping(
                _mapping(
                    _mapping(report_arms[arm_name], "report arm")["modes"],
                    "report modes",
                )[mode_name],
                "report mode",
            )["relocated"]["coherent"]["overall"]["witness_pointer"][
                "correct"
            ]
        )
        == int(path["row_exact_by_arm_mode"][f"{arm_name}:{mode_name}"])
        for arm_name in report_arms
        for mode_name in ("s0_qstruct", "s1_qstruct")
    )
    expected_arms = {"zero_update", *FITTED_ARMS}
    checks = {
        "checkpoint_schema_exact": checkpoint.get("schema") == SCHEMA,
        "evidence_schema_exact": evidence.get("schema") == EVIDENCE_SCHEMA,
        "report_schema_exact": report.get("schema") == REPORT_SCHEMA,
        "source_manifest_exact": checkpoint.get("source_manifest")
        == evidence.get("source_manifest")
        == report.get("source_manifest"),
        "seed_exact": checkpoint.get("seed") == evidence.get("seed") == report.get("seed"),
        "split_exact": checkpoint.get("split") == evidence.get("split") == report.get("split"),
        "parameter_certificate_exact": checkpoint.get("parameters")
        == report.get("parameters")
        == EXPECTED_PARAMETERS,
        "arms_exact": set(_mapping(checkpoint["arms"], "checkpoint arms"))
        == set(_mapping(evidence["arms"], "evidence arms"))
        == set(report_arms)
        == expected_arms,
        "shared_initialization_exact": shared_initialization,
        "gates_recompute_exact": recomputed_gates == report.get("gates"),
        "diagnosis_recomputes_exact": recomputed_diagnosis == report.get("diagnosis"),
        "decision_recomputes_exact": report.get("decision")
        == (
            "authorize_new_fresh_board_source"
            if all(recomputed_gates.values())
            else "reject_opcode_coupled_before_fresh_board"
        ),
        "path_map_exact": path["all_map_argmax_exact"],
        "path_complement_exact": path["all_coherent_complements_exact"],
        "path_probability_exact": path[
            "all_conditional_probabilities_normalized"
        ],
        "path_probability_softmax_exact": path[
            "all_conditional_probabilities_match_softmax"
        ],
        "path_route_metric_exact": route_metric_exact,
        "zero_scored_reads": int(report.get("development_accesses", -1)) == 0
        and int(report.get("confirmation_accesses", -1)) == 0
        and int(evidence.get("development_accesses", -1)) == 0
        and int(evidence.get("confirmation_accesses", -1)) == 0,
    }
    assessment = {
        "schema": ASSESSMENT_SCHEMA,
        "files": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "evidence_sha256": sha256_file(args.evidence),
            "report_sha256": sha256_file(args.report),
        },
        "checks": checks,
        "path_evidence": path,
        "all_checks_pass": all(checks.values()),
        "decision": report.get("decision"),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    atomic_json_save(assessment, args.out)
    print(json.dumps(assessment, sort_keys=True))


if __name__ == "__main__":
    main()
