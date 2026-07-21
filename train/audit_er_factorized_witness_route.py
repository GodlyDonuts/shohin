#!/usr/bin/env python3
"""Artifact-only audit of the frozen ER factorized witness canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Mapping

import torch


SCHEMA = "r12_er_factorized_witness_route_artifact_audit_v1"
CANARY_SCHEMA = "r12_er_factorized_witness_route_train_only_canary_v1"
SOURCE_COMMIT = "4643d1a51defe53397f9bed481051621d85c0b11"
SOURCE_MANIFEST_SHA256 = (
    "61c10b250dfaca8a3d2558b4afe25cfec86ad01919394c6cc9f8e35a732f9eeb"
)
SEED = 6_769_631_927_967_421_693
EXPECTED_SHA256 = {
    "compiler.pt": "e93bb4cff5f316616c7a02bce272112acf454f42f56f8b4ea07ffac6074318a2",
    "train_probe_evidence.pt": "11d931b37ad854de9976015fc1ff38522da0812776ad67ea7d408d43820889c6",
    "train_probe_report.json": "87ea12a28cfaf82c4556f1730da6778cf63df9e5c2ad3df8b09a86613786ccca",
}
ARM_MODES = ("treatment", "baseline", "structural_only", "shuffled_address")
EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 60_452_996,
    "motor": 0,
    "reader": 0,
    "complete_system": 185_534_660,
    "headroom_below_200m": 14_465_340,
    "trainable": 11_131_868,
}
EXPECTED_THRESHOLDS = {
    "packet": 0.85,
    "state": 0.85,
    "answer": 0.85,
    "joint": 0.85,
    "relation_rows": 0.90,
    "witness_pointer": 0.90,
    "events": 0.95,
    "halt": 0.95,
    "minimum_cardinality_joint": 0.75,
    "alpha_exact": 1.0,
    "oracle_route_transport_exact": 1.0,
}
OVERALL_METRICS = frozenset(
    {
        "answer",
        "binding_pointer",
        "cardinality",
        "events",
        "halt",
        "initial_pointer",
        "initial_rows",
        "joint",
        "line_pointer",
        "packet",
        "query",
        "query_pointer",
        "relation_rows",
        "rule_active",
        "state",
        "witness_pointer",
    }
)
AGGREGATE_METRICS = frozenset({"answer", "joint", "packet", "state"})


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_rate(metric: Mapping[str, object]) -> float:
    rows = int(metric["rows"])
    correct = int(metric["correct"])
    rate = float(metric["rate"])
    if rows <= 0 or correct < 0 or correct > rows:
        raise AssertionError("factorized metric counts differ")
    if not math.isclose(rate, correct / rows, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("factorized metric rate differs from counts")
    return rate


def _validate_metric_tree(value: object) -> int:
    if isinstance(value, dict) and {"correct", "rate", "rows"}.issubset(value):
        _metric_rate(value)
        return 1
    if isinstance(value, dict):
        return sum(_validate_metric_tree(child) for child in value.values())
    return 0


def _validate_metric_schema(metrics: Mapping[str, object]) -> int:
    if set(metrics) != {
        "overall",
        "by_cardinality",
        "by_depth",
        "by_renderer",
        "non_bijective",
    }:
        raise AssertionError("factorized metric groups differ")
    if set(metrics["overall"]) != OVERALL_METRICS:
        raise AssertionError("factorized overall metric schema differs")
    expected_groups = {
        "by_cardinality": {"3", "4", "5", "6"},
        "by_depth": {str(value) for value in range(1, 13)},
        "by_renderer": {
            "er-tt-d0w0e0q0-v1",
            "er-tt-d0w0e1q1-v1",
            "er-tt-d1w1e0q0-v1",
            "er-tt-d1w1e1q1-v1",
        },
    }
    overall = metrics["overall"]
    for group_name, expected_keys in expected_groups.items():
        group = metrics[group_name]
        if set(group) != expected_keys or any(
            set(value) != AGGREGATE_METRICS for value in group.values()
        ):
            raise AssertionError("factorized aggregate metric schema differs")
        for metric in AGGREGATE_METRICS:
            rows = sum(int(value[metric]["rows"]) for value in group.values())
            correct = sum(int(value[metric]["correct"]) for value in group.values())
            if rows != int(overall[metric]["rows"]) or correct != int(
                overall[metric]["correct"]
            ):
                raise AssertionError("factorized aggregate metric counts differ")
    non_bijective = metrics["non_bijective"]
    if set(non_bijective) != AGGREGATE_METRICS or any(
        non_bijective[name] != overall[name] for name in AGGREGATE_METRICS
    ):
        raise AssertionError("factorized non-bijective aggregate differs")
    leaves = _validate_metric_tree(metrics)
    if leaves != 100:
        raise AssertionError("factorized metric leaf count differs")
    return leaves


def _validate_exact_receipt(metric: Mapping[str, object], rows: int = 8_000) -> None:
    if (
        int(metric["exact"]) != rows
        or int(metric["rows"]) != rows
        or float(metric["rate"]) != 1.0
    ):
        raise AssertionError("factorized exact-rate receipt differs")


def _verify_source_manifest(repo_root: Path, manifest: Mapping[str, object]) -> int:
    if manifest["commit"] != SOURCE_COMMIT:
        raise AssertionError("factorized manifest commit differs")
    files = manifest["files"]
    for name, expected in files.items():
        payload = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{SOURCE_COMMIT}:{name}"]
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise AssertionError("factorized committed source file differs")
    return len(files)


def recompute_gates(report: Mapping[str, object]) -> dict[str, bool]:
    metrics = report["metrics"]
    controls = report["controls"]
    thresholds = report["thresholds"]
    contract = report["contract"]
    split = report["split"]
    alpha = report["alpha_invariance"]
    oracle = report["routing_diagnostics"]["oracle_route"]
    overall = metrics["overall"]
    by_cardinality = metrics["by_cardinality"]
    return {
        "family_split_exact_and_disjoint": (
            int(split["fit_families"]) == 10_000
            and int(split["probe_families"]) == 2_000
            and int(split["fit_rows"]) == 40_000
            and int(split["probe_rows"]) == 8_000
            and int(split["family_overlap"]) == 0
        ),
        "packet_state_answer_joint_at_least_85pct": all(
            float(overall[name]["rate"]) >= float(thresholds[name])
            for name in ("packet", "state", "answer", "joint")
        ),
        "relation_rows_at_least_90pct": float(overall["relation_rows"]["rate"])
        >= float(thresholds["relation_rows"]),
        "witness_pointers_at_least_90pct": float(overall["witness_pointer"]["rate"])
        >= float(thresholds["witness_pointer"]),
        "events_and_halt_at_least_95pct": all(
            float(overall[name]["rate"]) >= float(thresholds[name])
            for name in ("events", "halt")
        ),
        "minimum_cardinality_joint_at_least_75pct": min(
            float(value["joint"]["rate"]) for value in by_cardinality.values()
        )
        >= float(thresholds["minimum_cardinality_joint"]),
        "all_hard_fields_and_pointers_alpha_exact": float(alpha["complete"]["rate"])
        == float(thresholds["alpha_exact"])
        and int(alpha["complete"]["exact"]) == 8_000
        and int(alpha["complete"]["rows"]) == 8_000,
        "oracle_route_identity_transport_is_exact": all(
            float(oracle[name]["rate"])
            == float(thresholds["oracle_route_transport_exact"])
            and int(oracle[name]["exact"]) == 8_000
            and int(oracle[name]["rows"]) == 8_000
            for name in ("initial", "relations", "events", "joint")
        ),
        "confirmed_parent_unchanged": report["fit"]["treatment"][
            "frozen_parent_unchanged"
        ]
        is True,
        "witness_gain_over_same_seed_baseline_at_least_0_5pp": float(
            overall["witness_pointer"]["rate"]
        )
        - float(controls["baseline"]["overall"]["witness_pointer"]["rate"])
        >= 0.005,
        "witness_gain_over_shuffled_address_at_least_0_5pp": float(
            overall["witness_pointer"]["rate"]
        )
        - float(controls["shuffled_address"]["overall"]["witness_pointer"]["rate"])
        >= 0.005,
        "parameter_certificate_exact_and_below_200m": report["parameters"]
        == EXPECTED_PARAMETERS,
        "train_only_and_zero_scored_split_reads": contract["outcome_supervision"]
        is False
        and int(contract["development_reads"]) == 0
        and int(contract["confirmation_reads"]) == 0,
    }


def _grammar_diagnostic(state: Mapping[str, torch.Tensor]) -> dict[str, object]:
    table = state["er_fw_witness_address_bias"].float()
    gate = state["er_fw_witness_gate"].float()
    if tuple(table.shape) != (14, 12, 14) or tuple(gate.shape) != (12,):
        raise AssertionError("factorized address tensors differ")
    correct = 0
    rows = 0
    predictions: dict[str, list[int]] = {}
    for cardinality in range(3, 7):
        count = 1 + 2 * cardinality
        roles = tuple(range(cardinality)) + tuple(range(6, 6 + cardinality))
        effective = table[count, roles, :count].tanh()
        effective = effective * (4.0 * gate[list(roles)].tanh())[:, None]
        predicted = effective.argmax(-1).tolist()
        expected = list(range(1, 2 * cardinality + 1))
        correct += sum(left == right for left, right in zip(predicted, expected))
        rows += 2 * cardinality
        predictions[str(cardinality)] = predicted
    return {
        "correct": correct,
        "rows": rows,
        "rate": correct / rows,
        "predicted_ordinals_by_cardinality": predictions,
        "gate_l2": float(gate.norm()),
        "table_l2": float(table.norm()),
    }


def audit(run_dir: Path, repo_root: Path) -> dict[str, object]:
    paths = {name: run_dir / name for name in EXPECTED_SHA256}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_SHA256:
        raise AssertionError("factorized artifact identity differs")

    report = json.loads(paths["train_probe_report.json"].read_text())
    checkpoint = torch.load(
        paths["compiler.pt"], map_location="cpu", weights_only=False
    )
    evidence = torch.load(
        paths["train_probe_evidence.pt"], map_location="cpu", weights_only=False
    )
    if any(
        value["schema"] != CANARY_SCHEMA for value in (report, checkpoint, evidence)
    ):
        raise AssertionError("factorized artifact schema differs")
    if (
        report["source_commit"] != SOURCE_COMMIT
        or evidence["source_commit"] != SOURCE_COMMIT
    ):
        raise AssertionError("factorized source commit differs")
    if report["source_manifest"]["sha256"] != SOURCE_MANIFEST_SHA256:
        raise AssertionError("factorized source manifest differs")
    committed_source_files = _verify_source_manifest(
        repo_root, report["source_manifest"]
    )
    if any(int(value["seed"]) != SEED for value in (report, checkpoint, evidence)):
        raise AssertionError("factorized seed differs")
    if (
        report["artifacts"]
        != {
            "checkpoint_sha256": hashes["compiler.pt"],
            "evidence_sha256": hashes["train_probe_evidence.pt"],
        }
        or evidence["checkpoint_sha256"] != hashes["compiler.pt"]
    ):
        raise AssertionError("factorized embedded artifact hashes differ")
    if report["thresholds"] != EXPECTED_THRESHOLDS:
        raise AssertionError("factorized thresholds differ")
    if (
        tuple(report["arm_modes"]) != ARM_MODES
        or tuple(checkpoint["arm_modes"]) != ARM_MODES
    ):
        raise AssertionError("factorized arm modes differ")
    if (
        report["parameters"] != EXPECTED_PARAMETERS
        or checkpoint["parameters"] != EXPECTED_PARAMETERS
    ):
        raise AssertionError("factorized parameter certificate differs")
    if report["custody"] != {
        "train_only_probe_accesses": 1,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }:
        raise AssertionError("factorized report custody differs")
    if any(
        int(value[key]) != 0
        for value in (checkpoint, evidence)
        for key in ("development_accesses", "confirmation_accesses")
    ):
        raise AssertionError("factorized scored-split custody differs")

    common_initial = report["common_initial_trainable_state_sha256"]
    fit_seed = None
    frozen_digest = None
    grammar: dict[str, object] = {}
    for mode in ARM_MODES:
        arm = checkpoint["arms"][mode]
        if (
            arm["mode"] != mode
            or arm["initial_trainable_state_sha256"] != common_initial
        ):
            raise AssertionError("factorized matched initialization differs")
        if arm["fit"] != report["fit"][mode]:
            raise AssertionError("factorized checkpoint/report fit differs")
        if (
            int(arm["fit"]["updates"]) != 2_500
            or arm["fit"]["frozen_parent_unchanged"] is not True
        ):
            raise AssertionError("factorized fit contract differs")
        fit_seed = int(arm["fit"]["seed"]) if fit_seed is None else fit_seed
        frozen_digest = (
            arm["fit"]["frozen_digest"] if frozen_digest is None else frozen_digest
        )
        if (
            int(arm["fit"]["seed"]) != fit_seed
            or arm["fit"]["frozen_digest"] != frozen_digest
        ):
            raise AssertionError("factorized matched fit seeds or parent differ")
        grammar[mode] = _grammar_diagnostic(arm["compiler_trainable_state"])
        trainable_parameters = sum(
            int(tensor.numel()) for tensor in arm["compiler_trainable_state"].values()
        )
        if trainable_parameters != EXPECTED_PARAMETERS["trainable"]:
            raise AssertionError("factorized trainable parameter count differs")
    if not torch.equal(
        checkpoint["arms"]["baseline"]["compiler_trainable_state"][
            "er_fw_witness_gate"
        ],
        torch.zeros(12),
    ):
        raise AssertionError("factorized baseline gate is not exactly zero")

    metric_leaves = _validate_metric_schema(report["metrics"])
    if set(report["controls"]) != set(ARM_MODES) - {"treatment"}:
        raise AssertionError("factorized control schema differs")
    metric_leaves += sum(
        _validate_metric_schema(value) for value in report["controls"].values()
    )
    if metric_leaves != 400:
        raise AssertionError("factorized complete metric leaf count differs")
    recomputed_gates = recompute_gates(report)
    if recomputed_gates != report["gates"]:
        raise AssertionError("factorized gate recomputation differs")
    decision = (
        "authorize_fresh_factorized_witness_board"
        if all(recomputed_gates.values())
        else "reject_factorized_witness_before_fresh_board"
    )
    if decision != report["decision"] or report["all_gates_pass"] != all(
        recomputed_gates.values()
    ):
        raise AssertionError("factorized decision differs")

    canonical = evidence["canonical_predictions"]
    recoded = evidence["recoded_predictions"]
    if set(canonical) != set(recoded) or any(
        tuple(canonical[name].shape) != tuple(recoded[name].shape)
        or int(canonical[name].shape[0]) != 8_000
        for name in canonical
    ):
        raise AssertionError("factorized treatment evidence shapes differ")
    independently_recomputed_alpha: dict[str, dict[str, object]] = {}
    complete_mask = torch.ones(8_000, dtype=torch.bool)
    for name in sorted(canonical):
        equal = canonical[name].eq(recoded[name])
        row_equal = equal.reshape(8_000, -1).all(-1)
        complete_mask &= row_equal
        exact = int(row_equal.sum())
        independently_recomputed_alpha[name] = {
            "exact": exact,
            "rows": 8_000,
            "rate": exact / 8_000,
        }
        _validate_exact_receipt(report["alpha_invariance"][name])
        if report["alpha_invariance"][name] != independently_recomputed_alpha[name]:
            raise AssertionError("factorized alpha report differs from evidence")
    complete_exact = int(complete_mask.sum())
    independently_recomputed_alpha["complete"] = {
        "exact": complete_exact,
        "rows": 8_000,
        "rate": complete_exact / 8_000,
    }
    _validate_exact_receipt(report["alpha_invariance"]["complete"])
    if (
        report["alpha_invariance"]["complete"]
        != independently_recomputed_alpha["complete"]
    ):
        raise AssertionError("factorized complete alpha report differs")
    alpha_mask = evidence["alpha_complete_mask"]
    if (
        tuple(alpha_mask.shape) != (8_000,)
        or alpha_mask.dtype is not torch.bool
        or not torch.equal(alpha_mask, complete_mask)
    ):
        raise AssertionError("factorized alpha evidence differs")
    for name in ("initial", "relations", "events", "joint"):
        _validate_exact_receipt(report["routing_diagnostics"]["oracle_route"][name])

    overall = report["metrics"]["overall"]
    controls = report["controls"]
    result: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_sha256": hashes,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "committed_source_files_verified": committed_source_files,
        "seed": SEED,
        "fit_seed": fit_seed,
        "common_initial_trainable_state_sha256": common_initial,
        "frozen_parent_digest": frozen_digest,
        "custody": report["custody"],
        "metric_leaves_recounted": metric_leaves,
        "alpha_invariance_recomputed_from_evidence": independently_recomputed_alpha,
        "recomputed_gates": recomputed_gates,
        "decision": decision,
        "treatment_overall": {
            name: overall[name]
            for name in (
                "witness_pointer",
                "relation_rows",
                "packet",
                "state",
                "answer",
                "events",
            )
        },
        "control_overall": {
            mode: {
                name: controls[mode]["overall"][name]
                for name in (
                    "witness_pointer",
                    "relation_rows",
                    "packet",
                    "state",
                    "answer",
                    "events",
                )
            }
            for mode in controls
        },
        "minimum_cardinality_joint": min(
            float(value["joint"]["rate"])
            for value in report["metrics"]["by_cardinality"].values()
        ),
        "grammar_address_diagnostic": grammar,
        "limitations": [
            "No ER-TT board row or scored split was read by this audit.",
            "Control row-level predictions were not retained, so paired McNemar statistics cannot be reconstructed.",
            "Oracle-route targets were not retained; oracle exactness is an internally consistent producer receipt, not independently regenerated.",
            "Family disjointness, custody counters, frozen-parent integrity, and total compiler/base parameter counts are producer receipts bound by immutable hashes, not reconstructed facts.",
            "Reported probe metrics are schema-checked and aggregate-reconciled, but are not regenerated from board targets.",
        ],
        "conclusion": (
            "The immutable run is a valid negative result. The deterministic address "
            "table was learned, but soft residual identity transport remained far "
            "below the frozen absolute and minimum-cardinality gates. Do not open a "
            "fresh board or tune this route."
        ),
    }
    result["preimage_sha256"] = hashlib.sha256(
        canonical_json(result).encode()
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.repo_root)
    payload = canonical_json(result) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
