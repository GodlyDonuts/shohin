#!/usr/bin/env python3
"""Audit CTAA gate inputs and reject unresolved provenance contracts.

Revision 1 of this file accepted a caller-authored sidecar containing family
labels, parent mappings, pass bits, and a bootstrap seed.  That is an invalid
custody boundary: all of those values can be selected after outcomes are
visible.  The current assessment schema also discards the per-family oracle
provenance needed to reconstruct them independently.

This revision therefore does two things only:

1. validate and recompute every claim that current immutable artifacts can
   support; and
2. fail with ``UnresolvedContractError`` before any advancement statistic is
   computed.

The evaluator must remain closed until upstream producers commit the missing
provenance before outcome access.  It never accepts a metadata sidecar or a
caller-selected bootstrap seed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from ctaa_evaluation_io import sha256_file


ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v1"
FINITE_AUDIT_SCHEMA = "r12_ctaa_v2_finite_core_evaluation_v1"
RESOURCE_PROFILE_SCHEMA = "r12_ctaa_v2_resource_profile_v1"
CAPACITY_AUDIT_SCHEMA = "ctaa_matched_core_preflight_v1"
IMMUTABLE_PREFLIGHT_SCHEMA = "r12_ctaa_v2_immutable_artifact_preflight_v1"

ARMS = (
    "ctaa_closure",
    "oprc_closure",
    "ctaa_no_closure",
    "ctaa_shuffled_closure",
)
DATASETS = ("base", "intervention")
SEMANTIC_AXES = ("train", "development", "confirmation")
FACTORIAL_CELLS = frozenset(
    {"iii", "iih", "ihi", "ihh", "hii", "hih", "hhi", "hhh"}
)
PROGRAM_CLASSES = frozenset(
    {"stable_rank_two", "implicit_final_collapse", "explicit_final_collapse"}
)
DEPTHS = (16, 32)
RENDERERS = frozenset(range(16))
EXPECTED_PER_FACTORIAL_CLASS_DEPTH = 576
EXPECTED_BASE_FAMILIES = 27_648
EXPECTED_INTERVENTION_FAMILIES = 12_960
STRICT_SYSTEM_PARAMETER_LIMIT = 149_999_999

ASSESSMENT_KEYS = {
    "schema",
    "partition",
    "manifest_sha256",
    "access",
    "oracle_sha256",
    "runs",
    "capability_gate_computed",
}
RUN_KEYS = {"seed", "arm", "dataset", "evidence_commitment", "scores"}
SCORE_KEYS = {
    "overall",
    "by_factorial_cell",
    "by_program_class",
    "by_depth",
    "by_renderer",
    "by_action_active_prefix_accuracy",
    "by_step_quartile_active_prefix_accuracy",
    "intervention_relation_correct",
    "family_scores",
}
FAMILY_KEYS = {
    "family_id",
    "factorial_cell",
    "program_class",
    "depth",
    "renderer",
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
    "active_steps_correct",
    "active_steps_total",
}
BOOLEAN_FAMILY_KEYS = {
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
}
COMMITMENT_SHA_KEYS = {
    "program_predictions_sha256",
    "compiler_sha256",
    "program_source_sha256",
    "query_source_sha256",
    "packet_index_sha256",
    "execution_sha256",
    "core_sha256",
    "query_predictions_sha256",
    "answers_sha256",
    "evidence_sha256",
}
SHARED_COMMITMENT_KEYS = {
    "program_predictions_sha256",
    "compiler_sha256",
    "program_source_sha256",
    "query_source_sha256",
    "packet_index_sha256",
}
SHARED_FRONTEND_KEYS = {
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
}

UNRESOLVED_CONTRACTS = (
    "assessment_v1 family_scores omit oracle-derived parent_family_id and intervention relation for every family",
    "assessment_v1 family_scores omit family-level semantic-action and action-rank membership needed for clustered marginals",
    "assessment_v1 stores only pooled action/quartile percentages, which cannot be converted back into family outcomes",
    "no immutable pre-outcome receipt commits the hierarchical-bootstrap seed",
    "no capability-time resource/custody receipt binds parameter, recurrent-state, FLOP, source-deletion, and control-capacity audits to the assessed manifest and frozen cores",
)


class UnresolvedContractError(RuntimeError):
    """The available immutable schemas cannot safely support advancement."""

    def __init__(self, audit: Mapping[str, object]):
        self.audit = dict(audit)
        blockers = self.audit.get("unresolved_contracts", [])
        super().__init__(
            "CTAA advancement contract is unresolved: " + " | ".join(map(str, blockers))
        )


def _load_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"CTAA {label} is not a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"CTAA {label} SHA-256 commitment differs")
    return str(value)


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"CTAA {label} Boolean differs")
    return bool(value)


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(f"CTAA {label} integer differs")
    return int(value)


def _validate_commitment(value: object, rows: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("CTAA evidence commitment differs")
    for key in COMMITMENT_SHA_KEYS:
        _require_sha256(value.get(key), f"evidence {key}")
    if not isinstance(value.get("core_kind"), str) or not value["core_kind"]:
        raise ValueError("CTAA evidence core kind differs")
    if value.get("rows") != rows or value.get("oracle_access") != 0:
        raise ValueError("CTAA evidence rows or oracle-access custody differs")
    valid = _require_int(value.get("valid_packets"), "valid packet count")
    if valid > rows or any(
        value.get(key) != valid
        for key in ("executed_rows", "queried_rows", "answered_rows")
    ):
        raise ValueError("CTAA evidence stage counts differ")
    return value


def _validate_family_rows(
    value: object,
    *,
    dataset: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("CTAA family outcomes are missing")
    expected = EXPECTED_BASE_FAMILIES if dataset == "base" else EXPECTED_INTERVENTION_FAMILIES
    if len(value) != expected:
        raise ValueError(f"CTAA {dataset} family count differs")
    rows: dict[str, dict[str, object]] = {}
    geometry: Counter[tuple[str, str, int]] = Counter()
    for row in value:
        # Extra caller-authored mappings, labels, or bootstrap fields are not
        # extensions of assessment_v1; they are rejected as forged metadata.
        if not isinstance(row, dict) or set(row) != FAMILY_KEYS:
            raise ValueError("CTAA assessment_v1 family schema differs or contains forged labels")
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id or family_id in rows:
            raise ValueError("CTAA family ID identity differs")
        if row.get("factorial_cell") not in FACTORIAL_CELLS:
            raise ValueError("CTAA factorial-cell stratum differs")
        if row.get("program_class") not in PROGRAM_CLASSES:
            raise ValueError("CTAA program-class stratum differs")
        depth = _require_int(row.get("depth"), "family depth", minimum=1)
        renderer = _require_int(row.get("renderer"), "family renderer")
        if depth not in DEPTHS or renderer not in RENDERERS:
            raise ValueError("CTAA depth or renderer stratum differs")
        for key in BOOLEAN_FAMILY_KEYS:
            _require_bool(row.get(key), f"family {key}")
        total = _require_int(row.get("active_steps_total"), "active-step total", minimum=1)
        correct = _require_int(row.get("active_steps_correct"), "active-step correct")
        if total != depth or correct > total:
            raise ValueError("CTAA active-prefix family geometry differs")
        rows[family_id] = row
        geometry[(str(row["factorial_cell"]), str(row["program_class"]), depth)] += 1
    if dataset == "base":
        expected_geometry = {
            (cell, program_class, depth): EXPECTED_PER_FACTORIAL_CLASS_DEPTH
            for cell in FACTORIAL_CELLS
            for program_class in PROGRAM_CLASSES
            for depth in DEPTHS
        }
        if dict(geometry) != expected_geometry:
            raise ValueError("CTAA factorial/class/depth family geometry differs")
    return rows


def _validate_assessment(
    report: dict[str, object],
) -> tuple[
    list[int],
    dict[tuple[int, str, str], dict[str, object]],
    dict[tuple[int, str, str], dict[str, dict[str, object]]],
]:
    if set(report) != ASSESSMENT_KEYS or report.get("schema") != ASSESSMENT_SCHEMA:
        raise ValueError("CTAA assessment_v1 schema differs or contains outcome-aware metadata")
    if report.get("partition") not in {"development", "confirmation"}:
        raise ValueError("CTAA assessment partition differs")
    _require_sha256(report.get("manifest_sha256"), "assessment manifest")
    if report.get("capability_gate_computed") is not False:
        raise ValueError("CTAA assessment already asserts a capability gate")
    access = report.get("access")
    if (
        not isinstance(access, dict)
        or access.get("partition") != report["partition"]
        or access.get("access") != 1
    ):
        raise ValueError("CTAA assessment access receipt differs")
    runs = report.get("runs")
    if not isinstance(runs, dict) or len(runs) != 40:
        raise ValueError("CTAA assessment requires exactly five paired seeds by four arms by two datasets")

    indexed: dict[tuple[int, str, str], dict[str, object]] = {}
    families: dict[tuple[int, str, str], dict[str, dict[str, object]]] = {}
    for name, run in runs.items():
        if not isinstance(name, str) or not isinstance(run, dict) or set(run) != RUN_KEYS:
            raise ValueError("CTAA assessment run schema differs")
        seed = _require_int(run.get("seed"), "paired seed")
        arm = run.get("arm")
        dataset = run.get("dataset")
        if arm not in ARMS or dataset not in DATASETS:
            raise ValueError("CTAA assessment arm or dataset differs")
        key = (seed, str(arm), str(dataset))
        if key in indexed:
            raise ValueError("CTAA assessment duplicates a paired run")
        scores = run.get("scores")
        if not isinstance(scores, dict) or set(scores) != SCORE_KEYS:
            raise ValueError("CTAA assessment score schema differs or contains forged metadata")
        rows = _validate_family_rows(scores["family_scores"], dataset=str(dataset))
        overall = scores.get("overall")
        if not isinstance(overall, dict) or overall.get("rows") != len(rows):
            raise ValueError("CTAA assessment aggregate row count differs")
        commitment = _validate_commitment(run.get("evidence_commitment"), len(rows))
        indexed[key] = {**run, "evidence_commitment": commitment}
        families[key] = rows

    seeds = sorted({seed for seed, _, _ in indexed})
    expected_lattice = {
        (seed, arm, dataset)
        for seed in seeds
        for arm in ARMS
        for dataset in DATASETS
    }
    if len(seeds) != 5 or set(indexed) != expected_lattice:
        raise ValueError("CTAA paired 5x4x2 run lattice differs")

    reference_ids = {
        dataset: set(families[(seeds[0], ARMS[0], dataset)]) for dataset in DATASETS
    }
    for dataset in DATASETS:
        for seed in seeds:
            commitments = []
            for arm in ARMS:
                key = (seed, arm, dataset)
                if set(families[key]) != reference_ids[dataset]:
                    raise ValueError("CTAA paired family ID identity differs")
                commitments.append(indexed[key]["evidence_commitment"])
            for commitment_key in SHARED_COMMITMENT_KEYS:
                if len({value[commitment_key] for value in commitments}) != 1:
                    raise ValueError(f"CTAA paired {commitment_key} identity differs")
            for family_id in reference_ids[dataset]:
                signatures = {
                    tuple(families[(seed, arm, dataset)][family_id][field] for field in SHARED_FRONTEND_KEYS)
                    for arm in ARMS
                }
                if len(signatures) != 1:
                    raise ValueError("CTAA shared compiler outcomes differ across arms")

    compiler_hashes = set()
    core_hashes = set()
    for seed in seeds:
        per_seed_compilers = set()
        for arm in ARMS:
            base = indexed[(seed, arm, "base")]["evidence_commitment"]
            intervention = indexed[(seed, arm, "intervention")]["evidence_commitment"]
            if (
                base["core_sha256"] != intervention["core_sha256"]
                or base["core_kind"] != intervention["core_kind"]
            ):
                raise ValueError("CTAA base/intervention frozen-core identity differs")
            core_hashes.add(str(base["core_sha256"]))
            per_seed_compilers.add(str(base["compiler_sha256"]))
            per_seed_compilers.add(str(intervention["compiler_sha256"]))
        if len(per_seed_compilers) != 1:
            raise ValueError("CTAA base/intervention compiler identity differs")
        compiler_hashes.update(per_seed_compilers)
    if len(compiler_hashes) != 5 or len(core_hashes) != 20:
        raise ValueError("CTAA independently initialized compiler/core identity differs")

    for dataset in DATASETS:
        for source_key in ("program_source_sha256", "query_source_sha256"):
            values = {
                indexed[(seed, arm, dataset)]["evidence_commitment"][source_key]
                for seed in seeds
                for arm in ARMS
            }
            if len(values) != 1:
                raise ValueError(f"CTAA sealed {dataset} {source_key} differs")
    return seeds, indexed, families


def _validate_finite_audits(
    paths: Sequence[Path],
    *,
    indexed: Mapping[tuple[int, str, str], dict[str, object]],
) -> dict[str, object]:
    expected_by_core = {
        str(run["evidence_commitment"]["core_sha256"]): (seed, arm)
        for (seed, arm, dataset), run in indexed.items()
        if dataset == "base"
    }
    if len(paths) != 20 or len(expected_by_core) != 20:
        raise ValueError("CTAA requires exactly twenty uniquely core-bound finite audits")
    audited = {}
    for path in paths:
        value = _load_object(path, "finite-domain audit")
        if value.get("schema") != FINITE_AUDIT_SCHEMA or value.get("board_access") != 0:
            raise ValueError("CTAA finite-domain audit schema or custody differs")
        core_sha = _require_sha256(value.get("core_sha256"), "finite-audit core")
        if core_sha not in expected_by_core or core_sha in audited:
            raise ValueError("CTAA finite audit is duplicated or not bound to an assessed core")
        seed, arm = expected_by_core[core_sha]
        commitment = indexed[(seed, arm, "base")]["evidence_commitment"]
        if value.get("core_kind") != commitment["core_kind"]:
            raise ValueError("CTAA finite-audit core kind differs")
        axes = value.get("axes")
        gates = value.get("gates")
        if not isinstance(axes, dict) or not isinstance(gates, dict):
            raise ValueError("CTAA finite-domain axis receipt differs")
        if set(axes) != set(SEMANTIC_AXES) or set(gates) != set(SEMANTIC_AXES):
            raise ValueError("CTAA finite-domain axis coverage differs")
        recomputed = {}
        for axis in SEMANTIC_AXES:
            result = axes[axis]
            if (
                not isinstance(result, dict)
                or result.get("atomic_cases") != 243
                or result.get("two_action_cases") != 2_187
            ):
                raise ValueError("CTAA finite-domain case geometry differs")
            passed = all(
                result.get(metric) == 1.0
                for metric in (
                    "atomic_exact",
                    "two_action_exact",
                    "composition_exact",
                    "route_agreement",
                )
            )
            if gates[axis] is not passed:
                raise ValueError("CTAA finite-domain producer pass bit is forged")
            recomputed[axis] = passed
        all_pass = all(recomputed.values())
        if value.get("all_gates_pass") is not all_pass:
            raise ValueError("CTAA finite-domain aggregate pass bit is forged")
        audited[core_sha] = {
            "path_sha256": sha256_file(path),
            "seed": seed,
            "arm": arm,
            "axis_pass": recomputed,
            "all_pass": all_pass,
        }
    if set(audited) != set(expected_by_core):
        raise ValueError("CTAA finite-domain audit coverage differs")
    return audited


def _validate_resource_profile(value: dict[str, object]) -> dict[str, object]:
    if value.get("schema") != RESOURCE_PROFILE_SCHEMA:
        raise ValueError("CTAA resource-profile schema differs")
    ledger = value.get("parameter_ledger")
    cores = value.get("core_parameters")
    flops = value.get("transition_flops")
    state = value.get("state_contract")
    charge = value.get("evaluation_charge")
    if not all(isinstance(item, dict) for item in (ledger, cores, flops, state, charge)):
        raise ValueError("CTAA resource-profile structure differs")
    ledger_values = {key: _require_int(ledger.get(key), f"parameter {key}") for key in ("trunk", "compiler_adapter", "core", "total", "headroom")}
    ledger_exact = (
        ledger_values["total"]
        == ledger_values["trunk"] + ledger_values["compiler_adapter"] + ledger_values["core"]
        and ledger_values["headroom"]
        == STRICT_SYSTEM_PARAMETER_LIMIT - ledger_values["total"]
        and ledger_values["total"] <= STRICT_SYSTEM_PARAMETER_LIMIT
    )
    core_exact = (
        cores.get("closure_feature") == cores.get("outer_product_control") == 107_753
        and cores.get("exactly_matched") is True
        and ledger_values["core"] == 107_753
    )
    charged = max(
        _require_int(flops.get("closure_feature_analytic"), "treatment FLOPs"),
        _require_int(flops.get("outer_product_control_analytic"), "control FLOPs"),
    )
    flop_exact = (
        flops.get("charged_per_call") == charged
        and flops.get("treatment_padding_charge")
        == charged - flops.get("closure_feature_analytic")
        and flops.get("control_padding_charge")
        == charged - flops.get("outer_product_control_analytic")
        and charge.get("dual_route_core_calls_per_row") == 123
        and charge.get("charged_core_flops_per_row") == 123 * charged
    )
    state_exact = (
        state.get("matched_across_arms") is True
        and state.get("hard_packet_bytes_per_row") == 56
        and state.get("semantic_recurrent_state_bytes") == 3
        and state.get("implementation_recurrent_state_int64_bytes") == 24
        and state.get("halt_state_bytes") == 1
    )
    static_pass = (
        ledger_exact
        and core_exact
        and flop_exact
        and state_exact
        and value.get("qualified_memory_tensors") == 63
    )
    if value.get("all_static_gates_pass") is not static_pass:
        raise ValueError("CTAA resource-profile producer pass bit is forged")
    if value.get("board_seed_generated") is not False or value.get("oracle_access") != 0:
        raise ValueError("CTAA resource-profile custody differs")
    return {
        "base_sha256": _require_sha256(value.get("base_sha256"), "resource base"),
        "qualified_compiler_sha256": _require_sha256(
            value.get("qualified_compiler_sha256"), "qualified compiler"
        ),
        "parameter_ledger": ledger_values,
        "closure_feature_flops": flops["closure_feature_analytic"],
        "outer_product_flops": flops["outer_product_control_analytic"],
        "all_static_gates_pass": static_pass,
    }


def _validate_capacity_audit(value: dict[str, object]) -> dict[str, object]:
    if value.get("schema") != CAPACITY_AUDIT_SCHEMA:
        raise ValueError("CTAA matched-capacity audit schema differs")
    treatment = value.get("treatment")
    control = value.get("control")
    gates = value.get("gates")
    if not all(isinstance(item, dict) for item in (treatment, control, gates)):
        raise ValueError("CTAA matched-capacity audit structure differs")
    recomputed = {
        "parameters_exactly_matched": (
            value.get("treatment_parameters") == value.get("control_parameters") == 107_753
        ),
        "control_features_separate_all_pairs": value.get("unique_control_features") == 729,
        "closure_treatment_optimizes_exactly": treatment.get("exact_accuracy") == 1.0,
        "arbitrary_control_table_optimizes_exactly": control.get("exact_accuracy") == 1.0,
    }
    if gates != recomputed or value.get("all_gates_pass") is not all(recomputed.values()):
        raise ValueError("CTAA matched-capacity producer pass bits are forged")
    return {
        "treatment_parameters": value["treatment_parameters"],
        "control_parameters": value["control_parameters"],
        "all_gates_pass": all(recomputed.values()),
    }


def _validate_immutable_preflight(
    value: dict[str, object],
    *,
    resources: Mapping[str, object],
    capacity: Mapping[str, object],
) -> dict[str, object]:
    if value.get("schema") != IMMUTABLE_PREFLIGHT_SCHEMA:
        raise ValueError("CTAA immutable-preflight schema differs")
    base = value.get("base")
    compiler = value.get("qualified_compiler")
    core = value.get("core_match")
    if not all(isinstance(item, dict) for item in (base, compiler, core)):
        raise ValueError("CTAA immutable-preflight structure differs")
    gates = (
        base.get("sha256") == resources["base_sha256"]
        and base.get("strict_missing_keys") == []
        and base.get("strict_unexpected_keys") == []
        and compiler.get("sha256") == resources["qualified_compiler_sha256"]
        and compiler.get("memory_tensors_loaded") == compiler.get("memory_tensors_present") == 63
        and value.get("parameter_ledger") == resources["parameter_ledger"]
        and core.get("treatment_parameters") == capacity["treatment_parameters"]
        and core.get("control_parameters") == capacity["control_parameters"]
        and core.get("treatment_flops") == resources["closure_feature_flops"]
        and core.get("control_flops") == resources["outer_product_flops"]
        and value.get("board_artifact_written") is False
        and value.get("jobs_launched") is False
        and value.get("production_seed_generated") is False
    )
    if value.get("all_gates_pass") is not gates:
        raise ValueError("CTAA immutable-preflight producer pass bit is forged")
    return {
        "base_sha256": base["sha256"],
        "qualified_compiler_sha256": compiler["sha256"],
        "all_gates_pass": gates,
    }


def audit_current_contract(
    *,
    assessment_path: Path,
    finite_audit_paths: Sequence[Path],
    resource_profile_path: Path,
    capacity_audit_path: Path,
    immutable_preflight_path: Path,
) -> dict[str, object]:
    assessment = _load_object(assessment_path, "assessment")
    seeds, indexed, _families = _validate_assessment(assessment)
    finite = _validate_finite_audits(finite_audit_paths, indexed=indexed)
    resource = _validate_resource_profile(
        _load_object(resource_profile_path, "resource profile")
    )
    capacity = _validate_capacity_audit(
        _load_object(capacity_audit_path, "capacity audit")
    )
    immutable = _validate_immutable_preflight(
        _load_object(immutable_preflight_path, "immutable preflight"),
        resources=resource,
        capacity=capacity,
    )
    return {
        "schema": "r12_ctaa_v2_advancement_contract_audit_v2",
        "assessment_schema": assessment["schema"],
        "assessment_sha256": sha256_file(assessment_path),
        "manifest_sha256": assessment["manifest_sha256"],
        "seeds": seeds,
        "finite_audits": finite,
        "resource_profile_sha256": sha256_file(resource_profile_path),
        "capacity_audit_sha256": sha256_file(capacity_audit_path),
        "immutable_preflight_sha256": sha256_file(immutable_preflight_path),
        "resource_recomputation": resource,
        "capacity_recomputation": capacity,
        "immutable_recomputation": immutable,
        "caller_metadata_accepted": False,
        "caller_bootstrap_seed_accepted": False,
        "advancement_statistics_computed": False,
        "all_advancement_gates_pass": False,
        "contract_resolved": False,
        "unresolved_contracts": list(UNRESOLVED_CONTRACTS),
    }


def evaluate_advancement_gates(
    *,
    assessment_path: Path,
    finite_audit_paths: Sequence[Path],
    resource_profile_path: Path,
    capacity_audit_path: Path,
    immutable_preflight_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing existing CTAA advancement output: {output_path}")
    audit = audit_current_contract(
        assessment_path=assessment_path,
        finite_audit_paths=finite_audit_paths,
        resource_profile_path=resource_profile_path,
        capacity_audit_path=capacity_audit_path,
        immutable_preflight_path=immutable_preflight_path,
    )
    # Deliberately do not write a development-gate-shaped receipt.  A rejection
    # artifact could be mistaken for an authorization by downstream code.
    raise UnresolvedContractError(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--finite-audit", type=Path, action="append", required=True)
    parser.add_argument("--resource-profile", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--immutable-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate_advancement_gates(
            assessment_path=args.assessment,
            finite_audit_paths=args.finite_audit,
            resource_profile_path=args.resource_profile,
            capacity_audit_path=args.capacity_audit,
            immutable_preflight_path=args.immutable_preflight,
            output_path=args.output,
        )
    except UnresolvedContractError as error:
        print(json.dumps(error.audit, sort_keys=True))
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
