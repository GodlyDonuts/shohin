from __future__ import annotations

from copy import deepcopy
import hashlib
from inspect import signature
import json
from itertools import product
from pathlib import Path

import pytest

import evaluate_ctaa_advancement_gates as gates


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _family_row(
    family_id: str,
    *,
    index: int,
    cell: str,
    program_class: str,
    depth: int,
) -> dict[str, object]:
    return {
        "family_id": family_id,
        "factorial_cell": cell,
        "program_class": program_class,
        "depth": depth,
        "renderer": index % 16,
        "packet_valid": True,
        "cards_exact": True,
        "binding_exact": True,
        "initial_exact": True,
        "stop_exact": True,
        "schedule_exact": True,
        "program_exact": True,
        "query_exact": True,
        "halt_valid": True,
        "route_agreement": True,
        "prefix_exact": True,
        "terminal_exact": True,
        "answer_exact": True,
        "active_steps_correct": depth,
        "active_steps_total": depth,
    }


def _base_rows() -> list[dict[str, object]]:
    rows = []
    index = 0
    for cell, program_class, depth in product(
        sorted(gates.FACTORIAL_CELLS),
        sorted(gates.PROGRAM_CLASSES),
        gates.DEPTHS,
    ):
        for repeat in range(gates.EXPECTED_PER_FACTORIAL_CLASS_DEPTH):
            rows.append(
                _family_row(
                    f"base-{cell}-{program_class}-{depth}-{repeat}",
                    index=index,
                    cell=cell,
                    program_class=program_class,
                    depth=depth,
                )
            )
            index += 1
    return rows


def _intervention_rows(base: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for index in range(gates.EXPECTED_INTERVENTION_FAMILIES):
        parent = base[index % len(base)]
        rows.append(
            _family_row(
                f"intervention-{index}",
                index=index,
                cell=str(parent["factorial_cell"]),
                program_class=str(parent["program_class"]),
                depth=int(parent["depth"]),
            )
        )
    return rows


def _commitment(seed: int, arm: str, dataset: str, rows: int) -> dict[str, object]:
    return {
        "schema": "r12_ctaa_v2_raw_evidence_receipt_v1",
        "rows": rows,
        "valid_packets": rows,
        "executed_rows": rows,
        "queried_rows": rows,
        "answered_rows": rows,
        "program_predictions_sha256": _digest(f"program:{seed}:{dataset}"),
        "compiler_sha256": _digest(f"compiler:{seed}"),
        "program_source_sha256": _digest(f"program-source:{dataset}"),
        "query_source_sha256": _digest(f"query-source:{dataset}"),
        "packet_index_sha256": _digest(f"packet-index:{seed}:{dataset}"),
        "execution_sha256": _digest(f"execution:{seed}:{arm}:{dataset}"),
        "core_sha256": _digest(f"core:{seed}:{arm}"),
        "core_kind": "oprc" if arm == "oprc_closure" else "ctaa",
        "query_predictions_sha256": _digest(f"query:{seed}:{arm}:{dataset}"),
        "answers_sha256": _digest(f"answers:{seed}:{arm}:{dataset}"),
        "evidence_sha256": _digest(f"evidence:{seed}:{arm}:{dataset}"),
        "oracle_access": 0,
    }


def _scores(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "overall": {"rows": len(rows), "prefix_exact": 1.0},
        "by_factorial_cell": {},
        "by_program_class": {},
        "by_depth": {},
        "by_renderer": {},
        "by_action_active_prefix_accuracy": {},
        "by_step_quartile_active_prefix_accuracy": {},
        "intervention_relation_correct": {},
        "family_scores": rows,
    }


def _resource_profile() -> dict[str, object]:
    return {
        "schema": gates.RESOURCE_PROFILE_SCHEMA,
        "base_sha256": _digest("base"),
        "base_step": 300_000,
        "qualified_compiler_sha256": _digest("qualified"),
        "qualified_memory_tensors": 63,
        "parameter_ledger": {
            "trunk": 125_081_664,
            "compiler_adapter": 12_797_451,
            "core": 107_753,
            "total": 137_986_868,
            "headroom": 12_013_131,
        },
        "core_parameters": {
            "closure_feature": 107_753,
            "outer_product_control": 107_753,
            "exactly_matched": True,
        },
        "transition_flops": {
            "closure_feature_analytic": 215_530,
            "outer_product_control_analytic": 215_584,
            "charged_per_call": 215_584,
            "treatment_padding_charge": 54,
            "control_padding_charge": 0,
        },
        "state_contract": {
            "hard_packet_bytes_per_row": 56,
            "semantic_recurrent_state_bytes": 3,
            "implementation_recurrent_state_int64_bytes": 24,
            "halt_state_bytes": 1,
            "matched_across_arms": True,
        },
        "evaluation_charge": {
            "dual_route_core_calls_per_row": 123,
            "charged_core_flops_per_row": 26_516_832,
        },
        "runtime": None,
        "profile_depths": [1, 16, 32, 39],
        "board_seed_generated": False,
        "oracle_access": 0,
        "all_static_gates_pass": True,
    }


def _capacity_audit() -> dict[str, object]:
    return {
        "schema": gates.CAPACITY_AUDIT_SCHEMA,
        "treatment_parameters": 107_753,
        "control_parameters": 107_753,
        "unique_control_features": 729,
        "treatment": {"exact_accuracy": 1.0},
        "control": {"exact_accuracy": 1.0},
        "gates": {
            "parameters_exactly_matched": True,
            "control_features_separate_all_pairs": True,
            "closure_treatment_optimizes_exactly": True,
            "arbitrary_control_table_optimizes_exactly": True,
        },
        "all_gates_pass": True,
    }


def _immutable_preflight() -> dict[str, object]:
    return {
        "schema": gates.IMMUTABLE_PREFLIGHT_SCHEMA,
        "base": {
            "sha256": _digest("base"),
            "strict_missing_keys": [],
            "strict_unexpected_keys": [],
        },
        "qualified_compiler": {
            "sha256": _digest("qualified"),
            "memory_tensors_loaded": 63,
            "memory_tensors_present": 63,
        },
        "parameter_ledger": {
            "trunk": 125_081_664,
            "compiler_adapter": 12_797_451,
            "core": 107_753,
            "total": 137_986_868,
            "headroom": 12_013_131,
        },
        "core_match": {
            "treatment_parameters": 107_753,
            "control_parameters": 107_753,
            "treatment_flops": 215_530,
            "control_flops": 215_584,
        },
        "board_artifact_written": False,
        "jobs_launched": False,
        "production_seed_generated": False,
        "all_gates_pass": True,
    }


@pytest.fixture
def current_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(gates, "EXPECTED_PER_FACTORIAL_CLASS_DEPTH", 1)
    monkeypatch.setattr(gates, "EXPECTED_BASE_FAMILIES", 48)
    monkeypatch.setattr(gates, "EXPECTED_INTERVENTION_FAMILIES", 26)
    seeds = [101, 202, 303, 404, 505]
    base = _base_rows()
    intervention = _intervention_rows(base)
    runs = {}
    for seed in seeds:
        for arm in gates.ARMS:
            for dataset, template in (("base", base), ("intervention", intervention)):
                rows = deepcopy(template)
                name = f"{seed}-{arm}-{dataset}"
                runs[name] = {
                    "seed": seed,
                    "arm": arm,
                    "dataset": dataset,
                    "evidence_commitment": _commitment(seed, arm, dataset, len(rows)),
                    "scores": _scores(rows),
                }
    assessment = {
        "schema": gates.ASSESSMENT_SCHEMA,
        "partition": "development",
        "manifest_sha256": _digest("manifest"),
        "access": {"partition": "development", "access": 1},
        "oracle_sha256": {name: _digest(f"oracle:{name}") for name in runs},
        "runs": runs,
        "capability_gate_computed": False,
    }
    assessment_path = tmp_path / "assessment.json"
    _write(assessment_path, assessment)

    finite_paths = []
    finite_values = {}
    for seed in seeds:
        for arm in gates.ARMS:
            axes = {
                axis: {
                    "atomic_cases": 243,
                    "two_action_cases": 2_187,
                    "atomic_exact": 1.0,
                    "two_action_exact": 1.0,
                    "composition_exact": 1.0,
                    "route_agreement": 1.0,
                }
                for axis in gates.SEMANTIC_AXES
            }
            value = {
                "schema": gates.FINITE_AUDIT_SCHEMA,
                "core_sha256": _digest(f"core:{seed}:{arm}"),
                "core_kind": "oprc" if arm == "oprc_closure" else "ctaa",
                "device": "cuda",
                "axes": axes,
                "gates": {axis: True for axis in gates.SEMANTIC_AXES},
                "all_gates_pass": True,
                "board_access": 0,
            }
            path = tmp_path / f"finite-{seed}-{arm}.json"
            _write(path, value)
            finite_paths.append(path)
            finite_values[path] = value

    resource = _resource_profile()
    capacity = _capacity_audit()
    immutable = _immutable_preflight()
    resource_path = tmp_path / "resource.json"
    capacity_path = tmp_path / "capacity.json"
    immutable_path = tmp_path / "immutable.json"
    _write(resource_path, resource)
    _write(capacity_path, capacity)
    _write(immutable_path, immutable)
    return {
        "tmp_path": tmp_path,
        "seeds": seeds,
        "assessment": assessment,
        "assessment_path": assessment_path,
        "finite_paths": finite_paths,
        "finite_values": finite_values,
        "resource": resource,
        "resource_path": resource_path,
        "capacity": capacity,
        "capacity_path": capacity_path,
        "immutable": immutable,
        "immutable_path": immutable_path,
    }


def _audit(inputs: dict[str, object]) -> dict[str, object]:
    return gates.audit_current_contract(
        assessment_path=inputs["assessment_path"],
        finite_audit_paths=inputs["finite_paths"],
        resource_profile_path=inputs["resource_path"],
        capacity_audit_path=inputs["capacity_path"],
        immutable_preflight_path=inputs["immutable_path"],
    )


def _rewrite_assessment(inputs: dict[str, object]) -> None:
    _write(inputs["assessment_path"], inputs["assessment"])


def test_current_schema_is_a_documented_unresolved_contract_and_never_authorizes(
    current_contract: dict[str, object],
) -> None:
    audit = _audit(current_contract)
    assert audit["contract_resolved"] is False
    assert audit["advancement_statistics_computed"] is False
    assert audit["all_advancement_gates_pass"] is False
    assert audit["caller_metadata_accepted"] is False
    assert audit["caller_bootstrap_seed_accepted"] is False
    assert len(audit["unresolved_contracts"]) >= 5
    output = current_contract["tmp_path"] / "must-not-exist.json"
    with pytest.raises(gates.UnresolvedContractError, match="contract is unresolved") as caught:
        gates.evaluate_advancement_gates(
            assessment_path=current_contract["assessment_path"],
            finite_audit_paths=current_contract["finite_paths"],
            resource_profile_path=current_contract["resource_path"],
            capacity_audit_path=current_contract["capacity_path"],
            immutable_preflight_path=current_contract["immutable_path"],
            output_path=output,
        )
    assert caught.value.audit["all_advancement_gates_pass"] is False
    assert not output.exists()


def test_public_api_accepts_neither_sidecar_metadata_nor_bootstrap_seed() -> None:
    parameters = signature(gates.evaluate_advancement_gates).parameters
    assert "metadata_path" not in parameters
    assert "bootstrap_seed" not in parameters
    assert "family_annotations" not in parameters


@pytest.mark.parametrize("field", ["parent_family_id", "relation", "action_strata", "rank_strata"])
def test_rejects_forged_family_mappings_and_labels(
    current_contract: dict[str, object], field: str
) -> None:
    run = next(iter(current_contract["assessment"]["runs"].values()))
    run["scores"]["family_scores"][0][field] = "forged"
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="forged labels"):
        _audit(current_contract)


@pytest.mark.parametrize("field", ["bootstrap_seed", "advancement_metadata", "all_development_gates_pass"])
def test_rejects_outcome_selected_top_level_metadata(
    current_contract: dict[str, object], field: str
) -> None:
    current_contract["assessment"][field] = 7 if field == "bootstrap_seed" else True
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="outcome-aware metadata"):
        _audit(current_contract)


def test_rejects_outcome_selected_bootstrap_seed_inside_scores(
    current_contract: dict[str, object]
) -> None:
    run = next(iter(current_contract["assessment"]["runs"].values()))
    run["scores"]["bootstrap_seed"] = 123
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="forged metadata"):
        _audit(current_contract)


def test_missing_finite_receipt_fails_closed(current_contract: dict[str, object]) -> None:
    current_contract["finite_paths"].pop()
    with pytest.raises(ValueError, match="exactly twenty"):
        _audit(current_contract)


@pytest.mark.parametrize("receipt", ["resource", "capacity", "immutable"])
def test_missing_source_receipt_fails_closed(
    current_contract: dict[str, object], receipt: str
) -> None:
    Path(current_contract[f"{receipt}_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        _audit(current_contract)


def test_finite_audits_are_mapped_by_core_hash_not_caller_labels(
    current_contract: dict[str, object]
) -> None:
    current_contract["finite_paths"].reverse()
    audit = _audit(current_contract)
    mapped = audit["finite_audits"]
    assert len(mapped) == 20
    assert {entry["seed"] for entry in mapped.values()} == set(current_contract["seeds"])


def test_duplicate_or_foreign_finite_core_receipt_is_rejected(
    current_contract: dict[str, object]
) -> None:
    current_contract["finite_paths"][-1] = current_contract["finite_paths"][0]
    with pytest.raises(ValueError, match="duplicated"):
        _audit(current_contract)


def test_forged_finite_pass_bit_is_recomputed_and_rejected(
    current_contract: dict[str, object]
) -> None:
    path = current_contract["finite_paths"][0]
    value = current_contract["finite_values"][path]
    value["axes"]["development"]["atomic_exact"] = 0.5
    # Deliberately leave producer gate bits true.
    _write(path, value)
    with pytest.raises(ValueError, match="pass bit is forged"):
        _audit(current_contract)


def test_honest_finite_failure_is_audited_but_cannot_authorize(
    current_contract: dict[str, object]
) -> None:
    path = current_contract["finite_paths"][0]
    value = current_contract["finite_values"][path]
    value["axes"]["development"]["atomic_exact"] = 0.5
    value["gates"]["development"] = False
    value["all_gates_pass"] = False
    _write(path, value)
    audit = _audit(current_contract)
    core_sha = value["core_sha256"]
    assert audit["finite_audits"][core_sha]["all_pass"] is False
    assert audit["all_advancement_gates_pass"] is False


@pytest.mark.parametrize("receipt", ["resource", "capacity", "immutable"])
def test_forged_receipt_pass_booleans_are_recomputed_and_rejected(
    current_contract: dict[str, object], receipt: str
) -> None:
    if receipt == "resource":
        current_contract["resource"]["core_parameters"]["closure_feature"] = 1
        path = current_contract["resource_path"]
        value = current_contract["resource"]
    elif receipt == "capacity":
        current_contract["capacity"]["treatment"]["exact_accuracy"] = 0.5
        path = current_contract["capacity_path"]
        value = current_contract["capacity"]
    else:
        current_contract["immutable"]["jobs_launched"] = True
        path = current_contract["immutable_path"]
        value = current_contract["immutable"]
    _write(path, value)
    with pytest.raises(ValueError, match="pass bit|pass bits"):
        _audit(current_contract)


@pytest.mark.parametrize(
    "field",
    [
        "compiler_sha256",
        "program_predictions_sha256",
        "program_source_sha256",
        "query_source_sha256",
        "packet_index_sha256",
    ],
)
def test_rejects_paired_commitment_mutation(
    current_contract: dict[str, object], field: str
) -> None:
    run = next(
        run
        for run in current_contract["assessment"]["runs"].values()
        if run["arm"] == "oprc_closure" and run["dataset"] == "base"
    )
    run["evidence_commitment"][field] = _digest(f"forged:{field}")
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match=field):
        _audit(current_contract)


def test_rejects_family_id_mutation(current_contract: dict[str, object]) -> None:
    run = next(
        run
        for run in current_contract["assessment"]["runs"].values()
        if run["arm"] == "oprc_closure" and run["dataset"] == "base"
    )
    run["scores"]["family_scores"][0]["family_id"] = "forged-family"
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="family ID identity"):
        _audit(current_contract)


def test_current_pooled_relation_percentages_never_supply_family_provenance(
    current_contract: dict[str, object]
) -> None:
    for run in current_contract["assessment"]["runs"].values():
        if run["dataset"] == "intervention":
            run["scores"]["intervention_relation_correct"] = {
                "forged_relation": 1.0,
                "post_stop_poison": 1.0,
            }
    _rewrite_assessment(current_contract)
    audit = _audit(current_contract)
    assert audit["contract_resolved"] is False
    assert any("parent_family_id" in item for item in audit["unresolved_contracts"])
    assert audit["advancement_statistics_computed"] is False
