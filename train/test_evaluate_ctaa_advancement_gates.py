from __future__ import annotations

from copy import deepcopy
import hashlib
from inspect import signature
import json
from itertools import product
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import ctaa_access_registry as access_registry
import ctaa_bootstrap_seed_receipt as bootstrap
from ctaa_bootstrap_seed_receipt import build_receipt
from ctaa_evaluation_io import sha256_file
from ctaa_statistical_gate_spec import (
    StatisticalGateBindings,
    write_signed_statistical_gate_spec,
)
import evaluate_ctaa_advancement_gates as gates
from profile_ctaa_resources import (
    OBSERVATION_SCHEMA,
    PROFILE_ARMS,
    PROFILE_DEPTHS,
    PROFILE_PHASES,
    SHARED_BINDING_KEYS,
    _observation_digest,
    build_matched_arm_comparisons,
)
from test_ctaa_bootstrap_seed_receipt import (
    TEST_ROOT_PUBLIC,
    _beacon as _signed_test_beacon,
)


bootstrap.CUSTODY_ROOT_PUBLIC_KEY_HEX = TEST_ROOT_PUBLIC


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o444)


def _write_canonical(path: Path, value: object) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(access_registry.canonical_json_bytes(value) + b"\n")
    path.chmod(0o444)


def test_registry_public_key_requires_single_link_immutable_file(
    tmp_path: Path,
) -> None:
    key = tmp_path / "registry.pub"
    key.write_bytes(b"k" * 32)
    key.chmod(0o444)
    assert gates._load_registry_public_key(key) == b"k" * 32

    alias = tmp_path / "registry-alias.pub"
    alias.hardlink_to(key)
    with pytest.raises(ValueError, match="single-link immutable"):
        gates._load_registry_public_key(key)


def test_registry_public_key_rejects_symlink(tmp_path: Path) -> None:
    key = tmp_path / "registry.pub"
    key.write_bytes(b"k" * 32)
    key.chmod(0o444)
    alias = tmp_path / "registry-symlink.pub"
    alias.symlink_to(key.name)
    with pytest.raises(ValueError, match="single-link immutable"):
        gates._load_registry_public_key(alias)


def test_registry_public_key_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    key = real / "registry.pub"
    key.write_bytes(b"k" * 32)
    key.chmod(0o444)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="parent is missing or symlinked"):
        gates._load_registry_public_key(alias / key.name)


def _beacon() -> dict[str, object]:
    return _signed_test_beacon()


def _family_row(
    family_id: str,
    *,
    index: int,
    cell: str,
    program_class: str,
    depth: int,
    parent_family_id: str | None = None,
    relation: str | None = None,
) -> dict[str, object]:
    outcomes = [
        {
            "step": step,
            "opcode": step % 4,
            "semantic_action": [0, 1, 2],
            "action_rank": 3,
            "quartile": min(3, (4 * step) // depth) + 1,
            "correct": True,
        }
        for step in range(depth)
    ]
    return {
        "family_id": family_id,
        "cluster_family_id": parent_family_id or family_id,
        "parent_family_id": parent_family_id,
        "relation": relation,
        "expected_trace_equal": True if parent_family_id else None,
        "expected_terminal_equal": True if parent_family_id else None,
        "observed_trace_equal": True if parent_family_id else None,
        "observed_terminal_equal": True if parent_family_id else None,
        "relation_correct": True if parent_family_id else None,
        "factorial_cell": cell,
        "shift_order": cell.count("h"),
        "program_class": program_class,
        "depth": depth,
        "renderer": index % 16,
        "packet_valid": True,
        "cards_exact": True,
        "independent_binding_exact": True,
        "initial_exact": True,
        "stop_exact": True,
        "opcode_schedule_exact": True,
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
        "active_step_outcomes": outcomes,
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
                parent_family_id=str(parent["family_id"]),
                relation="post_stop_poison",
            )
        )
    return rows


def _commitment(seed: int, arm: str, dataset: str, rows: int) -> dict[str, object]:
    return {
        "schema": "r12_ctaa_v2_raw_evidence_receipt_v2",
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
    return gates._recompute_scores(rows)


def _resource_profile() -> dict[str, object]:
    shared = {
        "trunk_checkpoint_sha256": _digest("base"),
        "qualified_compiler_checkpoint_sha256": _digest("qualified"),
        "compiler_initial_adapter_sha256": _digest("compiler-adapter"),
        "tokenizer_sha256": _digest("tokenizer"),
        "compiler_training_source_sha256": _digest("compiler-training"),
        "atomic_training_source_sha256": _digest("atomic-training"),
        "closure_training_source_sha256": _digest("closure-training"),
        "curriculum_selection_plan_sha256": _digest("curriculum-plan"),
        "admission_device": "cuda:0",
    }
    assert set(shared) == set(SHARED_BINDING_KEYS) | {"admission_device"}
    bindings = {
        arm: {
            **shared,
            "core_checkpoint_sha256": _digest(f"resource-core:{arm}"),
            "core_kind": arm,
        }
        for arm in PROFILE_ARMS
    }
    measurements = []
    sequence = 0
    for arm in PROFILE_ARMS:
        for phase in PROFILE_PHASES:
            for depth in PROFILE_DEPTHS:
                sequence += 1
                elapsed_ns = 1_000_000 + sequence
                row: dict[str, object] = {
                    "schema": OBSERVATION_SCHEMA,
                    "arm": arm,
                    "phase": phase,
                    "active_depth": depth,
                    "device": "cpu" if phase == "curriculum_selection" else "cuda:0",
                    "batch_size": 64,
                    "repeats": 5,
                    "warmup_count": 3,
                    "elapsed_ns": elapsed_ns,
                    "milliseconds_per_iteration": elapsed_ns / 5 / 1_000_000.0,
                    "rows_per_second": 64 * 5 * 1_000_000_000.0 / elapsed_ns,
                    "peak_allocated_bytes": (
                        0 if phase == "curriculum_selection" else 10_000 + sequence
                    ),
                    "work_units_per_iteration": depth,
                    "bindings": bindings[arm],
                }
                row["observation_sha256"] = _observation_digest(row)
                measurements.append(row)
    comparisons = build_matched_arm_comparisons(
        measurements,
        expected_bindings=bindings,
    )
    runtime = {
        arm: {
            str(depth): next(
                row
                for row in measurements
                if row["arm"] == arm
                and row["phase"] == "inference"
                and row["active_depth"] == depth
            )
            for depth in PROFILE_DEPTHS
        }
        for arm in PROFILE_ARMS
    }
    return {
        "schema": gates.RESOURCE_PROFILE_SCHEMA,
        "base_sha256": _digest("base"),
        "base_step": 300_000,
        "qualified_compiler_sha256": _digest("qualified"),
        "qualified_memory_tensors": 63,
        "artifact_bindings": bindings,
        "parameter_ledger": {
            "trunk": 125_081_664,
            "compiler_adapter": 12_800_527,
            "core": 107_753,
            "total": 137_989_944,
            "headroom": 12_010_055,
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
            "hard_packet_bytes_per_row": 60,
            "semantic_recurrent_state_bytes": 3,
            "implementation_recurrent_state_int64_bytes": 24,
            "halt_state_bytes": 1,
            "matched_across_arms": True,
        },
        "evaluation_charge": {
            "dual_route_core_calls_per_row": 123,
            "charged_core_flops_per_row": 26_516_832,
            "note": "test matched route charge",
        },
        "runtime": runtime,
        "measurements": measurements,
        "matched_arm_comparisons": comparisons,
        "required_phases": list(PROFILE_PHASES),
        "profile_depths": list(PROFILE_DEPTHS),
        "board_seed_generated": False,
        "oracle_access": 0,
        "all_static_gates_pass": True,
        "all_resource_gates_pass": True,
        "all_gates_pass": True,
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
            "compiler_adapter": 12_800_527,
            "core": 107_753,
            "total": 137_989_944,
            "headroom": 12_010_055,
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
def current_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
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
    signing_key = Ed25519PrivateKey.from_private_bytes(b"g" * 32)
    registry_path = tmp_path / "access.jsonl"
    board_sha256 = _digest("board-tree")
    run_contract_sha256 = _digest("run-contract")
    run_contract_path = tmp_path / "run-contract.json"
    _write(
        run_contract_path,
        {
            "fixture": "run-contract",
            "manifest_sha256": _digest("manifest"),
            "board_sha256": board_sha256,
            "run_plan_sha256": _digest("run-plan"),
            "run_contract_sha256": run_contract_sha256,
            "training_seeds": seeds,
        },
    )
    runtime_bundle_path = tmp_path / "runtime-bundle.json"
    _write_canonical(runtime_bundle_path, {"fixture": "runtime-bundle"})
    runtime_bundle_sha256 = sha256_file(runtime_bundle_path)
    runtime_execution_set_path = tmp_path / "runtime-execution-set.json"
    _write_canonical(runtime_execution_set_path, {"fixture": "runtime-execution-set"})
    assessment_source_bundle_path = tmp_path / "assessor.pyz"
    assessment_source_bundle_path.write_bytes(b"fixture-assessor-bundle\n")
    assessment_source_bundle_path.chmod(0o444)
    python_executable_path = tmp_path / "python"
    bwrap_executable_path = tmp_path / "bwrap"
    python_executable_path.write_bytes(b"fixture-python\n")
    bwrap_executable_path.write_bytes(b"fixture-bwrap\n")
    python_executable_path.chmod(0o555)
    bwrap_executable_path.chmod(0o555)
    source_manifest = {
        "bundle_sha256": sha256_file(assessment_source_bundle_path),
        "python_interpreter": {"sha256": _digest("python")},
        "bwrap_executable": {"sha256": _digest("bwrap")},
    }
    assessment_source_manifest_path = tmp_path / "assessor.manifest.json"
    _write_canonical(assessment_source_manifest_path, source_manifest)
    execution_set_file_sha256 = sha256_file(runtime_execution_set_path)
    execution_set_sha256 = _digest("runtime-execution-set")
    monkeypatch.setattr(
        gates,
        "read_runtime_execution_set_with_replay",
        lambda *_args, **_kwargs: (
            {
                "run_contract_sha256": run_contract_sha256,
                "execution_set_sha256": execution_set_sha256,
            },
            execution_set_file_sha256,
        ),
    )
    monkeypatch.setattr(
        gates,
        "validate_assessment_source_bundle",
        lambda **_kwargs: source_manifest,
    )
    monkeypatch.setattr(
        gates,
        "_validate_loaded_runtime_bundle_with_replay",
        lambda _value, _path, **_kwargs: {
            "partition": "development",
            "manifest_sha256": _digest("manifest"),
            "run_contract_sha256": run_contract_sha256,
            "bundle_sha256": _digest("logical-runtime-bundle"),
            "entries": [],
        },
    )
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_receipt = build_receipt(
        source_commit="a" * 40,
        manifest_sha256=_digest("manifest"),
        gate_source_sha256=sha256_file(Path(gates.__file__)),
        statistics_source_sha256=sha256_file(
            Path(gates.__file__).with_name("ctaa_gate_statistics.py")
        ),
        beacon=_beacon(),
    )
    _write(bootstrap_path, bootstrap_receipt)
    statistical_gate_spec_path = tmp_path / "statistical-gate-spec.json"
    statistical_gate_spec_file_sha256 = write_signed_statistical_gate_spec(
        statistical_gate_spec_path,
        bindings=StatisticalGateBindings(
            manifest_sha256=_digest("manifest"),
            board_sha256=board_sha256,
            run_plan_sha256=_digest("run-plan"),
            run_contract_sha256=run_contract_sha256,
            runtime_bundle_file_sha256=runtime_bundle_sha256,
            runtime_bundle_sha256=_digest("logical-runtime-bundle"),
            runtime_execution_set_file_sha256=execution_set_file_sha256,
            runtime_execution_set_sha256=execution_set_sha256,
            assessment_source_bundle_sha256=sha256_file(
                assessment_source_bundle_path
            ),
            assessment_source_manifest_sha256=sha256_file(
                assessment_source_manifest_path
            ),
            bootstrap_seed_receipt_sha256=sha256_file(bootstrap_path),
            bootstrap_seed=bootstrap_receipt["bootstrap_seed"],
            training_seeds=tuple(seeds),
        ),
        signing_key=signing_key,
    )
    gate_spec_sha256 = json.loads(
        statistical_gate_spec_path.read_text()
    )["gate_spec_sha256"]
    assessment_path = tmp_path / "assessment.json"
    assessment_claim_path = tmp_path / "assessment-claim.json"
    public_key = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    claim_payload = {
        "schema": gates.ASSESSMENT_CLAIM_SCHEMA,
        "registry_id": "gate-test-registry",
        "access_id": "development-access",
        "spend_event_id": "development-spend",
        "commit_event_id": "development-assessment",
        "partition": "development",
        "manifest_sha256": _digest("manifest"),
        "board_sha256": board_sha256,
        "run_plan_sha256": _digest("run-plan"),
        "run_contract_sha256": run_contract_sha256,
        "bootstrap_seed_receipt_sha256": sha256_file(bootstrap_path),
        "bootstrap_seed": bootstrap_receipt["bootstrap_seed"],
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "execution_set_file_sha256": execution_set_file_sha256,
        "execution_set_sha256": execution_set_sha256,
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
        "assessment_source_bundle_sha256": sha256_file(assessment_source_bundle_path),
        "assessment_source_manifest_sha256": sha256_file(
            assessment_source_manifest_path
        ),
        "python_interpreter_sha256": _digest("python"),
        "bwrap_executable_sha256": _digest("bwrap"),
        "expected_previous_hash": access_registry.GENESIS_PREVIOUS_HASH,
        "assessment_output": str(assessment_path.absolute()),
        "assessor_argv_sha256": _digest("assessor-argv"),
        "signing_public_key": public_key.hex(),
    }
    claim = {
        "payload": claim_payload,
        "signature": signing_key.sign(
            access_registry.canonical_json_bytes(claim_payload)
        ).hex(),
    }
    assessment_claim_path.write_bytes(
        access_registry.canonical_json_bytes(claim) + b"\n"
    )
    assessment_claim_path.chmod(0o444)
    assessment_claim_sha256 = sha256_file(assessment_claim_path)
    spend_receipt = access_registry.append_access_spend(
        registry_path,
        signing_key=signing_key,
        registry_id="gate-test-registry",
        event_id="development-spend",
        access_id="development-access",
        partition="development",
        manifest_sha256=_digest("manifest"),
        board_sha256=board_sha256,
        run_contract_sha256=run_contract_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
        assessment_claim_sha256=assessment_claim_sha256,
        bootstrap_seed_receipt_sha256=sha256_file(bootstrap_path),
        bootstrap_seed=bootstrap_receipt["bootstrap_seed"],
        statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
        gate_spec_sha256=gate_spec_sha256,
        expected_previous_hash=access_registry.GENESIS_PREVIOUS_HASH,
    )
    spend_head_path = tmp_path / "spend-head.json"
    _write(spend_head_path, spend_receipt)
    spend_head_path.chmod(0o444)
    spend_event = access_registry.verify_registry_events(
        registry_path,
        signing_key.public_key(),
        expected_head_receipt=spend_receipt,
    )[-1]
    assessment = {
        "schema": gates.ASSESSMENT_SCHEMA,
        "partition": "development",
        "manifest_sha256": _digest("manifest"),
        "access": {
            "schema": "r12_ctaa_v2_assessment_access_v7",
            "registry_id": "gate-test-registry",
            "registry_head_receipt_sha256": sha256_file(spend_head_path),
            "registry_head_entry_hash": spend_event.entry_hash,
            "access_event_payload_sha256": hashlib.sha256(
                spend_event.canonical_payload
            ).hexdigest(),
            "access_id": "development-access",
            "partition": "development",
            "manifest_sha256": _digest("manifest"),
            "board_sha256": board_sha256,
            "run_contract_sha256": run_contract_sha256,
            "runtime_bundle_sha256": runtime_bundle_sha256,
            "assessment_claim_sha256": assessment_claim_sha256,
            "execution_set_file_sha256": execution_set_file_sha256,
            "execution_set_sha256": execution_set_sha256,
            "bootstrap_seed_receipt_sha256": sha256_file(bootstrap_path),
            "bootstrap_seed": bootstrap_receipt["bootstrap_seed"],
            "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
            "gate_spec_sha256": gate_spec_sha256,
            "access": 1,
        },
        "oracle_sha256": {name: _digest(f"oracle:{name}") for name in runs},
        "runs": runs,
        "capability_gate_computed": False,
    }
    _write(assessment_path, assessment)
    commit_receipt = access_registry.append_assessment_commit(
        registry_path,
        signing_key=signing_key,
        registry_id="gate-test-registry",
        event_id="development-assessment",
        access_id="development-access",
        assessment_sha256=sha256_file(assessment_path),
        statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
        gate_spec_sha256=gate_spec_sha256,
        expected_previous_hash=spend_event.entry_hash,
        expected_head_receipt=spend_receipt,
    )
    commit_head_path = tmp_path / "assessment-head.json"
    _write(commit_head_path, commit_receipt)
    commit_head_path.chmod(0o444)
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
        "assessment_claim_path": assessment_claim_path,
        "access_registry_path": registry_path,
        "access_spend_head_receipt_path": spend_head_path,
        "assessment_commit_head_receipt_path": commit_head_path,
        "registry_verification_key": public_key,
        "signing_key": signing_key,
        "bootstrap_path": bootstrap_path,
        "run_contract_path": run_contract_path,
        "runtime_bundle_path": runtime_bundle_path,
        "runtime_program_source_path": runtime_bundle_path,
        "runtime_query_source_path": runtime_bundle_path,
        "runtime_tokenizer_path": runtime_bundle_path,
        "runtime_execution_set_path": runtime_execution_set_path,
        "assessment_source_bundle_path": assessment_source_bundle_path,
        "assessment_source_manifest_path": assessment_source_manifest_path,
        "statistical_gate_spec_path": statistical_gate_spec_path,
        "python_executable_path": python_executable_path,
        "bwrap_executable_path": bwrap_executable_path,
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
        assessment_claim_path=inputs["assessment_claim_path"],
        access_registry_path=inputs["access_registry_path"],
        access_spend_head_receipt_path=inputs["access_spend_head_receipt_path"],
        assessment_commit_head_receipt_path=inputs[
            "assessment_commit_head_receipt_path"
        ],
        registry_verification_key=inputs["registry_verification_key"],
        finite_audit_paths=inputs["finite_paths"],
        resource_profile_path=inputs["resource_path"],
        capacity_audit_path=inputs["capacity_path"],
        immutable_preflight_path=inputs["immutable_path"],
        bootstrap_seed_receipt_path=inputs["bootstrap_path"],
        run_contract_path=inputs["run_contract_path"],
        runtime_bundle_path=inputs["runtime_bundle_path"],
        runtime_program_source_path=inputs["runtime_program_source_path"],
        runtime_query_source_path=inputs["runtime_query_source_path"],
        runtime_tokenizer_path=inputs["runtime_tokenizer_path"],
        runtime_execution_set_path=inputs["runtime_execution_set_path"],
        assessment_source_bundle_path=inputs["assessment_source_bundle_path"],
        assessment_source_manifest_path=inputs["assessment_source_manifest_path"],
        statistical_gate_spec_path=inputs["statistical_gate_spec_path"],
        python_executable_path=inputs["python_executable_path"],
        bwrap_executable_path=inputs["bwrap_executable_path"],
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
    assert audit["unresolved_contracts"] == list(gates.UNRESOLVED_CONTRACTS)
    assert audit["committed_bootstrap_seed_accepted"] is True
    output = current_contract["tmp_path"] / "must-not-exist.json"
    with pytest.raises(
        gates.UnresolvedContractError, match="contract is unresolved"
    ) as caught:
        gates.evaluate_advancement_gates(
            assessment_path=current_contract["assessment_path"],
            assessment_claim_path=current_contract["assessment_claim_path"],
            access_registry_path=current_contract["access_registry_path"],
            access_spend_head_receipt_path=current_contract[
                "access_spend_head_receipt_path"
            ],
            assessment_commit_head_receipt_path=current_contract[
                "assessment_commit_head_receipt_path"
            ],
            registry_verification_key=current_contract["registry_verification_key"],
            finite_audit_paths=current_contract["finite_paths"],
            resource_profile_path=current_contract["resource_path"],
            capacity_audit_path=current_contract["capacity_path"],
            immutable_preflight_path=current_contract["immutable_path"],
            bootstrap_seed_receipt_path=current_contract["bootstrap_path"],
            run_contract_path=current_contract["run_contract_path"],
            runtime_bundle_path=current_contract["runtime_bundle_path"],
            runtime_program_source_path=current_contract["runtime_program_source_path"],
            runtime_query_source_path=current_contract["runtime_query_source_path"],
            runtime_tokenizer_path=current_contract["runtime_tokenizer_path"],
            runtime_execution_set_path=current_contract["runtime_execution_set_path"],
            assessment_source_bundle_path=current_contract[
                "assessment_source_bundle_path"
            ],
            assessment_source_manifest_path=current_contract[
                "assessment_source_manifest_path"
            ],
            statistical_gate_spec_path=current_contract[
                "statistical_gate_spec_path"
            ],
            python_executable_path=current_contract["python_executable_path"],
            bwrap_executable_path=current_contract["bwrap_executable_path"],
            output_path=output,
        )
    assert caught.value.audit["all_advancement_gates_pass"] is False
    assert not output.exists()


def test_signed_assessment_claim_signature_is_independently_verified(
    current_contract: dict[str, object],
) -> None:
    claim_path = Path(current_contract["assessment_claim_path"])
    claim = json.loads(claim_path.read_text())
    claim["signature"] = "00" * 64
    _write_canonical(claim_path, claim)
    with pytest.raises(ValueError, match="claim signature"):
        _audit(current_contract)


def test_statistical_gate_spec_substitution_is_rejected_by_final_gate(
    current_contract: dict[str, object],
) -> None:
    path = Path(current_contract["statistical_gate_spec_path"])
    signing_key = current_contract["signing_key"]
    assert isinstance(signing_key, Ed25519PrivateKey)
    original = json.loads(path.read_text())
    bindings = dict(original["payload"]["bindings"])
    bindings["board_sha256"] = _digest("substituted-board")
    path.unlink()
    write_signed_statistical_gate_spec(
        path,
        bindings=bindings,
        signing_key=signing_key,
    )
    with pytest.raises(ValueError, match="statistical gate specification"):
        _audit(current_contract)


def test_execution_set_substitution_is_rejected_by_final_gate(
    current_contract: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gates,
        "read_runtime_execution_set_with_replay",
        lambda *_args, **_kwargs: (
            {
                "run_contract_sha256": current_contract["assessment"]["access"][
                    "run_contract_sha256"
                ],
                "execution_set_sha256": current_contract["assessment"]["access"][
                    "execution_set_sha256"
                ],
            },
            _digest("substituted-execution-set-file"),
        ),
    )
    with pytest.raises(ValueError, match="execution set binding"):
        _audit(current_contract)


def test_assessor_source_bundle_substitution_is_rejected_by_final_gate(
    current_contract: dict[str, object],
) -> None:
    path = Path(current_contract["assessment_source_bundle_path"])
    path.chmod(0o644)
    path.write_bytes(b"substituted-assessor-bundle\n")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="assessor source binding"):
        _audit(current_contract)


def test_signed_assessment_claim_binds_the_exact_run_plan(
    current_contract: dict[str, object],
) -> None:
    run_contract_path = Path(current_contract["run_contract_path"])
    _write(
        run_contract_path,
        {"fixture": "run-contract", "run_plan_sha256": _digest("substituted-plan")},
    )
    with pytest.raises(ValueError, match="claim run-plan binding"):
        _audit(current_contract)


def test_public_api_accepts_neither_sidecar_metadata_nor_bootstrap_seed() -> None:
    parameters = signature(gates.evaluate_advancement_gates).parameters
    assert "metadata_path" not in parameters
    assert "bootstrap_seed" not in parameters
    assert "bootstrap_seed_receipt_path" in parameters
    assert "assessment_claim_path" in parameters
    assert "family_annotations" not in parameters


@pytest.mark.parametrize(
    "field", ["action_strata", "rank_strata", "oracle_label", "bootstrap_seed"]
)
def test_rejects_forged_family_mappings_and_labels(
    current_contract: dict[str, object], field: str
) -> None:
    run = next(iter(current_contract["assessment"]["runs"].values()))
    run["scores"]["family_scores"][0][field] = "forged"
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="forged labels"):
        _audit(current_contract)


@pytest.mark.parametrize(
    "field", ["bootstrap_seed", "advancement_metadata", "all_development_gates_pass"]
)
def test_rejects_outcome_selected_top_level_metadata(
    current_contract: dict[str, object], field: str
) -> None:
    current_contract["assessment"][field] = 7 if field == "bootstrap_seed" else True
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="outcome-aware metadata"):
        _audit(current_contract)


def test_rejects_outcome_selected_bootstrap_seed_inside_scores(
    current_contract: dict[str, object],
) -> None:
    run = next(iter(current_contract["assessment"]["runs"].values()))
    run["scores"]["bootstrap_seed"] = 123
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="forged metadata"):
        _audit(current_contract)


def test_missing_finite_receipt_fails_closed(
    current_contract: dict[str, object],
) -> None:
    current_contract["finite_paths"].pop()
    with pytest.raises(ValueError, match="exactly twenty"):
        _audit(current_contract)


@pytest.mark.parametrize("receipt", ["resource", "capacity", "immutable", "bootstrap"])
def test_missing_source_receipt_fails_closed(
    current_contract: dict[str, object], receipt: str
) -> None:
    Path(current_contract[f"{receipt}_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        _audit(current_contract)


def test_writable_gate_input_is_rejected(current_contract: dict[str, object]) -> None:
    Path(current_contract["resource_path"]).chmod(0o644)
    with pytest.raises(ValueError, match="single-link immutable"):
        _audit(current_contract)


def test_symlinked_gate_input_is_rejected(current_contract: dict[str, object]) -> None:
    path = Path(current_contract["capacity_path"])
    backing = path.with_name("capacity-backing.json")
    path.rename(backing)
    path.symlink_to(backing.name)
    with pytest.raises(ValueError, match="single-link immutable"):
        _audit(current_contract)


def test_gate_reader_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    path = real / "input.json"
    path.write_text("{}\n")
    path.chmod(0o444)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="parent is missing or symlinked"):
        gates._read_file_once(linked / path.name, "test input")


def test_hardlinked_gate_input_is_rejected(current_contract: dict[str, object]) -> None:
    path = Path(current_contract["assessment_path"])
    path.with_name("assessment-alias.json").hardlink_to(path)
    with pytest.raises(ValueError, match="single-link immutable"):
        _audit(current_contract)


def test_substituted_spend_receipt_is_rejected(
    current_contract: dict[str, object],
) -> None:
    spend = Path(current_contract["access_spend_head_receipt_path"])
    commit = Path(current_contract["assessment_commit_head_receipt_path"])
    spend.chmod(0o644)
    spend.write_bytes(commit.read_bytes())
    spend.chmod(0o444)
    with pytest.raises(ValueError, match="retained spend receipt"):
        _audit(current_contract)


@pytest.mark.parametrize(
    ("path_key", "label"),
    [
        ("access_spend_head_receipt_path", "access-spend head receipt"),
        ("assessment_commit_head_receipt_path", "assessment-commit head receipt"),
        ("bootstrap_path", "bootstrap seed receipt"),
    ],
)
def test_receipt_decision_uses_exact_captured_bytes_after_path_replacement(
    current_contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    path_key: str,
    label: str,
) -> None:
    target = Path(current_contract[path_key])
    original_raw = target.read_bytes()
    original_reader = gates._read_file_once
    replaced = False

    def replace_after_capture(
        path: Path,
        current_label: str,
        *,
        require_read_only: bool = True,
    ) -> bytes:
        nonlocal replaced
        raw = original_reader(path, current_label, require_read_only=require_read_only)
        if Path(path) == target and current_label == label and not replaced:
            mode = target.stat().st_mode & 0o777
            target.chmod(0o600)
            target.write_bytes(b'{"substituted":true}\n')
            target.chmod(mode)
            replaced = True
        return raw

    monkeypatch.setattr(gates, "_read_file_once", replace_after_capture)
    audit = _audit(current_contract)
    assert replaced is True
    if path_key == "access_spend_head_receipt_path":
        expected = hashlib.sha256(original_raw).hexdigest()
        assert (
            current_contract["assessment"]["access"]["registry_head_receipt_sha256"]
            == expected
        )
    elif path_key == "bootstrap_path":
        assert (
            audit["bootstrap_seed_receipt_sha256"]
            == hashlib.sha256(original_raw).hexdigest()
        )


def test_registry_verification_uses_one_snapshot_across_all_checks(
    current_contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(current_contract["access_registry_path"])
    original_verify = gates.verify_registry
    original_events = gates.verify_registry_events
    observed_paths: list[Path] = []

    def verify_then_replace(path: Path, *args: object, **kwargs: object):
        result = original_verify(path, *args, **kwargs)
        observed_paths.append(Path(path))
        if len(observed_paths) == 1:
            source.write_bytes(b'{"substituted":true}\n')
        return result

    def observe_events(path: Path, *args: object, **kwargs: object):
        observed_paths.append(Path(path))
        return original_events(path, *args, **kwargs)

    monkeypatch.setattr(gates, "verify_registry", verify_then_replace)
    monkeypatch.setattr(gates, "verify_registry_events", observe_events)
    _audit(current_contract)
    assert len(observed_paths) == 3
    assert len(set(observed_paths)) == 1
    assert observed_paths[0] != source


def test_snapshot_canonicalizes_trusted_symlinked_temp_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = tmp_path / "real-temp"
    real_root.mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(gates.tempfile, "gettempdir", lambda: str(alias))

    with gates._immutable_snapshot(b"captured\n", "snapshot.json") as path:
        assert path.read_bytes() == b"captured\n"
        assert path.parent.parent == real_root
        assert alias not in path.parents


def test_runtime_bundle_replay_and_hash_use_the_same_captured_bytes(
    current_contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Path(current_contract["runtime_bundle_path"])
    original_raw = target.read_bytes()
    for key, name in (
        ("runtime_program_source_path", "program-source.jsonl"),
        ("runtime_query_source_path", "query-source.jsonl"),
        ("runtime_tokenizer_path", "tokenizer.json"),
    ):
        source = Path(current_contract["tmp_path"]) / name
        source.write_bytes(b'{"fixture":true}\n')
        source.chmod(0o444)
        current_contract[key] = source

    original_reader = gates._read_file_once
    original_validator = gates._validate_loaded_runtime_bundle_with_replay
    observed_bundle: dict[str, object] = {}

    def replace_after_capture(
        path: Path,
        label: str,
        *,
        require_read_only: bool = True,
    ) -> bytes:
        raw = original_reader(path, label, require_read_only=require_read_only)
        if Path(path) == target and label == "runtime bundle":
            target.chmod(0o600)
            target.write_bytes(b'{"substituted":true}\n')
            target.chmod(0o444)
        return raw

    def observe_bundle(
        value: dict[str, object], path: Path, **kwargs: object
    ) -> dict[str, object]:
        observed_bundle.update(value)
        return original_validator(value, path, **kwargs)

    monkeypatch.setattr(gates, "_read_file_once", replace_after_capture)
    monkeypatch.setattr(
        gates, "_validate_loaded_runtime_bundle_with_replay", observe_bundle
    )
    audit = _audit(current_contract)
    assert observed_bundle == {"fixture": "runtime-bundle"}
    assert audit["runtime_bundle_sha256"] == hashlib.sha256(original_raw).hexdigest()


@pytest.mark.parametrize(
    ("path_key", "label", "audit_key"),
    [
        ("resource_path", "resource profile", "resource_profile_sha256"),
        ("capacity_path", "capacity audit", "capacity_audit_sha256"),
        ("immutable_path", "immutable preflight", "immutable_preflight_sha256"),
    ],
)
def test_report_hash_uses_the_same_bytes_that_were_validated(
    current_contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    path_key: str,
    label: str,
    audit_key: str,
) -> None:
    target = Path(current_contract[path_key])
    original_raw = target.read_bytes()
    original_reader = gates._read_file_once

    def replace_after_capture(
        path: Path,
        current_label: str,
        *,
        require_read_only: bool = True,
    ) -> bytes:
        raw = original_reader(path, current_label, require_read_only=require_read_only)
        if Path(path) == target and current_label == label:
            target.chmod(0o600)
            target.write_bytes(b'{"substituted":true}\n')
            target.chmod(0o444)
        return raw

    monkeypatch.setattr(gates, "_read_file_once", replace_after_capture)
    audit = _audit(current_contract)
    assert audit[audit_key] == hashlib.sha256(original_raw).hexdigest()


def test_finite_audits_are_mapped_by_core_hash_not_caller_labels(
    current_contract: dict[str, object],
) -> None:
    current_contract["finite_paths"].reverse()
    audit = _audit(current_contract)
    mapped = audit["finite_audits"]
    assert len(mapped) == 20
    assert {entry["seed"] for entry in mapped.values()} == set(
        current_contract["seeds"]
    )


def test_duplicate_or_foreign_finite_core_receipt_is_rejected(
    current_contract: dict[str, object],
) -> None:
    current_contract["finite_paths"][-1] = current_contract["finite_paths"][0]
    with pytest.raises(ValueError, match="duplicated"):
        _audit(current_contract)


def test_forged_finite_pass_bit_is_recomputed_and_rejected(
    current_contract: dict[str, object],
) -> None:
    path = current_contract["finite_paths"][0]
    value = current_contract["finite_values"][path]
    value["axes"]["development"]["atomic_exact"] = 0.5
    # Deliberately leave producer gate bits true.
    _write(path, value)
    with pytest.raises(ValueError, match="pass bit is forged"):
        _audit(current_contract)


def test_honest_finite_failure_is_audited_but_cannot_authorize(
    current_contract: dict[str, object],
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
    with pytest.raises(ValueError, match="family ID identity|provenance"):
        _audit(current_contract)


def test_pooled_relation_percentages_must_match_family_provenance(
    current_contract: dict[str, object],
) -> None:
    for run in current_contract["assessment"]["runs"].values():
        if run["dataset"] == "intervention":
            run["scores"]["intervention_relation_correct"] = {
                "forged_relation": 1.0,
                "post_stop_poison": 1.0,
            }
    _rewrite_assessment(current_contract)
    with pytest.raises(ValueError, match="aggregates differ"):
        _audit(current_contract)
