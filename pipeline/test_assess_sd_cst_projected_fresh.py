from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import torch

from assess_sd_cst_projected_mechanics import (
    packet_arm,
    rotate_queries,
    shuffled_packet,
    swap_operand_suffix,
)
from assess_sd_cst_projected_fresh import (
    AssessmentError,
    _bind_auxiliary_artifacts,
    _validate_rows,
    assess,
    packet_exact,
    parse_output,
    parse_packet,
    sha256_file,
)
from build_sd_cst_board import build_all
from build_sd_cst_projected_board import PROTOCOL, _registration
from projected_sd_cst_fresh import (
    PROJECTED_TRAINABLE_NAMES,
    parse_projected_row,
    tape_to_json,
)
from sd_cst import HardLateQuery, HardProgramTape
from train_eval_sd_cst_projected_fresh import (
    CONFIG_SCHEMA,
    CONTROL_ARMS,
    DEVELOPMENT_SPLIT,
    EVALUATION_SCHEMA,
    THRESHOLDS,
    TRAINING_CONTRACT,
    _source_free_packet,
    _uniform_packet,
    safe_mutate_first_active,
    safe_perturb_post_stop,
)
from assess_sd_cst_projected_mechanics import semantic_rollout


def _packet(rows: int = 3):
    return {
        "initial_state": [0, 1, 2][:rows],
        "event_kind": [[0, 2, 0, 0, 0, 0, 0, 0] for _ in range(rows)],
        "event_identity": [[0] * 8 for _ in range(rows)],
        "amount": [[0] * 8 for _ in range(rows)],
        "query": [0] * rows,
    }


def test_packet_parser_and_exactness_score_active_fields_only():
    gold_raw = _packet()
    prediction_raw = _packet()
    prediction_raw["event_identity"][0][1] = 2  # STOP identity is unscored.
    gold = parse_packet(gold_raw, 3, "gold")
    prediction = parse_packet(prediction_raw, 3, "prediction")
    scores = packet_exact(prediction, gold)
    assert bool(scores["packet"].all())
    prediction_raw["event_identity"][0][0] = 2
    prediction = parse_packet(prediction_raw, 3, "prediction")
    scores = packet_exact(prediction, gold)
    assert not bool(scores["packet"][0])


def test_packet_parser_rejects_out_of_range_and_wrong_shape():
    value = _packet()
    value["query"][1] = 3
    with pytest.raises(AssessmentError):
        parse_packet(value, 3, "bad")
    value = _packet()
    value["event_kind"][0].pop()
    with pytest.raises(AssessmentError):
        parse_packet(value, 3, "bad")


def test_semantic_types_are_uint8_packets():
    packet = parse_packet(_packet(), 3, "packet")
    assert isinstance(packet[0], HardProgramTape)
    assert isinstance(packet[1], HardLateQuery)
    assert packet[0].initial_state.dtype == torch.uint8


@pytest.mark.parametrize("bad", [1.0, True, -1])
def test_packet_parser_rejects_json_numeric_coercions(bad):
    value = _packet()
    value["initial_state"][0] = bad
    with pytest.raises(AssessmentError):
        parse_packet(value, 3, "bad")


def test_output_parser_rejects_integer_as_boolean():
    value = {
        "final_state": [0, 1, 2],
        "answer": [0, 1, 2],
        "state_trajectory": [[0] * 8 for _ in range(3)],
        "alive_trajectory": [[1] * 8 for _ in range(3)],
    }
    with pytest.raises(AssessmentError):
        parse_output(value, 3, "bad")


def test_row_evidence_must_reparse_from_committed_raw_source():
    _, development, _ = build_all(
        train_rows=12,
        development_families=6,
        confirmation_families=6,
        seed=891,
    )
    evidence = []
    for raw in development:
        value = asdict(parse_projected_row(raw, DEVELOPMENT_SPLIT))
        value.pop("program_bytes")
        value.pop("query_bytes")
        evidence.append(json.loads(json.dumps(value)))
    registration = _registration(development)
    expected = {
        "row_count": 48,
        "family_count": 6,
        "variants": registration["variants"],
        "depths": [1, 2, 3, 4, 5, 6],
    }
    _validate_rows(evidence, expected, registration)
    evidence[0]["final_state"] = (int(evidence[0]["final_state"]) + 1) % 6
    with pytest.raises(AssessmentError, match="differs from its canonical source"):
        _validate_rows(evidence, expected, registration)


def test_auxiliary_tensor_artifacts_are_content_bound(tmp_path):
    arm = {
        "initial_state": torch.tensor([0, 1, 2], dtype=torch.uint8),
        "event_kind": torch.tensor(
            [[0, 2, 0, 0, 0, 0, 0, 0]] * 3,
            dtype=torch.uint8,
        ),
        "event_identity": torch.zeros((3, 8), dtype=torch.uint8),
        "amount": torch.zeros((3, 8), dtype=torch.uint8),
        "query": torch.zeros(3, dtype=torch.uint8),
        "control": "normal",
        "force_alive": False,
        "state_swap": None,
        "swap_after_step": 0,
    }
    outputs = {
        "final_state": torch.tensor([0, 1, 2], dtype=torch.uint8),
        "answer": torch.tensor([0, 1, 2], dtype=torch.int64),
        "state_trajectory": torch.zeros((3, 8), dtype=torch.uint8),
        "alive_trajectory": torch.ones((3, 8), dtype=torch.bool),
    }
    packets = tmp_path / "packets.pt"
    executor = tmp_path / "executor.pt"
    torch.save(
        {"schema": "r12_sd_cst_hard_packet_bundle_v1", "arms": {"x": arm}}, packets
    )
    torch.save(
        {"schema": "r12_sd_cst_hard_packet_outputs_v1", "outputs": {"x": outputs}},
        executor,
    )
    evaluation = {
        "rows": [{}, {}, {}],
        "artifact_hashes": {
            "hard_packets": sha256_file(packets),
            "executor_outputs": sha256_file(executor),
        },
        "packet_arms": {
            "x": {
                key: value.tolist()
                for key, value in arm.items()
                if isinstance(value, torch.Tensor)
            }
            | {
                "control": "normal",
                "force_alive": False,
                "state_swap": None,
                "swap_after_step": 0,
            }
        },
        "executor_outputs": {
            "x": {key: value.tolist() for key, value in outputs.items()}
        },
    }
    _bind_auxiliary_artifacts(evaluation, packets, executor)
    evaluation["executor_outputs"]["x"]["answer"][0] = 2
    with pytest.raises(AssessmentError, match="content differs"):
        _bind_auxiliary_artifacts(evaluation, packets, executor)


def test_perfect_system_contract_has_no_impossible_frozen_gate():
    _, development, _ = build_all(
        train_rows=96,
        development_families=288,
        confirmation_families=6,
        seed=31917,
    )
    rows = [parse_projected_row(row, DEVELOPMENT_SPLIT) for row in development]
    count = len(rows)
    gold_tape = HardProgramTape(
        torch.tensor([row.initial_state for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_kind for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_identity for row in rows], dtype=torch.uint8),
        torch.tensor([row.amount for row in rows], dtype=torch.uint8),
    )
    gold_query = HardLateQuery(
        torch.tensor(
            [row.query_position for row in rows],
            dtype=torch.uint8,
        )
    )
    uniform_tape, uniform_query = _uniform_packet(count)
    source_free_tape, source_free_query = _source_free_packet(rows)
    state_swap = torch.arange(count - 1, -1, -1, dtype=torch.long)
    compiled_packets = {
        "treatment": (gold_tape, gold_query),
        "row_shuffled_labels": (uniform_tape, uniform_query),
        "consumed_projected": (gold_tape, gold_query),
        "binding_source_free_compiler": (uniform_tape, gold_query),
    }
    packet_inputs = {
        name: packet_arm(tape, query)
        for name, (tape, query) in compiled_packets.items()
    }
    packet_inputs.update(
        {
            "uniform": packet_arm(uniform_tape, uniform_query),
            "source_free_packet": packet_arm(source_free_tape, source_free_query),
            "shuffled_packet": packet_arm(shuffled_packet(gold_tape), gold_query),
            "reset": packet_arm(gold_tape, gold_query, control="reset"),
            "freeze": packet_arm(gold_tape, gold_query, control="freeze"),
            "post_stop_perturbation": packet_arm(
                safe_perturb_post_stop(gold_tape),
                gold_query,
            ),
            "force_alive_post_stop": packet_arm(
                safe_perturb_post_stop(gold_tape),
                gold_query,
                force_alive=True,
            ),
            "operand_suffix_swap": packet_arm(
                swap_operand_suffix(gold_tape, 4),
                gold_query,
            ),
            "query_rotation": packet_arm(gold_tape, rotate_queries(gold_query)),
            "state_swap_after_step_0": packet_arm(
                gold_tape,
                gold_query,
                state_swap=state_swap,
                swap_after_step=0,
            ),
            "initial_state_rotation": packet_arm(
                safe_mutate_first_active(gold_tape, "initial_state"),
                gold_query,
            ),
            "event_kind_flip": packet_arm(
                safe_mutate_first_active(gold_tape, "kind"),
                gold_query,
            ),
            "event_identity_rotation": packet_arm(
                safe_mutate_first_active(gold_tape, "identity"),
                gold_query,
            ),
            "event_amount_flip": packet_arm(
                safe_mutate_first_active(gold_tape, "amount"),
                gold_query,
            ),
        }
    )

    def packet_json(arm):
        return tape_to_json(
            HardProgramTape(
                arm["initial_state"],
                arm["event_kind"],
                arm["event_identity"],
                arm["amount"],
            ),
            HardLateQuery(arm["query"]),
        ) | {
            "control": arm["control"],
            "force_alive": arm["force_alive"],
            "state_swap": (
                arm["state_swap"].tolist() if arm["state_swap"] is not None else None
            ),
            "swap_after_step": arm["swap_after_step"],
        }

    def execute(arm):
        output = semantic_rollout(
            HardProgramTape(
                arm["initial_state"],
                arm["event_kind"],
                arm["event_identity"],
                arm["amount"],
            ),
            HardLateQuery(arm["query"]),
            control=arm["control"],
            state_swap=arm["state_swap"],
            swap_after_step=arm["swap_after_step"],
            force_alive=arm["force_alive"],
        )
        return {
            "final_state": output[0].tolist(),
            "answer": output[1].tolist(),
            "state_trajectory": output[2].tolist(),
            "alive_trajectory": output[3].tolist(),
        }

    evidence = []
    for row in rows:
        value = asdict(row)
        value.pop("program_bytes")
        value.pop("query_bytes")
        evidence.append(json.loads(json.dumps(value)))
    treatment_pointers = {
        "line": [[start for start, _ in row.pointer_ranges] for row in rows],
        "binding": [[start for start, _ in row.binding_ranges] for row in rows],
        "initial_entity": [
            [start for start, _ in row.initial_entity_ranges] for row in rows
        ],
        "event_entity": [
            [start for start, _ in row.event_entity_ranges] for row in rows
        ],
    }
    zero_pointers = {
        "line": [[0] * 9 for _ in rows],
        "binding": [[0] * 3 for _ in rows],
        "initial_entity": [[0] * 3 for _ in rows],
        "event_entity": [[0] * 8 for _ in rows],
    }
    compiled = {
        name: {
            "packet": tape_to_json(tape, query),
            "pointers": treatment_pointers if name == "treatment" else zero_pointers,
            "source_poison_bit_identical": True,
        }
        for name, (tape, query) in compiled_packets.items()
    }
    artifact_hashes = {
        name: name[0] * 64
        for name in (
            "board",
            "checkpoint",
            "parent",
            "execution_core",
            "consumed_projected",
            "evaluator",
            "assessor",
        )
    }
    config = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "expected": {
            "evaluation_schema": EVALUATION_SCHEMA,
            "row_count": count,
            "family_count": 288,
            "family_size": 8,
            "variants": _registration(development)["variants"],
            "depths": [1, 2, 3, 4, 5, 6],
            "arms": list(compiled_packets),
            "control_arms": list(CONTROL_ARMS),
        },
        "registrations": {"development": _registration(development)},
        "thresholds": THRESHOLDS,
        "training_contract": TRAINING_CONTRACT,
        "parameter_caps": {"comparison": 150_000_000, "global": 200_000_000},
        "artifact_hashes": artifact_hashes,
        "split_hashes": {"development": "d" * 64},
        "expected_ledger_sha256": {"development": "e" * 64},
        "source": {"sha256": "s" * 64},
    }
    fit = {
        "updates": 3_000,
        "frozen_parent_unchanged": True,
        "initial_full_state_sha256": "i" * 64,
        "minibatch_order_sha256": "m" * 64,
    }
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "protocol": PROTOCOL,
        "split": DEVELOPMENT_SPLIT,
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "access_ledger": {"sha256": "e" * 64},
            "confirmation_authorization": None,
        },
        "artifact_hashes": artifact_hashes
        | {
            "split": "d" * 64,
            "gate_config": "g" * 64,
            "source_manifest": "s" * 64,
        },
        "parameters": {
            "nominal_complete_system": 146_057_595,
            "trainable": 6_748_897,
            "trainable_names": list(PROJECTED_TRAINABLE_NAMES),
        },
        "training": {
            "contract": TRAINING_CONTRACT,
            "fits": {"treatment": fit, "row_shuffled_labels": fit},
            "row_shuffled_label_mapping_sha256": "a" * 64,
        },
        "rows": evidence,
        "gold_packet": tape_to_json(gold_tape, gold_query),
        "compiled": compiled,
        "packet_arms": {name: packet_json(arm) for name, arm in packet_inputs.items()},
        "executor_outputs": {name: execute(arm) for name, arm in packet_inputs.items()},
        "source_deletion": {
            "program_elements_per_row": 25,
            "query_elements_per_row": 1,
            "dtype": "torch.uint8",
            "program_gpu_tensors_destroyed_before_query_compile": True,
            "host_evaluator_retains_hash_bound_row_evidence": True,
            "separate_typed_executor": True,
            "all_compiler_arms_source_poison_bit_identical": True,
        },
    }
    result = assess(evaluation, config)
    assert result["all_gates_pass"], {
        name: gate for name, gate in result["gates"].items() if not gate["pass"]
    }
