#!/usr/bin/env python3
"""Positive and adversarial tests for the independent SD-CST assessor."""

from __future__ import annotations

import base64
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from assess_sd_cst import (
    CONFIG_SCHEMA,
    EVAL_SCHEMA,
    EXPECTED_VARIANTS,
    PERMUTATIONS,
    SHA256_KEYS,
    THRESHOLD_KEYS,
    AssessmentError,
    _answer,
    _apply_action,
    _execute_program,
    _execute_state_swap,
    assess,
    row_ids_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


def slot(kind: int, role: int, amount: int, *, gold: bool = True) -> dict:
    result = {"kind_id": kind, "entity_role": role, "amount_id": amount}
    if gold:
        result["identity_and_amount_scored"] = kind != 2
    return result


def make_tape(initial: int, actions: list[tuple[int, int, int]], stop_at: int) -> dict:
    action_iter = iter(actions)
    return {
        "initial_state_id": initial,
        "event_slots": [
            slot(2, 0, 0)
            if index == stop_at
            else slot(*next(action_iter))
            for index in range(8)
        ],
    }


def family_tapes(
    rng: random.Random,
    family_index: int,
    depth: int,
    used_terminal_states: set[int],
) -> tuple[dict, dict, dict, dict]:
    action_domain = [
        (kind, role, amount)
        for kind in range(2)
        for role in range(3)
        for amount in range(2)
    ]
    for _ in range(100_000):
        actions = [rng.choice(action_domain) for _ in range(7)]
        canonical = make_tape(family_index, actions, depth)
        canonical_state = _execute_program(
            family_index, canonical["event_slots"], force_alive=False
        )
        if canonical_state in used_terminal_states:
            continue

        reordered_actions = list(actions)
        rng.shuffle(reordered_actions)
        if reordered_actions == actions:
            continue
        reordered = make_tape(family_index, reordered_actions, depth)
        reordered_state = _execute_program(
            family_index, reordered["event_slots"], force_alive=False
        )
        if _answer(reordered_state, 0) == _answer(canonical_state, 0):
            continue

        shifted = make_tape(family_index, actions, 7 - depth)
        shifted_state = _execute_program(
            family_index, shifted["event_slots"], force_alive=False
        )
        if _answer(shifted_state, 0) == _answer(canonical_state, 0):
            continue

        post_actions = list(actions[:depth]) + [
            rng.choice(action_domain) for _ in range(7 - depth)
        ]
        if post_actions[depth:] == actions[depth:]:
            continue
        post = make_tape(family_index, post_actions, depth)
        full_state = _execute_program(
            family_index, post["event_slots"], force_alive=True
        )
        if full_state == canonical_state:
            continue
        used_terminal_states.add(canonical_state)
        return canonical, reordered, shifted, post
    raise AssertionError("could not construct a separating synthetic family")


def predicted_tape(gold: dict) -> dict:
    return {
        "initial_state_id": gold["initial_state_id"],
        "event_slots": [
            {key: value for key, value in item.items() if key != "identity_and_amount_scored"}
            for item in gold["event_slots"]
        ],
    }


def certificates() -> dict:
    actions = []
    for state in range(6):
        for kind in range(2):
            for role in range(3):
                for amount in range(2):
                    actions.append({
                        "state_id": state,
                        "kind_id": kind,
                        "entity_role": role,
                        "amount_id": amount,
                        "predicted_state_id": _apply_action(state, kind, role, amount),
                        "predicted_alive": True,
                    })
    stops = [
        {
            "state_id": state,
            "predicted_state_id": state,
            "predicted_alive": False,
        }
        for state in range(6)
    ]
    action_domain = [
        (kind, role, amount)
        for kind in range(2)
        for role in range(3)
        for amount in range(2)
    ] + [(2, 0, 0)]
    dead = [
        {
            "state_id": state,
            "kind_id": kind,
            "entity_role": role,
            "amount_id": amount,
            "predicted_state_id": state,
            "predicted_alive": False,
        }
        for state in range(6)
        for kind, role, amount in action_domain
    ]
    readers = [
        {
            "state_id": state,
            "query_position": query,
            "predicted_answer_role": PERMUTATIONS[state][query],
        }
        for state in range(6)
        for query in range(3)
    ]
    return {
        "motor_state_action": actions,
        "motor_stop": stops,
        "dead_invariance": dead,
        "reader": readers,
    }


def evidence() -> tuple[dict, dict]:
    rows = []
    canonical_by_family = {}
    rng = random.Random(20260720)
    used_terminal_states: set[int] = set()
    for family_index, depth in enumerate(range(1, 7)):
        family_id = f"family-{family_index}"
        canonical_tape, reordered_tape, stop_tape, post_tape = family_tapes(
            rng, family_index, depth, used_terminal_states
        )
        state_id = _execute_program(
            canonical_tape["initial_state_id"],
            canonical_tape["event_slots"],
            force_alive=False,
        )
        canonical_by_family[family_id] = state_id
        for variant in EXPECTED_VARIANTS:
            tape = {
                "post_halt_suffix": post_tape,
                "order_counterfactual": reordered_tape,
                "stop_shift": stop_tape,
            }.get(variant, canonical_tape)
            row_depth = next(
                index for index, item in enumerate(tape["event_slots"])
                if item["kind_id"] == 2
            )
            query = 1 if variant == "query_swap" else 0
            oracle_state = _execute_program(
                tape["initial_state_id"], tape["event_slots"], force_alive=False
            )
            answer = _answer(oracle_state, query)
            interventions = {}
            if variant == "post_halt_suffix":
                full_state = _execute_program(
                    tape["initial_state_id"], tape["event_slots"], force_alive=True
                )
                interventions["force_alive"] = {
                    "final_state_id": full_state,
                    "answer_role": _answer(full_state, query),
                }
            rows.append({
                "id": f"{family_id}:{variant}",
                "family_id": family_id,
                "variant": variant,
                "depth": row_depth,
                "compiler_gold": copy.deepcopy(tape),
                "compiler_prediction": predicted_tape(tape),
                "late_query_gold": query,
                "late_query_prediction": query,
                "oracle": {"final_state_id": oracle_state, "answer_role": answer},
                "autonomous": {"final_state_id": oracle_state, "answer_role": answer},
                "interventions": interventions,
            })

    canonical_rows = [row for row in rows if row["variant"] == "canonical"]
    for recipient in canonical_rows:
        query = recipient["late_query_gold"]
        selected = None
        for donor in canonical_rows:
            receiver_stop = next(
                index for index, item in enumerate(
                    recipient["compiler_gold"]["event_slots"]
                ) if item["kind_id"] == 2
            )
            donor_stop = next(
                index for index, item in enumerate(
                    donor["compiler_gold"]["event_slots"]
                ) if item["kind_id"] == 2
            )
            for after_step in range(min(receiver_stop, donor_stop)):
                swapped_state = _execute_state_swap(
                    recipient["compiler_gold"], donor["compiler_gold"], after_step,
                )
                if (
                    swapped_state != recipient["oracle"]["final_state_id"]
                    and _answer(swapped_state, query)
                    != recipient["oracle"]["answer_role"]
                ):
                    selected = donor, after_step, swapped_state
                    break
            if selected is not None:
                break
        assert selected is not None
        donor, after_step, swapped_state = selected
        recipient["interventions"]["state_swap"] = {
            "donor_id": donor["id"],
            "after_step": after_step,
            "final_state_id": swapped_state,
            "answer_role": _answer(swapped_state, query),
        }

    poison_payload = base64.b64encode(b"identical discrete payload").decode("ascii")
    poison = [
        {
            "id": row["id"],
            "clean_program_tape_b64": poison_payload,
            "poisoned_program_tape_b64": poison_payload,
            "clean_late_query_b64": poison_payload,
            "poisoned_late_query_b64": poison_payload,
            "clean_rollout_b64": poison_payload,
            "poisoned_rollout_b64": poison_payload,
        }
        for row in rows
    ]
    hashes = {key: format(index + 1, "064x") for index, key in enumerate(SHA256_KEYS)}
    hashes["evaluator_sha256"] = sha256_file(ROOT / "assess_sd_cst.py")
    payload = {
        "schema": EVAL_SCHEMA,
        "protocol": "r12_sd_cst_v1_1",
        "split": "sd_cst_development",
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "confirmation_opened": False,
            "access_ledger": {"sha256": "a" * 64},
        },
        "artifact_hashes": dict(hashes),
        "parameters": {
            "base": 125_081_664,
            "compiler": 9_201_931,
            "motor": 19_206,
            "reader": 835,
            "total": 134_303_636,
            "excluded_trainable_parameters": 0,
            "complete_system": True,
        },
        "rows": rows,
        "source_poison": poison,
        "certificates": certificates(),
        "controls": {
            "shuffled_tape": {"cases": 48, "correct": 0},
            "source_free": {"cases": 48, "correct": 0},
            "oracle_tape": {"cases": 48, "correct": 48},
        },
    }
    thresholds = {key: 0.95 for key in THRESHOLD_KEYS}
    thresholds["source_poison_bit_identity"] = 1.0
    config = {
        "schema": CONFIG_SCHEMA,
        "expected": {
            "eval_schema": EVAL_SCHEMA,
            "protocol": payload["protocol"],
            "split": payload["split"],
            "row_count": len(rows),
            "family_count": 6,
            "family_size": 8,
            "row_ids_sha256": row_ids_sha256(row["id"] for row in rows),
            "depth_counts": {str(depth): 8 for depth in range(1, 7)},
            "variants": list(EXPECTED_VARIANTS),
        },
        "thresholds": thresholds,
        "controls": {
            "shuffled_tape": {
                "direction": "at_most", "threshold": 0.10, "min_cases": 48,
            },
            "source_free": {
                "direction": "at_most", "threshold": 0.10, "min_cases": 48,
            },
            "oracle_tape": {
                "direction": "at_least", "threshold": 0.99, "min_cases": 48,
            },
        },
        "expected_artifact_hashes": dict(hashes),
        "expected_access_ledger_sha256": "a" * 64,
        "parameter_cap": 150_000_000,
        "confirmation_accesses": 0,
    }
    return payload, config


def test_accepts_complete_exact_evidence():
    payload, config = evidence()
    result = assess(payload, config)
    assert result["all_gates_pass"]
    assert result["confirmation_authorized"]
    assert result["decision"] == "authorize_one_sealed_confirmation"
    assert result["certificates"]["motor_state_action"] == {
        "correct": 72, "total": 72, "accuracy": 1.0,
    }
    assert result["certificates"]["motor_stop"]["correct"] == 6
    assert result["certificates"]["dead_invariance"]["correct"] == 78
    assert result["certificates"]["reader"]["correct"] == 18
    assert result["autonomous"]["per_depth"]["6"]["answer"]["accuracy"] == 1.0
    assert result["causal"]["force_alive_suffix_oracle"]["accuracy"] == 1.0


def test_row_order_does_not_change_assessment():
    payload, config = evidence()
    random.Random(20260720).shuffle(payload["rows"])
    result = assess(payload, config)
    assert result["all_gates_pass"]


def test_below_threshold_is_a_recorded_rejection_not_missing_evidence():
    payload, config = evidence()
    for row in payload["rows"][:3]:
        row["autonomous"]["answer_role"] = (
            row["autonomous"]["answer_role"] + 1
        ) % 3
    result = assess(payload, config)
    assert not result["all_gates_pass"]
    assert not result["confirmation_authorized"]
    assert result["decision"].startswith("reject_sd_cst")
    assert not result["gates"]["autonomous_answer_overall"]["pass"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload, config: payload["rows"].append(copy.deepcopy(payload["rows"][0])), "row count mismatch"),
        (lambda payload, config: payload["rows"].pop(), "row count mismatch"),
        (lambda payload, config: payload["certificates"]["motor_state_action"].pop(), "72-cell product"),
        (lambda payload, config: payload["source_poison"].pop(), "does not cover"),
        (lambda payload, config: payload["custody"].update({"confirmation_accesses": 1}), "confirmation evidence"),
        (lambda payload, config: payload["custody"].update({"development_accesses": 2}), "development exactly once"),
        (lambda payload, config: payload["custody"]["access_ledger"].update({"sha256": "b" * 64}), "ledger does not match"),
        (lambda payload, config: payload["artifact_hashes"].update({"board_sha256": "f" * 64}), "hashes do not match"),
        (lambda payload, config: payload["parameters"].update({"total": 149_999_999}), "does not equal"),
    ],
)
def test_fail_closed_on_missing_duplicate_or_unbound_evidence(mutation, match):
    payload, config = evidence()
    mutation(payload, config)
    with pytest.raises(AssessmentError, match=match):
        assess(payload, config)


def test_duplicate_row_id_is_rejected_even_at_registered_count():
    payload, config = evidence()
    payload["rows"][1]["id"] = payload["rows"][0]["id"]
    with pytest.raises(AssessmentError, match="duplicate evaluation row id"):
        assess(payload, config)


def test_bad_certificate_value_cannot_pass_exact_gate():
    payload, config = evidence()
    record = payload["certificates"]["reader"][0]
    record["predicted_answer_role"] = (record["predicted_answer_role"] + 1) % 3
    result = assess(payload, config)
    assert not result["gates"]["reader_18_of_18"]["pass"]
    assert not result["all_gates_pass"]


def test_causal_pairs_and_source_poison_are_not_tautological():
    payload, config = evidence()
    query_swap = next(row for row in payload["rows"] if row["variant"] == "query_swap")
    suffix_slot = query_swap["compiler_gold"]["event_slots"][-1]
    suffix_slot["amount_id"] = 1 - suffix_slot["amount_id"]
    with pytest.raises(AssessmentError, match="query-swap programs differ"):
        assess(payload, config)

    payload, config = evidence()
    payload["source_poison"][0]["poisoned_rollout_b64"] = base64.b64encode(b"changed").decode("ascii")
    result = assess(payload, config)
    assert not result["gates"]["source_poison_bit_identity"]["pass"]
    assert not result["all_gates_pass"]


def test_incomplete_conditional_denominator_cannot_authorize_confirmation():
    payload, config = evidence()
    query_swap = next(row for row in payload["rows"] if row["variant"] == "query_swap")
    query_swap["late_query_prediction"] = (query_swap["late_query_prediction"] + 1) % 3
    result = assess(payload, config)
    conditional = result["causal"]["query_swap_answer_follow_query_conditional"]
    assert conditional["total"] == 5
    assert conditional["eligible_fraction"] == pytest.approx(5 / 6)
    assert not result["gates"]["query_swap_answer_follow_query_conditional"]["pass"]


def test_cli_hashes_inputs_and_refuses_duplicate_json_keys_without_output(tmp_path):
    payload, config = evidence()
    eval_path = tmp_path / "eval.json"
    config_path = tmp_path / "config.json"
    out_path = tmp_path / "assessment.json"
    eval_path.write_text(json.dumps(payload, sort_keys=True))
    config_path.write_text(json.dumps(config, sort_keys=True))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "assess_sd_cst.py"),
            "--eval", str(eval_path),
            "--config", str(config_path),
            "--out", str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    result = json.loads(out_path.read_text())
    assert len(result["evidence_sha256"]["evaluation"]) == 64
    assert len(result["evidence_sha256"]["gate_config"]) == 64

    bad_eval = tmp_path / "duplicate-key.json"
    bad_out = tmp_path / "must-not-exist.json"
    bad_eval.write_text('{"schema":"a","schema":"b"}')
    refused = subprocess.run(
        [
            sys.executable,
            str(ROOT / "assess_sd_cst.py"),
            "--eval", str(bad_eval),
            "--config", str(config_path),
            "--out", str(bad_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "duplicate JSON object key" in refused.stderr
    assert not bad_out.exists()
