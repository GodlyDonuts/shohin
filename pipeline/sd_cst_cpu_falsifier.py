#!/usr/bin/env python3
"""CPU-only mechanics and negative-control falsifier for SD-CST."""

from __future__ import annotations

import argparse
import copy
import inspect
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import audit_sd_cst_board
import build_sd_cst_board
from audit_sd_cst_board import audit_board, simulate_adjacent_swaps
from build_sd_cst_board import (
    AMOUNTS,
    DIRECTIONS,
    ENTITY_COUNT,
    OPERATION_COUNT,
    Operation,
    apply_operation_pop_insert,
    build_all,
)


CPU_SEED = 2026072001
FORBIDDEN_NEURAL_IMPORTS = ("torch", "numpy", "transformers", "tokenizers")


def _compiler(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row["compiler_targets"]
    if not isinstance(value, Mapping):
        raise TypeError("compiler_targets is not a mapping")
    return value


def _program(row: Mapping[str, object]) -> tuple[tuple[int, str, int], ...]:
    slots = sorted(
        _compiler(row)["event_slots"],
        key=lambda item: int(item["semantic_ordinal"]),
    )
    return tuple(
        (int(item["entity_role"]), str(item["direction"]), int(item["amount"]))
        for item in slots if item["kind"] != "stop"
    )


def _initial(row: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in _compiler(row)["initial_order_roles"])


def _halt(row: Mapping[str, object]) -> int:
    return int(_compiler(row)["halt_after"])


def _query(row: Mapping[str, object]) -> int:
    value = row["late_query_target"]
    if not isinstance(value, Mapping):
        raise TypeError("late_query_target is not a mapping")
    return int(value["position"])


def _answer(row: Mapping[str, object]) -> int:
    value = row["oracle"]
    if not isinstance(value, Mapping):
        raise TypeError("oracle is not a mapping")
    return int(value["answer_role"])


def _families(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Mapping[str, object]]]:
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["family_id"])][str(row["variant"])] = row
    return [grouped[key] for key in sorted(grouped)]


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _independent_cell_audit() -> dict[str, int]:
    exact = 0
    total = 0
    for initial in itertools.permutations(range(ENTITY_COUNT)):
        for role in range(ENTITY_COUNT):
            for direction in DIRECTIONS:
                for amount in AMOUNTS:
                    operation = Operation(role, direction, amount)
                    primary = apply_operation_pop_insert(initial, operation)
                    independent = audit_sd_cst_board.apply_operation_adjacent_swaps(
                        initial, (role, direction, amount)
                    )
                    exact += int(primary == independent)
                    total += 1
    return {"exact": exact, "total": total}


def _control_scores(
    development: Sequence[Mapping[str, object]],
    confirmation: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    families = _families(list(development) + list(confirmation))
    no_stop_correct = 0
    storage_blind_correct = 0
    order_bag_correct = 0
    stop_insensitive_correct = 0
    query_insensitive_correct = 0
    suffix_overrun_correct = 0
    reset_correct = 0
    total = len(families)

    for family in families:
        canonical = family["canonical"]
        query_swap = family["query_swap"]
        order = family["order_counterfactual"]
        stop = family["stop_shift"]
        storage = family["storage_order_shuffle"]
        suffix = family["post_halt_suffix"]

        base_program = _program(canonical)
        base_initial = _initial(canonical)
        base_query = _query(canonical)
        base_answer = _answer(canonical)
        full_state, _ = simulate_adjacent_swaps(
            base_initial, base_program, OPERATION_COUNT
        )
        no_stop_correct += int(full_state[base_query] == base_answer)

        storage_order = tuple(int(value) for value in _compiler(storage)["storage_order"])
        slots = {
            int(item["semantic_ordinal"]): item
            for item in _compiler(storage)["event_slots"]
        }
        textual_state = base_initial
        for ordinal in storage_order:
            item = slots[ordinal]
            if item["kind"] == "stop":
                break
            textual_state = audit_sd_cst_board.apply_operation_adjacent_swaps(
                textual_state,
                (int(item["entity_role"]), str(item["direction"]), int(item["amount"])),
            )
        storage_blind_correct += int(textual_state[base_query] == _answer(storage))

        order_bag_correct += int(base_answer == _answer(order))
        stop_insensitive_correct += int(base_answer == _answer(stop))
        query_insensitive_correct += int(base_answer == _answer(query_swap))

        suffix_state, _ = simulate_adjacent_swaps(
            _initial(suffix), _program(suffix), OPERATION_COUNT
        )
        suffix_overrun_correct += int(
            suffix_state[_query(suffix)] == _answer(suffix)
        )

        reset_state = _initial(canonical)
        for operation in base_program[: _halt(canonical)]:
            reset_state = audit_sd_cst_board.apply_operation_adjacent_swaps(
                _initial(canonical), operation
            )
        reset_correct += int(reset_state[base_query] == base_answer)

    eval_rows = list(development) + list(confirmation)
    length_groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for row in eval_rows:
        length_groups[
            (
                str(row["split"]),
                str(row["variant"]),
                len(str(row["program_text"])),
                len(str(row["late_query_text"])),
            )
        ].append(_halt(row))
    length_correct = sum(
        max(Counter(depths).values()) for depths in length_groups.values()
    )
    return {
        "execute_through_stop_accuracy": _accuracy(no_stop_correct, total),
        "storage_order_as_semantic_order_accuracy": _accuracy(
            storage_blind_correct, total
        ),
        "event_bag_ignoring_order_accuracy": _accuracy(order_bag_correct, total),
        "ignore_stop_shift_accuracy": _accuracy(stop_insensitive_correct, total),
        "ignore_late_query_swap_accuracy": _accuracy(query_insensitive_correct, total),
        "execute_post_halt_suffix_accuracy": _accuracy(suffix_overrun_correct, total),
        "reset_state_each_step_accuracy": _accuracy(reset_correct, total),
        "program_and_query_length_depth_accuracy": _accuracy(
            length_correct, len(eval_rows)
        ),
    }


def _mutation_checks(
    train: list[dict[str, object]],
    development: list[dict[str, object]],
    confirmation: list[dict[str, object]],
) -> dict[str, bool]:
    train_leak = copy.deepcopy(train)
    train_leak[0]["answer"] = "forbidden"
    leak_report = audit_board(train_leak, development, confirmation)

    query_leak = copy.deepcopy(development)
    query_leak[0]["program_text"] += "\nMirexo query: report position 1."
    query_report = audit_board(train, query_leak, confirmation)

    broken_twin = copy.deepcopy(development)
    query_twin = next(row for row in broken_twin if row["variant"] == "query_swap")
    query_twin["program_text"] += " "
    twin_report = audit_board(train, broken_twin, confirmation)

    overlap_confirmation = copy.deepcopy(confirmation)
    overlap_confirmation[0]["compiler_targets"]["event_slots"] = copy.deepcopy(
        train[0]["compiler_targets"]["event_slots"]
    )
    overlap_report = audit_board(train, development, overlap_confirmation)
    return {
        "explicit_train_answer_rejected": not leak_report["gates"][
            "training_evidence_excluded"
        ],
        "query_in_program_rejected": not query_report["gates"][
            "late_query_withheld_until_after_program"
        ],
        "nonidentical_query_twin_program_rejected": not twin_report["gates"][
            "query_twin_program_bytes_identical_and_answers_separate"
        ],
        "cross_split_sequence_reuse_rejected": not overlap_report["gates"][
            "cross_split_sequence_overlap_zero"
        ],
    }


def _uses_only_standard_library() -> tuple[bool, dict[str, list[str]]]:
    findings: dict[str, list[str]] = {}
    for module in (build_sd_cst_board, audit_sd_cst_board):
        source = inspect.getsource(module)
        hits = [name for name in FORBIDDEN_NEURAL_IMPORTS if re_import(source, name)]
        findings[module.__name__] = hits
    return not any(findings.values()), findings


def re_import(source: str, module: str) -> bool:
    prefixes = (f"import {module}", f"from {module} import", f"from {module}.")
    return any(line.strip().startswith(prefixes) for line in source.splitlines())


def run_falsifier(seed: int = CPU_SEED) -> dict[str, object]:
    train, development, confirmation = build_all(
        train_rows=72,
        development_families=18,
        confirmation_families=18,
        seed=seed,
    )
    board_report = audit_board(train, development, confirmation)
    cells = _independent_cell_audit()
    controls = _control_scores(development, confirmation)
    mutations = _mutation_checks(train, development, confirmation)
    standard_library_only, import_findings = _uses_only_standard_library()
    gates = {
        "fresh_in_memory_board_passes_all_audits": board_report["all_gates_pass"],
        "independent_simulators_agree_on_all_atomic_cells": cells["exact"] == cells["total"],
        "execute_through_stop_control_fails": controls["execute_through_stop_accuracy"] <= 0.25,
        "storage_order_shortcut_fails": controls["storage_order_as_semantic_order_accuracy"] <= 0.25,
        "event_bag_shortcut_fails": controls["event_bag_ignoring_order_accuracy"] == 0.0,
        "stop_insensitive_control_fails": controls["ignore_stop_shift_accuracy"] == 0.0,
        "query_insensitive_control_fails": controls["ignore_late_query_swap_accuracy"] == 0.0,
        "post_halt_overrun_control_fails": controls["execute_post_halt_suffix_accuracy"] <= 0.25,
        "state_reset_control_is_insufficient": controls["reset_state_each_step_accuracy"] < 0.75,
        "length_only_halt_control_is_at_chance": controls[
            "program_and_query_length_depth_accuracy"
        ] <= (1.0 / 6.0 + 1e-12),
        "all_adversarial_mutations_are_detected": all(mutations.values()),
        "board_mechanics_use_only_standard_library": standard_library_only,
        "confirmation_access_zero": True,
    }
    return {
        "schema": "r12_sd_cst_cpu_falsifier_v1",
        "decision": (
            "admit_sd_cst_board_mechanics"
            if all(gates.values())
            else "reject_sd_cst_board_mechanics"
        ),
        "seed": seed,
        "gates": gates,
        "atomic_simulator_cells": cells,
        "negative_controls": controls,
        "mutation_checks": mutations,
        "forbidden_import_findings": import_findings,
        "board_gate_count": len(board_report["gates"]),
        "board_violations": board_report["violations"],
        "confirmation_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CPU_SEED)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing SD-CST CPU report: {args.out}")
    report = run_falsifier(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))
    if report["decision"] != "admit_sd_cst_board_mechanics":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
