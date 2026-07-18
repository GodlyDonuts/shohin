#!/usr/bin/env python3
"""Independently replay sealed digitwise-factorial-v4 evaluation reports.

This verifier intentionally does not import the evaluator.  It reconstructs
stored parser outputs from retained generations, checks branch and pair fields,
rebuilds every metric group and accounting total, and computes paired factorial
contrasts across arms that share the frozen held-out order.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterable, Mapping


AUDIT = "shohin-digitwise-factorial-v4-full-eval-v2"
REGIME_BUDGETS = {
    "fit_w4": 300,
    "fit_w6": 300,
    "value_ood_w4": 300,
    "value_ood_w6": 300,
    "width_ood_w8": 300,
}
REGIME_WIDTHS = {
    "fit_w4": 4,
    "fit_w6": 6,
    "value_ood_w4": 4,
    "value_ood_w6": 6,
    "width_ood_w8": 8,
}
PAIR_COUNT = sum(REGIME_BUDGETS.values())
BRANCH_COUNT = 2 * PAIR_COUNT
STATE_KEYS = frozenset({"op", "w", "p", "c", "a", "b", "r", "z"})
STATE_RE = re.compile(
    r"(?mi)^\s*(dws:op=(add|sub);w=(\d+);p=(\d+);c=([01]);"
    r"a=(\d+);b=(\d+);r=(\d+);z=([01]))\s*$"
)
ANSWER_RE = re.compile(r"(?mi)^\s*answer=(-?\d+)\s*$")
GENERATION_KEYS = frozenset(
    {
        "text",
        "content_token_ids",
        "sampled_token_ids",
        "prompt_token_count",
        "stop_reason",
    }
)
ROW_KEYS = frozenset(
    {
        "position",
        "prompt",
        "input_state",
        "expected_state",
        "predicted_state",
        "correct",
        "generation",
    }
)
BRANCH_KEYS = frozenset(
    {
        "id",
        "regime",
        "operation",
        "width",
        "terminal_carry_class",
        "expected_answer",
        "initial_state",
        "transition_budget",
        "transition_calls",
        "prefix_exact_length",
        "fully_parseable",
        "state_closed_loop_exact",
        "terminal_transition_exact",
        "terminal_reached",
        "final_prompt_issued",
        "emitted_answer",
        "final_answer_correct",
        "closed_loop_success",
        "first_failure_position",
        "first_failure_reason",
        "emitted_token_count",
        "rows",
        "final_prompt",
        "final_generation",
    }
)
PAIR_KEYS = frozenset(
    {
        "id",
        "regime",
        "operation",
        "width",
        "normal_terminal_carry_class",
        "counterfactual_terminal_carry_class",
        "expected_answer_changed",
        "first_expected_state_divergence_position",
        "first_predicted_state_divergence_position",
        "prediction_diverged_at_expected_position",
        "state_intervention_at_expected_position",
        "both_state_closed_loop_exact",
        "both_final_answers_correct",
        "answer_intervention_success",
        "both_closed_loop_success",
        "normal",
        "counterfactual",
    }
)
BRANCH_METRICS = (
    "terminal_transition_exact",
    "state_closed_loop_exact",
    "final_answer_correct",
    "closed_loop_success",
)
PAIR_METRICS = ("both_state_closed_loop_exact", "both_closed_loop_success")


class ReplayError(ValueError):
    """Raised when retained evidence does not reconstruct exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def strict_int(value: Any, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    return value


def strict_bool(value: Any, label: str) -> bool:
    require(type(value) is bool, f"{label} must be a boolean")
    return value


def finite_tree(value: Any, label: str = "report") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        require(math.isfinite(value), f"{label} contains a non-finite float")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            require(type(key) is str, f"{label} contains a non-string key")
            finite_tree(child, f"{label}.{key}")
        return
    raise ReplayError(f"{label} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    require(type(value) is dict, f"{label} must be an object")
    require(set(value) == expected, f"{label} keys differ")


def value_lsf(digits: str) -> int:
    return sum(int(digit) * (10**index) for index, digit in enumerate(digits))


def validate_state(value: Any, label: str) -> dict[str, Any]:
    exact_keys(value, STATE_KEYS, label)
    require(value["op"] in {"add", "sub"}, f"{label}.op is invalid")
    width = strict_int(value["w"], f"{label}.w")
    position = strict_int(value["p"], f"{label}.p")
    carry = strict_int(value["c"], f"{label}.c")
    terminal = strict_int(value["z"], f"{label}.z")
    require(width > 0 and 0 <= position <= width, f"{label} position is invalid")
    require(carry in {0, 1} and terminal in {0, 1}, f"{label} bit is invalid")
    require(terminal == int(position == width), f"{label} terminal bit is invalid")
    for field in ("a", "b", "r"):
        tape = value[field]
        require(
            type(tape) is str
            and len(tape) == width
            and tape.isascii()
            and tape.isdigit(),
            f"{label}.{field} is invalid",
        )
    require(set(value["r"][position:]) <= {"0"}, f"{label} has a written suffix")
    require(position != 0 or carry == 0, f"{label} initial state carries")
    if value["op"] == "sub":
        require(value_lsf(value["a"]) >= value_lsf(value["b"]), f"{label} is negative")
    return dict(value)


def parse_state(text: Any) -> dict[str, Any] | None:
    matches = STATE_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, op, width, position, carry, left, right, result, terminal = matches[0]
    value = {
        "op": op,
        "w": int(width),
        "p": int(position),
        "c": int(carry),
        "a": left,
        "b": right,
        "r": result,
        "z": int(terminal),
    }
    try:
        return validate_state(value, "parsed_state")
    except ReplayError:
        return None


def parse_answer(text: Any) -> int | None:
    matches = ANSWER_RE.findall(str(text))
    return int(matches[0]) if len(matches) == 1 else None


def apply_microstep(state: Mapping[str, Any]) -> dict[str, Any]:
    current = validate_state(dict(state), "oracle_state")
    require(not current["z"], "oracle cannot step a terminal state")
    position = current["p"]
    left = int(current["a"][position])
    right = int(current["b"][position])
    carry = current["c"]
    if current["op"] == "add":
        total = left + right + carry
        digit, next_carry = total % 10, total // 10
    else:
        total = left - right - carry
        digit, next_carry = (total + 10) % 10, int(total < 0)
    result = list(current["r"])
    result[position] = str(digit)
    next_position = position + 1
    candidate = dict(current)
    candidate.update(
        {
            "p": next_position,
            "c": next_carry,
            "r": "".join(result),
            "z": int(next_position == current["w"]),
        }
    )
    return validate_state(candidate, "oracle_next_state")


def oracle_path(initial: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = dict(initial)
    path: list[dict[str, Any]] = []
    for _ in range(state["w"]):
        state = apply_microstep(state)
        path.append(state)
    return path


def state_answer(state: Mapping[str, Any]) -> int:
    terminal = validate_state(dict(state), "terminal_state")
    require(bool(terminal["z"]), "answer requested before terminal state")
    result = value_lsf(terminal["r"])
    if terminal["op"] == "add":
        return result + terminal["c"] * (10 ** terminal["w"])
    require(terminal["c"] == 0, "terminal subtraction still borrows")
    return result


def verify_generation(value: Any, label: str) -> int:
    exact_keys(value, GENERATION_KEYS, label)
    require(type(value["text"]) is str, f"{label}.text must be a string")
    for field in ("content_token_ids", "sampled_token_ids"):
        require(type(value[field]) is list, f"{label}.{field} must be a list")
        for index, token in enumerate(value[field]):
            require(
                type(token) is int and token >= 0,
                f"{label}.{field}[{index}] must be a token ID",
            )
    require(
        type(value["prompt_token_count"]) is int and value["prompt_token_count"] > 0,
        f"{label}.prompt_token_count is invalid",
    )
    require(type(value["stop_reason"]) is str, f"{label}.stop_reason is invalid")
    return len(value["sampled_token_ids"])


def first_difference(left: list[Any], right: list[Any]) -> int | None:
    require(len(left) == len(right), "paired sequences have unequal lengths")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return index
    return None


def expected_terminal_class(
    path: list[Mapping[str, Any]], initial: Mapping[str, Any]
) -> str:
    before = initial if len(path) == 1 else path[-2]
    after = path[-1]
    return f"{before['c']}{after['c']}"


def verify_branch(branch: Any, label: str) -> dict[str, Any]:
    exact_keys(branch, BRANCH_KEYS, label)
    width = strict_int(branch["width"], f"{label}.width")
    require(branch["regime"] in REGIME_WIDTHS, f"{label}.regime is invalid")
    require(REGIME_WIDTHS[branch["regime"]] == width, f"{label}.width mismatch")
    require(branch["operation"] in {"add", "sub"}, f"{label}.operation is invalid")
    require(type(branch["id"]) is str and branch["id"], f"{label}.id is invalid")
    expected_answer = strict_int(branch["expected_answer"], f"{label}.expected_answer")
    initial = parse_state(branch["initial_state"])
    require(initial is not None, f"{label}.initial_state is malformed")
    require(initial["op"] == branch["operation"], f"{label}.initial operation mismatch")
    require(
        initial["w"] == width and initial["p"] == 0, f"{label}.initial width mismatch"
    )
    expected_path = oracle_path(initial)
    require(
        state_answer(expected_path[-1]) == expected_answer,
        f"{label}.expected answer mismatch",
    )
    rows = branch["rows"]
    require(type(rows) is list and rows, f"{label}.rows must be nonempty")
    transition_budget = strict_int(
        branch["transition_budget"], f"{label}.transition_budget"
    )
    require(transition_budget == width, f"{label}.transition budget mismatch")
    require(len(rows) <= transition_budget, f"{label} has excess transition rows")

    previous: dict[str, Any] | None = initial
    first_failure_position: int | None = None
    first_failure_reason: str | None = None
    prefix_exact_length = 0
    emitted_tokens = 0
    for index, row in enumerate(rows):
        row_label = f"{label}.rows[{index}]"
        exact_keys(row, ROW_KEYS, row_label)
        require(
            strict_int(row["position"], f"{row_label}.position") == index,
            f"{row_label} position mismatch",
        )
        require(type(row["prompt"]) is str, f"{row_label}.prompt must be a string")
        input_state = validate_state(row["input_state"], f"{row_label}.input_state")
        expected = validate_state(row["expected_state"], f"{row_label}.expected_state")
        require(
            previous is not None and input_state == previous,
            f"{row_label} input chain mismatch",
        )
        require(expected["p"] == index + 1, f"{row_label} expected position mismatch")
        require(
            expected["op"] == branch["operation"] and expected["w"] == width,
            f"{row_label} expected identity mismatch",
        )
        require(
            expected == expected_path[index],
            f"{row_label} expected oracle state mismatch",
        )
        generation = row["generation"]
        emitted_tokens += verify_generation(generation, f"{row_label}.generation")
        parsed = parse_state(generation["text"])
        predicted = row["predicted_state"]
        if predicted is not None:
            predicted = validate_state(predicted, f"{row_label}.predicted_state")
        require(
            parsed == predicted,
            f"{row_label} stored parser output differs from generation",
        )
        correct = predicted == expected
        require(
            strict_bool(row["correct"], f"{row_label}.correct") == correct,
            f"{row_label} correctness mismatch",
        )
        if correct and first_failure_position is None:
            prefix_exact_length += 1
        elif first_failure_position is None:
            first_failure_position = index
            first_failure_reason = (
                "malformed_state" if predicted is None else "state_mismatch"
            )
        previous = predicted
        if predicted is None:
            require(
                index == len(rows) - 1, f"{row_label} malformed state did not terminate"
            )

    fully_parseable = len(rows) == transition_budget and all(
        row["predicted_state"] is not None for row in rows
    )
    state_exact = len(rows) == transition_budget and all(row["correct"] for row in rows)
    terminal_reached = previous is not None and bool(previous["z"])
    final_generation = branch["final_generation"]
    final_prompt = branch["final_prompt"]
    emitted_answer: int | None = None
    if terminal_reached:
        require(
            type(final_prompt) is str and final_prompt,
            f"{label}.final_prompt is missing",
        )
        require(final_generation is not None, f"{label}.final_generation is missing")
        emitted_tokens += verify_generation(
            final_generation, f"{label}.final_generation"
        )
        emitted_answer = parse_answer(final_generation["text"])
    else:
        require(
            final_prompt is None and final_generation is None,
            f"{label} issued a nonterminal final prompt",
        )
    final_correct = emitted_answer == expected_answer if terminal_reached else False
    if terminal_reached and first_failure_position is None and not final_correct:
        first_failure_position = transition_budget
        first_failure_reason = (
            "malformed_answer" if emitted_answer is None else "answer_mismatch"
        )
    elif not terminal_reached and first_failure_position is None:
        first_failure_position = transition_budget
        first_failure_reason = "nonterminal_after_budget"
    terminal_exact = len(rows) == transition_budget and bool(rows[-1]["correct"])
    success = state_exact and final_correct
    if success:
        first_failure_position = None
        first_failure_reason = None

    derived = {
        "terminal_carry_class": expected_terminal_class(expected_path, initial),
        "transition_calls": len(rows),
        "prefix_exact_length": prefix_exact_length,
        "fully_parseable": fully_parseable,
        "state_closed_loop_exact": state_exact,
        "terminal_transition_exact": terminal_exact,
        "terminal_reached": terminal_reached,
        "final_prompt_issued": final_generation is not None,
        "emitted_answer": emitted_answer,
        "final_answer_correct": final_correct,
        "closed_loop_success": success,
        "first_failure_position": first_failure_position,
        "first_failure_reason": first_failure_reason,
        "emitted_token_count": emitted_tokens,
    }
    for field, expected in derived.items():
        require(branch[field] == expected, f"{label}.{field} does not replay")
    return dict(branch)


def verify_pair(pair: Any, index: int) -> dict[str, Any]:
    label = f"transcripts[{index}]"
    exact_keys(pair, PAIR_KEYS, label)
    normal = verify_branch(pair["normal"], f"{label}.normal")
    counterfactual = verify_branch(pair["counterfactual"], f"{label}.counterfactual")
    require(pair["id"] == normal["id"], f"{label}.id mismatch")
    for field in ("regime", "operation", "width"):
        require(
            pair[field] == normal[field] == counterfactual[field],
            f"{label}.{field} mismatch",
        )
    normal_initial = parse_state(normal["initial_state"])
    counterfactual_initial = parse_state(counterfactual["initial_state"])
    require(
        normal_initial is not None and counterfactual_initial is not None,
        f"{label} initial state disappeared",
    )
    expected_normal = oracle_path(normal_initial)
    expected_counterfactual = oracle_path(counterfactual_initial)
    expected_divergence = first_difference(expected_normal, expected_counterfactual)
    predicted_normal = [row["predicted_state"] for row in normal["rows"]]
    predicted_counterfactual = [
        row["predicted_state"] for row in counterfactual["rows"]
    ]
    shared = min(len(predicted_normal), len(predicted_counterfactual))
    predicted_divergence = first_difference(
        predicted_normal[:shared], predicted_counterfactual[:shared]
    )
    expected_changed = normal["expected_answer"] != counterfactual["expected_answer"]
    answer_intervention = (
        expected_changed
        and normal["final_answer_correct"]
        and counterfactual["final_answer_correct"]
        and normal["emitted_answer"] != counterfactual["emitted_answer"]
    )
    prediction_at_expected = (
        expected_divergence is not None
        and expected_divergence < shared
        and predicted_normal[expected_divergence]
        != predicted_counterfactual[expected_divergence]
    )
    state_intervention = (
        prediction_at_expected
        and bool(normal["rows"][expected_divergence]["correct"])
        and bool(counterfactual["rows"][expected_divergence]["correct"])
    )
    derived = {
        "normal_terminal_carry_class": normal["terminal_carry_class"],
        "counterfactual_terminal_carry_class": counterfactual["terminal_carry_class"],
        "expected_answer_changed": expected_changed,
        "first_expected_state_divergence_position": expected_divergence,
        "first_predicted_state_divergence_position": predicted_divergence,
        "prediction_diverged_at_expected_position": prediction_at_expected,
        "state_intervention_at_expected_position": state_intervention,
        "both_state_closed_loop_exact": normal["state_closed_loop_exact"]
        and counterfactual["state_closed_loop_exact"],
        "both_final_answers_correct": normal["final_answer_correct"]
        and counterfactual["final_answer_correct"],
        "answer_intervention_success": answer_intervention,
        "both_closed_loop_success": normal["closed_loop_success"]
        and counterfactual["closed_loop_success"],
    }
    for field, expected in derived.items():
        require(pair[field] == expected, f"{label}.{field} does not replay")
    return dict(pair)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize_branches(branches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(branches)
    transition_budget = sum(int(row["transition_budget"]) for row in rows)
    exact_transitions = sum(
        int(step["correct"]) for row in rows for step in row["rows"]
    )
    parseable_transitions = sum(
        int(step["predicted_state"] is not None) for row in rows for step in row["rows"]
    )
    counts = {
        "branches": len(rows),
        "transition_budget": transition_budget,
        "transition_calls": sum(int(row["transition_calls"]) for row in rows),
        "parseable_transitions": parseable_transitions,
        "exact_transitions": exact_transitions,
        "exact_prefix_steps": sum(int(row["prefix_exact_length"]) for row in rows),
        "fully_parseable": sum(int(row["fully_parseable"]) for row in rows),
        "state_closed_loop_exact": sum(
            int(row["state_closed_loop_exact"]) for row in rows
        ),
        "terminal_transition_exact": sum(
            int(row["terminal_transition_exact"]) for row in rows
        ),
        "terminal_reached": sum(int(row["terminal_reached"]) for row in rows),
        "final_prompt_issued": sum(int(row["final_prompt_issued"]) for row in rows),
        "final_answer_parseable": sum(
            int(row["emitted_answer"] is not None) for row in rows
        ),
        "final_answer_correct": sum(int(row["final_answer_correct"]) for row in rows),
        "closed_loop_success": sum(int(row["closed_loop_success"]) for row in rows),
        "emitted_tokens": sum(int(row["emitted_token_count"]) for row in rows),
    }
    denominator = counts["branches"]
    rates = {
        "parseable_transition_per_budget": ratio(
            parseable_transitions, transition_budget
        ),
        "exact_transition_per_budget": ratio(exact_transitions, transition_budget),
        "exact_prefix_survival_per_budget": ratio(
            counts["exact_prefix_steps"], transition_budget
        ),
        "fully_parseable": ratio(counts["fully_parseable"], denominator),
        "state_closed_loop_exact": ratio(
            counts["state_closed_loop_exact"], denominator
        ),
        "terminal_transition_exact": ratio(
            counts["terminal_transition_exact"], denominator
        ),
        "terminal_reached": ratio(counts["terminal_reached"], denominator),
        "final_answer_correct": ratio(counts["final_answer_correct"], denominator),
        "closed_loop_success": ratio(counts["closed_loop_success"], denominator),
    }
    failures = Counter(
        "success"
        if row["first_failure_position"] is None
        else f"p{row['first_failure_position']}:{row['first_failure_reason']}"
        for row in rows
    )
    return {
        "counts": counts,
        "rates": rates,
        "first_failure_distribution": dict(sorted(failures.items())),
    }


def summarize_pairs(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(pairs)
    counts = {
        "pairs": len(rows),
        "expected_answer_changed": sum(
            int(row["expected_answer_changed"]) for row in rows
        ),
        "state_intervention_at_expected_position": sum(
            int(row["state_intervention_at_expected_position"]) for row in rows
        ),
        "both_state_closed_loop_exact": sum(
            int(row["both_state_closed_loop_exact"]) for row in rows
        ),
        "both_final_answers_correct": sum(
            int(row["both_final_answers_correct"]) for row in rows
        ),
        "answer_intervention_success": sum(
            int(row["answer_intervention_success"]) for row in rows
        ),
        "both_closed_loop_success": sum(
            int(row["both_closed_loop_success"]) for row in rows
        ),
    }
    rates = {
        name: ratio(value, counts["pairs"])
        for name, value in counts.items()
        if name != "pairs"
    }
    expected = Counter(
        "none"
        if row["first_expected_state_divergence_position"] is None
        else str(row["first_expected_state_divergence_position"])
        for row in rows
    )
    predicted = Counter(
        "none"
        if row["first_predicted_state_divergence_position"] is None
        else str(row["first_predicted_state_divergence_position"])
        for row in rows
    )
    return {
        "counts": counts,
        "rates": rates,
        "first_expected_state_divergence_distribution": dict(sorted(expected.items())),
        "first_predicted_state_divergence_distribution": dict(
            sorted(predicted.items())
        ),
    }


def grouped_summary(
    values: list[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
    summarize: Callable[[Iterable[Mapping[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[key(value)].append(value)
    return {name: summarize(grouped[name]) for name in sorted(grouped)}


def width_8_survival(branches: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in branches if row["width"] == 8]
    positions: list[dict[str, Any]] = []
    for position in range(8):
        attempted = sum(position < len(branch["rows"]) for branch in selected)
        parseable = sum(
            position < len(branch["rows"])
            and branch["rows"][position]["predicted_state"] is not None
            for branch in selected
        )
        exact = sum(
            position < len(branch["rows"]) and branch["rows"][position]["correct"]
            for branch in selected
        )
        survived = sum(branch["prefix_exact_length"] > position for branch in selected)
        positions.append(
            {
                "position": position,
                "branches": len(selected),
                "attempted": attempted,
                "parseable": parseable,
                "exact_transition": exact,
                "exact_prefix_survived": survived,
                "exact_prefix_survival_rate": ratio(survived, len(selected)),
            }
        )
    return {
        "branches": len(selected),
        "positions": positions,
        "terminal": summarize_branches(selected),
    }


def build_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    branches: list[Mapping[str, Any]] = []
    tagged: list[dict[str, Any]] = []
    for pair in records:
        for branch_name in ("normal", "counterfactual"):
            branch = pair[branch_name]
            branches.append(branch)
            tagged.append({**branch, "branch": branch_name})
    return {
        "branches": {
            "overall": summarize_branches(branches),
            "by_branch": grouped_summary(
                tagged, lambda row: str(row["branch"]), summarize_branches
            ),
            "by_regime": grouped_summary(
                branches, lambda row: str(row["regime"]), summarize_branches
            ),
            "by_operation": grouped_summary(
                branches, lambda row: str(row["operation"]), summarize_branches
            ),
            "by_width": grouped_summary(
                branches, lambda row: str(row["width"]), summarize_branches
            ),
            "by_terminal_carry_class": grouped_summary(
                branches,
                lambda row: f"{row['operation']}:{row['terminal_carry_class']}",
                summarize_branches,
            ),
            "by_operation_width_terminal_carry_class": grouped_summary(
                branches,
                lambda row: (
                    f"{row['operation']}|w{row['width']}|{row['terminal_carry_class']}"
                ),
                summarize_branches,
            ),
            "width_8_survival": width_8_survival(branches),
        },
        "pairs": {
            "overall": summarize_pairs(records),
            "by_regime": grouped_summary(
                records, lambda row: str(row["regime"]), summarize_pairs
            ),
            "by_operation": grouped_summary(
                records, lambda row: str(row["operation"]), summarize_pairs
            ),
            "by_width": grouped_summary(
                records, lambda row: str(row["width"]), summarize_pairs
            ),
            "by_terminal_carry_pair": grouped_summary(
                records,
                lambda row: (
                    f"{row['operation']}:{row['normal_terminal_carry_class']}->{row['counterfactual_terminal_carry_class']}"
                ),
                summarize_pairs,
            ),
        },
    }


def build_accounting(
    records: list[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    require(len(records) == PAIR_COUNT, "pair count differs from frozen budget")
    regime_counts = Counter(str(row["regime"]) for row in records)
    require(
        dict(sorted(regime_counts.items())) == REGIME_BUDGETS, "regime budget differs"
    )
    expected_budget = sum(
        2 * REGIME_BUDGETS[name] * width for name, width in REGIME_WIDTHS.items()
    )
    overall = metrics["branches"]["overall"]["counts"]
    require(overall["branches"] == BRANCH_COUNT, "branch count differs")
    require(
        overall["transition_budget"] == expected_budget, "transition budget differs"
    )
    require(
        overall["transition_calls"] <= expected_budget, "transition calls exceed budget"
    )
    require(overall["final_prompt_issued"] <= BRANCH_COUNT, "final calls exceed budget")
    return {
        "pairs": PAIR_COUNT,
        "branches": BRANCH_COUNT,
        "by_regime": dict(sorted(regime_counts.items())),
        "transition_budget": expected_budget,
        "transition_calls": overall["transition_calls"],
        "max_final_calls": BRANCH_COUNT,
        "actual_final_calls": overall["final_prompt_issued"],
        "max_generation_calls": expected_budget + BRANCH_COUNT,
        "actual_generation_calls": overall["transition_calls"]
        + overall["final_prompt_issued"],
        "early_termination_is_scored_failure_not_repaired": True,
    }


def branch_diagnostics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    branches = [pair[name] for pair in records for name in ("normal", "counterfactual")]
    serializer = [
        branch
        for branch in branches
        if branch["state_closed_loop_exact"] and not branch["final_answer_correct"]
    ]
    late_width_8: list[dict[str, Any]] = []
    for pair in records:
        for branch_name in ("normal", "counterfactual"):
            branch = pair[branch_name]
            if branch["width"] != 8 or len(branch["rows"]) != 8:
                continue
            if (
                not all(row["correct"] for row in branch["rows"][:7])
                or branch["rows"][7]["correct"]
            ):
                continue
            predicted = branch["rows"][7]["predicted_state"]
            expected = branch["rows"][7]["expected_state"]
            if predicted is None:
                fields = ["malformed"]
            else:
                fields = sorted(
                    field for field in STATE_KEYS if predicted[field] != expected[field]
                )
            late_width_8.append(
                {
                    "id": pair["id"],
                    "branch": branch_name,
                    "operation": branch["operation"],
                    "differing_fields": fields,
                    "expected_answer": branch["expected_answer"],
                    "emitted_answer": branch["emitted_answer"],
                }
            )
    return {
        "exact_state_wrong_final": {
            "count": len(serializer),
            "by_regime": dict(
                sorted(Counter(row["regime"] for row in serializer).items())
            ),
            "by_operation": dict(
                sorted(Counter(row["operation"] for row in serializer).items())
            ),
            "by_width": dict(
                sorted(
                    (str(key), value)
                    for key, value in Counter(
                        row["width"] for row in serializer
                    ).items()
                )
            ),
        },
        "width_8_exact_through_position_6_wrong_position_7": late_width_8,
    }


def load_and_replay(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular report: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    require(mode in {0o400, 0o444}, f"report is not sealed read-only: {path}")
    payload = path.read_bytes()
    try:
        report = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReplayError(f"non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayError(f"malformed report: {path}") from error
    finite_tree(report)
    require(
        payload == canonical_json_bytes(report), f"report is not canonical JSON: {path}"
    )
    require(
        report.get("audit") == AUDIT and report.get("status") == "complete",
        f"report header is invalid: {path}",
    )
    arm = report.get("arm")
    require(type(arm) is str and arm, f"report arm is invalid: {path}")
    transcripts = report.get("transcripts")
    require(type(transcripts) is list, f"transcripts are missing: {path}")
    require(
        report.get("transcript_count") == len(transcripts) == PAIR_COUNT,
        f"transcript count differs: {path}",
    )
    require(
        report.get("transcripts_sha256")
        == sha256_bytes(canonical_json_bytes(transcripts)),
        f"transcript hash differs: {path}",
    )
    verified = [verify_pair(pair, index) for index, pair in enumerate(transcripts)]
    metrics = build_metrics(verified)
    accounting = build_accounting(verified, metrics)
    require(report.get("metrics") == metrics, f"metrics do not replay: {path}")
    require(
        report.get("accounting") == accounting, f"accounting does not replay: {path}"
    )
    heldout = report.get("heldout")
    require(type(heldout) is dict, f"heldout contract is missing: {path}")
    require(
        heldout.get("pair_count") == PAIR_COUNT
        and heldout.get("branch_count") == BRANCH_COUNT,
        f"heldout count differs: {path}",
    )
    require(
        heldout.get("regime_budgets") == REGIME_BUDGETS,
        f"heldout regime budget differs: {path}",
    )
    selected_ids = [pair["id"] for pair in verified]
    require(len(set(selected_ids)) == len(selected_ids), f"duplicate pair IDs: {path}")
    require(
        heldout.get("selected_ids_sha256")
        == sha256_bytes(canonical_json_bytes(selected_ids)),
        f"selected ID commitment differs: {path}",
    )
    training = report.get("training")
    require(type(training) is dict, f"training contract is missing: {path}")
    require(training.get("arm") == arm, f"training arm differs: {path}")
    heldout_identity = {
        key: heldout.get(key)
        for key in (
            "episodes_sha256",
            "tokenizer_sha256",
            "tokenizer_bytes",
            "tokenizer_vocab_size",
            "pair_count",
            "branch_count",
            "regime_budgets",
            "selected_ids_sha256",
            "selection",
        )
    }
    return {
        "arm": arm,
        "path": str(path.resolve()),
        "mode": format(mode, "04o"),
        "report_sha256": sha256_bytes(payload),
        "transcripts_sha256": report["transcripts_sha256"],
        "heldout_identity": heldout_identity,
        "records": verified,
        "metrics": metrics,
        "accounting": accounting,
        "headline": {
            "branch_counts": metrics["branches"]["overall"]["counts"],
            "branch_rates": metrics["branches"]["overall"]["rates"],
            "pair_counts": metrics["pairs"]["overall"]["counts"],
            "pair_rates": metrics["pairs"]["overall"]["rates"],
            "width_8_survival": metrics["branches"]["width_8_survival"],
        },
        "diagnostics": branch_diagnostics(verified),
    }


def exact_mcnemar(left_only: int, right_only: int) -> float:
    require(left_only >= 0 and right_only >= 0, "negative discordance")
    total = left_only + right_only
    if total == 0:
        return 1.0
    low = min(left_only, right_only)
    return min(
        1.0, 2.0 * sum(math.comb(total, index) for index in range(low + 1)) / (2**total)
    )


def paired_boolean(
    left: list[Mapping[str, Any]], right: list[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    require(len(left) == len(right), "paired arm lengths differ")
    left_only = 0
    right_only = 0
    both = 0
    neither = 0
    for left_row, right_row in zip(left, right, strict=True):
        require(left_row["id"] == right_row["id"], "paired arm IDs differ")
        a = strict_bool(left_row[field], f"left.{field}")
        b = strict_bool(right_row[field], f"right.{field}")
        both += int(a and b)
        left_only += int(a and not b)
        right_only += int(not a and b)
        neither += int(not a and not b)
    total = len(left)
    return {
        "left_success": both + left_only,
        "right_success": both + right_only,
        "left_only_losses": left_only,
        "right_only_gains": right_only,
        "both_success": both,
        "neither_success": neither,
        "absolute_rate_delta": (right_only - left_only) / total,
        "mcnemar_exact_two_sided_p": exact_mcnemar(left_only, right_only),
    }


def arm_rows(result: Mapping[str, Any], level: str) -> list[Mapping[str, Any]]:
    records = result["records"]
    if level == "pairs":
        return records
    return [pair[name] for pair in records for name in ("normal", "counterfactual")]


def compare_arms(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    require(
        left["heldout_identity"] == right["heldout_identity"],
        "cross-arm heldout identity differs",
    )
    return {
        "branches": {
            field: paired_boolean(
                arm_rows(left, "branches"), arm_rows(right, "branches"), field
            )
            for field in BRANCH_METRICS
        },
        "pairs": {
            field: paired_boolean(
                arm_rows(left, "pairs"), arm_rows(right, "pairs"), field
            )
            for field in PAIR_METRICS
        },
    }


def factorial_analysis(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    needed = {"iid", "term", "width", "term_width"}
    if not needed <= set(results):
        return None
    comparisons = {
        "term_without_width__iid_to_term": compare_arms(
            results["iid"], results["term"]
        ),
        "term_with_width__width_to_term_width": compare_arms(
            results["width"], results["term_width"]
        ),
        "width_without_term__iid_to_width": compare_arms(
            results["iid"], results["width"]
        ),
        "width_with_term__term_to_term_width": compare_arms(
            results["term"], results["term_width"]
        ),
    }
    interaction: dict[str, dict[str, float]] = {"branches": {}, "pairs": {}}
    for level, fields in (("branches", BRANCH_METRICS), ("pairs", PAIR_METRICS)):
        for field in fields:
            term_without = comparisons["term_without_width__iid_to_term"][level][field][
                "absolute_rate_delta"
            ]
            term_with = comparisons["term_with_width__width_to_term_width"][level][
                field
            ]["absolute_rate_delta"]
            interaction[level][field] = term_with - term_without
    return {"comparisons": comparisons, "term_by_width_interaction": interaction}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replayed: dict[str, dict[str, Any]] = {}
    for path in args.reports:
        result = load_and_replay(path)
        require(result["arm"] not in replayed, f"duplicate arm: {result['arm']}")
        replayed[result["arm"]] = result
    summary = {
        "audit": "shohin-digitwise-factorial-v4-independent-replay-v1",
        "status": "complete",
        "arms": {
            arm: {
                key: value
                for key, value in result.items()
                if key not in {"records", "metrics", "accounting"}
            }
            for arm, result in sorted(replayed.items())
        },
        "factorial": factorial_analysis(replayed),
    }
    print(canonical_json_bytes(summary).decode("ascii"), end="")


if __name__ == "__main__":
    main()
