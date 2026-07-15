#!/usr/bin/env python3
"""Recompute the frozen raw-260k interaction evidence without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = {
    "manual": (
        "artifacts/eval_history/manual_capability_raw260k_20260715_mps.json",
        "42590202834294cea182821f09613503c5ca91f6a1676d020d9f2cc2100c0aac",
    ),
    "exploratory_modes": (
        "artifacts/eval_history/raw260k_continuation_modes_20260715_mps.json",
        "f462391f3351a8491955587c036e7579559deb6d52e3c44827a236f245d41290",
    ),
    "continuation": (
        "artifacts/eval_history/raw260k_continuation_confirmation_20260715_mps.json",
        "f333c8f54383c411813551bc2001077b88e49514923b76c3cfe0331e9fd6bb47",
    ),
    "continuation_assessment": (
        "artifacts/eval_history/raw260k_continuation_confirmation_20260715_mps.assessment.json",
        "058aa9dafdc741efc181e6377db5d46b233875504b4b4b6d92837a0db71ea62b",
    ),
    "ssc": (
        "artifacts/eval_history/raw260k_ssc_diagnostic_20260715_mps.json",
        "a152e85294d02173a697e29d8537bf4b53428d747d16c7e3baf692095d9b6a2f",
    ),
    "atomic_formats": (
        "artifacts/eval_history/raw260k_atomic_operation_formats_20260715_mps.json",
        "b33c26b3963296c0d97b2a6d3332c0be18af40f460137c25652b881824a1ca4b",
    ),
    "renderer_interchange": (
        "artifacts/eval_history/raw260k_renderer_interchange_20260715_mps.json",
        "963177139b6abb333710f0db19a521c341a039fce3f65743ebdd698be6f12170",
    ),
    "packet_interface": (
        "artifacts/eval_history/raw260k_residual_packet_interface_20260715_mps.json",
        "1ca48442013a69f8fa53e25a0e063ea38063d7cd9e245c731b2b5fa295e1376c",
    ),
    "updater_subskills": (
        "artifacts/eval_history/raw260k_updater_subskill_probe_20260715_mps.json",
        "4505602994a0e337b99359e580a6f2f04fad4d365b2dac59f4c339fac13a7593",
    ),
    "updater_subskills_assessment": (
        "artifacts/eval_history/raw260k_updater_subskill_probe_20260715_mps.assessment.json",
        "26da3205d50301f6a9accf27ed22d4a6d92d7efcc451460bb1a73bb02dcff536",
    ),
    "fresh_board": (
        "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json",
        "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474",
    ),
}

CHECKPOINT_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
CASE_PAYLOAD_SHA256 = "3bae0add841e403d01251ae6e6ff110f3c6a07324b28de1b671a59f012071f7c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hashed_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "rate": correct / total if total else None,
    }


def has_extra_text_after_first_line(response: str) -> bool:
    _first, separator, remainder = response.strip().partition("\n")
    return bool(separator and remainder.strip())


def apply_operation(state: int, operation: str, operand: int) -> int:
    if operation == "add":
        return state + operand
    if operation == "multiply":
        return state * operand
    if operation == "subtract":
        return state - operand
    if operation == "remainder":
        return state % operand
    raise ValueError(f"unknown operation: {operation}")


def contains_subsequence(values: Iterable[int], expected: Iterable[int]) -> bool:
    iterator = iter(values)
    return all(any(value == target for value in iterator) for target in expected)


def analyze_manual(data: dict[str, Any]) -> dict[str, Any]:
    require(
        data.get("audit") == "manual_capability_probe_v1", "manual audit schema differs"
    )
    require(data.get("cases") == 7, "manual case count differs")
    models = data.get("models")
    require(
        isinstance(models, list) and len(models) == 1,
        "expected one manual model record",
    )
    model = models[0]
    rows = model.get("rows")
    require(isinstance(rows, list) and len(rows) == 7, "manual rows differ")

    phases = ("initial", "review", "verified_fact", "state_reuse")
    recomputed = {
        phase: sum(bool(row[phase]["correct"]) for row in rows) for phase in phases
    }
    for phase, count in recomputed.items():
        require(model["summary"][phase] == count, f"manual {phase} summary differs")

    first_correct_phase = {}
    for row in rows:
        first_correct_phase[row["id"]] = next(
            (phase for phase in phases if row[phase]["correct"]), None
        )

    exact_state_prefix = sum(
        row["compact_state"]["response"].lstrip().startswith("state=") for row in rows
    )
    return {
        "cases": len(rows),
        "strict_scores": {
            phase: ratio(count, len(rows)) for phase, count in recomputed.items()
        },
        "initial_correct_ids": [row["id"] for row in rows if row["initial"]["correct"]],
        "first_correct_phase": first_correct_phase,
        "compact_state_exact_prefix": ratio(exact_state_prefix, len(rows)),
        "paired_answer_mode": model["summary"].get("paired_answer_mode", {}),
    }


def recompute_continuation_mode(
    rows: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    values = [row["modes"][mode] for row in rows]
    return {
        "final_correct": ratio(
            sum(value["final_correct"] for value in values), len(values)
        ),
        "contains_answer": ratio(
            sum(value["contains_answer"] for value in values), len(values)
        ),
        "intermediates_present": ratio(
            sum(value["intermediates_present"] for value in values), len(values)
        ),
        "termination_only_failures": sum(
            value["contains_answer"] and not value["final_correct"] for value in values
        ),
    }


def analyze_continuation(
    transcript: dict[str, Any], assessment: dict[str, Any], transcript_sha256: str
) -> dict[str, Any]:
    require(
        transcript.get("audit") == "raw_continuation_confirmation_v1",
        "continuation schema differs",
    )
    require(
        assessment.get("audit") == "raw_continuation_confirmation_assessment_v2",
        "continuation assessment schema differs",
    )
    require(
        assessment.get("source_sha256") == transcript_sha256,
        "assessment source hash differs",
    )
    require(
        transcript.get("cases_sha256") == CASE_PAYLOAD_SHA256,
        "case payload hash differs",
    )

    source_rows = transcript.get("rows")
    assessed_rows = assessment.get("rows")
    require(
        isinstance(source_rows, list) and len(source_rows) == 20,
        "continuation rows differ",
    )
    require(
        isinstance(assessed_rows, list) and len(assessed_rows) == 20,
        "assessment rows differ",
    )
    require(
        [row["id"] for row in source_rows] == [row["id"] for row in assessed_rows],
        "assessment row order differs",
    )

    modes = ("direct_qa", "bare_expression", "worked_completion")
    overall = {mode: recompute_continuation_mode(assessed_rows, mode) for mode in modes}
    for mode in modes:
        embedded = assessment["summary"][mode]
        for field in ("final_correct", "contains_answer", "intermediates_present"):
            require(
                overall[mode][field]["correct"] == embedded[field],
                f"corrected continuation {mode} {field} differs",
            )

    by_family: dict[str, Any] = {}
    for family in sorted({row["family"] for row in assessed_rows}):
        family_rows = [row for row in assessed_rows if row["family"] == family]
        by_family[family] = {
            mode: recompute_continuation_mode(family_rows, mode) for mode in modes
        }

    continued_after_segment = Counter()
    for source_row, assessed_row in zip(source_rows, assessed_rows):
        for mode in modes:
            response = source_row["modes"][mode]["response"].strip()
            segment = assessed_row["modes"][mode]["answer_segment"].strip()
            require(
                response.startswith(segment),
                f"corrected segment is not a response prefix: {source_row['id']} {mode}",
            )
            continued_after_segment[mode] += len(response) > len(segment)

    return {
        "cases": len(assessed_rows),
        "generations": len(assessed_rows) * len(modes),
        "overall": overall,
        "by_family": by_family,
        "continued_after_first_answer_segment": {
            mode: ratio(continued_after_segment[mode], len(assessed_rows))
            for mode in modes
        },
        "embedded_v1_direct_final_before_parser_repair": transcript["summary"][
            "direct_qa"
        ]["final_correct"],
        "corrected_v2_direct_final": assessment["summary"]["direct_qa"][
            "final_correct"
        ],
    }


def analyze_ssc(data: dict[str, Any]) -> dict[str, Any]:
    require(
        data.get("audit") == "source_scheduled_continuation_diagnostic_v1",
        "SSC schema differs",
    )
    rows = data.get("rows")
    require(isinstance(rows, list) and len(rows) == 20, "SSC rows differ")
    steps = [step for row in rows for step in row["steps"]]
    require(len(steps) == 55, "SSC step count differs")

    parse_success = sum(isinstance(step.get("predicted_state"), int) for step in steps)
    plus_one = sum(
        step.get("predicted_state") == step["input_state"] + 1 for step in steps
    )
    local_correct = sum(
        step.get("predicted_state")
        == apply_operation(step["input_state"], step["operation"], step["operand"])
        for step in steps
    )
    extra_text = sum(
        has_extra_text_after_first_line(step["response"]) for step in steps
    )

    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        family_steps = [
            step for row in rows if row["family"] == family for step in row["steps"]
        ]
        by_family[family] = {
            "calls": len(family_steps),
            "input_plus_one": ratio(
                sum(
                    step.get("predicted_state") == step["input_state"] + 1
                    for step in family_steps
                ),
                len(family_steps),
            ),
            "local_operation_correct": ratio(
                sum(
                    step.get("predicted_state")
                    == apply_operation(
                        step["input_state"], step["operation"], step["operand"]
                    )
                    for step in family_steps
                ),
                len(family_steps),
            ),
        }

    return {
        "cases": len(rows),
        "calls": len(steps),
        "parse_success": ratio(parse_success, len(steps)),
        "input_plus_one": ratio(plus_one, len(steps)),
        "local_operation_correct": ratio(local_correct, len(steps)),
        "first_transition_correct": ratio(
            sum(row["first_transition_correct"] for row in rows), len(rows)
        ),
        "full_chain_correct": ratio(
            sum(row["final_correct"] for row in rows), len(rows)
        ),
        "responses_with_extra_text_after_first_line": ratio(extra_text, len(steps)),
        "by_family": by_family,
    }


def chain_prefix_length(steps: list[dict[str, Any]]) -> int:
    length = 0
    for step in steps:
        if not step["gold_state_correct"]:
            break
        length += 1
    return length


def analyze_atomic_formats(data: dict[str, Any]) -> dict[str, Any]:
    require(
        data.get("audit") == "atomic_operation_format_matrix_v1",
        "atomic matrix schema differs",
    )
    rows = data.get("rows")
    require(isinstance(rows, list) and len(rows) == 20, "atomic matrix rows differ")
    formats = tuple(data["formats"])
    result: dict[str, Any] = {}

    for prompt_format in formats:
        atomic = [record for row in rows for record in row["atomic"][prompt_format]]
        chains = [row["chained"][prompt_format] for row in rows]
        chained_steps = [step for chain in chains for step in chain["steps"]]
        require(
            len(atomic) == 55 and len(chained_steps) == 55,
            f"{prompt_format} call counts differ",
        )

        all_calls = atomic + chained_steps
        parse_success = sum(
            isinstance(record.get("predicted_state"), int) for record in all_calls
        )
        extra_text = sum(
            has_extra_text_after_first_line(record["response"]) for record in all_calls
        )
        first_correct = sum(chain["steps"][0]["gold_state_correct"] for chain in chains)
        prefix_histogram = Counter(
            chain_prefix_length(chain["steps"]) for chain in chains
        )
        local_gold_cross = Counter(
            (step["local_operation_correct"], step["gold_state_correct"])
            for step in chained_steps
        )
        first_error = Counter()
        for chain in chains:
            for step in chain["steps"]:
                if not step["gold_state_correct"]:
                    first_error[step["operation"]] += 1
                    break

        by_family: dict[str, Any] = {}
        for family in sorted({row["family"] for row in rows}):
            family_rows = [row for row in rows if row["family"] == family]
            family_atomic = [
                record for row in family_rows for record in row["atomic"][prompt_format]
            ]
            family_chains = [row["chained"][prompt_format] for row in family_rows]
            family_steps = [step for chain in family_chains for step in chain["steps"]]
            by_family[family] = {
                "atomic_gold_state": ratio(
                    sum(record["correct"] for record in family_atomic),
                    len(family_atomic),
                ),
                "first_transition": ratio(
                    sum(
                        chain["steps"][0]["gold_state_correct"]
                        for chain in family_chains
                    ),
                    len(family_chains),
                ),
                "chained_local_operation": ratio(
                    sum(step["local_operation_correct"] for step in family_steps),
                    len(family_steps),
                ),
                "full_chain": ratio(
                    sum(chain["final_correct"] for chain in family_chains),
                    len(family_chains),
                ),
            }

        by_operation: dict[str, Any] = {}
        for operation in sorted({record["operation"] for record in atomic}):
            operation_atomic = [
                record for record in atomic if record["operation"] == operation
            ]
            operation_chained = [
                step for step in chained_steps if step["operation"] == operation
            ]
            by_operation[operation] = {
                "atomic_gold_state": ratio(
                    sum(record["correct"] for record in operation_atomic),
                    len(operation_atomic),
                ),
                "chained_local_operation": ratio(
                    sum(step["local_operation_correct"] for step in operation_chained),
                    len(operation_chained),
                ),
                "chained_gold_state": ratio(
                    sum(step["gold_state_correct"] for step in operation_chained),
                    len(operation_chained),
                ),
            }

        result[prompt_format] = {
            "model_calls": len(all_calls),
            "atomic_gold_state": ratio(
                sum(record["correct"] for record in atomic), len(atomic)
            ),
            "first_transition": ratio(first_correct, len(chains)),
            "chained_local_operation": ratio(
                sum(step["local_operation_correct"] for step in chained_steps),
                len(chained_steps),
            ),
            "chained_gold_state": ratio(
                sum(step["gold_state_correct"] for step in chained_steps),
                len(chained_steps),
            ),
            "full_chain": ratio(
                sum(chain["final_correct"] for chain in chains), len(chains)
            ),
            "parse_success": ratio(parse_success, len(all_calls)),
            "responses_with_extra_text_after_first_line": ratio(
                extra_text, len(all_calls)
            ),
            "correct_prefix_length_histogram": dict(sorted(prefix_histogram.items())),
            "local_gold_cross_tabulation": {
                f"local_{str(local).lower()}_gold_{str(gold).lower()}": count
                for (local, gold), count in sorted(local_gold_cross.items())
            },
            "first_error_operation": dict(sorted(first_error.items())),
            "by_family": by_family,
            "by_operation": by_operation,
        }

        embedded = data["summary"]["formats"][prompt_format]
        require(
            result[prompt_format]["atomic_gold_state"]["correct"]
            == embedded["atomic_correct"],
            f"{prompt_format} embedded atomic score differs",
        )
        require(
            result[prompt_format]["full_chain"]["correct"]
            == embedded["chains_final_correct"],
            f"{prompt_format} embedded chain score differs",
        )

    require(
        sum(item["model_calls"] for item in result.values()) == 330,
        "atomic model-call total differs",
    )
    return result


def analyze_renderer(data: dict[str, Any]) -> dict[str, Any]:
    require(
        data.get("audit") == "renderer_interchange_causal_audit_v1",
        "renderer schema differs",
    )
    crossed = [row for row in data["rows"] if row["crossed"]]
    local_wins = sum(row["winner"] == "local" for row in crossed)
    source_wins = sum(row["winner"] == "source" for row in crossed)
    ties = sum(row["winner"] == "tie" for row in crossed)
    minimum_margin = min(abs(row["local_minus_source_logprob"]) for row in crossed)
    summary = data["summary"]
    require(local_wins == summary["crossed_local_wins"], "renderer local wins differ")
    require(
        source_wins == summary["crossed_source_wins"], "renderer source wins differ"
    )
    return {
        "crossed_cells": len(crossed),
        "local_wins": local_wins,
        "source_wins": source_wins,
        "ties": ties,
        "minimum_absolute_margin": minimum_margin,
        "candidate_sequence_evaluations": data["resource_ledger"][
            "candidate_sequence_evaluations"
        ],
        "generated_tokens": data["resource_ledger"]["generated_tokens"],
    }


def analyze_packet(data: dict[str, Any]) -> dict[str, Any]:
    require(
        data.get("schema") == "raw_residual_packet_interface_probe_v1",
        "packet schema differs",
    )
    rows = {row["id"]: row for row in data["rows"]}
    require(len(rows) == 5, "packet row count differs")
    compiler_expected = {
        "compile_add_multiply_subtract": (25, 75, 64),
        "compile_multiply_subtract_add": (108, 95, 103),
    }
    integer_pattern = re.compile(r"-?\d+")
    compiler_arithmetic = 0
    compiler_packet_form = 0
    for row_id, expected in compiler_expected.items():
        response = rows[row_id]["response"]
        values = [int(value) for value in integer_pattern.findall(response)]
        compiler_arithmetic += contains_subsequence(values, expected)
        compiler_packet_form += bool(
            re.search(r"(?mi)^\s*State:\s*-?\d+\s*$", response)
            and re.search(r"(?mi)^\s*Plan:\s*.+$", response)
        )

    updater_expected = {
        "update_after_add": (25, "multiply 3; subtract 11"),
        "update_after_multiply": (75, "subtract 11"),
    }
    updater_correct = 0
    updater_repeats_observed = 0
    for row_id, (state, plan) in updater_expected.items():
        response = rows[row_id]["response"]
        updater_correct += bool(
            re.search(rf"(?mi)^\s*State:\s*{state}\s*$", response)
            and re.search(rf"(?mi)^\s*Plan:\s*{re.escape(plan)}\s*$", response)
        )
        updater_repeats_observed += f"Observed result: {state}" in response

    halt_response = rows["halt_after_subtract"]["response"]
    first_integer = integer_pattern.search(halt_response)
    halt_first_correct = bool(first_integer and int(first_integer.group()) == 64)
    halt_exact_only = halt_response.strip() == "64"

    return {
        "model_calls": data["model_calls"],
        "compiler_arithmetic_complete": ratio(
            compiler_arithmetic, len(compiler_expected)
        ),
        "compiler_packet_form": ratio(compiler_packet_form, len(compiler_expected)),
        "updater_exact_next_packet": ratio(updater_correct, len(updater_expected)),
        "updater_repeats_observed_result": ratio(
            updater_repeats_observed, len(updater_expected)
        ),
        "halt_first_integer_correct": ratio(int(halt_first_correct), 1),
        "halt_exact_integer_only": ratio(int(halt_exact_only), 1),
        "max_new_stop": ratio(
            sum(row["stop_reason"] == "max_new" for row in rows.values()), len(rows)
        ),
    }


def _updater_expected(prompt: str, kind: str) -> tuple[int | None, str | None]:
    observed_match = re.search(
        r"(?:observed result(?: is|:)|first operation produced)\s*(-?\d+)", prompt, re.I
    )
    observed = int(observed_match.group(1)) if observed_match else None
    plan_match = re.search(
        r"(?:plan (?:is|was)|Plan:)\s*(.+?)(?:\n|\. The first operation|\nObserved result:)",
        prompt,
    )
    plan = plan_match.group(1).strip().rstrip(".") if plan_match else None
    if kind == "copy_state":
        return observed, None
    require(plan is not None, f"cannot recover plan from updater prompt: {prompt!r}")
    operations = [operation.strip() for operation in plan.split(";")]
    require(len(operations) >= 2, "updater plan has fewer than two operations")
    return observed, "; ".join(operations[1:])


def _strict_updater_response(
    response: str, kind: str, observed: int | None, residual: str | None
) -> bool:
    normalized = response.strip()
    if kind == "copy_state":
        return observed is not None and normalized in {
            str(observed),
            f"State: {observed}",
            f"state={observed}",
        }
    if kind == "delete_head":
        return residual is not None and normalized in {residual, f"Plan: {residual}"}
    require(
        observed is not None and residual is not None,
        "joint updater expectation is incomplete",
    )
    canonical = f"State: {observed}\nPlan: {residual}"
    compact = f"state={observed}; plan={residual}"
    return normalized in {canonical, compact}


def analyze_updater_subskills(
    transcript: dict[str, Any], assessment: dict[str, Any], transcript_sha256: str
) -> dict[str, Any]:
    require(
        transcript.get("schema") == "raw260k_updater_subskill_probe_v1",
        "updater subskill transcript schema differs",
    )
    require(
        assessment.get("schema") == "raw260k_updater_subskill_assessment_v1",
        "updater subskill assessment schema differs",
    )
    require(
        assessment.get("raw_artifact_sha256") == transcript_sha256,
        "updater subskill assessment source hash differs",
    )
    rows = transcript.get("rows")
    assessed_rows = assessment.get("row_assessments")
    require(
        isinstance(rows, list) and len(rows) == 12, "updater transcript rows differ"
    )
    require(
        isinstance(assessed_rows, list) and len(assessed_rows) == len(rows),
        "updater assessment rows differ",
    )
    require(
        [row["id"] for row in rows] == [row["id"] for row in assessed_rows],
        "updater assessment row order differs",
    )
    require(transcript.get("model_calls") == len(rows), "updater call count differs")

    totals = Counter(row["kind"] for row in rows)
    recomputed = Counter()
    for row, assessed in zip(rows, assessed_rows, strict=True):
        observed, residual = _updater_expected(row["prompt"], row["kind"])
        strict_correct = _strict_updater_response(
            row["response"], row["kind"], observed, residual
        )
        recomputed[row["kind"]] += int(strict_correct)
        require(
            strict_correct == assessed["strict_correct"],
            f"updater assessment differs for {row['id']}",
        )

    summary = assessment.get("summary")
    require(isinstance(summary, dict), "updater assessment summary is missing")
    summary_fields = {
        "copy_state": "copy_state",
        "delete_head": "delete_head",
        "joint_natural": "joint_natural",
        "joint_packet": "joint_packet",
    }
    result: dict[str, Any] = {}
    for kind, prefix in summary_fields.items():
        require(summary[f"{prefix}_total"] == totals[kind], f"{kind} total differs")
        require(
            summary[f"{prefix}_correct"] == recomputed[kind], f"{kind} score differs"
        )
        result[kind] = ratio(recomputed[kind], totals[kind])

    strict_joint_correct = recomputed["joint_natural"] + recomputed["joint_packet"]
    strict_joint_total = totals["joint_natural"] + totals["joint_packet"]
    require(
        summary["strict_joint_correct"] == strict_joint_correct, "joint score differs"
    )
    require(summary["strict_joint_total"] == strict_joint_total, "joint total differs")
    result["strict_joint"] = ratio(strict_joint_correct, strict_joint_total)
    result["model_calls"] = len(rows)
    result["all_calls_hit_max_new"] = all(
        row["stop_reason"] == "max_new" for row in rows
    )
    return result


def analyze(root: Path = ROOT) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for name, (relative_path, expected_sha256) in ARTIFACTS.items():
        path = root / relative_path
        loaded[name] = read_hashed_json(path, expected_sha256)
        hashes[name] = {"path": relative_path, "sha256": expected_sha256}

    continuation_sha = ARTIFACTS["continuation"][1]
    for name in ("ssc", "atomic_formats"):
        require(
            loaded[name].get("source_cases_sha256") == continuation_sha,
            f"{name} is not bound to the frozen continuation transcript",
        )

    bound_names = (
        "exploratory_modes",
        "continuation",
        "ssc",
        "atomic_formats",
        "renderer_interchange",
        "packet_interface",
        "updater_subskills",
    )
    for name in bound_names:
        require(
            loaded[name].get("checkpoint_step") == 260000,
            f"{name} checkpoint step differs",
        )
        require(
            loaded[name].get("checkpoint_sha256") == CHECKPOINT_SHA256,
            f"{name} checkpoint hash differs",
        )
        require(
            loaded[name].get("tokenizer_sha256") == TOKENIZER_SHA256,
            f"{name} tokenizer hash differs",
        )

    fresh_board = loaded["fresh_board"]
    require(
        fresh_board.get("schema") == "source_scheduled_reasoning_confirmation_v1",
        "fresh board schema differs",
    )
    require(fresh_board.get("case_count") == 256, "fresh board case count differs")
    require(
        all("response" not in row for row in fresh_board["rows"]),
        "fresh board unexpectedly contains responses",
    )

    return {
        "artifact_hashes": hashes,
        "bindings": {
            "checkpoint_step": 260000,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "continuation_case_payload_sha256": CASE_PAYLOAD_SHA256,
        },
        "manual": analyze_manual(loaded["manual"]),
        "exploratory_modes": {
            "cases": loaded["exploratory_modes"]["case_count"],
            "modes": loaded["exploratory_modes"]["mode_count"],
            "embedded_summary": loaded["exploratory_modes"]["summary"],
        },
        "continuation": analyze_continuation(
            loaded["continuation"], loaded["continuation_assessment"], continuation_sha
        ),
        "ssc_next_state": analyze_ssc(loaded["ssc"]),
        "atomic_formats": analyze_atomic_formats(loaded["atomic_formats"]),
        "renderer_interchange": analyze_renderer(loaded["renderer_interchange"]),
        "packet_interface": analyze_packet(loaded["packet_interface"]),
        "updater_subskills": analyze_updater_subskills(
            loaded["updater_subskills"],
            loaded["updater_subskills_assessment"],
            ARTIFACTS["updater_subskills"][1],
        ),
        "fresh_confirmation_board": {
            "artifact_kind": "immutable_board_without_model_responses",
            "cases": fresh_board["case_count"],
            "families": fresh_board["per_family"],
            "rows_with_responses": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    indent = None if args.compact else 2
    print(json.dumps(report, indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
