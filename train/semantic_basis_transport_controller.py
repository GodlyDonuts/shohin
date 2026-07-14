"""Exact-carrier transport controller for semantic-basis episodes.

This module deliberately has no arithmetic, ledger rewriting, answer repair, or
state selection.  It may reject malformed model text, then substitutes the
remaining full model emission verbatim into a source-deleted downstream prompt.
Expected values belong to the frozen data and are used only for scoring.
"""
from __future__ import annotations

import re


LEDGER = re.compile(r"ledger:P=-?\d+;Q=-?\d+")
ANSWER = re.compile(r"answer=-?\d+")
PHASES = ("compile", "reflect", "update", "difference", "sum")


def exact_ledger(text):
    """Return the exact stripped model emission only when it is a full carrier."""
    emitted = str(text).strip()
    return emitted if LEDGER.fullmatch(emitted) else None


def exact_answer(text):
    """Return the exact stripped model emission only when it is a full answer."""
    emitted = str(text).strip()
    return emitted if ANSWER.fullmatch(emitted) else None


def phase_rows(rows):
    """Index one immutable five-phase episode without interpreting its values."""
    indexed = {}
    for row in rows:
        phase = row.get("phase")
        if phase not in PHASES or phase in indexed:
            raise ValueError("episode must have one row per semantic-basis phase")
        indexed[phase] = row
    if tuple(sorted(indexed)) != tuple(sorted(PHASES)):
        raise ValueError("episode is missing a semantic-basis phase")
    return indexed


def forward_carrier(prompt, expected_carrier, emitted_carrier):
    """Replace exactly one frozen carrier literal with a validated raw emission."""
    if exact_ledger(emitted_carrier) is None:
        raise ValueError("cannot forward malformed carrier")
    if prompt.count(expected_carrier) != 1:
        raise ValueError("prompt must contain exactly one frozen carrier")
    return prompt.replace(expected_carrier, emitted_carrier, 1)


def run_consumers(rows, ask, carrier, expected_difference, expected_sum):
    """Feed one exact carrier into both independent consumer prompts verbatim."""
    phases = phase_rows(rows)
    difference_prompt = forward_carrier(
        phases["difference"]["question"], phases["difference"]["expected_next_ledger"], carrier,
    )
    sum_prompt = forward_carrier(
        phases["sum"]["question"], phases["sum"]["expected_next_ledger"], carrier,
    )
    difference_response = ask(difference_prompt, 16)
    sum_response = ask(sum_prompt, 16)
    difference_exact = exact_answer(difference_response)
    sum_exact = exact_answer(sum_response)
    return {
        "carrier": carrier,
        "difference": {
            "prompt": difference_prompt,
            "response": difference_response,
            "exact_response": difference_exact,
            "expected": expected_difference,
            "correct": difference_exact == expected_difference,
        },
        "sum": {
            "prompt": sum_prompt,
            "response": sum_response,
            "exact_response": sum_exact,
            "expected": expected_sum,
            "correct": sum_exact == expected_sum,
        },
    }


def consumers_correct(result):
    return bool(result["difference"]["correct"] and result["sum"]["correct"])


def rollout_episode(rows, ask):
    """Run compile -> update -> two-consumer transport with a reportability check."""
    phases = phase_rows(rows)
    compile_response = ask(phases["compile"]["question"], 16)
    reflect_response = ask(phases["reflect"]["question"], 16)
    compile_carrier = exact_ledger(compile_response)
    reflect_carrier = exact_ledger(reflect_response)
    compile_correct = compile_carrier == phases["compile"]["response"]
    reflect_correct = reflect_carrier == phases["reflect"]["response"]
    result = {
        "episode_id": phases["compile"]["episode_id"],
        "compile": {
            "prompt": phases["compile"]["question"], "response": compile_response,
            "exact_response": compile_carrier, "expected": phases["compile"]["response"],
            "correct": compile_correct,
        },
        "reflect": {
            "prompt": phases["reflect"]["question"], "response": reflect_response,
            "exact_response": reflect_carrier, "expected": phases["reflect"]["response"],
            "correct": reflect_correct,
        },
        "reportability_equal": compile_carrier is not None and compile_carrier == reflect_carrier,
        "update": None,
        "consumers": None,
        "strict_success": False,
    }
    if compile_carrier is None:
        return result
    update_prompt = forward_carrier(
        phases["update"]["question"], phases["update"]["expected_ledger"], compile_carrier,
    )
    update_response = ask(update_prompt, 16)
    update_carrier = exact_ledger(update_response)
    update_correct = update_carrier == phases["update"]["response"]
    result["update"] = {
        "prompt": update_prompt, "response": update_response,
        "exact_response": update_carrier, "expected": phases["update"]["response"],
        "correct": update_correct,
    }
    if update_carrier is None:
        return result
    result["consumers"] = run_consumers(
        rows, ask, update_carrier, phases["difference"]["response"], phases["sum"]["response"],
    )
    result["strict_success"] = bool(
        compile_correct and reflect_correct and result["reportability_equal"] and update_correct
        and consumers_correct(result["consumers"])
    )
    return result
