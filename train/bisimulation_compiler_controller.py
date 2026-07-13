"""Transport-only evaluator controller for Counterfactual Bisimulation Compiler.

This module deliberately contains no world-state solver.  It may parse a
model-emitted CBC state, render a next prompt around that text, and compare the
result to generator-provided expectations after a rollout.  It never repairs,
selects, or computes a state transition or answer.
"""
from __future__ import annotations

import re

from bisimulation_compiler_protocol import (
    canonical_state,
    delta_prompt,
    parse_delta,
    parse_state,
    sum_query_prompt,
    update_prompt,
)


ANSWER = re.compile(r"(?mi)^\s*answer=(-?\d+)\s*$")


def parse_answer(text):
    matches = ANSWER.findall(str(text))
    return int(matches[-1]) if len(matches) == 1 else None


def _rollout_from_state(world, episode, ask, state):
    """Forward only a supplied model state; do not execute or repair it."""
    keys = tuple(episode["keys"])
    current = canonical_state(state, keys)
    rows = []
    for step in world["steps"]:
        revision = int(step["revision"])
        prompt = update_prompt(current, step["event"], episode["reference"], revision, episode["style"])
        response = ask(prompt)
        predicted = parse_state(response, keys)
        expected = parse_state(step["after_state"], keys)
        correct = predicted == expected
        row = {
            "revision": revision,
            "update_prompt": prompt,
            "update_response": response,
            "predicted_state": predicted,
            "expected_state": expected,
            "state_correct": correct,
        }
        rows.append(row)
        if not correct:
            return {
                "state_closed_loop": False,
                "terminal_state": None,
                "rows": rows,
                "final_response": "",
                "final_answer": None,
                "final_correct": False,
            }
        next_state = canonical_state(predicted, keys)
        inverse_prompt = delta_prompt(current, next_state, episode["reference"], revision, episode["style"])
        inverse_response = ask(inverse_prompt)
        row["inverse_delta_prompt"] = inverse_prompt
        row["inverse_delta_response"] = inverse_response
        row["predicted_delta"] = parse_delta(inverse_response, keys)
        row["expected_delta"] = step["operation"]
        row["inverse_delta_correct"] = row["predicted_delta"] == step["operation"]
        current = next_state
    final_prompt = sum_query_prompt(current, keys, episode["reference"], episode["style"])
    final_response = ask(final_prompt)
    final_answer = parse_answer(final_response)
    return {
        "state_closed_loop": True,
        "terminal_state": current,
        "rows": rows,
        "final_prompt": final_prompt,
        "final_response": final_response,
        "final_answer": final_answer,
        "final_correct": final_answer == int(world["query"]["answer"]),
    }


def rollout_world(world, episode, ask):
    """Compile twice, then use only the first emitted state for a rollout."""
    keys = tuple(episode["keys"])
    expected_initial = parse_state(world["initial_state"], keys)
    compilations = {}
    for variant in ("a", "b"):
        prompt = world["compile_prompts"][variant]
        response = ask(prompt)
        state = parse_state(response, keys)
        compilations[variant] = {
            "prompt": prompt,
            "response": response,
            "state": state,
            "correct": state == expected_initial,
        }
    first = compilations["a"]["state"]
    if first is None:
        return {
            "compilations": compilations,
            "compile_equal": False,
            "primary": {
                "state_closed_loop": False,
                "terminal_state": None,
                "rows": [],
                "final_response": "",
                "final_answer": None,
                "final_correct": False,
            },
            "interchange": None,
        }
    primary = _rollout_from_state(world, episode, ask, first)
    second = compilations["b"]["state"]
    interchange = None if second is None else _rollout_from_state(world, episode, ask, second)
    return {
        "compilations": compilations,
        "compile_equal": first == second,
        "primary": primary,
        "interchange": interchange,
    }


def evaluate_pair(episode, ask):
    """Score same-world interchange and counterfactual-state mismatch causally."""
    normal = rollout_world(episode["normal"], episode, ask)
    counterfactual = rollout_world(episode["counterfactual"], episode, ask)
    normal_primary = normal["primary"]
    counter_primary = counterfactual["primary"]
    keys = tuple(episode["keys"])
    cross_world = None
    if normal_primary["terminal_state"] is not None and counter_primary["terminal_state"] is not None:
        # The final query receives no original source text.  Replacing only its
        # carrier with the model-emitted counterfactual terminal state is the
        # actual interchange intervention; re-reading the normal state would
        # only restate the ordinary rollout score.
        prompt = sum_query_prompt(counter_primary["terminal_state"], keys, episode["reference"], episode["style"])
        response = ask(prompt)
        answer = parse_answer(response)
        cross_world = {
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "carrier": counter_primary["terminal_state"],
            "uses_counterfactual_answer": answer == int(episode["counterfactual"]["query"]["answer"]),
            "rejects_normal_answer": answer != int(episode["normal"]["query"]["answer"]),
        }
    return {
        "normal": normal,
        "counterfactual": counterfactual,
        "same_world_interchange_success": bool(
            normal["compilations"]["a"]["correct"] and normal["compilations"]["b"]["correct"] and
            normal["compile_equal"] and normal["interchange"] and normal["interchange"]["final_correct"]
        ),
        "counterfactual_interchange_success": bool(
            counterfactual["compilations"]["a"]["correct"] and counterfactual["compilations"]["b"]["correct"] and
            counterfactual["compile_equal"] and counterfactual["interchange"] and counterfactual["interchange"]["final_correct"]
        ),
        "cross_world_interchange": cross_world,
        "cross_world_counterfactual_success": bool(
            normal_primary["final_correct"] and counter_primary["final_correct"] and cross_world and
            cross_world["uses_counterfactual_answer"] and cross_world["rejects_normal_answer"]
        ),
        "claim_boundary": "Transport-only rollout; no semantic solver or state repair is used at runtime.",
    }
