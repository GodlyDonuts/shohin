"""Transport-only controller for token-native delta-ledger rollouts."""
from __future__ import annotations

from token_native_ledger_protocol import (
    canonical_delta,
    context_key,
    expected_answer,
    final_prompt,
    parse_delta,
    tape_from_text,
    transition_prompt,
)


def rollout_episode(episode, ask, prompt_style=None):
    """Forward exact model triples only; never calculate, repair, or select one."""
    tape = tape_from_text(episode["tape"])
    if tape is None:
        raise ValueError("invalid token-native ledger tape")
    style = episode["prompt_style"] if prompt_style is None else prompt_style
    prior = parse_delta(episode["initial_delta"])
    if prior is None:
        raise ValueError("invalid token-native initial delta")
    rows, emitted = [], []
    for index, expected_line in enumerate(episode["expected_deltas"]):
        expected = parse_delta(expected_line)
        if expected is None:
            raise ValueError("invalid expected token-native delta")
        prompt = transition_prompt(tape, prior, style=style)
        response = ask(prompt, 3)
        predicted = parse_delta(response)
        rows.append({
            "index": index,
            "prompt": prompt,
            "response": response,
            "prior_delta": canonical_delta(prior),
            "predicted_delta": predicted,
            "expected_delta": expected,
            "correct": predicted == expected,
        })
        if predicted is None:
            return {
                "success": False, "syntactic_closed_loop": False, "strict_closed_loop": False,
                "rows": rows, "emitted": emitted, "final_response": "", "final_correct": False,
            }
        prior = predicted
        emitted.append(canonical_delta(predicted))
    final_response = ask(final_prompt(emitted, context_key(tape), style=style), 48)
    final_correct = final_response.strip() == "answer={}".format(expected_answer(tape))
    strict = all(bool(row["correct"]) for row in rows)
    return {
        "success": strict and final_correct,
        "syntactic_closed_loop": True,
        "strict_closed_loop": strict,
        "rows": rows,
        "emitted": emitted,
        "final_response": final_response,
        "final_correct": final_correct,
    }
