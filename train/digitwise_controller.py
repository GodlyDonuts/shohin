"""Bounded controller for self-authored digitwise scratchpad rollouts."""
from __future__ import annotations

from digitwise_protocol import apply_microstep, final_prompt, microstep_prompt, parse_answer, parse_state, state_answer


def rollout_episode(episode, ask, prompt_style="heldout"):
    """Forward only model-emitted states; never execute or repair a transition."""
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("episode has invalid initial state")
    rows = []
    for index, expected_line in enumerate(episode["expected_states"]):
        expected = parse_state(expected_line)
        if expected is None:
            raise ValueError("episode has invalid expected state")
        prompt = microstep_prompt(state, style=prompt_style)
        response = ask(prompt)
        predicted = parse_state(response)
        correct = predicted == expected
        rows.append({
            "index": index,
            "prompt": prompt,
            "response": response,
            "input_state": state,
            "predicted_state": predicted,
            "expected_state": expected,
            "correct": correct,
        })
        if not correct:
            return {
                "success": False,
                "state_closed_loop": False,
                "state": state,
                "rows": rows,
                "final_response": "",
                "final_correct": False,
            }
        state = predicted
    prompt = final_prompt(state, style=prompt_style)
    final_response = ask(prompt)
    final_correct = parse_answer(final_response) == state_answer(state)
    return {
        "success": final_correct,
        "state_closed_loop": True,
        "state": state,
        "rows": rows,
        "final_response": final_response,
        "final_correct": final_correct,
    }
