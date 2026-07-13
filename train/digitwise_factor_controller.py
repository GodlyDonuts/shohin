"""Transport-only controller for factorized static-tape DRS rollouts."""
from __future__ import annotations

from digitwise_factor_protocol import (
    final_prompt,
    microstep_prompt,
    parse_answer,
    parse_register,
    parse_tape,
    register_answer,
)


def rollout_episode(episode, ask, prompt_style="heldout"):
    """Forward only model-emitted registers while retaining the original tape."""
    tape = parse_tape(episode["tape"])
    register = parse_register(episode["initial_register"], tape) if tape is not None else None
    if tape is None or register is None:
        raise ValueError("episode has invalid factorized initial state")
    rows = []
    for index, expected_line in enumerate(episode["expected_registers"]):
        expected = parse_register(expected_line, tape)
        if expected is None:
            raise ValueError("episode has invalid expected register")
        prompt = microstep_prompt(tape, register, style=prompt_style)
        response = ask(prompt)
        predicted = parse_register(response, tape)
        correct = predicted == expected
        rows.append({
            "index": index,
            "prompt": prompt,
            "response": response,
            "tape": tape,
            "input_register": register,
            "predicted_register": predicted,
            "expected_register": expected,
            "correct": correct,
        })
        if not correct:
            return {
                "success": False,
                "state_closed_loop": False,
                "tape": tape,
                "register": register,
                "rows": rows,
                "final_response": "",
                "final_correct": False,
            }
        register = predicted
    prompt = final_prompt(tape, register, style=prompt_style)
    final_response = ask(prompt)
    final_correct = parse_answer(final_response) == register_answer(tape, register)
    return {
        "success": final_correct,
        "state_closed_loop": True,
        "tape": tape,
        "register": register,
        "rows": rows,
        "final_response": final_response,
        "final_correct": final_correct,
    }
