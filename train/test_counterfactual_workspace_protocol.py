#!/usr/bin/env python3
"""Mechanical contracts for semantic, grammar-valid CWI reflection foils."""

from counterfactual_workspace_protocol import (
    FOIL_KINDS,
    make_semantic_foil,
    reflection_prompt,
    reflection_response,
    validate_reflection_record,
)
from digitwise_factor_protocol import apply_microstep, initial_register, initial_tape


def main():
    tape = initial_tape("add", 29, 16, 4)
    register = initial_register(tape)
    for kind in FOIL_KINDS:
        record = make_semantic_foil(tape, register, kind)
        assert validate_reflection_record(record)
        prompt = reflection_prompt(record)
        response = reflection_response(record)
        assert "Fixed tape:" in prompt and "Proposed register:" in prompt
        assert response.startswith("verdict=illegal;field={};at_p=0;".format(kind))
    legal = make_semantic_foil(tape, register, "carry")
    legal["candidate_register"] = apply_microstep(tape, register)
    legal["foil_kind"] = "legal"
    assert validate_reflection_record(legal)
    assert reflection_response(legal).startswith("verdict=legal;field=none;")

    terminal = initial_register(tape)
    for _ in range(4):
        terminal = apply_microstep(tape, terminal)
    try:
        make_semantic_foil(tape, terminal, "program_counter")
    except ValueError as exc:
        assert "terminal" in str(exc)
    else:
        raise AssertionError("terminal program-counter foil must be rejected")
    print("counterfactual workspace protocol contracts passed")


if __name__ == "__main__":
    main()
