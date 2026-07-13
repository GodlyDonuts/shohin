#!/usr/bin/env python3
"""Pure protocol checks for canonical working memory and transitions."""
from vrwm_protocol import (apply_operation, canonical_memory, parse_answer, parse_memory,
                           readout_prompt, repair_prompt, render_instruction, transition_prompt)


state = {"a": 3, "b": -2}
assert canonical_memory(state) == "wm:a=3;b=-2"
assert parse_memory("wm:a=3;b=-2") == state
assert parse_memory("x\nwm:a=3;b=-2\ny") == state
assert parse_memory("wm:a=3;b=-2\nwm:a=4;b=5") is None
assert parse_memory("wm:b=-2;a=3") is None
assert apply_operation(state, {"kind": "add_const", "target": "a", "value": 4}) == {"a": 7, "b": -2}
assert apply_operation(state, {"kind": "sub_var", "target": "a", "source": "b"}) == {"a": 5, "b": -2}
assert apply_operation(state, {"kind": "swap", "target": "a", "source": "b"}) == {"a": -2, "b": 3}
assert parse_answer("reasoning\nanswer=-17") == -17
op = {"kind": "add_var", "target": "a", "source": "b"}
assert render_instruction(op, style="default") == "add the current value of b to a"
assert render_instruction(op, style="paraphrase") == "replace a with a plus b"
assert render_instruction(op, style="semantic") == "set a to a + b"
assert "Working memory:" in transition_prompt(state, op)
assert "Registers:" in transition_prompt(state, op, style="paraphrase")
assert "State:" in transition_prompt(state, op, style="semantic")
assert "Inspect the register state" in readout_prompt(state, "a", style="semantic")
assert "Candidate:" in repair_prompt(state, op, {"a": 1, "b": -2}, style="semantic")
print("vrwm protocol checks: passed")
