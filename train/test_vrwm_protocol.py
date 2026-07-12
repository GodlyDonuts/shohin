#!/usr/bin/env python3
"""Pure protocol checks for canonical working memory and transitions."""
from vrwm_protocol import apply_operation, canonical_memory, parse_answer, parse_memory


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
print("vrwm protocol checks: passed")
