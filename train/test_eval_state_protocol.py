#!/usr/bin/env python3
"""Small contract checks for typed-state extraction and exact validation."""
from eval_state_protocol import state_from_response


assert state_from_response("state=n:12>19>76>63\nThe answer is 63.") == "state=n:12>19>76>63"
assert state_from_response("reasoning\nstate=base:8;digits:725;value:469\nThe answer is 469.") == "state=base:8;digits:725;value:469"
assert state_from_response("The answer is 63.") == ""
print("typed state protocol evaluator checks: passed")
