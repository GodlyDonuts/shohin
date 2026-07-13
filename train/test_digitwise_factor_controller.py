"""Prove the factorized controller transports registers without a solver repair."""
from digitwise_factor_controller import rollout_episode
from digitwise_factor_protocol import (
    apply_microstep,
    canonical_register,
    canonical_tape,
    initial_register,
    initial_tape,
)


def episode():
    tape = initial_tape("add", 29, 16, 4)
    register = initial_register(tape)
    expected = []
    for _ in range(4):
        register = apply_microstep(tape, register)
        expected.append(canonical_register(tape, register))
    return {
        "tape": canonical_tape(tape),
        "initial_register": canonical_register(tape, initial_register(tape)),
        "expected_registers": expected,
    }


item = episode()
responses = iter(item["expected_registers"] + ["answer=45"])
result = rollout_episode(item, lambda _prompt: next(responses), prompt_style="core")
assert result["success"] and result["state_closed_loop"]
assert all("dwt:" not in row["response"] for row in result["rows"])

responses = iter(["dwr:p=1;c=0;r=5000;z=0"])
failed = rollout_episode(item, lambda _prompt: next(responses), prompt_style="core")
assert not failed["success"] and not failed["state_closed_loop"]
assert len(failed["rows"]) == 1
print("digitwise factor controller checks: passed")
