"""Contract tests for the factorized static-tape recurrent register."""
from digitwise_factor_protocol import (
    apply_microstep,
    canonical_register,
    canonical_tape,
    initial_register,
    initial_tape,
    local_context,
    parse_register,
    parse_tape,
    register_answer,
)


add_tape = initial_tape("add", 29, 16, 4)
assert canonical_tape(add_tape) == "dwt:op=add;w=4;a=9200;b=6100"
register = initial_register(add_tape)
assert canonical_register(add_tape, register) == "dwr:p=0;c=0;r=0000;z=0"
for _ in range(4):
    register = apply_microstep(add_tape, register)
assert canonical_register(add_tape, register) == "dwr:p=4;c=0;r=5400;z=1"
assert register_answer(add_tape, register) == 45
assert local_context(add_tape, initial_register(add_tape)) == (4, "add", 0, 0, 9, 6)

sub_tape = initial_tape("sub", 1000, 1, 4)
sub_register = initial_register(sub_tape)
for _ in range(4):
    sub_register = apply_microstep(sub_tape, sub_register)
assert register_answer(sub_tape, sub_register) == 999

assert parse_tape(canonical_tape(add_tape)) == add_tape
assert parse_register(canonical_register(add_tape, register), add_tape) == register
assert parse_register("dwr:p=4;c=0;r=5400;z=1\ndwr:p=4;c=0;r=5400;z=1", add_tape) is None
assert parse_register("dwr:p=4;c=0;r=5400;z=1", sub_tape) == {"p": 4, "c": 0, "r": "5400", "z": 1}
print("digitwise factor protocol checks: passed")
