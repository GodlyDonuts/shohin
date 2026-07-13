"""Factorized static-tape recurrent register protocol.

The original DRS state rewrites immutable operand tapes on every model turn.
This protocol separates that fixed evidence from the compact state the model
must actually carry: the controller re-sends the original tape verbatim while
forwarding only a model-emitted register.  It never computes a transition,
repairs a register, or changes the tape.

Digits are least-significant first.  The tape is immutable problem context;
the register contains the program counter, carry/borrow, result tape, and
terminal bit.  This is a causal representation experiment, not a claim that
an external controller can solve arithmetic for the model.
"""
from __future__ import annotations

import re


TAPE_RE = re.compile(r"(?mi)^\s*(dwt:op=(add|sub);w=(\d+);a=(\d+);b=(\d+))\s*$")
REGISTER_RE = re.compile(r"(?mi)^\s*(dwr:p=(\d+);c=([01]);r=(\d+);z=([01]))\s*$")
ANSWER_RE = re.compile(r"(?mi)^\s*answer=(-?\d+)\s*$")
DIGIT_RE = re.compile(r"(?mi)^\s*digit=([0-9])\s*$")
OPERATIONS = ("add", "sub")
PROMPT_STYLES = ("core", "heldout")


def _value_lsf(digits):
    return sum(int(digit) * (10 ** index) for index, digit in enumerate(str(digits)))


def _digits_lsf(value, width):
    if not isinstance(value, int) or not isinstance(width, int) or width <= 0:
        raise ValueError("value and width must be positive integers")
    if value < 0 or value >= 10 ** width:
        raise ValueError("value does not fit the requested width")
    return "".join(str((value // (10 ** index)) % 10) for index in range(width))


def _validate_tape(tape):
    if set(tape) != {"op", "w", "a", "b"}:
        raise ValueError("invalid tape keys")
    if tape["op"] not in OPERATIONS:
        raise ValueError("invalid tape operation")
    width = int(tape["w"])
    if width <= 0:
        raise ValueError("invalid tape width")
    for key in ("a", "b"):
        value = str(tape[key])
        if len(value) != width or not value.isdigit():
            raise ValueError("invalid operand tape")
    if tape["op"] == "sub" and _value_lsf(tape["a"]) < _value_lsf(tape["b"]):
        raise ValueError("subtraction tape requires a nonnegative result")


def canonical_tape(tape):
    """Validate and render the immutable input tape."""
    normalized = {
        "op": str(tape["op"]),
        "w": int(tape["w"]),
        "a": str(tape["a"]),
        "b": str(tape["b"]),
    }
    _validate_tape(normalized)
    return "dwt:op={op};w={w};a={a};b={b}".format(**normalized)


def _validate_register(register, width):
    if set(register) != {"p", "c", "r", "z"}:
        raise ValueError("invalid register keys")
    width, position, carry, terminal = (int(width), int(register["p"]), int(register["c"]), int(register["z"]))
    if width <= 0 or position < 0 or position > width or carry not in (0, 1) or terminal not in (0, 1):
        raise ValueError("invalid register scalar")
    if terminal != int(position == width):
        raise ValueError("terminal flag does not match program counter")
    result = str(register["r"])
    if len(result) != width or not result.isdigit():
        raise ValueError("invalid result tape")
    if any(character != "0" for character in result[position:]):
        raise ValueError("unwritten result suffix is not zero")
    if position == 0 and carry:
        raise ValueError("initial register cannot carry")


def canonical_register(tape, register):
    """Validate and render only the model-carried recurrent state."""
    canonical_tape(tape)
    normalized = {
        "p": int(register["p"]),
        "c": int(register["c"]),
        "r": str(register["r"]),
        "z": int(register["z"]),
    }
    _validate_register(normalized, tape["w"])
    return "dwr:p={p};c={c};r={r};z={z}".format(**normalized)


def parse_tape(text):
    matches = TAPE_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, operation, width, left, right = matches[0]
    tape = {"op": operation, "w": int(width), "a": left, "b": right}
    try:
        canonical_tape(tape)
    except ValueError:
        return None
    return tape


def parse_register(text, tape):
    """Extract a valid register against a caller-supplied immutable tape."""
    canonical_tape(tape)
    matches = REGISTER_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, position, carry, result, terminal = matches[0]
    register = {"p": int(position), "c": int(carry), "r": result, "z": int(terminal)}
    try:
        canonical_register(tape, register)
    except ValueError:
        return None
    return register


def initial_tape(operation, left, right, width):
    if operation not in OPERATIONS:
        raise ValueError("invalid operation")
    if operation == "sub" and left < right:
        raise ValueError("subtraction requires left >= right")
    tape = {
        "op": operation,
        "w": int(width),
        "a": _digits_lsf(int(left), int(width)),
        "b": _digits_lsf(int(right), int(width)),
    }
    canonical_tape(tape)
    return tape


def initial_register(tape):
    canonical_tape(tape)
    register = {"p": 0, "c": 0, "r": "0" * int(tape["w"]), "z": 0}
    canonical_register(tape, register)
    return register


def apply_microstep(tape, register):
    """Solver-only transition used by generation and auditing, never rollout."""
    canonical_register(tape, register)
    if register["z"]:
        raise ValueError("cannot step a terminal register")
    position = register["p"]
    left, right, carry = int(tape["a"][position]), int(tape["b"][position]), int(register["c"])
    if tape["op"] == "add":
        total = left + right + carry
        digit, next_carry = total % 10, total // 10
    else:
        total = left - right - carry
        digit, next_carry = (total + 10) % 10, int(total < 0)
    result = list(register["r"])
    result[position] = str(digit)
    next_position = position + 1
    next_register = {
        "p": next_position,
        "c": next_carry,
        "r": "".join(result),
        "z": int(next_position == int(tape["w"])),
    }
    canonical_register(tape, next_register)
    return next_register


def local_context(tape, register):
    """Return the exact local information required for the next register update."""
    canonical_register(tape, register)
    if register["z"]:
        return None
    position = int(register["p"])
    return (
        int(tape["w"]), str(tape["op"]), position, int(register["c"]),
        int(tape["a"][position]), int(tape["b"][position]),
    )


def register_answer(tape, register):
    canonical_register(tape, register)
    if not register["z"]:
        raise ValueError("answer requested before terminal register")
    result = _value_lsf(register["r"])
    if tape["op"] == "add":
        return result + int(register["c"]) * (10 ** int(tape["w"]))
    if register["c"]:
        raise ValueError("terminal subtraction borrowed despite nonnegative invariant")
    return result


def register_digit(tape, register, position):
    canonical_register(tape, register)
    position = int(position)
    if position < 0 or position >= int(register["p"]):
        raise ValueError("digit has not been written")
    return int(register["r"][position])


def parse_answer(text):
    matches = ANSWER_RE.findall(str(text))
    return int(matches[-1]) if matches else None


def parse_digit(text):
    matches = DIGIT_RE.findall(str(text))
    return int(matches[-1]) if matches else None


def microstep_prompt(tape, register, style="core"):
    tape_line, register_line = canonical_tape(tape), canonical_register(tape, register)
    if style == "core":
        return (
            "Local register update. The tape is fixed evidence and must not be rewritten. Digits in a, b, and r "
            "are least-significant first. Use tape digit p with register c, write only r[p], and advance p.\n"
            "Tape: {}\nRegister: {}\nReturn exactly one dwr register line.\nAnswer:"
        ).format(tape_line, register_line)
    if style == "heldout":
        return (
            "Replay one local decimal rewrite. Keep the supplied operand tape fixed; the short register is the only "
            "state to advance. Position zero is least-significant and c is carry or borrow.\n"
            "Fixed tape: {}\nWorking register: {}\nEmit only the next canonical dwr line.\nResult:"
        ).format(tape_line, register_line)
    raise ValueError("unknown prompt style")


def digit_prompt(tape, register, position, style="core"):
    tape_line, register_line = canonical_tape(tape), canonical_register(tape, register)
    if style == "core":
        return (
            "Read one finished digit from this fixed-tape register. Position zero is least-significant.\n"
            "Tape: {}\nRegister: {}\nPosition: {}\nReturn only digit=<0-9>.\nAnswer:"
        ).format(tape_line, register_line, int(position))
    if style == "heldout":
        return (
            "Inspect the completed part of the short register under this unchanged operand tape. Index zero is "
            "least-significant.\nFixed tape: {}\nWorking register: {}\nRequested index: {}\n"
            "Emit only digit=<0-9>.\nResult:"
        ).format(tape_line, register_line, int(position))
    raise ValueError("unknown prompt style")


def final_prompt(tape, register, style="core"):
    tape_line, register_line = canonical_tape(tape), canonical_register(tape, register)
    if not register["z"]:
        raise ValueError("final prompt requires terminal register")
    if style == "core":
        return (
            "Read the completed decimal result from this terminal register. The result tape is least-significant "
            "first; include final carry for addition.\nTape: {}\nRegister: {}\n"
            "Return only answer=<integer>.\nAnswer:"
        ).format(tape_line, register_line)
    if style == "heldout":
        return (
            "Convert this finished fixed-tape local-rewrite record into its ordinary decimal result. The result "
            "tape is least-significant first and addition may have a final carry.\nFixed tape: {}\n"
            "Working register: {}\nEmit only answer=<integer>.\nResult:"
        ).format(tape_line, register_line)
    raise ValueError("unknown prompt style")
