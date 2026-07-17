"""Canonical protocol for a digitwise recurrent scratchpad.

The model holds a fixed-width decimal machine state.  Every recurrent turn
rewrites exactly one result digit using the current operand digits and a
carry/borrow bit.  The controller may carry a model-emitted state to the next
turn, but never executes, repairs, or selects a state.

Digits in ``a``, ``b``, and ``r`` are serialized least-significant first.
``p`` is the digit position to process next; ``z`` is one only for the terminal
state.  This is deliberately a discrete, locally verifiable protocol rather
than another continuous packet-memory interface.
"""

from __future__ import annotations

import re


STATE_RE = re.compile(
    r"(?mi)^\s*(dws:op=(add|sub);w=(\d+);p=(\d+);c=([01]);a=(\d+);b=(\d+);r=(\d+);z=([01]))\s*$"
)
ANSWER_RE = re.compile(r"(?mi)^\s*answer=(-?\d+)\s*$")
DIGIT_RE = re.compile(r"(?mi)^\s*digit=([0-9])\s*$")
OPERATIONS = ("add", "sub")
PROMPT_STYLES = ("core", "heldout")


def _digits_lsf(value, width):
    if not isinstance(value, int) or not isinstance(width, int) or width <= 0:
        raise ValueError("value and width must be positive integers")
    if value < 0 or value >= 10**width:
        raise ValueError("value does not fit the requested width")
    return "".join(str((value // (10**index)) % 10) for index in range(width))


def _value_lsf(digits):
    return sum(int(digit) * (10**index) for index, digit in enumerate(str(digits)))


def _validate_state(state):
    required = {"op", "w", "p", "c", "a", "b", "r", "z"}
    if set(state) != required:
        raise ValueError("invalid state keys")
    if state["op"] not in OPERATIONS:
        raise ValueError("invalid operation")
    width, position, carry, terminal = (
        int(state[name]) for name in ("w", "p", "c", "z")
    )
    if (
        width <= 0
        or position < 0
        or position > width
        or carry not in (0, 1)
        or terminal not in (0, 1)
    ):
        raise ValueError("invalid scalar state field")
    if terminal != int(position == width):
        raise ValueError("terminal flag does not match position")
    for name in ("a", "b", "r"):
        value = str(state[name])
        if len(value) != width or not value.isdigit():
            raise ValueError("invalid digit tape")
    if any(character != "0" for character in state["r"][position:]):
        raise ValueError("unwritten result tape suffix is not zero")
    if position == 0 and carry:
        raise ValueError("initial state cannot carry")
    if state["op"] == "sub" and _value_lsf(state["a"]) < _value_lsf(state["b"]):
        raise ValueError("subtraction protocol requires nonnegative results")


def canonical_state(state):
    """Validate and render the only accepted scratchpad state serialization."""
    normalized = {
        "op": str(state["op"]),
        "w": int(state["w"]),
        "p": int(state["p"]),
        "c": int(state["c"]),
        "a": str(state["a"]),
        "b": str(state["b"]),
        "r": str(state["r"]),
        "z": int(state["z"]),
    }
    _validate_state(normalized)
    return "dws:op={op};w={w};p={p};c={c};a={a};b={b};r={r};z={z}".format(**normalized)


def parse_state(text):
    """Extract exactly one valid canonical DWS state from model output."""
    matches = STATE_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, operation, width, position, carry, a_tape, b_tape, result_tape, terminal = (
        matches[0]
    )
    state = {
        "op": operation,
        "w": int(width),
        "p": int(position),
        "c": int(carry),
        "a": a_tape,
        "b": b_tape,
        "r": result_tape,
        "z": int(terminal),
    }
    try:
        canonical_state(state)
    except ValueError:
        return None
    return state


def initial_state(operation, left, right, width):
    """Build an exact initial state.  This solver is never used by rollout control."""
    if operation not in OPERATIONS:
        raise ValueError("invalid operation")
    if operation == "sub" and left < right:
        raise ValueError("subtraction requires left >= right")
    state = {
        "op": operation,
        "w": int(width),
        "p": 0,
        "c": 0,
        "a": _digits_lsf(int(left), int(width)),
        "b": _digits_lsf(int(right), int(width)),
        "r": "0" * int(width),
        "z": 0,
    }
    canonical_state(state)
    return state


def apply_microstep(state):
    """Apply one local decimal transition without mutating the input state."""
    canonical_state(state)
    if state["z"]:
        raise ValueError("cannot step a terminal state")
    position = state["p"]
    left, right, carry = (
        int(state["a"][position]),
        int(state["b"][position]),
        int(state["c"]),
    )
    if state["op"] == "add":
        total = left + right + carry
        digit, next_carry = total % 10, total // 10
    else:
        total = left - right - carry
        digit, next_carry = (total + 10) % 10, int(total < 0)
    result = list(state["r"])
    result[position] = str(digit)
    next_position = position + 1
    next_state = dict(state)
    next_state.update(
        {
            "p": next_position,
            "c": next_carry,
            "r": "".join(result),
            "z": int(next_position == state["w"]),
        }
    )
    canonical_state(next_state)
    return next_state


def state_answer(state):
    """Return the arithmetic result encoded by a terminal state."""
    canonical_state(state)
    if not state["z"]:
        raise ValueError("answer requested before terminal state")
    result = _value_lsf(state["r"])
    if state["op"] == "add":
        return result + int(state["c"]) * (10 ** state["w"])
    if state["c"]:
        raise ValueError("terminal subtraction borrowed despite nonnegative invariant")
    return result


def state_digit(state, position):
    """Read a computed result digit.  Position zero is least significant."""
    canonical_state(state)
    position = int(position)
    if position < 0 or position >= state["p"]:
        raise ValueError("digit has not been written")
    return int(state["r"][position])


def parse_answer(text):
    matches = ANSWER_RE.findall(str(text))
    return int(matches[0]) if len(matches) == 1 else None


def parse_digit(text):
    matches = DIGIT_RE.findall(str(text))
    return int(matches[-1]) if matches else None


def microstep_prompt(state, style="core"):
    line = canonical_state(state)
    if style == "core":
        return (
            "Microstate update. Digits in a, b, and r are least-significant first. "
            "Use the digit at p with c, write only r[p], then advance p by one.\n"
            "State: {}\nReturn exactly one dws state line.\nAnswer:"
        ).format(line)
    if style == "heldout":
        return (
            "Replay one local decimal rewrite. The retained machine record uses position zero for the "
            "least-significant digit; c is the carry or borrow. Change one r digit and the control fields.\n"
            "Machine record: {}\nEmit only the next canonical dws line.\nResult:"
        ).format(line)
    raise ValueError("unknown prompt style")


def digit_prompt(state, position, style="core"):
    line = canonical_state(state)
    if style == "core":
        return (
            "Read a computed digit from this scratchpad. Position zero is least-significant.\n"
            "State: {}\nPosition: {}\nReturn only digit=<0-9>.\nAnswer:"
        ).format(line, int(position))
    if style == "heldout":
        return (
            "Inspect the retained machine record and report one finished result digit. Index zero means "
            "the least-significant place.\nMachine record: {}\nRequested index: {}\nEmit only digit=<0-9>.\nResult:"
        ).format(line, int(position))
    raise ValueError("unknown prompt style")


def final_prompt(state, style="core"):
    line = canonical_state(state)
    if not state["z"]:
        raise ValueError("final prompt requires terminal state")
    if style == "core":
        return (
            "Read the completed decimal result from this terminal scratchpad. The result tape is "
            "least-significant first; include final carry for addition.\n"
            "State: {}\nReturn only answer=<integer>.\nAnswer:"
        ).format(line)
    if style == "heldout":
        return (
            "Convert this finished local-rewrite record into its ordinary decimal result. The r tape is "
            "least-significant first and addition may have a final carry.\n"
            "Machine record: {}\nEmit only answer=<integer>.\nResult:"
        ).format(line)
    raise ValueError("unknown prompt style")
