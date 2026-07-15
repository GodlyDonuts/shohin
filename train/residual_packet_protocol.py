"""Pure protocol helpers for source-deleted residual packet control.

The runtime-facing helpers in this module only parse, render, select the first
operation, delete that operation, and transport observed integers. Arithmetic
helpers are provided for frozen-board generation and independent audits; a
source-blind controller must not call them while evaluating a model.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType


OPERATIONS = ("add", "multiply", "subtract")
TRAIN_SOURCE_TEMPLATE_IDS = ("train_0", "train_1", "train_2", "train_3")
RESERVED_SOURCE_TEMPLATE_ID = "reserved"
SOURCE_TEMPLATE_IDS = TRAIN_SOURCE_TEMPLATE_IDS + (RESERVED_SOURCE_TEMPLATE_ID,)

_SOURCE_TEMPLATE_ITEMS = (
    (
        "train_0",
        "Begin with {state}. Execute this sequence from left to right: {clauses}.",
    ),
    (
        "train_1",
        "Take {state} as the running value. Perform, in sequence: {clauses}.",
    ),
    (
        "train_2",
        "The starting number is {state}. Make these changes in order: {clauses}.",
    ),
    (
        "train_3",
        "Set the running total to {state}. Follow the listed commands: {clauses}.",
    ),
    (
        RESERVED_SOURCE_TEMPLATE_ID,
        "Initialize the value to {state}. Apply these instructions in order: {clauses}.",
    ),
)
SOURCE_TEMPLATES = MappingProxyType(dict(_SOURCE_TEMPLATE_ITEMS))
TRAIN_SOURCE_TEMPLATES = MappingProxyType(
    {name: SOURCE_TEMPLATES[name] for name in TRAIN_SOURCE_TEMPLATE_IDS}
)
RESERVED_SOURCE_TEMPLATE = SOURCE_TEMPLATES[RESERVED_SOURCE_TEMPLATE_ID]

_ASCII_WHITESPACE = " \t\n\r\f\v"
_INTEGER_PATTERN = r"(?:0|[1-9]\d*|-[1-9]\d*)"
_POSITIVE_INTEGER_PATTERN = r"(?:[1-9]\d*)"
_OPERATION_PATTERN = (
    r"(?:(?:add|multiply|subtract) " + _POSITIVE_INTEGER_PATTERN + r")"
)
_OPERATION_RE = re.compile(r"\A(" + "|".join(OPERATIONS) + r") ([1-9]\d*)\Z", re.ASCII)
_PACKET_RE = re.compile(
    r"\AState: (" + _INTEGER_PATTERN + r")\nPlan: ("
    + _OPERATION_PATTERN
    + r"(?:; "
    + _OPERATION_PATTERN
    + r")*)\Z",
    re.ASCII,
)
_ANSWER_RE = re.compile(r"\AAnswer: (" + _INTEGER_PATTERN + r")\Z", re.ASCII)

# This is the frozen native parser used by the Problem/Work executor. It is
# intentionally different from the exact packet and answer channel parsers.
INTEGER = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?![A-Za-z0-9_,]|\.\d)"
)


def _require_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_integer(value, name="value"):
    return str(_require_integer(value, name))


def normalize_operation(operation, operand=None):
    """Return one validated operation as ``(name, positive_operand)``."""
    if operand is None:
        if (
            isinstance(operation, (str, bytes))
            or not isinstance(operation, (tuple, list))
            or len(operation) != 2
        ):
            raise ValueError("operation must be a two-item tuple or list")
        operation, operand = operation
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation!r}")
    operand = _require_integer(operand, "operand")
    if operand <= 0:
        raise ValueError("operand must be positive")
    return operation, operand


def normalize_plan(plan, require_nonempty=True):
    """Return a validated immutable operation plan."""
    if isinstance(plan, (str, bytes)):
        raise ValueError("plan must be an iterable of structured operations")
    try:
        normalized = tuple(normalize_operation(operation) for operation in plan)
    except TypeError as error:
        raise ValueError("plan must be an iterable of structured operations") from error
    if require_nonempty and not normalized:
        raise ValueError("packet plans must contain at least one operation")
    return normalized


def render_operation(operation, operand=None):
    """Render one operation in the exact packet grammar."""
    name, value = normalize_operation(operation, operand)
    return f"{name} {value}"


def parse_operation(text):
    """Parse one exact operation, returning ``None`` on any deviation."""
    if not isinstance(text, str):
        return None
    match = _OPERATION_RE.fullmatch(text)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def canonical_packet(state, plan):
    """Render the only valid nonterminal State/Plan packet."""
    state_text = _canonical_integer(state, "state")
    operations = normalize_plan(plan)
    return f"State: {state_text}\nPlan: " + "; ".join(
        render_operation(operation) for operation in operations
    )


def canonical_answer(value):
    """Render the only valid terminal answer channel."""
    return f"Answer: {_canonical_integer(value, 'answer')}"


def _model_output_body(text):
    if not isinstance(text, str):
        return None
    return text.strip(_ASCII_WHITESPACE)


def parse_packet(text):
    """Parse exactly one canonical packet with no extra non-whitespace text."""
    body = _model_output_body(text)
    if body is None:
        return None
    match = _PACKET_RE.fullmatch(body)
    if match is None:
        return None
    plan = tuple(parse_operation(item) for item in match.group(2).split("; "))
    if any(operation is None for operation in plan):
        return None
    return {"state": int(match.group(1)), "plan": plan}


def parse_answer(text):
    """Parse exactly one canonical answer with no extra non-whitespace text."""
    body = _model_output_body(text)
    if body is None:
        return None
    match = _ANSWER_RE.fullmatch(body)
    return int(match.group(1)) if match is not None else None


def parse_controller_output(text):
    """Parse a packet or answer channel without accepting mixed channels."""
    packet = parse_packet(text)
    if packet is not None:
        return {"channel": "packet", "state": packet["state"], "plan": packet["plan"]}
    answer = parse_answer(text)
    if answer is not None:
        return {"channel": "answer", "answer": answer}
    return None


def _coerce_packet(packet):
    if isinstance(packet, str):
        parsed = parse_packet(packet)
        if parsed is None:
            raise ValueError("invalid packet text")
        return parsed
    if not isinstance(packet, Mapping) or set(packet) != {"state", "plan"}:
        raise ValueError("packet must contain exactly state and plan")
    state = _require_integer(packet["state"], "state")
    plan = normalize_plan(packet["plan"])
    return {"state": state, "plan": plan}


def compiler_prompt(source):
    """Render the frozen compiler prompt for one printable ASCII source line."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    if source != source.strip(" ") or any(ord(character) < 32 or ord(character) > 126 for character in source):
        raise ValueError("source must be one trimmed printable ASCII line")
    return f"Problem: {source}\nCompile only the execution packet.\nPacket:"


def update_prompt(packet, observed_result):
    """Render the frozen source-free updater prompt."""
    current = _coerce_packet(packet)
    observed_text = _canonical_integer(observed_result, "observed result")
    return (
        f"Packet:\n{canonical_packet(current['state'], current['plan'])}\n"
        f"Observed result: {observed_text}\nNext packet:"
    )


def expected_update(packet, observed_result):
    """Copy an observation and delete exactly one operation without arithmetic."""
    current = _coerce_packet(packet)
    observed_result = _require_integer(observed_result, "observed result")
    residual = current["plan"][1:]
    if residual:
        return canonical_packet(observed_result, residual)
    return canonical_answer(observed_result)


def parse_exact_update(packet, observed_result, response):
    """Return a parsed update only when state, deletion, and channel are exact."""
    expected = expected_update(packet, observed_result)
    body = _model_output_body(response)
    if body != expected:
        return None
    return parse_controller_output(response)


def update_is_exact(packet, observed_result, response):
    """Report whether a model response is the unique valid next output."""
    return parse_exact_update(packet, observed_result, response) is not None


def apply_operation(state, operation, operand=None):
    """Apply one operation for board construction or independent scoring."""
    state = _require_integer(state, "state")
    name, value = normalize_operation(operation, operand)
    if name == "add":
        return state + value
    if name == "multiply":
        return state * value
    return state - value


def trajectory(initial_state, plan):
    """Return the initial state followed by every mathematical next state."""
    state = _require_integer(initial_state, "initial state")
    operations = normalize_plan(plan, require_nonempty=False)
    states = [state]
    for operation in operations:
        state = apply_operation(state, operation)
        states.append(state)
    return tuple(states)


def apply_plan(initial_state, plan):
    """Return the final mathematical state for a structured plan."""
    return trajectory(initial_state, plan)[-1]


def packet_trajectory(initial_state, plan):
    """Render the gold compiler packet, updater packets, and terminal answer."""
    operations = normalize_plan(plan)
    states = trajectory(initial_state, operations)
    outputs = [
        canonical_packet(states[index], operations[index:])
        for index in range(len(operations))
    ]
    outputs.append(canonical_answer(states[-1]))
    return tuple(outputs)


def source_clause(operation, operand=None):
    """Render one natural-language clause for a frozen source template."""
    name, value = normalize_operation(operation, operand)
    if name == "multiply":
        return f"multiply by {value}"
    return f"{name} {value}"


def render_source_clauses(plan):
    """Render the semicolon-separated clauses shared by all source templates."""
    return "; ".join(source_clause(operation) for operation in normalize_plan(plan))


def render_source(initial_state, plan, template_id):
    """Render one source from the frozen training or reserved template set."""
    initial_state = _require_integer(initial_state, "initial state")
    if initial_state <= 0:
        raise ValueError("source initial state must be positive")
    if template_id not in SOURCE_TEMPLATES:
        raise ValueError(f"unknown source template: {template_id!r}")
    return SOURCE_TEMPLATES[template_id].format(
        state=initial_state,
        clauses=render_source_clauses(plan),
    )


def operation_clause(value, operation, operand=None):
    """Render the immutable native arithmetic executor question."""
    value = _require_integer(value, "value")
    name, operand = normalize_operation(operation, operand)
    if name == "add":
        return f"Compute {value} plus {operand}."
    if name == "subtract":
        return f"Compute {value} minus {operand}."
    return f"Compute {value} times {operand}."


def format_atomic_prompt(value, operation, operand=None):
    """Render one native Problem/Work call to the immutable executor."""
    return f"Problem: {operation_clause(value, operation, operand)}\nWork:"


atomic_prompt = format_atomic_prompt


def first_nonempty_line(text):
    """Return the stripped first nonempty response line."""
    if not isinstance(text, str):
        return ""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def parse_first_line_integer(text):
    """Parse only the last integer on the first nonempty executor line."""
    values = INTEGER.findall(first_nonempty_line(text))
    return int(values[-1].replace(",", "")) if values else None


parse_first_line_final = parse_first_line_integer
compute_trajectory = trajectory
