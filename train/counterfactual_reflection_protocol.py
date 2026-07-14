"""Strict protocol primitives for the conditional reflection ablation.

This is deliberately only a CPU-only protocol definition.  It does not create
training data, run inference, or claim that a model has an internal state.  The
later experiment compares a direct-answer learner with a matched learner that
is additionally trained on a *counterfactual* interruption: a prompt asking
what state it would report before answering.  Normal evaluation never asks for
that report.

The module keeps the experimental boundary explicit:

* direct answers are source-visible and must not include a state carrier;
* reflection answers are exact carriers and never include a direct answer;
* neutral continuations have comparable fixed structure but contain no numeric
  task state;
* source-dropped consumers receive only a literal model-emitted carrier.

Generators and evaluators may calculate expected values from :class:`Record`.
Controllers may only validate a full emission and replace one frozen literal.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


STATE = re.compile(r"state:P=(-?\d+);Q=(-?\d+);D=(-?\d+)")
ANSWER = re.compile(r"answer=(-?\d+)")
# This deliberately retains the field/digit surface while fixing all values.
# It is therefore not a source-specific state and cannot answer a held-out task.
# A future builder must still measure actual tokenizer lengths and resample to
# equalize the token budget with the reflection arm.
NEUTRAL = "note:P=000;Q=000;D=000"


@dataclass(frozen=True)
class Record:
    """A solver-rendered two-value source record used by future data builders."""

    p: int
    q: int
    delta: int
    domain: str
    first_label: str
    second_label: str

    def __post_init__(self) -> None:
        if self.delta == 0:
            raise ValueError("delta must distinguish the before and after state")
        if not all(value >= 0 for value in (self.p, self.q, self.delta)):
            raise ValueError("protocol values must be non-negative")
        if not all(value and "\n" not in value for value in (self.domain, self.first_label, self.second_label)):
            raise ValueError("text fields must be non-empty single-line values")


def state(record: Record) -> str:
    """Return the canonical portable state used only as a supervised target."""
    return "state:P={};Q={};D={}".format(record.p + record.delta, record.q, record.delta)


def expected_answer(record: Record, operation: str) -> str:
    """Return a data/evaluator oracle; never call this from a rollout controller."""
    updated_p = record.p + record.delta
    if operation == "sum":
        value = updated_p + record.q
    elif operation == "difference":
        value = updated_p - record.q
    else:
        raise ValueError("unknown operation: {}".format(operation))
    return "answer={}".format(value)


def source_prefix(record: Record, *, heldout: bool) -> str:
    """Render an ordinary language record without exposing a carrier."""
    if heldout:
        return (
            "At a {domain}, the first operational role is {first} with value {p}; "
            "the second operational role is {second} with value {q}. "
            "A planned change raises the first role by {delta}."
        ).format(
            domain=record.domain,
            first=record.first_label,
            second=record.second_label,
            p=record.p,
            q=record.q,
            delta=record.delta,
        )
    return (
        "A {domain} record names {first} first and {second} second. "
        "It records {first}={p}, {second}={q}, and raises {first} by {delta}."
    ).format(
        domain=record.domain,
        first=record.first_label,
        second=record.second_label,
        p=record.p,
        q=record.q,
        delta=record.delta,
    )


def direct_question(record: Record, operation: str, *, heldout: bool) -> str:
    """Ask the ordinary task path, which must not request a state report."""
    if operation not in {"sum", "difference"}:
        raise ValueError("unknown operation: {}".format(operation))
    directive = (
        "After the change, return only answer=<integer> for first plus second."
        if operation == "sum"
        else "After the change, return only answer=<integer> for first minus second."
    )
    return "{} {}".format(source_prefix(record, heldout=heldout), directive)


def reflection_question(record: Record, *, heldout: bool) -> str:
    """Ask a counterfactual interruption, never used on the ordinary path."""
    return (
        "{} Before answering the task, an external interruption asks what exact "
        "portable state would be needed after the change. Emit only that state."
    ).format(source_prefix(record, heldout=heldout))


def neutral_question(record: Record, *, heldout: bool) -> str:
    """A length-stable auxiliary continuation with no numeric state target."""
    return (
        "{} Before answering the task, an external interruption asks for a "
        "non-numeric continuation marker. Emit only the marker."
    ).format(source_prefix(record, heldout=heldout))


def consumer_question(record: Record, operation: str, carrier: str, *, heldout: bool) -> str:
    """Render a source-dropped downstream question around exactly one carrier."""
    if exact_state(carrier) is None:
        raise ValueError("consumer requires a full exact carrier")
    if operation not in {"sum", "difference"}:
        raise ValueError("unknown operation: {}".format(operation))
    wording = (
        "Return only answer=<integer> for P plus Q."
        if operation == "sum"
        else "Return only answer=<integer> for P minus Q."
    )
    prefix = "The original record is unavailable." if not heldout else "The original description has been erased."
    return "{} Received portable state:\n{}\n{}".format(prefix, carrier, wording)


def exact_state(text: str) -> str | None:
    """Accept an emitted state only when the entire stripped response matches."""
    emitted = str(text).strip()
    return emitted if STATE.fullmatch(emitted) else None


def exact_answer(text: str) -> str | None:
    """Accept an emitted answer only when the entire stripped response matches."""
    emitted = str(text).strip()
    return emitted if ANSWER.fullmatch(emitted) else None


def forward_state(prompt: str, expected_state: str, emitted_state: str) -> str:
    """Literal-only forwarding primitive for a future source-dropped evaluator."""
    if exact_state(emitted_state) is None:
        raise ValueError("cannot forward malformed state")
    if prompt.count(expected_state) != 1:
        raise ValueError("prompt must contain exactly one frozen state literal")
    return prompt.replace(expected_state, emitted_state, 1)
