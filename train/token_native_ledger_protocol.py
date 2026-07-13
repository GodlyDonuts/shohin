"""Token-native append ledger for isolated recurrent-state research.

Each model-produced arithmetic transition is exactly three already-existing
tokenizer special tokens.  In fixed order they encode the next position,
carry/borrow, and result digit.  The format deliberately avoids new tokenizer
entries and model changes: it is a representation-length control, not a
hidden controller or a claim of latent reasoning.

The controller may only transport an emitted triple verbatim.  Arithmetic in
this module is restricted to dataset construction and independent scoring.
"""
from __future__ import annotations

import hashlib

from digitwise_factor_protocol import (
    apply_microstep,
    canonical_tape,
    initial_register,
    parse_tape,
    register_answer,
)


# IDs 2..11 in the frozen Shohin tokenizer.  They are retained as literal
# tokens by generation when skip_special_tokens=False, so no vocabulary resize
# or embedding initialization is required.
CODE_TOKENS = (
    "<think>", "</think>", "<code>", "</code>", "<answer>", "</answer>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|correct|>",
)
PROMPT_STYLES = ("core", "heldout")


def _value_to_token(value):
    value = int(value)
    if not 0 <= value < len(CODE_TOKENS):
        raise ValueError("microcode value must be in [0, 9]")
    return CODE_TOKENS[value]


def _token_to_value(text, start):
    for value, token in enumerate(CODE_TOKENS):
        if str(text).startswith(token, start):
            return value, start + len(token)
    return None, start


def canonical_delta(delta):
    """Render a fixed three-token `(next_position, carry, digit)` carrier."""
    normalized = {key: int(delta[key]) for key in ("p", "c", "d")}
    if not 0 <= normalized["p"] <= 9 or normalized["c"] not in (0, 1) or not 0 <= normalized["d"] <= 9:
        raise ValueError("invalid token-native delta")
    return "".join(_value_to_token(normalized[key]) for key in ("p", "c", "d"))


def parse_delta(text):
    """Accept exactly one three-token carrier and reject all surrounding text."""
    text, cursor, values = str(text), 0, []
    for _ in range(3):
        value, cursor = _token_to_value(text, cursor)
        if value is None:
            return None
        values.append(value)
    if cursor != len(text):
        return None
    result = {"p": values[0], "c": values[1], "d": values[2]}
    try:
        if canonical_delta(result) != text:
            return None
    except ValueError:
        return None
    return result


def initial_delta():
    """The only controller-supplied carrier is a fixed, non-arithmetic start."""
    return {"p": 0, "c": 0, "d": 0}


def context_key(tape):
    """Opaque fixed identifier derived only from immutable tape text."""
    return "tnl-" + hashlib.sha256(canonical_tape(tape).encode()).hexdigest()[:16]


def expected_delta(tape, position, carry):
    """Solver-only local transition used by generation and independent scoring."""
    canonical_tape(tape)
    position, carry = int(position), int(carry)
    width = int(tape["w"])
    if position < 0 or position >= width or carry not in (0, 1):
        raise ValueError("invalid local microcode context")
    register = initial_register(tape)
    for _ in range(position):
        register = apply_microstep(tape, register)
    if int(register["c"]) != carry:
        raise ValueError("carry does not match this tape position")
    advanced = apply_microstep(tape, register)
    return {"p": int(advanced["p"]), "c": int(advanced["c"]), "d": int(advanced["r"][position])}


def expected_answer(tape):
    """Solver-only final answer used by generation/audit, never by rollout."""
    canonical_tape(tape)
    register = initial_register(tape)
    for _ in range(int(tape["w"])):
        register = apply_microstep(tape, register)
    return register_answer(tape, register)


def transition_prompt(tape, prior_delta, style="core"):
    """Ask for the next atomic carrier using only fixed tape plus last emission."""
    tape_line, prior_line, key = canonical_tape(tape), canonical_delta(prior_delta), context_key(tape)
    if style == "core":
        return (
            "Key {}. Tape plus p,c,d carrier. Emit three special tokens.\n"
            "Tape: {}\nPrior: {}\nAnswer:"
        ).format(key, tape_line, prior_line)
    if style == "heldout":
        return (
            "Key {}. Fixed tape plus p,c,d carrier. Emit next triple.\n"
            "Fixed tape: {}\nCarrier: {}\nResult:"
        ).format(key, tape_line, prior_line)
    raise ValueError("unknown prompt style")


def final_prompt(deltas, key, style="core"):
    """Read only model-authored token triples; the operand tape is intentionally absent."""
    triples = [canonical_delta(parse_delta(item) if isinstance(item, str) else item) for item in deltas]
    if not triples or not str(key).startswith("tnl-"):
        raise ValueError("final token-native ledger cannot be empty")
    # A three-token carrier has deliberately tiny entropy.  Repeat the opaque
    # immutable key between triples so train and held-out final prompts cannot
    # share a long carrier substring.  The key is never decoded, generated, or
    # used for arithmetic; it only bounds the ledger to its fixed tape.
    rendered = "".join("{}:{}".format(key, triple) for triple in triples)
    if style == "core":
        return (
            "Key {}. Decode p,c,d triples low-order first. Return answer=<integer>.\n"
            "Ledger: {}\nAnswer:"
        ).format(key, rendered)
    if style == "heldout":
        return (
            "Key {}. Recover integer from p,c,d triples low-order first. Emit answer=<integer>.\n"
            "Checkpoints: {}\nResult:"
        ).format(key, rendered)
    raise ValueError("unknown prompt style")


def tape_from_text(text):
    """Small public adapter used by the transport-only controller."""
    return parse_tape(text)
