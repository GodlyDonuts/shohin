#!/usr/bin/env python3
"""Counterfactual-reflection protocol for a model-authored local register.

This is the training-time reflection component of Counterfactual Workspace
Induction (CWI).  It does not execute a transition at runtime.  During data
construction and auditing only, it creates grammar-valid candidate updates that
violate exactly one semantic invariant, then renders a value-bound reflection
target.  At evaluation, the reflection prompt is absent: the model must make
the ordinary direct register transition.

The protocol is deliberately built on the static-tape representation so a
failure can be localized to dynamic-state transport rather than repeated
copying of immutable operands.  It is not a claim that a model has a workspace
or that reflection is useful before a direct-transition primitive exists.
"""

from __future__ import annotations

from copy import deepcopy

from digitwise_factor_protocol import (
    apply_microstep,
    canonical_register,
    canonical_tape,
    initial_tape,
)


FOIL_KINDS = ("carry", "result_digit", "program_counter", "tape")


def _value_lsf(digits: str) -> int:
    return sum(int(digit) * (10 ** index) for index, digit in enumerate(str(digits)))


def _tape_with_one_changed_operand(tape):
    """Return a distinct grammar-valid tape of the same operation and width."""

    canonical_tape(tape)
    width = int(tape["w"])
    maximum = 10 ** width - 1
    left, right = _value_lsf(tape["a"]), _value_lsf(tape["b"])
    if tape["op"] == "add":
        if left < maximum:
            left += 1
        elif right > 0:
            right -= 1
        else:
            left -= 1
    elif left < maximum:
        left += 1
    elif right > 0:
        right -= 1
    else:
        right += 1
    changed = initial_tape(tape["op"], left, right, width)
    if changed == tape:
        raise AssertionError("tape foil did not change an operand")
    return changed


def make_semantic_foil(tape, register, kind):
    """Create one canonical near miss and retain the legal successor.

    The candidate always parses as a valid tape/register.  It is invalid only
    because it violates the semantic one-step transition for the original
    fixed tape and register.
    """

    if kind not in FOIL_KINDS:
        raise ValueError("unknown CWI foil kind: {}".format(kind))
    fixed_tape = deepcopy(tape)
    canonical_tape(fixed_tape)
    legal = apply_microstep(fixed_tape, register)
    candidate_tape, candidate_register = deepcopy(fixed_tape), deepcopy(legal)
    position = int(register["p"])
    if kind == "carry":
        candidate_register["c"] = 1 - int(legal["c"])
    elif kind == "result_digit":
        result = list(str(legal["r"]))
        result[position] = str((int(result[position]) + 1) % 10)
        candidate_register["r"] = "".join(result)
    elif kind == "program_counter":
        if int(legal["p"]) >= int(fixed_tape["w"]):
            raise ValueError("terminal transition has no grammar-valid program-counter foil")
        candidate_register["p"] = int(legal["p"]) + 1
        candidate_register["z"] = int(candidate_register["p"] == int(fixed_tape["w"]))
    else:
        candidate_tape = _tape_with_one_changed_operand(fixed_tape)
    canonical_tape(candidate_tape)
    canonical_register(candidate_tape, candidate_register)
    return {
        "fixed_tape": fixed_tape,
        "previous_register": deepcopy(register),
        "legal_register": legal,
        "candidate_tape": candidate_tape,
        "candidate_register": candidate_register,
        "foil_kind": kind,
    }


def _reflection_values(record):
    fixed_tape = record["fixed_tape"]
    legal = record["legal_register"]
    candidate_tape = record["candidate_tape"]
    candidate = record["candidate_register"]
    position = int(record["previous_register"]["p"])
    if candidate_tape != fixed_tape:
        return "illegal", "tape", fixed_tape["a"] + "/" + fixed_tape["b"], candidate_tape["a"] + "/" + candidate_tape["b"]
    if candidate == legal:
        return "legal", "none", "next", "next"
    if int(candidate["p"]) != int(legal["p"]):
        return "illegal", "program_counter", str(legal["p"]), str(candidate["p"])
    if int(candidate["c"]) != int(legal["c"]):
        return "illegal", "carry", str(legal["c"]), str(candidate["c"])
    if str(candidate["r"]) != str(legal["r"]):
        differing = [index for index, pair in enumerate(zip(str(candidate["r"]), str(legal["r"]))) if pair[0] != pair[1]]
        if differing != [position]:
            raise ValueError("result foil changed more than the active digit")
        return "illegal", "result_digit", str(legal["r"])[position], str(candidate["r"])[position]
    raise ValueError("candidate is neither legal nor a single-field CWI foil")


def reflection_prompt(record, style="core"):
    """Render a reflection-only prompt with no direct transition target."""

    for key in ("fixed_tape", "previous_register", "candidate_tape", "candidate_register", "legal_register"):
        if key not in record:
            raise ValueError("CWI record lacks {}".format(key))
    fixed = canonical_tape(record["fixed_tape"])
    previous = canonical_register(record["fixed_tape"], record["previous_register"])
    proposed_tape = canonical_tape(record["candidate_tape"])
    proposed = canonical_register(record["candidate_tape"], record["candidate_register"])
    if style == "core":
        return (
            "Audit a proposed local register update. The fixed tape is immutable. A legal update uses tape digit p "
            "with c, writes only r[p], and advances p once.\n"
            "Fixed tape: {}\nPrevious register: {}\nProposed tape: {}\nProposed register: {}\n"
            "Report whether it is legal. Name the one violated field and its expected and observed values.\nAnswer:"
        ).format(fixed, previous, proposed_tape, proposed)
    if style == "heldout":
        return (
            "Check one candidate rewrite against its frozen decimal tape. The operand record must not change; use the "
            "current index and carry/borrow for exactly one result-digit step.\n"
            "Frozen tape: {}\nCurrent register: {}\nCandidate tape: {}\nCandidate register: {}\n"
            "Give the verdict plus the single wrong field, expected value, and observed value.\nResult:"
        ).format(fixed, previous, proposed_tape, proposed)
    raise ValueError("unknown CWI prompt style")


def reflection_response(record):
    """Render a value-bound reflection target, never a direct register target."""

    verdict, field, expected, observed = _reflection_values(record)
    position = int(record["previous_register"]["p"])
    return "verdict={};field={};at_p={};expected={};observed={}".format(
        verdict, field, position, expected, observed
    )


def validate_reflection_record(record):
    """Independently rederive the semantic verdict and validate prompt fields."""

    fixed = record["fixed_tape"]
    previous = record["previous_register"]
    canonical_tape(fixed)
    canonical_register(fixed, previous)
    expected = apply_microstep(fixed, previous)
    if expected != record["legal_register"]:
        raise ValueError("CWI legal register is not the one-step successor")
    canonical_tape(record["candidate_tape"])
    canonical_register(record["candidate_tape"], record["candidate_register"])
    verdict, field, _, _ = _reflection_values(record)
    if record.get("foil_kind") not in ("legal",) + FOIL_KINDS:
        raise ValueError("invalid CWI foil metadata")
    if record["foil_kind"] == "legal" and verdict != "legal":
        raise ValueError("legal CWI record has an illegal verdict")
    if record["foil_kind"] != "legal" and (verdict != "illegal" or field != record["foil_kind"]):
        raise ValueError("CWI foil verdict does not match its semantic field")
    return True
