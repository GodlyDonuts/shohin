"""Pure protocol for a counterfactual bisimulation compiler experiment.

The protocol is intentionally solver-free at inference time.  A controller may
transport model-emitted ``cbc:`` states, parse their fixed grammar, and compare
them after the run.  Generation and auditing use the semantic helpers below,
but a future controller must never call them to repair a state or answer a
query.
"""
from __future__ import annotations

import re
from typing import Mapping


STATE = re.compile(r"^cbc:([a-z][a-z0-9_]*=-?\d+(?:;[a-z][a-z0-9_]*=-?\d+)*)$")
DELTA = re.compile(r"^cbc-delta:(add|sub|move|swap);(.+)$")


def _keys(keys):
    result = tuple(str(key) for key in keys)
    if len(result) != 2 or len(set(result)) != 2 or any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in result):
        raise ValueError("CBC requires two distinct lowercase field keys")
    return result


def copy_values(values: Mapping[str, object], keys):
    keys = _keys(keys)
    if set(values) != set(keys):
        raise ValueError("values do not match the declared CBC keys")
    return {key: int(values[key]) for key in keys}


def canonical_state(values: Mapping[str, object], keys) -> str:
    values = copy_values(values, keys)
    return "cbc:" + ";".join("{}={}".format(key, values[key]) for key in _keys(keys))


def parse_state(text: object, keys):
    keys = _keys(keys)
    candidates = [line.strip() for line in str(text).splitlines() if line.strip().startswith("cbc:")]
    if len(candidates) != 1 or not STATE.fullmatch(candidates[0]):
        return None
    pairs = candidates[0][4:].split(";")
    if len(pairs) != len(keys):
        return None
    values = {}
    for pair, key in zip(pairs, keys):
        found_key, value = pair.split("=", 1)
        if found_key != key or found_key in values:
            return None
        values[found_key] = int(value)
    return values if set(values) == set(keys) else None


def apply_operation(values: Mapping[str, object], operation: Mapping[str, object], keys):
    values = copy_values(values, keys)
    kind = str(operation["kind"])
    if kind in {"add", "sub"}:
        key, amount = str(operation["key"]), int(operation["value"])
        if key not in values or amount <= 0:
            raise ValueError("invalid one-field operation")
        values[key] = values[key] + amount if kind == "add" else values[key] - amount
    elif kind == "move":
        source, target, amount = str(operation["source"]), str(operation["target"]), int(operation["value"])
        if source not in values or target not in values or source == target or amount <= 0:
            raise ValueError("invalid move operation")
        values[source] -= amount
        values[target] += amount
    elif kind == "swap":
        left, right = str(operation["left"]), str(operation["right"])
        if left not in values or right not in values or left == right:
            raise ValueError("invalid swap operation")
        values[left], values[right] = values[right], values[left]
    else:
        raise ValueError("unknown CBC operation")
    return values


def canonical_delta(operation: Mapping[str, object], keys) -> str:
    keys = _keys(keys)
    kind = str(operation["kind"])
    if kind in {"add", "sub"}:
        key, amount = str(operation["key"]), int(operation["value"])
        if key not in keys or amount <= 0:
            raise ValueError("invalid one-field delta")
        return "cbc-delta:{};key={};value={}".format(kind, key, amount)
    if kind == "move":
        source, target, amount = str(operation["source"]), str(operation["target"]), int(operation["value"])
        if source not in keys or target not in keys or source == target or amount <= 0:
            raise ValueError("invalid move delta")
        return "cbc-delta:move;from={};to={};value={}".format(source, target, amount)
    if kind == "swap":
        left, right = str(operation["left"]), str(operation["right"])
        if left not in keys or right not in keys or left == right:
            raise ValueError("invalid swap delta")
        return "cbc-delta:swap;left={};right={}".format(left, right)
    raise ValueError("unknown CBC delta")


def parse_delta(text: object, keys):
    keys = _keys(keys)
    candidates = [line.strip() for line in str(text).splitlines() if line.strip().startswith("cbc-delta:")]
    if len(candidates) != 1:
        return None
    match = DELTA.fullmatch(candidates[0])
    if not match:
        return None
    kind, fields = match.groups()
    values = {}
    for part in fields.split(";"):
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        if key in values:
            return None
        values[key] = value
    try:
        if kind in {"add", "sub"} and set(values) == {"key", "value"}:
            result = {"kind": kind, "key": values["key"], "value": int(values["value"])}
        elif kind == "move" and set(values) == {"from", "to", "value"}:
            result = {"kind": kind, "source": values["from"], "target": values["to"], "value": int(values["value"])}
        elif kind == "swap" and set(values) == {"left", "right"}:
            result = {"kind": kind, "left": values["left"], "right": values["right"]}
        else:
            return None
        canonical_delta(result, keys)
        return result
    except ValueError:
        return None


def render_event(operation: Mapping[str, object], item: str, style: str) -> str:
    kind = str(operation["kind"])
    if style not in {"train", "heldout"}:
        raise ValueError("unknown CBC wording style")
    if kind in {"add", "sub"}:
        verb = "adds" if kind == "add" else "removes"
        if style == "heldout":
            return "The {} register {} {} {}.".format(operation["key"], verb, operation["value"], item)
        return "{} {} {} {}.".format("Add" if kind == "add" else "Subtract", operation["value"], item, "to {}".format(operation["key"]) if kind == "add" else "from {}".format(operation["key"]))
    if kind == "move":
        if style == "heldout":
            return "Relocate {} {} from {} into {}.".format(operation["value"], item, operation["source"], operation["target"])
        return "Move {} {} from {} to {}.".format(operation["value"], item, operation["source"], operation["target"])
    if kind == "swap":
        if style == "heldout":
            return "Exchange the stored quantities for {} and {}.".format(operation["left"], operation["right"])
        return "Swap the values of {} and {}.".format(operation["left"], operation["right"])
    raise ValueError("unknown CBC operation")


def compile_prompt(values: Mapping[str, object], keys, domain: str, item: str, reference: str, style: str, variant: str) -> str:
    values = copy_values(values, keys)
    left, right = _keys(keys)
    if style == "train":
        if variant == "a":
            return (
                "Question: Convert this factual {} record into a compact causal state. Reference {} is not a quantity. "
                "{} contains {} {} and {} contains {} {}. Return only cbc:{}=<integer>;{}=<integer>.\nAnswer:"
            ).format(domain, reference, left, values[left], item, right, values[right], item, left, right)
        return (
            "Task: Preserve both quantities from this {} note before its prose is discarded. Identifier {} is not a value. "
            "The {} count is {} {} while the {} count is {} {}. Emit exactly cbc:{}=<integer>;{}=<integer>.\nResult:"
        ).format(domain, reference, left, values[left], item, right, values[right], item, left, right)
    if variant == "a":
        return (
            "Archive instruction: encode this {} inventory without retaining its sentence. Record {} is only a label. "
            "It lists {} {} beside {} and {} {} beside {}. Produce one cbc:{}=<integer>;{}=<integer> line.\nOutput:"
        ).format(domain, reference, values[left], item, left, values[right], item, right, left, right)
    return (
        "A {} ledger tagged {} will be removed after compression. Its named totals are {}={} {} and {}={} {}. "
        "Write only the canonical cbc:{}=<integer>;{}=<integer> carrier.\nResponse:"
    ).format(domain, reference, left, values[left], item, right, values[right], item, left, right)


def update_prompt(state: str, event: str, reference: str, revision: int, style: str) -> str:
    if style == "train":
        return (
            "Question: The source record is gone. Continue reference {} revision {} using only this causal state.\n"
            "State: {}\nEvent: {}\nReturn only the next cbc state.\nAnswer:"
        ).format(reference, revision, state, event)
    return (
        "Retained carrier only, source discarded. For archive {} update {}, apply the new event.\n"
        "Carrier: {}\nIncoming event: {}\nEmit exactly one next cbc state.\nOutput:"
    ).format(reference, revision, state, event)


def delta_prompt(before: str, after: str, reference: str, revision: int, style: str) -> str:
    if style == "train":
        return (
            "Question: Identify the one causal update between two compact states for reference {} revision {}.\n"
            "Before: {}\nAfter: {}\nReturn only the cbc-delta line.\nAnswer:"
        ).format(reference, revision, before, after)
    return (
        "Inspect adjacent retained carriers from archive {} update {}.\nPrior carrier: {}\nLater carrier: {}\n"
        "Emit exactly one cbc-delta line.\nOutput:"
    ).format(reference, revision, before, after)


def sum_query_prompt(state: str, keys, reference: str, style: str) -> str:
    left, right = _keys(keys)
    if style == "train":
        return (
            "Question: The original record is unavailable. From this causal state for reference {}, report {} plus {}.\n"
            "State: {}\nReturn only answer=<integer>.\nAnswer:"
        ).format(reference, left, right, state)
    return (
        "Only a retained carrier remains for archive {}. What is the combined {} and {} quantity?\n"
        "Carrier: {}\nEmit only answer=<integer>.\nOutput:"
    ).format(reference, left, right, state)

