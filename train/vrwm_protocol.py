"""Pure protocol for Verified Recursive Working Memory (VRWM).

VRWM is an external bounded-memory controller for small models.  The model sees
one natural-language instruction plus a canonical two-variable working-memory
line, emits the next line, and a controller carries that emitted state to the
next instruction.  The controller never supplies an answer or corrects a
model-produced state; it only parses valid states and retrieves the next
instruction from the program store.
"""
from __future__ import annotations

import re


MEMORY_RE = re.compile(r"(?mi)^\s*(wm:a=(-?\d+);b=(-?\d+))\s*$")
ANSWER_RE = re.compile(r"(?i)answer\s*=\s*(-?\d+)")
VARIABLES = ("a", "b")
PROMPT_STYLES = ("default", "paraphrase", "semantic")


def canonical_memory(values):
    """Return the only accepted serialization for a two-integer memory."""
    if set(values) != set(VARIABLES):
        raise ValueError(f"expected variables {VARIABLES}, got {sorted(values)}")
    return f"wm:a={int(values['a'])};b={int(values['b'])}"


def parse_memory(text):
    """Extract one canonical working-memory line, or return ``None``."""
    matches = MEMORY_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, a, b = matches[0]
    return {"a": int(a), "b": int(b)}


def parse_answer(text):
    values = ANSWER_RE.findall(str(text))
    return int(values[-1]) if values else None


def validate_operation(operation):
    kind = operation.get("kind")
    target = operation.get("target")
    if target not in VARIABLES:
        raise ValueError(f"invalid target: {target!r}")
    if kind in {"add_const", "sub_const"}:
        if not isinstance(operation.get("value"), int):
            raise ValueError(f"missing integer value for {kind}")
        return
    if kind in {"add_var", "sub_var"}:
        if operation.get("source") not in VARIABLES:
            raise ValueError(f"invalid source: {operation.get('source')!r}")
        return
    if kind == "swap":
        if target != "a" or operation.get("source") != "b":
            raise ValueError("swap must be represented as target=a, source=b")
        return
    raise ValueError(f"unknown operation kind: {kind!r}")


def apply_operation(values, operation):
    """Apply one operation without mutating the caller's memory."""
    validate_operation(operation)
    result = {name: int(values[name]) for name in VARIABLES}
    kind, target = operation["kind"], operation["target"]
    if kind == "add_const":
        result[target] += operation["value"]
    elif kind == "sub_const":
        result[target] -= operation["value"]
    elif kind == "add_var":
        result[target] += result[operation["source"]]
    elif kind == "sub_var":
        result[target] -= result[operation["source"]]
    else:  # swap
        result["a"], result["b"] = result["b"], result["a"]
    return result


def render_instruction(operation, style="default"):
    """Render a structured operation in a fixed or held-out language style."""
    validate_operation(operation)
    kind, target = operation["kind"], operation["target"]
    if style == "default":
        if kind == "add_const":
            return f"increase {target} by {operation['value']}"
        if kind == "sub_const":
            return f"decrease {target} by {operation['value']}"
        if kind == "add_var":
            return f"add the current value of {operation['source']} to {target}"
        if kind == "sub_var":
            return f"subtract the current value of {operation['source']} from {target}"
        return "swap the values of a and b"
    if style == "paraphrase":
        if kind == "add_const":
            return f"add {operation['value']} to register {target}"
        if kind == "sub_const":
            return f"subtract {operation['value']} from register {target}"
        if kind == "add_var":
            return f"replace {target} with {target} plus {operation['source']}"
        if kind == "sub_var":
            return f"replace {target} with {target} minus {operation['source']}"
        return "exchange the contents of registers a and b"
    if style == "semantic":
        if kind == "add_const":
            return f"set {target} to {target} + {operation['value']}"
        if kind == "sub_const":
            return f"set {target} to {target} - {operation['value']}"
        if kind == "add_var":
            return f"set {target} to {target} + {operation['source']}"
        if kind == "sub_var":
            return f"set {target} to {target} - {operation['source']}"
        return "set the pair (a, b) to (b, a)"
    raise ValueError(f"unknown prompt style: {style!r}")


def transition_prompt(memory, operation, style="default"):
    instruction = render_instruction(operation, style=style)
    if style == "default":
        return (
            "Question: Apply exactly one instruction to the working memory.\n"
            f"Working memory: {canonical_memory(memory)}\n"
            f"Instruction: {instruction}.\n"
            "Return only the next canonical working-memory line in the form "
            "wm:a=<integer>;b=<integer>.\nAnswer:"
        )
    if style == "paraphrase":
        return (
            "Task: Update the two registers once using the command below.\n"
            f"Registers: {canonical_memory(memory)}\n"
            f"Command: {instruction}.\n"
            "Emit just the resulting wm:a=<integer>;b=<integer> line.\nResult:"
        )
    if style == "semantic":
        return (
            "Compute exactly one register update.\n"
            f"State: {canonical_memory(memory)}\n"
            f"Rule: {instruction}.\n"
            "Write only the new wm:a=<integer>;b=<integer> state.\nAnswer:"
        )
    raise ValueError(f"unknown prompt style: {style!r}")


def repair_prompt(memory, operation, proposal, style="default"):
    """Ask the model to verify its own proposed next state without a solver."""
    instruction = render_instruction(operation, style=style)
    proposed = canonical_memory(proposal)
    if style == "default":
        return (
            "Question: Verify a proposed next working-memory state.\n"
            f"Working memory: {canonical_memory(memory)}\n"
            f"Instruction: {instruction}.\n"
            f"Proposed next state: {proposed}\n"
            "Check the update, then return the corrected canonical working-memory line.\nAnswer:"
        )
    if style == "paraphrase":
        return (
            "Task: Check the proposed register update.\n"
            f"Registers before: {canonical_memory(memory)}\n"
            f"Command: {instruction}.\n"
            f"Proposal: {proposed}\n"
            "Emit the verified wm:a=<integer>;b=<integer> state.\nResult:"
        )
    if style == "semantic":
        return (
            "Validate the candidate result for one register rule.\n"
            f"State before: {canonical_memory(memory)}\n"
            f"Rule: {instruction}.\n"
            f"Candidate: {proposed}\n"
            "Write the correct wm:a=<integer>;b=<integer> state.\nAnswer:"
        )
    raise ValueError(f"unknown prompt style: {style!r}")


def readout_prompt(memory, variable, style="default"):
    if variable not in VARIABLES:
        raise ValueError(f"invalid readout variable: {variable!r}")
    if style == "default":
        return (
            f"Question: Read the value of {variable} from this working memory.\n"
            f"Working memory: {canonical_memory(memory)}\n"
            f"Return only answer=<integer>.\nAnswer:"
        )
    if style == "paraphrase":
        return (
            f"Task: Look up register {variable}.\n"
            f"Registers: {canonical_memory(memory)}\n"
            "Emit just answer=<integer>.\nResult:"
        )
    if style == "semantic":
        return (
            f"Inspect the register state and report {variable}.\n"
            f"State: {canonical_memory(memory)}\n"
            "Write answer=<integer>.\nAnswer:"
        )
    raise ValueError(f"unknown prompt style: {style!r}")
