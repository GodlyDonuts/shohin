"""Append-only delta-ledger protocol for bounded recurrent arithmetic.

The model emits short digit/carry deltas, not a rewritten full machine state.
At fixed boundaries it emits a compact block of its own previous deltas.  The
controller is allowed only to transport exact model text and schedule fixed
turns; it never computes, repairs, or selects arithmetic content.
"""
from __future__ import annotations

import hashlib
import re


BASE_RE = re.compile(r"(?mi)^\s*(adl:op=(add|sub);w=(\d+);a=(\d+);b=(\d+))\s*$")
DELTA_RE = re.compile(r"(?mi)^\s*(adl:step=(\d+);d=([0-9]);c=([01]))\s*$")
BLOCK_RE = re.compile(r"(?mi)^\s*(adl:block=(\d+);digits=(\d+);c=([01]))\s*$")
ANSWER_RE = re.compile(r"(?mi)^\s*answer=(-?\d+)\s*$")
OPERATIONS = ("add", "sub")
PROMPT_STYLES = ("core", "heldout")


def _digits_lsf(value, width):
    if value < 0 or value >= 10 ** width:
        raise ValueError("value does not fit width")
    return "".join(str((value // 10 ** position) % 10) for position in range(width))


def _value_lsf(digits):
    return sum(int(digit) * 10 ** position for position, digit in enumerate(str(digits)))


def canonical_base(base):
    base = {"op": str(base["op"]), "w": int(base["w"]), "a": str(base["a"]), "b": str(base["b"])}
    if base["op"] not in OPERATIONS or base["w"] <= 0:
        raise ValueError("invalid base")
    if any(len(base[field]) != base["w"] or not base[field].isdigit() for field in ("a", "b")):
        raise ValueError("invalid base tape")
    if base["op"] == "sub" and _value_lsf(base["a"]) < _value_lsf(base["b"]):
        raise ValueError("negative subtraction is outside v1")
    return "adl:op={op};w={w};a={a};b={b}".format(**base)


def parse_base(text):
    matches = BASE_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, operation, width, a_tape, b_tape = matches[0]
    base = {"op": operation, "w": int(width), "a": a_tape, "b": b_tape}
    try:
        canonical_base(base)
    except ValueError:
        return None
    return base


def canonical_delta(delta):
    delta = {"step": int(delta["step"]), "d": int(delta["d"]), "c": int(delta["c"])}
    if delta["step"] < 0 or not 0 <= delta["d"] <= 9 or delta["c"] not in (0, 1):
        raise ValueError("invalid delta")
    return "adl:step={step};d={d};c={c}".format(**delta)


def parse_delta(text):
    matches = DELTA_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, step, digit, carry = matches[0]
    delta = {"step": int(step), "d": int(digit), "c": int(carry)}
    try:
        canonical_delta(delta)
    except ValueError:
        return None
    return delta


def canonical_block(block):
    block = {"block": int(block["block"]), "digits": str(block["digits"]), "c": int(block["c"])}
    if block["block"] < 0 or not block["digits"] or not block["digits"].isdigit() or block["c"] not in (0, 1):
        raise ValueError("invalid block")
    return "adl:block={block};digits={digits};c={c}".format(**block)


def parse_block(text):
    matches = BLOCK_RE.findall(str(text))
    if len(matches) != 1:
        return None
    _, index, digits, carry = matches[0]
    block = {"block": int(index), "digits": digits, "c": int(carry)}
    try:
        canonical_block(block)
    except ValueError:
        return None
    return block


def initial_base(operation, left, right, width):
    if operation not in OPERATIONS or (operation == "sub" and left < right):
        raise ValueError("invalid operands")
    base = {"op": operation, "w": int(width), "a": _digits_lsf(int(left), int(width)), "b": _digits_lsf(int(right), int(width))}
    canonical_base(base)
    return base


def expected_delta(base, step, carry):
    canonical_base(base)
    step, carry = int(step), int(carry)
    if step < 0 or step >= base["w"] or carry not in (0, 1):
        raise ValueError("invalid local transition")
    left, right = int(base["a"][step]), int(base["b"][step])
    total = left + right + carry if base["op"] == "add" else left - right - carry
    return {"step": step, "d": total % 10 if total >= 0 else (total + 10) % 10, "c": total // 10 if base["op"] == "add" else int(total < 0)}


def expected_block(index, deltas):
    parsed = [parse_delta(item) if isinstance(item, str) else item for item in deltas]
    if not parsed or any(item is None for item in parsed):
        raise ValueError("empty or malformed delta block")
    return {"block": int(index), "digits": "".join(str(item["d"]) for item in parsed), "c": int(parsed[-1]["c"])}


def expected_answer(base):
    canonical_base(base)
    left, right = _value_lsf(base["a"]), _value_lsf(base["b"])
    return left + right if base["op"] == "add" else left - right


def parse_answer(text):
    matches = ANSWER_RE.findall(str(text))
    return int(matches[-1]) if matches else None


def _ledger_key(base):
    # Fixed identifier derived from immutable input only. It is never sampled,
    # computed from a model response, or used to select content.
    return "k" + hashlib.sha256(canonical_base(base).encode()).hexdigest()[:16]


def _lines(items, key):
    # Bind both ends of every short record to its immutable episode key. This
    # prevents generic multi-delta suffixes from becoming accidental train/eval
    # n-gram overlap while leaving the model output itself short.
    return " | ".join("ctx={} {} ctx={}".format(key, item, key) for item in items) if items else "-"


def transition_prompt(base, blocks, live, step, style="core"):
    base_line = canonical_base(base)
    key = _ledger_key(base)
    if style == "core":
        return (
            "Append one local decimal delta. a and b are least-significant first. "
            "Use the scheduled step and the last carried c from the retained ledger.\n"
            "Base: {base}\nBlocks: {blocks}\nLive: {live}\nScheduled step: {step}\n"
            "Emit only adl:step=<n>;d=<0-9>;c=<0|1>.\nAnswer:"
        ).format(base=base_line, blocks=_lines(blocks, key), live=_lines(live, key), step=int(step))
    if style == "heldout":
        return (
            "Replay the next atomic decimal rewrite. Tapes use low-order digit first; preserve the retained "
            "ledger and add only this scheduled digit/carry record.\n"
            "Machine: {base}\nCompressed blocks: {blocks}\nOpen records: {live}\nNext index: {step}\n"
            "Return exactly adl:step=<n>;d=<0-9>;c=<0|1>.\nResult:"
        ).format(base=base_line, blocks=_lines(blocks, key), live=_lines(live, key), step=int(step))
    raise ValueError("unknown prompt style")


def compact_prompt(base, blocks, live, block_index, style="core"):
    base_line = canonical_base(base)
    key = _ledger_key(base)
    if style == "core":
        return (
            "Compact exactly these model-authored live deltas into one ledger block. Copy their digits in order "
            "and retain the final c; do not solve or add new content.\n"
            "Base: {base}\nEarlier blocks: {blocks}\nLive: {live}\nBlock index: {index}\n"
            "Emit only adl:block=<n>;digits=<digits>;c=<0|1>.\nAnswer:"
        ).format(base=base_line, blocks=_lines(blocks, key), live=_lines(live, key), index=int(block_index))
    if style == "heldout":
        return (
            "Fold the open retained records into one short checkpoint. Preserve their digit order and last carry.\n"
            "Machine: {base}\nPrior checkpoints: {blocks}\nRecords to fold: {live}\nCheckpoint index: {index}\n"
            "Return exactly adl:block=<n>;digits=<digits>;c=<0|1>.\nResult:"
        ).format(base=base_line, blocks=_lines(blocks, key), live=_lines(live, key), index=int(block_index))
    raise ValueError("unknown prompt style")


def final_prompt(base, blocks, style="core"):
    base_line = canonical_base(base)
    key = _ledger_key(base)
    if style == "core":
        return (
            "Read the completed ordinary decimal result from the model-authored ledger blocks. Block digits are "
            "least-significant first; addition may retain a final carry.\n"
            "Base: {base}\nBlocks: {blocks}\nReturn only answer=<integer>.\nAnswer:"
        ).format(base=base_line, blocks=_lines(blocks, key))
    if style == "heldout":
        return (
            "Convert the retained compact checkpoints into the finished decimal result. Their digits are low-order first.\n"
            "Machine: {base}\nCheckpoints: {blocks}\nEmit only answer=<integer>.\nResult:"
        ).format(base=base_line, blocks=_lines(blocks, key))
    raise ValueError("unknown prompt style")
