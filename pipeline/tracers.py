"""Execution-trace tracers: turn a verified RG (question, answer) into a genuine
worked-steps document (master plan §6.3 — "execution-trace pretraining").

Every trace is RE-VERIFIED: the trace's computed result must equal RG's verified
answer, else the item is dropped. So only correct derivations survive — rejection
sampling on our own reasoning. This is what makes the corpus safe to pretrain on.

Add a family by writing trace_<family>(question, answer, metadata) -> str | None
and registering it in TRACERS. Parsing can be loose; the verify-and-drop net keeps
only the traces that actually reproduce the answer.
"""
import re
from math import gcd as _gcd


def _expr(question):
    """Pull the arithmetic expression: after the last ':' , before the '='."""
    q = question
    if ":" in q:
        q = q.rsplit(":", 1)[1]
    return q.split("=")[0].strip()


def _tokens(expr):
    return re.findall(r"\d+\.?\d*|[+\-*/]", expr)


def _g(x):
    return f"{x:g}"


def trace_chain_sum(question, answer, md):
    toks = _tokens(_expr(question))
    if len(toks) < 3:
        return None
    try:
        total = int(toks[0])
        steps, i = [], 1
        while i + 1 <= len(toks) - 1 + 1 and i + 1 <= len(toks):
            op, val = toks[i], int(toks[i + 1])
            i += 2
            new = total + val if op == "+" else total - val
            steps.append(f"{total} {op} {val} = {new}")
            total = new
    except (ValueError, IndexError):
        return None
    if str(total) != str(answer).strip():
        return None
    return " ; ".join(steps)


def trace_decimal_chain_sum(question, answer, md):
    toks = _tokens(_expr(question))
    if len(toks) < 3:
        return None
    try:
        total = float(toks[0])
        steps, i = [], 1
        while i + 1 <= len(toks):
            op, val = toks[i], float(toks[i + 1])
            i += 2
            new = (total + val if op == "+" else total - val if op == "-"
                   else total * val if op == "*" else total / val)
            steps.append(f"{_g(total)} {op} {_g(val)} = {_g(new)}")
            total = new
        if abs(total - float(str(answer).strip())) > 1e-4:
            return None
    except (ValueError, IndexError, ZeroDivisionError):
        return None
    return " ; ".join(steps)


def trace_gcd(question, answer, md):
    ints = re.findall(r"-?\d+", question)
    if len(ints) < 2:
        return None
    x, y = abs(int(ints[0])), abs(int(ints[1]))
    steps = []
    while y:
        steps.append(f"gcd({x}, {y}): {x} mod {y} = {x % y}")
        x, y = y, x % y
    if str(x) != str(answer).strip():
        return None
    return " ; ".join(steps) + f" ; gcd = {x}"


def trace_lcm(question, answer, md):
    ints = re.findall(r"-?\d+", question)
    if len(ints) < 2:
        return None
    a, b = abs(int(ints[0])), abs(int(ints[1]))
    if a == 0 or b == 0:
        return None
    g = _gcd(a, b)
    lcm = a // g * b
    if str(lcm) != str(answer).strip():
        return None
    return f"gcd({a}, {b}) = {g} ; lcm = {a}*{b} / {g} = {lcm}"


TRACERS = {
    "chain_sum": trace_chain_sum,
    "decimal_chain_sum": trace_decimal_chain_sum,
    "gcd": trace_gcd,
    "lcm": trace_lcm,
}


def make_document(question, trace, answer):
    """Render the pretraining document: question + verified worked steps + answer."""
    return f"{question}\n<think>{trace}</think>\n<answer>{str(answer).strip()}</answer>"
