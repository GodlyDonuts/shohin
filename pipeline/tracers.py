"""Execution-trace tracers: turn a verified RG (question, answer) into a genuine
worked-steps document (master plan §6.3 — "execution-trace pretraining").

Every trace is RE-VERIFIED: its computed result must equal RG's verified answer,
else the item is dropped. Only correct derivations survive — rejection sampling on
our own reasoning. This is what makes the corpus safe to pretrain on.

Arithmetic uses a real AST post-order evaluator, so precedence and parentheses are
respected and each binary op is logged in the order it's actually computed. Add a
family by writing trace_<family>(question, answer, metadata) -> str | None and
registering it in TRACERS; loose parsing is fine — the verify-and-drop net keeps
only traces that reproduce the answer.
"""
import re, ast
from math import gcd as _gcd

_ALLOWED = set("0123456789.+-*/() ")


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:g}" if isinstance(x, float) else str(x)


def _extract_expr(q):
    s = q
    for trig in ("problem:", "Calculate", "Compute", "evaluate", ":"):
        if trig in s:
            s = s.split(trig, 1)[1]
            break
    s = s.split("=")[0]
    return s.strip().rstrip(".").strip()


def _eval_steps(node, steps):
    if isinstance(node, ast.Expression):
        return _eval_steps(node.body, steps)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_steps(node.operand, steps)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        l = _eval_steps(node.left, steps)
        r = _eval_steps(node.right, steps)
        t = type(node.op)
        if t is ast.Add:
            res, op = l + r, "+"
        elif t is ast.Sub:
            res, op = l - r, "-"
        elif t is ast.Mult:
            res, op = l * r, "*"
        elif t is ast.Div:
            if r == 0:
                raise ZeroDivisionError
            res, op = l / r, "/"
        else:
            raise ValueError
        steps.append(f"{_fmt(l)} {op} {_fmt(r)} = {_fmt(res)}")
        return res
    raise ValueError("unsupported")


def trace_arithmetic(question, answer, md):
    """chain_sum / decimal_chain_sum / basic_arithmetic / decimal_arithmetic."""
    expr = _extract_expr(question)
    if not expr or any(c not in _ALLOWED for c in expr):
        return None
    try:
        steps = []
        val = _eval_steps(ast.parse(expr, mode="eval"), steps)
    except Exception:
        return None
    if not steps:
        return None
    try:
        if abs(float(val) - float(str(answer).strip())) > 1e-4:
            return None
    except ValueError:
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
    if not a or not b:
        return None
    g = _gcd(a, b)
    lcm = a // g * b
    if str(lcm) != str(answer).strip():
        return None
    return f"gcd({a}, {b}) = {g} ; lcm = {a}*{b} / {g} = {lcm}"


TRACERS = {
    "chain_sum": trace_arithmetic,
    "decimal_chain_sum": trace_arithmetic,
    "basic_arithmetic": trace_arithmetic,
    "decimal_arithmetic": trace_arithmetic,
    "gcd": trace_gcd,
    "lcm": trace_lcm,
}


def make_document(question, trace, answer):
    return f"{question}\n<think>{trace}</think>\n<answer>{str(answer).strip()}</answer>"
