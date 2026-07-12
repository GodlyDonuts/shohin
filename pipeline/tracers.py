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
import ast
import json
import re
from fractions import Fraction
from math import gcd as _gcd

_ALLOWED = set("0123456789.+-*/() ")


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.12g}" if isinstance(x, float) else str(x)


def _extract_expr(q):
    s = q
    # Some Reasoning-Gym decimal prompts put a prose instruction first and the
    # arithmetic expression alone on their final line (for example, "7.4-5/8 =
    # ?"). Prefer that syntactically safe line before applying legacy triggers.
    for line in reversed(s.splitlines()):
        candidate = line.strip()
        if "=" in candidate:
            candidate = candidate.split("=", 1)[0].strip()
        if candidate and all(char in _ALLOWED for char in candidate):
            return candidate
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


def _plain_answer(value):
    return re.sub(r"[^a-z0-9./-]", "", str(value).lower())


def trace_fraction_simplification(question, answer, md):
    m = re.search(r"\$?(-?\d+)\s*/\s*(-?\d+)\$?", question)
    if not m:
        return None
    numerator, denominator = int(m.group(1)), int(m.group(2))
    if denominator == 0:
        return None
    frac = Fraction(numerator, denominator)
    result = f"{frac.numerator}/{frac.denominator}"
    if _plain_answer(answer) != result:
        return None
    return (f"gcd({abs(numerator)}, {abs(denominator)}) = "
            f"{abs(numerator) // abs(frac.numerator) if frac.numerator else abs(denominator)} ; "
            f"divide numerator and denominator to get {result}")


def _linear_coeff(expr, variable):
    """Return a*x+b for a restricted arithmetic expression, or raise ValueError."""
    tree = ast.parse(expr.replace("^", "**"), mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(0), Fraction(node.value)
        if isinstance(node, ast.Name) and node.id == variable:
            return Fraction(1), Fraction(0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a, b = walk(node.operand)
            return (a, b) if isinstance(node.op, ast.UAdd) else (-a, -b)
        if isinstance(node, ast.BinOp):
            a1, b1 = walk(node.left)
            a2, b2 = walk(node.right)
            if isinstance(node.op, ast.Add):
                return a1 + a2, b1 + b2
            if isinstance(node.op, ast.Sub):
                return a1 - a2, b1 - b2
            if isinstance(node.op, ast.Mult):
                if a1 and a2:
                    raise ValueError("nonlinear")
                return a1 * b2 + a2 * b1, b1 * b2
            if isinstance(node.op, ast.Div):
                if a2:
                    raise ValueError("variable denominator")
                return a1 / b2, b1 / b2
        raise ValueError("unsupported")

    return walk(tree)


def _frac_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def trace_simple_equations(question, answer, md):
    m = re.search(r"(?:satisfies|that)\s*:\s*(.+?)\s*=\s*(.+?)(?:\n|$)", question, flags=re.I)
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip()
    variables = sorted(set(re.findall(r"\b[a-zA-Z]\b", left + " " + right)))
    if len(variables) != 1:
        return None
    var = variables[0]
    try:
        a1, b1 = _linear_coeff(left, var)
        a2, b2 = _linear_coeff(right, var)
        coefficient, constant = a1 - a2, b2 - b1
        if not coefficient:
            return None
        solution = constant / coefficient
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    result = _frac_text(solution)
    if _plain_answer(answer) != _plain_answer(result):
        return None
    return (f"Move constants: ({_frac_text(a1)} - {_frac_text(a2)})*{var} = "
            f"{_frac_text(b2)} - {_frac_text(b1)} = {_frac_text(constant)} ; "
            f"{var} = {_frac_text(constant)} / {_frac_text(coefficient)} = {result}")


def _from_base(digits, base):
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    value = 0
    for char in digits.lower():
        digit = alphabet.index(char)
        if digit >= base:
            raise ValueError("invalid digit")
        value = value * base + digit
    return value


def _to_base(value, base):
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = []
    while value:
        value, digit = divmod(value, base)
        out.append(alphabet[digit])
    return "".join(reversed(out))


def trace_base_conversion(question, answer, md):
    m = re.search(r"base-(\d+) number\s+([0-9a-z]+)\s+to\s+base-(\d+)", question, flags=re.I)
    if not m:
        return None
    source_base, digits, target_base = int(m.group(1)), m.group(2), int(m.group(3))
    try:
        decimal = _from_base(digits, source_base)
        result = _to_base(decimal, target_base)
    except ValueError:
        return None
    if _plain_answer(answer) != _plain_answer(result):
        return None
    return f"{digits}_{source_base} = {decimal}_{10} ; {decimal}_{10} = {result}_{target_base}"


def trace_letter_counting(question, answer, md):
    m = re.search(r'letter\s+"(.+?)"\s+appear in the text:\s+"(.*?)"', question, flags=re.I)
    if not m:
        return None
    letter, text = m.group(1), m.group(2)
    count = text.lower().count(letter.lower())
    if _plain_answer(answer) != str(count):
        return None
    return f"Count '{letter}' in '{text}' = {count}"


def trace_spell_backward(question, answer, md):
    m = re.search(r"Spell this word backward.*?:\s*([A-Za-z]+)\s*$", question, flags=re.S)
    if not m:
        return None
    word = m.group(1)
    result = word[::-1]
    if str(answer).strip() != result:
        return None
    return f"Reverse the characters of {word}: {result}"


def trace_isomorphic_strings(question, answer, md):
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if not lines:
        return None
    pair = lines[-1].split()
    if len(pair) != 2:
        return None
    left, right = pair
    forward, reverse = {}, {}
    ok = len(left) == len(right)
    for a, b in zip(left, right):
        if (a in forward and forward[a] != b) or (b in reverse and reverse[b] != a):
            ok = False
            break
        forward[a], reverse[b] = b, a
    result = str(ok)
    if _plain_answer(answer) != result.lower():
        return None
    pairs = ", ".join(f"{a}->{b}" for a, b in forward.items())
    return f"Mapping {pairs} is one-to-one, so the strings are {result}"


def trace_word_sorting(question, answer, md):
    m = re.search(r"sort these words in (ascending|descending).*?:\s*(.+)$", question, flags=re.I | re.S)
    if not m:
        return None
    direction = m.group(1).lower()
    words = [word.strip() for word in m.group(2).strip().split(",") if word.strip()]
    if not words:
        return None
    result_words = sorted(words, reverse=(direction == "descending"))
    result = ", ".join(result_words)
    if str(answer).strip() != result:
        return None
    return f"ASCII sort ({direction}): {result}"


def trace_group_anagrams(question, answer, md):
    m = re.search(r"Group the following list of words into anagrams:\s*(\[.*\])\s*$", question, flags=re.S)
    if not m:
        return None
    try:
        words = json.loads(m.group(1))
        groups = {}
        for word in words:
            groups.setdefault("".join(sorted(word)), []).append(word)
        result_groups = [sorted(group) for _, group in sorted(groups.items())]
        gold_groups = [sorted(group) for group in json.loads(answer)]
    except (json.JSONDecodeError, TypeError):
        return None
    if sorted(result_groups) != sorted(gold_groups):
        return None
    result = json.dumps(result_groups, separators=(",", ":"))
    return f"Group words by sorted-letter key: {result}"


def trace_string_insertion(question, answer, md):
    candidates = re.findall(r"pattern:\s*([ABCDE]+)", question, flags=re.I)
    if not candidates:
        return None
    source = candidates[-1].upper()
    inserts = {"ABCD": "A", "BCDE": "B", "CDEA": "C", "DEAB": "D", "EABC": "E"}
    out, i = [], 0
    while i < len(source):
        block = source[i:i + 4]
        if block in inserts:
            out.extend(block)
            out.append(inserts[block])
            i += 4
        else:
            out.append(source[i])
            i += 1
    result = "".join(out)
    if str(answer).strip() != result:
        return None
    return f"Apply the insertion automaton left-to-right: {source} -> {result}"


TRACERS = {
    "chain_sum": trace_arithmetic,
    "decimal_chain_sum": trace_arithmetic,
    "basic_arithmetic": trace_arithmetic,
    "decimal_arithmetic": trace_arithmetic,
    "gcd": trace_gcd,
    "lcm": trace_lcm,
    "fraction_simplification": trace_fraction_simplification,
    "simple_equations": trace_simple_equations,
    "base_conversion": trace_base_conversion,
    "letter_counting": trace_letter_counting,
    "spell_backward": trace_spell_backward,
    "isomorphic_strings": trace_isomorphic_strings,
    "word_sorting": trace_word_sorting,
    "group_anagrams": trace_group_anagrams,
    "string_insertion": trace_string_insertion,
}


def make_document(question, trace, answer):
    return f"{question}\n<think>{trace}</think>\n<answer>{str(answer).strip()}</answer>"
