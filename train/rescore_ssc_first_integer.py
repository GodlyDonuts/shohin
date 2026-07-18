#!/usr/bin/env python3
"""Offline first-integer / reaches-answer rescore of SSC confirmation whole decode.

Does not change the frozen last-integer evaluator. Reads an existing result JSON
with whole_problem_work.response fields and reports:
  - last-integer scored (frozen contract echo)
  - first-integer correct
  - answer appears anywhere as a standalone integer token
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


INT_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+")
HEADER_RE = re.compile(r"(?:^|\n)(?:Question|Problem)\s*(?:\d+\s*)?:", re.I)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_segment(response: str) -> str:
    match = HEADER_RE.search(response)
    return response[: match.start()] if match else response


def last_integer(text: str) -> int | None:
    matches = list(INT_RE.finditer(text))
    return int(matches[-1].group(0)) if matches else None


def first_integer(text: str) -> int | None:
    match = INT_RE.search(text)
    return int(match.group(0)) if match else None


def answer_appears(text: str, answer: int) -> bool:
    target = str(answer)
    for m in INT_RE.finditer(text):
        if m.group(0) == target or (target[0] != "-" and m.group(0).lstrip("0") == target.lstrip("0") and m.group(0) != "-"):
            # Exact digit-string match preferred
            if m.group(0) == target:
                return True
            if m.group(0).lstrip("+") == target:
                return True
    # Also accept exact token equality after int parse
    for m in INT_RE.finditer(text):
        try:
            if int(m.group(0)) == answer:
                return True
        except ValueError:
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    rows = data["rows"]
    by_family: dict[str, Counter] = {}
    totals: Counter = Counter()

    per_row = []
    for row in rows:
        fam = row["family"]
        ans = int(row["answer"])
        whole = row["whole_problem_work"]
        resp = whole["response"]
        seg = answer_segment(resp)
        last_i = last_integer(seg)
        first_i = first_integer(seg)
        appears = answer_appears(seg, ans)
        last_ok = last_i == ans
        first_ok = first_i == ans
        frozen_ok = bool(whole.get("correct"))
        flags = {
            "family": fam,
            "id": row["id"],
            "answer": ans,
            "frozen_last_integer_correct": frozen_ok,
            "rescored_last_integer_correct": last_ok,
            "first_integer_correct": first_ok,
            "answer_appears_in_segment": appears,
            "parser_loss": appears and not last_ok,
            "first_int": first_i,
            "last_int": last_i,
        }
        per_row.append(flags)
        c = by_family.setdefault(fam, Counter())
        for k in (
            "frozen_last_integer_correct",
            "rescored_last_integer_correct",
            "first_integer_correct",
            "answer_appears_in_segment",
            "parser_loss",
        ):
            c[k] += int(flags[k])
            totals[k] += int(flags[k])
        c["n"] += 1
        totals["n"] += 1

    summary = {
        "protocol": "R12-SSC-FIRST-INTEGER-OFFLINE",
        "source_result": str(args.result),
        "source_sha256": sha256_file(args.result),
        "n": totals["n"],
        "totals": dict(totals),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "rates": {
            "frozen_last": totals["frozen_last_integer_correct"] / totals["n"],
            "first_integer": totals["first_integer_correct"] / totals["n"],
            "answer_appears": totals["answer_appears_in_segment"] / totals["n"],
            "parser_loss": totals["parser_loss"] / totals["n"],
        },
    }
    out = {"summary": summary, "rows": per_row}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["decision_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.write_text(json.dumps({"summary": summary, "rows": per_row}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
