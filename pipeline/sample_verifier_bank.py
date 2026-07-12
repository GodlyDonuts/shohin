#!/usr/bin/env python3
"""Freeze a balanced, train-only answer-labeled bank for verifier rollouts."""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def valid_row(row):
    question = str(row.get("question") or row.get("problem") or "").strip()
    answer = str(row.get("answer") or "").strip()
    family = str(row.get("family") or "unknown").strip()
    return question, answer, family


def normalized_question(question):
    return " ".join(re.findall(r"\w+", question.lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.per_family <= 0:
        raise SystemExit("per-family must be positive")

    rng = random.Random(args.seed)
    reservoirs = defaultdict(list)
    seen = Counter()
    seen_questions = set()
    malformed = 0
    for line in open(args.input, errors="replace"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        question, answer, family = valid_row(row)
        if not question or not answer:
            malformed += 1
            continue
        question_key = normalized_question(question)
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        seen[family] += 1
        selected = reservoirs[family]
        clean = {"question": question, "answer": answer, "family": family, "source": "rg_v4_verifier_train"}
        if len(selected) < args.per_family:
            selected.append(clean)
        else:
            replacement = rng.randrange(seen[family])
            if replacement < args.per_family:
                selected[replacement] = clean

    rows = [row for family in sorted(reservoirs) for row in reservoirs[family]]
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".partial")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite verifier bank: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "out": str(output), "rows": len(rows), "families": len(reservoirs),
        "per_family": args.per_family, "malformed_or_missing": malformed,
        "selected_by_family": {family: len(rows) for family, rows in sorted(reservoirs.items())},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
