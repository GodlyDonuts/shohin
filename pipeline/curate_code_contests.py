#!/usr/bin/env python3
"""Curate execution-verified Python 3 CodeContests train examples for SFT.

The source is a public test-backed programming-contest corpus. Only its train
split is streamed. Each retained Python 3 reference solution must pass a small,
bounded set of public/generated I/O cases under the same resource limits used by
the APPS curator. Output is atomic and decontaminated against Shohin's held-out
HumanEval and MBPP prompts.
"""
import argparse
import ast
import json
import os
from pathlib import Path

from curate_apps import build_test_grams, has_eval_overlap, run_solution


PYTHON3 = 3


def collect_cases(row, max_tests, max_case_chars):
    pairs = []
    for key in ("public_tests", "generated_tests", "private_tests"):
        tests = row.get(key) or {}
        inputs, outputs = tests.get("input") or [], tests.get("output") or []
        for stdin, expected in zip(inputs, outputs):
            stdin, expected = str(stdin), str(expected)
            if len(stdin) > max_case_chars or len(expected) > max_case_chars:
                continue
            pairs.append((stdin, expected))
            if len(pairs) >= max_tests:
                return pairs
    return pairs or None


def python3_solutions(row, max_code_chars):
    solutions = row.get("solutions") or {}
    languages = solutions.get("language") or []
    code_items = solutions.get("solution") or []
    for language, code in zip(languages, code_items):
        if int(language) != PYTHON3:
            continue
        code = str(code).strip()
        if not code or len(code) > max_code_chars:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        yield code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--max-seen", type=int, default=500)
    ap.add_argument("--max-kept", type=int, default=250)
    ap.add_argument("--max-tests", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--max-code-chars", type=int, default=20_000)
    ap.add_argument("--max-case-chars", type=int, default=20_000)
    ap.add_argument("--ngram", type=int, default=13)
    args = ap.parse_args()

    from datasets import load_dataset

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp_path.exists():
        raise SystemExit(f"refusing stale partial output: {tmp_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_grams = build_test_grams(args.evals, args.ngram)
    dataset = load_dataset("Imandra/code_contests", split="train", streaming=True)
    seen = kept = 0
    drops = {key: 0 for key in ("eval_overlap", "missing_cases", "missing_python3", "execution")}
    with open(tmp_path, "w") as out:
        for row in dataset:
            if seen >= args.max_seen or kept >= args.max_kept:
                break
            seen += 1
            question = str(row.get("description") or "").strip()
            if not question or has_eval_overlap(question, test_grams, args.ngram):
                drops["eval_overlap"] += 1
                continue
            cases = collect_cases(row, args.max_tests, args.max_case_chars)
            if not cases:
                drops["missing_cases"] += 1
                continue
            selected = None
            for code in python3_solutions(row, args.max_code_chars):
                if run_solution(code, cases, args.timeout):
                    selected = code
                    break
            if selected is None:
                drops["missing_python3"] += 1
                continue
            out.write(json.dumps({
                "question": question,
                "response": selected,
                "source": "code_contests_train",
                "name": row.get("name"),
                "difficulty": row.get("difficulty"),
                "verified_cases": len(cases),
            }, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp_path, out_path)
    print(json.dumps({
        "dataset": "Imandra/code_contests",
        "split": "train",
        "seen": seen,
        "kept": kept,
        "dropped": drops,
        "out": str(out_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
