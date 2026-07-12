#!/usr/bin/env python3
"""Curate a bounded execution-verified TACO-derived Python SFT pilot.

The source exposes an English programming problem, candidate Python solutions,
and stdin/stdout tests. A row is useful only when a syntax-valid candidate
passes a bounded subset of supplied tests under the resource-limited runner
shared with the APPS and CodeContests curators. The output stays isolated from
every frozen SFT mix until its own quality report passes.
"""
import argparse
import ast
import json
import os
import re
from pathlib import Path

from curate_apps import build_test_grams, has_eval_overlap, run_solution


WORD = re.compile(r"\w+")


def normalized_question(question):
    """Match the punctuation-insensitive identity used by the SFT quality gate."""
    return " ".join(WORD.findall(str(question).lower()))


def parse_cases(raw, max_tests, max_case_chars):
    """Return bounded stdin/stdout cases from TACO's input_output JSON."""
    try:
        item = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if item.get("fn_name") or not isinstance(item.get("inputs"), list) or not isinstance(item.get("outputs"), list):
        return None
    pairs = []
    if max_tests < 0:
        raise ValueError("max_tests must be non-negative")
    for stdin, expected in zip(item["inputs"], item["outputs"]):
        stdin, expected = str(stdin), str(expected)
        if len(stdin) > max_case_chars or len(expected) > max_case_chars:
            continue
        pairs.append((stdin, expected))
        if max_tests and len(pairs) >= max_tests:
            break
    return pairs or None


def python_solutions(row, max_code_chars):
    """Yield syntax-valid Python candidate programs in source order."""
    for value in row.get("solutions") or []:
        code = str(value).strip()
        if not code or len(code) > max_code_chars:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        yield code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--dataset", default="likaixin/TACO-verified")
    parser.add_argument("--max-seen", type=int, default=500)
    parser.add_argument("--max-kept", type=int, default=250)
    parser.add_argument("--max-tests", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-code-chars", type=int, default=20_000)
    parser.add_argument("--max-case-chars", type=int, default=20_000)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000,
                        help="streaming shuffle buffer; 0 preserves source order")
    args = parser.parse_args()
    if args.max_seen <= 0 or args.max_kept <= 0 or args.max_tests <= 0:
        raise ValueError("max-seen, max-kept, and max-tests must be positive")
    if args.shuffle_buffer < 0:
        raise ValueError("shuffle-buffer must be non-negative")

    from datasets import load_dataset

    out_path = Path(args.out)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    if out_path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_grams = build_test_grams(args.evals, args.ngram)
    stream = load_dataset(args.dataset, split="train", streaming=True)
    # A prefix of a streaming competitive-programming dataset can be ordered by
    # source, difficulty, or upload time. Sample a deterministic reservoir-like
    # window instead, while the later full audit always replays selected IDs
    # against the canonical unshuffled source.
    if args.shuffle_buffer:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    seen = kept = 0
    drops = {key: 0 for key in (
        "eval_overlap", "missing_cases", "missing_python", "execution", "duplicate_question",
    )}
    kept_questions = set()
    with partial.open("w") as out:
        for row in stream:
            if seen >= args.max_seen or kept >= args.max_kept:
                break
            seen += 1
            question = str(row.get("question") or "").strip()
            if not question or has_eval_overlap(question, test_grams, args.ngram):
                drops["eval_overlap"] += 1
                continue
            cases = parse_cases(row.get("input_output"), args.max_tests, args.max_case_chars)
            if not cases:
                drops["missing_cases"] += 1
                continue
            candidates = list(python_solutions(row, args.max_code_chars))
            if not candidates:
                drops["missing_python"] += 1
                continue
            selected = next((code for code in candidates if run_solution(code, cases, args.timeout)), None)
            if selected is None:
                drops["execution"] += 1
                continue
            question_key = normalized_question(question)
            if question_key in kept_questions:
                drops["duplicate_question"] += 1
                continue
            kept_questions.add(question_key)
            out.write(json.dumps({
                "question": question,
                "response": selected,
                "source": "taco_verified_train",
                "problem_id": row.get("id"),
                "difficulty": row.get("difficulty"),
                "verified_cases": len(cases),
            }, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(partial, out_path)
    print(json.dumps({
        "dataset": args.dataset,
        "split": "train",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "seen": seen,
        "kept": kept,
        "dropped": drops,
        "out": str(out_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
