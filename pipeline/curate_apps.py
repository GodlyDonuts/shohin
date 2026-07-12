#!/usr/bin/env python3
"""Curate execution-verified APPS train examples for code-reasoning SFT.

APPS supplies English programming problems, Python reference solutions, and
input/output tests. This script uses only the train split, filters evaluation
overlap against the Shohin HumanEval/MBPP sets, and retains a candidate only if
it executes correctly on several supplied test cases under strict process
limits. The output format is compatible with ``train/sft.py`` but is produced
separately from the frozen SFT mix, so it can be quality-audited before use.
"""
import argparse
import ast
import json
import os
import re
import resource
import subprocess
import sys
import tempfile


WORD = re.compile(r"\w+")


def grams(text, n=13):
    words = WORD.findall(str(text).lower())
    if len(words) < n:
        yield " ".join(words)
    else:
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])


def build_test_grams(evals_dir, n=13):
    grams_set = set()
    for name in ("mbpp_full.jsonl", "humaneval_full.jsonl"):
        path = os.path.join(evals_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as src:
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (row.get("text") or row.get("prompt") or row.get("question") or "")
                grams_set.update(grams(prompt, n))
    return grams_set


def has_eval_overlap(question, gram_set, n=13):
    return any(gram in gram_set for gram in grams(question, n))


def normalize_output(text):
    return "\n".join(line.rstrip() for line in str(text).strip().splitlines()).strip()


def _limits():
    # A few macOS resource constants reject reduced hard limits in preexec_fn;
    # the Linux compute nodes accept them. Apply every independent cap we can
    # rather than making verification unavailable on either platform.
    for kind, soft, hard in (
        (resource.RLIMIT_CPU, 3, 4),
        (resource.RLIMIT_AS, 1_024 * 1024 * 1024, 1_024 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 4 * 1024 * 1024, 4 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64, 64),
        (resource.RLIMIT_NPROC, 16, 16),
    ):
        try:
            resource.setrlimit(kind, (soft, hard))
        except (OSError, ValueError):
            pass


def run_solution(code, cases, timeout):
    """Run a stdin/stdout program against supplied APPS cases with hard limits."""
    with tempfile.TemporaryDirectory(prefix="shohin_apps_") as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as dst:
            dst.write(code)
            dst.write("\n")
        for stdin, expected in cases:
            try:
                result = subprocess.run(
                    [sys.executable, "-I", path],
                    input=stdin,
                    capture_output=True,
                    cwd=tmp,
                    text=True,
                    timeout=timeout,
                    preexec_fn=_limits,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
                )
            except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
                return False
            if result.returncode != 0:
                return False
            if normalize_output(result.stdout) != normalize_output(expected):
                return False
    return True


def parse_cases(raw, max_tests, max_case_chars):
    try:
        io = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if io.get("fn_name") or not isinstance(io.get("inputs"), list) or not isinstance(io.get("outputs"), list):
        return None
    pairs = []
    for stdin, expected in zip(io["inputs"], io["outputs"]):
        stdin, expected = str(stdin), str(expected)
        if len(stdin) > max_case_chars or len(expected) > max_case_chars:
            continue
        pairs.append((stdin, expected))
        if len(pairs) >= max_tests:
            break
    return pairs or None


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
    ap.add_argument("--revision", default="main")
    ap.add_argument("--difficulties", nargs="*", default=["introductory", "interview"])
    args = ap.parse_args()

    from datasets import load_dataset

    wanted = {value.lower() for value in args.difficulties}
    test_grams = build_test_grams(args.evals, args.ngram)
    # Recent datasets releases intentionally reject Hub loading scripts. The
    # official APPS repository exposes the same public train JSONL directly,
    # which avoids executing remote code and lets us stream only the rows we
    # audit. The train-only URL is part of the provenance recorded below.
    train_url = f"hf://datasets/codeparrot/apps@{args.revision}/train.jsonl"
    dataset = load_dataset("json", data_files=train_url, split="train", streaming=True)
    seen = kept = 0
    drops = {key: 0 for key in ("difficulty", "eval_overlap", "missing_cases", "syntax", "solution", "execution")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for row in dataset:
            if seen >= args.max_seen or kept >= args.max_kept:
                break
            seen += 1
            difficulty = str(row.get("difficulty") or "").lower()
            if wanted and difficulty not in wanted:
                drops["difficulty"] += 1
                continue
            question = str(row.get("question") or "").strip()
            if not question or has_eval_overlap(question, test_grams, args.ngram):
                drops["eval_overlap"] += 1
                continue
            cases = parse_cases(row.get("input_output"), args.max_tests, args.max_case_chars)
            if not cases:
                drops["missing_cases"] += 1
                continue
            try:
                solutions = json.loads(row.get("solutions") or "[]")
            except (TypeError, json.JSONDecodeError):
                drops["solution"] += 1
                continue
            selected = None
            for solution in solutions[:12]:
                code = str(solution).strip()
                if not code or len(code) > args.max_code_chars:
                    continue
                try:
                    ast.parse(code)
                except SyntaxError:
                    drops["syntax"] += 1
                    continue
                if run_solution(code, cases, args.timeout):
                    selected = code
                    break
            if selected is None:
                drops["execution"] += 1
                continue
            out.write(json.dumps({
                "question": question,
                "response": selected,
                "source": "apps_train",
                "difficulty": difficulty,
                "problem_id": row.get("problem_id"),
                "verified_cases": len(cases),
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({
        "dataset": "codeparrot/apps",
        "revision": args.revision,
        "train_url": train_url,
        "seen": seen,
        "kept": kept,
        "dropped": drops,
        "out": args.out,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
