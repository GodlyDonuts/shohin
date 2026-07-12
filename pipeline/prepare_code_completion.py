#!/usr/bin/env python3
"""Convert verified code SFT rows into raw-completion-form examples.

The canonical code sources are execution-verified before this stage. This converter
does not claim a new semantic proof; it preserves the original code while teaching
the model a prompt contract closer to HumanEval-style continuation. Function rows
use a task comment plus their source through the function header, then train on the
indented body. Program rows retain the full program as the completion after a task
comment. Output is atomic and remains a separate frozen-candidate input.
"""
import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path


WORD = re.compile(r"\w+")


def norm_question(question):
    return " ".join(WORD.findall(str(question).lower()))


def question_hash(question):
    return hashlib.sha1(norm_question(question).encode("utf-8", "ignore")).hexdigest()[:16]


def get_question(row):
    for key in ("question", "problem", "prompt", "instruction", "text", "description"):
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def get_code(row):
    for key in ("response", "solution", "completion", "output", "code"):
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def split_function(question, code):
    """Return (completion_prompt, completion) while preserving a function body indent."""
    tree = ast.parse(code)
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if function is None or not function.body:
        return None
    lines = code.splitlines(keepends=True)
    body_line = function.body[0].lineno
    prefix = "".join(lines[:body_line - 1])
    completion = "".join(lines[body_line - 1:]).rstrip()
    if not prefix or not completion:
        return None
    reconstructed = prefix + completion
    ast.parse(reconstructed)
    return f"# Task: {question}\n{prefix}", completion


def completion_form(question, code):
    try:
        function = split_function(question, code)
    except SyntaxError:
        return None
    if function is not None:
        return function
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return f"# Task: {question}\n", code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = malformed = duplicate = 0
    hashes = set()
    sources = {}
    with open(tmp, "w") as dst:
        for input_path in args.inputs:
            path = Path(input_path)
            if not path.is_file():
                raise SystemExit(f"input missing: {path}")
            with open(path, errors="replace") as src:
                for line in src:
                    if not line.strip():
                        continue
                    seen += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    question, code = get_question(row), get_code(row)
                    if not question or not code:
                        malformed += 1
                        continue
                    item = completion_form(question, code)
                    if item is None:
                        malformed += 1
                        continue
                    key = question_hash(question)
                    if key in hashes:
                        duplicate += 1
                        continue
                    hashes.add(key)
                    prompt, completion = item
                    source = str(row.get("source") or path.stem)
                    completion_source = f"{source}_completion"
                    dst.write(json.dumps({
                        "question": question,
                        "completion_prompt": prompt,
                        "response": completion,
                        "source": completion_source,
                    }, ensure_ascii=False) + "\n")
                    sources[completion_source] = sources.get(completion_source, 0) + 1
                    kept += 1
    os.replace(tmp, out)
    print(json.dumps({
        "inputs": args.inputs,
        "out": str(out),
        "seen": seen,
        "kept": kept,
        "malformed_or_unusable": malformed,
        "duplicate_questions": duplicate,
        "source_counts": sources,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
