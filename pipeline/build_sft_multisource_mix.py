#!/usr/bin/env python3
"""Freeze a multi-source SFT candidate without rewriting its group labels.

``train/sft.py`` can consume multiple JSONL files, but a frozen union is easier
to audit for cross-source prompt duplication and to bind to a downstream SFT
job. This builder deliberately preserves each row's ``training_group`` and
``completion_prompt`` so a low-share auxiliary skill cannot be relabeled as
general math during sample-weighted training.
"""
import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path


WORD = re.compile(r"\w+")
QUESTION_FIELDS = ("question", "problem", "prompt", "instruction")
RESPONSE_FIELDS = ("response", "solution", "completion", "output", "answer")
EVAL_QUESTION_FIELDS = QUESTION_FIELDS + ("task", "text")


def first_text(row, fields):
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalized_question(text):
    return " ".join(WORD.findall(str(text).lower()))


def grams(text, n):
    words = WORD.findall(str(text).lower())
    if len(words) < n:
        if words:
            yield " ".join(words)
        return
    for index in range(len(words) - n + 1):
        yield " ".join(words[index:index + n])


def load_eval_prompts(patterns, n):
    """Return frozen public-eval prompt identity and n-gram sets.

    Multi-source mixes are often built after their component datasets.  The
    public benchmark boundary must therefore be re-applied to the union rather
    than inferred from older component reports.
    """
    exact, ngrams, paths = set(), set(), []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            names = sorted(pattern_path.parent.glob(pattern_path.name))
        else:
            names = sorted(Path().glob(pattern))
        for path in names:
            if not path.is_file():
                continue
            paths.append(str(path.resolve()))
            with path.open(errors="replace") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = first_text(row, EVAL_QUESTION_FIELDS)
                    key = normalized_question(question)
                    if key:
                        exact.add(key)
                        ngrams.update(grams(question, n))
    return exact, ngrams, paths


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--eval-glob",
        nargs="*",
        default=[],
        help="optional public-eval JSONL globs whose prompt identity/grams are hard-filtered",
    )
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()
    if args.ngram <= 0:
        raise SystemExit("--ngram must be positive")

    out = Path(args.out)
    report_path = Path(args.report)
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists() or report_path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite an existing multi-source candidate")
    for path in args.inputs:
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise SystemExit(f"required input missing or empty: {path}")

    eval_exact, eval_ngrams, eval_paths = load_eval_prompts(args.eval_glob, args.ngram)

    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    source_rows = collections.Counter()
    group_rows = collections.Counter()
    input_rows = collections.Counter()
    malformed = missing = duplicates = eval_exact_drops = eval_ngram_drops = 0
    kept = 0
    with partial.open("w") as target:
        for path in args.inputs:
            with open(path, errors="replace") as source:
                for line in source:
                    if not line.strip():
                        continue
                    input_rows[path] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    question = first_text(row, QUESTION_FIELDS)
                    response = first_text(row, RESPONSE_FIELDS)
                    group = str(row.get("training_group") or "").strip()
                    if not question or not response or not group:
                        missing += 1
                        continue
                    key = normalized_question(question)
                    if not key:
                        missing += 1
                        continue
                    if key in eval_exact:
                        eval_exact_drops += 1
                        continue
                    if any(gram in eval_ngrams for gram in grams(question, args.ngram)):
                        eval_ngram_drops += 1
                        continue
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    kept += 1
                    source_rows[str(row.get("source") or "unknown")] += 1
                    group_rows[group] += 1
    os.replace(partial, out)
    report = {
        "schema": "shohin-sft-multisource-mix-v1",
        "inputs": [str(Path(path).resolve()) for path in args.inputs],
        "input_sha256": {str(Path(path).resolve()): sha256(path) for path in args.inputs},
        "out": str(out.resolve()),
        "out_sha256": sha256(out),
        "input_rows": dict(input_rows),
        "valid_rows": kept,
        "malformed_json_rows": malformed,
        "missing_question_response_or_group": missing,
        "duplicate_normalized_questions_dropped": duplicates,
        "eval_filter": {
            "enabled": bool(args.eval_glob),
            "patterns": args.eval_glob,
            "paths": eval_paths,
            "ngram": args.ngram,
            "eval_exact_questions": len(eval_exact),
            "eval_ngrams": len(eval_ngrams),
            "exact_prompt_drops": eval_exact_drops,
            "ngram_prompt_drops": eval_ngram_drops,
        },
        "source_rows": dict(source_rows),
        "training_group_rows": dict(group_rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "valid_rows": kept,
        "duplicates": duplicates,
        "groups": dict(group_rows),
        "out": str(out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
