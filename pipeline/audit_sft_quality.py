#!/usr/bin/env python3
"""Audit a frozen SFT JSONL before committing compute to fine-tuning.

This is intentionally format-agnostic: it measures the completion field that
``train/sft.py`` will actually consume and reports source balance, trace length,
and answer-format coverage. It does not certify semantic correctness; source
generators must provide that guarantee separately.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


WORD = re.compile(r"\w+")
QUESTION_FIELDS = ("question", "problem", "prompt", "instruction")
RESPONSE_FIELDS = ("response", "solution", "completion", "output", "answer")
EVAL_FILES = ("gsm8k.jsonl", "math500.jsonl", "gsm8k_platinum.jsonl",
              "humaneval_full.jsonl", "mbpp_full.jsonl")


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
        yield " ".join(words)
    else:
        for index in range(len(words) - n + 1):
            yield " ".join(words[index:index + n])


def load_eval_overlap(evals_dir, n):
    exact, ngrams = set(), set()
    for name in EVAL_FILES:
        path = Path(evals_dir) / name
        if not path.exists():
            continue
        with path.open(errors="replace") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = first_text(row, QUESTION_FIELDS)
                key = normalized_question(question)
                if key:
                    exact.add(key)
                    ngrams.update(grams(question, n))
    return exact, ngrams


def percentile(values, p):
    if not values:
        return 0
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--evals", default=None, help="optional eval directory for independent overlap audit")
    ap.add_argument("--ngram", type=int, default=13)
    args = ap.parse_args()

    source_counts = collections.Counter()
    training_group_counts = collections.Counter()
    lengths = []
    exact_questions = set()
    malformed = missing = duplicates = 0
    markers = collections.Counter()
    eval_exact, eval_grams = load_eval_overlap(args.evals, args.ngram) if args.evals else (set(), set())
    exact_eval_hits = ngram_eval_hits = 0

    with open(args.data, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            question = first_text(row, QUESTION_FIELDS)
            response = first_text(row, RESPONSE_FIELDS)
            if not question or not response:
                missing += 1
                continue
            normalized = normalized_question(question)
            key = hashlib.sha1(normalized.encode()).hexdigest()
            if key in exact_questions:
                duplicates += 1
            exact_questions.add(key)
            if normalized in eval_exact:
                exact_eval_hits += 1
            if eval_grams and any(gram in eval_grams for gram in grams(question, args.ngram)):
                ngram_eval_hits += 1
            source_counts[str(row.get("source") or "unknown")] += 1
            training_group_counts[str(row.get("training_group") or "unclassified")] += 1
            lengths.append(len(WORD.findall(response)))
            for marker in ("The answer is", "\\boxed", "<think>", "<code>", "```", "Therefore"):
                if marker in response:
                    markers[marker] += 1

    lengths.sort()
    report = {
        "data": str(Path(args.data)),
        "valid_rows": len(lengths),
        "malformed_rows": malformed,
        "missing_question_or_response": missing,
        "duplicate_normalized_questions": duplicates,
        "response_words": {
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths, default=0),
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "source_counts": dict(source_counts.most_common()),
        "training_group_counts": dict(training_group_counts.most_common()),
        "format_markers": dict(markers),
        "eval_overlap": {
            "enabled": bool(args.evals),
            "eval_exact_questions": len(eval_exact),
            "eval_ngrams": len(eval_grams),
            "exact_prompt_hits": exact_eval_hits,
            "ngram_prompt_hits": ngram_eval_hits,
        },
    }
    out = args.out or str(Path(args.data).with_suffix(".quality.json"))
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
