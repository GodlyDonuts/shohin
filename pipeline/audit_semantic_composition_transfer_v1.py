#!/usr/bin/env python3
"""Audit the evaluation-only semantic composition transfer suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from generate_semantic_composition_transfer_v1 import FAMILIES, normalized_question


ANSWER = re.compile(r"The answer is (-?\d+)\.")
WORD = re.compile(r"\w+")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_ngrams(text, width=13):
    words = WORD.findall(text.lower())
    return {" ".join(words[index:index + width]) for index in range(len(words) - width + 1)}


def rows(path):
    result = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("question", "response", "answer", "family", "source")
            if any(not str(row.get(key, "")).strip() for key in required):
                raise ValueError("missing required field at line {}".format(line_number))
            result.append(row)
    if not result:
        raise ValueError("suite is empty")
    return result


def audit(suite, train):
    suite_rows, train_rows = rows(suite), rows(train)
    normalized = [normalized_question(row["question"]) for row in suite_rows]
    duplicate = len(normalized) - len(set(normalized))
    train_questions = {normalized_question(row["question"]) for row in train_rows}
    exact_hits = sum(question in train_questions for question in normalized)
    train_grams = set()
    for row in train_rows:
        train_grams.update(word_ngrams(row["question"]))
    ngram_hits = sum(bool(word_ngrams(row["question"]) & train_grams) for row in suite_rows)
    malformed, incorrect_answer = 0, 0
    families = Counter()
    for row in suite_rows:
        families[row["family"]] += 1
        match = ANSWER.search(row["response"])
        if not row["response"].startswith("<think>") or match is None:
            malformed += 1
            continue
        incorrect_answer += int(match.group(1) != str(row["answer"]))
    return {
        "schema": "semantic_composition_transfer_v1_audit",
        "suite": str(suite),
        "suite_sha256": sha256_file(suite),
        "training_reference": str(train),
        "training_reference_sha256": sha256_file(train),
        "rows": len(suite_rows),
        "families": dict(sorted(families.items())),
        "malformed_rows": malformed,
        "response_answer_mismatches": incorrect_answer,
        "duplicate_normalized_questions": duplicate,
        "exact_question_hits_against_training": exact_hits,
        "question_13gram_hits_against_training": ngram_hits,
        "admitted": (
            set(families) == set(FAMILIES)
            and malformed == 0
            and incorrect_answer == 0
            and duplicate == 0
            and exact_hits == 0
            and ngram_hits == 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--training-reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite audit: {}".format(out))
    result = audit(args.suite, args.training_reference)
    if not result["admitted"]:
        raise SystemExit("semantic composition transfer audit failed: {}".format(json.dumps(result, sort_keys=True)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
