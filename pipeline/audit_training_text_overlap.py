#!/usr/bin/env python3
"""Audit every text field consumed by SFT against every live eval prompt.

Prompt-only filtering is insufficient when a teacher trace can quote an eval
question in its completion. This scanner stores no training text in its report:
it records only line/source/field identifiers for bounded forensic review.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


WORD = re.compile(r"\w+")
EVAL_FIELDS = ("question", "problem", "prompt", "instruction", "task", "text")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def grams(text, n):
    words = WORD.findall(str(text).lower())
    if not words:
        return
    if len(words) < n:
        yield " ".join(words)
        return
    for index in range(len(words) - n + 1):
        yield " ".join(words[index:index + n])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_index(evals_dir, n):
    exact, ngrams, files = set(), set(), []
    for path in sorted(Path(evals_dir).glob("*.jsonl")):
        files.append(str(path))
        with path.open(errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = next((row[field] for field in EVAL_FIELDS if row.get(field)), "")
                clean = normalized(prompt)
                if clean:
                    exact.add(clean)
                    ngrams.update(grams(clean, n))
    return exact, ngrams, files


def first_overlap(text, exact, ngrams, n):
    clean = normalized(text)
    if not clean:
        return None
    if clean in exact:
        return "exact"
    if any(gram in ngrams for gram in grams(clean, n)):
        return "ngram"
    return None


def audit_rows(rows, fields, exact, ngrams, n, max_examples=20):
    valid_rows = malformed = 0
    missing_fields = Counter()
    exact_rows = ngram_rows = 0
    exact_by_field = Counter()
    ngram_by_field = Counter()
    examples = []
    for lineno, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            malformed += 1
            continue
        valid_rows += 1
        row_exact = row_ngram = False
        for field in fields:
            value = row.get(field)
            if value is None or not str(value).strip():
                missing_fields[field] += 1
                continue
            hit = first_overlap(value, exact, ngrams, n)
            if hit == "exact":
                row_exact = True
                exact_by_field[field] += 1
            elif hit == "ngram":
                row_ngram = True
                ngram_by_field[field] += 1
            if hit and len(examples) < max_examples:
                examples.append({
                    "line": lineno,
                    "field": field,
                    "kind": hit,
                    "source": str(row.get("source") or "unknown"),
                    "training_group": str(row.get("training_group") or "unclassified"),
                })
        exact_rows += int(row_exact)
        ngram_rows += int(row_ngram)
    return {
        "valid_rows": valid_rows,
        "malformed_rows": malformed,
        "missing_fields": dict(sorted(missing_fields.items())),
        "overlap": {
            "exact_rows": exact_rows,
            "ngram_rows": ngram_rows,
            "exact_hits_by_field": dict(sorted(exact_by_field.items())),
            "ngram_hits_by_field": dict(sorted(ngram_by_field.items())),
            "examples": examples,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fields", nargs="+", default=("question", "response", "completion_prompt"))
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--require-zero", action="store_true")
    args = parser.parse_args()
    if args.ngram <= 0:
        raise ValueError("--ngram must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {out}")

    exact, ngrams, files = load_eval_index(args.evals, args.ngram)
    malformed_json = [0]

    def rows():
        with open(args.data, errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    malformed_json[0] += 1

    summary = audit_rows(rows(), tuple(args.fields), exact, ngrams, args.ngram)
    summary["malformed_json_rows"] = malformed_json[0]
    report = {
        "schema": "shohin-training-text-overlap-v1",
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256(args.data),
        "fields": list(args.fields),
        "ngram": args.ngram,
        "eval_files": files,
        "eval_exact_prompts": len(exact),
        "eval_ngrams": len(ngrams),
        **summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    partial.replace(out)
    overlap = report["overlap"]
    print(json.dumps({
        "data": report["data"],
        "valid_rows": report["valid_rows"],
        "exact_rows": overlap["exact_rows"],
        "ngram_rows": overlap["ngram_rows"],
        "out": str(out),
    }, sort_keys=True))
    if args.require_zero and (overlap["exact_rows"] or overlap["ngram_rows"]):
        raise SystemExit("training-text eval overlap gate failed")


if __name__ == "__main__":
    main()
