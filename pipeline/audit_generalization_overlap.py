#!/usr/bin/env python3
"""Prove a frozen SFT file is lexically disjoint from a direct interview.

Public benchmark decontamination does not prove that a custom capability
interview is held out. This audit reads the literal ``CASES`` declaration from
the evaluator's source through ``ast.literal_eval`` rather than importing the
model/evaluator stack, then checks normalized prompt identity and word n-grams
against a candidate JSONL.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


WORD = re.compile(r"\w+")
QUESTION_FIELDS = ("question", "problem", "prompt", "instruction")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def grams(text, n):
    words = WORD.findall(str(text).lower())
    if len(words) < n:
        yield " ".join(words)
    else:
        for index in range(len(words) - n + 1):
            yield " ".join(words[index:index + n])


def first_question(row):
    for field in QUESTION_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_cases(path):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets):
            cases = ast.literal_eval(node.value)
            if not isinstance(cases, list) or not cases:
                raise ValueError("CASES must be a non-empty literal list")
            for case in cases:
                if not isinstance(case, dict) or not case.get("id") or not case.get("question"):
                    raise ValueError("every CASES row must have id and question")
            return cases
    raise ValueError(f"literal CASES assignment not found in {path}")


def load_cases_json(path):
    value = json.loads(Path(path).read_text())
    cases = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON must contain a non-empty list")
    for case in cases:
        if not isinstance(case, dict) or not case.get("id") or not case.get("question"):
            raise ValueError("every JSON case needs id and question")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--case-source", default="")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument(
        "--case-regimes", nargs="*", default=[],
        help="optional case['regime'] values to audit; useful for factorized held-out suites",
    )
    args = parser.parse_args()
    if args.ngram <= 0:
        raise ValueError("ngram must be positive")
    if bool(args.case_source) == bool(args.cases_json):
        raise ValueError("supply exactly one of --case-source or --cases-json")

    if args.cases_json:
        cases = load_cases_json(args.cases_json)
        case_source = args.cases_json
    else:
        cases = load_cases(args.case_source)
        case_source = args.case_source
    regimes = sorted(set(args.case_regimes))
    if regimes:
        cases = [case for case in cases if str(case.get("regime", "")) in regimes]
        if not cases:
            raise ValueError("no cases matched --case-regimes")
    exact = {normalized(case["question"]): str(case["id"]) for case in cases}
    case_grams = {}
    for case in cases:
        for gram in grams(case["question"], args.ngram):
            case_grams.setdefault(gram, str(case["id"]))

    rows = malformed = missing = exact_hits = ngram_hits = 0
    examples = []
    with open(args.data, errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            question = first_question(row)
            if not question:
                missing += 1
                continue
            rows += 1
            hit_ids = set()
            if (case_id := exact.get(normalized(question))) is not None:
                exact_hits += 1
                hit_ids.add(case_id)
            for gram in grams(question, args.ngram):
                if (case_id := case_grams.get(gram)) is not None:
                    hit_ids.add(case_id)
            if hit_ids:
                ngram_hits += 1
                if len(examples) < 20:
                    examples.append({"line": line_number, "case_ids": sorted(hit_ids)})

    report = {
        "audit": "generalization_interview_overlap_v1",
        "data": str(Path(args.data)),
        "data_sha256": sha256(args.data),
        "case_source": str(Path(case_source)),
        "case_source_sha256": sha256(case_source),
        "cases": len(cases),
        "case_regimes": regimes,
        "ngram": args.ngram,
        "valid_rows": rows,
        "malformed_rows": malformed,
        "missing_question": missing,
        "exact_prompt_hits": exact_hits,
        "ngram_prompt_hits": ngram_hits,
        "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
