#!/usr/bin/env python3
"""Re-run every supplied TACO test for a bounded curated code artifact.

The first curation pass uses a few tests to cheaply identify plausible programs.
This audit is the admission gate: it finds the source record for every retained
program and executes all bounded supplied stdin/stdout cases before emitting a
new immutable derivative. It never mutates its input.
"""
import argparse
import json
import os
from pathlib import Path

from curate_apps import run_solution
from curate_taco_verified import parse_cases


def read_candidates(path):
    by_id = {}
    malformed = missing = duplicate_ids = 0
    with open(path, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            problem_id = row.get("problem_id")
            response = str(row.get("response") or "").strip()
            if problem_id is None or not response:
                missing += 1
                continue
            key = str(problem_id)
            if key in by_id:
                duplicate_ids += 1
                continue
            by_id[key] = row
    if malformed or missing or duplicate_ids:
        raise SystemExit(
            f"invalid input: malformed={malformed} missing={missing} duplicate_ids={duplicate_ids}"
        )
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="likaixin/TACO-verified")
    parser.add_argument("--max-case-chars", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-source-rows", type=int, default=0,
                        help="0 scans until every input problem is found")
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_case_chars <= 0 or args.max_source_rows < 0:
        raise ValueError("timeout and max-case-chars must be positive; max-source-rows non-negative")

    from datasets import load_dataset

    candidates = read_candidates(args.input)
    out_path = Path(args.out)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    if out_path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    found = kept = source_rows = 0
    drops = {key: 0 for key in ("missing_cases", "execution", "source_unmatched")}
    stream = load_dataset(args.dataset, split="train", streaming=True)
    with partial.open("w") as out:
        for source in stream:
            source_rows += 1
            if args.max_source_rows and source_rows > args.max_source_rows:
                break
            key = str(source.get("id"))
            row = candidates.pop(key, None)
            if row is None:
                continue
            found += 1
            cases = parse_cases(source.get("input_output"), max_tests=0,
                                max_case_chars=args.max_case_chars)
            if not cases:
                drops["missing_cases"] += 1
                continue
            if not run_solution(str(row["response"]), cases, args.timeout):
                drops["execution"] += 1
                continue
            clean = dict(row)
            clean["full_verified_cases"] = len(cases)
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            kept += 1
            if not candidates:
                break
    drops["source_unmatched"] = len(candidates)
    if candidates:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"[taco-full-audit] source records missing for {len(candidates)} input rows")
    os.replace(partial, out_path)
    print(json.dumps({
        "dataset": args.dataset,
        "input_rows": found + len(candidates),
        "source_rows_scanned": source_rows,
        "matched_source_rows": found,
        "kept": kept,
        "dropped": drops,
        "out": str(out_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
