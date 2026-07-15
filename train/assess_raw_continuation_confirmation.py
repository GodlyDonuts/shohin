#!/usr/bin/env python3
"""Hash-bound assessor for immutable raw continuation confirmation transcripts."""

import argparse
import hashlib
import json
import re
from pathlib import Path


NEXT_SECTION = re.compile(
    r"\n\n(?=(?:Problem|Question)(?:\s+\d+)?\s*:|#{1,6}\s|Example(?:\s+\d+)?\s*:)",
    flags=re.IGNORECASE,
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_segment(response):
    return NEXT_SECTION.split(response, maxsplit=1)[0].strip()


def score(response, case):
    segment = answer_segment(response)
    values = [int(value) for value in re.findall(r"(?<![A-Za-z0-9_])-?\d+", segment)]
    answer = int(case["answer"])
    return {
        "answer_segment": segment,
        "integers": values,
        "leading_correct": bool(values) and values[0] == answer,
        "final_correct": bool(values) and values[-1] == answer,
        "contains_answer": answer in values,
        "intermediates_present": all(int(value) in values for value in case["required_intermediates"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text())
    modes = list(source["rows"][0]["modes"])
    rows = []
    for row in source["rows"]:
        assessed_modes = {
            mode: score(row["modes"][mode]["response"], row)
            for mode in modes
        }
        rows.append({
            "id": row["id"],
            "family": row["family"],
            "answer": row["answer"],
            "modes": assessed_modes,
        })

    summary = {
        mode: {
            metric: sum(row["modes"][mode][metric] for row in rows)
            for metric in ("leading_correct", "final_correct", "contains_answer", "intermediates_present")
        }
        for mode in modes
    }
    families = sorted({row["family"] for row in rows})
    by_family = {
        family: {
            mode: {
                metric: sum(
                    row["modes"][mode][metric]
                    for row in rows if row["family"] == family
                )
                for metric in ("final_correct", "contains_answer", "intermediates_present")
            }
            for mode in modes
        }
        for family in families
    }
    result = {
        "audit": "raw_continuation_confirmation_assessment_v2",
        "source": args.input,
        "source_sha256": sha256_file(args.input),
        "source_audit": source.get("audit"),
        "source_cases_sha256": source.get("cases_sha256"),
        "source_script_sha256": source.get("script_sha256"),
        "checkpoint_step": source.get("checkpoint_step"),
        "case_count": len(rows),
        "summary": summary,
        "by_family": by_family,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "by_family": by_family}, indent=2))


if __name__ == "__main__":
    main()
