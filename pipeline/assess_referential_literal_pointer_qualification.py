#!/usr/bin/env python3
"""Assess the frozen one-shot conventional compiler qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "base_sha256": "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6",
    "adapter_file_sha256": "747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991",
    "data_sha256": "06deeb39ac8c6ceb74003f6e503361401c58ead445558485e30a51d8c6d9358e",
    "report_sha256": "1467b089f964b5078f444f5d1c91228dcd3ee0a40792987b319f62dc7e98023d",
    "rows": 8192,
    "groups": 2048,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess(result):
    overall = result["overall"]
    groups = result["group_summary"]
    gates = {
        "answer_accuracy": {
            "value": float(overall["answer_accuracy"]),
            "floor": 0.99,
        },
        "semantic_program_exact": {
            "value": float(overall["semantic_program_exact"]),
            "floor": 0.99,
        },
        "full_pointer_exact": {
            "value": float(overall["full_pointer_exact"]),
            "floor": 0.99,
        },
        "kind_accuracy": {
            "value": float(overall["kind_accuracy"]),
            "floor": 0.999,
        },
        "initial_joint": {
            "value": float(overall["initial_joint"]),
            "floor": 0.99,
        },
        "all_four_exact": {
            "value": int(groups["all_four_full_pointer_exact"]),
            "floor": 2000,
        },
    }
    for gate in gates.values():
        gate["pass"] = gate["value"] >= gate["floor"]
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output_path = Path(args.out)
    if output_path.exists():
        raise SystemExit("refusing existing qualification assessment")
    result = json.loads(Path(args.result).read_text())
    report = json.loads(Path(args.report).read_text())
    if result.get("split") != "development_compositional":
        raise SystemExit("qualification split mismatch")
    if result.get("confirmation_access") != 0 or report.get("confirmation_access") != 0:
        raise SystemExit("qualification accessed sealed confirmation")
    if result.get("oracle", "none") != "none":
        raise SystemExit("qualification used an oracle")
    for field in ("base_sha256", "adapter_file_sha256", "data_sha256", "report_sha256"):
        if result.get(field) != EXPECTED[field]:
            raise SystemExit("qualification {} mismatch".format(field))
    if result["overall"].get("rows") != EXPECTED["rows"]:
        raise SystemExit("qualification row count mismatch")
    if result["group_summary"].get("groups") != EXPECTED["groups"]:
        raise SystemExit("qualification group count mismatch")
    if not report.get("all_gates_pass"):
        raise SystemExit("qualification structural report failed")
    if sha256_file(args.report) != EXPECTED["report_sha256"]:
        raise SystemExit("qualification report bytes mismatch")
    gates = assess(result)
    passed = all(gate["pass"] for gate in gates.values())
    assessment = {
        "schema": "r12_referential_literal_pointer_qualification_assessment_v1",
        "expected_identity": EXPECTED,
        "result_sha256": sha256_file(args.result),
        "gates": gates,
        "qualified": passed,
        "decision": (
            "qualify_conventional_compiler_for_isolated_stage_b_development"
            if passed else "reject_compiler_integration"
        ),
        "claim_boundary": (
            "One-shot known-atom compiler qualification only. No sealed factorized "
            "confirmation, executor, halt, autonomous rollout, reasoning, or novelty claim."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": assessment["decision"],
        "out": str(output_path.resolve()),
        "qualified": passed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
