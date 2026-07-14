#!/usr/bin/env python3
"""Make a conservative decision record for behavior-preserving SFT evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected object JSON: {}".format(path))
    return value


def summary_for_manual(report: dict, checkpoint_name: str) -> dict:
    for model in report.get("models", []):
        if Path(str(model.get("checkpoint", ""))).name == checkpoint_name:
            return dict(model.get("summary") or {})
    raise ValueError("manual report lacks checkpoint {}".format(checkpoint_name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-primitives", required=True)
    parser.add_argument("--candidate-primitives", required=True)
    parser.add_argument("--raw-rg", required=True)
    parser.add_argument("--candidate-rg", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--deep", required=True)
    parser.add_argument("--raw-trace", required=True)
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--raw-checkpoint-name", default="best_step200000.pt")
    parser.add_argument("--candidate-checkpoint-name", default="sft_ep1.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in (
        "raw-primitives", "candidate-primitives", "raw-rg", "candidate-rg", "manual", "deep",
        "raw-trace", "candidate-trace",
    )}
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output")
    if not all(path.is_file() for path in paths.values()):
        raise SystemExit("all evidence paths must exist")
    raw_primitive = read_json(str(paths["raw-primitives"]))
    candidate_primitive = read_json(str(paths["candidate-primitives"]))
    raw_rg = read_json(str(paths["raw-rg"]))
    candidate_rg = read_json(str(paths["candidate-rg"]))
    manual = read_json(str(paths["manual"]))
    deep = read_json(str(paths["deep"]))
    raw_trace = read_json(str(paths["raw-trace"]))
    candidate_trace = read_json(str(paths["candidate-trace"]))
    raw_manual = summary_for_manual(manual, args.raw_checkpoint_name)
    candidate_manual = summary_for_manual(manual, args.candidate_checkpoint_name)
    direct_preserved = all(
        int(candidate_manual.get(field, 0)) >= int(raw_manual.get(field, 0))
        for field in ("initial", "verified_fact")
    )
    primitive_gain = float(candidate_primitive.get("accuracy", 0.0)) - float(raw_primitive.get("accuracy", 0.0))
    rg_gain = float(candidate_rg.get("accuracy", 0.0)) - float(raw_rg.get("accuracy", 0.0))
    families = candidate_primitive.get("by_contract", {}).get("answer", {}).get("families", {})
    arithmetic_and_base = {
        name: float(families.get(name, {}).get("accuracy", 0.0))
        for name in ("arithmetic", "base_conversion")
    }
    operation_floor = all(value >= 0.10 for value in arithmetic_and_base.values())
    deep_summary = dict(deep.get("model", {}).get("summary") or {})
    raw_visible_traces = int(raw_trace.get("summary", {}).get("correct_trace_and_final", 0))
    candidate_visible_traces = int(candidate_trace.get("summary", {}).get("correct_trace_and_final", 0))
    visible_trace_gain = candidate_visible_traces - raw_visible_traces
    visible_reasoning_present = candidate_visible_traces >= 1 and visible_trace_gain >= 1
    # A positive decision is deliberately modest: it permits further direct
    # investigations, never broad promotion or a reasoning claim.
    decision = "reject_retention_candidate"
    if direct_preserved and primitive_gain >= 0.10 and rg_gain >= 0.02 and operation_floor and visible_reasoning_present:
        decision = "bounded_behavior_preserving_visible_reasoning_signal"
    report = {
        "audit": "assess_retention_sft_v1",
        "decision": decision,
        "claim_boundary": (
            "A bounded signal is not general reasoning. Promotion requires independently read direct transcripts "
            "and fresh cross-family evidence beyond these fixed gates."
        ),
        "raw": {
            "primitive_accuracy": raw_primitive.get("accuracy"),
            "rg_accuracy": raw_rg.get("accuracy"),
            "manual": raw_manual,
        },
        "candidate": {
            "primitive_accuracy": candidate_primitive.get("accuracy"),
            "rg_accuracy": candidate_rg.get("accuracy"),
            "manual": candidate_manual,
            "deep": deep_summary,
            "arithmetic_and_base_accuracy": arithmetic_and_base,
            "visible_trace_and_final": candidate_visible_traces,
        },
        "gates": {
            "direct_decode_preserved": direct_preserved,
            "primitive_gain": primitive_gain,
            "rg_gain": rg_gain,
            "arithmetic_and_base_operation_floor": operation_floor,
            "raw_visible_trace_and_final": raw_visible_traces,
            "visible_trace_gain": visible_trace_gain,
            "visible_reasoning_present": visible_reasoning_present,
        },
        "evidence_sha256": {name: sha256_file(path) for name, path in sorted(paths.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
