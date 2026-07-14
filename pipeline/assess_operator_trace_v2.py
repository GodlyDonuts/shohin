#!/usr/bin/env python3
"""Assess whether a factorized operator-trace SFT has bounded transfer.

This is a CPU-only decision record.  It deliberately cannot submit training or
modify any checkpoint.  The candidate must improve separately on wording,
value, and combined transfer while preserving direct behavior and clearing
arithmetic/base-operation floors.  Passing is a bounded skill result, never a
flagship-promotion condition or evidence of general intelligence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REGIME_GAIN_MIN = {"wording": 15, "value": 15, "full": 10}
PER_FAMILY_JOINT_MIN = 5
MAX_RG_REGRESSION = 0.02
PAIR_RESPONSE_MARKERS = (
    re.compile(r"\bproblem\s+a\s*:", re.IGNORECASE),
    re.compile(r"\bproblem\s+b\s*:", re.IGNORECASE),
    re.compile(r"\bthe\s+answers\s+are\s+a\s*=", re.IGNORECASE),
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def factor_summary(report: dict) -> dict:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 900:
        raise ValueError("factor report must contain exactly 900 retained rows")
    regimes: dict[str, Counter] = defaultdict(Counter)
    families: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        regime = row.get("regime")
        family = row.get("family")
        if regime not in REGIME_GAIN_MIN or not isinstance(family, str):
            raise ValueError("factor report row lacks a recognized regime/family")
        regimes[regime]["cases"] += 1
        regimes[regime]["joint"] += int(bool(row.get("correct_trace_and_final")))
        key = f"{regime}:{family}"
        families[key]["cases"] += 1
        families[key]["joint"] += int(bool(row.get("correct_trace_and_final")))
    if set(regimes) != set(REGIME_GAIN_MIN) or any(stats["cases"] != 300 for stats in regimes.values()):
        raise ValueError("factor report must have 300 rows in each regime")
    if len(families) != 9 or any(stats["cases"] != 100 for stats in families.values()):
        raise ValueError("factor report must have nine 100-row regime-family cells")
    return {
        "regimes": {name: dict(values) for name, values in sorted(regimes.items())},
        "regime_families": {name: dict(values) for name, values in sorted(families.items())},
    }


def primitive_accuracy(report: dict, family: str) -> float:
    payload = report.get("summary", report)
    direct = payload.get("by_family", {}).get(family, {})
    if direct.get("accuracy") is not None:
        return float(direct["accuracy"])
    # ``eval_contract_primitives.py`` emits contract-nested families.  The
    # currently admitted primitive gate uses exactly one answer contract, but
    # reject an ambiguous report rather than silently selecting a contract.
    matches = [
        stats for contract in payload.get("by_contract", {}).values()
        if isinstance(contract, dict)
        for stats in [contract.get("families", {}).get(family, {})]
        if stats.get("accuracy") is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"primitive report lacks one unambiguous {family} accuracy")
    return float(matches[0]["accuracy"])


def manual_summary(report: dict) -> dict:
    models = report.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("manual report must compare exactly two checkpoints")
    result = {}
    for model in models:
        summary = dict(model["summary"])
        # Old transcript reports predate the explicit response-mode field, so
        # derive the same safety signal from their immutable raw responses.
        observed = {condition: 0 for condition in (
            "initial", "review", "verified_fact", "compact_state", "state_reuse"
        )}
        for row in model.get("rows", []):
            for condition in observed:
                response = str(row.get(condition, {}).get("response") or "")
                observed[condition] += int(any(pattern.search(response) for pattern in PAIR_RESPONSE_MARKERS))
        summary.setdefault("paired_answer_mode", observed)
        result[model["checkpoint"]] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-factor", required=True)
    parser.add_argument("--candidate-factor", required=True)
    parser.add_argument("--baseline-primitives", required=True)
    parser.add_argument("--candidate-primitives", required=True)
    parser.add_argument("--baseline-rg", required=True)
    parser.add_argument("--candidate-rg", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--baseline-fixed-trace", required=True)
    parser.add_argument("--candidate-fixed-trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {key: value for key, value in vars(args).items() if key != "out"}
    if any(not Path(path).is_file() for path in paths.values()):
        missing = [key for key, path in paths.items() if not Path(path).is_file()]
        raise SystemExit("missing evidence: " + ", ".join(missing))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite an assessment")

    baseline_factor = factor_summary(load(args.baseline_factor))
    candidate_factor = factor_summary(load(args.candidate_factor))
    baseline_primitive = load(args.baseline_primitives)
    candidate_primitive = load(args.candidate_primitives)
    baseline_rg = load(args.baseline_rg)
    candidate_rg = load(args.candidate_rg)
    manual = manual_summary(load(args.manual))
    baseline_trace = load(args.baseline_fixed_trace)
    candidate_trace = load(args.candidate_fixed_trace)

    baseline_key, candidate_key = list(manual)
    baseline_manual, candidate_manual = manual[baseline_key], manual[candidate_key]
    gains = {
        regime: candidate_factor["regimes"][regime]["joint"] - baseline_factor["regimes"][regime]["joint"]
        for regime in REGIME_GAIN_MIN
    }
    factor_gains_pass = all(gains[name] >= required for name, required in REGIME_GAIN_MIN.items())
    factor_family_pass = all(
        candidate_factor["regime_families"][name]["joint"] >= PER_FAMILY_JOINT_MIN
        for name in candidate_factor["regime_families"]
    )
    arithmetic_base = {
        family: primitive_accuracy(candidate_primitive, family)
        for family in ("arithmetic", "base_conversion")
    }
    operation_floor = all(value >= 0.10 for value in arithmetic_base.values())
    rg_nonregression = float(candidate_rg["accuracy"]) >= float(baseline_rg["accuracy"]) - MAX_RG_REGRESSION
    direct_preserved = (
        int(candidate_manual.get("initial", -1)) >= int(baseline_manual.get("initial", -1))
        and int(candidate_manual.get("verified_fact", -1)) >= int(baseline_manual.get("verified_fact", -1))
    )
    paired_response_free = not any(candidate_manual.get("paired_answer_mode", {}).values())
    trace_nonregression = int(candidate_trace["summary"].get("correct_trace_and_final", 0)) >= int(
        baseline_trace["summary"].get("correct_trace_and_final", 0)
    )
    accepted = all((
        factor_gains_pass, factor_family_pass, operation_floor, rg_nonregression, direct_preserved,
        paired_response_free, trace_nonregression,
    ))
    result = {
        "audit": "assess_operator_trace_v2",
        "decision": "bounded_operator_binding_transfer" if accepted else "reject_operator_trace_candidate",
        "claim_boundary": (
            "Acceptance demonstrates only bounded transfer on a frozen factorized operator/state suite. "
            "It does not authorize flagship modification or establish general reasoning."
        ),
        "factor": {"baseline": baseline_factor, "candidate": candidate_factor, "joint_gains": gains},
        "candidate": {
            "arithmetic_and_base_accuracy": arithmetic_base,
            "rg_accuracy": float(candidate_rg["accuracy"]),
            "manual": candidate_manual,
            "fixed_trace_joint": int(candidate_trace["summary"].get("correct_trace_and_final", 0)),
        },
        "baseline": {
            "rg_accuracy": float(baseline_rg["accuracy"]),
            "manual": baseline_manual,
            "fixed_trace_joint": int(baseline_trace["summary"].get("correct_trace_and_final", 0)),
        },
        "gates": {
            "factor_regime_gains": factor_gains_pass,
            "factor_regime_gain_minimums": REGIME_GAIN_MIN,
            "factor_all_nine_cells_at_least_five_joint": factor_family_pass,
            "arithmetic_and_base_operation_floor": operation_floor,
            "rg_nonregression": rg_nonregression,
            "direct_decode_preserved": direct_preserved,
            "paired_response_mode_absent": paired_response_free,
            "fixed_trace_nonregression": trace_nonregression,
        },
        "evidence_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
