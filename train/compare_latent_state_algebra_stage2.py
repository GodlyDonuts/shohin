#!/usr/bin/env python3
"""Second-stage causal gate for latent-state-algebra training controls.

Stage one establishes that a verified packet can beat answer-only, zero-packet,
and shuffled-source controls.  This stage is intentionally separate so it
cannot change an already-running primary comparison.  It asks the stronger
question: does the advantage require the *correct* pair relation and state
code, rather than merely another auxiliary loss?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compare_latent_state_algebra import (
    MODES,
    metric,
    pair_metric,
    references,
    verify_matching,
    verify_pair_coverage,
    load_report,
)


MARGIN = 0.05


def normal_rows(report, predicate=lambda row: True):
    return [row for row in report["rows"] if row["mode"] == "normal" and predicate(row)]


def require_matched_reports(reports):
    """Bind every evaluator report to one held-out split and reference set."""
    names = sorted(reports)
    anchor = reports[names[0]]
    for name in names[1:]:
        verify_matching(anchor, reports[name])
        verify_pair_coverage(anchor, reports[name])
    for report in reports.values():
        for mode in MODES:
            if references(report, mode) != references(anchor, "normal"):
                raise ValueError("every mode must cover the complete common reference set")


def load_lsa_metadata(report):
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    memory = checkpoint.get("source_dropping_memory")
    algebra = checkpoint.get("latent_state_algebra")
    if not isinstance(memory, dict) or not isinstance(algebra, dict):
        raise ValueError("{} lacks latent-state-algebra metadata".format(report["checkpoint"]))
    return {"memory": memory, "algebra": algebra}


def require_training_controls(metadata):
    """Verify the two stage-two controls differ in exactly their causal label path."""
    candidate = metadata["candidate"]
    pair = metadata["pair_shuffled"]
    permuted = metadata["state_permuted"]
    control = metadata["answer_only"]

    expected_memory = ("init", "data", "data_sha256", "slots", "max_chunks", "seed", "updates", "batch_size")
    for field in expected_memory:
        values = {name: item["memory"].get(field) for name, item in metadata.items()}
        if len(set(values.values())) != 1:
            raise ValueError("training metadata differs for {}: {}".format(field, values))
    for name, item in metadata.items():
        if item["memory"].get("source_present_at_decode") is not False:
            raise ValueError("{} does not certify source-removed decoding".format(name))
    if candidate["algebra"].get("pair_mode") != "verified" or candidate["algebra"].get("state_mode") != "verified":
        raise ValueError("candidate must use verified pair and state supervision")
    if pair["algebra"].get("pair_mode") != "shuffled" or pair["algebra"].get("state_mode") != "verified":
        raise ValueError("pair control must shuffle only pair supervision")
    if permuted["algebra"].get("pair_mode") != "verified" or permuted["algebra"].get("state_mode") != "permuted":
        raise ValueError("state control must permute only state-code supervision")
    if any(float(value) != 0.0 for value in control["algebra"].get("weights", {}).values()):
        raise ValueError("answer-only control must have exact zero auxiliary weights")
    weights = candidate["algebra"].get("weights")
    if not isinstance(weights, dict) or any(float(value) <= 0.0 for value in weights.values()):
        raise ValueError("candidate must retain every nonzero algebra weight")
    if pair["algebra"].get("weights") != weights or permuted["algebra"].get("weights") != weights:
        raise ValueError("causal controls must retain the candidate's auxiliary weights")


def control_metric(reports, predicate):
    candidate = metric(normal_rows(reports["candidate"], predicate))
    controls = {
        "answer_only": metric(normal_rows(reports["answer_only"], predicate)),
        "candidate_zero_packet": metric([row for row in reports["candidate"]["rows"] if row["mode"] == "zero" and predicate(row)]),
        "candidate_shuffled_source": metric([row for row in reports["candidate"]["rows"] if row["mode"] == "shuffled" and predicate(row)]),
        "pair_shuffled": metric(normal_rows(reports["pair_shuffled"], predicate)),
        "state_permuted": metric(normal_rows(reports["state_permuted"], predicate)),
    }
    ceiling = max(item["accuracy"] for item in controls.values())
    return {
        "candidate_normal": candidate,
        "controls": controls,
        "control_max_accuracy": ceiling,
        "normal_margin": candidate["accuracy"] - ceiling,
    }


def pair_control_metric(reports, pair_kind):
    candidate, _ = pair_metric(reports["candidate"]["rows"], "normal", pair_kind)
    controls = {
        "answer_only": pair_metric(reports["answer_only"]["rows"], "normal", pair_kind)[0],
        "candidate_zero_packet": pair_metric(reports["candidate"]["rows"], "zero", pair_kind)[0],
        "candidate_shuffled_source": pair_metric(reports["candidate"]["rows"], "shuffled", pair_kind)[0],
        "pair_shuffled": pair_metric(reports["pair_shuffled"]["rows"], "normal", pair_kind)[0],
        "state_permuted": pair_metric(reports["state_permuted"]["rows"], "normal", pair_kind)[0],
    }
    ceiling = max(item["accuracy"] for item in controls.values())
    return {
        "candidate_normal": candidate,
        "controls": controls,
        "control_max_accuracy": ceiling,
        "normal_margin": candidate["accuracy"] - ceiling,
    }


def compare(reports, metadata):
    expected = {"answer_only", "candidate", "pair_shuffled", "state_permuted"}
    if set(reports) != expected or set(metadata) != expected:
        raise ValueError("stage-two comparator requires exactly {}".format(sorted(expected)))
    require_matched_reports(reports)
    require_training_controls(metadata)

    regimes = sorted({row["eval_regime"] for row in reports["candidate"]["rows"]})
    by_regime = {
        regime: control_metric(reports, lambda row, regime=regime: row["eval_regime"] == regime)
        for regime in regimes
    }
    ood = control_metric(reports, lambda row: row["eval_regime"] in {"length_ood", "language_ood"})
    pairs = {kind: pair_control_metric(reports, kind) for kind in ("equivalent", "intervention")}
    gates = {
        "fit_margin_at_least_5pp_over_all_controls": by_regime.get("fit_iid", {"normal_margin": float("-inf")})["normal_margin"] >= MARGIN,
        "length_language_margin_at_least_5pp_over_all_controls": ood["normal_margin"] >= MARGIN,
        "equivalent_pair_margin_at_least_5pp_over_all_controls": pairs["equivalent"]["normal_margin"] >= MARGIN,
        "intervention_pair_margin_at_least_5pp_over_all_controls": pairs["intervention"]["normal_margin"] >= MARGIN,
    }
    return {
        "audit": "latent_state_algebra_stage2_causal_comparison_v1",
        "margin_threshold": MARGIN,
        "reports": {name: report["checkpoint"] for name, report in reports.items()},
        "by_regime": by_regime,
        "length_language_combined": ood,
        "pair_results": pairs,
        "gates": gates,
        "advance_latent_state_algebra_beyond_controls": all(gates.values()),
        "claim_boundary": (
            "Passing supports a source-free packet effect that depends on verified pair and state-code "
            "supervision on this held-out algebra task only. It is not broad reasoning or flagship evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-only", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pair-shuffled", required=True)
    parser.add_argument("--state-permuted", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    reports = {
        "answer_only": load_report(args.answer_only),
        "candidate": load_report(args.candidate),
        "pair_shuffled": load_report(args.pair_shuffled),
        "state_permuted": load_report(args.state_permuted),
    }
    metadata = {name: load_lsa_metadata(report) for name, report in reports.items()}
    result = compare(reports, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[lsa-stage2] " + json.dumps({
        "advance": result["advance_latent_state_algebra_beyond_controls"], "gates": result["gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
