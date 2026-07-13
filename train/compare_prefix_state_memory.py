#!/usr/bin/env python3
"""Locked causal comparison for prefix-supervised source-free memory.

The candidate must beat answer-only, zero/shuffled packet ablations, and a
training control whose solver-recomputed prefix labels were shuffled across
examples.  This prevents a lower answer loss or a state probe fit from being
mistaken for retained-state evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compare_latent_state_algebra import MODES, metric, pair_metric, references, verify_matching, verify_pair_coverage, load_report


FIT_MARGIN = 0.10
OOD_MARGIN = 0.05
PAIR_MARGIN = 0.10


def normal_rows(report, predicate=lambda row: True):
    return [row for row in report["rows"] if row["mode"] == "normal" and predicate(row)]


def metadata(report):
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    memory = checkpoint.get("source_dropping_memory")
    prefix = checkpoint.get("prefix_state_supervision")
    if not isinstance(memory, dict) or not isinstance(prefix, dict):
        raise ValueError("{} lacks prefix-state metadata".format(report["checkpoint"]))
    return {"memory": memory, "prefix": prefix}


def verify_metadata(metadata_by_name):
    expected = {"answer_only", "candidate", "prefix_shuffled"}
    if set(metadata_by_name) != expected:
        raise ValueError("prefix comparator requires {}".format(sorted(expected)))
    fields = ("init", "data", "data_sha256", "slots", "max_chunks", "seed", "updates", "batch_size")
    for field in fields:
        values = {name: item["memory"].get(field) for name, item in metadata_by_name.items()}
        if len(set(values.values())) != 1:
            raise ValueError("training metadata differs for {}: {}".format(field, values))
    for name, item in metadata_by_name.items():
        if item["memory"].get("source_present_at_decode") is not False:
            raise ValueError("{} does not certify source-removed decoding".format(name))
        if int(item["prefix"].get("state_scale", 0)) != 256:
            raise ValueError("{} lacks the expected normalized state scale".format(name))
    candidate = metadata_by_name["candidate"]["prefix"]
    shuffled = metadata_by_name["prefix_shuffled"]["prefix"]
    answer_only = metadata_by_name["answer_only"]["prefix"]
    if candidate.get("prefix_mode") != "verified" or shuffled.get("prefix_mode") != "shuffled":
        raise ValueError("candidate/control prefix target modes are invalid")
    weights = candidate.get("weights")
    if not isinstance(weights, dict) or any(float(value) <= 0.0 for value in weights.values()):
        raise ValueError("candidate must use nonzero prefix state and delta supervision")
    if shuffled.get("weights") != weights:
        raise ValueError("shuffled-prefix control must retain candidate weights")
    if any(float(value) != 0.0 for value in answer_only.get("weights", {}).values()):
        raise ValueError("answer-only control must have exact zero prefix weights")


def verify_reports(reports):
    anchor = reports["candidate"]
    for name, report in reports.items():
        if name != "candidate":
            verify_matching(anchor, report)
            verify_pair_coverage(anchor, report)
        for mode in MODES:
            if references(report, mode) != references(anchor, "normal"):
                raise ValueError("{}:{} does not cover the common held-out references".format(name, mode))


def group(reports, predicate):
    candidate = metric(normal_rows(reports["candidate"], predicate))
    controls = {
        "answer_only": metric(normal_rows(reports["answer_only"], predicate)),
        "candidate_zero_packet": metric([row for row in reports["candidate"]["rows"] if row["mode"] == "zero" and predicate(row)]),
        "candidate_shuffled_source": metric([row for row in reports["candidate"]["rows"] if row["mode"] == "shuffled" and predicate(row)]),
        "prefix_shuffled": metric(normal_rows(reports["prefix_shuffled"], predicate)),
    }
    ceiling = max(item["accuracy"] for item in controls.values())
    return {"candidate_normal": candidate, "controls": controls, "control_max_accuracy": ceiling, "normal_margin": candidate["accuracy"] - ceiling}


def pair_group(reports, pair_kind):
    candidate, _ = pair_metric(reports["candidate"]["rows"], "normal", pair_kind)
    controls = {
        "answer_only": pair_metric(reports["answer_only"]["rows"], "normal", pair_kind)[0],
        "candidate_zero_packet": pair_metric(reports["candidate"]["rows"], "zero", pair_kind)[0],
        "candidate_shuffled_source": pair_metric(reports["candidate"]["rows"], "shuffled", pair_kind)[0],
        "prefix_shuffled": pair_metric(reports["prefix_shuffled"]["rows"], "normal", pair_kind)[0],
    }
    ceiling = max(item["accuracy"] for item in controls.values())
    return {"candidate_normal": candidate, "controls": controls, "control_max_accuracy": ceiling, "normal_margin": candidate["accuracy"] - ceiling}


def compare(reports, metadata_by_name=None):
    verify_reports(reports)
    if metadata_by_name is None:
        metadata_by_name = {name: metadata(report) for name, report in reports.items()}
    verify_metadata(metadata_by_name)
    regimes = sorted({row["eval_regime"] for row in reports["candidate"]["rows"]})
    by_regime = {regime: group(reports, lambda row, regime=regime: row["eval_regime"] == regime) for regime in regimes}
    ood = group(reports, lambda row: row["eval_regime"] in {"length_ood", "language_ood"})
    pairs = {kind: pair_group(reports, kind) for kind in ("equivalent", "intervention")}
    gates = {
        "fit_margin_at_least_10pp": by_regime.get("fit_iid", {"normal_margin": float("-inf")})["normal_margin"] >= FIT_MARGIN,
        "length_language_margin_at_least_5pp": ood["normal_margin"] >= OOD_MARGIN,
        "equivalent_pair_margin_at_least_10pp": pairs["equivalent"]["normal_margin"] >= PAIR_MARGIN,
        "intervention_pair_margin_at_least_10pp": pairs["intervention"]["normal_margin"] >= PAIR_MARGIN,
    }
    return {
        "audit": "prefix_state_memory_causal_comparison_v1",
        "reports": {name: report["checkpoint"] for name, report in reports.items()},
        "by_regime": by_regime, "length_language_combined": ood, "pair_results": pairs,
        "gates": gates, "advance_prefix_state_memory": all(gates.values()),
        "claim_boundary": "Passing supports a source-free prefix-state packet effect on this held-out task only; it is not broad reasoning or flagship evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-only", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prefix-shuffled", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    reports = {
        "answer_only": load_report(args.answer_only),
        "candidate": load_report(args.candidate),
        "prefix_shuffled": load_report(args.prefix_shuffled),
    }
    result = compare(reports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[prefix-state-compare] " + json.dumps({"advance": result["advance_prefix_state_memory"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
