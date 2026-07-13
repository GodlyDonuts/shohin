#!/usr/bin/env python3
"""Locked gate for semantic packet readback at every source boundary.

The verified-prefix model must beat both packet ablations and two equal-work
training controls: replaying the final query at every prefix and assigning
another example's readback labels to each prefix.  Passing remains a narrow
retained-state result and only justifies the separate final-answer/pair gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIT_MARGIN = 0.10
OOD_MARGIN = 0.05
MODES = ("normal", "zero", "shuffled")


def load_report(path):
    report = json.loads(Path(path).read_text())
    if report.get("audit") != "causal_prefix_readback_heldout_v1":
        raise ValueError("{} is not a causal-prefix-readback report".format(path))
    rows = report.get("rows")
    required = ("mode", "reference", "eval_regime", "chunk_count", "prefix_index", "key", "correct")
    if not isinstance(rows, list) or not rows or any(any(key not in row for key in required) for row in rows):
        raise ValueError("{} has incomplete readback rows".format(path))
    return report


def metric(rows):
    if not rows:
        raise ValueError("empty comparison group")
    correct = sum(bool(row["correct"]) for row in rows)
    return {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def references(report, mode):
    return {
        (row["reference"], int(row["prefix_index"]), str(row["key"]))
        for row in report["rows"] if row["mode"] == mode
    }


def memory_metadata(report):
    metadata = report.get("checkpoint_memory_metadata")
    if not isinstance(metadata, dict) or metadata.get("source_present_at_decode") is not False:
        raise ValueError("report lacks source-removed checkpoint metadata")
    keys = ("init", "data", "data_sha256", "slots", "max_chunks", "seed", "updates", "batch_size")
    if any(key not in metadata for key in keys):
        raise ValueError("report lacks matched-memory metadata")
    return {key: metadata[key] for key in keys}


def readback_metadata(report):
    metadata = report.get("checkpoint_readback_metadata")
    if not isinstance(metadata, dict) or metadata.get("decoder_readback_at_every_prefix") is not True:
        raise ValueError("report lacks causal-prefix-readback metadata")
    return metadata


def verify_matching(verified, final_replay, label_shuffled):
    reports = {"verified": verified, "final_replay": final_replay, "label_shuffled": label_shuffled}
    expected_data, expected_seed, expected_refs, expected_memory = None, None, None, None
    for name, report in reports.items():
        if report.get("data_sha256") is None or report.get("seed") is None:
            raise ValueError("{} lacks data binding".format(name))
        metadata = memory_metadata(report)
        if expected_data is None:
            expected_data, expected_seed, expected_memory = report["data_sha256"], report["seed"], metadata
        elif (report["data_sha256"], report["seed"], metadata) != (expected_data, expected_seed, expected_memory):
            raise ValueError("readback reports do not share init/data/seed/update metadata")
        for mode in MODES:
            found = references(report, mode)
            if expected_refs is None:
                expected_refs = found
            elif found != expected_refs:
                raise ValueError("readback reports or modes cover different prefix references")
    modes = {name: readback_metadata(report).get("readback_mode") for name, report in reports.items()}
    if modes != {"verified": "verified", "final_replay": "replicated-final", "label_shuffled": "shuffled"}:
        raise ValueError("reports do not bind the required verified/final-replay/label-shuffle controls")
    if readback_metadata(final_replay).get("equal_decoder_work_control") is not True:
        raise ValueError("final replay is not marked as equal decoder work")


def compare(verified, final_replay, label_shuffled):
    verify_matching(verified, final_replay, label_shuffled)
    candidate_rows = verified["rows"]
    replay_rows = final_replay["rows"]
    shuffled_rows = label_shuffled["rows"]

    def group(predicate):
        candidate = metric([row for row in candidate_rows if row["mode"] == "normal" and predicate(row)])
        controls = {
            "equal_work_final_replay": metric([row for row in replay_rows if row["mode"] == "normal" and predicate(row)]),
            "shuffled_readback_labels": metric([row for row in shuffled_rows if row["mode"] == "normal" and predicate(row)]),
            "candidate_zero_packet": metric([row for row in candidate_rows if row["mode"] == "zero" and predicate(row)]),
            "candidate_shuffled_source": metric([row for row in candidate_rows if row["mode"] == "shuffled" and predicate(row)]),
        }
        ceiling = max(item["accuracy"] for item in controls.values())
        return {
            "candidate_normal": candidate,
            "controls": controls,
            "control_max_accuracy": ceiling,
            "normal_margin": candidate["accuracy"] - ceiling,
        }

    regimes = sorted({row["eval_regime"] for row in candidate_rows})
    chunks = sorted({int(row["chunk_count"]) for row in candidate_rows})
    prefixes = sorted({int(row["prefix_index"]) for row in candidate_rows})
    keys = sorted({str(row["key"]) for row in candidate_rows})
    by_regime = {regime: group(lambda row, regime=regime: row["eval_regime"] == regime) for regime in regimes}
    by_chunk = {str(chunk): group(lambda row, chunk=chunk: int(row["chunk_count"]) == chunk) for chunk in chunks}
    by_prefix = {str(prefix): group(lambda row, prefix=prefix: int(row["prefix_index"]) == prefix) for prefix in prefixes}
    by_key = {key: group(lambda row, key=key: str(row["key"]) == key) for key in keys}
    ood = group(lambda row: row["eval_regime"] in {"length_ood", "language_ood"})
    gates = {
        "fit_margin_at_least_10pp": by_regime.get("fit_iid", {"normal_margin": float("-inf")})["normal_margin"] >= FIT_MARGIN,
        "length_language_margin_at_least_5pp": ood["normal_margin"] >= OOD_MARGIN,
        "at_least_three_chunk_counts_beat_controls": sum(item["normal_margin"] > 0 for item in by_chunk.values()) >= 3,
        "at_least_two_prefix_depths_beat_controls": sum(item["normal_margin"] > 0 for item in by_prefix.values()) >= 2,
        "both_register_keys_beat_controls": len(by_key) >= 2 and all(item["normal_margin"] > 0 for item in by_key.values()),
    }
    return {
        "audit": "causal_prefix_readback_comparison_v1",
        "verified_checkpoint": verified.get("checkpoint"),
        "final_replay_checkpoint": final_replay.get("checkpoint"),
        "label_shuffled_checkpoint": label_shuffled.get("checkpoint"),
        "data_sha256": verified.get("data_sha256"),
        "fit_margin_threshold": FIT_MARGIN,
        "ood_margin_threshold": OOD_MARGIN,
        "by_regime": by_regime,
        "length_language_combined": ood,
        "by_chunk": by_chunk,
        "by_prefix": by_prefix,
        "by_key": by_key,
        "gates": gates,
        "advance_causal_prefix_readback": all(gates.values()),
        "claim_boundary": (
            "Passing supports source-free semantic readback from intermediate packets on this held-out task only. "
            "It must still pass a separate final-answer and paired-intervention gate before any broader claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--final-replay", required=True)
    parser.add_argument("--label-shuffled", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    result = compare(load_report(args.verified), load_report(args.final_replay), load_report(args.label_shuffled))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[causal-prefix-compare] " + json.dumps({
        "advance": result["advance_causal_prefix_readback"], "gates": result["gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
