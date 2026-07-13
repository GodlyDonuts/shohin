#!/usr/bin/env python3
"""Locked held-out gate for dense source-free latent-state algebra.

The candidate must beat an answer-only matched control, zeroed packet, and
shuffled-source controls.  It must also solve complete equivalent and
intervention pairs.  Passing is narrow retained-state evidence only; it is not
a broad reasoning or flagship-promotion result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


FIT_MARGIN = 0.10
OOD_MARGIN = 0.05
PAIR_MARGIN = 0.10
MODES = ("normal", "zero", "shuffled")


def load_report(path):
    report = json.loads(Path(path).read_text())
    if report.get("audit") != "source_dropping_memory_heldout_v1":
        raise ValueError("{} is not a source-memory held-out report".format(path))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("{} has no rows".format(path))
    required = ("mode", "reference", "eval_regime", "chunk_count", "correct")
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("{} has incomplete rows".format(path))
    return report


def metric(rows):
    if not rows:
        raise ValueError("empty comparison group")
    correct = sum(bool(row["correct"]) for row in rows)
    return {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def references(report, mode):
    return {row["reference"] for row in report["rows"] if row["mode"] == mode}


def verify_matching(control, candidate):
    if control.get("data_sha256") != candidate.get("data_sha256"):
        raise ValueError("control and candidate use different held-out data")
    if control.get("seed") != candidate.get("seed"):
        raise ValueError("control and candidate use different selection seeds")
    expected = None
    for report in (control, candidate):
        for mode in MODES:
            found = references(report, mode)
            if expected is None:
                expected = found
            elif found != expected:
                raise ValueError("modes/models do not cover identical source references")


def pair_metric(rows, mode, pair_kind):
    grouped = defaultdict(list)
    for row in rows:
        if row["mode"] == mode and row.get("pair_kind") == pair_kind and row.get("pair_id"):
            grouped[row["pair_id"]].append(row)
    complete = {
        pair_id: pair for pair_id, pair in grouped.items()
        if len(pair) == 2 and {row.get("pair_member") for row in pair} == {"a", "b"}
    }
    if pair_kind == "equivalent":
        correct = sum(
            all(bool(row["correct"]) for row in pair) and len({row.get("prediction") for row in pair}) == 1
            for pair in complete.values()
        )
    elif pair_kind == "intervention":
        correct = sum(
            all(bool(row["correct"]) for row in pair) and len({row.get("prediction") for row in pair}) == 2
            for pair in complete.values()
        )
    else:
        raise ValueError("unknown pair kind")
    return {"pairs": len(complete), "correct": correct, "accuracy": correct / len(complete) if complete else None}, set(complete)


def verify_pair_coverage(control, candidate):
    for pair_kind in ("equivalent", "intervention"):
        expected = None
        for report in (control, candidate):
            for mode in MODES:
                _, found = pair_metric(report["rows"], mode, pair_kind)
                if expected is None:
                    expected = found
                elif found != expected:
                    raise ValueError("{} pairs are not matched across modes/models".format(pair_kind))
        if not expected:
            raise ValueError("no complete {} pairs were retained".format(pair_kind))


def compare(control, candidate):
    verify_matching(control, candidate)
    verify_pair_coverage(control, candidate)
    control_rows, candidate_rows = control["rows"], candidate["rows"]

    def group(predicate):
        normal = metric([row for row in candidate_rows if row["mode"] == "normal" and predicate(row)])
        controls = {
            "answer_only_control": metric([row for row in control_rows if row["mode"] == "normal" and predicate(row)]),
            "candidate_zero_packet": metric([row for row in candidate_rows if row["mode"] == "zero" and predicate(row)]),
            "candidate_shuffled_source": metric([row for row in candidate_rows if row["mode"] == "shuffled" and predicate(row)]),
        }
        ceiling = max(item["accuracy"] for item in controls.values())
        return {
            "candidate_normal": normal,
            "controls": controls,
            "control_max_accuracy": ceiling,
            "normal_margin": normal["accuracy"] - ceiling,
        }

    regimes = sorted({row["eval_regime"] for row in candidate_rows})
    chunks = sorted({int(row["chunk_count"]) for row in candidate_rows})
    query_kinds = sorted({row.get("query_kind") for row in candidate_rows if row.get("query_kind")})
    by_regime = {regime: group(lambda row, regime=regime: row["eval_regime"] == regime) for regime in regimes}
    by_chunk = {str(chunk): group(lambda row, chunk=chunk: int(row["chunk_count"]) == chunk) for chunk in chunks}
    by_query = {kind: group(lambda row, kind=kind: row.get("query_kind") == kind) for kind in query_kinds}
    ood = group(lambda row: row["eval_regime"] in {"length_ood", "language_ood"})

    pair_results = {}
    for pair_kind in ("equivalent", "intervention"):
        candidate_normal, _ = pair_metric(candidate_rows, "normal", pair_kind)
        controls = {
            "answer_only_control": pair_metric(control_rows, "normal", pair_kind)[0],
            "candidate_zero_packet": pair_metric(candidate_rows, "zero", pair_kind)[0],
            "candidate_shuffled_source": pair_metric(candidate_rows, "shuffled", pair_kind)[0],
        }
        ceiling = max(item["accuracy"] for item in controls.values())
        pair_results[pair_kind] = {
            "candidate_normal": candidate_normal,
            "controls": controls,
            "control_max_accuracy": ceiling,
            "normal_margin": candidate_normal["accuracy"] - ceiling,
        }

    chunk_wins = sum(item["normal_margin"] > 0 for item in by_chunk.values())
    query_wins = sum(item["normal_margin"] > 0 for item in by_query.values())
    gates = {
        "fit_margin_at_least_10pp": by_regime.get("fit_iid", {"normal_margin": float("-inf")})["normal_margin"] >= FIT_MARGIN,
        "length_language_margin_at_least_5pp": ood["normal_margin"] >= OOD_MARGIN,
        "at_least_three_chunk_counts_beat_controls": chunk_wins >= 3,
        "at_least_two_query_kinds_beat_controls": query_wins >= 2,
        "equivalent_pair_margin_at_least_10pp": pair_results["equivalent"]["normal_margin"] >= PAIR_MARGIN,
        "intervention_pair_margin_at_least_10pp": pair_results["intervention"]["normal_margin"] >= PAIR_MARGIN,
    }
    return {
        "audit": "latent_state_algebra_matched_comparison_v1",
        "control_checkpoint": control.get("checkpoint"),
        "candidate_checkpoint": candidate.get("checkpoint"),
        "data_sha256": candidate.get("data_sha256"),
        "fit_margin_threshold": FIT_MARGIN,
        "ood_margin_threshold": OOD_MARGIN,
        "pair_margin_threshold": PAIR_MARGIN,
        "by_regime": by_regime,
        "length_language_combined": ood,
        "by_chunk": by_chunk,
        "by_query_kind": by_query,
        "pair_results": pair_results,
        "chunk_count_wins": chunk_wins,
        "query_kind_wins": query_wins,
        "gates": gates,
        "advance_dense_latent_state_algebra": all(gates.values()),
        "claim_boundary": (
            "Passing supports a causal source-free retained-state effect on this held-out paired task only. "
            "It is not broad reasoning and cannot modify flagship pretraining without separate transfer evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    result = compare(load_report(args.control), load_report(args.candidate))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[lsa-compare] " + json.dumps({
        "gates": result["gates"], "advance": result["advance_dense_latent_state_algebra"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
