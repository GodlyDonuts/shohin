#!/usr/bin/env python3
"""Compare matched no-slot and source-packet memory evaluations without cherry-picking.

M1 may advance only when normal source-derived packets beat all three controls:
M0 no-slot, M1 zeroed packet, and M1 shuffled source order. The result is a
small-model memory gate, not a broad reasoning verdict.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODES = ("normal", "zero", "shuffled")
FIT_MARGIN = 0.15
OOD_MARGIN = 0.05


def load_report(path):
    report = json.loads(Path(path).read_text())
    if report.get("audit") != "source_dropping_memory_heldout_v1":
        raise ValueError("{} is not a source-memory held-out report".format(path))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("{} has no evaluation rows".format(path))
    required = ("mode", "eval_regime", "chunk_count", "reference", "correct")
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("{} has incomplete evaluation rows".format(path))
    return report


def accuracy(rows):
    if not rows:
        raise ValueError("empty comparison group")
    correct = sum(bool(row["correct"]) for row in rows)
    return {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def by_mode(rows, mode, predicate):
    return accuracy([row for row in rows if row["mode"] == mode and predicate(row)])


def matching_normal_rows(report):
    rows = report["rows"]
    references = {row["reference"] for row in rows if row["mode"] == "normal"}
    for mode in MODES:
        found = {row["reference"] for row in rows if row["mode"] == mode}
        if found != references:
            raise ValueError("mode {} does not cover the same source references".format(mode))
    return references


def compare(m0_report, m1_report):
    if m0_report.get("data_sha256") != m1_report.get("data_sha256"):
        raise ValueError("M0 and M1 evaluated different held-out data")
    if m0_report.get("seed") != m1_report.get("seed"):
        raise ValueError("M0 and M1 used different selection seeds")
    if matching_normal_rows(m0_report) != matching_normal_rows(m1_report):
        raise ValueError("M0 and M1 evaluated different source references")

    m0_rows, m1_rows = m0_report["rows"], m1_report["rows"]
    regimes = sorted({row["eval_regime"] for row in m1_rows})
    chunks = sorted({int(row["chunk_count"]) for row in m1_rows})
    query_kinds = sorted({row.get("query_kind") for row in m1_rows if row.get("query_kind")})

    def group(predicate):
        normal = by_mode(m1_rows, "normal", predicate)
        controls = {
            "m0_no_slot": by_mode(m0_rows, "normal", predicate),
            "m1_zero_packet": by_mode(m1_rows, "zero", predicate),
            "m1_shuffled_source": by_mode(m1_rows, "shuffled", predicate),
        }
        control_max = max(item["accuracy"] for item in controls.values())
        return {
            "m1_normal": normal,
            "controls": controls,
            "control_max_accuracy": control_max,
            "normal_margin": normal["accuracy"] - control_max,
        }

    by_regime = {regime: group(lambda row, regime=regime: row["eval_regime"] == regime) for regime in regimes}
    by_chunk = {str(chunk): group(lambda row, chunk=chunk: int(row["chunk_count"]) == chunk) for chunk in chunks}
    by_query = {
        query_kind: group(lambda row, query_kind=query_kind: row.get("query_kind") == query_kind)
        for query_kind in query_kinds
    }
    ood_rows = {"length_ood", "language_ood"}
    ood = group(lambda row: row["eval_regime"] in ood_rows)
    fit_margin = by_regime["fit_iid"]["normal_margin"] if "fit_iid" in by_regime else float("-inf")
    chunk_wins = sum(item["normal_margin"] > 0 for item in by_chunk.values())
    query_wins = sum(item["normal_margin"] > 0 for item in by_query.values())
    gates = {
        "fit_margin_at_least_15pp": fit_margin >= FIT_MARGIN,
        "length_language_margin_at_least_5pp": ood["normal_margin"] >= OOD_MARGIN,
        "at_least_three_chunk_counts_beat_controls": chunk_wins >= 3,
        "at_least_two_query_kinds_beat_controls": query_wins >= 2,
    }
    return {
        "audit": "source_dropping_memory_matched_comparison_v1",
        "m0_checkpoint": m0_report.get("checkpoint"),
        "m1_checkpoint": m1_report.get("checkpoint"),
        "data_sha256": m1_report.get("data_sha256"),
        "fit_margin_threshold": FIT_MARGIN,
        "ood_margin_threshold": OOD_MARGIN,
        "by_regime": by_regime,
        "length_language_combined": ood,
        "by_chunk": by_chunk,
        "by_query_kind": by_query,
        "chunk_count_wins": chunk_wins,
        "query_kind_wins": query_wins,
        "gates": gates,
        "advance_answer_only_source_packet": all(gates.values()),
        "claim_boundary": (
            "Passing shows a causal advantage on this held-out source-removal memory task only. "
            "It does not establish broad reasoning or justify modifying flagship pretraining."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", required=True)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))
    result = compare(load_report(args.m0), load_report(args.m1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[source-memory-compare] " + json.dumps({
        "gates": result["gates"],
        "advance": result["advance_answer_only_source_packet"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
