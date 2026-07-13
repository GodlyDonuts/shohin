#!/usr/bin/env python3
"""Synthetic contracts for the prefix-state causal comparator."""

import json
from pathlib import Path

from compare_prefix_state_memory import compare


def metadata(mode="verified", zero=False):
    return {
        "memory": {
            "init": "raw.pt", "data": "data.jsonl", "data_sha256": "smoke-data", "slots": 8,
            "max_chunks": 8, "seed": 17, "updates": 100, "batch_size": 4,
            "source_present_at_decode": False,
        },
        "prefix": {
            "state_scale": 256, "prefix_mode": mode,
            "weights": {"state": 0.0 if zero else 1.0, "delta": 0.0 if zero else 0.5},
        },
    }


def report(rows, checkpoint):
    return {
        "audit": "source_dropping_memory_heldout_v1", "data_sha256": "smoke-data", "seed": 17,
        "checkpoint": checkpoint, "rows": rows,
    }


def main():
    source = Path("/tmp/shohin-lsa-smoke/eval.jsonl")
    if not source.exists():
        raise SystemExit("run the generator smoke before this test")
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    reports = {}
    for name, normal_correct in {"candidate": True, "answer_only": False, "prefix_shuffled": False}.items():
        rows = []
        for mode in ("normal", "zero", "shuffled"):
            for case in cases:
                correct = mode == "normal" and normal_correct
                rows.append({
                    "mode": mode, "reference": case["pair_id"] + "-" + case["pair_member"],
                    "eval_regime": case["eval_regime"], "chunk_count": case["chunk_count"],
                    "query_kind": case["query_kind"], "pair_id": case["pair_id"],
                    "pair_kind": case["pair_kind"], "pair_member": case["pair_member"],
                    "prediction": int(case["answer"]) if correct else -999, "correct": correct,
                })
        reports[name] = report(rows, name + ".pt")
    metadatas = {
        "candidate": metadata(), "answer_only": metadata(zero=True), "prefix_shuffled": metadata(mode="shuffled"),
    }
    result = compare(reports, metadatas)
    assert result["advance_prefix_state_memory"] and all(result["gates"].values())
    reports["prefix_shuffled"] = reports["candidate"]
    result = compare(reports, metadatas)
    assert not result["advance_prefix_state_memory"]
    print("prefix-state comparator contract passed")


if __name__ == "__main__":
    main()
