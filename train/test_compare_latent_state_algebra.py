#!/usr/bin/env python3
"""Synthetic contract test for the preregistered LSA comparator."""

import json
from pathlib import Path

from compare_latent_state_algebra import compare


def report(rows):
    return {
        "audit": "source_dropping_memory_heldout_v1",
        "data_sha256": "smoke-data",
        "seed": 17,
        "checkpoint": "smoke.pt",
        "rows": rows,
    }


def main():
    source = Path("/tmp/shohin-lsa-smoke/eval.jsonl")
    if not source.exists():
        raise SystemExit("run the generator smoke before this test")
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    candidate, control = [], []
    for mode in ("normal", "zero", "shuffled"):
        for case in cases:
            base = {
                "mode": mode,
                "reference": case["pair_id"] + "-" + case["pair_member"],
                "eval_regime": case["eval_regime"],
                "chunk_count": case["chunk_count"],
                "query_kind": case["query_kind"],
                "pair_id": case["pair_id"],
                "pair_kind": case["pair_kind"],
                "pair_member": case["pair_member"],
                "expected": int(case["answer"]),
            }
            candidate.append({
                **base,
                "prediction": int(case["answer"]) if mode == "normal" else -999,
                "correct": mode == "normal",
            })
            control.append({**base, "prediction": -999, "correct": False})
    result = compare(report(control), report(candidate))
    assert result["advance_dense_latent_state_algebra"]
    assert all(result["gates"].values())
    assert result["pair_results"]["equivalent"]["candidate_normal"]["correct"] > 0
    assert result["pair_results"]["intervention"]["candidate_normal"]["correct"] > 0
    print("latent state algebra comparator contract passed")


if __name__ == "__main__":
    main()
