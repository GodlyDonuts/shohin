#!/usr/bin/env python3
"""Deterministic gates for the matched source-memory comparison report."""

from compare_source_dropping_memory import compare


def report(slots, normal_correct, counterfactual=False):
    rows = []
    for mode in ("normal", "zero", "shuffled"):
        for chunk_count in (2, 3, 4, 5, 6, 8):
            for query_kind in ("read", "sum", "difference"):
                for index in range(5):
                    correct = mode == "normal" and normal_correct and index < 4
                    rows.append({
                        "mode": mode,
                        "eval_regime": "fit_iid" if chunk_count <= 4 else "length_ood",
                        "chunk_count": chunk_count,
                        "query_kind": query_kind,
                        "reference": "{}:{}:{}".format(chunk_count, query_kind, index),
                        "correct": correct,
                    })
        if counterfactual:
            for variant, prediction in (("a", 3), ("b", 5)):
                rows.append({
                    "mode": mode,
                    "eval_regime": "fit_iid",
                    "chunk_count": 2,
                    "query_kind": "read",
                    "reference": "pair:{}".format(variant),
                    "counterfactual_id": "pair",
                    "counterfactual_variant": variant,
                    "prediction": prediction if normal_correct and mode == "normal" else None,
                    "correct": normal_correct and mode == "normal",
                })
    return {
        "audit": "source_dropping_memory_heldout_v1",
        "checkpoint": "slots-{}".format(slots),
        "data_sha256": "data",
        "seed": 7,
        "rows": rows,
    }


def main():
    passed = compare(report(0, False), report(8, True))
    assert passed["advance_answer_only_source_packet"]
    failed = compare(report(0, False), report(8, False))
    assert not failed["advance_answer_only_source_packet"]
    assert not failed["gates"]["fit_margin_at_least_15pp"]
    ledger = compare(report(0, False, True), report(8, True, True))
    assert ledger["gates"]["counterfactual_pair_margin_at_least_10pp"]
    print("source-memory comparison tests passed")


if __name__ == "__main__":
    main()
