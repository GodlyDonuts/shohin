#!/usr/bin/env python3
"""Synthetic contracts for the recurrent scratch comparison."""

from compare_causal_recurrent_scratch import compare
from eval_causal_recurrent_scratch_nll import condition_specs


REGIMES = ("all", "fit_iid", "depth_ood", "language_ood", "full_ood")


def report(mode, normal, t1, reset_nll, zero, shuffled, exact):
    meta = {
        "mode": mode, "base_sha256": "b", "data_sha256": "d", "seed": 1,
        "layer": 3, "slots": 4, "width": 16, "steps": 4, "batch_size": 8,
        "selected_examples_per_epoch": 100, "updates": 20, "adapter_parameters": 10,
        "workspace_topk": 8, "workspace_temperature": 0.2,
        "workspace_basis": "frozen_normalized_unembedding_topk_v1",
        "initial_adapter_sha256": "i",
        "reset_aggregation": "mean_of_identical_independent_executions_v1",
    }
    summary = {}
    values = {
        "normal_trained_depth": normal,
        "normal_t1": t1,
        "reset_trained_depth": reset_nll,
        "zero_trained_depth": zero,
        "shuffled_trained_depth": shuffled,
        "disabled": zero,
    }
    for condition, nll in values.items():
        summary[condition] = {
            regime: {"mean_nll": nll, "sequence_accuracy": exact if condition == "normal_trained_depth" else 0.1}
            for regime in REGIMES
        }
    return {
        "adapter_metadata": meta, "base_sha256": "b", "data_sha256": "d",
        "per_depth": 4, "seed": 1, "summary": summary,
    }


def main():
    recurrent_conditions = {row[0]: row for row in condition_specs("recurrent", 4)}
    reset_conditions = {row[0]: row for row in condition_specs("reset", 4)}
    assert recurrent_conditions["normal_trained_depth"][2:] == (True, 4)
    assert reset_conditions["normal_trained_depth"][2:] == (False, 4)
    assert recurrent_conditions["reset_trained_depth"][2:] == (False, 4)

    recurrent = report("recurrent", 0.8, 1.0, 0.95, 1.1, 1.2, 0.4)
    reset = report("reset", 1.0, 1.0, 1.0, 1.1, 1.2, 0.1)
    result = compare(recurrent, reset)
    assert result["advance_to_autoregressive_eval"]
    recurrent["summary"]["normal_trained_depth"]["depth_ood"]["mean_nll"] = 0.99
    result = compare(recurrent, reset)
    assert not result["advance_to_autoregressive_eval"]
    assert not result["gates"]["depth_ood_nll_advantage_at_least_0_03"]
    print("causal recurrent scratch comparison tests passed")


if __name__ == "__main__":
    main()
