#!/usr/bin/env python3
"""Locked comparison for recurrent versus reset scratch adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric(report, condition, regime, key):
    return float(report["summary"][condition][regime][key])


def compare(recurrent, reset):
    rec_meta = recurrent["adapter_metadata"]
    reset_meta = reset["adapter_metadata"]
    matched = all(rec_meta.get(key) == reset_meta.get(key) for key in (
        "base_sha256", "data_sha256", "seed", "layer", "slots", "width",
        "steps", "batch_size", "selected_examples_per_epoch", "updates",
        "adapter_parameters", "workspace_topk", "workspace_temperature", "workspace_basis",
        "initial_adapter_sha256", "reset_aggregation",
    )) and rec_meta.get("mode") == "recurrent" and reset_meta.get("mode") == "reset"
    same_eval = all(recurrent.get(key) == reset.get(key) for key in (
        "base_sha256", "data_sha256", "per_depth", "seed",
    ))
    rec_normal = metric(recurrent, "normal_trained_depth", "all", "mean_nll")
    reset_normal = metric(reset, "normal_trained_depth", "all", "mean_nll")
    strongest_state_control = min(
        metric(recurrent, "zero_trained_depth", "all", "mean_nll"),
        metric(recurrent, "shuffled_trained_depth", "all", "mean_nll"),
    )
    depth_normal = metric(recurrent, "normal_trained_depth", "depth_ood", "mean_nll")
    depth_t1 = metric(recurrent, "normal_t1", "depth_ood", "mean_nll")
    depth_reset = metric(recurrent, "reset_trained_depth", "depth_ood", "mean_nll")
    fit_advantage = metric(reset, "normal_trained_depth", "fit_iid", "mean_nll") - metric(
        recurrent, "normal_trained_depth", "fit_iid", "mean_nll",
    )
    depth_advantage = metric(reset, "normal_trained_depth", "depth_ood", "mean_nll") - depth_normal
    state_margin = strongest_state_control - rec_normal
    recurrence_margin = min(depth_t1, depth_reset) - depth_normal
    exact_regime_wins = sum(
        metric(recurrent, "normal_trained_depth", regime, "sequence_accuracy")
        > metric(reset, "normal_trained_depth", regime, "sequence_accuracy")
        for regime in ("fit_iid", "depth_ood", "language_ood", "full_ood")
    )
    gates = {
        "matched_training": matched,
        "matched_evaluation": same_eval,
        "fit_iid_nll_advantage_at_least_0_05": fit_advantage >= 0.05,
        "depth_ood_nll_advantage_at_least_0_03": depth_advantage >= 0.03,
        "state_necessity_nll_margin_at_least_0_05": state_margin >= 0.05,
        "within_model_recurrence_nll_margin_at_least_0_03": recurrence_margin >= 0.03,
        "exact_sequence_wins_at_least_two_regimes": exact_regime_wins >= 2,
    }
    return {
        "audit": "causal_recurrent_scratch_comparison_v1",
        "metrics": {
            "fit_iid_recurrent_over_reset_nll": fit_advantage,
            "depth_ood_recurrent_over_reset_nll": depth_advantage,
            "state_necessity_nll_margin": state_margin,
            "within_model_recurrence_nll_margin": recurrence_margin,
            "exact_sequence_regime_wins": exact_regime_wins,
        },
        "gates": gates,
        "advance_to_autoregressive_eval": all(gates.values()),
        "claim_boundary": (
            "A pass admits autoregressive causal evaluation only. It is not evidence of broad reasoning."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recurrent", required=True)
    parser.add_argument("--reset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))
    recurrent, reset = json.load(open(args.recurrent)), json.load(open(args.reset))
    result = compare(recurrent, reset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
