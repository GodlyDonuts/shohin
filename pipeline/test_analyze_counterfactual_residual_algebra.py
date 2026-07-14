#!/usr/bin/env python3
"""Synthetic contract tests for the CRA failure taxonomy."""
import json
import tempfile
from pathlib import Path

from analyze_counterfactual_residual_algebra import classify, load_behavior, load_nll


def behavior(path, strict, normal=80, counter=80, zero=0, shuffle=0):
    rows = 100
    path.write_text(json.dumps({"rows": rows, "summary": {
        "normal_correct": normal,
        "paraphrase_correct": normal,
        "counterfactual_correct": counter,
        "zero_recreates_normal": zero,
        "shuffle_recreates_normal": shuffle,
        "strict_causal": strict,
    }}))


def nll(path, counter_margin):
    path.write_text(json.dumps({"rows": 100, "summary": {
        "normal_margin": 1.0,
        "counterfactual_margin": counter_margin,
        "paraphrase_margin": 1.0,
        "zero_margin": 1.0,
        "shuffle_margin": 1.0,
    }, "paired_directional": 80, "strict_directional": 80}))


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    paths = {name: root / (name + ".json") for name in ("combined", "train", "language", "values", "delta", "query", "two", "nll")}
    behavior(paths["combined"], strict=10, normal=70, counter=5)
    behavior(paths["train"], strict=80)
    behavior(paths["language"], strict=10)
    for name in ("values", "delta", "query", "two"):
        behavior(paths[name], strict=70)
    nll(paths["nll"], counter_margin=-0.4)
    factors = {name: load_behavior(paths[name]) for name in ("language", "values", "delta", "query", "two")}
    diagnoses, recommendation = classify(load_behavior(paths["combined"]), load_behavior(paths["train"]), load_nll(paths["nll"]), factors)
    assert "counterfactual_sign_not_functional" in diagnoses
    assert recommendation == "paired_sign_discrimination_is_the_only_justified_follow_up"

    behavior(paths["train"], strict=5, normal=10, counter=5)
    diagnoses, recommendation = classify(load_behavior(paths["combined"]), load_behavior(paths["train"]), load_nll(paths["nll"]), factors)
    assert "not_learned_in_distribution" in diagnoses
    assert recommendation == "close_residual_algebra_at_this_capacity"

print("CRA failure taxonomy checks: passed")
