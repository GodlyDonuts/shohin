#!/usr/bin/env python3
"""Focused generator contracts for matched operator counterfactual reflection."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generate_operator_counterfactual_reflection_v1 import (
    CONTRACT_NEUTRAL,
    CONTRACT_REFLECTION,
    STATE_WIDTH,
    build_rows,
)


def main() -> None:
    reflection, neutral = build_rows(per_family=8, seed=7)
    assert len(reflection) == len(neutral) == 48
    assert {row["contract"] for row in reflection} == {CONTRACT_REFLECTION}
    assert {row["contract"] for row in neutral} == {CONTRACT_NEUTRAL}
    reflection_by_episode = {
        (row["family"], tuple(row["operations"]), tuple(sorted(row["counterfactual"].items()))): row
        for row in reflection
    }
    neutral_by_episode = {
        (row["family"], tuple(row["operations"]), tuple(sorted(row["counterfactual"].items()))): row
        for row in neutral
    }
    assert reflection_by_episode.keys() == neutral_by_episode.keys()
    for key, reflected in reflection_by_episode.items():
        control = neutral_by_episode[key]
        target = reflected["response"]
        baseline = control["response"]
        assert "<reflect>" in target and "</reflect>" in target
        assert "old_op=" in target and "new_op=" in target
        assert f"state_before={'0' * STATE_WIDTH}" in baseline
        assert f"counterfactual_after={'0' * STATE_WIDTH}" in baseline
        state = reflected["counterfactual"]
        assert state["state_before"] >= 0 and state["counterfactual_after"] >= 0
    print("operator counterfactual reflection generator checks: passed")


if __name__ == "__main__":
    main()
