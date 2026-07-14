#!/usr/bin/env python3
"""Frozen probe split and deterministic schedule contracts for R6 training."""

from train_future_effect_compiler import (
    HELDOUT_PROBE_INDICES,
    TRAIN_PROBE_INDICES,
    selected_probe_indices,
)


def main():
    assert len(TRAIN_PROBE_INDICES) == 48
    assert len(HELDOUT_PROBE_INDICES) == 16
    assert set(TRAIN_PROBE_INDICES).isdisjoint(HELDOUT_PROBE_INDICES)
    assert set(TRAIN_PROBE_INDICES).union(HELDOUT_PROBE_INDICES) == set(range(64))
    assert selected_probe_indices(7, 11, 4) == selected_probe_indices(7, 11, 4)
    covered = set()
    for step in range(48):
        covered.update(selected_probe_indices(step, 0, 4))
    assert covered == set(TRAIN_PROBE_INDICES)
    print("future-effect training probe contracts passed")


if __name__ == "__main__":
    main()
