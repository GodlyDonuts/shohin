from __future__ import annotations

import inspect

from pilot_er_addressed_marginal_train_canary import (
    CONTRACT,
    FROZEN_SOURCE_PATHS,
    SCHEMA,
    THRESHOLDS,
    compute_addressed_gates,
    main,
)
from pilot_er_addressed_marginal_relation_adapter import EXPECTED_PARAMETERS


def test_addressed_canary_preserves_budget_thresholds_and_custody() -> None:
    assert SCHEMA == "r12_er_addressed_marginal_train_only_canary_v1"
    assert CONTRACT["fit_families"] == 10_000
    assert CONTRACT["probe_families"] == 2_000
    assert CONTRACT["updates"] == 2_500
    assert CONTRACT["outcome_supervision"] is False
    assert CONTRACT["development_reads"] == 0
    assert CONTRACT["confirmation_reads"] == 0
    assert THRESHOLDS["witness_pointer"] == 0.90
    assert THRESHOLDS["relation_rows"] == 0.90
    assert THRESHOLDS["joint"] == 0.85


def test_addressed_canary_freezes_all_new_runtime_paths() -> None:
    required = {
        "R12_ER_ADDRESSED_MARGINAL_ROUTE_PREREG.md",
        "train/er_addressed_marginal_relation_adapter.py",
        "train/pilot_er_addressed_marginal_relation_adapter.py",
        "train/pilot_er_addressed_marginal_train_canary.py",
        "train/test_er_addressed_marginal_relation_adapter.py",
        "train/test_pilot_er_addressed_marginal_train_canary.py",
        "train/jobs/er_addressed_marginal_train_canary.sbatch",
    }
    assert required.issubset(FROZEN_SOURCE_PATHS)


def test_addressed_main_loads_only_training_split() -> None:
    source = inspect.getsource(main)
    assert 'filename="train.jsonl"' in source
    assert 'split=TRAIN_SPLIT' in source
    assert 'filename="development.jsonl"' not in source
    assert 'filename="confirmation.jsonl"' not in source


def test_addressed_gate_uses_addressed_parameter_certificate() -> None:
    source = inspect.getsource(compute_addressed_gates)
    assert "== EXPECTED_PARAMETERS" in source
    assert EXPECTED_PARAMETERS["complete_system"] < 200_000_000
