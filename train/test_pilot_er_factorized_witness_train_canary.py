from __future__ import annotations

import inspect

from pilot_er_factorized_witness_route_adapter import EXPECTED_PARAMETERS
from pilot_er_factorized_witness_train_canary import (
    ARM_MODES,
    CONTRACT,
    FROZEN_SOURCE_PATHS,
    MINIMUM_WITNESS_GAIN,
    SCHEMA,
    THRESHOLDS,
    compute_factorized_gates,
    main,
)


def test_factorized_canary_preserves_budget_thresholds_and_custody() -> None:
    assert SCHEMA == "r12_er_factorized_witness_route_train_only_canary_v1"
    assert CONTRACT["fit_families"] == 10_000
    assert CONTRACT["probe_families"] == 2_000
    assert CONTRACT["updates"] == 2_500
    assert CONTRACT["outcome_supervision"] is False
    assert CONTRACT["development_reads"] == 0
    assert CONTRACT["confirmation_reads"] == 0
    assert THRESHOLDS["witness_pointer"] == 0.90
    assert THRESHOLDS["relation_rows"] == 0.90
    assert THRESHOLDS["joint"] == 0.85
    assert ARM_MODES == (
        "treatment",
        "baseline",
        "structural_only",
        "shuffled_address",
    )
    assert MINIMUM_WITNESS_GAIN == 0.005


def test_factorized_canary_freezes_all_new_runtime_paths() -> None:
    required = {
        "R12_ER_FACTORIZED_WITNESS_ROUTE_PREREG.md",
        "train/er_factorized_witness_route_adapter.py",
        "train/pilot_er_factorized_witness_route_adapter.py",
        "train/pilot_er_factorized_witness_train_canary.py",
        "train/test_er_factorized_witness_route_adapter.py",
        "train/test_pilot_er_factorized_witness_train_canary.py",
        "train/jobs/er_factorized_witness_train_canary.sbatch",
    }
    assert required.issubset(FROZEN_SOURCE_PATHS)


def test_factorized_main_loads_only_training_split() -> None:
    source = inspect.getsource(main)
    assert 'filename="train.jsonl"' in source
    assert "split=TRAIN_SPLIT" in source
    assert 'filename="development.jsonl"' not in source
    assert 'filename="confirmation.jsonl"' not in source


def test_factorized_gate_uses_its_own_parameter_certificate() -> None:
    source = inspect.getsource(compute_factorized_gates)
    assert "== EXPECTED_PARAMETERS" in source
    assert 'controls["baseline"]' in source
    assert 'controls["shuffled_address"]' in source
    assert "MINIMUM_WITNESS_GAIN" in source
    assert EXPECTED_PARAMETERS["complete_system"] < 200_000_000


def test_factorized_main_fits_matched_arms_before_probe_evaluation() -> None:
    source = inspect.getsource(main)
    fit_loop = source.index("for mode in ARM_MODES:")
    checkpoint_write = source.index("atomic_torch_save(checkpoint, checkpoint_path)")
    probe_scoring = source.index("scored_probe =")
    evaluation = source.index("evaluate_arm(")
    assert fit_loop < checkpoint_write < probe_scoring < evaluation
    assert "common_initial_sha256" in source
    assert "model.set_route_mode(mode)" in source
    assert '"compiler_trainable_state": trainable_state(model)' in source
