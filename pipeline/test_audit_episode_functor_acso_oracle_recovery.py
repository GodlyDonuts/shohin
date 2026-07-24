from __future__ import annotations

import math
import os

import pytest
import torch

from pipeline.audit_episode_functor_acso_oracle_recovery import (
    ACSOOracleRecoveryError,
    CYCLES,
    DEFAULT_MARGINS,
    Fault,
    MachineTables,
    OutputReservation,
    _aggregate,
    _margin_gate,
    _permutations,
    _publish,
    _recode,
    _recoding_differences,
    _reservation_valid,
    _reserve_output,
    _targets,
    audit_fault,
    build_hankel_codebook,
    deep_fault_inventory,
    explicit_causal_adjoint,
    run_arm,
)


def _machine() -> MachineTables:
    return MachineTables(
        transitions=(
            tuple((state + 1) % 8 for state in range(8)),
            tuple((state + 2) % 8 for state in range(8)),
            (1, 0, 3, 2, 5, 4, 7, 6),
        ),
        observations=(
            (0, 0, 1, 1, 2, 2, 3, 3),
            (0, 0, 1, 1, 2, 2, 3, 3),
        ),
    )


def test_deep_fault_inventory_contains_only_one_step_ambiguities() -> None:
    machine = _machine()
    faults = deep_fault_inventory(machine)
    assert faults
    assert len(set(faults)) == len(faults)
    for fault in faults:
        correct = machine.transitions[fault.action][fault.state]
        assert tuple(
            machine.observations[observer][correct]
            for observer in range(2)
        ) == tuple(
            machine.observations[observer][fault.wrong]
            for observer in range(2)
        )
        codebook = build_hankel_codebook(machine, max_depth=3)
        assert codebook.base[correct] != codebook.base[fault.wrong]


def test_one_step_control_has_oracle_fixed_point() -> None:
    machine = _machine()
    target_base, target_derivative = _targets(machine)
    transition = torch.full((1, 3, 8, 8), -30.0)
    observer = torch.full((1, 2, 8, 4), -30.0)
    for action, row in enumerate(machine.transitions):
        for state, destination in enumerate(row):
            transition[0, action, state, destination] = 30.0
    for relation, row in enumerate(machine.observations):
        for state, answer in enumerate(row):
            observer[0, relation, state, answer] = 30.0
    result = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
        routing_mode="one-step-control",
    )
    assert abs(float(result.base_innovation)) < 1e-7
    assert abs(float(result.derivative_innovation)) < 1e-7
    assert float(result.transition_logit_adjoint.abs().max()) < 1e-7
    assert float(result.observer_logit_adjoint.abs().max()) < 1e-7


def test_singleton_fixture_records_every_cycle_and_recodes_exactly() -> None:
    machine = _machine()
    fault = deep_fault_inventory(machine)[0]
    treatment = run_arm(
        machine,
        fault,
        margin=0.1,
        routing_mode="causal",
    )
    control = run_arm(
        machine,
        fault,
        margin=0.1,
        routing_mode="one-step-control",
    )
    assert len(treatment.cycles) == CYCLES + 1
    assert len(control.cycles) == CYCLES + 1
    assert not treatment.cycles[0].exact_machine
    assert not treatment.cycles[0].fault_row_recovered
    assert all(
        math.isfinite(cycle.total_innovation)
        for cycle in (*treatment.cycles, *control.cycles)
    )
    recoded_machine, recoded_fault = _recode(
        machine,
        fault,
        _permutations("synthetic-fixture"),
    )
    recoded_treatment = run_arm(
        recoded_machine,
        recoded_fault,
        margin=0.1,
        routing_mode="causal",
    )
    recoded_control = run_arm(
        recoded_machine,
        recoded_fault,
        margin=0.1,
        routing_mode="one-step-control",
    )
    treatment_mismatches, treatment_delta = _recoding_differences(
        treatment,
        recoded_treatment,
    )
    control_mismatches, control_delta = _recoding_differences(
        control,
        recoded_control,
    )
    assert treatment_mismatches == 0
    assert control_mismatches == 0
    assert treatment_delta <= 1e-6
    assert control_delta <= 1e-6


def test_fault_dataclass_is_hashable() -> None:
    assert len({Fault(0, 1, 2), Fault(0, 1, 2)}) == 1


def test_aggregate_and_recoded_monotonicity_are_serialized() -> None:
    machine = _machine()
    fault = deep_fault_inventory(machine)[0]
    evidence = [
        audit_fault(
            "synthetic-fixture",
            machine,
            fault,
            margin=margin,
        )
        for margin in DEFAULT_MARGINS
    ]
    for row in evidence:
        assert (
            row["recoded_treatment"]["monotonic_violations"]
            >= 0
        )
    margins, global_receipt, outcome = _aggregate(
        evidence,
        {"synthetic-fixture": machine},
    )
    assert len(margins) == len(DEFAULT_MARGINS)
    assert global_receipt["fault_case_count"] == len(evidence)
    assert global_receipt["gate"] == (
        "pass" if outcome else "fail"
    )


def test_output_reservation_is_exclusive_and_publication_no_clobber(
    tmp_path,
) -> None:
    output = tmp_path / "report.json"
    reservation = _reserve_output(output)
    with pytest.raises(
        ACSOOracleRecoveryError,
        match="reservation already exists",
    ):
        _reserve_output(output)
    report = {"decision": "fixture", "payload_sha256": "0" * 64}
    _publish(reservation, report)
    assert output.is_file()
    assert reservation.lock.is_file()
    assert not reservation.temporary.exists()
    with pytest.raises(
        ACSOOracleRecoveryError,
        match="not fresh",
    ):
        _reserve_output(output)


def test_forged_reservation_is_rejected(tmp_path) -> None:
    output = tmp_path / "forged.json"
    lock = tmp_path / "forged.reserve"
    lock.write_text("{}", encoding="ascii")
    descriptor = os.open(lock, os.O_RDONLY)
    stat = os.fstat(descriptor)
    forged = OutputReservation(
        output=output,
        temporary=tmp_path / "forged.tmp",
        lock=lock,
        token="forged",
        descriptor=descriptor,
        device=stat.st_dev,
        inode=stat.st_ino,
        lock_sha256="0" * 64,
    )
    try:
        assert not _reservation_valid(forged)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("treatment_rate", 0.999999),
        ("row_rate", 0.999999),
        ("control_rate", 0.200001),
        ("monotonic_violations", 1),
        ("recoded_monotonic_violations", 1),
        ("final_ties", 1),
        ("recoding_mismatches", 1),
        ("recoding_delta", 1.000001e-6),
        ("all_worlds_pass", False),
        ("treatment_rate", float("nan")),
    ),
)
def test_margin_gate_fails_every_preregistered_boundary(
    field,
    value,
) -> None:
    arguments = {
        "treatment_rate": 1.0,
        "row_rate": 1.0,
        "control_rate": 0.0,
        "monotonic_violations": 0,
        "recoded_monotonic_violations": 0,
        "final_ties": 0,
        "recoding_mismatches": 0,
        "recoding_delta": 0.0,
        "all_worlds_pass": True,
    }
    assert _margin_gate(**arguments)
    arguments[field] = value
    assert not _margin_gate(**arguments)


def test_margin_gate_accepts_exact_inclusive_boundaries() -> None:
    assert _margin_gate(
        treatment_rate=1.0,
        row_rate=1.0,
        control_rate=0.20,
        monotonic_violations=0,
        recoded_monotonic_violations=0,
        final_ties=0,
        recoding_mismatches=0,
        recoding_delta=1e-6,
        all_worlds_pass=True,
    )
