#!/usr/bin/env python3
"""Source-bound deep-fault oracle recovery audit for ACSO mechanics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
from typing import cast

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "train"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from episode_functor_causal_syndrome_observer import (  # noqa: E402
    behavioral_closure,
    explicit_causal_adjoint,
)
from pipeline.audit_episode_functor_identifiable_board import (  # noqa: E402
    DEFAULT_COUNTS,
)
from pipeline.episode_functor_hankel_geometry import (  # noqa: E402
    enumerate_action_words,
)
from pipeline.episode_functor_hankel_shift import (  # noqa: E402
    build_hankel_codebook,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    ACTION_COUNT,
    ANSWER_COUNT,
    IdentifiableMachine,
    OBSERVER_COUNT,
    STATE_COUNT,
    generate_pilot_rows,
)


SCHEMA = "efc-acso-deep-fault-oracle-recovery/v2"
DEFAULT_SEED = "efc-identifiable-pilot-v1"
DEFAULT_MARGINS = (0.05, 0.10, 0.20)
CYCLES = 4
STEP = 0.1
MONOTONIC_TOLERANCE = 1e-7
RECODING_TOLERANCE = 1e-6
MINIMUM_RECOVERY = 1.0
MINIMUM_CONTROL_GAP = 0.80
EXPECTED_WORLDS = 200
EXPECTED_ELIGIBLE_WORLDS = 88
EXPECTED_FAULTS = 672
BOUND_FILES = (
    "R12_EFC_ACSO_ORACLE_RECOVERY_PROTOCOL.md",
    "pipeline/audit_episode_functor_identifiable_board.py",
    "pipeline/audit_episode_functor_acso_oracle_recovery.py",
    "pipeline/episode_functor_hankel_geometry.py",
    "pipeline/episode_functor_hankel_shift.py",
    "pipeline/episode_functor_identifiable_board.py",
    "train/episode_functor_causal_syndrome_observer.py",
    "train/episode_functor_constrained_transport.py",
)
EXPECTED_FAULT_DISTRIBUTION = {0: 112, 6: 64, 12: 24}


class ACSOOracleRecoveryError(ValueError):
    """The preregistered recovery audit failed closed."""


@dataclass(frozen=True, slots=True)
class MachineTables:
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Fault:
    action: int
    state: int
    wrong: int


@dataclass(frozen=True, slots=True)
class CycleEvidence:
    base_innovation: float
    derivative_innovation: float
    total_innovation: float
    exact_machine: bool
    fault_row_recovered: bool
    any_tied_row: bool


@dataclass(frozen=True, slots=True)
class ArmResult:
    cycles: tuple[CycleEvidence, ...]


@dataclass(frozen=True, slots=True)
class OutputReservation:
    output: Path
    temporary: Path
    lock: Path
    token: str
    descriptor: int
    device: int
    inode: int
    lock_sha256: str


_ACTIVE_RESERVATIONS: dict[str, OutputReservation] = {}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tables(machine: IdentifiableMachine) -> MachineTables:
    return MachineTables(
        transitions=machine.transitions,
        observations=machine.observations,
    )


def deep_fault_inventory(machine: MachineTables) -> tuple[Fault, ...]:
    codebook = build_hankel_codebook(
        cast(IdentifiableMachine, machine),
        max_depth=3,
    )
    faults = []
    for action in range(ACTION_COUNT):
        for state in range(STATE_COUNT):
            correct = machine.transitions[action][state]
            correct_observation = tuple(
                machine.observations[observer][correct]
                for observer in range(OBSERVER_COUNT)
            )
            for wrong in range(STATE_COUNT):
                if wrong == correct:
                    continue
                wrong_observation = tuple(
                    machine.observations[observer][wrong]
                    for observer in range(OBSERVER_COUNT)
                )
                if (
                    wrong_observation == correct_observation
                    and codebook.base[correct]
                    != codebook.base[wrong]
                ):
                    faults.append(Fault(action, state, wrong))
    result = tuple(faults)
    if len(set(result)) != len(result):
        raise ACSOOracleRecoveryError("deep-fault inventory is duplicated")
    return result


def _machine_logits(
    machine: MachineTables,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if margin <= 0.0:
        raise ACSOOracleRecoveryError("logit margin is not positive")
    transition = torch.full(
        (1, ACTION_COUNT, STATE_COUNT, STATE_COUNT),
        -margin,
        dtype=torch.float32,
    )
    observer = torch.full(
        (1, OBSERVER_COUNT, STATE_COUNT, ANSWER_COUNT),
        -margin,
        dtype=torch.float32,
    )
    for action, row in enumerate(machine.transitions):
        for state, destination in enumerate(row):
            transition[0, action, state, destination] = margin
    for relation, row in enumerate(machine.observations):
        for state, answer in enumerate(row):
            observer[0, relation, state, answer] = margin
    return transition, observer


def _fault_logits(
    machine: MachineTables,
    fault: Fault,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    transition, observer = _machine_logits(machine, margin)
    correct = machine.transitions[fault.action][fault.state]
    transition[0, fault.action, fault.state, correct] = -margin
    transition[0, fault.action, fault.state, fault.wrong] = margin
    return transition, observer


def _targets(
    machine: MachineTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    codebook = build_hankel_codebook(
        cast(IdentifiableMachine, machine),
        max_depth=3,
    )
    return (
        F.one_hot(
            torch.tensor(codebook.base, dtype=torch.long),
            ANSWER_COUNT,
        )[None].to(torch.float32),
        F.one_hot(
            torch.tensor(codebook.derivative, dtype=torch.long),
            ANSWER_COUNT,
        )[None].to(torch.float32),
    )


def _js_per_example(
    model: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    tiny = torch.finfo(model.dtype).tiny
    midpoint = 0.5 * (model + target)
    model_term = model * (
        model.clamp_min(tiny).log()
        - midpoint.clamp_min(tiny).log()
    )
    target_term = torch.where(
        target.gt(0),
        target
        * (
            target.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        ),
        torch.zeros_like(target),
    )
    return 0.5 * (model_term + target_term).flatten(1).sum(1) / (
        model[0].numel() // model.shape[-1]
    )


def _innovation_components(
    transition: torch.Tensor,
    observer: torch.Tensor,
    target_base: torch.Tensor,
    target_derivative: torch.Tensor,
    *,
    routing_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    closure = behavioral_closure(
        transition,
        observer,
        max_depth=3,
    )
    if routing_mode == "causal":
        base = closure.base
        derivative = closure.derivative
        base_target = target_base
        derivative_target = target_derivative
    elif routing_mode == "one-step-control":
        words = enumerate_action_words(3)
        base_indices = torch.tensor(
            tuple(
                index
                for index, word in enumerate(words)
                if len(word) <= 1
            ),
            dtype=torch.long,
        )
        derivative_indices = torch.tensor(
            tuple(
                index
                for index, word in enumerate(words)
                if len(word) == 0
            ),
            dtype=torch.long,
        )
        base = closure.base.index_select(2, base_indices)
        derivative = closure.derivative.index_select(
            3,
            derivative_indices,
        )
        base_target = target_base.index_select(2, base_indices)
        derivative_target = target_derivative.index_select(
            3,
            derivative_indices,
        )
    else:
        raise ACSOOracleRecoveryError("audit routing mode differs")
    return (
        _js_per_example(base, base_target),
        _js_per_example(derivative, derivative_target),
    )


def _row_normalized(adjoint: torch.Tensor) -> torch.Tensor:
    scale = adjoint.abs().amax(-1, keepdim=True)
    return torch.where(
        scale.gt(0),
        adjoint / scale.clamp_min(torch.finfo(adjoint.dtype).tiny),
        torch.zeros_like(adjoint),
    )


def _hard_evidence(
    transition: torch.Tensor,
    observer: torch.Tensor,
    machine: MachineTables,
    fault: Fault,
) -> tuple[bool, bool, bool]:
    if transition.shape[0] != 1 or observer.shape[0] != 1:
        raise ACSOOracleRecoveryError(
            "official audit requires batch size one"
        )
    transition_top = transition.topk(2, dim=-1)
    observer_top = observer.topk(2, dim=-1)
    transition_tied = transition_top.values[..., 0].eq(
        transition_top.values[..., 1]
    )
    observer_tied = observer_top.values[..., 0].eq(
        observer_top.values[..., 1]
    )
    transition_hard = transition_top.indices[..., 0]
    observer_hard = observer_top.indices[..., 0]
    expected_transition = torch.tensor(
        machine.transitions,
        dtype=torch.long,
    )[None]
    expected_observer = torch.tensor(
        machine.observations,
        dtype=torch.long,
    )[None]
    any_tie = bool(transition_tied.any() or observer_tied.any())
    exact = bool(
        transition_hard.eq(expected_transition).all()
        and observer_hard.eq(expected_observer).all()
        and not any_tie
    )
    row_recovered = (
        int(transition_hard[0, fault.action, fault.state])
        == machine.transitions[fault.action][fault.state]
        and not bool(transition_tied[0, fault.action, fault.state])
    )
    return exact, row_recovered, any_tie


def run_arm(
    machine: MachineTables,
    fault: Fault,
    *,
    margin: float,
    routing_mode: str,
) -> ArmResult:
    transition, observer = _fault_logits(machine, fault, margin)
    target_base, target_derivative = _targets(machine)
    initial_exact, initial_row, initial_tie = _hard_evidence(
        transition,
        observer,
        machine,
        fault,
    )
    if initial_exact or initial_row or initial_tie:
        raise ACSOOracleRecoveryError(
            "initial fault is not one wrong untied row"
        )
    evidence = []
    for cycle in range(CYCLES + 1):
        base, derivative = _innovation_components(
            transition,
            observer,
            target_base,
            target_derivative,
            routing_mode=routing_mode,
        )
        exact, row, tied = _hard_evidence(
            transition,
            observer,
            machine,
            fault,
        )
        base_value = float(base[0])
        derivative_value = float(derivative[0])
        values = (base_value, derivative_value)
        if not all(math.isfinite(value) for value in values):
            raise ACSOOracleRecoveryError(
                "audit innovation is nonfinite"
            )
        evidence.append(
            CycleEvidence(
                base_innovation=base_value,
                derivative_innovation=derivative_value,
                total_innovation=base_value + derivative_value,
                exact_machine=exact,
                fault_row_recovered=row,
                any_tied_row=tied,
            )
        )
        if cycle == CYCLES:
            break
        adjoint = explicit_causal_adjoint(
            transition,
            observer,
            target_base,
            target_derivative,
            max_depth=3,
            routing_mode=routing_mode,
        )
        transition = transition - STEP * _row_normalized(
            adjoint.transition_logit_adjoint
        )
        observer = observer - STEP * _row_normalized(
            adjoint.observer_logit_adjoint
        )
    return ArmResult(cycles=tuple(evidence))


def _permutations(world_id: str) -> tuple[tuple[int, ...], ...]:
    generator = random.Random(
        int.from_bytes(
            sha256(f"acso-v2-recoding:{world_id}".encode("ascii")).digest(),
            "big",
        )
    )
    result = []
    for size in (
        STATE_COUNT,
        ACTION_COUNT,
        OBSERVER_COUNT,
        ANSWER_COUNT,
    ):
        permutation = list(range(size))
        generator.shuffle(permutation)
        if permutation == list(range(size)):
            permutation = permutation[1:] + permutation[:1]
        result.append(tuple(permutation))
    return tuple(result)


def _recode(
    machine: MachineTables,
    fault: Fault,
    permutations: tuple[tuple[int, ...], ...],
) -> tuple[MachineTables, Fault]:
    state, action, observer, answer = permutations
    transitions = [
        [0 for _ in range(STATE_COUNT)]
        for _ in range(ACTION_COUNT)
    ]
    observations = [
        [0 for _ in range(STATE_COUNT)]
        for _ in range(OBSERVER_COUNT)
    ]
    for old_action in range(ACTION_COUNT):
        for old_state in range(STATE_COUNT):
            transitions[action[old_action]][state[old_state]] = state[
                machine.transitions[old_action][old_state]
            ]
    for old_observer in range(OBSERVER_COUNT):
        for old_state in range(STATE_COUNT):
            observations[observer[old_observer]][state[old_state]] = answer[
                machine.observations[old_observer][old_state]
            ]
    return (
        MachineTables(
            transitions=tuple(tuple(row) for row in transitions),
            observations=tuple(tuple(row) for row in observations),
        ),
        Fault(
            action=action[fault.action],
            state=state[fault.state],
            wrong=state[fault.wrong],
        ),
    )


def _recoding_differences(
    original: ArmResult,
    recoded: ArmResult,
) -> tuple[int, float]:
    mismatches = 0
    maximum_delta = 0.0
    for left, right in zip(
        original.cycles,
        recoded.cycles,
        strict=True,
    ):
        mismatches += int(
            (
                left.exact_machine,
                left.fault_row_recovered,
                left.any_tied_row,
            )
            != (
                right.exact_machine,
                right.fault_row_recovered,
                right.any_tied_row,
            )
        )
        maximum_delta = max(
            maximum_delta,
            abs(left.base_innovation - right.base_innovation),
            abs(
                left.derivative_innovation
                - right.derivative_innovation
            ),
            abs(left.total_innovation - right.total_innovation),
        )
    return mismatches, maximum_delta


def _monotonic_violations(result: ArmResult) -> int:
    return sum(
        right.total_innovation
        > left.total_innovation + MONOTONIC_TOLERANCE
        for left, right in zip(
            result.cycles,
            result.cycles[1:],
        )
    )


def audit_fault(
    world_id: str,
    machine: MachineTables,
    fault: Fault,
    *,
    margin: float,
) -> dict[str, object]:
    treatment = run_arm(
        machine,
        fault,
        margin=margin,
        routing_mode="causal",
    )
    control = run_arm(
        machine,
        fault,
        margin=margin,
        routing_mode="one-step-control",
    )
    recoded_machine, recoded_fault = _recode(
        machine,
        fault,
        _permutations(world_id),
    )
    recoded_treatment = run_arm(
        recoded_machine,
        recoded_fault,
        margin=margin,
        routing_mode="causal",
    )
    recoded_control = run_arm(
        recoded_machine,
        recoded_fault,
        margin=margin,
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
    return {
        "world_id": world_id,
        "margin": margin,
        "fault": asdict(fault),
        "treatment": {
            "cycles": [
                asdict(cycle) for cycle in treatment.cycles
            ],
            "monotonic_violations": _monotonic_violations(
                treatment
            ),
        },
        "control": {
            "cycles": [
                asdict(cycle) for cycle in control.cycles
            ],
            "monotonic_violations": _monotonic_violations(control),
        },
        "recoded_treatment": {
            "cycles": [
                asdict(cycle)
                for cycle in recoded_treatment.cycles
            ],
            "monotonic_violations": _monotonic_violations(
                recoded_treatment
            ),
        },
        "recoded_control": {
            "cycles": [
                asdict(cycle)
                for cycle in recoded_control.cycles
            ],
            "monotonic_violations": _monotonic_violations(
                recoded_control
            ),
        },
        "recoding_decision_mismatches": (
            treatment_mismatches + control_mismatches
        ),
        "maximum_recoding_innovation_delta": max(
            treatment_delta,
            control_delta,
        ),
    }


def _board() -> dict[str, MachineTables]:
    rows = generate_pilot_rows(
        seed=DEFAULT_SEED,
        counts=DEFAULT_COUNTS,
    )
    return {
        row.world_id: _tables(row.machine)
        for row in rows
    }


def _board_manifest(
    machines: dict[str, MachineTables],
) -> tuple[str, list[dict[str, object]]]:
    rows = [
        {
            "world_id": world_id,
            "transitions": machines[world_id].transitions,
            "observations": machines[world_id].observations,
            "eligible_faults": len(
                deep_fault_inventory(machines[world_id])
            ),
        }
        for world_id in sorted(machines)
    ]
    return sha256(_canonical_json_bytes(rows)).hexdigest(), rows


def _source_receipt() -> tuple[str, list[dict[str, object]], bool]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
    ).strip()
    rows = []
    matched = True
    for relative in BOUND_FILES:
        path = ROOT / relative
        local_sha = _sha256_file(path)
        try:
            committed = subprocess.check_output(
                ("git", "show", f"HEAD:{relative}"),
                cwd=ROOT,
            )
        except subprocess.CalledProcessError:
            committed = b""
        commit_sha = sha256(committed).hexdigest()
        row_match = local_sha == commit_sha
        matched = matched and row_match
        rows.append(
            {
                "path": relative,
                "sha256": local_sha,
                "head_sha256": commit_sha,
                "matches_head": row_match,
            }
        )
    return commit, rows, matched


def _margin_gate(
    *,
    treatment_rate: float,
    row_rate: float,
    control_rate: float,
    monotonic_violations: int,
    recoded_monotonic_violations: int,
    final_ties: int,
    recoding_mismatches: int,
    recoding_delta: float,
    all_worlds_pass: bool,
) -> bool:
    numeric = (
        treatment_rate,
        row_rate,
        control_rate,
        recoding_delta,
    )
    return bool(
        all(math.isfinite(value) for value in numeric)
        and treatment_rate >= MINIMUM_RECOVERY
        and row_rate >= MINIMUM_RECOVERY
        and treatment_rate - control_rate
        >= MINIMUM_CONTROL_GAP
        and monotonic_violations == 0
        and recoded_monotonic_violations == 0
        and final_ties == 0
        and recoding_mismatches == 0
        and recoding_delta <= RECODING_TOLERANCE
        and all_worlds_pass
    )


def _aggregate(
    evidence: list[dict[str, object]],
    machines: dict[str, MachineTables],
) -> tuple[list[dict[str, object]], dict[str, object], bool]:
    margin_receipts = []
    all_go = True
    represented_worlds = {
        world_id
        for world_id, machine in machines.items()
        if deep_fault_inventory(machine)
    }
    for margin in DEFAULT_MARGINS:
        selected = [
            row for row in evidence if row["margin"] == margin
        ]
        treatment_exact = sum(
            bool(row["treatment"]["cycles"][-1]["exact_machine"])
            for row in selected
        )
        treatment_row = sum(
            bool(
                row["treatment"]["cycles"][-1][
                    "fault_row_recovered"
                ]
            )
            for row in selected
        )
        control_exact = sum(
            bool(row["control"]["cycles"][-1]["exact_machine"])
            for row in selected
        )
        monotonic = sum(
            int(row["treatment"]["monotonic_violations"])
            for row in selected
        )
        recoded_monotonic = sum(
            int(
                row["recoded_treatment"][
                    "monotonic_violations"
                ]
            )
            for row in selected
        )
        ties = sum(
            bool(row["treatment"]["cycles"][-1]["any_tied_row"])
            for row in selected
        )
        recoding_mismatches = sum(
            int(row["recoding_decision_mismatches"])
            for row in selected
        )
        recoding_delta = max(
            float(row["maximum_recoding_innovation_delta"])
            for row in selected
        )
        count = len(selected)
        world_gates = []
        for world_id in sorted(represented_worlds):
            world_rows = [
                row
                for row in selected
                if row["world_id"] == world_id
            ]
            world_count = len(world_rows)
            world_treatment = sum(
                bool(
                    row["treatment"]["cycles"][-1][
                        "exact_machine"
                    ]
                )
                for row in world_rows
            )
            world_control = sum(
                bool(
                    row["control"]["cycles"][-1][
                        "exact_machine"
                    ]
                )
                for row in world_rows
            )
            world_go = (
                world_treatment == world_count
                and (
                    world_treatment / world_count
                    - world_control / world_count
                )
                >= MINIMUM_CONTROL_GAP
            )
            world_gates.append(
                {
                    "world_id": world_id,
                    "fault_count": world_count,
                    "treatment_exact_recovered": world_treatment,
                    "control_exact_recovered": world_control,
                    "gate": "pass" if world_go else "fail",
                }
            )
        treatment_rate = treatment_exact / count
        row_rate = treatment_row / count
        control_rate = control_exact / count
        margin_go = _margin_gate(
            treatment_rate=treatment_rate,
            row_rate=row_rate,
            control_rate=control_rate,
            monotonic_violations=monotonic,
            recoded_monotonic_violations=recoded_monotonic,
            final_ties=ties,
            recoding_mismatches=recoding_mismatches,
            recoding_delta=recoding_delta,
            all_worlds_pass=all(
                row["gate"] == "pass" for row in world_gates
            ),
        )
        all_go = all_go and margin_go
        margin_receipts.append(
            {
                "margin": margin,
                "fault_count": count,
                "treatment_exact_recovery": treatment_rate,
                "treatment_row_recovery": row_rate,
                "control_exact_recovery": control_rate,
                "treatment_control_exact_gap": (
                    treatment_rate - control_rate
                ),
                "treatment_monotonic_violations": monotonic,
                "recoded_treatment_monotonic_violations": (
                    recoded_monotonic
                ),
                "treatment_final_ties": ties,
                "recoding_decision_mismatches": recoding_mismatches,
                "maximum_recoding_innovation_delta": recoding_delta,
                "world_gates": world_gates,
                "gate": "pass" if margin_go else "fail",
            }
        )
    total_faults = len(evidence)
    global_treatment = sum(
        bool(row["treatment"]["cycles"][-1]["exact_machine"])
        for row in evidence
    )
    global_control = sum(
        bool(row["control"]["cycles"][-1]["exact_machine"])
        for row in evidence
    )
    global_receipt = {
        "fault_case_count": total_faults,
        "treatment_exact_recovery": global_treatment / total_faults,
        "control_exact_recovery": global_control / total_faults,
        "treatment_control_exact_gap": (
            global_treatment / total_faults
            - global_control / total_faults
        ),
        "gate": "pass" if all_go else "fail",
    }
    return margin_receipts, global_receipt, all_go


def _reserve_output(output: Path) -> OutputReservation:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    lock = output.with_suffix(output.suffix + ".reserve")
    if output.exists() or temporary.exists():
        raise ACSOOracleRecoveryError("output path is not fresh")
    token = secrets.token_hex(32)
    payload = _canonical_json_bytes(
        {
            "schema": "efc-acso-output-reservation/v1",
            "output": str(output),
            "token": token,
        }
    )
    try:
        descriptor = os.open(
            lock,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ACSOOracleRecoveryError(
            "output reservation already exists"
        ) from exc
    os.write(descriptor, payload)
    os.fsync(descriptor)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if output.exists() or temporary.exists():
        os.close(descriptor)
        lock.unlink()
        raise ACSOOracleRecoveryError(
            "output appeared during reservation"
        )
    stat = os.fstat(descriptor)
    reservation = OutputReservation(
        output=output,
        temporary=temporary,
        lock=lock,
        token=token,
        descriptor=descriptor,
        device=stat.st_dev,
        inode=stat.st_ino,
        lock_sha256=_sha256_file(lock),
    )
    _ACTIVE_RESERVATIONS[token] = reservation
    return reservation


def _reservation_valid(
    reservation: OutputReservation | None,
    *,
    allow_temporary: bool = False,
) -> bool:
    if (
        reservation is None
        or _ACTIVE_RESERVATIONS.get(reservation.token)
        is not reservation
        or not reservation.lock.is_file()
        or reservation.output.exists()
        or (
            not allow_temporary
            and reservation.temporary.exists()
        )
    ):
        return False
    try:
        descriptor_stat = os.fstat(reservation.descriptor)
        path_stat = reservation.lock.stat()
        payload = json.loads(reservation.lock.read_text("ascii"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        descriptor_stat.st_dev == reservation.device
        and descriptor_stat.st_ino == reservation.inode
        and path_stat.st_dev == reservation.device
        and path_stat.st_ino == reservation.inode
        and payload
        == {
            "schema": "efc-acso-output-reservation/v1",
            "output": str(reservation.output),
            "token": reservation.token,
        }
        and _sha256_file(reservation.lock)
        == reservation.lock_sha256
    )


def _publish(
    reservation: OutputReservation,
    report: dict[str, object],
) -> None:
    if not _reservation_valid(reservation):
        raise ACSOOracleRecoveryError(
            "output reservation is no longer valid"
        )
    payload = _canonical_json_bytes(report) + b"\n"
    descriptor = os.open(
        reservation.temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not _reservation_valid(
            reservation,
            allow_temporary=True,
        ):
            raise ACSOOracleRecoveryError(
                "output reservation changed before publication"
            )
        os.link(reservation.temporary, reservation.output)
        directory = os.open(reservation.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        reservation.temporary.unlink()
        directory = os.open(reservation.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.close(reservation.descriptor)
        _ACTIVE_RESERVATIONS.pop(reservation.token, None)
    except Exception:
        raise


def audit_board(
    *,
    reservation: OutputReservation,
) -> dict[str, object]:
    if not _reservation_valid(reservation):
        raise ACSOOracleRecoveryError(
            "official audit lacks a valid output reservation"
        )
    machines = _board()
    eligible = {
        world_id: deep_fault_inventory(machine)
        for world_id, machine in machines.items()
    }
    eligible_worlds = sum(bool(faults) for faults in eligible.values())
    fault_count = sum(len(faults) for faults in eligible.values())
    fault_distribution = {
        count: sum(
            len(faults) == count for faults in eligible.values()
        )
        for count in sorted({len(faults) for faults in eligible.values()})
    }
    board_sha, board_rows = _board_manifest(machines)
    commit, source_rows, source_match = _source_receipt()
    counts_match = (
        dict(DEFAULT_COUNTS)
        == {
            "confirmation": 24,
            "development": 32,
            "mechanics": 48,
            "train": 96,
        }
        and len(machines) == EXPECTED_WORLDS
        and eligible_worlds == EXPECTED_ELIGIBLE_WORLDS
        and fault_count == EXPECTED_FAULTS
        and fault_distribution == EXPECTED_FAULT_DISTRIBUTION
    )
    if not source_match:
        raise ACSOOracleRecoveryError(
            "bound source differs before outcome evaluation"
        )
    if not counts_match:
        raise ACSOOracleRecoveryError(
            "board inventory differs before outcome evaluation"
        )
    expected_evidence_keys = {
        (
            margin,
            world_id,
            fault.action,
            fault.state,
            fault.wrong,
        )
        for margin in DEFAULT_MARGINS
        for world_id in sorted(machines)
        for fault in eligible[world_id]
    }
    evidence = [
        audit_fault(
            world_id,
            machines[world_id],
            fault,
            margin=margin,
        )
        for margin in DEFAULT_MARGINS
        for world_id in sorted(machines)
        for fault in eligible[world_id]
    ]
    actual_evidence_keys = {
        (
            float(row["margin"]),
            str(row["world_id"]),
            int(row["fault"]["action"]),
            int(row["fault"]["state"]),
            int(row["fault"]["wrong"]),
        )
        for row in evidence
    }
    evidence_binding_pass = (
        len(evidence) == len(expected_evidence_keys)
        and len(actual_evidence_keys) == len(evidence)
        and actual_evidence_keys == expected_evidence_keys
        and len(evidence)
        == EXPECTED_FAULTS * len(DEFAULT_MARGINS)
    )
    if not evidence_binding_pass:
        raise ACSOOracleRecoveryError(
            "evaluated evidence inventory differs"
        )
    margin_receipts, global_receipt, outcome_go = _aggregate(
        evidence,
        machines,
    )
    custody_pass = _reservation_valid(reservation)
    bindings_go = (
        source_match
        and counts_match
        and evidence_binding_pass
        and custody_pass
    )
    evidence_sha = sha256(_canonical_json_bytes(evidence)).hexdigest()
    report = {
        "schema": SCHEMA,
        "git_commit": commit,
        "seed": DEFAULT_SEED,
        "counts": DEFAULT_COUNTS,
        "world_count": len(machines),
        "eligible_world_count": eligible_worlds,
        "fault_count": fault_count,
        "fault_distribution": fault_distribution,
        "margins": list(DEFAULT_MARGINS),
        "cycles": CYCLES,
        "step": STEP,
        "primary_cases": len(evidence),
        "recoded_cases": len(evidence),
        "thresholds": {
            "monotonic_tolerance": MONOTONIC_TOLERANCE,
            "recoding_tolerance": RECODING_TOLERANCE,
            "minimum_recovery": MINIMUM_RECOVERY,
            "minimum_control_gap": MINIMUM_CONTROL_GAP,
        },
        "source_receipt": source_rows,
        "source_binding_pass": source_match,
        "board_manifest_sha256": board_sha,
        "board_manifest": board_rows,
        "count_binding_pass": counts_match,
        "output_custody": {
            "output": str(reservation.output),
            "reservation": str(reservation.lock),
            "reservation_sha256": reservation.lock_sha256,
            "reservation_device": reservation.device,
            "reservation_inode": reservation.inode,
            "pass": custody_pass,
        },
        "fault_evidence_sha256": evidence_sha,
        "evidence_binding_pass": evidence_binding_pass,
        "fault_evidence": evidence,
        "margin_receipts": margin_receipts,
        "global_receipt": global_receipt,
        "decision": (
            "deep_fault_oracle_go"
            if outcome_go and bindings_go
            else "deep_fault_oracle_no_go"
        ),
        "claim_boundary": (
            "oracle_targeted_synthetic_deep_fault_recovery_only;"
            "no_hsc_fit_no_source_compilation_no_transfer_no_reasoning"
        ),
    }
    report["payload_sha256"] = sha256(
        _canonical_json_bytes(report)
    ).hexdigest()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reservation = _reserve_output(args.output)
    report = audit_board(reservation=reservation)
    _publish(reservation, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["decision"],
                "payload_sha256": report["payload_sha256"],
                "fault_evidence_sha256": report[
                    "fault_evidence_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
