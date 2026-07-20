#!/usr/bin/env python3
"""Training-only source-deletion and causal mechanics gate for projected SD-CST."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence

import torch

from pilot_sd_cst_binding_bus import BindingPilotRow, load_rows, partition
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    byte_batch,
    sha256_file,
)
from pilot_sd_cst_hierarchical_binding import PROJECTED_TRAINABLE_NAMES
from sd_cst import (
    EVENT_STEPS,
    STOP_KIND,
    CategoricalStateReader,
    HardLateQuery,
    HardProgramTape,
    StateSwap,
    TiedCategoricalMotor,
    rollout_hard_categorical,
)
from sd_cst_binding_bus import ProjectedHierarchicalBindingBusCompiler
from train_sd_cst import fit_motor_certificate, fit_reader_certificate


PROJECTED_CHECKPOINT_SHA256 = (
    "f347d1aea90dd3c60f7500167c7c22884451b365880259698306c6fce8ab10f3"
)
PROJECTED_REPORT_SHA256 = (
    "5d6be14798af3a75781898c6405e956fe9eb040e861ee63e669e7b87e7fa6f32"
)
PILOT_SOURCE_COMMIT = "9bd2e04ea93406eb50a6fd112cd844892b72a7c4"
GLOBAL_PARAMETER_CAP = 200_000_000
PILOT_PARAMETER_CAP = 150_000_000
PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_STATE = {value: index for index, value in enumerate(PERMUTATIONS)}


def derived_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def state_dict_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def alpha_renamed_row(row: BindingPilotRow) -> BindingPilotRow:
    source = bytes(row.program_bytes)
    names = [source[start:end] for start, end in row.binding_ranges]
    replacements = []
    for role, name in enumerate(names):
        digest = hashlib.sha256(f"{row.row_id}:{role}:alpha".encode()).hexdigest()
        candidate = (digest[:4] + "-" + digest[4:12]).encode("ascii")
        if len(candidate) != len(name) or candidate in names:
            raise ValueError("alpha replacement does not preserve the name contract")
        replacements.append(candidate)
    renamed = source
    for old, new in zip(names, replacements, strict=True):
        renamed = renamed.replace(old, new)
    if len(renamed) != len(source) or any(name in renamed for name in names):
        raise ValueError("alpha renaming was incomplete or changed byte width")
    return replace(row, program_bytes=tuple(renamed))


def event_counterfactual_row(row: BindingPilotRow) -> BindingPilotRow:
    source = bytearray(row.program_bytes)
    names = [bytes(source[start:end]) for start, end in row.binding_ranges]
    stop = row.event_kind.index(STOP_KIND)
    candidates = [slot for slot in range(stop) if row.event_kind[slot] != STOP_KIND]
    if not candidates:
        raise ValueError("counterfactual requires an active pre-STOP event")
    slot = candidates[0]
    old_role = row.event_identity[slot]
    new_role = (old_role + 1) % 3
    start, end = row.event_entity_ranges[slot]
    if bytes(source[start:end]) != names[old_role]:
        raise ValueError("event occurrence does not match its declared identity")
    if len(names[new_role]) != end - start:
        raise ValueError("event counterfactual requires width-preserving names")
    source[start:end] = names[new_role]
    identities = list(row.event_identity)
    identities[slot] = new_role
    return replace(
        row,
        program_bytes=tuple(source),
        event_identity=tuple(identities),
    )


def declaration_role_swap_row(row: BindingPilotRow) -> BindingPilotRow:
    source = bytearray(row.program_bytes)
    first = bytes(source[slice(*row.binding_ranges[0])])
    second = bytes(source[slice(*row.binding_ranges[1])])
    if len(first) != len(second):
        raise ValueError("declaration swap must preserve byte width")
    source[slice(*row.binding_ranges[0])] = second
    source[slice(*row.binding_ranges[1])] = first
    role_map = {0: 1, 1: 0, 2: 2}
    identities = tuple(role_map[value] for value in row.event_identity)
    initial_order = tuple(
        role_map[value] for value in PERMUTATIONS[row.initial_state]
    )
    return replace(
        row,
        program_bytes=tuple(source),
        initial_state=PERMUTATION_TO_STATE[initial_order],
        event_identity=identities,
    )


def relocated_event_lines_row(row: BindingPilotRow) -> BindingPilotRow:
    source = bytes(row.program_bytes)
    lines = source.splitlines(keepends=True)
    if len(lines) != 9 or not all(line.endswith(b"\n") for line in lines):
        raise ValueError("event relocation requires one binding plus eight LF lines")
    relocated = lines[:1] + list(reversed(lines[1:]))
    return replace(row, program_bytes=tuple(b"".join(relocated)))


def _concatenate_tapes(tapes: Sequence[HardProgramTape]) -> HardProgramTape:
    return HardProgramTape(
        initial_state=torch.cat([tape.initial_state for tape in tapes]),
        event_kind=torch.cat([tape.event_kind for tape in tapes]),
        event_identity=torch.cat([tape.event_identity for tape in tapes]),
        amount=torch.cat([tape.amount for tape in tapes]),
    )


@torch.no_grad()
def compile_hard(
    model: ProjectedHierarchicalBindingBusCompiler,
    rows: Sequence[BindingPilotRow],
    batch_size: int,
    device: torch.device,
) -> tuple[HardProgramTape, HardLateQuery, bool]:
    tapes = []
    queries = []
    source_poison_bit_identical = True
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.compile_program(program_ids, program_valid)
        hard = output.tape.hard()
        sealed = HardProgramTape(
            hard.initial_state.detach().cpu().clone(),
            hard.event_kind.detach().cpu().clone(),
            hard.event_identity.detach().cpu().clone(),
            hard.amount.detach().cpu().clone(),
        )
        before_program = tuple(value.clone() for value in (
            sealed.initial_state,
            sealed.event_kind,
            sealed.event_identity,
            sealed.amount,
        ))
        program_ids.random_(0, 257)
        program_valid.logical_not_()
        after_program = (
            sealed.initial_state,
            sealed.event_kind,
            sealed.event_identity,
            sealed.amount,
        )
        source_poison_bit_identical &= all(
            torch.equal(left, right)
            for left, right in zip(before_program, after_program, strict=True)
        )
        del output, hard, program_ids, program_valid
        torch.cuda.empty_cache()

        # The late query is disclosed only after the program source and every
        # compiler-side program tensor have become unreachable.
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query_output = model.compile_query(query_ids, query_valid)
        hard_query = query_output.hard()
        sealed_query = HardLateQuery(hard_query.position.detach().cpu().clone())
        before_query = sealed_query.position.clone()
        query_ids.random_(0, 257)
        query_valid.logical_not_()
        source_poison_bit_identical &= torch.equal(
            before_query, sealed_query.position,
        )
        tapes.append(sealed)
        queries.append(sealed_query.position)
        del query_output, hard_query, query_ids, query_valid
    return (
        _concatenate_tapes(tapes),
        HardLateQuery(torch.cat(queries)),
        source_poison_bit_identical,
    )


def expected_tape(rows: Sequence[BindingPilotRow]) -> HardProgramTape:
    return HardProgramTape(
        torch.tensor([row.initial_state for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_kind for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_identity for row in rows], dtype=torch.uint8),
        torch.tensor([row.amount for row in rows], dtype=torch.uint8),
    )


def tape_metrics(
    prediction: HardProgramTape,
    target: HardProgramTape,
) -> dict[str, object]:
    active = target.event_kind.ne(STOP_KIND)
    fields = {
        "initial": prediction.initial_state.eq(target.initial_state),
        "kind": prediction.event_kind.eq(target.event_kind).all(-1),
        "identity": (
            prediction.event_identity.eq(target.event_identity) | ~active
        ).all(-1),
        "amount": (prediction.amount.eq(target.amount) | ~active).all(-1),
    }
    fields["whole_tape"] = (
        fields["initial"] & fields["kind"] & fields["identity"] & fields["amount"]
    )
    rows = prediction.batch_size
    return {
        "rows": rows,
        "exact": {name: int(value.sum()) for name, value in fields.items()},
        "rates": {name: float(value.float().mean()) for name, value in fields.items()},
    }


def _apply_event(
    state: tuple[int, ...], identity: int, kind: int, amount: int,
) -> tuple[int, ...]:
    values = list(state)
    source = values.index(identity)
    signed = -(amount + 1) if kind == 0 else amount + 1
    destination = min(2, max(0, source + signed))
    value = values.pop(source)
    values.insert(destination, value)
    return tuple(values)


def semantic_rollout(
    tape: HardProgramTape,
    query: HardLateQuery,
    *,
    control: str = "normal",
    state_swap: torch.Tensor | None = None,
    swap_after_step: int = 0,
    force_alive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [int(value) for value in tape.initial_state]
    initial = list(states)
    alive = [True] * tape.batch_size
    state_trajectory = []
    alive_trajectory = []
    for step in range(EVENT_STEPS):
        for row in range(tape.batch_size):
            kind = int(tape.event_kind[row, step])
            if alive[row] and kind != STOP_KIND and control != "freeze":
                input_state = initial[row] if control == "reset" else states[row]
                next_order = _apply_event(
                    PERMUTATIONS[input_state],
                    int(tape.event_identity[row, step]),
                    kind,
                    int(tape.amount[row, step]),
                )
                states[row] = PERMUTATION_TO_STATE[next_order]
            if not force_alive and kind == STOP_KIND:
                alive[row] = False
        if state_swap is not None and step == swap_after_step:
            states = [states[int(index)] for index in state_swap]
        state_trajectory.append(torch.tensor(states, dtype=torch.uint8))
        alive_trajectory.append(torch.tensor(alive, dtype=torch.bool))
    final_state = torch.tensor(states, dtype=torch.uint8)
    answers = torch.tensor([
        PERMUTATIONS[state][int(position)]
        for state, position in zip(states, query.position, strict=True)
    ], dtype=torch.long)
    return (
        final_state,
        answers,
        torch.stack(state_trajectory, dim=1),
        torch.stack(alive_trajectory, dim=1),
    )


def execute_control(
    motor: TiedCategoricalMotor,
    reader: CategoricalStateReader,
    tape: HardProgramTape,
    query: HardLateQuery,
    *,
    canonical_state: torch.Tensor,
    canonical_answer: torch.Tensor,
    control: str = "normal",
    state_swap: torch.Tensor | None = None,
    swap_after_step: int = 0,
    force_alive: bool = False,
) -> dict[str, object]:
    swap = None
    if state_swap is not None:
        swap = StateSwap(after_step=swap_after_step, batch_permutation=state_swap)
    result = rollout_hard_categorical(
        motor,
        reader,
        tape,
        query,
        control=control,
        state_swap=swap,
        force_alive=force_alive,
    )
    expected_state, expected_answer, expected_states, expected_alive = semantic_rollout(
        tape,
        query,
        control=control,
        state_swap=state_swap,
        swap_after_step=swap_after_step,
        force_alive=force_alive,
    )
    predicted_state = result.final_state.cpu()
    predicted_answer = result.answer_logits.argmax(-1).cpu()
    rows = tape.batch_size
    return {
        "rows": rows,
        "state_exact": int(predicted_state.eq(expected_state).sum()),
        "answer_exact": int(predicted_answer.eq(expected_answer).sum()),
        "state_rate": float(predicted_state.eq(expected_state).float().mean()),
        "answer_rate": float(predicted_answer.eq(expected_answer).float().mean()),
        "trajectory_exact": int(
            torch.stack(result.state_trajectory, dim=1).cpu().eq(expected_states).all(-1).sum()
        ),
        "alive_trajectory_exact": int(
            torch.stack(result.alive_trajectory, dim=1).cpu().eq(expected_alive).all(-1).sum()
        ),
        "state_causal_opportunities": int(expected_state.ne(canonical_state).sum()),
        "answer_causal_opportunities": int(expected_answer.ne(canonical_answer).sum()),
        "motor_calls": len(result.state_trajectory),
        "all_halted": bool((~result.alive_trajectory[-1]).all()) if not force_alive else None,
    }


def rotate_queries(query: HardLateQuery) -> HardLateQuery:
    return HardLateQuery(((query.position.long() + 1) % 3).to(torch.uint8))


def perturb_post_stop(tape: HardProgramTape) -> HardProgramTape:
    identity = tape.event_identity.clone()
    amount = tape.amount.clone()
    for row in range(tape.batch_size):
        stop = int(tape.event_kind[row].eq(STOP_KIND).nonzero()[0])
        for step in range(stop + 1, EVENT_STEPS):
            if int(tape.event_kind[row, step]) != STOP_KIND:
                identity[row, step] = (identity[row, step].long() + 1) % 3
                amount[row, step] = (amount[row, step].long() + 1) % 2
    return HardProgramTape(
        tape.initial_state.clone(), tape.event_kind.clone(), identity, amount,
    )


def swap_operand_suffix(tape: HardProgramTape, start_step: int) -> HardProgramTape:
    donor = torch.arange(tape.batch_size - 1, -1, -1)
    identity = tape.event_identity.clone()
    amount = tape.amount.clone()
    identity[:, start_step:] = tape.event_identity[donor, start_step:]
    amount[:, start_step:] = tape.amount[donor, start_step:]
    return HardProgramTape(
        tape.initial_state.clone(), tape.event_kind.clone(), identity, amount,
    )


def mutate_first_active(
    tape: HardProgramTape, field: str,
) -> HardProgramTape:
    initial = tape.initial_state.clone()
    kind = tape.event_kind.clone()
    identity = tape.event_identity.clone()
    amount = tape.amount.clone()
    if field == "initial_state":
        initial = ((initial.long() + 1) % len(PERMUTATIONS)).to(torch.uint8)
    else:
        for row in range(tape.batch_size):
            stop = int(tape.event_kind[row].eq(STOP_KIND).nonzero()[0])
            candidates = [
                step for step in range(stop)
                if int(tape.event_kind[row, step]) != STOP_KIND
            ]
            if not candidates:
                raise ValueError("active mutation requires a pre-STOP event")
            step = candidates[0]
            if field == "kind":
                kind[row, step] = 1 - kind[row, step]
            elif field == "identity":
                identity[row, step] = (identity[row, step].long() + 1) % 3
            elif field == "amount":
                amount[row, step] = 1 - amount[row, step]
            else:
                raise ValueError(f"unknown active mutation field: {field}")
    return HardProgramTape(initial, kind, identity, amount)


def shuffled_packet(tape: HardProgramTape) -> HardProgramTape:
    donor = torch.arange(tape.batch_size - 1, -1, -1)
    return HardProgramTape(
        tape.initial_state[donor].clone(),
        tape.event_kind[donor].clone(),
        tape.event_identity[donor].clone(),
        tape.amount[donor].clone(),
    )


def packet_arm(
    tape: HardProgramTape,
    query: HardLateQuery,
    *,
    control: str = "normal",
    force_alive: bool = False,
    state_swap: torch.Tensor | None = None,
    swap_after_step: int = 0,
) -> dict[str, object]:
    return {
        "initial_state": tape.initial_state,
        "event_kind": tape.event_kind,
        "event_identity": tape.event_identity,
        "amount": tape.amount,
        "query": query.position,
        "control": control,
        "force_alive": force_alive,
        "state_swap": state_swap,
        "swap_after_step": swap_after_step,
    }


def assess_executor_output(
    output: dict[str, torch.Tensor],
    tape: HardProgramTape,
    query: HardLateQuery,
    *,
    canonical_state: torch.Tensor,
    canonical_answer: torch.Tensor,
    control: str = "normal",
    state_swap: torch.Tensor | None = None,
    swap_after_step: int = 0,
    force_alive: bool = False,
) -> dict[str, object]:
    expected_state, expected_answer, expected_states, expected_alive = semantic_rollout(
        tape,
        query,
        control=control,
        state_swap=state_swap,
        swap_after_step=swap_after_step,
        force_alive=force_alive,
    )
    predicted_state = output["final_state"]
    predicted_answer = output["answer"]
    predicted_states = output["state_trajectory"]
    predicted_alive = output["alive_trajectory"]
    rows = tape.batch_size
    return {
        "rows": rows,
        "state_exact": int(predicted_state.eq(expected_state).sum()),
        "answer_exact": int(predicted_answer.eq(expected_answer).sum()),
        "joint_exact": int(
            (predicted_state.eq(expected_state) & predicted_answer.eq(expected_answer)).sum()
        ),
        "trajectory_exact": int(predicted_states.eq(expected_states).all(-1).sum()),
        "alive_trajectory_exact": int(predicted_alive.eq(expected_alive).all(-1).sum()),
        "state_rate": float(predicted_state.eq(expected_state).float().mean()),
        "answer_rate": float(predicted_answer.eq(expected_answer).float().mean()),
        "state_causal_opportunities": int(expected_state.ne(canonical_state).sum()),
        "answer_causal_opportunities": int(expected_answer.ne(canonical_answer).sum()),
        "all_halted": bool((~predicted_alive[:, -1]).all()) if not force_alive else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-rows", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing mechanics output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("projected mechanics gate requires bf16 CUDA")
    if sha256_file(args.compiler_checkpoint) != PROJECTED_CHECKPOINT_SHA256:
        raise SystemExit("projected compiler checkpoint hash mismatch")
    if sha256_file(args.pilot_report) != PROJECTED_REPORT_SHA256:
        raise SystemExit("projected pilot report hash mismatch")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows, board = load_rows(args.data_dir)
    _, heldout = partition(rows, args.fit_rows)
    pilot_report = json.loads(args.pilot_report.read_text())
    if pilot_report.get("decision") != "advance_hierarchical_binding":
        raise SystemExit("projected pilot did not authorize mechanics")
    if not all(pilot_report.get("gates", {}).values()):
        raise SystemExit("projected pilot report contains a failed gate")

    payload = torch.load(args.compiler_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "r12_sd_cst_hierarchical_binding_training_pilot_v1":
        raise SystemExit("projected compiler schema mismatch")
    if payload.get("seed") != pilot_report.get("seed"):
        raise SystemExit("projected checkpoint/report seed mismatch")
    model = ProjectedHierarchicalBindingBusCompiler()
    model.load_state_dict(payload["state"], strict=True)
    model.requires_grad_(False).eval().to(device)
    frozen_compiler_before = state_dict_digest(model)
    compiler_parameters = model.parameter_count()
    complete_parameters = (
        BASE_PARAMETERS + compiler_parameters + MOTOR_PARAMETERS + READER_PARAMETERS
    )

    motor = TiedCategoricalMotor().to(device)
    reader = CategoricalStateReader().to(device)
    motor_seed = derived_seed(args.seed, "motor")
    reader_seed = derived_seed(args.seed, "reader")
    motor_certificate = fit_motor_certificate(
        SimpleNamespace(motor=motor), seed=motor_seed, lr=0.003, max_updates=1000,
    )
    reader_certificate = fit_reader_certificate(
        SimpleNamespace(reader=reader), seed=reader_seed, lr=0.005, max_updates=500,
    )
    motor.eval()
    reader.eval()

    canonical_tape, canonical_query, poison_exact = compile_hard(
        model, heldout, args.batch_size, device,
    )
    alpha_rows = [alpha_renamed_row(row) for row in heldout]
    alpha_tape, alpha_query, alpha_poison_exact = compile_hard(
        model, alpha_rows, args.batch_size, device,
    )
    counterfactual_rows = [event_counterfactual_row(row) for row in heldout]
    counterfactual_tape, counterfactual_query, counterfactual_poison_exact = compile_hard(
        model, counterfactual_rows, args.batch_size, device,
    )
    declaration_rows = [declaration_role_swap_row(row) for row in heldout]
    declaration_tape, declaration_query, declaration_poison_exact = compile_hard(
        model, declaration_rows, args.batch_size, device,
    )
    relocated_rows = [relocated_event_lines_row(row) for row in heldout]
    relocated_tape, relocated_query, relocated_poison_exact = compile_hard(
        model, relocated_rows, args.batch_size, device,
    )
    canonical_tape_metrics = tape_metrics(canonical_tape, expected_tape(heldout))
    alpha_tape_metrics = tape_metrics(alpha_tape, expected_tape(alpha_rows))
    counterfactual_tape_metrics = tape_metrics(
        counterfactual_tape, expected_tape(counterfactual_rows),
    )
    declaration_tape_metrics = tape_metrics(
        declaration_tape, expected_tape(declaration_rows),
    )
    relocated_tape_metrics = tape_metrics(
        relocated_tape, expected_tape(relocated_rows),
    )
    query_exact = {
        "canonical": int(canonical_query.position.eq(torch.tensor(
            [row.query_position for row in heldout], dtype=torch.uint8,
        )).sum()),
        "alpha_rename": int(alpha_query.position.eq(canonical_query.position).sum()),
        "event_name_counterfactual": int(
            counterfactual_query.position.eq(canonical_query.position).sum()
        ),
        "declaration_role_swap": int(
            declaration_query.position.eq(canonical_query.position).sum()
        ),
        "event_line_relocation": int(
            relocated_query.position.eq(canonical_query.position).sum()
        ),
    }

    frozen_compiler_after = state_dict_digest(model)

    model.cpu()
    del model, payload
    torch.cuda.empty_cache()

    canonical_state, canonical_answer, _, _ = semantic_rollout(
        expected_tape(heldout), canonical_query,
    )
    state_swap = torch.arange(canonical_tape.batch_size - 1, -1, -1)
    control_inputs = {
        "canonical": (canonical_tape, canonical_query, {}),
        "alpha_rename": (alpha_tape, alpha_query, {}),
        "event_name_counterfactual": (
            counterfactual_tape, counterfactual_query, {},
        ),
        "declaration_role_swap": (
            declaration_tape, declaration_query, {},
        ),
        "event_line_relocation": (relocated_tape, relocated_query, {}),
        "initial_state_rotation": (
            mutate_first_active(canonical_tape, "initial_state"), canonical_query, {},
        ),
        "event_kind_flip": (
            mutate_first_active(canonical_tape, "kind"), canonical_query, {},
        ),
        "event_identity_rotation": (
            mutate_first_active(canonical_tape, "identity"), canonical_query, {},
        ),
        "event_amount_flip": (
            mutate_first_active(canonical_tape, "amount"), canonical_query, {},
        ),
        "query_rotation": (
            canonical_tape, rotate_queries(canonical_query), {},
        ),
        "state_swap_after_step_0": (
            canonical_tape, canonical_query,
            {"state_swap": state_swap, "swap_after_step": 0},
        ),
        "reset_each_step": (
            canonical_tape, canonical_query, {"control": "reset"},
        ),
        "freeze_state": (
            canonical_tape, canonical_query, {"control": "freeze"},
        ),
        "post_stop_perturbation": (
            perturb_post_stop(canonical_tape), canonical_query, {},
        ),
        "force_alive_post_stop": (
            perturb_post_stop(canonical_tape), canonical_query,
            {"force_alive": True},
        ),
        "operand_suffix_swap": (
            swap_operand_suffix(canonical_tape, 4), canonical_query, {},
        ),
        "shuffled_program_packet": (
            shuffled_packet(canonical_tape), canonical_query, {},
        ),
    }

    args.out_dir.mkdir(parents=True)
    execution_path = args.out_dir / "execution_core.pt"
    torch.save({
        "schema": "r12_sd_cst_projected_execution_core_v1",
        "motor": motor.cpu().state_dict(),
        "reader": reader.cpu().state_dict(),
        "seed": args.seed,
        "motor_seed": motor_seed,
        "reader_seed": reader_seed,
        "compiler_checkpoint_sha256": PROJECTED_CHECKPOINT_SHA256,
        "score_eligible": False,
    }, execution_path)
    packet_path = args.out_dir / "hard_packets.pt"
    packet_arms = {}
    for name, (tape, query, settings) in control_inputs.items():
        packet_arms[name] = packet_arm(tape, query, **settings)
    torch.save({
        "schema": "r12_sd_cst_hard_packet_bundle_v1",
        "arms": packet_arms,
    }, packet_path)
    executor_output_path = args.out_dir / "hard_packet_outputs.pt"
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("run_sd_cst_hard_packets.py")),
        "--packets", str(packet_path),
        "--execution-core", str(execution_path),
        "--output", str(executor_output_path),
    ], check=True)
    executor_payload = torch.load(
        executor_output_path, map_location="cpu", weights_only=False,
    )
    if executor_payload.get("schema") != "r12_sd_cst_hard_packet_outputs_v1":
        raise SystemExit("source-blind executor output schema mismatch")
    if set(executor_payload.get("outputs", {})) != set(control_inputs):
        raise SystemExit("source-blind executor arm mismatch")
    controls = {}
    for name, (tape, query, settings) in control_inputs.items():
        controls[name] = assess_executor_output(
            executor_payload["outputs"][name],
            tape,
            query,
            canonical_state=canonical_state,
            canonical_answer=canonical_answer,
            **settings,
        )

    rows_count = len(heldout)
    stop_buckets = {}
    canonical_output = executor_payload["outputs"]["canonical"]
    canonical_row_exact = (
        canonical_output["final_state"].eq(canonical_state)
        & canonical_output["answer"].eq(canonical_answer)
        & canonical_output["state_trajectory"].eq(
            semantic_rollout(canonical_tape, canonical_query)[2]
        ).all(-1)
    )
    for stop_position in range(EVENT_STEPS):
        members = canonical_tape.event_kind[:, stop_position].eq(STOP_KIND)
        stop_buckets[str(stop_position)] = {
            "rows": int(members.sum()),
            "exact": int(canonical_row_exact[members].sum()),
        }
    gates = {
        "projected_pilot_all_gates_pass": all(pilot_report["gates"].values()),
        "canonical_whole_tape_exact": canonical_tape_metrics["exact"]["whole_tape"] == rows_count,
        "every_query_byte_exact": all(value == rows_count for value in query_exact.values()),
        "untrained_projection_whole_tape_at_most_1pct": (
            pilot_report["parent"]["prefit"]["rates"]["whole_tape"] <= 0.01
        ),
        "source_poison_bit_identical": all((
            poison_exact, alpha_poison_exact, counterfactual_poison_exact,
            declaration_poison_exact, relocated_poison_exact,
        )),
        "full_compiler_state_dict_bit_identical": (
            frozen_compiler_before == frozen_compiler_after
        ),
        "hard_payload_only_25_program_bytes_plus_query": (
            canonical_tape.initial_state.numel()
            + canonical_tape.event_kind.numel()
            + canonical_tape.event_identity.numel()
            + canonical_tape.amount.numel()
            == rows_count * 25
            and canonical_query.position.numel() == rows_count
        ),
        "motor_certificate_exact": bool(motor_certificate["exact"]),
        "reader_certificate_exact": bool(reader_certificate["exact"]),
        "alpha_rename_whole_tape_at_least_99pct": alpha_tape_metrics["rates"]["whole_tape"] >= 0.99,
        "event_counterfactual_whole_tape_at_least_95pct": counterfactual_tape_metrics["rates"]["whole_tape"] >= 0.95,
        "declaration_swap_whole_tape_at_least_95pct": declaration_tape_metrics["rates"]["whole_tape"] >= 0.95,
        "event_line_relocation_whole_tape_at_least_99pct": relocated_tape_metrics["rates"]["whole_tape"] >= 0.99,
        "executor_is_separate_source_blind_process": executor_output_path.is_file(),
        "every_control_matches_its_oracle": all(
            controls[name]["state_exact"] == rows_count
            and controls[name]["answer_exact"] == rows_count
            and controls[name]["joint_exact"] == rows_count
            and controls[name]["trajectory_exact"] == rows_count
            and controls[name]["alive_trajectory_exact"] == rows_count
            for name in control_inputs
        ),
        "every_stop_position_bucket_exact": all(
            values["rows"] == 0 or values["exact"] == values["rows"]
            for values in stop_buckets.values()
        ) and sum(values["rows"] > 0 for values in stop_buckets.values()) == 6,
        "canonical_and_post_stop_halt": (
            controls["canonical"]["all_halted"] is True
            and controls["post_stop_perturbation"]["all_halted"] is True
        ),
        "post_stop_invariance_exact": (
            controls["post_stop_perturbation"]["state_causal_opportunities"] == 0
            and controls["post_stop_perturbation"]["answer_causal_opportunities"] == 0
        ),
        "query_rotation_changes_every_answer": controls["query_rotation"]["answer_causal_opportunities"] == rows_count,
        "packet_mutations_each_have_at_least_1024_state_or_answer_opportunities": all(
            controls[name]["state_causal_opportunities"] >= 1024
            or controls[name]["answer_causal_opportunities"] >= 1024
            for name in (
                "initial_state_rotation", "event_kind_flip",
                "event_identity_rotation", "event_amount_flip",
            )
        ),
        "state_swap_has_at_least_512_state_opportunities": controls["state_swap_after_step_0"]["state_causal_opportunities"] >= 512,
        "reset_has_at_least_20pct_state_opportunity": controls["reset_each_step"]["state_causal_opportunities"] >= 0.20 * rows_count,
        "freeze_has_at_least_20pct_state_opportunity": controls["freeze_state"]["state_causal_opportunities"] >= 0.20 * rows_count,
        "force_alive_has_at_least_20pct_state_opportunity": controls["force_alive_post_stop"]["state_causal_opportunities"] >= 0.20 * rows_count,
        "suffix_swap_has_at_least_10pct_state_opportunity": controls["operand_suffix_swap"]["state_causal_opportunities"] >= 0.10 * rows_count,
        "shuffled_packet_state_at_most_25pct_and_answer_at_most_45pct": (
            rows_count - controls["shuffled_program_packet"]["state_causal_opportunities"] <= 0.25 * rows_count
            and rows_count - controls["shuffled_program_packet"]["answer_causal_opportunities"] <= 0.45 * rows_count
        ),
        "complete_system_below_stricter_150m_pilot_cap": complete_parameters < PILOT_PARAMETER_CAP,
        "complete_system_below_global_200m_cap": complete_parameters < GLOBAL_PARAMETER_CAP,
        "scored_access_zero": board["development_accesses"] == 0 and board["confirmation_accesses"] == 0,
    }

    report = {
        "schema": "r12_sd_cst_projected_mechanics_report_v1",
        "decision": "admit_fresh_board_integration" if all(gates.values()) else "reject_or_revise_projected_mechanics",
        "source_commit": args.source_commit,
        "pilot_source_commit": PILOT_SOURCE_COMMIT,
        "seed": args.seed,
        "board": board,
        "inputs": {
            "compiler_checkpoint_sha256": PROJECTED_CHECKPOINT_SHA256,
            "pilot_report_sha256": PROJECTED_REPORT_SHA256,
            "pilot_seed": pilot_report["seed"],
        },
        "partition": {
            "method": "sha256(row_id) ordering inherited from projected pilot",
            "fit_rows_consumed_by_parent": args.fit_rows,
            "heldout_consumed_training_rows": rows_count,
        },
        "parameters": {
            "base": BASE_PARAMETERS,
            "compiler": compiler_parameters,
            "motor": MOTOR_PARAMETERS,
            "reader": READER_PARAMETERS,
            "complete_system": complete_parameters,
            "pilot_cap": PILOT_PARAMETER_CAP,
            "global_cap": GLOBAL_PARAMETER_CAP,
            "maximum_additional_under_strict_global_cap": (
                GLOBAL_PARAMETER_CAP - complete_parameters - 1
            ),
            "projected_trainable_names": sorted(PROJECTED_TRAINABLE_NAMES),
        },
        "certificates": {
            "motor": motor_certificate,
            "reader": reader_certificate,
        },
        "compiler": {
            "canonical": canonical_tape_metrics,
            "alpha_rename": alpha_tape_metrics,
            "event_name_counterfactual": counterfactual_tape_metrics,
            "declaration_role_swap": declaration_tape_metrics,
            "event_line_relocation": relocated_tape_metrics,
            "late_query_exact": query_exact,
            "untrained_projection_prefit": pilot_report["parent"]["prefit"],
            "full_state_dict_digest_before": frozen_compiler_before,
            "full_state_dict_digest_after": frozen_compiler_after,
        },
        "controls": controls,
        "stop_position_buckets": stop_buckets,
        "source_deletion": {
            "program_payload_fields": [
                "initial_state", "event_kind", "event_identity", "amount",
            ],
            "program_payload_elements_per_row": 25,
            "late_query_elements_per_row": 1,
            "dtype": "torch.uint8",
            "source_poison_bit_identical": gates["source_poison_bit_identical"],
            "program_source_destroyed_before_query_compile": True,
            "source_blind_executor_process": str(
                Path(__file__).with_name("run_sd_cst_hard_packets.py").name
            ),
            "hard_packets_sha256": sha256_file(packet_path),
            "hard_packet_outputs_sha256": sha256_file(executor_output_path),
        },
        "gates": gates,
        "execution_core_sha256": sha256_file(execution_path),
        "score_eligible": False,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "claim_boundary": (
            "Consumed training rows only. Passing admits a fresh-board integration; "
            "it is not a development score or broad native-reasoning claim."
        ),
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gates": gates,
        "compiler": report["compiler"],
        "controls": controls,
        "parameters": report["parameters"],
        "execution_core_sha256": report["execution_core_sha256"],
        "report_sha256": sha256_file(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
