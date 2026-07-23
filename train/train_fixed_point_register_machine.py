#!/usr/bin/env python3
"""Train the query-blind register controller on gold source-deleted packets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from contrastive_fixed_point_board import generate_rows, validate_row
from equivariant_relation_register_machine import (
    CONTINUE,
    HALT,
    MAX_OBJECTS,
    READ_ONLY_REGISTERS,
    REGISTER_COUNT,
    DeletedRelationRegisterPacket,
    ControllerAction,
    EquivariantRelationRegisterMachine,
    LateRelationRegisterQuery,
    RelationOperation,
    controller_parameter_receipt,
)


CHECKPOINT_SCHEMA = "fixed_point_register_machine_v1"


def _row_tensor(
    rows: Sequence[dict[str, object]],
    field: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [row[field] for row in rows],
        dtype=torch.float,
        device=device,
    )


def batch_from_rows(
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
) -> tuple[
    DeletedRelationRegisterPacket,
    LateRelationRegisterQuery,
    torch.Tensor,
    torch.Tensor,
]:
    if not rows:
        raise ValueError("fixed-point batch is empty")
    for row in rows:
        validate_row(row)
    packet = DeletedRelationRegisterPacket(
        cardinality=torch.tensor(
            [row["cardinality"] for row in rows],
            dtype=torch.uint8,
            device=device,
        ),
        registers=_row_tensor(rows, "input_registers", device=device),
    )
    query = LateRelationRegisterQuery(
        register=torch.tensor(
            [row["query"]["register"] for row in rows],  # type: ignore[index]
            dtype=torch.long,
            device=device,
        ),
        position=torch.tensor(
            [row["query"]["position"] for row in rows],  # type: ignore[index]
            dtype=torch.long,
            device=device,
        ),
    )
    target_registers = _row_tensor(
        rows,
        "target_registers",
        device=device,
    )
    target_answer = _row_tensor(rows, "answer_bits", device=device)
    return packet, query, target_registers, target_answer


def fixed_point_loss(
    machine: EquivariantRelationRegisterMachine,
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
    hard: bool,
    halt_weight: float = 0.0,
    runtime_weight: float = 0.0,
    teacher_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if halt_weight < 0 or runtime_weight < 0 or teacher_weight < 0:
        raise ValueError("fixed-point loss weights differ")
    packet, query, target_registers, target_answer = batch_from_rows(
        rows,
        device=device,
    )
    result = machine(packet, query, hard=hard)
    active_objects = (
        torch.arange(MAX_OBJECTS, device=device)[None]
        < packet.cardinality.long()[:, None]
    )
    active_square = active_objects[:, :, None] & active_objects[:, None, :]
    work_mask = active_square[:, None].expand(
        -1,
        REGISTER_COUNT - READ_ONLY_REGISTERS,
        -1,
        -1,
    )
    work_target = target_registers[:, READ_ONLY_REGISTERS:]
    work_prediction = result.final_registers[:, READ_ONLY_REGISTERS:]
    state_error = (work_prediction - work_target).square()
    state_positive = work_mask & work_target.bool()
    state_negative = work_mask & ~work_target.bool()
    state = 0.5 * (
        state_error[state_positive].mean()
        + state_error[state_negative].mean()
    )
    answer_values = (result.answer - target_answer).square()
    answer_positive = active_objects & target_answer.bool()
    answer_negative = active_objects & ~target_answer.bool()
    answer = 0.5 * (
        answer_values[answer_positive].mean()
        + answer_values[answer_negative].mean()
    )
    alive = result.alive_trajectory[-1].mean()
    expected_runtime = torch.stack(result.alive_trajectory, dim=1).mean()
    teacher = teacher_action_loss(result.actions, rows, device=device)
    total = (
        state
        + answer
        + halt_weight * alive
        + runtime_weight * expected_runtime
        + teacher_weight * teacher
    )
    return total, {
        "state": state,
        "answer": answer,
        "alive": alive,
        "expected_runtime": expected_runtime,
        "teacher": teacher,
    }


def _teacher_program(
    row: dict[str, object],
) -> list[tuple[int, int, int, int, int, int]]:
    program = [
        (int(RelationOperation.IDENTITY), 2, 2, 3, CONTINUE, 0),
    ]
    program.extend(
        (int(RelationOperation.EXPAND), 0, 3, 3, CONTINUE, 0)
        for _ in range(int(row["a_depth"]))
    )
    program.append(
        (int(RelationOperation.IDENTITY), 2, 2, 4, CONTINUE, 1)
    )
    program.extend(
        (int(RelationOperation.EXPAND), 1, 4, 4, CONTINUE, 1)
        for _ in range(int(row["b_depth"]))
    )
    program.extend(
        (
            (
                int(RelationOperation.DIFFERENCE),
                3,
                4,
                5,
                CONTINUE,
                2,
            ),
            (int(RelationOperation.CLEAR), 0, 0, 3, HALT, 2),
        )
    )
    return program


def teacher_action_loss(
    actions: Sequence[ControllerAction],
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not actions or not rows:
        raise ValueError("fixed-point teacher batch differs")
    losses = []
    for row_index, row in enumerate(rows):
        program = _teacher_program(row)
        if len(program) > len(actions):
            raise ValueError("fixed-point teacher exceeds controller horizon")
        for step, target in enumerate(program):
            action = actions[step]
            halt_target = target[4]
            losses.append(
                F.cross_entropy(
                    action.halt_logits[row_index : row_index + 1],
                    torch.tensor([halt_target], device=device),
                )
            )
            losses.append(
                F.cross_entropy(
                    action.phase_logits[row_index : row_index + 1],
                    torch.tensor([target[5]], device=device),
                )
            )
            if halt_target == HALT:
                continue
            for logits, label in (
                (action.operation_logits, target[0]),
                (action.left_logits, target[1]),
                (action.right_logits, target[2]),
                (
                    action.destination_logits,
                    target[3] - READ_ONLY_REGISTERS,
                ),
            ):
                losses.append(
                    F.cross_entropy(
                        logits[row_index : row_index + 1],
                        torch.tensor([label], device=device),
                    )
                )
    return torch.stack(losses).mean().to(device)


def training_curriculum(
    update: int,
    updates: int,
) -> tuple[bool, float, float, float]:
    if updates < 1 or not 0 <= update < updates:
        raise ValueError("fixed-point curriculum differs")
    hard = update >= updates // 10
    teacher_decay_start = updates // 2
    teacher_decay_end = 4 * updates // 5
    if update < teacher_decay_start:
        teacher_weight = 1.0
    elif update < teacher_decay_end:
        progress = (update - teacher_decay_start) / max(
            1,
            teacher_decay_end - teacher_decay_start,
        )
        teacher_weight = 1.0 - 0.9 * progress
    else:
        teacher_weight = 0.1
    halt_progress = max(
        0.0,
        (update - teacher_decay_end)
        / max(1.0, updates - teacher_decay_end),
    )
    halt_weight = 0.1 * halt_progress
    runtime_weight = 0.001 if hard else 0.0
    return hard, teacher_weight, halt_weight, runtime_weight


@torch.inference_mode()
def evaluate(
    machine: EquivariantRelationRegisterMachine,
    rows: Sequence[dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    totals: Counter[str] = Counter()
    by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    actions: Counter[str] = Counter()
    machine.eval()
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        packet, query, target_registers, target_answer = batch_from_rows(
            selected,
            device=device,
        )
        result = machine(packet, query, hard=True)
        active_objects = (
            torch.arange(MAX_OBJECTS, device=device)[None]
            < packet.cardinality.long()[:, None]
        )
        active_square = (
            active_objects[:, :, None] & active_objects[:, None, :]
        )
        work_exact = (
            result.final_registers[:, READ_ONLY_REGISTERS:]
            .ge(0.5)
            .eq(target_registers[:, READ_ONLY_REGISTERS:].bool())
            | ~active_square[:, None]
        ).all(dim=(-1, -2, -3))
        answer_exact = (
            result.answer.ge(0.5).eq(target_answer.bool())
            | ~active_objects
        ).all(-1)
        halt_exact = result.halted_by_deadline
        joint = work_exact & answer_exact & halt_exact
        for index, row in enumerate(selected):
            values = {
                "work": int(work_exact[index]),
                "answer": int(answer_exact[index]),
                "halt": int(halt_exact[index]),
                "joint": int(joint[index]),
            }
            totals.update(values)
            totals["rows"] += 1
            depth = int(row["a_depth"])
            by_depth[depth].update(values)
            by_depth[depth]["rows"] += 1
        for step, action in enumerate(result.actions):
            operation = action.operation.argmax(-1).tolist()
            left = action.left.argmax(-1).tolist()
            right = action.right.argmax(-1).tolist()
            destination = action.destination.argmax(-1).tolist()
            halt = action.halt.argmax(-1).tolist()
            phase = action.phase.argmax(-1).tolist()
            for values in zip(
                operation,
                left,
                right,
                destination,
                halt,
                phase,
                strict=True,
            ):
                actions[f"{step}:{values}"] += 1
    count = totals["rows"]
    return {
        "rows": count,
        "work_accuracy": totals["work"] / count,
        "answer_accuracy": totals["answer"] / count,
        "halt_accuracy": totals["halt"] / count,
        "joint_accuracy": totals["joint"] / count,
        "by_depth": {
            str(depth): {
                "rows": values["rows"],
                "work_accuracy": values["work"] / values["rows"],
                "answer_accuracy": values["answer"] / values["rows"],
                "halt_accuracy": values["halt"] / values["rows"],
                "joint_accuracy": values["joint"] / values["rows"],
            }
            for depth, values in sorted(by_depth.items())
        },
        "action_histogram": dict(sorted(actions.items())),
    }


def cosine_scale(step: int, updates: int, warmup: int) -> float:
    if updates < 1 or not 0 <= warmup < updates:
        raise ValueError("fixed-point schedule differs")
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, updates - warmup - 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"fixed-point output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"fixed-point report exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-rows", type=int, default=8_192)
    parser.add_argument("--development-rows", type=int, default=1_024)
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--controller-width", type=int, default=512)
    parser.add_argument("--controller-layers", type=int, default=3)
    parser.add_argument("--maximum-steps", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.train_rows < args.batch_size
        or args.development_rows < 1
        or args.updates < 1
        or args.learning_rate <= 0
        or args.clip <= 0
        or args.log_every < 1
    ):
        raise SystemExit("fixed-point training arguments differ")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("fixed-point training has no CUDA")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_rows = generate_rows(
        split="train",
        count=args.train_rows,
        seed=args.seed,
    )
    development_rows = generate_rows(
        split="development",
        count=args.development_rows,
        seed=args.seed + 1,
    )
    train_hash = hashlib.sha256(
        "\n".join(canonical_row(row) for row in train_rows).encode()
    ).hexdigest()
    development_hash = hashlib.sha256(
        "\n".join(canonical_row(row) for row in development_rows).encode()
    ).hexdigest()
    machine = EquivariantRelationRegisterMachine(
        controller_width=args.controller_width,
        controller_layers=args.controller_layers,
        maximum_steps=args.maximum_steps,
    ).to(device)
    parameters = list(machine.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    order = list(range(len(train_rows)))
    random.Random(args.seed + 7).shuffle(order)
    cursor = 0
    losses: list[float] = []
    start_time = time.monotonic()
    machine.train()
    for update in range(args.updates):
        if cursor + args.batch_size > len(order):
            random.Random(args.seed + update + 17).shuffle(order)
            cursor = 0
        selected = [
            train_rows[index]
            for index in order[cursor : cursor + args.batch_size]
        ]
        cursor += args.batch_size
        scale = cosine_scale(update, args.updates, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.zero_grad(set_to_none=True)
        hard, teacher_weight, halt_weight, runtime_weight = (
            training_curriculum(update, args.updates)
        )
        loss, components = fixed_point_loss(
            machine,
            selected,
            device=device,
            hard=hard,
            halt_weight=halt_weight,
            runtime_weight=runtime_weight,
            teacher_weight=teacher_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("fixed-point loss became non-finite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("fixed-point gradient became non-finite")
        optimizer.step()
        losses.append(float(loss.detach()))
        if update == 0 or (update + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "loss": losses[-1],
                        "mean_recent_loss": sum(
                            losses[-args.log_every :]
                        ) / min(len(losses), args.log_every),
                        "gradient_norm": float(norm),
                        "hard_actions": hard,
                        "halt_weight": halt_weight,
                        "teacher_weight": teacher_weight,
                        **{
                            name: float(value.detach())
                            for name, value in components.items()
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_metrics = evaluate(
        machine,
        train_rows,
        batch_size=args.batch_size,
        device=device,
    )
    development_metrics = evaluate(
        machine,
        development_rows,
        batch_size=args.batch_size,
        device=device,
    )
    configuration = {
        "controller_width": args.controller_width,
        "controller_layers": args.controller_layers,
        "maximum_steps": args.maximum_steps,
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "seed": args.seed,
        "configuration": configuration,
        "parameter_receipt": controller_parameter_receipt(machine),
        "train_sha256": train_hash,
        "development_sha256": development_hash,
        "machine": {
            name: value.detach().cpu()
            for name, value in machine.state_dict().items()
        },
    }
    atomic_save(checkpoint, args.out)
    report = {
        "schema": "fixed_point_register_machine_report_v1",
        "claim_boundary": (
            "Gold-packet learned-executor development result only; no language "
            "compiler, confirmation access, or general-reasoning claim."
        ),
        "seed": args.seed,
        "configuration": configuration,
        "parameter_receipt": checkpoint["parameter_receipt"],
        "checkpoint_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "train_sha256": train_hash,
        "development_sha256": development_hash,
        "train_metrics": train_metrics,
        "development_metrics": development_metrics,
        "initial_loss_mean": sum(losses[: min(100, len(losses))])
        / min(100, len(losses)),
        "final_loss_mean": sum(losses[-min(100, len(losses)) :])
        / min(100, len(losses)),
        "elapsed_seconds": time.monotonic() - start_time,
        "confirmation_accesses": 0,
    }
    atomic_json(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def canonical_row(row: object) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
