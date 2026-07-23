"""Train the AHRF without a host executor or host convergence test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from autocatalytic_hysteretic_relation_field import (
    FEEDBACK_ROLE,
    AHRFRollout,
    AutocatalyticHystereticRelationField,
    SourceDeletedRelationGraph,
)
from contextual_bekic_graph_machine import PROGRAM_VARIABLES
from contextualize_bekic_program import contextualize_simultaneous_packet
from contrastive_bekic_program_orbits import (
    DEVELOPMENT_CELLS,
    ISOLATED_COUNTERFACTUAL_ARMS,
    assert_split_disjoint,
    evaluate_simultaneous,
    fixed_point_pressure,
    generate_train_development,
    select_isolated_counterfactual_input,
    select_machine_input,
)
from tensorize_contextual_ahrf import (
    AHRF_NODE_FEATURE_DIM,
    tensorized_contextual_to_ahrf,
)
from tensorize_contextual_bekic import (
    tensorize_contextual_packets,
    tensorize_target_environment,
)


SCORE_ARMS = ("p", "p_prime", "p_eq", *ISOLATED_COUNTERFACTUAL_ARMS)
DEFAULT_SEED = 2026072336
SOURCE_PATHS = (
    "R12_AHRF_PREREG.md",
    "pipeline/bekic_relational_fixed_point_board.py",
    "pipeline/contextualize_bekic_program.py",
    "pipeline/contrastive_bekic_program_orbits.py",
    "train/autocatalytic_hysteretic_relation_field.py",
    "train/contextual_bekic_graph_machine.py",
    "train/equivariant_relation_register_machine.py",
    "train/tensorize_contextual_ahrf.py",
    "train/tensorize_contextual_bekic.py",
    "train/train_autocatalytic_hysteretic_relation_field.py",
)


@dataclass(frozen=True, slots=True)
class AHRFTrainConfig:
    seed: int = DEFAULT_SEED
    train_orbits: int = 8
    development_orbits: int = 10
    train_renderers: int = 2
    steps: int = 1_000
    halt_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 3e-4
    halt_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    card_rounds: int = 2
    max_steps: int = 64
    hard_fraction: float = 0.10
    write_weight: float = 1e-4
    device: str = "auto"
    binder_checkpoint: str | None = None
    control: str = "treatment"


@dataclass(frozen=True, slots=True)
class AHRFBoard:
    graph: SourceDeletedRelationGraph
    targets: torch.Tensor
    roots: torch.Tensor
    labels: tuple[tuple[str, str, str, int], ...]
    max_expression_depth: int
    max_convergence_updates: int
    minimum_safety_steps: int

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_receipt(root: Path) -> dict[str, str]:
    return {
        relative: _sha256(root / relative)
        for relative in SOURCE_PATHS
    }


def _source_packet(
    row: dict[str, object],
    arm: str,
) -> dict[str, object]:
    if arm in {"p", "p_prime", "p_eq"}:
        return select_machine_input(
            row,
            arm=arm,
            form="simultaneous",
        )
    return select_isolated_counterfactual_input(
        row,
        arm=arm,
        form="simultaneous",
    )


def _target(source: dict[str, object]) -> torch.Tensor:
    variables = [str(item) for item in source["program"]["variables"]]
    return tensorize_target_environment(
        evaluate_simultaneous(source),
        variables,
        cardinality=int(source["cardinality"]),
    )


def _expression_depth(expression: object) -> int:
    if not isinstance(expression, dict):
        raise ValueError("expression differs")
    children: list[object] = []
    if isinstance(expression.get("child"), dict):
        children.append(expression["child"])
    if isinstance(expression.get("children"), list):
        children.extend(expression["children"])
    return 1 + max(
        (_expression_depth(child) for child in children),
        default=0,
    )


def build_board(config: AHRFTrainConfig) -> AHRFBoard:
    train, development = generate_train_development(
        train_count=config.train_orbits,
        development_count=config.development_orbits,
        seed=config.seed,
    )
    assert_split_disjoint(train, development)
    if {
        str(row["axes"]["cell"])
        for row in development
    } != set(DEVELOPMENT_CELLS):
        raise ValueError("development board does not cover every cell")
    contextual: list[dict[str, object]] = []
    targets: list[torch.Tensor] = []
    labels: list[tuple[str, str, str, int]] = []
    max_expression_depth = 0
    max_convergence_updates = 0
    minimum_safety_steps = 0
    for split, rows, renderers in (
        ("train", train, config.train_renderers),
        ("development", development, 1),
    ):
        for row_index, row in enumerate(rows):
            cell = str(row["axes"]["cell"])
            for arm_index, arm in enumerate(SCORE_ARMS):
                source = _source_packet(row, arm)
                program = source.get("program")
                if not isinstance(program, dict):
                    raise ValueError("board program differs")
                equations = program.get("equations")
                if not isinstance(equations, list):
                    raise ValueError("board equations differ")
                depth = max(
                    _expression_depth(equation["expression"])
                    for equation in equations
                )
                updates = int(
                    fixed_point_pressure(source)["convergence_updates"]
                )
                max_expression_depth = max(max_expression_depth, depth)
                max_convergence_updates = max(
                    max_convergence_updates,
                    updates,
                )
                minimum_safety_steps = max(
                    minimum_safety_steps,
                    depth * (updates + 1),
                )
                for renderer in range(renderers):
                    contextual.append(
                        contextualize_simultaneous_packet(
                            source,
                            seed=(
                                config.seed
                                + (1_000_000 if split == "train" else 2_000_000)
                                + row_index * 10_000
                                + arm_index * 100
                                + renderer
                            ),
                        )
                    )
                    targets.append(_target(source))
                    labels.append((split, cell, arm, renderer))
    tensors = tensorize_contextual_packets(contextual)
    return AHRFBoard(
        graph=tensorized_contextual_to_ahrf(tensors),
        targets=torch.stack(targets),
        roots=tensors.packet.equation_root,
        labels=tuple(labels),
        max_expression_depth=max_expression_depth,
        max_convergence_updates=max_convergence_updates,
        minimum_safety_steps=minimum_safety_steps,
    )


def _index_graph(
    graph: SourceDeletedRelationGraph,
    indices: torch.Tensor,
) -> SourceDeletedRelationGraph:
    return SourceDeletedRelationGraph(
        **{
            field: getattr(graph, field).index_select(0, indices)
            for field in SourceDeletedRelationGraph.__dataclass_fields__
        }
    )


def _move_graph(
    graph: SourceDeletedRelationGraph,
    device: torch.device,
) -> SourceDeletedRelationGraph:
    return SourceDeletedRelationGraph(
        **{
            field: getattr(graph, field).to(device)
            for field in SourceDeletedRelationGraph.__dataclass_fields__
        }
    )


def _apply_control(
    graph: SourceDeletedRelationGraph,
    control: str,
) -> SourceDeletedRelationGraph:
    if control in {
        "treatment",
        "no_hysteresis",
        "generic_recurrence",
        "false_triad",
        "zero_triad",
    }:
        return graph
    if control == "no_feedback":
        edges = graph.argument_edges.clone()
        edges[..., FEEDBACK_ROLE] = False
        return replace(graph, argument_edges=edges)
    if control == "shuffled_cards":
        attachment = torch.zeros_like(graph.node_card_mask)
        arity = graph.argument_mask.long().sum(-1)
        slot_arity = torch.where(
            graph.witness_mask,
            arity,
            torch.full_like(arity, -1),
        ).amax(2)
        for batch_index in range(graph.node_card_mask.shape[0]):
            for value in (0, 1, 2):
                slots = slot_arity[batch_index].eq(value).nonzero().flatten()
                if len(slots) < 2:
                    attachment[
                        batch_index,
                        :,
                        slots,
                    ] = graph.node_card_mask[
                        batch_index,
                        :,
                        slots,
                    ]
                    continue
                targets = slots.roll(1)
                for source, target in zip(slots, targets, strict=True):
                    attachment[
                        batch_index,
                        :,
                        target,
                    ] = graph.node_card_mask[
                        batch_index,
                        :,
                        source,
                    ]
        return replace(graph, node_card_mask=attachment)
    raise ValueError("AHRF control differs")


def _root_facts(
    facts: torch.Tensor,
    roots: torch.Tensor,
) -> torch.Tensor:
    if facts.ndim == 4:
        return facts.gather(
            1,
            roots[..., None, None].expand(
                -1,
                -1,
                facts.shape[-2],
                facts.shape[-1],
            ),
        )
    if facts.ndim == 5:
        return facts.gather(
            2,
            roots[:, None, :, None, None].expand(
                -1,
                facts.shape[1],
                -1,
                facts.shape[-2],
                facts.shape[-1],
            ),
        )
    raise ValueError("root fact rank differs")


def _target_mask(
    object_mask: torch.Tensor,
) -> torch.Tensor:
    square = (
        object_mask[:, :, None]
        & object_mask[:, None, :]
    )
    return square[:, None].expand(-1, PROGRAM_VARIABLES, -1, -1)


def _terminal_loss(
    rollout: AHRFRollout,
    roots: torch.Tensor,
    targets: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    hard_events: bool,
) -> torch.Tensor:
    prediction = _root_facts(rollout.terminal_facts, roots)
    mask = _target_mask(object_mask)
    if hard_events:
        return F.mse_loss(
            prediction[mask],
            targets[mask],
        )
    return F.binary_cross_entropy(
        prediction[mask].clamp(1e-6, 1.0 - 1e-6),
        targets[mask],
    )


def _transfer_binder(
    model: AutocatalyticHystereticRelationField,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    state = checkpoint.get("model_state")
    if (
        checkpoint.get("protocol")
        != "contextual_witness_equivariant_binder_v1"
        or not isinstance(config, dict)
        or not isinstance(state, dict)
        or int(config.get("width", -1)) != model.hidden_dim
        or int(config.get("rounds", -1)) != model.card_rounds
        or str(config.get("architecture", "equivariant"))
        != "equivariant"
        or str(config.get("triad_mode", "learned")) != "learned"
    ):
        raise ValueError("binder warm-start contract differs")
    destination = model.state_dict()
    copied: list[str] = []
    for source_prefix, destination_prefix in (
        ("pair_input.", "card_encoder.pair_input."),
        ("pair_rounds.", "card_encoder.pair_rounds."),
        ("witness_encoder.", "card_encoder.witness_encoder."),
    ):
        for name, value in state.items():
            if not name.startswith(source_prefix):
                continue
            target_name = destination_prefix + name[len(source_prefix):]
            if (
                target_name not in destination
                or destination[target_name].shape != value.shape
            ):
                raise ValueError("binder warm-start tensor differs")
            destination[target_name].copy_(value)
            copied.append(target_name)
    if not copied:
        raise ValueError("binder warm-start copied no parameters")
    model.load_state_dict(destination)
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "copied_tensors": sorted(copied),
        "copied_parameters": sum(
            destination[name].numel() for name in copied
        ),
    }


def _sample_indices(
    available: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selection = torch.randint(
        available.numel(),
        (batch_size,),
        generator=generator,
    )
    return available[selection]


def _exact_receipt(
    model: AutocatalyticHystereticRelationField,
    board: AHRFBoard,
    indices: torch.Tensor,
    device: torch.device,
    enable_halt: bool = True,
    batch_size: int = 4,
) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str]] = Counter()
    correct: Counter[tuple[str, str, str]] = Counter()
    halted: Counter[tuple[str, str, str]] = Counter()
    exact_total = 0
    halted_total = 0
    safety_total = 0
    halt_steps: list[int] = []
    for chunk in indices.split(batch_size):
        graph = _move_graph(_index_graph(board.graph, chunk), device)
        targets = board.targets.index_select(0, chunk).to(device)
        roots = board.roots.index_select(0, chunk).to(device)
        with torch.no_grad():
            rollout = model(
                graph,
                hard_events=True,
                enable_halt=enable_halt,
            )
        prediction = _root_facts(rollout.terminal_facts, roots)
        mask = _target_mask(graph.object_mask)
        exact = (
            (prediction.eq(targets) | ~mask)
            .flatten(1)
            .all(-1)
        )
        labels = [board.labels[int(index)] for index in chunk]
        for label, is_exact, is_halted in zip(
            labels,
            exact.tolist(),
            rollout.learned_halted.tolist(),
            strict=True,
        ):
            key = label[:3]
            counts[key] += 1
            correct[key] += int(is_exact)
            halted[key] += int(is_halted)
        exact_total += int(exact.sum())
        halted_total += int(rollout.learned_halted.sum())
        safety_total += int(rollout.safety_exhausted.sum())
        halt_steps.extend(rollout.halt_step.cpu().tolist())
    return {
        "exact": exact_total,
        "count": len(indices),
        "exact_rate": exact_total / len(indices),
        "learned_halted": halted_total,
        "safety_exhausted": safety_total,
        "halt_steps": halt_steps,
        "metrics": {
            ":".join(key): {
                "exact": correct[key],
                "halted": halted[key],
                "count": counts[key],
            }
            for key in sorted(counts)
        },
    }


def train_ahrf(
    config: AHRFTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    root = Path(__file__).resolve().parents[1]
    sources = _source_receipt(root)
    device = _resolve_device(config.device)
    board = build_board(config)
    if config.max_steps < board.minimum_safety_steps:
        raise ValueError(
            "AHRF safety horizon is below the board propagation envelope"
        )
    board = replace(
        board,
        graph=_apply_control(board.graph, config.control),
    )
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=AHRF_NODE_FEATURE_DIM,
        hidden_dim=config.hidden_dim,
        card_rounds=config.card_rounds,
        max_steps=config.max_steps,
        hysteresis=config.control != "no_hysteresis",
        use_card_conditioning=(
            config.control != "generic_recurrence"
        ),
        triad_mode={
            "false_triad": "false",
            "zero_triad": "zero",
        }.get(config.control, "learned"),
    ).to(device)
    warm_start = None
    if config.binder_checkpoint is not None:
        warm_start = _transfer_binder(
            model,
            Path(config.binder_checkpoint),
        )
    with torch.no_grad():
        model.halt_head.bias.fill_(-8.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_indices = torch.tensor(
        [
            index
            for index, label in enumerate(board.labels)
            if label[0] == "train"
        ],
        dtype=torch.long,
    )
    development_indices = torch.tensor(
        [
            index
            for index, label in enumerate(board.labels)
            if label[0] == "development"
        ],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(config.seed + 1)
    hard_start = max(
        0,
        config.steps - round(config.steps * config.hard_fraction),
    )
    trace: list[dict[str, int | float | bool]] = []
    model.train()
    for step in range(config.steps):
        indices = _sample_indices(
            train_indices,
            batch_size=config.batch_size,
            generator=generator,
        )
        graph = _move_graph(_index_graph(board.graph, indices), device)
        targets = board.targets.index_select(0, indices).to(device)
        roots = board.roots.index_select(0, indices).to(device)
        hard = step >= hard_start
        rollout = model(
            graph,
            hard_events=hard,
            enable_halt=False,
        )
        terminal = _terminal_loss(
            rollout,
            roots,
            targets,
            graph.object_mask,
            hard_events=hard,
        )
        write = rollout.write_probabilities.mean()
        loss = terminal + config.write_weight * write
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (
            step == 0
            or step + 1 == config.steps
            or (step + 1) % max(config.steps // 20, 1) == 0
        ):
            trace.append(
                {
                    "step": step + 1,
                    "hard": hard,
                    "loss": float(loss.detach().cpu()),
                    "terminal": float(terminal.detach().cpu()),
                    "write": float(write.detach().cpu()),
                }
            )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.halt_head.parameters():
        parameter.requires_grad_(True)
    halt_optimizer = torch.optim.AdamW(
        model.halt_head.parameters(),
        lr=config.halt_learning_rate,
        weight_decay=0.0,
    )
    halt_trace: list[dict[str, int | float]] = []
    for step in range(config.halt_steps):
        indices = _sample_indices(
            train_indices,
            batch_size=config.batch_size,
            generator=generator,
        )
        graph = _move_graph(_index_graph(board.graph, indices), device)
        targets = board.targets.index_select(0, indices).to(device)
        roots = board.roots.index_select(0, indices).to(device)
        rollout = model(
            graph,
            hard_events=True,
            enable_halt=False,
            return_history=True,
        )
        assert rollout.fact_history is not None
        root_history = _root_facts(
            rollout.fact_history[:, 1:],
            roots,
        )
        mask = _target_mask(graph.object_mask)[:, None]
        ready = (
            (root_history.eq(targets[:, None]) | ~mask)
            .flatten(2)
            .all(-1)
            .to(rollout.halt_logits.dtype)
        )
        halt_loss = F.binary_cross_entropy_with_logits(
            rollout.halt_logits,
            ready,
        )
        halt_optimizer.zero_grad(set_to_none=True)
        halt_loss.backward()
        halt_optimizer.step()
        if (
            step == 0
            or step + 1 == config.halt_steps
            or (step + 1) % max(config.halt_steps // 10, 1) == 0
        ):
            halt_trace.append(
                {
                    "step": step + 1,
                    "loss": float(halt_loss.detach().cpu()),
                    "ready_cells": int(ready.sum().detach().cpu()),
                    "examples": int(ready.shape[0]),
                }
            )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    train_receipt = _exact_receipt(
        model,
        board,
        train_indices,
        device,
        batch_size=config.batch_size,
    )
    development_receipt = _exact_receipt(
        model,
        board,
        development_indices,
        device,
        batch_size=config.batch_size,
    )
    fixed_deadline_receipt = _exact_receipt(
        model,
        board,
        development_indices,
        device,
        enable_halt=False,
        batch_size=config.batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ahrf.pt"
    report_path = output_dir / "report.json"
    if _source_receipt(root) != sources:
        raise RuntimeError("AHRF source receipt drifted during training")
    checkpoint = {
        "protocol": "autocatalytic_hysteretic_relation_field_v1",
        "config": asdict(config),
        "parameter_receipt": asdict(model.parameter_receipt()),
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "source_sha256": sources,
        "warm_start": warm_start,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "protocol": checkpoint["protocol"],
        "claim_boundary": (
            "This standalone AHRF receives only host-compiled, source-deleted "
            "graph/card tensors, owns fact updates and halt, and calls no host "
            "executor or convergence test. It is not yet integrated with the "
            "Shohin trunk. This board remains synthetic relational fixed-point "
            "reasoning; general reasoning requires integration, cross-family, "
            "and language transfer."
        ),
        "config": asdict(config),
        "device": str(device),
        "board": {
            "examples": len(board),
            "train_examples": len(train_indices),
            "development_examples": len(development_indices),
            "score_arms": list(SCORE_ARMS),
            "max_expression_depth": board.max_expression_depth,
            "max_convergence_updates": board.max_convergence_updates,
            "minimum_safety_steps": board.minimum_safety_steps,
        },
        "parameter_receipt": checkpoint["parameter_receipt"],
        "warm_start": warm_start,
        "trace": trace,
        "halt_trace": halt_trace,
        "train": train_receipt,
        "development": development_receipt,
        "development_fixed_deadline": fixed_deadline_receipt,
        "source_sha256": sources,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-orbits", type=int, default=8)
    parser.add_argument("--development-orbits", type=int, default=10)
    parser.add_argument("--train-renderers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--halt-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--halt-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--card-rounds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--hard-fraction", type=float, default=0.10)
    parser.add_argument("--write-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--binder-checkpoint")
    parser.add_argument(
        "--control",
        choices=(
            "treatment",
            "no_feedback",
            "no_hysteresis",
            "shuffled_cards",
            "generic_recurrence",
        ),
        default="treatment",
    )
    args = parser.parse_args()
    config = AHRFTrainConfig(
        seed=args.seed,
        train_orbits=args.train_orbits,
        development_orbits=args.development_orbits,
        train_renderers=args.train_renderers,
        steps=args.steps,
        halt_steps=args.halt_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        halt_learning_rate=args.halt_learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        card_rounds=args.card_rounds,
        max_steps=args.max_steps,
        hard_fraction=args.hard_fraction,
        write_weight=args.write_weight,
        device=args.device,
        binder_checkpoint=args.binder_checkpoint,
        control=args.control,
    )
    report = train_ahrf(config, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "train": report["train"],
                "development": report["development"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
