"""Train the AHRF without a host executor or host convergence test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from autocatalytic_hysteretic_relation_field import (
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
    max_steps: int = 16
    hard_fraction: float = 0.10
    write_weight: float = 1e-4
    device: str = "auto"
    binder_checkpoint: str | None = None


@dataclass(frozen=True, slots=True)
class AHRFBoard:
    graph: SourceDeletedRelationGraph
    targets: torch.Tensor
    roots: torch.Tensor
    labels: tuple[tuple[str, str, str, int], ...]

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
    for split, rows, renderers in (
        ("train", train, config.train_renderers),
        ("development", development, 1),
    ):
        for row_index, row in enumerate(rows):
            cell = str(row["axes"]["cell"])
            for arm_index, arm in enumerate(SCORE_ARMS):
                source = _source_packet(row, arm)
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
) -> torch.Tensor:
    prediction = _root_facts(rollout.terminal_facts, roots)
    mask = _target_mask(object_mask)
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
        ("card_classifier.0.", "card_encoder.slot_encoder.0."),
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
) -> dict[str, Any]:
    graph = _move_graph(_index_graph(board.graph, indices), device)
    targets = board.targets.index_select(0, indices).to(device)
    roots = board.roots.index_select(0, indices).to(device)
    with torch.no_grad():
        rollout = model(
            graph,
            hard_events=True,
            enable_halt=True,
            return_history=True,
        )
    prediction = _root_facts(rollout.terminal_facts, roots)
    mask = _target_mask(graph.object_mask)
    exact = (
        (prediction.eq(targets) | ~mask)
        .flatten(1)
        .all(-1)
    )
    labels = [board.labels[int(index)] for index in indices]
    counts: Counter[tuple[str, str, str]] = Counter()
    correct: Counter[tuple[str, str, str]] = Counter()
    halted: Counter[tuple[str, str, str]] = Counter()
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
    return {
        "exact": int(exact.sum()),
        "count": len(indices),
        "exact_rate": float(exact.float().mean()),
        "learned_halted": int(rollout.learned_halted.sum()),
        "safety_exhausted": int(rollout.safety_exhausted.sum()),
        "halt_steps": rollout.halt_step.cpu().tolist(),
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
    device = _resolve_device(config.device)
    board = build_board(config)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=AHRF_NODE_FEATURE_DIM,
        hidden_dim=config.hidden_dim,
        card_rounds=config.card_rounds,
        max_steps=config.max_steps,
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
    )
    development_receipt = _exact_receipt(
        model,
        board,
        development_indices,
        device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ahrf.pt"
    report_path = output_dir / "report.json"
    root = Path(__file__).resolve().parents[1]
    sources = {
        relative: _sha256(root / relative)
        for relative in (
            "train/autocatalytic_hysteretic_relation_field.py",
            "train/tensorize_contextual_ahrf.py",
            "train/train_autocatalytic_hysteretic_relation_field.py",
            "pipeline/contextualize_bekic_program.py",
            "pipeline/contrastive_bekic_program_orbits.py",
        )
    }
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
            "AHRF receives only source-deleted graph/card tensors, owns fact "
            "updates and halt, and calls no host executor or convergence test. "
            "This board remains synthetic relational fixed-point reasoning; "
            "general reasoning requires cross-family and language transfer."
        ),
        "config": asdict(config),
        "device": str(device),
        "board": {
            "examples": len(board),
            "train_examples": len(train_indices),
            "development_examples": len(development_indices),
            "score_arms": list(SCORE_ARMS),
        },
        "parameter_receipt": checkpoint["parameter_receipt"],
        "warm_start": warm_start,
        "trace": trace,
        "halt_trace": halt_trace,
        "train": train_receipt,
        "development": development_receipt,
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
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--hard-fraction", type=float, default=0.10)
    parser.add_argument("--write-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--binder-checkpoint")
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
