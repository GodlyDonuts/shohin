"""Evaluate a learned contextual binder inside the private Bekić executor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from contextual_bekic_graph_machine import (
    MAX_OPERATION_SLOTS,
    ContextualBekicGraphMachine,
    LateContextualProgramQuery,
)
from contextual_witness_equivariant_binder import (
    ContextualWitnessEquivariantBinder,
    ContextualWitnessStatisticsBinder,
)
from contextualize_bekic_program import (
    contextualize_simultaneous_packet,
    identify_contextual_slots,
)
from contrastive_bekic_program_orbits import (
    ISOLATED_COUNTERFACTUAL_ARMS,
    assert_split_disjoint,
    evaluate_simultaneous,
    generate_train_development,
    select_isolated_counterfactual_input,
    select_machine_input,
)
from tensorize_contextual_bekic import (
    tensorize_contextual_packets,
    tensorize_target_environment,
)


SCORE_ARMS = ("p", "p_prime", "p_eq", *ISOLATED_COUNTERFACTUAL_ARMS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("protocol") != "contextual_witness_equivariant_binder_v1":
        raise ValueError("binder checkpoint protocol differs")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("binder checkpoint payload differs")
    architecture = str(config.get("architecture", "equivariant"))
    if architecture == "equivariant":
        model = ContextualWitnessEquivariantBinder(
            width=int(config["width"]),
            rounds=int(config["rounds"]),
            triad_mode=str(config.get("triad_mode", "learned")),
        )
    elif architecture == "statistics":
        model = ContextualWitnessStatisticsBinder(
            width=int(config["width"]),
        )
    else:
        raise ValueError("binder checkpoint architecture differs")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


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


def _gold_assignment(
    contextual: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    identified = identify_contextual_slots(contextual)
    output = torch.full(
        (MAX_OPERATION_SLOTS,),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, card in enumerate(contextual["operation_cards"]):
        output[index] = identified[str(card["slot"])]
    return output


def _target(
    source: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    variables = [str(item) for item in source["program"]["variables"]]
    environment = evaluate_simultaneous(source)
    return tensorize_target_environment(
        environment,
        variables,
        cardinality=int(source["cardinality"]),
        device=device,
    )


def _metrics(
    labels: list[tuple[str, str, str]],
    values: list[bool],
) -> dict[str, dict[str, int | float]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    correct: Counter[tuple[str, str, str]] = Counter()
    for label, value in zip(labels, values, strict=True):
        counts[label] += 1
        correct[label] += int(value)
    return {
        ":".join(label): {
            "correct": correct[label],
            "count": counts[label],
            "accuracy": correct[label] / counts[label],
        }
        for label in sorted(counts)
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    seed: int,
    train_count: int,
    development_count: int,
    device: torch.device,
) -> dict[str, Any]:
    model, checkpoint = _load_model(checkpoint_path, device)
    train, development = generate_train_development(
        train_count=train_count,
        development_count=development_count,
        seed=seed,
    )
    assert_split_disjoint(train, development)
    machine = ContextualBekicGraphMachine().to(device)
    labels: list[tuple[str, str, str]] = []
    binding_exact: list[bool] = []
    terminal_exact: list[bool] = []
    rejected_active_cards = 0
    active_cards = 0
    per_class_total = Counter()
    per_class_correct = Counter()

    for split, rows in (("train", train), ("development", development)):
        for row_index, row in enumerate(rows):
            cell = str(row["axes"]["cell"])
            for arm_index, arm in enumerate(SCORE_ARMS):
                source = _source_packet(row, arm)
                contextual = contextualize_simultaneous_packet(
                    source,
                    seed=seed
                    + 1_000_000
                    + row_index * 100
                    + arm_index,
                )
                tensors = tensorize_contextual_packets(
                    [contextual],
                    device=device,
                )
                with torch.no_grad():
                    binding = model(
                        tensors.witness_left,
                        tensors.witness_right,
                        tensors.witness_output,
                        tensors.witness_mask,
                        tensors.argument_mask,
                        tensors.object_mask,
                        hard=True,
                    )
                gold = _gold_assignment(contextual, device)
                prediction = binding.logits.argmax(-1)[0]
                active = gold.ge(0)
                exact_binding = bool(prediction[active].eq(gold[active]).all())
                labels.append((split, cell, arm))
                binding_exact.append(exact_binding)
                active_cards += int(active.sum())
                rejected_active_cards += int(binding.rejected[0, active].sum())
                for class_index in gold[active].tolist():
                    per_class_total[int(class_index)] += 1
                for class_index in gold[
                    active & prediction.eq(gold)
                ].tolist():
                    per_class_correct[int(class_index)] += 1

                if bool(binding.rejected[0, active].any()):
                    terminal_exact.append(False)
                    continue
                query = LateContextualProgramQuery(
                    variable=torch.zeros(1, dtype=torch.long, device=device),
                    position=torch.zeros(1, dtype=torch.long, device=device),
                )
                with torch.no_grad():
                    rollout = machine(
                        tensors.packet,
                        binding.discrete_assignment,
                        query,
                    )
                terminal_exact.append(
                    bool(
                        rollout.terminal_variables[0]
                        .eq(_target(source, device))
                        .all()
                    )
                )

    return {
        "schema": "contextual_witness_bekic_integration_eval_v1",
        "claim_boundary": (
            "The learned score path receives only source-deleted cards and graph "
            "tensors. Analytic primitive identification is used after prediction "
            "for scoring only. The private executor remains a bounded host "
            "mechanism, so this is a contextual-binding result, not general "
            "reasoning."
        ),
        "seed": seed,
        "train_orbits": len(train),
        "development_orbits": len(development),
        "score_arms": list(SCORE_ARMS),
        "packets": len(labels),
        "active_cards": active_cards,
        "rejected_active_cards": rejected_active_cards,
        "binding_exact_packets": sum(binding_exact),
        "binding_exact_rate": sum(binding_exact) / len(binding_exact),
        "terminal_exact_packets": sum(terminal_exact),
        "terminal_exact_rate": sum(terminal_exact) / len(terminal_exact),
        "binding_metrics": _metrics(labels, binding_exact),
        "terminal_metrics": _metrics(labels, terminal_exact),
        "per_class": {
            str(class_index): {
                "correct": per_class_correct[class_index],
                "count": per_class_total[class_index],
                "accuracy": per_class_correct[class_index]
                / max(per_class_total[class_index], 1),
            }
            for class_index in range(5)
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "config": checkpoint["config"],
            "parameter_receipt": checkpoint["parameter_receipt"],
            "source_sha256": checkpoint["source_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026072331)
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--development-count", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing report: {args.out}")
    report = evaluate_checkpoint(
        args.checkpoint,
        seed=args.seed,
        train_count=args.train_count,
        development_count=args.development_count,
        device=torch.device(args.device),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
