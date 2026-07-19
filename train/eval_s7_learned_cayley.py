#!/usr/bin/env python3
"""One-read S7 development evaluation with frozen causal controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from s6_contextual_affine_law_inducer import ContextualAffineLawInducer
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI
from s7_learned_cayley_law import compile_destination, pop_insert


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha(states: list[list[int]]) -> str:
    payload = json.dumps(states, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_rows(data_dir: Path, name: str, report: dict[str, object]) -> list[dict[str, object]]:
    path = data_dir / name
    if _sha256(path) != report["files"][name]["sha256"]:
        raise SystemExit(f"S7 evaluation hash mismatch: {name}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _cards(row: dict[str, object], mode: str, successor: tuple[int, ...]) -> dict[str, dict[str, int]]:
    cards = {
        str(name): {
            "card_y0": int(value["card_y0"]),
            "card_y1": int(value["card_y1"]),
        }
        for name, value in row["law_cards"].items()
    }
    if mode == "normal":
        return cards
    if mode == "deranged":
        names = sorted(cards)
        values = [cards[name] for name in names]
        return {name: values[(index + 1) % len(values)] for index, name in enumerate(names)}
    if mode == "one_witness":
        return {
            name: {
                "card_y0": value["card_y0"],
                "card_y1": successor[value["card_y0"]],
            }
            for name, value in cards.items()
        }
    raise ValueError(f"unknown S7 card mode: {mode}")


def _score_states(rows: list[dict[str, object]], states: list[list[int]], samples: list[dict[str, object]]) -> dict[str, object]:
    state_correct = 0
    answer_correct = 0
    depth: dict[str, dict[str, int | float]] = {}
    for row, state in zip(rows, states, strict=True):
        expected = [int(value) for value in row["final_state"]]
        query = int(row["query_position"])
        state_ok = state == expected
        answer_ok = state[query] == int(row["answer"])
        state_correct += int(state_ok)
        answer_correct += int(answer_ok)
        key = str(row["depth"])
        cell = depth.setdefault(key, {"correct": 0, "total": 0, "accuracy": 0.0})
        cell["correct"] = int(cell["correct"]) + int(state_ok)
        cell["total"] = int(cell["total"]) + 1
    for cell in depth.values():
        cell["accuracy"] = int(cell["correct"]) / int(cell["total"])
    return {
        "state_correct": state_correct,
        "state_total": len(rows),
        "state_accuracy": state_correct / len(rows),
        "answer_correct": answer_correct,
        "answer_total": len(rows),
        "answer_accuracy": answer_correct / len(rows),
        "depth_state": dict(sorted(depth.items(), key=lambda item: int(item[0]))),
        "predicted_state_sha256": _state_sha(states),
        "samples": samples[:5],
    }


def _run_generator(
    rows: list[dict[str, object]],
    generators: dict[int, tuple[int, ...]],
    zeros: dict[int, int],
    card_mode: str = "normal",
    reset_state: bool = False,
) -> dict[str, object]:
    final_states: list[list[int]] = []
    samples: list[dict[str, object]] = []
    for row in rows:
        modulus = int(row["modulus"])
        successor = generators[modulus]
        zero = zeros[modulus]
        cards = _cards(row, card_mode, successor)
        initial = [int(value) for value in row["initial_state"]]
        state = list(initial)
        transitions: list[list[int]] = []
        for event in row["events"]:
            if reset_state:
                state = list(initial)
            identity = int(event["identity"])
            source = state.index(identity)
            card = cards[str(event["operation"])]
            destination = compile_destination(
                successor,
                zero,
                card["card_y0"],
                card["card_y1"],
                source,
            )
            state = list(pop_insert(state, identity, destination))
            transitions.append(list(state))
        final_states.append(state)
        samples.append(
            {
                "row_id": row["row_id"],
                "expected_state": row["final_state"],
                "predicted_state": state,
                "expected_answer": row["answer"],
                "predicted_answer": state[int(row["query_position"])],
                "transitions": transitions,
            }
        )
    return _score_states(rows, final_states, samples)


def _run_transformer(
    rows: list[dict[str, object]],
    model: ContextualAffineLawInducer,
    device: torch.device,
) -> dict[str, object]:
    states = [[int(value) for value in row["initial_state"]] for row in rows]
    transitions: list[list[list[int]]] = [[] for _ in rows]
    max_depth = max(int(row["depth"]) for row in rows)
    model.eval()
    with torch.no_grad():
        for step in range(max_depth):
            active = [index for index, row in enumerate(rows) if step < len(row["events"])]
            moduli: list[int] = []
            y0s: list[int] = []
            y1s: list[int] = []
            locations: list[int] = []
            identities: list[int] = []
            for index in active:
                row = rows[index]
                event = row["events"][step]
                identity = int(event["identity"])
                card = row["law_cards"][str(event["operation"])]
                moduli.append(int(row["modulus"]))
                y0s.append(int(card["card_y0"]))
                y1s.append(int(card["card_y1"]))
                locations.append(states[index].index(identity))
                identities.append(identity)
            logits = model(
                torch.tensor(moduli, dtype=torch.long, device=device),
                torch.tensor(y0s, dtype=torch.long, device=device),
                torch.tensor(y1s, dtype=torch.long, device=device),
                torch.tensor(locations, dtype=torch.long, device=device),
            )
            destinations = logits.argmax(-1).detach().cpu().tolist()
            for index, identity, destination in zip(
                active, identities, destinations, strict=True
            ):
                states[index] = list(pop_insert(states[index], identity, int(destination)))
                transitions[index].append(list(states[index]))
    samples = [
        {
            "row_id": row["row_id"],
            "expected_state": row["final_state"],
            "predicted_state": state,
            "expected_answer": row["answer"],
            "predicted_answer": state[int(row["query_position"])],
            "transitions": history,
        }
        for row, state, history in zip(rows, states, transitions, strict=True)
    ]
    return _score_states(rows, states, samples)


def _atomic_score(
    rows: list[dict[str, object]],
    generators: dict[int, tuple[int, ...]],
    zeros: dict[int, int],
    transformer: ContextualAffineLawInducer,
    device: torch.device,
) -> dict[str, object]:
    treatment_correct = 0
    for row in rows:
        modulus = int(row["modulus"])
        predicted = compile_destination(
            generators[modulus],
            zeros[modulus],
            int(row["card_y0"]),
            int(row["card_y1"]),
            int(row["current_location"]),
        )
        treatment_correct += int(predicted == int(row["destination"]))
    with torch.no_grad():
        transformer_predictions = transformer(
            torch.tensor([int(row["modulus"]) for row in rows], dtype=torch.long, device=device),
            torch.tensor([int(row["card_y0"]) for row in rows], dtype=torch.long, device=device),
            torch.tensor([int(row["card_y1"]) for row in rows], dtype=torch.long, device=device),
            torch.tensor(
                [int(row["current_location"]) for row in rows],
                dtype=torch.long,
                device=device,
            ),
        ).argmax(-1).detach().cpu().tolist()
    transformer_correct = sum(
        int(prediction == int(row["destination"]))
        for prediction, row in zip(transformer_predictions, rows, strict=True)
    )
    return {
        "treatment": {
            "correct": treatment_correct,
            "total": len(rows),
            "accuracy": treatment_correct / len(rows),
        },
        "ordinary_transformer": {
            "correct": transformer_correct,
            "total": len(rows),
            "accuracy": transformer_correct / len(rows),
        },
    }


def _rename_operations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    renamed = copy.deepcopy(rows)
    for row in renamed:
        mapping = {
            old: f"recoded_{index:03d}"
            for index, old in enumerate(sorted(row["law_cards"]))
        }
        row["law_cards"] = {
            mapping[old]: card for old, card in row["law_cards"].items()
        }
        for event in row["events"]:
            event["operation"] = mapping[event["operation"]]
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 evaluation: {args.out}")

    report = json.loads((args.data_dir / "report.json").read_text())
    if report.get("decision") != "admit_s7_learned_cayley_board":
        raise SystemExit("S7 board is not admitted")
    rows = _load_rows(args.data_dir, "development.jsonl", report)
    atomic_rows = _load_rows(args.data_dir, "atomic_development.jsonl", report)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s7_learned_cayley_checkpoint_v1":
        raise SystemExit("S7 checkpoint schema mismatch")
    if checkpoint["board_report_sha256"] != _sha256(args.data_dir / "report.json"):
        raise SystemExit("S7 checkpoint/board mismatch")

    device = torch.device(args.device)
    treatment = LearnedCayleyGenerator().to(device)
    false_generator = LearnedCayleyGenerator().to(device)
    transformer = ContextualAffineLawInducer().to(device)
    treatment.load_state_dict(checkpoint["treatment_state"])
    false_generator.load_state_dict(checkpoint["false_generator_state"])
    transformer.load_state_dict(checkpoint["ordinary_transformer_state"])
    true_successors = {
        modulus: treatment.discrete_successor(modulus) for modulus in PRIMARY_MODULI
    }
    true_zeros = {modulus: treatment.discrete_zero(modulus) for modulus in PRIMARY_MODULI}
    false_successors = {
        modulus: false_generator.discrete_successor(modulus)
        for modulus in PRIMARY_MODULI
    }
    false_zeros = {
        modulus: false_generator.discrete_zero(modulus) for modulus in PRIMARY_MODULI
    }

    treatment_score = _run_generator(rows, true_successors, true_zeros)
    renamed_score = _run_generator(
        _rename_operations(rows), true_successors, true_zeros
    )
    host = {
        **_score_states(
            rows,
            [[int(value) for value in row["final_state"]] for row in rows],
            [],
        ),
        "samples": [],
    }
    evaluation = {
        "schema": "r12_s7_learned_cayley_development_evaluation_v1",
        "checkpoint_sha256": _sha256(args.checkpoint),
        "board_report_sha256": _sha256(args.data_dir / "report.json"),
        "parameters": {
            "treatment": checkpoint["treatment_parameters"],
            "whole_system": checkpoint["treatment_total_system_parameters"],
            "ordinary_transformer": checkpoint["ordinary_transformer_parameters"],
        },
        "fit": {
            "treatment": checkpoint["treatment_fit"],
            "false_generator": checkpoint["false_generator_fit"],
            "ordinary_transformer": checkpoint["ordinary_transformer_fit"],
        },
        "training_contract": checkpoint["training_contract"],
        "atomic_development": _atomic_score(
            atomic_rows, true_successors, true_zeros, transformer, device
        ),
        "arms": {
            "host": host,
            "treatment": treatment_score,
            "ordinary_transformer": _run_transformer(rows, transformer, device),
            "stride_two_generator": _run_generator(
                rows, false_successors, false_zeros
            ),
            "deranged_card": _run_generator(
                rows, true_successors, true_zeros, card_mode="deranged"
            ),
            "one_witness": _run_generator(
                rows, true_successors, true_zeros, card_mode="one_witness"
            ),
            "state_reset": _run_generator(
                rows, true_successors, true_zeros, reset_state=True
            ),
        },
        "nonce_operation_invariance": {
            "all_rows_bit_identical": (
                renamed_score["predicted_state_sha256"]
                == treatment_score["predicted_state_sha256"]
            ),
            "original_state_sha256": treatment_score["predicted_state_sha256"],
            "renamed_state_sha256": renamed_score["predicted_state_sha256"],
        },
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "atomic": evaluation["atomic_development"],
                "treatment": {
                    "state_accuracy": treatment_score["state_accuracy"],
                    "answer_accuracy": treatment_score["answer_accuracy"],
                    "depth_state": treatment_score["depth_state"],
                },
                "ordinary_transformer_state": evaluation["arms"]["ordinary_transformer"]["state_accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
