#!/usr/bin/env python3
"""Run the sole S6 development read with frozen causal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from s6_contextual_affine_law import (
    ADMITTED_MODULI,
    infer_affine_law,
    pop_insert,
    split_laws,
)
from s6_contextual_affine_law_inducer import (
    ContextualAffineLawInducer,
    LawIdMemorizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _score_programs(
    rows: list[dict[str, object]],
    treatment: ContextualAffineLawInducer,
    law_id_control: LawIdMemorizer,
    device: torch.device,
    arm: str,
) -> dict[str, object]:
    states = [tuple(int(value) for value in row["initial_state"]) for row in rows]
    initial_states = list(states)
    transitions: list[list[list[int]]] = [[] for _ in rows]
    max_depth = max(int(row["depth"]) for row in rows)
    for step in range(max_depth):
        active = [index for index, row in enumerate(rows) if step < int(row["depth"])]
        if arm == "state_reset":
            for index in active:
                states[index] = initial_states[index]

        exact_destinations: list[int] = []
        moduli: list[int] = []
        card_y0: list[int] = []
        card_y1: list[int] = []
        locations: list[int] = []
        identities: list[int] = []
        for index in active:
            row = rows[index]
            event = row["events"][step]
            identity = int(event["identity"])
            name = str(event["operation"])
            cards = row["law_cards"]
            if arm == "deranged_card":
                names = sorted(cards)
                rotated = {name_: cards[names[(offset + 1) % len(names)]] for offset, name_ in enumerate(names)}
                card = rotated[name]
            else:
                card = cards[name]
            modulus = int(row["modulus"])
            location = states[index].index(identity)
            law = infer_affine_law(
                modulus, int(card["card_y0"]), int(card["card_y1"])
            )
            exact_destinations.append(law.destination(location))
            moduli.append(modulus)
            card_y0.append(int(card["card_y0"]))
            card_y1.append(
                int(card["card_y0"])
                if arm == "one_witness"
                else int(card["card_y1"])
            )
            locations.append(location)
            identities.append(identity)

        if arm == "host":
            destinations = exact_destinations
        else:
            with torch.no_grad():
                modulus_tensor = torch.tensor(moduli, dtype=torch.long, device=device)
                location_tensor = torch.tensor(locations, dtype=torch.long, device=device)
                if arm == "law_id":
                    logits = law_id_control(
                        modulus_tensor,
                        torch.full_like(modulus_tensor, law_id_control.oov_law_id),
                        location_tensor,
                    )
                else:
                    logits = treatment(
                        modulus_tensor,
                        torch.tensor(card_y0, dtype=torch.long, device=device),
                        torch.tensor(card_y1, dtype=torch.long, device=device),
                        location_tensor,
                    )
                destinations = [int(value) for value in logits.argmax(-1).cpu().tolist()]

        for local_index, row_index in enumerate(active):
            states[row_index] = pop_insert(
                states[row_index], identities[local_index], destinations[local_index]
            )
            transitions[row_index].append(list(states[row_index]))

    state_correct = 0
    answer_correct = 0
    depth_scores: dict[str, dict[str, int]] = {}
    multi_law_state_correct = 0
    multi_law_total = 0
    samples: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        expected_state = tuple(int(value) for value in row["final_state"])
        query_position = int(row["query_position"])
        state_ok = states[index] == expected_state
        answer = states[index][query_position]
        answer_ok = answer == int(row["answer"])
        state_correct += int(state_ok)
        answer_correct += int(answer_ok)
        depth = str(row["depth"])
        bucket = depth_scores.setdefault(depth, {"correct": 0, "total": 0})
        bucket["correct"] += int(state_ok)
        bucket["total"] += 1
        if int(row["distinct_laws"]) >= 2:
            multi_law_state_correct += int(state_ok)
            multi_law_total += 1
        if len(samples) < 5:
            samples.append(
                {
                    "row_id": row["row_id"],
                    "predicted_state": list(states[index]),
                    "expected_state": list(expected_state),
                    "predicted_answer": answer,
                    "expected_answer": int(row["answer"]),
                    "transitions": transitions[index],
                }
            )
    total = len(rows)
    predicted_state_sha256 = hashlib.sha256(
        json.dumps([list(state) for state in states], separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "state_correct": state_correct,
        "state_total": total,
        "state_accuracy": state_correct / total,
        "answer_correct": answer_correct,
        "answer_total": total,
        "answer_accuracy": answer_correct / total,
        "depth_state": {
            depth: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
            }
            for depth, counts in sorted(depth_scores.items(), key=lambda item: int(item[0]))
        },
        "multi_law_state_correct": multi_law_state_correct,
        "multi_law_total": multi_law_total,
        "multi_law_state_accuracy": multi_law_state_correct / multi_law_total,
        "predicted_state_sha256": predicted_state_sha256,
        "samples": samples,
    }


def _atomic_development_score(
    model: ContextualAffineLawInducer, device: torch.device
) -> dict[str, object]:
    rows: list[tuple[int, int, int, int, int]] = []
    for modulus in ADMITTED_MODULI:
        for law in split_laws(modulus)["development"]:
            for location in range(modulus):
                rows.append((modulus, law.card[0], law.card[1], location, law.destination(location)))
    with torch.no_grad():
        logits = model(
            torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row[2] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row[3] for row in rows], dtype=torch.long, device=device),
        )
    predictions = logits.argmax(-1).cpu()
    targets = torch.tensor([row[4] for row in rows])
    correct = int((predictions == targets).sum().item())
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def _renamed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    renamed: list[dict[str, object]] = []
    for row in rows:
        names = sorted(row["law_cards"])
        mapping = {name: f"renamed_{index}" for index, name in enumerate(reversed(names))}
        copy = dict(row)
        copy["law_cards"] = {
            mapping[name]: card for name, card in row["law_cards"].items()
        }
        copy["events"] = [
            {"operation": mapping[event["operation"]], "identity": event["identity"]}
            for event in row["events"]
        ]
        renamed.append(copy)
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S6 evaluation: {args.out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("S6 evaluation requires allocated CUDA")

    board_report = json.loads((args.data_dir / "report.json").read_text())
    for filename in ("development.jsonl", "scale_diagnostic.jsonl"):
        if _sha256(args.data_dir / filename) != board_report["files"][filename]["sha256"]:
            raise SystemExit(f"S6 {filename} hash mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s6_contextual_affine_law_checkpoint_v1":
        raise SystemExit("invalid S6 checkpoint schema")
    if checkpoint["development_accesses"] != 0 or checkpoint["confirmation_accesses"] != 0:
        raise SystemExit("S6 checkpoint access contract violated")
    if checkpoint["board_report_sha256"] != _sha256(args.data_dir / "report.json"):
        raise SystemExit("S6 checkpoint/board mismatch")

    treatment = ContextualAffineLawInducer().to(device)
    treatment.load_state_dict(checkpoint["treatment_state"], strict=True)
    treatment.eval()
    law_id_control = LawIdMemorizer(checkpoint["train_law_count"]).to(device)
    law_id_control.load_state_dict(checkpoint["law_id_control_state"], strict=True)
    law_id_control.eval()

    primary = _load_rows(args.data_dir / "development.jsonl")
    diagnostic = _load_rows(args.data_dir / "scale_diagnostic.jsonl")
    arms = {
        name: _score_programs(primary, treatment, law_id_control, device, name)
        for name in (
            "host",
            "treatment",
            "deranged_card",
            "one_witness",
            "state_reset",
            "law_id",
        )
    }
    renamed = _score_programs(
        _renamed_rows(primary), treatment, law_id_control, device, "treatment"
    )
    name_invariance = (
        arms["treatment"]["predicted_state_sha256"]
        == renamed["predicted_state_sha256"]
    )
    if not name_invariance:
        raise SystemExit("S6 nonce-law rename changed treatment score")

    diagnostic_score = _score_programs(
        diagnostic, treatment, law_id_control, device, "treatment"
    )
    evaluation = {
        "schema": "r12_s6_contextual_affine_law_development_eval_v1",
        "board_report_sha256": _sha256(args.data_dir / "report.json"),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "development_accesses": 1,
        "confirmation_accesses": 0,
        "atomic_development": _atomic_development_score(treatment, device),
        "arms": arms,
        "nonce_name_invariance": {
            "all_rows_bit_identical": name_invariance,
            "original_state_sha256": arms["treatment"]["predicted_state_sha256"],
            "renamed_state_sha256": renamed["predicted_state_sha256"],
        },
        "scale_diagnostic": diagnostic_score,
        "fit": {
            "treatment": checkpoint["treatment_fit"],
            "law_id_control": checkpoint["law_id_control_fit"],
        },
        "parameters": {
            "treatment": checkpoint["treatment_parameters"],
            "whole_system": checkpoint["total_system_parameters"],
            "law_id_control": checkpoint["law_id_control_parameters"],
        },
        "training_contract": checkpoint["training_contract"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "atomic": evaluation["atomic_development"],
                "treatment": arms["treatment"],
                "diagnostic_state_accuracy": diagnostic_score["state_accuracy"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
