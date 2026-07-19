#!/usr/bin/env python3
"""Sole unchanged-weight S7 confirmation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval_s7_learned_cayley import (
    _load_rows,
    _rename_operations,
    _run_generator,
    _run_transformer,
    _score_states,
    _sha256,
)
from s6_contextual_affine_law_inducer import ContextualAffineLawInducer
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI


EXPECTED_CHECKPOINT_SHA256 = (
    "c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607"
)
EXPECTED_DEVELOPMENT_ASSESSMENT_SHA256 = (
    "2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4"
)
EXPECTED_CONFIRMATION_SHA256 = (
    "c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-assessment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 confirmation evaluation: {args.out}")
    if _sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("S7 confirmation checkpoint hash mismatch")
    if (
        _sha256(args.development_assessment)
        != EXPECTED_DEVELOPMENT_ASSESSMENT_SHA256
    ):
        raise SystemExit("S7 development assessment hash mismatch")
    development_assessment = json.loads(args.development_assessment.read_text())
    if (
        development_assessment.get("decision")
        != "qualify_s7_learned_cayley_for_fresh_confirmation"
    ):
        raise SystemExit("S7 development did not authorize confirmation")

    report = json.loads((args.data_dir / "report.json").read_text())
    confirmation_path = args.data_dir / "confirmation.sealed.jsonl"
    if _sha256(confirmation_path) != EXPECTED_CONFIRMATION_SHA256:
        raise SystemExit("S7 sealed confirmation hash mismatch")
    rows = _load_rows(args.data_dir, "confirmation.sealed.jsonl", report)
    if len(rows) != 2048:
        raise SystemExit("S7 confirmation row count mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s7_learned_cayley_checkpoint_v1":
        raise SystemExit("S7 checkpoint schema mismatch")
    if checkpoint["board_report_sha256"] != _sha256(args.data_dir / "report.json"):
        raise SystemExit("S7 confirmation checkpoint/board mismatch")

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
    true_zeros = {
        modulus: treatment.discrete_zero(modulus) for modulus in PRIMARY_MODULI
    }
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
    host_states = [[int(value) for value in row["final_state"]] for row in rows]
    host = _score_states(rows, host_states, [])
    host["samples"] = []
    evaluation = {
        "schema": "r12_s7_learned_cayley_confirmation_evaluation_v1",
        "checkpoint_sha256": _sha256(args.checkpoint),
        "development_assessment_sha256": _sha256(args.development_assessment),
        "board_report_sha256": _sha256(args.data_dir / "report.json"),
        "confirmation_sha256": _sha256(confirmation_path),
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
        "confirmation_accesses": 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "treatment_state": treatment_score["state_accuracy"],
                "treatment_answer": treatment_score["answer_accuracy"],
                "ordinary_transformer_state": evaluation["arms"][
                    "ordinary_transformer"
                ]["state_accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
