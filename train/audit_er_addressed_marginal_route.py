#!/usr/bin/env python3
"""Read-only scale audit for a closed addressed-marginal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, load_trainable_state
from er_relation_tensor_training import evaluate_arm, load_board_receipt, load_split
from pilot_er_addressed_marginal_relation_adapter import (
    initialize_addressed_marginal_relation,
)
from pilot_er_addressed_marginal_train_canary import SCHEMA as CHECKPOINT_SCHEMA
from pilot_er_dual_stream_train_canary import score_train_row, split_train_families
from pilot_sd_cst_byte_addressed import sha256_file


SCHEMA = "r12_er_addressed_marginal_route_read_only_audit_v1"
ORDINAL_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def atomic_json_save(value: object, path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def reduced_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "overall": metrics["overall"],
        "by_cardinality": metrics["by_cardinality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmed-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-assessment", type=Path, required=True)
    parser.add_argument("--witness-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--witness-confirmation-assessment", type=Path, required=True
    )
    parser.add_argument("--closed-checkpoint", type=Path, required=True)
    parser.add_argument("--closed-checkpoint-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing existing audit output: {args.out}")
    if sha256_file(args.closed_checkpoint) != args.closed_checkpoint_sha256:
        raise SystemExit("closed addressed checkpoint hash differs")
    closed = torch.load(args.closed_checkpoint, map_location="cpu", weights_only=False)
    if closed.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("closed addressed checkpoint schema differs")
    if int(closed.get("development_accesses", -1)) != 0 or int(
        closed.get("confirmation_accesses", -1)
    ) != 0:
        raise SystemExit("closed addressed checkpoint custody differs")

    board = load_board_receipt(args.data_dir)
    rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    _, probe_rows, split_receipt = split_train_families(
        rows,
        derived_seed(int(closed["seed"]), "dual-stream-train-probe-split"),
    )
    if split_receipt != closed["split"]:
        raise SystemExit("closed addressed probe split differs")
    scored_probe = [score_train_row(row) for row in probe_rows]

    model, parameters, _, receipt = initialize_addressed_marginal_relation(
        joint_checkpoint=args.joint_checkpoint,
        physical_checkpoint=args.physical_checkpoint,
        v1_checkpoint=args.v1_checkpoint,
        v1_2_checkpoint=args.v1_2_checkpoint,
        confirmed_checkpoint=args.confirmed_checkpoint,
        confirmation_assessment=args.confirmation_assessment,
        witness_checkpoint=args.witness_checkpoint,
        witness_confirmation_assessment=args.witness_confirmation_assessment,
        seed=int(closed["seed"]),
        device=torch.device("cuda"),
    )
    if parameters != closed["parameters"]:
        raise SystemExit("closed addressed parameter certificate differs")
    load_trainable_state(model, closed["compiler_trainable_state"])
    ordinal = model.er_am_candidate_ordinal_embedding.weight.detach().clone()
    count = model.er_am_candidate_count_embedding.weight.detach().clone()

    arms: dict[str, object] = {}
    with torch.no_grad():
        for ordinal_scale in ORDINAL_SCALES:
            model.er_am_candidate_ordinal_embedding.weight.copy_(
                ordinal * ordinal_scale
            )
            model.er_am_candidate_count_embedding.weight.copy_(count)
            arms[f"ordinal_{ordinal_scale:g}_count_1"] = reduced_metrics(
                evaluate_arm(
                    model,
                    scored_probe,
                    batch_size=args.batch_size,
                    include_raw=False,
                    include_invariances=False,
                )
            )
        model.er_am_candidate_ordinal_embedding.weight.copy_(ordinal)
        model.er_am_candidate_count_embedding.weight.zero_()
        arms["ordinal_1_count_0"] = reduced_metrics(
            evaluate_arm(
                model,
                scored_probe,
                batch_size=args.batch_size,
                include_raw=False,
                include_invariances=False,
            )
        )
        model.er_am_candidate_ordinal_embedding.weight.zero_()
        arms["ordinal_0_count_0"] = reduced_metrics(
            evaluate_arm(
                model,
                scored_probe,
                batch_size=args.batch_size,
                include_raw=False,
                include_invariances=False,
            )
        )

    report: dict[str, object] = {
        "schema": SCHEMA,
        "closed_checkpoint_sha256": args.closed_checkpoint_sha256,
        "closed_source_commit": closed["source_manifest"]["commit"],
        "seed": int(closed["seed"]),
        "split": split_receipt,
        "parameters": parameters,
        "parent_receipt_sha256": hashlib.sha256(
            canonical_json(receipt).encode()
        ).hexdigest(),
        "ordinal_scales": list(ORDINAL_SCALES),
        "arms": arms,
        "custody": {
            "training_probe_read_only_audit": 1,
            "development_accesses": 0,
            "confirmation_accesses": 0,
        },
        "decision_boundary": (
            "This post-hoc scale audit diagnoses the closed train-only checkpoint. "
            "No arm is score-eligible and no result can authorize a fresh board."
        ),
    }
    atomic_json_save(report, args.out)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "report_sha256": sha256_file(args.out),
                "arms": {
                    name: {
                        metric: value["overall"][metric]["rate"]
                        for metric in (
                            "witness_pointer",
                            "relation_rows",
                            "joint",
                            "state",
                            "answer",
                        )
                    }
                    for name, value in arms.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
