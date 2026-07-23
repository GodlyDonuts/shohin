#!/usr/bin/env python3
"""Oracle-only assessor for committed CTAA binding-completion predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from ctaa_binding_completion import (
    audit_parity_rows,
    materialize_factorized,
    materialize_whole,
)
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from ctaa_compiler_training import TokenizedCompilerRow
from predict_ctaa_binding_completion import (
    PREDICTION_KEYS,
    SCHEMA as PREDICTION_SCHEMA,
    SEED_PREDICTION_KEYS,
    load_seed_freeze,
)
from train_ctaa_binding_completion import (
    metrics_from_logits,
    safe_torch_load,
    sha256_file,
    tensor_sha256,
    validate_frozen_seed,
    write_once,
)
from ctaa_trunk_compiler import HardCTAAPacket


SCHEMA = "r12_ctaa_a4_binding_completion_assessment_v1"
BOARD_SCHEMA = "r12_ctaa_a4_binding_completion_board_v1"
ORACLE_KEYS = {
    "row_id",
    "family_id",
    "query_source",
    "action_cards",
    "opcode_to_card",
    "initial_state",
    "opcode_schedule",
    "schedule",
    "query_position",
    "renderer",
}


def open_oracle_once(
    path: Path,
    expected_sha256: str,
) -> tuple[list[dict[str, object]], str]:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != expected_sha256:
        raise ValueError("CTAA completion oracle commitment differs")
    rows = []
    for line_number, line in enumerate(encoded.splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != ORACLE_KEYS:
            raise ValueError(
                f"CTAA completion oracle row {line_number} schema differs"
            )
        rows.append(value)
    if not rows:
        raise ValueError("CTAA completion oracle is empty")
    return rows, digest


def claim_oracle_access(
    path: Path,
    *,
    admission_sha256: str,
    oracle_sha256: str,
    prediction_sha256: str,
    assessment_output: Path,
    code_commit: str,
    protocol_source_sha256: str,
) -> str:
    payload = {
        "schema": "r12_ctaa_a4_binding_completion_oracle_access_v1",
        "admission_sha256": admission_sha256,
        "oracle_sha256": oracle_sha256,
        "prediction_sha256": prediction_sha256,
        "assessment_output": str(assessment_output.resolve()),
        "code_commit": code_commit,
        "protocol_source_sha256": protocol_source_sha256,
        "access_number": 1,
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.chmod(0o600)
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def oracle_bindings(
    rows: Sequence[dict[str, object]],
) -> torch.Tensor:
    result = torch.tensor([row["opcode_to_card"] for row in rows], dtype=torch.long)
    synthetic = [
        TokenizedCompilerRow(
            program_ids=(index + 2,),
            query_ids=(2,),
            action_cards=((0, 1, 2),) * 4,
            opcode_to_card=tuple(int(item) for item in binding),
            initial_state=(0, 1, 2),
            opcode_schedule=(4, *([0] * 40)),
            schedule=(4, *([binding[0]] * 40)),
            query_position=0,
        )
        for index, binding in enumerate(result.tolist())
    ]
    audit_parity_rows(synthetic, expected_parity=1)
    return result


def packet_metrics(
    common_logits: dict[str, torch.Tensor],
    binding_logits: torch.Tensor,
    oracle_rows: Sequence[dict[str, object]],
    *,
    arm: str,
) -> dict[str, float]:
    cards = common_logits["action_cards"].argmax(-1).to(torch.uint8)
    initial = common_logits["initial_state"].argmax(-1).to(torch.uint8)
    opcode_schedule = common_logits["opcode_schedule"].argmax(-1).to(torch.uint8)
    binding = (
        materialize_whole(binding_logits)
        if arm == "whole"
        else materialize_factorized(binding_logits)
    ).to(torch.uint8)
    oracle_cards = torch.tensor(
        [row["action_cards"] for row in oracle_rows],
        dtype=torch.uint8,
    )
    oracle_initial = torch.tensor(
        [row["initial_state"] for row in oracle_rows],
        dtype=torch.uint8,
    )
    oracle_opcode = torch.tensor(
        [row["opcode_schedule"] for row in oracle_rows],
        dtype=torch.uint8,
    )
    oracle_schedule = torch.tensor(
        [row["schedule"] for row in oracle_rows],
        dtype=torch.uint8,
    )
    valid = opcode_schedule.eq(4).sum(-1).eq(1)
    resolved = binding.long().gather(
        1,
        opcode_schedule.long().clamp_max(3),
    ).to(torch.uint8)
    resolved = torch.where(
        opcode_schedule.eq(4),
        opcode_schedule,
        resolved,
    )
    active = ~opcode_schedule.eq(4)
    persistent_excitation = torch.stack(
        [
            opcode_schedule.eq(opcode).logical_and(active).any(1)
            for opcode in range(4)
        ],
        dim=1,
    ).all(1)
    counterfactual_binding = binding.clone()
    counterfactual_binding[:, 0], counterfactual_binding[:, 1] = (
        binding[:, 1],
        binding[:, 0],
    )
    counterfactual_resolved = counterfactual_binding.long().gather(
        1,
        opcode_schedule.long().clamp_max(3),
    ).to(torch.uint8)
    counterfactual_resolved = torch.where(
        opcode_schedule.eq(4),
        opcode_schedule,
        counterfactual_resolved,
    )
    counterfactual_effect = (
        counterfactual_resolved.ne(resolved).logical_and(active).any(1)
    )
    cards_exact = cards.eq(oracle_cards).flatten(1).all(1)
    binding_exact = binding.eq(
        torch.tensor(
            [row["opcode_to_card"] for row in oracle_rows],
            dtype=torch.uint8,
        )
    ).all(1)
    initial_exact = initial.eq(oracle_initial).all(1)
    opcode_exact = opcode_schedule.eq(oracle_opcode).all(1)
    schedule_exact = resolved.eq(oracle_schedule).all(1)
    if bool(valid.any()):
        packet = HardCTAAPacket(
            action_cards=cards[valid],
            opcode_to_card=binding[valid],
            initial_state=initial[valid],
            opcode_schedule=opcode_schedule[valid],
        )
        if packet.bytes_per_row != 60:
            raise AssertionError("CTAA completion materialized packet size differs")
    program_exact = (
        valid
        & cards_exact
        & binding_exact
        & initial_exact
        & opcode_exact
        & schedule_exact
    )
    return {
        "packet_valid": float(valid.float().mean()),
        "cards_exact": float(cards_exact.float().mean()),
        "binding_exact": float(binding_exact.float().mean()),
        "initial_exact": float(initial_exact.float().mean()),
        "opcode_schedule_exact": float(opcode_exact.float().mean()),
        "resolved_schedule_exact": float(schedule_exact.float().mean()),
        "opcode_persistent_excitation": float(
            persistent_excitation.float().mean()
        ),
        "binding_counterfactual_effect": float(
            counterfactual_effect.float().mean()
        ),
        "program_exact": float(program_exact.float().mean()),
        "packet_bytes_per_valid_row": 60.0,
    }


def assess(
    *,
    prediction_path: Path,
    admission_path: Path,
    board_manifest_path: Path,
    confirmation_oracle_path: Path,
    output: Path,
    device_name: str,
) -> dict[str, object]:
    admission = load_admission(admission_path)
    require_admitted_protocol_source(admission)
    require_admitted_artifact_path(
        output,
        admission,
        "assessment_artifact_name",
    )
    require_admitted_artifact_path(
        prediction_path,
        admission,
        "prediction_artifact_name",
    )
    if output.exists():
        raise FileExistsError(
            f"refusing existing CTAA completion assessment: {output}"
        )
    prediction, prediction_sha256 = safe_torch_load(prediction_path)
    admission_sha256 = sha256_file(admission_path)
    if (
        set(prediction) != PREDICTION_KEYS
        or prediction.get("schema") != PREDICTION_SCHEMA
        or prediction.get("confirmation_oracle_access") != 0
        or prediction.get("admission_sha256") != admission_sha256
    ):
        raise ValueError("CTAA completion prediction commitment differs")
    manifest = json.loads(board_manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != BOARD_SCHEMA:
        raise ValueError("CTAA completion assessment board schema differs")
    if prediction.get("board_manifest_sha256") != sha256_file(
        board_manifest_path
    ):
        raise ValueError("CTAA completion prediction board differs")
    if sha256_file(board_manifest_path) != admission["board_manifest_sha256"]:
        raise ValueError("CTAA completion admission board differs")
    if manifest.get("confirmation_odd_oracle_sha256") != admission[
        "confirmation_oracle_sha256"
    ]:
        raise ValueError("CTAA completion admission oracle differs")

    seed_freeze_path = (
        Path(str(admission["custody_root"]))
        / str(admission["seed_freeze_manifest_name"])
    )
    freeze_records, freeze_sha256 = load_seed_freeze(
        seed_freeze_path,
        admission_sha256=admission_sha256,
    )
    if freeze_sha256 != prediction["seed_freeze_sha256"]:
        raise ValueError("CTAA completion prediction seed freeze differs")
    seed_predictions = prediction.get("seed_predictions")
    if not isinstance(seed_predictions, list) or len(seed_predictions) != 5:
        raise ValueError("CTAA completion prediction seed lattice differs")
    frozen_seeds = []
    for index, (predicted, record, expected_seed) in enumerate(
        zip(
            seed_predictions,
            freeze_records,
            admission["seeds"],
            strict=True,
        )
    ):
        if (
            not isinstance(predicted, dict)
            or set(predicted) != SEED_PREDICTION_KEYS
            or predicted.get("seed") != expected_seed
            or record.get("seed") != expected_seed
            or record.get("index") != index
            or predicted.get("frozen_seed_sha256")
            != record.get("artifact_sha256")
        ):
            raise ValueError("CTAA completion seed prediction identity differs")
        seed_path = Path(str(predicted["frozen_seed_path"]))
        if (
            seed_path.resolve().parent
            != Path(str(admission["custody_root"]))
            or seed_path.name != record.get("artifact_name")
        ):
            raise ValueError("CTAA completion frozen seed custody differs")
        frozen, _ = safe_torch_load(
            seed_path,
            expected_sha256=str(record["artifact_sha256"]),
        )
        validate_frozen_seed(
            frozen,
            admission=admission,
            admission_sha256=admission_sha256,
            expected_seed=int(expected_seed),
        )
        frozen_seeds.append(frozen)

    ledger_path = (
        Path(str(admission["custody_root"]))
        / str(admission["oracle_access_ledger_name"])
    )
    ledger_sha256 = claim_oracle_access(
        ledger_path,
        admission_sha256=admission_sha256,
        oracle_sha256=str(manifest["confirmation_odd_oracle_sha256"]),
        prediction_sha256=prediction_sha256,
        assessment_output=output,
        code_commit=str(admission["code_commit"]),
        protocol_source_sha256=str(admission["protocol_source_sha256"]),
    )
    oracle_rows, oracle_sha256 = open_oracle_once(
        confirmation_oracle_path,
        str(manifest["confirmation_odd_oracle_sha256"]),
    )
    row_ids = [str(row["row_id"]) for row in oracle_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("CTAA completion oracle row identities overlap")
    if row_ids != prediction.get("ordered_row_ids"):
        raise ValueError("CTAA completion oracle row order differs")
    family_ids = [str(row["family_id"]) for row in oracle_rows]
    if family_ids != prediction.get("ordered_family_ids"):
        raise ValueError("CTAA completion oracle order differs")
    if len(oracle_rows) != manifest.get("confirmation_odd_oracle_rows_written"):
        raise ValueError("CTAA completion oracle count differs")
    bindings = oracle_bindings(oracle_rows)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA completion assessment requires available CUDA")

    seed_results = []
    for predicted, frozen in zip(
        seed_predictions,
        frozen_seeds,
        strict=True,
    ):
        confirmation_slots = predicted["confirmation_slot_cache"].float()
        if tensor_sha256(confirmation_slots) != predicted.get(
            "confirmation_slot_cache_sha256"
        ):
            raise ValueError("CTAA completion confirmation cache hash differs")
        if confirmation_slots.shape[0] != bindings.shape[0]:
            raise ValueError("CTAA completion confirmation cache rows differ")
        main_metrics = {
            arm: metrics_from_logits(
                logits,
                bindings,
                arm=arm,
            )
            for arm, logits in predicted["arm_logits"].items()
        }
        program_metrics = {
            arm: packet_metrics(
                predicted["common_program_logits"],
                logits,
                oracle_rows,
                arm=arm,
            )
            for arm, logits in predicted["arm_logits"].items()
        }
        probe_metrics = {
            label: metrics_from_logits(
                logits,
                bindings,
                arm="global_structured",
            )
            for label, logits in predicted["single_slot_probe_logits"].items()
        }
        chimera_metrics = frozen["training"].get(
            "a4_derived_odd_chimera_metrics"
        )
        if not isinstance(chimera_metrics, dict):
            raise ValueError("CTAA completion frozen chimera receipt differs")
        seed_results.append(
            {
                "seed": int(predicted["seed"]),
                "confirmation_metrics": main_metrics,
                "program_packet_metrics": program_metrics,
                "single_slot_probe_metrics": probe_metrics,
                "two_slot_chimera_metrics": chimera_metrics,
            }
        )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "binding_completion_diagnostic_not_recurrent_reasoning"
        ),
        "prediction_sha256": sha256_file(prediction_path),
        "admission_sha256": sha256_file(admission_path),
        "board_manifest_sha256": sha256_file(board_manifest_path),
        "confirmation_oracle_sha256": oracle_sha256,
        "oracle_access_ledger_sha256": ledger_sha256,
        "rows": len(oracle_rows),
        "ordered_row_ids": row_ids,
        "ordered_family_ids": family_ids,
        "oracle_rows": oracle_rows,
        "oracle_bindings": bindings,
        "seed_results": seed_results,
        "confirmation_source_access_in_prediction": 1,
        "confirmation_oracle_access": 1,
        "all_s4_capacity_gate": "pending_separate_disposable_post_assessment_job",
        "valid_for_binding_attribution": False,
    }
    digest = write_once(output, payload)
    return {
        "assessment_sha256": digest,
        "rows": len(oracle_rows),
        "seeds": [result["seed"] for result in seed_results],
        "confirmation_oracle_access": 1,
        "valid_for_binding_attribution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = assess(
        prediction_path=args.predictions,
        admission_path=args.admission,
        board_manifest_path=args.board_manifest,
        confirmation_oracle_path=args.confirmation_oracle,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
