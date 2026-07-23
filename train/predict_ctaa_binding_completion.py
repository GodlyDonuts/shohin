#!/usr/bin/env python3
"""Source-only prediction for five frozen CTAA A4 completion seeds.

This process accepts no oracle path and cannot compute accuracy. It opens the
sealed odd source once, commits raw logits and slot caches for every frozen
seed, and exits before any assessor may open labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from tokenizers import Tokenizer

from ctaa_artifact_loader import (
    TOKENIZER_SHA256,
    load_raw_trunk,
    require_sha256,
)
from ctaa_binding_completion import (
    FactorizedBindingReadout,
    GlobalStructuredBindingReadout,
    SingleSlotFullBindingProbe,
    WholePermutationReadout,
)
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from freeze_ctaa_binding_completion_seeds import (
    SCHEMA as SEED_FREEZE_SCHEMA,
)
from ctaa_trunk_compiler import TrunkCausalCTAACompiler
from train_ctaa_binding_completion import (
    load_state,
    safe_torch_load,
    sha256_file,
    tensor_sha256,
    validate_frozen_seed,
    write_once,
)


SCHEMA = "r12_ctaa_a4_binding_completion_predictions_v1"
BOARD_SCHEMA = "r12_ctaa_a4_binding_completion_board_v1"
SOURCE_KEYS = {"row_id", "family_id", "program_source"}
PREDICTION_KEYS = {
    "schema",
    "claim_boundary",
    "base_sha256",
    "admission_sha256",
    "seed_freeze_sha256",
    "tokenizer_sha256",
    "board_manifest_sha256",
    "confirmation_source_sha256",
    "ordered_row_ids",
    "ordered_family_ids",
    "ordered_program_sha256",
    "seed_predictions",
    "confirmation_source_access",
    "confirmation_oracle_access",
}
SEED_PREDICTION_KEYS = {
    "seed",
    "frozen_seed_path",
    "frozen_seed_sha256",
    "confirmation_slot_cache",
    "confirmation_slot_cache_sha256",
    "common_program_logits",
    "arm_logits",
    "single_slot_probe_logits",
}


def load_source_rows(
    path: Path,
    tokenizer: Tokenizer,
    max_length: int,
) -> tuple[list[str], list[str], list[tuple[int, ...]], list[str]]:
    row_ids = []
    family_ids = []
    token_rows = []
    source_hashes = []
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != SOURCE_KEYS:
                raise ValueError(
                    f"CTAA completion source row {line_number} schema differs"
                )
            source = str(value["program_source"])
            ids = tokenizer.encode(source).ids
            if not ids or len(ids) > max_length:
                raise ValueError(
                    f"CTAA completion source row {line_number} length differs"
                )
            row_ids.append(str(value["row_id"]))
            family_ids.append(str(value["family_id"]))
            token_rows.append(tuple(ids))
            source_hashes.append(
                hashlib.sha256(source.encode("utf-8")).hexdigest()
            )
    if not family_ids:
        raise ValueError("CTAA completion source file is empty")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("CTAA completion source row identities overlap")
    return row_ids, family_ids, token_rows, source_hashes


def open_source_once(
    path: Path,
    tokenizer: Tokenizer,
    max_length: int,
    expected_sha256: str,
) -> tuple[list[str], list[str], list[tuple[int, ...]], list[str], str]:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != expected_sha256:
        raise ValueError("CTAA completion source commitment differs")
    row_ids = []
    family_ids = []
    token_rows = []
    source_hashes = []
    for line_number, line in enumerate(encoded.splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != SOURCE_KEYS:
            raise ValueError(
                f"CTAA completion source row {line_number} schema differs"
            )
        source = str(value["program_source"])
        ids = tokenizer.encode(source).ids
        if not ids or len(ids) > max_length:
            raise ValueError(
                f"CTAA completion source row {line_number} length differs"
            )
        row_ids.append(str(value["row_id"]))
        family_ids.append(str(value["family_id"]))
        token_rows.append(tuple(ids))
        source_hashes.append(hashlib.sha256(source.encode("utf-8")).hexdigest())
    if not family_ids:
        raise ValueError("CTAA completion source file is empty")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("CTAA completion source row identities overlap")
    return row_ids, family_ids, token_rows, source_hashes, digest


def collate_source_ids(
    rows: Sequence[Sequence[int]],
    *,
    device: torch.device,
    padding_id: int = 1,
) -> torch.Tensor:
    if not rows:
        raise ValueError("CTAA completion source batch is empty")
    width = max(len(row) for row in rows)
    result = torch.full(
        (len(rows), width),
        padding_id,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
    return result


def restore_adapter_state(
    compiler: TrunkCausalCTAACompiler,
    state: Mapping[str, torch.Tensor],
) -> None:
    own = compiler.state_dict()
    expected = {name for name in own if not name.startswith("model.")}
    if set(state) != expected:
        raise ValueError("CTAA completion frozen compiler state differs")
    with torch.no_grad():
        for name, value in state.items():
            if own[name].shape != value.shape:
                raise ValueError("CTAA completion frozen compiler geometry differs")
            own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))


@torch.inference_mode()
def extract_source_slots(
    compiler: TrunkCausalCTAACompiler,
    token_rows: Sequence[Sequence[int]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    compiler.eval()
    slots = []
    cards = []
    initial = []
    opcode_schedule = []
    for start in range(0, len(token_rows), batch_size):
        ids = collate_source_ids(
            token_rows[start : start + batch_size],
            device=device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            bundle = compiler.encode_source(ids)
            program = compiler.compile_program_from_residuals(bundle)
            slots.append(
                compiler.binding_relation_slots_from_residuals(
                    bundle
                ).float().cpu()
            )
            cards.append(program.action_cards.float().cpu())
            initial.append(program.initial_state.float().cpu())
            opcode_schedule.append(program.opcode_schedule.float().cpu())
    return torch.cat(slots), {
        "action_cards": torch.cat(cards),
        "initial_state": torch.cat(initial),
        "opcode_schedule": torch.cat(opcode_schedule),
    }


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    slots: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model = model.to(device).eval()
    result = []
    for start in range(0, slots.shape[0], batch_size):
        result.append(
            model(slots[start : start + batch_size].to(device)).float().cpu()
        )
    return torch.cat(result)


def load_seed_freeze(
    path: Path,
    *,
    admission_sha256: str,
) -> tuple[list[dict[str, object]], str]:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    value = json.loads(encoded)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "admission_sha256",
            "code_commit",
            "protocol_source_sha256",
            "seed_records",
            "confirmation_source_access",
            "confirmation_oracle_access",
        }
        or value.get("schema") != SEED_FREEZE_SCHEMA
        or value.get("admission_sha256") != admission_sha256
        or value.get("confirmation_source_access") != 0
        or value.get("confirmation_oracle_access") != 0
    ):
        raise ValueError("CTAA completion seed freeze differs")
    records = value.get("seed_records")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("CTAA completion seed freeze lattice differs")
    record_keys = {
        "index",
        "seed",
        "artifact_name",
        "artifact_sha256",
        "train_cache_bundle_sha256",
        "common_compiler_state_sha256",
    }
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != record_keys
            or record.get("index") != index
            or not isinstance(record.get("artifact_name"), str)
            or not isinstance(record.get("artifact_sha256"), str)
        ):
            raise ValueError("CTAA completion seed freeze record differs")
    return records, digest


def predict(
    *,
    base_path: Path,
    tokenizer_path: Path,
    admission_path: Path,
    seed_freeze_manifest_path: Path,
    board_manifest_path: Path,
    confirmation_source_path: Path,
    frozen_seed_paths: Sequence[Path],
    output: Path,
    device_name: str,
) -> dict[str, object]:
    if len(frozen_seed_paths) != 5:
        raise ValueError("CTAA completion prediction bundle geometry differs")
    admission = load_admission(admission_path)
    batch_size = int(admission["batch_size"])
    require_admitted_protocol_source(admission)
    require_admitted_artifact_path(
        output,
        admission,
        "prediction_artifact_name",
    )
    require_admitted_artifact_path(
        seed_freeze_manifest_path,
        admission,
        "seed_freeze_manifest_name",
    )
    require_sha256(tokenizer_path, TOKENIZER_SHA256, "tokenizer")
    manifest = json.loads(board_manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != BOARD_SCHEMA:
        raise ValueError("CTAA completion prediction board schema differs")
    commitments = {
        "base_sha256": sha256_file(base_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "board_manifest_sha256": sha256_file(board_manifest_path),
        "confirmation_source_sha256": str(
            manifest["confirmation_odd_source_sha256"]
        ),
        "confirmation_oracle_sha256": str(
            manifest["confirmation_odd_oracle_sha256"]
        ),
    }
    for key, observed in commitments.items():
        if observed != admission[key]:
            raise ValueError(f"CTAA completion admission commitment differs: {key}")
    admission_sha256 = sha256_file(admission_path)
    freeze_records, freeze_sha256 = load_seed_freeze(
        seed_freeze_manifest_path,
        admission_sha256=admission_sha256,
    )
    if [
        record.get("artifact_name")
        for record in freeze_records
    ] != list(admission["seed_artifact_names"]):
        raise ValueError("CTAA completion seed freeze artifacts differ")
    seeds = []
    for path, record, expected_seed in zip(
        frozen_seed_paths,
        freeze_records,
        admission["seeds"],
        strict=True,
    ):
        if (
            path.resolve().parent != Path(str(admission["custody_root"]))
            or path.name != record.get("artifact_name")
            or record.get("seed") != expected_seed
        ):
            raise ValueError("CTAA completion seed freeze identity differs")
        value, observed_sha256 = safe_torch_load(
            path,
            expected_sha256=str(record["artifact_sha256"]),
        )
        if observed_sha256 != record["artifact_sha256"]:
            raise AssertionError("CTAA completion seed freeze hash changed")
        validate_frozen_seed(
            value,
            admission=admission,
            admission_sha256=admission_sha256,
            expected_seed=int(expected_seed),
        )
        seeds.append((path, value))
    seed_numbers = [int(value["training"]["seed"]) for _, value in seeds]
    if set(seed_numbers) != set(admission["seeds"]):
        raise ValueError("CTAA completion prediction seeds differ from admission")
    artifact_names = {path.name for path, _ in seeds}
    if artifact_names != set(admission["seed_artifact_names"]):
        raise ValueError("CTAA completion seed artifacts differ from admission")
    if any(
        value["training"].get("admission_sha256") != admission_sha256
        for _, value in seeds
    ):
        raise ValueError("CTAA completion seed admission commitments differ")
    seeds.sort(key=lambda item: int(item[1]["training"]["seed"]))
    common_board = {
        value["training"]["board_manifest_sha256"]
        for _, value in seeds
    }
    if common_board != {sha256_file(board_manifest_path)}:
        raise ValueError("CTAA completion frozen board commitments differ")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trunk, base_receipt = load_raw_trunk(base_path)
    (
        row_ids,
        family_ids,
        token_rows,
        source_hashes,
        source_sha256,
    ) = open_source_once(
        confirmation_source_path,
        tokenizer,
        trunk.cfg.seq_len,
        str(manifest["confirmation_odd_source_sha256"]),
    )
    if len(family_ids) != manifest.get("confirmation_odd_source_rows_written"):
        raise ValueError("CTAA completion source count differs")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA completion prediction requires available CUDA")
    compiler = TrunkCausalCTAACompiler(trunk).to(device)
    seed_predictions = []
    arm_factories = {
        "factorized": FactorizedBindingReadout,
        "global_structured": GlobalStructuredBindingReadout,
        "whole": WholePermutationReadout,
    }
    for path, frozen in seeds:
        training = frozen["training"]
        if training["base_sha256"] != base_receipt.sha256:
            raise ValueError("CTAA completion frozen base differs")
        restore_adapter_state(compiler, frozen["common_compiler_state"])
        slots, common_program_logits = extract_source_slots(
            compiler,
            token_rows,
            batch_size=batch_size,
            device=device,
        )
        arm_logits = {}
        for arm, factory in arm_factories.items():
            model = factory()
            load_state(model, frozen["arm_states"][arm])
            arm_logits[arm] = predict_logits(
                model,
                slots,
                batch_size=batch_size,
                device=device,
            )
        probe_logits = {}
        for slot_index in range(4):
            label = f"single_slot_{slot_index}"
            probe = SingleSlotFullBindingProbe(slot_index)
            load_state(probe, frozen["single_slot_probe_states"][label])
            probe_logits[label] = predict_logits(
                probe,
                slots,
                batch_size=batch_size,
                device=device,
            )
        seed_predictions.append(
            {
                "seed": int(training["seed"]),
                "frozen_seed_path": str(path),
                "frozen_seed_sha256": sha256_file(path),
                "confirmation_slot_cache": slots,
                "confirmation_slot_cache_sha256": tensor_sha256(slots),
                "common_program_logits": common_program_logits,
                "arm_logits": arm_logits,
                "single_slot_probe_logits": probe_logits,
            }
        )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": "source_only_predictions_no_oracle_access",
        "base_sha256": base_receipt.sha256,
        "admission_sha256": admission_sha256,
        "seed_freeze_sha256": freeze_sha256,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "board_manifest_sha256": sha256_file(board_manifest_path),
        "confirmation_source_sha256": source_sha256,
        "ordered_row_ids": row_ids,
        "ordered_family_ids": family_ids,
        "ordered_program_sha256": source_hashes,
        "seed_predictions": seed_predictions,
        "confirmation_source_access": 1,
        "confirmation_oracle_access": 0,
    }
    digest = write_once(output, payload)
    return {
        "prediction_sha256": digest,
        "rows": len(family_ids),
        "seeds": sorted(seed_numbers),
        "confirmation_source_access": 1,
        "confirmation_oracle_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed-freeze-manifest", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-source", type=Path, required=True)
    parser.add_argument(
        "--frozen-seed",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = predict(
        base_path=args.base,
        tokenizer_path=args.tokenizer,
        admission_path=args.admission,
        seed_freeze_manifest_path=args.seed_freeze_manifest,
        board_manifest_path=args.board_manifest,
        confirmation_source_path=args.confirmation_source,
        frozen_seed_paths=args.frozen_seed,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
