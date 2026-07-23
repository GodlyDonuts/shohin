#!/usr/bin/env python3
"""Train and freeze one A4-only CTAA binding-completion seed.

This stage has no confirmation path and cannot open odd source or oracle data.
The decisive comparison uses identical cached slot tensors, identical
four-cell targets, and identical minibatch order.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from ctaa_artifact_loader import (
    STRICT_SYSTEM_LIMIT,
    TOKENIZER_SHA256,
    load_qualified_memory_state,
    load_raw_trunk,
    require_sha256,
    verify_complete_system_parameters,
)
from ctaa_binding_completion import (
    ACTION_COUNT,
    BINDINGS,
    READOUT_PARAMETERS,
    FactorizedBindingReadout,
    GlobalStructuredBindingReadout,
    SingleSlotFullBindingProbe,
    WholePermutationReadout,
    audit_parity_rows,
    factorized_loss,
    materialize_factorized,
    materialize_whole,
    permutation_parity,
    readout_resource_receipt,
    whole_loss,
)
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_protocol_source,
)
from ctaa_compiler_training import (
    TokenizedCompilerRow,
    collate_compiler_rows,
    parse_train_row,
)
from ctaa_neural_core import ClosureFeatureTransitionCore
from ctaa_trunk_compiler import TrunkCausalCTAACompiler


SCHEMA = "r12_ctaa_a4_binding_completion_training_v1"
BOARD_SCHEMA = "r12_ctaa_a4_binding_completion_board_v1"
SEED_TOP_LEVEL_KEYS = {
    "schema",
    "claim_boundary",
    "common_compiler_state",
    "discarded_qualifier_state",
    "arm_states",
    "single_slot_probe_states",
    "train_slot_cache",
    "train_bindings",
    "train_ordered_family_ids",
    "train_ordered_program_sha256",
    "training",
}
SEED_TRAINING_KEYS = {
    "seed",
    "admission_sha256",
    "qualifier_updates",
    "readout_updates",
    "batch_size",
    "learning_rate",
    "minimum_train_exact",
    "base_sha256",
    "base_step",
    "qualified_compiler_sha256",
    "qualified_memory_tensors",
    "tokenizer_sha256",
    "board_manifest_sha256",
    "train_sha256",
    "train_audit",
    "readout_resources",
    "parameter_ledger",
    "qualifier_last",
    "qualifier_metrics",
    "common_compiler_metrics",
    "train_slot_cache_sha256",
    "train_cache_bundle_sha256",
    "common_compiler_state_sha256",
    "arm_training",
    "a4_derived_odd_chimera_metrics",
    "single_slot_probe_training",
    "development_access",
    "confirmation_source_access",
    "confirmation_oracle_access",
    "whole_control_role",
}
ARM_NAMES = {"factorized", "global_structured", "whole"}
PROBE_NAMES = {f"single_slot_{index}" for index in range(ACTION_COUNT)}
TRAINABLE_COMPILER_PREFIXES = (
    "early_memory_norm.",
    "early_memory_projection.",
    "late_memory_norm.",
    "late_memory_projection.",
    "memory_encoder.",
    "program_queries",
    "query_query",
    "decoder.",
    "decoder_norm.",
    "tuple_head.",
    "event_head.",
    "query_head.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def tagged_seed(seed: int, label: str) -> int:
    payload = f"ctaa-a4-completion-v1|{seed}|{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(json.dumps(list(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def tensor_mapping_sha256(value: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def safe_torch_load(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("CTAA completion artifact changed during read")
    encoded = b"".join(chunks)
    observed = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError("CTAA completion artifact hash differs before load")
    value = torch.load(
        io.BytesIO(encoded),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(value, dict):
        raise ValueError("CTAA completion artifact schema differs")
    return value, observed


def cache_bundle_sha256(
    *,
    slots: torch.Tensor,
    bindings: torch.Tensor,
    family_ids: Sequence[str],
    program_hashes: Sequence[str],
    compiler_state: Mapping[str, torch.Tensor],
    train_sha256: str,
    configuration: Mapping[str, object],
) -> str:
    if not (
        slots.shape[0]
        == bindings.shape[0]
        == len(family_ids)
        == len(program_hashes)
    ):
        raise ValueError("CTAA completion cache commitment geometry differs")
    payload = {
        "slots_sha256": tensor_sha256(slots),
        "bindings_sha256": tensor_sha256(bindings),
        "family_ids": list(family_ids),
        "program_hashes": list(program_hashes),
        "compiler_state_sha256": tensor_mapping_sha256(compiler_state),
        "train_sha256": train_sha256,
        "configuration": dict(configuration),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_seed(
    value: dict[str, object],
    *,
    admission: Mapping[str, object],
    admission_sha256: str,
    expected_seed: int,
) -> dict[str, object]:
    if set(value) != SEED_TOP_LEVEL_KEYS or value.get("schema") != SCHEMA:
        raise ValueError("CTAA completion frozen seed schema differs")
    training = value.get("training")
    if not isinstance(training, dict) or set(training) != SEED_TRAINING_KEYS:
        raise ValueError("CTAA completion frozen seed receipt differs")
    expected_receipt = {
        "seed": expected_seed,
        "admission_sha256": admission_sha256,
        "qualifier_updates": admission["qualifier_updates"],
        "readout_updates": admission["readout_updates"],
        "batch_size": admission["batch_size"],
        "learning_rate": admission["learning_rate"],
        "minimum_train_exact": admission["minimum_train_exact"],
        "base_sha256": admission["base_sha256"],
        "qualified_compiler_sha256": admission["qualified_compiler_sha256"],
        "tokenizer_sha256": admission["tokenizer_sha256"],
        "board_manifest_sha256": admission["board_manifest_sha256"],
        "train_sha256": admission["train_even_sha256"],
        "development_access": 0,
        "confirmation_source_access": 0,
        "confirmation_oracle_access": 0,
        "whole_control_role": "support_starved_lookup_negative_only",
    }
    for key, expected in expected_receipt.items():
        if training.get(key) != expected:
            raise ValueError(
                f"CTAA completion frozen seed receipt differs: {key}"
            )
    arm_states = value.get("arm_states")
    arm_training = training.get("arm_training")
    probe_states = value.get("single_slot_probe_states")
    probe_training = training.get("single_slot_probe_training")
    if (
        not isinstance(arm_states, dict)
        or set(arm_states) != ARM_NAMES
        or not isinstance(arm_training, dict)
        or set(arm_training) != ARM_NAMES
        or not isinstance(probe_states, dict)
        or set(probe_states) != PROBE_NAMES
        or not isinstance(probe_training, dict)
        or set(probe_training) != PROBE_NAMES
    ):
        raise ValueError("CTAA completion frozen seed arm lattice differs")
    minimum = float(admission["minimum_train_exact"])
    qualifier_metrics = training.get("qualifier_metrics")
    common_metrics = training.get("common_compiler_metrics")
    chimera_metrics = training.get("a4_derived_odd_chimera_metrics")
    if (
        not isinstance(qualifier_metrics, dict)
        or float(qualifier_metrics.get("projected_binding_exact", -1.0))
        < minimum
        or not isinstance(common_metrics, dict)
        or set(common_metrics)
        != {
            "cards_exact",
            "initial_exact",
            "opcode_schedule_exact",
            "query_exact",
        }
        or any(float(metric) < minimum for metric in common_metrics.values())
        or not isinstance(chimera_metrics, dict)
        or set(chimera_metrics) != ARM_NAMES
        or float(
            chimera_metrics["factorized"].get(
                "projected_binding_exact",
                -1.0,
            )
        )
        < float(admission["minimum_chimera_exact"])
    ):
        raise ValueError("CTAA completion frozen seed fit gate differs")
    for arm in ARM_NAMES:
        receipt = arm_training[arm]
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("metrics"), dict)
            or float(
                receipt["metrics"].get("projected_binding_exact", -1.0)
            )
            < minimum
        ):
            raise ValueError("CTAA completion frozen seed arm fit differs")
    for label in PROBE_NAMES:
        receipt = probe_training[label]
        if (
            not isinstance(receipt, dict)
            or receipt.get("a4_fit_qualified") is not True
            or not isinstance(receipt.get("metrics"), dict)
            or float(
                receipt["metrics"].get("projected_binding_exact", -1.0)
            )
            < minimum
        ):
            raise ValueError("CTAA completion frozen seed probe fit differs")
    if training.get("readout_resources") != readout_resource_receipt():
        raise ValueError("CTAA completion frozen seed resource receipt differs")
    slots = value.get("train_slot_cache")
    bindings = value.get("train_bindings")
    compiler_state = value.get("common_compiler_state")
    family_ids = value.get("train_ordered_family_ids")
    program_hashes = value.get("train_ordered_program_sha256")
    if (
        not isinstance(slots, torch.Tensor)
        or not isinstance(bindings, torch.Tensor)
        or not isinstance(compiler_state, dict)
        or not isinstance(family_ids, list)
        or not isinstance(program_hashes, list)
    ):
        raise ValueError("CTAA completion frozen seed cache schema differs")
    observed_cache = cache_bundle_sha256(
        slots=slots,
        bindings=bindings,
        family_ids=family_ids,
        program_hashes=program_hashes,
        compiler_state=compiler_state,
        train_sha256=str(training["train_sha256"]),
        configuration={
            "seed": expected_seed,
            "qualifier_updates": admission["qualifier_updates"],
            "readout_updates": admission["readout_updates"],
            "batch_size": admission["batch_size"],
            "learning_rate": admission["learning_rate"],
            "minimum_train_exact": admission["minimum_train_exact"],
        },
    )
    if (
        tensor_sha256(slots) != training.get("train_slot_cache_sha256")
        or observed_cache != training.get("train_cache_bundle_sha256")
        or tensor_mapping_sha256(compiler_state)
        != training.get("common_compiler_state_sha256")
    ):
        raise ValueError("CTAA completion frozen seed commitment differs")
    return value


def load_rows(
    path: Path,
    tokenizer: Tokenizer,
    max_length: int,
) -> tuple[list[TokenizedCompilerRow], list[dict[str, object]]]:
    parsed = []
    raw = []
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                parsed.append(parse_train_row(value, tokenizer, max_length))
                raw.append(value)
            except Exception as error:
                raise ValueError(
                    f"CTAA completion row {line_number} failed: {path}"
                ) from error
    if not parsed:
        raise ValueError(f"CTAA completion file is empty: {path}")
    return parsed, raw


def validate_orbit_pair(
    train_raw: Sequence[Mapping[str, object]],
    confirmation_raw: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    partitions: dict[int, defaultdict[str, list[Mapping[str, object]]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for parity, rows in ((0, train_raw), (1, confirmation_raw)):
        for row in rows:
            binding = tuple(int(item) for item in row["opcode_to_card"])
            if permutation_parity(binding) != parity:
                raise ValueError("CTAA completion file parity differs")
            partitions[parity][str(row["family_id"])].append(row)
    if set(partitions[0]) != set(partitions[1]):
        raise ValueError("CTAA completion family partitions differ")
    invariant_keys = (
        "query_source",
        "action_cards",
        "initial_state",
        "schedule",
        "query_position",
        "renderer",
    )
    for family_id in partitions[0]:
        train = partitions[0][family_id]
        confirmation = partitions[1][family_id]
        if len(train) != 12 or len(confirmation) != 12:
            raise ValueError("CTAA completion orbit half-size differs")
        combined = [*train, *confirmation]
        bindings = {
            tuple(int(item) for item in row["opcode_to_card"])
            for row in combined
        }
        if bindings != set(BINDINGS):
            raise ValueError("CTAA completion paired orbit does not span S4")
        for key in invariant_keys:
            values = {
                json.dumps(row[key], sort_keys=True, separators=(",", ":"))
                for row in combined
            }
            if len(values) != 1:
                raise ValueError(
                    f"CTAA completion paired orbit changes invariant {key}"
                )
    train_sources = {str(row["program_source"]) for row in train_raw}
    confirmation_sources = {
        str(row["program_source"]) for row in confirmation_raw
    }
    if train_sources.intersection(confirmation_sources):
        raise ValueError("CTAA completion paired sources overlap")
    return {
        "families": len(partitions[0]),
        "rows_per_family_per_partition": 12,
        "combined_bindings_per_family": 24,
        "program_source_overlap": 0,
    }


def build_two_slot_chimeras(
    train_slots: torch.Tensor,
    train_bindings: torch.Tensor,
    train_raw: Sequence[Mapping[str, object]],
    *,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose odd targets exclusively from independently chosen A4 slots."""

    if limit < 1:
        raise ValueError("CTAA completion chimera limit differs")
    if (
        len(train_raw) != train_slots.shape[0]
        or len(train_raw) != train_bindings.shape[0]
    ):
        raise ValueError("CTAA completion chimera cache geometry differs")
    family_indices: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_raw):
        family_indices[str(row["family_id"])].append(index)
    chimeras = []
    targets = []
    for train_index, row in enumerate(train_raw):
        binding = tuple(int(item) for item in row["opcode_to_card"])
        for left, right in itertools.combinations(range(ACTION_COUNT), 2):
            target = list(binding)
            target[left], target[right] = (
                target[right],
                target[left],
            )
            if permutation_parity(target) != 1:
                raise AssertionError("CTAA completion chimera target parity differs")
            candidates_by_slot = [
                [
                    index
                    for index in family_indices[str(row["family_id"])]
                    if int(train_bindings[index, slot]) == card
                ]
                for slot, card in enumerate(target)
            ]

            def assign_donors(
                slot: int,
                used: set[int],
            ) -> list[int] | None:
                if slot == ACTION_COUNT:
                    return []
                for candidate in candidates_by_slot[slot]:
                    if candidate in used:
                        continue
                    suffix = assign_donors(slot + 1, used | {candidate})
                    if suffix is not None:
                        return [candidate, *suffix]
                return None

            donors = assign_donors(0, set())
            if donors is None:
                raise ValueError(
                    "CTAA completion independent A4 chimera donor is missing"
                )
            opcode_slots = torch.stack(
                [
                    train_slots[donor_index, slot]
                    for slot, donor_index in enumerate(donors)
                ]
            )
            chimera = torch.cat(
                (opcode_slots, train_slots[train_index, ACTION_COUNT:])
            )
            chimeras.append(chimera)
            targets.append(torch.tensor(target, dtype=train_bindings.dtype))
            if len(chimeras) == limit:
                return torch.stack(chimeras), torch.stack(targets)
    if not chimeras:
        raise ValueError("CTAA completion chimera board is empty")
    return torch.stack(chimeras), torch.stack(targets)


def capture_adapter_state(
    compiler: TrunkCausalCTAACompiler,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in compiler.state_dict().items()
        if not name.startswith("model.")
    }


def configure_compiler_training(
    compiler: TrunkCausalCTAACompiler,
) -> list[nn.Parameter]:
    compiler.requires_grad_(False)
    selected = []
    for name, parameter in compiler.named_parameters():
        if name.startswith("model."):
            continue
        if name.startswith(TRAINABLE_COMPILER_PREFIXES):
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise AssertionError("CTAA completion compiler path is empty")
    return selected


def fixed_schedule(
    row_count: int,
    updates: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        row_count,
        (updates, batch_size),
        generator=generator,
    )


def train_qualifier(
    compiler: TrunkCausalCTAACompiler,
    rows: Sequence[TokenizedCompilerRow],
    schedule: torch.Tensor,
    *,
    learning_rate: float,
    device: torch.device,
) -> tuple[GlobalStructuredBindingReadout, dict[str, float]]:
    qualifier = GlobalStructuredBindingReadout().to(device)
    parameters = [*configure_compiler_training(compiler), *qualifier.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    last = {}
    compiler.train()
    qualifier.train()
    for indices in schedule:
        batch = collate_compiler_rows(
            [rows[index] for index in indices.tolist()],
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            bundle = compiler.encode_source(batch.program_ids)
            program = compiler.compile_program_from_residuals(bundle)
            binding_logits = qualifier(
                compiler.binding_relation_slots_from_residuals(bundle)
            )
            query_logits = compiler.compile_query(batch.query_ids)
            binding = factorized_loss(
                binding_logits,
                batch.opcode_to_card,
            )
            cards = F.cross_entropy(
                program.action_cards.reshape(-1, 3),
                batch.action_cards.reshape(-1),
            )
            initial = F.cross_entropy(
                program.initial_state.reshape(-1, 3),
                batch.initial_state.reshape(-1),
            )
            opcode_schedule = F.cross_entropy(
                program.opcode_schedule.reshape(-1, 5),
                batch.opcode_schedule.reshape(-1),
            )
            query = F.cross_entropy(query_logits, batch.query_position)
            loss = binding + cards + initial + opcode_schedule + query
        if not torch.isfinite(loss):
            raise FloatingPointError("CTAA completion qualifier loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach()),
            "binding": float(binding.detach()),
            "cards": float(cards.detach()),
            "initial": float(initial.detach()),
            "opcode_schedule": float(opcode_schedule.detach()),
            "query": float(query.detach()),
        }
    return qualifier, last


@torch.inference_mode()
def extract_slot_cache(
    compiler: TrunkCausalCTAACompiler,
    rows: Sequence[TokenizedCompilerRow],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    compiler.eval()
    slots = []
    bindings = []
    for start in range(0, len(rows), batch_size):
        batch = collate_compiler_rows(
            rows[start : start + batch_size],
            device=device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            current = compiler.binding_relation_slots(batch.program_ids)
        slots.append(current.float().cpu())
        bindings.append(batch.opcode_to_card.cpu())
    return torch.cat(slots), torch.cat(bindings)


@torch.inference_mode()
def evaluate_common_compiler(
    compiler: TrunkCausalCTAACompiler,
    rows: Sequence[TokenizedCompilerRow],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    compiler.eval()
    correct = {
        "cards_exact": 0,
        "initial_exact": 0,
        "opcode_schedule_exact": 0,
        "query_exact": 0,
    }
    for start in range(0, len(rows), batch_size):
        batch = collate_compiler_rows(
            rows[start : start + batch_size],
            device=device,
        )
        program = compiler.compile_program(batch.program_ids)
        query = compiler.compile_query(batch.query_ids)
        correct["cards_exact"] += int(
            program.action_cards.argmax(-1)
            .eq(batch.action_cards)
            .flatten(1)
            .all(1)
            .sum()
        )
        correct["initial_exact"] += int(
            program.initial_state.argmax(-1)
            .eq(batch.initial_state)
            .all(1)
            .sum()
        )
        correct["opcode_schedule_exact"] += int(
            program.opcode_schedule.argmax(-1)
            .eq(batch.opcode_schedule)
            .all(1)
            .sum()
        )
        correct["query_exact"] += int(
            query.argmax(-1).eq(batch.query_position).sum()
        )
    return {
        key: value / len(rows)
        for key, value in correct.items()
    }


def train_readout(
    readout: nn.Module,
    slots: torch.Tensor,
    bindings: torch.Tensor,
    schedule: torch.Tensor,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    readout = readout.to(device).train()
    optimizer = torch.optim.AdamW(
        readout.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    last = {}
    for indices in schedule:
        selected_slots = slots.index_select(0, indices).to(device)
        selected_bindings = bindings.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = readout(selected_slots)
            loss = loss_function(logits, selected_bindings)
        if not torch.isfinite(loss):
            raise FloatingPointError("CTAA completion readout loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        optimizer.step()
        last = {"loss": float(loss.detach())}
    state = {
        name: value.detach().cpu().clone()
        for name, value in readout.state_dict().items()
    }
    return state, last


@torch.inference_mode()
def metrics_from_logits(
    logits: torch.Tensor,
    bindings: torch.Tensor,
    *,
    arm: str,
) -> dict[str, object]:
    if logits.device != bindings.device:
        raise ValueError("CTAA completion metric devices differ")
    if arm == "whole":
        loss = whole_loss(logits, bindings)
        predicted = materialize_whole(logits)
        raw = predicted
        top = logits.topk(2, dim=-1).values
        margin = top[:, 0] - top[:, 1]
    else:
        loss = factorized_loss(logits, bindings)
        predicted = materialize_factorized(logits)
        raw = logits.argmax(-1)
        top = logits.topk(2, dim=-1).values
        margin = top[:, :, 0] - top[:, :, 1]
    expected = torch.arange(ACTION_COUNT)[None].expand(raw.shape[0], -1)
    expected = expected.to(raw.device)
    raw_valid = raw.sort(-1).values.eq(expected).all(-1)
    projected_exact = predicted.eq(bindings).all(-1)
    raw_exact = raw.eq(bindings).all(-1)
    per_binding = {}
    for binding in BINDINGS:
        mask = bindings.eq(torch.tensor(binding)).all(-1)
        per_binding["".join(str(item) for item in binding)] = {
            "rows": int(mask.sum()),
            "projected_exact": float(projected_exact[mask].float().mean()),
            "raw_exact": float(raw_exact[mask].float().mean()),
            "projection_rescue": float(
                (projected_exact[mask] & ~raw_exact[mask]).float().mean()
            ),
        }
    predicted_odd = torch.tensor(
        [permutation_parity(row.tolist()) for row in predicted],
        device=predicted.device,
    )
    return {
        "rows": int(bindings.shape[0]),
        "nll": float(loss),
        "projected_binding_exact": float(projected_exact.float().mean()),
        "projected_local_cell_accuracy": float(
            predicted.eq(bindings).float().mean()
        ),
        "raw_local_cell_accuracy": float(raw.eq(bindings).float().mean()),
        "raw_binding_exact": float(raw_exact.float().mean()),
        "raw_assignment_valid": float(raw_valid.float().mean()),
        "projection_rescue": float(
            (projected_exact & ~raw_exact).float().mean()
        ),
        "predicted_odd_fraction": float(predicted_odd.float().mean()),
        "mean_logit_margin": float(margin.float().mean()),
        "per_binding": per_binding,
    }


@torch.inference_mode()
def evaluate_readout(
    readout: nn.Module,
    slots: torch.Tensor,
    bindings: torch.Tensor,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    readout = readout.to(device).eval()
    logits = []
    for start in range(0, slots.shape[0], batch_size):
        logits.append(
            readout(slots[start : start + batch_size].to(device)).float().cpu()
        )
    return metrics_from_logits(torch.cat(logits), bindings.cpu(), arm=arm)


def load_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(state), strict=True)


def write_once(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.chmod(0o600)
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def train(
    *,
    base_path: Path,
    qualified_path: Path,
    tokenizer_path: Path,
    board_manifest_path: Path,
    train_path: Path,
    output: Path,
    seed: int,
    qualifier_updates: int,
    readout_updates: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    minimum_train_exact: float,
    admission_sha256: str,
) -> dict[str, object]:
    if (
        qualifier_updates < 1
        or readout_updates < 1
        or batch_size < 1
        or learning_rate <= 0
        or not 0 < minimum_train_exact <= 1
    ):
        raise ValueError("CTAA completion training configuration differs")
    require_sha256(tokenizer_path, TOKENIZER_SHA256, "tokenizer")
    manifest = json.loads(board_manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != BOARD_SCHEMA:
        raise ValueError("CTAA completion board manifest schema differs")
    if sha256_file(train_path) != manifest.get("train_even_sha256"):
        raise ValueError("CTAA completion train hash differs")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trunk, base_receipt = load_raw_trunk(base_path)
    qualified = load_qualified_memory_state(qualified_path)
    random.seed(seed)
    torch.manual_seed(tagged_seed(seed, "common-compiler"))
    compiler = TrunkCausalCTAACompiler(trunk)
    loaded = compiler.initialize_qualified_memory(qualified)
    core = ClosureFeatureTransitionCore()
    base_ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        core.unique_parameters,
    )
    resources = readout_resource_receipt()
    decisive_total = base_ledger["total"] + READOUT_PARAMETERS
    if decisive_total > STRICT_SYSTEM_LIMIT:
        raise ValueError("CTAA completion decisive system exceeds parameter limit")
    parameter_ledger = {
        **base_ledger,
        "readout": READOUT_PARAMETERS,
        "total_with_readout": decisive_total,
        "headroom_with_readout": STRICT_SYSTEM_LIMIT - decisive_total,
        "whole_negative_readout": resources["whole_parameters"],
        "total_with_whole_negative": (
            base_ledger["total"] + int(resources["whole_parameters"])
        ),
        "headroom_with_whole_negative": (
            STRICT_SYSTEM_LIMIT
            - base_ledger["total"]
            - int(resources["whole_parameters"])
        ),
    }

    train_rows, train_raw = load_rows(
        train_path,
        tokenizer,
        trunk.cfg.seq_len,
    )
    train_audit = audit_parity_rows(train_rows, expected_parity=0)
    if len(train_rows) != manifest.get("train_even_rows_written"):
        raise ValueError("CTAA completion train row count differs")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA completion training requires available CUDA")
    compiler.to(device)
    qualifier_schedule = fixed_schedule(
        len(train_rows),
        qualifier_updates,
        batch_size,
        tagged_seed(seed, "qualifier-batches"),
    )
    torch.manual_seed(tagged_seed(seed, "qualifier-readout"))
    qualifier, qualifier_last = train_qualifier(
        compiler,
        train_rows,
        qualifier_schedule,
        learning_rate=learning_rate,
        device=device,
    )
    compiler.requires_grad_(False)
    compiler_state = capture_adapter_state(compiler)
    qualifier_state = {
        name: value.detach().cpu().clone()
        for name, value in qualifier.state_dict().items()
    }
    train_slots, train_bindings = extract_slot_cache(
        compiler,
        train_rows,
        batch_size=batch_size,
        device=device,
    )
    train_cache_sha256 = tensor_sha256(train_slots)
    train_program_hashes = [
        hashlib.sha256(str(row["program_source"]).encode("utf-8")).hexdigest()
        for row in train_raw
    ]
    train_family_ids = [str(row["family_id"]) for row in train_raw]
    train_cache_bundle_sha256 = cache_bundle_sha256(
        slots=train_slots,
        bindings=train_bindings,
        family_ids=train_family_ids,
        program_hashes=train_program_hashes,
        compiler_state=compiler_state,
        train_sha256=sha256_file(train_path),
        configuration={
            "seed": seed,
            "qualifier_updates": qualifier_updates,
            "readout_updates": readout_updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "minimum_train_exact": minimum_train_exact,
        },
    )
    qualifier_metrics = evaluate_readout(
        qualifier,
        train_slots,
        train_bindings,
        arm="global_structured",
        batch_size=batch_size,
        device=device,
    )
    if qualifier_metrics["projected_binding_exact"] < minimum_train_exact:
        raise RuntimeError("CTAA completion common qualifier failed A4 fit gate")
    common_compiler_metrics = evaluate_common_compiler(
        compiler,
        train_rows,
        batch_size=batch_size,
        device=device,
    )
    if any(
        value < minimum_train_exact
        for value in common_compiler_metrics.values()
    ):
        raise RuntimeError("CTAA completion common compiler failed A4 fit gate")
    readout_schedule = fixed_schedule(
        len(train_rows),
        readout_updates,
        batch_size,
        tagged_seed(seed, "readout-batches"),
    )
    arm_specs: dict[
        str,
        tuple[
            Callable[[], nn.Module],
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ],
    ] = {
        "factorized": (FactorizedBindingReadout, factorized_loss),
        "global_structured": (GlobalStructuredBindingReadout, factorized_loss),
        "whole": (WholePermutationReadout, whole_loss),
    }
    arm_states = {}
    arm_training = {}
    for arm, (factory, loss_function) in arm_specs.items():
        if tensor_sha256(train_slots) != train_cache_sha256:
            raise RuntimeError("CTAA completion A4 cache changed before arm")
        torch.manual_seed(tagged_seed(seed, f"{arm}-readout"))
        state, last = train_readout(
            factory(),
            train_slots,
            train_bindings,
            readout_schedule,
            loss_function=loss_function,
            learning_rate=learning_rate,
            device=device,
        )
        model = factory()
        load_state(model, state)
        metrics = evaluate_readout(
            model,
            train_slots,
            train_bindings,
            arm=arm,
            batch_size=batch_size,
            device=device,
        )
        if metrics["projected_binding_exact"] < minimum_train_exact:
            raise RuntimeError(f"CTAA completion {arm} failed A4 fit gate")
        arm_states[arm] = state
        arm_training[arm] = {"last": last, "metrics": metrics}
        if tensor_sha256(train_slots) != train_cache_sha256:
            raise RuntimeError("CTAA completion A4 cache changed after arm")
    chimera_slots, chimera_bindings = build_two_slot_chimeras(
        train_slots,
        train_bindings,
        train_raw,
        limit=min(4_096, len(train_rows) * 6),
    )
    chimera_metrics = {}
    for arm, (factory, _) in arm_specs.items():
        model = factory()
        load_state(model, arm_states[arm])
        chimera_metrics[arm] = evaluate_readout(
            model,
            chimera_slots,
            chimera_bindings,
            arm=arm,
            batch_size=batch_size,
            device=device,
        )
    probe_states = {}
    probe_training = {}
    for slot_index in range(ACTION_COUNT):
        label = f"single_slot_{slot_index}"
        torch.manual_seed(tagged_seed(seed, f"{label}-probe"))
        state, last = train_readout(
            SingleSlotFullBindingProbe(slot_index),
            train_slots,
            train_bindings,
            readout_schedule,
            loss_function=factorized_loss,
            learning_rate=learning_rate,
            device=device,
        )
        probe = SingleSlotFullBindingProbe(slot_index)
        load_state(probe, state)
        metrics = evaluate_readout(
            probe,
            train_slots,
            train_bindings,
            arm="global_structured",
            batch_size=batch_size,
            device=device,
        )
        if metrics["projected_binding_exact"] < minimum_train_exact:
            raise RuntimeError(
                f"CTAA completion {label} failed A4 probe fit gate"
            )
        probe_states[label] = state
        probe_training[label] = {
            "last": last,
            "metrics": metrics,
            "a4_fit_qualified": (
                metrics["projected_binding_exact"] >= minimum_train_exact
            ),
        }

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "a4_binding_completion_diagnostic_only_not_recurrent_reasoning"
        ),
        "common_compiler_state": compiler_state,
        "discarded_qualifier_state": qualifier_state,
        "arm_states": arm_states,
        "single_slot_probe_states": probe_states,
        "train_slot_cache": train_slots,
        "train_bindings": train_bindings,
        "train_ordered_family_ids": train_family_ids,
        "train_ordered_program_sha256": train_program_hashes,
        "training": {
            "seed": seed,
            "admission_sha256": admission_sha256,
            "qualifier_updates": qualifier_updates,
            "readout_updates": readout_updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "minimum_train_exact": minimum_train_exact,
            "base_sha256": base_receipt.sha256,
            "base_step": base_receipt.step,
            "qualified_compiler_sha256": sha256_file(qualified_path),
            "qualified_memory_tensors": len(loaded),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "board_manifest_sha256": sha256_file(board_manifest_path),
            "train_sha256": sha256_file(train_path),
            "train_audit": train_audit,
            "readout_resources": resources,
            "parameter_ledger": parameter_ledger,
            "qualifier_last": qualifier_last,
            "qualifier_metrics": qualifier_metrics,
            "common_compiler_metrics": common_compiler_metrics,
            "train_slot_cache_sha256": train_cache_sha256,
            "train_cache_bundle_sha256": train_cache_bundle_sha256,
            "common_compiler_state_sha256": tensor_mapping_sha256(
                compiler_state
            ),
            "arm_training": arm_training,
            "a4_derived_odd_chimera_metrics": chimera_metrics,
            "single_slot_probe_training": probe_training,
            "development_access": 0,
            "confirmation_source_access": 0,
            "confirmation_oracle_access": 0,
            "whole_control_role": "support_starved_lookup_negative_only",
        },
    }
    digest = write_once(output, payload)
    return {
        "checkpoint_sha256": digest,
        **payload["training"],  # type: ignore[dict-item]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    admission = load_admission(args.admission)
    require_admitted_protocol_source(admission)
    if not 0 <= args.seed_index < 5:
        raise ValueError("CTAA completion admission seed index differs")
    if args.output.name != admission["seed_artifact_names"][args.seed_index]:
        raise ValueError("CTAA completion admission seed output differs")
    if args.output.resolve().parent != Path(str(admission["custody_root"])):
        raise ValueError("CTAA completion admission seed custody root differs")
    commitments = {
        "base_sha256": sha256_file(args.base),
        "qualified_compiler_sha256": sha256_file(args.qualified_compiler),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "board_manifest_sha256": sha256_file(args.board_manifest),
        "train_even_sha256": sha256_file(args.train),
    }
    for key, observed in commitments.items():
        if observed != admission[key]:
            raise ValueError(f"CTAA completion admission commitment differs: {key}")
    report = train(
        base_path=args.base,
        qualified_path=args.qualified_compiler,
        tokenizer_path=args.tokenizer,
        board_manifest_path=args.board_manifest,
        train_path=args.train,
        output=args.output,
        seed=int(admission["seeds"][args.seed_index]),
        qualifier_updates=int(admission["qualifier_updates"]),
        readout_updates=int(admission["readout_updates"]),
        batch_size=int(admission["batch_size"]),
        learning_rate=float(admission["learning_rate"]),
        device_name=args.device,
        minimum_train_exact=float(admission["minimum_train_exact"]),
        admission_sha256=sha256_file(args.admission),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
