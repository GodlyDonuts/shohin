"""Shared mechanics for fresh-board complete physical SD-CST qualification."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    cosine_scale,
    sha256_file,
)
from pilot_sd_cst_complete_physical_record_bus import PHYSICAL_CHECKPOINT_SHA256
from pilot_sd_cst_complete_physical_record_bus_v1_1 import V1_CHECKPOINT_SHA256
from pilot_sd_cst_complete_physical_record_bus_v1_2 import (
    initialize_model as initialize_v1_2_parent,
)
from pilot_sd_cst_physical_record_bus import JOINT_CHECKPOINT_SHA256
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from pilot_sd_cst_renderer_orbit import OrbitPilotRow, evaluate, loss_groups
from projected_sd_cst_fresh import (
    EXECUTION_CORE_SHA256,
    ProjectedFreshRow,
    as_binding_row,
    compile_fresh_rows,
    parse_projected_row,
    permute_training_labels,
    state_dict_digest,
)
from sd_cst_complete_physical_record_bus import _freeze_to_declared
from sd_cst_complete_physical_record_bus_v1_2 import (
    CompletePhysicalRecordBusCompilerV1_2,
    occurrence_head_trainable_names,
)
from sd_cst_physical_record_bus import physical_record_trainable_names


GLOBAL_PARAMETER_CAP = 200_000_000
V1_2_CHECKPOINT_SHA256 = (
    "eaa83df068ca0545ddd578f36d4d3f269334e83553f0730f844e46e107323c18"
)
TRAINING_CONTRACT = {
    "semantic_families": 12_000,
    "views_per_family": 4,
    "rows": 48_000,
    "epochs": 2,
    "family_batch_size": 8,
    "rows_per_update": 32,
    "updates": 3_000,
    "lr": 2e-4,
    "warmup": 100,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "gradient_clip": 1.0,
    "renderer_consistency_weight": 1.0,
    "schedule": "cosine_to_zero",
}


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode("ascii")).digest()[:8], "big"
    )


def fresh_trainable_names(
    model: CompletePhysicalRecordBusCompilerV1_2,
) -> frozenset[str]:
    query = {
        name for name, _ in model.named_parameters() if name.startswith("local_query_")
    }
    if len(query) != 8:
        raise ValueError("fresh local-query tensor contract differs")
    declared = (
        physical_record_trainable_names(model)
        | occurrence_head_trainable_names(model)
        | frozenset(query)
    )
    if any(name.startswith("local_declaration_") for name in declared):
        raise ValueError("obsolete bilinear declaration tensors are trainable")
    return declared


def initialize_model(
    joint_checkpoint: Path,
    physical_checkpoint: Path,
    v1_checkpoint: Path,
    v1_2_checkpoint: Path,
    device: torch.device,
) -> tuple[CompletePhysicalRecordBusCompilerV1_2, dict[str, object], str]:
    if sha256_file(v1_2_checkpoint) != V1_2_CHECKPOINT_SHA256:
        raise ValueError("fresh v1.2 endpoint hash differs")
    endpoint = torch.load(v1_2_checkpoint, map_location="cpu", weights_only=False)
    if type(endpoint.get("seed")) is not int:
        raise ValueError("fresh v1.2 endpoint seed differs")
    torch.manual_seed(int(endpoint["seed"]))
    model, _, _, _, query_state_sha256 = initialize_v1_2_parent(
        joint_checkpoint,
        physical_checkpoint,
        v1_checkpoint,
        torch.device("cpu"),
    )
    if (
        endpoint.get("schema") != "r12_sd_cst_complete_physical_record_bus_pilot_v1_2"
        or endpoint.get("joint_checkpoint_sha256") != JOINT_CHECKPOINT_SHA256
        or endpoint.get("physical_checkpoint_sha256") != PHYSICAL_CHECKPOINT_SHA256
        or endpoint.get("v1_checkpoint_sha256") != V1_CHECKPOINT_SHA256
        or endpoint.get("development_accesses") != 0
        or endpoint.get("confirmation_accesses") != 0
    ):
        raise ValueError("fresh v1.2 endpoint receipt differs")
    occurrence_names = occurrence_head_trainable_names(model)
    occurrence_state = endpoint.get("occurrence_state")
    if not isinstance(occurrence_state, Mapping) or set(occurrence_state) != set(
        occurrence_names
    ):
        raise ValueError("fresh v1.2 occurrence state differs")
    current = model.state_dict()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("local_declaration_"):
                parameter.zero_()
    for name, tensor in occurrence_state.items():
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise ValueError("fresh v1.2 occurrence tensor differs")
        current[name].copy_(tensor)
    if query_state_sha256 != endpoint.get("query_state_sha256"):
        raise ValueError("fresh v1.2 active query endpoint differs")

    declared = fresh_trainable_names(model)
    trainable = _freeze_to_declared(model, declared)
    frozen_digest = frozen_state_digest(model, declared)
    model.to(device)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    if complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("fresh complete system reaches 200M cap")
    parameters: dict[str, object] = {
        "base": BASE_PARAMETERS,
        "compiler": compiler,
        "motor": MOTOR_PARAMETERS,
        "reader": READER_PARAMETERS,
        "complete_system": complete,
        "headroom": GLOBAL_PARAMETER_CAP - complete,
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": list(trainable),
    }
    return model, parameters, frozen_digest


def load_rows(path: Path, split: str) -> list[ProjectedFreshRow]:
    rows = [
        parse_projected_row(json.loads(line), split)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    expected = 48_000 if split == "sd_cst_train" else 2_048
    if len(rows) != expected:
        raise ValueError("fresh split row count differs")
    return rows


def orbit_row(row: ProjectedFreshRow) -> OrbitPilotRow:
    raw = json.loads(row.raw_row_canonical_json)
    span = raw.get("late_query_target", {}).get("byte_span")
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError("fresh query span is absent")
    return OrbitPilotRow(
        binding=as_binding_row(row),
        query_span=(int(span[0]), int(span[1])),
        renderer=row.variant,
        semantic_id=str(row.family_id),
    )


def group_rows(rows: Sequence[ProjectedFreshRow]) -> list[list[OrbitPilotRow]]:
    families: dict[str, list[ProjectedFreshRow]] = {}
    for row in rows:
        if row.family_id is None:
            raise ValueError("fresh row lacks family ID")
        families.setdefault(row.family_id, []).append(row)
    groups = []
    for family in sorted(families):
        values = sorted(families[family], key=lambda row: row.variant)
        if len(values) != 4 or len({row.variant for row in values}) != 4:
            raise ValueError("fresh family does not contain four renderer views")
        groups.append([orbit_row(row) for row in values])
    return groups


def permute_family_labels(
    rows: Sequence[ProjectedFreshRow], seed: int
) -> tuple[list[ProjectedFreshRow], str]:
    derangements = ((1, 2, 0), (2, 0, 1))
    output = []
    mapping = []
    for row in rows:
        if row.family_id is None:
            raise ValueError("fresh control row lacks family ID")
        digest = hashlib.sha256(
            f"{seed}:{row.family_id}:family-false-labels".encode("utf-8")
        ).digest()
        permutation = derangements[int.from_bytes(digest[:8], "big") % 2]
        output.append(permute_training_labels(row, permutation))
        mapping.append(f"{row.row_id}:{','.join(map(str, permutation))}\n")
    return output, hashlib.sha256("".join(mapping).encode("utf-8")).hexdigest()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(
    model: CompletePhysicalRecordBusCompilerV1_2,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = fresh_trainable_names(model)
    if set(state) != set(expected):
        raise ValueError("fresh trainable state keys differ")
    current = model.state_dict()
    with torch.no_grad():
        for name, tensor in state.items():
            if (
                tensor.shape != current[name].shape
                or tensor.dtype != current[name].dtype
            ):
                raise ValueError("fresh trainable state tensor differs")
            current[name].copy_(tensor)


def fit_arm(
    model: CompletePhysicalRecordBusCompilerV1_2,
    rows: Sequence[ProjectedFreshRow],
    *,
    seed: int,
) -> dict[str, object]:
    groups = group_rows(rows)
    if len(groups) != int(TRAINING_CONTRACT["semantic_families"]):
        raise ValueError("fresh fit semantic-family count differs")
    device = next(model.parameters()).device
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    declared = fresh_trainable_names(model)
    frozen_before = frozen_state_digest(model, declared)
    initial_digest = state_dict_digest(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(TRAINING_CONTRACT["lr"]),
        betas=tuple(TRAINING_CONTRACT["betas"]),
        weight_decay=float(TRAINING_CONTRACT["weight_decay"]),
    )
    total_updates = int(TRAINING_CONTRACT["updates"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_scale(
            step,
            total_updates,
            int(TRAINING_CONTRACT["warmup"]),
        ),
    )
    rng = random.Random(seed)
    history = []
    update = 0
    for epoch in range(int(TRAINING_CONTRACT["epochs"])):
        model.train()
        order = list(range(len(groups)))
        rng.shuffle(order)
        totals: Counter[str] = Counter()
        seen = 0
        family_batch = int(TRAINING_CONTRACT["family_batch_size"])
        for start in range(0, len(order), family_batch):
            batch = [groups[index] for index in order[start : start + family_batch]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = loss_groups(
                    model,
                    batch,
                    device,
                    float(TRAINING_CONTRACT["renderer_consistency_weight"]),
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(TRAINING_CONTRACT["gradient_clip"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite fresh compiler gradient")
            optimizer.step()
            scheduler.step()
            update += 1
            row_count = sum(len(group) for group in batch)
            seen += row_count
            for name, value in pieces.items():
                totals[name] += value * row_count
        history.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "fit_losses": {
                    name: value / seen for name, value in sorted(totals.items())
                },
            }
        )
    if update != total_updates:
        raise RuntimeError("fresh compiler update count differs")
    metrics = evaluate(model, groups, 16, device)
    frozen_after = frozen_state_digest(model, declared)
    return {
        "seed": seed,
        "updates": update,
        "initial_full_state_sha256": initial_digest,
        "final_full_state_sha256": state_dict_digest(model),
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parent_unchanged": frozen_before == frozen_after,
        "history": history,
        "train_metrics": metrics,
    }


def endpoint_hashes() -> dict[str, str]:
    return {
        "joint": JOINT_CHECKPOINT_SHA256,
        "physical": PHYSICAL_CHECKPOINT_SHA256,
        "v1": V1_CHECKPOINT_SHA256,
        "v1_2": V1_2_CHECKPOINT_SHA256,
        "execution_core": EXECUTION_CORE_SHA256,
    }


def compile_rows(
    model: CompletePhysicalRecordBusCompilerV1_2,
    rows: Sequence[ProjectedFreshRow],
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    return compile_fresh_rows(model, rows, batch_size, device)
