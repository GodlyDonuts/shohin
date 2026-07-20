"""Shared mechanics for the projected SD-CST fresh-board experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import itertools
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from pilot_sd_cst_binding_bus import (
    BindingPilotRow,
    evaluate as evaluate_compiler,
)
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    _line_ranges,
    batches,
    byte_batch,
    cosine_scale,
    sha256_file,
)
from pilot_sd_cst_hierarchical_binding import (
    PARENT_SHA256,
    PROJECTED_TRAINABLE_NAMES,
    binding_loss,
    freeze_parent,
    frozen_parameter_digest,
    load_parent_state,
)
from sd_cst import EVENT_STEPS, STOP_KIND, HardLateQuery, HardProgramTape
from sd_cst_binding_bus import ProjectedHierarchicalBindingBusCompiler


GLOBAL_PARAMETER_CAP = 200_000_000
COMPARISON_PARAMETER_CAP = 150_000_000
EXECUTION_CORE_SHA256 = (
    "166ca6f81dd962b06a94f7a3661921a410760090ed1b750d78ec1b0f610113f1"
)
CONSUMED_PROJECTED_SHA256 = (
    "f347d1aea90dd3c60f7500167c7c22884451b365880259698306c6fce8ab10f3"
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_STATE = {value: index for index, value in enumerate(PERMUTATIONS)}
TRAINING_CONTRACT = {
    "rows": 48_000,
    "epochs": 4,
    "batch_size": 64,
    "updates": 3_000,
    "lr": 3e-4,
    "warmup": 100,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "gradient_clip": 1.0,
    "schedule": "cosine_to_zero",
}


@dataclass(frozen=True, slots=True)
class ProjectedFreshRow:
    row_id: str
    raw_row_sha256: str
    raw_row_canonical_json: str
    split: str
    variant: str
    family_id: str | None
    template_id: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    pointer_ranges: tuple[tuple[int, int], ...]
    binding_ranges: tuple[tuple[int, int], ...]
    initial_entity_ranges: tuple[tuple[int, int], ...]
    event_entity_ranges: tuple[tuple[int, int], ...]
    initial_state: int
    event_kind: tuple[int, ...]
    event_identity: tuple[int, ...]
    amount: tuple[int, ...]
    query_position: int
    halt_after: int
    final_state: int | None
    answer_role: int | None
    full_suffix_state: int | None
    active_state_trajectory: tuple[int, ...] | None


def canonical_json(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode("ascii")).digest()[:8], "big"
    )


def state_dict_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    if set(state) != set(PROJECTED_TRAINABLE_NAMES):
        raise ValueError("projected trainable-state key mismatch")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            target = named[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"projected tensor contract mismatch: {name}")
            target.copy_(value)


def _find_within(source: bytes, needle: bytes, start: int, end: int) -> tuple[int, int]:
    found = source.find(needle, start, end)
    if found < 0:
        raise ValueError(f"name occurrence absent from declared region: {needle!r}")
    return found, found + len(needle)


def _advance_order(
    state: tuple[int, ...],
    identity: int,
    kind: int,
    amount: int,
) -> tuple[int, ...]:
    values = list(state)
    source = values.index(identity)
    signed = -(amount + 1) if kind == 0 else amount + 1
    destination = min(2, max(0, source + signed))
    value = values.pop(source)
    values.insert(destination, value)
    return tuple(values)


def parse_projected_row(
    row: Mapping[str, object],
    split: str,
) -> ProjectedFreshRow:
    if row.get("split") != split:
        raise ValueError("projected row split mismatch")
    if split == "sd_cst_train" and "oracle" in row:
        raise ValueError("projected training row contains outcome supervision")
    targets = row.get("compiler_targets")
    query = row.get("late_query_target")
    if not isinstance(targets, Mapping) or not isinstance(query, Mapping):
        raise ValueError("projected row lacks compiler/query targets")
    slots = targets.get("event_slots")
    if not isinstance(slots, list) or len(slots) != EVENT_STEPS:
        raise ValueError("projected row requires eight event slots")
    slots = sorted(slots, key=lambda item: int(item["semantic_ordinal"]))
    kind = tuple(int(item["kind_id"]) for item in slots)
    if kind.count(STOP_KIND) != 1:
        raise ValueError("projected row requires exactly one STOP")
    identity = tuple(int(item.get("entity_role", 0)) for item in slots)
    amount = tuple(int(item.get("amount_id", 0)) for item in slots)
    initial_order = tuple(int(value) for value in targets["initial_order_roles"])
    initial_state = int(targets["initial_state_id"])
    if PERMUTATION_TO_STATE.get(initial_order) != initial_state:
        raise ValueError("initial-state category mismatch")
    text = str(row["program_text"])
    source = text.encode("utf-8")
    lines = _line_ranges(text)
    storage_order = [int(value) for value in targets["storage_order"]]
    if sorted(storage_order) != list(range(1, EVENT_STEPS + 1)):
        raise ValueError("storage order is not an event permutation")
    pointer_ranges = [lines[0]]
    for ordinal in range(1, EVENT_STEPS + 1):
        pointer_ranges.append(lines[1 + storage_order.index(ordinal)])

    bindings = sorted(
        targets["entity_bindings"], key=lambda item: int(item["entity_role"])
    )
    names = [str(item["entity"]).encode("utf-8") for item in bindings]
    line_start, line_end = lines[0]
    markers = [
        source.find(value, line_start, line_end) for value in (b"initial ", b"lineup ")
    ]
    markers = [value for value in markers if value >= 0]
    if len(markers) != 1:
        raise ValueError("binding line requires exactly one initial-order marker")
    marker = markers[0]
    marker_width = (
        len(b"initial ")
        if source[marker : marker + len(b"initial ")] == b"initial "
        else len(b"lineup ")
    )
    binding_ranges = tuple(
        _find_within(source, name, line_start, marker) for name in names
    )
    initial_ranges = []
    cursor = marker + marker_width
    for role in initial_order:
        found = _find_within(source, names[role], cursor, line_end)
        initial_ranges.append(found)
        cursor = found[1]
    event_ranges = []
    for slot, (start, end) in enumerate(pointer_ranges[1:]):
        if kind[slot] == STOP_KIND:
            event_ranges.append((0, 0))
        else:
            event_ranges.append(_find_within(source, names[identity[slot]], start, end))

    final_state = answer_role = full_suffix_state = None
    active_trajectory = None
    oracle = row.get("oracle")
    if split != "sd_cst_train":
        if not isinstance(oracle, Mapping):
            raise ValueError("projected evaluation row lacks oracle")
        final_order = tuple(int(value) for value in oracle["final_state_roles"])
        final_state = PERMUTATION_TO_STATE[final_order]
        answer_role = int(oracle["answer_role"])
        active_trajectory = tuple(
            PERMUTATION_TO_STATE[tuple(int(value) for value in order)]
            for order in oracle["active_trajectory_roles"]
        )
        full_order = initial_order
        for event_kind, event_identity, event_amount in zip(
            kind,
            identity,
            amount,
            strict=True,
        ):
            if event_kind != STOP_KIND:
                full_order = _advance_order(
                    full_order,
                    event_identity,
                    event_kind,
                    event_amount,
                )
        full_suffix_state = PERMUTATION_TO_STATE[full_order]

    return ProjectedFreshRow(
        row_id=str(row["id"]),
        raw_row_sha256=hashlib.sha256(
            canonical_json(dict(row)).encode("utf-8")
        ).hexdigest(),
        raw_row_canonical_json=canonical_json(dict(row)),
        split=split,
        variant=str(row.get("variant", "")),
        family_id=str(row["family_id"]) if row.get("family_id") is not None else None,
        template_id=str(row.get("template_id", "")),
        program_bytes=tuple(source),
        query_bytes=tuple(str(row["late_query_text"]).encode("utf-8")),
        pointer_ranges=tuple(pointer_ranges),
        binding_ranges=binding_ranges,
        initial_entity_ranges=tuple(initial_ranges),
        event_entity_ranges=tuple(event_ranges),
        initial_state=initial_state,
        event_kind=kind,
        event_identity=identity,
        amount=amount,
        query_position=int(query["position"]),
        halt_after=int(targets["halt_after"]),
        final_state=final_state,
        answer_role=answer_role,
        full_suffix_state=full_suffix_state,
        active_state_trajectory=active_trajectory,
    )


def as_binding_row(row: ProjectedFreshRow) -> BindingPilotRow:
    return BindingPilotRow(
        row_id=row.row_id,
        program_bytes=row.program_bytes,
        query_bytes=row.query_bytes,
        pointer_ranges=row.pointer_ranges,
        binding_ranges=row.binding_ranges,
        initial_entity_ranges=row.initial_entity_ranges,
        event_entity_ranges=row.event_entity_ranges,
        initial_state=row.initial_state,
        event_kind=row.event_kind,
        event_identity=row.event_identity,
        amount=row.amount,
        query_position=row.query_position,
    )


def permute_training_labels(
    row: ProjectedFreshRow,
    permutation: tuple[int, int, int],
) -> ProjectedFreshRow:
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("label control requires a role permutation")
    inverse = tuple(permutation.index(label) for label in range(3))
    initial_order = tuple(permutation[role] for role in PERMUTATIONS[row.initial_state])
    return replace(
        row,
        binding_ranges=tuple(row.binding_ranges[inverse[label]] for label in range(3)),
        initial_state=PERMUTATION_TO_STATE[initial_order],
        event_identity=tuple(permutation[role] for role in row.event_identity),
    )


def row_shuffled_permutation(seed: int, row_id: str) -> tuple[int, int, int]:
    """Select an independent deterministic role permutation for one row."""
    digest = hashlib.sha256(f"{seed}:{row_id}:row-labels".encode("utf-8")).digest()
    return PERMUTATIONS[int.from_bytes(digest[:8], "big") % len(PERMUTATIONS)]


def row_shuffled_mapping_digest(
    rows: Sequence[ProjectedFreshRow],
    seed: int,
) -> str:
    return hashlib.sha256(
        "".join(
            f"{row.row_id}:{','.join(map(str, row_shuffled_permutation(seed, row.row_id)))}\n"
            for row in rows
        ).encode("utf-8")
    ).hexdigest()


def initialize_model(
    parent_checkpoint: Path,
    seed: int,
    device: torch.device,
) -> tuple[ProjectedHierarchicalBindingBusCompiler, dict[str, object]]:
    if sha256_file(parent_checkpoint) != PARENT_SHA256:
        raise ValueError("fresh projected parent checkpoint hash mismatch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ProjectedHierarchicalBindingBusCompiler()
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "r12_sd_cst_byte_addressed_training_pilot_v1":
        raise ValueError("fresh projected parent schema mismatch")
    missing = load_parent_state(model, payload["state"])
    trainable = freeze_parent(model, PROJECTED_TRAINABLE_NAMES)
    if set(missing) != set(PROJECTED_TRAINABLE_NAMES):
        raise ValueError(
            "parent missing keys do not equal projected trainable contract"
        )
    model.to(device)
    compiler_parameters = model.parameter_count()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    complete = (
        BASE_PARAMETERS + compiler_parameters + MOTOR_PARAMETERS + READER_PARAMETERS
    )
    if complete >= COMPARISON_PARAMETER_CAP or complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("fresh projected system exceeds frozen parameter contract")
    return model, {
        "parent_sha256": PARENT_SHA256,
        "compiler": compiler_parameters,
        "trainable": trainable_parameters,
        "trainable_names": list(trainable),
        "nominal_base": BASE_PARAMETERS,
        "motor": MOTOR_PARAMETERS,
        "reader": READER_PARAMETERS,
        "nominal_complete_system": complete,
        "active_projected_plus_core": compiler_parameters
        + MOTOR_PARAMETERS
        + READER_PARAMETERS,
        "comparison_cap": COMPARISON_PARAMETER_CAP,
        "global_cap": GLOBAL_PARAMETER_CAP,
    }


def minibatch_order_digest(rows: int, seed: int, epochs: int, batch_size: int) -> str:
    digest = hashlib.sha256()
    for epoch in range(epochs):
        for indices in batches(rows, batch_size, seed, epoch):
            digest.update(torch.tensor(indices, dtype=torch.int32).numpy().tobytes())
    return digest.hexdigest()


def fit_projected_arm(
    model: ProjectedHierarchicalBindingBusCompiler,
    rows: Sequence[ProjectedFreshRow],
    *,
    seed: int,
    batch_size: int = 64,
    epochs: int = 4,
    lr: float = 3e-4,
    warmup: int = 100,
    clip: float = 1.0,
) -> dict[str, object]:
    if len(rows) != TRAINING_CONTRACT["rows"]:
        raise ValueError("fresh projected arm requires exactly 48,000 rows")
    device = next(model.parameters()).device
    binding_rows = [as_binding_row(row) for row in rows]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    frozen_before = frozen_parameter_digest(model)
    initial_digest = state_dict_digest(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    total_updates = epochs * math.ceil(len(rows) / batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_scale(step, total_updates, warmup),
    )
    history = []
    update = 0
    for epoch in range(epochs):
        model.train()
        totals: Counter[str] = Counter()
        seen = 0
        for indices in batches(len(rows), batch_size, seed, epoch):
            batch = [binding_rows[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = binding_loss(model, batch, device)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite fresh projected gradient")
            optimizer.step()
            scheduler.step()
            update += 1
            seen += len(batch)
            for name, value in pieces.items():
                totals[name] += value * len(batch)
        history.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "fit_losses": {
                    name: value / seen for name, value in sorted(totals.items())
                },
            }
        )
    if update != TRAINING_CONTRACT["updates"]:
        raise RuntimeError("fresh projected update count changed")
    final_metrics = evaluate_compiler(model, binding_rows, 128, device)
    frozen_after = frozen_parameter_digest(model)
    return {
        "seed": seed,
        "updates": update,
        "minibatch_order_sha256": minibatch_order_digest(
            len(rows),
            seed,
            epochs,
            batch_size,
        ),
        "initial_full_state_sha256": initial_digest,
        "final_full_state_sha256": state_dict_digest(model),
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_parent_unchanged": frozen_before == frozen_after,
        "history": history,
        "train_metrics": final_metrics,
    }


def _concatenate_tapes(tapes: Sequence[HardProgramTape]) -> HardProgramTape:
    return HardProgramTape(
        torch.cat([value.initial_state for value in tapes]),
        torch.cat([value.event_kind for value in tapes]),
        torch.cat([value.event_identity for value in tapes]),
        torch.cat([value.amount for value in tapes]),
    )


@torch.no_grad()
def compile_fresh_rows(
    model: ProjectedHierarchicalBindingBusCompiler,
    rows: Sequence[ProjectedFreshRow],
    batch_size: int,
    device: torch.device,
    *,
    source_free_binding: bool = False,
) -> dict[str, object]:
    model.eval()
    tapes = []
    queries = []
    pointers = {
        name: []
        for name in (
            "line",
            "binding",
            "initial_entity",
            "event_entity",
        )
    }
    source_poison_exact = True
    for start in range(0, len(rows), batch_size):
        fresh = rows[start : start + batch_size]
        batch = [as_binding_row(row) for row in fresh]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = (
                model.compile_program_source_free_binding(program_ids, program_valid)
                if source_free_binding
                else model.compile_program(program_ids, program_valid)
            )
        hard = output.tape.hard()
        sealed = HardProgramTape(
            hard.initial_state.detach().cpu().to(torch.uint8).clone(),
            hard.event_kind.detach().cpu().to(torch.uint8).clone(),
            hard.event_identity.detach().cpu().to(torch.uint8).clone(),
            hard.amount.detach().cpu().to(torch.uint8).clone(),
        )
        pointers["line"].append(output.line_pointer_logits.argmax(-1).detach().cpu())
        pointers["binding"].append(
            output.binding_pointer_logits.argmax(-1).detach().cpu()
        )
        pointers["initial_entity"].append(
            output.initial_entity_pointer_logits.argmax(-1).detach().cpu()
        )
        pointers["event_entity"].append(
            output.event_entity_pointer_logits.argmax(-1).detach().cpu()
        )
        before = tuple(
            value.clone()
            for value in (
                sealed.initial_state,
                sealed.event_kind,
                sealed.event_identity,
                sealed.amount,
            )
        )
        program_ids.random_(0, 257)
        program_valid.logical_not_()
        source_poison_exact &= all(
            torch.equal(left, right)
            for left, right in zip(
                before,
                (
                    sealed.initial_state,
                    sealed.event_kind,
                    sealed.event_identity,
                    sealed.amount,
                ),
                strict=True,
            )
        )
        del output, hard, program_ids, program_valid
        torch.cuda.empty_cache()

        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query_output = model.compile_query(query_ids, query_valid)
        hard_query = query_output.hard()
        sealed_query = HardLateQuery(
            hard_query.position.detach().cpu().to(torch.uint8).clone()
        )
        query_before = sealed_query.position.clone()
        query_ids.random_(0, 257)
        query_valid.logical_not_()
        source_poison_exact &= torch.equal(query_before, sealed_query.position)
        tapes.append(sealed)
        queries.append(sealed_query.position)
        del query_output, hard_query, query_ids, query_valid
    return {
        "tape": _concatenate_tapes(tapes),
        "query": HardLateQuery(torch.cat(queries)),
        "pointers": {name: torch.cat(values) for name, values in pointers.items()},
        "source_poison_bit_identical": source_poison_exact,
    }


def constant_source_rows(rows: Sequence[ProjectedFreshRow]) -> list[ProjectedFreshRow]:
    return [
        replace(
            row,
            program_bytes=tuple([32] * len(row.program_bytes)),
            query_bytes=tuple([32] * len(row.query_bytes)),
        )
        for row in rows
    ]


def tape_to_json(tape: HardProgramTape, query: HardLateQuery) -> dict[str, object]:
    return {
        "initial_state": tape.initial_state.tolist(),
        "event_kind": tape.event_kind.tolist(),
        "event_identity": tape.event_identity.tolist(),
        "amount": tape.amount.tolist(),
        "query": query.position.tolist(),
    }


def outputs_to_json(outputs: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {name: value.tolist() for name, value in outputs.items()}


def expected_tape(
    rows: Sequence[ProjectedFreshRow],
) -> tuple[HardProgramTape, HardLateQuery]:
    return HardProgramTape(
        torch.tensor([row.initial_state for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_kind for row in rows], dtype=torch.uint8),
        torch.tensor([row.event_identity for row in rows], dtype=torch.uint8),
        torch.tensor([row.amount for row in rows], dtype=torch.uint8),
    ), HardLateQuery(
        torch.tensor(
            [row.query_position for row in rows],
            dtype=torch.uint8,
        )
    )
