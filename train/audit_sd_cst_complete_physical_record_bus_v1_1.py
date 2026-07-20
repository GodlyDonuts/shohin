#!/usr/bin/env python3
"""Read-only six-slot confusion audit for the failed declaration-key repair."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from pilot_sd_cst_binding_bus import span_mask
from pilot_sd_cst_byte_addressed import byte_batch, sha256_file
from pilot_sd_cst_complete_physical_record_bus_v1_1 import initialize_model
from pilot_sd_cst_renderer_orbit import (
    expand_orbit,
    load_consumed_train,
    partition_rows,
)
from sd_cst_complete_physical_record_bus_v1_1 import (
    declaration_repair_trainable_names,
)
from sd_cst_renderer_orbit import HELD_OUT_RENDERERS


ENDPOINT_SHA256 = (
    "46697b3942fdfd2edfec06cea6cb119ad507adcdee4a99330fd06bc79e5b3e88"
)
SLOT_NAMES = (
    "binding_role_0",
    "binding_role_1",
    "binding_role_2",
    "initial_occurrence_0",
    "initial_occurrence_1",
    "initial_occurrence_2",
)
PREDICTED_CATEGORIES = SLOT_NAMES + ("other",)


def classify_top_indices(
    all_span_masks: torch.Tensor,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    """Map each query's selected byte to one of six occurrence spans or other."""
    if all_span_masks.ndim != 3 or top_indices.ndim != 2:
        raise ValueError("six-slot audit shapes differ")
    if (
        all_span_masks.shape[0] != top_indices.shape[0]
        or all_span_masks.shape[1] != len(SLOT_NAMES)
    ):
        raise ValueError("six-slot audit batch or category count differs")
    selected = all_span_masks.gather(
        -1,
        top_indices[:, None].expand(-1, len(SLOT_NAMES), -1),
    )
    has_category = selected.any(1)
    category = selected.float().argmax(1)
    return torch.where(
        has_category,
        category,
        torch.full_like(category, len(SLOT_NAMES)),
    )


def _empty_slot() -> dict[str, object]:
    return {
        "rows": 0,
        "top1_exact": 0,
        "target_probability_sum": 0.0,
        "target_nll_sum": 0.0,
        "confusion": Counter(),
    }


def _finalize_slot(raw: dict[str, object]) -> dict[str, object]:
    rows = int(raw["rows"])
    if rows <= 0:
        raise ValueError("six-slot audit has an empty slot")
    confusion = raw["confusion"]
    if not isinstance(confusion, Counter):
        raise TypeError("six-slot audit confusion has invalid type")
    return {
        "rows": rows,
        "top1_exact": int(raw["top1_exact"]),
        "top1_rate": int(raw["top1_exact"]) / rows,
        "mean_target_probability": float(raw["target_probability_sum"]) / rows,
        "mean_target_nll": float(raw["target_nll_sum"]) / rows,
        "confusion": {
            name: int(confusion[name]) for name in PREDICTED_CATEGORIES
        },
    }


@torch.no_grad()
def audit_renderer(
    model: torch.nn.Module,
    rows: list[object],
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    slots = {name: _empty_slot() for name in SLOT_NAMES}
    all_three_binding = 0
    all_three_initial = 0
    total_rows = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        binding_rows = [row.binding for row in batch]  # type: ignore[attr-defined]
        ids, valid = byte_batch(binding_rows, "program_bytes", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.compile_program(ids, valid)  # type: ignore[attr-defined]
        binding_mask, _ = span_mask(
            binding_rows,
            "binding_ranges",
            3,
            ids.shape[1],
            device,
        )
        initial_mask, _ = span_mask(
            binding_rows,
            "initial_entity_ranges",
            3,
            ids.shape[1],
            device,
        )
        masks = torch.cat([binding_mask, initial_mask], dim=1)
        logits = torch.cat(
            [
                output.binding_pointer_logits,
                output.initial_entity_pointer_logits,
            ],
            dim=1,
        ).float()
        probabilities = logits.softmax(-1)
        log_probabilities = logits.log_softmax(-1)
        target_counts = masks.sum(-1).clamp_min(1)
        target_probability = (probabilities * masks).sum(-1)
        target_nll = -torch.where(
            masks,
            log_probabilities,
            torch.zeros_like(log_probabilities),
        ).sum(-1) / target_counts
        top = logits.argmax(-1)
        exact = masks.gather(-1, top[..., None]).squeeze(-1)
        categories = classify_top_indices(masks, top)
        all_three_binding += int(exact[:, :3].all(-1).sum())
        all_three_initial += int(exact[:, 3:].all(-1).sum())
        total_rows += len(batch)
        for slot_index, slot_name in enumerate(SLOT_NAMES):
            raw = slots[slot_name]
            raw["rows"] = int(raw["rows"]) + len(batch)
            raw["top1_exact"] = int(raw["top1_exact"]) + int(
                exact[:, slot_index].sum()
            )
            raw["target_probability_sum"] = float(
                raw["target_probability_sum"]
            ) + float(target_probability[:, slot_index].sum())
            raw["target_nll_sum"] = float(raw["target_nll_sum"]) + float(
                target_nll[:, slot_index].sum()
            )
            confusion = raw["confusion"]
            if not isinstance(confusion, Counter):
                raise TypeError("six-slot audit confusion has invalid type")
            confusion.update(
                PREDICTED_CATEGORIES[int(index)]
                for index in categories[:, slot_index].cpu().tolist()
            )
    return {
        "rows": total_rows,
        "all_three_binding": all_three_binding,
        "all_three_binding_rate": all_three_binding / total_rows,
        "all_three_initial": all_three_initial,
        "all_three_initial_rate": all_three_initial / total_rows,
        "slots": {name: _finalize_slot(raw) for name, raw in slots.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing six-slot audit output: {args.out}")
    if sha256_file(args.endpoint) != ENDPOINT_SHA256:
        raise SystemExit("six-slot audit endpoint hash differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("six-slot audit requires bf16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(0)
    model, parameters, excluded_digest, _, query_state_sha256 = initialize_model(
        args.joint_checkpoint,
        args.physical_checkpoint,
        args.v1_checkpoint,
        device,
    )
    endpoint = torch.load(args.endpoint, map_location="cpu", weights_only=False)
    if endpoint.get("schema") != "r12_sd_cst_complete_physical_record_bus_pilot_v1_1":
        raise SystemExit("six-slot audit endpoint schema differs")
    if (
        endpoint.get("development_accesses") != 0
        or endpoint.get("confirmation_accesses") != 0
    ):
        raise SystemExit("six-slot audit endpoint has scored access")
    state = endpoint.get("declaration_state")
    names = declaration_repair_trainable_names(model)
    if not isinstance(state, dict) or set(state) != names:
        raise SystemExit("six-slot audit endpoint state differs")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or not names.isdisjoint(missing):
        raise SystemExit("six-slot audit endpoint load differs")
    if endpoint.get("excluded_state_digest") != excluded_digest:
        raise SystemExit("six-slot audit excluded-state digest differs")
    model.eval()

    source_rows = load_consumed_train(args.train_jsonl)
    _, heldout_source = partition_rows(source_rows, 12_000, 2_000)
    groups = expand_orbit(heldout_source, HELD_OUT_RENDERERS)
    per_renderer = {}
    for renderer_index, renderer in enumerate(HELD_OUT_RENDERERS):
        per_renderer[renderer.name] = audit_renderer(
            model,
            [group[renderer_index] for group in groups],
            args.batch_size,
            device,
        )
    report = {
        "schema": "r12_sd_cst_complete_physical_record_bus_v1_1_slot_audit_v1",
        "endpoint_sha256": ENDPOINT_SHA256,
        "query_state_sha256": query_state_sha256,
        "parameters": parameters,
        "per_renderer": per_renderer,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": "Read-only consumed-training heldout pointer confusion "
        "audit; no scored split or reasoning claim.",
    }
    args.out.parent.mkdir(parents=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(args.out),
                "minimum_all_three_binding": min(
                    value["all_three_binding_rate"]
                    for value in per_renderer.values()
                ),
                "minimum_all_three_initial": min(
                    value["all_three_initial_rate"]
                    for value in per_renderer.values()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
