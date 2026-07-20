#!/usr/bin/env python3
"""Post-hoc consumed-training audit for the rejected joint renderer compiler."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from pilot_sd_cst_binding_bus import span_mask
from pilot_sd_cst_byte_addressed import byte_batch, labels, sha256_file
from pilot_sd_cst_renderer_native_program import initialize_model
from pilot_sd_cst_renderer_orbit import (
    OrbitPilotRow,
    expand_orbit,
    load_consumed_train,
    partition_rows,
)
from projected_sd_cst_fresh import exact_one_stop_map
from sd_cst import STOP_KIND
from sd_cst_renderer_orbit import HELD_OUT_RENDERERS, TRAIN_RENDERERS


JOINT_CHECKPOINT_SHA256 = (
    "4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d"
)
JOINT_SEED = 6795424534800881443


def _rates_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    fields = {
        "line_slot": "line_slots",
        "kind_slot": "kind_slots",
        "amount_active": "active_slots",
        "identity_active": "active_slots",
        "event_pointer_active": "active_slots",
        "gold_line_kind_slot": "kind_slots",
        "gold_line_amount_active": "active_slots",
        "gold_event_identity_active": "active_slots",
    }
    result: dict[str, float] = {}
    for numerator, denominator in fields.items():
        total = int(counts[denominator])
        if total <= 0:
            raise ValueError(f"renderer-native diagnostic has no {denominator}")
        result[numerator] = int(counts[numerator]) / total
    return result


def _gold_span_logits(mask: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(torch.float32).min
    return torch.where(
        mask,
        torch.zeros((), device=mask.device),
        torch.full((), floor, device=mask.device),
    )


@torch.no_grad()
def evaluate_slots(
    model: torch.nn.Module,
    groups: Sequence[Sequence[OrbitPilotRow]],
    family_batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    by_renderer: dict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(groups), family_batch_size):
        rows = [
            row for group in groups[start : start + family_batch_size] for row in group
        ]
        binding_rows = [row.binding for row in rows]
        ids, valid = byte_batch(binding_rows, "program_bytes", device)
        target = labels(binding_rows, device)
        output = model.compile_program(ids, valid)
        _, orbit_memory = model._encode_components(ids, valid)

        line_mask, _ = span_mask(
            binding_rows,
            "pointer_ranges",
            9,
            ids.shape[1],
            device,
        )
        event_mask, event_active = span_mask(
            binding_rows,
            "event_entity_ranges",
            8,
            ids.shape[1],
            device,
        )
        active = target["kind"].ne(STOP_KIND)
        if not bool(event_active.eq(active).all()):
            raise ValueError("event span activity differs from kind activity")

        free_kind = exact_one_stop_map(output.tape.event_kind)
        free_amount = output.tape.amount.argmax(-1)
        free_identity = output.tape.event_identity.argmax(-1)
        free_line = line_mask.gather(
            -1,
            output.line_pointer_logits.argmax(-1)[..., None],
        ).squeeze(-1)
        free_event = event_mask.gather(
            -1,
            output.event_entity_pointer_logits.argmax(-1)[..., None],
        ).squeeze(-1)

        gold_line_weights = line_mask.float()
        gold_line_weights = gold_line_weights / gold_line_weights.sum(
            -1,
            keepdim=True,
        ).clamp_min(1)
        gold_slots = torch.einsum(
            "bsl,blw->bsw",
            gold_line_weights.to(orbit_memory.dtype),
            orbit_memory,
        )
        gold_slots = model.native_slot_norm(model.native_slot_encoder(gold_slots))
        gold_events = gold_slots[:, 1:]
        gold_line_kind = exact_one_stop_map(model.native_kind_head(gold_events).float())
        gold_line_amount = model.native_amount_head(gold_events).argmax(-1)

        bindings = model._fingerprints(
            ids,
            valid,
            output.binding_pointer_logits,
        )
        gold_event_entities = model._fingerprints(
            ids,
            valid,
            _gold_span_logits(event_mask),
        )
        gold_event_matches = model.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bef,brf->ber",
            gold_event_entities,
            bindings,
        )
        gold_event_identity = gold_event_matches.argmax(-1)

        values = {
            "line_slot": free_line,
            "kind_slot": free_kind.eq(target["kind"]),
            "amount_active": free_amount.eq(target["amount"]),
            "identity_active": free_identity.eq(target["identity"]),
            "event_pointer_active": free_event,
            "gold_line_kind_slot": gold_line_kind.eq(target["kind"]),
            "gold_line_amount_active": gold_line_amount.eq(target["amount"]),
            "gold_event_identity_active": gold_event_identity.eq(target["identity"]),
        }
        for index, row in enumerate(rows):
            counts = by_renderer[row.renderer]
            counts["rows"] += 1
            counts["line_slots"] += 9
            counts["kind_slots"] += 8
            active_count = int(active[index].sum())
            counts["active_slots"] += active_count
            counts["line_slot"] += int(values["line_slot"][index].sum())
            counts["kind_slot"] += int(values["kind_slot"][index].sum())
            for field in (
                "amount_active",
                "identity_active",
                "event_pointer_active",
                "gold_line_amount_active",
                "gold_event_identity_active",
            ):
                counts[field] += int((values[field][index] & active[index]).sum())
            counts["gold_line_kind_slot"] += int(
                values["gold_line_kind_slot"][index].sum()
            )

    aggregate: Counter[str] = Counter()
    renderers: dict[str, object] = {}
    for renderer, counts in sorted(by_renderer.items()):
        aggregate.update(counts)
        renderers[renderer] = {
            "counts": dict(sorted(counts.items())),
            "rates": _rates_from_counts(counts),
        }
    return {
        "aggregate": {
            "counts": dict(sorted(aggregate.items())),
            "rates": _rates_from_counts(aggregate),
        },
        "renderers": renderers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--orbit-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing diagnostic output: {args.out}")
    if sha256_file(args.joint_checkpoint) != JOINT_CHECKPOINT_SHA256:
        raise SystemExit("joint diagnostic checkpoint hash differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("joint diagnostic requires bf16 CUDA")
    device = torch.device("cuda")
    source_rows = load_consumed_train(args.train_jsonl)
    fit_source, heldout_source = partition_rows(source_rows, 12_000, 2_000)
    fit_groups = expand_orbit(fit_source, TRAIN_RENDERERS)
    heldout_groups = expand_orbit(heldout_source, HELD_OUT_RENDERERS)

    torch.manual_seed(JOINT_SEED)
    model, parameters, _ = initialize_model(
        args.orbit_checkpoint,
        device,
        train_shared_orbit=True,
    )
    initial = {
        "fit": evaluate_slots(
            model,
            fit_groups,
            args.family_batch_size,
            device,
        ),
        "heldout": evaluate_slots(
            model,
            heldout_groups,
            args.family_batch_size,
            device,
        ),
    }
    checkpoint = torch.load(
        args.joint_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("schema") != "r12_sd_cst_renderer_native_joint_pilot_v1":
        raise SystemExit("joint diagnostic checkpoint schema differs")
    if (
        checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
    ):
        raise SystemExit("joint diagnostic checkpoint has scored access")
    model.load_state_dict(checkpoint["state"])
    final = {
        "fit": evaluate_slots(
            model,
            fit_groups,
            args.family_batch_size,
            device,
        ),
        "heldout": evaluate_slots(
            model,
            heldout_groups,
            args.family_batch_size,
            device,
        ),
    }
    report = {
        "schema": "r12_sd_cst_renderer_native_joint_posthoc_audit_v1",
        "status": "posthoc_consumed_training_diagnostic",
        "source": {
            "train_sha256": sha256_file(args.train_jsonl),
            "orbit_checkpoint_sha256": sha256_file(args.orbit_checkpoint),
            "joint_checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
        },
        "parameters": parameters,
        "initial": initial,
        "final": final,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": "Consumed training rows and model-state interventions only; "
        "not a fresh score or reasoning result.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"final": final, "parameters": parameters}, sort_keys=True))


if __name__ == "__main__":
    main()
