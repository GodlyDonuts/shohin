#!/usr/bin/env python3
"""Training-only renderer-orbit and late-query grounding pilot.

This pilot consumes only the already-consumed projected-v2 training split. It
fits on even-parity renderer combinations and evaluates odd-parity combinations
over disjoint latent programs. It cannot open development or confirmation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from pilot_sd_cst_binding_bus import (
    BindingPilotRow,
    span_mask,
    uniform_span_loss,
)
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    byte_batch,
    cosine_scale,
    labels,
    sha256_file,
)
from pilot_sd_cst_hierarchical_binding import (
    PROJECTED_TRAINABLE_NAMES,
    load_parent_state,
)
from projected_sd_cst_fresh import (
    exact_one_stop_map,
    load_trainable_state,
    parse_projected_row,
    as_binding_row,
)
from sd_cst import STOP_KIND
from sd_cst_renderer_orbit import (
    HELD_OUT_RENDERERS,
    TRAIN_RENDERERS,
    RendererOrbitElement,
    render_row,
)
from sd_cst_renderer_orbit_frontend import (
    RendererOrbitGroundedCompiler,
    freeze_to_renderer_orbit_front_end,
    renderer_orbit_trainable_names,
)


PARENT_SHA256 = "e5f87a1d5b22d24250a6aac6fb7c70b4a77dbdf01bd5f5c509020a3584dfa6f9"
V2_CHECKPOINT_SHA256 = (
    "1d338651e381c6bd36982adca0e0edf36147c54c101ebc63e37ffea431a645fd"
)
CONSUMED_TRAIN_SHA256 = (
    "b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25"
)
GLOBAL_PARAMETER_CAP = 200_000_000


@dataclass(frozen=True, slots=True)
class OrbitPilotRow:
    binding: BindingPilotRow
    query_span: tuple[int, int]
    renderer: str
    semantic_id: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _source_digest(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_consumed_train(path: Path) -> list[dict[str, object]]:
    if sha256_file(path) != CONSUMED_TRAIN_SHA256:
        raise ValueError("renderer-orbit consumed training hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 48_000:
        raise ValueError("renderer-orbit pilot requires 48,000 consumed train rows")
    if any(row.get("split") != "sd_cst_train" or "oracle" in row for row in rows):
        raise ValueError("renderer-orbit pilot input is not compiler-only training")
    return rows


def partition_rows(
    rows: Sequence[Mapping[str, object]],
    fit_semantics: int,
    heldout_semantics: int,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    if fit_semantics <= 0 or heldout_semantics <= 0:
        raise ValueError("renderer-orbit partitions must be positive")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["id"]).encode("utf-8")).digest(),
    )
    if fit_semantics + heldout_semantics > len(ordered):
        raise ValueError("renderer-orbit partition exceeds consumed rows")
    return ordered[:fit_semantics], ordered[
        fit_semantics : fit_semantics + heldout_semantics
    ]


def expand_orbit(
    rows: Sequence[Mapping[str, object]],
    renderers: Sequence[RendererOrbitElement],
) -> list[list[OrbitPilotRow]]:
    groups: list[list[OrbitPilotRow]] = []
    for row in rows:
        semantic_id = str(row["id"])
        group: list[OrbitPilotRow] = []
        for renderer in renderers:
            rendered = render_row(
                row,
                renderer,
                row_id=f"{semantic_id}::{renderer.name}",
                family_id=semantic_id,
            )
            parsed = parse_projected_row(rendered, "sd_cst_train")
            target = rendered["late_query_target"]
            query_span = tuple(map(int, target["byte_span"]))
            group.append(
                OrbitPilotRow(
                    binding=as_binding_row(parsed),
                    query_span=query_span,  # type: ignore[arg-type]
                    renderer=renderer.name,
                    semantic_id=semantic_id,
                )
            )
        groups.append(group)
    return groups


def initialize_model(
    parent_checkpoint: Path,
    v2_checkpoint: Path,
    device: torch.device,
) -> tuple[RendererOrbitGroundedCompiler, dict[str, object]]:
    if sha256_file(parent_checkpoint) != PARENT_SHA256:
        raise ValueError("renderer-orbit parent checkpoint hash differs")
    if sha256_file(v2_checkpoint) != V2_CHECKPOINT_SHA256:
        raise ValueError("renderer-orbit v2 checkpoint hash differs")
    parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    v2 = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)
    if parent.get("schema") != "r12_sd_cst_byte_addressed_training_pilot_v1":
        raise ValueError("renderer-orbit parent schema differs")
    if v2.get("schema") != "r12_sd_cst_projected_fresh_checkpoint_v2":
        raise ValueError("renderer-orbit v2 schema differs")

    model = RendererOrbitGroundedCompiler()
    missing = set(load_parent_state(model, parent["state"]))
    expected_missing = (
        set(PROJECTED_TRAINABLE_NAMES)
        | set(renderer_orbit_trainable_names(model))
        | {"permutations"}
    )
    if missing != expected_missing:
        raise ValueError("renderer-orbit parent missing-key contract differs")
    load_trainable_state(model, v2["arms"]["treatment"]["trainable_state"])
    trainable = freeze_to_renderer_orbit_front_end(model)
    model.to(device)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    if complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("renderer-orbit complete system reaches global cap")
    return model, {
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


def _query_span_mask(
    rows: Sequence[OrbitPilotRow],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        start, end = row.query_span
        if start < 0 or end <= start or end > width:
            raise ValueError("query span is outside rendered query")
        mask[index, start:end] = True
    return mask


def _orbit_consistency(logits: torch.Tensor, families: int, views: int) -> torch.Tensor:
    shape = (families, views) + tuple(logits.shape[1:])
    values = logits.float().reshape(shape)
    log_probs = values.log_softmax(-1)
    probs = log_probs.exp()
    mean = probs.mean(1, keepdim=True).clamp_min(1e-8)
    return (probs * (log_probs - mean.log())).sum(-1).mean()


def loss_groups(
    model: RendererOrbitGroundedCompiler,
    groups: Sequence[Sequence[OrbitPilotRow]],
    device: torch.device,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not groups or len({len(group) for group in groups}) != 1:
        raise ValueError("renderer-orbit loss requires equal nonempty view groups")
    views = len(groups[0])
    rows = [row for group in groups for row in group]
    binding_rows = [row.binding for row in rows]
    program_ids, program_valid = byte_batch(binding_rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(binding_rows, "query_bytes", device)
    target = labels(binding_rows, device)
    program = model.compile_program(program_ids, program_valid)
    query = model.compile_query_with_evidence(query_ids, query_valid)
    tape = program.tape
    active = target["kind"].ne(STOP_KIND).reshape(-1)

    pieces: dict[str, torch.Tensor] = {
        "initial": F.cross_entropy(tape.initial_state, target["initial"]),
        "kind": F.cross_entropy(
            tape.event_kind.reshape(-1, 3),
            target["kind"].reshape(-1),
            weight=torch.tensor([1.0, 1.0, 4.0], device=device),
        ),
        "identity": F.cross_entropy(
            tape.event_identity.reshape(-1, 3)[active],
            target["identity"].reshape(-1)[active],
        ),
        "amount": F.cross_entropy(
            tape.amount.reshape(-1, 2)[active],
            target["amount"].reshape(-1)[active],
        ),
        "query": F.cross_entropy(query.query.logits, target["query"]),
    }
    line_mask, line_active = span_mask(
        binding_rows, "pointer_ranges", 9, program_ids.shape[1], device
    )
    binding_mask, binding_active = span_mask(
        binding_rows, "binding_ranges", 3, program_ids.shape[1], device
    )
    initial_mask, initial_active = span_mask(
        binding_rows, "initial_entity_ranges", 3, program_ids.shape[1], device
    )
    event_mask, event_active = span_mask(
        binding_rows, "event_entity_ranges", 8, program_ids.shape[1], device
    )
    pieces.update(
        {
            "line_address": uniform_span_loss(
                program.line_pointer_logits, line_mask, line_active
            ),
            "binding_address": uniform_span_loss(
                program.binding_pointer_logits, binding_mask, binding_active
            ),
            "initial_entity_address": uniform_span_loss(
                program.initial_entity_pointer_logits, initial_mask, initial_active
            ),
            "event_entity_address": uniform_span_loss(
                program.event_entity_pointer_logits, event_mask, event_active
            ),
            "query_address": uniform_span_loss(
                query.pointer_logits,
                _query_span_mask(rows, query_ids.shape[1], device),
                torch.ones(len(rows), dtype=torch.bool, device=device),
            ),
        }
    )
    consistency = sum(
        _orbit_consistency(value, len(groups), views)
        for value in (
            tape.initial_state,
            tape.event_kind,
            tape.event_identity,
            tape.amount,
            query.query.logits,
        )
    )
    pieces["orbit_consistency"] = consistency
    total = sum(value for name, value in pieces.items() if name != "orbit_consistency")
    total = total + consistency_weight * consistency
    pieces["total"] = total
    return total, {name: float(value.detach()) for name, value in pieces.items()}


@torch.no_grad()
def evaluate(
    model: RendererOrbitGroundedCompiler,
    groups: Sequence[Sequence[OrbitPilotRow]],
    family_batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(groups), family_batch_size):
        rows = [
            row for group in groups[start : start + family_batch_size] for row in group
        ]
        binding_rows = [row.binding for row in rows]
        program_ids, program_valid = byte_batch(binding_rows, "program_bytes", device)
        query_ids, query_valid = byte_batch(binding_rows, "query_bytes", device)
        target = labels(binding_rows, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            program = model.compile_program(program_ids, program_valid)
            query = model.compile_query_with_evidence(query_ids, query_valid)
        tape = program.tape
        kind = exact_one_stop_map(tape.event_kind)
        identity = tape.event_identity.argmax(-1)
        amount = tape.amount.argmax(-1)
        active = target["kind"].ne(STOP_KIND)
        exact = {
            "initial": tape.initial_state.argmax(-1).eq(target["initial"]),
            "kind": kind.eq(target["kind"]).all(-1),
            "identity": (identity.eq(target["identity"]) | ~active).all(-1),
            "amount": (amount.eq(target["amount"]) | ~active).all(-1),
            "query": query.query.logits.argmax(-1).eq(target["query"]),
        }
        line_mask, _ = span_mask(
            binding_rows, "pointer_ranges", 9, program_ids.shape[1], device
        )
        binding_mask, _ = span_mask(
            binding_rows, "binding_ranges", 3, program_ids.shape[1], device
        )
        initial_mask, _ = span_mask(
            binding_rows, "initial_entity_ranges", 3, program_ids.shape[1], device
        )
        event_mask, event_active = span_mask(
            binding_rows, "event_entity_ranges", 8, program_ids.shape[1], device
        )

        def pointer_exact(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            selected = logits.argmax(-1)
            return mask.gather(-1, selected[..., None]).squeeze(-1)

        exact["line_pointer"] = pointer_exact(
            program.line_pointer_logits, line_mask
        ).all(-1)
        exact["binding_pointer"] = pointer_exact(
            program.binding_pointer_logits, binding_mask
        ).all(-1)
        exact["initial_pointer"] = pointer_exact(
            program.initial_entity_pointer_logits, initial_mask
        ).all(-1)
        event_pointer = pointer_exact(program.event_entity_pointer_logits, event_mask)
        exact["event_pointer"] = (event_pointer | ~event_active).all(-1)
        exact["query_pointer"] = pointer_exact(
            query.pointer_logits,
            _query_span_mask(rows, query_ids.shape[1], device),
        )
        exact["whole_tape"] = (
            exact["initial"] & exact["kind"] & exact["identity"] & exact["amount"]
        )
        exact["packet"] = exact["whole_tape"] & exact["query"]
        for index, row in enumerate(rows):
            bucket = counts[row.renderer]
            bucket["rows"] += 1
            for name, values in exact.items():
                bucket[name] += int(values[index])
    return {
        renderer: {
            "rows": values["rows"],
            "exact": {
                name: value for name, value in sorted(values.items()) if name != "rows"
            },
            "rates": {
                name: value / values["rows"]
                for name, value in sorted(values.items())
                if name != "rows"
            },
        }
        for renderer, values in sorted(counts.items())
    }


def _minimum_rate(metrics: Mapping[str, object], field: str) -> float:
    return min(float(item["rates"][field]) for item in metrics.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-semantics", type=int, default=12_000)
    parser.add_argument("--heldout-semantics", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--family-batch-size", type=int, default=8)
    parser.add_argument("--eval-family-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing renderer-orbit output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("renderer-orbit pilot requires bf16 CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    source_rows = load_consumed_train(args.train_jsonl)
    fit_source, heldout_source = partition_rows(
        source_rows, args.fit_semantics, args.heldout_semantics
    )
    fit_groups = expand_orbit(fit_source, TRAIN_RENDERERS)
    heldout_groups = expand_orbit(heldout_source, HELD_OUT_RENDERERS)
    model, parameters = initialize_model(
        args.parent_checkpoint, args.v2_checkpoint, device
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    updates_per_epoch = math.ceil(len(fit_groups) / args.family_batch_size)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_scale(step, total_updates, args.warmup),
    )
    rng = random.Random(args.seed)
    history: list[dict[str, object]] = []
    update = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(fit_groups)))
        rng.shuffle(order)
        totals: Counter[str] = Counter()
        seen = 0
        for start in range(0, len(order), args.family_batch_size):
            groups = [
                fit_groups[index]
                for index in order[start : start + args.family_batch_size]
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = loss_groups(
                    model, groups, device, args.consistency_weight
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite renderer-orbit gradient")
            optimizer.step()
            scheduler.step()
            update += 1
            rows_seen = sum(len(group) for group in groups)
            seen += rows_seen
            for name, value in pieces.items():
                totals[name] += value * rows_seen
        heldout = evaluate(model, heldout_groups, args.eval_family_batch_size, device)
        record = {
            "epoch": epoch + 1,
            "updates": update,
            "fit_losses": {
                name: value / seen for name, value in sorted(totals.items())
            },
            "heldout_min_rates": {
                field: _minimum_rate(heldout, field)
                for field in ("packet", "whole_tape", "query", "query_pointer")
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    fit = evaluate(model, fit_groups, args.eval_family_batch_size, device)
    heldout = evaluate(model, heldout_groups, args.eval_family_batch_size, device)
    gates = {
        "heldout_initial_at_least_95pct": _minimum_rate(heldout, "initial") >= 0.95,
        "heldout_kind_at_least_95pct": _minimum_rate(heldout, "kind") >= 0.95,
        "heldout_identity_at_least_90pct": _minimum_rate(heldout, "identity") >= 0.90,
        "heldout_amount_at_least_95pct": _minimum_rate(heldout, "amount") >= 0.95,
        "heldout_query_at_least_99pct": _minimum_rate(heldout, "query") >= 0.99,
        "heldout_query_pointer_at_least_99pct": (
            _minimum_rate(heldout, "query_pointer") >= 0.99
        ),
        "heldout_packet_at_least_80pct": _minimum_rate(heldout, "packet") >= 0.80,
        "complete_system_below_200m": parameters["complete_system"]
        < GLOBAL_PARAMETER_CAP,
        "scored_access_zero": True,
    }
    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save(
        {
            "schema": "r12_sd_cst_renderer_orbit_training_pilot_v1",
            "seed": args.seed,
            "state": model.state_dict(),
            "trainable_names": parameters["trainable_names"],
            "development_accesses": 0,
            "confirmation_accesses": 0,
            "score_eligible": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": "r12_sd_cst_renderer_orbit_training_pilot_report_v1",
        "decision": (
            "advance_renderer_orbit_to_fresh_board"
            if all(gates.values())
            else "reject_or_revise_renderer_orbit_front_end"
        ),
        "seed": args.seed,
        "source": {
            "train_sha256": CONSUMED_TRAIN_SHA256,
            "parent_sha256": PARENT_SHA256,
            "v2_checkpoint_sha256": V2_CHECKPOINT_SHA256,
            "fit_source_sha256": _source_digest(fit_source),
            "heldout_source_sha256": _source_digest(heldout_source),
        },
        "partition": {
            "method": "sha256(row_id) ordering",
            "fit_semantics": len(fit_source),
            "heldout_semantics": len(heldout_source),
            "fit_views_per_semantic": len(TRAIN_RENDERERS),
            "heldout_views_per_semantic": len(HELD_OUT_RENDERERS),
        },
        "training": {
            "epochs": args.epochs,
            "updates": update,
            "family_batch_size": args.family_batch_size,
            "lr": args.lr,
            "warmup": args.warmup,
            "consistency_weight": args.consistency_weight,
            "elapsed_seconds": time.time() - started,
        },
        "parameters": parameters,
        "history": history,
        "fit": fit,
        "heldout": heldout,
        "gates": gates,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": (
            "Consumed training rows only; renderer/query identifiability pilot, "
            "not a reasoning score."
        ),
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "parameters": parameters,
                "heldout_min_packet": _minimum_rate(heldout, "packet"),
                "heldout_min_query": _minimum_rate(heldout, "query"),
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
