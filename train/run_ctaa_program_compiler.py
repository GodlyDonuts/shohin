#!/usr/bin/env python3
"""Compile program-only CTAA source without opening query or oracle files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    packet_valid_mask,
    sha256_file,
    validate_program_predictions,
    write_torch_once,
)
from ctaa_frozen_compiler import load_frozen_compiler, load_source_rows, token_batches


@torch.inference_mode()
def compile_program_sources(
    compiler,
    tokenizer,
    sources: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cards = []
    initial = []
    schedules = []
    for ids in token_batches(
        sources,
        tokenizer,
        batch_size=batch_size,
        max_length=compiler.model.cfg.seq_len,
        padding_id=compiler.padding_id,
        device=device,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = compiler.compile_program(ids)
        cards.append(logits.action_cards.argmax(-1).to(torch.uint8).cpu())
        initial.append(logits.initial_state.argmax(-1).to(torch.uint8).cpu())
        schedules.append(logits.schedule.argmax(-1).to(torch.uint8).cpu())
    return torch.cat(cards), torch.cat(initial), torch.cat(schedules)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--program-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    family_ids, sources = load_source_rows(args.program_source, "program_source")
    bundle = load_frozen_compiler(
        base_path=args.base,
        qualified_path=args.qualified_compiler,
        tokenizer_path=args.tokenizer,
        compiler_path=args.compiler,
        device_name=args.device,
    )
    cards, initial, schedule = compile_program_sources(
        bundle.compiler,
        bundle.tokenizer,
        sources,
        batch_size=args.batch_size,
        device=bundle.device,
    )
    payload = validate_program_predictions(
        {
            "schema": PROGRAM_PREDICTION_SCHEMA,
            "family_ids": family_ids,
            "program_source_sha256": sha256_file(args.program_source),
            "compiler_sha256": bundle.compiler_sha256,
            "action_cards": cards,
            "initial_state": initial,
            "schedule": schedule,
            "packet_valid": packet_valid_mask(schedule),
        }
    )
    digest = write_torch_once(args.output, payload)
    report = {
        "schema": PROGRAM_PREDICTION_SCHEMA,
        "rows": len(family_ids),
        "packet_valid": int(payload["packet_valid"].sum()),
        "packet_invalid": int((~payload["packet_valid"]).sum()),
        "output_sha256": digest,
        "development_access": 0,
        "confirmation_access": 0,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

