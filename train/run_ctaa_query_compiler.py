#!/usr/bin/env python3
"""Compile sealed query source only after packet execution is immutable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctaa_evaluation_io import (
    QUERY_PREDICTION_SCHEMA,
    read_packet_index,
    sha256_file,
    validate_query_predictions,
    write_torch_once,
)
from ctaa_frozen_compiler import load_frozen_compiler, load_source_rows, token_batches
from run_ctaa_packet_executor import EXECUTION_SCHEMA


def validate_execution_binding(execution_path: Path, packet_index: dict[str, object]) -> dict[str, object]:
    execution = torch.load(execution_path, map_location="cpu", weights_only=True)
    if not isinstance(execution, dict) or execution.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("CTAA late-query execution schema differs")
    valid_ids = packet_index["valid_family_ids"]
    state_route = execution.get("state_route")
    if (
        not valid_ids
        or not isinstance(state_route, torch.Tensor)
        or state_route.ndim != 3
        or state_route.shape[0] != len(valid_ids)
        or execution.get("packet_sha256") != packet_index["packet_sha256"]
    ):
        raise ValueError("CTAA late-query execution binding differs")
    if execution_path.stat().st_mode & 0o222:
        raise PermissionError("CTAA execution receipt is not immutable")
    return execution


@torch.inference_mode()
def compile_query_sources(
    compiler,
    tokenizer,
    sources: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    positions = []
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
            logits = compiler.compile_query(ids)
        positions.append(logits.argmax(-1).to(torch.uint8).cpu())
    return torch.cat(positions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--packet-index", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--query-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    packet_index = read_packet_index(args.packet_index)
    validate_execution_binding(args.execution, packet_index)
    all_ids, all_sources = load_source_rows(args.query_source, "query_source")
    expected_ids = set(packet_index["valid_family_ids"] + packet_index["invalid_family_ids"])
    if set(all_ids) != expected_ids:
        raise ValueError("CTAA query-source family set differs from packet compile")
    source_by_id = dict(zip(all_ids, all_sources, strict=True))
    valid_ids = packet_index["valid_family_ids"]
    sources = [source_by_id[family_id] for family_id in valid_ids]
    bundle = load_frozen_compiler(
        base_path=args.base,
        qualified_path=args.qualified_compiler,
        tokenizer_path=args.tokenizer,
        compiler_path=args.compiler,
        device_name=args.device,
    )
    positions = compile_query_sources(
        bundle.compiler,
        bundle.tokenizer,
        sources,
        batch_size=args.batch_size,
        device=bundle.device,
    )
    payload = validate_query_predictions(
        {
            "schema": QUERY_PREDICTION_SCHEMA,
            "family_ids": valid_ids,
            "query_source_sha256": sha256_file(args.query_source),
            "compiler_sha256": bundle.compiler_sha256,
            "execution_sha256": sha256_file(args.execution),
            "positions": positions,
        }
    )
    digest = write_torch_once(args.output, payload)
    print(
        json.dumps(
            {
                "schema": QUERY_PREDICTION_SCHEMA,
                "rows": len(valid_ids),
                "output_sha256": digest,
                "development_access": 0,
                "confirmation_access": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

