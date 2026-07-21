#!/usr/bin/env python3
"""Seal hard query bytes after validating the immutable execution binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctaa_evaluation_io import (
    read_packet_index,
    read_query_predictions,
    sha256_file,
)
from ctaa_packet_io import write_query_file
from ctaa_trunk_compiler import HardCTAAQuery
from run_ctaa_query_compiler import validate_execution_binding


def seal_queries(
    predictions_path: Path,
    packet_index_path: Path,
    execution_path: Path,
    output_path: Path,
) -> dict[str, object]:
    packet_index = read_packet_index(packet_index_path)
    validate_execution_binding(execution_path, packet_index)
    predictions = read_query_predictions(predictions_path)
    if (
        predictions["family_ids"] != packet_index["valid_family_ids"]
        or predictions["execution_sha256"] != sha256_file(execution_path)
    ):
        raise ValueError("CTAA sealed-query row or execution binding differs")
    receipt = write_query_file(
        output_path,
        HardCTAAQuery(position=predictions["positions"]),
    )
    return {
        **receipt,
        "query_predictions_sha256": sha256_file(predictions_path),
        "execution_sha256": sha256_file(execution_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--packet-index", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seal_queries(
                args.predictions,
                args.packet_index,
                args.execution,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

