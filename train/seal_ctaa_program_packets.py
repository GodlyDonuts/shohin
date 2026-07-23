#!/usr/bin/env python3
"""Seal only valid source-compiler rows into fixed CTAA packet bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctaa_evaluation_io import (
    PACKET_INDEX_SCHEMA,
    read_program_predictions,
    sha256_file,
    validate_packet_index,
    write_json_once,
)
from ctaa_packet_io import write_packet_file
from ctaa_trunk_compiler import HardCTAAPacket


def seal_predictions(predictions_path: Path, packet_path: Path, index_path: Path) -> dict[str, object]:
    if packet_path.exists() or index_path.exists():
        raise FileExistsError("refusing existing CTAA packet-sealing output")
    predictions = read_program_predictions(predictions_path)
    valid = predictions["packet_valid"]
    indices = valid.nonzero(as_tuple=False).flatten().tolist()
    family_ids = predictions["family_ids"]
    packet_sha: str | None = None
    if indices:
        packet = HardCTAAPacket(
            action_cards=predictions["action_cards"][indices],
            opcode_to_card=predictions["opcode_to_card"][indices],
            initial_state=predictions["initial_state"][indices],
            opcode_schedule=predictions["opcode_schedule"][indices],
        )
        packet_sha = str(write_packet_file(packet_path, packet)["sha256"])
    index = validate_packet_index(
        {
            "schema": PACKET_INDEX_SCHEMA,
            "program_predictions_sha256": sha256_file(predictions_path),
            "packet_sha256": packet_sha,
            "valid_family_ids": [family_ids[index] for index in indices],
            "valid_source_indices": indices,
            "invalid_family_ids": [
                family_id
                for index, family_id in enumerate(family_ids)
                if not bool(valid[index])
            ],
        }
    )
    index_sha = write_json_once(index_path, index)
    return {
        "schema": PACKET_INDEX_SCHEMA,
        "rows": len(family_ids),
        "valid_rows": len(indices),
        "invalid_rows": len(family_ids) - len(indices),
        "packet_sha256": packet_sha,
        "index_sha256": index_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seal_predictions(args.predictions, args.packet, args.index),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
