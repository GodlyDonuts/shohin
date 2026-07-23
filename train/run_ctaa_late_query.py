#!/usr/bin/env python3
"""Apply a separately disclosed query byte to a committed CTAA execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctaa_packet_io import read_query_file, sha256_file
from run_ctaa_packet_executor import EXECUTION_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing CTAA answer artifact: {args.output}")
    execution = torch.load(args.execution, map_location="cpu", weights_only=True)
    if not isinstance(execution, dict) or execution.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("CTAA execution artifact schema differs")
    state_route = execution.get("state_route")
    if not isinstance(state_route, torch.Tensor) or state_route.ndim != 3:
        raise ValueError("CTAA execution state geometry differs")
    halted = execution.get("halted")
    composed_states = execution.get("composed_states")
    if (
        not isinstance(halted, torch.Tensor)
        or halted.shape != state_route.shape[:2]
        or halted.dtype != torch.bool
        or not isinstance(composed_states, torch.Tensor)
        or composed_states.shape != state_route.shape
    ):
        raise ValueError("CTAA execution receipt geometry differs")
    rising = halted[:, 1:].to(torch.int8) - halted[:, :-1].to(torch.int8)
    if (
        bool(halted[:, 0].any())
        or not bool(halted[:, -1].all())
        or not bool(rising.eq(1).sum(1).eq(1).all())
        or bool(rising.lt(0).any())
    ):
        raise ValueError("CTAA execution STOP receipt differs")
    if not torch.equal(state_route, composed_states):
        raise ValueError("CTAA execution routes disagree")
    query = read_query_file(args.query)
    if query.position.shape[0] != state_route.shape[0]:
        raise ValueError("CTAA late-query batch differs")
    terminal = state_route[:, -1].long()
    answers = terminal.gather(1, query.position.long()[:, None]).squeeze(1)
    payload = {
        "schema": "ctaa_late_query_answer_v2",
        "execution_sha256": sha256_file(args.execution),
        "query_sha256": sha256_file(args.query),
        "answers": answers.tolist(),
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA answer temporary: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.chmod(0o444)
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    args.output.chmod(0o444)
    print(f"answer_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
