#!/usr/bin/env python3
"""Compile development worlds into source-deleted Shohin workspace states.

This process receives no development queries, labels, cluster metadata, or
offline oracle products.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

import pipeline.episode_workspace_custody as custody_module
from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    DEVELOPMENT_WORLD_SCHEMA,
    REPOSITORY_ROOT,
    WORLD_TOKENS,
    abort_atomic_bundle,
    atomic_bundle_directory,
    committed_source_receipt,
    file_sha256,
    finish_atomic_bundle,
    fsync_directory,
    read_jsonl_verified,
    verify_landlock_stage,
    write_json_fsync,
)
from causal_bind_select_workspace import CausalWorkspaceConfig, WorkspaceState
from workspace_checkpoint import (
    CHECKPOINT_SOURCE_PATH,
    MODEL_SOURCE_PATH,
    PROTECTED_BASE_STATE_SHA256,
    WORKSPACE_SOURCE_PATH,
    load_protected_workspace_model,
    load_workspace_delta,
    runtime_source_manifest,
    state_dict_sha256,
)
import workspace_state_custody as state_custody_module
from workspace_state_custody import (
    COMPILER_SOURCE_RECEIPT_SCHEMA,
    save_compiled_states,
)


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "train/flagship_out/ckpt_0300000.pt"
COMPILE_REPORT_SCHEMA = "episode_causal_workspace_compile_v1"
COMPILE_BUNDLE_SCHEMA = "episode_causal_workspace_compile_bundle_v1"


class EpisodeWorkspaceCompileError(ValueError):
    """A world-only compile or publication invariant failed."""


def _integer_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise EpisodeWorkspaceCompileError("world tokens must be an integer list")
    return tuple(value)


def load_worlds(
    path: Path, expected_sha256: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    rows = read_jsonl_verified(path, expected_sha256)
    if len(rows) != 192:
        raise EpisodeWorkspaceCompileError("world source must contain 192 rows")
    worlds: list[tuple[str, tuple[int, ...]]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"schema", "world_id", "world_tokens"}:
            raise EpisodeWorkspaceCompileError("world row has unexpected fields")
        if row.get("schema") != DEVELOPMENT_WORLD_SCHEMA:
            raise EpisodeWorkspaceCompileError("world row schema is invalid")
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or len(world_id) != 64:
            raise EpisodeWorkspaceCompileError("world ID is invalid")
        if world_id in seen:
            raise EpisodeWorkspaceCompileError("world ID is duplicated")
        seen.add(world_id)
        tokens = _integer_tuple(row.get("world_tokens"))
        if len(tokens) != WORLD_TOKENS:
            raise EpisodeWorkspaceCompileError("world length drifted")
        worlds.append((world_id, tokens))
    return tuple(worlds)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EpisodeWorkspaceCompileError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EpisodeWorkspaceCompileError("MPS requested but unavailable")
    return device


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _source_receipt(expected_sha256: str) -> dict[str, object]:
    source = Path(__file__).resolve()
    try:
        receipt = committed_source_receipt(
            source,
            expected_sha256,
            (
                Path(custody_module.__file__),
                Path(state_custody_module.__file__),
                WORKSPACE_SOURCE_PATH,
                MODEL_SOURCE_PATH,
                CHECKPOINT_SOURCE_PATH,
            ),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceCompileError(str(exc)) from exc
    return {**receipt, "runtime_source_manifest": runtime_source_manifest()}


def run(args: argparse.Namespace) -> dict[str, object]:
    landlock_receipt = verify_landlock_stage("compiler", args.deny_probe)
    source_before = _source_receipt(args.expected_source_sha256)
    worlds = load_worlds(args.worlds, args.expected_worlds_sha256)
    device = _resolve_device(args.device)
    model, protected_receipt = load_protected_workspace_model(
        args.checkpoint,
        CausalWorkspaceConfig(),
    )
    load_workspace_delta(
        args.workspace_delta,
        model,
        protected_receipt,
        expected_sha256=args.expected_delta_sha256,
    )
    model.to(device)
    model.eval()
    base_before = state_dict_sha256(model.base.state_dict())
    states: dict[str, WorkspaceState] = {}
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(worlds), args.batch_size):
            subset = worlds[start : start + args.batch_size]
            idx = torch.tensor(
                [tokens for _, tokens in subset],
                dtype=torch.long,
                device=device,
            )
            with _autocast(device):
                state = model.compile_world_state(idx)
            if state.token_position != WORLD_TOKENS or not state.sealed:
                raise EpisodeWorkspaceCompileError("compiler emitted an invalid state")
            for row, (world_id, _) in enumerate(subset):
                states[world_id] = WorkspaceState(
                    slots=state.slots[row : row + 1].detach().clone(),
                    token_position=state.token_position,
                    sealed=True,
                )
    if len(states) != 192:
        raise EpisodeWorkspaceCompileError("compiler state cardinality drifted")
    base_after = state_dict_sha256(model.base.state_dict())
    if base_before != base_after or base_after != PROTECTED_BASE_STATE_SHA256:
        raise EpisodeWorkspaceCompileError("protected base changed while compiling")
    source_after = _source_receipt(args.expected_source_sha256)
    if source_after != source_before:
        raise EpisodeWorkspaceCompileError("source receipt changed while compiling")

    report = {
        "schema": COMPILE_REPORT_SCHEMA,
        "claim_scope": (
            "world-only source deletion stage; no query, label, neural-fit, "
            "reasoning, or continuation-pretraining claim"
        ),
        "source": source_before,
        "process_id": os.getpid(),
        "landlock_receipt": landlock_receipt,
        "compiler_visible_input": {
            "path": str(args.worlds.absolute()),
            "sha256": args.expected_worlds_sha256,
            "worlds": len(worlds),
        },
        "forbidden_inputs_opened": [],
        "workspace_delta": {
            "path": str(args.workspace_delta.absolute()),
            "sha256": args.expected_delta_sha256,
        },
        "protected_checkpoint": asdict(protected_receipt),
        "protected_base_sha256_before": base_before,
        "protected_base_sha256_after": base_after,
        "worlds_compiled": len(states),
        "token_position": WORLD_TOKENS,
        "source_tokens_serialized": False,
        "query_tokens_seen": False,
        "labels_seen": False,
        "elapsed_seconds": time.monotonic() - started,
        "pretraining_started": False,
        "continuation_pretraining_authorized": False,
    }
    staging, lock = atomic_bundle_directory(args.output)
    try:
        state_path = staging / "compiled_states.pt"
        state_sha256 = save_compiled_states(
            state_path,
            states,
            model=model,
            protected_receipt=protected_receipt,
            workspace_delta_sha256=args.expected_delta_sha256,
            world_source_sha256=args.expected_worlds_sha256,
            compiler_source_receipt={
                "schema": COMPILER_SOURCE_RECEIPT_SCHEMA,
                **source_before,
                "process_id": os.getpid(),
                "landlock_receipt": landlock_receipt,
            },
        )
        report = {**report, "compiled_states_sha256": state_sha256}
        report_path = staging / "compile_report.json"
        write_json_fsync(report_path, report)
        manifest = {
            "schema": COMPILE_BUNDLE_SCHEMA,
            "files": {
                "compiled_states.pt": state_sha256,
                "compile_report.json": file_sha256(report_path),
            },
            "source_tokens_serialized": False,
            "pretraining_started": False,
        }
        write_json_fsync(staging / "bundle_manifest.json", manifest)
        fsync_directory(staging)
        finish_atomic_bundle(staging, args.output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worlds",
        type=Path,
        default=DEFAULT_CUSTODY_BUNDLE / "development_worlds.jsonl",
    )
    parser.add_argument("--expected-worlds-sha256", required=True)
    parser.add_argument("--workspace-delta", type=Path, required=True)
    parser.add_argument("--expected-delta-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise EpisodeWorkspaceCompileError("batch size must be positive")
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "worlds_compiled": report["worlds_compiled"],
                "compiled_states_sha256": report["compiled_states_sha256"],
                "pretraining_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
