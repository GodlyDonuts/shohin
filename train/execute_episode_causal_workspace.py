#!/usr/bin/env python3
"""Execute late EPISODE queries from source-deleted Shohin workspace states.

This process receives no world tokens, targets, candidate-state sets, cluster
metadata, or offline oracle products. It freezes label-blind hard predictions
and full answer-position logits for a later one-access assessor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Literal

import torch

import pipeline.episode_action_binding_board as episode_board
import pipeline.episode_workspace_custody as custody_module
from pipeline.episode_action_binding_board import ANSWER, EOS
from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    DEVELOPMENT_QUERY_SCHEMA,
    REPOSITORY_ROOT,
    abort_atomic_bundle,
    atomic_bundle_directory,
    canonical_json,
    committed_source_receipt,
    file_sha256,
    finish_atomic_bundle,
    fsync_directory,
    read_jsonl_verified,
    verify_landlock_stage,
    write_json_fsync,
    write_jsonl_fsync,
)
from causal_bind_select_workspace import (
    CausalWorkspaceConfig,
    WorkspaceControls,
    WorkspaceState,
)
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
from workspace_state_custody import load_compiled_states


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "train/flagship_out/ckpt_0300000.pt"
EXECUTION_REPORT_SCHEMA = "episode_causal_workspace_execution_v1"
EXECUTION_BUNDLE_SCHEMA = "episode_causal_workspace_execution_bundle_v1"
LOGITS_SCHEMA = "episode_causal_workspace_answer_logits_v1"
ControlName = Literal[
    "treatment",
    "zero_workspace",
    "uniform_binding",
    "uniform_operator",
    "binding_permutation",
    "operator_permutation",
    "selected_slot_scramble",
    "discarded_slot_scramble",
]
CONTROL_NAMES: tuple[ControlName, ...] = (
    "treatment",
    "zero_workspace",
    "uniform_binding",
    "uniform_operator",
    "binding_permutation",
    "operator_permutation",
    "selected_slot_scramble",
    "discarded_slot_scramble",
)


class EpisodeWorkspaceExecutionError(ValueError):
    """A query-only execution or publication invariant failed."""


def _integer_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise EpisodeWorkspaceExecutionError("query tokens must be an integer list")
    return tuple(value)


def load_queries(
    path: Path, expected_sha256: str
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    rows = read_jsonl_verified(path, expected_sha256)
    if len(rows) != 384:
        raise EpisodeWorkspaceExecutionError("query source must contain 384 rows")
    queries: list[tuple[str, str, tuple[int, ...]]] = []
    seen: set[str] = set()
    worlds: Counter[str] = Counter()
    for row in rows:
        if set(row) != {
            "schema",
            "packet_sha256",
            "world_id",
            "query_tokens",
        }:
            raise EpisodeWorkspaceExecutionError("query row has unexpected fields")
        if row.get("schema") != DEVELOPMENT_QUERY_SCHEMA:
            raise EpisodeWorkspaceExecutionError("query row schema is invalid")
        packet_sha256 = row.get("packet_sha256")
        world_id = row.get("world_id")
        if not isinstance(packet_sha256, str) or len(packet_sha256) != 64:
            raise EpisodeWorkspaceExecutionError("packet digest is invalid")
        if not isinstance(world_id, str) or len(world_id) != 64:
            raise EpisodeWorkspaceExecutionError("world ID is invalid")
        if packet_sha256 in seen:
            raise EpisodeWorkspaceExecutionError("packet digest is duplicated")
        seen.add(packet_sha256)
        worlds[world_id] += 1
        query = _integer_tuple(row.get("query_tokens"))
        if len(query) not in {13, 15}:
            raise EpisodeWorkspaceExecutionError(
                "development query length is outside depth 5-6"
            )
        if query[-2:] != (ANSWER, EOS) or query.count(ANSWER) != 1:
            raise EpisodeWorkspaceExecutionError("query grammar drifted")
        queries.append((packet_sha256, world_id, query))
    if len(worlds) != 192 or set(worlds.values()) != {2}:
        raise EpisodeWorkspaceExecutionError(
            "each source-deleted world must have exactly two late queries"
        )
    return tuple(sorted(queries))


def _control(name: ControlName) -> WorkspaceControls | None:
    if name == "treatment":
        return None
    if name == "zero_workspace":
        return WorkspaceControls(zero_workspace=True)
    if name == "uniform_binding":
        return WorkspaceControls(uniform_binding=True)
    if name == "uniform_operator":
        return WorkspaceControls(uniform_operator=True)
    if name == "binding_permutation":
        return WorkspaceControls(binding_permutation=(1, 2, 3, 0))
    if name == "operator_permutation":
        return WorkspaceControls(operator_permutation=(1, 2, 3, 0))
    if name == "selected_slot_scramble":
        return WorkspaceControls(scramble_selected_slot=True)
    if name == "discarded_slot_scramble":
        return WorkspaceControls(scramble_discarded_slots=True)
    raise EpisodeWorkspaceExecutionError(f"unknown control {name}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EpisodeWorkspaceExecutionError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EpisodeWorkspaceExecutionError("MPS requested but unavailable")
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
                Path(episode_board.__file__),
                Path(state_custody_module.__file__),
                WORKSPACE_SOURCE_PATH,
                MODEL_SOURCE_PATH,
                CHECKPOINT_SOURCE_PATH,
            ),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceExecutionError(str(exc)) from exc
    return {**receipt, "runtime_source_manifest": runtime_source_manifest()}


def execute_control(
    model,
    queries: tuple[tuple[str, str, tuple[int, ...]], ...],
    states: dict[str, WorkspaceState],
    *,
    name: ControlName,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    by_length: dict[int, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
    for query in queries:
        by_length[len(query[2])].append(query)
    output_by_digest: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    controls = _control(name)
    with torch.no_grad():
        for length in sorted(by_length):
            values = by_length[length]
            for start in range(0, len(values), batch_size):
                subset = values[start : start + batch_size]
                query_idx = torch.tensor(
                    [query for _, _, query in subset],
                    dtype=torch.long,
                    device=device,
                )
                slot_tensor = torch.cat(
                    [states[world_id].slots for _, world_id, _ in subset],
                    dim=0,
                ).to(device=device)
                workspace = WorkspaceState(
                    slots=slot_tensor,
                    token_position=145,
                    sealed=True,
                )
                with _autocast(device):
                    logits, diagnostics = model.execute_workspace_state(
                        workspace,
                        query_idx,
                        controls=controls,
                    )
                answer_logits = logits[:, length - 2].float().cpu()
                binding_indices = diagnostics.bindings.argmax(dim=-1).cpu()
                operator_indices = diagnostics.operator_probabilities.argmax(
                    dim=-1
                ).cpu()
                if name != "uniform_binding" and not torch.all(
                    (diagnostics.bindings == 0) | (diagnostics.bindings == 1)
                ):
                    raise EpisodeWorkspaceExecutionError(
                        "binding route is not exactly one-hot"
                    )
                if name != "uniform_operator" and not torch.all(
                    (diagnostics.operator_probabilities == 0)
                    | (diagnostics.operator_probabilities == 1)
                ):
                    raise EpisodeWorkspaceExecutionError(
                        "operator route is not exactly one-hot"
                    )
                for row, (packet_sha256, _, _) in enumerate(subset):
                    output_by_digest[packet_sha256] = (
                        answer_logits[row],
                        binding_indices[row, :length],
                        operator_indices[row, :length],
                    )
    packet_order = [packet_sha256 for packet_sha256, _, _ in queries]
    if set(output_by_digest) != set(packet_order):
        raise EpisodeWorkspaceExecutionError("execution output coverage drifted")
    answer_logits = torch.stack(
        [output_by_digest[digest][0] for digest in packet_order]
    )
    bindings = torch.full((len(packet_order), 15), -1, dtype=torch.int64)
    operators = torch.full((len(packet_order), 15), -1, dtype=torch.int64)
    prediction_rows: list[dict[str, object]] = []
    for row, digest in enumerate(packet_order):
        binding = output_by_digest[digest][1]
        operator = output_by_digest[digest][2]
        bindings[row, : binding.numel()] = binding
        operators[row, : operator.numel()] = operator
        prediction_rows.append(
            {
                "control": name,
                "packet_sha256": digest,
                "predicted_token": int(answer_logits[row].argmax().item()),
            }
        )
    return answer_logits, bindings, operators, prediction_rows


def _write_logits_fsync(
    path: Path,
    *,
    name: ControlName,
    packet_order: list[str],
    answer_logits: torch.Tensor,
    bindings: torch.Tensor,
    operators: torch.Tensor,
    receipts: dict[str, object],
) -> None:
    payload = {
        "schema": LOGITS_SCHEMA,
        "control": name,
        "packet_sha256": packet_order,
        "answer_logits": answer_logits,
        "binding_indices": bindings,
        "operator_indices": operators,
        "receipts": receipts,
        "targets_seen": False,
        "candidate_sets_seen": False,
        "pretraining_started": False,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0:
        raise EpisodeWorkspaceExecutionError("batch size must be positive")
    landlock_receipt = verify_landlock_stage("executor", args.deny_probe)
    source_before = _source_receipt(args.expected_source_sha256)
    queries = load_queries(args.queries, args.expected_queries_sha256)
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
    states, state_receipt = load_compiled_states(
        args.compiled_states,
        model=model,
        protected_receipt=protected_receipt,
        expected_sha256=args.expected_states_sha256,
        expected_workspace_delta_sha256=args.expected_delta_sha256,
        expected_world_source_sha256=args.expected_worlds_sha256,
        expected_compiler_source_sha256=args.expected_compiler_source_sha256,
        expected_repository_commit=args.expected_repository_commit,
    )
    if {world_id for _, world_id, _ in queries} != set(states):
        raise EpisodeWorkspaceExecutionError(
            "query and compiled-state world coverage differ"
        )
    model.to(device)
    model.eval()
    base_before = state_dict_sha256(model.base.state_dict())
    started = time.monotonic()
    staging, lock = atomic_bundle_directory(args.output)
    try:
        packet_order = [packet_sha256 for packet_sha256, _, _ in queries]
        prediction_rows: list[dict[str, object]] = []
        file_hashes: dict[str, str] = {}
        causal_logits: dict[str, torch.Tensor] = {}
        control_receipts = {
            "query_source_sha256": args.expected_queries_sha256,
            "compiled_states_sha256": args.expected_states_sha256,
            "workspace_delta_sha256": args.expected_delta_sha256,
        }
        for name in CONTROL_NAMES:
            answer_logits, bindings, operators, rows = execute_control(
                model,
                queries,
                states,
                name=name,
                batch_size=args.batch_size,
                device=device,
            )
            logits_name = f"answer_logits_{name}.pt"
            logits_path = staging / logits_name
            _write_logits_fsync(
                logits_path,
                name=name,
                packet_order=packet_order,
                answer_logits=answer_logits,
                bindings=bindings,
                operators=operators,
                receipts=control_receipts,
            )
            file_hashes[logits_name] = file_sha256(logits_path)
            prediction_rows.extend(rows)
            if name in {"treatment", "discarded_slot_scramble"}:
                causal_logits[name] = answer_logits
            print(
                canonical_json(
                    {
                        "event": "label_blind_execution",
                        "control": name,
                        "packets": len(rows),
                    }
                ),
                flush=True,
            )
        if not torch.equal(
            causal_logits["treatment"],
            causal_logits["discarded_slot_scramble"],
        ):
            raise EpisodeWorkspaceExecutionError(
                "discarded-slot scramble changed hard-routed answer logits"
            )
        prediction_path = staging / "label_blind_predictions.jsonl"
        write_jsonl_fsync(prediction_path, prediction_rows)
        file_hashes[prediction_path.name] = file_sha256(prediction_path)
        base_after = state_dict_sha256(model.base.state_dict())
        if base_before != base_after or base_after != PROTECTED_BASE_STATE_SHA256:
            raise EpisodeWorkspaceExecutionError(
                "protected base changed during execution"
            )
        source_after = _source_receipt(args.expected_source_sha256)
        if source_after != source_before:
            raise EpisodeWorkspaceExecutionError(
                "source receipt changed during execution"
            )
        report = {
            "schema": EXECUTION_REPORT_SCHEMA,
            "claim_scope": (
                "query-only label-blind source-deleted execution; no assessment, "
                "reasoning, or continuation-pretraining claim"
            ),
            "source": source_before,
            "process_id": os.getpid(),
            "landlock_receipt": landlock_receipt,
            "executor_visible_inputs": {
                "queries": {
                    "path": str(args.queries.absolute()),
                    "sha256": args.expected_queries_sha256,
                },
                "compiled_states": {
                    "path": str(args.compiled_states.absolute()),
                    "sha256": args.expected_states_sha256,
                },
                "workspace_delta": {
                    "path": str(args.workspace_delta.absolute()),
                    "sha256": args.expected_delta_sha256,
                },
            },
            "forbidden_inputs_opened": [],
            "compiled_state_receipt": state_receipt,
            "protected_checkpoint": asdict(protected_receipt),
            "protected_base_sha256_before": base_before,
            "protected_base_sha256_after": base_after,
            "packets": len(queries),
            "controls": list(CONTROL_NAMES),
            "discarded_slot_logits_bit_identical": True,
            "hard_argmax": True,
            "retry_or_repair": False,
            "candidate_mask": False,
            "targets_seen": False,
            "candidate_sets_seen": False,
            "world_tokens_seen": False,
            "elapsed_seconds": time.monotonic() - started,
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        }
        report_path = staging / "execution_report.json"
        write_json_fsync(report_path, report)
        file_hashes[report_path.name] = file_sha256(report_path)
        manifest = {
            "schema": EXECUTION_BUNDLE_SCHEMA,
            "files": file_hashes,
            "targets_seen": False,
            "candidate_sets_seen": False,
            "world_tokens_seen": False,
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
        "--queries",
        type=Path,
        default=DEFAULT_CUSTODY_BUNDLE / "development_queries.jsonl",
    )
    parser.add_argument("--expected-queries-sha256", required=True)
    parser.add_argument("--compiled-states", type=Path, required=True)
    parser.add_argument("--expected-states-sha256", required=True)
    parser.add_argument("--expected-worlds-sha256", required=True)
    parser.add_argument("--workspace-delta", type=Path, required=True)
    parser.add_argument("--expected-delta-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-compiler-source-sha256", required=True)
    parser.add_argument("--expected-repository-commit", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "packets": report["packets"],
                "controls": report["controls"],
                "pretraining_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
