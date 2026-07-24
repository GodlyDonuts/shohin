#!/usr/bin/env python3
"""Fit only Shohin's causal workspace on a train-only EPISODE artifact.

This process is intentionally unable to open development worlds, queries,
labels, or offline metadata. It loads the protected step-300k trunk read-only,
freezes every base parameter, and serializes only the workspace delta.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import Tensor

import pipeline.episode_action_binding_board as episode_board
import pipeline.episode_workspace_custody as custody_module
from pipeline.episode_action_binding_board import ANSWER, EOS
from pipeline.episode_workspace_custody import (
    CASES_PER_CLUSTER,
    DEFAULT_CUSTODY_BUNDLE,
    REPOSITORY_ROOT,
    TRAIN_GROUP_SCHEMA,
    WORLD_TOKENS,
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
)
from causal_bind_select_workspace import (
    CausalWorkspaceConfig,
    CausalWorkspaceGPT,
    freeze_protected_base,
    trainable_workspace_parameters,
)
from workspace_checkpoint import (
    CHECKPOINT_SOURCE_PATH,
    MODEL_SOURCE_PATH,
    PROTECTED_BASE_STATE_SHA256,
    PROTECTED_CHECKPOINT_SHA256,
    WORKSPACE_SOURCE_PATH,
    load_protected_workspace_model,
    runtime_source_manifest,
    save_workspace_delta,
    state_dict_sha256,
)


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "train/flagship_out/ckpt_0300000.pt"
FIT_REPORT_SCHEMA = "episode_causal_workspace_fit_v1"
FIT_BUNDLE_SCHEMA = "episode_causal_workspace_fit_bundle_v1"
EXPECTED_ARM_INPUTS = {
    "true": {
        "name": "train_true_groups.jsonl",
        "sha256": "80d7e6e503d4aebbda506fcb3d321f1a91db556f7fe1bd2ab8e6ee92d2fbec27",
    },
    "shuffled": {
        "name": "train_shuffled_groups.jsonl",
        "sha256": "5917049c910cdc2beae667165465052c71329033be30532a4cec5a04fe419038",
    },
}


class EpisodeWorkspaceFitError(ValueError):
    """A train-only source, optimization, or publication invariant failed."""


def validate_frozen_arm_input(
    arm: str,
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    if arm not in EXPECTED_ARM_INPUTS:
        raise EpisodeWorkspaceFitError("unknown fit arm")
    expected = EXPECTED_ARM_INPUTS[arm]
    if path.name != expected["name"]:
        raise EpisodeWorkspaceFitError("fit arm references the wrong frozen input")
    if expected_sha256 != expected["sha256"]:
        raise EpisodeWorkspaceFitError("fit arm hash differs from the frozen ledger")
    return dict(expected)


@dataclass(frozen=True)
class FitExample:
    packet_sha256: str
    world_tokens: tuple[int, ...]
    query_tokens: tuple[int, ...]
    target_token: int


@dataclass(frozen=True)
class FitGroup:
    examples: tuple[FitExample, ...]
    query_length: int


@dataclass(frozen=True)
class FitBatch:
    world_idx: Tensor
    query_idx: Tensor
    targets: Tensor
    answer_index: int
    packet_digests: tuple[str, ...]


@dataclass(frozen=True)
class FitConfig:
    updates: int = 800
    groups_per_batch: int = 8
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 2026072347
    log_interval: int = 25

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise EpisodeWorkspaceFitError("updates must be positive")
        if self.groups_per_batch <= 0:
            raise EpisodeWorkspaceFitError("groups per batch must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise EpisodeWorkspaceFitError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise EpisodeWorkspaceFitError(
                "weight decay must be finite and nonnegative"
            )
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise EpisodeWorkspaceFitError("gradient clip must be finite and positive")
        if self.log_interval <= 0:
            raise EpisodeWorkspaceFitError("log interval must be positive")


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise EpisodeWorkspaceFitError(f"{label} must be an integer list")
    return tuple(value)


def load_train_groups(path: Path, expected_sha256: str) -> tuple[FitGroup, ...]:
    """Open exactly one optimization artifact and no custody manifest."""

    rows = read_jsonl_verified(path, expected_sha256)
    if len(rows) != 256:
        raise EpisodeWorkspaceFitError("train artifact must contain 256 groups")
    groups: list[FitGroup] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"schema", "examples"}:
            raise EpisodeWorkspaceFitError("train group has unexpected fields")
        if row.get("schema") != TRAIN_GROUP_SCHEMA:
            raise EpisodeWorkspaceFitError("train group schema is invalid")
        raw_examples = row.get("examples")
        if not isinstance(raw_examples, list) or len(raw_examples) != CASES_PER_CLUSTER:
            raise EpisodeWorkspaceFitError(
                "train group is not an indivisible six-case cluster"
            )
        examples: list[FitExample] = []
        for value in raw_examples:
            if not isinstance(value, dict) or set(value) != {
                "packet_sha256",
                "world_tokens",
                "query_tokens",
                "target_token",
            }:
                raise EpisodeWorkspaceFitError("train example has unexpected fields")
            digest = value.get("packet_sha256")
            target = value.get("target_token")
            if not isinstance(digest, str) or len(digest) != 64:
                raise EpisodeWorkspaceFitError("packet digest is invalid")
            if digest in seen:
                raise EpisodeWorkspaceFitError("packet digest is duplicated")
            seen.add(digest)
            if not isinstance(target, int) or isinstance(target, bool):
                raise EpisodeWorkspaceFitError("target token is invalid")
            world = _integer_tuple(value.get("world_tokens"), "world_tokens")
            query = _integer_tuple(value.get("query_tokens"), "query_tokens")
            if len(world) != WORLD_TOKENS:
                raise EpisodeWorkspaceFitError("train world length drifted")
            if query[-2:] != (ANSWER, EOS) or query.count(ANSWER) != 1:
                raise EpisodeWorkspaceFitError("train query grammar drifted")
            examples.append(
                FitExample(
                    packet_sha256=digest,
                    world_tokens=world,
                    query_tokens=query,
                    target_token=target,
                )
            )
        query_lengths = {len(example.query_tokens) for example in examples}
        if len(query_lengths) != 1:
            raise EpisodeWorkspaceFitError("one train group mixes active query lengths")
        query_length = query_lengths.pop()
        if query_length not in {7, 9, 11}:
            raise EpisodeWorkspaceFitError("train query length is outside depth 2-4")
        groups.append(
            FitGroup(
                examples=tuple(examples),
                query_length=query_length,
            )
        )
    if len(seen) != 1_536:
        raise EpisodeWorkspaceFitError("train packet cardinality drifted")
    return tuple(groups)


class LengthBucketScheduler:
    """Epoch-complete batches of whole clusters with one active length."""

    def __init__(
        self,
        groups: tuple[FitGroup, ...],
        *,
        groups_per_batch: int,
        seed: int,
    ):
        buckets: dict[int, list[FitGroup]] = defaultdict(list)
        for group in groups:
            buckets[group.query_length].append(group)
        if set(buckets) != {7, 9, 11}:
            raise EpisodeWorkspaceFitError("train length buckets are incomplete")
        if any(groups_per_batch > len(values) for values in buckets.values()):
            raise EpisodeWorkspaceFitError("groups per batch exceeds a length bucket")
        self._buckets = dict(buckets)
        self._groups_per_batch = groups_per_batch
        self._rng = random.Random(seed)
        self._epoch_batches: list[tuple[FitGroup, ...]] = []
        self._cursor = 0
        self.epochs_built = 0

    def _build_epoch(self) -> None:
        batches: list[tuple[FitGroup, ...]] = []
        for length in sorted(self._buckets):
            values = self._buckets[length].copy()
            self._rng.shuffle(values)
            for start in range(0, len(values), self._groups_per_batch):
                batches.append(tuple(values[start : start + self._groups_per_batch]))
        self._rng.shuffle(batches)
        self._epoch_batches = batches
        self._cursor = 0
        self.epochs_built += 1

    def next(self) -> tuple[FitGroup, ...]:
        if self._cursor >= len(self._epoch_batches):
            self._build_epoch()
        selected = self._epoch_batches[self._cursor]
        self._cursor += 1
        if len({group.query_length for group in selected}) != 1:
            raise EpisodeWorkspaceFitError("scheduler mixed active query lengths")
        return selected


def make_fit_batch(
    groups: tuple[FitGroup, ...],
    *,
    device: torch.device,
) -> FitBatch:
    examples = tuple(example for group in groups for example in group.examples)
    if not examples:
        raise EpisodeWorkspaceFitError("cannot construct an empty fit batch")
    query_lengths = {len(example.query_tokens) for example in examples}
    if len(query_lengths) != 1:
        raise EpisodeWorkspaceFitError("fit batch mixes active query lengths")
    query_length = query_lengths.pop()
    batch_size = len(examples)
    world_idx = torch.empty(
        batch_size,
        WORLD_TOKENS,
        dtype=torch.long,
        device=device,
    )
    query_idx = torch.empty(
        batch_size,
        query_length,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (batch_size, query_length),
        -1,
        dtype=torch.long,
        device=device,
    )
    answer_index = query_length - 2
    for row, example in enumerate(examples):
        world_idx[row] = torch.tensor(
            example.world_tokens,
            dtype=torch.long,
            device=device,
        )
        query_idx[row] = torch.tensor(
            example.query_tokens,
            dtype=torch.long,
            device=device,
        )
        targets[row, answer_index] = example.target_token
    if int((targets != -1).sum().item()) != batch_size:
        raise EpisodeWorkspaceFitError(
            "fit batch must supervise exactly one position per packet"
        )
    if not torch.all(query_idx[:, answer_index] == ANSWER):
        raise EpisodeWorkspaceFitError("fit target is not aligned to ANSWER")
    return FitBatch(
        world_idx=world_idx,
        query_idx=query_idx,
        targets=targets,
        answer_index=answer_index,
        packet_digests=tuple(example.packet_sha256 for example in examples),
    )


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EpisodeWorkspaceFitError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EpisodeWorkspaceFitError("MPS requested but unavailable")
    return device


def _source_receipt(expected_sha256: str) -> dict[str, object]:
    source = Path(__file__).resolve()
    try:
        receipt = committed_source_receipt(
            source,
            expected_sha256,
            (
                Path(custody_module.__file__),
                Path(episode_board.__file__),
                WORKSPACE_SOURCE_PATH,
                MODEL_SOURCE_PATH,
                CHECKPOINT_SOURCE_PATH,
            ),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceFitError(str(exc)) from exc
    return {**receipt, "runtime_source_manifest": runtime_source_manifest()}


def evaluate_train_fit(
    model: CausalWorkspaceGPT,
    groups: tuple[FitGroup, ...],
    *,
    device: torch.device,
    groups_per_batch: int,
) -> dict[str, object]:
    model.base.eval()
    model.workspace.eval()
    correct = 0
    total = 0
    weighted_loss = 0.0
    by_length: dict[int, list[FitGroup]] = defaultdict(list)
    for group in groups:
        by_length[group.query_length].append(group)
    with torch.no_grad():
        for length in sorted(by_length):
            values = by_length[length]
            for start in range(0, len(values), groups_per_batch):
                batch = make_fit_batch(
                    tuple(values[start : start + groups_per_batch]),
                    device=device,
                )
                with _autocast(device):
                    logits, loss = model.forward_staged(
                        batch.world_idx,
                        batch.query_idx,
                        targets=batch.targets,
                    )
                if loss is None or not torch.isfinite(loss):
                    raise EpisodeWorkspaceFitError("train evaluation loss is invalid")
                row = torch.arange(len(batch.packet_digests), device=device)
                predicted = logits[row, batch.answer_index].argmax(dim=-1)
                target = batch.targets[row, batch.answer_index]
                correct += int((predicted == target).sum().item())
                count = len(batch.packet_digests)
                total += count
                weighted_loss += float(loss.detach().cpu()) * count
    return {
        "packets": {
            "correct": correct,
            "total": total,
            "rate": correct / total,
        },
        "answer_position_nll": weighted_loss / total,
    }


def fit_workspace(
    model: CausalWorkspaceGPT,
    groups: tuple[FitGroup, ...],
    config: FitConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    freeze_protected_base(model)
    model.base.eval()
    model.workspace.train()
    parameters = trainable_workspace_parameters(model)
    optimizer_kwargs: dict[str, object] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
    scheduler = LengthBucketScheduler(
        groups,
        groups_per_batch=config.groups_per_batch,
        seed=config.seed,
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    sampled: set[str] = set()
    appearances = 0
    started = time.monotonic()
    for update in range(1, config.updates + 1):
        batch = make_fit_batch(scheduler.next(), device=device)
        sampled.update(batch.packet_digests)
        appearances += len(batch.packet_digests)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            _, loss, _, _ = model.forward_mechanism_fit(
                batch.world_idx,
                batch.query_idx,
                targets=batch.targets,
            )
        if not torch.isfinite(loss):
            raise EpisodeWorkspaceFitError(f"nonfinite loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            config.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise EpisodeWorkspaceFitError(
                f"nonfinite gradient norm at update {update}"
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        if update == 1 or update % config.log_interval == 0:
            print(
                canonical_json(
                    {
                        "event": "workspace_fit",
                        "update": update,
                        "updates": config.updates,
                        "loss": losses[-1],
                        "gradient_norm": gradient_norms[-1],
                        "unique_packets_seen": len(sampled),
                        "read_gate": float(model.workspace.read_gate.detach().cpu()),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                ),
                flush=True,
            )
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise EpisodeWorkspaceFitError("protected base received a gradient")
    return {
        "updates": config.updates,
        "sampled_packet_appearances": appearances,
        "unique_packets_seen": len(sampled),
        "unique_packet_fraction": len(sampled) / 1_536,
        "scheduler_epochs_built": scheduler.epochs_built,
        "initial_update_loss": losses[0],
        "final_update_loss": losses[-1],
        "minimum_update_loss": min(losses),
        "mean_last_25_update_loss": (sum(losses[-25:]) / min(25, len(losses))),
        "maximum_gradient_norm_before_clip": max(gradient_norms),
        "read_gate": float(model.workspace.read_gate.detach().cpu()),
        "elapsed_seconds": time.monotonic() - started,
        "optimizer_state_serialized": False,
        "protected_base_gradient_tensors": 0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    landlock_receipt = verify_landlock_stage("fit", args.deny_probe)
    expected_arm_input = validate_frozen_arm_input(
        args.arm,
        args.train_groups,
        args.expected_train_sha256,
    )
    config = FitConfig(
        updates=args.updates,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    source_before = _source_receipt(args.expected_source_sha256)
    groups = load_train_groups(args.train_groups, args.expected_train_sha256)
    device = _resolve_device(args.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
    model, protected_receipt = load_protected_workspace_model(
        args.checkpoint,
        CausalWorkspaceConfig(),
    )
    if protected_receipt.checkpoint_sha256 != PROTECTED_CHECKPOINT_SHA256:
        raise EpisodeWorkspaceFitError("protected checkpoint receipt drifted")
    if protected_receipt.base_state_sha256 != PROTECTED_BASE_STATE_SHA256:
        raise EpisodeWorkspaceFitError("protected base-state receipt drifted")
    model.to(device)
    base_before = state_dict_sha256(model.base.state_dict())
    workspace_initial = state_dict_sha256(model.workspace.state_dict())
    train_before = evaluate_train_fit(
        model,
        groups,
        device=device,
        groups_per_batch=config.groups_per_batch,
    )
    training = fit_workspace(model, groups, config, device=device)
    train_after = evaluate_train_fit(
        model,
        groups,
        device=device,
        groups_per_batch=config.groups_per_batch,
    )
    base_after = state_dict_sha256(model.base.state_dict())
    if base_after != base_before or base_after != PROTECTED_BASE_STATE_SHA256:
        raise EpisodeWorkspaceFitError("protected base changed during fitting")
    source_after = _source_receipt(args.expected_source_sha256)
    if source_after != source_before:
        raise EpisodeWorkspaceFitError("source receipt changed during fitting")

    report = {
        "schema": FIT_REPORT_SCHEMA,
        "claim_scope": (
            "train-only synthetic mechanism fit; no development access, "
            "reasoning, language, or continuation-pretraining claim"
        ),
        "arm": args.arm,
        "source": source_before,
        "process_id": os.getpid(),
        "landlock_receipt": landlock_receipt,
        "optimizer_visible_input": {
            "path": str(args.train_groups.absolute()),
            "sha256": args.expected_train_sha256,
            "frozen_arm_binding": dict(expected_arm_input),
            "rows": len(groups),
            "packets": 1_536,
        },
        "forbidden_inputs_opened": [],
        "protected_checkpoint": asdict(protected_receipt),
        "protected_base_sha256_before": base_before,
        "protected_base_sha256_after": base_after,
        "workspace_initial_state_sha256": workspace_initial,
        "workspace_config": asdict(model.workspace_config),
        "fit_config": asdict(config),
        "train_before": train_before,
        "training": training,
        "train_after": train_after,
        "optimizer_state_serialized": False,
        "pretraining_started": False,
        "continuation_pretraining_authorized": False,
    }
    staging, lock = atomic_bundle_directory(args.output)
    try:
        delta_path = staging / "workspace_delta.pt"
        delta_sha256 = save_workspace_delta(
            delta_path,
            model,
            protected_receipt,
        )
        report = {
            **report,
            "workspace_delta_sha256": delta_sha256,
        }
        report_path = staging / "fit_report.json"
        write_json_fsync(report_path, report)
        manifest = {
            "schema": FIT_BUNDLE_SCHEMA,
            "files": {
                "workspace_delta.pt": delta_sha256,
                "fit_report.json": file_sha256(report_path),
            },
            "arm": args.arm,
            "optimizer_state_serialized": False,
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
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
        "--train-groups",
        type=Path,
        default=DEFAULT_CUSTODY_BUNDLE / "train_true_groups.jsonl",
    )
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    parser.add_argument("--arm", choices=("true", "shuffled"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026072347)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "arm": report["arm"],
                "train_before": report["train_before"],
                "train_after": report["train_after"],
                "pretraining_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
