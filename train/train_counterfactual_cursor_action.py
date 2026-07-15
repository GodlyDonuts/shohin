#!/usr/bin/env python3
"""Train all six frozen R12 cursor-action canary arms in one isolated run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from counterfactual_cursor_action_data import (
    IMPLEMENTATION_PATHS,
    CanarySplit,
    load_canary,
)
from counterfactual_cursor_action_objectives import (
    FROZEN_LABEL_TOKEN_IDS,
    relation_losses,
)
from counterfactual_cursor_action_training import (
    ARMS,
    adapter_state_payload,
    build_adapter,
    encode_before_final_block,
    freeze_base,
    logits_from_final_block_cache,
)
from model import GPT, GPTConfig


SCHEMA = "counterfactual_cursor_action_adapter_v1"
MANIFEST_SCHEMA = "counterfactual_cursor_action_training_manifest_v1"
BASE_STEP = 260000
BASE_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
SEED = 2026071506
EPOCHS = 4
LEARNING_RATE = 0.01
MIN_LR_RATIO = 0.1
WARMUP_UPDATES = 50
BETAS = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
CURSOR_MARGIN = 1.0
RELATION_COEFFICIENTS = {
    "orbit_interchange": 1.0,
    "ordinary_loss": 0.0,
    "relation_sham": 1.0,
    "source_only": 0.0,
    "cursor_table": 0.0,
    "text_cursor_lora": 0.0,
}
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FrozenInputCache:
    hidden: torch.Tensor
    prompt_last: torch.Tensor
    max_tokens: int
    source_token_count: int
    source_token_bytes: int
    padded_token_positions: int
    cache_bytes: int


@dataclass(frozen=True)
class CompiledTrainingUnit:
    unit_id: str
    swap_index: int
    renderer_count: int
    source_indices: tuple[int, ...]
    cell_indices: tuple[int, ...]
    cursors: tuple[int, ...]
    target_indices: tuple[int, ...]
    target_token_ids: tuple[int, ...]

    @property
    def examples(self) -> int:
        return len(self.cell_indices)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_state_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_implementation_identity(
    commit: str, file_sha256: Mapping[str, str],
) -> None:
    hashes = dict(file_sha256)
    if set(hashes) != set(IMPLEMENTATION_PATHS):
        raise ValueError("implementation hash ledger has the wrong file set")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("implementation commit is malformed")
    for relative in IMPLEMENTATION_PATHS:
        live_path = ROOT / relative
        reject_symlink_components(live_path, f"implementation file {relative}")
        if sha256_file(live_path) != hashes[relative]:
            raise ValueError(f"live implementation differs from canary: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != hashes[relative]:
            raise ValueError(f"committed implementation differs from canary: {relative}")


def reject_symlink_components(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for alias, target in {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
        Path("/etc"): Path("/private/etc"),
    }.items():
        if (
            alias.is_symlink()
            and alias.resolve() == target
            and (absolute == alias or alias in absolute.parents)
        ):
            absolute = target / absolute.relative_to(alias)
            break
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError(f"{label} contains a symlink path component: {current}")
    return absolute


def require_regular_file(path: Path, label: str) -> None:
    absolute = reject_symlink_components(path, label)
    info = os.lstat(absolute)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file")
    if info.st_mode & 0o222:
        raise ValueError(f"{label} must be read-only")


def compile_training_units(split: CanarySplit) -> tuple[CompiledTrainingUnit, ...]:
    if split.name != "train" or split.counts.training_units != 288:
        raise ValueError("cursor-action trainer requires the exact frozen training split")
    sham = split.relations.relation_sham
    if (
        sham.cursor_target_rotation,
        sham.adjacent_cursor_rotation,
        sham.renderer_cursor_rotation,
    ) != (1, 1, 1):
        raise ValueError("relation-sham mapping differs from the frozen local rotation")
    compiled = []
    cell_multiplicity = [0] * len(split.cells)
    for unit in split.training_units:
        if len(unit.adjacent_pairs) != 6:
            raise ValueError("training unit must contain six canonical renderers")
        if tuple(pair.renderer_id for pair in unit.adjacent_pairs) != tuple(range(6)):
            raise ValueError("training-unit renderers are not in canonical order")
        source_indices = []
        cell_indices = []
        cursors = []
        target_indices = []
        target_token_ids = []
        for side in ("left", "right"):
            for pair in unit.adjacent_pairs:
                if pair.swap_index != unit.swap_index:
                    raise ValueError("training-unit swap identity mismatch")
                left_source = split.source_by_id[pair.left_source_id]
                right_source = split.source_by_id[pair.right_source_id]
                if (
                    left_source.permutation_id != unit.left_permutation_id
                    or right_source.permutation_id != unit.right_permutation_id
                ):
                    raise ValueError("training-unit permutation identity mismatch")
                expected_right = list(left_source.operation_order)
                expected_right[unit.swap_index], expected_right[unit.swap_index + 1] = (
                    expected_right[unit.swap_index + 1], expected_right[unit.swap_index]
                )
                if tuple(expected_right) != right_source.operation_order:
                    raise ValueError("training-unit sources are not the declared adjacent swap")
                source_id = (
                    pair.left_source_id if side == "left" else pair.right_source_id
                )
                source_index = split.source_index_by_id[source_id]
                for cursor in range(5):
                    cell_index = split.cell_index_by_key[(source_id, cursor)]
                    cell = split.cells[cell_index]
                    cell_multiplicity[cell_index] += 1
                    source_indices.append(source_index)
                    cell_indices.append(cell_index)
                    cursors.append(cursor)
                    target_indices.append(cell.target_index)
                    target_token_ids.append(cell.target_token_id)
        result = CompiledTrainingUnit(
            unit_id=unit.unit_id,
            swap_index=unit.swap_index,
            renderer_count=len(unit.adjacent_pairs),
            source_indices=tuple(source_indices),
            cell_indices=tuple(cell_indices),
            cursors=tuple(cursors),
            target_indices=tuple(target_indices),
            target_token_ids=tuple(target_token_ids),
        )
        if result.examples != 60:
            raise ValueError("training unit must contain exactly 60 examples")
        expected_labels = set(range(5))
        for offset in range(0, result.examples, 5):
            if set(result.target_indices[offset:offset + 5]) != expected_labels:
                raise ValueError("a source does not expose all five selector labels")
        compiled.append(result)
    if len(compiled) != 288 or len({unit.unit_id for unit in compiled}) != 288:
        raise ValueError("training-unit identity mismatch")
    if set(cell_multiplicity) != {3}:
        raise ValueError("training-unit graph does not weight every cell exactly three times")
    return tuple(compiled)


def epoch_unit_orders(units: int) -> tuple[tuple[int, ...], ...]:
    result = []
    for epoch in range(EPOCHS):
        order = list(range(units))
        random.Random(SEED + epoch).shuffle(order)
        result.append(tuple(order))
    return tuple(result)


def lr_scale(update: int, total_updates: int) -> float:
    if update < WARMUP_UPDATES:
        return (update + 1) / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES) / max(1, total_updates - WARMUP_UPDATES - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def build_input_cache(
    model: GPT, token_sequences: Iterable[tuple[int, ...]], *, cache_batch_size: int,
) -> FrozenInputCache:
    sequences = tuple(token_sequences)
    if not sequences or cache_batch_size <= 0:
        raise ValueError("cache input and batch size must be nonempty")
    if any(not sequence for sequence in sequences):
        raise ValueError("cache input contains an empty sequence")
    max_tokens = max(len(sequence) for sequence in sequences)
    if max_tokens > model.cfg.seq_len:
        raise ValueError("cache input exceeds model context")
    hidden_cache = None
    positions = torch.tensor(
        [len(sequence) - 1 for sequence in sequences], dtype=torch.long, device="cuda",
    )
    for start in range(0, len(sequences), cache_batch_size):
        batch = sequences[start:start + cache_batch_size]
        input_ids = torch.zeros((len(batch), max_tokens), dtype=torch.long, device="cuda")
        for local, sequence in enumerate(batch):
            input_ids[local, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device="cuda",
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encode_before_final_block(model, input_ids)
        if hidden_cache is None:
            hidden_cache = torch.empty(
                (len(sequences), max_tokens, model.cfg.d_model),
                dtype=hidden.dtype, device="cuda",
            )
        hidden_cache[start:start + len(batch)].copy_(hidden)
    if hidden_cache is None:
        raise AssertionError("cache allocation did not occur")
    return FrozenInputCache(
        hidden=hidden_cache,
        prompt_last=positions,
        max_tokens=max_tokens,
        source_token_count=sum(len(sequence) for sequence in sequences),
        source_token_bytes=sum(len(sequence) for sequence in sequences) * 8,
        padded_token_positions=len(sequences) * max_tokens,
        cache_bytes=hidden_cache.numel() * hidden_cache.element_size(),
    )


def select_unit_cache(
    cache: FrozenInputCache, indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long, device="cuda")
    return cache.hidden.index_select(0, index), cache.prompt_last.index_select(0, index)


def write_torch_exclusive_read_only(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing artifact: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary artifact: {temporary}")
    torch.save(payload, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, 0o444)
    os.link(temporary, path)
    os.unlink(temporary)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json_exclusive_read_only(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing artifact: {path}")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def train_arm(
    model: GPT,
    arm: str,
    cache: FrozenInputCache,
    compiled_units: tuple[CompiledTrainingUnit, ...],
    unit_orders: tuple[tuple[int, ...], ...],
    label_token_ids: torch.Tensor,
    bindings: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter, spec = build_adapter(arm, model.cfg, SEED)
    adapter = adapter.to("cuda")
    initial_adapter_sha256 = hash_state_dict(adapter.state_dict())
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    total_updates = len(compiled_units) * EPOCHS
    relation_coefficient = RELATION_COEFFICIENTS[arm]
    epoch_history = []
    started = time.time()
    update = 0
    relation_counts = None
    adapter.train()
    for epoch, order in enumerate(unit_orders):
        sums = {name: 0.0 for name in ("loss", "ce", "cursor", "adjacent", "renderer", "gnorm")}
        epoch_started = time.time()
        for unit_index in order:
            unit = compiled_units[unit_index]
            cache_indices = unit.cell_indices if arm == "text_cursor_lora" else unit.source_indices
            prefix_hidden, prompt_last = select_unit_cache(cache, cache_indices)
            cursor = None if arm == "text_cursor_lora" else torch.tensor(
                unit.cursors, dtype=torch.long, device="cuda",
            )
            target_indices = torch.tensor(
                unit.target_indices, dtype=torch.long, device="cuda",
            )
            target_token_ids = torch.tensor(
                unit.target_token_ids, dtype=torch.long, device="cuda",
            )
            optimizer.param_groups[0]["lr"] = LEARNING_RATE * lr_scale(update, total_updates)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full_logits, action_logits = logits_from_final_block_cache(
                    model, prefix_hidden, prompt_last, adapter, arm, cursor,
                    label_token_ids,
                )
                shaped_logits = action_logits.reshape(2, unit.renderer_count, 5, 5)
                shaped_labels = target_indices.reshape(2, unit.renderer_count, 5)
                relations = relation_losses(
                    shaped_logits, shaped_labels, swap_index=unit.swap_index,
                    sham=arm == "relation_sham", cursor_margin=CURSOR_MARGIN,
                )
                ce = F.cross_entropy(full_logits.float(), target_token_ids)
                loss = ce + relation_coefficient * relations.total()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {arm} loss at update {update}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                tuple(adapter.parameters()), GRADIENT_CLIP,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(f"non-finite {arm} gradient at update {update}")
            optimizer.step()
            counts = (
                relations.cursor_pairs, relations.adjacent_pairs, relations.renderer_pairs,
            )
            if relation_counts is None:
                relation_counts = counts
            elif relation_counts != counts:
                raise RuntimeError("relation pair counts changed across updates")
            values = {
                "loss": loss, "ce": ce,
                "cursor": relations.cursor_interchange,
                "adjacent": relations.adjacent_equivariance,
                "renderer": relations.renderer_invariance,
                "gnorm": gradient_norm,
            }
            for name, value in values.items():
                sums[name] += float(value.detach())
            if update % 50 == 0 or update + 1 == total_updates:
                print(
                    "[ccaa-train] arm={} update={}/{} epoch={} loss={:.5f} ce={:.5f} "
                    "cursor={:.5f} adjacent={:.5f} renderer={:.5f} gnorm={:.4f} lr={:.6g}".format(
                        arm, update + 1, total_updates, epoch + 1,
                        float(loss.detach()), float(ce.detach()),
                        float(relations.cursor_interchange.detach()),
                        float(relations.adjacent_equivariance.detach()),
                        float(relations.renderer_invariance.detach()),
                        float(gradient_norm), optimizer.param_groups[0]["lr"],
                    ), flush=True,
                )
            update += 1
        epoch_history.append({
            "epoch": epoch + 1,
            "updates": len(order),
            "seconds": time.time() - epoch_started,
            **{f"mean_{name}": value / len(order) for name, value in sums.items()},
        })
    if update != total_updates or relation_counts is None:
        raise RuntimeError("training update accounting mismatch")
    torch.cuda.synchronize()
    final_adapter_sha256 = hash_state_dict(adapter.state_dict())
    if final_adapter_sha256 == initial_adapter_sha256:
        raise RuntimeError(f"{arm} adapter did not change")
    adapter_payload = adapter_state_payload(adapter, spec)
    examples_per_update = compiled_units[0].examples
    final_block_token_positions = total_updates * examples_per_update * cache.max_tokens
    relation_pairs_per_update = sum(relation_counts)
    adapter_projection_positions = (
        final_block_token_positions if arm == "text_cursor_lora"
        else total_updates * examples_per_update
    )
    resource_ledger = {
        "trainable_scalars": spec.parameters,
        "active_trainable_scalars": (
            5 * (model.cfg.d_model // model.cfg.n_head)
            if arm == "cursor_table" else spec.parameters
        ),
        "inactive_trainable_scalars": (
            3 * (model.cfg.d_model // model.cfg.n_head)
            if arm == "cursor_table" else 0
        ),
        "base_trainable_scalars": 0,
        "retained_cursor_bits_selector": spec.retained_cursor_bits,
        "retained_phase_bits_selector": 0,
        "retained_bits_future_one_call": spec.retained_cursor_bits + (
            1 if spec.retained_cursor_bits else 0
        ),
        "adapter_dtype": "float32",
        "base_autocast_dtype": "bfloat16",
        "source_token_count": cache.source_token_count,
        "source_token_storage_bytes_int64": cache.source_token_bytes,
        "padded_cache_token_positions": cache.padded_token_positions,
        "pre_final_hidden_cache_bytes": cache.cache_bytes,
        "unique_training_cells": 5760,
        "training_examples_with_repetition": total_updates * examples_per_update,
        "oracle_calls": 0,
        "fixed_training_compute_proxy": {
            "final_block_token_positions": final_block_token_positions,
            "full_vocab_last_position_projections": total_updates * examples_per_update,
            "relation_pair_evaluations": total_updates * relation_pairs_per_update,
            "adapter_projection_positions": adapter_projection_positions,
        },
        "inference_compute_proxy_per_cell": {
            "full_model_token_positions": "prompt_length",
            "full_vocab_last_position_projections": 1,
            "adapter_projection_positions": (
                "prompt_length" if arm == "text_cursor_lora" else 1
            ),
        },
        "sequential_token_depth": cache.max_tokens,
        "external_memory": "ordinary prompt/KV plus declared cursor bits only",
        "external_execution": 0,
    }
    training = {
        "seed": SEED,
        "epochs": EPOCHS,
        "updates": total_updates,
        "units_per_epoch": len(compiled_units),
        "examples_per_update": examples_per_update,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "minimum_lr_ratio": MIN_LR_RATIO,
        "warmup_updates": WARMUP_UPDATES,
        "betas": list(BETAS),
        "epsilon": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "cursor_margin": CURSOR_MARGIN,
        "action_ce_weight": 1.0,
        "relation_coefficient": relation_coefficient,
        "relation_mapping": "deranged" if arm == "relation_sham" else "true",
        "relation_pairs_per_update": {
            "cursor_interchange": relation_counts[0],
            "adjacent_equivariance": relation_counts[1],
            "renderer_invariance": relation_counts[2],
        },
        "epoch_history": epoch_history,
        "elapsed_seconds": time.time() - started,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {
        "schema": SCHEMA,
        "arm": arm,
        **adapter_payload,
        "bindings": bindings,
        "training": training,
        "resource_ledger": resource_ledger,
    }
    arm_directory = output_root / arm
    arm_directory.mkdir(mode=0o700)
    artifact = arm_directory / "adapter.pt"
    write_torch_exclusive_read_only(artifact, payload)
    artifact_sha256 = sha256_file(artifact)
    os.chmod(arm_directory, 0o555)
    print(
        f"[ccaa-train] saved arm={arm} sha256={artifact_sha256} "
        f"final_adapter_sha256={final_adapter_sha256}", flush=True,
    )
    del adapter, optimizer
    torch.cuda.empty_cache()
    return {
        "arm": arm,
        "artifact": str(artifact.relative_to(output_root)),
        "artifact_sha256": artifact_sha256,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "trainable_scalars": spec.parameters,
        "updates": total_updates,
        "relation_coefficient": relation_coefficient,
        "fixed_training_compute_proxy": resource_ledger["fixed_training_compute_proxy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cursor-action training requires CUDA")
    output_parent = reject_symlink_components(args.out.parent, "output parent")
    if not output_parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {output_parent}")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing existing output root: {args.out}")
    for path, label in (
        (args.base, "base checkpoint"), (args.canary, "canary"),
        (args.audit, "canary audit"), (args.tokenizer, "tokenizer"),
    ):
        require_regular_file(path, label)
    base_sha256 = sha256_file(args.base)
    if base_sha256 != BASE_SHA256:
        raise SystemExit("base checkpoint SHA-256 mismatch")
    dataset = load_canary(
        args.canary, args.audit, args.tokenizer,
        requested_model_fields={
            "sidecar": ("prompt_token_ids", "cursor"),
            "text_control": ("text_prompt_token_ids",),
        },
    )
    if dataset.implementation_commit is None:
        raise SystemExit("persistent canary lacks a committed implementation identity")
    verify_implementation_identity(
        dataset.implementation_commit,
        dataset.implementation_file_sha256,
    )
    checkpoint = torch.load(
        args.base, map_location="cpu", weights_only=False, mmap=True,
    )
    if checkpoint.get("step") != BASE_STEP:
        raise SystemExit("base checkpoint step mismatch")
    cfg = GPTConfig(**checkpoint["cfg"])
    if cfg.n_loop != 1 or cfg.d_model != 576 or cfg.n_head != 9:
        raise SystemExit("base architecture is outside cursor-action v1")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    freeze_base(model)
    split = dataset.split("train")
    compiled_units = compile_training_units(split)
    unit_orders = epoch_unit_orders(len(compiled_units))
    all_sidecar_examples = split.sidecar_examples()
    sidecar_source_examples = tuple(
        all_sidecar_examples[source_index * 5] for source_index in range(len(split.sources))
    )
    for source_index, example in enumerate(sidecar_source_examples):
        for cursor in range(5):
            candidate = all_sidecar_examples[source_index * 5 + cursor]
            if candidate.model_input.prompt_token_ids != example.model_input.prompt_token_ids:
                raise SystemExit("sidecar source prompt differs across cursor interventions")
    text_examples = split.text_examples()
    label_token_ids = torch.tensor(
        FROZEN_LABEL_TOKEN_IDS, dtype=torch.long, device="cuda",
    )
    bindings = {
        "base_sha256": base_sha256,
        "base_step": BASE_STEP,
        "canary_sha256": dataset.canary_file_sha256,
        "canary_payload_sha256": dataset.payload_sha256,
        "audit_sha256": dataset.audit_file_sha256,
        "tokenizer_sha256": dataset.tokenizer_sha256,
        "implementation_commit": dataset.implementation_commit,
    }
    args.out.mkdir(parents=True, mode=0o700)
    results = []
    sidecar_cache = build_input_cache(
        model,
        (example.model_input.prompt_token_ids for example in sidecar_source_examples),
        cache_batch_size=args.cache_batch_size,
    )
    print(
        f"[ccaa-train] sidecar cache sources={len(sidecar_source_examples)} "
        f"tokens={sidecar_cache.max_tokens} bytes={sidecar_cache.cache_bytes}", flush=True,
    )
    for arm in ARMS[:-1]:
        results.append(train_arm(
            model, arm, sidecar_cache, compiled_units, unit_orders,
            label_token_ids, bindings, args.out,
        ))
    del sidecar_cache
    torch.cuda.empty_cache()
    text_cache = build_input_cache(
        model,
        (example.model_input.text_prompt_token_ids for example in text_examples),
        cache_batch_size=args.cache_batch_size,
    )
    print(
        f"[ccaa-train] text cache cells={len(text_examples)} "
        f"tokens={text_cache.max_tokens} bytes={text_cache.cache_bytes}", flush=True,
    )
    results.append(train_arm(
        model, ARMS[-1], text_cache, compiled_units, unit_orders,
        label_token_ids, bindings, args.out,
    ))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "arms": results,
        "arm_order": list(ARMS),
        "bindings": bindings,
        "all_arms_complete": len(results) == len(ARMS),
        "score_bearing_evaluation_performed": False,
    }
    write_json_exclusive_read_only(args.out / "training_manifest.json", manifest)
    os.chmod(args.out, 0o555)
    print(
        f"[ccaa-train] complete arms={len(results)} manifest_sha256="
        f"{sha256_file(args.out / 'training_manifest.json')}", flush=True,
    )


if __name__ == "__main__":
    main()
