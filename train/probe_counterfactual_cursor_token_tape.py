#!/usr/bin/env python3
"""Probe distributed prompt-token state with an externally supplied cursor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from counterfactual_cursor_action_dev_view import LABEL_TOKEN_IDS, load_development_view
from model import GPT, GPTConfig
from probe_counterfactual_cursor_readout import (
    BASE_SHA256,
    BASE_STEP,
    BATCH_SIZE,
    BETAS,
    DEV_VIEW_AUDIT_SHA256,
    DEV_VIEW_SHA256,
    EPOCHS,
    EPSILON,
    GRADIENT_CLIP,
    LEARNING_RATE,
    SplitFeatures,
    _base_evidence,
    _distribution,
    _metric_counts,
    _state_payload,
    build_input_cache,
    freeze_base,
    reject_symlink_components,
    require_read_only,
    sha256_file,
    write_exclusive_read_only,
)


SCHEMA = "counterfactual_cursor_token_tape_dev_v1"
SEED = 2026071602
ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "R12_CURSOR_TOKEN_TAPE_DIAGNOSTIC.md",
    "train/model.py",
    "train/counterfactual_cursor_action_dev_view.py",
    "train/probe_counterfactual_cursor_readout.py",
    "train/probe_counterfactual_cursor_token_tape.py",
    "train/test_probe_counterfactual_cursor_token_tape.py",
    "train/jobs/probe_counterfactual_cursor_token_tape.sbatch",
)


@dataclass(frozen=True)
class TapeSplit:
    features: SplitFeatures
    embedding_source: torch.Tensor
    pre_source: torch.Tensor
    valid_mask: torch.Tensor


class TokenTapeReadout(nn.Module):
    """Cursor-conditioned attention with shared or cursor-specific values."""

    def __init__(self, features: int, *, cursor_query: bool, cursor_value: bool):
        super().__init__()
        self.features = features
        self.cursor_query = cursor_query
        self.cursor_value = cursor_value
        query_count = 5 if cursor_query else 1
        self.query = nn.Embedding(query_count, features)
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        if cursor_value:
            self.weight = nn.Parameter(torch.zeros(5, 5, features))
            self.bias = nn.Parameter(torch.zeros(5, 5))
        else:
            self.value = nn.Linear(features, 5)
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)

    def forward_with_attention(
        self, tape: torch.Tensor, valid_mask: torch.Tensor, cursor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tape.ndim != 3 or tape.shape[-1] != self.features:
            raise ValueError("token tape has the wrong shape")
        if valid_mask.shape != tape.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("token tape mask has the wrong shape or dtype")
        if cursor.shape != (len(tape),) or cursor.dtype != torch.long:
            raise ValueError("cursor must be an int64 batch vector")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every token tape row must contain a valid token")
        query_index = cursor if self.cursor_query else torch.zeros_like(cursor)
        query = self.query(query_index)
        attention_logits = torch.einsum("btd,bd->bt", tape, query) / math.sqrt(self.features)
        attention_logits = attention_logits.masked_fill(~valid_mask, -torch.inf)
        attention = F.softmax(attention_logits, dim=-1)
        pooled = torch.einsum("bt,btd->bd", attention, tape)
        if self.cursor_value:
            scores = torch.einsum("bad,bd->ba", self.weight[cursor], pooled) + self.bias[cursor]
        else:
            scores = self.value(pooled)
        return scores, attention

    def forward(
        self, tape: torch.Tensor, valid_mask: torch.Tensor, cursor: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_attention(tape, valid_mask, cursor)[0]


class MeanJointTapeReadout(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5, 5, features))
        self.bias = nn.Parameter(torch.zeros(5, 5))

    def forward(
        self, tape: torch.Tensor, valid_mask: torch.Tensor, cursor: torch.Tensor,
    ) -> torch.Tensor:
        count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (tape * valid_mask[:, :, None]).sum(dim=1) / count
        return torch.einsum("bad,bd->ba", self.weight[cursor], pooled) + self.bias[cursor]


class CursorOnlyTapeReadout(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        del features
        self.table = nn.Embedding(5, 5)
        nn.init.zeros_(self.table.weight)

    def forward(
        self, tape: torch.Tensor, valid_mask: torch.Tensor, cursor: torch.Tensor,
    ) -> torch.Tensor:
        del tape, valid_mask
        return self.table(cursor)


def verify_implementation(commit: str) -> dict[str, str]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("implementation commit is malformed")
    marker = reject_symlink_components(ROOT / ".implementation_commit")
    if not marker.is_file() or marker.stat().st_mode & 0o222:
        raise ValueError("immutable implementation marker is missing")
    if marker.read_text(encoding="ascii").strip() != commit:
        raise ValueError("implementation marker differs from requested commit")
    hashes = {}
    for relative in IMPLEMENTATION_FILES:
        path = reject_symlink_components(ROOT / relative)
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError(f"exported implementation is not immutable: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


@torch.no_grad()
def extract_tape_split(
    model: GPT, split, labels: torch.Tensor, cache_batch_size: int,
) -> TapeSplit:
    sequences = tuple(source.prompt_token_ids for source in split.sources)
    cache = build_input_cache(model, sequences, cache_batch_size=cache_batch_size)
    token_grid = torch.arange(cache.max_tokens, device="cuda")
    valid_mask = token_grid[None, :].le(cache.prompt_last[:, None])
    input_ids = torch.zeros(
        (len(sequences), cache.max_tokens), dtype=torch.long, device="cuda",
    )
    for index, sequence in enumerate(sequences):
        input_ids[index, :len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device="cuda",
        )
    embedding_source = model.tok(input_ids).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cos = model.cos[:cache.max_tokens].to("cuda")
        sin = model.sin[:cache.max_tokens].to("cuda")
        final, _ = model.blocks[-1](cache.hidden, cos, sin)
        post_source = model.norm(final).float()
        row = torch.arange(len(split.sources), device="cuda")
        last = post_source[row, cache.prompt_last]
        full_logits = model.head(last.to(model.head.weight.dtype)).float()
    action_source, non_lse_source, non_max_source = _base_evidence(full_logits, labels)
    source_index = torch.tensor(
        [split.source_index_by_id[cell.source_id] for cell in split.cells],
        dtype=torch.long,
        device="cuda",
    )
    cursor = torch.tensor([cell.cursor for cell in split.cells], dtype=torch.long, device="cuda")
    target = torch.tensor(
        [cell.target_index for cell in split.cells], dtype=torch.long, device="cuda",
    )
    renderer = torch.tensor(
        [split.sources[index].renderer_id for index in source_index.tolist()],
        dtype=torch.long,
        device="cuda",
    )
    permutation = torch.tensor(
        [split.sources[index].permutation_id for index in source_index.tolist()],
        dtype=torch.long,
        device="cuda",
    )
    features = SplitFeatures(
        name=split.name,
        pre=cache.hidden[
            torch.arange(len(split.sources), device="cuda"), cache.prompt_last
        ].float().index_select(0, source_index),
        post=last.float().index_select(0, source_index),
        cursor=cursor,
        target=target,
        source_index=source_index,
        source_ids=tuple(source.source_id for source in split.sources),
        renderer_ids=renderer,
        permutation_ids=permutation,
        base_action_logits=action_source.index_select(0, source_index),
        nonaction_logsumexp=non_lse_source.index_select(0, source_index),
        nonaction_max=non_max_source.index_select(0, source_index),
    )
    return TapeSplit(features, embedding_source, cache.hidden.float(), valid_mask)


def standardize_tapes(
    train: torch.Tensor,
    train_mask: torch.Tensor,
    development: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = train[train_mask]
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (train - mean) / std, (development - mean) / std, mean, std


def rms_normalize_tape(tape: torch.Tensor) -> torch.Tensor:
    return tape / tape.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()


def position_tape(sources: int, tokens: int, features: int, device: str) -> torch.Tensor:
    if features % 2:
        raise ValueError("position tape requires an even feature count")
    position = torch.arange(tokens, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(features // 2, device=device, dtype=torch.float32)
        / max(features // 2 - 1, 1)
    )[None, :]
    basis = torch.cat((torch.sin(position * frequency), torch.cos(position * frequency)), dim=1)
    return basis[None, :, :].expand(sources, -1, -1).contiguous()


def derangement(size: int, seed: int, device: str) -> torch.Tensor:
    if size < 2:
        raise ValueError("derangement requires at least two sources")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.arange(size)
    for _ in range(100):
        candidate = torch.randperm(size, generator=generator)
        if bool(candidate.ne(identity).all()):
            return candidate.to(device)
    raise RuntimeError("failed to construct deterministic derangement")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _batch_inputs(
    tape: torch.Tensor, mask: torch.Tensor, split: SplitFeatures, index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = split.source_index.index_select(0, index)
    return (
        tape.index_select(0, source),
        mask.index_select(0, source),
        split.cursor.index_select(0, index),
    )


def train_tape_readout(
    *,
    name: str,
    seed: int,
    model: nn.Module,
    train_tape: torch.Tensor,
    train_mask: torch.Tensor,
    development_tape: torch.Tensor,
    development_mask: torch.Tensor,
    train: SplitFeatures,
    development: SplitFeatures,
) -> tuple[nn.Module, dict[str, object]]:
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=0.0,
    )
    losses = []
    updates = 0
    for epoch in range(EPOCHS):
        order = list(range(len(train.target)))
        random.Random(seed + epoch).shuffle(order)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = torch.tensor(order[start:start + BATCH_SIZE], device="cuda")
            inputs = _batch_inputs(train_tape, train_mask, train, index)
            scores = model(*inputs)
            loss = F.cross_entropy(scores, train.target.index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                raise RuntimeError(f"non-finite training state for {name}")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            updates += 1
        losses.append(loss_sum / len(order))
    model.eval()
    return model, {
        "name": name,
        "seed": seed,
        "trainable_scalars": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "state": _state_payload(model),
        "train": evaluate_restricted(
            model, train_tape, train_mask, train, include_examples=False,
        ),
        "development": evaluate_restricted(
            model, development_tape, development_mask, development, include_examples=True,
        ),
    }


@torch.no_grad()
def _scores_and_attention(
    model: nn.Module,
    tape: torch.Tensor,
    mask: torch.Tensor,
    split: SplitFeatures,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    source_tape = tape.index_select(0, split.source_index)
    source_mask = mask.index_select(0, split.source_index)
    if isinstance(model, TokenTapeReadout):
        return model.forward_with_attention(source_tape, source_mask, split.cursor)
    return model(source_tape, source_mask, split.cursor), None


@torch.no_grad()
def evaluate_restricted(
    model: nn.Module,
    tape: torch.Tensor,
    mask: torch.Tensor,
    split: SplitFeatures,
    *,
    include_examples: bool,
) -> dict[str, object]:
    scores, attention = _scores_and_attention(model, tape, mask, split)
    top, prediction = scores.max(dim=-1)
    unique = scores.eq(top[:, None]).sum(dim=-1).eq(1)
    correct = unique & prediction.eq(split.target)
    restricted = _metric_counts(correct, split)
    per_cursor = {}
    for cursor in range(5):
        selected = split.cursor.eq(cursor)
        numerator = int(correct[selected].sum())
        denominator = int(selected.sum())
        per_cursor[str(cursor)] = {
            "numerator": numerator,
            "denominator": denominator,
            "proportion": numerator / denominator,
        }
    operation = split.cursor.lt(4)
    operation_numerator = int(correct[operation].sum())
    operation_denominator = int(operation.sum())
    restricted["per_cursor_cell_accuracy"] = per_cursor
    restricted["operation_only_cell_accuracy"] = {
        "numerator": operation_numerator,
        "denominator": operation_denominator,
        "proportion": operation_numerator / operation_denominator,
    }
    result = {"restricted": restricted}
    if attention is not None:
        safe = attention.clamp_min(torch.finfo(attention.dtype).tiny)
        entropy = -(attention * safe.log()).sum(dim=-1)
        peak = attention.max(dim=-1)
        result["attention"] = {
            "entropy": _distribution(entropy),
            "peak_mass": _distribution(peak.values),
            "peak_position": _distribution(peak.indices.float()),
        }
    if isinstance(model, TokenTapeReadout):
        result["query_norm"] = _distribution(model.query.weight.detach().norm(dim=-1))
    if include_examples:
        source_index = split.source_index.detach().cpu().tolist()
        examples = []
        for index in range(len(split.target)):
            item = {
                "source_id": split.source_ids[source_index[index]],
                "renderer_id": int(split.renderer_ids[index]),
                "permutation_id": int(split.permutation_ids[index]),
                "cursor": int(split.cursor[index]),
                "target_index": int(split.target[index]),
                "prediction": int(prediction[index]),
                "unique": bool(unique[index]),
                "correct": bool(correct[index]),
            }
            if attention is not None:
                peak = attention[index].max(dim=-1)
                item["attention_peak_position"] = int(peak.indices)
                item["attention_peak_mass"] = float(peak.values)
            examples.append(item)
        result["examples"] = examples
    return result


def _proportion(arm: Mapping[str, object], metric: str) -> float:
    return float(arm["readout"]["development"]["restricted"][metric]["proportion"])


def _replicate_status(arm: Mapping[str, object]) -> dict[str, bool]:
    train_fit = (
        float(arm["readout"]["train"]["restricted"]["cell_accuracy"]["proportion"])
        >= 0.99
    )
    development = arm["readout"]["development"]["restricted"]
    subgroup_pass = (
        all(
            float(item["proportion"]) >= 0.95
            for item in development["per_renderer_cell_accuracy"].values()
        )
        and all(
            float(development["per_cursor_cell_accuracy"][str(cursor)]["proportion"])
            >= 0.95
            for cursor in range(4)
        )
    )
    development_pass = (
        float(development["cell_accuracy"]["proportion"]) >= 0.95
        and float(development["exact_five_action_groups"]["proportion"]) >= 0.90
        and subgroup_pass
    )
    return {"train_fit": train_fit, "development_pass": development_pass}


def decision(arms: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    source_ok = _proportion(arms["source_only_tape"], "cell_accuracy") <= 0.20
    cursor_ok = _proportion(arms["cursor_only"], "cell_accuracy") <= 0.40
    family_status = {}
    for family in ("pre_final_shared", "pre_final_cursor_specific"):
        statuses = [
            _replicate_status(arms[f"{family}_seed{index}"]) for index in range(3)
        ]
        family_status[family] = {
            "train_fit_replicates": sum(item["train_fit"] for item in statuses),
            "development_pass_replicates": sum(
                item["train_fit"] and item["development_pass"] for item in statuses
            ),
            "replicates": statuses,
        }
    shared_development = [
        _proportion(arms[f"pre_final_shared_seed{index}"], "cell_accuracy")
        for index in range(3)
    ]
    matched_controls = {
        name: _proportion(arms[name], "cell_accuracy")
        for name in ("embedding_shared", "position_shared", "source_deranged_shared")
    }
    shared_deep_margin = statistics.median(shared_development) - max(
        matched_controls.values()
    )
    shared_pass = (
        family_status["pre_final_shared"]["development_pass_replicates"] >= 2
        and shared_deep_margin >= 0.10
        and source_ok
        and cursor_ok
    )
    specific_pass = (
        family_status["pre_final_cursor_specific"]["development_pass_replicates"] >= 2
        and source_ok
        and cursor_ok
    )
    embedding_pass = (
        _replicate_status(arms["embedding_shared"])["train_fit"]
        and _replicate_status(arms["embedding_shared"])["development_pass"]
        and source_ok
        and cursor_ok
    )
    mean_pass = (
        _replicate_status(arms["mean_joint"])["train_fit"]
        and _replicate_status(arms["mean_joint"])["development_pass"]
        and source_ok
        and cursor_ok
    )
    position_status = _replicate_status(arms["position_shared"])
    deranged_status = _replicate_status(arms["source_deranged_shared"])
    shortcut_detected = (
        position_status["train_fit"] and position_status["development_pass"]
    ) or (
        deranged_status["train_fit"] and deranged_status["development_pass"]
    )
    optimization_inconclusive = (
        family_status["pre_final_shared"]["train_fit_replicates"] < 2
        and family_status["pre_final_cursor_specific"]["train_fit_replicates"] < 2
    )
    control_failure = not source_ok or not cursor_ok
    if control_failure:
        result = "structural_control_failure"
    elif shortcut_detected:
        result = "matched_control_shortcut_detected"
    elif shared_pass:
        result = "deep_shared_attention_bottleneck_available"
    elif embedding_pass:
        result = "lexical_embedding_attention_bottleneck_available"
    elif specific_pass:
        result = "cursor_specific_token_tape_upper_bound_available"
    elif mean_pass:
        result = "mean_pooled_token_tape_readout_available"
    elif optimization_inconclusive:
        result = "token_tape_optimization_inconclusive"
    else:
        result = "external_cursor_token_tape_no_go"
    return {
        "source_only_control_respects_ceiling": source_ok,
        "cursor_only_control_respects_ceiling": cursor_ok,
        "structural_control_failure": control_failure,
        "family_status": family_status,
        "matched_control_development_accuracy": matched_controls,
        "shared_median_development_accuracy": statistics.median(shared_development),
        "shared_deep_state_margin_over_best_matched_control": shared_deep_margin,
        "shared_deep_state_pass": shared_pass,
        "embedding_only_pass": embedding_pass,
        "cursor_specific_upper_bound_pass": specific_pass,
        "mean_joint_pass": mean_pass,
        "matched_control_shortcut_detected": shortcut_detected,
        "optimization_inconclusive": optimization_inconclusive,
        "decision": result,
        "reasoning_claim_authorized": False,
        "internal_cursor_claim_authorized": False,
        "unseen_permutation_claim_authorized": False,
        "actuation_claim_authorized": False,
        "confirmation_reuse_authorized": False,
    }


def make_readout(kind: str, features: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if kind == "shared":
        return TokenTapeReadout(features, cursor_query=True, cursor_value=False)
    if kind == "cursor_specific":
        return TokenTapeReadout(features, cursor_query=True, cursor_value=True)
    if kind == "mean_joint":
        return MeanJointTapeReadout(features)
    if kind == "source_only":
        return TokenTapeReadout(features, cursor_query=False, cursor_value=False)
    if kind == "cursor_only":
        return CursorOnlyTapeReadout(features)
    raise ValueError(f"unknown token-tape readout kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--development-view-audit", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    arguments = parser.parse_args()
    if not torch.cuda.is_available() or arguments.cache_batch_size <= 0:
        raise SystemExit("token-tape diagnostic requires CUDA and a positive cache batch")
    if (
        len(arguments.runtime_sha256) != 64
        or any(char not in "0123456789abcdef" for char in arguments.runtime_sha256)
    ):
        raise SystemExit("runtime SHA-256 is malformed")
    inherited = [name for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD") if os.getenv(name)]
    if inherited:
        raise SystemExit(f"unsafe inherited runtime variables: {','.join(inherited)}")
    base = require_read_only(arguments.base, BASE_SHA256, "base")
    implementation_hashes = verify_implementation(arguments.implementation_commit)
    dataset = load_development_view(
        arguments.development_view,
        arguments.development_view_audit,
        expected_view_sha256=DEV_VIEW_SHA256,
        expected_audit_sha256=DEV_VIEW_AUDIT_SHA256,
    )
    payload = torch.load(base, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("step") != BASE_STEP or not isinstance(payload.get("cfg"), dict):
        raise ValueError("base checkpoint metadata mismatch")
    cfg = GPTConfig(**payload["cfg"])
    if cfg.n_loop != 1 or cfg.d_model != 576 or cfg.n_head != 9:
        raise ValueError("base architecture is outside the frozen diagnostic")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    model = GPT(cfg)
    model.load_state_dict(payload["model"], strict=True)
    freeze_base(model)
    model = model.to("cuda")
    del payload
    labels = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long, device="cuda")
    train = extract_tape_split(
        model, dataset.split("train"), labels, arguments.cache_batch_size,
    )
    development = extract_tape_split(
        model, dataset.split("development"), labels, arguments.cache_batch_size,
    )
    train_pre, dev_pre, pre_mean, pre_std = standardize_tapes(
        train.pre_source, train.valid_mask, development.pre_source,
    )
    train_embedding, dev_embedding, embedding_mean, embedding_std = standardize_tapes(
        train.embedding_source, train.valid_mask, development.embedding_source,
    )
    train_position_raw = position_tape(
        len(train.features.source_ids), train.pre_source.shape[1], cfg.d_model, "cuda",
    )
    development_position_raw = position_tape(
        len(development.features.source_ids),
        development.pre_source.shape[1],
        cfg.d_model,
        "cuda",
    )
    train_position, dev_position, position_mean, position_std = standardize_tapes(
        train_position_raw, train.valid_mask, development_position_raw,
    )
    train_derangement = derangement(len(train.features.source_ids), SEED + 500, "cuda")
    development_derangement = derangement(
        len(development.features.source_ids), SEED + 501, "cuda",
    )
    train_deranged = train_pre.index_select(0, train_derangement)
    train_deranged_mask = train.valid_mask.index_select(0, train_derangement)
    development_deranged = dev_pre.index_select(0, development_derangement)
    development_deranged_mask = development.valid_mask.index_select(
        0, development_derangement,
    )
    train_rms = rms_normalize_tape(train.pre_source)
    development_rms = rms_normalize_tape(development.pre_source)
    del model

    arms = {}
    specifications = []
    for index, seed in enumerate((SEED, SEED + 1, SEED + 2)):
        specifications.extend((
            (
                f"pre_final_shared_seed{index}", "shared", seed,
                train_pre, train.valid_mask, dev_pre, development.valid_mask,
                "train_standardized_pre_final_token_tape",
            ),
            (
                f"pre_final_cursor_specific_seed{index}", "cursor_specific", seed,
                train_pre, train.valid_mask, dev_pre, development.valid_mask,
                "train_standardized_pre_final_token_tape",
            ),
        ))
    specifications.extend((
        (
            "embedding_shared", "shared", SEED,
            train_embedding, train.valid_mask, dev_embedding, development.valid_mask,
            "train_standardized_input_embedding_tape",
        ),
        (
            "position_shared", "shared", SEED,
            train_position, train.valid_mask, dev_position, development.valid_mask,
            "train_standardized_sinusoidal_position_tape",
        ),
        (
            "source_deranged_shared", "shared", SEED,
            train_deranged, train_deranged_mask,
            development_deranged, development_deranged_mask,
            "source_deranged_train_standardized_pre_final_tape",
        ),
        (
            "pre_final_raw_shared", "shared", SEED,
            train.pre_source, train.valid_mask,
            development.pre_source, development.valid_mask,
            "raw_pre_final_token_tape",
        ),
        (
            "pre_final_rms_shared", "shared", SEED,
            train_rms, train.valid_mask, development_rms, development.valid_mask,
            "per_token_rms_pre_final_token_tape",
        ),
        (
            "mean_joint", "mean_joint", SEED,
            train_pre, train.valid_mask, dev_pre, development.valid_mask,
            "train_standardized_pre_final_token_tape",
        ),
        (
            "source_only_tape", "source_only", SEED,
            train_pre, train.valid_mask, dev_pre, development.valid_mask,
            "train_standardized_pre_final_token_tape",
        ),
        (
            "cursor_only", "cursor_only", SEED,
            train_pre, train.valid_mask, dev_pre, development.valid_mask,
            "no_token_state",
        ),
    ))
    for (
        name, kind, seed, train_tape, train_mask, development_tape,
        development_mask, surface,
    ) in specifications:
        print(f"[token-tape] fitting {name}", flush=True)
        _, readout_report = train_tape_readout(
            name=name,
            seed=seed,
            model=make_readout(kind, cfg.d_model, seed),
            train_tape=train_tape,
            train_mask=train_mask,
            development_tape=development_tape,
            development_mask=development_mask,
            train=train.features,
            development=development.features,
        )
        arms[name] = {
            "feature_surface": surface,
            "readout": readout_report,
        }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "base_sha256": BASE_SHA256,
            "base_step": BASE_STEP,
            "development_view_sha256": DEV_VIEW_SHA256,
            "development_view_audit_sha256": DEV_VIEW_AUDIT_SHA256,
            "source_canary_sha256": dataset.source_canary_sha256,
            "source_audit_sha256": dataset.source_audit_sha256,
            "tokenizer_sha256": dataset.tokenizer_sha256,
            "implementation_commit": arguments.implementation_commit,
            "implementation_file_sha256": implementation_hashes,
            "runtime_sha256": arguments.runtime_sha256,
        },
        "data_contract": {
            "allowed_splits": ["train", "development"],
            "runtime_artifact_contains_confirmation": False,
            "train_cells": len(train.features.target),
            "development_cells": len(development.features.target),
        },
        "optimization": {
            "family_seeds": [SEED, SEED + 1, SEED + 2],
            "control_seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "epsilon": EPSILON,
            "gradient_clip": GRADIENT_CLIP,
        },
        "feature_standardization": {
            "pre_mean": pre_mean.detach().cpu().tolist(),
            "pre_std": pre_std.detach().cpu().tolist(),
            "embedding_mean": embedding_mean.detach().cpu().tolist(),
            "embedding_std": embedding_std.detach().cpu().tolist(),
            "position_mean": position_mean.detach().cpu().tolist(),
            "position_std": position_std.detach().cpu().tolist(),
            "train_derived_scalars": 3_456,
            "raw_and_per_token_rms_controls_present": True,
        },
        "derangement": {
            "train_seed": SEED + 500,
            "development_seed": SEED + 501,
            "train_index_sha256": tensor_sha256(train_derangement),
            "development_index_sha256": tensor_sha256(development_derangement),
        },
        "runtime": {
            "pythonpath_pythonhome_ld_preload_absent": True,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_module_path": str(Path(torch.__file__).resolve()),
            "model_module_path": str(Path(sys.modules["model"].__file__).resolve()),
            "probe_dependency_module_path": str(
                Path(sys.modules["probe_counterfactual_cursor_readout"].__file__).resolve()
            ),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "arms": arms,
    }
    report["gate"] = decision(arms)
    digest = write_exclusive_read_only(arguments.out, report)
    print(json.dumps({
        "out": str(arguments.out),
        "sha256": digest,
        "decision": report["gate"]["decision"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
