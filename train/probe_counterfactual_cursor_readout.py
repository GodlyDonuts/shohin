#!/usr/bin/env python3
"""Development-only representation/readout decomposition for R12 cursor action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from counterfactual_cursor_action_dev_view import LABEL_TOKEN_IDS, load_development_view
from model import GPT, GPTConfig


SCHEMA = "counterfactual_cursor_readout_actuation_dev_v1"
BASE_STEP = 260000
BASE_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
DEV_VIEW_SHA256 = "24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9"
DEV_VIEW_AUDIT_SHA256 = "33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25"
SEED = 2026071601
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.03
BETAS = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 5.0
ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "R12_CURSOR_READOUT_ACTUATION_DIAGNOSTIC.md",
    "train/model.py",
    "train/counterfactual_cursor_action_dev_view.py",
    "train/probe_counterfactual_cursor_readout.py",
    "train/test_probe_counterfactual_cursor_readout.py",
    "train/jobs/probe_counterfactual_cursor_readout.sbatch",
)


@dataclass(frozen=True)
class FrozenInputCache:
    hidden: torch.Tensor
    prompt_last: torch.Tensor
    max_tokens: int


@dataclass(frozen=True)
class SplitFeatures:
    name: str
    pre: torch.Tensor
    post: torch.Tensor
    cursor: torch.Tensor
    target: torch.Tensor
    source_index: torch.Tensor
    source_ids: tuple[str, ...]
    renderer_ids: torch.Tensor
    permutation_ids: torch.Tensor
    base_action_logits: torch.Tensor
    nonaction_logsumexp: torch.Tensor
    nonaction_max: torch.Tensor


class JointCursorReadout(nn.Module):
    """Independent five-action linear readout at each supplied cursor."""

    def __init__(self, features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5, 5, features))
        self.bias = nn.Parameter(torch.zeros(5, 5))

    def forward(self, hidden: torch.Tensor, cursor: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.weight.shape[2]:
            raise ValueError("hidden shape is incompatible with joint readout")
        if cursor.dtype != torch.long or cursor.shape != (hidden.shape[0],):
            raise ValueError("cursor must be an int64 batch vector")
        if bool(((cursor < 0) | (cursor > 4)).any()):
            raise ValueError("cursor is outside [0,4]")
        weight = self.weight[cursor]
        return torch.einsum("bad,bd->ba", weight, hidden) + self.bias[cursor]


class SourceOnlyReadout(nn.Module):
    """Five-action linear source classifier shared across cursor values."""

    def __init__(self, features: int):
        super().__init__()
        self.linear = nn.Linear(features, 5)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor, cursor: torch.Tensor) -> torch.Tensor:
        del cursor
        return self.linear(hidden)


class CursorOnlyReadout(nn.Module):
    """Five-by-five cursor table independent of source features."""

    def __init__(self, features: int):
        super().__init__()
        del features
        self.table = nn.Embedding(5, 5)
        nn.init.zeros_(self.table.weight)

    def forward(self, hidden: torch.Tensor, cursor: torch.Tensor) -> torch.Tensor:
        del hidden
        return self.table(cursor)


class FrozenScoreCalibrator(nn.Module):
    """Two-scalar positive gain plus common action-logit bias."""

    def __init__(self):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(0.54132485))
        self.beta = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    def forward(self, frozen_scores: torch.Tensor) -> torch.Tensor:
        return self.alpha * frozen_scores + self.beta


def freeze_base(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def encode_before_final_block(model: GPT, input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("input_ids must be an int64 [batch,tokens] tensor")
    if model.cfg.n_loop != 1 or input_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("input is outside the frozen architecture")
    hidden = model.tok(input_ids)
    cos = model.cos[:input_ids.shape[1]].to(hidden.device)
    sin = model.sin[:input_ids.shape[1]].to(hidden.device)
    for block in model.blocks[:-1]:
        hidden, _ = block(hidden, cos, sin)
    return hidden.detach()


@torch.no_grad()
def build_input_cache(
    model: GPT, token_sequences: tuple[tuple[int, ...], ...], *, cache_batch_size: int,
) -> FrozenInputCache:
    if not token_sequences or cache_batch_size <= 0:
        raise ValueError("cache inputs and batch size must be nonempty")
    if any(not sequence for sequence in token_sequences):
        raise ValueError("cache input contains an empty sequence")
    max_tokens = max(len(sequence) for sequence in token_sequences)
    if max_tokens > model.cfg.seq_len:
        raise ValueError("cache input exceeds model context")
    hidden_cache = None
    positions = torch.tensor(
        [len(sequence) - 1 for sequence in token_sequences],
        dtype=torch.long,
        device="cuda",
    )
    for start in range(0, len(token_sequences), cache_batch_size):
        batch = token_sequences[start:start + cache_batch_size]
        input_ids = torch.zeros((len(batch), max_tokens), dtype=torch.long, device="cuda")
        for local, sequence in enumerate(batch):
            input_ids[local, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device="cuda",
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encode_before_final_block(model, input_ids)
        if hidden_cache is None:
            hidden_cache = torch.empty(
                (len(token_sequences), max_tokens, model.cfg.d_model),
                dtype=hidden.dtype,
                device="cuda",
            )
        hidden_cache[start:start + len(batch)].copy_(hidden)
    if hidden_cache is None:
        raise AssertionError("cache allocation did not occur")
    return FrozenInputCache(hidden_cache, positions, max_tokens)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_components(path: str | Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
    return absolute


def require_read_only(path: str | Path, expected_sha256: str, label: str) -> Path:
    absolute = reject_symlink_components(path)
    metadata = absolute.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
        raise ValueError(f"{label} must be a read-only regular file")
    observed = sha256_file(absolute)
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return absolute


def verify_implementation(commit: str) -> dict[str, str]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("implementation commit is malformed")
    marker = reject_symlink_components(ROOT / ".implementation_commit")
    if not marker.is_file() or marker.stat().st_mode & 0o222:
        raise ValueError("immutable implementation marker is missing")
    if marker.read_text(encoding="ascii").strip() != commit:
        raise ValueError("implementation marker differs from requested commit")
    result = {}
    for relative in IMPLEMENTATION_FILES:
        live = reject_symlink_components(ROOT / relative)
        if not live.is_file() or live.stat().st_mode & 0o222:
            raise ValueError(f"exported implementation is not immutable: {relative}")
        result[relative] = sha256_file(live)
    return result


def exact_full_vocabulary_loss(
    base_action_logits: torch.Tensor,
    nonaction_logsumexp: torch.Tensor,
    action_delta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Exact CE after adding deltas only to the five action-token logits."""
    adjusted = base_action_logits + action_delta
    action_logsumexp = torch.logsumexp(adjusted, dim=-1)
    denominator = torch.logaddexp(nonaction_logsumexp, action_logsumexp)
    target_logit = adjusted.gather(1, target[:, None]).squeeze(1)
    return (denominator - target_logit).mean()


def standardize(
    train: torch.Tensor, other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (train - mean) / std, (other - mean) / std, mean, std


def _base_evidence(
    full_logits: torch.Tensor, label_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = full_logits.index_select(1, label_token_ids)
    masked = full_logits.clone()
    masked[:, label_token_ids] = -torch.inf
    return actions, torch.logsumexp(masked, dim=-1), masked.max(dim=-1).values


@torch.no_grad()
def extract_split(
    model: GPT, split, label_token_ids: torch.Tensor, cache_batch_size: int,
) -> SplitFeatures:
    if split.name not in {"train", "development"}:
        raise ValueError("diagnostic may only extract train/development")
    source_sequences = tuple(source.prompt_token_ids for source in split.sources)
    cache = build_input_cache(model, source_sequences, cache_batch_size=cache_batch_size)
    positions = cache.prompt_last
    row = torch.arange(len(split.sources), device="cuda")
    pre_source = cache.hidden[row, positions].float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cos = model.cos[:cache.max_tokens].to("cuda")
        sin = model.sin[:cache.max_tokens].to("cuda")
        final, _ = model.blocks[-1](cache.hidden, cos, sin)
        post_source = model.norm(final[row, positions]).float()
        full_source_logits = model.head(post_source.to(model.head.weight.dtype)).float()
    action_source, non_lse_source, non_max_source = _base_evidence(
        full_source_logits, label_token_ids,
    )
    source_index = torch.tensor(
        [split.source_index_by_id[cell.source_id] for cell in split.cells],
        dtype=torch.long,
        device="cuda",
    )
    cursor = torch.tensor([cell.cursor for cell in split.cells], dtype=torch.long, device="cuda")
    target = torch.tensor(
        [cell.target_index for cell in split.cells], dtype=torch.long, device="cuda",
    )
    renderer_ids = torch.tensor(
        [split.sources[index].renderer_id for index in source_index.tolist()],
        dtype=torch.long,
        device="cuda",
    )
    permutation_ids = torch.tensor(
        [split.sources[index].permutation_id for index in source_index.tolist()],
        dtype=torch.long,
        device="cuda",
    )
    return SplitFeatures(
        name=split.name,
        pre=pre_source.index_select(0, source_index),
        post=post_source.index_select(0, source_index),
        cursor=cursor,
        target=target,
        source_index=source_index,
        source_ids=tuple(source.source_id for source in split.sources),
        renderer_ids=renderer_ids,
        permutation_ids=permutation_ids,
        base_action_logits=action_source.index_select(0, source_index),
        nonaction_logsumexp=non_lse_source.index_select(0, source_index),
        nonaction_max=non_max_source.index_select(0, source_index),
    )


def _metric_counts(correct: torch.Tensor, split: SplitFeatures) -> dict[str, object]:
    correct_cpu = correct.detach().cpu()
    source_cpu = split.source_index.detach().cpu()
    renderer_cpu = split.renderer_ids.detach().cpu()
    exact = []
    for source in range(len(split.source_ids)):
        exact.append(bool(correct_cpu[source_cpu == source].all()))
    per_renderer = {}
    for renderer in sorted(set(renderer_cpu.tolist())):
        mask = renderer_cpu == renderer
        numerator = int(correct_cpu[mask].sum())
        per_renderer[str(renderer)] = {
            "numerator": numerator,
            "denominator": int(mask.sum()),
            "proportion": numerator / int(mask.sum()),
        }
    numerator = int(correct_cpu.sum())
    exact_numerator = sum(exact)
    return {
        "cell_accuracy": {
            "numerator": numerator,
            "denominator": len(correct_cpu),
            "proportion": numerator / len(correct_cpu),
        },
        "exact_five_action_groups": {
            "numerator": exact_numerator,
            "denominator": len(exact),
            "proportion": exact_numerator / len(exact),
        },
        "per_renderer_cell_accuracy": per_renderer,
    }


@torch.no_grad()
def evaluate_restricted(
    model: nn.Module, hidden: torch.Tensor, split: SplitFeatures,
) -> tuple[nn.Module, dict[str, object]]:
    restricted_scores = model(hidden, split.cursor)
    restricted_top, restricted = restricted_scores.max(dim=-1)
    restricted_unique = restricted_scores.eq(restricted_top[:, None]).sum(dim=-1).eq(1)
    return {
        "restricted": _metric_counts(
            restricted_unique & restricted.eq(split.target), split,
        ),
    }


def _state_payload(model: nn.Module) -> dict[str, object]:
    digest = hashlib.sha256()
    tensors = {}
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
        tensors[key] = tensor.tolist()
    return {"sha256": digest.hexdigest(), "tensors": tensors}


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.quantile(0.5)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


@torch.no_grad()
def evaluate_calibrated(
    readout: nn.Module,
    calibrator: FrozenScoreCalibrator,
    hidden: torch.Tensor,
    split: SplitFeatures,
    *,
    include_examples: bool,
) -> dict[str, object]:
    scores = readout(hidden, split.cursor)
    delta = calibrator(scores)
    adjusted = split.base_action_logits + delta
    restricted_top, restricted_index = scores.max(dim=-1)
    restricted_unique = scores.eq(restricted_top[:, None]).sum(dim=-1).eq(1)
    action_score, action_index = adjusted.max(dim=-1)
    action_unique = adjusted.eq(action_score[:, None]).sum(dim=-1).eq(1)
    action_wins = action_unique & action_score.gt(split.nonaction_max)
    full_correct = action_wins & action_index.eq(split.target)
    row = torch.arange(len(split.target), device=split.target.device)
    base_target_margin = (
        split.base_action_logits[row, split.target] - split.nonaction_max
    )
    adjusted_target_margin = adjusted[row, split.target] - split.nonaction_max
    result = {
        "restricted": _metric_counts(
            restricted_unique & restricted_index.eq(split.target), split,
        ),
        "full_vocabulary": _metric_counts(full_correct, split),
        "action_token_wins": {
            "numerator": int(action_wins.sum()),
            "denominator": len(action_score),
            "proportion": float(action_wins.float().mean()),
        },
        "base_target_minus_best_nonaction": _distribution(base_target_margin),
        "adjusted_target_minus_best_nonaction": _distribution(adjusted_target_margin),
        "delta_linf": _distribution(delta.abs().max(dim=-1).values),
        "delta_l2": _distribution(delta.norm(dim=-1)),
        "restricted_correct_class_margin": _distribution(
            scores[row, split.target]
            - scores.masked_fill(
                F.one_hot(split.target, num_classes=5).bool(), -torch.inf,
            ).max(dim=-1).values
        ),
    }
    if include_examples:
        source_index = split.source_index.detach().cpu().tolist()
        result["examples"] = [
            {
                "source_id": split.source_ids[source_index[index]],
                "renderer_id": int(split.renderer_ids[index]),
                "permutation_id": int(split.permutation_ids[index]),
                "cursor": int(split.cursor[index]),
                "target_index": int(split.target[index]),
                "restricted_prediction": int(restricted_index[index]),
                "restricted_unique": bool(restricted_unique[index]),
                "action_prediction": int(action_index[index]),
                "action_wins_full_vocabulary": bool(action_wins[index]),
                "full_vocabulary_correct": bool(full_correct[index]),
                "base_target_minus_best_nonaction": float(base_target_margin[index]),
                "adjusted_target_minus_best_nonaction": float(adjusted_target_margin[index]),
                "delta_linf": float(delta[index].abs().max()),
            }
            for index in range(len(split.target))
        ]
    return result


def train_readout(
    *,
    name: str,
    model_type: str,
    train_hidden: torch.Tensor,
    development_hidden: torch.Tensor,
    train: SplitFeatures,
    development: SplitFeatures,
) -> tuple[nn.Module, dict[str, object]]:
    constructors = {
        "joint": JointCursorReadout,
        "source_only": SourceOnlyReadout,
        "cursor_only": CursorOnlyReadout,
    }
    model = constructors[model_type](train_hidden.shape[1]).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    updates = 0
    epoch_losses = []
    for epoch in range(EPOCHS):
        order = list(range(len(train.target)))
        random.Random(SEED + epoch).shuffle(order)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = torch.tensor(order[start:start + BATCH_SIZE], device="cuda")
            scores = model(
                train_hidden.index_select(0, index), train.cursor.index_select(0, index),
            )
            loss = F.cross_entropy(scores, train.target.index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                raise RuntimeError(f"non-finite optimization state for {name}")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            updates += 1
        epoch_losses.append(loss_sum / len(order))
    model.eval()
    report = {
        "name": name,
        "model_type": model_type,
        "objective": "restricted_then_two_scalar_full_vocabulary_calibration",
        "trainable_scalars": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "first_epoch_loss": epoch_losses[0],
        "final_epoch_loss": epoch_losses[-1],
        "state": _state_payload(model),
        "train": evaluate_restricted(model, train_hidden, train),
        "development": evaluate_restricted(model, development_hidden, development),
    }
    return model, report


def calibrate_readout(
    *,
    readout: nn.Module,
    train_hidden: torch.Tensor,
    development_hidden: torch.Tensor,
    train: SplitFeatures,
    development: SplitFeatures,
) -> dict[str, object]:
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_scores = readout(train_hidden, train.cursor).detach()
    calibrator = FrozenScoreCalibrator().to("cuda")
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=0.0,
    )
    losses = []
    updates = 0
    for epoch in range(EPOCHS):
        order = list(range(len(train.target)))
        random.Random(SEED + 10_000 + epoch).shuffle(order)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = torch.tensor(order[start:start + BATCH_SIZE], device="cuda")
            delta = calibrator(train_scores.index_select(0, index))
            loss = exact_full_vocabulary_loss(
                train.base_action_logits.index_select(0, index),
                train.nonaction_logsumexp.index_select(0, index),
                delta,
                train.target.index_select(0, index),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(calibrator.parameters(), GRADIENT_CLIP)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                raise RuntimeError("non-finite calibration state")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            updates += 1
        losses.append(loss_sum / len(order))
    calibrator.eval()
    return {
        "trainable_scalars": 2,
        "updates": updates,
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "alpha": float(calibrator.alpha.detach()),
        "beta": float(calibrator.beta.detach()),
        "state": _state_payload(calibrator),
        "train": evaluate_calibrated(
            readout, calibrator, train_hidden, train, include_examples=False,
        ),
        "development": evaluate_calibrated(
            readout, calibrator, development_hidden, development, include_examples=True,
        ),
    }


def _score(arm: Mapping[str, object], phase: str, mode: str, metric: str) -> float:
    return float(arm[phase]["development"][mode][metric]["proportion"])


def decision(arms: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    surfaces = {}
    for surface in ("pre", "post"):
        joint = arms[f"{surface}_joint"]
        source_only = arms[f"{surface}_source_only"]
        cursor_only = arms["cursor_only"]
        source_control_ok = (
            _score(source_only, "readout", "restricted", "cell_accuracy") <= 0.21
        )
        cursor_control_ok = (
            _score(cursor_only, "readout", "restricted", "cell_accuracy") <= 0.41
        )
        representation = (
            float(joint["readout"]["train"]["restricted"]["cell_accuracy"]["proportion"])
            >= 0.99
            and _score(joint, "readout", "restricted", "cell_accuracy") >= 0.95
            and _score(joint, "readout", "restricted", "exact_five_action_groups") >= 0.90
            and source_control_ok
            and cursor_control_ok
        )
        per_renderer = joint["calibration"]["development"]["full_vocabulary"][
            "per_renderer_cell_accuracy"
        ]
        calibrated_fit = (
            _score(joint, "calibration", "full_vocabulary", "cell_accuracy") >= 0.95
            and _score(joint, "calibration", "full_vocabulary", "exact_five_action_groups")
            >= 0.90
            and all(float(item["proportion"]) >= 0.95 for item in per_renderer.values())
            and _score(joint, "calibration", "full_vocabulary", "cell_accuracy")
            - _score(source_only, "calibration", "full_vocabulary", "cell_accuracy")
            >= 0.10
            and _score(joint, "calibration", "full_vocabulary", "cell_accuracy")
            - _score(cursor_only, "calibration", "full_vocabulary", "cell_accuracy")
            >= 0.10
        )
        surfaces[surface] = {
            "oracle_cursor_indexed_linear_separability": representation,
            "source_only_control_respects_ceiling": source_control_ok,
            "cursor_only_control_respects_ceiling": cursor_control_ok,
            "two_scalar_full_vocabulary_calibration_fits": calibrated_fit,
            "actuation_claim_authorized": False,
        }
    admitted = [
        surface for surface, result in surfaces.items()
        if result["oracle_cursor_indexed_linear_separability"]
        and result["two_scalar_full_vocabulary_calibration_fits"]
    ]
    return {
        "surfaces": surfaces,
        "admitted_surfaces": admitted,
        "decision": (
            "evidence_for_held_out_permutation_v2_prereg"
            if admitted else "cursor_indexed_linear_readout_no_go"
        ),
        "reasoning_claim_authorized": False,
        "internal_cursor_claim_authorized": False,
        "compositional_generalization_claim_authorized": False,
        "actuation_claim_authorized": False,
        "confirmation_reuse_authorized": False,
    }


def write_exclusive_read_only(path: Path, payload: object) -> str:
    destination = reject_symlink_components(path)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing existing output: {destination}")
    parent = reject_symlink_components(destination.parent)
    if not parent.is_dir():
        raise ValueError(f"output parent is not a directory: {parent}")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
        os.unlink(temporary)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


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
    if not torch.cuda.is_available():
        raise SystemExit("cursor readout diagnostic requires CUDA")
    if arguments.cache_batch_size <= 0:
        raise SystemExit("cache batch size must be positive")
    if len(arguments.runtime_sha256) != 64:
        raise SystemExit("runtime SHA-256 is malformed")

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
    train = extract_split(model, dataset.split("train"), labels, arguments.cache_batch_size)
    development = extract_split(
        model, dataset.split("development"), labels, arguments.cache_batch_size,
    )

    train_pre, dev_pre, pre_mean, pre_std = standardize(train.pre, development.pre)
    train_post, dev_post, post_mean, post_std = standardize(train.post, development.post)
    feature_sets = {"pre": (train_pre, dev_pre), "post": (train_post, dev_post)}
    arms = {}
    for surface, (train_hidden, dev_hidden) in feature_sets.items():
        for model_type in ("joint", "source_only"):
            name = f"{surface}_{model_type}"
            print(f"[cursor-readout] fitting {name}", flush=True)
            readout, readout_report = train_readout(
                name=name,
                model_type=model_type,
                train_hidden=train_hidden,
                development_hidden=dev_hidden,
                train=train,
                development=development,
            )
            arms[name] = {
                "readout": readout_report,
                "calibration": calibrate_readout(
                    readout=readout,
                    train_hidden=train_hidden,
                    development_hidden=dev_hidden,
                    train=train,
                    development=development,
                ),
            }
    print("[cursor-readout] fitting cursor_only", flush=True)
    readout, readout_report = train_readout(
        name="cursor_only",
        model_type="cursor_only",
        train_hidden=train_post,
        development_hidden=dev_post,
        train=train,
        development=development,
    )
    arms["cursor_only"] = {
        "readout": readout_report,
        "calibration": calibrate_readout(
            readout=readout,
            train_hidden=train_post,
            development_hidden=dev_post,
            train=train,
            development=development,
        ),
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
            "view_builder_implementation_commit": dataset.implementation_commit,
            "implementation_commit": arguments.implementation_commit,
            "implementation_file_sha256": implementation_hashes,
            "runtime_sha256": arguments.runtime_sha256,
        },
        "data_contract": {
            "allowed_splits": ["train", "development"],
            "runtime_artifact_contains_confirmation": False,
            "source_canary_path_provided_to_process": False,
            "train_cells": len(train.target),
            "development_cells": len(development.target),
            "train_sources": len(train.source_ids),
            "development_sources": len(development.source_ids),
        },
        "optimization": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "epsilon": EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "feature_standardization": {
            "pre_mean": pre_mean.detach().cpu().tolist(),
            "pre_std": pre_std.detach().cpu().tolist(),
            "post_mean": post_mean.detach().cpu().tolist(),
            "post_std": post_std.detach().cpu().tolist(),
        },
        "runtime": {
            "torch_version": torch.__version__,
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
        "admitted_surfaces": report["gate"]["admitted_surfaces"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
