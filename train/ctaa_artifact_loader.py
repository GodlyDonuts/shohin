"""Fail-closed immutable artifact loading for the CTAA falsifier."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch

from model import GPT, GPTConfig


RAW_CHECKPOINT_SHA256 = "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
QUALIFIED_COMPILER_SHA256 = "747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991"
RAW_STEP = 300_000
RAW_UNIQUE_PARAMETERS = 125_081_664
STRICT_SYSTEM_LIMIT = 149_999_999
QUALIFIED_MEMORY_TENSORS = 63


@dataclass(frozen=True)
class ImmutableArtifactReceipt:
    path: str
    sha256: str
    step: int
    unique_parameters: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"CTAA {label} SHA-256 differs")
    return observed


def load_raw_trunk(
    checkpoint_path: Path,
    *,
    expected_sha256: str = RAW_CHECKPOINT_SHA256,
    expected_step: int = RAW_STEP,
    expected_parameters: int = RAW_UNIQUE_PARAMETERS,
) -> tuple[GPT, ImmutableArtifactReceipt]:
    digest = require_sha256(checkpoint_path, expected_sha256, "raw checkpoint")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "cfg",
        "data_seed",
        "data_stream_generation",
        "data_stream_seed",
        "model",
        "step",
    }:
        raise ValueError("CTAA raw checkpoint schema differs")
    cfg = payload.get("cfg")
    state = payload.get("model")
    if not isinstance(cfg, dict) or not isinstance(state, Mapping):
        raise ValueError("CTAA raw checkpoint payload differs")
    if payload.get("step") != expected_step or cfg.get("n_loop") != 1:
        raise ValueError("CTAA raw checkpoint step/loop differs")
    model = GPT(GPTConfig(**cfg))
    incompatible = model.load_state_dict(state, strict=True)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ValueError("CTAA raw checkpoint strict load differs")
    model.requires_grad_(False).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != expected_parameters:
        raise ValueError("CTAA raw checkpoint parameter count differs")
    return model, ImmutableArtifactReceipt(
        path=str(checkpoint_path),
        sha256=digest,
        step=int(payload["step"]),
        unique_parameters=parameters,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_qualified_memory_state(
    compiler_path: Path,
    *,
    expected_sha256: str = QUALIFIED_COMPILER_SHA256,
    expected_base_sha256: str = RAW_CHECKPOINT_SHA256,
    expected_tokenizer_sha256: str = TOKENIZER_SHA256,
    expected_tensors: int = QUALIFIED_MEMORY_TENSORS,
) -> dict[str, torch.Tensor]:
    require_sha256(compiler_path, expected_sha256, "qualified compiler")
    payload = torch.load(compiler_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"compiler", "adapter_state"}:
        raise ValueError("CTAA qualified compiler schema differs")
    metadata = payload.get("compiler")
    state = payload.get("adapter_state")
    if not isinstance(metadata, dict) or not isinstance(state, Mapping):
        raise ValueError("CTAA qualified compiler payload differs")
    required_metadata = {
        "base_sha256": expected_base_sha256,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "base_step": RAW_STEP,
        "layer": 19,
        "width": 384,
        "heads": 8,
        "encoder_layers": 5,
        "ff": 1408,
        "ordinary_tagger": True,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError("CTAA qualified compiler metadata differs")
    prefixes = ("memory_norm.", "memory_projection.", "memory_encoder.")
    memory = {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
        if isinstance(name, str)
        and name.startswith(prefixes)
        and isinstance(value, torch.Tensor)
    }
    if len(memory) != expected_tensors:
        raise ValueError("CTAA qualified memory tensor count differs")
    return memory


def verify_complete_system_parameters(
    trunk: GPT,
    compiler_adapter_parameters: int,
    core_parameters: int,
    *,
    expected_total: int = 137_989_944,
) -> dict[str, int]:
    trunk_parameters = sum(parameter.numel() for parameter in trunk.parameters())
    total = trunk_parameters + compiler_adapter_parameters + core_parameters
    if total != expected_total or total > STRICT_SYSTEM_LIMIT:
        raise ValueError("CTAA complete-system parameter ledger differs")
    return {
        "trunk": trunk_parameters,
        "compiler_adapter": compiler_adapter_parameters,
        "core": core_parameters,
        "total": total,
        "headroom": STRICT_SYSTEM_LIMIT - total,
    }
