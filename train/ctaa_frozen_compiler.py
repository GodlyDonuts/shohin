"""Strict loader and source batching for a frozen CTAA compiler."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer

from ctaa_artifact_loader import (
    TOKENIZER_SHA256,
    load_qualified_memory_state,
    load_raw_trunk,
    require_sha256,
    verify_complete_system_parameters,
)
from ctaa_neural_core import ClosureFeatureTransitionCore
from ctaa_trunk_compiler import TrunkCausalCTAACompiler
from ctaa_evaluation_io import sha256_file
from train_ctaa_compiler import SCHEMA as COMPILER_TRAINING_SCHEMA


@dataclass(frozen=True)
class FrozenCompilerBundle:
    compiler: TrunkCausalCTAACompiler
    tokenizer: Tokenizer
    device: torch.device
    compiler_sha256: str
    parameter_ledger: dict[str, int]


def load_frozen_compiler(
    *,
    base_path: Path,
    qualified_path: Path,
    tokenizer_path: Path,
    compiler_path: Path,
    device_name: str,
) -> FrozenCompilerBundle:
    require_sha256(tokenizer_path, TOKENIZER_SHA256, "tokenizer")
    trunk, base_receipt = load_raw_trunk(base_path)
    qualified = load_qualified_memory_state(qualified_path)
    compiler = TrunkCausalCTAACompiler(trunk)
    compiler.initialize_qualified_memory(qualified)
    payload = torch.load(compiler_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "adapter_state", "training"}
        or payload.get("schema") != COMPILER_TRAINING_SCHEMA
        or not isinstance(payload.get("adapter_state"), dict)
        or not isinstance(payload.get("training"), dict)
    ):
        raise ValueError("CTAA frozen compiler checkpoint schema differs")
    training = payload["training"]
    expected_bindings = {
        "base_sha256": base_receipt.sha256,
        "base_step": base_receipt.step,
        "qualified_compiler_sha256": sha256_file(qualified_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "adapter_parameters": compiler.adapter_num_parameters,
    }
    if any(training.get(key) != value for key, value in expected_bindings.items()):
        raise ValueError("CTAA frozen compiler checkpoint bindings differ")
    own = compiler.state_dict()
    expected_adapter = {name for name in own if not name.startswith("model.")}
    adapter_state = payload["adapter_state"]
    if set(adapter_state) != expected_adapter:
        raise ValueError("CTAA frozen compiler adapter keys differ")
    with torch.no_grad():
        for name in sorted(expected_adapter):
            value = adapter_state[name]
            if not isinstance(value, torch.Tensor) or value.shape != own[name].shape:
                raise ValueError("CTAA frozen compiler adapter tensor differs")
            own[name].copy_(value)
    ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        ClosureFeatureTransitionCore().unique_parameters,
    )
    if training.get("parameter_ledger") != ledger:
        raise ValueError("CTAA frozen compiler parameter ledger differs")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA frozen compiler requires available CUDA")
    compiler.to(device).eval()
    return FrozenCompilerBundle(
        compiler=compiler,
        tokenizer=Tokenizer.from_file(str(tokenizer_path)),
        device=device,
        compiler_sha256=sha256_file(compiler_path),
        parameter_ledger=ledger,
    )


def load_source_rows(path: Path, source_key: str) -> tuple[list[str], list[str]]:
    expected = {"family_id", source_key}
    family_ids: list[str] = []
    sources: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError(f"CTAA source row {line_number} schema differs")
            family_id = value["family_id"]
            source = value[source_key]
            if not isinstance(family_id, str) or not family_id or not isinstance(source, str) or not source:
                raise ValueError(f"CTAA source row {line_number} value differs")
            family_ids.append(family_id)
            sources.append(source)
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise ValueError("CTAA source family IDs differ")
    return family_ids, sources


def token_batches(
    sources: Iterable[str],
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    max_length: int,
    padding_id: int,
    device: torch.device,
) -> Iterable[torch.Tensor]:
    if batch_size < 1:
        raise ValueError("CTAA compiler evaluation batch size differs")
    encoded = [tuple(tokenizer.encode(source).ids) for source in sources]
    if any(not row or len(row) > max_length for row in encoded):
        raise ValueError("CTAA compiler evaluation token length differs")
    for start in range(0, len(encoded), batch_size):
        rows = encoded[start : start + batch_size]
        width = max(len(row) for row in rows)
        batch = torch.full((len(rows), width), padding_id, dtype=torch.long)
        for index, row in enumerate(rows):
            batch[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        yield batch.to(device)

