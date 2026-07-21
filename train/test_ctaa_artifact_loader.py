from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from ctaa_artifact_loader import (
    load_qualified_memory_state,
    load_raw_trunk,
    require_sha256,
    verify_complete_system_parameters,
)
from model import GPT, GPTConfig


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tiny_checkpoint(path: Path) -> tuple[GPT, str]:
    model = GPT(
        GPTConfig(
            vocab_size=32,
            n_layer=2,
            n_head=3,
            n_kv_head=1,
            d_model=24,
            d_ff=48,
            seq_len=16,
            n_loop=1,
            zloss=0.0,
        )
    )
    torch.save(
        {
            "cfg": model.cfg.__dict__,
            "model": model.state_dict(),
            "step": 7,
            "data_seed": 1,
            "data_stream_generation": 0,
            "data_stream_seed": 2,
        },
        path,
    )
    return model, file_sha(path)


def test_raw_loader_requires_hash_schema_step_loop_and_exact_parameter_count(tmp_path: Path) -> None:
    path = tmp_path / "base.pt"
    source, digest = tiny_checkpoint(path)
    expected = sum(parameter.numel() for parameter in source.parameters())
    loaded, receipt = load_raw_trunk(
        path,
        expected_sha256=digest,
        expected_step=7,
        expected_parameters=expected,
    )
    assert receipt.sha256 == digest
    assert receipt.unique_parameters == expected
    assert receipt.missing_keys == receipt.unexpected_keys == ()
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="SHA-256"):
        require_sha256(path, "0" * 64, "test")


def test_qualified_memory_loader_checks_metadata_and_exact_tensor_scope(tmp_path: Path) -> None:
    path = tmp_path / "compiler.pt"
    state = {
        "memory_norm.weight": torch.ones(3),
        "memory_projection.weight": torch.ones(2, 3),
        "decoder.weight": torch.ones(4, 4),
    }
    metadata = {
        "base_sha256": "b",
        "tokenizer_sha256": "t",
        "base_step": 300_000,
        "layer": 19,
        "width": 384,
        "heads": 8,
        "encoder_layers": 5,
        "ff": 1408,
        "ordinary_tagger": True,
    }
    torch.save({"compiler": metadata, "adapter_state": state}, path)
    memory = load_qualified_memory_state(
        path,
        expected_sha256=file_sha(path),
        expected_base_sha256="b",
        expected_tokenizer_sha256="t",
        expected_tensors=2,
    )
    assert set(memory) == {"memory_norm.weight", "memory_projection.weight"}
    metadata["layer"] = 18
    torch.save({"compiler": metadata, "adapter_state": state}, path)
    with pytest.raises(ValueError, match="metadata"):
        load_qualified_memory_state(
            path,
            expected_sha256=file_sha(path),
            expected_base_sha256="b",
            expected_tokenizer_sha256="t",
            expected_tensors=2,
        )


def test_complete_parameter_ledger_fails_closed() -> None:
    model = GPT(
        GPTConfig(
            vocab_size=32,
            n_layer=1,
            n_head=3,
            n_kv_head=1,
            d_model=24,
            d_ff=48,
            seq_len=8,
            zloss=0.0,
        )
    )
    trunk = sum(parameter.numel() for parameter in model.parameters())
    receipt = verify_complete_system_parameters(
        model,
        100,
        9,
        expected_total=trunk + 109,
    )
    assert receipt["total"] == trunk + 109
    with pytest.raises(ValueError, match="ledger"):
        verify_complete_system_parameters(model, 100, 9, expected_total=trunk + 110)
