from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_pointer_compiler import (  # noqa: E402
    BYTE_PAD_ID,
    DirectByteEFCCompiler,
    MAX_KEY_OCCURRENCES,
    MAX_SOURCE_BYTES,
    MAX_UNIQUE_KEYS,
    PointerCompilerError,
    collate_sources,
    scan_source,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    GrammarFactors,
    SOURCE_FACTOR_COMBINATIONS,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _payload(values: tuple[int, int, int]) -> bytes:
    machine = generate_machine(
        seed="efc-pointer-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-pointer-test-v1",
        split="mechanics",
        index=0,
    )
    return encode_source(evidence, GrammarFactors(*values))


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_generic_scanner_copies_all_opaque_keys_exactly(
    values: tuple[int, int, int],
) -> None:
    source = scan_source(_payload(values))
    assert len(source.unique_keys) == 13
    assert len(source.spans) == 99
    for (start, end), key in zip(
        source.spans,
        source.occurrence_keys,
        strict=True,
    ):
        token = source.payload[start:end]
        value = int(token[1:], 16 if token.startswith(b"h") else 10)
        assert key == value.to_bytes(8, "little")


def test_collation_is_bounded_padded_and_exact() -> None:
    sources = [
        scan_source(_payload((0, 0, 0))),
        scan_source(_payload((1, 1, 1))),
    ]
    batch = collate_sources(sources)
    assert batch.byte_ids.shape[0] == 2
    assert batch.byte_ids.shape[1] == max(len(source.payload) for source in sources)
    assert batch.span_bounds.shape == (2, MAX_KEY_OCCURRENCES, 2)
    assert batch.unique_key_bytes.shape == (2, MAX_UNIQUE_KEYS, 8)
    assert batch.unique_key_valid.sum(1).tolist() == [13, 13]
    for row, source in enumerate(sources):
        assert bytes(batch.byte_ids[row, : len(source.payload)].tolist()) == source.payload
        assert bool(
            batch.byte_ids[row, len(source.payload) :].eq(BYTE_PAD_ID).all()
        )
        assert (
            batch.unique_key_bytes[row, :13].flatten().tolist()
            == list(b"".join(source.unique_keys))
        )


def test_scanner_fails_closed_on_missing_overflow_and_oversize() -> None:
    with pytest.raises(PointerCompilerError, match="key geometry"):
        scan_source(b"BEGIN-EFC\nEND-EFC\n")
    too_many = b" ".join(
        f"h{value + 1:016x}".encode("ascii")
        for value in range(MAX_UNIQUE_KEYS + 1)
    )
    with pytest.raises(PointerCompilerError, match="too many unique"):
        scan_source(too_many)
    with pytest.raises(PointerCompilerError, match="byte length"):
        scan_source(b"x" * (MAX_SOURCE_BYTES + 1))
    with pytest.raises(PointerCompilerError, match="uint64"):
        scan_source(b"d18446744073709551616")


def test_direct_baseline_forward_has_attached_machine_gradients() -> None:
    torch.manual_seed(11)
    batch = collate_sources(
        (
            scan_source(_payload((0, 0, 0))),
            scan_source(_payload((1, 1, 1))),
        )
    )
    compiler = DirectByteEFCCompiler(
        width=64,
        convolution_layers=2,
        decoder_layers=1,
        heads=4,
        feedforward=128,
    )
    output = compiler(batch)
    assert output.machine.state_active.shape == (2, 16, 2)
    assert output.machine.action_active.shape == (2, 8, 2)
    assert output.machine.observer_active.shape == (2, 8, 2)
    assert output.machine.action_next.shape == (2, 8, 16, 16)
    assert output.machine.observer_answer.shape == (2, 8, 16, 5)
    assert output.key_assignment_logits.shape == (2, 32, 32)
    assert output.unique_key_valid.sum(1).tolist() == [13, 13]
    assert compiler.parameter_count() < 2_000_000

    machine_loss = sum(
        tensor.square().mean()
        for tensor in (
            output.machine.state_active,
            output.machine.action_active,
            output.machine.observer_active,
            output.machine.action_next,
            output.machine.observer_answer,
            output.key_assignment_logits,
        )
    )
    machine_loss.backward()
    missing = [
        name
        for name, parameter in compiler.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.parameters()
        if parameter.grad is not None
    )


def test_direct_baseline_rejects_odd_width_position_geometry() -> None:
    with pytest.raises(PointerCompilerError, match="geometry"):
        DirectByteEFCCompiler(width=33, heads=3)
