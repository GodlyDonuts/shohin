from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ctaa_packet_io import (
    read_packet_file,
    read_query_file,
    write_packet_file,
    write_query_file,
)
from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery


def sample_packet() -> HardCTAAPacket:
    schedule = [0, 1, 2, 3, 4, *([0] * 36)]
    return HardCTAAPacket(
        action_cards=torch.tensor(
            [
                [[1, 0, 2], [2, 2, 0], [0, 1, 1], [2, 0, 1]],
                [[0, 0, 2], [2, 1, 0], [1, 2, 2], [1, 0, 1]],
            ],
            dtype=torch.uint8,
        ),
        opcode_to_card=torch.tensor(
            [[2, 0, 3, 1], [1, 3, 0, 2]], dtype=torch.uint8
        ),
        initial_state=torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.uint8),
        opcode_schedule=torch.tensor([schedule, schedule], dtype=torch.uint8),
    )


def test_packet_binary_round_trip_is_exact_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "packet.bin"
    original = sample_packet()
    receipt = write_packet_file(path, original)
    restored = read_packet_file(path)
    assert receipt["rows"] == 2
    assert receipt["bytes_per_row"] == 60
    assert path.stat().st_mode & 0o222 == 0
    assert torch.equal(restored.action_cards, original.action_cards)
    assert torch.equal(restored.opcode_to_card, original.opcode_to_card)
    assert torch.equal(restored.initial_state, original.initial_state)
    assert torch.equal(restored.opcode_schedule, original.opcode_schedule)
    assert torch.equal(restored.resolved_schedule, original.resolved_schedule)
    with pytest.raises(FileExistsError):
        write_packet_file(path, original)


def test_late_query_is_a_physically_separate_one_byte_artifact(tmp_path: Path) -> None:
    path = tmp_path / "query.bin"
    original = HardCTAAQuery(position=torch.tensor([2, 0], dtype=torch.uint8))
    receipt = write_query_file(path, original)
    restored = read_query_file(path)
    assert receipt["bytes_per_row"] == 1
    assert torch.equal(restored.position, original.position)


def test_packet_reader_rejects_corrupt_header_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "packet.bin"
    write_packet_file(path, sample_packet())
    payload = bytearray(path.read_bytes())
    path.chmod(0o644)
    path.write_bytes(payload[:-1])
    path.chmod(0o444)
    with pytest.raises(ValueError, match="geometry"):
        read_packet_file(path)
    path.chmod(0o644)
    path.write_bytes(b"WRONG" + bytes(payload[5:]))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="magic"):
        read_packet_file(path)


def test_packet_reader_rejects_writable_symlink_and_hardlink(tmp_path: Path) -> None:
    path = tmp_path / "packet.bin"
    write_packet_file(path, sample_packet())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="single-link immutable"):
        read_packet_file(path)
    path.chmod(0o444)
    alias = tmp_path / "packet-alias.bin"
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="single-link immutable"):
        read_packet_file(path)
    alias.unlink()
    backing = tmp_path / "packet-backing.bin"
    path.rename(backing)
    path.symlink_to(backing.name)
    with pytest.raises(ValueError, match="single-link immutable"):
        read_packet_file(path)
