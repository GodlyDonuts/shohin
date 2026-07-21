from __future__ import annotations

from pathlib import Path

import pytest

import ctaa_process_sandbox as sandbox


def test_linux_sandbox_masks_board_and_keeps_output_writable(tmp_path, monkeypatch) -> None:
    board = tmp_path / "board"
    output = tmp_path / "output"
    board.mkdir()
    output.mkdir()
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/bwrap")
    command = sandbox.hidden_board_command(
        ["python3", "stage.py"],
        writable_root=output,
        board_root=board,
    )
    assert command[0] == "/usr/bin/bwrap"
    assert command[command.index("--tmpfs") + 1] == str(board.resolve())
    assert command[command.index("--bind") + 1] == str(output.resolve())
    assert command[-2:] == ["python3", "stage.py"]


def test_sandbox_rejects_overlapping_roots(tmp_path: Path) -> None:
    board = tmp_path / "board"
    output = board / "output"
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="overlap"):
        sandbox.hidden_board_command(
            ["true"],
            writable_root=output,
            board_root=board,
        )
