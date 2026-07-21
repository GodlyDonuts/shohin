"""OS-level board hiding for CTAA evaluator subprocesses."""

from __future__ import annotations

import platform
from pathlib import Path
import shutil


def hidden_board_command(
    command: list[str],
    *,
    writable_root: Path,
    board_root: Path,
) -> list[str]:
    """Return a command that cannot read any file in the sealed board root."""
    writable = writable_root.resolve()
    board = board_root.resolve()
    if writable == board or board in writable.parents or writable in board.parents:
        raise ValueError("CTAA sandbox roots overlap")
    system = platform.system()
    if system == "Linux":
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("CTAA physical custody requires bubblewrap on Linux")
        return [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            str(writable),
            str(writable),
            "--tmpfs",
            str(board),
            "--",
            *command,
        ]
    if system == "Darwin":
        sandbox = shutil.which("sandbox-exec")
        if sandbox is None:
            raise RuntimeError("CTAA physical custody requires sandbox-exec on macOS")
        escaped = str(board).replace('"', '\\"')
        policy = (
            '(version 1)(allow default)(deny file-read* file-write* '
            f'(subpath "{escaped}"))'
        )
        return [sandbox, "-p", policy, *command]
    raise RuntimeError(f"CTAA physical custody is unsupported on {system}")
