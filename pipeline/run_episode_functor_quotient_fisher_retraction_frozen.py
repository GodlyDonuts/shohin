#!/usr/bin/env python3
"""Execute the QFCR audit from a detached public Git object worktree.

Official use pipes this exact committed blob to isolated Python:

    git show HEAD:pipeline/run_episode_functor_quotient_fisher_retraction_frozen.py \
      | python3 -I - --repository /absolute/repo --output /absolute/report \
          --beacon-snapshot /absolute/snapshot

The bootstrap imports no Shohin code. It verifies public HEAD, creates a
detached worktree from that Git object, and only then imports outcome code in
a second isolated interpreter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


AUDITOR = "pipeline/audit_episode_functor_quotient_fisher_retraction.py"


class FrozenBootstrapError(RuntimeError):
    """The public source bootstrap contract failed closed."""


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repository,
        text=True,
    ).strip()


def _absolute_regular_input(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise FrozenBootstrapError(f"{label} must be absolute")
    if path.is_symlink():
        raise FrozenBootstrapError(
            f"{label} must be a nonsymlink regular file"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise FrozenBootstrapError(
            f"{label} must be a nonsymlink regular file"
        )
    return resolved


def _absolute_output(path: Path) -> Path:
    if not path.is_absolute():
        raise FrozenBootstrapError("output must be absolute")
    resolved_parent = path.parent.resolve()
    if not resolved_parent.is_dir() or path.exists():
        raise FrozenBootstrapError(
            "output parent differs or output already exists"
        )
    return resolved_parent / path.name


def main() -> None:
    if not sys.flags.isolated:
        raise FrozenBootstrapError(
            "bootstrap requires Python isolated mode"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--environment-output", type=Path)
    parser.add_argument(
        "--beacon-snapshot",
        type=Path,
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    if (
        not args.repository.is_absolute()
        or args.repository.is_symlink()
        or not repository.is_dir()
    ):
        raise FrozenBootstrapError(
            "repository must be an absolute nonsymlink directory"
        )
    environment_mode = args.environment_output is not None
    if environment_mode:
        if args.output is not None or args.beacon_snapshot is not None:
            raise FrozenBootstrapError(
                "environment mode cannot run an audit"
            )
        output = _absolute_output(args.environment_output)
        snapshot = None
    else:
        if args.output is None or args.beacon_snapshot is None:
            raise FrozenBootstrapError(
                "audit mode requires output and beacon snapshot"
            )
        output = _absolute_output(args.output)
        snapshot = _absolute_regular_input(
            args.beacon_snapshot,
            "beacon snapshot",
        )
    head = _git(repository, "rev-parse", "HEAD")
    remote = _git(
        repository,
        "ls-remote",
        "origin",
        "refs/heads/main",
    )
    if not remote or remote.split()[0] != head:
        raise FrozenBootstrapError(
            "HEAD is not public origin/main"
        )
    status = _git(repository, "status", "--porcelain")
    if status:
        raise FrozenBootstrapError(
            "source repository must be clean"
        )
    with tempfile.TemporaryDirectory(
        prefix="shohin-qfcr-public-"
    ) as temporary:
        worktree = Path(temporary) / "source"
        subprocess.run(
            (
                "git",
                "worktree",
                "add",
                "--detach",
                str(worktree),
                head,
            ),
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            child_arguments = (
                (
                    sys.executable,
                    "-I",
                    str(worktree / AUDITOR),
                    "--environment-output",
                    str(output),
                )
                if environment_mode
                else (
                    sys.executable,
                    "-I",
                    str(worktree / AUDITOR),
                    "--output",
                    str(output),
                    "--beacon-snapshot",
                    str(snapshot),
                )
            )
            child = subprocess.run(
                child_arguments,
                cwd=worktree,
                check=False,
            )
            if child.returncode != 0:
                raise FrozenBootstrapError(
                    "detached official audit failed"
                )
        finally:
            subprocess.run(
                (
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ),
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


if __name__ == "__main__":
    main()
