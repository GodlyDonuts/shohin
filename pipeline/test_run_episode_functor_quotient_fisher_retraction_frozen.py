from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import pipeline.run_episode_functor_quotient_fisher_retraction_frozen as frozen
from pipeline.run_episode_functor_quotient_fisher_retraction_frozen import (
    FrozenBootstrapError,
    _absolute_output,
    _absolute_regular_input,
)


def test_bootstrap_requires_absolute_regular_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}\n", encoding="ascii")
    assert _absolute_regular_input(snapshot, "snapshot") == snapshot
    with pytest.raises(FrozenBootstrapError):
        _absolute_regular_input(
            Path("snapshot.json"),
            "snapshot",
        )


def test_bootstrap_rejects_symlink_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}\n", encoding="ascii")
    link = tmp_path / "link.json"
    link.symlink_to(snapshot)
    with pytest.raises(FrozenBootstrapError):
        _absolute_regular_input(link, "snapshot")


def test_bootstrap_requires_fresh_absolute_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    assert _absolute_output(output) == output
    output.write_text("{}\n", encoding="ascii")
    with pytest.raises(FrozenBootstrapError):
        _absolute_output(output)
    with pytest.raises(FrozenBootstrapError):
        _absolute_output(Path("report.json"))


def test_committed_blob_bootstrap_executes_detached_worktree(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(
        ("git", "init", "--bare", str(remote)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ("git", "init", "-b", "main", str(repository)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    auditor = (
        repository
        / "pipeline/audit_episode_functor_quotient_fisher_retraction.py"
    )
    auditor.parent.mkdir(parents=True)
    auditor.write_text(
        "\n".join(
            (
                "import argparse",
                "from pathlib import Path",
                "import subprocess",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--beacon-snapshot', required=True)",
                "args = parser.parse_args()",
                "branch = subprocess.check_output(",
                "    ('git', 'branch', '--show-current'), text=True",
                ").strip()",
                "if branch:",
                "    raise SystemExit('child was not detached')",
                "Path(args.output).write_text('detached\\n', encoding='ascii')",
                "",
            )
        ),
        encoding="ascii",
    )
    committed_bootstrap = (
        repository
        / "pipeline/run_episode_functor_quotient_fisher_retraction_frozen.py"
    )
    committed_bootstrap.write_text(
        Path(frozen.__file__).read_text("ascii"),
        encoding="ascii",
    )
    subprocess.run(
        ("git", "add", "."),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=QFCR Test",
            "-c",
            "user.email=qfcr@example.invalid",
            "commit",
            "-m",
            "fixture",
        ),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ("git", "remote", "add", "origin", str(remote)),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "push", "-u", "origin", "main"),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}\n", encoding="ascii")
    output = tmp_path / "report.json"
    source = subprocess.check_output(
        (
            "git",
            "show",
            "HEAD:pipeline/"
            "run_episode_functor_quotient_fisher_retraction_frozen.py",
        ),
        cwd=repository,
        text=True,
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-",
            "--repository",
            str(repository),
            "--output",
            str(output),
            "--beacon-snapshot",
            str(snapshot),
        ),
        input=source,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_text("ascii") == "detached\n"
    worktrees = subprocess.check_output(
        ("git", "worktree", "list", "--porcelain"),
        cwd=repository,
        text=True,
    )
    assert worktrees.count("worktree ") == 1
