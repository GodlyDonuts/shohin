#!/usr/bin/env python3
"""Stage a minimal immutable source tree for confined EPISODE processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pipeline.episode_workspace_custody import (
    REPOSITORY_ROOT,
    abort_atomic_bundle,
    atomic_bundle_directory,
    file_sha256,
    finish_atomic_bundle,
    fsync_directory,
    write_json_fsync,
)


SOURCE_BUNDLE_SCHEMA = "shohin_episode_workspace_source_bundle_v1"
SOURCE_PATHS = (
    "pipeline/episode_action_binding_board.py",
    "pipeline/generate_episode_action_binding_corpus.py",
    "pipeline/episode_workspace_custody.py",
    "pipeline/assess_episode_causal_workspace.py",
    "pipeline/decide_episode_causal_workspace.py",
    "pipeline/stage_episode_workspace_sources.py",
    "pipeline/publish_episode_workspace_experiment.py",
    "train/model.py",
    "train/causal_bind_select_workspace.py",
    "train/workspace_checkpoint.py",
    "train/workspace_state_custody.py",
    "train/landlock_stage_exec.py",
    "train/fit_episode_causal_workspace.py",
    "train/compile_episode_causal_workspace.py",
    "train/execute_episode_causal_workspace.py",
    "train/jobs/episode_causal_workspace_pilot.sbatch",
)


class SourceStagingError(ValueError):
    """The committed source tree cannot be staged exactly."""


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceStagingError(f"git {' '.join(arguments)} failed") from exc


def stage_sources(expected_commit: str, output: Path) -> dict[str, object]:
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or _git("rev-parse", "HEAD").decode("ascii").strip() != expected_commit
    ):
        raise SourceStagingError("expected commit is not the checked-out HEAD")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *SOURCE_PATHS],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    staging, lock = atomic_bundle_directory(output)
    try:
        source_manifest: dict[str, str] = {}
        for relative in SOURCE_PATHS:
            committed = _git("show", f"{expected_commit}:{relative}")
            current = (REPOSITORY_ROOT / relative).read_bytes()
            if current != committed:
                raise SourceStagingError(f"working source differs for {relative}")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(committed)
                handle.flush()
                os.fsync(handle.fileno())
            source_manifest[relative] = hashlib.sha256(committed).hexdigest()
        receipt = {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "repository_commit": expected_commit,
            "source_manifest": source_manifest,
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        }
        receipt_path = staging / "source_receipt.json"
        write_json_fsync(receipt_path, receipt)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        finish_atomic_bundle(staging, output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    return {
        **receipt,
        "source_receipt_sha256": file_sha256(output / "source_receipt.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = stage_sources(args.expected_commit, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
