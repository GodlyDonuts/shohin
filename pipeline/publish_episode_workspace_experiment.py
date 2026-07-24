#!/usr/bin/env python3
"""Validate and atomically publish one complete EPISODE reasoning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat

from pipeline.episode_workspace_custody import (
    EpisodeCustodyError,
    file_sha256,
    fsync_directory,
    publish_directory_noreplace,
    read_json_verified,
    write_json_fsync,
)


EXPERIMENT_SCHEMA = "shohin_episode_causal_workspace_experiment_v1"
DECISION_SCHEMA = "episode_causal_workspace_decision_v1"
EXECUTION_FILES = frozenset(
    {
        "label_blind_predictions.jsonl",
        "execution_report.json",
        "answer_logits_treatment.pt",
        "answer_logits_zero_workspace.pt",
        "answer_logits_uniform_binding.pt",
        "answer_logits_uniform_operator.pt",
        "answer_logits_binding_permutation.pt",
        "answer_logits_operator_permutation.pt",
        "answer_logits_selected_slot_scramble.pt",
        "answer_logits_discarded_slot_scramble.pt",
    }
)
EXPECTED_STAGE_BUNDLES = {
    "true_fit": (
        "episode_causal_workspace_fit_bundle_v1",
        frozenset({"fit_report.json", "workspace_delta.pt"}),
        ("fit_report.json", "episode_causal_workspace_fit_v1"),
    ),
    "shuffled_fit": (
        "episode_causal_workspace_fit_bundle_v1",
        frozenset({"fit_report.json", "workspace_delta.pt"}),
        ("fit_report.json", "episode_causal_workspace_fit_v1"),
    ),
    "true_compiled": (
        "episode_causal_workspace_compile_bundle_v1",
        frozenset({"compile_report.json", "compiled_states.pt"}),
        ("compile_report.json", "episode_causal_workspace_compile_v1"),
    ),
    "shuffled_compiled": (
        "episode_causal_workspace_compile_bundle_v1",
        frozenset({"compile_report.json", "compiled_states.pt"}),
        ("compile_report.json", "episode_causal_workspace_compile_v1"),
    ),
    "true_execution": (
        "episode_causal_workspace_execution_bundle_v1",
        EXECUTION_FILES,
        ("execution_report.json", "episode_causal_workspace_execution_v1"),
    ),
    "shuffled_execution": (
        "episode_causal_workspace_execution_bundle_v1",
        EXECUTION_FILES,
        ("execution_report.json", "episode_causal_workspace_execution_v1"),
    ),
    "true_assessment": (
        "episode_causal_workspace_assessment_bundle_v1",
        frozenset({"assessment_report.json", "assessed_predictions.jsonl"}),
        ("assessment_report.json", "episode_causal_workspace_assessment_v1"),
    ),
    "shuffled_assessment": (
        "episode_causal_workspace_assessment_bundle_v1",
        frozenset({"assessment_report.json", "assessed_predictions.jsonl"}),
        ("assessment_report.json", "episode_causal_workspace_assessment_v1"),
    ),
    "decision": (
        "episode_causal_workspace_decision_bundle_v1",
        frozenset({"decision.json"}),
        ("decision.json", DECISION_SCHEMA),
    ),
}
EXPECTED_STAGE_DIRECTORIES = frozenset(EXPECTED_STAGE_BUNDLES)


class ExperimentPublicationError(ValueError):
    """The hidden experiment is incomplete, mutable, or internally inconsistent."""


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ExperimentPublicationError(
            f"experiment staging root cannot be inspected: {path}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ExperimentPublicationError(
            "experiment staging root must be a non-symlink directory"
        )
    return metadata.st_dev, metadata.st_ino


def _validate_tree(staging: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(staging.rglob("*")):
        relative = str(path.relative_to(staging))
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ExperimentPublicationError(f"symlink is forbidden: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ExperimentPublicationError(
                f"non-regular or multiply linked file is forbidden: {relative}"
            )
        files[relative] = file_sha256(path)
    return files


def _validate_stage_manifests(
    staging: Path,
    tree_files: dict[str, str],
) -> None:
    for stage, (expected_schema, expected_files, report_spec) in sorted(
        EXPECTED_STAGE_BUNDLES.items()
    ):
        root = staging / stage
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ExperimentPublicationError(f"stage directory is missing: {stage}")
        manifest_path = root / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise ExperimentPublicationError(f"stage manifest is missing: {stage}")
        manifest_relative = str(manifest_path.relative_to(staging))
        expected_manifest_sha256 = tree_files.get(manifest_relative)
        if expected_manifest_sha256 is None:
            raise ExperimentPublicationError(f"stage manifest is unsealed: {stage}")
        try:
            manifest = read_json_verified(manifest_path, expected_manifest_sha256)
        except EpisodeCustodyError as exc:
            raise ExperimentPublicationError(str(exc)) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != expected_schema
            or manifest.get("pretraining_started") is not False
        ):
            raise ExperimentPublicationError(f"stage manifest flags differ: {stage}")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ExperimentPublicationError(f"stage file ledger is invalid: {stage}")
        expected_names = set(expected_files) | {"bundle_manifest.json"}
        actual_names = {path.name for path in root.iterdir()}
        if actual_names != expected_names:
            raise ExperimentPublicationError(f"stage file set differs: {stage}")
        for name, expected in files.items():
            relative = f"{stage}/{name}"
            if (
                not isinstance(name, str)
                or not isinstance(expected, str)
                or tree_files.get(relative) != expected
            ):
                raise ExperimentPublicationError(
                    f"stage artifact hash differs: {stage}/{name}"
                )
        report_name, report_schema = report_spec
        report_relative = f"{stage}/{report_name}"
        try:
            report = read_json_verified(
                root / report_name,
                tree_files[report_relative],
            )
        except (EpisodeCustodyError, KeyError) as exc:
            raise ExperimentPublicationError(
                f"stage report is unsealed: {stage}"
            ) from exc
        if (
            not isinstance(report, dict)
            or report.get("schema") != report_schema
            or report.get("pretraining_started") is not False
            or report.get("continuation_pretraining_authorized") is not False
        ):
            raise ExperimentPublicationError(
                f"stage report flags differ: {stage}"
            )


def publish_experiment(
    staging: Path,
    output: Path,
    *,
    expected_commit: str,
    expected_source_receipt_sha256: str,
    expected_decision_sha256: str,
    slurm_job_id: str,
    slurm_node: str,
) -> dict[str, object]:
    staging = staging.absolute()
    output = output.absolute()
    staging_identity = _directory_identity(staging)
    if staging.parent != output.parent or not staging.name.startswith("."):
        raise ExperimentPublicationError(
            "staging must be a hidden sibling of the final output"
        )
    if output.exists() or output.is_symlink():
        raise ExperimentPublicationError("final experiment path already exists")
    if {path.name for path in staging.iterdir()} != {
        "source",
        *EXPECTED_STAGE_DIRECTORIES,
    }:
        raise ExperimentPublicationError("hidden experiment top-level set differs")
    source_receipt_path = staging / "source" / "source_receipt.json"
    source_receipt = read_json_verified(
        source_receipt_path,
        expected_source_receipt_sha256,
    )
    if (
        not isinstance(source_receipt, dict)
        or source_receipt.get("repository_commit") != expected_commit
        or source_receipt.get("pretraining_started") is not False
        or source_receipt.get("continuation_pretraining_authorized") is not False
    ):
        raise ExperimentPublicationError("source receipt differs")
    source_manifest = source_receipt.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise ExperimentPublicationError("source file manifest is missing")
    actual_source_files = {
        str(path.relative_to(staging / "source"))
        for path in (staging / "source").rglob("*")
        if path.is_file() and path != source_receipt_path
    }
    if actual_source_files != set(source_manifest):
        raise ExperimentPublicationError("staged source file set differs")
    existing_files = _validate_tree(staging)
    if (
        existing_files.get("source/source_receipt.json")
        != expected_source_receipt_sha256
    ):
        raise ExperimentPublicationError("source receipt hash differs from invocation")
    for relative, expected in source_manifest.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or existing_files.get(f"source/{relative}") != expected
        ):
            raise ExperimentPublicationError(
                f"staged source hash differs: {relative}"
            )
    _validate_stage_manifests(staging, existing_files)
    decision_path = staging / "decision" / "decision.json"
    if existing_files.get("decision/decision.json") != expected_decision_sha256:
        raise ExperimentPublicationError("decision hash differs from invocation")
    decision = read_json_verified(decision_path, expected_decision_sha256)
    if (
        not isinstance(decision, dict)
        or decision.get("schema") != DECISION_SCHEMA
        or decision.get("pretraining_started") is not False
        or decision.get("continuation_pretraining_authorized") is not False
        or decision.get("reasoning_promotion_authorized") is not False
    ):
        raise ExperimentPublicationError("decision exceeds the bounded tuning scope")
    if _validate_tree(staging) != existing_files:
        raise ExperimentPublicationError("experiment files changed during validation")
    manifest = {
        "schema": EXPERIMENT_SCHEMA,
        "repository_commit": expected_commit,
        "slurm_job_id": slurm_job_id,
        "slurm_node": slurm_node,
        "files": existing_files,
        "decision_sha256": expected_decision_sha256,
        "reasoning_promotion_authorized": False,
        "continuation_pretraining_authorized": False,
        "pretraining_started": False,
    }
    manifest_path = staging / "experiment_manifest.json"
    write_json_fsync(manifest_path, manifest)
    fsync_directory(staging)
    published_files = {
        **existing_files,
        "experiment_manifest.json": file_sha256(manifest_path),
    }
    if _directory_identity(staging) != staging_identity:
        raise ExperimentPublicationError(
            "experiment staging root changed during validation"
        )
    if _validate_tree(staging) != published_files:
        raise ExperimentPublicationError(
            "experiment files changed before publication"
        )
    _validate_stage_manifests(staging, published_files)
    final_decision = read_json_verified(decision_path, expected_decision_sha256)
    if final_decision != decision:
        raise ExperimentPublicationError("decision changed before publication")
    publish_directory_noreplace(staging, output)
    return {
        **manifest,
        "experiment_manifest_sha256": file_sha256(
            output / "experiment_manifest.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--slurm-node", required=True)
    args = parser.parse_args()
    report = publish_experiment(
        args.staging,
        args.output,
        expected_commit=args.expected_commit,
        expected_source_receipt_sha256=args.expected_source_receipt_sha256,
        expected_decision_sha256=args.expected_decision_sha256,
        slurm_job_id=args.slurm_job_id,
        slurm_node=args.slurm_node,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
