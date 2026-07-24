#!/usr/bin/env python3
"""Combine true/shuffled EPISODE workspace arms into one bounded decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pipeline.episode_workspace_custody as custody_module
from pipeline.episode_workspace_custody import (
    abort_atomic_bundle,
    atomic_bundle_directory,
    committed_source_receipt,
    EpisodeCustodyError,
    file_sha256,
    finish_atomic_bundle,
    fsync_directory,
    json_sha256,
    read_json_verified,
    verify_landlock_stage,
    write_json_fsync,
)


DECISION_SCHEMA = "episode_causal_workspace_decision_v1"
DECISION_BUNDLE_SCHEMA = "episode_causal_workspace_decision_bundle_v1"
EXPECTED_ARM_INPUTS = {
    "true": {
        "name": "train_true_groups.jsonl",
        "sha256": "80d7e6e503d4aebbda506fcb3d321f1a91db556f7fe1bd2ab8e6ee92d2fbec27",
    },
    "shuffled": {
        "name": "train_shuffled_groups.jsonl",
        "sha256": "5917049c910cdc2beae667165465052c71329033be30532a4cec5a04fe419038",
    },
}
LANDLOCK_POLICY_SCHEMA = "shohin_landlock_stage_policy_v1"
LANDLOCK_READ_FILE = 1 << 2
CUSTODY_FILE_NAMES = frozenset(
    {
        "custody_manifest.json",
        "train_true_groups.jsonl",
        "train_shuffled_groups.jsonl",
        "development_worlds.jsonl",
        "development_queries.jsonl",
        "development_assessor.jsonl",
    }
)


class EpisodeWorkspaceDecisionError(ValueError):
    """A cross-arm receipt or decision invariant failed."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _landlock_pass(
    value: object,
    *,
    stage: str,
    process_id: object,
    denied_path: Path,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "stage",
        "enforced",
        "dumpable",
        "abi",
        "policy_sha256",
        "canonical_policy",
        "process_id",
        "denied_probe_receipt",
    }:
        return False
    abi = value.get("abi")
    policy_sha256 = value.get("policy_sha256")
    canonical_policy = value.get("canonical_policy")
    if (
        value.get("schema") != "shohin_landlock_stage_receipt_v1"
        or value.get("stage") != stage
        or value.get("enforced") is not True
        or value.get("dumpable") is not False
        or value.get("process_id") != process_id
        or not isinstance(abi, int)
        or isinstance(abi, bool)
        or not 3 <= abi <= 10
        or not _is_sha256(policy_sha256)
        or not _canonical_policy_pass(
            canonical_policy,
            stage=stage,
            abi=abi,
            policy_sha256=str(policy_sha256),
        )
    ):
        return False
    denied = value.get("denied_probe_receipt")
    if not isinstance(denied, dict) or set(denied) != {
        "schema",
        "stage",
        "process_id",
        "operation",
        "path",
        "path_name",
        "path_sha256",
        "denied",
        "errno",
    }:
        return False
    denied_path_text = str(denied_path)
    denied_errno = denied.get("errno")
    return bool(
        denied.get("schema") == "shohin_landlock_denied_probe_receipt_v1"
        and denied.get("stage") == stage
        and denied.get("process_id") == process_id
        and denied.get("operation") == "open_read"
        and denied.get("path") == denied_path_text
        and Path(denied_path_text).is_absolute()
        and denied.get("path_name") == denied_path.name
        and denied.get("path_sha256")
        == hashlib.sha256(os.fsencode(denied_path_text)).hexdigest()
        and denied.get("denied") is True
        and isinstance(denied_errno, int)
        and not isinstance(denied_errno, bool)
        and denied_errno in {1, 13}
    )


def _canonical_policy_pass(
    value: object,
    *,
    stage: str,
    abi: int,
    policy_sha256: str,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "landlock_abi",
        "stage",
        "handled_access_fs",
        "handled_access_fs_names",
        "rules",
    }:
        return False
    handled = value.get("handled_access_fs")
    names = value.get("handled_access_fs_names")
    rules = value.get("rules")
    if (
        value.get("schema") != LANDLOCK_POLICY_SCHEMA
        or value.get("landlock_abi") != abi
        or value.get("stage") != stage
        or not isinstance(handled, int)
        or isinstance(handled, bool)
        or handled <= 0
        or not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or not isinstance(rules, list)
        or json_sha256(value) != policy_sha256
    ):
        return False
    previous = ""
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {
            "path",
            "object_type",
            "st_dev",
            "st_ino",
            "allowed_access_fs",
            "allowed_access_fs_names",
        }:
            return False
        path = rule.get("path")
        access = rule.get("allowed_access_fs")
        access_names = rule.get("allowed_access_fs_names")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or os.path.normpath(path) != path
            or path <= previous
            or rule.get("object_type") not in {"file", "directory"}
            or not isinstance(rule.get("st_dev"), int)
            or isinstance(rule.get("st_dev"), bool)
            or int(rule["st_dev"]) < 0
            or not isinstance(rule.get("st_ino"), int)
            or isinstance(rule.get("st_ino"), bool)
            or int(rule["st_ino"]) <= 0
            or not isinstance(access, int)
            or isinstance(access, bool)
            or access <= 0
            or access & ~handled
            or not isinstance(access_names, list)
            or not all(isinstance(name, str) for name in access_names)
        ):
            return False
        previous = path
    return True


def _policy_grants_read(policy: object, target: Path) -> bool:
    if not isinstance(policy, dict) or not isinstance(policy.get("rules"), list):
        return True
    target_text = str(target)
    for rule in policy["rules"]:
        if not isinstance(rule, dict):
            return True
        access = rule.get("allowed_access_fs")
        path_text = rule.get("path")
        if (
            not isinstance(access, int)
            or not access & LANDLOCK_READ_FILE
            or not isinstance(path_text, str)
        ):
            continue
        rule_path = Path(path_text)
        if rule.get("object_type") == "file" and path_text == target_text:
            return True
        if rule.get("object_type") == "directory" and (
            target == rule_path or target.is_relative_to(rule_path)
        ):
            return True
    return False


def _absolute_report_path(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        return None
    return path


def _load_json_verified(
    path: Path,
    expected_sha256: str,
) -> dict[str, object]:
    try:
        value = read_json_verified(path, expected_sha256)
    except EpisodeCustodyError as exc:
        raise EpisodeWorkspaceDecisionError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EpisodeWorkspaceDecisionError(f"{path.name} is not an object")
    if value.get("pretraining_started") is not False:
        raise EpisodeWorkspaceDecisionError(
            f"{path.name} has an invalid pretraining flag"
        )
    return value


def _metric(container: dict[str, object], key: str) -> float:
    value = container.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("rate"), (int, float)):
        raise EpisodeWorkspaceDecisionError(f"metric {key} is missing")
    return float(value["rate"])


def _validate_fit_arm_binding(fit: dict[str, object], arm: str) -> None:
    expected = EXPECTED_ARM_INPUTS[arm]
    visible = fit.get("optimizer_visible_input")
    if (
        not isinstance(visible, dict)
        or visible.get("sha256") != expected["sha256"]
        or visible.get("frozen_arm_binding") != expected
        or Path(str(visible.get("path", ""))).name != expected["name"]
        or visible.get("rows") != 256
        or visible.get("packets") != 1_536
    ):
        raise EpisodeWorkspaceDecisionError(
            f"{arm} fit is not bound to its frozen optimization ledger"
        )


def _process_deletion_pass(
    fit: dict[str, object],
    assessment: dict[str, object],
) -> bool:
    execution = assessment.get("execution_report")
    if not isinstance(execution, dict):
        return False
    compiled = execution.get("compiled_state_receipt")
    if not isinstance(compiled, dict):
        return False
    compiler_source = compiled.get("compiler_source_receipt")
    if not isinstance(compiler_source, dict):
        return False
    process_ids = (
        fit.get("process_id"),
        compiler_source.get("process_id"),
        execution.get("process_id"),
        assessment.get("process_id"),
    )
    if any(not isinstance(value, int) or value <= 0 for value in process_ids):
        return False
    if len(set(process_ids)) != 4:
        return False
    landlock_receipts = (
        fit.get("landlock_receipt"),
        compiler_source.get("landlock_receipt"),
        execution.get("landlock_receipt"),
        assessment.get("landlock_receipt"),
    )
    visible_fit = fit.get("optimizer_visible_input")
    protected = fit.get("protected_checkpoint")
    executor_inputs = execution.get("executor_visible_inputs")
    if (
        not isinstance(visible_fit, dict)
        or not isinstance(protected, dict)
        or not isinstance(executor_inputs, dict)
    ):
        return False
    train_path = _absolute_report_path(visible_fit.get("path"))
    checkpoint_path = _absolute_report_path(protected.get("checkpoint_path"))
    queries = executor_inputs.get("queries")
    if not isinstance(queries, dict):
        return False
    query_path = _absolute_report_path(queries.get("path"))
    if (
        train_path is None
        or checkpoint_path is None
        or query_path is None
        or len(checkpoint_path.parents) < 3
    ):
        return False
    custody_root = train_path.parent
    repository_root = checkpoint_path.parents[2]
    denied_paths = (
        custody_root / "development_assessor.jsonl",
        query_path,
        custody_root / "development_worlds.jsonl",
        custody_root / "development_worlds.jsonl",
    )
    if not all(
        (
            _landlock_pass(
                landlock_receipts[0],
                stage="fit",
                process_id=process_ids[0],
                denied_path=denied_paths[0],
            ),
            _landlock_pass(
                landlock_receipts[1],
                stage="compiler",
                process_id=process_ids[1],
                denied_path=denied_paths[1],
            ),
            _landlock_pass(
                landlock_receipts[2],
                stage="executor",
                process_id=process_ids[2],
                denied_path=denied_paths[2],
            ),
            _landlock_pass(
                landlock_receipts[3],
                stage="assessor",
                process_id=process_ids[3],
                denied_path=denied_paths[3],
            ),
        )
    ):
        return False
    policies = {
        str(receipt["policy_sha256"])
        for receipt in landlock_receipts
        if isinstance(receipt, dict)
    }
    if len(policies) != 4:
        return False
    allowed_custody_names = (
        {train_path.name},
        {"development_worlds.jsonl"},
        {"development_queries.jsonl"},
        {"development_assessor.jsonl"},
    )
    protected_repository_paths = {
        repository_root / ".git",
        repository_root / ".env",
    }
    for receipt, allowed_names in zip(
        landlock_receipts,
        allowed_custody_names,
        strict=True,
    ):
        if not isinstance(receipt, dict):
            return False
        sensitive = {
            custody_root / name
            for name in CUSTODY_FILE_NAMES - allowed_names
        } | protected_repository_paths
        if any(
            _policy_grants_read(receipt.get("canonical_policy"), path)
            for path in sensitive
        ):
            return False
    if compiled.get("source_tokens_serialized") is not False:
        return False
    if compiled.get("query_tokens_seen") is not False:
        return False
    if compiled.get("labels_seen") is not False:
        return False
    if execution.get("world_tokens_seen") is not False:
        return False
    if execution.get("targets_seen") is not False:
        return False
    if execution.get("candidate_sets_seen") is not False:
        return False
    if assessment.get("assessor_source", {}).get("open_count") != 1:
        return False
    return fit.get("workspace_delta_sha256") == compiled.get("workspace_delta_sha256")


def build_decision(
    true_fit: dict[str, object],
    shuffled_fit: dict[str, object],
    true_assessment: dict[str, object],
    shuffled_assessment: dict[str, object],
) -> dict[str, object]:
    if true_fit.get("arm") != "true" or shuffled_fit.get("arm") != "shuffled":
        raise EpisodeWorkspaceDecisionError("fit arm labels are invalid")
    _validate_fit_arm_binding(true_fit, "true")
    _validate_fit_arm_binding(shuffled_fit, "shuffled")
    if true_fit.get("fit_config") != shuffled_fit.get("fit_config"):
        raise EpisodeWorkspaceDecisionError(
            "true and shuffled fit configurations differ"
        )
    if true_fit.get("workspace_initial_state_sha256") != shuffled_fit.get(
        "workspace_initial_state_sha256"
    ):
        raise EpisodeWorkspaceDecisionError(
            "true and shuffled arms did not share initialization"
        )
    if true_fit.get("protected_checkpoint") != shuffled_fit.get("protected_checkpoint"):
        raise EpisodeWorkspaceDecisionError(
            "true and shuffled arms reference different bases"
        )
    true_controls = true_assessment.get("control_summaries")
    shuffled_controls = shuffled_assessment.get("control_summaries")
    if not isinstance(true_controls, dict) or not isinstance(shuffled_controls, dict):
        raise EpisodeWorkspaceDecisionError("control summaries are missing")
    treatment = true_controls["treatment"]
    shuffled_treatment = shuffled_controls["treatment"]
    if not isinstance(treatment, dict) or not isinstance(shuffled_treatment, dict):
        raise EpisodeWorkspaceDecisionError("treatment summaries are invalid")
    by_depth = treatment.get("by_depth")
    if not isinstance(by_depth, dict):
        raise EpisodeWorkspaceDecisionError("depth metrics are missing")
    treatment_packet = _metric(treatment, "packets")
    selected_cost = treatment_packet - _metric(
        true_controls["selected_slot_scramble"],
        "packets",
    )
    discarded_cost = treatment_packet - _metric(
        true_controls["discarded_slot_scramble"],
        "packets",
    )
    depth_floor = min(_metric(by_depth, "5"), _metric(by_depth, "6"))
    process_deletion = _process_deletion_pass(
        true_fit,
        true_assessment,
    ) and _process_deletion_pass(shuffled_fit, shuffled_assessment)
    tuning_gates = {
        "packet_accuracy_at_least_90_percent": treatment_packet >= 0.90,
        "cyclic_triple_accuracy_at_least_90_percent": (
            _metric(treatment, "complete_cyclic_triples") >= 0.90
        ),
        "order_pair_accuracy_at_least_85_percent": (
            _metric(treatment, "complete_order_pairs") >= 0.85
        ),
        "depth_five_six_floor_at_least_80_percent": depth_floor >= 0.80,
        "selected_slot_scramble_cost_at_least_40_points": (selected_cost >= 0.40),
        "discarded_slot_scramble_cost_at_most_2_points": (abs(discarded_cost) <= 0.02),
        "shuffled_target_training_at_most_40_percent": (
            _metric(shuffled_treatment, "packets") <= 0.40
        ),
        "process_level_source_deletion": process_deletion,
    }
    all_tuning_gates_pass = all(tuning_gates.values())
    remaining_confirmation_gates = {
        "post_compile_source_poison_bit_identity": False,
        "three_fresh_seed_replication": False,
        "unopened_confirmation_manifest": False,
    }
    next_action = (
        "source_poison_and_two_replications"
        if all_tuning_gates_pass
        else "reject_architecture_on_inspected_tuning_board"
    )
    return {
        "schema": DECISION_SCHEMA,
        "claim_scope": (
            "inspected synthetic tuning decision only; not broad reasoning, "
            "language reasoning, or continuation pretraining"
        ),
        "true_treatment": treatment,
        "shuffled_treatment": shuffled_treatment,
        "tuning_gates": tuning_gates,
        "all_tuning_gates_pass": all_tuning_gates_pass,
        "remaining_confirmation_gates": remaining_confirmation_gates,
        "selected_slot_scramble_cost": selected_cost,
        "discarded_slot_scramble_cost": discarded_cost,
        "depth_five_six_floor": depth_floor,
        "next_action": next_action,
        "reasoning_promotion_authorized": False,
        "continuation_pretraining_authorized": False,
        "pretraining_started": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    landlock_receipt = verify_landlock_stage("decision", args.deny_probe)
    try:
        source_before = committed_source_receipt(
            Path(__file__).resolve(),
            args.expected_source_sha256,
            (Path(custody_module.__file__),),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceDecisionError(str(exc)) from exc
    true_fit = _load_json_verified(args.true_fit, args.true_fit_sha256)
    shuffled_fit = _load_json_verified(
        args.shuffled_fit,
        args.shuffled_fit_sha256,
    )
    true_assessment = _load_json_verified(
        args.true_assessment,
        args.true_assessment_sha256,
    )
    shuffled_assessment = _load_json_verified(
        args.shuffled_assessment,
        args.shuffled_assessment_sha256,
    )
    decision = {
        **build_decision(
            true_fit,
            shuffled_fit,
            true_assessment,
            shuffled_assessment,
        ),
        "source": source_before,
        "process_id": os.getpid(),
        "landlock_receipt": landlock_receipt,
        "inputs": {
            "true_fit": args.true_fit_sha256,
            "shuffled_fit": args.shuffled_fit_sha256,
            "true_assessment": args.true_assessment_sha256,
            "shuffled_assessment": args.shuffled_assessment_sha256,
        },
    }
    staging, lock = atomic_bundle_directory(args.output)
    try:
        decision_path = staging / "decision.json"
        write_json_fsync(decision_path, decision)
        manifest = {
            "schema": DECISION_BUNDLE_SCHEMA,
            "files": {"decision.json": file_sha256(decision_path)},
            "reasoning_promotion_authorized": False,
            "continuation_pretraining_authorized": False,
            "pretraining_started": False,
        }
        write_json_fsync(staging / "bundle_manifest.json", manifest)
        fsync_directory(staging)
        finish_atomic_bundle(staging, args.output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-fit", type=Path, required=True)
    parser.add_argument("--true-fit-sha256", required=True)
    parser.add_argument("--shuffled-fit", type=Path, required=True)
    parser.add_argument("--shuffled-fit-sha256", required=True)
    parser.add_argument("--true-assessment", type=Path, required=True)
    parser.add_argument("--true-assessment-sha256", required=True)
    parser.add_argument("--shuffled-assessment", type=Path, required=True)
    parser.add_argument("--shuffled-assessment-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = run(args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
