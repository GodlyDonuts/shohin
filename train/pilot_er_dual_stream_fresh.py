#!/usr/bin/env python3
"""Fit matched ordinal-route arms and consume one fresh development read."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

import torch

from build_er_relation_tensor_board import DEVELOPMENT_SPLIT, PROTOCOL, TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, load_trainable_state, trainable_state
from er_dual_stream_fresh_scoring import (
    SCORING_ARMS,
    evaluate_fresh_treatment,
    evaluate_source_free,
    fit_fresh_arm,
)
from er_relation_tensor_training import (
    TRAINING_CONTRACT,
    evaluate_arm,
    load_board_receipt,
    load_split,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_dual_stream_relation_adapter import (
    EXPECTED_PARAMETERS,
    initialize_dual_stream_relation,
)
from pilot_er_relation_tensor import (
    FROZEN_SOURCE_PATHS as ER_TT_SOURCE_PATHS,
    atomic_json_save,
    atomic_torch_save,
    release_cuda,
    runtime_manifest,
)
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_renderer_native_program import frozen_state_digest


CHECKPOINT_SCHEMA = "r12_er_dual_stream_fresh_checkpoint_v1"
EVIDENCE_SCHEMA = "r12_er_dual_stream_fresh_development_evidence_v1"
REPORT_SCHEMA = "r12_er_dual_stream_fresh_development_report_v1"
ACCESS_SCHEMA = "r12_er_dual_stream_fresh_development_access_v1"
BOARD_VARIANT = "ordinal_route_neutral_distractor_v1"
BOARD_SOURCE_COMMIT = "627b6c3d97a885017041aacb5971874680e1b289"
BOARD_REPORT_SHA256 = (
    "6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f"
)
CANARY_SOURCE_COMMIT = "c0d3d37218b0c65ba6c1baf7722d7381ee3be92e"
CANARY_SEED = 1790361034717866861
CANARY_CHECKPOINT_SHA256 = (
    "99be7b89e0b7dfe35f745abf1320c6640ad61f2fb62624b288fb8f9502cd97e7"
)
CANARY_SCHEMA = "r12_er_dual_stream_train_only_canary_v1_2"
THRESHOLDS = {
    "packet_overall": 0.95,
    "state_overall": 0.95,
    "answer_overall": 0.95,
    "joint_overall": 0.95,
    "field_overall": 0.95,
    "pointer_overall": 0.95,
    "joint_min_cardinality": 0.90,
    "joint_min_depth": 0.90,
    "joint_min_renderer": 0.90,
    "joint_non_bijective": 0.95,
    "treatment_advantage": 0.50,
    "negative_max": 0.35,
    "source_free_joint_max": 0.10,
    "invariance_exact": 1.0,
}
FROZEN_SOURCE_PATHS = tuple(
    sorted(
        set(
            ER_TT_SOURCE_PATHS
            + (
                "R12_ER_DUAL_STREAM_RELATION_REPAIR_PREREG.md",
                "R12_ER_DUAL_STREAM_ORDINAL_ROUTE_PREREG.md",
                "R12_ER_DUAL_STREAM_TRAIN_CANARY_RESULT.md",
                "R12_ER_DUAL_STREAM_FRESH_BOARD_PREREG.md",
                "R12_ER_DUAL_STREAM_FRESH_BOARD_RECEIPT.md",
                "R12_ER_DUAL_STREAM_FRESH_SCORE_PREREG.md",
                "train/er_dual_stream_relation_adapter.py",
                "train/pilot_er_dual_stream_relation_adapter.py",
                "train/pilot_er_dual_stream_train_canary.py",
                "train/er_dual_stream_fresh_scoring.py",
                "train/pilot_er_dual_stream_fresh.py",
                "train/assess_er_dual_stream_fresh.py",
                "train/test_er_dual_stream_fresh_scoring.py",
                "train/test_pilot_er_dual_stream_fresh.py",
                "train/test_assess_er_dual_stream_fresh.py",
                "train/jobs/er_dual_stream_fresh.sbatch",
            )
        )
    )
)


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("fresh dual-stream scientific source is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("fresh dual-stream source is not an ancestor")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"fresh dual-stream source omits path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"fresh dual-stream runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def consume_development_access(
    data_dir: Path, board: Mapping[str, object], source_commit: str
) -> dict[str, str]:
    split_sha = str(board["files"]["development.jsonl"]["sha256"])
    payload = (
        json.dumps(
            {
                "schema": ACCESS_SCHEMA,
                "protocol": PROTOCOL,
                "board_variant": BOARD_VARIANT,
                "split": DEVELOPMENT_SPLIT,
                "board_report_sha256": BOARD_REPORT_SHA256,
                "split_sha256": split_sha,
                "scientific_source_commit": source_commit,
                "board_source_commit": BOARD_SOURCE_COMMIT,
                "access_number": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    directory = data_dir / "access"
    directory.mkdir(exist_ok=True)
    path = directory / f"er_dual_stream_fresh_development_{split_sha}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _load_canary(path: Path) -> dict[str, object]:
    if sha256_file(path) != CANARY_CHECKPOINT_SHA256:
        raise RuntimeError("fresh dual-stream canary checkpoint hash differs")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError("fresh dual-stream canary checkpoint is not a mapping")
    if (
        value.get("schema") != CANARY_SCHEMA
        or value.get("seed") != CANARY_SEED
        or value.get("parameters") != EXPECTED_PARAMETERS
        or value.get("development_accesses") != 0
        or value.get("confirmation_accesses") != 0
        or value.get("source_manifest", {}).get("commit") != CANARY_SOURCE_COMMIT
        or not isinstance(value.get("compiler_trainable_state"), Mapping)
    ):
        raise RuntimeError("fresh dual-stream canary identity differs")
    return value


def initialize_system(
    args: argparse.Namespace,
    device: torch.device,
    canary: Mapping[str, object],
) -> tuple[torch.nn.Module, dict[str, int], str, dict[str, object]]:
    model, parameters, frozen_digest, receipt = initialize_dual_stream_relation(
        joint_checkpoint=args.joint_checkpoint,
        physical_checkpoint=args.physical_checkpoint,
        v1_checkpoint=args.v1_checkpoint,
        v1_2_checkpoint=args.v1_2_checkpoint,
        confirmed_checkpoint=args.confirmed_checkpoint,
        confirmation_assessment=args.confirmation_assessment,
        witness_checkpoint=args.witness_checkpoint,
        witness_confirmation_assessment=args.witness_confirmation_assessment,
        seed=CANARY_SEED,
        device=device,
    )
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError("fresh dual-stream parameter certificate differs")
    trainable = canary["compiler_trainable_state"]
    if not isinstance(trainable, Mapping):
        raise RuntimeError("fresh dual-stream canary trainable state is absent")
    load_trainable_state(model, trainable)  # type: ignore[arg-type]
    loaded_digest = state_dict_digest(trainable_state(model))
    expected_digest = state_dict_digest(dict(trainable))  # type: ignore[arg-type]
    if loaded_digest != expected_digest:
        raise RuntimeError("fresh dual-stream canary state did not load exactly")
    if frozen_digest != canary.get("fit", {}).get("frozen_digest"):
        raise RuntimeError("fresh dual-stream excluded parent differs from canary")
    output_receipt = {
        **receipt,
        "qualified_canary_checkpoint_sha256": CANARY_CHECKPOINT_SHA256,
        "qualified_canary_source_commit": CANARY_SOURCE_COMMIT,
        "qualified_canary_seed": CANARY_SEED,
        "qualified_canary_trainable_state_sha256": loaded_digest,
    }
    return model, parameters, frozen_digest, output_receipt


def _minimum_group(metrics: Mapping[str, object], group: str, field: str) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise RuntimeError(f"fresh dual-stream {group} metrics are absent")
    return min(float(value[field]["rate"]) for value in values.values())


def compute_gates(
    metrics: Mapping[str, Mapping[str, object]], checkpoint: Mapping[str, object]
) -> dict[str, bool]:
    treatment = metrics["treatment"]
    controls = [metrics["family_deranged"], metrics["equality_ablated"]]
    overall = treatment["overall"]
    invariance = treatment["invariance"]
    interventions = treatment["interventions"]
    return {
        "treatment_packet_state_answer_joint_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS[f"{name}_overall"]
            for name in ("packet", "state", "answer", "joint")
        ),
        "all_semantic_fields_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["field_overall"]
            for name in (
                "cardinality",
                "initial_rows",
                "relation_rows",
                "rule_active",
                "events",
                "halt",
                "query",
            )
        ),
        "all_active_pointers_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["pointer_overall"]
            for name in (
                "line_pointer",
                "binding_pointer",
                "initial_pointer",
                "witness_pointer",
                "query_pointer",
            )
        ),
        "minimum_cardinality_joint_at_least_90pct": _minimum_group(
            treatment, "by_cardinality", "joint"
        )
        >= THRESHOLDS["joint_min_cardinality"],
        "minimum_depth_joint_at_least_90pct": _minimum_group(
            treatment, "by_depth", "joint"
        )
        >= THRESHOLDS["joint_min_depth"],
        "minimum_renderer_joint_at_least_90pct": _minimum_group(
            treatment, "by_renderer", "joint"
        )
        >= THRESHOLDS["joint_min_renderer"],
        "non_bijective_joint_at_least_95pct": float(
            treatment["non_bijective"]["joint"]["rate"]
        )
        >= THRESHOLDS["joint_non_bijective"],
        "treatment_packet_and_joint_advantage_at_least_50pp": all(
            float(overall[name]["rate"])
            - float(control["overall"][name]["rate"])
            >= THRESHOLDS["treatment_advantage"]
            for control in controls
            for name in ("packet", "joint")
        ),
        "negative_packet_and_joint_at_most_35pct": all(
            float(control["overall"][name]["rate"]) <= THRESHOLDS["negative_max"]
            for control in controls
            for name in ("packet", "joint")
        ),
        "source_free_joint_at_most_10pct": float(
            metrics["source_free"]["overall"]["joint"]["rate"]
        )
        <= THRESHOLDS["source_free_joint_max"],
        "all_alpha_distractor_and_storage_invariances_exact": all(
            int(value["exact"]) == int(value["rows"]) == 2_048
            and float(value["rate"]) == THRESHOLDS["invariance_exact"]
            for value in invariance.values()
        ),
        "all_causal_interventions_exact_and_effective": all(
            int(value["eligible"]) > 0
            and int(value["exact_on_eligible"]) == int(value["eligible"])
            and int(value["sensitive"]) > 0
            and int(value["changed_on_sensitive"]) == int(value["sensitive"])
            for value in interventions.values()
        ),
        "all_arms_share_qualified_canary_initialization": all(
            arm["initial_state_sha256"]
            == checkpoint["shared_initial_state_sha256"]
            for arm in checkpoint["arms"].values()
        )
        and checkpoint["qualified_canary_checkpoint_sha256"]
        == CANARY_CHECKPOINT_SHA256,
        "confirmed_parent_unchanged": all(
            arm["fit"]["frozen_parent_unchanged"] is True
            for arm in checkpoint["arms"].values()
        ),
        "parameter_certificate_exact_and_below_200m": checkpoint["parameters"]
        == EXPECTED_PARAMETERS,
        "motor_and_reader_are_parameter_free": all(
            arm["fit"]["motor_parameters"] == 0
            and arm["fit"]["reader_parameters"] == 0
            for arm in checkpoint["arms"].values()
        ),
        "outcome_supervision_zero_and_development_one_confirmation_zero": TRAINING_CONTRACT[
            "outcome_supervision"
        ]
        is False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--canary-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmed-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-assessment", type=Path, required=True)
    parser.add_argument("--witness-checkpoint", type=Path, required=True)
    parser.add_argument("--witness-confirmation-assessment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing existing fresh dual-stream output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("fresh dual-stream qualification requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if (
        board.get("report_sha256") != BOARD_REPORT_SHA256
        or board.get("board_variant") != BOARD_VARIANT
    ):
        raise SystemExit("fresh dual-stream board identity differs")
    canary = _load_canary(args.canary_checkpoint)
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda")
    fit_seed = derived_seed(args.seed, "fresh-shared-fit-order-and-control")
    arms: dict[str, dict[str, object]] = {}
    parameters: dict[str, int] | None = None
    parent_receipt: dict[str, object] | None = None
    initial_digest: str | None = None

    for arm_name in SCORING_ARMS:
        model, arm_parameters, frozen_digest, receipt = initialize_system(
            args, device, canary
        )
        current_initial = state_dict_digest(trainable_state(model))
        if initial_digest is None:
            initial_digest = current_initial
            parameters = arm_parameters
            parent_receipt = receipt
        elif (
            current_initial != initial_digest
            or arm_parameters != parameters
            or receipt != parent_receipt
        ):
            raise RuntimeError("fresh dual-stream arms differ at initialization")
        declared = frozenset(receipt["trainable_names"])
        fit = fit_fresh_arm(
            model,
            train_rows,
            seed=fit_seed,
            arm=arm_name,
            frozen_digest=frozen_digest,
            digest_fn=lambda value, names=declared: frozen_state_digest(value, names),
        )
        arms[arm_name] = {
            "fit": fit,
            "initial_state_sha256": current_initial,
            "compiler_trainable_state": trainable_state(model),
        }
        release_cuda(model)

    if parameters is None or parent_receipt is None or initial_digest is None:
        raise RuntimeError("fresh dual-stream produced no fitted arms")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "board_variant": BOARD_VARIANT,
        "scientific_source_commit": args.source_commit,
        "source_manifest": source,
        "board_source_commit": BOARD_SOURCE_COMMIT,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "board_files": board["files"],
        "qualified_canary_checkpoint_sha256": CANARY_CHECKPOINT_SHA256,
        "qualified_canary_source_commit": CANARY_SOURCE_COMMIT,
        "qualified_canary_seed": CANARY_SEED,
        "training_seed": args.seed,
        "shared_initial_state_sha256": initial_digest,
        "training_contract": TRAINING_CONTRACT,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "arms": arms,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    checkpoint_path = args.out_dir / "compiler.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    ledger = consume_development_access(args.data_dir, board, args.source_commit)
    development_rows = load_split(
        args.data_dir,
        board,
        filename="development.jsonl",
        split=DEVELOPMENT_SPLIT,
        expected=2_048,
    )
    metrics: dict[str, Mapping[str, object]] = {}
    raw_arms: dict[str, object] = {}
    invariant_raw: dict[str, object] | None = None
    for arm_name in SCORING_ARMS:
        model, arm_parameters, _, receipt = initialize_system(args, device, canary)
        if arm_parameters != parameters or receipt != parent_receipt:
            raise RuntimeError("fresh dual-stream evaluation reconstruction differs")
        load_trainable_state(model, arms[arm_name]["compiler_trainable_state"])
        if arm_name == "treatment":
            treatment, raw, invariant_raw = evaluate_fresh_treatment(
                model, development_rows, batch_size=args.batch_size
            )
            source_free, source_free_raw = evaluate_source_free(
                model, development_rows, batch_size=args.batch_size
            )
            metrics[arm_name] = treatment
            metrics["source_free"] = source_free
            raw_arms[arm_name] = raw
            raw_arms["source_free"] = source_free_raw
        else:
            result = evaluate_arm(
                model,
                development_rows,
                batch_size=args.batch_size,
                include_raw=True,
                include_invariances=False,
            )
            raw = result.pop("raw")
            metrics[arm_name] = result
            raw_arms[arm_name] = raw
        release_cuda(model)
    if invariant_raw is None:
        raise RuntimeError("fresh dual-stream invariance evidence is absent")

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": PROTOCOL,
        "board_variant": BOARD_VARIANT,
        "scientific_source_commit": args.source_commit,
        "checkpoint_sha256": checkpoint_sha,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "development_sha256": board["files"]["development.jsonl"]["sha256"],
        "arms": raw_arms,
        "invariance": invariant_raw,
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    evidence_path = args.out_dir / "development_evidence.pt"
    atomic_torch_save(evidence, evidence_path)
    evidence_sha = sha256_file(evidence_path)
    gates = compute_gates(metrics, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "board_variant": BOARD_VARIANT,
        "scientific_source_commit": args.source_commit,
        "runtime": runtime_manifest(),
        "training_seed": args.seed,
        "training_contract": TRAINING_CONTRACT,
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_one_sealed_confirmation"
            if all(gates.values())
            else "reject_er_dual_stream_fresh_v1"
        ),
        "artifacts": {
            "checkpoint_sha256": checkpoint_sha,
            "evidence_sha256": evidence_sha,
            "development_ledger_sha256": ledger["sha256"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "development_ledger": ledger,
        },
        "claim_boundary": (
            "Passing establishes bounded fresh neutral-namespace episodic relation "
            "compilation, source-deleted recurrent composition, halt, and query readout "
            "under irrelevant names. It does not establish free-form grounding, arbitrary "
            "program induction, arithmetic, planning, or general reasoning."
        ),
    }
    report_path = args.out_dir / "development_report.json"
    atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checkpoint_sha256": checkpoint_sha,
                "evidence_sha256": evidence_sha,
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )
    gc.collect()


if __name__ == "__main__":
    main()
