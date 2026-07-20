#!/usr/bin/env python3
"""Train, freeze gates, and one-read evaluate projected SD-CST fresh boards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping, Sequence

import torch

from assess_sd_cst_projected_mechanics import (
    packet_arm,
    rotate_queries,
    shuffled_packet,
    swap_operand_suffix,
)
from build_sd_cst_projected_board import (
    BOARD_SCHEMA,
    PARENT_BOARD_REPORT_SHA256,
    PARENT_BOARD_TRAIN_SHA256,
    PROTOCOL,
)
from projected_sd_cst_fresh import (
    COMPARISON_PARAMETER_CAP,
    CONSUMED_PROJECTED_SHA256,
    EXECUTION_CORE_SHA256,
    GLOBAL_PARAMETER_CAP,
    PARENT_SHA256,
    PROJECTED_TRAINABLE_NAMES,
    TRAINING_CONTRACT,
    ProjectedFreshRow,
    canonical_json,
    compile_fresh_rows,
    derived_seed,
    expected_tape,
    fit_projected_arm,
    frozen_parameter_digest,
    initialize_model,
    load_trainable_state,
    outputs_to_json,
    parse_projected_row,
    permute_training_labels,
    row_shuffled_permutation,
    row_shuffled_mapping_digest,
    sha256_file,
    state_dict_digest,
    tape_to_json,
    trainable_state,
)
from sd_cst import EVENT_STEPS, STOP_KIND, HardLateQuery, HardProgramTape


CHECKPOINT_SCHEMA = "r12_sd_cst_projected_fresh_checkpoint_v1"
CONFIG_SCHEMA = "r12_sd_cst_projected_fresh_gate_config_v1"
EVALUATION_SCHEMA = "r12_sd_cst_projected_fresh_evaluation_v1"
ASSESSMENT_SCHEMA = "r12_sd_cst_projected_fresh_assessment_v1"
TRAIN_SPLIT = "sd_cst_train"
DEVELOPMENT_SPLIT = "sd_cst_development"
CONFIRMATION_SPLIT = "sd_cst_confirmation"
FROZEN_SOURCE_PATHS = (
    "R12_SD_CST_PROJECTED_MECHANICS_PREREG.md",
    "R12_SD_CST_PROJECTED_BINDING_PILOT_PREREG.md",
    "R12_SD_CST_PROJECTED_FRESH_BOARD_PREREG.md",
    "pipeline/assess_sd_cst_projected_fresh.py",
    "pipeline/audit_sd_cst_board.py",
    "pipeline/build_sd_cst_board.py",
    "pipeline/build_sd_cst_projected_board.py",
    "pipeline/test_assess_sd_cst_projected_fresh.py",
    "pipeline/test_build_sd_cst_projected_board.py",
    "train/assess_sd_cst_projected_mechanics.py",
    "train/pilot_sd_cst_binding_bus.py",
    "train/pilot_sd_cst_byte_addressed.py",
    "train/pilot_sd_cst_hierarchical_binding.py",
    "train/projected_sd_cst_fresh.py",
    "train/run_sd_cst_hard_packets.py",
    "train/sd_cst.py",
    "train/sd_cst_binding_bus.py",
    "train/sd_cst_byte_addressed.py",
    "train/model.py",
    "train/train_sd_cst.py",
    "train/test_projected_sd_cst_fresh.py",
    "train/test_run_sd_cst_hard_packets.py",
    "train/train_eval_sd_cst_projected_fresh.py",
    "train/jobs/sd_cst_projected_fresh.sbatch",
)
THRESHOLDS = {
    "exact_packet_overall": 0.95,
    "exact_packet_min_variant": 0.90,
    "exact_packet_min_depth": 0.90,
    "field_overall": 0.98,
    "pointer_overall": 0.90,
    "pointer_min_variant": 0.80,
    "autonomous_overall": 0.90,
    "autonomous_min_variant": 0.90,
    "autonomous_min_depth": 0.85,
    "conditional_execution": 1.0,
    "treatment_advantage": 0.50,
    "pair_eligibility": 0.85,
    "paired_consistency": 1.0,
    "causal_oracle": 1.0,
    "causal_min_opportunity": 0.15,
    "query_min_opportunity": 0.85,
    "negative_state_max": 0.35,
    "negative_answer_max": 0.45,
    "reset_freeze_max": 0.75,
    "source_deletion": 1.0,
}
CONTROL_ARMS = (
    "uniform",
    "source_free_packet",
    "shuffled_packet",
    "reset",
    "freeze",
    "post_stop_perturbation",
    "force_alive_post_stop",
    "operand_suffix_swap",
    "query_rotation",
    "state_swap_after_step_0",
    "initial_state_rotation",
    "event_kind_flip",
    "event_identity_rotation",
    "event_amount_flip",
)


def runtime_manifest() -> dict[str, object]:
    value = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda)
        if torch.version.cuda is not None
        else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        value.update(
            {
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    value["sha256"] = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return value


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
    if resolved.returncode:
        raise RuntimeError("projected source commit is unavailable")
    commit = resolved.stdout.strip()
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise RuntimeError("projected source commit is not an ancestor of HEAD")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{commit}:{relative}").returncode:
            raise RuntimeError(f"source commit omits frozen path: {relative}")
        if git("diff", "--quiet", commit, "--", relative).returncode:
            raise RuntimeError(f"runtime bytes differ from source commit: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return value


def _ledger_bytes(
    *,
    split: str,
    board_sha: str,
    split_sha: str,
    source_commit: str,
) -> bytes:
    value = {
        "schema": "r12_sd_cst_projected_fresh_access_v1",
        "protocol": PROTOCOL,
        "split": split,
        "board_report_sha256": board_sha,
        "split_sha256": split_sha,
        "source_commit": source_commit,
        "access_number": 1,
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _consume_access(
    ledger_dir: Path,
    *,
    split: str,
    board_sha: str,
    split_sha: str,
    source_commit: str,
) -> dict[str, str]:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"projected_fresh_{split}_{split_sha}.json"
    encoded = _ledger_bytes(
        split=split,
        board_sha=board_sha,
        split_sha=split_sha,
        source_commit=source_commit,
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _read_board_report(data_dir: Path, source_commit: str) -> dict[str, object]:
    report = json.loads((data_dir / "report.json").read_text())
    if (
        report.get("schema") != BOARD_SCHEMA
        or report.get("protocol") != PROTOCOL
        or report.get("all_gates_pass") is not True
        or report.get("source_commit") != source_commit
        or int(report.get("development_accesses", -1)) != 0
        or int(report.get("confirmation_accesses", -1)) != 0
    ):
        raise RuntimeError("projected fresh board receipt is not admitted")
    return report


def load_rows(
    data_dir: Path,
    split: str,
    source_commit: str,
) -> tuple[list[ProjectedFreshRow], dict[str, object]]:
    report = _read_board_report(data_dir, source_commit)
    filenames = {
        TRAIN_SPLIT: "train.jsonl",
        DEVELOPMENT_SPLIT: "development.jsonl",
        CONFIRMATION_SPLIT: "confirmation.sealed.jsonl",
    }
    filename = filenames[split]
    expected = str(report["files"][filename]["sha256"])
    report_sha = sha256_file(data_dir / "report.json")
    ledger = None
    if split != TRAIN_SPLIT:
        ledger = _consume_access(
            data_dir.resolve() / "access",
            split=split,
            board_sha=report_sha,
            split_sha=expected,
            source_commit=source_commit,
        )
    path = data_dir / filename
    if sha256_file(path) != expected:
        raise RuntimeError("projected split hash differs from receipt")
    rows = [
        parse_projected_row(json.loads(line), split)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    expected_rows = 48_000 if split == TRAIN_SPLIT else 2_304
    if len(rows) != expected_rows:
        raise RuntimeError("projected split row count changed")
    return rows, {
        "board_report_sha256": report_sha,
        "split_sha256": expected,
        "filename": filename,
        "seed": report["seed"],
        "registration": report[
            "development_registration"
            if split == DEVELOPMENT_SPLIT
            else "confirmation_registration"
            if split == CONFIRMATION_SPLIT
            else "development_registration"
        ],
        "access_ledger": ledger,
    }


def train_main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"refusing existing projected checkpoint: {args.out}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("projected fresh training requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    rows, board = load_rows(args.data_dir, TRAIN_SPLIT, args.source_commit)
    device = torch.device("cuda")
    init_seed = derived_seed(args.seed, "shared-initialization")
    order_seed = derived_seed(args.seed, "shared-minibatch-order")
    row_shuffle_seed = derived_seed(args.seed, "row-shuffled-labels")

    treatment, parameters = initialize_model(args.parent_checkpoint, init_seed, device)
    treatment_fit = fit_projected_arm(treatment, rows, seed=order_seed)
    treatment_state = trainable_state(treatment)
    treatment.cpu()
    del treatment
    torch.cuda.empty_cache()

    row_permutations = [
        row_shuffled_permutation(row_shuffle_seed, row.row_id) for row in rows
    ]
    control_rows = [
        permute_training_labels(row, permutation)
        for row, permutation in zip(rows, row_permutations, strict=True)
    ]
    row_shuffle_digest = row_shuffled_mapping_digest(rows, row_shuffle_seed)
    label_control, control_parameters = initialize_model(
        args.parent_checkpoint,
        init_seed,
        device,
    )
    label_fit = fit_projected_arm(label_control, control_rows, seed=order_seed)
    label_state = trainable_state(label_control)
    label_control.cpu()
    del label_control
    torch.cuda.empty_cache()
    if parameters != control_parameters:
        raise RuntimeError("matched projected arms have different parameter reports")
    if (
        treatment_fit["initial_full_state_sha256"]
        != label_fit["initial_full_state_sha256"]
    ):
        raise RuntimeError("matched projected arms did not share initialization")
    if treatment_fit["minibatch_order_sha256"] != label_fit["minibatch_order_sha256"]:
        raise RuntimeError("matched projected arms did not share minibatch order")
    if (
        not treatment_fit["frozen_parent_unchanged"]
        or not label_fit["frozen_parent_unchanged"]
    ):
        raise RuntimeError("projected training changed a frozen parent tensor")

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "source": source,
        "runtime": runtime_manifest(),
        "board": board,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "execution_core_sha256": EXECUTION_CORE_SHA256,
        "seeds": {
            "master": args.seed,
            "shared_initialization": init_seed,
            "shared_minibatch_order": order_seed,
            "row_shuffled_labels": row_shuffle_seed,
        },
        "row_shuffled_label_mapping_sha256": row_shuffle_digest,
        "training_contract": TRAINING_CONTRACT,
        "parameters": parameters,
        "arms": {
            "treatment": {"fit": treatment_fit, "trainable_state": treatment_state},
            "row_shuffled_labels": {
                "fit": label_fit,
                "trainable_state": label_state,
            },
        },
        "score_selection": "epoch_4_only",
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(
        json.dumps(
            {
                "saved": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
                "parameters": parameters,
                "treatment_train": treatment_fit["train_metrics"],
                "row_shuffled_labels_train": label_fit["train_metrics"],
            },
            sort_keys=True,
        )
    )


def config_main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"refusing existing projected gate config: {args.out}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("projected checkpoint schema mismatch")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    if checkpoint["source"]["sha256"] != source["sha256"]:
        raise SystemExit("projected source changed after training")
    report = _read_board_report(args.data_dir, args.source_commit)
    report_path = args.data_dir / "report.json"
    if report.get("inherited_parent_board") != {
        "report_sha256": PARENT_BOARD_REPORT_SHA256,
        "train_sha256": PARENT_BOARD_TRAIN_SHA256,
    }:
        raise SystemExit(
            "projected board does not bind the inherited parent train split"
        )
    board = checkpoint.get("board", {})
    parameters = checkpoint.get("parameters", {})
    arms = checkpoint.get("arms", {})
    seeds = checkpoint.get("seeds", {})
    master_seed = seeds.get("master")
    if type(master_seed) is not int or master_seed < 0:
        raise SystemExit("projected checkpoint master seed is invalid")
    expected_seeds = {
        "master": master_seed,
        "shared_initialization": derived_seed(master_seed, "shared-initialization"),
        "shared_minibatch_order": derived_seed(master_seed, "shared-minibatch-order"),
        "row_shuffled_labels": derived_seed(master_seed, "row-shuffled-labels"),
    }
    if seeds != expected_seeds:
        raise SystemExit("projected checkpoint derived seeds differ")
    train_rows, train_board = load_rows(args.data_dir, TRAIN_SPLIT, args.source_commit)
    expected_mapping_sha = row_shuffled_mapping_digest(
        train_rows,
        int(seeds["row_shuffled_labels"]),
    )
    if (
        checkpoint.get("protocol") != PROTOCOL
        or checkpoint.get("parent_checkpoint_sha256") != PARENT_SHA256
        or checkpoint.get("execution_core_sha256") != EXECUTION_CORE_SHA256
        or checkpoint.get("training_contract") != TRAINING_CONTRACT
        or checkpoint.get("score_selection") != "epoch_4_only"
        or checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
        or board.get("board_report_sha256") != sha256_file(report_path)
        or board.get("split_sha256") != report["files"]["train.jsonl"]["sha256"]
        or board.get("seed") != report.get("seed")
        or train_board.get("board_report_sha256") != board.get("board_report_sha256")
        or train_board.get("split_sha256") != board.get("split_sha256")
        or int(parameters.get("nominal_complete_system", -1)) != 146_057_595
        or int(parameters.get("trainable", -1)) != 6_748_897
        or set(parameters.get("trainable_names", [])) != set(PROJECTED_TRAINABLE_NAMES)
        or set(arms) != {"treatment", "row_shuffled_labels"}
        or not isinstance(checkpoint.get("row_shuffled_label_mapping_sha256"), str)
        or checkpoint["row_shuffled_label_mapping_sha256"] != expected_mapping_sha
    ):
        raise SystemExit("projected checkpoint violates legal training contract")
    fits = {name: value.get("fit", {}) for name, value in arms.items()}
    if any(
        fit.get("updates") != 3_000 or fit.get("frozen_parent_unchanged") is not True
        for fit in fits.values()
    ) or (
        fits["treatment"].get("initial_full_state_sha256")
        != fits["row_shuffled_labels"].get("initial_full_state_sha256")
        or fits["treatment"].get("minibatch_order_sha256")
        != fits["row_shuffled_labels"].get("minibatch_order_sha256")
    ):
        raise SystemExit(
            "projected checkpoint fit evidence violates matched-arm contract"
        )
    for arm, value in arms.items():
        state = value.get("trainable_state", {})
        if set(state) != set(PROJECTED_TRAINABLE_NAMES):
            raise SystemExit(f"projected checkpoint trainable keys differ: {arm}")
        if any(
            not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise SystemExit(f"projected checkpoint trainable state is invalid: {arm}")
    treatment_state = arms["treatment"]["trainable_state"]
    control_state = arms["row_shuffled_labels"]["trainable_state"]
    if any(
        treatment_state[name].shape != control_state[name].shape
        or treatment_state[name].dtype != control_state[name].dtype
        for name in PROJECTED_TRAINABLE_NAMES
    ):
        raise SystemExit("projected matched-arm tensor contracts differ")
    if sha256_file(args.execution_core) != EXECUTION_CORE_SHA256:
        raise SystemExit("projected execution core hash mismatch")
    if sha256_file(args.parent_checkpoint) != PARENT_SHA256:
        raise SystemExit("projected parent checkpoint hash mismatch")
    if sha256_file(args.consumed_checkpoint) != CONSUMED_PROJECTED_SHA256:
        raise SystemExit("consumed projected diagnostic hash mismatch")
    reference_model, reference_parameters = initialize_model(
        args.parent_checkpoint,
        int(seeds["shared_initialization"]),
        torch.device("cpu"),
    )
    reference_trainable = {
        name: parameter
        for name, parameter in reference_model.named_parameters()
        if parameter.requires_grad
    }
    reference_initial_digest = state_dict_digest(reference_model)
    reference_frozen_digest = frozen_parameter_digest(reference_model)
    if parameters != reference_parameters or any(
        fits[name].get("initial_full_state_sha256") != reference_initial_digest
        for name in fits
    ):
        raise SystemExit("projected checkpoint initialization/accounting differs")
    for arm, value in arms.items():
        for name, tensor in value["trainable_state"].items():
            target = reference_trainable[name]
            if tensor.shape != target.shape or tensor.dtype != target.dtype:
                raise SystemExit(
                    f"projected checkpoint trainable tensor contract differs: {arm}:{name}"
                )
        load_trainable_state(reference_model, value["trainable_state"])
        if (
            state_dict_digest(reference_model)
            != fits[arm].get("final_full_state_sha256")
            or fits[arm].get("frozen_digest_before") != reference_frozen_digest
            or fits[arm].get("frozen_digest_after") != reference_frozen_digest
        ):
            raise SystemExit(f"projected checkpoint final-state receipt differs: {arm}")
    del reference_model
    assessor = args.repo_root / "pipeline/assess_sd_cst_projected_fresh.py"
    evaluator = args.repo_root / "train/train_eval_sd_cst_projected_fresh.py"
    output = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "source": source,
        "expected": {
            "evaluation_schema": EVALUATION_SCHEMA,
            "row_count": 2_304,
            "family_count": 288,
            "family_size": 8,
            "variants": report["development_registration"]["variants"],
            "depths": [1, 2, 3, 4, 5, 6],
            "arms": [
                "treatment",
                "row_shuffled_labels",
                "consumed_projected",
                "binding_source_free_compiler",
            ],
            "control_arms": list(CONTROL_ARMS),
        },
        "registrations": {
            "development": report["development_registration"],
            "confirmation": report["confirmation_registration"],
        },
        "thresholds": THRESHOLDS,
        "training_contract": TRAINING_CONTRACT,
        "parameter_caps": {
            "comparison": COMPARISON_PARAMETER_CAP,
            "global": GLOBAL_PARAMETER_CAP,
        },
        "artifact_hashes": {
            "board": sha256_file(report_path),
            "checkpoint": sha256_file(args.checkpoint),
            "parent": PARENT_SHA256,
            "execution_core": EXECUTION_CORE_SHA256,
            "consumed_projected": CONSUMED_PROJECTED_SHA256,
            "evaluator": sha256_file(evaluator),
            "assessor": sha256_file(assessor),
        },
        "canonical_data_dir": str(args.data_dir.resolve()),
        "split_hashes": {
            "development": report["files"]["development.jsonl"]["sha256"],
            "confirmation": report["files"]["confirmation.sealed.jsonl"]["sha256"],
        },
        "expected_ledger_sha256": {
            "development": hashlib.sha256(
                _ledger_bytes(
                    split=DEVELOPMENT_SPLIT,
                    board_sha=sha256_file(report_path),
                    split_sha=report["files"]["development.jsonl"]["sha256"],
                    source_commit=args.source_commit,
                )
            ).hexdigest(),
            "confirmation": hashlib.sha256(
                _ledger_bytes(
                    split=CONFIRMATION_SPLIT,
                    board_sha=sha256_file(report_path),
                    split_sha=report["files"]["confirmation.sealed.jsonl"]["sha256"],
                    source_commit=args.source_commit,
                )
            ).hexdigest(),
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "saved": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
                "scored_split_opened": False,
            },
            sort_keys=True,
        )
    )


def _uniform_packet(rows: int) -> tuple[HardProgramTape, HardLateQuery]:
    kind = torch.zeros((rows, EVENT_STEPS), dtype=torch.uint8)
    kind[:, 0] = STOP_KIND
    return HardProgramTape(
        torch.zeros(rows, dtype=torch.uint8),
        kind,
        torch.zeros((rows, EVENT_STEPS), dtype=torch.uint8),
        torch.zeros((rows, EVENT_STEPS), dtype=torch.uint8),
    ), HardLateQuery(torch.zeros(rows, dtype=torch.uint8))


def _source_free_packet(
    rows: Sequence[ProjectedFreshRow],
) -> tuple[HardProgramTape, HardLateQuery]:
    count = len(rows)
    initial = torch.empty(count, dtype=torch.uint8)
    kind = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    identity = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    amount = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    query = torch.empty(count, dtype=torch.uint8)
    for index, row in enumerate(rows):
        digest = hashlib.sha256(
            ("projected-source-free:" + row.row_id).encode()
        ).digest()
        initial[index] = digest[0] % 6
        stop = digest[1] % EVENT_STEPS
        for step in range(EVENT_STEPS):
            kind[index, step] = (
                STOP_KIND if step == stop else digest[2 + step] % STOP_KIND
            )
            identity[index, step] = digest[10 + step] % 3
            amount[index, step] = digest[18 + step] % 2
        query[index] = digest[26] % 3
    return HardProgramTape(initial, kind, identity, amount), HardLateQuery(query)


def safe_perturb_post_stop(tape: HardProgramTape) -> HardProgramTape:
    """Perturb only after the first predicted STOP; malformed rows remain un-repaired."""
    identity = tape.event_identity.clone()
    amount = tape.amount.clone()
    for row in range(tape.batch_size):
        stops = tape.event_kind[row].eq(STOP_KIND).nonzero(as_tuple=False).flatten()
        if not stops.numel():
            continue
        for step in range(int(stops[0]) + 1, EVENT_STEPS):
            if int(tape.event_kind[row, step]) != STOP_KIND:
                identity[row, step] = (identity[row, step].long() + 1) % 3
                amount[row, step] = (amount[row, step].long() + 1) % 2
    return HardProgramTape(
        tape.initial_state.clone(),
        tape.event_kind.clone(),
        identity,
        amount,
    )


def safe_mutate_first_active(tape: HardProgramTape, field: str) -> HardProgramTape:
    """Mutate a predicted pre-STOP field without imposing a valid STOP grammar."""
    initial = tape.initial_state.clone()
    kind = tape.event_kind.clone()
    identity = tape.event_identity.clone()
    amount = tape.amount.clone()
    if field == "initial_state":
        initial = ((initial.long() + 1) % 6).to(torch.uint8)
    else:
        for row in range(tape.batch_size):
            stops = tape.event_kind[row].eq(STOP_KIND).nonzero(as_tuple=False).flatten()
            bound = int(stops[0]) if stops.numel() else EVENT_STEPS
            candidates = [
                step
                for step in range(bound)
                if int(tape.event_kind[row, step]) != STOP_KIND
            ]
            if not candidates:
                continue
            step = candidates[0]
            if field == "kind":
                kind[row, step] = 1 - kind[row, step]
            elif field == "identity":
                identity[row, step] = (identity[row, step].long() + 1) % 3
            elif field == "amount":
                amount[row, step] = 1 - amount[row, step]
            else:
                raise ValueError(f"unknown projected mutation field: {field}")
    return HardProgramTape(initial, kind, identity, amount)


def _load_arm_model(
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    arm: str,
    device: torch.device,
):
    model, parameters = initialize_model(
        args.parent_checkpoint,
        int(checkpoint["seeds"]["shared_initialization"]),
        device,
    )
    load_trainable_state(model, checkpoint["arms"][arm]["trainable_state"])
    model.requires_grad_(False).eval()
    return model, parameters


def _compiled_json(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "packet": tape_to_json(value["tape"], value["query"]),
        "pointers": {
            name: tensor.tolist() for name, tensor in value["pointers"].items()
        },
        "source_poison_bit_identical": bool(value["source_poison_bit_identical"]),
    }


def _row_evidence(row: ProjectedFreshRow) -> dict[str, object]:
    value = asdict(row)
    for key in ("program_bytes", "query_bytes"):
        value.pop(key)
    return value


def _authorize_confirmation(
    path: Path | None,
    gate_config: Path,
    development_evaluation: Path | None,
    development_packets: Path | None,
    development_executor: Path | None,
) -> dict[str, object]:
    required = (
        path,
        development_evaluation,
        development_packets,
        development_executor,
    )
    if any(value is None for value in required):
        raise SystemExit(
            "confirmation requires the complete development evidence chain"
        )
    from assess_sd_cst_projected_fresh import assess_files

    value = json.loads(path.read_text())
    recomputed = assess_files(
        development_evaluation,
        gate_config,
        development_packets,
        development_executor,
    )
    if (
        value.get("schema") != ASSESSMENT_SCHEMA
        or value.get("split") != DEVELOPMENT_SPLIT
        or value.get("decision") != "authorize_one_sealed_confirmation"
        or value.get("all_gates_pass") is not True
        or value.get("confirmation_authorized") is not True
        or value.get("gate_config_sha256") != sha256_file(gate_config)
        or value.get("evaluation_sha256") != sha256_file(development_evaluation)
        or canonical_json(value) != canonical_json(recomputed)
    ):
        raise SystemExit("development assessment did not authorize confirmation")
    return {
        "assessment_sha256": sha256_file(path),
        "development_evaluation_sha256": sha256_file(development_evaluation),
        "development_packets_sha256": sha256_file(development_packets),
        "development_executor_sha256": sha256_file(development_executor),
    }


@torch.no_grad()
def evaluate_main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"refusing existing projected evaluation: {args.out}")
    packet_path = args.out.with_suffix(".packets.pt")
    output_path = args.out.with_suffix(".executor.pt")
    for path in (packet_path, output_path):
        if path.exists():
            raise SystemExit(f"refusing existing projected auxiliary output: {path}")
    config = json.loads(args.gate_config.read_text())
    config_sha = sha256_file(args.gate_config)
    if config.get("schema") != CONFIG_SCHEMA:
        raise SystemExit("projected gate config schema mismatch")
    if str(args.data_dir.resolve()) != config.get("canonical_data_dir"):
        raise SystemExit("projected board path differs from canonical gate path")
    if sha256_file(Path(__file__)) != config["artifact_hashes"]["evaluator"]:
        raise SystemExit("projected evaluator differs from gate config")
    assessor_path = args.repo_root / "pipeline/assess_sd_cst_projected_fresh.py"
    if sha256_file(assessor_path) != config["artifact_hashes"]["assessor"]:
        raise SystemExit("projected assessor differs from gate config")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("projected checkpoint schema mismatch")
    if sha256_file(args.checkpoint) != config["artifact_hashes"]["checkpoint"]:
        raise SystemExit("projected checkpoint differs from gate config")
    if sha256_file(args.parent_checkpoint) != config["artifact_hashes"]["parent"]:
        raise SystemExit("projected parent differs from gate config")
    if sha256_file(args.execution_core) != config["artifact_hashes"]["execution_core"]:
        raise SystemExit("projected execution core differs from gate config")
    if (
        sha256_file(args.consumed_checkpoint)
        != config["artifact_hashes"]["consumed_projected"]
    ):
        raise SystemExit("consumed projected diagnostic differs from gate config")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    if source["sha256"] != config["source"]["sha256"]:
        raise SystemExit("projected source changed after gate freeze")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("projected evaluation requires bf16 CUDA")
    authorization = None
    if args.split == CONFIRMATION_SPLIT:
        authorization = _authorize_confirmation(
            args.authorization,
            args.gate_config,
            args.development_evaluation,
            args.development_packets,
            args.development_executor,
        )
    rows, board = load_rows(
        args.data_dir,
        args.split,
        args.source_commit,
    )
    expected_ledger = config["expected_ledger_sha256"][
        "development" if args.split == DEVELOPMENT_SPLIT else "confirmation"
    ]
    if board["access_ledger"]["sha256"] != expected_ledger:
        raise SystemExit("projected access ledger differs from gate config")
    expected_split = config["split_hashes"][
        "development" if args.split == DEVELOPMENT_SPLIT else "confirmation"
    ]
    if board["split_sha256"] != expected_split:
        raise SystemExit("projected scored split differs from gate config")
    device = torch.device("cuda")

    compiled = {}
    parameters = None
    for arm in ("treatment", "row_shuffled_labels"):
        model, arm_parameters = _load_arm_model(args, checkpoint, arm, device)
        compiled[arm] = compile_fresh_rows(model, rows, args.batch_size, device)
        if arm == "treatment":
            compiled["binding_source_free_compiler"] = compile_fresh_rows(
                model,
                rows,
                args.batch_size,
                device,
                source_free_binding=True,
            )
        parameters = arm_parameters if parameters is None else parameters
        if parameters != arm_parameters:
            raise RuntimeError("projected evaluation parameter mismatch")
        model.cpu()
        del model
        torch.cuda.empty_cache()

    consumed_payload = torch.load(
        args.consumed_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    consumed_model, _ = initialize_model(
        args.parent_checkpoint,
        int(consumed_payload["seed"]),
        device,
    )
    consumed_model.load_state_dict(consumed_payload["state"], strict=True)
    consumed_model.requires_grad_(False).eval()
    compiled["consumed_projected"] = compile_fresh_rows(
        consumed_model,
        rows,
        args.batch_size,
        device,
    )
    consumed_model.cpu()
    del consumed_model, consumed_payload
    torch.cuda.empty_cache()

    treatment_tape = compiled["treatment"]["tape"]
    treatment_query = compiled["treatment"]["query"]
    uniform_tape, uniform_query = _uniform_packet(len(rows))
    sf_tape, sf_query = _source_free_packet(rows)
    state_swap = torch.arange(len(rows) - 1, -1, -1, dtype=torch.long)
    packet_inputs = {
        name: packet_arm(value["tape"], value["query"])
        for name, value in compiled.items()
    }
    packet_inputs.update(
        {
            "uniform": packet_arm(uniform_tape, uniform_query),
            "source_free_packet": packet_arm(sf_tape, sf_query),
            "shuffled_packet": packet_arm(
                shuffled_packet(treatment_tape), treatment_query
            ),
            "reset": packet_arm(treatment_tape, treatment_query, control="reset"),
            "freeze": packet_arm(treatment_tape, treatment_query, control="freeze"),
            "post_stop_perturbation": packet_arm(
                safe_perturb_post_stop(treatment_tape),
                treatment_query,
            ),
            "force_alive_post_stop": packet_arm(
                safe_perturb_post_stop(treatment_tape),
                treatment_query,
                force_alive=True,
            ),
            "operand_suffix_swap": packet_arm(
                swap_operand_suffix(treatment_tape, 4),
                treatment_query,
            ),
            "query_rotation": packet_arm(
                treatment_tape, rotate_queries(treatment_query)
            ),
            "state_swap_after_step_0": packet_arm(
                treatment_tape,
                treatment_query,
                state_swap=state_swap,
                swap_after_step=0,
            ),
            "initial_state_rotation": packet_arm(
                safe_mutate_first_active(treatment_tape, "initial_state"),
                treatment_query,
            ),
            "event_kind_flip": packet_arm(
                safe_mutate_first_active(treatment_tape, "kind"),
                treatment_query,
            ),
            "event_identity_rotation": packet_arm(
                safe_mutate_first_active(treatment_tape, "identity"),
                treatment_query,
            ),
            "event_amount_flip": packet_arm(
                safe_mutate_first_active(treatment_tape, "amount"),
                treatment_query,
            ),
        }
    )
    torch.save(
        {
            "schema": "r12_sd_cst_hard_packet_bundle_v1",
            "arms": packet_inputs,
        },
        packet_path,
    )
    subprocess.run(
        [
            sys.executable,
            str(args.repo_root / "train/run_sd_cst_hard_packets.py"),
            "--packets",
            str(packet_path),
            "--execution-core",
            str(args.execution_core),
            "--output",
            str(output_path),
        ],
        check=True,
    )
    executor = torch.load(output_path, map_location="cpu", weights_only=True)
    if executor.get("schema") != "r12_sd_cst_hard_packet_outputs_v1" or set(
        executor.get("outputs", {})
    ) != set(packet_inputs):
        raise SystemExit("projected source-blind executor output mismatch")

    gold_tape, gold_query = expected_tape(rows)
    output = {
        "schema": EVALUATION_SCHEMA,
        "protocol": PROTOCOL,
        "split": args.split,
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": int(args.split == CONFIRMATION_SPLIT),
            "access_ledger": board["access_ledger"],
            "confirmation_authorization": authorization,
        },
        "artifact_hashes": {
            "source_manifest": source["sha256"],
            "board": board["board_report_sha256"],
            "split": board["split_sha256"],
            "checkpoint": sha256_file(args.checkpoint),
            "parent": sha256_file(args.parent_checkpoint),
            "gate_config": config_sha,
            "execution_core": sha256_file(args.execution_core),
            "consumed_projected": sha256_file(args.consumed_checkpoint),
            "evaluator": sha256_file(Path(__file__)),
            "assessor": sha256_file(
                args.repo_root / "pipeline/assess_sd_cst_projected_fresh.py"
            ),
            "hard_packets": sha256_file(packet_path),
            "executor_outputs": sha256_file(output_path),
        },
        "parameters": parameters,
        "training": {
            "contract": checkpoint["training_contract"],
            "seeds": checkpoint["seeds"],
            "row_shuffled_label_mapping_sha256": checkpoint[
                "row_shuffled_label_mapping_sha256"
            ],
            "fits": {
                arm: checkpoint["arms"][arm]["fit"]
                for arm in ("treatment", "row_shuffled_labels")
            },
        },
        "rows": [_row_evidence(row) for row in rows],
        "gold_packet": tape_to_json(gold_tape, gold_query),
        "compiled": {name: _compiled_json(value) for name, value in compiled.items()},
        "packet_arms": {
            name: tape_to_json(
                HardProgramTape(
                    value["initial_state"],
                    value["event_kind"],
                    value["event_identity"],
                    value["amount"],
                ),
                HardLateQuery(value["query"]),
            )
            | {
                "control": value["control"],
                "force_alive": value["force_alive"],
                "state_swap": (
                    value["state_swap"].tolist()
                    if value["state_swap"] is not None
                    else None
                ),
                "swap_after_step": value["swap_after_step"],
            }
            for name, value in packet_inputs.items()
        },
        "executor_outputs": {
            name: outputs_to_json(value) for name, value in executor["outputs"].items()
        },
        "source_deletion": {
            "program_elements_per_row": 25,
            "query_elements_per_row": 1,
            "dtype": "torch.uint8",
            "program_gpu_tensors_destroyed_before_query_compile": True,
            "host_evaluator_retains_hash_bound_row_evidence": True,
            "separate_typed_executor": True,
            "all_compiler_arms_source_poison_bit_identical": all(
                bool(value["source_poison_bit_identical"])
                for value in compiled.values()
            ),
        },
        "runtime": runtime_manifest(),
        "claim_boundary": (
            "Bounded fresh-board source-deleted state transport only; the nominal "
            "Shohin trunk is inactive in the projected compiler forward path."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "saved": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
                "split": args.split,
                "rows": len(rows),
                "executor_arms": len(packet_inputs),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    train = modes.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--parent-checkpoint", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--repo-root", type=Path, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--seed", type=int, required=True)

    config = modes.add_parser("config")
    config.add_argument("--data-dir", type=Path, required=True)
    config.add_argument("--checkpoint", type=Path, required=True)
    config.add_argument("--parent-checkpoint", type=Path, required=True)
    config.add_argument("--execution-core", type=Path, required=True)
    config.add_argument("--consumed-checkpoint", type=Path, required=True)
    config.add_argument("--out", type=Path, required=True)
    config.add_argument("--repo-root", type=Path, required=True)
    config.add_argument("--source-commit", required=True)

    evaluate = modes.add_parser("evaluate")
    evaluate.add_argument(
        "--split", choices=(DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT), required=True
    )
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--parent-checkpoint", type=Path, required=True)
    evaluate.add_argument("--execution-core", type=Path, required=True)
    evaluate.add_argument("--consumed-checkpoint", type=Path, required=True)
    evaluate.add_argument("--gate-config", type=Path, required=True)
    evaluate.add_argument("--authorization", type=Path)
    evaluate.add_argument("--development-evaluation", type=Path)
    evaluate.add_argument("--development-packets", type=Path)
    evaluate.add_argument("--development-executor", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--repo-root", type=Path, required=True)
    evaluate.add_argument("--source-commit", required=True)
    evaluate.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "train":
        train_main(args)
    elif args.mode == "config":
        config_main(args)
    elif args.mode == "evaluate":
        evaluate_main(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
