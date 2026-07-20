#!/usr/bin/env python3
"""Train and one-read evaluate the fresh complete physical SD-CST compiler."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
)
from build_sd_cst_complete_physical_fresh_board import BOARD_SCHEMA, PROTOCOL
from pilot_sd_cst_byte_addressed import sha256_file
from projected_sd_cst_fresh import expected_tape
from sd_cst import STOP_KIND, HardLateQuery, HardProgramTape
from sd_cst_complete_physical_fresh import (
    TRAINING_CONTRACT,
    compile_rows,
    derived_seed,
    endpoint_hashes,
    fit_arm,
    initialize_model,
    load_rows,
    load_trainable_state,
    permute_family_labels,
    trainable_state,
)
from train_eval_sd_cst_projected_fresh import (
    safe_mutate_first_active,
    safe_perturb_post_stop,
)


CHECKPOINT_SCHEMA = "r12_sd_cst_complete_physical_fresh_checkpoint_v1_3"
REPORT_SCHEMA = "r12_sd_cst_complete_physical_fresh_development_report_v1_3"
TRAIN_SPLIT = "sd_cst_train"
DEVELOPMENT_SPLIT = "sd_cst_development"
FROZEN_SOURCE_PATHS = (
    "R12_SD_CST_COMPLETE_PHYSICAL_FRESH_V1_3_PREREG.md",
    "pipeline/build_sd_cst_complete_physical_fresh_board.py",
    "pipeline/sd_cst_complete_physical_fresh_renderers.py",
    "train/sd_cst_complete_physical_fresh.py",
    "train/pilot_sd_cst_complete_physical_fresh.py",
    "train/assess_sd_cst_complete_physical_fresh.py",
    "train/sd_cst_complete_physical_record_bus_v1_2.py",
    "train/sd_cst_complete_physical_record_bus.py",
    "train/sd_cst_physical_record_bus.py",
    "train/projected_sd_cst_fresh.py",
    "train/run_sd_cst_hard_packets.py",
    "train/sd_cst.py",
    "train/jobs/sd_cst_complete_physical_fresh.sbatch",
)
THRESHOLDS = {
    "fit_packet_min_renderer": 0.99,
    "packet_overall": 0.90,
    "packet_min_renderer": 0.85,
    "state_overall": 0.90,
    "answer_overall": 0.90,
    "joint_overall": 0.90,
    "joint_min_renderer": 0.85,
    "field_overall": 0.95,
    "pointer_overall": 0.90,
    "treatment_packet_advantage": 0.50,
    "row_shuffled_packet_max": 0.25,
    "negative_state_max": 0.35,
    "reset_freeze_state_max": 0.75,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def runtime_manifest() -> dict[str, object]:
    value: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
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
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
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
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("fresh scientific source commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("fresh scientific source is not an ancestor of HEAD")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"fresh source omits frozen path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"fresh runtime differs from source: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def _read_board(data_dir: Path, source_commit: str) -> dict[str, object]:
    report = json.loads((data_dir / "report.json").read_text())
    if (
        report.get("schema") != BOARD_SCHEMA
        or report.get("protocol") != PROTOCOL
        or report.get("source_commit") != source_commit
        or report.get("all_gates_pass") is not True
        or report.get("development_accesses") != 0
        or report.get("confirmation_accesses") != 0
    ):
        raise RuntimeError("fresh board receipt is not admitted")
    return report


def _ledger_bytes(*, board_sha: str, split_sha: str, source_commit: str) -> bytes:
    return (
        json.dumps(
            {
                "schema": "r12_sd_cst_complete_physical_fresh_access_v1_3",
                "protocol": PROTOCOL,
                "split": DEVELOPMENT_SPLIT,
                "board_report_sha256": board_sha,
                "split_sha256": split_sha,
                "source_commit": source_commit,
                "access_number": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _consume_development_access(
    data_dir: Path, board: Mapping[str, object], source_commit: str
) -> dict[str, str]:
    board_sha = sha256_file(data_dir / "report.json")
    split_sha = str(board["files"]["development.jsonl"]["sha256"])
    payload = _ledger_bytes(
        board_sha=board_sha,
        split_sha=split_sha,
        source_commit=source_commit,
    )
    directory = data_dir / "access"
    directory.mkdir(exist_ok=True)
    path = directory / f"complete_physical_fresh_development_{split_sha}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _minimum_fit_packet(fit: Mapping[str, object]) -> float:
    metrics = fit.get("train_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise RuntimeError("fresh fit renderer metrics are absent")
    rates = []
    for renderer, value in metrics.items():
        if not isinstance(renderer, str) or not isinstance(value, Mapping):
            raise RuntimeError("fresh fit renderer metric differs")
        renderer_rates = value.get("rates")
        if not isinstance(renderer_rates, Mapping) or "packet" not in renderer_rates:
            raise RuntimeError("fresh fit renderer packet rate is absent")
        rates.append(float(renderer_rates["packet"]))
    return min(rates)


def _exact_packet(
    prediction: tuple[HardProgramTape, HardLateQuery],
    gold: tuple[HardProgramTape, HardLateQuery],
) -> dict[str, torch.Tensor]:
    tape, query = prediction
    target, target_query = gold
    active = target.event_kind.ne(STOP_KIND)
    fields = {
        "initial": tape.initial_state.eq(target.initial_state),
        "kind": tape.event_kind.eq(target.event_kind).all(-1),
        "identity": (tape.event_identity.eq(target.event_identity) | ~active).all(-1),
        "amount": (tape.amount.eq(target.amount) | ~active).all(-1),
        "query": query.position.eq(target_query.position),
    }
    fields["packet"] = torch.stack(list(fields.values())).all(0)
    return fields


def _pointer_exact(
    pointers: Mapping[str, torch.Tensor], rows: Sequence[object]
) -> dict[str, torch.Tensor]:
    specs = {
        "line": (9, "pointer_ranges"),
        "binding": (3, "binding_ranges"),
        "initial_entity": (3, "initial_entity_ranges"),
        "event_entity": (8, "event_entity_ranges"),
    }
    output = {}
    for name, (slots, attribute) in specs.items():
        prediction = pointers[name]
        if tuple(prediction.shape) != (len(rows), slots):
            raise RuntimeError(f"fresh pointer shape differs: {name}")
        exact = torch.ones(len(rows), dtype=torch.bool)
        for index, row in enumerate(rows):
            ranges = getattr(row, attribute)
            for slot, (start, end) in enumerate(ranges):
                if end <= start:
                    if name == "event_entity" and row.event_kind[slot] == STOP_KIND:
                        continue
                    raise RuntimeError("fresh active pointer span is empty")
                exact[index] &= start <= int(prediction[index, slot]) < end
        output[name] = exact
    return output


def _rate(values: torch.Tensor) -> float:
    return float(values.float().mean())


def _summary(values: Mapping[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "correct": int(value.sum()),
            "rows": value.numel(),
            "rate": _rate(value),
        }
        for name, value in values.items()
    }


def _grouped(
    values: torch.Tensor, rows: Sequence[object], attribute: str
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(getattr(row, attribute))].append(index)
    return {
        key: {
            "correct": int(values[indices].sum()),
            "rows": len(indices),
            "rate": _rate(values[indices]),
        }
        for key, indices in sorted(groups.items())
    }


def _load_arm(
    args: argparse.Namespace,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
):
    model, parameters, _ = initialize_model(
        args.joint_checkpoint,
        args.physical_checkpoint,
        args.v1_checkpoint,
        args.v1_2_checkpoint,
        device,
    )
    load_trainable_state(model, state)
    model.requires_grad_(False).eval()
    return model, parameters


def _execute(
    args: argparse.Namespace,
    out_dir: Path,
    compiled: Mapping[str, Mapping[str, object]],
    gold: tuple[HardProgramTape, HardLateQuery],
) -> tuple[dict[str, object], Path, Path]:
    treatment_tape = compiled["treatment"]["tape"]
    treatment_query = compiled["treatment"]["query"]
    uniform_kind = torch.zeros_like(treatment_tape.event_kind)
    uniform_kind[:, 0] = STOP_KIND
    uniform_tape = HardProgramTape(
        torch.zeros_like(treatment_tape.initial_state),
        uniform_kind,
        torch.zeros_like(treatment_tape.event_identity),
        torch.zeros_like(treatment_tape.amount),
    )
    uniform_query = HardLateQuery(torch.zeros_like(treatment_query.position))
    inputs = {
        name: packet_arm(value["tape"], value["query"])
        for name, value in compiled.items()
    }
    inputs.update(
        {
            "gold": packet_arm(*gold),
            "uniform": packet_arm(uniform_tape, uniform_query),
            "shuffled_packet": packet_arm(
                shuffled_packet(treatment_tape), treatment_query
            ),
            "reset": packet_arm(treatment_tape, treatment_query, control="reset"),
            "freeze": packet_arm(treatment_tape, treatment_query, control="freeze"),
            "post_stop_perturbation": packet_arm(
                safe_perturb_post_stop(treatment_tape), treatment_query
            ),
            "force_alive_post_stop": packet_arm(
                safe_perturb_post_stop(treatment_tape),
                treatment_query,
                force_alive=True,
            ),
            "query_rotation": packet_arm(
                treatment_tape, rotate_queries(treatment_query)
            ),
            "initial_state_rotation": packet_arm(
                safe_mutate_first_active(treatment_tape, "initial_state"),
                treatment_query,
            ),
            "event_kind_flip": packet_arm(
                safe_mutate_first_active(treatment_tape, "kind"), treatment_query
            ),
            "event_identity_rotation": packet_arm(
                safe_mutate_first_active(treatment_tape, "identity"), treatment_query
            ),
            "event_amount_flip": packet_arm(
                safe_mutate_first_active(treatment_tape, "amount"), treatment_query
            ),
        }
    )
    packet_path = out_dir / "development_packets.pt"
    output_path = out_dir / "development_executor.pt"
    torch.save(
        {"schema": "r12_sd_cst_hard_packet_bundle_v1", "arms": inputs}, packet_path
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
    outputs = torch.load(output_path, map_location="cpu", weights_only=True)
    if outputs.get("schema") != "r12_sd_cst_hard_packet_outputs_v1":
        raise RuntimeError("fresh source-blind executor schema differs")
    return outputs["outputs"], packet_path, output_path


def _save_evidence(
    path: Path,
    compiled: Mapping[str, Mapping[str, object]],
    rows: Sequence[object],
) -> None:
    pointer_specs = {
        "line": "pointer_ranges",
        "binding": "binding_ranges",
        "initial_entity": "initial_entity_ranges",
        "event_entity": "event_entity_ranges",
    }
    renderer_names = sorted({str(row.variant) for row in rows})
    renderer_to_index = {name: index for index, name in enumerate(renderer_names)}
    torch.save(
        {
            "schema": "r12_sd_cst_complete_physical_fresh_evidence_v1_3",
            "pointers": {
                arm: {
                    name: value.detach().cpu().long().clone()
                    for name, value in output["pointers"].items()
                }
                for arm, output in compiled.items()
            },
            "pointer_ranges": {
                name: torch.tensor(
                    [getattr(row, attribute) for row in rows], dtype=torch.long
                )
                for name, attribute in pointer_specs.items()
            },
            "renderer_names": renderer_names,
            "renderer_index": torch.tensor(
                [renderer_to_index[str(row.variant)] for row in rows],
                dtype=torch.uint8,
            ),
            "halt_after": torch.tensor(
                [int(row.halt_after) for row in rows], dtype=torch.uint8
            ),
            "source_poison_bit_identical": {
                arm: bool(output["source_poison_bit_identical"])
                for arm, output in compiled.items()
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--execution-core", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing fresh output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("fresh physical qualification requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = _read_board(args.data_dir, args.source_commit)
    if sha256_file(args.execution_core) != endpoint_hashes()["execution_core"]:
        raise SystemExit("fresh execution core hash differs")
    train_path = args.data_dir / "train.jsonl"
    if sha256_file(train_path) != board["files"]["train.jsonl"]["sha256"]:
        raise SystemExit("fresh train split hash differs")
    train_rows = load_rows(train_path, TRAIN_SPLIT)
    device = torch.device("cuda")
    order_seed = derived_seed(args.seed, "shared-minibatch-order")
    shuffle_seed = derived_seed(args.seed, "family-shuffled-labels")

    treatment, parameters, frozen_digest = initialize_model(
        args.joint_checkpoint,
        args.physical_checkpoint,
        args.v1_checkpoint,
        args.v1_2_checkpoint,
        device,
    )
    treatment_fit = fit_arm(treatment, train_rows, seed=order_seed)
    treatment_state = trainable_state(treatment)
    treatment.cpu()
    del treatment
    torch.cuda.empty_cache()

    shuffled_rows, mapping_digest = permute_family_labels(train_rows, shuffle_seed)
    control, control_parameters, control_frozen_digest = initialize_model(
        args.joint_checkpoint,
        args.physical_checkpoint,
        args.v1_checkpoint,
        args.v1_2_checkpoint,
        device,
    )
    control_fit = fit_arm(control, shuffled_rows, seed=order_seed)
    control_state = trainable_state(control)
    control.cpu()
    del control
    torch.cuda.empty_cache()
    if parameters != control_parameters or frozen_digest != control_frozen_digest:
        raise RuntimeError("fresh matched arms differ before training")
    if (
        treatment_fit["initial_full_state_sha256"]
        != control_fit["initial_full_state_sha256"]
    ):
        raise RuntimeError("fresh matched arms did not share initialization")

    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "source": source,
        "runtime": runtime_manifest(),
        "board_report_sha256": sha256_file(args.data_dir / "report.json"),
        "train_sha256": sha256_file(train_path),
        "endpoint_hashes": endpoint_hashes(),
        "parameters": parameters,
        "training_contract": TRAINING_CONTRACT,
        "seeds": {
            "master": args.seed,
            "shared_minibatch_order": order_seed,
            "family_shuffled_labels": shuffle_seed,
        },
        "family_shuffled_mapping_sha256": mapping_digest,
        "arms": {
            "treatment": {"fit": treatment_fit, "trainable_state": treatment_state},
            "row_shuffled_labels": {
                "fit": control_fit,
                "trainable_state": control_state,
            },
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    torch.save(checkpoint, checkpoint_path)
    gate_config = {
        "schema": "r12_sd_cst_complete_physical_fresh_gate_config_v1_3",
        "protocol": PROTOCOL,
        "source_manifest_sha256": source["sha256"],
        "board_report_sha256": checkpoint["board_report_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_sha256": board["files"]["development.jsonl"]["sha256"],
        "confirmation_sha256": board["files"]["confirmation.sealed.jsonl"]["sha256"],
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "training_contract": TRAINING_CONTRACT,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    config_path = args.out_dir / "gate_config.json"
    config_path.write_text(json.dumps(gate_config, indent=2, sort_keys=True) + "\n")

    ledger = _consume_development_access(args.data_dir, board, args.source_commit)
    development_path = args.data_dir / "development.jsonl"
    if sha256_file(development_path) != gate_config["development_sha256"]:
        raise RuntimeError("fresh development split hash differs after access")
    rows = load_rows(development_path, DEVELOPMENT_SPLIT)
    compiled = {}
    for arm, state in (
        ("treatment", treatment_state),
        ("row_shuffled_labels", control_state),
    ):
        model, arm_parameters = _load_arm(args, state, device)
        if arm_parameters != parameters:
            raise RuntimeError("fresh evaluation parameter report differs")
        compiled[arm] = compile_rows(model, rows, args.batch_size, device)
        model.cpu()
        del model
        torch.cuda.empty_cache()

    gold = expected_tape(rows)
    evidence_path = args.out_dir / "development_evidence.pt"
    _save_evidence(evidence_path, compiled, rows)
    executor, packet_path, executor_path = _execute(args, args.out_dir, compiled, gold)
    final = torch.tensor([row.final_state for row in rows], dtype=torch.uint8)
    answer = torch.tensor([row.answer_role for row in rows], dtype=torch.long)
    arm_metrics = {}
    arm_exact = {}
    for arm in ("treatment", "row_shuffled_labels"):
        packet = (compiled[arm]["tape"], compiled[arm]["query"])
        packet_fields = _exact_packet(packet, gold)
        pointers = _pointer_exact(compiled[arm]["pointers"], rows)
        state = executor[arm]["final_state"].eq(final)
        answer_ok = executor[arm]["answer"].eq(answer)
        exact = (
            packet_fields
            | pointers
            | {
                "state": state,
                "answer": answer_ok,
                "joint": state & answer_ok,
            }
        )
        arm_exact[arm] = exact
        arm_metrics[arm] = {
            "overall": _summary(exact),
            "packet_by_renderer": _grouped(packet_fields["packet"], rows, "variant"),
            "joint_by_renderer": _grouped(exact["joint"], rows, "variant"),
            "packet_by_depth": _grouped(packet_fields["packet"], rows, "halt_after"),
            "joint_by_depth": _grouped(exact["joint"], rows, "halt_after"),
            "source_poison_bit_identical": bool(
                compiled[arm]["source_poison_bit_identical"]
            ),
        }

    gold_exact = executor["gold"]["final_state"].eq(final) & executor["gold"][
        "answer"
    ].eq(answer)
    treatment_packet = arm_exact["treatment"]["packet"]
    conditional = (
        arm_exact["treatment"]["joint"][treatment_packet]
        if bool(treatment_packet.any())
        else torch.zeros(1, dtype=torch.bool)
    )
    post_stop_invariant = all(
        torch.equal(
            executor["treatment"][name], executor["post_stop_perturbation"][name]
        )
        for name in ("final_state", "answer", "state_trajectory", "alive_trajectory")
    )
    control_metrics = {
        name: {
            "state_rate": _rate(value["final_state"].eq(final)),
            "answer_rate": _rate(value["answer"].eq(answer)),
        }
        for name, value in executor.items()
        if name not in ("treatment", "row_shuffled_labels", "gold")
    }
    treatment = arm_metrics["treatment"]
    shuffled = arm_metrics["row_shuffled_labels"]
    gates = {
        "fit_packet_min_renderer_at_least_99pct": _minimum_fit_packet(treatment_fit)
        >= THRESHOLDS["fit_packet_min_renderer"],
        "packet_overall_at_least_90pct": treatment["overall"]["packet"]["rate"]
        >= THRESHOLDS["packet_overall"],
        "packet_min_renderer_at_least_85pct": min(
            value["rate"] for value in treatment["packet_by_renderer"].values()
        )
        >= THRESHOLDS["packet_min_renderer"],
        "state_answer_joint_at_least_90pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS[f"{name}_overall"]
            for name in ("state", "answer", "joint")
        ),
        "joint_min_renderer_at_least_85pct": min(
            value["rate"] for value in treatment["joint_by_renderer"].values()
        )
        >= THRESHOLDS["joint_min_renderer"],
        "all_packet_fields_at_least_95pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS["field_overall"]
            for name in ("initial", "kind", "identity", "amount", "query")
        ),
        "all_pointers_at_least_90pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS["pointer_overall"]
            for name in ("line", "binding", "initial_entity", "event_entity")
        ),
        "treatment_packet_advantage_at_least_50pp": treatment["overall"]["packet"][
            "rate"
        ]
        - shuffled["overall"]["packet"]["rate"]
        >= THRESHOLDS["treatment_packet_advantage"],
        "row_shuffled_packet_at_most_25pct": shuffled["overall"]["packet"]["rate"]
        <= THRESHOLDS["row_shuffled_packet_max"],
        "gold_executor_exact": bool(gold_exact.all()),
        "conditional_execution_exact": bool(conditional.all()),
        "post_stop_perturbation_invariant": post_stop_invariant,
        "shuffled_packet_state_at_most_35pct": control_metrics["shuffled_packet"][
            "state_rate"
        ]
        <= THRESHOLDS["negative_state_max"],
        "reset_and_freeze_state_at_most_75pct": all(
            control_metrics[name]["state_rate"] <= THRESHOLDS["reset_freeze_state_max"]
            for name in ("reset", "freeze")
        ),
        "source_deleted_before_separate_execution": all(
            bool(compiled[arm]["source_poison_bit_identical"]) for arm in compiled
        ),
        "complete_system_below_200m": int(parameters["complete_system"]) < 200_000_000,
        "frozen_state_unchanged": treatment_fit["frozen_parent_unchanged"]
        and control_fit["frozen_parent_unchanged"],
        "development_one_confirmation_zero": True,
    }
    decision = (
        "authorize_one_sealed_confirmation"
        if all(gates.values())
        else "reject_complete_physical_fresh_v1_3"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "thresholds": THRESHOLDS,
        "source": source,
        "runtime": runtime_manifest(),
        "board": {
            "report_sha256": checkpoint["board_report_sha256"],
            "development_sha256": gate_config["development_sha256"],
            "registration": board["development_registration"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "development_ledger": ledger,
        },
        "parameters": parameters,
        "training": {
            "contract": TRAINING_CONTRACT,
            "seeds": checkpoint["seeds"],
            "family_shuffled_mapping_sha256": mapping_digest,
            "fits": {
                "treatment": treatment_fit,
                "row_shuffled_labels": control_fit,
            },
        },
        "metrics": arm_metrics,
        "controls": control_metrics,
        "source_deletion": {
            "program_elements_per_row": 25,
            "query_elements_per_row": 1,
            "separate_typed_executor": True,
            "program_source_poisoned_after_packet_seal": True,
            "packet_path_sha256": sha256_file(packet_path),
            "evidence_path_sha256": sha256_file(evidence_path),
            "executor_path_sha256": sha256_file(executor_path),
        },
        "artifacts": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "gate_config_sha256": sha256_file(config_path),
            "packet_sha256": sha256_file(packet_path),
            "evidence_sha256": sha256_file(evidence_path),
            "executor_sha256": sha256_file(executor_path),
        },
        "claim_boundary": (
            "Fresh finite renderer/name transfer into a source-deleted categorical "
            "executor. Passing is not broad natural-language or general reasoning."
        ),
    }
    report_path = args.out_dir / "development_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "treatment": treatment["overall"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
