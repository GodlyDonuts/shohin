#!/usr/bin/env python3
"""Run the one-read sealed confirmation for complete physical fresh v1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

import torch

from build_sd_cst_board import CONFIRMATION_SPLIT
from build_sd_cst_complete_physical_fresh_board import BOARD_SCHEMA, PROTOCOL
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_complete_physical_fresh import (
    THRESHOLDS,
    _exact_packet,
    _execute,
    _grouped,
    _load_arm,
    _minimum_fit_packet,
    _pointer_exact,
    _rate,
    _save_evidence,
    _summary,
    runtime_manifest,
    source_manifest,
)
from projected_sd_cst_fresh import expected_tape
from sd_cst_complete_physical_fresh import compile_rows, load_rows


REPORT_SCHEMA = "r12_sd_cst_complete_physical_fresh_confirmation_report_v1_3"
ACCESS_SCHEMA = "r12_sd_cst_complete_physical_fresh_confirmation_access_v1_3"
CHECKPOINT_SCHEMA = "r12_sd_cst_complete_physical_fresh_checkpoint_v1_3"
CONFIG_SCHEMA = "r12_sd_cst_complete_physical_fresh_gate_config_v1_3"
DEVELOPMENT_ASSESSMENT_SCHEMA = "r12_sd_cst_complete_physical_fresh_assessment_v1_3"
SCIENTIFIC_SOURCE_COMMIT = "eed66757c47e126b6566ee269bc73b0c0cef4fab"
BOARD_REPORT_SHA256 = "fd487cdf7c30cf945ace152e389aaf5c354b8b6a55555c2acc6f046e8ed00b24"
CHECKPOINT_SHA256 = "a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8"
GATE_CONFIG_SHA256 = "ab466c339b77d4193cbdbc383a2c9a28bd4ce6afcf9579f3a714b301e8d9a990"
DEVELOPMENT_REPORT_SHA256 = "7dc048cc9ad16e1e326c7e4180fb06539428a4518c4d79440a92b794754b6bc2"
DEVELOPMENT_ASSESSMENT_SHA256 = (
    "1c5fad49a6eba6c2d76420945166e78b807d002f947a616c542c8a85ba35e497"
)
DEVELOPMENT_LEDGER_SHA256 = (
    "15a9edd09f008084c0533672e57af8da96935da3dc1ca86edbaa4200e2f499e0"
)
CONFIRMATION_SHA256 = "6186fb8c83c9863db2844f5eb537194a713c5ab16d2a41f1f88f6e3742f02165"
EVALUATOR_SOURCE_PATHS = (
    "R12_SD_CST_COMPLETE_PHYSICAL_FRESH_V1_3_CONFIRMATION_PREREG.md",
    "train/confirm_sd_cst_complete_physical_fresh.py",
    "train/assess_sd_cst_complete_physical_confirmation.py",
    "train/jobs/sd_cst_complete_physical_confirmation.sbatch",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def evaluator_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
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
        raise RuntimeError("confirmation evaluator commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("confirmation evaluator is not an ancestor of HEAD")
    hashes = {}
    for relative in EVALUATOR_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"confirmation evaluator omits path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"confirmation evaluator runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"confirmation JSON is not an object: {path}")
    return value


def _verify_authorization(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if sha256_file(args.data_dir / "report.json") != BOARD_REPORT_SHA256:
        raise RuntimeError("confirmation board report differs")
    board = _load_json(args.data_dir / "report.json")
    if (
        board.get("schema") != BOARD_SCHEMA
        or board.get("protocol") != PROTOCOL
        or board.get("source_commit") != SCIENTIFIC_SOURCE_COMMIT
        or board.get("all_gates_pass") is not True
        or board.get("development_accesses") != 0
        or board.get("confirmation_accesses") != 0
        or board["files"]["confirmation.sealed.jsonl"]["sha256"]
        != CONFIRMATION_SHA256
    ):
        raise RuntimeError("confirmation board identity differs")
    expected = {
        args.checkpoint: CHECKPOINT_SHA256,
        args.gate_config: GATE_CONFIG_SHA256,
        args.development_report: DEVELOPMENT_REPORT_SHA256,
        args.development_assessment: DEVELOPMENT_ASSESSMENT_SHA256,
        args.development_ledger: DEVELOPMENT_LEDGER_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"confirmation authorization artifact differs: {path}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = _load_json(args.gate_config)
    assessment = _load_json(args.development_assessment)
    ledger = _load_json(args.development_ledger)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or config.get("schema") != CONFIG_SCHEMA
        or config.get("protocol") != PROTOCOL
        or config.get("confirmation_sha256") != CONFIRMATION_SHA256
        or assessment.get("schema") != DEVELOPMENT_ASSESSMENT_SCHEMA
        or assessment.get("protocol") != PROTOCOL
        or assessment.get("decision") != "authorize_one_sealed_confirmation"
        or assessment.get("all_gates_pass") is not True
        or assessment.get("custody")
        != {"development_accesses": 1, "confirmation_accesses": 0}
        or ledger.get("split") != "sd_cst_development"
        or ledger.get("access_number") != 1
    ):
        raise RuntimeError("development authorization is not exact")
    access_files = sorted((args.data_dir / "access").glob("*.json"))
    if access_files != [args.development_ledger]:
        raise RuntimeError("confirmation pre-access ledger set differs")
    return board, checkpoint, assessment


def _consume_confirmation_access(
    data_dir: Path,
    board: Mapping[str, object],
    evaluator_source_commit: str,
) -> dict[str, str]:
    payload = (
        json.dumps(
            {
                "schema": ACCESS_SCHEMA,
                "protocol": PROTOCOL,
                "split": CONFIRMATION_SPLIT,
                "board_report_sha256": BOARD_REPORT_SHA256,
                "split_sha256": CONFIRMATION_SHA256,
                "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
                "evaluator_source_commit": evaluator_source_commit,
                "access_number": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path = data_dir / "access" / f"complete_physical_fresh_confirmation_{CONFIRMATION_SHA256}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _metrics(compiled, executor, gold, rows):
    final = executor["gold"]["final_state"]
    answer = executor["gold"]["answer"]
    arm_metrics = {}
    arm_exact = {}
    for arm in ("treatment", "row_shuffled_labels"):
        fields = _exact_packet((compiled[arm]["tape"], compiled[arm]["query"]), gold)
        pointers = _pointer_exact(compiled[arm]["pointers"], rows)
        state = executor[arm]["final_state"].eq(final)
        answer_ok = executor[arm]["answer"].eq(answer)
        exact = fields | pointers | {"state": state, "answer": answer_ok, "joint": state & answer_ok}
        arm_exact[arm] = exact
        arm_metrics[arm] = {
            "overall": _summary(exact),
            "packet_by_renderer": _grouped(fields["packet"], rows, "variant"),
            "joint_by_renderer": _grouped(exact["joint"], rows, "variant"),
            "packet_by_depth": _grouped(fields["packet"], rows, "halt_after"),
            "joint_by_depth": _grouped(exact["joint"], rows, "halt_after"),
            "source_poison_bit_identical": bool(compiled[arm]["source_poison_bit_identical"]),
        }
    controls = {
        name: {
            "state_rate": _rate(value["final_state"].eq(final)),
            "answer_rate": _rate(value["answer"].eq(answer)),
        }
        for name, value in executor.items()
        if name not in ("treatment", "row_shuffled_labels", "gold")
    }
    return arm_metrics, arm_exact, controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--development-assessment", type=Path, required=True)
    parser.add_argument("--development-ledger", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--execution-core", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluator-source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing confirmation output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("complete physical confirmation requires bf16 CUDA")
    evaluator = evaluator_manifest(args.repo_root.resolve(), args.evaluator_source_commit)
    scientific = source_manifest(args.repo_root.resolve(), SCIENTIFIC_SOURCE_COMMIT)
    board, checkpoint, development_assessment = _verify_authorization(args)
    args.out_dir.mkdir(parents=True)
    authorization_path = args.out_dir / "authorization.json"
    authorization = {
        "schema": "r12_sd_cst_complete_physical_fresh_confirmation_authorization_v1_3",
        "protocol": PROTOCOL,
        "scientific_source": scientific,
        "evaluator_source": evaluator,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gate_config_sha256": GATE_CONFIG_SHA256,
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "development_assessment_sha256": DEVELOPMENT_ASSESSMENT_SHA256,
        "development_ledger_sha256": DEVELOPMENT_LEDGER_SHA256,
        "development_decision": development_assessment["decision"],
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    ledger = _consume_confirmation_access(args.data_dir, board, args.evaluator_source_commit)
    confirmation_path = args.data_dir / "confirmation.sealed.jsonl"
    if sha256_file(confirmation_path) != CONFIRMATION_SHA256:
        raise RuntimeError("sealed confirmation hash differs after access")
    rows = load_rows(confirmation_path, CONFIRMATION_SPLIT)
    device = torch.device("cuda")
    compiled = {}
    for arm in ("treatment", "row_shuffled_labels"):
        model, parameters = _load_arm(args, checkpoint["arms"][arm]["trainable_state"], device)
        compiled[arm] = compile_rows(model, rows, args.batch_size, device)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    gold = expected_tape(rows)
    evidence_path = args.out_dir / "confirmation_evidence.pt"
    _save_evidence(evidence_path, compiled, rows)
    executor, packet_path, executor_path = _execute(args, args.out_dir, compiled, gold)
    confirmation_packet_path = args.out_dir / "confirmation_packets.pt"
    confirmation_executor_path = args.out_dir / "confirmation_executor.pt"
    packet_path.replace(confirmation_packet_path)
    executor_path.replace(confirmation_executor_path)
    packet_path = confirmation_packet_path
    executor_path = confirmation_executor_path
    metrics, exact, controls = _metrics(compiled, executor, gold, rows)
    expected_final = torch.tensor([row.final_state for row in rows], dtype=torch.uint8)
    expected_answer = torch.tensor([row.answer_role for row in rows], dtype=torch.long)
    treatment = metrics["treatment"]
    shuffled = metrics["row_shuffled_labels"]
    post_stop = all(
        torch.equal(executor["treatment"][name], executor["post_stop_perturbation"][name])
        for name in ("final_state", "answer", "state_trajectory", "alive_trajectory")
    )
    gates = {
        "development_authorization_exact": True,
        "fit_packet_min_renderer_at_least_99pct": _minimum_fit_packet(checkpoint["arms"]["treatment"]["fit"]) >= THRESHOLDS["fit_packet_min_renderer"],
        "packet_overall_at_least_90pct": treatment["overall"]["packet"]["rate"] >= THRESHOLDS["packet_overall"],
        "packet_min_renderer_at_least_85pct": min(v["rate"] for v in treatment["packet_by_renderer"].values()) >= THRESHOLDS["packet_min_renderer"],
        "state_answer_joint_at_least_90pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS[f"{name}_overall"] for name in ("state", "answer", "joint")),
        "joint_min_renderer_at_least_85pct": min(v["rate"] for v in treatment["joint_by_renderer"].values()) >= THRESHOLDS["joint_min_renderer"],
        "all_packet_fields_at_least_95pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS["field_overall"] for name in ("initial", "kind", "identity", "amount", "query")),
        "all_pointers_at_least_90pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS["pointer_overall"] for name in ("line", "binding", "initial_entity", "event_entity")),
        "treatment_packet_advantage_at_least_50pp": treatment["overall"]["packet"]["rate"] - shuffled["overall"]["packet"]["rate"] >= THRESHOLDS["treatment_packet_advantage"],
        "row_shuffled_packet_at_most_25pct": shuffled["overall"]["packet"]["rate"] <= THRESHOLDS["row_shuffled_packet_max"],
        "gold_executor_exact": bool(
            executor["gold"]["final_state"].eq(expected_final).all()
            and executor["gold"]["answer"].eq(expected_answer).all()
        ),
        "conditional_execution_exact": bool(exact["treatment"]["joint"][exact["treatment"]["packet"]].all()),
        "post_stop_perturbation_invariant": post_stop,
        "shuffled_packet_state_at_most_35pct": controls["shuffled_packet"]["state_rate"] <= THRESHOLDS["negative_state_max"],
        "reset_and_freeze_state_at_most_75pct": all(controls[name]["state_rate"] <= THRESHOLDS["reset_freeze_state_max"] for name in ("reset", "freeze")),
        "source_deleted_before_separate_execution": all(bool(compiled[arm]["source_poison_bit_identical"]) for arm in compiled),
        "complete_system_below_200m": int(parameters["complete_system"]) < 200_000_000,
        "frozen_state_unchanged": all(checkpoint["arms"][arm]["fit"]["frozen_parent_unchanged"] for arm in ("treatment", "row_shuffled_labels")),
        "confirmation_one_after_development_one": True,
    }
    decision = "confirm_complete_physical_fresh_v1_3" if all(gates.values()) else "reject_complete_physical_fresh_confirmation_v1_3"
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "metrics": metrics,
        "controls": controls,
        "custody": {"development_accesses": 1, "confirmation_accesses": 1, "confirmation_ledger": ledger},
        "authorization": authorization,
        "artifacts": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "authorization_sha256": sha256_file(authorization_path),
            "packet_sha256": sha256_file(packet_path),
            "evidence_sha256": sha256_file(evidence_path),
            "executor_sha256": sha256_file(executor_path),
        },
        "runtime": runtime_manifest(),
        "claim_boundary": "Confirmed fresh finite renderer/name compilation into a source-deleted categorical executor; not broad natural-language or general reasoning.",
    }
    report_path = args.out_dir / "confirmation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "report": str(report_path), "sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
