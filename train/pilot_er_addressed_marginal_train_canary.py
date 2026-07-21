#!/usr/bin/env python3
"""Train-only falsifier for occurrence-addressed marginal ER-TT routing."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, trainable_state
from er_relation_tensor_training import evaluate_arm, load_board_receipt, load_split
from pilot_er_addressed_marginal_relation_adapter import (
    EXPECTED_PARAMETERS,
    initialize_addressed_marginal_relation,
)
from pilot_er_dual_stream_train_canary import (
    BOARD_REPORT_SHA256,
    CONTRACT,
    FROZEN_SOURCE_PATHS as MARGINAL_FROZEN_SOURCE_PATHS,
    THRESHOLDS,
    alpha_metrics,
    alpha_predictions,
    atomic_json_save,
    atomic_torch_save,
    fit_train_only,
    oracle_route_transport_metrics,
    runtime_manifest,
    score_train_row,
    split_train_families,
    alpha_recode_row,
)
from pilot_sd_cst_byte_addressed import sha256_file


SCHEMA = "r12_er_addressed_marginal_train_only_canary_v1"
FROZEN_SOURCE_PATHS = tuple(
    sorted(
        set(
            MARGINAL_FROZEN_SOURCE_PATHS
            + (
                "R12_ER_ADDRESSED_MARGINAL_ROUTE_PREREG.md",
                "train/er_addressed_marginal_relation_adapter.py",
                "train/pilot_er_addressed_marginal_relation_adapter.py",
                "train/pilot_er_addressed_marginal_train_canary.py",
                "train/test_er_addressed_marginal_relation_adapter.py",
                "train/test_pilot_er_addressed_marginal_train_canary.py",
                "train/jobs/er_addressed_marginal_train_canary.sbatch",
            )
        )
    )
)


def compute_addressed_gates(
    metrics: Mapping[str, object],
    alpha: Mapping[str, object],
    parameters: Mapping[str, int],
    fit: Mapping[str, object],
    split: Mapping[str, object],
    oracle_route: Mapping[str, object],
) -> dict[str, bool]:
    """Apply the inherited scientific gates with this adapter's certificate."""
    overall = metrics["overall"]
    cardinality = metrics["by_cardinality"]
    return {
        "family_split_exact_and_disjoint": split
        == {
            **split,
            "fit_families": int(CONTRACT["fit_families"]),
            "probe_families": int(CONTRACT["probe_families"]),
            "fit_rows": int(CONTRACT["fit_rows"]),
            "probe_rows": int(CONTRACT["probe_rows"]),
            "family_overlap": 0,
        },
        "packet_state_answer_joint_at_least_85pct": all(
            float(overall[name]["rate"]) >= float(THRESHOLDS[name])
            for name in ("packet", "state", "answer", "joint")
        ),
        "relation_rows_at_least_90pct": float(
            overall["relation_rows"]["rate"]
        )
        >= float(THRESHOLDS["relation_rows"]),
        "witness_pointers_at_least_90pct": float(
            overall["witness_pointer"]["rate"]
        )
        >= float(THRESHOLDS["witness_pointer"]),
        "events_and_halt_at_least_95pct": all(
            float(overall[name]["rate"]) >= float(THRESHOLDS[name])
            for name in ("events", "halt")
        ),
        "minimum_cardinality_joint_at_least_75pct": min(
            float(value["joint"]["rate"])
            for value in cardinality.values()
        )
        >= float(THRESHOLDS["minimum_cardinality_joint"]),
        "all_hard_fields_and_pointers_alpha_exact": float(
            alpha["complete"]["rate"]
        )
        == float(THRESHOLDS["alpha_exact"]),
        "oracle_route_identity_transport_is_exact": all(
            float(oracle_route[name]["rate"])
            == float(THRESHOLDS["oracle_route_transport_exact"])
            for name in ("initial", "relations", "events", "joint")
        ),
        "confirmed_parent_unchanged": fit["frozen_parent_unchanged"] is True,
        "parameter_certificate_exact_and_below_200m": dict(parameters)
        == EXPECTED_PARAMETERS,
        "train_only_and_zero_scored_split_reads": CONTRACT[
            "outcome_supervision"
        ]
        is False
        and int(CONTRACT["development_reads"]) == 0
        and int(CONTRACT["confirmation_reads"]) == 0,
    }


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
        raise RuntimeError("addressed marginal scientific source is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("addressed marginal source is not an ancestor")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"addressed marginal source omits: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"addressed marginal runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
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
        raise SystemExit(f"refusing existing addressed output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("addressed marginal canary requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise SystemExit("addressed marginal board identity differs")
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    fit_rows, probe_rows, split_receipt = split_train_families(
        train_rows, derived_seed(args.seed, "dual-stream-train-probe-split")
    )
    if (
        len(fit_rows) != int(CONTRACT["fit_rows"])
        or len(probe_rows) != int(CONTRACT["probe_rows"])
    ):
        raise RuntimeError("addressed marginal row split differs")

    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda")
    model, parameters, frozen_digest, parent_receipt = (
        initialize_addressed_marginal_relation(
            joint_checkpoint=args.joint_checkpoint,
            physical_checkpoint=args.physical_checkpoint,
            v1_checkpoint=args.v1_checkpoint,
            v1_2_checkpoint=args.v1_2_checkpoint,
            confirmed_checkpoint=args.confirmed_checkpoint,
            confirmation_assessment=args.confirmation_assessment,
            witness_checkpoint=args.witness_checkpoint,
            witness_confirmation_assessment=(
                args.witness_confirmation_assessment
            ),
            seed=args.seed,
            device=device,
        )
    )
    trainable_names = frozenset(parent_receipt["trainable_names"])
    fit = fit_train_only(
        model,
        fit_rows,
        seed=derived_seed(args.seed, "addressed-marginal-fit-order"),
        frozen_digest=frozen_digest,
        trainable_names=trainable_names,
    )
    checkpoint = {
        "schema": SCHEMA,
        "source_manifest": source,
        "seed": args.seed,
        "contract": CONTRACT,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "split": split_receipt,
        "fit": fit,
        "compiler_trainable_state": trainable_state(model),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    checkpoint_path = args.out_dir / "compiler.pt"
    atomic_torch_save(checkpoint, checkpoint_path)

    scored_probe = [score_train_row(row) for row in probe_rows]
    metrics = evaluate_arm(
        model,
        scored_probe,
        batch_size=args.batch_size,
        include_raw=False,
        include_invariances=False,
    )
    oracle_route, oracle_predictions = oracle_route_transport_metrics(
        model, scored_probe, batch_size=args.batch_size
    )
    recoded_probe = [
        alpha_recode_row(row, "addressed-marginal-neutral-alpha")
        for row in scored_probe
    ]
    canonical_predictions = alpha_predictions(
        model, scored_probe, batch_size=args.batch_size
    )
    recoded_predictions = alpha_predictions(
        model, recoded_probe, batch_size=args.batch_size
    )
    alpha = alpha_metrics(canonical_predictions, recoded_predictions)
    complete_mask = alpha.pop("complete_mask")
    evidence = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "canonical_predictions": canonical_predictions,
        "recoded_predictions": recoded_predictions,
        "alpha_complete_mask": complete_mask,
        "oracle_route_predictions": oracle_predictions,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    evidence_path = args.out_dir / "train_probe_evidence.pt"
    atomic_torch_save(evidence, evidence_path)
    gates = compute_addressed_gates(
        metrics, alpha, parameters, fit, split_receipt, oracle_route
    )
    overall = metrics["overall"]
    report = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "source_manifest": source,
        "runtime": runtime_manifest(),
        "seed": args.seed,
        "contract": CONTRACT,
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "split": split_receipt,
        "fit": fit,
        "metrics": metrics,
        "routing_diagnostics": {
            "oracle_route": oracle_route,
            "soft_route": {
                name: overall[name]
                for name in (
                    "initial_rows",
                    "relation_rows",
                    "events",
                    "binding_pointer",
                    "initial_pointer",
                    "witness_pointer",
                )
            },
        },
        "alpha_invariance": alpha,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_fresh_addressed_marginal_board"
            if all(gates.values())
            else "reject_addressed_marginal_before_fresh_board"
        ),
        "artifacts": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evidence_sha256": sha256_file(evidence_path),
        },
        "custody": {
            "train_only_probe_accesses": 1,
            "development_accesses": 0,
            "confirmation_accesses": 0,
        },
        "claim_boundary": (
            "Passing admits only a fresh-board test of learned occurrence-address "
            "routing and exact identity transport. It is not a scored-split or "
            "broad reasoning claim."
        ),
    }
    report_path = args.out_dir / "train_probe_report.json"
    atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "evidence_sha256": sha256_file(evidence_path),
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
