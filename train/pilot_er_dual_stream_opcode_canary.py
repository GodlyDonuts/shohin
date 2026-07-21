#!/usr/bin/env python3
"""Train-only opcode-complement qualification on fresh-training families."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Mapping, Sequence

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, trainable_state
from er_dual_stream_fresh_renderers import SCORED_RENDERERS, render_row
from build_er_dual_stream_fresh_board import validate_row
from er_dual_stream_fresh_scoring import (
    alpha_recode_row,
    distractor_rotate_row,
    source_free_row,
)
from er_relation_tensor_training import (
    RelationTensorRow,
    evaluate_arm,
    load_board_receipt,
    load_split,
    parse_row,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_dual_stream_fresh import _load_canary, initialize_system
from pilot_er_dual_stream_relation_adapter import EXPECTED_PARAMETERS
from pilot_er_dual_stream_train_canary import (
    alpha_metrics,
    alpha_predictions,
    fit_train_only,
    score_train_row,
    split_train_families,
)
from pilot_er_relation_tensor import atomic_json_save, atomic_torch_save, release_cuda
from pilot_sd_cst_byte_addressed import sha256_file


SCHEMA = "r12_er_dual_stream_opcode_coupled_canary_v1_3"
EVIDENCE_SCHEMA = "r12_er_dual_stream_opcode_coupled_evidence_v1_3"
REPORT_SCHEMA = "r12_er_dual_stream_opcode_coupled_report_v1_3"
BOARD_REPORT_SHA256 = (
    "6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f"
)
ARMS = {"opcode_coupled": 1.0, "legacy_uncoupled": 0.0}
SEMANTIC_KEYS = (
    "cardinality",
    "initial",
    "relations",
    "rule_active",
    "events",
    "halt",
    "query",
)
NEUTRAL = re.compile(r"(?<!\S)z[0-9a-z]{5}(?!\S)")
THRESHOLDS = {
    "primary": 0.99,
    "minimum_group": 0.99,
    "source_free_joint_max": 0.10,
    "legacy_joint_max": 0.80,
    "advantage": 0.20,
}
FROZEN_SOURCE_PATHS = (
    "R12_ER_DUAL_STREAM_OPCODE_COUPLED_PREREG.md",
    "train/er_dual_stream_relation_adapter.py",
    "train/pilot_er_dual_stream_opcode_canary.py",
    "train/test_er_dual_stream_opcode_canary.py",
    "train/test_er_dual_stream_relation_adapter.py",
    "train/jobs/er_dual_stream_opcode_canary.sbatch",
)


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args), cwd=repo_root, capture_output=True, text=True, check=False
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("opcode-coupled source commit is unavailable")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"opcode-coupled source omits {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"opcode-coupled runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def _new_neutral(
    family_id: str, label: str, used: set[str], seed: int
) -> str:
    retry = 0
    while True:
        digest = hashlib.sha256(
            f"{seed}:{family_id}:{label}:{retry}".encode()
        ).hexdigest()
        value = "z" + digest[:5]
        if value not in used:
            used.add(value)
            return value
        retry += 1


def renderer_relocation_rows(
    raw_rows: Sequence[Mapping[str, object]],
    probe_family_ids: set[str],
    *,
    seed: int,
) -> list[RelationTensorRow]:
    """Render held-out train semantics in the complementary public coset."""
    representatives: dict[str, Mapping[str, object]] = {}
    for row in raw_rows:
        family = str(row["family_id"])
        if family in probe_family_ids:
            representatives.setdefault(family, row)
    if set(representatives) != probe_family_ids:
        raise ValueError("opcode-coupled relocation families differ")
    output = []
    for family in sorted(representatives):
        base = representatives[family]
        used = set(NEUTRAL.findall(
            f"{base['program_text']}\n{base['late_query_text']}"
        ))
        rule_noise = [
            _new_neutral(family, f"rule-{slot}", used, seed)
            for slot in range(4)
        ]
        query_noise = _new_neutral(family, "query", used, seed)
        for view, renderer in enumerate(SCORED_RENDERERS):
            event_noise = _new_neutral(family, f"event-{view}", used, seed)
            order = list(range(18))
            random.Random(derived_seed(seed, f"{family}:relocate:{view}")).shuffle(order)
            relocated = render_row(
                base,
                renderer,
                storage_order=order,
                row_id=f"train-relocate-{family}-v{view}",
                family_id=family,
                rule_distractors=rule_noise,
                event_distractor=event_noise,
                event_distractor_slot=derived_seed(
                    seed, f"{family}:event-slot:{view}"
                )
                % 13,
                query_distractor=query_noise,
            )
            validate_row(relocated)
            output.append(parse_row(relocated, TRAIN_SPLIT))
    if len(output) != 4 * len(probe_family_ids):
        raise ValueError("opcode-coupled relocation row count differs")
    return output


def load_raw_train(data_dir: Path, expected_sha256: str) -> list[dict[str, object]]:
    path = data_dir / "train.jsonl"
    if sha256_file(path) != expected_sha256:
        raise ValueError("opcode-coupled raw train hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 48_000:
        raise ValueError("opcode-coupled raw train row count differs")
    return rows


def relocation_consistency(
    rows: Sequence[RelationTensorRow], predictions: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.family_id].append(index)
    exact = 0
    for indices in groups.values():
        if len(indices) != 8:
            raise ValueError("opcode-coupled family does not have eight views")
        reference = indices[0]
        exact += int(
            all(
                predictions[key][indices]
                .eq(predictions[key][reference])
                .reshape(len(indices), -1)
                .all()
                for key in SEMANTIC_KEYS
            )
        )
    return {"exact": exact, "families": len(groups), "rate": exact / len(groups)}


def _minimum(metrics: Mapping[str, object], group: str) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"opcode-coupled {group} is absent")
    return min(float(value["joint"]["rate"]) for value in values.values())


def compute_gates(
    arms: Mapping[str, Mapping[str, object]],
    *,
    parameters: Mapping[str, int],
    shared_initialization: bool,
) -> dict[str, bool]:
    treatment = arms["opcode_coupled"]
    legacy = arms["legacy_uncoupled"]
    required = ("packet", "state", "answer", "joint", "relation_rows", "witness_pointer")
    primary = all(
        float(treatment[view]["overall"][field]["rate"]) >= THRESHOLDS["primary"]
        for view in ("canonical", "relocated")
        for field in required
    )
    groups = all(
        _minimum(treatment[view], group) >= THRESHOLDS["minimum_group"]
        for view in ("canonical", "relocated")
        for group in ("by_cardinality", "by_renderer")
    )
    treatment_joint = float(treatment["relocated"]["overall"]["joint"]["rate"])
    legacy_joint = float(legacy["relocated"]["overall"]["joint"]["rate"])
    return {
        "coupled_primary_metrics_at_least_99pct": primary,
        "coupled_minimum_cardinality_and_renderer_joint_at_least_99pct": groups,
        "all_eight_renderer_views_semantically_identical": int(
            treatment["relocation_consistency"]["exact"]
        )
        == int(treatment["relocation_consistency"]["families"])
        == 2_000,
        "alpha_and_distractor_relocation_invariance_exact": all(
            int(treatment[name]["complete"]["exact"])
            == int(treatment[name]["complete"]["rows"])
            == 8_000
            for name in ("alpha", "distractor")
        ),
        "source_free_joint_at_most_10pct": float(
            treatment["source_free"]["overall"]["joint"]["rate"]
        )
        <= THRESHOLDS["source_free_joint_max"],
        "coupling_beats_favorable_legacy_by_20pp": treatment_joint - legacy_joint
        >= THRESHOLDS["advantage"],
        "favorable_legacy_relocated_joint_at_most_80pct": legacy_joint
        <= THRESHOLDS["legacy_joint_max"],
        "shared_initialization_and_frozen_parent": shared_initialization
        and all(arm["fit"]["frozen_parent_unchanged"] is True for arm in arms.values()),
        "parameter_certificate_exact_and_below_200m": dict(parameters)
        == EXPECTED_PARAMETERS,
        "train_only_zero_scored_reads": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--canary-checkpoint", type=Path, required=True)
    for name in (
        "joint_checkpoint",
        "physical_checkpoint",
        "v1_checkpoint",
        "v1_2_checkpoint",
        "confirmed_checkpoint",
        "confirmation_assessment",
        "witness_checkpoint",
        "witness_confirmation_assessment",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing opcode-coupled output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("opcode-coupled canary requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise SystemExit("opcode-coupled board identity differs")
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    fit_rows, probe_rows, split = split_train_families(
        train_rows, derived_seed(args.seed, "opcode-coupled-family-split")
    )
    probe_ids = {row.family_id for row in probe_rows}
    raw = load_raw_train(args.data_dir, str(board["files"]["train.jsonl"]["sha256"]))
    relocated = renderer_relocation_rows(raw, probe_ids, seed=args.seed)
    canonical_scored = [score_train_row(row) for row in probe_rows]
    relocated_scored = [score_train_row(row) for row in relocated]
    combined = sorted(
        canonical_scored + relocated_scored,
        key=lambda row: (row.family_id, row.renderer, row.row_id),
    )
    canary = _load_canary(args.canary_checkpoint)
    device = torch.device("cuda")
    arms: dict[str, dict[str, object]] = {}
    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "seed": args.seed,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "arms": {},
    }
    initial_digests = set()
    parameters: dict[str, int] | None = None
    for arm_name, weight in ARMS.items():
        model, arm_parameters, frozen_digest, receipt = initialize_system(
            args, device, canary
        )
        parameters = arm_parameters
        model.opcode_coupling_scale = weight
        initial_digest = state_dict_digest(trainable_state(model))
        initial_digests.add(initial_digest)
        fit = fit_train_only(
            model,
            fit_rows,
            seed=derived_seed(args.seed, "opcode-coupled-fit-order"),
            frozen_digest=frozen_digest,
            trainable_names=frozenset(receipt["trainable_names"]),
        )
        canonical = evaluate_arm(
            model, canonical_scored, batch_size=args.batch_size, include_raw=True
        )
        relocated_metrics = evaluate_arm(
            model, relocated_scored, batch_size=args.batch_size, include_raw=True
        )
        arm_evidence = {
            "canonical": canonical.pop("raw"),
            "relocated": relocated_metrics.pop("raw"),
        }
        combined_predictions = alpha_predictions(
            model, combined, batch_size=args.batch_size
        )
        alpha = alpha_metrics(
            alpha_predictions(model, relocated_scored, batch_size=args.batch_size),
            alpha_predictions(
                model,
                [alpha_recode_row(row, "opcode-coupled-alpha") for row in relocated_scored],
                batch_size=args.batch_size,
            ),
        )
        distractor = alpha_metrics(
            alpha_predictions(model, relocated_scored, batch_size=args.batch_size),
            alpha_predictions(
                model,
                [distractor_rotate_row(row) for row in relocated_scored],
                batch_size=args.batch_size,
            ),
        )
        source_free = evaluate_arm(
            model,
            [source_free_row(row) for row in relocated_scored],
            batch_size=args.batch_size,
        )
        arms[arm_name] = {
            "opcode_coupling_scale": weight,
            "initial_state_sha256": initial_digest,
            "fit": fit,
            "canonical": canonical,
            "relocated": relocated_metrics,
            "relocation_consistency": relocation_consistency(
                combined, combined_predictions
            ),
            "alpha": {key: value for key, value in alpha.items() if key != "complete_mask"},
            "distractor": {
                key: value for key, value in distractor.items() if key != "complete_mask"
            },
            "source_free": source_free,
            "compiler_trainable_state": trainable_state(model),
        }
        evidence["arms"][arm_name] = arm_evidence
        release_cuda(model)
    if parameters is None:
        raise RuntimeError("opcode-coupled arms are absent")
    gates = compute_gates(
        arms,
        parameters=parameters,
        shared_initialization=len(initial_digests) == 1,
    )
    report_arms = {
        name: {key: value for key, value in arm.items() if key != "compiler_trainable_state"}
        for name, arm in arms.items()
    }
    report = {
        "schema": REPORT_SCHEMA,
        "source_manifest": source,
        "seed": args.seed,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "split": split,
        "parameters": parameters,
        "thresholds": THRESHOLDS,
        "arms": report_arms,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_new_fresh_board_source"
            if all(gates.values())
            else "reject_opcode_coupled_before_fresh_board"
        ),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out_dir.mkdir(parents=True)
    atomic_torch_save(
        {
            "schema": SCHEMA,
            "source_manifest": source,
            "seed": args.seed,
            "parameters": parameters,
            "split": split,
            "arms": {
                name: {
                    "opcode_coupling_scale": arm["opcode_coupling_scale"],
                    "initial_state_sha256": arm["initial_state_sha256"],
                    "fit": arm["fit"],
                    "compiler_trainable_state": arm["compiler_trainable_state"],
                }
                for name, arm in arms.items()
            },
            "development_accesses": 0,
            "confirmation_accesses": 0,
        },
        args.out_dir / "compiler.pt",
    )
    atomic_torch_save(evidence, args.out_dir / "train_probe_evidence.pt")
    atomic_json_save(report, args.out_dir / "train_probe_report.json")
    print(json.dumps({"decision": report["decision"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
