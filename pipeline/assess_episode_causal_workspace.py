#!/usr/bin/env python3
"""One-access assessment of frozen label-blind EPISODE predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path

import torch

import pipeline.episode_workspace_custody as custody_module
from pipeline.episode_workspace_custody import (
    ASSESSOR_ROW_SCHEMA,
    DEFAULT_CUSTODY_BUNDLE,
    abort_atomic_bundle,
    atomic_bundle_directory,
    canonical_json,
    committed_source_receipt,
    EpisodeCustodyError,
    file_sha256,
    finish_atomic_bundle,
    fsync_directory,
    read_json_verified,
    read_jsonl_verified,
    verify_landlock_stage,
    write_json_fsync,
    write_jsonl_fsync,
)


ASSESSMENT_SCHEMA = "episode_causal_workspace_assessment_v1"
ASSESSMENT_BUNDLE_SCHEMA = "episode_causal_workspace_assessment_bundle_v1"
EXECUTION_BUNDLE_SCHEMA = "episode_causal_workspace_execution_bundle_v1"
LOGITS_SCHEMA = "episode_causal_workspace_answer_logits_v1"
CONTROL_NAMES = (
    "treatment",
    "zero_workspace",
    "uniform_binding",
    "uniform_operator",
    "binding_permutation",
    "operator_permutation",
    "selected_slot_scramble",
    "discarded_slot_scramble",
)


class EpisodeWorkspaceAssessmentError(ValueError):
    """An assessor input, metric, or publication invariant failed."""


def _read_assessor_once(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Hash and parse the assessor ledger from exactly one file open."""

    with path.open("rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise EpisodeWorkspaceAssessmentError(
            f"assessor ledger hash mismatch: {actual}, expected {expected_sha256}"
        )
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeWorkspaceAssessmentError(
                f"assessor ledger line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise EpisodeWorkspaceAssessmentError(
                f"assessor ledger line {line_number} is not an object"
            )
        rows.append(value)
    return tuple(rows)


def load_assessor_rows(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]:
    rows = _read_assessor_once(path, expected_sha256)
    if len(rows) != 384:
        raise EpisodeWorkspaceAssessmentError("assessor ledger must contain 384 rows")
    seen: set[str] = set()
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    worlds: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if set(row) != {
            "schema",
            "packet_sha256",
            "target_token",
            "state_tokens",
            "cluster_id",
            "cluster_index",
            "query_variant",
            "binding_shift",
            "query_depth",
            "world_id",
        }:
            raise EpisodeWorkspaceAssessmentError("assessor row has unexpected fields")
        if row.get("schema") != ASSESSOR_ROW_SCHEMA:
            raise EpisodeWorkspaceAssessmentError("assessor row schema is invalid")
        digest = row.get("packet_sha256")
        cluster_id = row.get("cluster_id")
        world_id = row.get("world_id")
        target = row.get("target_token")
        state_tokens = row.get("state_tokens")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EpisodeWorkspaceAssessmentError("packet digest is invalid")
        if digest in seen:
            raise EpisodeWorkspaceAssessmentError(
                "assessor packet digest is duplicated"
            )
        seen.add(digest)
        if not isinstance(cluster_id, str) or not cluster_id:
            raise EpisodeWorkspaceAssessmentError("cluster ID is invalid")
        if not isinstance(world_id, str) or len(world_id) != 64:
            raise EpisodeWorkspaceAssessmentError("world ID is invalid")
        if not isinstance(target, int) or isinstance(target, bool):
            raise EpisodeWorkspaceAssessmentError("target token is invalid")
        if (
            not isinstance(state_tokens, list)
            or len(state_tokens) != 8
            or len(set(state_tokens)) != 8
            or target not in state_tokens
        ):
            raise EpisodeWorkspaceAssessmentError("candidate-state domain is invalid")
        if row.get("query_variant") not in {"primary", "reordered"}:
            raise EpisodeWorkspaceAssessmentError("query variant is invalid")
        if row.get("binding_shift") not in {0, 1, 2}:
            raise EpisodeWorkspaceAssessmentError("binding shift is invalid")
        if row.get("query_depth") not in {5, 6}:
            raise EpisodeWorkspaceAssessmentError("query depth is invalid")
        clusters[cluster_id].append(row)
        worlds[world_id].append(row)
    if len(clusters) != 64 or any(len(values) != 6 for values in clusters.values()):
        raise EpisodeWorkspaceAssessmentError(
            "assessor six-case clusters are incomplete"
        )
    if len(worlds) != 192 or any(len(values) != 2 for values in worlds.values()):
        raise EpisodeWorkspaceAssessmentError(
            "assessor late-query world pairs are incomplete"
        )
    return rows


def _torch_load_verified(path: Path, expected_sha256: str) -> object:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise EpisodeWorkspaceAssessmentError(
                f"{path.name} hash mismatch: {actual}, expected {expected_sha256}"
            )
        handle.seek(0)
        return torch.load(handle, map_location="cpu", weights_only=True)


def load_execution_bundle(
    path: Path,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, dict[str, object]],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    manifest_path = path / "bundle_manifest.json"
    try:
        manifest = read_json_verified(manifest_path, expected_manifest_sha256)
    except EpisodeCustodyError as exc:
        raise EpisodeWorkspaceAssessmentError(str(exc)) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != EXECUTION_BUNDLE_SCHEMA
    ):
        raise EpisodeWorkspaceAssessmentError("execution bundle manifest is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EpisodeWorkspaceAssessmentError("execution bundle file ledger is missing")
    expected_files = {
        "label_blind_predictions.jsonl",
        "execution_report.json",
        *(f"answer_logits_{name}.pt" for name in CONTROL_NAMES),
    }
    if set(files) != expected_files:
        raise EpisodeWorkspaceAssessmentError("execution bundle file set differs")
    if any(not isinstance(expected, str) for expected in files.values()):
        raise EpisodeWorkspaceAssessmentError("execution artifact hash is invalid")
    try:
        predictions = read_jsonl_verified(
            path / "label_blind_predictions.jsonl",
            str(files["label_blind_predictions.jsonl"]),
        )
    except EpisodeCustodyError as exc:
        raise EpisodeWorkspaceAssessmentError(str(exc)) from exc
    for value in predictions:
        if set(value) != {
            "control",
            "packet_sha256",
            "predicted_token",
        }:
            raise EpisodeWorkspaceAssessmentError(
                "label-blind prediction row is invalid"
            )
    if len(predictions) != 384 * len(CONTROL_NAMES):
        raise EpisodeWorkspaceAssessmentError(
            "label-blind prediction cardinality drifted"
        )
    logits: dict[str, dict[str, object]] = {}
    for name in CONTROL_NAMES:
        payload = _torch_load_verified(
            path / f"answer_logits_{name}.pt",
            str(files[f"answer_logits_{name}.pt"]),
        )
        if not isinstance(payload, dict) or payload.get("schema") != LOGITS_SCHEMA:
            raise EpisodeWorkspaceAssessmentError(f"{name} logits payload is invalid")
        if payload.get("control") != name:
            raise EpisodeWorkspaceAssessmentError(f"{name} logits control differs")
        if (
            payload.get("targets_seen") is not False
            or payload.get("candidate_sets_seen") is not False
            or payload.get("pretraining_started") is not False
        ):
            raise EpisodeWorkspaceAssessmentError(
                f"{name} logits custody flags are invalid"
            )
        packet_order = payload.get("packet_sha256")
        answer_logits = payload.get("answer_logits")
        if (
            not isinstance(packet_order, list)
            or len(packet_order) != 384
            or len(set(packet_order)) != 384
            or not isinstance(answer_logits, torch.Tensor)
            or answer_logits.shape != (384, 32768)
            or not torch.isfinite(answer_logits).all()
        ):
            raise EpisodeWorkspaceAssessmentError(
                f"{name} logits dimensions are invalid"
            )
        logits[name] = payload
    try:
        report = read_json_verified(
            path / "execution_report.json",
            str(files["execution_report.json"]),
        )
    except EpisodeCustodyError as exc:
        raise EpisodeWorkspaceAssessmentError(str(exc)) from exc
    if (
        not isinstance(report, dict)
        or report.get("targets_seen") is not False
        or report.get("candidate_sets_seen") is not False
        or report.get("world_tokens_seen") is not False
        or report.get("pretraining_started") is not False
    ):
        raise EpisodeWorkspaceAssessmentError(
            "execution report custody flags are invalid"
        )
    return logits, tuple(predictions), report


def _rate(correct: int, total: int) -> dict[str, object]:
    if total <= 0 or not 0 <= correct <= total:
        raise EpisodeWorkspaceAssessmentError("metric denominator is invalid")
    return {"correct": correct, "total": total, "rate": correct / total}


def summarize_control(
    assessor_rows: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
    logits_payload: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predicted = {
        str(row["packet_sha256"]): int(row["predicted_token"])
        for row in prediction_rows
    }
    if len(predicted) != 384:
        raise EpisodeWorkspaceAssessmentError("one control has duplicate predictions")
    packet_order = logits_payload["packet_sha256"]
    answer_logits = logits_payload["answer_logits"]
    if not isinstance(packet_order, list) or not isinstance(
        answer_logits, torch.Tensor
    ):
        raise EpisodeWorkspaceAssessmentError("logits payload types differ")
    logit_by_digest = {
        str(digest): answer_logits[index] for index, digest in enumerate(packet_order)
    }
    if set(predicted) != set(logit_by_digest) or set(predicted) != {
        str(row["packet_sha256"]) for row in assessor_rows
    }:
        raise EpisodeWorkspaceAssessmentError(
            "prediction/label/logit digest coverage differs"
        )
    assessed: list[dict[str, object]] = []
    total_nll = 0.0
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in assessor_rows:
        digest = str(row["packet_sha256"])
        token = predicted[digest]
        logits = logit_by_digest[digest]
        if int(logits.argmax().item()) != token:
            raise EpisodeWorkspaceAssessmentError(
                "prediction disagrees with frozen logits"
            )
        target = int(row["target_token"])
        nll = float(torch.logsumexp(logits, dim=-1) - logits[target])
        correct = token == target
        candidate = token in row["state_tokens"]
        value = {
            "packet_sha256": digest,
            "predicted_token": token,
            "target_token": target,
            "correct": correct,
            "candidate_state": candidate,
            "answer_position_nll": nll,
            "cluster_id": row["cluster_id"],
            "query_variant": row["query_variant"],
            "binding_shift": row["binding_shift"],
            "query_depth": row["query_depth"],
            "world_id": row["world_id"],
        }
        assessed.append(value)
        clusters[str(row["cluster_id"])].append(value)
        total_nll += nll
    packet_correct = sum(bool(row["correct"]) for row in assessed)
    candidate_count = sum(bool(row["candidate_state"]) for row in assessed)
    exact_clusters = 0
    exact_triples = 0
    exact_pairs = 0
    for values in clusters.values():
        cells = {
            (str(row["query_variant"]), int(row["binding_shift"])): bool(row["correct"])
            for row in values
        }
        exact_clusters += int(all(cells.values()))
        exact_triples += sum(
            int(all(cells[(variant, shift)] for shift in range(3)))
            for variant in ("primary", "reordered")
        )
        exact_pairs += sum(
            int(cells[("primary", shift)] and cells[("reordered", shift)])
            for shift in range(3)
        )

    def subset_rate(key: str, value: object) -> dict[str, object]:
        subset = [row for row in assessed if row[key] == value]
        return _rate(sum(bool(row["correct"]) for row in subset), len(subset))

    summary = {
        "packets": _rate(packet_correct, len(assessed)),
        "candidate_state_predictions": _rate(candidate_count, len(assessed)),
        "complete_six_case_clusters": _rate(exact_clusters, len(clusters)),
        "complete_cyclic_triples": _rate(exact_triples, len(clusters) * 2),
        "complete_order_pairs": _rate(exact_pairs, len(clusters) * 3),
        "by_depth": {
            "5": subset_rate("query_depth", 5),
            "6": subset_rate("query_depth", 6),
        },
        "by_query_variant": {
            "primary": subset_rate("query_variant", "primary"),
            "reordered": subset_rate("query_variant", "reordered"),
        },
        "by_binding_shift": {
            str(shift): subset_rate("binding_shift", shift) for shift in range(3)
        },
        "answer_position_nll": total_nll / len(assessed),
        "hard_argmax": True,
        "retry_or_repair": False,
        "candidate_mask": False,
    }
    return summary, assessed


def _metric(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, Mapping) or not isinstance(
        value.get("rate"), (int, float)
    ):
        raise EpisodeWorkspaceAssessmentError(f"metric {key} is missing")
    return float(value["rate"])


def promotion_diagnostics(
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    treatment = summaries["treatment"]
    treatment_packet = _metric(treatment, "packets")
    selected_cost = treatment_packet - _metric(
        summaries["selected_slot_scramble"], "packets"
    )
    discarded_cost = treatment_packet - _metric(
        summaries["discarded_slot_scramble"], "packets"
    )
    depth = treatment.get("by_depth")
    if not isinstance(depth, Mapping):
        raise EpisodeWorkspaceAssessmentError("depth metrics are missing")
    depth_floor = min(
        _metric(depth, "5"),
        _metric(depth, "6"),
    )
    gates = {
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
        "shuffled_target_training_at_most_40_percent": False,
        "process_level_source_deletion": False,
        "post_compile_source_poison_bit_identity": False,
        "three_fresh_seed_replication": False,
        "unopened_confirmation_manifest": False,
    }
    return {
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "selected_slot_scramble_cost": selected_cost,
        "discarded_slot_scramble_cost": discarded_cost,
        "depth_five_six_floor": depth_floor,
        "reasoning_promotion_authorized": False,
        "continuation_pretraining_authorized": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    landlock_receipt = verify_landlock_stage("assessor", args.deny_probe)
    try:
        source_before = committed_source_receipt(
            Path(__file__).resolve(),
            args.expected_source_sha256,
            (Path(custody_module.__file__),),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceAssessmentError(str(exc)) from exc
    assessor_rows = load_assessor_rows(
        args.assessor,
        args.expected_assessor_sha256,
    )
    logits, prediction_rows, execution_report = load_execution_bundle(
        args.execution_bundle,
        args.expected_execution_manifest_sha256,
    )
    by_control: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        control = row.get("control")
        if control not in CONTROL_NAMES:
            raise EpisodeWorkspaceAssessmentError("prediction control is invalid")
        by_control[str(control)].append(row)
    if set(by_control) != set(CONTROL_NAMES):
        raise EpisodeWorkspaceAssessmentError("prediction controls are incomplete")
    summaries: dict[str, dict[str, object]] = {}
    assessed_rows: list[dict[str, object]] = []
    for control in CONTROL_NAMES:
        summary, rows = summarize_control(
            assessor_rows,
            by_control[control],
            logits[control],
        )
        summaries[control] = summary
        assessed_rows.extend({"control": control, **row} for row in rows)
        print(
            canonical_json(
                {
                    "event": "one_access_assessment",
                    "control": control,
                    "packets": summary["packets"],
                    "clusters": summary["complete_six_case_clusters"],
                }
            ),
            flush=True,
        )
    diagnostics = promotion_diagnostics(summaries)
    try:
        source_after = committed_source_receipt(
            Path(__file__).resolve(),
            args.expected_source_sha256,
            (Path(custody_module.__file__),),
        )
    except ValueError as exc:
        raise EpisodeWorkspaceAssessmentError(str(exc)) from exc
    if source_after != source_before:
        raise EpisodeWorkspaceAssessmentError(
            "assessor source receipt changed during assessment"
        )
    report = {
        "schema": ASSESSMENT_SCHEMA,
        "claim_scope": (
            "inspected synthetic development tuning assessment only; "
            "not unopened confirmation, broad reasoning, language reasoning, "
            "or continuation pretraining"
        ),
        "source": source_before,
        "process_id": os.getpid(),
        "landlock_receipt": landlock_receipt,
        "assessor_source": {
            "path": str(args.assessor.absolute()),
            "sha256": args.expected_assessor_sha256,
            "open_count": 1,
            "rows": len(assessor_rows),
        },
        "execution_bundle": {
            "path": str(args.execution_bundle.absolute()),
            "manifest_sha256": args.expected_execution_manifest_sha256,
        },
        "execution_report": execution_report,
        "control_summaries": summaries,
        "promotion_diagnostics": diagnostics,
        "scored_split_opened": True,
        "optimizer_state_serialized": False,
        "pretraining_started": False,
        "continuation_pretraining_authorized": False,
    }
    staging, lock = atomic_bundle_directory(args.output)
    try:
        report_path = staging / "assessment_report.json"
        write_json_fsync(report_path, report)
        assessed_path = staging / "assessed_predictions.jsonl"
        write_jsonl_fsync(assessed_path, assessed_rows)
        manifest = {
            "schema": ASSESSMENT_BUNDLE_SCHEMA,
            "files": {
                "assessment_report.json": file_sha256(report_path),
                "assessed_predictions.jsonl": file_sha256(assessed_path),
            },
            "assessor_source_open_count": 1,
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        }
        write_json_fsync(staging / "bundle_manifest.json", manifest)
        fsync_directory(staging)
        finish_atomic_bundle(staging, args.output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assessor",
        type=Path,
        default=DEFAULT_CUSTODY_BUNDLE / "development_assessor.jsonl",
    )
    parser.add_argument("--expected-assessor-sha256", required=True)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--expected-execution-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "treatment": report["control_summaries"]["treatment"],
                "promotion_diagnostics": report["promotion_diagnostics"],
                "pretraining_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
