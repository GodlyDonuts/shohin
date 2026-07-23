"""No-training trajectory audit for a completed frozen AHRF run.

The auditor reconstructs the board and architecture from the completed run
receipt, verifies every available custody hash, and performs exactly one
model-owned rollout per batch and evaluation mode. It never feeds an
intermediate state back to the model and never changes model parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

import torch

from autocatalytic_hysteretic_relation_field import (
    AHRFRollout,
    AutocatalyticHystereticRelationField,
)
from tensorize_contextual_ahrf import AHRF_NODE_FEATURE_DIM
from train_autocatalytic_hysteretic_relation_field import (
    SCORE_ARMS,
    SOURCE_PATHS,
    AHRFBoard,
    AHRFTrainConfig,
    _apply_control,
    _index_graph,
    _move_graph,
    _resolve_device,
    _root_facts,
    _source_receipt,
    _target_mask,
    _transfer_binder,
    build_board,
)


PROTOCOL = "autocatalytic_hysteretic_relation_field_v1"
AUDIT_PROTOCOL = "ahrf_post_run_trajectory_audit_v1"
MILESTONE_STEP = 64
MODES = {
    "soft_fixed_deadline": (False, False),
    "hard_fixed_deadline": (True, False),
    "hard_learned_halt": (True, True),
}
CHECKPOINT_KEYS = {
    "protocol",
    "config",
    "parameter_receipt",
    "model_state",
    "source_sha256",
    "warm_start",
}
REPORT_KEYS = {
    "protocol",
    "claim_boundary",
    "config",
    "device",
    "board",
    "parameter_receipt",
    "warm_start",
    "trace",
    "halt_trace",
    "train",
    "development",
    "development_fixed_deadline",
    "source_sha256",
    "checkpoint",
}


class AHRFTrajectoryAuditError(RuntimeError):
    """Raised when run custody or trajectory evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AHRFTrajectoryAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_normalize(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise AHRFTrajectoryAuditError("receipt is not JSON-compatible") from error


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AHRFTrajectoryAuditError(
            f"cannot load completed AHRF report: {path}"
        ) from error
    _require(isinstance(value, dict), "completed AHRF report is not an object")
    return value


def _existing_regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise AHRFTrajectoryAuditError(f"{label} is unavailable: {path}") from error
    _require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    return resolved


def _config_from_receipt(value: object) -> AHRFTrainConfig:
    _require(isinstance(value, dict), "AHRF config receipt is not an object")
    expected = {field.name for field in fields(AHRFTrainConfig)}
    _require(set(value) == expected, "AHRF config receipt fields drifted")
    try:
        config = AHRFTrainConfig(**value)
    except (TypeError, ValueError) as error:
        raise AHRFTrajectoryAuditError("AHRF config receipt is invalid") from error
    _require(config.max_steps >= MILESTONE_STEP, "AHRF horizon ends before step 64")
    return config


def _expected_board_receipt(
    board: AHRFBoard,
    train_indices: Sequence[int],
    development_indices: Sequence[int],
) -> dict[str, Any]:
    return {
        "examples": len(board),
        "train_examples": len(train_indices),
        "development_examples": len(development_indices),
        "score_arms": list(SCORE_ARMS),
        "max_expression_depth": board.max_expression_depth,
        "max_convergence_updates": board.max_convergence_updates,
        "minimum_safety_steps": board.minimum_safety_steps,
    }


def _build_model(config: AHRFTrainConfig) -> AutocatalyticHystereticRelationField:
    return AutocatalyticHystereticRelationField(
        node_feature_dim=AHRF_NODE_FEATURE_DIM,
        hidden_dim=config.hidden_dim,
        card_rounds=config.card_rounds,
        max_steps=config.max_steps,
        hysteresis=config.control != "no_hysteresis",
        use_card_conditioning=config.control != "generic_recurrence",
        triad_mode={
            "false_triad": "false",
            "zero_triad": "zero",
        }.get(config.control, "learned"),
    )


def _resolve_warm_start(
    *,
    repository_root: Path,
    recorded: Mapping[str, Any] | None,
    override: Path | None,
) -> Path | None:
    if recorded is None:
        _require(override is None, "warm-start override supplied for a cold run")
        return None
    _require(
        set(recorded)
        == {
            "path",
            "sha256",
            "copied_tensors",
            "copied_parameters",
        },
        "warm-start receipt fields drifted",
    )
    recorded_path = recorded["path"]
    _require(isinstance(recorded_path, str), "warm-start path receipt differs")
    candidate = override if override is not None else Path(recorded_path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return _existing_regular_file(candidate, "warm-start checkpoint")


def _verify_warm_start(
    *,
    config: AHRFTrainConfig,
    recorded: object,
    warm_start_path: Path | None,
) -> dict[str, Any] | None:
    if recorded is None:
        _require(
            config.binder_checkpoint is None,
            "config names a binder but the warm-start receipt is absent",
        )
        _require(warm_start_path is None, "unexpected warm-start checkpoint")
        return None
    _require(isinstance(recorded, dict), "warm-start receipt is not an object")
    _require(
        config.binder_checkpoint == recorded.get("path"),
        "config and warm-start path receipts differ",
    )
    _require(warm_start_path is not None, "warm-start checkpoint is unavailable")
    observed_sha256 = sha256_file(warm_start_path)
    _require(
        observed_sha256 == recorded.get("sha256"),
        "warm-start checkpoint hash drifted",
    )
    probe_model = _build_model(config)
    observed = _transfer_binder(probe_model, warm_start_path)
    for key in ("sha256", "copied_tensors", "copied_parameters"):
        _require(
            observed[key] == recorded.get(key),
            f"warm-start {key} receipt drifted",
        )
    return {
        "recorded_path": recorded["path"],
        "verified_path": str(warm_start_path),
        "sha256": observed_sha256,
        "copied_tensors": list(recorded["copied_tensors"]),
        "copied_parameters": int(recorded["copied_parameters"]),
    }


def classify_trajectory_history(
    binary_history: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    halt_step: int,
    safety_exhausted: bool,
    milestone_step: int = MILESTONE_STEP,
) -> dict[str, Any]:
    """Classify one already-binarized root trajectory without model execution."""

    _require(binary_history.dtype == torch.bool, "trajectory history is not binary")
    _require(target.dtype == torch.bool, "trajectory target is not binary")
    _require(mask.dtype == torch.bool, "trajectory mask is not binary")
    _require(binary_history.ndim >= 2, "trajectory history rank differs")
    _require(
        binary_history.shape[1:] == target.shape == mask.shape,
        "trajectory target/mask geometry differs",
    )
    _require(
        0 <= milestone_step < binary_history.shape[0],
        "trajectory does not include the requested milestone",
    )
    _require(
        halt_step == -1 or 1 <= halt_step < binary_history.shape[0],
        "trajectory halt step is outside its history",
    )
    _require(
        safety_exhausted == (halt_step == -1),
        "trajectory halt and safety status disagree",
    )

    exact = (
        (binary_history.eq(target.unsqueeze(0)) | ~mask.unsqueeze(0)).flatten(1).all(-1)
    )
    exact_steps = exact.nonzero().flatten().tolist()
    first_exact = int(exact_steps[0]) if exact_steps else None
    last_exact = int(exact_steps[-1]) if exact_steps else None

    false_transition = (
        (
            ~binary_history[:-1]
            & binary_history[1:]
            & ~target.unsqueeze(0)
            & mask.unsqueeze(0)
        )
        .flatten(1)
        .any(-1)
    )
    first_false_write = None
    if first_exact is not None:
        candidates = (
            (
                false_transition
                & torch.arange(
                    1,
                    binary_history.shape[0],
                    device=false_transition.device,
                ).gt(first_exact)
            )
            .nonzero()
            .flatten()
        )
        if candidates.numel():
            first_false_write = int(candidates[0]) + 1

    halt_exactness = None if halt_step == -1 else bool(exact[halt_step])
    return {
        "first_exact_step": first_exact,
        "last_exact_step": last_exact,
        "ever_exact": bool(exact.any()),
        "exact_at_milestone": bool(exact[milestone_step]),
        "final_exact": bool(exact[-1]),
        "first_false_write_after_first_exact": first_false_write,
        "halt_step": halt_step,
        "halt_exactness": halt_exactness,
        "safety_exhausted": bool(safety_exhausted),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_present(values: Iterable[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def aggregate_trajectory_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(records)
    ever_exact = sum(bool(row["ever_exact"]) for row in records)
    halted = sum(int(row["halt_step"] != -1) for row in records)
    halt_exact = sum(row["halt_exactness"] is True for row in records)
    counts = {
        "count": count,
        "ever_exact": ever_exact,
        "exact_at_step64": sum(bool(row["exact_at_step64"]) for row in records),
        "final_exact": sum(bool(row["final_exact"]) for row in records),
        "false_write_after_first_exact": sum(
            row["first_false_write_after_first_exact"] is not None for row in records
        ),
        "halted": halted,
        "halt_exact": halt_exact,
        "safety_exhausted": sum(bool(row["safety_exhausted"]) for row in records),
    }
    return {
        **counts,
        "rates": {
            key: _rate(value, count) for key, value in counts.items() if key != "count"
        },
        "halt_precision": _rate(halt_exact, halted),
        "halt_recall": _rate(halt_exact, ever_exact),
        "mean_first_exact_step": _mean_present(
            row["first_exact_step"] for row in records
        ),
        "mean_last_exact_step": _mean_present(
            row["last_exact_step"] for row in records
        ),
        "mean_false_write_step": _mean_present(
            row["first_false_write_after_first_exact"] for row in records
        ),
        "mean_halt_step": _mean_present(
            None if row["halt_step"] == -1 else int(row["halt_step"]) for row in records
        ),
    }


def _group_aggregates(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        label = ":".join(str(row[key]) for key in keys)
        grouped.setdefault(label, []).append(row)
    return {
        label: aggregate_trajectory_records(rows)
        for label, rows in sorted(grouped.items())
    }


def summarize_trajectory_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "overall": aggregate_trajectory_records(records),
        "by_split": _group_aggregates(records, ("split",)),
        "by_arm": _group_aggregates(records, ("arm",)),
        "by_cell": _group_aggregates(records, ("cell",)),
        "by_cell_arm": _group_aggregates(records, ("cell", "arm")),
        "by_split_cell_arm": _group_aggregates(
            records,
            ("split", "cell", "arm"),
        ),
    }


def _validate_rollout(
    rollout: AHRFRollout,
    *,
    graph: Any,
    hard_events: bool,
    enable_halt: bool,
    max_steps: int,
) -> None:
    histories = {
        "fact": rollout.fact_history,
        "membrane": rollout.membrane_history,
        "evidence": rollout.evidence_history,
        "halted": rollout.halted_history,
    }
    _require(
        all(value is not None for value in histories.values()),
        "AHRF rollout omitted required histories",
    )
    fact_history = histories["fact"]
    membrane_history = histories["membrane"]
    evidence_history = histories["evidence"]
    halted_history = histories["halted"]
    assert fact_history is not None
    assert membrane_history is not None
    assert evidence_history is not None
    assert halted_history is not None
    batch = graph.node_features.shape[0]
    nodes = graph.node_features.shape[1]
    objects = graph.object_mask.shape[1]
    expected_facts = (batch, max_steps + 1, nodes, objects, objects)
    _require(fact_history.shape == expected_facts, "fact history geometry differs")
    _require(
        evidence_history.shape == expected_facts,
        "evidence history geometry differs",
    )
    _require(
        membrane_history.shape[:5] == expected_facts,
        "membrane history geometry differs",
    )
    _require(
        membrane_history.shape[-1] > 0,
        "membrane history width is empty",
    )
    _require(
        halted_history.shape == (batch, max_steps + 1),
        "halted history geometry differs",
    )
    _require(
        rollout.halt_logits.shape == (batch, max_steps),
        "halt logit history geometry differs",
    )
    _require(
        rollout.halt_probabilities.shape == (batch, max_steps),
        "halt probability history geometry differs",
    )
    _require(
        rollout.write_probabilities.shape
        == (
            batch,
            max_steps,
            nodes,
            objects,
            objects,
        ),
        "write probability history geometry differs",
    )
    _require(
        torch.equal(fact_history[:, 0], graph.seed_facts),
        "fact history does not begin at the frozen seed",
    )
    _require(
        torch.equal(evidence_history[:, 0], graph.seed_facts),
        "evidence history does not begin at the frozen seed",
    )
    _require(
        torch.equal(fact_history[:, -1], rollout.terminal_facts),
        "terminal fact state differs from history",
    )
    _require(
        torch.equal(membrane_history[:, -1], rollout.terminal_membrane),
        "terminal membrane differs from history",
    )
    _require(
        torch.equal(evidence_history[:, -1], rollout.terminal_evidence),
        "terminal evidence differs from history",
    )
    _require(not bool(halted_history[:, 0].any()), "halt history starts halted")
    _require(
        not bool((halted_history[:, :-1] & ~halted_history[:, 1:]).any()),
        "halt history is not absorbing",
    )
    _require(
        torch.equal(halted_history[:, -1], rollout.learned_halted),
        "terminal halt state differs from history",
    )
    _require(
        torch.equal(rollout.safety_exhausted, ~rollout.learned_halted),
        "halt and safety terminal states disagree",
    )
    finite = (
        fact_history,
        membrane_history,
        evidence_history,
        rollout.halt_logits,
        rollout.halt_probabilities,
        rollout.write_probabilities,
    )
    _require(
        all(bool(torch.isfinite(value).all()) for value in finite),
        "AHRF history contains non-finite values",
    )
    _require(
        bool(
            rollout.halt_probabilities.ge(0).all()
            and rollout.halt_probabilities.le(1).all()
            and rollout.write_probabilities.ge(0).all()
            and rollout.write_probabilities.le(1).all()
        ),
        "AHRF event probabilities leave [0, 1]",
    )
    _require(
        torch.allclose(
            rollout.halt_probabilities,
            rollout.halt_logits.sigmoid(),
            atol=1e-7,
            rtol=1e-6,
        ),
        "halt probability history differs from logits",
    )
    if hard_events:
        _require(
            bool(fact_history.eq(fact_history.round()).all()),
            "hard fact history is not binary",
        )
        _require(
            bool(evidence_history.eq(evidence_history.round()).all()),
            "hard evidence history is not binary",
        )
    else:
        _require(
            bool(
                fact_history.ge(0).all()
                and fact_history.le(1).all()
                and evidence_history.ge(0).all()
                and evidence_history.le(1).all()
            ),
            "soft fact/evidence history leaves [0, 1]",
        )
    transitions = halted_history[:, 1:] & ~halted_history[:, :-1]
    for index in range(batch):
        transition_steps = transitions[index].nonzero().flatten() + 1
        expected_step = int(transition_steps[0]) if transition_steps.numel() else -1
        _require(
            int(rollout.halt_step[index]) == expected_step,
            "recorded halt step differs from halt history",
        )
        if expected_step != -1:
            _require(
                bool(
                    fact_history[index, expected_step:]
                    .eq(fact_history[index, expected_step])
                    .all()
                ),
                "facts changed after the absorbing halt",
            )
            _require(
                bool(
                    evidence_history[index, expected_step:]
                    .eq(evidence_history[index, expected_step])
                    .all()
                ),
                "evidence changed after the absorbing halt",
            )
            _require(
                bool(
                    membrane_history[index, expected_step:]
                    .eq(membrane_history[index, expected_step])
                    .all()
                ),
                "membrane changed after the absorbing halt",
            )
    if not enable_halt:
        _require(
            not bool(rollout.learned_halted.any()),
            "fixed-deadline rollout halted",
        )
        _require(
            bool(rollout.halt_step.eq(-1).all()),
            "fixed-deadline rollout recorded a halt step",
        )
        _require(
            bool(rollout.safety_exhausted.all()),
            "fixed-deadline rollout did not reach the safety horizon",
        )


def _chunk_records(
    *,
    board: AHRFBoard,
    indices: torch.Tensor,
    rollout: AHRFRollout,
    roots: torch.Tensor,
    targets: torch.Tensor,
    object_mask: torch.Tensor,
    hard_events: bool,
) -> list[dict[str, Any]]:
    assert rollout.fact_history is not None
    root_history = _root_facts(rollout.fact_history, roots)
    binary_history = root_history.eq(1.0) if hard_events else root_history.ge(0.5)
    mask = _target_mask(object_mask)
    records: list[dict[str, Any]] = []
    for local_index, board_index in enumerate(indices.tolist()):
        split, cell, arm, renderer = board.labels[board_index]
        classified = classify_trajectory_history(
            binary_history[local_index],
            targets[local_index].eq(1.0),
            mask[local_index],
            halt_step=int(rollout.halt_step[local_index]),
            safety_exhausted=bool(rollout.safety_exhausted[local_index]),
        )
        classified["exact_at_step64"] = classified.pop("exact_at_milestone")
        records.append(
            {
                "example_index": board_index,
                "split": split,
                "cell": cell,
                "arm": arm,
                "renderer": renderer,
                **classified,
            }
        )
    return records


def _evaluate_mode(
    *,
    model: AutocatalyticHystereticRelationField,
    board: AHRFBoard,
    device: torch.device,
    batch_size: int,
    hard_events: bool,
    enable_halt: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_indices = torch.arange(len(board), dtype=torch.long)
    for indices in all_indices.split(batch_size):
        graph = _move_graph(_index_graph(board.graph, indices), device)
        roots = board.roots.index_select(0, indices).to(device)
        targets = board.targets.index_select(0, indices).to(device)
        with torch.inference_mode():
            rollout = model(
                graph,
                hard_events=hard_events,
                enable_halt=enable_halt,
                return_history=True,
            )
        _validate_rollout(
            rollout,
            graph=graph,
            hard_events=hard_events,
            enable_halt=enable_halt,
            max_steps=model.max_steps,
        )
        records.extend(
            _chunk_records(
                board=board,
                indices=indices,
                rollout=rollout,
                roots=roots,
                targets=targets,
                object_mask=graph.object_mask,
                hard_events=hard_events,
            )
        )
    _require(
        [row["example_index"] for row in records] == list(range(len(board))),
        "trajectory records are incomplete or reordered",
    )
    return records


def _terminal_receipt(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    selected = [records[index] for index in indices]
    counts: Counter[tuple[str, str, str]] = Counter()
    correct: Counter[tuple[str, str, str]] = Counter()
    halted: Counter[tuple[str, str, str]] = Counter()
    for row in selected:
        key = (str(row["split"]), str(row["cell"]), str(row["arm"]))
        counts[key] += 1
        correct[key] += int(bool(row["final_exact"]))
        halted[key] += int(int(row["halt_step"]) != -1)
    exact = sum(bool(row["final_exact"]) for row in selected)
    learned_halted = sum(int(row["halt_step"] != -1) for row in selected)
    safety_exhausted = sum(bool(row["safety_exhausted"]) for row in selected)
    return {
        "exact": exact,
        "count": len(selected),
        "exact_rate": exact / len(selected),
        "learned_halted": learned_halted,
        "safety_exhausted": safety_exhausted,
        "halt_steps": [int(row["halt_step"]) for row in selected],
        "metrics": {
            ":".join(key): {
                "exact": correct[key],
                "halted": halted[key],
                "count": counts[key],
            }
            for key in sorted(counts)
        },
    }


def _verify_recorded_terminal_receipts(
    *,
    report: Mapping[str, Any],
    mode_records: Mapping[str, Sequence[Mapping[str, Any]]],
    train_indices: Sequence[int],
    development_indices: Sequence[int],
) -> dict[str, bool]:
    comparisons = {
        "train_hard_learned_halt": (
            report["train"],
            _terminal_receipt(
                mode_records["hard_learned_halt"],
                train_indices,
            ),
        ),
        "development_hard_learned_halt": (
            report["development"],
            _terminal_receipt(
                mode_records["hard_learned_halt"],
                development_indices,
            ),
        ),
        "development_hard_fixed_deadline": (
            report["development_fixed_deadline"],
            _terminal_receipt(
                mode_records["hard_fixed_deadline"],
                development_indices,
            ),
        ),
    }
    result: dict[str, bool] = {}
    for label, (recorded, observed) in comparisons.items():
        _require(recorded == observed, f"recorded terminal receipt drifted: {label}")
        result[label] = True
    return result


def _snapshot_inputs(
    *,
    repository_root: Path,
    report_path: Path,
    checkpoint_path: Path,
    warm_start_path: Path | None,
) -> dict[str, Any]:
    return {
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "warm_start_sha256": (
            sha256_file(warm_start_path) if warm_start_path is not None else None
        ),
        "source_sha256": _source_receipt(repository_root),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }


def _atomic_publish_json(output: Path, payload: Mapping[str, Any]) -> str:
    _require(not output.exists(), f"audit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.exists(), f"audit output already exists: {output}")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise AHRFTrajectoryAuditError(
                f"audit output appeared during publication: {output}"
            ) from error
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _ensure_separate_output(
    output: Path,
    *,
    report_path: Path,
    checkpoint_path: Path,
) -> Path:
    output = output.expanduser().resolve()
    source_directories = {report_path.parent, checkpoint_path.parent}
    _require(output not in {report_path, checkpoint_path}, "audit output aliases input")
    _require(
        not any(
            output == directory or output.is_relative_to(directory)
            for directory in source_directories
        ),
        "audit output must be separate from the completed AHRF run directory",
    )
    _require(not output.exists(), f"audit output already exists: {output}")
    return output


def audit_completed_run(
    *,
    report_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    warm_start_override: Path | None = None,
    device_name: str = "recorded",
    batch_size: int = 2,
) -> tuple[dict[str, Any], str]:
    _require(batch_size >= 1, "audit batch size must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    report_path = _existing_regular_file(report_path, "completed AHRF report")
    checkpoint_path = _existing_regular_file(
        checkpoint_path,
        "completed AHRF checkpoint",
    )
    output_path = _ensure_separate_output(
        output_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )
    report = _load_json_object(report_path)
    _require(set(report) == REPORT_KEYS, "completed AHRF report schema drifted")
    _require(report.get("protocol") == PROTOCOL, "AHRF report protocol differs")
    checkpoint_receipt = report.get("checkpoint")
    _require(
        isinstance(checkpoint_receipt, dict)
        and set(checkpoint_receipt) == {"path", "sha256"},
        "AHRF checkpoint receipt differs",
    )
    input_checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(
        checkpoint_receipt["sha256"] == input_checkpoint_sha256,
        "completed AHRF checkpoint hash drifted",
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise AHRFTrajectoryAuditError(
            "cannot load completed AHRF checkpoint"
        ) from error
    _require(isinstance(checkpoint, dict), "AHRF checkpoint is not an object")
    _require(set(checkpoint) == CHECKPOINT_KEYS, "AHRF checkpoint schema drifted")
    _require(checkpoint.get("protocol") == PROTOCOL, "checkpoint protocol differs")
    _require(checkpoint["config"] == report["config"], "config receipts differ")
    _require(
        _json_normalize(checkpoint["parameter_receipt"]) == report["parameter_receipt"],
        "parameter receipts differ",
    )
    _require(
        checkpoint["source_sha256"] == report["source_sha256"],
        "source receipts differ between checkpoint and report",
    )
    _require(
        checkpoint["warm_start"] == report["warm_start"],
        "warm-start receipts differ between checkpoint and report",
    )
    _require(
        set(report["source_sha256"]) == set(SOURCE_PATHS),
        "AHRF source receipt path set drifted",
    )
    live_sources = _source_receipt(repository_root)
    _require(
        live_sources == report["source_sha256"],
        "live frozen AHRF sources drifted",
    )

    config = _config_from_receipt(report["config"])
    board = build_board(config)
    board = AHRFBoard(
        graph=_apply_control(board.graph, config.control),
        targets=board.targets,
        roots=board.roots,
        labels=board.labels,
        max_expression_depth=board.max_expression_depth,
        max_convergence_updates=board.max_convergence_updates,
        minimum_safety_steps=board.minimum_safety_steps,
    )
    train_indices = [
        index for index, label in enumerate(board.labels) if label[0] == "train"
    ]
    development_indices = [
        index for index, label in enumerate(board.labels) if label[0] == "development"
    ]
    _require(
        report["board"]
        == _expected_board_receipt(board, train_indices, development_indices),
        "reconstructed AHRF board receipt drifted",
    )
    _require(
        config.max_steps >= board.minimum_safety_steps,
        "recorded AHRF safety horizon is below the reconstructed board",
    )

    recorded_warm_start = report["warm_start"]
    warm_start_path = _resolve_warm_start(
        repository_root=repository_root,
        recorded=recorded_warm_start,
        override=warm_start_override,
    )
    warm_start_receipt = _verify_warm_start(
        config=config,
        recorded=recorded_warm_start,
        warm_start_path=warm_start_path,
    )
    input_snapshot = _snapshot_inputs(
        repository_root=repository_root,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        warm_start_path=warm_start_path,
    )
    _require(
        input_snapshot["checkpoint_sha256"] == checkpoint_receipt["sha256"],
        "checkpoint changed before rollout",
    )
    _require(
        input_snapshot["source_sha256"] == report["source_sha256"],
        "source changed before rollout",
    )

    model = _build_model(config)
    _require(
        _json_normalize(asdict(model.parameter_receipt()))
        == report["parameter_receipt"],
        "reconstructed AHRF parameter receipt drifted",
    )
    model_state = checkpoint["model_state"]
    _require(isinstance(model_state, dict), "AHRF model state is not an object")
    _require(
        all(isinstance(value, torch.Tensor) for value in model_state.values()),
        "AHRF model state contains a non-tensor",
    )
    _require(
        all(bool(torch.isfinite(value).all()) for value in model_state.values()),
        "AHRF model state contains a non-finite tensor",
    )
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise AHRFTrajectoryAuditError("AHRF model state geometry drifted") from error
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_device_name = (
        str(report["device"]) if device_name == "recorded" else device_name
    )
    device = _resolve_device(selected_device_name)
    model = model.to(device)
    model.eval()
    _require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "AHRF audit model was not frozen",
    )

    mode_records: dict[str, list[dict[str, Any]]] = {}
    for mode, (hard_events, enable_halt) in MODES.items():
        mode_records[mode] = _evaluate_mode(
            model=model,
            board=board,
            device=device,
            batch_size=batch_size,
            hard_events=hard_events,
            enable_halt=enable_halt,
        )
    recorded_receipts = _verify_recorded_terminal_receipts(
        report=report,
        mode_records=mode_records,
        train_indices=train_indices,
        development_indices=development_indices,
    )

    final_snapshot = _snapshot_inputs(
        repository_root=repository_root,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        warm_start_path=warm_start_path,
    )
    _require(final_snapshot == input_snapshot, "audit inputs drifted during rollout")
    _require(not output_path.exists(), "audit output appeared during rollout")

    examples = []
    for index, label in enumerate(board.labels):
        split, cell, arm, renderer = label
        examples.append(
            {
                "example_index": index,
                "split": split,
                "cell": cell,
                "arm": arm,
                "renderer": renderer,
                "modes": {
                    mode: {
                        key: value
                        for key, value in records[index].items()
                        if key
                        not in {
                            "example_index",
                            "split",
                            "cell",
                            "arm",
                            "renderer",
                        }
                    }
                    for mode, records in mode_records.items()
                },
            }
        )
    payload: dict[str, Any] = {
        "protocol": AUDIT_PROTOCOL,
        "claim_boundary": (
            "Post-run, no-training trajectory measurement of the completed "
            "AHRF checkpoint. Rollouts are model-owned and receive no host "
            "intervention, convergence feedback, repair, or intermediate-state "
            "replay. Soft exactness is measured after a fixed >=0.5 threshold."
        ),
        "custody": {
            "repository_root": str(repository_root),
            "report_path": str(report_path),
            "checkpoint_path": str(checkpoint_path),
            "output_path": str(output_path),
            "input_snapshot": input_snapshot,
            "warm_start": warm_start_receipt,
            "recorded_terminal_receipts_verified": recorded_receipts,
        },
        "config": asdict(config),
        "device": str(device),
        "batch_size": batch_size,
        "board": report["board"],
        "mode_contract": {
            mode: {
                "hard_events": hard_events,
                "enable_halt": enable_halt,
                "history_threshold": 1.0 if hard_events else 0.5,
                "host_intervention": False,
                "host_feedback": False,
            }
            for mode, (hard_events, enable_halt) in MODES.items()
        },
        "trajectory_metric_contract": {
            "step_zero_included": True,
            "milestone_step": MILESTONE_STEP,
            "soft_binary_threshold": 0.5,
            "first_false_write_after_first_exact": (
                "first post-exact 0-to-1 threshold transition in an active "
                "root cell whose target is zero"
            ),
            "halt_precision": "exact halts / all learned halts",
            "halt_recall": "exact halts / trajectories ever exact",
        },
        "summaries": {
            mode: summarize_trajectory_records(records)
            for mode, records in mode_records.items()
        },
        "examples": examples,
    }
    payload["audit_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    file_sha256 = _atomic_publish_json(output_path, payload)
    return payload, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a completed AHRF trajectory without training.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument(
        "--device",
        default="recorded",
        help="recorded, auto, cpu, mps, cuda, or another torch device",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    payload, file_sha256 = audit_completed_run(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        warm_start_override=args.warm_start,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "file_sha256": file_sha256,
                "audit_payload_sha256": payload["audit_payload_sha256"],
                "summaries": payload["summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
