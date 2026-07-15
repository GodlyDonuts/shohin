#!/usr/bin/env python3
"""Fail-closed Stage-A scorer for the frozen WGRQ CPU falsifier.

The input is a ``wgrq_stage_a_score_manifest_v1`` JSON object containing one
hash-bound symbolic audit and exactly 60 fit records (five arms by twelve
seeds).  Every fit record has the following shape::

    {
      "arm": "wgrq_shortest",
      "seed": 17011,
      "checkpoint": {"path": "...", "sha256": "..."},
      "evaluation": {"path": "...", "sha256": "..."}
    }

Paths are relative to the manifest.  Evaluation artifacts use schema
``wgrq_confirmation_evaluation_v1`` and contain three strata, each with 1,024
rows keyed by ``committed_episode_id``.  The scorer joins those IDs across all
arms and seeds before applying a crossed two-way paired bootstrap.

All decision constants are fixed in this module.  There are deliberately no
CLI switches for thresholds, bootstrap replicates, bootstrap seed, or a
fallback scoring path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_SCHEMA = "wgrq_stage_a_score_manifest_v1"
SYMBOLIC_SCHEMA = "wgrq_symbolic_audit_v1"
EVALUATION_SCHEMA = "wgrq_confirmation_evaluation_v1"
DECISION_SCHEMA = "wgrq_stage_a_decision_v1"

SEEDS = (
    17011,
    27103,
    38119,
    49201,
    50311,
    61403,
    72503,
    83609,
    94709,
    105019,
    116027,
    127031,
)
ARMS = (
    "wgrq_shortest",
    "active_answer_only",
    "uniform_witness",
    "relation_sham",
    "privileged_edge",
)
ARM_ALIASES = {
    "wgrq_shortest": "wgrq_shortest",
    "wgrq-shortest": "wgrq_shortest",
    "active_answer_only": "active_answer_only",
    "active-answer-only": "active_answer_only",
    "uniform_witness": "uniform_witness",
    "uniform-witness": "uniform_witness",
    "relation_sham": "relation_sham",
    "relation-sham": "relation_sham",
    "privileged_edge": "privileged_edge",
    "privileged-edge": "privileged_edge",
}
STRATA = ("length_ood", "scale_ood", "full_ood")

EPISODES_PER_STRATUM = 1_024
EXPECTED_FITS = len(SEEDS) * len(ARMS)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CONFIDENCE_PERCENT = 95
BOOTSTRAP_LOWER_TAIL_PERCENT = 5
BOOTSTRAP_CHUNK_REPLICATES = 256
BOOTSTRAP_SEED_LABEL = b"wgrq-stage-a-two-way-paired-bootstrap-v1"
BOOTSTRAP_SEED = int.from_bytes(
    hashlib.sha256(BOOTSTRAP_SEED_LABEL).digest()[:16], "big"
)
PRIVILEGED_CEILING_PERCENT = 99
WGRQ_FLOOR_PERCENT = 95
CONTROL_MARGIN_PERCENT = 5
PAIRED_SEED_REQUIRED = 10

SYMBOLIC_SCALES = (3, 6)
SYMBOLIC_SCALE_GATES = (
    "physical_transitions_and_reversibility",
    "future_equivalence_exact",
    "quotient_cardinality_exact",
    "quotient_transitions_representative_independent",
    "shortest_witness_depths_exact",
    "tight_maximum_witness_depth",
    "canonical_serialization_exact",
)
CANCELLATION_GATES = (
    "ff_rn_identity",
    "frf_rn_minus_1_nonidentity",
    "fr_noncommutes_rf",
    "global_complement_observationally_null",
    "equal_count_identity",
)
GENERATION_GATES = (
    "labels_balanced_within_declared_strata_where_possible",
    "unavoidable_parity_obstruction_reported",
)
PROTOCOL_GATES = (
    "ordinary_oracle_answers_match",
    "training_transcript_byte_identical",
    "source_and_cache_absent",
    "resource_contract_match",
    "checkpoint_frozen_before_confirmation",
    "process_deletion_passed",
    "masked_bits_zero",
    "packet_reuse_byte_identical",
    "all_branches_complete",
)

HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON_ARTIFACT_BYTES = 512 * 1024 * 1024
CLAIM_BOUNDARY = (
    "A GO supports only the frozen synthetic-family neural optimization claim "
    "in R12_WGRQ_CPU_PREREG.md; it is not a new state, algorithm, oracle-rate, "
    "language-transfer, or general-reasoning result."
)


class EvidenceError(ValueError):
    """A fail-closed evidence-contract violation."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("invalid_json_shape", f"{label} must be a JSON object")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError("invalid_integer", f"{label} must be an integer")
    return value


def _binary(value: Any, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise EvidenceError(
        "invalid_episode_exact", f"{label} must be boolean or integer 0/1"
    )


def _canonical_arm(value: Any, label: str = "arm") -> str:
    if not isinstance(value, str) or value not in ARM_ALIASES:
        raise EvidenceError(
            "unknown_arm", f"{label} is not one of the five frozen arms"
        )
    return ARM_ALIASES[value]


def _expected_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise EvidenceError(
            "invalid_sha256", f"{label} must be a lowercase SHA-256 hex digest"
        )
    return value


def _artifact_reference(value: Any, label: str) -> tuple[str, str]:
    reference = _require_object(value, label)
    raw_path = reference.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError(
            "invalid_artifact_path", f"{label}.path must be a nonempty string"
        )
    return raw_path, _expected_hash(reference.get("sha256"), f"{label}.sha256")


def _path_from_manifest(raw_path: str, manifest_directory: Path) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = manifest_directory / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _read_regular_file(path: Path, *, capture: bool) -> tuple[bytes | None, str]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot open regular artifact {path}: {exc}"
        ) from exc

    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(
                "artifact_not_regular", f"artifact is not a regular file: {path}"
            )
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if capture and total > MAX_JSON_ARTIFACT_BYTES:
                raise EvidenceError(
                    "json_artifact_too_large",
                    f"JSON artifact exceeds size limit: {path}",
                )
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise EvidenceError(
                "artifact_changed_during_read",
                f"artifact changed while being read: {path}",
            )
    finally:
        os.close(descriptor)
    return (b"".join(chunks) if chunks is not None else None), digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one stable, non-symlink regular file."""

    return _read_regular_file(Path(path), capture=False)[1]


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(
                    "duplicate_json_key", f"{label} repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceError("nonfinite_json_number", f"{label} contains {value}")

    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid_json", f"cannot parse {label}: {exc}") from exc
    return _require_object(value, label)


def _verify_json_reference(
    value: Any,
    label: str,
    manifest_directory: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    raw_path, expected = _artifact_reference(value, label)
    path = _path_from_manifest(raw_path, manifest_directory)
    raw, actual = _read_regular_file(path, capture=True)
    if actual != expected:
        raise EvidenceError(
            "artifact_hash_mismatch",
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}",
        )
    assert raw is not None
    return _parse_json_bytes(raw, label), {"path": raw_path, "sha256": actual}


def _verify_binary_reference(
    value: Any,
    label: str,
    manifest_directory: Path,
) -> dict[str, str]:
    raw_path, expected = _artifact_reference(value, label)
    path = _path_from_manifest(raw_path, manifest_directory)
    _, actual = _read_regular_file(path, capture=False)
    if actual != expected:
        raise EvidenceError(
            "artifact_hash_mismatch",
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}",
        )
    return {"path": raw_path, "sha256": actual}


def minimum_symbolic_checks(n: int) -> int:
    return (n - 1) * (2 ** (2 * n)) + 3 * (2**n)


def verify_symbolic_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate the symbolic evidence shape and return all gate failures."""

    if audit.get("schema") != SYMBOLIC_SCHEMA:
        raise EvidenceError(
            "symbolic_schema_mismatch", f"symbolic schema must be {SYMBOLIC_SCHEMA}"
        )
    reported_pass = audit.get("passed")
    if not isinstance(reported_pass, bool):
        raise EvidenceError("invalid_symbolic_pass", "symbolic passed must be boolean")

    scales = _require_object(audit.get("scales"), "symbolic.scales")
    if set(scales) != {str(n) for n in SYMBOLIC_SCALES}:
        raise EvidenceError(
            "symbolic_scale_set_mismatch", "symbolic scales must be exactly n=3 and n=6"
        )

    failures: list[str] = []
    scale_results: dict[str, Any] = {}
    for n in SYMBOLIC_SCALES:
        label = f"symbolic.scales.{n}"
        scale = _require_object(scales[str(n)], label)
        check_count = _strict_int(scale.get("check_count"), f"{label}.check_count")
        class_count = _strict_int(
            scale.get("quotient_class_count"), f"{label}.quotient_class_count"
        )
        class_size = _strict_int(
            scale.get("quotient_class_size"), f"{label}.quotient_class_size"
        )
        max_depth = _strict_int(
            scale.get("maximum_shortest_witness_depth"),
            f"{label}.maximum_shortest_witness_depth",
        )
        expected_checks = minimum_symbolic_checks(n)
        expected_classes = 2 ** (n - 1)
        if check_count < expected_checks:
            failures.append(f"n{n}:check_count_below_{expected_checks}")
        if class_count != expected_classes:
            failures.append(f"n{n}:quotient_class_count_not_{expected_classes}")
        if class_size != 2:
            failures.append(f"n{n}:quotient_class_size_not_2")
        if max_depth != n - 2:
            failures.append(f"n{n}:maximum_shortest_witness_depth_not_{n - 2}")

        gates = _require_object(scale.get("gates"), f"{label}.gates")
        for gate in SYMBOLIC_SCALE_GATES:
            value = gates.get(gate)
            if value is not True:
                if value is not None and not isinstance(value, bool):
                    raise EvidenceError(
                        "invalid_symbolic_gate", f"{label}.gates.{gate} must be boolean"
                    )
                failures.append(f"n{n}:{gate}")
        scale_results[str(n)] = {
            "check_count": check_count,
            "minimum_check_count": expected_checks,
            "quotient_class_count": class_count,
            "quotient_class_size": class_size,
            "maximum_shortest_witness_depth": max_depth,
        }

    cancellations = _require_object(
        audit.get("cancellation_controls"), "symbolic.cancellation_controls"
    )
    for gate in CANCELLATION_GATES:
        value = cancellations.get(gate)
        if value is not True:
            if value is not None and not isinstance(value, bool):
                raise EvidenceError(
                    "invalid_cancellation_gate",
                    f"cancellation control {gate} must be boolean",
                )
            failures.append(f"cancellation:{gate}")
    generation = _require_object(
        audit.get("generation_controls"), "symbolic.generation_controls"
    )
    for gate in GENERATION_GATES:
        value = generation.get(gate)
        if value is not True:
            if value is not None and not isinstance(value, bool):
                raise EvidenceError(
                    "invalid_generation_gate",
                    f"generation control {gate} must be boolean",
                )
            failures.append(f"generation:{gate}")
    if not reported_pass:
        failures.append("symbolic_reported_pass_false")
    return {
        "passed": not failures,
        "failures": failures,
        "scales": scale_results,
        "cancellation_controls_passed": all(
            cancellations.get(gate) is True for gate in CANCELLATION_GATES
        ),
        "generation_controls_passed": all(
            generation.get(gate) is True for gate in GENERATION_GATES
        ),
    }


def _validate_fit_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceError(
            "manifest_schema_mismatch", f"manifest schema must be {MANIFEST_SCHEMA}"
        )
    fits = manifest.get("fits")
    if not isinstance(fits, list) or len(fits) != EXPECTED_FITS:
        actual = len(fits) if isinstance(fits, list) else "non-list"
        raise EvidenceError(
            "fit_count_mismatch",
            f"manifest must contain exactly {EXPECTED_FITS} fits, got {actual}",
        )

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for position, raw_fit in enumerate(fits):
        fit = _require_object(raw_fit, f"fits[{position}]")
        arm = _canonical_arm(fit.get("arm"), f"fits[{position}].arm")
        seed = _strict_int(fit.get("seed"), f"fits[{position}].seed")
        if seed not in SEEDS:
            raise EvidenceError(
                "unexpected_seed", f"fits[{position}].seed is not frozen"
            )
        _artifact_reference(fit.get("checkpoint"), f"fits[{position}].checkpoint")
        _artifact_reference(fit.get("evaluation"), f"fits[{position}].evaluation")
        key = (arm, seed)
        if key in indexed:
            raise EvidenceError(
                "duplicate_fit", f"duplicate fit for arm={arm}, seed={seed}"
            )
        indexed[key] = fit

    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise EvidenceError(
            "fit_matrix_mismatch",
            f"fit matrix mismatch; missing={missing}, extra={extra}",
        )
    return indexed


def _parse_evaluation(
    evaluation: dict[str, Any],
    *,
    expected_arm: str,
    expected_seed: int,
    checkpoint_sha256: str,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    label = f"evaluation[{expected_arm},{expected_seed}]"
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise EvidenceError(
            "evaluation_schema_mismatch", f"{label} schema must be {EVALUATION_SCHEMA}"
        )
    if _canonical_arm(evaluation.get("arm"), f"{label}.arm") != expected_arm:
        raise EvidenceError(
            "evaluation_arm_mismatch", f"{label} arm does not match its fit"
        )
    if _strict_int(evaluation.get("seed"), f"{label}.seed") != expected_seed:
        raise EvidenceError(
            "evaluation_seed_mismatch", f"{label} seed does not match its fit"
        )
    reported_checkpoint = _expected_hash(
        evaluation.get("checkpoint_sha256"), f"{label}.checkpoint_sha256"
    )
    if reported_checkpoint != checkpoint_sha256:
        raise EvidenceError(
            "evaluation_checkpoint_mismatch",
            f"{label} is bound to a different checkpoint",
        )

    protocol = _require_object(
        evaluation.get("protocol_gates"), f"{label}.protocol_gates"
    )
    protocol_failures: list[str] = []
    for gate in PROTOCOL_GATES:
        value = protocol.get(gate)
        if value is not True:
            if value is not None and not isinstance(value, bool):
                raise EvidenceError(
                    "invalid_protocol_gate",
                    f"{label}.protocol_gates.{gate} must be boolean",
                )
            protocol_failures.append(f"{expected_arm}:{expected_seed}:{gate}")

    strata = _require_object(evaluation.get("strata"), f"{label}.strata")
    if set(strata) != set(STRATA):
        raise EvidenceError(
            "evaluation_strata_mismatch",
            f"{label} must contain exactly the three frozen strata",
        )
    parsed: dict[str, dict[str, int]] = {}
    for stratum in STRATA:
        block = _require_object(strata[stratum], f"{label}.strata.{stratum}")
        episodes = block.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != EPISODES_PER_STRATUM:
            actual = len(episodes) if isinstance(episodes, list) else "non-list"
            raise EvidenceError(
                "episode_count_mismatch",
                f"{label}.{stratum} must contain {EPISODES_PER_STRATUM} episodes, got {actual}",
            )
        rows: dict[str, int] = {}
        for position, raw_episode in enumerate(episodes):
            episode = _require_object(
                raw_episode, f"{label}.{stratum}.episodes[{position}]"
            )
            episode_id = episode.get("committed_episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise EvidenceError(
                    "invalid_committed_episode_id",
                    f"{label}.{stratum}.episodes[{position}] lacks a nonempty committed_episode_id",
                )
            if episode_id in rows:
                raise EvidenceError(
                    "duplicate_committed_episode_id",
                    f"{label}.{stratum} repeats committed episode {episode_id!r}",
                )
            rows[episode_id] = _binary(
                episode.get("episode_exact"),
                f"{label}.{stratum}.episodes[{position}].episode_exact",
            )
        parsed[stratum] = rows
    return parsed, protocol_failures


def _hash_commitment(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_evidence(
    manifest: dict[str, Any],
    manifest_directory: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify every artifact and assemble [stratum, arm, seed, episode]."""

    indexed = _validate_fit_index(manifest)
    symbolic_audit, symbolic_binding = _verify_json_reference(
        manifest.get("symbolic_audit"), "symbolic_audit", manifest_directory
    )
    symbolic_result = verify_symbolic_audit(symbolic_audit)

    checkpoints: dict[tuple[str, int], dict[str, str]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            fit = indexed[(arm, seed)]
            checkpoints[(arm, seed)] = _verify_binary_reference(
                fit.get("checkpoint"), f"checkpoint[{arm},{seed}]", manifest_directory
            )

    evaluations: dict[tuple[str, int], dict[str, dict[str, int]]] = {}
    evaluation_bindings: dict[tuple[str, int], dict[str, str]] = {}
    protocol_failures: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            fit = indexed[(arm, seed)]
            evaluation, binding = _verify_json_reference(
                fit.get("evaluation"), f"evaluation[{arm},{seed}]", manifest_directory
            )
            parsed, failures = _parse_evaluation(
                evaluation,
                expected_arm=arm,
                expected_seed=seed,
                checkpoint_sha256=checkpoints[(arm, seed)]["sha256"],
            )
            evaluations[(arm, seed)] = parsed
            evaluation_bindings[(arm, seed)] = binding
            protocol_failures.extend(failures)

    reference_pair = (ARMS[0], SEEDS[0])
    episode_ids: dict[str, list[str]] = {}
    for stratum in STRATA:
        reference = set(evaluations[reference_pair][stratum])
        for arm in ARMS:
            for seed in SEEDS:
                candidate = set(evaluations[(arm, seed)][stratum])
                if candidate != reference:
                    raise EvidenceError(
                        "committed_episode_pairing_mismatch",
                        f"{stratum} committed episode IDs differ for arm={arm}, seed={seed}",
                    )
        episode_ids[stratum] = sorted(reference)

    values = np.empty(
        (len(STRATA), len(ARMS), len(SEEDS), EPISODES_PER_STRATUM),
        dtype=np.uint8,
    )
    for stratum_index, stratum in enumerate(STRATA):
        ids = episode_ids[stratum]
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                rows = evaluations[(arm, seed)][stratum]
                values[stratum_index, arm_index, seed_index] = [
                    rows[episode_id] for episode_id in ids
                ]

    bindings: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            bindings.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": checkpoints[(arm, seed)],
                    "evaluation": evaluation_bindings[(arm, seed)],
                }
            )
    checkpoint_commitment = _hash_commitment(
        [
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "sha256": row["checkpoint"]["sha256"],
            }
            for row in bindings
        ]
    )
    evaluation_commitment = _hash_commitment(
        [
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "sha256": row["evaluation"]["sha256"],
            }
            for row in bindings
        ]
    )
    verification = {
        "status": "verified",
        "expected_fits": EXPECTED_FITS,
        "fits_verified": len(bindings),
        "checkpoint_hashes_verified": len(bindings),
        "evaluation_hashes_verified": len(bindings),
        "episodes_per_stratum_per_fit": EPISODES_PER_STRATUM,
        "paired_seed_episode_matrix_complete": True,
        "symbolic_audit": symbolic_binding,
        "symbolic_gate": symbolic_result,
        "protocol_gate_passed": not protocol_failures,
        "protocol_gate_failures": protocol_failures,
        "checkpoint_set_sha256": checkpoint_commitment,
        "evaluation_set_sha256": evaluation_commitment,
        "artifact_bindings": bindings,
    }
    return values, verification


def _uniform_indices(
    bit_generator: np.random.PCG64,
    *,
    rows: int,
    draws: int,
    upper: int,
) -> np.ndarray:
    """Draw unbiased indices using only PCG64 raw uint64 output."""

    if rows <= 0 or draws <= 0 or upper <= 0:
        raise ValueError("rows, draws, and upper must be positive")
    count = rows * draws
    if upper & (upper - 1) == 0:
        raw = bit_generator.random_raw(count)
        indices = raw & np.uint64(upper - 1)
        return indices.astype(np.int64, copy=False).reshape(rows, draws)

    max_acceptable = np.uint64(((1 << 64) // upper) * upper - 1)
    accepted = np.empty(count, dtype=np.uint64)
    filled = 0
    while filled < count:
        raw = bit_generator.random_raw(count - filled)
        valid = raw[raw <= max_acceptable]
        take = min(valid.size, count - filled)
        accepted[filled : filled + take] = valid[:take]
        filled += take
    indices = accepted % np.uint64(upper)
    return indices.astype(np.int64, copy=False).reshape(rows, draws)


def _indices_to_counts(indices: np.ndarray, upper: int) -> np.ndarray:
    rows = indices.shape[0]
    offsets = np.arange(rows, dtype=np.int64)[:, None] * upper
    flattened = (indices + offsets).reshape(-1)
    counts = np.bincount(flattened, minlength=rows * upper).reshape(rows, upper)
    return counts.astype(np.uint16, copy=False)


def paired_two_way_bootstrap(
    episode_exact: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the frozen crossed bootstrap with arm pairing retained.

    Seed IDs are sampled once per replicate and shared across all strata.
    Committed episode IDs are sampled within stratum and shared across every
    arm and sampled seed.  The returned lower bound is the nearest-rank 5th
    percentile of the replicate-wise minimum over all twelve G constraints.
    """

    values = np.asarray(episode_exact)
    if (
        values.ndim != 4
        or values.shape[0] != len(STRATA)
        or values.shape[1] != len(ARMS)
    ):
        raise EvidenceError(
            "bootstrap_shape_mismatch",
            "episode_exact must have shape [3 strata, 5 arms, seeds, episodes]",
        )
    if replicates <= 0:
        raise EvidenceError(
            "invalid_bootstrap_replicates", "bootstrap replicates must be positive"
        )
    if not np.all((values == 0) | (values == 1)):
        raise EvidenceError(
            "bootstrap_nonbinary_input", "bootstrap episode values must be binary"
        )

    stratum_count, _, seed_count, episode_count = values.shape
    denominator = seed_count * episode_count
    wgrq = values[:, 0].astype(np.int16)
    contrasts = np.stack(
        (
            wgrq,
            wgrq - values[:, 1].astype(np.int16),
            wgrq - values[:, 2].astype(np.int16),
            wgrq - values[:, 3].astype(np.int16),
        ),
        axis=1,
    )

    bit_generator = np.random.PCG64(rng_seed)
    seed_indices = _uniform_indices(
        bit_generator,
        rows=replicates,
        draws=seed_count,
        upper=seed_count,
    )
    seed_counts = _indices_to_counts(seed_indices, seed_count)
    resample_digest = hashlib.sha256(b"wgrq-two-way-resamples-v1\0")
    resample_digest.update(b"seed-counts\0")
    resample_digest.update(np.asarray(seed_counts, dtype="<u2").tobytes(order="C"))

    g_scaled = np.full(replicates, np.iinfo(np.int64).max, dtype=np.int64)
    offsets = np.array(
        [
            WGRQ_FLOOR_PERCENT,
            CONTROL_MARGIN_PERCENT,
            CONTROL_MARGIN_PERCENT,
            CONTROL_MARGIN_PERCENT,
        ],
        dtype=np.int64,
    )
    for stratum_index in range(stratum_count):
        resample_digest.update(STRATA[stratum_index].encode("ascii") + b"\0")
        flattened_contrasts = (
            contrasts[stratum_index]
            .reshape(4 * seed_count, episode_count)
            .astype(np.float32)
        )
        for start in range(0, replicates, BOOTSTRAP_CHUNK_REPLICATES):
            stop = min(start + BOOTSTRAP_CHUNK_REPLICATES, replicates)
            chunk_size = stop - start
            episode_indices = _uniform_indices(
                bit_generator,
                rows=chunk_size,
                draws=episode_count,
                upper=episode_count,
            )
            episode_counts = _indices_to_counts(episode_indices, episode_count)
            resample_digest.update(
                np.asarray(episode_counts, dtype="<u2").tobytes(order="C")
            )

            episode_sums = episode_counts.astype(np.float32) @ flattened_contrasts.T
            episode_sums = episode_sums.reshape(chunk_size, 4, seed_count)
            sampled = np.sum(
                episode_sums * seed_counts[start:stop, None, :].astype(np.float32),
                axis=2,
                dtype=np.float32,
            )
            numerators = np.rint(sampled).astype(np.int64)
            if not np.array_equal(sampled, numerators.astype(np.float32)):
                raise EvidenceError(
                    "bootstrap_integer_accumulation_failed",
                    "bootstrap did not preserve exact integer accumulations",
                )
            scaled_margins = 100 * numerators - offsets[None, :] * denominator
            g_scaled[start:stop] = np.minimum(
                g_scaled[start:stop], scaled_margins.min(axis=1)
            )

    lower_rank = max(1, math.ceil(replicates * BOOTSTRAP_LOWER_TAIL_PERCENT / 100))
    lower_index = lower_rank - 1
    lower_scaled = int(np.partition(g_scaled, lower_index)[lower_index])
    scaled_denominator = 100 * denominator
    replicate_digest = hashlib.sha256(
        np.asarray(g_scaled, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    return {
        "status": "complete",
        "algorithm": "crossed_two_way_paired_cluster_bootstrap_v1",
        "seed_sampling": "resample frozen seed IDs with replacement; shared by strata and arms",
        "episode_sampling": "resample committed episode IDs within stratum; shared by seeds and arms",
        "rng": "numpy.PCG64.random_raw with unbiased uint64 rejection sampling",
        "rng_seed_hex": f"{rng_seed:032x}",
        "rng_seed_label": BOOTSTRAP_SEED_LABEL.decode("ascii"),
        "replicates": replicates,
        "confidence_percent": BOOTSTRAP_CONFIDENCE_PERCENT,
        "lower_tail_percent": BOOTSTRAP_LOWER_TAIL_PERCENT,
        "lower_order_statistic_index_zero_based": lower_index,
        "simultaneous_g_lower_bound": lower_scaled / scaled_denominator,
        "simultaneous_g_lower_bound_scaled_numerator": lower_scaled,
        "simultaneous_g_lower_bound_scaled_denominator": scaled_denominator,
        "resample_counts_sha256": resample_digest.hexdigest(),
        "replicate_g_sha256": replicate_digest,
    }


def compute_point_results(episode_exact: np.ndarray) -> dict[str, Any]:
    values = np.asarray(episode_exact)
    if (
        values.ndim != 4
        or values.shape[0] != len(STRATA)
        or values.shape[1] != len(ARMS)
    ):
        raise EvidenceError(
            "point_shape_mismatch", "point score matrix has the wrong shape"
        )
    _, _, seed_count, episode_count = values.shape
    total = seed_count * episode_count
    counts = values.sum(axis=(2, 3), dtype=np.int64)

    estimates: dict[str, Any] = {}
    g_components: dict[str, Any] = {}
    point_g_scaled: int | None = None
    component_names = (
        "wgrq_minus_0_95",
        "wgrq_minus_active_answer_only_minus_0_05",
        "wgrq_minus_uniform_witness_minus_0_05",
        "wgrq_minus_relation_sham_minus_0_05",
    )
    for stratum_index, stratum in enumerate(STRATA):
        estimates[stratum] = {}
        for arm_index, arm in enumerate(ARMS):
            exact = int(counts[stratum_index, arm_index])
            estimates[stratum][arm] = {
                "exact_episodes": exact,
                "total_episodes": total,
                "rate": exact / total,
            }
        wgrq_count = int(counts[stratum_index, 0])
        numerators = (
            wgrq_count,
            wgrq_count - int(counts[stratum_index, 1]),
            wgrq_count - int(counts[stratum_index, 2]),
            wgrq_count - int(counts[stratum_index, 3]),
        )
        offsets = (
            WGRQ_FLOOR_PERCENT,
            CONTROL_MARGIN_PERCENT,
            CONTROL_MARGIN_PERCENT,
            CONTROL_MARGIN_PERCENT,
        )
        g_components[stratum] = {}
        for name, numerator, offset in zip(component_names, numerators, offsets):
            scaled = 100 * numerator - offset * total
            g_components[stratum][name] = {
                "scaled_numerator": scaled,
                "scaled_denominator": 100 * total,
                "value": scaled / (100 * total),
            }
            point_g_scaled = (
                scaled if point_g_scaled is None else min(point_g_scaled, scaled)
            )

    privileged_by_stratum: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(STRATA):
        exact = int(counts[stratum_index, ARMS.index("privileged_edge")])
        passed = 100 * exact >= PRIVILEGED_CEILING_PERCENT * total
        privileged_by_stratum[stratum] = {
            "exact_episodes": exact,
            "total_episodes": total,
            "rate": exact / total,
            "passed": passed,
        }

    full_index = STRATA.index("full_ood")
    per_seed: list[dict[str, Any]] = []
    wins = 0
    for seed_index in range(seed_count):
        wgrq_exact = int(
            values[full_index, ARMS.index("wgrq_shortest"), seed_index].sum()
        )
        aao_exact = int(
            values[full_index, ARMS.index("active_answer_only"), seed_index].sum()
        )
        difference = wgrq_exact - aao_exact
        passed = 100 * difference >= CONTROL_MARGIN_PERCENT * episode_count
        wins += int(passed)
        seed_value: int | str = (
            SEEDS[seed_index] if seed_count == len(SEEDS) else seed_index
        )
        per_seed.append(
            {
                "seed": seed_value,
                "wgrq_exact": wgrq_exact,
                "active_answer_only_exact": aao_exact,
                "difference_numerator": difference,
                "difference_denominator": episode_count,
                "difference": difference / episode_count,
                "beats_by_at_least_0_05": passed,
            }
        )

    assert point_g_scaled is not None
    return {
        "episode_exact": estimates,
        "g_components": g_components,
        "point_g": {
            "scaled_numerator": point_g_scaled,
            "scaled_denominator": 100 * total,
            "value": point_g_scaled / (100 * total),
        },
        "privileged_ceiling": {
            "threshold": PRIVILEGED_CEILING_PERCENT / 100,
            "by_stratum": privileged_by_stratum,
            "passed": all(row["passed"] for row in privileged_by_stratum.values()),
        },
        "full_ood_paired_seed_rule": {
            "required": PAIRED_SEED_REQUIRED,
            "total": seed_count,
            "margin": CONTROL_MARGIN_PERCENT / 100,
            "wins": wins,
            "passed": wins >= PAIRED_SEED_REQUIRED,
            "per_seed": per_seed,
        },
    }


def _requirements() -> dict[str, Any]:
    return {
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "strata": list(STRATA),
        "fits": EXPECTED_FITS,
        "episodes_per_stratum_per_fit": EPISODES_PER_STRATUM,
        "privileged_ceiling_minimum": PRIVILEGED_CEILING_PERCENT / 100,
        "wgrq_floor": WGRQ_FLOOR_PERCENT / 100,
        "control_margin": CONTROL_MARGIN_PERCENT / 100,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE_PERCENT / 100,
        "simultaneous_g_lower_bound_strictly_above": 0.0,
        "full_ood_paired_seed_wins_required": PAIRED_SEED_REQUIRED,
        "score_dependent_fallback": False,
    }


def _evidence_rejection(
    manifest_path: Path,
    manifest_sha256: str | None,
    error: EvidenceError,
) -> dict[str, Any]:
    return {
        "schema": DECISION_SCHEMA,
        "decision": "NO_GO",
        "go": False,
        "reasons": ["evidence_contract_failed", error.code],
        "failure_detail": error.detail,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "requirements": _requirements(),
        "verification": {"status": "failed"},
        "bootstrap": {
            "status": "not_run_evidence_failure",
            "replicates": 0,
            "required_replicates": BOOTSTRAP_REPLICATES,
            "score_dependent_fallback_used": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "output_contract": {
            "exclusive_create": True,
            "overwrite": False,
            "mode": "0444",
        },
    }


def score_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify and score a manifest, returning an immutable-output payload."""

    path = Path(manifest_path)
    manifest_sha256: str | None = None
    try:
        raw, manifest_sha256 = _read_regular_file(path, capture=True)
        assert raw is not None
        manifest = _parse_json_bytes(raw, "manifest")
        values, verification = verify_evidence(manifest, path.parent)
        point = compute_point_results(values)
        bootstrap = paired_two_way_bootstrap(
            values,
            replicates=BOOTSTRAP_REPLICATES,
            rng_seed=BOOTSTRAP_SEED,
        )
    except EvidenceError as exc:
        return _evidence_rejection(path, manifest_sha256, exc)
    except Exception as exc:
        error = EvidenceError(
            "scoring_execution_failed",
            f"scoring failed without fallback ({type(exc).__name__}): {exc}",
        )
        return _evidence_rejection(path, manifest_sha256, error)

    reasons: list[str] = []
    if not verification["symbolic_gate"]["passed"]:
        reasons.append("symbolic_gate_failed")
    if not verification["protocol_gate_passed"]:
        reasons.append("protocol_gate_failed")
    if not point["privileged_ceiling"]["passed"]:
        reasons.append("privileged_ceiling_below_0_99")
    if bootstrap["simultaneous_g_lower_bound_scaled_numerator"] <= 0:
        reasons.append("simultaneous_g_lower_bound_not_strictly_positive")
    if not point["full_ood_paired_seed_rule"]["passed"]:
        reasons.append("full_ood_paired_seed_rule_failed")

    go = not reasons
    return {
        "schema": DECISION_SCHEMA,
        "decision": "GO" if go else "NO_GO",
        "go": go,
        "reasons": reasons,
        "manifest": {"path": str(path), "sha256": manifest_sha256},
        "requirements": _requirements(),
        "verification": verification,
        "point_results": point,
        "bootstrap": bootstrap,
        "score_dependent_fallback_used": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_contract": {
            "exclusive_create": True,
            "overwrite": False,
            "mode": "0444",
        },
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_immutable_json(path: str | Path, payload: dict[str, Any]) -> str:
    """Create a decision once, fsync it, and remove all write bits."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating immutable decision")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Frozen Stage-A score manifest"
    )
    parser.add_argument("--out", required=True, help="New immutable decision JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing decision: {args.out}")
    decision = score_manifest(args.manifest)
    decision_sha256 = write_immutable_json(args.out, decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "decision_sha256": decision_sha256,
                "reasons": decision["reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
