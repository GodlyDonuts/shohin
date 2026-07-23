"""Signed, outcome-free preregistration for CTAA statistical advancement gates.

This module freezes and authenticates a statistical decision contract.  It does
not read model outcomes, compute a capability statistic, authorize confirmation,
or authorize a capability claim.  A valid record is therefore necessary but
never sufficient for CTAA advancement.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ctaa_intervention_protocol import (
    MANDATORY_OPERATIONS,
    OPERATION_SPECS,
    RUNTIME_PANEL_SIZE,
)


RECORD_SCHEMA = "r12_ctaa_signed_statistical_gate_spec_record_v1"
SPEC_SCHEMA = "r12_ctaa_statistical_gate_spec_v1"
SIGNATURE_DOMAIN = b"shohin.ctaa.statistical-gate-spec.v1\x00"
SEED_COUNT = 5
BOOTSTRAP_DRAWS = 100_000
RUNTIME_ANCHORS = 864
RENDERER_ROWS = 108
ONE_SIDED_ALPHA = 0.05

_MAX_FILE_BYTES = 4 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_RECORD_KEYS = frozenset({"schema", "payload", "signature", "gate_spec_sha256"})
_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "bindings",
        "partition_policy",
        "end_to_end_family_metric",
        "absolute_gates",
        "compiler_frontend_gates",
        "finite_core_audits",
        "runtime_intervention_gates",
        "paired_arm_effects",
        "bootstrap",
        "multiple_testing",
        "feasibility",
        "scope",
        "signing_public_key",
    }
)
_BINDING_KEYS = frozenset(
    {
        "manifest_sha256",
        "board_sha256",
        "run_plan_sha256",
        "run_contract_sha256",
        "runtime_bundle_file_sha256",
        "runtime_bundle_sha256",
        "runtime_execution_set_file_sha256",
        "runtime_execution_set_sha256",
        "assessment_source_bundle_sha256",
        "assessment_source_manifest_sha256",
        "bootstrap_seed_receipt_sha256",
        "bootstrap_seed",
        "training_seeds",
    }
)
_HASH_BINDINGS = tuple(sorted(_BINDING_KEYS - {"bootstrap_seed", "training_seeds"}))

_EXACT_INVARIANCE_OUTCOMES = frozenset(
    {
        "terminal_invariance",
        "active_prefix_invariance",
        "prefix_before_exposure_invariance",
    }
)
_CUSTODY_OPERATIONS = frozenset({"source_deletion", "query_isolation"})
_ZERO_ABLATIONS = frozenset({"h19_zero", "h29_zero"})


class StatisticalGateSpecError(ValueError):
    """The statistical preregistration failed closed."""


@dataclass(frozen=True)
class StatisticalGateBindings:
    """Content bindings that must be known before outcomes are opened."""

    manifest_sha256: str
    board_sha256: str
    run_plan_sha256: str
    run_contract_sha256: str
    runtime_bundle_file_sha256: str
    runtime_bundle_sha256: str
    runtime_execution_set_file_sha256: str
    runtime_execution_set_sha256: str
    assessment_source_bundle_sha256: str
    assessment_source_manifest_sha256: str
    bootstrap_seed_receipt_sha256: str
    bootstrap_seed: int
    training_seeds: tuple[int, int, int, int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "board_sha256": self.board_sha256,
            "run_plan_sha256": self.run_plan_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "runtime_bundle_file_sha256": self.runtime_bundle_file_sha256,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "runtime_execution_set_file_sha256": (
                self.runtime_execution_set_file_sha256
            ),
            "runtime_execution_set_sha256": self.runtime_execution_set_sha256,
            "assessment_source_bundle_sha256": self.assessment_source_bundle_sha256,
            "assessment_source_manifest_sha256": (
                self.assessment_source_manifest_sha256
            ),
            "bootstrap_seed_receipt_sha256": self.bootstrap_seed_receipt_sha256,
            "bootstrap_seed": self.bootstrap_seed,
            "training_seeds": list(self.training_seeds),
        }


def canonical_json_bytes(value: object) -> bytes:
    """Return the only JSON byte representation accepted by this module."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise StatisticalGateSpecError(
            "gate specification is not canonical JSON"
        ) from error


def _canonical_clone(value: object) -> object:
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _HEX for character in value)
    )


def _require_hash(value: object, label: str) -> str:
    if not _is_lower_hex(value, 64):
        raise StatisticalGateSpecError(f"invalid {label} SHA-256")
    return str(value)


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise StatisticalGateSpecError(f"{label} schema differs")
    return dict(value)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StatisticalGateSpecError(f"duplicate gate specification key: {key}")
        result[key] = value
    return result


def _decode_object(raw: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise StatisticalGateSpecError(
            f"non-finite gate specification constant: {value}"
        )

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StatisticalGateSpecError("malformed gate specification JSON") from error
    if not isinstance(value, dict):
        raise StatisticalGateSpecError("gate specification root must be an object")
    return value


def _path_text(path: Path) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise StatisticalGateSpecError("gate specification path differs") from error
    if not isinstance(raw, str) or "\x00" in raw:
        raise StatisticalGateSpecError("gate specification path differs")
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw or raw == "/":
        raise StatisticalGateSpecError(
            "gate specification path must be absolute and normalized"
        )
    if any(component in {"", ".", ".."} for component in raw.split("/")[1:]):
        raise StatisticalGateSpecError("gate specification path is unsafe")
    return raw


def _open_parent_directory(path: Path) -> tuple[int, str]:
    raw = _path_text(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise StatisticalGateSpecError("O_NOFOLLOW is required")
    components = raw.split("/")[1:]
    name = components[-1]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise StatisticalGateSpecError(
                    "gate specification parent is missing, symlinked, or not a directory"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise StatisticalGateSpecError(
                    "gate specification parent is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, name
    except Exception:
        os.close(descriptor)
        raise


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_immutable_bytes(path: Path) -> bytes:
    parent_descriptor, name = _open_parent_directory(path)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise StatisticalGateSpecError(
                "gate specification input is missing or symlinked"
            ) from error
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_size <= 0
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise StatisticalGateSpecError(
                "gate specification input must be read-only, regular, and single-link"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise StatisticalGateSpecError(
                    "gate specification input changed during read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise StatisticalGateSpecError("gate specification input grew during read")
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise StatisticalGateSpecError(
                "gate specification input changed during read"
            )
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _write_immutable_bytes(path: Path, raw: bytes) -> None:
    if not raw or len(raw) > _MAX_FILE_BYTES:
        raise StatisticalGateSpecError("gate specification output size differs")
    parent_descriptor, name = _open_parent_directory(path)
    temporary = f".{name}.ctaa-stat-{os.getpid()}-{secrets.token_hex(12)}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise StatisticalGateSpecError(
                    "gate specification write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o222
            or metadata.st_size != len(raw)
        ):
            raise StatisticalGateSpecError(
                "gate specification immutable write verification failed"
            )
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"refusing existing gate specification: {path}"
            ) from None
        linked = True
        os.unlink(temporary, dir_fd=parent_descriptor)
        temporary = ""
        final = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_mode & 0o222
            or final.st_size != len(raw)
        ):
            raise StatisticalGateSpecError(
                "published gate specification is not immutable"
            )
        os.fsync(parent_descriptor)
    except OSError as error:
        if linked:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if isinstance(error, FileExistsError):
            raise
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise StatisticalGateSpecError(
                "gate specification path contains a symlink"
            ) from error
        raise
    except Exception:
        if linked:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _public_key_bytes(verification_key: bytes | Ed25519PublicKey) -> bytes:
    if isinstance(verification_key, Ed25519PublicKey):
        raw = verification_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    elif isinstance(verification_key, bytes):
        raw = verification_key
    else:
        raise StatisticalGateSpecError("unsupported Ed25519 public key type")
    if len(raw) != 32:
        raise StatisticalGateSpecError("Ed25519 public key must be 32 bytes")
    try:
        Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as error:
        raise StatisticalGateSpecError("invalid Ed25519 public key") from error
    return raw


def _signing_public_key(signing_key: Ed25519PrivateKey) -> bytes:
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise TypeError("signing_key must be an Ed25519PrivateKey")
    return _public_key_bytes(signing_key.public_key())


def _validate_bindings(value: object) -> dict[str, object]:
    bindings = _exact_mapping(value, _BINDING_KEYS, "gate specification bindings")
    for key in _HASH_BINDINGS:
        _require_hash(bindings[key], key)
    bootstrap_seed = bindings["bootstrap_seed"]
    if type(bootstrap_seed) is not int or not 0 <= int(bootstrap_seed) < 2**64:
        raise StatisticalGateSpecError("bootstrap seed differs")
    seeds = bindings["training_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != SEED_COUNT
        or any(type(seed) is not int or not 0 <= int(seed) < 2**64 for seed in seeds)
        or len(set(seeds)) != SEED_COUNT
        or seeds != sorted(seeds)
    ):
        raise StatisticalGateSpecError(
            "training seeds must be exactly five distinct ordered values"
        )
    return bindings


def _partition_policy() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_partition_policy_v1",
        "ordered_partitions": ["development", "confirmation"],
        "development_access_limit": 1,
        "confirmation_access_limit": 1,
        "confirmation_requires_all_development_primary_gates": True,
        "same_signed_gate_spec_required": True,
        "threshold_changes_after_development_forbidden": True,
        "confirmation_remains_sealed_on_development_failure": True,
    }


def _end_to_end_metric() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_end_to_end_family_metric_v1",
        "name": "end_to_end_family_success",
        "definition": "prefix_exact AND terminal_exact AND answer_exact",
        "required_boolean_fields": [
            "prefix_exact",
            "terminal_exact",
            "answer_exact",
        ],
        "failure_value": 0,
        "all_failures_retained_in_denominator": True,
        "missing_or_invalid_outcome_value": 0,
    }


def _absolute_gates() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_absolute_gate_policy_v1",
        "metric": "end_to_end_family_success",
        "strata": [
            {
                "name": name,
                "minimum_rate": 0.95,
                "minimum_one_sided_cp_lower": 0.90,
                "alpha": ONE_SIDED_ALPHA,
            }
            for name in ("training_seed", "factorial_cell", "depth")
        ],
        "all_strata_must_pass": True,
        "pooling_across_failed_strata_forbidden": True,
    }


def _compiler_frontend_gates() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_compiler_frontend_gate_policy_v1",
        "minimum_component_rate": 0.99,
        "components": [
            "packet_valid",
            "independent_binding_exact",
            "initial_exact",
            "stop_exact",
            "schedule_exact",
            "program_exact",
            "query_exact",
        ],
        "all_components_must_pass": True,
        "binding_metric": {
            "name": "independent_binding_exact",
            "definition": (
                "predicted opcode-to-card bindings equal oracle opcode-to-card "
                "bindings for every opcode, scored independently of card contents"
            ),
            "forbidden_aliases": ["cards_exact"],
            "cards_exact_is_binding_metric": False,
        },
    }


def _finite_core_audits() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_exact_finite_core_audit_policy_v1",
        "receipt_count": 20,
        "receipt_crossing": "five_training_seeds_x_four_arms",
        "semantic_axes": [
            {
                "name": axis,
                "required_atomic_cases": 243,
                "required_atomic_exact": 1.0,
                "required_two_action_cases": 2_187,
                "required_two_action_exact": 1.0,
                "required_composition_exact": 1.0,
                "required_route_agreement": 1.0,
            }
            for axis in ("train", "development", "confirmation")
        ],
        "finite_enumerations_not_treated_as_repeated_samples": True,
        "all_receipts_and_axes_must_pass": True,
    }


def _runtime_criteria(operation: str, outcome: str) -> list[dict[str, object]]:
    criteria: list[dict[str, object]] = []
    if outcome in _EXACT_INVARIANCE_OUTCOMES:
        criteria.append(
            {
                "kind": "exact_invariance",
                "metric": "relation_correct",
                "required_rate": 1.0,
            }
        )
    elif operation in _CUSTODY_OPERATIONS:
        criteria.append(
            {
                "kind": "custody_exact",
                "metric": "custody_receipt_valid",
                "required_rate": 1.0,
            }
        )
    elif outcome == "route_agreement":
        criteria.append(
            {
                "kind": "custody_exact",
                "metric": "state_route_composed_route_agreement",
                "required_rate": 1.0,
            }
        )
    else:
        criteria.append(
            {
                "kind": "one_sided_clopper_pearson",
                "metric": "relation_correct",
                "alpha": ONE_SIDED_ALPHA,
                "minimum_lower_bound": 0.95,
            }
        )
    if operation in _ZERO_ABLATIONS:
        criteria.append(
            {
                "kind": "paired_directional_drop",
                "metric": "parent_minus_zero_ablation_end_to_end_family_success",
                "minimum_drop": 0.10,
            }
        )
    return criteria


def _runtime_intervention_gates() -> dict[str, object]:
    operations = []
    for operation in MANDATORY_OPERATIONS:
        specification = OPERATION_SPECS[operation]
        outcome = specification.expected.outcome.value
        operations.append(
            {
                "operation": operation,
                "expected_outcome": outcome,
                "anchor_count": RUNTIME_ANCHORS,
                "criteria": _runtime_criteria(operation, outcome),
            }
        )
    return {
        "schema": "r12_ctaa_runtime_statistical_gate_policy_v1",
        "operation_order": list(MANDATORY_OPERATIONS),
        "operations": operations,
        "all_operations_must_pass": True,
        "operation_pooling_forbidden": True,
        "failed_attempts_retained_as_zero": True,
    }


def _paired_arm_effects() -> dict[str, object]:
    comparisons = [
        ("ctaa_closure", "oprc_closure"),
        ("ctaa_closure", "ctaa_no_closure"),
        ("ctaa_closure", "ctaa_shuffled_closure"),
    ]
    return {
        "schema": "r12_ctaa_paired_arm_effect_policy_v1",
        "metric": "end_to_end_family_success",
        "comparisons": [
            {
                "treatment": treatment,
                "control": control,
                "minimum_crossed_bootstrap_lower_effect": 0.10,
                "alternative": "greater",
            }
            for treatment, control in comparisons
        ],
        "all_comparisons_must_pass": True,
        "unpaired_or_pooled_comparisons_forbidden": True,
    }


def _bootstrap_policy() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_crossed_bootstrap_policy_v1",
        "method": "crossed_training_seed_by_shared_family_root",
        "training_seed_count": SEED_COUNT,
        "family_root_field": "cluster_family_id",
        "frozen_strata": ["factorial_cell", "program_class", "depth"],
        "pairing": (
            "each sampled seed and shared family-root draw is reused across every "
            "paired arm before differencing"
        ),
        "seed_resampling": "sample_training_seeds_with_replacement",
        "family_resampling": (
            "sample_shared_family_roots_with_replacement_within_each_frozen_stratum"
        ),
        "independent_within_seed_family_resampling": False,
        "draws": BOOTSTRAP_DRAWS,
        "dtype": "float64",
        "numpy_bit_generator": "numpy.random.PCG64",
        "seed_binding": "bindings.bootstrap_seed",
        "seed_receipt_binding": "bindings.bootstrap_seed_receipt_sha256",
        "missing_or_failed_family_value": 0,
    }


def _multiple_testing_policy() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_multiple_testing_policy_v1",
        "primary_claim": {
            "method": "intersection_union",
            "all_primary_gates_must_pass": True,
            "holm_applied": False,
            "families": [
                "absolute_end_to_end",
                "compiler_frontend",
                "finite_core",
                "runtime_interventions",
                "paired_arm_effects",
            ],
        },
        "secondary_families": [
            {
                "name": "compiler_component_diagnostics",
                "method": "holm",
                "alpha": ONE_SIDED_ALPHA,
            },
            {
                "name": "factorial_axis_diagnostics",
                "method": "holm",
                "alpha": ONE_SIDED_ALPHA,
            },
            {
                "name": "runtime_operation_diagnostics",
                "method": "holm",
                "alpha": ONE_SIDED_ALPHA,
            },
        ],
        "holm_outside_explicit_secondary_families_forbidden": True,
    }


def _feasibility_policy() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_gate_feasibility_policy_v1",
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "runtime_anchor_count": RUNTIME_ANCHORS,
        "renderer_rows": RENDERER_ROWS,
        "renderer_level_claim_mode": "descriptive_only",
        "renderer_level_primary_or_secondary_claims_forbidden": True,
        "perfect_success_cp_lower_formula": "alpha**(1/n)",
    }


def _scope_policy() -> dict[str, object]:
    return {
        "schema": "r12_ctaa_statistical_gate_scope_v1",
        "purpose": "outcome_free_preregistration_only",
        "consumes_outcomes": False,
        "computes_capability": False,
        "authorizes_training": False,
        "authorizes_confirmation": False,
        "authorizes_capability_claim": False,
    }


def _expected_policy() -> dict[str, object]:
    return {
        "partition_policy": _partition_policy(),
        "end_to_end_family_metric": _end_to_end_metric(),
        "absolute_gates": _absolute_gates(),
        "compiler_frontend_gates": _compiler_frontend_gates(),
        "finite_core_audits": _finite_core_audits(),
        "runtime_intervention_gates": _runtime_intervention_gates(),
        "paired_arm_effects": _paired_arm_effects(),
        "bootstrap": _bootstrap_policy(),
        "multiple_testing": _multiple_testing_policy(),
        "feasibility": _feasibility_policy(),
        "scope": _scope_policy(),
    }


def _perfect_success_cp_lower(total: int, alpha: float) -> float:
    if total < 1 or not 0.0 < alpha < 1.0:
        raise StatisticalGateSpecError("Clopper-Pearson feasibility inputs differ")
    return math.exp(math.log(alpha) / total)


def validate_feasibility(payload: Mapping[str, object]) -> None:
    """Reject signed but statistically impossible or underpowered claims."""

    feasibility = payload.get("feasibility")
    if not isinstance(feasibility, Mapping):
        raise StatisticalGateSpecError("feasibility policy schema differs")
    alpha = feasibility.get("one_sided_alpha")
    anchors = feasibility.get("runtime_anchor_count")
    if type(alpha) is not float or type(anchors) is not int:
        raise StatisticalGateSpecError("feasibility policy inputs differ")
    maximum_lower = _perfect_success_cp_lower(int(anchors), float(alpha))

    runtime = payload.get("runtime_intervention_gates")
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("operations"), list
    ):
        raise StatisticalGateSpecError("runtime gate feasibility schema differs")
    for operation in runtime["operations"]:
        if not isinstance(operation, Mapping) or not isinstance(
            operation.get("criteria"), list
        ):
            raise StatisticalGateSpecError(
                "runtime operation feasibility schema differs"
            )
        if operation.get("anchor_count") != anchors:
            raise StatisticalGateSpecError(
                "runtime operations must use exactly 864 anchors"
            )
        for criterion in operation["criteria"]:
            if not isinstance(criterion, Mapping):
                raise StatisticalGateSpecError("runtime criterion schema differs")
            if criterion.get("kind") == "one_sided_clopper_pearson":
                threshold = criterion.get("minimum_lower_bound")
                if type(threshold) is not float or float(threshold) > maximum_lower:
                    raise StatisticalGateSpecError(
                        "Clopper-Pearson lower threshold is infeasible at 864 anchors"
                    )

    if (
        feasibility.get("renderer_rows") != RENDERER_ROWS
        or feasibility.get("renderer_level_claim_mode") != "descriptive_only"
        or feasibility.get("renderer_level_primary_or_secondary_claims_forbidden")
        is not True
    ):
        raise StatisticalGateSpecError(
            "renderer-level claims are underpowered by 108 rows"
        )
    absolute = payload.get("absolute_gates")
    if isinstance(absolute, Mapping) and isinstance(absolute.get("strata"), list):
        if any(
            isinstance(item, Mapping) and item.get("name") == "renderer"
            for item in absolute["strata"]
        ):
            raise StatisticalGateSpecError(
                "renderer-level claims are underpowered by 108 rows"
            )


def _validate_policy(payload: Mapping[str, object]) -> None:
    validate_feasibility(payload)
    expected = _expected_policy()
    for key, value in expected.items():
        if payload.get(key) != value:
            if key == "compiler_frontend_gates":
                binding = payload.get(key)
                if isinstance(binding, Mapping):
                    metric = binding.get("binding_metric")
                    if isinstance(metric, Mapping) and (
                        metric.get("name") == "cards_exact"
                        or metric.get("cards_exact_is_binding_metric") is True
                    ):
                        raise StatisticalGateSpecError(
                            "cards_exact is not an independent binding metric"
                        )
            if key == "bootstrap":
                raise StatisticalGateSpecError(
                    "crossed seed x shared family-root bootstrap semantics differ"
                )
            raise StatisticalGateSpecError(f"{key} policy differs")


def _bindings_dict(
    value: StatisticalGateBindings | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, StatisticalGateBindings):
        result = value.as_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise StatisticalGateSpecError("expected gate bindings differ")
    return _validate_bindings(result)


def make_signed_statistical_gate_spec(
    *,
    bindings: StatisticalGateBindings | Mapping[str, object],
    signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Construct and sign the sole accepted outcome-free gate specification."""

    raw_key = _signing_public_key(signing_key)
    payload: dict[str, object] = {
        "schema": SPEC_SCHEMA,
        "bindings": _bindings_dict(bindings),
        **_expected_policy(),
        "signing_public_key": raw_key.hex(),
    }
    signature = signing_key.sign(SIGNATURE_DOMAIN + canonical_json_bytes(payload)).hex()
    record = {
        "schema": RECORD_SCHEMA,
        "payload": payload,
        "signature": signature,
        "gate_spec_sha256": _sha256_bytes(canonical_json_bytes(payload)),
    }
    return validate_signed_statistical_gate_spec(
        record,
        verification_key=raw_key,
        expected_bindings=bindings,
    )


def validate_signed_statistical_gate_spec(
    value: Mapping[str, object],
    *,
    verification_key: bytes | Ed25519PublicKey,
    expected_bindings: StatisticalGateBindings | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Authenticate and validate a frozen preregistration without reading outcomes."""

    record = _exact_mapping(value, _RECORD_KEYS, "signed gate specification record")
    if record["schema"] != RECORD_SCHEMA:
        raise StatisticalGateSpecError(
            "signed gate specification record schema differs"
        )
    payload = _exact_mapping(
        record["payload"], _PAYLOAD_KEYS, "statistical gate specification"
    )
    if payload["schema"] != SPEC_SCHEMA:
        raise StatisticalGateSpecError("statistical gate specification schema differs")
    bindings = _validate_bindings(payload["bindings"])
    if expected_bindings is not None and bindings != _bindings_dict(expected_bindings):
        raise StatisticalGateSpecError("statistical gate binding substitution detected")
    raw_key = _public_key_bytes(verification_key)
    if payload["signing_public_key"] != raw_key.hex():
        raise StatisticalGateSpecError("statistical gate signing key differs")
    signature = record["signature"]
    if not _is_lower_hex(signature, 128):
        raise StatisticalGateSpecError("malformed Ed25519 signature")
    try:
        Ed25519PublicKey.from_public_bytes(raw_key).verify(
            bytes.fromhex(str(signature)),
            SIGNATURE_DOMAIN + canonical_json_bytes(payload),
        )
    except InvalidSignature as error:
        raise StatisticalGateSpecError(
            "Ed25519 signature verification failed"
        ) from error
    expected_hash = _sha256_bytes(canonical_json_bytes(payload))
    if record["gate_spec_sha256"] != expected_hash:
        raise StatisticalGateSpecError("statistical gate specification hash differs")
    _validate_policy(payload)
    return {
        "schema": RECORD_SCHEMA,
        "payload": payload,
        "signature": signature,
        "gate_spec_sha256": expected_hash,
    }


def write_signed_statistical_gate_spec(
    path: Path,
    *,
    bindings: StatisticalGateBindings | Mapping[str, object],
    signing_key: Ed25519PrivateKey,
) -> str:
    """Publish one immutable signed preregistration and return its file hash."""

    record = make_signed_statistical_gate_spec(
        bindings=bindings,
        signing_key=signing_key,
    )
    raw = canonical_json_bytes(record) + b"\n"
    _write_immutable_bytes(Path(path), raw)
    if _read_immutable_bytes(Path(path)) != raw:
        raise StatisticalGateSpecError("published gate specification differs")
    return _sha256_bytes(raw)


def read_signed_statistical_gate_spec_with_sha(
    path: Path,
    *,
    verification_key: bytes | Ed25519PublicKey,
    expected_bindings: StatisticalGateBindings | Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    """Read a canonical immutable preregistration and verify all bindings."""

    raw = _read_immutable_bytes(Path(path))
    value = _decode_object(raw)
    if raw != canonical_json_bytes(value) + b"\n":
        raise StatisticalGateSpecError("gate specification file is not canonical JSON")
    verified = validate_signed_statistical_gate_spec(
        value,
        verification_key=verification_key,
        expected_bindings=expected_bindings,
    )
    return verified, _sha256_bytes(raw)


def read_signed_statistical_gate_spec(
    path: Path,
    *,
    verification_key: bytes | Ed25519PublicKey,
    expected_bindings: StatisticalGateBindings | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Read and authenticate an immutable preregistration."""

    verified, _ = read_signed_statistical_gate_spec_with_sha(
        path,
        verification_key=verification_key,
        expected_bindings=expected_bindings,
    )
    return verified


if RUNTIME_PANEL_SIZE != RUNTIME_ANCHORS:  # pragma: no cover - import invariant
    raise RuntimeError("CTAA runtime anchor contract differs")
