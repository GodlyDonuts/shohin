"""Immutable five-seed custody bundle for CTAA runtime interventions.

Runtime interventions are a causal sidecar, not extra scored examples.  This
module binds exactly one 864-anchor/29-operation plan and its raw evidence to
each of the five ``ctaa_closure`` base runs without changing the locked 40,608
row assessment denominator.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Mapping, Sequence

from ctaa_intervention_protocol import (
    LOCKED_SCORED_ROW_COUNT,
    MANDATORY_OPERATIONS,
    RUNTIME_PANEL_SIZE,
    RuntimeInterventionPlan,
    plan_to_dict,
    validate_runtime_intervention_plan,
)
from ctaa_run_contract import (
    RUN_CONTRACT_SCHEMA,
    RUN_COUNT,
    SEED_COUNT,
    canonical_sha256,
)
from ctaa_runtime_evidence import read_runtime_evidence, validate_runtime_evidence
from ctaa_runtime_plan_replay import load_runtime_replay_rows, replay_runtime_plan


BUNDLE_SCHEMA = "r12_ctaa_runtime_bundle_v1"
ENTRY_SCHEMA = "r12_ctaa_runtime_bundle_entry_v1"
TREATMENT_ARM = "ctaa_closure"
BASE_DATASET = "base"
ATTEMPT_COUNT_PER_SEED = RUNTIME_PANEL_SIZE * len(MANDATORY_OPERATIONS)

_HEX = frozenset("0123456789abcdef")
_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "partition",
        "manifest_sha256",
        "board_sha256",
        "run_plan_sha256",
        "bootstrap_seed_receipt_sha256",
        "bootstrap_seed",
        "training_seeds",
        "arms",
        "datasets",
        "run_count",
        "oracle_files",
        "runs",
        "run_contract_sha256",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "schema",
        "training_seed",
        "compiler_sha256",
        "core_sha256",
        "core_kind",
        "base_raw_evidence_receipt_sha256",
        "runtime_plan_filename",
        "runtime_plan_file_sha256",
        "runtime_plan_sha256",
        "runtime_evidence_filename",
        "runtime_evidence_file_sha256",
        "runtime_evidence_sha256",
        "entry_sha256",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "partition",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "bootstrap_seed_receipt_sha256",
        "selection_seed",
        "selection_seed_receipt_sha256",
        "arm_id",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "runtime_implementation_sha256",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order_sha256",
        "seed_count",
        "runtime_panel_size_per_seed",
        "operation_count",
        "attempt_count_per_seed",
        "scored_row_count",
        "runtime_attempts_affect_scored_denominator",
        "oracle_access",
        "entries",
        "bundle_sha256",
    }
)


class RuntimeBundleError(ValueError):
    """A runtime bundle or one of its immutable members differs."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeBundleError("CTAA runtime bundle is not canonical JSON") from error


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_hash(value: object, label: str) -> str:
    if not _is_hash(value):
        raise RuntimeBundleError(f"CTAA runtime bundle {label} hash differs")
    return str(value)


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeBundleError(f"CTAA runtime bundle {label} schema differs")
    return dict(value)


def _safe_filename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RuntimeBundleError(f"CTAA runtime bundle {label} filename differs")
    pure = PurePosixPath(value)
    if pure.name != value or value in {".", ".."} or "\x00" in value:
        raise RuntimeBundleError(f"CTAA runtime bundle {label} filename is unsafe")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeBundleError(f"CTAA runtime bundle duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise RuntimeBundleError(
                f"CTAA runtime {label} cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeBundleError(f"CTAA runtime {label} contains a symlink")


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeBundleError(f"CTAA runtime {label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        raise RuntimeBundleError(
            f"CTAA runtime {label} is not a single-link read-only file"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731 - compact immutable-file identity
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(metadata) != identity(before)
        or identity(before) != identity(after)
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise RuntimeBundleError(f"CTAA runtime {label} changed while being read")
    return b"".join(chunks)


def _decode_object(raw: bytes, label: str) -> dict[str, object]:
    def reject_nonfinite(value: str) -> None:
        raise RuntimeBundleError(f"CTAA runtime {label} has non-finite JSON: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeBundleError(f"CTAA runtime {label} JSON differs") from error
    if not isinstance(value, dict):
        raise RuntimeBundleError(f"CTAA runtime {label} root differs")
    return value


def _read_canonical_object(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = _read_immutable_bytes(path, label)
    value = _decode_object(raw, label)
    if raw != (_canonical_json(value) + "\n").encode("ascii"):
        raise RuntimeBundleError(f"CTAA runtime {label} is not canonical JSON")
    return value, hashlib.sha256(raw).hexdigest()


def read_runtime_plan(path: Path) -> tuple[RuntimeInterventionPlan, str]:
    value, digest = _read_canonical_object(path, "intervention plan")
    plan = validate_runtime_intervention_plan(value)
    if value != plan_to_dict(plan):
        raise RuntimeBundleError("CTAA runtime intervention plan normalization differs")
    return plan, digest


def _validate_contract(value: Mapping[str, object]) -> dict[str, object]:
    contract = _exact_mapping(value, _CONTRACT_KEYS, "run contract")
    if (
        contract["schema"] != RUN_CONTRACT_SCHEMA
        or contract["run_count"] != RUN_COUNT
        or not isinstance(contract["runs"], list)
        or len(contract["runs"]) != RUN_COUNT
    ):
        raise RuntimeBundleError("CTAA runtime run contract identity differs")
    expected_hash = canonical_sha256(
        {key: item for key, item in contract.items() if key != "run_contract_sha256"}
    )
    if contract["run_contract_sha256"] != expected_hash:
        raise RuntimeBundleError("CTAA runtime run contract commitment differs")
    return contract


def _treatment_runs(contract: Mapping[str, object]) -> dict[int, dict[str, object]]:
    rows = contract["runs"]
    assert isinstance(rows, list)
    selected: dict[int, dict[str, object]] = {}
    for value in rows:
        if not isinstance(value, dict):
            raise RuntimeBundleError("CTAA runtime run contract entry differs")
        if value.get("arm") != TREATMENT_ARM or value.get("dataset") != BASE_DATASET:
            continue
        seed = value.get("seed")
        if type(seed) is not int or seed < 0 or seed in selected:
            raise RuntimeBundleError("CTAA runtime treatment seed set differs")
        selected[seed] = value
    expected_seeds = contract.get("training_seeds")
    if (
        not isinstance(expected_seeds, list)
        or len(selected) != SEED_COUNT
        or sorted(selected) != expected_seeds
    ):
        raise RuntimeBundleError("CTAA runtime treatment seed coverage differs")
    return selected


def _entry_payload(
    *,
    run: Mapping[str, object],
    plan: RuntimeInterventionPlan,
    evidence: Mapping[str, object],
    plan_filename: str,
    plan_file_sha256: str,
    evidence_filename: str,
    evidence_file_sha256: str,
) -> dict[str, object]:
    training = run.get("core_training")
    if not isinstance(training, Mapping):
        raise RuntimeBundleError("CTAA runtime treatment core metadata differs")
    binding = plan.bindings
    expected = {
        "training_seed": run.get("seed"),
        "compiler_sha256": run.get("compiler_sha256"),
        "core_sha256": training.get("core_sha256"),
        "core_kind": training.get("core_kind"),
        "base_raw_evidence_receipt_sha256": run.get("raw_evidence_receipt_sha256"),
    }
    observed = {
        "training_seed": binding.training_seed,
        "compiler_sha256": binding.compiler_sha256,
        "core_sha256": binding.core_sha256,
        "core_kind": binding.core_kind,
        "base_raw_evidence_receipt_sha256": (binding.base_raw_evidence_receipt_sha256),
    }
    if observed != expected:
        raise RuntimeBundleError("CTAA runtime plan differs from treatment run")
    for key, item in evidence.items():
        if key in observed and item != observed[key]:
            raise RuntimeBundleError(f"CTAA runtime evidence {key} binding differs")
    evidence_sha = _require_hash(evidence.get("evidence_sha256"), "evidence")
    payload: dict[str, object] = {
        "schema": ENTRY_SCHEMA,
        **observed,
        "runtime_plan_filename": _safe_filename(plan_filename, "plan"),
        "runtime_plan_file_sha256": _require_hash(plan_file_sha256, "plan file"),
        "runtime_plan_sha256": plan.plan_sha256,
        "runtime_evidence_filename": _safe_filename(evidence_filename, "evidence"),
        "runtime_evidence_file_sha256": _require_hash(
            evidence_file_sha256, "evidence file"
        ),
        "runtime_evidence_sha256": evidence_sha,
    }
    payload["entry_sha256"] = _canonical_hash(payload)
    return payload


def make_runtime_bundle(
    *,
    run_contract: Mapping[str, object],
    artifacts: Sequence[
        tuple[
            RuntimeInterventionPlan,
            Mapping[str, object],
            str,
            str,
            str,
            str,
        ]
    ],
) -> dict[str, object]:
    """Build a logical bundle from already validated plan/evidence artifacts."""

    contract = _validate_contract(run_contract)
    treatment = _treatment_runs(contract)
    entries = []
    plans: list[RuntimeInterventionPlan] = []
    filenames: set[str] = set()
    for (
        plan,
        evidence,
        plan_name,
        plan_file_sha,
        evidence_name,
        evidence_file_sha,
    ) in artifacts:
        frozen = validate_runtime_intervention_plan(plan)
        evidence = validate_runtime_evidence(evidence, frozen)
        seed = frozen.bindings.training_seed
        if seed not in treatment:
            raise RuntimeBundleError("CTAA runtime plan seed is not a treatment seed")
        for name in (plan_name, evidence_name):
            _safe_filename(name, "artifact")
            if name in filenames:
                raise RuntimeBundleError("CTAA runtime artifact filename repeats")
            filenames.add(name)
        entries.append(
            _entry_payload(
                run=treatment[seed],
                plan=frozen,
                evidence=evidence,
                plan_filename=plan_name,
                plan_file_sha256=plan_file_sha,
                evidence_filename=evidence_name,
                evidence_file_sha256=evidence_file_sha,
            )
        )
        plans.append(frozen)
    entries.sort(key=lambda row: int(row["training_seed"]))
    plans.sort(key=lambda plan: plan.bindings.training_seed)
    if len(entries) != SEED_COUNT or [
        row["training_seed"] for row in entries
    ] != sorted(treatment):
        raise RuntimeBundleError("CTAA runtime bundle seed coverage differs")
    shared_fields = (
        "board_manifest_sha256",
        "board_tree_sha256",
        "run_contract_sha256",
        "selection_seed",
        "selection_seed_receipt_sha256",
        "arm_id",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "runtime_implementation_sha256",
        "batch_order_sha256",
    )
    reference = plans[0].bindings
    if any(
        getattr(plan.bindings, key) != getattr(reference, key)
        for plan in plans[1:]
        for key in shared_fields
    ) or any(
        plan.anchor_panel_sha256 != plans[0].anchor_panel_sha256
        or plan.donor_registry_sha256 != plans[0].donor_registry_sha256
        for plan in plans[1:]
    ):
        raise RuntimeBundleError("CTAA runtime five-seed panel binding differs")
    if (
        reference.board_manifest_sha256 != contract["manifest_sha256"]
        or reference.board_tree_sha256 != contract["board_sha256"]
        or reference.run_contract_sha256 != contract["run_contract_sha256"]
        or reference.arm_id != TREATMENT_ARM
        or reference.partition.value != contract["partition"]
    ):
        raise RuntimeBundleError("CTAA runtime bundle differs from run contract")
    bundle: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "partition": contract["partition"],
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "selection_seed": reference.selection_seed,
        "selection_seed_receipt_sha256": reference.selection_seed_receipt_sha256,
        "arm_id": TREATMENT_ARM,
        "tokenizer_sha256": reference.tokenizer_sha256,
        "base_checkpoint_sha256": reference.base_checkpoint_sha256,
        "runtime_implementation_sha256": reference.runtime_implementation_sha256,
        "anchor_panel_sha256": plans[0].anchor_panel_sha256,
        "donor_registry_sha256": plans[0].donor_registry_sha256,
        "batch_order_sha256": reference.batch_order_sha256,
        "seed_count": SEED_COUNT,
        "runtime_panel_size_per_seed": RUNTIME_PANEL_SIZE,
        "operation_count": len(MANDATORY_OPERATIONS),
        "attempt_count_per_seed": ATTEMPT_COUNT_PER_SEED,
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "oracle_access": 0,
        "entries": entries,
    }
    bundle["bundle_sha256"] = _canonical_hash(bundle)
    return validate_runtime_bundle(bundle, run_contract=contract)


def validate_runtime_bundle(
    value: Mapping[str, object], *, run_contract: Mapping[str, object]
) -> dict[str, object]:
    contract = _validate_contract(run_contract)
    treatment = _treatment_runs(contract)
    bundle = _exact_mapping(value, _BUNDLE_KEYS, "root")
    expected_top = {
        "schema": BUNDLE_SCHEMA,
        "partition": contract["partition"],
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "arm_id": TREATMENT_ARM,
        "seed_count": SEED_COUNT,
        "runtime_panel_size_per_seed": RUNTIME_PANEL_SIZE,
        "operation_count": len(MANDATORY_OPERATIONS),
        "attempt_count_per_seed": ATTEMPT_COUNT_PER_SEED,
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "oracle_access": 0,
    }
    for key, expected in expected_top.items():
        if bundle[key] != expected:
            raise RuntimeBundleError(f"CTAA runtime bundle {key} differs")
    for key in (
        "selection_seed_receipt_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "runtime_implementation_sha256",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order_sha256",
    ):
        _require_hash(bundle[key], key)
    if type(bundle["selection_seed"]) is not int or bundle["selection_seed"] < 0:
        raise RuntimeBundleError("CTAA runtime bundle selection seed differs")
    rows = bundle["entries"]
    if not isinstance(rows, list) or len(rows) != SEED_COUNT:
        raise RuntimeBundleError("CTAA runtime bundle entry count differs")
    entries = []
    filenames: set[str] = set()
    for value in rows:
        row = _exact_mapping(value, _ENTRY_KEYS, "entry")
        seed = row["training_seed"]
        if type(seed) is not int or seed not in treatment:
            raise RuntimeBundleError("CTAA runtime bundle entry seed differs")
        run = treatment[seed]
        training = run.get("core_training")
        if not isinstance(training, Mapping):
            raise RuntimeBundleError("CTAA runtime bundle core metadata differs")
        expected = {
            "schema": ENTRY_SCHEMA,
            "training_seed": seed,
            "compiler_sha256": run.get("compiler_sha256"),
            "core_sha256": training.get("core_sha256"),
            "core_kind": training.get("core_kind"),
            "base_raw_evidence_receipt_sha256": run.get("raw_evidence_receipt_sha256"),
        }
        if any(row[key] != item for key, item in expected.items()):
            raise RuntimeBundleError("CTAA runtime bundle treatment binding differs")
        for key in (
            "compiler_sha256",
            "core_sha256",
            "base_raw_evidence_receipt_sha256",
            "runtime_plan_file_sha256",
            "runtime_plan_sha256",
            "runtime_evidence_file_sha256",
            "runtime_evidence_sha256",
        ):
            _require_hash(row[key], key)
        for key in ("runtime_plan_filename", "runtime_evidence_filename"):
            name = _safe_filename(row[key], key)
            if name in filenames:
                raise RuntimeBundleError(
                    "CTAA runtime bundle artifact filename repeats"
                )
            filenames.add(name)
        expected_entry_sha = _canonical_hash(
            {key: item for key, item in row.items() if key != "entry_sha256"}
        )
        if row["entry_sha256"] != expected_entry_sha:
            raise RuntimeBundleError("CTAA runtime bundle entry commitment differs")
        entries.append(row)
    if [row["training_seed"] for row in entries] != sorted(treatment):
        raise RuntimeBundleError("CTAA runtime bundle entry order/coverage differs")
    expected_bundle_sha = _canonical_hash(
        {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    )
    if bundle["bundle_sha256"] != expected_bundle_sha:
        raise RuntimeBundleError("CTAA runtime bundle commitment differs")
    bundle["entries"] = entries
    return bundle


def read_runtime_bundle(
    bundle_path: Path, *, run_contract: Mapping[str, object]
) -> dict[str, object]:
    """Read the bundle and every referenced plan/evidence before oracle access."""

    bundle_value, _ = _read_canonical_object(bundle_path, "bundle")
    bundle = validate_runtime_bundle(bundle_value, run_contract=run_contract)
    root = Path(os.path.abspath(bundle_path)).parent
    artifacts = []
    for entry in bundle["entries"]:
        assert isinstance(entry, dict)
        plan_path = root / str(entry["runtime_plan_filename"])
        evidence_path = root / str(entry["runtime_evidence_filename"])
        if plan_path.parent != root or evidence_path.parent != root:
            raise RuntimeBundleError("CTAA runtime bundle member escapes package root")
        plan, plan_file_sha = read_runtime_plan(plan_path)
        if (
            plan_file_sha != entry["runtime_plan_file_sha256"]
            or plan.plan_sha256 != entry["runtime_plan_sha256"]
            or plan.bindings.training_seed != entry["training_seed"]
        ):
            raise RuntimeBundleError("CTAA runtime plan member differs")
        evidence = read_runtime_evidence(
            evidence_path,
            plan,
            expected_file_sha256=str(entry["runtime_evidence_file_sha256"]),
        )
        if evidence.get("evidence_sha256") != entry["runtime_evidence_sha256"]:
            raise RuntimeBundleError("CTAA runtime evidence member differs")
        artifacts.append(
            (
                plan,
                evidence,
                str(entry["runtime_plan_filename"]),
                plan_file_sha,
                str(entry["runtime_evidence_filename"]),
                str(entry["runtime_evidence_file_sha256"]),
            )
        )
    rebuilt = make_runtime_bundle(run_contract=run_contract, artifacts=artifacts)
    if rebuilt != bundle:
        raise RuntimeBundleError("CTAA runtime bundle member recomputation differs")
    return bundle


def read_runtime_bundle_with_replay(
    bundle_path: Path,
    *,
    run_contract: Mapping[str, object],
    program_path: Path,
    query_path: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    """Validate every member and independently replay all concrete recipes."""

    bundle = read_runtime_bundle(bundle_path, run_contract=run_contract)
    root = Path(os.path.abspath(bundle_path)).parent
    for entry in bundle["entries"]:
        if not isinstance(entry, Mapping):
            raise RuntimeBundleError("CTAA runtime replay entry differs")
        plan_path = root / str(entry["runtime_plan_filename"])
        plan, _ = read_runtime_plan(plan_path)
        try:
            rows = load_runtime_replay_rows(
                plan=plan,
                program_path=program_path,
                query_path=query_path,
                tokenizer_path=tokenizer_path,
            )
            replay = replay_runtime_plan(plan, rows)
        except ValueError as error:
            raise RuntimeBundleError(
                "CTAA runtime plan semantic replay failed"
            ) from error
        if replay.plan_sha256 != entry["runtime_plan_sha256"] or (
            replay.attempt_count != ATTEMPT_COUNT_PER_SEED
        ):
            raise RuntimeBundleError("CTAA runtime plan semantic replay differs")
    return bundle


def _publish_once(path: Path, payload: bytes) -> None:
    target = Path(os.path.abspath(path))
    _reject_symlink_components(target.parent, "bundle output parent")
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing CTAA runtime bundle: {target}")
    if not stat.S_ISDIR(target.parent.lstat().st_mode):
        raise RuntimeBundleError("CTAA runtime bundle output parent differs")
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeBundleError("CTAA runtime bundle write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            os.chmod(temporary, 0o600, follow_symlinks=False)
            temporary.unlink()


def write_runtime_bundle(
    path: Path,
    *,
    run_contract: Mapping[str, object],
    artifacts: Sequence[
        tuple[
            RuntimeInterventionPlan,
            Mapping[str, object],
            str,
            str,
            str,
            str,
        ]
    ],
) -> str:
    bundle = make_runtime_bundle(run_contract=run_contract, artifacts=artifacts)
    payload = (_canonical_json(bundle) + "\n").encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    _publish_once(path, payload)
    published, observed = _read_canonical_object(path, "published bundle")
    if observed != digest or published != bundle:
        raise RuntimeBundleError("CTAA runtime published bundle differs")
    return digest
