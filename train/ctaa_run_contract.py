#!/usr/bin/env python3
"""Build an immutable, outcome-free contract for one CTAA assessment access.

The contract is deliberately constructed before oracle access.  It opens only
the sealed board manifest, a precommitted run plan, and committed raw-evidence
receipts.  Oracle paths are derived from the manifest file map and are never
opened, stated, or resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping

from ctaa_bootstrap_seed_receipt import validate_receipt as validate_bootstrap_receipt


MANIFEST_SCHEMA = "r12_ctaa_v2_manifest_v2"
RAW_EVIDENCE_RECEIPT_SCHEMA = "r12_ctaa_v2_raw_evidence_receipt_v2"
CORE_TRAINING_SCHEMA = "r12_ctaa_v2_core_training_v1"
RUN_PLAN_SCHEMA = "r12_ctaa_v2_run_plan_v1"
RUN_INPUT_SCHEMA = "r12_ctaa_v2_run_input_v1"
RUN_BINDING_SCHEMA = "r12_ctaa_v2_run_binding_v1"
RUN_CONTRACT_SCHEMA = "r12_ctaa_v2_run_contract_v1"
BOARD_TREE_SCHEMA = "r12_ctaa_v2_board_tree_v1"

ARMS = (
    "ctaa_closure",
    "oprc_closure",
    "ctaa_no_closure",
    "ctaa_shuffled_closure",
)
DATASETS = ("base", "intervention")
PARTITIONS = ("development", "confirmation")
SEED_COUNT = 5
RUN_COUNT = SEED_COUNT * len(ARMS) * len(DATASETS)

_HEX = frozenset("0123456789abcdef")
_MANIFEST_KEYS = frozenset({"schema", "seed", "files"})
_RUN_PLAN_KEYS = frozenset(
    {
        "schema",
        "partition",
        "expected_manifest_sha256",
        "expected_board_sha256",
        "runs",
    }
)
_RUN_INPUT_KEYS = frozenset(
    {
        "schema",
        "seed",
        "arm",
        "dataset",
        "evidence_receipt_path",
        "parent_evidence_receipt_path",
        "core_training",
    }
)
_CORE_TRAINING_KEYS = frozenset(
    {
        "core_sha256",
        "core_kind",
        "training_schema",
        "training_seed",
        "training_arm",
        "atomic_sha256",
        "closure_sha256",
        "updates",
        "batch_size",
        "learning_rate",
    }
)
_RAW_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "rows",
        "valid_packets",
        "executed_rows",
        "queried_rows",
        "answered_rows",
        "program_predictions_sha256",
        "compiler_sha256",
        "program_source_sha256",
        "query_source_sha256",
        "packet_index_sha256",
        "execution_sha256",
        "core_sha256",
        "core_kind",
        "core_training",
        "query_predictions_sha256",
        "query_positions_sha256",
        "answers_sha256",
        "evidence_sha256",
        "oracle_access",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "program_predictions_sha256",
        "packet_index_sha256",
        "execution_sha256",
        "query_predictions_sha256",
        "query_positions_sha256",
        "answers_sha256",
        "evidence_sha256",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "program_filename",
        "program_sha256",
        "query_filename",
        "query_sha256",
        "oracle_filename",
        "oracle_sha256",
    }
)
_RUN_BINDING_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "seed",
        "arm",
        "dataset",
        "raw_evidence_receipt_sha256",
        "compiler_sha256",
        "sealed_sources",
        "evidence_artifacts",
        "core_training",
        "parent_run_id",
        "parent_evidence_receipt_sha256",
        "parent_evidence_sha256",
    }
)
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


class RunContractError(ValueError):
    """A CTAA pre-access run-contract input or receipt is invalid."""


def canonical_json(value: object) -> str:
    """Return the sole JSON encoding used for CTAA logical commitments."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _require_hash(value: object, label: str) -> str:
    if not _is_hash(value):
        raise RunContractError(f"CTAA {label} SHA-256 differs")
    return str(value)


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RunContractError(f"CTAA {label} schema differs")
    return value


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunContractError("CTAA JSON contains a duplicate key")
        result[key] = value
    return result


def _open_parent_directory(path: Path, label: str) -> tuple[Path, int]:
    """Open every lexical parent component without following symlinks."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(os.path.sep, flags)
        try:
            for component in absolute.parent.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as error:
        raise RunContractError(f"CTAA {label} cannot be opened safely") from error
    return absolute, descriptor


def _read_committed_bytes(
    path: Path, label: str, *, require_read_only_parent: bool = False
) -> bytes:
    """Read one immutable file through a held, no-symlink parent descriptor."""

    absolute, parent_descriptor = _open_parent_directory(path, label)
    try:
        parent_before = os.fstat(parent_descriptor)
        if require_read_only_parent and (
            not stat.S_ISDIR(parent_before.st_mode) or parent_before.st_mode & 0o222
        ):
            raise RunContractError(f"CTAA {label} directory is not read-only")
        try:
            metadata = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise RunContractError(f"CTAA {label} is unavailable") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_mode & 0o222
            or metadata.st_nlink != 1
        ):
            raise RunContractError(f"CTAA {label} is not a single-link read-only file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise RunContractError(f"CTAA {label} cannot be opened safely") from error
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
        parent_after = os.fstat(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or before.st_dev != metadata.st_dev
        or before.st_ino != metadata.st_ino
        or before.st_size != metadata.st_size
        or before.st_mtime_ns != metadata.st_mtime_ns
        or before.st_ctime_ns != metadata.st_ctime_ns
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise RunContractError(f"CTAA {label} changed while being read")
    return b"".join(chunks)


def _decode_json(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunContractError(f"CTAA {label} JSON differs") from error
    if not isinstance(value, dict):
        raise RunContractError(f"CTAA {label} JSON root differs")
    return value


def _read_committed_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    data = _read_committed_bytes(path, label)
    return _decode_json(data, label), _sha256_bytes(data)


def _safe_manifest_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RunContractError("CTAA manifest filename differs")
    pure = PurePosixPath(value)
    if pure.name != value or value in {".", ".."} or "\x00" in value:
        raise RunContractError("CTAA manifest filename is unsafe")
    return value


def _manifest_bindings(
    manifest_path: Path,
) -> tuple[dict[str, object], str, str, dict[str, str]]:
    manifest, manifest_sha = _read_committed_json(manifest_path, "board manifest")
    _exact_mapping(manifest, _MANIFEST_KEYS, "board manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or type(manifest["seed"]) is not int:
        raise RunContractError("CTAA board manifest identity differs")
    if int(manifest["seed"]) < 0 or not isinstance(manifest["files"], dict):
        raise RunContractError("CTAA board manifest metadata differs")
    files: dict[str, str] = {}
    for raw_name, raw_hash in manifest["files"].items():
        name = _safe_manifest_name(raw_name)
        files[name] = _require_hash(raw_hash, f"manifest file {name}")
    if not files or len(files) != len(manifest["files"]):
        raise RunContractError("CTAA board manifest file map differs")
    tree = {
        "schema": BOARD_TREE_SCHEMA,
        "files": [{"name": name, "sha256": files[name]} for name in sorted(files)],
    }
    return manifest, manifest_sha, canonical_sha256(tree), files


def inspect_sealed_manifest(manifest_path: Path) -> dict[str, str]:
    """Return pre-access registry anchors without opening any board member file."""

    _, manifest_sha, board_sha, _ = _manifest_bindings(manifest_path)
    return {"manifest_sha256": manifest_sha, "board_sha256": board_sha}


def _source_names(partition: str, dataset: str) -> dict[str, str]:
    infix = "" if dataset == "base" else "_intervention"
    prefix = f"{partition}{infix}"
    return {
        "program_filename": f"{prefix}_program.jsonl",
        "query_filename": f"{prefix}_query.jsonl",
        "oracle_filename": f"{prefix}_oracle.jsonl",
    }


def _sealed_sources(
    partition: str, dataset: str, files: Mapping[str, str]
) -> dict[str, str]:
    names = _source_names(partition, dataset)
    result = {
        "program_filename": names["program_filename"],
        "program_sha256": files.get(names["program_filename"], ""),
        "query_filename": names["query_filename"],
        "query_sha256": files.get(names["query_filename"], ""),
        "oracle_filename": names["oracle_filename"],
        "oracle_sha256": files.get(names["oracle_filename"], ""),
    }
    _exact_mapping(result, _SOURCE_KEYS, "sealed source binding")
    for key in ("program_sha256", "query_sha256", "oracle_sha256"):
        _require_hash(result[key], f"sealed {dataset} {key}")
    return result


def _validate_core_training(value: object, label: str) -> dict[str, object]:
    core = _exact_mapping(value, _CORE_TRAINING_KEYS, label)
    for key in ("core_sha256", "atomic_sha256", "closure_sha256"):
        _require_hash(core[key], f"{label} {key}")
    if core["training_schema"] != CORE_TRAINING_SCHEMA:
        raise RunContractError(f"CTAA {label} training schema differs")
    if core["training_arm"] not in ARMS:
        raise RunContractError(f"CTAA {label} arm differs")
    expected_kind = (
        "outer_product_control"
        if core["training_arm"] == "oprc_closure"
        else "closure_feature"
    )
    if core["core_kind"] != expected_kind:
        raise RunContractError(f"CTAA {label} core kind differs")
    if (
        type(core["training_seed"]) is not int
        or int(core["training_seed"]) < 0
        or type(core["updates"]) is not int
        or int(core["updates"]) < 1
        or type(core["batch_size"]) is not int
        or int(core["batch_size"]) < 1
        or type(core["learning_rate"]) not in {int, float}
        or isinstance(core["learning_rate"], bool)
        or not math.isfinite(float(core["learning_rate"]))
        or float(core["learning_rate"]) <= 0.0
    ):
        raise RunContractError(f"CTAA {label} training metadata differs")
    return core


def _nullable_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_hash(value, label)


def _validate_raw_receipt(value: object, label: str) -> dict[str, object]:
    receipt = _exact_mapping(value, _RAW_RECEIPT_KEYS, label)
    if (
        receipt["schema"] != RAW_EVIDENCE_RECEIPT_SCHEMA
        or receipt["oracle_access"] != 0
    ):
        raise RunContractError(f"CTAA {label} identity or oracle custody differs")
    counts = (
        receipt["rows"],
        receipt["valid_packets"],
        receipt["executed_rows"],
        receipt["queried_rows"],
        receipt["answered_rows"],
    )
    if any(type(item) is not int or int(item) < 0 for item in counts):
        raise RunContractError(f"CTAA {label} row accounting differs")
    rows, valid, executed, queried, answered = (int(item) for item in counts)
    if rows < 1 or not (0 <= answered <= queried <= executed <= valid <= rows):
        raise RunContractError(f"CTAA {label} row accounting differs")
    for key in (
        "program_predictions_sha256",
        "compiler_sha256",
        "program_source_sha256",
        "packet_index_sha256",
        "core_sha256",
        "evidence_sha256",
    ):
        _require_hash(receipt[key], f"{label} {key}")
    for key in (
        "query_source_sha256",
        "execution_sha256",
        "query_predictions_sha256",
        "query_positions_sha256",
        "answers_sha256",
    ):
        _nullable_hash(receipt[key], f"{label} {key}")
    core = _validate_core_training(receipt["core_training"], f"{label} core training")
    if (
        receipt["core_sha256"] != core["core_sha256"]
        or receipt["core_kind"] != core["core_kind"]
    ):
        raise RunContractError(f"CTAA {label} core binding differs")
    if (executed == 0) != (receipt["execution_sha256"] is None):
        raise RunContractError(f"CTAA {label} execution commitment differs")
    query_hashes_absent = (
        receipt["query_predictions_sha256"] is None
        and receipt["query_positions_sha256"] is None
    )
    query_hashes_present = (
        receipt["query_predictions_sha256"] is not None
        and receipt["query_positions_sha256"] is not None
    )
    if (queried == 0 and not query_hashes_absent) or (
        queried > 0 and not query_hashes_present
    ):
        raise RunContractError(f"CTAA {label} query commitment differs")
    if (answered == 0) != (receipt["answers_sha256"] is None):
        raise RunContractError(f"CTAA {label} answer commitment differs")
    if receipt["query_source_sha256"] is None and any((executed, queried, answered)):
        raise RunContractError(f"CTAA {label} disclosed-query commitment differs")
    return receipt


def _receipt_path(value: object, plan_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RunContractError(f"CTAA {label} path differs")
    path = Path(value)
    if not path.is_absolute():
        path = plan_root / path
    if path.name != "receipt.json":
        raise RunContractError(f"CTAA {label} is not a raw-evidence receipt")
    return path


def _run_id(seed: int, arm: str, dataset: str) -> str:
    return f"seed-{seed}:{arm}:{dataset}"


def _read_raw_receipt(path: Path, label: str) -> tuple[dict[str, object], str]:
    data = _read_committed_bytes(path, label, require_read_only_parent=True)
    value, digest = _decode_json(data, label), _sha256_bytes(data)
    return _validate_raw_receipt(value, label), digest


def _artifact_binding(receipt: Mapping[str, object]) -> dict[str, object]:
    result = {key: receipt[key] for key in sorted(_ARTIFACT_KEYS)}
    _exact_mapping(result, _ARTIFACT_KEYS, "evidence artifact binding")
    return result


def _validate_plan_run(value: object, plan_root: Path) -> dict[str, object]:
    run = _exact_mapping(value, _RUN_INPUT_KEYS, "run-plan entry")
    if run["schema"] != RUN_INPUT_SCHEMA:
        raise RunContractError("CTAA run-plan entry identity differs")
    if type(run["seed"]) is not int or int(run["seed"]) < 0:
        raise RunContractError("CTAA run-plan seed differs")
    if run["arm"] not in ARMS or run["dataset"] not in DATASETS:
        raise RunContractError("CTAA run-plan arm or dataset differs")
    core = _validate_core_training(run["core_training"], "run-plan core training")
    if core["training_seed"] != run["seed"] or core["training_arm"] != run["arm"]:
        raise RunContractError("CTAA run-plan core metadata differs")
    receipt_path = _receipt_path(
        run["evidence_receipt_path"], plan_root, "evidence receipt"
    )
    parent_value = run["parent_evidence_receipt_path"]
    if run["dataset"] == "base":
        if parent_value is not None:
            raise RunContractError("CTAA base run cannot name parent evidence")
        parent_path = None
    else:
        parent_path = _receipt_path(parent_value, plan_root, "parent evidence receipt")
    return {
        **run,
        "core_training": core,
        "receipt_path": receipt_path,
        "parent_path": parent_path,
    }


def _make_run_binding(
    *,
    run: Mapping[str, object],
    receipt: Mapping[str, object],
    receipt_sha: str,
    sources: Mapping[str, str],
    parent_run_id: str | None,
    parent_receipt_sha: str | None,
    parent_evidence_sha: str | None,
) -> dict[str, object]:
    return {
        "schema": RUN_BINDING_SCHEMA,
        "run_id": _run_id(int(run["seed"]), str(run["arm"]), str(run["dataset"])),
        "seed": run["seed"],
        "arm": run["arm"],
        "dataset": run["dataset"],
        "raw_evidence_receipt_sha256": receipt_sha,
        "compiler_sha256": receipt["compiler_sha256"],
        "sealed_sources": dict(sources),
        "evidence_artifacts": _artifact_binding(receipt),
        "core_training": dict(receipt["core_training"]),
        "parent_run_id": parent_run_id,
        "parent_evidence_receipt_sha256": parent_receipt_sha,
        "parent_evidence_sha256": parent_evidence_sha,
    }


def _build_payload(
    manifest_path: Path,
    run_plan_path: Path,
    bootstrap_seed_receipt_path: Path,
) -> dict[str, object]:
    _, manifest_sha, board_sha, files = _manifest_bindings(manifest_path)
    bootstrap_receipt, bootstrap_receipt_sha = _read_committed_json(
        bootstrap_seed_receipt_path, "bootstrap seed receipt"
    )
    try:
        bootstrap = validate_bootstrap_receipt(
            bootstrap_receipt,
            manifest_sha256=manifest_sha,
            gate_source_sha256=_sha256_bytes(
                Path(__file__)
                .with_name("evaluate_ctaa_advancement_gates.py")
                .read_bytes()
            ),
            statistics_source_sha256=_sha256_bytes(
                Path(__file__).with_name("ctaa_gate_statistics.py").read_bytes()
            ),
        )
    except ValueError as error:
        raise RunContractError(
            "CTAA bootstrap receipt binding was substituted"
        ) from error
    plan, plan_sha = _read_committed_json(run_plan_path, "run plan")
    _exact_mapping(plan, _RUN_PLAN_KEYS, "run plan")
    if plan["schema"] != RUN_PLAN_SCHEMA or plan["partition"] not in PARTITIONS:
        raise RunContractError("CTAA run-plan identity differs")
    if (
        plan["expected_manifest_sha256"] != manifest_sha
        or plan["expected_board_sha256"] != board_sha
    ):
        raise RunContractError("CTAA sealed manifest or board tree was substituted")
    if not isinstance(plan["runs"], list) or len(plan["runs"]) != RUN_COUNT:
        raise RunContractError(f"CTAA run plan must contain exactly {RUN_COUNT} runs")
    partition = str(plan["partition"])
    source_bindings = {
        dataset: _sealed_sources(partition, dataset, files) for dataset in DATASETS
    }
    parsed = [_validate_plan_run(value, run_plan_path.parent) for value in plan["runs"]]
    identities = [
        (int(run["seed"]), str(run["arm"]), str(run["dataset"])) for run in parsed
    ]
    if len(set(identities)) != RUN_COUNT:
        raise RunContractError("CTAA run identities repeat")
    seeds = sorted({seed for seed, _, _ in identities})
    expected = {
        (seed, arm, dataset) for seed in seeds for arm in ARMS for dataset in DATASETS
    }
    if len(seeds) != SEED_COUNT or set(identities) != expected:
        raise RunContractError("CTAA run plan is not five complete paired seeds")

    loaded: dict[
        tuple[int, str, str], tuple[dict[str, object], str, dict[str, object]]
    ] = {}
    for run in parsed:
        identity = (int(run["seed"]), str(run["arm"]), str(run["dataset"]))
        receipt, receipt_sha = _read_raw_receipt(
            Path(run["receipt_path"]), f"raw-evidence receipt for {_run_id(*identity)}"
        )
        if receipt["core_training"] != run["core_training"]:
            raise RunContractError(
                "CTAA raw receipt/core-training metadata was swapped"
            )
        sources = source_bindings[identity[2]]
        query_bound = receipt["query_source_sha256"] in {None, sources["query_sha256"]}
        if (
            receipt["program_source_sha256"] != sources["program_sha256"]
            or not query_bound
        ):
            raise RunContractError("CTAA raw evidence is not bound to sealed sources")
        loaded[identity] = (receipt, receipt_sha, run)

    receipt_hashes = [receipt_sha for _, receipt_sha, _ in loaded.values()]
    evidence_hashes = [
        str(receipt["evidence_sha256"]) for receipt, _, _ in loaded.values()
    ]
    if len(set(receipt_hashes)) != RUN_COUNT or len(set(evidence_hashes)) != RUN_COUNT:
        raise RunContractError("CTAA raw-evidence identities repeat")

    atomic_hashes = set()
    closure_hashes = set()
    budget_shapes = set()
    compiler_by_seed: dict[int, str] = {}
    compiled_identities = set()
    core_identities = set()
    bindings: list[dict[str, object]] = []
    for seed in seeds:
        seed_compilers = {
            str(loaded[(seed, arm, dataset)][0]["compiler_sha256"])
            for arm in ARMS
            for dataset in DATASETS
        }
        if len(seed_compilers) != 1:
            raise RunContractError(
                "CTAA paired seed does not share one frozen compiler"
            )
        compiler_by_seed[seed] = next(iter(seed_compilers))
        for dataset in DATASETS:
            compiled = {
                (
                    loaded[(seed, arm, dataset)][0]["program_predictions_sha256"],
                    loaded[(seed, arm, dataset)][0]["packet_index_sha256"],
                )
                for arm in ARMS
            }
            if len(compiled) != 1:
                raise RunContractError(
                    "CTAA arms do not share compiled program packets"
                )
            compiled_identities.add(next(iter(compiled)))
        for arm in ARMS:
            base_receipt, base_sha, _ = loaded[(seed, arm, "base")]
            child_receipt, child_sha, child_run = loaded[(seed, arm, "intervention")]
            if (
                base_receipt["core_training"] != child_receipt["core_training"]
                or base_receipt["core_sha256"] != child_receipt["core_sha256"]
            ):
                raise RunContractError("CTAA base/intervention core binding differs")
            parent, parent_sha = _read_raw_receipt(
                Path(child_run["parent_path"]),
                f"parent evidence receipt for {_run_id(seed, arm, 'intervention')}",
            )
            if parent_sha != base_sha or parent != base_receipt:
                raise RunContractError(
                    "CTAA intervention parent evidence differs from paired base"
                )
            core = dict(base_receipt["core_training"])
            core_identities.add(str(core["core_sha256"]))
            atomic_hashes.add(str(core["atomic_sha256"]))
            closure_hashes.add(str(core["closure_sha256"]))
            budget_shapes.add(
                (core["updates"], core["batch_size"], core["learning_rate"])
            )
            bindings.append(
                _make_run_binding(
                    run=loaded[(seed, arm, "base")][2],
                    receipt=base_receipt,
                    receipt_sha=base_sha,
                    sources=source_bindings["base"],
                    parent_run_id=None,
                    parent_receipt_sha=None,
                    parent_evidence_sha=None,
                )
            )
            bindings.append(
                _make_run_binding(
                    run=child_run,
                    receipt=child_receipt,
                    receipt_sha=child_sha,
                    sources=source_bindings["intervention"],
                    parent_run_id=_run_id(seed, arm, "base"),
                    parent_receipt_sha=base_sha,
                    parent_evidence_sha=str(base_receipt["evidence_sha256"]),
                )
            )
    if len(set(compiler_by_seed.values())) != SEED_COUNT:
        raise RunContractError("CTAA independently trained compiler identities repeat")
    if len(compiled_identities) != SEED_COUNT * len(DATASETS):
        raise RunContractError(
            "CTAA compiled packet identities repeat across seed/dataset"
        )
    if len(core_identities) != SEED_COUNT * len(ARMS):
        raise RunContractError("CTAA core identities repeat across seed/arm")
    if len(atomic_hashes) != 1 or len(closure_hashes) != 1 or len(budget_shapes) != 1:
        raise RunContractError("CTAA matched core-training inputs or budgets differ")

    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    bindings.sort(
        key=lambda item: (
            int(item["seed"]),
            arm_order[str(item["arm"])],
            dataset_order[str(item["dataset"])],
        )
    )
    oracle_files = {
        dataset: {
            "filename": source_bindings[dataset]["oracle_filename"],
            "sha256": source_bindings[dataset]["oracle_sha256"],
        }
        for dataset in DATASETS
    }
    payload: dict[str, object] = {
        "schema": RUN_CONTRACT_SCHEMA,
        "partition": partition,
        "manifest_sha256": manifest_sha,
        "board_sha256": board_sha,
        "run_plan_sha256": plan_sha,
        "bootstrap_seed_receipt_sha256": bootstrap_receipt_sha,
        "bootstrap_seed": bootstrap["bootstrap_seed"],
        "training_seeds": seeds,
        "arms": list(ARMS),
        "datasets": list(DATASETS),
        "run_count": RUN_COUNT,
        "oracle_files": oracle_files,
        "runs": bindings,
    }
    return {**payload, "run_contract_sha256": canonical_sha256(payload)}


def _validate_contract_shape(value: object) -> dict[str, object]:
    contract = _exact_mapping(value, _CONTRACT_KEYS, "run contract")
    if (
        contract["schema"] != RUN_CONTRACT_SCHEMA
        or contract["partition"] not in PARTITIONS
        or contract["arms"] != list(ARMS)
        or contract["datasets"] != list(DATASETS)
        or contract["run_count"] != RUN_COUNT
        or not isinstance(contract["training_seeds"], list)
        or len(contract["training_seeds"]) != SEED_COUNT
        or not isinstance(contract["runs"], list)
        or len(contract["runs"]) != RUN_COUNT
    ):
        raise RunContractError("CTAA run-contract identity differs")
    for key in (
        "manifest_sha256",
        "board_sha256",
        "run_plan_sha256",
        "bootstrap_seed_receipt_sha256",
        "run_contract_sha256",
    ):
        _require_hash(contract[key], f"run contract {key}")
    if (
        type(contract["bootstrap_seed"]) is not int
        or int(contract["bootstrap_seed"]) < 0
    ):
        raise RunContractError("CTAA run-contract bootstrap seed differs")
    if not isinstance(contract["oracle_files"], dict) or set(
        contract["oracle_files"]
    ) != set(DATASETS):
        raise RunContractError("CTAA run-contract oracle identity set differs")
    for value in contract["oracle_files"].values():
        row = _exact_mapping(
            value, frozenset({"filename", "sha256"}), "oracle identity"
        )
        _safe_manifest_name(row["filename"])
        _require_hash(row["sha256"], "oracle identity")
    seen = set()
    for value in contract["runs"]:
        row = _exact_mapping(value, _RUN_BINDING_KEYS, "run binding")
        if row["schema"] != RUN_BINDING_SCHEMA:
            raise RunContractError("CTAA run-binding identity differs")
        identity = (row["seed"], row["arm"], row["dataset"])
        if (
            type(row["seed"]) is not int
            or row["arm"] not in ARMS
            or row["dataset"] not in DATASETS
            or row["run_id"]
            != _run_id(int(row["seed"]), str(row["arm"]), str(row["dataset"]))
            or identity in seen
        ):
            raise RunContractError("CTAA run-binding identity differs")
        seen.add(identity)
        _require_hash(row["raw_evidence_receipt_sha256"], "run evidence receipt")
        _require_hash(row["compiler_sha256"], "run compiler")
        sources = _exact_mapping(
            row["sealed_sources"], _SOURCE_KEYS, "run sealed sources"
        )
        for key in ("program_sha256", "query_sha256", "oracle_sha256"):
            _require_hash(sources[key], f"run sealed source {key}")
        artifacts = _exact_mapping(
            row["evidence_artifacts"], _ARTIFACT_KEYS, "run evidence artifacts"
        )
        for key, item in artifacts.items():
            _nullable_hash(item, f"run evidence artifact {key}")
        _validate_core_training(row["core_training"], "run-binding core training")
        if row["dataset"] == "base":
            if any(
                row[key] is not None
                for key in (
                    "parent_run_id",
                    "parent_evidence_receipt_sha256",
                    "parent_evidence_sha256",
                )
            ):
                raise RunContractError("CTAA base run has parent evidence")
        else:
            if row["parent_run_id"] != _run_id(
                int(row["seed"]), str(row["arm"]), "base"
            ):
                raise RunContractError("CTAA intervention parent identity differs")
            _require_hash(
                row["parent_evidence_receipt_sha256"], "parent evidence receipt"
            )
            _require_hash(row["parent_evidence_sha256"], "parent evidence")
    unhashed = {
        key: value for key, value in contract.items() if key != "run_contract_sha256"
    }
    if contract["run_contract_sha256"] != canonical_sha256(unhashed):
        raise RunContractError("CTAA run-contract canonical commitment differs")
    return contract


def _write_read_only_once(path: Path, value: Mapping[str, object]) -> None:
    """Publish canonical JSON through an O_EXCL link, then retain mode 0444."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing CTAA run contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        data = (canonical_json(dict(value)) + "\n").encode("ascii")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()


def _read_published_contract(path: Path) -> dict[str, object]:
    data = _read_committed_bytes(path, "published run contract")
    contract = _decode_json(data, "published run contract")
    if data != (canonical_json(contract) + "\n").encode("ascii"):
        raise RunContractError("CTAA published run contract is not canonical JSON")
    return contract


def create_run_contract(
    *,
    manifest_path: Path,
    run_plan_path: Path,
    bootstrap_seed_receipt_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Validate all pre-access inputs and publish one immutable contract."""

    payload = _validate_contract_shape(
        _build_payload(manifest_path, run_plan_path, bootstrap_seed_receipt_path)
    )
    _write_read_only_once(output_path, payload)
    written = _read_published_contract(output_path)
    if written != payload:
        raise RunContractError("CTAA published run contract differs")
    return payload


def validate_run_contract(
    *,
    contract_path: Path,
    manifest_path: Path,
    run_plan_path: Path,
    bootstrap_seed_receipt_path: Path,
) -> dict[str, object]:
    """Recompute every source binding and compare it with a published contract."""

    contract = _read_published_contract(contract_path)
    validated = _validate_contract_shape(contract)
    expected = _build_payload(manifest_path, run_plan_path, bootstrap_seed_receipt_path)
    if validated != expected:
        raise RunContractError("CTAA run contract differs from pre-access inputs")
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--bootstrap-seed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_run_contract(
        manifest_path=args.manifest,
        run_plan_path=args.run_plan,
        bootstrap_seed_receipt_path=args.bootstrap_seed_receipt,
        output_path=args.output,
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
