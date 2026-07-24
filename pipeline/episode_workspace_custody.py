"""Custody primitives for the source-deleted EPISODE workspace experiment.

This module materializes physically disjoint optimization, world-compilation,
query-execution, and assessment inputs. It contains no neural training code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from pipeline.episode_action_binding_board import (
    ANSWER,
    EOS,
    ModelPacket,
    split_world_and_query,
    visible_table_oracle,
    world_commitment,
)
from pipeline.generate_episode_action_binding_corpus import verify_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CORPUS = (
    REPOSITORY_ROOT
    / "artifacts/r12/episode_action_binding_corpus_v1_1532fe2_seed2026072309"
)
DEFAULT_CUSTODY_BUNDLE = (
    REPOSITORY_ROOT
    / "artifacts/r12/episode_workspace_custody_v1_1532fe2_seed2026072309"
)
SOURCE_FILE_SHA256 = {
    "bundle_manifest.json": (
        "d2e6ad2f52bf8ec3355be2f4b29762ca4d106b711446c81d1099864834f84cac"
    ),
    "manifest.json": (
        "9eb3289d9d64b09d4886c71f1a9d8dd7167479f5c3d042a6ebb9bd6de4de53ed"
    ),
    "model_packets.jsonl": (
        "3f726a4bd15aeccbcda8999459434fdf01153495d7196a9f307184bb5a570445"
    ),
    "target_labels.jsonl": (
        "a550870c61fae98dbab36a781e25df0d752991f2a20e3b26301fdef15d30c045"
    ),
    "offline_ledger.jsonl": (
        "c5c73108eaa23d2524e54c8fe7fa0233536c0df45c35e9c91c5b362386a2bf42"
    ),
}
SOURCE_LOGICAL_SHA256 = {
    "model_payload_sha256": (
        "f975291b22560e07cfa5e636133cb62c5688cfb8b39b8786812ea40610807323"
    ),
    "target_labels_sha256": (
        "aa0afa01882e24f9c3d708c8cdd2d9f7bb5978a9dae79c2f4c90c5c355972283"
    ),
    "offline_ledger_sha256": (
        "3ae85435030c80ac58a4ed16cab5d50e28250826994377a588aa1690b7e2ecec"
    ),
}
CUSTODY_SCHEMA = "episode_workspace_custody_v1"
LANDLOCK_RECEIPT_SCHEMA = "shohin_landlock_stage_receipt_v1"
DENIED_PROBE_RECEIPT_SCHEMA = "shohin_landlock_denied_probe_receipt_v1"
TRAIN_GROUP_SCHEMA = "episode_workspace_train_group_v1"
DEVELOPMENT_WORLD_SCHEMA = "episode_workspace_development_world_v1"
DEVELOPMENT_QUERY_SCHEMA = "episode_workspace_development_query_v1"
ASSESSOR_ROW_SCHEMA = "episode_workspace_assessor_row_v1"
CASES_PER_CLUSTER = 6
WORLD_TOKENS = 145
TRAIN_PARTITION = "train"
DEVELOPMENT_PARTITION = "development"


class EpisodeCustodyError(ValueError):
    """A corpus, split, hash, or publication invariant failed."""


@dataclass(frozen=True)
class SourceExample:
    packet_sha256: str
    partition: str
    world_tokens: tuple[int, ...]
    query_tokens: tuple[int, ...]
    target_token: int
    cluster_id: str
    cluster_index: int
    query_variant: str
    binding_shift: int
    query_depth: int
    state_tokens: tuple[int, ...]
    world_commitment: str


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def verify_landlock_stage(stage: str, deny_probe: Path) -> dict[str, object]:
    """Prove this process is confined and cannot read one real forbidden input."""

    if os.environ.get("SHOHIN_LANDLOCK_ENFORCED") != "1":
        raise EpisodeCustodyError("stage is not running under enforced Landlock")
    if os.environ.get("SHOHIN_LANDLOCK_STAGE") != stage:
        raise EpisodeCustodyError("Landlock stage identity differs")
    try:
        abi = int(os.environ["SHOHIN_LANDLOCK_ABI"])
    except (KeyError, ValueError) as exc:
        raise EpisodeCustodyError("Landlock ABI receipt is invalid") from exc
    if abi <= 0:
        raise EpisodeCustodyError("Landlock ABI must be positive")
    if _process_dumpable() != 0:
        raise EpisodeCustodyError("confined stage is dumpable")
    policy_sha256 = os.environ.get("SHOHIN_LANDLOCK_POLICY_SHA256", "")
    if len(policy_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in policy_sha256
    ):
        raise EpisodeCustodyError("Landlock policy SHA-256 is invalid")
    policy_path_text = os.environ.get("SHOHIN_LANDLOCK_POLICY_PATH")
    if not policy_path_text:
        raise EpisodeCustodyError("Landlock policy receipt path is missing")
    canonical_policy = read_json_verified(
        Path(policy_path_text),
        policy_sha256,
    )
    if (
        not isinstance(canonical_policy, dict)
        or canonical_policy.get("schema") != "shohin_landlock_stage_policy_v1"
        or canonical_policy.get("stage") != stage
        or canonical_policy.get("landlock_abi") != abi
        or not isinstance(canonical_policy.get("rules"), list)
    ):
        raise EpisodeCustodyError("Landlock canonical policy differs")
    try:
        with deny_probe.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise EpisodeCustodyError(
                "forbidden-input probe failed for a reason other than access denial"
            ) from exc
        denied_errno = exc.errno
    else:
        raise EpisodeCustodyError("Landlock allowed a forbidden-input probe")
    process_id = os.getpid()
    return {
        "schema": LANDLOCK_RECEIPT_SCHEMA,
        "stage": stage,
        "enforced": True,
        "dumpable": False,
        "abi": abi,
        "policy_sha256": policy_sha256,
        "canonical_policy": canonical_policy,
        "process_id": process_id,
        "denied_probe_receipt": {
            "schema": DENIED_PROBE_RECEIPT_SCHEMA,
            "stage": stage,
            "process_id": process_id,
            "operation": "open_read",
            "path": str(deny_probe.absolute()),
            "path_name": deny_probe.name,
            "path_sha256": hashlib.sha256(
                os.fsencode(deny_probe.absolute())
            ).hexdigest(),
            "denied": True,
            "errno": denied_errno,
        },
    }


def _process_dumpable() -> int:
    if not sys.platform.startswith("linux"):
        raise EpisodeCustodyError("Landlock stage verification requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "prctl"):
        raise EpisodeCustodyError("prctl is unavailable")
    libc.prctl.restype = ctypes.c_int
    result = libc.prctl(3, 0, 0, 0, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise EpisodeCustodyError(
            f"PR_GET_DUMPABLE failed: {os.strerror(error)}"
        )
    return int(result)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_bytes(path: Path, expected_sha256: str) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise EpisodeCustodyError(
            f"{path.name} hash mismatch: {actual}, expected {expected_sha256}"
        )
    return raw


def _parse_jsonl_bytes(path: Path, raw: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EpisodeCustodyError(
                f"{path.name}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise EpisodeCustodyError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as handle:
        raw = handle.read()
    return _parse_jsonl_bytes(path, raw)


def read_jsonl_verified(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    return _parse_jsonl_bytes(path, _read_verified_bytes(path, expected_sha256))


def read_json_verified(path: Path, expected_sha256: str) -> object:
    raw = _read_verified_bytes(path, expected_sha256)
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeCustodyError(f"{path.name} is invalid JSON") from exc


def write_jsonl_fsync(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_fsync(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory_noreplace(staging: Path, output: Path) -> None:
    """Atomically publish a regular directory with a kernel no-replace flag."""

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    staging.chmod(0o555)
    fsync_directory(staging)
    fsync_directory(staging.parent)
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staging)
    destination = os.fsencode(output)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source, destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source, -100, destination, 0x00000001)
    else:
        raise OSError(
            errno.ENOTSUP,
            "kernel no-replace directory publication is unavailable",
        )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to replace {output}")
        raise OSError(error, os.strerror(error), str(output))
    output.chmod(0o555)
    fsync_directory(output.parent)


def atomic_bundle_directory(output: Path) -> tuple[Path, Path]:
    """Create a hidden immutable payload path and its exclusive lock."""

    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace {output}")
    lock = output.with_name(f".{output.name}.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.payload.",
            dir=output.parent,
        )
    )
    return staging, lock


def abort_atomic_bundle(staging: Path, lock: Path) -> None:
    if staging.exists():
        staging.chmod(0o755)
        for path in staging.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
    shutil.rmtree(staging, ignore_errors=True)
    lock.unlink(missing_ok=True)


def finish_atomic_bundle(staging: Path, output: Path, lock: Path) -> None:
    try:
        publish_directory_noreplace(staging, output)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    lock.unlink(missing_ok=True)
    fsync_directory(output.absolute().parent)


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise EpisodeCustodyError(f"{key} must be a nonempty string")
    return value


def _required_integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EpisodeCustodyError(f"{key} must be an integer")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise EpisodeCustodyError(f"{label} must be an integer list")
    return tuple(value)


def _unique_digest_map(
    rows: Sequence[dict[str, object]],
    *,
    expected_keys: set[str],
    label: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if set(row) != expected_keys:
            raise EpisodeCustodyError(f"{label} row has unexpected fields")
        digest = _required_string(row, "packet_sha256")
        if digest in result:
            raise EpisodeCustodyError(f"{label} packet digest is duplicated")
        result[digest] = row
    return result


def _cluster_groups(
    examples: Sequence[SourceExample],
) -> dict[str, tuple[SourceExample, ...]]:
    pending: dict[str, list[SourceExample]] = defaultdict(list)
    for example in examples:
        pending[example.cluster_id].append(example)
    return {
        cluster_id: tuple(
            sorted(
                values,
                key=lambda item: (item.query_variant, item.binding_shift),
            )
        )
        for cluster_id, values in sorted(pending.items())
    }


def _load_source_corpus(
    path: Path,
) -> tuple[
    tuple[SourceExample, ...],
    tuple[SourceExample, ...],
    dict[str, object],
]:
    path = path.resolve()
    if not path.is_dir():
        raise EpisodeCustodyError("source corpus directory is missing")
    verified_source_bytes = {
        name: _read_verified_bytes(path / name, expected)
        for name, expected in SOURCE_FILE_SHA256.items()
    }
    verified_bundle = verify_bundle(path)
    manifest = json.loads(verified_source_bytes["manifest.json"])
    if not isinstance(manifest, dict):
        raise EpisodeCustodyError("source manifest is not an object")
    if any(manifest.get(key) != value for key, value in SOURCE_LOGICAL_SHA256.items()):
        raise EpisodeCustodyError("source logical payload hash drifted")

    packet_rows = _parse_jsonl_bytes(
        path / "model_packets.jsonl",
        verified_source_bytes["model_packets.jsonl"],
    )
    target_rows = _parse_jsonl_bytes(
        path / "target_labels.jsonl",
        verified_source_bytes["target_labels.jsonl"],
    )
    offline_rows = _parse_jsonl_bytes(
        path / "offline_ledger.jsonl",
        verified_source_bytes["offline_ledger.jsonl"],
    )
    expected_total = int(manifest["total_packets"])
    if not (
        len(packet_rows) == len(target_rows) == len(offline_rows) == expected_total
    ):
        raise EpisodeCustodyError("source ledgers have different cardinality")
    targets = _unique_digest_map(
        target_rows,
        expected_keys={"packet_sha256", "target_token"},
        label="target",
    )
    offline = _unique_digest_map(
        offline_rows,
        expected_keys={
            "packet_sha256",
            "partition",
            "cluster_id",
            "cluster_index",
            "query_variant",
            "binding_shift",
            "target_token",
            "query_start_state",
            "query_action_indices",
            "physical_operators",
            "state_tokens",
            "action_tokens",
            "world_commitment",
        },
        label="offline",
    )

    examples: list[SourceExample] = []
    seen: set[str] = set()
    for packet_row in packet_rows:
        if set(packet_row) != {
            "packet_sha256",
            "partition",
            "tokens",
            "attention_mask",
        }:
            raise EpisodeCustodyError("model packet has unexpected fields")
        digest = _required_string(packet_row, "packet_sha256")
        if digest in seen:
            raise EpisodeCustodyError("model packet digest is duplicated")
        seen.add(digest)
        target_row = targets.get(digest)
        metadata = offline.get(digest)
        if target_row is None or metadata is None:
            raise EpisodeCustodyError("source digest join is incomplete")
        partition = _required_string(packet_row, "partition")
        if partition not in {TRAIN_PARTITION, DEVELOPMENT_PARTITION}:
            raise EpisodeCustodyError("unknown partition")
        if metadata.get("partition") != partition:
            raise EpisodeCustodyError("metadata partition disagrees")
        tokens = _integer_tuple(packet_row.get("tokens"), "tokens")
        attention_mask = _integer_tuple(
            packet_row.get("attention_mask"),
            "attention_mask",
        )
        if attention_mask != (1,) * len(tokens):
            raise EpisodeCustodyError("source packet mask is not fully active")
        packet = ModelPacket(tokens=tokens, attention_mask=attention_mask)
        world, query = split_world_and_query(packet)
        if len(world) != WORLD_TOKENS:
            raise EpisodeCustodyError("world is not exactly 145 tokens")
        if query[-2:] != (ANSWER, EOS) or query.count(ANSWER) != 1:
            raise EpisodeCustodyError("query ANSWER/EOS grammar drifted")
        target = _required_integer(target_row, "target_token")
        if _required_integer(metadata, "target_token") != target:
            raise EpisodeCustodyError("target ledgers disagree")
        if visible_table_oracle(packet) != target:
            raise EpisodeCustodyError("independent visible oracle disagrees")
        commitment = world_commitment(packet)
        if metadata.get("world_commitment") != commitment:
            raise EpisodeCustodyError("world commitment disagrees")
        state_tokens = _integer_tuple(metadata.get("state_tokens"), "state_tokens")
        if len(state_tokens) != 8 or len(set(state_tokens)) != 8:
            raise EpisodeCustodyError("state token domain is malformed")
        if target not in state_tokens:
            raise EpisodeCustodyError("target is outside state token domain")
        depth = len(
            _integer_tuple(
                metadata.get("query_action_indices"),
                "query_action_indices",
            )
        )
        if len(query) != 2 * depth + 3:
            raise EpisodeCustodyError("query length does not encode its depth")
        if partition == TRAIN_PARTITION and depth not in {2, 3, 4}:
            raise EpisodeCustodyError("train depth is outside 2-4")
        if partition == DEVELOPMENT_PARTITION and depth not in {5, 6}:
            raise EpisodeCustodyError("development depth is outside 5-6")
        examples.append(
            SourceExample(
                packet_sha256=digest,
                partition=partition,
                world_tokens=world,
                query_tokens=query,
                target_token=target,
                cluster_id=_required_string(metadata, "cluster_id"),
                cluster_index=_required_integer(metadata, "cluster_index"),
                query_variant=_required_string(metadata, "query_variant"),
                binding_shift=_required_integer(metadata, "binding_shift"),
                query_depth=depth,
                state_tokens=state_tokens,
                world_commitment=commitment,
            )
        )
    if seen != set(targets) or seen != set(offline):
        raise EpisodeCustodyError("source ledgers cover different packets")

    train = tuple(item for item in examples if item.partition == TRAIN_PARTITION)
    development = tuple(
        item for item in examples if item.partition == DEVELOPMENT_PARTITION
    )
    _validate_partition(
        train,
        expected_clusters=int(manifest["train_clusters"]),
    )
    _validate_partition(
        development,
        expected_clusters=int(manifest["development_clusters"]),
    )
    if {item.packet_sha256 for item in train} & {
        item.packet_sha256 for item in development
    }:
        raise EpisodeCustodyError("train/development packet overlap")
    source_receipt = {
        "source_path": str(path.relative_to(REPOSITORY_ROOT)),
        "source_file_sha256": dict(SOURCE_FILE_SHA256),
        "source_logical_sha256": dict(SOURCE_LOGICAL_SHA256),
        "verified_bundle": verified_bundle,
        "source_manifest": manifest,
        "train_packets": len(train),
        "development_packets": len(development),
        "train_clusters": len(_cluster_groups(train)),
        "development_clusters": len(_cluster_groups(development)),
        "packet_overlap": 0,
        "reported_operator_family_overlap": int(
            manifest["exact_operator_family_overlap"]
        ),
    }
    return train, development, source_receipt


def _validate_partition(
    examples: Sequence[SourceExample],
    *,
    expected_clusters: int,
) -> None:
    groups = _cluster_groups(examples)
    if len(groups) != expected_clusters:
        raise EpisodeCustodyError("partition cluster count drifted")
    for cluster_id, values in groups.items():
        if len(values) != CASES_PER_CLUSTER:
            raise EpisodeCustodyError(
                f"cluster {cluster_id} is not a complete six-case group"
            )
        if len({len(item.query_tokens) for item in values}) != 1:
            raise EpisodeCustodyError(f"cluster {cluster_id} mixes query lengths")
        expected_cells = {
            (variant, shift)
            for variant in ("primary", "reordered")
            for shift in range(3)
        }
        cells = {(item.query_variant, item.binding_shift) for item in values}
        if cells != expected_cells:
            raise EpisodeCustodyError(
                f"cluster {cluster_id} lacks a six-case factorial surface"
            )
        commitments: dict[int, set[str]] = defaultdict(set)
        for item in values:
            commitments[item.binding_shift].add(item.world_commitment)
        if any(len(values) != 1 for values in commitments.values()):
            raise EpisodeCustodyError(
                f"cluster {cluster_id} violates late-query world identity"
            )


def _opaque_world_id(commitment: str) -> str:
    return hashlib.sha256(
        f"episode-workspace-world-v1:{commitment}".encode("ascii")
    ).hexdigest()


def _train_group_rows(
    examples: Sequence[SourceExample],
    *,
    shuffled: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for values in _cluster_groups(examples).values():
        by_cell = {(item.query_variant, item.binding_shift): item for item in values}
        payload: list[dict[str, object]] = []
        for item in values:
            target = item.target_token
            if shuffled:
                donor = by_cell[(item.query_variant, (item.binding_shift + 1) % 3)]
                target = donor.target_token
                if target == item.target_token or target not in item.state_tokens:
                    raise EpisodeCustodyError(
                        "shuffled train target is not an in-domain derangement"
                    )
            payload.append(
                {
                    "packet_sha256": item.packet_sha256,
                    "world_tokens": list(item.world_tokens),
                    "query_tokens": list(item.query_tokens),
                    "target_token": target,
                }
            )
        rows.append(
            {
                "schema": TRAIN_GROUP_SCHEMA,
                "examples": payload,
            }
        )
    return rows


def _development_rows(
    examples: Sequence[SourceExample],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    worlds: dict[str, tuple[int, ...]] = {}
    queries: list[dict[str, object]] = []
    assessor: list[dict[str, object]] = []
    for item in examples:
        world_id = _opaque_world_id(item.world_commitment)
        previous = worlds.setdefault(world_id, item.world_tokens)
        if previous != item.world_tokens:
            raise EpisodeCustodyError("world ID aliases different token sources")
        queries.append(
            {
                "schema": DEVELOPMENT_QUERY_SCHEMA,
                "packet_sha256": item.packet_sha256,
                "world_id": world_id,
                "query_tokens": list(item.query_tokens),
            }
        )
        assessor.append(
            {
                "schema": ASSESSOR_ROW_SCHEMA,
                "packet_sha256": item.packet_sha256,
                "target_token": item.target_token,
                "state_tokens": list(item.state_tokens),
                "cluster_id": item.cluster_id,
                "cluster_index": item.cluster_index,
                "query_variant": item.query_variant,
                "binding_shift": item.binding_shift,
                "query_depth": item.query_depth,
                "world_id": world_id,
            }
        )
    world_rows = [
        {
            "schema": DEVELOPMENT_WORLD_SCHEMA,
            "world_id": world_id,
            "world_tokens": list(tokens),
        }
        for world_id, tokens in sorted(worlds.items())
    ]
    queries.sort(key=lambda row: str(row["packet_sha256"]))
    assessor.sort(key=lambda row: str(row["packet_sha256"]))
    return world_rows, queries, assessor


def materialize_custody_bundle(
    source: Path,
    output: Path,
    *,
    generator_source_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Materialize disjoint stage inputs from the already frozen source board."""

    train, development, source_receipt = _load_source_corpus(source)
    true_groups = _train_group_rows(train, shuffled=False)
    shuffled_groups = _train_group_rows(train, shuffled=True)
    worlds, queries, assessor = _development_rows(development)
    if len(worlds) != 192:
        raise EpisodeCustodyError("expected exactly 192 unique development worlds")
    if len(queries) != 384 or len(assessor) != 384:
        raise EpisodeCustodyError("development packet cardinality drifted")

    staging, lock = atomic_bundle_directory(output)
    try:
        files: dict[str, str] = {}
        rows_by_name = {
            "train_true_groups.jsonl": true_groups,
            "train_shuffled_groups.jsonl": shuffled_groups,
            "development_worlds.jsonl": worlds,
            "development_queries.jsonl": queries,
            "development_assessor.jsonl": assessor,
        }
        for name, rows in rows_by_name.items():
            path = staging / name
            write_jsonl_fsync(path, rows)
            files[name] = file_sha256(path)
        manifest = {
            "schema": CUSTODY_SCHEMA,
            "claim_scope": (
                "physical input custody only; no neural fit, evaluation, "
                "reasoning, or continuation-pretraining claim"
            ),
            "generator_source": dict(generator_source_receipt),
            "source": source_receipt,
            "files": files,
            "logical_sha256": {
                name: json_sha256(rows) for name, rows in rows_by_name.items()
            },
            "counts": {
                "train_true_groups": len(true_groups),
                "train_shuffled_groups": len(shuffled_groups),
                "train_packets_per_arm": len(train),
                "development_worlds": len(worlds),
                "development_queries": len(queries),
                "development_assessor_rows": len(assessor),
            },
            "optimizer_visible_files": [
                "train_true_groups.jsonl",
                "train_shuffled_groups.jsonl",
            ],
            "compiler_visible_files": ["development_worlds.jsonl"],
            "executor_visible_files": ["development_queries.jsonl"],
            "assessor_visible_files": ["development_assessor.jsonl"],
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        }
        manifest_path = staging / "custody_manifest.json"
        write_json_fsync(manifest_path, manifest)
        fsync_directory(staging)
        finish_atomic_bundle(staging, output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    return {
        **manifest,
        "custody_manifest_sha256": file_sha256(output / "custody_manifest.json"),
    }


def load_custody_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = path / "custody_manifest.json"
    value = read_json_verified(manifest_path, expected_sha256)
    if not isinstance(value, dict) or value.get("schema") != CUSTODY_SCHEMA:
        raise EpisodeCustodyError("custody manifest schema is invalid")
    if value.get("pretraining_started") is not False:
        raise EpisodeCustodyError("custody manifest pretraining flag is invalid")
    return value


def committed_source_receipt(
    primary: Path,
    expected_primary_sha256: str,
    dependencies: Sequence[Path],
) -> dict[str, object]:
    """Bind a stage to clean, committed local sources and their byte hashes."""

    primary = primary.resolve()
    paths = tuple(dict.fromkeys((primary, *(path.resolve() for path in dependencies))))
    manifest = {
        str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in paths
    }
    primary_relative = str(primary.relative_to(REPOSITORY_ROOT))
    if manifest[primary_relative] != expected_primary_sha256:
        raise EpisodeCustodyError(
            "primary stage source hash differs from its frozen invocation"
        )
    relative_paths = [str(path.relative_to(REPOSITORY_ROOT)) for path in paths]
    if os.environ.get("SHOHIN_LANDLOCK_ENFORCED") == "1":
        return _sealed_committed_source_receipt(
            manifest=manifest,
            primary_relative=primary_relative,
            expected_primary_sha256=expected_primary_sha256,
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", *relative_paths],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *relative_paths],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EpisodeCustodyError(
            "stage sources are not clean files in the current commit"
        ) from exc
    return {
        "primary_path": primary_relative,
        "primary_sha256": expected_primary_sha256,
        "repository_commit": commit,
        "local_source_manifest": manifest,
    }


def _sealed_committed_source_receipt(
    *,
    manifest: Mapping[str, str],
    primary_relative: str,
    expected_primary_sha256: str,
) -> dict[str, object]:
    receipt_path_text = os.environ.get("SHOHIN_SOURCE_RECEIPT_PATH")
    expected_receipt_sha256 = os.environ.get("SHOHIN_SOURCE_RECEIPT_SHA256")
    if not receipt_path_text or not expected_receipt_sha256:
        raise EpisodeCustodyError("sealed source receipt environment is missing")
    value = read_json_verified(
        Path(receipt_path_text),
        expected_receipt_sha256,
    )
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "repository_commit",
        "source_manifest",
        "pretraining_started",
        "continuation_pretraining_authorized",
    }:
        raise EpisodeCustodyError("sealed source receipt fields differ")
    if (
        value["schema"] != "shohin_episode_workspace_source_bundle_v1"
        or value["pretraining_started"] is not False
        or value["continuation_pretraining_authorized"] is not False
    ):
        raise EpisodeCustodyError("sealed source receipt flags are invalid")
    commit = value["repository_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise EpisodeCustodyError("sealed source commit is invalid")
    sealed_manifest = value["source_manifest"]
    if not isinstance(sealed_manifest, dict):
        raise EpisodeCustodyError("sealed source manifest is invalid")
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for path, digest in sealed_manifest.items()
    ):
        raise EpisodeCustodyError("sealed source manifest entry is invalid")
    for path, actual in manifest.items():
        if sealed_manifest.get(path) != actual:
            raise EpisodeCustodyError(
                f"sealed source receipt differs for {path}"
            )
    if manifest.get(primary_relative) != expected_primary_sha256:
        raise EpisodeCustodyError("sealed primary source hash differs")
    return {
        "primary_path": primary_relative,
        "primary_sha256": expected_primary_sha256,
        "repository_commit": commit,
        "local_source_manifest": dict(manifest),
    }
