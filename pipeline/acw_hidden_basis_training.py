"""Public-only trainer utilities for the ACW hidden-basis falsifier.

This module must not import the hidden-basis generator.  Canonical scored runs
consume a trainer bundle containing only the required ``public`` arrays and a
frozen curriculum.  Any visible ``oracle`` directory is a fail-closed error.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.addressed_categorical_workspace import (
    AddressedContinuousTrackSModel,
    AnswerMotorControl,
    CategoricalTrackSModel,
    DenseCategoricalTrackSModel,
    GRUTrackSModel,
    PacketTokenTransformerTrackSModel,
    SourceRetainedTrackSModel,
    packet_to_symbols,
    symbols_to_packet,
    trainable_parameters,
)


ARM_IDS = (
    "acw",
    "dense_categorical",
    "addressed_continuous",
    "gru",
    "packet_token_transformer",
    "answer_motor",
    "source_retained",
)
EXPECTED_PARAMETERS = {
    "acw": 26_008,
    "dense_categorical": 26_250,
    "addressed_continuous": 26_008,
    "gru": 26_036,
    "packet_token_transformer": 25_872,
    "answer_motor": 25_939,
    "source_retained": 166_801,
}
PUBLIC_ARRAYS = (
    "public/event_features.npy",
    "public/event_addresses.npy",
    "public/train/source_features.npy",
    "public/train/event_ids.npy",
    "public/train/lengths.npy",
    "public/train/initial_queries.npy",
    "public/train/initial_answers.npy",
)
TRAINING_PROTOCOL = "R12-ACW-TRAINER-v2"
GENERATOR_PROTOCOL = "R12-ACW-HIDDEN-BASIS-v2"
BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v4"
PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v3"
PILOT_COMPARISON_PROTOCOL = "R12-ACW-PILOT-REPLAY-COMPARISON-v3"
BUNDLE_KEYS = {
    "protocol",
    "source_manifest_payload_sha256",
    "seed_identity",
    "data_replay_verification",
    "query_schedule_sha256",
    "query_schedule_kind",
    "pilot_report_payload_sha256",
    "pilot_report_sha256",
    "pilot_replay_comparison_payload_sha256",
    "pilot_replay_comparison_sha256",
    "pilot_artifacts",
    "arrays",
    "files",
    "oracle_paths_exported",
    "payload_sha256",
}
BUNDLE_DATA_REPLAY_KEYS = {
    "protocol",
    "seed_identity",
    "seed_fingerprint",
    "source_manifest_payload_sha256",
    "regenerated_manifest_payload_sha256",
    "array_registry_sha256",
    "arrays_verified",
    "public_arrays_verified",
    "oracle_arrays_verified",
}
BUNDLE_PILOT_ARTIFACTS = (
    "pilot/report.json",
    "pilot/replay_comparison.json",
    "pilot/cgb_schedule.jsonl",
    "pilot/uniform_schedule.jsonl",
)
BUNDLE_ARTIFACT_RECORD_KEYS = {"bytes", "sha256"}
CANONICAL_BUNDLE_BLOCK = (
    "canonical trainer bundles are disabled until the verified public pilot "
    "artifact registry is committed and pushed as an external anchor"
)
STATE_AUXILIARY_WEIGHT = 4.0
PILOT_SEED = 2026071600
DEVELOPMENT_SEEDS = (2026071601, 2026071602, 2026071603)
CONFIRMATION_COMMITMENTS = (
    "35102b3974877e8547b9b9c74156c63b71d467820f752301be21721b0f58e9a1",
    "737a6d6a76c3cdbfd07d84c83cfec5491cf13afeb8e077421af789cb652baa7f",
    "0e60eb70f2193ea57710db1f2cf9d6f93cf9b8e310b1b2cf5f4ea2694851854d",
)
ACW_SCIENTIFIC_PATHS = (
    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md",
    "R12_GOAL_CONDITIONED_VERSION_SPACE_CONTROLLER_PREREG.md",
    "pipeline/addressed_categorical_workspace.py",
    "pipeline/audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/generate_acw_hidden_basis.py",
    "pipeline/acw_nist_beacon.py",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/freeze_acw_curriculum.py",
    "pipeline/evaluate_acw_hidden_basis.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/test_addressed_categorical_workspace.py",
    "pipeline/test_audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/test_generate_acw_hidden_basis.py",
    "pipeline/test_acw_nist_beacon.py",
    "pipeline/testdata/acw_nist_beacon_snapshot.json",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_freeze_acw_curriculum.py",
    "pipeline/test_evaluate_acw_hidden_basis.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
    "pipeline/jobs/run_acw_pilot_stokes.sbatch",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


def _json_object(raw: bytes, label: str) -> dict:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_hash_bound_json(path: Path, label: str) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = _json_object(raw, label)
    payload = dict(value)
    recorded = _sha256(payload.pop("payload_sha256", None), f"{label} payload")
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError(f"{label} payload hash mismatch")
    return value, raw


def curriculum_query_schedule_sha256(path: Path) -> str:
    """Derive the consumed query schedule from exact curriculum rows."""

    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("curriculum must be nonempty newline-framed JSONL")
    digest = hashlib.sha256()
    for index, line in enumerate(raw.splitlines()):
        row = _json_object(line, f"curriculum row {index}")
        if set(row) != {"history_id", "query_id", "answer", "round"}:
            raise ValueError("curriculum row has the wrong schema")
        if canonical_json_bytes(row) != line:
            raise ValueError("curriculum row is not canonical JSON")
        schedule_row = {
            "history_id": row["history_id"],
            "query_id": row["query_id"],
            "round": row["round"],
        }
        digest.update(canonical_json_bytes(schedule_row) + b"\n")
    return digest.hexdigest()


def _validate_artifact_record(
    root: Path,
    relative: str,
    record: object,
) -> tuple[Path, bytes, str]:
    if not isinstance(record, dict) or set(record) != BUNDLE_ARTIFACT_RECORD_KEYS:
        raise ValueError(f"trainer bundle artifact record differs: {relative}")
    expected_bytes = record["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ValueError(f"trainer bundle artifact byte count differs: {relative}")
    expected_hash = _sha256(record["sha256"], f"trainer bundle {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    observed_hash = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_bytes or observed_hash != expected_hash:
        raise ValueError(f"trainer bundle artifact differs: {relative}")
    return path, raw, observed_hash


def validate_trainer_bundle_contract(root: Path, manifest: dict) -> dict:
    """Fail closed until a post-pilot Git anchor exists."""

    del root, manifest
    raise RuntimeError(CANONICAL_BUNDLE_BLOCK)


def _validate_unanchored_trainer_bundle_structure(root: Path, manifest: dict) -> dict:
    """Exercise v4 structure without granting canonical training admission."""

    root = root.resolve()
    if set(manifest) != BUNDLE_KEYS:
        raise ValueError("canonical trainer bundle has the wrong exact schema")
    if manifest.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("canonical trainer bundle has the wrong protocol")
    payload = dict(manifest)
    recorded_payload = _sha256(
        payload.pop("payload_sha256", None), "trainer bundle manifest payload"
    )
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded_payload:
        raise ValueError("dataset manifest payload hash mismatch")
    source_payload = _sha256(
        manifest["source_manifest_payload_sha256"], "source manifest payload"
    )
    seed_identity = manifest["seed_identity"]
    expected_optimizer_seed(seed_identity)
    schedule_kind = manifest["query_schedule_kind"]
    if schedule_kind not in {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
        raise ValueError("canonical trainer bundle lacks a registered curriculum kind")
    schedule_hash = _sha256(
        manifest["query_schedule_sha256"], "trainer bundle query schedule"
    )

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {"curriculum.jsonl"}:
        raise ValueError("canonical trainer bundle file registry differs")
    curriculum_record = files["curriculum.jsonl"]
    if not isinstance(curriculum_record, dict) or set(curriculum_record) != {
        "bytes",
        "rows",
        "sha256",
    }:
        raise ValueError("canonical trainer curriculum record differs")
    curriculum_path = root / "curriculum.jsonl"
    if not curriculum_path.is_file():
        raise FileNotFoundError(curriculum_path)
    curriculum_raw = curriculum_path.read_bytes()
    curriculum_hash = hashlib.sha256(curriculum_raw).hexdigest()
    expected_curriculum_bytes = curriculum_record["bytes"]
    if (
        isinstance(expected_curriculum_bytes, bool)
        or not isinstance(expected_curriculum_bytes, int)
        or expected_curriculum_bytes != len(curriculum_raw)
        or curriculum_hash
        != _sha256(curriculum_record["sha256"], "trainer curriculum file")
    ):
        raise ValueError("canonical trainer curriculum artifact differs")
    rows = curriculum_record.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("canonical trainer curriculum row count differs")
    if len(curriculum_raw.splitlines()) != rows:
        raise ValueError("canonical trainer curriculum framing differs")
    derived_schedule_hash = curriculum_query_schedule_sha256(curriculum_path)
    if derived_schedule_hash != schedule_hash:
        raise ValueError("curriculum-derived query schedule hash differs")

    replay = manifest["data_replay_verification"]
    if seed_identity.get("kind") == "development":
        if not isinstance(replay, dict) or set(replay) != BUNDLE_DATA_REPLAY_KEYS:
            raise ValueError("canonical trainer data replay schema differs")
        if (
            replay["protocol"] != "R12-ACW-DATA-REPLAY-v1"
            or replay["seed_identity"] != seed_identity
            or replay["source_manifest_payload_sha256"] != source_payload
            or replay["regenerated_manifest_payload_sha256"] != source_payload
        ):
            raise ValueError("canonical trainer data replay binding differs")
        _sha256(replay["seed_fingerprint"], "trainer bundle seed fingerprint")
        _sha256(replay["array_registry_sha256"], "trainer array registry")
        for name in (
            "arrays_verified",
            "public_arrays_verified",
            "oracle_arrays_verified",
        ):
            value = replay[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("canonical trainer data replay count differs")
    elif seed_identity.get("kind") == "confirmation":
        if replay is not None:
            raise ValueError("confirmation bundle may not claim public-seed replay")
    else:
        raise ValueError("canonical trainer bundle has an unregistered domain")

    arrays = manifest["arrays"]
    if not isinstance(arrays, dict) or set(arrays) != set(PUBLIC_ARRAYS):
        raise ValueError("canonical trainer public array registry differs")
    for relative, record in arrays.items():
        if not isinstance(record, dict) or set(record) != {
            "bytes",
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"canonical trainer array record differs: {relative}")
    if (
        isinstance(manifest["oracle_paths_exported"], bool)
        or manifest["oracle_paths_exported"] != 0
    ):
        raise ValueError("canonical trainer bundle exports oracle paths")
    if any(
        "oracle" in part.lower()
        for path in root.rglob("*")
        for part in path.relative_to(root).parts
    ):
        raise ValueError("canonical trainer bundle contains an oracle-named path")

    registry = manifest["pilot_artifacts"]
    if not isinstance(registry, dict) or set(registry) != set(BUNDLE_PILOT_ARTIFACTS):
        raise ValueError("canonical trainer pilot-artifact registry differs")
    opened = {
        relative: _validate_artifact_record(root, relative, registry[relative])
        for relative in BUNDLE_PILOT_ARTIFACTS
    }
    report, report_raw = _load_hash_bound_json(
        opened["pilot/report.json"][0], "trainer bundle pilot report"
    )
    comparison, comparison_raw = _load_hash_bound_json(
        opened["pilot/replay_comparison.json"][0],
        "trainer bundle pilot comparison",
    )
    if (
        report.get("protocol") != PILOT_PROTOCOL
        or report.get("payload_sha256")
        != _sha256(manifest["pilot_report_payload_sha256"], "pilot report payload")
        or hashlib.sha256(report_raw).hexdigest()
        != _sha256(manifest["pilot_report_sha256"], "pilot report file")
    ):
        raise ValueError("trainer bundle pilot report binding differs")
    if (
        comparison.get("protocol") != PILOT_COMPARISON_PROTOCOL
        or comparison.get("payload_sha256")
        != _sha256(
            manifest["pilot_replay_comparison_payload_sha256"],
            "pilot comparison payload",
        )
        or hashlib.sha256(comparison_raw).hexdigest()
        != _sha256(manifest["pilot_replay_comparison_sha256"], "pilot comparison file")
    ):
        raise ValueError("trainer bundle pilot comparison binding differs")

    pilot_identity = report.get("scientific_identity")
    if not isinstance(pilot_identity, dict) or set(pilot_identity) != {
        "scientific_commit",
        "scientific_path_sha256",
    }:
        raise ValueError("trainer bundle pilot scientific identity differs")
    commit = pilot_identity["scientific_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("trainer bundle pilot scientific commit differs")
    path_hashes = pilot_identity["scientific_path_sha256"]
    if not isinstance(path_hashes, dict) or not path_hashes:
        raise ValueError("trainer bundle pilot scientific paths differ")
    for relative, digest in path_hashes.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("trainer bundle pilot scientific path differs")
        _sha256(digest, f"trainer bundle pilot scientific path {relative}")

    schedules = report.get("schedules")
    expected_schedules = {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}
    if not isinstance(schedules, dict) or set(schedules) != expected_schedules:
        raise ValueError("trainer bundle pilot schedule registry differs")
    for name in expected_schedules:
        record = schedules[name]
        if not isinstance(record, dict) or set(record) != {
            "bytes",
            "rows",
            "sha256",
        }:
            raise ValueError("trainer bundle pilot schedule record differs")
        schedule_raw = opened[f"pilot/{name}"][1]
        if (
            len(schedule_raw) != record["bytes"]
            or len(schedule_raw.splitlines()) != record["rows"]
            or hashlib.sha256(schedule_raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("trainer bundle pilot schedule differs from report")
    if hashlib.sha256(opened[f"pilot/{schedule_kind}"][1]).hexdigest() != schedule_hash:
        raise ValueError("trainer bundle selected pilot schedule differs")

    common_files = comparison.get("common_files")
    expected_common = {"report.json", *expected_schedules}
    if not isinstance(common_files, dict) or set(common_files) != expected_common:
        raise ValueError("trainer bundle pilot comparison file registry differs")
    recomputed = comparison.get("independent_recomputation_sha256")
    if not isinstance(recomputed, dict) or set(recomputed) != expected_common:
        raise ValueError("trainer bundle pilot recomputation registry differs")
    for name in expected_common:
        relative = f"pilot/{name}"
        raw = opened[relative][1]
        record = common_files[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or record["bytes"] != len(raw)
            or record["sha256"] != hashlib.sha256(raw).hexdigest()
            or recomputed[name] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("trainer bundle pilot comparison binding differs")
    if (
        comparison.get("reports_byte_identical") is not True
        or comparison.get("schedules_byte_identical") is not True
        or comparison.get("independently_recomputed") is not True
        or comparison.get("dataset_manifest_payload_sha256")
        != report.get("dataset_manifest_payload_sha256")
        or comparison.get("scientific_identity") != report.get("scientific_identity")
    ):
        raise ValueError("trainer bundle pilot comparison differs from report")
    return {
        "payload_sha256": recorded_payload,
        "curriculum_sha256": curriculum_hash,
        "query_schedule_sha256": derived_schedule_hash,
        "pilot_report_payload_sha256": report["payload_sha256"],
        "pilot_replay_comparison_payload_sha256": comparison["payload_sha256"],
        "pilot_scientific_identity": pilot_identity,
        "pilot_artifacts_opened": len(opened),
    }


def scientific_identity(*, require_clean: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *ACW_SCIENTIFIC_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("ACW scientific paths are not clean in Git")
    hashes = {}
    for relative in ACW_SCIENTIFIC_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        working_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        committed_hash = hashlib.sha256(committed).hexdigest()
        if require_clean and working_hash != committed_hash:
            raise RuntimeError(
                f"ACW scientific working file differs from HEAD: {relative}"
            )
        hashes[relative] = working_hash
    if require_clean:
        remote = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        if not remote or remote[0] != commit:
            raise RuntimeError("ACW scientific HEAD must equal pushed origin/main")
    return {"scientific_commit": commit, "scientific_path_sha256": hashes}


def _load_array(root: Path, relative: str, manifest: dict) -> np.ndarray:
    record = manifest.get("arrays", {}).get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"manifest lacks required public array: {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"public array hash mismatch: {relative}")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"public array schema mismatch: {relative}")
    return array


@dataclass(frozen=True)
class PublicTrainingData:
    root: Path
    manifest_payload_sha256: str
    event_features: torch.Tensor
    event_addresses: torch.Tensor
    source_features: torch.Tensor
    event_ids: torch.Tensor
    lengths: torch.Tensor
    initial_queries: torch.Tensor
    initial_answers: torch.Tensor
    bound_curriculum_sha256: str | None
    query_schedule_sha256: str | None
    query_schedule_kind: str | None
    pilot_report_payload_sha256: str | None
    source_manifest_payload_sha256: str
    seed_identity: dict

    @property
    def histories(self) -> int:
        return int(self.lengths.shape[0])


def load_public_training_data(
    root: Path,
    *,
    reject_oracle: bool = True,
) -> PublicTrainingData:
    root = root.resolve()
    if reject_oracle and (root / "oracle").exists():
        raise RuntimeError("scored trainer bundle exposes forbidden oracle files")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    allowed_protocols = {GENERATOR_PROTOCOL, BUNDLE_PROTOCOL}
    if manifest.get("protocol") not in allowed_protocols:
        raise ValueError("wrong ACW public-data manifest protocol")
    bundle_contract = None
    if reject_oracle:
        if manifest.get("protocol") != BUNDLE_PROTOCOL:
            raise ValueError("canonical trainer requires a public-only trainer bundle")
        bundle_contract = validate_trainer_bundle_contract(root, manifest)
    payload = dict(manifest)
    recorded_payload = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded_payload:
        raise ValueError("dataset manifest payload hash mismatch")
    arrays = {
        relative: _load_array(root, relative, manifest) for relative in PUBLIC_ARRAYS
    }
    event_features = torch.from_numpy(arrays[PUBLIC_ARRAYS[0]].copy()).float()
    event_addresses = torch.from_numpy(arrays[PUBLIC_ARRAYS[1]].copy()).long()
    source_features = torch.from_numpy(arrays[PUBLIC_ARRAYS[2]].copy()).float()
    event_ids = torch.from_numpy(arrays[PUBLIC_ARRAYS[3]].copy()).long()
    lengths = torch.from_numpy(arrays[PUBLIC_ARRAYS[4]].copy()).long()
    initial_queries = torch.from_numpy(arrays[PUBLIC_ARRAYS[5]].copy()).long()
    initial_answers = torch.from_numpy(arrays[PUBLIC_ARRAYS[6]].copy()).long()
    histories = len(lengths)
    if source_features.shape != (histories, 96):
        raise ValueError("public source feature shape mismatch")
    if event_ids.shape[0] != histories or initial_queries.shape != (histories, 2):
        raise ValueError("public training history shape mismatch")
    if initial_answers.shape != initial_queries.shape:
        raise ValueError("initial query/answer shape mismatch")
    if event_features.shape != (48, 96) or event_addresses.shape != (48,):
        raise ValueError("public event bank shape mismatch")
    if bool(((event_ids >= 48) | (event_ids < -1)).any()):
        raise ValueError("event ID is outside the public event bank")
    return PublicTrainingData(
        root=root,
        manifest_payload_sha256=recorded_payload,
        event_features=event_features,
        event_addresses=event_addresses,
        source_features=source_features,
        event_ids=event_ids,
        lengths=lengths,
        initial_queries=initial_queries,
        initial_answers=initial_answers,
        bound_curriculum_sha256=(
            bundle_contract["curriculum_sha256"]
            if bundle_contract is not None
            else manifest.get("files", {}).get("curriculum.jsonl", {}).get("sha256")
        ),
        query_schedule_sha256=(
            bundle_contract["query_schedule_sha256"]
            if bundle_contract is not None
            else manifest.get("query_schedule_sha256")
        ),
        query_schedule_kind=manifest.get("query_schedule_kind"),
        pilot_report_payload_sha256=manifest.get("pilot_report_payload_sha256"),
        source_manifest_payload_sha256=manifest.get(
            "source_manifest_payload_sha256",
            recorded_payload,
        ),
        seed_identity=dict(manifest.get("seed_identity", {})),
    )


def expected_optimizer_seed(seed_identity: dict) -> int:
    kind = seed_identity.get("kind")
    if kind == "pilot":
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("public optimizer seed identity has the wrong schema")
        if int(seed_identity["seed"]) != PILOT_SEED:
            raise ValueError("pilot optimizer seed is outside the frozen registry")
        return PILOT_SEED
    if kind == "development":
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("public optimizer seed identity has the wrong schema")
        seed = int(seed_identity["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError(
                "development optimizer seed is outside the frozen registry"
            )
        return seed
    if kind == "confirmation":
        if set(seed_identity) != {"kind", "index", "commitment"}:
            raise ValueError(
                "confirmation optimizer seed identity has the wrong schema"
            )
        index = int(seed_identity["index"])
        if not 0 <= index < len(CONFIRMATION_COMMITMENTS):
            raise ValueError(
                "confirmation optimizer index is outside the frozen registry"
            )
        if seed_identity["commitment"] != CONFIRMATION_COMMITMENTS[index]:
            raise ValueError(
                "confirmation optimizer commitment is outside the frozen registry"
            )
        material = b"R12-ACW-OPT-v1\x00" + str(seed_identity["commitment"]).encode(
            "ascii"
        )
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63
    raise ValueError("unknown optimizer seed identity")


@dataclass(frozen=True)
class Curriculum:
    history_ids: torch.Tensor
    query_ids: torch.Tensor
    answers: torch.Tensor
    rounds: torch.Tensor

    def validate(self, histories: int, *, canonical: bool) -> None:
        count = len(self.history_ids)
        if any(
            len(field) != count for field in (self.query_ids, self.answers, self.rounds)
        ):
            raise ValueError("curriculum columns have different lengths")
        if bool(((self.history_ids < 0) | (self.history_ids >= histories)).any()):
            raise ValueError("curriculum history ID is outside the public data")
        if bool(((self.query_ids < 0) | (self.query_ids >= 24)).any()):
            raise ValueError("curriculum query ID is outside the public bank")
        if bool(((self.answers < 0) | (self.answers >= 17)).any()):
            raise ValueError("curriculum answer is outside F_17")
        if bool(((self.rounds < 0) | (self.rounds > 12)).any()):
            raise ValueError("curriculum round is outside [0,12]")
        pairs = torch.stack((self.history_ids, self.query_ids), dim=1)
        if len(torch.unique(pairs, dim=0)) != count:
            raise ValueError("curriculum repeats a history/query pair")
        if canonical:
            if histories != 4096 or count != 57_344:
                raise ValueError("canonical curriculum count mismatch")
            per_history = torch.bincount(self.history_ids, minlength=histories)
            if not torch.equal(per_history, torch.full_like(per_history, 14)):
                raise ValueError("canonical curriculum must have 14 labels per history")
            round_counts = torch.bincount(self.rounds, minlength=13)
            expected_round_counts = torch.full_like(round_counts, histories)
            expected_round_counts[0] = 2 * histories
            if not torch.equal(round_counts, expected_round_counts):
                raise ValueError(
                    "canonical curriculum must start with two labels per history "
                    "and add one label per history in rounds 1..12"
                )


def initial_curriculum(data: PublicTrainingData) -> Curriculum:
    histories = torch.arange(data.histories).repeat_interleave(2)
    return Curriculum(
        history_ids=histories,
        query_ids=data.initial_queries.reshape(-1),
        answers=data.initial_answers.reshape(-1),
        rounds=torch.zeros(data.histories * 2, dtype=torch.long),
    )


def load_curriculum(path: Path) -> Curriculum:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed curriculum row {line_number}") from error
            if set(record) != {"history_id", "query_id", "answer", "round"}:
                raise ValueError(f"wrong curriculum schema at row {line_number}")
            records.append(record)
    if not records:
        raise ValueError("curriculum is empty")
    return Curriculum(
        history_ids=torch.tensor(
            [row["history_id"] for row in records], dtype=torch.long
        ),
        query_ids=torch.tensor([row["query_id"] for row in records], dtype=torch.long),
        answers=torch.tensor([row["answer"] for row in records], dtype=torch.long),
        rounds=torch.tensor([row["round"] for row in records], dtype=torch.long),
    )


def model_for_arm(arm: str) -> torch.nn.Module:
    if arm == "acw":
        model = CategoricalTrackSModel()
    elif arm == "dense_categorical":
        model = DenseCategoricalTrackSModel()
    elif arm == "addressed_continuous":
        model = AddressedContinuousTrackSModel()
    elif arm == "gru":
        model = GRUTrackSModel()
    elif arm == "packet_token_transformer":
        model = PacketTokenTransformerTrackSModel()
    elif arm == "answer_motor":
        model = AnswerMotorControl()
    elif arm == "source_retained":
        model = SourceRetainedTrackSModel()
    else:
        raise ValueError(f"unknown ACW arm: {arm}")
    if trainable_parameters(model) != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError("arm parameter count drifted from preregistration")
    return model


def initialized_model_for_arm(arm: str, seed: int) -> torch.nn.Module:
    """Construct an arm only after its declared RNG seed is installed."""

    set_determinism(seed)
    return model_for_arm(arm)


def arm_resource_ledger(arm: str, model: torch.nn.Module) -> dict:
    ledgers = {
        "acw": (3 * np.log2(17), 3, 3 * 17 * 4, 0, "uint8"),
        "dense_categorical": (3 * np.log2(17), 3, 3 * 17 * 4, 0, "uint8"),
        "addressed_continuous": (96.0, 12, 12, 0, "float32"),
        "gru": (39 * 32.0, 156, 156, 0, "float32"),
        "packet_token_transformer": (
            3 * np.log2(17),
            3,
            3 * 17 * 4,
            7 * 24 * 4,
            "uint8",
        ),
        "answer_motor": (192 * 32.0, 768, 768, 0, "float32"),
        "source_retained": (224 * 32.0, 896, 896, 0, "float32"),
    }
    semantic_bits, persistent_bytes, training_bytes, transient_bytes, dtype = ledgers[
        arm
    ]
    return {
        "trainable_parameters": trainable_parameters(model),
        "semantic_state_bits": float(semantic_bits),
        "persistent_evaluation_bytes": persistent_bytes,
        "persistent_evaluation_dtype": dtype,
        "persistent_training_state_bytes": training_bytes,
        "declared_transient_token_bytes": transient_bytes,
        "parameter_matched_primary": arm != "source_retained",
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _profiler_measurements(profiler: torch.profiler.profile) -> dict:
    events = profiler.events()
    inventory: dict[str, dict[str, int | str]] = {}
    for event in events:
        name = str(event.name)
        entry = inventory.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "operator_reported_flops": 0,
                "positive_allocation_bytes": 0,
                "positive_self_allocation_bytes": 0,
            },
        )
        entry["calls"] = int(entry["calls"]) + 1
        entry["operator_reported_flops"] = int(entry["operator_reported_flops"]) + int(
            event.flops or 0
        )
        entry["positive_allocation_bytes"] = int(
            entry["positive_allocation_bytes"]
        ) + max(0, int(event.cpu_memory_usage))
        entry["positive_self_allocation_bytes"] = int(
            entry["positive_self_allocation_bytes"]
        ) + max(0, int(event.self_cpu_memory_usage))
    ordered_inventory = [inventory[name] for name in sorted(inventory)]
    positive_allocations = [max(0, int(event.cpu_memory_usage)) for event in events]
    positive_self_allocations = [
        max(0, int(event.self_cpu_memory_usage)) for event in events
    ]
    return {
        "profiler_event_count": len(events),
        "operator_inventory": ordered_inventory,
        "uncounted_operator_names": [
            str(entry["name"])
            for entry in ordered_inventory
            if int(entry["operator_reported_flops"]) == 0
        ],
        "operator_inventory_complete": True,
        "operator_reported_flops": sum(int(event.flops or 0) for event in events),
        "largest_operator_allocation_bytes": max(positive_allocations, default=0),
        "largest_self_operator_allocation_bytes": max(
            positive_self_allocations,
            default=0,
        ),
        "total_positive_operator_allocations_bytes": sum(positive_allocations),
        "flop_counting_contract": (
            "PyTorch CPU profiler operator-reported FLOPs; unsupported operators are "
            "listed as uncounted rather than imputed."
        ),
        "transient_memory_contract": (
            "Runtime CPU profiler allocations; largest operator and self-operator "
            "allocations are reported without claiming allocator-wide liveness."
        ),
    }


def profile_answer_step_resources(
    model: torch.nn.Module,
    arm: str,
    data: PublicTrainingData,
    curriculum: Curriculum,
    *,
    batch_size: int,
) -> dict:
    profile_model = copy.deepcopy(model).train()
    count = min(batch_size, len(curriculum.history_ids))
    selected = torch.arange(count)
    optimizer = torch.optim.AdamW(
        profile_model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )

    def training_step() -> None:
        logits = forward_logits(
            profile_model,
            arm,
            data,
            curriculum.history_ids[selected],
            curriculum.query_ids[selected],
            training=True,
        )
        loss = F.cross_entropy(logits.float(), curriculum.answers[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    training_step()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        profile_memory=True,
        acc_events=True,
    ) as profiler:
        training_step()
    report = {
        "scope": "one complete forward+backward+AdamW answer-loss update",
        "batch_size": count,
        "active_events": int(data.lengths[curriculum.history_ids[selected]].sum()),
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "optimizer_included": True,
    }
    report.update(_profiler_measurements(profiler))
    return report


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _frozen_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def profile_inference_resources(
    model: torch.nn.Module,
    arm: str,
    data: PublicTrainingData,
    curriculum: Curriculum,
    *,
    batch_size: int,
) -> dict:
    profile_model = copy.deepcopy(model).eval()
    count = min(batch_size, len(curriculum.history_ids))
    selected = torch.arange(count)

    def inference_step() -> torch.Tensor:
        with torch.no_grad():
            return forward_logits(
                profile_model,
                arm,
                data,
                curriculum.history_ids[selected],
                curriculum.query_ids[selected],
                training=False,
                literal_symbols=arm
                in {"acw", "dense_categorical", "packet_token_transformer"},
            )

    inference_step()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        profile_memory=True,
        acc_events=True,
    ) as profiler:
        inference_step()
    report = {
        "scope": "one source-deleted literal-state inference batch including all events and reader",
        "batch_size": count,
        "active_events": int(data.lengths[curriculum.history_ids[selected]].sum()),
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }
    report.update(_profiler_measurements(profiler))
    return report


def _history_events(
    data: PublicTrainingData,
    history_ids: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event_id = data.event_ids[history_ids, step]
    active = step < data.lengths[history_ids]
    safe_id = event_id.clamp_min(0)
    hidden = data.event_features[safe_id]
    address = data.event_addresses[safe_id]
    return hidden, address, active


def _mean_event_features(
    data: PublicTrainingData,
    history_ids: torch.Tensor,
) -> torch.Tensor:
    batch = len(history_ids)
    total = torch.zeros(batch, 96, dtype=data.event_features.dtype)
    for step in range(data.event_ids.shape[1]):
        hidden, _, active = _history_events(data, history_ids, step)
        total = total + hidden * active.unsqueeze(1)
    denominator = data.lengths[history_ids].clamp_min(1).unsqueeze(1)
    return total / denominator


def recurrent_state(
    model: torch.nn.Module,
    arm: str,
    data: PublicTrainingData,
    history_ids: torch.Tensor,
    *,
    training: bool,
    literal_symbols: bool = False,
):
    source = data.source_features[history_ids]
    max_steps = data.event_ids.shape[1]
    if arm == "answer_motor":
        return source, _mean_event_features(data, history_ids)
    if arm == "source_retained":
        state = model.encode_source(source)
        for step in range(max_steps):
            hidden, address, active = _history_events(data, history_ids, step)
            updated = model.update(state, hidden, address)
            state = torch.where(active.unsqueeze(1), updated, state)
        return state, source
    if arm == "acw":
        workspace = model.workspace
        if literal_symbols:
            state = workspace.encode_source_symbols(source)
            for step in range(max_steps):
                hidden, address, active = _history_events(data, history_ids, step)
                event = workspace.encode_event_symbols(hidden)
                updated = workspace.update_symbols(state, event, address)
                state = torch.where(active.unsqueeze(1), updated, state)
            return state
        state = workspace.encode_source(source, straight_through=training)
        for step in range(max_steps):
            hidden, address, active = _history_events(data, history_ids, step)
            event = workspace.encode_event(hidden, straight_through=training)
            updated = workspace.update(
                state,
                event,
                address,
                straight_through=training,
            )
            state = torch.where(active[:, None, None], updated, state)
        return state
    if arm in {"dense_categorical", "packet_token_transformer"}:
        state = model.encode_source(source, straight_through=training)
        if literal_symbols:
            state = packet_to_symbols(state)
        for step in range(max_steps):
            hidden, address, active = _history_events(data, history_ids, step)
            event = model.encode_event(hidden, straight_through=training)
            if literal_symbols:
                float_state = symbols_to_packet(
                    state,
                    17,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                event = symbols_to_packet(
                    packet_to_symbols(event),
                    17,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                updated = model.update(
                    float_state,
                    event,
                    address,
                    straight_through=False,
                )
                updated = packet_to_symbols(updated)
                state = torch.where(active.unsqueeze(1), updated, state)
            else:
                updated = model.update(
                    state,
                    event,
                    address,
                    straight_through=training,
                )
                state = torch.where(active[:, None, None], updated, state)
        return state
    if arm == "addressed_continuous":
        state = model.encode_source(source)
        for step in range(max_steps):
            hidden, address, active = _history_events(data, history_ids, step)
            event = model.encode_event(hidden, straight_through=training)
            updated = model.update(state, event, address)
            state = torch.where(active.unsqueeze(1), updated, state)
        return state
    if arm == "gru":
        state = model.encode_source(source)
        for step in range(max_steps):
            hidden, address, active = _history_events(data, history_ids, step)
            updated = model.update(state, hidden, address)
            state = torch.where(active.unsqueeze(1), updated, state)
        return state
    raise AssertionError("unreachable arm")


def forward_logits(
    model: torch.nn.Module,
    arm: str,
    data: PublicTrainingData,
    history_ids: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    training: bool,
    literal_symbols: bool = False,
) -> torch.Tensor:
    state = recurrent_state(
        model,
        arm,
        data,
        history_ids,
        training=training,
        literal_symbols=literal_symbols,
    )
    if arm == "answer_motor":
        return model(state[0], state[1], query_ids)
    if arm == "source_retained":
        return model.read(state[0], state[1], query_ids)
    if arm == "acw":
        if literal_symbols:
            return model.reader(model.workspace.packet_delta_symbols(state), query_ids)
        return model.read(state, query_ids)
    if arm in {"dense_categorical", "packet_token_transformer"} and literal_symbols:
        state = symbols_to_packet(state, 17, dtype=torch.float32)
    return model.read(state, query_ids)


@dataclass(frozen=True)
class DirectStateTruth:
    source_manifest_payload_sha256: str
    source_states: torch.Tensor
    trajectory_states: torch.Tensor


def load_direct_state_truth(root: Path) -> DirectStateTruth:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError("oracle manifest payload hash mismatch")
    source = _load_array(root, "oracle/train/source_states.npy", manifest)
    trajectory = _load_array(root, "oracle/train/trajectory_states.npy", manifest)
    if trajectory.shape != (len(source), 9, 3) or source.shape != (len(source), 3):
        raise ValueError("direct-state oracle trajectory shape mismatch")
    if not np.array_equal(source, trajectory[:, 0]):
        raise ValueError("direct-state source and trajectory origin disagree")
    return DirectStateTruth(
        source_manifest_payload_sha256=recorded,
        source_states=torch.from_numpy(source.copy()).long(),
        trajectory_states=torch.from_numpy(trajectory.copy()).long(),
    )


def direct_state_forward(
    model: CategoricalTrackSModel,
    data: PublicTrainingData,
    truth: DirectStateTruth,
    history_ids: torch.Tensor,
    query_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = data.source_features[history_ids]
    packet = model.workspace.encode_source(source, straight_through=True)
    losses = []
    weights = []

    def add_state_loss(
        observed: torch.Tensor, target: torch.Tensor, active: torch.Tensor
    ) -> None:
        one_hot = F.one_hot(target.clamp_min(0), 17).to(observed.dtype)
        per_history = 0.5 * (observed - one_hot).square().sum(dim=-1).mean(dim=-1)
        losses.append((per_history * active).sum())
        weights.append(active.sum())

    active = torch.ones(len(history_ids), dtype=packet.dtype)
    add_state_loss(packet, truth.trajectory_states[history_ids, 0], active)
    for step in range(data.event_ids.shape[1]):
        hidden, address, active_bool = _history_events(data, history_ids, step)
        event = model.workspace.encode_event(hidden, straight_through=True)
        updated = model.workspace.update(
            packet,
            event,
            address,
            straight_through=True,
        )
        packet = torch.where(active_bool[:, None, None], updated, packet)
        add_state_loss(
            packet,
            truth.trajectory_states[history_ids, step + 1],
            active_bool.to(packet.dtype),
        )
    state_loss = torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1)
    logits = model.read(packet, query_ids)
    return logits, state_loss


def profile_direct_state_step_resources(
    model: CategoricalTrackSModel,
    data: PublicTrainingData,
    truth: DirectStateTruth,
    curriculum: Curriculum,
    *,
    batch_size: int,
) -> dict:
    profile_model = copy.deepcopy(model).train()
    count = min(batch_size, len(curriculum.history_ids))
    selected = torch.arange(count)
    optimizer = torch.optim.AdamW(
        profile_model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )

    def training_step() -> None:
        logits, state_loss = direct_state_forward(
            profile_model,
            data,
            truth,
            curriculum.history_ids[selected],
            curriculum.query_ids[selected],
        )
        answer_loss = F.cross_entropy(logits.float(), curriculum.answers[selected])
        loss = answer_loss + STATE_AUXILIARY_WEIGHT * state_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    training_step()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        profile_memory=True,
        acc_events=True,
    ) as profiler:
        training_step()
    report = {
        "scope": "one complete direct-state forward+backward+AdamW update",
        "batch_size": count,
        "active_events": int(data.lengths[curriculum.history_ids[selected]].sum()),
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "optimizer_included": True,
        "state_auxiliary_weight": STATE_AUXILIARY_WEIGHT,
    }
    report.update(_profiler_measurements(profiler))
    return report


def train_direct_state_model(
    model: CategoricalTrackSModel,
    data: PublicTrainingData,
    truth: DirectStateTruth,
    curriculum: Curriculum,
    *,
    seed: int,
    updates_per_round: int = 200,
    final_updates: int = 800,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    weight_decay: float = 0.0001,
    canonical: bool = True,
) -> dict:
    if data.source_manifest_payload_sha256 != truth.source_manifest_payload_sha256:
        raise ValueError(
            "public bundle and direct-state oracle are from different domains"
        )
    curriculum.validate(data.histories, canonical=canonical)
    resource_measurements = (
        {
            "training": profile_direct_state_step_resources(
                model,
                data,
                truth,
                curriculum,
                batch_size=batch_size,
            ),
            "inference": profile_inference_resources(
                model,
                "acw",
                data,
                curriculum,
                batch_size=batch_size,
            ),
        }
        if canonical
        else None
    )
    started = time.perf_counter()
    generator = set_determinism(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    losses = []
    state_losses = []
    updates = 0

    def run_updates(eligible: torch.Tensor, count: int) -> None:
        nonlocal updates
        for _ in range(count):
            selected = eligible[
                torch.randint(len(eligible), (batch_size,), generator=generator)
            ]
            logits, state_loss = direct_state_forward(
                model,
                data,
                truth,
                curriculum.history_ids[selected],
                curriculum.query_ids[selected],
            )
            answer_loss = F.cross_entropy(
                logits.float(),
                curriculum.answers[selected],
            )
            loss = answer_loss + STATE_AUXILIARY_WEIGHT * state_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            state_losses.append(float(state_loss.detach()))
            updates += 1

    for round_index in range(13):
        eligible = torch.nonzero(curriculum.rounds <= round_index).flatten()
        run_updates(eligible, updates_per_round)
    run_updates(torch.arange(len(curriculum.history_ids)), final_updates)
    return {
        "updates": updates,
        "labels": len(curriculum.history_ids),
        "oracle_source_manifest_payload_sha256": truth.source_manifest_payload_sha256,
        "state_auxiliary_weight": STATE_AUXILIARY_WEIGHT,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "state_loss_first": state_losses[0],
        "state_loss_last": state_losses[-1],
        "wall_seconds": time.perf_counter() - started,
        "resource_ledger": arm_resource_ledger("acw", model),
        "resource_measurements": resource_measurements,
        "equal_label_compute_comparison": False,
    }


def set_determinism(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def train_model(
    model: torch.nn.Module,
    arm: str,
    data: PublicTrainingData,
    curriculum: Curriculum,
    *,
    seed: int,
    updates_per_round: int = 200,
    final_updates: int = 800,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    weight_decay: float = 0.0001,
    canonical: bool = True,
) -> dict:
    curriculum.validate(data.histories, canonical=canonical)
    resource_measurements = (
        {
            "training": profile_answer_step_resources(
                model,
                arm,
                data,
                curriculum,
                batch_size=batch_size,
            ),
            "inference": profile_inference_resources(
                model,
                arm,
                data,
                curriculum,
                batch_size=batch_size,
            ),
        }
        if canonical
        else None
    )
    started = time.perf_counter()
    generator = set_determinism(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    losses = []
    updates = 0
    label_efficiency = []
    label_efficiency_models = []

    def run_updates(eligible: torch.Tensor, count: int) -> None:
        nonlocal updates
        for _ in range(count):
            selected = eligible[
                torch.randint(len(eligible), (batch_size,), generator=generator)
            ]
            logits = forward_logits(
                model,
                arm,
                data,
                curriculum.history_ids[selected],
                curriculum.query_ids[selected],
                training=True,
            )
            loss = F.cross_entropy(logits.float(), curriculum.answers[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1

    for round_index in range(13):
        eligible = torch.nonzero(curriculum.rounds <= round_index).flatten()
        run_updates(eligible, updates_per_round)
        if canonical and round_index < 12:
            state = _frozen_model_state(model)
            label_efficiency.append(
                {
                    "round": round_index,
                    "labels": int(len(eligible)),
                    "optimizer_updates": updates,
                    "model_tensor_sha256": _tensor_state_sha256(state),
                }
            )
            label_efficiency_models.append(state)
    run_updates(torch.arange(len(curriculum.history_ids)), final_updates)
    if canonical:
        state = _frozen_model_state(model)
        label_efficiency.append(
            {
                "round": 12,
                "labels": len(curriculum.history_ids),
                "optimizer_updates": updates,
                "model_tensor_sha256": _tensor_state_sha256(state),
            }
        )
        label_efficiency_models.append(state)
    return {
        "updates": updates,
        "labels": len(curriculum.history_ids),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "wall_seconds": time.perf_counter() - started,
        "resource_ledger": arm_resource_ledger(arm, model),
        "resource_measurements": resource_measurements,
        "label_efficiency": label_efficiency,
        "_label_efficiency_models": label_efficiency_models,
    }


def write_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    arm: str,
    seed: int,
    data: PublicTrainingData,
    curriculum_sha256: str,
    training_report: dict,
    scientific_identity_record: dict | None = None,
) -> dict:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_report = dict(training_report)
    label_efficiency_models = training_report.pop("_label_efficiency_models", None)
    payload = {
        "protocol": TRAINING_PROTOCOL,
        "arm": arm,
        "seed": seed,
        "dataset_manifest_payload_sha256": data.manifest_payload_sha256,
        "source_manifest_payload_sha256": data.source_manifest_payload_sha256,
        "curriculum_sha256": curriculum_sha256,
        "query_schedule_sha256": data.query_schedule_sha256,
        "query_schedule_kind": data.query_schedule_kind,
        "pilot_report_payload_sha256": data.pilot_report_payload_sha256,
        "parameters": trainable_parameters(model),
        "training_report": training_report,
        "label_efficiency_models": label_efficiency_models,
        "scientific_identity": scientific_identity_record,
        "model": model.state_dict(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARM_IDS, "direct_state_acw"), required=True)
    parser.add_argument("--oracle-dataset", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load_public_training_data(args.bundle, reject_oracle=True)
    if args.seed != expected_optimizer_seed(data.seed_identity):
        raise ValueError(
            "optimizer seed does not match the trainer-bundle domain identity"
        )
    curriculum_hash = file_sha256(args.curriculum)
    if data.bound_curriculum_sha256 is None:
        raise ValueError("trainer bundle does not bind curriculum.jsonl")
    if curriculum_hash != data.bound_curriculum_sha256:
        raise ValueError("curriculum does not match the trainer-bundle binding")
    curriculum = load_curriculum(args.curriculum)
    if args.arm == "direct_state_acw":
        if args.oracle_dataset is None:
            raise ValueError("direct_state_acw requires --oracle-dataset")
        model = initialized_model_for_arm("acw", args.seed)
        truth = load_direct_state_truth(args.oracle_dataset)
        training_report = train_direct_state_model(
            model,
            data,
            truth,
            curriculum,
            seed=args.seed,
            canonical=True,
        )
    else:
        if args.oracle_dataset is not None:
            raise ValueError("scored arms may not receive --oracle-dataset")
        model = initialized_model_for_arm(args.arm, args.seed)
        training_report = train_model(
            model,
            args.arm,
            data,
            curriculum,
            seed=args.seed,
            canonical=True,
        )
    checkpoint = write_checkpoint(
        args.out,
        model,
        arm=args.arm,
        seed=args.seed,
        data=data,
        curriculum_sha256=curriculum_hash,
        training_report=training_report,
        scientific_identity_record=scientific_identity(require_clean=True),
    )
    print(
        f"[acw-train] arm={args.arm} updates={training_report['updates']} "
        f"sha256={checkpoint['sha256']}"
    )


if __name__ == "__main__":
    main()
