"""Replay-verify ACW data and freeze reproducible public trainer curricula.

Canonical pilot and development datasets are regenerated from their registered
public seed material before use.  A pilot schedule is frozen only after two
isolated executions produce byte-identical reports and schedules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.acw_hidden_basis_training import (
    CANONICAL_BUNDLE_BLOCK,
    CONFIRMATION_COMMITMENTS,
    PUBLIC_ARRAYS,
    Curriculum,
    PublicTrainingData,
    canonical_json_bytes,
    curriculum_query_schedule_sha256,
    file_sha256,
    forward_logits,
    initialized_model_for_arm,
    load_public_training_data,
    recurrent_state,
    scientific_identity,
)
from pipeline.generate_acw_hidden_basis import (
    ADAPTATION_HISTORIES as GENERATOR_ADAPTATION_HISTORIES,
    DEVELOPMENT_SEEDS,
    EVALUATION_DEPTHS as GENERATOR_EVALUATION_DEPTHS,
    EVALUATION_HISTORIES as GENERATOR_EVALUATION_HISTORIES,
    GENERATOR_PROTOCOL,
    PILOT_SEED as GENERATOR_PILOT_SEED,
    TRAIN_HISTORIES as GENERATOR_TRAIN_HISTORIES,
    development_seed_material,
    generate_dataset,
)


PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v4"
SCHEDULE_PROTOCOL = "R12-ACW-QUERY-SCHEDULE-v3"
BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v4"
DATA_REPLAY_PROTOCOL = "R12-ACW-DATA-REPLAY-v1"
PILOT_EXECUTION_PROTOCOL = "R12-ACW-PILOT-REPLAY-EXECUTION-v3"
PILOT_COMPARISON_PROTOCOL = "R12-ACW-PILOT-REPLAY-COMPARISON-v4"
PILOT_ORCHESTRATION_PROTOCOL = "R12-ACW-PILOT-ORCHESTRATION-v1"
PILOT_SEED = 2026071600
UNIFORM_SEED = 2026071604
PUBLIC_QUERIES = 24
REFINEMENT_ROUNDS = 12
MAX_GROUPS_PER_ROUND = 512
MAX_CANDIDATE_EVALUATIONS = 147_456
CANONICAL_HISTORIES = 4096
CANONICAL_LABELS = 57_344
CANONICAL_UPDATES_PER_ROUND = 200
CANONICAL_FINAL_UPDATES = 800
CANONICAL_BATCH_SIZE = 256
CANONICAL_TOTAL_UPDATES = (
    REFINEMENT_ROUNDS + 1
) * CANONICAL_UPDATES_PER_ROUND + CANONICAL_FINAL_UPDATES
CANONICAL_PILOT_DATASET = "artifacts/r12/acw_pilot_domain_v3"
CANONICAL_PILOT_REPLAY_A = "artifacts/r12/acw_cgbr_pilot_v4_replay_a"
CANONICAL_PILOT_REPLAY_B = "artifacts/r12/acw_cgbr_pilot_v4_replay_b"
CANONICAL_PILOT_OUTPUT = "artifacts/r12/acw_cgbr_pilot_v4"

if GENERATOR_PILOT_SEED != PILOT_SEED:
    raise RuntimeError("freezer and generator pilot seed registries differ")


def _load_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != recorded:
        raise ValueError("dataset manifest payload hash mismatch")
    if manifest.get("protocol") != GENERATOR_PROTOCOL:
        raise ValueError("wrong ACW generator manifest protocol")
    return manifest


def _load_bound_array(root: Path, manifest: dict, relative: str) -> np.ndarray:
    record = manifest.get("arrays", {}).get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"manifest lacks array: {relative}")
    path = root / relative
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"array hash mismatch: {relative}")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"array schema mismatch: {relative}")
    return array


def _registered_public_seed_material(
    seed_identity: dict,
    *,
    allowed_kinds: set[str],
) -> bytes:
    kind = seed_identity.get("kind")
    if kind not in allowed_kinds:
        raise ValueError(f"registered deterministic replay forbids {kind!r} data")
    if kind == "pilot":
        if seed_identity != {"kind": "pilot", "seed": PILOT_SEED}:
            raise ValueError("pilot identity is not the registered public seed")
        return development_seed_material(PILOT_SEED)
    if kind == "development":
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("development seed identity has the wrong schema")
        seed = int(seed_identity["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError("development seed is outside the public registry")
        return development_seed_material(seed)
    raise ValueError("only pilot and development domains have public replay material")


def _regenerate_registered_dataset(
    out: Path,
    seed_material: bytes,
    seed_identity: dict,
) -> dict:
    return generate_dataset(
        out,
        seed_material,
        seed_identity=seed_identity,
        train_count=GENERATOR_TRAIN_HISTORIES,
        adaptation_count=GENERATOR_ADAPTATION_HISTORIES,
        evaluation_count=GENERATOR_EVALUATION_HISTORIES,
        evaluation_depths=GENERATOR_EVALUATION_DEPTHS,
    )


def _data_files(root: Path) -> set[str]:
    files = set()
    for prefix in ("public", "oracle"):
        directory = root / prefix
        if directory.exists():
            files.update(
                str(path.relative_to(root))
                for path in directory.rglob("*")
                if path.is_file()
            )
    return files


def verify_registered_dataset(
    root: Path,
    *,
    allowed_kinds: set[str],
) -> dict:
    """Regenerate a public-seed domain and compare every public/oracle array."""
    root = root.resolve()
    manifest = _load_manifest(root)
    if manifest.get("protocol") != GENERATOR_PROTOCOL:
        raise ValueError("dataset does not use the registered generator protocol")
    seed_identity = manifest.get("seed_identity")
    if not isinstance(seed_identity, dict):
        raise ValueError("dataset lacks a registered seed identity")
    seed_material = _registered_public_seed_material(
        seed_identity,
        allowed_kinds=allowed_kinds,
    )
    expected_fingerprint = hashlib.sha256(seed_material).hexdigest()
    if manifest.get("seed_fingerprint") != expected_fingerprint:
        raise ValueError("dataset seed fingerprint fails deterministic replay")

    with tempfile.TemporaryDirectory(prefix="acw-data-replay-") as temporary:
        replay_root = Path(temporary) / "dataset"
        expected = _regenerate_registered_dataset(
            replay_root,
            seed_material,
            dict(seed_identity),
        )
        observed_arrays = manifest.get("arrays")
        expected_arrays = expected.get("arrays")
        if not isinstance(observed_arrays, dict) or set(observed_arrays) != set(
            expected_arrays or {}
        ):
            raise ValueError("dataset array registry fails deterministic replay")
        expected_files = set(expected_arrays)
        if _data_files(root) != expected_files:
            raise ValueError("dataset public/oracle files fail deterministic replay")
        if _data_files(replay_root) != expected_files:
            raise RuntimeError(
                "registered generator emitted an unexpected file registry"
            )

        public_count = 0
        oracle_count = 0
        for relative in sorted(expected_files):
            observed = _load_bound_array(root, manifest, relative)
            regenerated = _load_bound_array(replay_root, expected, relative)
            if not np.array_equal(observed, regenerated):
                raise ValueError(
                    f"dataset array differs from deterministic replay: {relative}"
                )
            public_count += relative.startswith("public/")
            oracle_count += relative.startswith("oracle/")

        if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
            raise ValueError("dataset manifest differs from deterministic replay")

    return {
        "protocol": DATA_REPLAY_PROTOCOL,
        "seed_identity": dict(seed_identity),
        "seed_fingerprint": expected_fingerprint,
        "source_manifest_payload_sha256": manifest["payload_sha256"],
        "regenerated_manifest_payload_sha256": expected["payload_sha256"],
        "array_registry_sha256": hashlib.sha256(
            canonical_json_bytes(expected["arrays"])
        ).hexdigest(),
        "arrays_verified": len(expected["arrays"]),
        "public_arrays_verified": public_count,
        "oracle_arrays_verified": oracle_count,
    }


def load_oracle_truth(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    root = root.resolve()
    manifest = _load_manifest(root)
    states = _load_bound_array(root, manifest, "oracle/train/final_states.npy")
    answers = _load_bound_array(root, manifest, "oracle/train/public_answers.npy")
    if states.shape != (answers.shape[0], 3) or answers.shape[1] != PUBLIC_QUERIES:
        raise ValueError("oracle training truth has the wrong shape")
    return states, answers, manifest


def _initial_rows(data: PublicTrainingData, oracle_answers: np.ndarray) -> list[dict]:
    rows = []
    for history_id in range(data.histories):
        for query_id, answer in zip(
            data.initial_queries[history_id].tolist(),
            data.initial_answers[history_id].tolist(),
            strict=True,
        ):
            if int(oracle_answers[history_id, query_id]) != answer:
                raise ValueError("public initial answer disagrees with oracle truth")
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": int(query_id),
                    "round": 0,
                }
            )
    return rows


def _rows_to_curriculum(rows: list[dict], oracle_answers: np.ndarray) -> Curriculum:
    return Curriculum(
        history_ids=torch.tensor([row["history_id"] for row in rows], dtype=torch.long),
        query_ids=torch.tensor([row["query_id"] for row in rows], dtype=torch.long),
        answers=torch.tensor(
            [oracle_answers[row["history_id"], row["query_id"]] for row in rows],
            dtype=torch.long,
        ),
        rounds=torch.tensor([row["round"] for row in rows], dtype=torch.long),
    )


def validate_query_schedule(
    rows: list[dict],
    histories: int,
    *,
    refinement_rounds: int,
    canonical: bool,
) -> None:
    if any(set(row) != {"history_id", "query_id", "round"} for row in rows):
        raise ValueError("query schedule has the wrong schema")
    pairs = set()
    per_history = [0] * histories
    round_counts = [0] * (refinement_rounds + 1)
    for row in rows:
        history_id = row["history_id"]
        query_id = row["query_id"]
        round_index = row["round"]
        if not 0 <= history_id < histories or not 0 <= query_id < PUBLIC_QUERIES:
            raise ValueError("query schedule index is outside the public domain")
        if not 0 <= round_index <= refinement_rounds:
            raise ValueError("query schedule round is outside the configured range")
        pair = (history_id, query_id)
        if pair in pairs:
            raise ValueError("query schedule repeats a history/query pair")
        pairs.add(pair)
        per_history[history_id] += 1
        round_counts[round_index] += 1
    expected_per_history = 2 + refinement_rounds
    if per_history != [expected_per_history] * histories:
        raise ValueError("query schedule multiplicity differs across histories")
    expected_rounds = [2 * histories] + [histories] * refinement_rounds
    if round_counts != expected_rounds:
        raise ValueError("query schedule round multiplicity is invalid")
    if canonical and (
        histories != CANONICAL_HISTORIES
        or len(rows) != CANONICAL_LABELS
        or refinement_rounds != REFINEMENT_ROUNDS
    ):
        raise ValueError("canonical query schedule dimensions are invalid")


def _unused_query(
    used: set[int],
    *,
    seed: int,
    round_index: int,
    history_id: int,
    domain: str,
) -> int:
    available = sorted(set(range(PUBLIC_QUERIES)) - used)
    if not available:
        raise RuntimeError("history exhausted the public query bank")
    material = (
        b"R12-ACW-UNUSED-QUERY-v1\x00"
        + domain.encode("ascii")
        + b"\x00"
        + seed.to_bytes(8, "big")
        + round_index.to_bytes(2, "big")
        + history_id.to_bytes(8, "big")
    )
    rank = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return available[rank % len(available)]


def select_refinement_round(
    packets: np.ndarray,
    final_states: np.ndarray,
    oracle_answers: np.ndarray,
    rows: list[dict],
    *,
    round_index: int,
    max_groups: int = MAX_GROUPS_PER_ROUND,
) -> tuple[list[dict], dict]:
    histories = len(packets)
    if packets.shape != (histories, 3) or packets.dtype != np.uint8:
        raise ValueError("pilot packets must be literal uint8 triples")
    if final_states.shape != (histories, 3):
        raise ValueError("final states do not match pilot packets")
    used = [set() for _ in range(histories)]
    for row in rows:
        used[row["history_id"]].add(row["query_id"])

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for history_id, packet in enumerate(packets):
        grouped.setdefault(tuple(int(value) for value in packet), []).append(history_id)
    collisions = []
    for packet, members in grouped.items():
        residual_classes = len(
            {tuple(int(value) for value in final_states[h]) for h in members}
        )
        if residual_classes > 1:
            collisions.append((residual_classes, packet, min(members), members))
    collisions.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned: dict[int, int] = {}
    candidate_evaluations = 0
    selected_witnesses = 0
    witness_exhausted = 0
    query_bank_unresolved = 0
    for residual_classes, packet, _, members in collisions[:max_groups]:
        del residual_classes, packet
        common_unused = set(range(PUBLIC_QUERIES))
        for history_id in members:
            common_unused.difference_update(used[history_id])
        if not common_unused:
            witness_exhausted += 1
            continue
        best_query = None
        best_distinct = -1
        for query_id in sorted(common_unused):
            distinct = len({int(oracle_answers[h, query_id]) for h in members})
            candidate_evaluations += 1
            if distinct > best_distinct:
                best_query = query_id
                best_distinct = distinct
        if best_distinct < 2:
            query_bank_unresolved += 1
            continue
        selected_witnesses += 1
        for history_id in members:
            assigned[history_id] = int(best_query)

    new_rows = []
    filler_histories = 0
    for history_id in range(histories):
        query_id = assigned.get(history_id)
        if query_id is None:
            filler_histories += 1
            query_id = _unused_query(
                used[history_id],
                seed=PILOT_SEED,
                round_index=round_index,
                history_id=history_id,
                domain="CGBR-FILLER",
            )
        if query_id in used[history_id]:
            raise RuntimeError("refinement selected an already-used query")
        new_rows.append(
            {
                "history_id": history_id,
                "query_id": query_id,
                "round": round_index,
            }
        )
    report = {
        "round": round_index,
        "packet_classes": len(grouped),
        "cross_residual_collision_groups": len(collisions),
        "groups_scanned": min(len(collisions), max_groups),
        "selected_witnesses": selected_witnesses,
        "witness_exhausted_groups": witness_exhausted,
        "query_bank_unresolved_groups": query_bank_unresolved,
        "candidate_evaluations": candidate_evaluations,
        "filler_histories": filler_histories,
    }
    return new_rows, report


def build_uniform_schedule(
    initial_rows: list[dict],
    histories: int,
    *,
    refinement_rounds: int,
) -> list[dict]:
    rows = list(initial_rows)
    used = [set() for _ in range(histories)]
    for row in rows:
        used[row["history_id"]].add(row["query_id"])
    for round_index in range(1, refinement_rounds + 1):
        for history_id in range(histories):
            query_id = _unused_query(
                used[history_id],
                seed=UNIFORM_SEED,
                round_index=round_index,
                history_id=history_id,
                domain="UNIFORM",
            )
            used[history_id].add(query_id)
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": query_id,
                    "round": round_index,
                }
            )
    return sorted(rows, key=lambda row: (row["history_id"], row["query_id"]))


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        encoded = canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hard_packets(
    model: torch.nn.Module,
    data: PublicTrainingData,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    packets = []
    with torch.no_grad():
        for start in range(0, data.histories, batch_size):
            history_ids = torch.arange(start, min(start + batch_size, data.histories))
            packets.append(
                recurrent_state(
                    model,
                    "acw",
                    data,
                    history_ids,
                    training=False,
                    literal_symbols=True,
                )
                .cpu()
                .numpy()
            )
    model.train()
    result = np.concatenate(packets, axis=0)
    if result.dtype != np.uint8:
        raise RuntimeError("pilot packet extraction lost literal uint8 persistence")
    return result


def run_pilot(
    root: Path,
    *,
    seed: int = PILOT_SEED,
    refinement_rounds: int = REFINEMENT_ROUNDS,
    updates_per_round: int = CANONICAL_UPDATES_PER_ROUND,
    final_updates: int = CANONICAL_FINAL_UPDATES,
    batch_size: int = CANONICAL_BATCH_SIZE,
    max_groups: int = MAX_GROUPS_PER_ROUND,
    canonical: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    replay_verification = None
    if canonical:
        canonical_values = (
            seed == PILOT_SEED,
            refinement_rounds == REFINEMENT_ROUNDS,
            updates_per_round == CANONICAL_UPDATES_PER_ROUND,
            final_updates == CANONICAL_FINAL_UPDATES,
            batch_size == CANONICAL_BATCH_SIZE,
            max_groups == MAX_GROUPS_PER_ROUND,
        )
        if not all(canonical_values):
            raise ValueError("canonical pilot hyperparameters are frozen internally")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        replay_verification = verify_registered_dataset(
            root,
            allowed_kinds={"pilot"},
        )
    data = load_public_training_data(root, reject_oracle=False)
    final_states, oracle_answers, manifest = load_oracle_truth(root)
    if canonical and manifest.get("seed_identity") != {
        "kind": "pilot",
        "seed": PILOT_SEED,
    }:
        raise ValueError("canonical curriculum pilot requires the frozen pilot dataset")
    if len(final_states) != data.histories:
        raise ValueError("public and oracle training history counts differ")
    rows = _initial_rows(data, oracle_answers)
    model = initialized_model_for_arm("acw", seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )
    round_reports = []
    total_updates = 0

    def optimize(count: int) -> list[float]:
        nonlocal total_updates
        curriculum = _rows_to_curriculum(rows, oracle_answers)
        losses = []
        for _ in range(count):
            selected = torch.randint(
                len(curriculum.history_ids),
                (batch_size,),
                generator=generator,
            )
            logits = forward_logits(
                model,
                "acw",
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
            total_updates += 1
        return losses

    for round_index in range(refinement_rounds + 1):
        losses = optimize(updates_per_round)
        report = {
            "round": round_index,
            "labels_before_update": len(rows),
            "optimizer_updates": updates_per_round,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }
        if round_index < refinement_rounds:
            packets = _hard_packets(model, data, batch_size=batch_size)
            report["packet_sha256"] = hashlib.sha256(packets.tobytes()).hexdigest()
            additions, selection = select_refinement_round(
                packets,
                final_states,
                oracle_answers,
                rows,
                round_index=round_index + 1,
                max_groups=max_groups,
            )
            rows.extend(additions)
            report["selection"] = selection
        round_reports.append(report)
    final_losses = optimize(final_updates)
    cgb_schedule = sorted(rows, key=lambda row: (row["history_id"], row["query_id"]))
    uniform_schedule = build_uniform_schedule(
        _initial_rows(data, oracle_answers),
        data.histories,
        refinement_rounds=refinement_rounds,
    )
    validate_query_schedule(
        cgb_schedule,
        data.histories,
        refinement_rounds=refinement_rounds,
        canonical=canonical,
    )
    validate_query_schedule(
        uniform_schedule,
        data.histories,
        refinement_rounds=refinement_rounds,
        canonical=canonical,
    )
    candidate_evaluations = sum(
        item.get("selection", {}).get("candidate_evaluations", 0)
        for item in round_reports
    )
    if canonical and candidate_evaluations > MAX_CANDIDATE_EVALUATIONS:
        raise RuntimeError("pilot exceeded its preregistered oracle-query cap")
    report = {
        "protocol": PILOT_PROTOCOL,
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "model_arm": "acw",
        "deterministic_algorithms": canonical,
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
        },
        "pilot_seed": seed,
        "uniform_seed": UNIFORM_SEED,
        "dataset_manifest_payload_sha256": manifest["payload_sha256"],
        "histories": data.histories,
        "refinement_rounds": refinement_rounds,
        "updates_per_round": updates_per_round,
        "final_updates": final_updates,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "max_groups_per_round": max_groups,
        "labels": len(cgb_schedule),
        "candidate_evaluations": candidate_evaluations,
        "rounds": round_reports,
        "final_loss_first": final_losses[0] if final_losses else None,
        "final_loss_last": final_losses[-1] if final_losses else None,
        "model_tensor_sha256": _tensor_state_sha256(model),
        "dataset_replay_verification": replay_verification,
        "claim_boundary": (
            "Non-scored curriculum pilot only. This is neither a scored architecture "
            "result nor evidence of reasoning."
        ),
    }
    return cgb_schedule, uniform_schedule, report


def _schedule_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _published_pilot_files(
    cgb_schedule: list[dict],
    uniform_schedule: list[dict],
    report: dict,
    *,
    identity: dict,
) -> tuple[dict[str, bytes], dict]:
    files = {
        "cgb_schedule.jsonl": _schedule_bytes(cgb_schedule),
        "uniform_schedule.jsonl": _schedule_bytes(uniform_schedule),
    }
    published = dict(report)
    published["scientific_identity"] = identity
    published["schedules"] = {
        name: {
            "bytes": len(payload),
            "rows": len(cgb_schedule)
            if name.startswith("cgb_")
            else len(uniform_schedule),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in files.items()
    }
    published["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(published)
    ).hexdigest()
    files["report.json"] = canonical_json_bytes(published) + b"\n"
    return files, published


def _parse_scontrol_fields(snapshot: str) -> dict[str, str]:
    fields = {}
    for token in shlex.split(snapshot):
        if "=" in token:
            name, value = token.split("=", 1)
            fields[name] = value
    return fields


def _slurm_snapshot(*, required: bool) -> dict | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        if required:
            raise RuntimeError("canonical pilot execution requires SLURM_JOB_ID")
        return None
    if not job_id.isdigit():
        raise RuntimeError("SLURM_JOB_ID must be numeric")
    command = ["scontrol", "show", "job", "-o", job_id]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if required:
            raise RuntimeError(
                "canonical pilot could not query its Slurm job"
            ) from error
        return None
    snapshot = result.stdout.strip()
    if not snapshot or f"JobId={job_id}" not in snapshot:
        raise RuntimeError("Slurm job snapshot does not bind the active job ID")
    fields = _parse_scontrol_fields(snapshot)
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    node_list = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if required and (
        fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or not cpus
        or not cpus.isdigit()
        or int(fields.get("NumCPUs", "0")) != int(cpus)
        or not node_list
        or fields.get("NodeList") != node_list
    ):
        raise RuntimeError("live Slurm allocation differs from the pilot environment")
    return {
        "command": command,
        "stdout": snapshot,
        "stdout_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "allocation": {
            "job_id": fields.get("JobId"),
            "job_state": fields.get("JobState"),
            "num_cpus": int(fields.get("NumCPUs", "0")),
            "node_list": fields.get("NodeList"),
        },
    }


def execute_pilot_replay(
    root: Path,
    out: Path,
    *,
    replay_id: str,
    canonical: bool,
    pilot_kwargs: dict | None = None,
    hold_fd: int | None = None,
) -> dict:
    """Run and publish one replay while measuring the process that did the work."""
    if replay_id not in {"a", "b"}:
        raise ValueError("pilot replay ID must be 'a' or 'b'")
    if canonical and hold_fd is None:
        raise RuntimeError("canonical pilot child requires a parent-held release pipe")
    if hold_fd is not None:
        os.fstat(hold_fd)
    started_time_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    identity = (
        scientific_identity(require_clean=True)
        if canonical
        else {
            "scientific_commit": "noncanonical-test",
            "scientific_path_sha256": {"noncanonical-test": "0" * 64},
        }
    )
    cgb, uniform, report = run_pilot(
        root,
        canonical=canonical,
        **(pilot_kwargs or {}),
    )
    finished_monotonic_ns = time.monotonic_ns()
    finished_time_ns = time.time_ns()
    files, report = _published_pilot_files(
        cgb,
        uniform,
        report,
        identity=identity,
    )
    execution = {
        "protocol": PILOT_EXECUTION_PROTOCOL,
        "replay_id": replay_id,
        "execution_nonce": secrets.token_hex(32),
        "process_id": os.getpid(),
        "hostname": socket.getfqdn(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "started_time_ns": started_time_ns,
        "finished_time_ns": finished_time_ns,
        "elapsed_wall_ns": finished_time_ns - started_time_ns,
        "elapsed_monotonic_ns": finished_monotonic_ns - started_monotonic_ns,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cpu_count": int(os.cpu_count() or 0),
        "slurm_cpus_per_task": (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK")
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_snapshot": _slurm_snapshot(required=canonical),
        "dataset_manifest_payload_sha256": report["dataset_manifest_payload_sha256"],
        "scientific_identity": identity,
        "report_sha256": hashlib.sha256(files["report.json"]).hexdigest(),
        "schedule_sha256": {
            name: record["sha256"] for name, record in report["schedules"].items()
        },
    }
    execution["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(execution)
    ).hexdigest()
    files["execution.json"] = canonical_json_bytes(execution) + b"\n"
    _write_pilot_output(out, files)
    if hold_fd is not None:
        try:
            if os.read(hold_fd, 1) != b"1":
                raise RuntimeError("pilot child received an invalid parent release")
        finally:
            os.close(hold_fd)
    return report


def _write_pilot_output(out: Path, files: dict[str, bytes]) -> None:
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        for name, payload in files.items():
            (partial / name).write_bytes(payload)
        for path in partial.iterdir():
            path.chmod(0o444)
        partial.replace(out)
        out.chmod(0o555)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def load_query_schedule(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed query-schedule row {line_number}"
                ) from error
            rows.append(row)
    return rows


def _load_hash_bound_json(path: Path, *, label: str) -> dict:
    record = json.loads(path.read_text())
    payload = dict(record)
    recorded = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError(f"{label} payload hash mismatch")
    return record


def _validate_replay_report(path: Path, *, canonical: bool) -> dict:
    path = path.resolve()
    report = _load_hash_bound_json(path, label="pilot report")
    if report.get("protocol") != PILOT_PROTOCOL:
        raise ValueError("wrong pilot report protocol")
    identity = report.get("scientific_identity")
    if not isinstance(identity, dict) or set(identity) != {
        "scientific_commit",
        "scientific_path_sha256",
    }:
        raise ValueError("pilot report lacks a scientific identity")
    if canonical and identity != scientific_identity(require_clean=True):
        raise ValueError(
            "pilot report scientific identity differs from the executing code"
        )
    schedules = report.get("schedules")
    if set(schedules or {}) != {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
        raise ValueError("pilot report schedule registry is incomplete")
    refinement_rounds = int(report.get("refinement_rounds", -1))
    for name, record in schedules.items():
        if set(record) != {"bytes", "rows", "sha256"}:
            raise ValueError("pilot schedule binding has the wrong schema")
        schedule_path = path.parent / name
        if (
            not schedule_path.is_file()
            or schedule_path.stat().st_size != record["bytes"]
            or file_sha256(schedule_path) != record["sha256"]
        ):
            raise ValueError("pilot schedule file differs from its report binding")
        rows = load_query_schedule(schedule_path)
        if len(rows) != record["rows"]:
            raise ValueError("pilot schedule row count differs from its report binding")
        validate_query_schedule(
            rows,
            int(report.get("histories", -1)),
            refinement_rounds=refinement_rounds,
            canonical=canonical,
        )
    if report.get("labels") != schedules["cgb_schedule.jsonl"]["rows"]:
        raise ValueError("pilot label count differs from its schedule")
    if canonical:
        expected = {
            "schedule_protocol": SCHEDULE_PROTOCOL,
            "model_arm": "acw",
            "deterministic_algorithms": True,
            "optimizer": {
                "kind": "AdamW",
                "learning_rate": 0.003,
                "weight_decay": 0.0001,
            },
            "pilot_seed": PILOT_SEED,
            "uniform_seed": UNIFORM_SEED,
            "histories": CANONICAL_HISTORIES,
            "refinement_rounds": REFINEMENT_ROUNDS,
            "updates_per_round": CANONICAL_UPDATES_PER_ROUND,
            "final_updates": CANONICAL_FINAL_UPDATES,
            "total_updates": CANONICAL_TOTAL_UPDATES,
            "batch_size": CANONICAL_BATCH_SIZE,
            "max_groups_per_round": MAX_GROUPS_PER_ROUND,
            "labels": CANONICAL_LABELS,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "pilot report differs from frozen canonical hyperparameters"
            )
        if int(report.get("candidate_evaluations", -1)) > MAX_CANDIDATE_EVALUATIONS:
            raise ValueError("pilot report exceeds the frozen oracle-query cap")
        replay = report.get("dataset_replay_verification")
        expected_keys = {
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
        if not isinstance(replay, dict) or set(replay) != expected_keys:
            raise ValueError("pilot report lacks complete data replay verification")
        if (
            replay["protocol"] != DATA_REPLAY_PROTOCOL
            or replay["seed_identity"] != {"kind": "pilot", "seed": PILOT_SEED}
            or replay["source_manifest_payload_sha256"]
            != report.get("dataset_manifest_payload_sha256")
            or replay["regenerated_manifest_payload_sha256"]
            != report.get("dataset_manifest_payload_sha256")
            or min(
                int(replay["arrays_verified"]),
                int(replay["public_arrays_verified"]),
                int(replay["oracle_arrays_verified"]),
            )
            <= 0
        ):
            raise ValueError("pilot report data replay verification is inconsistent")
    return report


def _validate_execution(
    root: Path,
    report: dict,
    *,
    replay_id: str,
    canonical: bool,
    require_live_scheduler: bool = False,
) -> dict:
    execution_path = root / "execution.json"
    execution = _load_hash_bound_json(execution_path, label="pilot execution")
    required = {
        "protocol",
        "replay_id",
        "execution_nonce",
        "process_id",
        "hostname",
        "python_executable",
        "python_version",
        "torch_version",
        "numpy_version",
        "started_time_ns",
        "finished_time_ns",
        "elapsed_wall_ns",
        "elapsed_monotonic_ns",
        "peak_rss_kib",
        "cpu_count",
        "slurm_cpus_per_task",
        "slurm_job_id",
        "slurm_snapshot",
        "dataset_manifest_payload_sha256",
        "scientific_identity",
        "report_sha256",
        "schedule_sha256",
        "payload_sha256",
    }
    nonce = execution.get("execution_nonce")
    if (
        set(execution) != required
        or execution.get("protocol") != PILOT_EXECUTION_PROTOCOL
    ):
        raise ValueError("pilot execution receipt has the wrong schema")
    if (
        execution.get("replay_id") != replay_id
        or not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or int(execution.get("process_id", 0)) <= 0
        or not isinstance(execution.get("hostname"), str)
        or not execution["hostname"]
        or not isinstance(execution.get("python_executable"), str)
        or not execution["python_executable"]
        or not isinstance(execution.get("python_version"), str)
        or not execution["python_version"]
        or not isinstance(execution.get("torch_version"), str)
        or not execution["torch_version"]
        or not isinstance(execution.get("numpy_version"), str)
        or not execution["numpy_version"]
        or int(execution.get("started_time_ns", 0)) <= 0
        or int(execution.get("finished_time_ns", 0)) <= 0
        or int(execution.get("finished_time_ns", 0))
        <= int(execution.get("started_time_ns", 0))
        or int(execution.get("elapsed_wall_ns", 0))
        != int(execution.get("finished_time_ns", 0))
        - int(execution.get("started_time_ns", 0))
        or int(execution.get("elapsed_monotonic_ns", 0)) <= 0
        or int(execution.get("peak_rss_kib", 0)) <= 0
        or int(execution.get("cpu_count", 0)) <= 0
    ):
        raise ValueError("pilot execution receipt is invalid")
    if canonical:
        job_id = execution.get("slurm_job_id")
        cpus = execution.get("slurm_cpus_per_task")
        snapshot = execution.get("slurm_snapshot")
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or not isinstance(cpus, int)
            or cpus <= 0
            or not isinstance(snapshot, dict)
            or set(snapshot)
            != {
                "command",
                "stdout",
                "stdout_sha256",
                "allocation",
            }
            or snapshot["command"] != ["scontrol", "show", "job", "-o", job_id]
            or f"JobId={job_id}" not in snapshot["stdout"]
            or hashlib.sha256(snapshot["stdout"].encode("utf-8")).hexdigest()
            != snapshot["stdout_sha256"]
            or not isinstance(snapshot["allocation"], dict)
            or snapshot["allocation"].get("job_id") != job_id
            or snapshot["allocation"].get("job_state") != "RUNNING"
            or snapshot["allocation"].get("num_cpus") != cpus
            or not isinstance(snapshot["allocation"].get("node_list"), str)
            or not snapshot["allocation"]["node_list"]
            or execution["elapsed_wall_ns"] <= 0
            or abs(
                int(execution["elapsed_wall_ns"])
                - int(execution["elapsed_monotonic_ns"])
            )
            > max(5_000_000_000, int(execution["elapsed_monotonic_ns"]) // 20)
        ):
            raise ValueError("canonical pilot lacks measured Slurm execution evidence")
        if require_live_scheduler:
            current = _slurm_snapshot(required=True)
            if (
                current is None
                or current["allocation"] != snapshot["allocation"]
                or os.environ.get("SLURM_JOB_ID") != job_id
                or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != cpus
                or execution["hostname"] != socket.getfqdn()
            ):
                raise ValueError(
                    "canonical pilot execution differs from the live Slurm allocation"
                )
    if (
        execution["dataset_manifest_payload_sha256"]
        != report["dataset_manifest_payload_sha256"]
        or execution["scientific_identity"] != report["scientific_identity"]
        or execution["report_sha256"] != file_sha256(root / "report.json")
        or execution["schedule_sha256"]
        != {name: record["sha256"] for name, record in report["schedules"].items()}
    ):
        raise ValueError("pilot execution receipt differs from replay output")
    return execution


def _validate_replay_output(
    root: Path,
    *,
    canonical: bool,
    replay_id: str,
    require_live_scheduler: bool = False,
) -> tuple[dict, dict]:
    root = root.resolve()
    expected_files = {
        "cgb_schedule.jsonl",
        "uniform_schedule.jsonl",
        "report.json",
        "execution.json",
    }
    if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
        raise ValueError("pilot replay output file registry is incomplete")
    report = _validate_replay_report(root / "report.json", canonical=canonical)
    execution = _validate_execution(
        root,
        report,
        replay_id=replay_id,
        canonical=canonical,
        require_live_scheduler=require_live_scheduler,
    )
    return report, execution


def _canonical_path(relative: str) -> Path:
    return (Path(__file__).resolve().parents[1] / relative).resolve()


def _recompute_pilot_files(
    dataset_root: Path,
    report: dict,
    *,
    canonical: bool,
) -> tuple[dict[str, bytes], dict]:
    replay_kwargs = {
        "seed": int(report["pilot_seed"]),
        "refinement_rounds": int(report["refinement_rounds"]),
        "updates_per_round": int(report["updates_per_round"]),
        "final_updates": int(report["final_updates"]),
        "batch_size": int(report["batch_size"]),
        "max_groups": int(report["max_groups_per_round"]),
    }
    cgb, uniform, recomputed_report = run_pilot(
        dataset_root,
        canonical=canonical,
        **replay_kwargs,
    )
    return _published_pilot_files(
        cgb,
        uniform,
        recomputed_report,
        identity=report["scientific_identity"],
    )


def _process_parent_pid(process_id: int) -> int:
    status = Path(f"/proc/{process_id}/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
    result = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(process_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _validate_live_child_processes(
    children: list[dict],
    executions: tuple[dict, dict],
    replay_roots: tuple[Path, Path],
) -> list[dict]:
    if len(children) != 2:
        raise RuntimeError("canonical pilot requires two parent-owned children")
    records = []
    for expected_id, child, execution, replay_root in zip(
        ("a", "b"),
        children,
        executions,
        replay_roots,
        strict=True,
    ):
        process = child.get("process")
        if (
            child.get("replay_id") != expected_id
            or not isinstance(process, subprocess.Popen)
            or process.poll() is not None
            or process.pid != execution["process_id"]
            or _process_parent_pid(process.pid) != os.getpid()
            or child.get("command") != process.args
            or int(child.get("started_time_ns", 0)) <= 0
            or int(child.get("ready_time_ns", 0)) < int(execution["finished_time_ns"])
            or child.get("execution_sha256")
            != file_sha256(replay_root / "execution.json")
            or execution["slurm_job_id"] != os.environ.get("SLURM_JOB_ID")
        ):
            raise RuntimeError(
                "canonical replay is not a live parent-observed child execution"
            )
        records.append(
            {
                "replay_id": expected_id,
                "command": list(child["command"]),
                "observed_process_id": process.pid,
                "observed_parent_process_id": os.getpid(),
                "started_time_ns": int(child["started_time_ns"]),
                "ready_time_ns": int(child["ready_time_ns"]),
                "execution_sha256": child["execution_sha256"],
            }
        )
    return records


def _release_child_processes(
    children: list[dict],
    records: list[dict],
) -> None:
    for child in children:
        release_fd = int(child["release_fd"])
        os.write(release_fd, b"1")
        os.close(release_fd)
        child["release_fd"] = -1
    for child, record in zip(children, records, strict=True):
        process = child["process"]
        stdout, _ = process.communicate(timeout=60)
        if process.returncode != 0:
            raise RuntimeError("canonical pilot child failed after parent release")
        record["finished_time_ns"] = time.time_ns()
        record["return_code"] = process.returncode
        record["stdout_sha256"] = hashlib.sha256(stdout.encode("utf-8")).hexdigest()


def _validate_historical_orchestration(
    orchestration: dict,
    executions: list[dict],
    replay_records: list[dict],
) -> None:
    if (
        set(orchestration or {})
        != {
            "protocol",
            "parent_process_id",
            "hostname",
            "slurm_snapshot",
            "children",
        }
        or orchestration.get("protocol") != PILOT_ORCHESTRATION_PROTOCOL
    ):
        raise ValueError("pilot orchestration record has the wrong schema")
    parent_process_id = int(orchestration.get("parent_process_id", 0))
    hostname = orchestration.get("hostname")
    snapshot = orchestration.get("slurm_snapshot")
    allocation = snapshot.get("allocation") if isinstance(snapshot, dict) else None
    if (
        parent_process_id <= 0
        or not isinstance(hostname, str)
        or not hostname
        or not isinstance(snapshot, dict)
        or set(snapshot) != {"command", "stdout", "stdout_sha256", "allocation"}
        or hashlib.sha256(snapshot["stdout"].encode("utf-8")).hexdigest()
        != snapshot["stdout_sha256"]
        or not isinstance(allocation, dict)
        or set(allocation) != {"job_id", "job_state", "num_cpus", "node_list"}
        or not isinstance(allocation["job_id"], str)
        or not allocation["job_id"].isdigit()
        or allocation["job_state"] != "RUNNING"
        or int(allocation["num_cpus"]) <= 0
        or not isinstance(allocation["node_list"], str)
        or not allocation["node_list"]
        or snapshot["command"]
        != ["scontrol", "show", "job", "-o", allocation["job_id"]]
        or f"JobId={allocation['job_id']}" not in snapshot["stdout"]
    ):
        raise ValueError("pilot orchestration identity is invalid")
    children = orchestration.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("pilot orchestration lacks two child records")
    expected_child_keys = {
        "replay_id",
        "command",
        "observed_process_id",
        "observed_parent_process_id",
        "started_time_ns",
        "ready_time_ns",
        "execution_sha256",
        "finished_time_ns",
        "return_code",
        "stdout_sha256",
    }
    for replay_id, child, execution, replay_record in zip(
        ("a", "b"),
        children,
        executions,
        replay_records,
        strict=True,
    ):
        command = child.get("command")
        if (
            set(child) != expected_child_keys
            or child.get("replay_id") != replay_id
            or not isinstance(command, list)
            or len(command) != 8
            or command[1:6]
            != [
                "-m",
                "pipeline.freeze_acw_curriculum",
                "pilot-replay-internal",
                "--replay-id",
                replay_id,
            ]
            or command[6] != "--hold-fd"
            or not str(command[7]).isdigit()
            or Path(command[0]).resolve()
            != Path(execution["python_executable"]).resolve()
            or child["observed_process_id"] != execution["process_id"]
            or child["observed_parent_process_id"] != parent_process_id
            or child["execution_sha256"] != replay_record.get("execution_sha256")
            or child["return_code"] != 0
            or not (
                int(child["started_time_ns"])
                <= int(execution["started_time_ns"])
                < int(execution["finished_time_ns"])
                <= int(child["ready_time_ns"])
                <= int(child["finished_time_ns"])
            )
            or execution["hostname"] != hostname
            or execution["slurm_snapshot"]["allocation"] != snapshot["allocation"]
            or not isinstance(child["stdout_sha256"], str)
            or len(child["stdout_sha256"]) != 64
        ):
            raise ValueError("pilot orchestration child binding is invalid")


def freeze_pilot_replays(
    first: Path,
    second: Path,
    out: Path,
    *,
    dataset_root: Path,
    canonical: bool = True,
    canonical_children: list[dict] | None = None,
) -> dict:
    first = first.resolve()
    second = second.resolve()
    out = out.resolve()
    dataset_root = dataset_root.resolve()
    if len({first, second, out}) != 3:
        raise ValueError("pilot replay and frozen output paths must be distinct")
    if canonical and canonical_children is None:
        raise RuntimeError(
            "canonical pilot freeze requires live parent-owned child processes"
        )
    if canonical and (
        dataset_root != _canonical_path(CANONICAL_PILOT_DATASET)
        or first != _canonical_path(CANONICAL_PILOT_REPLAY_A)
        or second != _canonical_path(CANONICAL_PILOT_REPLAY_B)
        or out != _canonical_path(CANONICAL_PILOT_OUTPUT)
    ):
        raise ValueError("canonical pilot replay paths are frozen")
    first_report, first_execution = _validate_replay_output(
        first,
        canonical=canonical,
        replay_id="a",
        require_live_scheduler=canonical,
    )
    second_report, second_execution = _validate_replay_output(
        second,
        canonical=canonical,
        replay_id="b",
        require_live_scheduler=canonical,
    )
    if first_execution["execution_nonce"] == second_execution["execution_nonce"]:
        raise ValueError("pilot replays do not have independent execution receipts")
    if canonical and (
        first_execution["process_id"] == second_execution["process_id"]
        or first_execution["slurm_job_id"] != second_execution["slurm_job_id"]
        or first_execution["hostname"] != second_execution["hostname"]
    ):
        raise ValueError("canonical pilot replays lack two child-process executions")
    orchestration_records = None
    if canonical:
        orchestration_records = _validate_live_child_processes(
            canonical_children or [],
            (first_execution, second_execution),
            (first, second),
        )
    common_names = ("report.json", "cgb_schedule.jsonl", "uniform_schedule.jsonl")
    for name in common_names:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise ValueError(f"pilot replays are not byte-identical: {name}")
    if first_report != second_report:
        raise AssertionError("byte-identical pilot reports parsed differently")
    executing_identity = None
    if canonical:
        executing_identity = scientific_identity(require_clean=True)
        if executing_identity != first_report["scientific_identity"]:
            raise ValueError("pilot replay identity changed before recomputation")

    expected_files, expected_report = _recompute_pilot_files(
        dataset_root,
        first_report,
        canonical=canonical,
    )
    for name in common_names:
        if (first / name).read_bytes() != expected_files[name]:
            raise ValueError(
                f"pilot replay differs from independent recomputation: {name}"
            )
    if first_report != expected_report:
        raise AssertionError(
            "independently reconstructed pilot report parsed differently"
        )
    if canonical and scientific_identity(require_clean=True) != executing_identity:
        raise RuntimeError("ACW scientific identity changed during recomputation")
    if canonical:
        second_observation = _validate_live_child_processes(
            canonical_children or [],
            (first_execution, second_execution),
            (first, second),
        )
        if second_observation != orchestration_records:
            raise RuntimeError("canonical child identity changed during recomputation")
        _release_child_processes(canonical_children or [], orchestration_records or [])
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        common_files = {}
        for name in common_names:
            destination = partial / name
            shutil.copyfile(first / name, destination)
            common_files[name] = {
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        comparison = {
            "protocol": PILOT_COMPARISON_PROTOCOL,
            "reports_byte_identical": True,
            "schedules_byte_identical": True,
            "independently_recomputed": True,
            "independent_recomputation_sha256": {
                name: hashlib.sha256(expected_files[name]).hexdigest()
                for name in common_names
            },
            "dataset_manifest_payload_sha256": first_report[
                "dataset_manifest_payload_sha256"
            ],
            "scientific_identity": first_report["scientific_identity"],
            "orchestration": (
                {
                    "protocol": PILOT_ORCHESTRATION_PROTOCOL,
                    "parent_process_id": os.getpid(),
                    "hostname": socket.getfqdn(),
                    "slurm_snapshot": _slurm_snapshot(required=True),
                    "children": orchestration_records,
                }
                if canonical
                else None
            ),
            "common_files": common_files,
            "replays": [
                {
                    "replay_id": "a",
                    "path": CANONICAL_PILOT_REPLAY_A if canonical else str(first),
                    "execution_sha256": file_sha256(first / "execution.json"),
                    "execution_payload_sha256": first_execution["payload_sha256"],
                },
                {
                    "replay_id": "b",
                    "path": CANONICAL_PILOT_REPLAY_B if canonical else str(second),
                    "execution_sha256": file_sha256(second / "execution.json"),
                    "execution_payload_sha256": second_execution["payload_sha256"],
                },
            ],
        }
        comparison["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(comparison)
        ).hexdigest()
        comparison_path = partial / "replay_comparison.json"
        comparison_path.write_bytes(canonical_json_bytes(comparison) + b"\n")
        for path in partial.iterdir():
            path.chmod(0o444)
        partial.replace(out)
        out.chmod(0o555)
        return comparison
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _launch_held_replay(replay_id: str) -> dict:
    read_fd, write_fd = os.pipe()
    command = [
        sys.executable,
        "-m",
        "pipeline.freeze_acw_curriculum",
        "pilot-replay-internal",
        "--replay-id",
        replay_id,
        "--hold-fd",
        str(read_fd),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    started_time_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            pass_fds=(read_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        os.close(read_fd)
    return {
        "replay_id": replay_id,
        "command": command,
        "process": process,
        "release_fd": write_fd,
        "started_time_ns": started_time_ns,
    }


def _wait_for_held_replay(child: dict, *, timeout_seconds: int = 82_800) -> None:
    replay_id = str(child["replay_id"])
    out = _canonical_path(
        CANONICAL_PILOT_REPLAY_A if replay_id == "a" else CANONICAL_PILOT_REPLAY_B
    )
    deadline = time.monotonic() + timeout_seconds
    process = child["process"]
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(
                f"canonical pilot child {replay_id} exited before custody check: "
                f"return_code={return_code} stdout_sha256="
                f"{hashlib.sha256(stdout.encode('utf-8')).hexdigest()}"
            )
        execution_path = out / "execution.json"
        if execution_path.is_file():
            child["ready_time_ns"] = time.time_ns()
            child["execution_sha256"] = file_sha256(execution_path)
            return
        time.sleep(0.25)
    raise TimeoutError(f"canonical pilot child {replay_id} did not become ready")


def _cleanup_held_replays(children: list[dict]) -> None:
    for child in children:
        release_fd = int(child.get("release_fd", -1))
        if release_fd >= 0:
            try:
                os.write(release_fd, b"1")
            except OSError:
                pass
            finally:
                os.close(release_fd)
                child["release_fd"] = -1
        process = child.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if (
            isinstance(process, subprocess.Popen)
            and process.stdout is not None
            and not process.stdout.closed
        ):
            process.stdout.close()


def run_canonical_pilot() -> dict:
    """Own canonical data generation, two child fits, recomputation, and freeze."""
    start_identity = scientific_identity(require_clean=True)
    dataset_root = _canonical_path(CANONICAL_PILOT_DATASET)
    first = _canonical_path(CANONICAL_PILOT_REPLAY_A)
    second = _canonical_path(CANONICAL_PILOT_REPLAY_B)
    out = _canonical_path(CANONICAL_PILOT_OUTPUT)
    for path in (dataset_root, first, second, out):
        if path.exists() or path.with_name(path.name + ".partial").exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    _regenerate_registered_dataset(
        dataset_root,
        development_seed_material(PILOT_SEED),
        {"kind": "pilot", "seed": PILOT_SEED},
    )
    verify_registered_dataset(dataset_root, allowed_kinds={"pilot"})
    children = []
    try:
        children = [_launch_held_replay(replay_id) for replay_id in ("a", "b")]
        for child in children:
            _wait_for_held_replay(child)
        comparison = freeze_pilot_replays(
            first,
            second,
            out,
            dataset_root=dataset_root,
            canonical=True,
            canonical_children=children,
        )
    finally:
        _cleanup_held_replays(children)
    if scientific_identity(require_clean=True) != start_identity:
        raise RuntimeError("ACW scientific identity changed during pilot execution")
    load_pilot_report(out / "report.json")
    return comparison


def load_pilot_report(path: Path) -> dict:
    path = path.resolve()
    if path != _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json":
        raise ValueError("canonical pilot report path is frozen")
    report = _validate_replay_report(path, canonical=True)
    comparison = _load_hash_bound_json(
        path.parent / "replay_comparison.json",
        label="pilot replay comparison",
    )
    if comparison.get("protocol") != PILOT_COMPARISON_PROTOCOL:
        raise ValueError("wrong pilot replay comparison protocol")
    if (
        comparison.get("reports_byte_identical") is not True
        or comparison.get("schedules_byte_identical") is not True
        or comparison.get("independently_recomputed") is not True
        or comparison.get("dataset_manifest_payload_sha256")
        != report["dataset_manifest_payload_sha256"]
        or comparison.get("scientific_identity") != report["scientific_identity"]
    ):
        raise ValueError("pilot replay comparison differs from the frozen report")
    common_files = comparison.get("common_files")
    if set(common_files or {}) != {
        "report.json",
        "cgb_schedule.jsonl",
        "uniform_schedule.jsonl",
    }:
        raise ValueError("pilot replay comparison common-file registry is incomplete")
    for name, record in common_files.items():
        target = path.parent / name
        if set(record) != {"bytes", "sha256"} or (
            target.stat().st_size != record["bytes"]
            or file_sha256(target) != record["sha256"]
        ):
            raise ValueError("frozen pilot file differs from replay comparison")
    recomputation_hashes = comparison.get("independent_recomputation_sha256")
    if set(recomputation_hashes or {}) != set(common_files):
        raise ValueError("pilot comparison lacks independent recomputation hashes")
    expected_files, expected_report = _recompute_pilot_files(
        _canonical_path(CANONICAL_PILOT_DATASET),
        report,
        canonical=True,
    )
    if scientific_identity(require_clean=True) != report["scientific_identity"]:
        raise RuntimeError("ACW scientific identity changed during report replay")
    for name in common_files:
        if (path.parent / name).read_bytes() != expected_files[
            name
        ] or recomputation_hashes[name] != hashlib.sha256(
            expected_files[name]
        ).hexdigest():
            raise ValueError(
                f"frozen pilot differs from fresh independent recomputation: {name}"
            )
    if report != expected_report:
        raise AssertionError("freshly recomputed pilot report parsed differently")
    replay_records = comparison.get("replays")
    if not isinstance(replay_records, list) or len(replay_records) != 2:
        raise ValueError("pilot replay comparison lacks two executions")
    historical_executions = []
    for expected_id, expected_path, record in zip(
        ("a", "b"),
        (CANONICAL_PILOT_REPLAY_A, CANONICAL_PILOT_REPLAY_B),
        replay_records,
        strict=True,
    ):
        if (
            set(record)
            != {
                "replay_id",
                "path",
                "execution_sha256",
                "execution_payload_sha256",
            }
            or record.get("replay_id") != expected_id
            or record.get("path") != expected_path
        ):
            raise ValueError("pilot replay comparison source registry differs")
        replay_root = _canonical_path(expected_path)
        replay_report, execution = _validate_replay_output(
            replay_root,
            canonical=True,
            replay_id=expected_id,
        )
        if replay_report != report or (
            file_sha256(replay_root / "execution.json")
            != record.get("execution_sha256")
            or execution["payload_sha256"] != record.get("execution_payload_sha256")
        ):
            raise ValueError("pilot replay source differs from frozen comparison")
        historical_executions.append(execution)
    _validate_historical_orchestration(
        comparison.get("orchestration"),
        historical_executions,
        replay_records,
    )
    return report


def _validate_scored_seed_identity(seed_identity: dict) -> None:
    kind = seed_identity.get("kind")
    if kind == "development":
        if seed_identity != {"kind": "development", "seed": seed_identity.get("seed")}:
            raise ValueError("scored development identity has the wrong schema")
        if int(seed_identity["seed"]) not in DEVELOPMENT_SEEDS:
            raise ValueError(
                "scored development identity is outside the frozen registry"
            )
        return
    if kind == "confirmation":
        if set(seed_identity) != {"kind", "index", "commitment"}:
            raise ValueError("scored confirmation identity has the wrong schema")
        index = int(seed_identity["index"])
        if not 0 <= index < len(CONFIRMATION_COMMITMENTS):
            raise ValueError(
                "scored confirmation identity is outside the frozen registry"
            )
        if seed_identity["commitment"] != CONFIRMATION_COMMITMENTS[index]:
            raise ValueError(
                "scored confirmation commitment is outside the frozen registry"
            )
        return
    raise ValueError(
        "canonical scored bundle may not use a pilot or unregistered domain"
    )


def _write_array(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": file_sha256(path),
    }


def _require_committed_pilot_anchor() -> None:
    raise RuntimeError(CANONICAL_BUNDLE_BLOCK)


def build_trainer_bundle(
    dataset_root: Path,
    schedule_path: Path,
    out: Path,
    *,
    canonical: bool = True,
    pilot_report_path: Path | None = None,
) -> dict:
    dataset_root = dataset_root.resolve()
    schedule_path = schedule_path.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if canonical:
        _require_committed_pilot_anchor()
    source_manifest = _load_manifest(dataset_root)
    data_replay_verification = None
    if canonical:
        _validate_scored_seed_identity(source_manifest.get("seed_identity", {}))
        if source_manifest["seed_identity"].get("kind") == "development":
            data_replay_verification = verify_registered_dataset(
                dataset_root,
                allowed_kinds={"development"},
            )
    data = load_public_training_data(dataset_root, reject_oracle=False)
    _, oracle_answers, loaded_manifest = load_oracle_truth(dataset_root)
    if loaded_manifest != source_manifest:
        raise RuntimeError("dataset manifest changed during bundle construction")
    pilot_report = None
    pilot_comparison = None
    schedule_kind = "noncanonical"
    if canonical:
        if pilot_report_path is None:
            raise ValueError(
                "canonical trainer bundle requires the frozen pilot report"
            )
        pilot_report_path = pilot_report_path.resolve()
        pilot_report = load_pilot_report(pilot_report_path)
        pilot_comparison = _load_hash_bound_json(
            pilot_report_path.parent / "replay_comparison.json",
            label="pilot replay comparison",
        )
        schedule_kind = schedule_path.name
        if schedule_kind not in {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
            raise ValueError("canonical schedule has an unregistered filename")
        if schedule_path != pilot_report_path.parent / schedule_kind:
            raise ValueError(
                "canonical schedule is not the pilot report's bound sibling"
            )
        schedule_record = pilot_report["schedules"][schedule_kind]
        if set(schedule_record) != {"bytes", "rows", "sha256"}:
            raise ValueError("pilot schedule binding has the wrong schema")
        if (
            schedule_path.stat().st_size != schedule_record["bytes"]
            or file_sha256(schedule_path) != schedule_record["sha256"]
            or schedule_record["rows"] != CANONICAL_LABELS
        ):
            raise ValueError("canonical schedule differs from the frozen pilot report")
    schedule = load_query_schedule(schedule_path)
    validate_query_schedule(
        schedule,
        data.histories,
        refinement_rounds=REFINEMENT_ROUNDS
        if canonical
        else max(row["round"] for row in schedule),
        canonical=canonical,
    )
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    arrays = {}
    try:
        round_zero = [row for row in schedule if row["round"] == 0]
        initial_queries = np.empty((data.histories, 2), dtype=np.int8)
        initial_answers = np.empty_like(initial_queries)
        grouped_initial = [[] for _ in range(data.histories)]
        for row in round_zero:
            grouped_initial[row["history_id"]].append(row["query_id"])
        for history_id, queries in enumerate(grouped_initial):
            queries = sorted(queries)
            if len(queries) != 2:
                raise ValueError("bundle schedule lacks two round-zero queries")
            initial_queries[history_id] = queries
            initial_answers[history_id] = oracle_answers[history_id, queries]

        for relative in PUBLIC_ARRAYS:
            destination = partial / relative
            if relative.endswith("initial_queries.npy"):
                arrays[relative] = _write_array(destination, initial_queries)
            elif relative.endswith("initial_answers.npy"):
                arrays[relative] = _write_array(destination, initial_answers)
            else:
                source = dataset_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                source_record = source_manifest["arrays"][relative]
                if file_sha256(destination) != source_record["sha256"]:
                    raise RuntimeError(f"bundle copy hash mismatch: {relative}")
                arrays[relative] = dict(source_record)

        curriculum_rows = [
            {
                "history_id": row["history_id"],
                "query_id": row["query_id"],
                "answer": int(oracle_answers[row["history_id"], row["query_id"]]),
                "round": row["round"],
            }
            for row in schedule
        ]
        curriculum_path = partial / "curriculum.jsonl"
        curriculum_path.write_bytes(
            b"".join(canonical_json_bytes(row) + b"\n" for row in curriculum_rows)
        )
        if curriculum_query_schedule_sha256(curriculum_path) != file_sha256(
            schedule_path
        ):
            raise RuntimeError(
                "curriculum-derived schedule differs from the selected pilot schedule"
            )
        files = {
            "curriculum.jsonl": {
                "bytes": curriculum_path.stat().st_size,
                "rows": len(curriculum_rows),
                "sha256": file_sha256(curriculum_path),
            }
        }
        pilot_artifacts = None
        if canonical:
            pilot_artifacts = {}
            pilot_sources = {
                "pilot/report.json": pilot_report_path,
                "pilot/replay_comparison.json": (
                    pilot_report_path.parent / "replay_comparison.json"
                ),
                "pilot/cgb_schedule.jsonl": (
                    pilot_report_path.parent / "cgb_schedule.jsonl"
                ),
                "pilot/uniform_schedule.jsonl": (
                    pilot_report_path.parent / "uniform_schedule.jsonl"
                ),
            }
            for relative, source in pilot_sources.items():
                destination = partial / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                pilot_artifacts[relative] = {
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(destination),
                }
        manifest = {
            "protocol": BUNDLE_PROTOCOL,
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "seed_identity": source_manifest["seed_identity"],
            "data_replay_verification": data_replay_verification,
            "query_schedule_sha256": file_sha256(schedule_path),
            "query_schedule_kind": schedule_kind,
            "pilot_report_payload_sha256": (
                pilot_report["payload_sha256"] if pilot_report is not None else None
            ),
            "pilot_report_sha256": (
                file_sha256(pilot_report_path)
                if pilot_report_path is not None
                else None
            ),
            "pilot_replay_comparison_payload_sha256": (
                pilot_comparison["payload_sha256"]
                if pilot_comparison is not None
                else None
            ),
            "pilot_replay_comparison_sha256": (
                file_sha256(pilot_report_path.parent / "replay_comparison.json")
                if pilot_report_path is not None
                else None
            ),
            "pilot_artifacts": pilot_artifacts,
            "arrays": arrays,
            "files": files,
            "oracle_paths_exported": 0,
        }
        manifest["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        (partial / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        if any(
            "oracle" in str(path.relative_to(partial)).lower()
            for path in partial.rglob("*")
        ):
            raise RuntimeError("trainer bundle contains an oracle-named path")
        for path in partial.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        partial.replace(out)
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        out.chmod(0o555)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pilot-run")
    replay = subparsers.add_parser("pilot-replay-internal", help=argparse.SUPPRESS)
    replay.add_argument("--replay-id", choices=("a", "b"), required=True)
    replay.add_argument("--hold-fd", type=int, required=True, help=argparse.SUPPRESS)
    subparsers.add_parser("verify-pilot")
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--dataset", type=Path, required=True)
    bundle.add_argument("--schedule", type=Path, required=True)
    bundle.add_argument("--pilot-report", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pilot-run":
        comparison = run_canonical_pilot()
        print(
            "[acw-pilot-run] byte_identical=1 independently_recomputed=1 "
            f"payload_sha256={comparison['payload_sha256']}"
        )
    elif args.command == "pilot-replay-internal":
        expected_out = (
            CANONICAL_PILOT_REPLAY_A
            if args.replay_id == "a"
            else CANONICAL_PILOT_REPLAY_B
        )
        report = execute_pilot_replay(
            _canonical_path(CANONICAL_PILOT_DATASET),
            _canonical_path(expected_out),
            replay_id=args.replay_id,
            canonical=True,
            hold_fd=args.hold_fd,
        )
        print(
            f"[acw-pilot-replay-{args.replay_id}] labels={report['labels']} "
            f"payload_sha256={report['payload_sha256']}"
        )
    elif args.command == "verify-pilot":
        report = load_pilot_report(
            _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
        )
        print(
            f"[acw-pilot-verify] complete=1 payload_sha256={report['payload_sha256']}"
        )
    else:
        manifest = build_trainer_bundle(
            args.dataset,
            args.schedule,
            args.out,
            canonical=True,
            pilot_report_path=args.pilot_report,
        )
        print(f"[acw-bundle] payload_sha256={manifest['payload_sha256']}")


if __name__ == "__main__":
    main()
