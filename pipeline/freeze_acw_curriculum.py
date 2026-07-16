"""Freeze the ACW pilot query schedule and public-only scored trainer bundles.

The pilot is the only learned model whose hard packet may be inspected by the
development oracle.  Scored arms receive only the resulting query schedule and
the answer labels instantiated for their own domain.  Canonical pilot execution
claims an immutable one-use ledger before optimization starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.acw_hidden_basis_training import (
    PUBLIC_ARRAYS,
    Curriculum,
    PublicTrainingData,
    canonical_json_bytes,
    file_sha256,
    forward_logits,
    initialized_model_for_arm,
    load_public_training_data,
    recurrent_state,
    scientific_identity,
)


PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v1"
SCHEDULE_PROTOCOL = "R12-ACW-QUERY-SCHEDULE-v1"
BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v1"
PILOT_SEED = 2026071600
UNIFORM_SEED = 2026071604
PUBLIC_QUERIES = 24
REFINEMENT_ROUNDS = 12
MAX_GROUPS_PER_ROUND = 512
MAX_CANDIDATE_EVALUATIONS = 147_456
CANONICAL_HISTORIES = 4096
CANONICAL_LABELS = 57_344


def _load_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != recorded:
        raise ValueError("dataset manifest payload hash mismatch")
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
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
        raise ValueError(f"array schema mismatch: {relative}")
    return array


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
            rows.append({
                "history_id": history_id,
                "query_id": int(query_id),
                "round": 0,
            })
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
    rows: list[dict], histories: int, *, refinement_rounds: int, canonical: bool,
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
    used: set[int], *, seed: int, round_index: int, history_id: int, domain: str,
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
        residual_classes = len({tuple(int(value) for value in final_states[h]) for h in members})
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
        new_rows.append({
            "history_id": history_id,
            "query_id": query_id,
            "round": round_index,
        })
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
    initial_rows: list[dict], histories: int, *, refinement_rounds: int,
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
            rows.append({
                "history_id": history_id,
                "query_id": query_id,
                "round": round_index,
            })
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
    model: torch.nn.Module, data: PublicTrainingData, *, batch_size: int,
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
                ).cpu().numpy()
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
    updates_per_round: int = 200,
    final_updates: int = 800,
    batch_size: int = 256,
    max_groups: int = MAX_GROUPS_PER_ROUND,
    canonical: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    data = load_public_training_data(root, reject_oracle=False)
    final_states, oracle_answers, manifest = load_oracle_truth(root)
    if canonical and manifest.get("seed_identity") != {"kind": "pilot", "seed": PILOT_SEED}:
        raise ValueError("canonical curriculum pilot requires the frozen pilot dataset")
    if len(final_states) != data.histories:
        raise ValueError("public and oracle training history counts differ")
    rows = _initial_rows(data, oracle_answers)
    model = initialized_model_for_arm("acw", seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=0.0001,
    )
    round_reports = []
    total_updates = 0

    def optimize(count: int) -> list[float]:
        nonlocal total_updates
        curriculum = _rows_to_curriculum(rows, oracle_answers)
        losses = []
        for _ in range(count):
            selected = torch.randint(
                len(curriculum.history_ids), (batch_size,), generator=generator,
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
        cgb_schedule, data.histories,
        refinement_rounds=refinement_rounds,
        canonical=canonical,
    )
    validate_query_schedule(
        uniform_schedule, data.histories,
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
        "pilot_seed": seed,
        "uniform_seed": UNIFORM_SEED,
        "dataset_manifest_payload_sha256": manifest["payload_sha256"],
        "histories": data.histories,
        "refinement_rounds": refinement_rounds,
        "updates_per_round": updates_per_round,
        "final_updates": final_updates,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "labels": len(cgb_schedule),
        "candidate_evaluations": candidate_evaluations,
        "rounds": round_reports,
        "final_loss_first": final_losses[0] if final_losses else None,
        "final_loss_last": final_losses[-1] if final_losses else None,
        "model_tensor_sha256": _tensor_state_sha256(model),
        "claim_boundary": (
            "Non-scored curriculum pilot only. This is neither a scored architecture "
            "result nor evidence of reasoning."
        ),
    }
    return cgb_schedule, uniform_schedule, report


def _claim_execution_ledger(path: Path, root: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(root.resolve())
    record = {
        "protocol": "R12-ACW-PILOT-EXECUTION-CLAIM-v1",
        "dataset_manifest_payload_sha256": manifest["payload_sha256"],
        "scientific_identity": scientific_identity(require_clean=True),
    }
    data = canonical_json_bytes(record) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _schedule_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def publish_pilot(
    out: Path,
    cgb_schedule: list[dict],
    uniform_schedule: list[dict],
    report: dict,
    *,
    identity: dict,
) -> dict:
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        cgb_path = partial / "cgb_schedule.jsonl"
        uniform_path = partial / "uniform_schedule.jsonl"
        cgb_path.write_bytes(_schedule_bytes(cgb_schedule))
        uniform_path.write_bytes(_schedule_bytes(uniform_schedule))
        report = dict(report)
        report["scientific_identity"] = identity
        report["schedules"] = {
            "cgb_schedule.jsonl": {
                "bytes": cgb_path.stat().st_size,
                "rows": len(cgb_schedule),
                "sha256": file_sha256(cgb_path),
            },
            "uniform_schedule.jsonl": {
                "bytes": uniform_path.stat().st_size,
                "rows": len(uniform_schedule),
                "sha256": file_sha256(uniform_path),
            },
        }
        report["payload_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
        (partial / "report.json").write_bytes(canonical_json_bytes(report) + b"\n")
        for path in partial.iterdir():
            path.chmod(0o444)
        partial.replace(out)
        out.chmod(0o555)
        return report
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
                raise ValueError(f"malformed query-schedule row {line_number}") from error
            rows.append(row)
    return rows


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


def build_trainer_bundle(
    dataset_root: Path,
    schedule_path: Path,
    out: Path,
    *,
    canonical: bool = True,
) -> dict:
    dataset_root = dataset_root.resolve()
    schedule_path = schedule_path.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    data = load_public_training_data(dataset_root, reject_oracle=False)
    _, oracle_answers, source_manifest = load_oracle_truth(dataset_root)
    schedule = load_query_schedule(schedule_path)
    validate_query_schedule(
        schedule,
        data.histories,
        refinement_rounds=REFINEMENT_ROUNDS if canonical else max(row["round"] for row in schedule),
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
        files = {
            "curriculum.jsonl": {
                "bytes": curriculum_path.stat().st_size,
                "rows": len(curriculum_rows),
                "sha256": file_sha256(curriculum_path),
            }
        }
        manifest = {
            "protocol": BUNDLE_PROTOCOL,
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "seed_identity": source_manifest["seed_identity"],
            "query_schedule_sha256": file_sha256(schedule_path),
            "arrays": arrays,
            "files": files,
            "oracle_paths_exported": 0,
        }
        manifest["payload_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        (partial / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        if any("oracle" in str(path.relative_to(partial)).lower() for path in partial.rglob("*")):
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
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--dataset", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.add_argument("--execution-ledger", type=Path, required=True)
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--dataset", type=Path, required=True)
    bundle.add_argument("--schedule", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pilot":
        _claim_execution_ledger(args.execution_ledger, args.dataset)
        cgb, uniform, report = run_pilot(args.dataset, canonical=True)
        report = publish_pilot(
            args.out,
            cgb,
            uniform,
            report,
            identity=scientific_identity(require_clean=True),
        )
        print(
            f"[acw-pilot] labels={report['labels']} "
            f"payload_sha256={report['payload_sha256']}"
        )
    else:
        manifest = build_trainer_bundle(
            args.dataset, args.schedule, args.out, canonical=True,
        )
        print(f"[acw-bundle] payload_sha256={manifest['payload_sha256']}")


if __name__ == "__main__":
    main()
