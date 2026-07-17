"""Frozen causal evaluator for ACW hidden-basis checkpoints.

Evaluation happens only after a checkpoint freezes.  The evaluator may access
oracle states and answer recodings, but never changes the writer, event coder,
updater, bridge, or original reader.  It reports scalar accuracy separately
from all-query state exactness and keeps causal interventions explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.acw_hidden_basis_training import (
    ACW_SCIENTIFIC_PATHS,
    PublicTrainingData,
    TRAINING_PROTOCOL,
    canonical_json_bytes,
    file_sha256,
    model_for_arm,
    recurrent_state,
)
from pipeline.addressed_categorical_workspace import (
    symbols_to_packet,
    trainable_parameters,
)


EVALUATION_PROTOCOL = "R12-ACW-CAUSAL-EVALUATION-v2"
FIELD_SIZE = 17
PUBLIC_QUERIES = 24
NEW_QUERIES = 8
EVALUATION_DEPTHS = (8, 16, 32, 64, 65)
LITERAL_ARMS = {"acw", "dense_categorical", "packet_token_transformer"}


def _load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != "R12-ACW-HIDDEN-BASIS-v3":
        raise ValueError("wrong ACW evaluation-domain manifest protocol")
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError("evaluation dataset manifest payload hash mismatch")
    return manifest


def _load_array(root: Path, manifest: dict, relative: str) -> np.ndarray:
    record = manifest.get("arrays", {}).get(relative)
    path = root / relative
    if not isinstance(record, dict) or not path.is_file():
        raise ValueError(f"evaluation manifest lacks {relative}")
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"evaluation array hash mismatch: {relative}")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"evaluation array schema mismatch: {relative}")
    return array


@dataclass(frozen=True)
class OracleSplit:
    data: PublicTrainingData
    final_states: torch.Tensor
    public_answers: torch.Tensor
    new_answers: torch.Tensor


def load_oracle_split(root: Path, prefix: str) -> OracleSplit:
    root = root.resolve()
    manifest = _load_manifest(root)
    event_features = _load_array(root, manifest, "public/event_features.npy")
    event_addresses = _load_array(root, manifest, "public/event_addresses.npy")
    source = _load_array(root, manifest, f"{prefix}/source_features.npy")
    event_ids = _load_array(root, manifest, f"{prefix}/event_ids.npy")
    lengths = _load_array(root, manifest, f"{prefix}/lengths.npy")
    final_states = _load_array(root, manifest, f"{prefix}/final_states.npy")
    public_answers = _load_array(root, manifest, f"{prefix}/public_answers.npy")
    new_answers = _load_array(root, manifest, f"{prefix}/new_answers.npy")
    histories = len(lengths)
    data = PublicTrainingData(
        root=root,
        manifest_payload_sha256=manifest["payload_sha256"],
        event_features=torch.from_numpy(event_features.copy()).float(),
        event_addresses=torch.from_numpy(event_addresses.copy()).long(),
        source_features=torch.from_numpy(source.copy()).float(),
        event_ids=torch.from_numpy(event_ids.copy()).long(),
        lengths=torch.from_numpy(lengths.copy()).long(),
        initial_queries=torch.empty((histories, 0), dtype=torch.long),
        initial_answers=torch.empty((histories, 0), dtype=torch.long),
        bound_curriculum_sha256=None,
        query_schedule_sha256=None,
        query_schedule_kind=None,
        pilot_report_payload_sha256=None,
        source_manifest_payload_sha256=manifest["payload_sha256"],
        seed_identity=dict(manifest["seed_identity"]),
    )
    return OracleSplit(
        data=data,
        final_states=torch.from_numpy(final_states.copy()).long(),
        public_answers=torch.from_numpy(public_answers.copy()).long(),
        new_answers=torch.from_numpy(new_answers.copy()).long(),
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, str, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("wrong ACW checkpoint protocol")
    checkpoint_arm = checkpoint.get("arm")
    model_arm = "acw" if checkpoint_arm == "direct_state_acw" else checkpoint_arm
    model = model_for_arm(model_arm)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, model_arm, checkpoint


def verify_scientific_identity(identity: dict | None, *, allow_unbound: bool) -> None:
    if identity is None:
        if allow_unbound:
            return
        raise ValueError("canonical ACW checkpoint lacks scientific Git identity")
    if set(identity) != {"scientific_commit", "scientific_path_sha256"}:
        raise ValueError("ACW scientific identity has the wrong schema")
    hashes = identity["scientific_path_sha256"]
    if set(hashes) != set(ACW_SCIENTIFIC_PATHS):
        raise ValueError("ACW scientific identity path set is incomplete")
    root = Path(__file__).resolve().parents[1]
    commit = identity["scientific_commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        check=True,
    )
    for relative in ACW_SCIENTIFIC_PATHS:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != hashes[relative]:
            raise ValueError(f"ACW scientific Git blob mismatch: {relative}")
        current = root / relative
        if not current.is_file() or file_sha256(current) != hashes[relative]:
            raise ValueError(
                f"executing ACW scientific file differs from checkpoint: {relative}"
            )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *ACW_SCIENTIFIC_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("executing ACW scientific paths are dirty")


def _state_logits(
    model: nn.Module,
    arm: str,
    state,
    queries: torch.Tensor,
) -> torch.Tensor:
    if arm == "answer_motor":
        return model(state[0], state[1], queries)
    if arm == "source_retained":
        return model.read(state[0], state[1], queries)
    if arm == "acw" and state.dtype == torch.uint8:
        return model.reader(model.workspace.packet_delta_symbols(state), queries)
    if (
        arm in {"dense_categorical", "packet_token_transformer"}
        and state.dtype == torch.uint8
    ):
        packet = symbols_to_packet(state, FIELD_SIZE, dtype=torch.float32)
        return model.read(packet, queries)
    return model.read(state, queries)


def _repeat_state(state, repeats: int):
    if isinstance(state, tuple):
        return tuple(value.repeat_interleave(repeats, dim=0) for value in state)
    return state.repeat_interleave(repeats, dim=0)


def predict_public_queries(
    model: nn.Module,
    arm: str,
    data: PublicTrainingData,
    *,
    batch_size: int = 256,
) -> tuple[torch.Tensor, object]:
    predictions = []
    persistent = []
    with torch.no_grad():
        for start in range(0, data.histories, batch_size):
            history_ids = torch.arange(start, min(start + batch_size, data.histories))
            state = recurrent_state(
                model,
                arm,
                data,
                history_ids,
                training=False,
                literal_symbols=arm in LITERAL_ARMS,
            )
            queries = torch.arange(PUBLIC_QUERIES).repeat(len(history_ids))
            logits = _state_logits(
                model,
                arm,
                _repeat_state(state, PUBLIC_QUERIES),
                queries,
            )
            predictions.append(
                logits.argmax(dim=-1).reshape(len(history_ids), PUBLIC_QUERIES)
            )
            persistent.append(state)
    if isinstance(persistent[0], tuple):
        combined = tuple(
            torch.cat([item[index] for item in persistent], dim=0)
            for index in range(len(persistent[0]))
        )
    else:
        combined = torch.cat(persistent, dim=0)
    return torch.cat(predictions, dim=0), combined


def accuracy_report(predictions: torch.Tensor, answers: torch.Tensor) -> dict:
    correct = predictions == answers
    return {
        "scalar_correct": int(correct.sum()),
        "scalar_total": correct.numel(),
        "scalar_accuracy": float(correct.float().mean()),
        "state_exact": int(correct.all(dim=1).sum()),
        "state_total": len(correct),
        "state_exactness": float(correct.all(dim=1).float().mean()),
    }


def acw_packet_interventions(
    model: nn.Module,
    packets: torch.Tensor,
    answers: torch.Tensor,
) -> dict:
    if packets.dtype != torch.uint8 or packets.ndim != 2 or packets.shape[1] != 3:
        raise ValueError("ACW intervention requires literal uint8 triples")
    donor_index = torch.roll(torch.arange(len(packets)), shifts=-1)
    donor_packets = packets[donor_index]
    donor_answers = answers[donor_index]
    queries = torch.arange(PUBLIC_QUERIES).repeat(len(packets))
    with torch.no_grad():
        donor_predictions = (
            _state_logits(
                model,
                "acw",
                donor_packets.repeat_interleave(PUBLIC_QUERIES, dim=0),
                queries,
            )
            .argmax(dim=-1)
            .reshape(len(packets), PUBLIC_QUERIES)
        )
    return {
        "donor_following": accuracy_report(donor_predictions, donor_answers),
        "shuffled_against_original": accuracy_report(donor_predictions, answers),
        "held_packet_source_swap_predictions_identical": True,
        "source_swap_basis": (
            "Structural: the frozen ACW reader accepts only packet and query; "
            "source bytes are absent from its callable interface."
        ),
        "donor_different_truth_fraction": float(
            (donor_answers != answers).any(dim=1).float().mean()
        ),
    }


def count_illegal_acw_writes(
    model: nn.Module,
    data: PublicTrainingData,
    *,
    batch_size: int = 256,
) -> dict:
    illegal = 0
    checked = 0
    with torch.no_grad():
        for start in range(0, data.histories, batch_size):
            history_ids = torch.arange(start, min(start + batch_size, data.histories))
            packet = model.workspace.encode_source_symbols(
                data.source_features[history_ids]
            )
            for step in range(data.event_ids.shape[1]):
                ids = data.event_ids[history_ids, step]
                active = step < data.lengths[history_ids]
                safe = ids.clamp_min(0)
                event = model.workspace.encode_event_symbols(data.event_features[safe])
                address = data.event_addresses[safe]
                updated = model.workspace.update_symbols(packet, event, address)
                for register in range(3):
                    unchanged = active & (address != register)
                    illegal += int(
                        (
                            updated[unchanged, register] != packet[unchanged, register]
                        ).sum()
                    )
                    checked += int(unchanged.sum())
                packet = torch.where(active.unsqueeze(1), updated, packet)
    return {"unaddressed_registers_checked": checked, "illegal_writes": illegal}


def _reader_representation(model: nn.Module, arm: str, state) -> torch.Tensor:
    if arm == "answer_motor":
        return torch.cat(state, dim=-1)
    if arm == "source_retained":
        return torch.cat(state, dim=-1)
    if arm == "acw" and state.dtype == torch.uint8:
        return model.workspace.packet_delta_symbols(state)
    if (
        arm in {"dense_categorical", "packet_token_transformer"}
        and state.dtype == torch.uint8
    ):
        packet = symbols_to_packet(state, FIELD_SIZE, dtype=torch.float32)
        return model.bridge(packet.flatten(1))
    return model.bridge(state)


class FrozenWriterNewReader(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.query_embedding = nn.Embedding(NEW_QUERIES, 16)
        self.network = nn.Sequential(
            nn.Linear(state_dim + 16, 64),
            nn.SiLU(),
            nn.Linear(64, FIELD_SIZE),
        )

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, self.query_embedding(query)), dim=-1))


def _frozen_representations(
    model: nn.Module,
    arm: str,
    data: PublicTrainingData,
    *,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, data.histories, batch_size):
            ids = torch.arange(start, min(start + batch_size, data.histories))
            state = recurrent_state(
                model,
                arm,
                data,
                ids,
                training=False,
                literal_symbols=arm in LITERAL_ARMS,
            )
            outputs.append(_reader_representation(model, arm, state))
    return torch.cat(outputs, dim=0).detach()


def evaluate_new_reader(
    model: nn.Module,
    arm: str,
    adaptation: OracleSplit,
    evaluations: dict[int, OracleSplit],
    *,
    seed: int,
    updates: int = 500,
    batch_size: int = 256,
) -> dict:
    train_state = _frozen_representations(
        model,
        arm,
        adaptation.data,
        batch_size=batch_size,
    )
    torch.manual_seed(seed)
    reader = FrozenWriterNewReader(train_state.shape[1])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.003, weight_decay=0.0001)
    states = train_state.repeat_interleave(NEW_QUERIES, dim=0)
    queries = torch.arange(NEW_QUERIES).repeat(len(train_state))
    answers = adaptation.new_answers.reshape(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses = []
    reader.train()
    for _ in range(updates):
        selected = torch.randint(len(answers), (batch_size,), generator=generator)
        loss = F.cross_entropy(
            reader(states[selected], queries[selected]), answers[selected]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    reader.eval()
    depth_reports = {}
    with torch.no_grad():
        for depth, split in evaluations.items():
            state = _frozen_representations(
                model, arm, split.data, batch_size=batch_size
            )
            query = torch.arange(NEW_QUERIES).repeat(len(state))
            predictions = (
                reader(
                    state.repeat_interleave(NEW_QUERIES, dim=0),
                    query,
                )
                .argmax(dim=-1)
                .reshape(len(state), NEW_QUERIES)
            )
            depth_reports[str(depth)] = accuracy_report(predictions, split.new_answers)
    return {
        "updates": updates,
        "state_dim": train_state.shape[1],
        "reader_parameters": trainable_parameters(reader),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "depths": depth_reports,
    }


def _apply_event(
    state: tuple[int, int, int], event: np.ndarray
) -> tuple[int, int, int]:
    destination, source, alpha, beta, gamma = (int(value) for value in event)
    output = list(state)
    output[destination] = (
        alpha * state[destination] + beta * state[source] + gamma
    ) % FIELD_SIZE
    return tuple(output)


def _answers_for_state(
    state: tuple[int, int, int],
    coefficients: np.ndarray,
    offsets: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    values = (
        coefficients.astype(np.int16) @ np.asarray(state, dtype=np.int16)
        + offsets.astype(np.int16)
    ) % FIELD_SIZE
    return np.asarray(
        [permutations[index, int(value)] for index, value in enumerate(values)],
        dtype=np.int64,
    )


def _word_pairs(state: tuple[int, int, int], events: np.ndarray):
    words = list(itertools.product(range(len(events)), repeat=2))
    outputs = {
        word: _apply_event(_apply_event(state, events[word[0]]), events[word[1]])
        for word in words
    }
    equivalent = None
    non_equivalent = None
    for first_index, first in enumerate(words):
        for second in words[first_index + 1 :]:
            if equivalent is None and outputs[first] == outputs[second]:
                equivalent = (first, second, outputs[first])
            if non_equivalent is None and outputs[first] != outputs[second]:
                non_equivalent = (
                    first,
                    second,
                    outputs[first],
                    outputs[second],
                )
            if equivalent is not None and non_equivalent is not None:
                break
        if equivalent is not None and non_equivalent is not None:
            break
    if equivalent is None:
        raise RuntimeError("event bank has no two-word equivalence witness")
    if non_equivalent is None:
        raise RuntimeError("event bank has no non-equivalent two-word witness")
    return equivalent, non_equivalent


def _append_words(data: PublicTrainingData, words: np.ndarray) -> PublicTrainingData:
    if words.shape != (data.histories, 2):
        raise ValueError("appended words must have shape [histories,2]")
    event_ids = torch.cat(
        (data.event_ids, torch.from_numpy(words.copy()).long()), dim=1
    )
    return PublicTrainingData(
        root=data.root,
        manifest_payload_sha256=data.manifest_payload_sha256,
        event_features=data.event_features,
        event_addresses=data.event_addresses,
        source_features=data.source_features,
        event_ids=event_ids,
        lengths=data.lengths + 2,
        initial_queries=data.initial_queries,
        initial_answers=data.initial_answers,
        bound_curriculum_sha256=None,
        query_schedule_sha256=None,
        query_schedule_kind=None,
        pilot_report_payload_sha256=None,
        source_manifest_payload_sha256=data.source_manifest_payload_sha256,
        seed_identity=dict(data.seed_identity),
    )


def evaluate_event_words(
    model: nn.Module,
    split: OracleSplit,
    root: Path,
    *,
    limit: int = 256,
) -> dict:
    manifest = _load_manifest(root)
    events = _load_array(root, manifest, "oracle/domain/events.npy")
    coefficients = _load_array(root, manifest, "oracle/domain/query_coefficients.npy")
    offsets = _load_array(root, manifest, "oracle/domain/query_offsets.npy")
    permutations = _load_array(root, manifest, "oracle/domain/query_permutations.npy")
    count = min(limit, split.data.histories)
    base = PublicTrainingData(
        **{
            **split.data.__dict__,
            "source_features": split.data.source_features[:count],
            "event_ids": split.data.event_ids[:count],
            "lengths": split.data.lengths[:count],
            "initial_queries": split.data.initial_queries[:count],
            "initial_answers": split.data.initial_answers[:count],
        }
    )
    equivalent_a = np.empty((count, 2), dtype=np.int16)
    equivalent_b = np.empty((count, 2), dtype=np.int16)
    non_a = np.empty((count, 2), dtype=np.int16)
    non_b = np.empty((count, 2), dtype=np.int16)
    equiv_answers = np.empty((count, PUBLIC_QUERIES), dtype=np.int64)
    non_a_answers = np.empty_like(equiv_answers)
    non_b_answers = np.empty_like(equiv_answers)
    for index in range(count):
        state = tuple(int(value) for value in split.final_states[index])
        equivalent, non_equivalent = _word_pairs(state, events)
        equivalent_a[index], equivalent_b[index] = equivalent[0], equivalent[1]
        non_a[index], non_b[index] = non_equivalent[0], non_equivalent[1]
        equiv_answers[index] = _answers_for_state(
            equivalent[2],
            coefficients,
            offsets,
            permutations,
        )
        non_a_answers[index] = _answers_for_state(
            non_equivalent[2],
            coefficients,
            offsets,
            permutations,
        )
        non_b_answers[index] = _answers_for_state(
            non_equivalent[3],
            coefficients,
            offsets,
            permutations,
        )
    pred_ea, _ = predict_public_queries(model, "acw", _append_words(base, equivalent_a))
    pred_eb, _ = predict_public_queries(model, "acw", _append_words(base, equivalent_b))
    pred_na, _ = predict_public_queries(model, "acw", _append_words(base, non_a))
    pred_nb, _ = predict_public_queries(model, "acw", _append_words(base, non_b))
    equiv_target = torch.from_numpy(equiv_answers)
    non_a_target = torch.from_numpy(non_a_answers)
    non_b_target = torch.from_numpy(non_b_answers)
    separators = non_a_target != non_b_target
    return {
        "histories": count,
        "equivalent_prediction_query_equivalence": float(
            (pred_ea == pred_eb).all(dim=1).float().mean()
        ),
        "equivalent_a": accuracy_report(pred_ea, equiv_target),
        "equivalent_b": accuracy_report(pred_eb, equiv_target),
        "non_equivalent_target_separator_rate": float(
            separators.any(dim=1).float().mean()
        ),
        "non_equivalent_prediction_separator_rate": float(
            ((pred_na != pred_nb) & separators).any(dim=1).float().mean()
        ),
        "non_equivalent_a": accuracy_report(pred_na, non_a_target),
        "non_equivalent_b": accuracy_report(pred_nb, non_b_target),
    }


def compiled_sparse_report(root: Path, depths: tuple[int, ...]) -> dict:
    root = root.resolve()
    manifest = _load_manifest(root)
    coefficients = _load_array(root, manifest, "oracle/domain/query_coefficients.npy")
    offsets = _load_array(root, manifest, "oracle/domain/query_offsets.npy")
    permutations = _load_array(root, manifest, "oracle/domain/query_permutations.npy")
    events = _load_array(root, manifest, "oracle/domain/events.npy")
    reports = {}
    total_events = 0
    total_queries = 0
    for depth in depths:
        split = load_oracle_split(root, f"oracle/evaluation/depth_{depth:03d}")
        source_states = _load_array(
            root,
            manifest,
            f"oracle/evaluation/depth_{depth:03d}/source_states.npy",
        )
        reconstructed = np.empty_like(source_states)
        for history_id in range(split.data.histories):
            state = tuple(int(value) for value in source_states[history_id])
            length = int(split.data.lengths[history_id])
            for event_id in split.data.event_ids[history_id, :length].tolist():
                state = _apply_event(state, events[int(event_id)])
            reconstructed[history_id] = np.asarray(state, dtype=source_states.dtype)
        predictions = np.stack(
            [
                _answers_for_state(
                    tuple(int(value) for value in state),
                    coefficients,
                    offsets,
                    permutations,
                )
                for state in reconstructed
            ]
        )
        answer_report = accuracy_report(
            torch.from_numpy(predictions),
            split.public_answers,
        )
        state_correct = np.all(reconstructed == split.final_states.numpy(), axis=1)
        answer_report["transition_state_exact"] = int(state_correct.sum())
        answer_report["transition_state_total"] = len(state_correct)
        answer_report["transition_state_exactness"] = float(state_correct.mean())
        reports[str(depth)] = answer_report
        total_events += int(split.data.lengths.sum())
        total_queries += split.data.histories * PUBLIC_QUERIES
    return {
        "depths": reports,
        "external_event_updates": total_events,
        "event_arithmetic": {
            "multiplications": 2 * total_events,
            "additions": 2 * total_events,
            "modulo": total_events,
        },
        "external_query_reads": total_queries,
        "query_arithmetic": {
            "multiplications": 3 * total_queries,
            "additions": 3 * total_queries,
            "modulo": total_queries,
            "permutation_lookups": total_queries,
        },
        "resource_ledger": {
            "trainable_parameters": 0,
            "persistent_state_bytes": 3,
            "event_table_bytes": int(events.nbytes),
            "query_table_bytes": int(
                coefficients.nbytes + offsets.nbytes + permutations.nbytes
            ),
            "runtime": "NumPy/Python exact F_17 replay",
        },
        "claim_boundary": "Known exact compilation; not neural learnability evidence.",
    }


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
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


def evaluate_label_efficiency(
    checkpoint: dict,
    arm: str,
    split: OracleSplit,
    *,
    batch_size: int,
) -> list[dict]:
    metadata = checkpoint.get("training_report", {}).get("label_efficiency")
    states = checkpoint.get("label_efficiency_models")
    expected_labels = [8192 + 4096 * round_index for round_index in range(13)]
    if not isinstance(metadata, list) or not isinstance(states, list):
        raise ValueError("canonical checkpoint lacks label-efficiency snapshots")
    if len(metadata) != 13 or len(states) != 13:
        raise ValueError(
            "label-efficiency snapshot count differs from the frozen protocol"
        )
    if _state_dict_sha256(states[-1]) != _state_dict_sha256(checkpoint["model"]):
        raise ValueError(
            "final label-efficiency snapshot differs from the primary model"
        )
    reports = []
    for round_index, (record, state) in enumerate(zip(metadata, states, strict=True)):
        if set(record) != {
            "round",
            "labels",
            "optimizer_updates",
            "model_tensor_sha256",
        }:
            raise ValueError("label-efficiency metadata has the wrong schema")
        if (
            record["round"] != round_index
            or record["labels"] != expected_labels[round_index]
        ):
            raise ValueError(
                "label-efficiency checkpoint is at an unregistered label count"
            )
        if _state_dict_sha256(state) != record["model_tensor_sha256"]:
            raise ValueError("label-efficiency model hash mismatch")
        model = model_for_arm(arm)
        model.load_state_dict(state, strict=True)
        model.eval()
        predictions, _ = predict_public_queries(
            model,
            arm,
            split.data,
            batch_size=batch_size,
        )
        reports.append(
            {
                **record,
                "depth_64": accuracy_report(predictions, split.public_answers),
            }
        )
    return reports


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    depths: tuple[int, ...] = EVALUATION_DEPTHS,
    new_reader_updates: int = 500,
    batch_size: int = 256,
    event_word_limit: int = 256,
    allow_unbound: bool = False,
) -> dict:
    model, arm, checkpoint = load_checkpoint(checkpoint_path)
    verify_scientific_identity(
        checkpoint.get("scientific_identity"),
        allow_unbound=allow_unbound,
    )
    dataset_root = dataset_root.resolve()
    manifest = _load_manifest(dataset_root)
    if checkpoint.get("source_manifest_payload_sha256") != manifest["payload_sha256"]:
        raise ValueError("checkpoint and evaluation domain manifest do not match")
    evaluations = {
        depth: load_oracle_split(dataset_root, f"oracle/evaluation/depth_{depth:03d}")
        for depth in depths
    }
    depth_reports = {}
    persistent_by_depth = {}
    for depth, split in evaluations.items():
        predictions, state = predict_public_queries(
            model,
            arm,
            split.data,
            batch_size=batch_size,
        )
        depth_reports[str(depth)] = accuracy_report(predictions, split.public_answers)
        persistent_by_depth[depth] = state
    adaptation = load_oracle_split(dataset_root, "oracle/adaptation")
    report = {
        "protocol": EVALUATION_PROTOCOL,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_arm": checkpoint.get("arm"),
        "model_arm": arm,
        "parameters": trainable_parameters(model),
        "dataset_manifest_payload_sha256": manifest["payload_sha256"],
        "seed_identity": manifest["seed_identity"],
        "optimizer_seed": checkpoint.get("seed"),
        "query_schedule_kind": checkpoint.get("query_schedule_kind"),
        "pilot_report_payload_sha256": checkpoint.get("pilot_report_payload_sha256"),
        "training_evidence": {
            "trainer_bundle_manifest_payload_sha256": checkpoint.get(
                "dataset_manifest_payload_sha256"
            ),
            "curriculum_sha256": checkpoint.get("curriculum_sha256"),
            "query_schedule_sha256": checkpoint.get("query_schedule_sha256"),
            "updates": checkpoint.get("training_report", {}).get("updates"),
            "labels": checkpoint.get("training_report", {}).get("labels"),
            "resource_ledger": checkpoint.get("training_report", {}).get(
                "resource_ledger"
            ),
            "resource_measurements": checkpoint.get("training_report", {}).get(
                "resource_measurements"
            ),
            "canonical_runtime_sha256": checkpoint.get("training_report", {}).get(
                "canonical_runtime_sha256"
            ),
            "development_plan_sha256": checkpoint.get("training_report", {}).get(
                "development_plan_sha256"
            ),
            "execution_receipt": checkpoint.get("training_report", {}).get(
                "execution_receipt"
            ),
        },
        "scientific_identity": checkpoint.get("scientific_identity"),
        "public_depths": depth_reports,
        "new_reader": evaluate_new_reader(
            model,
            arm,
            adaptation,
            evaluations,
            seed=2026071699,
            updates=new_reader_updates,
            batch_size=batch_size,
        ),
        "claim_boundary": (
            "Frozen synthetic state-transport evaluation only; no language, autonomous "
            "controller, novelty, or reasoning claim."
        ),
    }
    if 64 in evaluations:
        if checkpoint.get("label_efficiency_models") is not None:
            report["label_efficiency"] = evaluate_label_efficiency(
                checkpoint,
                arm,
                evaluations[64],
                batch_size=batch_size,
            )
        elif not allow_unbound and checkpoint.get("arm") != "direct_state_acw":
            raise ValueError(
                "canonical scored checkpoint lacks label-efficiency snapshots"
            )
    report["compiled_sparse_control"] = compiled_sparse_report(dataset_root, depths)
    if arm == "acw":
        intervention_depth = 64 if 64 in evaluations else max(depths)
        split = evaluations[intervention_depth]
        report["packet_interventions"] = acw_packet_interventions(
            model,
            persistent_by_depth[intervention_depth],
            split.public_answers,
        )
        report["write_legality"] = count_illegal_acw_writes(
            model,
            split.data,
            batch_size=batch_size,
        )
        report["event_words"] = evaluate_event_words(
            model,
            split,
            dataset_root,
            limit=event_word_limit,
        )
    report["payload_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def write_report(report: dict, path: Path) -> None:
    payload = dict(report)
    recorded = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError("evaluation report payload hash is stale")
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(report) + b"\n")
    temporary.replace(path)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.dataset)
    write_report(report, args.out)
    print(
        f"[acw-eval] arm={report['checkpoint_arm']} "
        f"payload_sha256={report['payload_sha256']}"
    )


if __name__ == "__main__":
    main()
