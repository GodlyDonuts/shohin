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
import subprocess
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
TRAINING_PROTOCOL = "R12-ACW-TRAINER-v1"
STATE_AUXILIARY_WEIGHT = 4.0
ACW_SCIENTIFIC_PATHS = (
    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md",
    "pipeline/addressed_categorical_workspace.py",
    "pipeline/audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/generate_acw_hidden_basis.py",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/freeze_acw_curriculum.py",
    "pipeline/evaluate_acw_hidden_basis.py",
    "pipeline/test_addressed_categorical_workspace.py",
    "pipeline/test_audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/test_generate_acw_hidden_basis.py",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_freeze_acw_curriculum.py",
    "pipeline/test_evaluate_acw_hidden_basis.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def scientific_identity(*, require_clean: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *ACW_SCIENTIFIC_PATHS],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("ACW scientific paths are not clean in Git")
    hashes = {}
    for relative in ACW_SCIENTIFIC_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
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
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
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
    source_manifest_payload_sha256: str
    seed_identity: dict

    @property
    def histories(self) -> int:
        return int(self.lengths.shape[0])


def load_public_training_data(
    root: Path, *, reject_oracle: bool = True,
) -> PublicTrainingData:
    root = root.resolve()
    if reject_oracle and (root / "oracle").exists():
        raise RuntimeError("scored trainer bundle exposes forbidden oracle files")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
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
            manifest.get("files", {}).get("curriculum.jsonl", {}).get("sha256")
        ),
        source_manifest_payload_sha256=manifest.get(
            "source_manifest_payload_sha256", recorded_payload,
        ),
        seed_identity=dict(manifest.get("seed_identity", {})),
    )


def expected_optimizer_seed(seed_identity: dict) -> int:
    kind = seed_identity.get("kind")
    if kind in {"development", "pilot"}:
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("public optimizer seed identity has the wrong schema")
        return int(seed_identity["seed"])
    if kind == "confirmation":
        if set(seed_identity) != {"kind", "index", "commitment"}:
            raise ValueError("confirmation optimizer seed identity has the wrong schema")
        material = (
            b"R12-ACW-OPT-v1\x00" + str(seed_identity["commitment"]).encode("ascii")
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
        if any(len(field) != count for field in (self.query_ids, self.answers, self.rounds)):
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
        history_ids=torch.tensor([row["history_id"] for row in records], dtype=torch.long),
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
            3 * np.log2(17), 3, 3 * 17 * 4, 7 * 24 * 4, "uint8",
        ),
        "answer_motor": (192 * 32.0, 768, 768, 0, "float32"),
        "source_retained": (224 * 32.0, 896, 896, 0, "float32"),
    }
    semantic_bits, persistent_bytes, training_bytes, transient_bytes, dtype = ledgers[arm]
    return {
        "trainable_parameters": trainable_parameters(model),
        "semantic_state_bits": float(semantic_bits),
        "persistent_evaluation_bytes": persistent_bytes,
        "persistent_evaluation_dtype": dtype,
        "persistent_training_state_bytes": training_bytes,
        "declared_transient_token_bytes": transient_bytes,
        "parameter_matched_primary": arm != "source_retained",
    }


def profile_answer_training_step_flops(
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
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True,
    ) as profiler:
        logits = forward_logits(
            profile_model,
            arm,
            data,
            curriculum.history_ids[selected],
            curriculum.query_ids[selected],
            training=True,
        )
        loss = F.cross_entropy(logits.float(), curriculum.answers[selected])
        loss.backward()
    supported_flops = sum(int(event.flops or 0) for event in profiler.key_averages())
    return {
        "scope": "one forward+backward answer-loss step; operator-reported FLOPs only",
        "batch_size": count,
        "active_events": int(data.lengths[curriculum.history_ids[selected]].sum()),
        "supported_flops": supported_flops,
        "unsupported_ops_are_not_imputed": True,
    }


def _history_events(
    data: PublicTrainingData, history_ids: torch.Tensor, step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event_id = data.event_ids[history_ids, step]
    active = step < data.lengths[history_ids]
    safe_id = event_id.clamp_min(0)
    hidden = data.event_features[safe_id]
    address = data.event_addresses[safe_id]
    return hidden, address, active


def _mean_event_features(
    data: PublicTrainingData, history_ids: torch.Tensor,
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
                state, event, address, straight_through=training,
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
                    state, 17, dtype=hidden.dtype, device=hidden.device,
                )
                event = symbols_to_packet(
                    packet_to_symbols(event), 17, dtype=hidden.dtype, device=hidden.device,
                )
                updated = model.update(
                    float_state, event, address, straight_through=False,
                )
                updated = packet_to_symbols(updated)
                state = torch.where(active.unsqueeze(1), updated, state)
            else:
                updated = model.update(
                    state, event, address, straight_through=training,
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

    def add_state_loss(observed: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> None:
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
            packet, event, address, straight_through=True,
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
    started = time.perf_counter()
    if data.source_manifest_payload_sha256 != truth.source_manifest_payload_sha256:
        raise ValueError("public bundle and direct-state oracle are from different domains")
    curriculum.validate(data.histories, canonical=canonical)
    generator = set_determinism(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
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
                logits.float(), curriculum.answers[selected],
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
        "flop_profile": {
            "scope": "direct-state diagnostic excluded from equal-label compute comparison",
            "supported_flops": None,
            "unsupported_ops_are_not_imputed": True,
        },
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
    started = time.perf_counter()
    curriculum.validate(data.histories, canonical=canonical)
    flop_profile = (
        profile_answer_training_step_flops(
            model, arm, data, curriculum, batch_size=batch_size,
        )
        if canonical
        else None
    )
    generator = set_determinism(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    losses = []
    updates = 0

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
    run_updates(torch.arange(len(curriculum.history_ids)), final_updates)
    return {
        "updates": updates,
        "labels": len(curriculum.history_ids),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "wall_seconds": time.perf_counter() - started,
        "resource_ledger": arm_resource_ledger(arm, model),
        "flop_profile": flop_profile,
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
    payload = {
        "protocol": TRAINING_PROTOCOL,
        "arm": arm,
        "seed": seed,
        "dataset_manifest_payload_sha256": data.manifest_payload_sha256,
        "source_manifest_payload_sha256": data.source_manifest_payload_sha256,
        "curriculum_sha256": curriculum_sha256,
        "parameters": trainable_parameters(model),
        "training_report": training_report,
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
        raise ValueError("optimizer seed does not match the trainer-bundle domain identity")
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
            model, data, truth, curriculum, seed=args.seed, canonical=True,
        )
    else:
        if args.oracle_dataset is not None:
            raise ValueError("scored arms may not receive --oracle-dataset")
        model = initialized_model_for_arm(args.arm, args.seed)
        training_report = train_model(
            model, args.arm, data, curriculum, seed=args.seed, canonical=True,
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
