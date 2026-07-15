#!/usr/bin/env python3
"""Train the five frozen CPU arms for the DWEPR WGRQ falsifier.

This module consumes the independently audited Stage-A JSONL transcript.  It
does not generate episodes or score confirmation data.  Uniform probes,
privileged endpoint bits, the wrong-partner permutation, and all epoch batches
are deterministic functions of the immutable public transcript and frozen
seed, and are prepared before model initialization.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

try:
    from wgrq_state_machine import (
        EXPECTED_PARAMETER_COUNT,
        FLIP,
        PACKET_BITS,
        ROTATE,
        HardBitDWEPRLearner,
        make_scale_mask,
    )
except ImportError:  # pragma: no cover - package-style import in external tests
    from train.wgrq_state_machine import (
        EXPECTED_PARAMETER_COUNT,
        FLIP,
        PACKET_BITS,
        ROTATE,
        HardBitDWEPRLearner,
        make_scale_mask,
    )


EPISODE_SCHEMA = "wgrq_stage_a_episode_v1"
RICH_EPISODE_SCHEMA = "wgrq_falsifier_v1"
GENERATOR_REPORT_SCHEMA = "wgrq_falsifier_v1_report"
AUDIT_NAME = "wgrq_stage_a_independent_admission_v1"
TRAIN_SCALES = (4, 6, 8)
LENGTH_BANDS = {"le_2n": 2, "le_8n": 8}
EPISODES_PER_CELL = 3_072
TRAIN_EPISODES = 18_432
HISTORIES_PER_EPISODE = 4
SHAM_RELATIONS = 2
PROBES_PER_HISTORY = 8
ANSWERS_PER_EPISODE = HISTORIES_PER_EPISODE * PROBES_PER_HISTORY
TRAIN_ANSWER_CALLS = 589_824

BATCH_SIZE = 64
EPOCHS = 4
BATCHES_PER_EPOCH = TRAIN_EPISODES // BATCH_SIZE
TOTAL_UPDATES = EPOCHS * BATCHES_PER_EPOCH
WARMUP_UPDATES = 64

LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
EPSILON = 1e-8
MATRIX_WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

FROZEN_SEEDS = (
    17_011,
    27_103,
    38_119,
    49_201,
    50_311,
    61_403,
    72_503,
    83_609,
    94_709,
    105_019,
    116_027,
    127_031,
)

WGRQ_SHORTEST = "WGRQ-shortest"
ACTIVE_ANSWER_ONLY = "active-answer-only"
UNIFORM_WITNESS = "uniform-witness"
RELATION_SHAM = "relation-sham"
PRIVILEGED_EDGE = "privileged-edge"
ARM_NAMES = (
    WGRQ_SHORTEST,
    ACTIVE_ANSWER_ONLY,
    UNIFORM_WITNESS,
    RELATION_SHAM,
    PRIVILEGED_EDGE,
)

# A zero coefficient never suppresses construction of the corresponding loss.
ARM_COEFFICIENTS = {
    WGRQ_SHORTEST: {"answer": 0.75, "relation_shortest": 0.25, "commitment": 0.01},
    ACTIVE_ANSWER_ONLY: {"answer": 1.00, "commitment": 0.01},
    UNIFORM_WITNESS: {"answer": 0.75, "relation_uniform": 0.25, "commitment": 0.01},
    RELATION_SHAM: {"answer": 0.75, "relation_sham": 0.25, "commitment": 0.01},
    PRIVILEGED_EDGE: {"answer": 0.75, "privileged_edge": 0.25, "commitment": 0.01},
}

EPISODE_FIELDS = {
    "schema",
    "episode_id",
    "n",
    "length_band",
    "cell_index",
    "probe_rotations",
    "histories",
    "equivalence_labels",
    "witness_masks",
    "record_sha256",
}
HISTORY_FIELDS = {"events", "answers"}
RICH_EPISODE_FIELDS = {
    "schema",
    "episode_id",
    "split",
    "global_episode_index",
    "batch_index",
    "batch_offset",
    "cell",
    "event_code_bits",
    "gadget",
    "probe_rotations",
    "histories",
    "equivalence_label_matrix",
    "pairs",
    "first_distinguishing_witness_mask",
    "uniform_probe_index",
    "uniform_probe_mask",
    "balance",
    "oracle_call_span",
}
RICH_HISTORY_FIELDS = {
    "history_index",
    "history_id",
    "role",
    "events",
    "source_length",
    "event_counts",
    "history_sha256",
    "probes",
    "canonical_edge_bits_from_public_answers",
}
RICH_PROBE_FIELDS = {
    "probe_index",
    "continuation_rotations",
    "continuation",
    "answer",
    "oracle_call_id",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def record_hash(row: dict[str, Any]) -> str:
    payload = dict(row)
    payload.pop("record_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_model_state(model: HardBitDWEPRLearner) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class FrozenInputBinding:
    transcript: Path
    audit_report: Path
    transcript_sha256: str
    audit_report_sha256: str

    def assert_unchanged(self) -> None:
        if sha256_file(self.transcript) != self.transcript_sha256:
            raise RuntimeError("training transcript changed after its hash was frozen")
        if sha256_file(self.audit_report) != self.audit_report_sha256:
            raise RuntimeError("audit report changed after its hash was frozen")


@dataclasses.dataclass(frozen=True)
class WGRQHistory:
    events: tuple[int, ...]
    answers: tuple[int, ...]
    endpoint_answer: int
    canonical_code: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class WGRQEpisode:
    episode_id: str
    record_sha256: str
    n: int
    length_band: str
    cell_index: int
    batch_index: int
    batch_offset: int
    probe_rotations: tuple[int, ...]
    histories: tuple[WGRQHistory, ...]
    shortest_witness_mask: tuple[int, ...]
    uniform_probe_index: int


@dataclasses.dataclass(frozen=True)
class FrozenWGRQDataset:
    episodes: tuple[WGRQEpisode, ...]
    sham_equivalent_partner: tuple[int, ...]
    sham_non_equivalent_partner: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class WGRQBatch:
    source_events: torch.Tensor
    source_mask: torch.Tensor
    probe_events: torch.Tensor
    probe_mask: torch.Tensor
    answers: torch.Tensor
    scale_mask: torch.Tensor
    canonical_codes: torch.Tensor
    shortest_witness_mask: torch.Tensor
    uniform_probe_index: torch.Tensor
    sham_permutation: torch.Tensor


@dataclasses.dataclass(frozen=True)
class WGRQForwardPass:
    probe_logits: torch.Tensor
    final_source_probabilities: torch.Tensor
    commitment: torch.Tensor


def bind_frozen_inputs(transcript: str | Path, audit_report: str | Path) -> FrozenInputBinding:
    transcript_path = Path(transcript)
    report_path = Path(audit_report)
    if not transcript_path.is_file() or not report_path.is_file():
        raise ValueError("transcript and audit report must both be existing files")
    transcript_sha256 = sha256_file(transcript_path)
    report_sha256 = sha256_file(report_path)
    with report_path.open("r", encoding="utf-8") as source:
        report = json.load(source)
    admitted = False
    if isinstance(report, dict) and report.get("audit") == AUDIT_NAME:
        artifact = report.get("artifact")
        call_ledger = report.get("call_ledger_expected")
        admitted = bool(
            report.get("all_checks_pass") is True
            and isinstance(artifact, dict)
            and artifact.get("sha256") == transcript_sha256
            and artifact.get("records_valid") == TRAIN_EPISODES
            and isinstance(call_ledger, dict)
            and call_ledger.get("ordinary_one_bit_answer_calls") == TRAIN_ANSWER_CALLS
        )
    elif isinstance(report, dict) and report.get("schema") == GENERATOR_REPORT_SCHEMA:
        transcript_artifact = report.get("artifacts", {}).get("transcript")
        totals = report.get("totals")
        admitted = bool(
            report.get("passed") is True
            and isinstance(transcript_artifact, dict)
            and transcript_artifact.get("sha256") == transcript_sha256
            and transcript_artifact.get("rows") == TRAIN_EPISODES
            and isinstance(totals, dict)
            and totals.get("ordinary_one_bit_read_calls") == TRAIN_ANSWER_CALLS
        )
    if not admitted:
        raise ValueError("audit report does not admit and bind this exact WGRQ transcript")
    binding = FrozenInputBinding(
        transcript=transcript_path,
        audit_report=report_path,
        transcript_sha256=transcript_sha256,
        audit_report_sha256=report_sha256,
    )
    binding.assert_unchanged()
    return binding


def _canonical_code_from_answers(answers: Sequence[int], n: int) -> tuple[int, ...]:
    if len(answers) != PROBES_PER_HISTORY:
        raise ValueError("canonical target requires the public eight-probe answer bank")
    active = tuple(int(answer) for answer in answers[: n - 1])
    return active + (0,) * (PACKET_BITS - len(active))


def deterministic_uniform_probe(record_sha256: str) -> int:
    try:
        record_bytes = bytes.fromhex(record_sha256)
    except ValueError as error:
        raise ValueError("record hash must be hexadecimal") from error
    if len(record_bytes) != 32:
        raise ValueError("record hash must be SHA-256")
    digest = hashlib.sha256(b"wgrq-uniform-probe-v1\0" + record_bytes).digest()
    return digest[0] & (PROBES_PER_HISTORY - 1)


def _exact_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _bits(value: Any, length: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} bits")
    result = tuple(_exact_int(bit, label) for bit in value)
    if any(bit not in (0, 1) for bit in result):
        raise ValueError(f"{label} must be binary")
    return result


def _normalize_flat_episode(row: dict[str, Any], transcript_index: int) -> WGRQEpisode:
    if not isinstance(row, dict) or set(row) != EPISODE_FIELDS:
        raise ValueError("episode fields differ from the audited public contract")
    if row["schema"] != EPISODE_SCHEMA:
        raise ValueError("episode schema differs from the frozen contract")
    if not isinstance(row["record_sha256"], str) or row["record_sha256"] != record_hash(row):
        raise ValueError("episode record hash mismatch")

    n = _exact_int(row["n"], "n")
    if n not in TRAIN_SCALES:
        raise ValueError("training scale must be one of 4, 6, 8")
    length_band = row["length_band"]
    if length_band not in LENGTH_BANDS:
        raise ValueError("unknown source-length band")
    cell_index = _exact_int(row["cell_index"], "cell_index")
    if not 0 <= cell_index < EPISODES_PER_CELL:
        raise ValueError("cell index lies outside the frozen cell")
    expected_id = f"wgrq-n{n:02d}-{length_band}-{cell_index:06d}"
    if row["episode_id"] != expected_id:
        raise ValueError("episode ID is not canonical")

    probe_rotations = tuple(_exact_int(value, "probe rotation") for value in row["probe_rotations"])
    expected_probes = tuple(index % n for index in range(PROBES_PER_HISTORY))
    if probe_rotations != expected_probes:
        raise ValueError("probe rotations differ from the fixed eight-probe bank")

    histories_value = row["histories"]
    if not isinstance(histories_value, list) or len(histories_value) != HISTORIES_PER_EPISODE:
        raise ValueError("episode must contain exactly four histories")
    histories = []
    multiplier = LENGTH_BANDS[length_band]
    minimum = 0 if multiplier == 2 else 2 * n + 1
    maximum = multiplier * n
    for value in histories_value:
        if not isinstance(value, dict) or set(value) != HISTORY_FIELDS:
            raise ValueError("history fields differ from the public contract")
        if not isinstance(value["events"], list) or any(event not in ("R", "F") for event in value["events"]):
            raise ValueError("history contains an invalid public event")
        events = tuple(ROTATE if event == "R" else FLIP for event in value["events"])
        if not minimum <= len(events) <= maximum:
            raise ValueError("history lies outside its declared source-length band")
        answers = _bits(value["answers"], PROBES_PER_HISTORY, "history answers")
        histories.append(
            WGRQHistory(
                events=events,
                answers=answers,
                endpoint_answer=answers[0],
                canonical_code=_canonical_code_from_answers(answers, n),
            )
        )

    if row["equivalence_labels"] != [1, 0]:
        raise ValueError("episode pair roles or equivalence labels changed")
    masks = row["witness_masks"]
    if not isinstance(masks, list) or len(masks) != 2:
        raise ValueError("episode must contain exactly two witness masks")
    equivalent_mask = _bits(masks[0], PROBES_PER_HISTORY, "equivalent witness mask")
    shortest_mask = _bits(masks[1], PROBES_PER_HISTORY, "non-equivalent witness mask")
    if any(equivalent_mask) or sum(shortest_mask) != 1:
        raise ValueError("witness masks do not match the frozen pair roles")
    if histories[0].answers != histories[1].answers:
        raise ValueError("equivalent histories disagree on a public answer")
    differing = tuple(
        index for index in range(PROBES_PER_HISTORY)
        if histories[2].answers[index] != histories[3].answers[index]
    )
    if not differing or shortest_mask.index(1) != differing[0] or differing[0] > n - 2:
        raise ValueError("non-equivalent witness mask is not the first distinguishing probe")

    return WGRQEpisode(
        episode_id=expected_id,
        record_sha256=row["record_sha256"],
        n=n,
        length_band=length_band,
        cell_index=cell_index,
        batch_index=transcript_index // BATCH_SIZE,
        batch_offset=transcript_index % BATCH_SIZE,
        probe_rotations=probe_rotations,
        histories=tuple(histories),
        shortest_witness_mask=shortest_mask,
        uniform_probe_index=deterministic_uniform_probe(row["record_sha256"]),
    )


def _normalize_rich_episode(row: dict[str, Any], transcript_index: int) -> WGRQEpisode:
    if set(row) != RICH_EPISODE_FIELDS or row.get("schema") != RICH_EPISODE_SCHEMA:
        raise ValueError("rich episode fields differ from the frozen producer contract")
    if row.get("split") != "training":
        raise ValueError("WGRQ trainer may consume only the training split")
    global_index = _exact_int(row["global_episode_index"], "global episode index")
    batch_index = _exact_int(row["batch_index"], "batch index")
    batch_offset = _exact_int(row["batch_offset"], "batch offset")
    if (
        global_index != transcript_index
        or batch_index != transcript_index // BATCH_SIZE
        or batch_offset != transcript_index % BATCH_SIZE
    ):
        raise ValueError("producer order or frozen batch membership is discontinuous")

    cell = row["cell"]
    if not isinstance(cell, dict) or set(cell) != {
        "n", "length_band", "source_length_ceiling", "cell_episode_index"
    }:
        raise ValueError("rich episode cell differs from the producer contract")
    n = _exact_int(cell["n"], "n")
    length_band = cell["length_band"]
    cell_index = _exact_int(cell["cell_episode_index"], "cell episode index")
    if n not in TRAIN_SCALES or length_band not in LENGTH_BANDS:
        raise ValueError("rich episode lies outside the frozen training cells")
    if cell["source_length_ceiling"] != LENGTH_BANDS[length_band] * n:
        raise ValueError("source-length ceiling changed")
    if not 0 <= cell_index < EPISODES_PER_CELL:
        raise ValueError("cell episode index lies outside the frozen cell")
    expected_id = "wgrq-train-n{:02d}-{}-{:06d}".format(
        n, length_band.replace("_", ""), cell_index
    )
    if row["episode_id"] != expected_id:
        raise ValueError("rich episode ID is not canonical")
    if row["event_code_bits"] != {"R": [1, 0], "F": [0, 1]}:
        raise ValueError("two-bit event code changed")

    probe_rotations = tuple(_exact_int(value, "probe rotation") for value in row["probe_rotations"])
    expected_probes = tuple(index % n for index in range(PROBES_PER_HISTORY))
    if probe_rotations != expected_probes:
        raise ValueError("probe rotations differ from the fixed eight-probe bank")
    histories_value = row["histories"]
    if not isinstance(histories_value, list) or len(histories_value) != HISTORIES_PER_EPISODE:
        raise ValueError("rich episode must contain exactly four histories")
    roles = ("equivalent_a", "equivalent_b", "different_a", "different_b")
    histories = []
    first_call = transcript_index * ANSWERS_PER_EPISODE
    for history_index, value in enumerate(histories_value):
        if not isinstance(value, dict) or set(value) != RICH_HISTORY_FIELDS:
            raise ValueError("rich history fields differ from the producer contract")
        if (
            value["history_index"] != history_index
            or value["role"] != roles[history_index]
            or value["history_id"] != f"{expected_id}:h{history_index}"
        ):
            raise ValueError("rich history identity or role changed")
        names = value["events"]
        if not isinstance(names, list) or any(event not in ("R", "F") for event in names):
            raise ValueError("rich history contains an invalid public event")
        events = tuple(ROTATE if event == "R" else FLIP for event in names)
        if value["source_length"] != len(events) or len(events) > LENGTH_BANDS[length_band] * n:
            raise ValueError("rich history source length changed")
        if value["event_counts"] != {"R": names.count("R"), "F": names.count("F")}:
            raise ValueError("rich history event-count ledger changed")
        history_digest = hashlib.sha256("".join(names).encode("ascii")).hexdigest()
        if value["history_sha256"] != history_digest:
            raise ValueError("rich history hash mismatch")

        probes = value["probes"]
        if not isinstance(probes, list) or len(probes) != PROBES_PER_HISTORY:
            raise ValueError("rich history must contain exactly eight probes")
        answers = []
        for probe_index, probe in enumerate(probes):
            if not isinstance(probe, dict) or set(probe) != RICH_PROBE_FIELDS:
                raise ValueError("rich probe fields differ from the producer contract")
            rotations = probe_rotations[probe_index]
            expected_call = first_call + history_index * PROBES_PER_HISTORY + probe_index
            if (
                probe["probe_index"] != probe_index
                or probe["continuation_rotations"] != rotations
                or probe["continuation"] != ["R"] * rotations
                or probe["oracle_call_id"] != expected_call
            ):
                raise ValueError("rich probe geometry or call ledger changed")
            answer = _exact_int(probe["answer"], "probe answer")
            if answer not in (0, 1):
                raise ValueError("ordinary probe answer must be one bit")
            answers.append(answer)
        canonical = _canonical_code_from_answers(answers, n)
        if value["canonical_edge_bits_from_public_answers"] != list(canonical[: n - 1]):
            raise ValueError("canonical edge target is not derived from public answers")
        histories.append(
            WGRQHistory(
                events=events,
                answers=tuple(answers),
                endpoint_answer=answers[0],
                canonical_code=canonical,
            )
        )

    signatures = [history.canonical_code[: n - 1] for history in histories]
    expected_matrix = [
        [int(left == right) for right in signatures]
        for left in signatures
    ]
    if row["equivalence_label_matrix"] != expected_matrix:
        raise ValueError("rich equivalence matrix is not answer-derived")
    differing = [
        index for index in range(PROBES_PER_HISTORY)
        if histories[2].answers[index] != histories[3].answers[index]
    ]
    if not differing or differing[0] > n - 2:
        raise ValueError("rich non-equivalent pair lacks a valid shortest witness")
    shortest_mask = tuple(int(index == differing[0]) for index in range(PROBES_PER_HISTORY))
    expected_pairs = {
        "equivalent": {"history_indices": [0, 1], "label": 1},
        "non_equivalent": {
            "history_indices": [2, 3],
            "label": 0,
            "shortest_witness_depth": differing[0],
            "first_distinguishing_probe_index": differing[0],
            "first_distinguishing_witness_mask": list(shortest_mask),
        },
    }
    if row["pairs"] != expected_pairs or row["first_distinguishing_witness_mask"] != list(shortest_mask):
        raise ValueError("rich pair or witness declaration changed")
    uniform_probe_index = _exact_int(row["uniform_probe_index"], "uniform probe index")
    if not 0 <= uniform_probe_index < PROBES_PER_HISTORY:
        raise ValueError("uniform probe index lies outside the frozen bank")
    expected_uniform_mask = [
        int(index == uniform_probe_index) for index in range(PROBES_PER_HISTORY)
    ]
    if row["uniform_probe_mask"] != expected_uniform_mask:
        raise ValueError("uniform probe mask does not select its frozen index")
    if row["oracle_call_span"] != {
        "first_call_id": first_call,
        "last_call_id": first_call + ANSWERS_PER_EPISODE - 1,
        "ordinary_one_bit_read_calls": ANSWERS_PER_EPISODE,
    }:
        raise ValueError("rich episode ordinary-call span changed")

    row_digest = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
    return WGRQEpisode(
        episode_id=expected_id,
        record_sha256=row_digest,
        n=n,
        length_band=length_band,
        cell_index=cell_index,
        batch_index=batch_index,
        batch_offset=batch_offset,
        probe_rotations=probe_rotations,
        histories=tuple(histories),
        shortest_witness_mask=shortest_mask,
        uniform_probe_index=uniform_probe_index,
    )


def normalize_episode(row: dict[str, Any], transcript_index: int = 0) -> WGRQEpisode:
    schema = row.get("schema") if isinstance(row, dict) else None
    if schema == RICH_EPISODE_SCHEMA:
        return _normalize_rich_episode(row, transcript_index)
    if schema == EPISODE_SCHEMA:
        return _normalize_flat_episode(row, transcript_index)
    raise ValueError("unknown WGRQ training episode schema")


def _strict_json_line(raw_line: bytes, line_number: int) -> dict[str, Any]:
    if not raw_line.endswith(b"\n") or raw_line in (b"\n", b"\r\n"):
        raise ValueError(f"blank or unterminated transcript line {line_number}")
    try:
        text = raw_line[:-1].decode("ascii")
        row = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid transcript JSON on line {line_number}") from error
    if not isinstance(row, dict) or raw_line != canonical_json_bytes(row) + b"\n":
        raise ValueError(f"transcript line {line_number} is not canonical JSONL")
    return row


def _history_sham_stratum(episode: WGRQEpisode, history_index: int) -> tuple[Any, ...]:
    history = episode.histories[history_index]
    return (
        episode.n,
        episode.length_band,
        len(history.events),
        history.events.count(FLIP),
        history.events.count(ROTATE),
        history.endpoint_answer,
        history.answers,
    )


def _sham_permutation(episodes: Sequence[WGRQEpisode], history_index: int, domain: bytes) -> tuple[int, ...]:
    frozen_batches: dict[int, list[int]] = collections.defaultdict(list)
    for index, episode in enumerate(episodes):
        frozen_batches[episode.batch_index].append(index)
    permutation = [-1] * len(episodes)
    for batch_index, indices in frozen_batches.items():
        if len(indices) != BATCH_SIZE:
            raise ValueError(f"frozen batch {batch_index} does not contain 64 episodes")
        ordered = sorted(
            indices,
            key=lambda index: (
                _history_sham_stratum(episodes[index], history_index),
                hashlib.sha256(
                    domain + b"\0" + bytes.fromhex(episodes[index].record_sha256)
                ).digest(),
            ),
        )
        for position, index in enumerate(ordered):
            permutation[index] = ordered[(position + 1) % len(ordered)]
    if sorted(permutation) != list(range(len(episodes))) or any(index == partner for index, partner in enumerate(permutation)):
        raise RuntimeError("wrong-partner construction did not produce a derangement")
    if any(
        episodes[index].batch_index != episodes[partner].batch_index
        for index, partner in enumerate(permutation)
    ):
        raise RuntimeError("wrong-partner construction escaped a frozen batch")
    return tuple(permutation)


def finalize_dataset(episodes: Sequence[WGRQEpisode]) -> FrozenWGRQDataset:
    if len(episodes) != TRAIN_EPISODES:
        raise ValueError(f"training transcript must contain exactly {TRAIN_EPISODES:,} episodes")
    identifiers = [episode.episode_id for episode in episodes]
    hashes = [episode.record_sha256 for episode in episodes]
    if len(set(identifiers)) != len(identifiers) or len(set(hashes)) != len(hashes):
        raise ValueError("episode IDs and record hashes must be unique")
    for transcript_index, episode in enumerate(episodes):
        if (
            episode.batch_index != transcript_index // BATCH_SIZE
            or episode.batch_offset != transcript_index % BATCH_SIZE
        ):
            raise ValueError("frozen producer batches are not contiguous blocks of 64")
    cells = collections.Counter((episode.n, episode.length_band) for episode in episodes)
    expected = {
        (n, length_band): EPISODES_PER_CELL
        for n in TRAIN_SCALES
        for length_band in LENGTH_BANDS
    }
    if dict(cells) != expected:
        raise ValueError("every scale/length cell must contain exactly 3,072 episodes")
    if len(episodes) * ANSWERS_PER_EPISODE != TRAIN_ANSWER_CALLS:
        raise RuntimeError("ordinary training answer-call accounting mismatch")
    equivalent_partner = _sham_permutation(episodes, 1, b"wgrq-relation-sham-equivalent-v1")
    non_equivalent_partner = _sham_permutation(episodes, 3, b"wgrq-relation-sham-non-equivalent-v1")
    return FrozenWGRQDataset(
        episodes=tuple(episodes),
        sham_equivalent_partner=equivalent_partner,
        sham_non_equivalent_partner=non_equivalent_partner,
    )


def load_frozen_dataset(binding: FrozenInputBinding) -> FrozenWGRQDataset:
    binding.assert_unchanged()
    rows = []
    with binding.transcript.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            rows.append(_strict_json_line(raw_line, line_number))
    episodes = tuple(normalize_episode(row, index) for index, row in enumerate(rows))
    dataset = finalize_dataset(episodes)
    binding.assert_unchanged()
    return dataset


def frozen_training_batches(seed: int) -> tuple[tuple[int, ...], ...]:
    if seed not in FROZEN_SEEDS:
        raise ValueError("seed is not one of the 12 frozen paired seeds")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    for _ in range(EPOCHS):
        batch_order = torch.randperm(BATCHES_PER_EPOCH, generator=generator).tolist()
        for batch_index in batch_order:
            first = batch_index * BATCH_SIZE
            batches.append(tuple(range(first, first + BATCH_SIZE)))
    if len(batches) != TOTAL_UPDATES or any(len(batch) != BATCH_SIZE for batch in batches):
        raise RuntimeError("frozen update-order accounting mismatch")
    return tuple(batches)


def collate_frozen_batch(dataset: FrozenWGRQDataset, indices: Sequence[int]) -> WGRQBatch:
    if len(indices) != BATCH_SIZE or len(set(indices)) != BATCH_SIZE:
        raise ValueError("each frozen update must contain 64 distinct episodes")
    selected = [dataset.episodes[int(index)] for index in indices]
    frozen_batch_ids = {episode.batch_index for episode in selected}
    if len(frozen_batch_ids) != 1:
        raise ValueError("an update may not split or combine frozen producer batches")
    max_source = max(len(history.events) for episode in selected for history in episode.histories)
    max_probe = max(max(episode.probe_rotations) for episode in selected)

    source_events = torch.zeros(BATCH_SIZE, HISTORIES_PER_EPISODE, max_source, dtype=torch.long)
    source_mask = torch.zeros_like(source_events, dtype=torch.bool)
    probe_events = torch.full((BATCH_SIZE, PROBES_PER_HISTORY, max_probe), ROTATE, dtype=torch.long)
    probe_mask = torch.zeros_like(probe_events, dtype=torch.bool)
    answers = torch.empty(BATCH_SIZE, HISTORIES_PER_EPISODE, PROBES_PER_HISTORY, dtype=torch.float32)
    scale_masks = torch.empty(BATCH_SIZE, PACKET_BITS, dtype=torch.float32)
    canonical_codes = torch.empty(BATCH_SIZE, HISTORIES_PER_EPISODE, PACKET_BITS, dtype=torch.float32)
    witness_masks = torch.empty(BATCH_SIZE, PROBES_PER_HISTORY, dtype=torch.float32)
    uniform_indices = torch.empty(BATCH_SIZE, dtype=torch.long)
    sham_permutation = torch.empty(BATCH_SIZE, SHAM_RELATIONS, dtype=torch.long)
    local_index = {int(global_index): offset for offset, global_index in enumerate(indices)}

    for batch_index, (global_index, episode) in enumerate(zip(indices, selected)):
        scale_masks[batch_index] = make_scale_mask(episode.n)
        answers[batch_index] = torch.tensor(
            [history.answers for history in episode.histories], dtype=torch.float32
        )
        canonical_codes[batch_index] = torch.tensor(
            [history.canonical_code for history in episode.histories], dtype=torch.float32
        )
        witness_masks[batch_index] = torch.tensor(episode.shortest_witness_mask, dtype=torch.float32)
        uniform_indices[batch_index] = episode.uniform_probe_index
        sham_permutation[batch_index, 0] = local_index[
            dataset.sham_equivalent_partner[int(global_index)]
        ]
        sham_permutation[batch_index, 1] = local_index[
            dataset.sham_non_equivalent_partner[int(global_index)]
        ]
        for history_index, history in enumerate(episode.histories):
            length = len(history.events)
            if length:
                source_events[batch_index, history_index, :length] = torch.tensor(history.events)
                source_mask[batch_index, history_index, :length] = True
        for probe_index, rotations in enumerate(episode.probe_rotations):
            probe_mask[batch_index, probe_index, :rotations] = True
    return WGRQBatch(
        source_events=source_events,
        source_mask=source_mask,
        probe_events=probe_events,
        probe_mask=probe_mask,
        answers=answers,
        scale_mask=scale_masks,
        canonical_codes=canonical_codes,
        shortest_witness_mask=witness_masks,
        uniform_probe_index=uniform_indices,
        sham_permutation=sham_permutation,
    )


def validate_batch(batch: WGRQBatch) -> None:
    batch_size = batch.answers.shape[0]
    if batch.source_events.shape[:2] != (batch_size, HISTORIES_PER_EPISODE):
        raise ValueError("source events must have shape [batch,4,steps]")
    if batch.source_mask.shape != batch.source_events.shape:
        raise ValueError("source mask must match source events")
    if batch.probe_events.shape[:2] != (batch_size, PROBES_PER_HISTORY):
        raise ValueError("probe events must have shape [batch,8,steps]")
    if batch.probe_mask.shape != batch.probe_events.shape:
        raise ValueError("probe mask must match probe events")
    if batch.answers.shape != (batch_size, HISTORIES_PER_EPISODE, PROBES_PER_HISTORY):
        raise ValueError("answers must have shape [batch,4,8]")
    if batch.scale_mask.shape != (batch_size, PACKET_BITS):
        raise ValueError("scale masks must have shape [batch,15]")
    if batch.canonical_codes.shape != (batch_size, HISTORIES_PER_EPISODE, PACKET_BITS):
        raise ValueError("canonical targets must have shape [batch,4,15]")
    if batch.shortest_witness_mask.shape != (batch_size, PROBES_PER_HISTORY):
        raise ValueError("shortest-witness masks must have shape [batch,8]")
    if not bool((batch.shortest_witness_mask.sum(dim=-1) == 1).all()):
        raise ValueError("each shortest-witness mask must be one-hot")
    if batch.uniform_probe_index.shape != (batch_size,) or not bool(
        ((batch.uniform_probe_index >= 0) & (batch.uniform_probe_index < PROBES_PER_HISTORY)).all()
    ):
        raise ValueError("uniform indices must select one frozen probe")
    if batch.sham_permutation.shape != (batch_size, SHAM_RELATIONS):
        raise ValueError("sham permutation must select two batch-local partners")
    for relation in range(SHAM_RELATIONS):
        values = batch.sham_permutation[:, relation].tolist()
        if sorted(values) != list(range(batch_size)) or any(index == partner for index, partner in enumerate(values)):
            raise ValueError("each sham relation must be a batch-local derangement")
    if not bool(((batch.answers == 0) | (batch.answers == 1)).all()):
        raise ValueError("ordinary answers must be binary")
    inactive = (batch.scale_mask[:, None, :] == 0).expand_as(batch.canonical_codes)
    if not bool((batch.canonical_codes.masked_select(inactive) == 0).all()):
        raise ValueError("masked canonical targets must be zero")


def _commitment_sum(
    probabilities: torch.Tensor,
    transition_mask: torch.Tensor,
    scale_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = transition_mask.to(probabilities.dtype).unsqueeze(-1) * scale_mask.to(probabilities.dtype).unsqueeze(-2)
    return (probabilities * (1.0 - probabilities) * weights).sum(), weights.sum()


def forward_training_batch(model: HardBitDWEPRLearner, batch: WGRQBatch) -> WGRQForwardPass:
    validate_batch(batch)
    batch_size, history_count, source_steps = batch.source_events.shape
    probe_steps = batch.probe_events.shape[-1]
    source_scale = batch.scale_mask[:, None, :].expand(-1, history_count, -1).reshape(-1, PACKET_BITS)
    source = model.encode(
        batch.source_events.reshape(-1, source_steps),
        source_scale,
        batch.source_mask.reshape(-1, source_steps),
    )
    source_packets = source.packet.reshape(batch_size, history_count, PACKET_BITS)
    source_probabilities = source.probabilities.reshape(
        batch_size, history_count, source_steps, PACKET_BITS
    )
    if source_steps:
        source_lengths = batch.source_mask.sum(dim=-1)
        final_indices = (source_lengths - 1).clamp_min(0)[..., None, None].expand(-1, -1, 1, PACKET_BITS)
        final_probabilities = source_probabilities.gather(2, final_indices).squeeze(2)
        final_probabilities = torch.where(
            source_lengths[..., None] > 0, final_probabilities, source_packets
        )
    else:
        final_probabilities = source_packets

    branch_packets = source_packets[:, :, None, :].expand(-1, -1, PROBES_PER_HISTORY, -1).reshape(-1, PACKET_BITS)
    branch_scale = batch.scale_mask[:, None, None, :].expand(
        -1, history_count, PROBES_PER_HISTORY, -1
    ).reshape(-1, PACKET_BITS)
    branch_events = batch.probe_events[:, None, :, :].expand(
        -1, history_count, -1, -1
    ).reshape(-1, probe_steps)
    branch_mask = batch.probe_mask[:, None, :, :].expand(
        -1, history_count, -1, -1
    ).reshape(-1, probe_steps)
    probes = model.rollout(branch_packets, branch_events, branch_scale, branch_mask)
    all_logits = model.read_logits(probes.packet, branch_scale).reshape(
        batch_size, history_count, PROBES_PER_HISTORY
    )
    probe_probabilities = probes.probabilities.reshape(
        batch_size, history_count, PROBES_PER_HISTORY, probe_steps, PACKET_BITS
    )

    source_sum, source_count = _commitment_sum(
        source_probabilities,
        batch.source_mask,
        batch.scale_mask[:, None, :].expand(-1, history_count, -1),
    )
    probe_sum, probe_count = _commitment_sum(
        probe_probabilities,
        batch.probe_mask[:, None, :, :].expand(-1, history_count, -1, -1),
        batch.scale_mask[:, None, None, :].expand(-1, history_count, PROBES_PER_HISTORY, -1),
    )
    denominator = source_count + probe_count
    if float(denominator) <= 0:
        raise ValueError("batch contains no active transition bits")
    return WGRQForwardPass(
        probe_logits=all_logits,
        final_source_probabilities=final_probabilities,
        commitment=(source_sum + probe_sum) / denominator,
    )


def bernoulli_js_from_logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("behavior logits must have matching shapes")
    left_probability = torch.sigmoid(left)
    right_probability = torch.sigmoid(right)
    mixture = 0.5 * (left_probability + right_probability)
    epsilon = torch.finfo(left_probability.dtype).eps

    def kl(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(epsilon, 1.0 - epsilon)
        center = mixture.clamp(epsilon, 1.0 - epsilon)
        return probability * (probability.log() - center.log()) + (1.0 - probability) * (
            torch.log1p(-probability) - torch.log1p(-center)
        )

    return 0.5 * (kl(left_probability) + kl(right_probability))


def separation_hinge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("separation logits must have matching shapes")
    return F.relu(1.0 - (left - right).abs())


def compute_all_loss_terms(
    forward: WGRQForwardPass,
    batch: WGRQBatch,
    arm: str,
) -> dict[str, torch.Tensor]:
    """Eagerly construct all answer, relation, sham, and ceiling tensors."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown frozen WGRQ arm {arm!r}")
    logits = forward.probe_logits
    if logits.shape != batch.answers.shape:
        raise ValueError("model outputs do not match the frozen probe geometry")

    answer = F.binary_cross_entropy_with_logits(logits, batch.answers)
    equivalent_js = bernoulli_js_from_logits(logits[:, 0], logits[:, 1]).mean()
    hinges = separation_hinge(logits[:, 2], logits[:, 3])
    separation_shortest = (hinges * batch.shortest_witness_mask).sum(dim=-1).mean()
    uniform_mask = F.one_hot(
        batch.uniform_probe_index, num_classes=PROBES_PER_HISTORY
    ).to(logits.dtype)
    separation_uniform = (hinges * uniform_mask).sum(dim=-1).mean()
    relation_shortest = 0.5 * (equivalent_js + separation_shortest)
    relation_uniform = 0.5 * (equivalent_js + separation_uniform)

    sham_equivalent_right = logits[batch.sham_permutation[:, 0], 1]
    sham_non_equivalent_right = logits[batch.sham_permutation[:, 1], 3]
    sham_equivalent_js = bernoulli_js_from_logits(
        logits[:, 0], sham_equivalent_right
    ).mean()
    sham_hinges = separation_hinge(logits[:, 2], sham_non_equivalent_right)
    sham_separation = (sham_hinges * batch.shortest_witness_mask).sum(dim=-1).mean()
    relation_sham = 0.5 * (sham_equivalent_js + sham_separation)

    active_codes = batch.scale_mask[:, None, :].expand_as(batch.canonical_codes)
    edge_elements = F.binary_cross_entropy(
        forward.final_source_probabilities.clamp(1e-7, 1.0 - 1e-7),
        batch.canonical_codes,
        reduction="none",
    )
    privileged_edge = (edge_elements * active_codes).sum() / active_codes.sum()

    terms = {
        "answer": answer,
        "equivalent_js": equivalent_js,
        "separation_shortest": separation_shortest,
        "separation_uniform": separation_uniform,
        "relation_shortest": relation_shortest,
        "relation_uniform": relation_uniform,
        "sham_equivalent_js": sham_equivalent_js,
        "sham_separation": sham_separation,
        "relation_sham": relation_sham,
        "commitment": forward.commitment,
        "privileged_edge": privileged_edge,
    }
    total = sum(
        terms[name] * coefficient
        for name, coefficient in ARM_COEFFICIENTS[arm].items()
    )
    return {"total": total, **terms}


def build_optimizer(model: HardBitDWEPRLearner) -> torch.optim.AdamW:
    model.assert_contract()
    matrices = []
    vectors = []
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            raise ValueError(f"learner parameter {name} unexpectedly has gradients disabled")
        if parameter.ndim == 2:
            matrices.append(parameter)
        elif parameter.ndim == 1:
            vectors.append(parameter)
        else:
            raise ValueError(f"unexpected parameter rank for {name}")
    return torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": MATRIX_WEIGHT_DECAY},
            {"params": vectors, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=0.0,
    )


def learning_rate_for_update(update_index: int) -> float:
    update_index = int(update_index)
    if not 0 <= update_index < TOTAL_UPDATES:
        raise ValueError(f"update index must lie in [0, {TOTAL_UPDATES})")
    completed = update_index + 1
    if completed <= WARMUP_UPDATES:
        scale = completed / WARMUP_UPDATES
    else:
        progress = (completed - WARMUP_UPDATES) / (TOTAL_UPDATES - WARMUP_UPDATES)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * scale


def run_training_update(
    model: HardBitDWEPRLearner,
    optimizer: torch.optim.AdamW,
    batch: WGRQBatch,
    arm: str,
    update_index: int,
) -> dict[str, float]:
    learning_rate = learning_rate_for_update(update_index)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    forward = forward_training_batch(model, batch)
    terms = compute_all_loss_terms(forward, batch, arm)
    if not all(bool(torch.isfinite(value)) for value in terms.values()):
        raise RuntimeError(f"non-finite WGRQ loss at update {update_index}")
    terms["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError(f"non-finite WGRQ gradient at update {update_index}")
    optimizer.step()
    stats = {name: float(value.detach()) for name, value in terms.items()}
    stats["gradient_norm"] = float(gradient_norm)
    stats["learning_rate"] = learning_rate
    return stats


def _batch_order_sha256(batches: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for index in batch:
            digest.update(int(index).to_bytes(4, "big"))
    return digest.hexdigest()


def train_frozen_fit(
    dataset: FrozenWGRQDataset,
    binding: FrozenInputBinding,
    arm: str,
    seed: int,
) -> tuple[HardBitDWEPRLearner, dict[str, Any]]:
    if arm not in ARM_NAMES or seed not in FROZEN_SEEDS:
        raise ValueError("arm and seed must come from the frozen preregistration")
    if len(dataset.episodes) != TRAIN_EPISODES:
        raise ValueError("dataset does not contain the frozen episode count")

    batches = frozen_training_batches(seed)
    binding.assert_unchanged()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = HardBitDWEPRLearner().cpu()
    optimizer = build_optimizer(model)
    initial_hash = hash_model_state(model)
    final_stats: dict[str, float] = {}

    model.train()
    for update_index, indices in enumerate(batches):
        if update_index % BATCHES_PER_EPOCH == 0:
            binding.assert_unchanged()
        batch = collate_frozen_batch(dataset, indices)
        final_stats = run_training_update(model, optimizer, batch, arm, update_index)
    binding.assert_unchanged()

    metadata = {
        "protocol": "wgrq_dwepr_cpu_stage_a_v1",
        "arm": arm,
        "arm_coefficients": ARM_COEFFICIENTS[arm],
        "seed": seed,
        "paired_seeds": list(FROZEN_SEEDS),
        "transcript_sha256": binding.transcript_sha256,
        "audit_report_sha256": binding.audit_report_sha256,
        "episodes": TRAIN_EPISODES,
        "ordinary_answer_calls": TRAIN_ANSWER_CALLS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "updates": TOTAL_UPDATES,
        "batch_order_sha256": _batch_order_sha256(batches),
        "warmup_updates": WARMUP_UPDATES,
        "cosine_decay_to_zero": True,
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "epsilon": EPSILON,
        "matrix_weight_decay": MATRIX_WEIGHT_DECAY,
        "vector_weight_decay": 0.0,
        "gradient_clip": GRADIENT_CLIP,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_dtype": "torch.float32",
        "packet_bits": PACKET_BITS,
        "all_loss_terms_eager": True,
        "sham_partner_contract": (
            "batch-local cyclic derangement after lexicographic ordering by scale,"
            "length-band,event-count,fixed-sensor-endpoint,answer-signature,SHA256"
        ),
        "initial_model_sha256": initial_hash,
        "final_model_sha256": hash_model_state(model),
        "final_update": final_stats,
    }
    return model, metadata


def save_frozen_checkpoint(
    path: str | Path,
    model: HardBitDWEPRLearner,
    metadata: dict[str, Any],
) -> str:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing checkpoint: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {output.parent}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary checkpoint: {temporary}")
    try:
        torch.save(
            {
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "wgrq_cpu": metadata,
            },
            temporary,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", "--data", dest="transcript", required=True)
    parser.add_argument("--audit-report", "--audit", dest="audit_report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--seed", required=True, type=int, choices=FROZEN_SEEDS)
    args = parser.parse_args()

    binding = bind_frozen_inputs(args.transcript, args.audit_report)
    dataset = load_frozen_dataset(binding)
    model, metadata = train_frozen_fit(dataset, binding, args.arm, args.seed)
    binding.assert_unchanged()
    checkpoint_sha256 = save_frozen_checkpoint(args.out, model, metadata)
    print(json.dumps({"wgrq_cpu": metadata, "checkpoint_sha256": checkpoint_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
