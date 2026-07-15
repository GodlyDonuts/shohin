#!/usr/bin/env python3
"""Generate and evaluate the frozen WGRQ Stage-A CPU confirmation board.

Confirmation acquisition is gated on an exact manifest of all 60 final
checkpoint hashes.  Evaluation transports only a 15-bit packet from a fresh
writer child to separate fresh reader children.  Each reader receives fixed
weights, a public scale mask, the packet, and one continuation; it never
imports or receives the oracle, source history, identifiers, or verifier.

The only scoring unit emitted by this module is ``episode_exact``.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from wgrq_state_machine import (  # noqa: E402
    FLIP,
    PACKET_BITS,
    ROTATE,
    HardBitDWEPRLearner,
    WGRQReader,
    WGRQWriter,
    deserialize_packet,
    make_scale_mask,
)


CHECKPOINT_SCHEMA = "wgrq_checkpoint_hashes_v1"
CONFIRMATION_SCHEMA = "wgrq_stage_a_confirmation_v1"
REPORT_SCHEMA = "wgrq_confirmation_evaluation_v1"
ARMS = (
    "WGRQ-shortest",
    "active-answer-only",
    "uniform-witness",
    "relation-sham",
    "privileged-edge",
)
ARM_ALIASES = {
    "WGRQ-shortest": "WGRQ-shortest",
    "wgrq-shortest": "WGRQ-shortest",
    "wgrq_shortest": "WGRQ-shortest",
    "active-answer-only": "active-answer-only",
    "active_answer_only": "active-answer-only",
    "uniform-witness": "uniform-witness",
    "uniform_witness": "uniform-witness",
    "relation-sham": "relation-sham",
    "relation_sham": "relation-sham",
    "privileged-edge": "privileged-edge",
    "privileged_edge": "privileged-edge",
}
SCORE_ARM_NAMES = {
    "WGRQ-shortest": "wgrq-shortest",
    "active-answer-only": "active-answer-only",
    "uniform-witness": "uniform-witness",
    "relation-sham": "relation-sham",
    "privileged-edge": "privileged-edge",
}
SEEDS = (
    17011,
    27103,
    38119,
    49201,
    50311,
    61403,
    72503,
    83609,
    94709,
    105019,
    116027,
    127031,
)
EPISODES_PER_STRATUM = 1_024
HISTORIES_PER_EPISODE = 4
BRANCHES_PER_HISTORY = 32
ANSWERS_PER_EPISODE = HISTORIES_PER_EPISODE * BRANCHES_PER_HISTORY
TOTAL_CONFIRMATION_ANSWERS = 3 * EPISODES_PER_STRATUM * ANSWERS_PER_EPISODE
CONFIRMATION_SEED = hashlib.sha256(
    b"R12_WGRQ_CPU_PREREG.md\x00Stage-A confirmation v1"
).digest()

HISTORY_ROLES = (
    "equivalent_a",
    "equivalent_b",
    "non_equivalent_a",
    "non_equivalent_b",
)
WRITER_TASK_KEYS = frozenset(("role", "scale_mask", "source_events"))
READER_TASK_KEYS = frozenset(("role", "scale_mask", "packet", "continuation"))
ORACLE_MODULE_NAMES = frozenset(
    (
        "wgrq_residual_oracle",
        "generate_wgrq_falsifier_v1",
        "audit_wgrq_falsifier_v1",
        "score_wgrq_falsifier_v1",
    )
)


@dataclass(frozen=True)
class ConfirmationStratum:
    name: str
    n: int
    maximum_source_length: int
    minimum_source_length: int


STRATA = (
    ConfirmationStratum("length_ood", 8, 64 * 8, 8 * 8 + 1),
    ConfirmationStratum("scale_ood", 16, 8 * 16, 0),
    ConfirmationStratum("full_ood", 16, 64 * 16, 8 * 16 + 1),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    if "document_sha256" in document:
        raise ValueError("document is already sealed")
    sealed = dict(document)
    sealed["document_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return sealed


def _verify_document_seal(document: Mapping[str, Any]) -> None:
    claimed = document.get("document_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("sealed document is missing document_sha256")
    core = {key: value for key, value in document.items() if key != "document_sha256"}
    if sha256_bytes(canonical_json_bytes(core)) != claimed:
        raise ValueError("sealed document hash mismatch")


def _write_new_json(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite confirmation artifact: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii") + b"\n"
    with partial.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)


def _expected_checkpoint_pairs() -> frozenset[tuple[str, int]]:
    return frozenset((arm, seed) for arm in ARMS for seed in SEEDS)


def _canonical_arm(value: object) -> str:
    try:
        return ARM_ALIASES[str(value)]
    except KeyError as error:
        raise ValueError("unknown WGRQ arm: {!r}".format(value)) from error


def _checkpoint_rows(value: object) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("checkpoints")
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError("checkpoint index must contain a checkpoints list")
    return list(value)


def _build_checkpoint_hash_document(
    rows: Sequence[Mapping[str, Any]],
    expected_pairs: frozenset[tuple[str, int]],
) -> dict[str, Any]:
    normalized = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not {"arm", "seed", "path"}.issubset(row):
            raise ValueError("checkpoint entry requires arm, seed, and path")
        arm, seed = _canonical_arm(row["arm"]), int(row["seed"])
        pair = (arm, seed)
        if pair in seen:
            raise ValueError("duplicate checkpoint entry: {} seed {}".format(arm, seed))
        seen.add(pair)
        path = Path(str(row["path"])).expanduser().resolve()
        if not path.is_file():
            raise ValueError("checkpoint is not a regular file: {}".format(path))
        normalized.append(
            {
                "arm": arm,
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        extra = sorted(seen - expected_pairs)
        raise ValueError("checkpoint grid mismatch; missing={} extra={}".format(missing, extra))
    normalized.sort(key=lambda row: (ARMS.index(row["arm"]) if row["arm"] in ARMS else row["arm"], row["seed"]))
    return _seal_document(
        {
            "schema": CHECKPOINT_SCHEMA,
            "frozen": True,
            "arms": list(ARMS) if expected_pairs == _expected_checkpoint_pairs() else sorted({row["arm"] for row in normalized}),
            "seeds": list(SEEDS) if expected_pairs == _expected_checkpoint_pairs() else sorted({row["seed"] for row in normalized}),
            "checkpoint_count": len(normalized),
            "checkpoints": normalized,
        }
    )


def freeze_checkpoint_hashes(index_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Hash and freeze the exact preregistered 5-arm by 12-seed grid."""
    raw = json.loads(Path(index_path).read_text())
    document = _build_checkpoint_hash_document(_checkpoint_rows(raw), _expected_checkpoint_pairs())
    indexed = _validate_checkpoint_hash_document(
        document,
        _expected_checkpoint_pairs(),
        verify_files=True,
    )
    contracts = [
        _validate_final_checkpoint(indexed[pair]["path"], pair[0], pair[1])
        for pair in sorted(indexed, key=lambda item: (ARMS.index(item[0]), item[1]))
    ]
    transcript_hashes = {contract["transcript_sha256"] for contract in contracts}
    audit_hashes = {contract["audit_report_sha256"] for contract in contracts}
    if len(transcript_hashes) != 1 or len(audit_hashes) != 1:
        raise ValueError("all 60 checkpoints must bind the same byte-identical transcript and audit")
    _validate_checkpoint_hash_document(
        document,
        _expected_checkpoint_pairs(),
        verify_files=True,
    )
    core = {key: value for key, value in document.items() if key != "document_sha256"}
    core.update(
        {
            "checkpoint_contract_validated": True,
            "training_transcript_sha256": next(iter(transcript_hashes)),
            "training_audit_report_sha256": next(iter(audit_hashes)),
            "final_model_hashes_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "arm": contract["arm"],
                            "seed": contract["seed"],
                            "final_model_sha256": contract["final_model_sha256"],
                        }
                        for contract in contracts
                    ]
                )
            ),
        }
    )
    document = _seal_document(core)
    _write_new_json(output_path, document)
    return document


def _validate_checkpoint_hash_document(
    document: Mapping[str, Any],
    expected_pairs: frozenset[tuple[str, int]],
    *,
    verify_files: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    _verify_document_seal(document)
    if document.get("schema") != CHECKPOINT_SCHEMA or document.get("frozen") is not True:
        raise ValueError("checkpoint hashes are not a frozen WGRQ manifest")
    rows = document.get("checkpoints")
    if not isinstance(rows, list) or int(document.get("checkpoint_count", -1)) != len(rows):
        raise ValueError("checkpoint hash manifest count mismatch")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"arm", "seed", "path", "sha256"}:
            raise ValueError("invalid checkpoint hash entry")
        arm, seed = _canonical_arm(row["arm"]), int(row["seed"])
        pair = (arm, seed)
        digest = str(row["sha256"])
        if pair in indexed or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("invalid or duplicate checkpoint hash entry")
        path = Path(str(row["path"]))
        if not path.is_absolute():
            raise ValueError("frozen checkpoint paths must be absolute")
        if verify_files and (not path.is_file() or sha256_file(path) != digest):
            raise ValueError("frozen checkpoint changed or disappeared: {}".format(path))
        indexed[pair] = dict(row)
    if frozenset(indexed) != expected_pairs:
        raise ValueError("frozen checkpoint manifest does not cover the exact checkpoint grid")
    return indexed


def load_checkpoint_hashes(path: str | Path, *, verify_files: bool = True) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    document = json.loads(Path(path).read_text())
    indexed = _validate_checkpoint_hash_document(
        document,
        _expected_checkpoint_pairs(),
        verify_files=verify_files,
    )
    if document.get("checkpoint_contract_validated") is not True:
        raise ValueError("checkpoint hash manifest lacks final-checkpoint contract validation")
    contracts = [
        _validate_final_checkpoint(indexed[pair]["path"], pair[0], pair[1])
        for pair in sorted(indexed, key=lambda item: (ARMS.index(item[0]), item[1]))
    ]
    transcript_hashes = {contract["transcript_sha256"] for contract in contracts}
    audit_hashes = {contract["audit_report_sha256"] for contract in contracts}
    if transcript_hashes != {document.get("training_transcript_sha256")}:
        raise ValueError("frozen checkpoints no longer share the recorded training transcript")
    if audit_hashes != {document.get("training_audit_report_sha256")}:
        raise ValueError("frozen checkpoints no longer share the recorded training audit")
    model_commitment = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "arm": contract["arm"],
                    "seed": contract["seed"],
                    "final_model_sha256": contract["final_model_sha256"],
                }
                for contract in contracts
            ]
        )
    )
    if model_commitment != document.get("final_model_hashes_sha256"):
        raise ValueError("final checkpoint model hashes changed after freeze")
    return document, indexed


class Sha256PRF:
    """Counter-mode SHA256 with unbiased finite-bank selection."""

    def __init__(self, seed: bytes, domain: str) -> None:
        if not isinstance(seed, bytes) or not seed:
            raise ValueError("PRF seed must be nonempty bytes")
        try:
            self._domain = str(domain).encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("PRF domain must be ASCII") from error
        self._seed = seed
        self._counter = 0

    def block(self) -> bytes:
        if self._counter >= 1 << 64:
            raise OverflowError("PRF counter exhausted")
        result = hashlib.sha256(
            self._seed + b"\x00" + self._domain + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        return result

    def randbelow(self, size: int) -> int:
        size = int(size)
        if size <= 0:
            raise ValueError("finite bank size must be positive")
        limit = (1 << 256) - ((1 << 256) % size)
        while True:
            candidate = int.from_bytes(self.block(), "big")
            if candidate < limit:
                return candidate % size

    def shuffle(self, values: Sequence[Any]) -> list[Any]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            selected = self.randbelow(index + 1)
            result[index], result[selected] = result[selected], result[index]
        return result


def _oracle_module():
    return importlib.import_module("wgrq_residual_oracle")


class OracleFacade:
    """Narrow adapter over the shared symbolic oracle module."""

    def __init__(self, module: Any | None = None) -> None:
        self.module = module if module is not None else _oracle_module()

    @staticmethod
    def _event_names(events: Sequence[int]) -> tuple[str, ...]:
        names = []
        for event in events:
            if int(event) == ROTATE:
                names.append("R")
            elif int(event) == FLIP:
                names.append("F")
            else:
                raise ValueError("event code must be ROTATE=0 or FLIP=1")
        return tuple(names)

    def state_after(self, n: int, events: Sequence[int]):
        names = self._event_names(events)
        return self.module.apply_word(0, names, n)

    def answer_after(self, state: Any, continuation: Sequence[int], n: int) -> int:
        names = self._event_names(continuation)
        return int(self.module.read(self.module.apply_word(state, names, n), n))


def _bits_from_integer(value: int, n: int) -> tuple[int, ...]:
    return tuple((int(value) >> index) & 1 for index in range(n))


def _history_for_state(bits: Sequence[int]) -> list[int]:
    history = []
    for bit in bits:
        if int(bit) not in (0, 1):
            raise ValueError("physical state bits must be binary")
        if bit:
            history.append(FLIP)
        history.append(ROTATE)
    return history


def _edge_difference_state(n: int, witness_depth: int, second_edge: int) -> tuple[int, ...]:
    if not (0 <= witness_depth < second_edge < n):
        raise ValueError("invalid two-edge witness construction")
    edges = [0] * n
    edges[witness_depth] = 1
    edges[second_edge] = 1
    bits = [0]
    for edge in edges[:-1]:
        bits.append(bits[-1] ^ edge)
    if (bits[-1] ^ bits[0]) != edges[-1]:
        raise AssertionError("edge construction did not close around the ring")
    return tuple(bits)


def _pad_identity(
    history: Sequence[int],
    minimum_length: int,
    maximum_length: int,
    prf: Sha256PRF,
) -> list[int]:
    history = list(history)
    lower = max(len(history), int(minimum_length))
    if lower > maximum_length:
        raise ValueError("canonical source history exceeds stratum maximum")
    if (lower - len(history)) % 2:
        lower += 1
    candidates = list(range(lower, maximum_length + 1, 2))
    if not candidates:
        raise ValueError("no parity-compatible source length in stratum")
    target = candidates[prf.randbelow(len(candidates))]
    history.extend((FLIP, FLIP) * ((target - len(history)) // 2))
    return history


def _make_confirmation_episode(
    stratum: ConfirmationStratum,
    index: int,
    oracle: OracleFacade,
) -> dict[str, Any]:
    n = stratum.n
    prf = Sha256PRF(CONFIRMATION_SEED, "{}:{}".format(stratum.name, index))
    equivalent_base = _bits_from_integer(prf.randbelow(1 << n), n)
    equivalent_complement = tuple(1 ^ bit for bit in equivalent_base)
    non_equivalent_base = _bits_from_integer(prf.randbelow(1 << n), n)
    if stratum.name == "full_ood":
        witness_depth = n - 2 if index == 0 else (index - 1) % (n - 1)
    else:
        witness_depth = index % (n - 1)
    second_edge = witness_depth + 1 + prf.randbelow(n - witness_depth - 1)
    difference = _edge_difference_state(n, witness_depth, second_edge)
    non_equivalent_donor = tuple(
        bit ^ delta for bit, delta in zip(non_equivalent_base, difference)
    )
    physical_states = (
        equivalent_base,
        equivalent_complement,
        non_equivalent_base,
        non_equivalent_donor,
    )
    histories = [
        _pad_identity(
            _history_for_state(bits),
            stratum.minimum_source_length,
            stratum.maximum_source_length,
            prf,
        )
        for bits in physical_states
    ]
    endpoints = [oracle.state_after(n, history) for history in histories]
    if histories[0] == histories[1]:
        raise RuntimeError("confirmation equivalent histories are not distinct")

    rotations = list(range(n)) * (BRANCHES_PER_HISTORY // n)
    rotations = prf.shuffle(rotations)
    continuations = [[ROTATE] * count for count in rotations]
    answers = []
    oracle_answer_calls = 0
    for endpoint in endpoints:
        history_answers = []
        for continuation in continuations:
            answer = oracle.answer_after(endpoint, continuation, n)
            oracle_answer_calls += 1
            if answer not in (0, 1):
                raise RuntimeError("ordinary WGRQ oracle READ was not one bit")
            history_answers.append(answer)
        answers.append(history_answers)
    witness_indices = [position for position, count in enumerate(rotations) if count == witness_depth]
    if not witness_indices or any(answers[2][position] == answers[3][position] for position in witness_indices):
        raise RuntimeError("selected non-equivalent witness does not distinguish donor histories")
    if any(answers[0][position] != answers[1][position] for position in range(BRANCHES_PER_HISTORY)):
        raise RuntimeError("equivalent histories disagree on a confirmation branch")
    signatures = []
    for history_answers in answers:
        signature = []
        for rotation in range(n):
            values = {
                history_answers[position]
                for position, count in enumerate(rotations)
                if count == rotation
            }
            if len(values) != 1:
                raise RuntimeError("repeated public READ branches disagree")
            signature.append(next(iter(values)))
        signatures.append(signature)
    public_differences = [
        signatures[2][rotation] ^ signatures[3][rotation]
        for rotation in range(n - 1)
    ]
    if not any(public_differences) or public_differences.index(1) != witness_depth:
        raise RuntimeError("public answers do not derive the scheduled shortest witness")
    if signatures[0][: n - 1] != signatures[1][: n - 1]:
        raise RuntimeError("public answers do not derive the scheduled equivalence label")
    if oracle_answer_calls != ANSWERS_PER_EPISODE:
        raise AssertionError("confirmation episode oracle-call count drifted")

    return {
        "id": "{}-{:04d}".format(stratum.name, index),
        "stratum": stratum.name,
        "n": n,
        "maximum_source_length": stratum.maximum_source_length,
        "equivalent_pair": [0, 1],
        "non_equivalent_pair": [2, 3],
        "non_equivalent_witness_depth": witness_depth,
        "continuations": continuations,
        "histories": [
            {
                "role": role,
                "source_events": history,
                "expected_answers": expected,
            }
            for role, history, expected in zip(HISTORY_ROLES, histories, answers)
        ],
        "ordinary_oracle_answer_calls": oracle_answer_calls,
    }


def _build_confirmation_document(
    checkpoint_hash_document: Mapping[str, Any],
    *,
    episodes_per_stratum: int,
    oracle: OracleFacade | None = None,
) -> dict[str, Any]:
    if episodes_per_stratum <= 0:
        raise ValueError("episodes_per_stratum must be positive")
    oracle = oracle or OracleFacade()
    episodes = []
    for stratum in STRATA:
        for index in range(episodes_per_stratum):
            episodes.append(_make_confirmation_episode(stratum, index, oracle))
    total_calls = sum(int(episode["ordinary_oracle_answer_calls"]) for episode in episodes)
    expected_calls = len(STRATA) * episodes_per_stratum * ANSWERS_PER_EPISODE
    if total_calls != expected_calls:
        raise AssertionError("confirmation acquisition call count drifted")
    document = _seal_document(
        {
            "schema": CONFIRMATION_SCHEMA,
            "checkpoint_hashes_document_sha256": checkpoint_hash_document["document_sha256"],
            "generation_prf": "SHA256(seed || 0x00 || ASCII(domain) || uint64_be(counter))",
            "generation_seed_sha256": sha256_bytes(CONFIRMATION_SEED),
            "episodes_per_stratum": episodes_per_stratum,
            "histories_per_episode": HISTORIES_PER_EPISODE,
            "branches_per_history": BRANCHES_PER_HISTORY,
            "score_unit": "episode_exact",
            "strata": [
                {
                    "name": stratum.name,
                    "n": stratum.n,
                    "minimum_source_length": stratum.minimum_source_length,
                    "maximum_source_length": stratum.maximum_source_length,
                }
                for stratum in STRATA
            ],
            "ordinary_oracle_answer_calls": total_calls,
            "oracle_call_ledger": {
                "ordinary_one_bit_answer_calls": total_calls,
                "equivalence_oracle_calls": 0,
                "witness_oracle_calls": 0,
                "counterexample_oracle_calls": 0,
                "model_dependent_mining_calls": 0,
                "hidden_state_ids_serialized": 0,
            },
            "episodes": episodes,
        }
    )
    _validate_confirmation_document(
        document,
        checkpoint_hash_document,
        expected_episodes_per_stratum=episodes_per_stratum,
    )
    return document


def generate_confirmation(
    checkpoint_hashes_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Acquire the single immutable confirmation board after all hashes freeze."""
    checkpoint_document, _ = load_checkpoint_hashes(checkpoint_hashes_path, verify_files=True)
    document = _build_confirmation_document(
        checkpoint_document,
        episodes_per_stratum=EPISODES_PER_STRATUM,
    )
    if int(document["ordinary_oracle_answer_calls"]) != TOTAL_CONFIRMATION_ANSWERS:
        raise AssertionError("full confirmation must contain exactly 393,216 ordinary answers")
    after_document, _ = load_checkpoint_hashes(checkpoint_hashes_path, verify_files=True)
    if after_document["document_sha256"] != checkpoint_document["document_sha256"]:
        raise RuntimeError("checkpoint hash manifest changed during confirmation acquisition")
    _write_new_json(output_path, document)
    return document


def _validate_confirmation_document(
    document: Mapping[str, Any],
    checkpoint_hash_document: Mapping[str, Any],
    *,
    expected_episodes_per_stratum: int,
) -> None:
    _verify_document_seal(document)
    if document.get("schema") != CONFIRMATION_SCHEMA:
        raise ValueError("not a WGRQ Stage-A confirmation document")
    if document.get("checkpoint_hashes_document_sha256") != checkpoint_hash_document.get("document_sha256"):
        raise ValueError("confirmation was not generated after this frozen checkpoint grid")
    if document.get("generation_seed_sha256") != sha256_bytes(CONFIRMATION_SEED):
        raise ValueError("confirmation generation seed changed")
    if document.get("score_unit") != "episode_exact":
        raise ValueError("episode_exact must be the sole confirmation scoring unit")
    if int(document.get("episodes_per_stratum", -1)) != expected_episodes_per_stratum:
        raise ValueError("confirmation episode count changed")
    if int(document.get("histories_per_episode", -1)) != HISTORIES_PER_EPISODE:
        raise ValueError("confirmation history count changed")
    if int(document.get("branches_per_history", -1)) != BRANCHES_PER_HISTORY:
        raise ValueError("confirmation branch count changed")
    expected_specs = [
        {
            "name": stratum.name,
            "n": stratum.n,
            "minimum_source_length": stratum.minimum_source_length,
            "maximum_source_length": stratum.maximum_source_length,
        }
        for stratum in STRATA
    ]
    if document.get("strata") != expected_specs:
        raise ValueError("confirmation strata changed")
    expected_ledger = {
        "ordinary_one_bit_answer_calls": len(STRATA)
        * expected_episodes_per_stratum
        * ANSWERS_PER_EPISODE,
        "equivalence_oracle_calls": 0,
        "witness_oracle_calls": 0,
        "counterexample_oracle_calls": 0,
        "model_dependent_mining_calls": 0,
        "hidden_state_ids_serialized": 0,
    }
    if document.get("oracle_call_ledger") != expected_ledger:
        raise ValueError("confirmation oracle-call ledger changed")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(STRATA) * expected_episodes_per_stratum:
        raise ValueError("confirmation episode geometry changed")

    by_stratum = collections.Counter()
    witness_depths: dict[str, set[int]] = collections.defaultdict(set)
    total_calls = 0
    spec_by_name = {stratum.name: stratum for stratum in STRATA}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("confirmation episode must be an object")
        required_keys = {
            "id",
            "stratum",
            "n",
            "maximum_source_length",
            "equivalent_pair",
            "non_equivalent_pair",
            "non_equivalent_witness_depth",
            "continuations",
            "histories",
            "ordinary_oracle_answer_calls",
        }
        if set(episode) != required_keys:
            raise ValueError("confirmation episode fields changed")
        name = str(episode["stratum"])
        if name not in spec_by_name:
            raise ValueError("unknown confirmation stratum")
        spec = spec_by_name[name]
        if int(episode["n"]) != spec.n or int(episode["maximum_source_length"]) != spec.maximum_source_length:
            raise ValueError("confirmation episode scale changed")
        by_stratum[name] += 1
        witness = int(episode["non_equivalent_witness_depth"])
        if not 0 <= witness <= spec.n - 2:
            raise ValueError("confirmation witness depth is outside the determining bank")
        witness_depths[name].add(witness)
        if episode["equivalent_pair"] != [0, 1] or episode["non_equivalent_pair"] != [2, 3]:
            raise ValueError("confirmation pair roles changed")
        continuations = episode["continuations"]
        if not isinstance(continuations, list) or len(continuations) != BRANCHES_PER_HISTORY:
            raise ValueError("each committed history requires exactly 32 branches")
        rotation_counts = []
        for continuation in continuations:
            if not isinstance(continuation, list) or any(int(event) != ROTATE for event in continuation):
                raise ValueError("confirmation continuations must be rotation-only READ branches")
            rotation_counts.append(len(continuation))
        expected_rotation_counts = sorted(list(range(spec.n)) * (BRANCHES_PER_HISTORY // spec.n))
        if sorted(rotation_counts) != expected_rotation_counts:
            raise ValueError("confirmation continuation bank changed")
        histories = episode["histories"]
        if not isinstance(histories, list) or len(histories) != HISTORIES_PER_EPISODE:
            raise ValueError("confirmation episode requires exactly four histories")
        if [history.get("role") for history in histories if isinstance(history, Mapping)] != list(HISTORY_ROLES):
            raise ValueError("confirmation history roles changed")
        for history in histories:
            if not isinstance(history, Mapping) or set(history) != {"role", "source_events", "expected_answers"}:
                raise ValueError("invalid confirmation history")
            source_events = history["source_events"]
            expected_answers = history["expected_answers"]
            if not isinstance(source_events, list) or any(int(event) not in (ROTATE, FLIP) for event in source_events):
                raise ValueError("invalid confirmation source history")
            if not spec.minimum_source_length <= len(source_events) <= spec.maximum_source_length:
                raise ValueError("confirmation source history is outside its OOD band")
            if not isinstance(expected_answers, list) or len(expected_answers) != BRANCHES_PER_HISTORY:
                raise ValueError("confirmation history must have exactly 32 ordinary answers")
            if any(int(answer) not in (0, 1) for answer in expected_answers):
                raise ValueError("confirmation ordinary answers must be one bit")
        witness_positions = [position for position, count in enumerate(rotation_counts) if count == witness]
        if not witness_positions or any(
            histories[2]["expected_answers"][position] == histories[3]["expected_answers"][position]
            for position in witness_positions
        ):
            raise ValueError("non-equivalent donor witness no longer distinguishes")
        if any(
            histories[0]["expected_answers"][position] != histories[1]["expected_answers"][position]
            for position in range(BRANCHES_PER_HISTORY)
        ):
            raise ValueError("equivalent histories no longer have interchangeable answers")
        calls = int(episode["ordinary_oracle_answer_calls"])
        if calls != ANSWERS_PER_EPISODE:
            raise ValueError("confirmation episode must contain exactly 128 ordinary answers")
        total_calls += calls

    if any(by_stratum[stratum.name] != expected_episodes_per_stratum for stratum in STRATA):
        raise ValueError("confirmation does not contain 1,024 episodes in every stratum")
    for stratum in STRATA:
        required_depths = set(range(stratum.n - 1))
        if expected_episodes_per_stratum >= stratum.n - 1 and witness_depths[stratum.name] != required_depths:
            raise ValueError("confirmation does not cover every shortest-witness depth")
    if spec_by_name["full_ood"].n - 2 not in witness_depths["full_ood"]:
        raise ValueError("full OOD confirmation omits the tight n-2 witness")
    expected_calls = len(STRATA) * expected_episodes_per_stratum * ANSWERS_PER_EPISODE
    if total_calls != expected_calls or int(document.get("ordinary_oracle_answer_calls", -1)) != expected_calls:
        raise ValueError("confirmation ordinary-answer ledger changed")


def _torch_load(path_or_buffer: Any) -> Any:
    try:
        return torch.load(path_or_buffer, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path_or_buffer, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    candidates = []
    if isinstance(checkpoint, Mapping):
        for key in ("model", "model_state", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                candidates.append(value)
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            candidates.append(checkpoint)
    for candidate in candidates:
        if candidate and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in candidate.items()):
            return candidate
    raise ValueError("checkpoint does not contain a WGRQ model state dict")


def _validate_checkpoint_metadata(checkpoint: Any, arm: str, seed: int) -> None:
    if not isinstance(checkpoint, Mapping):
        return
    metadata_candidates = [checkpoint]
    for key in ("wgrq", "wgrq_cpu", "wgrq_stage_a"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            metadata_candidates.append(value)
    for metadata in metadata_candidates:
        if "arm" in metadata and _canonical_arm(metadata["arm"]) != _canonical_arm(arm):
            raise ValueError("checkpoint arm metadata does not match frozen manifest")
        if "seed" in metadata and int(metadata["seed"]) != int(seed):
            raise ValueError("checkpoint seed metadata does not match frozen manifest")
        if "packet_bits" in metadata and int(metadata["packet_bits"]) != PACKET_BITS:
            raise ValueError("checkpoint packet width is not exactly 15 bits")


def _training_model_state_hash(model: HardBitDWEPRLearner) -> str:
    """Match the final_model_sha256 contract written by train_wgrq_cpu.py."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(str(value.dtype).encode("ascii") + b"\x00")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\x00")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _lowercase_sha256(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return digest


def _validate_final_checkpoint(path: str | Path, arm: str, seed: int) -> dict[str, Any]:
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("wgrq_cpu"), Mapping):
        raise ValueError("final checkpoint lacks wgrq_cpu metadata")
    metadata = checkpoint["wgrq_cpu"]
    _validate_checkpoint_metadata(checkpoint, arm, seed)
    required_contract = {
        "protocol": "wgrq_dwepr_cpu_stage_a_v1",
        "episodes": 18_432,
        "ordinary_answer_calls": 589_824,
        "batch_size": 64,
        "epochs": 4,
        "updates": 1_152,
        "warmup_updates": 64,
        "parameter_count": 5_136,
        "parameter_dtype": "torch.float32",
        "packet_bits": PACKET_BITS,
        "all_loss_terms_eager": True,
    }
    for key, expected in required_contract.items():
        if metadata.get(key) != expected:
            raise ValueError("final checkpoint {} contract changed".format(key))
    if tuple(int(value) for value in metadata.get("paired_seeds", ())) != SEEDS:
        raise ValueError("final checkpoint paired-seed grid changed")
    state_dict = _extract_state_dict(checkpoint)
    model = HardBitDWEPRLearner().cpu().eval()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("final checkpoint model state is incompatible")
    final_hash = _training_model_state_hash(model)
    if _lowercase_sha256(metadata.get("final_model_sha256"), "final_model_sha256") != final_hash:
        raise ValueError("final checkpoint model hash does not match its fixed weights")
    return {
        "arm": _canonical_arm(arm),
        "seed": int(seed),
        "transcript_sha256": _lowercase_sha256(metadata.get("transcript_sha256"), "transcript_sha256"),
        "audit_report_sha256": _lowercase_sha256(metadata.get("audit_report_sha256"), "audit_report_sha256"),
        "final_model_sha256": final_hash,
    }


def checkpoint_weights_wire(path: str | Path, arm: str, seed: int) -> tuple[str, str]:
    checkpoint = _torch_load(path)
    _validate_checkpoint_metadata(checkpoint, arm, seed)
    state_dict = _extract_state_dict(checkpoint)
    model = HardBitDWEPRLearner().cpu().eval()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("checkpoint model state is incompatible with the frozen WGRQ learner")
    model.assert_contract()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=False)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), sha256_bytes(payload)


def _model_from_wire(weights_b64: str, weights_sha256: str) -> HardBitDWEPRLearner:
    try:
        payload = base64.b64decode(weights_b64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError("invalid fixed-weight transport") from error
    if sha256_bytes(payload) != weights_sha256:
        raise ValueError("fixed weights changed across the process boundary")
    state_dict = _torch_load(io.BytesIO(payload))
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()
    ):
        raise ValueError("fixed-weight transport did not contain a state dict")
    model = HardBitDWEPRLearner().cpu().eval()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("fixed weights are incompatible with the WGRQ learner")
    model.assert_contract()
    return model


def _model_fingerprint(model: HardBitDWEPRLearner) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _packet_bytes(packet: Sequence[int]) -> bytes:
    frozen = tuple(int(value) for value in packet)
    if len(frozen) != PACKET_BITS or any(value not in (0, 1) for value in frozen):
        raise ValueError("packet must serialize exactly 15 binary values")
    return bytes(frozen)


def _mask_from_task(task: Mapping[str, Any]) -> torch.Tensor:
    values = task.get("scale_mask")
    if not isinstance(values, list) or len(values) != PACKET_BITS:
        raise ValueError("public scale mask must contain exactly 15 values")
    mask = torch.tensor(values, dtype=torch.float32)
    active = int(mask.sum().item())
    expected = make_scale_mask(active + 1)
    if not torch.equal(mask, expected):
        raise ValueError("public scale mask is not a canonical prefix mask")
    return mask


def _writer_child_task(model: HardBitDWEPRLearner, task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != WRITER_TASK_KEYS or task.get("role") != "write":
        raise ValueError("writer task fields are not allowlisted")
    source_events = task.get("source_events")
    if not isinstance(source_events, list) or any(int(event) not in (ROTATE, FLIP) for event in source_events):
        raise ValueError("writer requires one binary-coded source history")
    scale_mask = _mask_from_task(task)
    before_weights = _model_fingerprint(model)
    writer = WGRQWriter(model, scale_mask)
    packet = writer.write(torch.tensor(source_events, dtype=torch.long))
    packet_payload = _packet_bytes(packet)
    after_weights = _model_fingerprint(model)
    return {
        "role": "writer_result",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "input_keys": sorted(task),
        "packet": list(packet),
        "packet_sha256": sha256_bytes(packet_payload),
        "exactly_15_serialized_bits": len(packet) == PACKET_BITS,
        "masked_bits_zero": all(
            int(packet[index]) == 0 for index, active in enumerate(scale_mask.tolist()) if not active
        ),
        "model_weights_unchanged": before_weights == after_weights,
    }


def _oracle_modules_absent() -> bool:
    loaded = {name.rsplit(".", 1)[-1] for name in sys.modules}
    return loaded.isdisjoint(ORACLE_MODULE_NAMES)


def _reader_child_task(model: HardBitDWEPRLearner, task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != READER_TASK_KEYS or task.get("role") != "read":
        raise ValueError("reader task fields are not allowlisted")
    if not _oracle_modules_absent():
        raise RuntimeError("reader process imported an oracle, generator, auditor, or verifier")
    scale_mask = _mask_from_task(task)
    packet = tuple(task.get("packet", ()))
    packet_payload = _packet_bytes(packet)
    deserialize_packet(packet, scale_mask, device="cpu")
    continuation = task.get("continuation")
    if not isinstance(continuation, list) or any(int(event) not in (ROTATE, FLIP) for event in continuation):
        raise ValueError("reader requires exactly one valid continuation")

    packet_digest = sha256_bytes(packet_payload)
    before_weights = _model_fingerprint(model)
    before_rng = torch.random.get_rng_state().clone()
    model_buffers_absent = not any(True for _ in model.named_buffers())
    reader = WGRQReader(model, scale_mask)
    probability = reader.read_probability(
        packet,
        torch.tensor(continuation, dtype=torch.long),
    )
    prediction = int(probability >= 0.5)
    packet_digest_after = sha256_bytes(_packet_bytes(packet))
    after_weights = _model_fingerprint(model)
    rng_state_unchanged = torch.equal(before_rng, torch.random.get_rng_state())
    packet_reuse_identical = packet_digest_after == packet_digest
    return {
        "role": "reader_result",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "input_keys": sorted(task),
        "prediction": prediction,
        "packet_sha256_before": packet_digest,
        "packet_sha256_after": packet_digest_after,
        "packet_reuse_identical": packet_reuse_identical,
        "exactly_15_serialized_bits": len(packet) == PACKET_BITS,
        "masked_bits_zero": all(
            int(packet[index]) == 0 for index, active in enumerate(scale_mask.tolist()) if not active
        ),
        "forbidden_reader_inputs_absent": set(task) == READER_TASK_KEYS,
        "oracle_modules_absent": _oracle_modules_absent(),
        "model_weights_unchanged": before_weights == after_weights,
        "model_buffers_absent": model_buffers_absent,
        "rng_state_unchanged": rng_state_unchanged,
        "cross_branch_memory_absent": packet_reuse_identical
        and before_weights == after_weights
        and model_buffers_absent
        and rng_state_unchanged,
    }


def _write_broker_response(response: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _broker_main(role: str) -> int:
    if role not in ("writer", "reader") or not hasattr(os, "fork"):
        raise RuntimeError("WGRQ process brokers require POSIX fork and an explicit role")
    torch.set_num_threads(1)
    initialization_line = sys.stdin.readline()
    if not initialization_line:
        raise RuntimeError("broker did not receive fixed weights")
    initialization = json.loads(initialization_line)
    if set(initialization) != {"role", "weights_b64", "weights_sha256"} or initialization.get("role") != "init":
        raise ValueError("invalid broker initialization")
    if role == "reader" and not _oracle_modules_absent():
        raise RuntimeError("reader broker imported a forbidden oracle-side module")
    model = _model_from_wire(
        str(initialization["weights_b64"]),
        str(initialization["weights_sha256"]),
    )
    fixed_fingerprint = _model_fingerprint(model)
    initialization = None
    initialization_line = ""
    _write_broker_response(
        {
            "role": role + "_broker_ready",
            "pid": os.getpid(),
            "oracle_modules_absent": _oracle_modules_absent(),
            "fixed_weights_sha256": fixed_fingerprint,
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        task = json.loads(line)
        if task == {"role": "stop"}:
            _write_broker_response({"role": role + "_broker_stopped", "pid": os.getpid()})
            return 0
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                result = _writer_child_task(model, task) if role == "writer" else _reader_child_task(model, task)
                result["broker_pid"] = os.getppid()
                payload = canonical_json_bytes(result) + b"\n"
            except BaseException as error:
                payload = canonical_json_bytes(
                    {
                        "role": role + "_error",
                        "pid": os.getpid(),
                        "broker_pid": os.getppid(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                ) + b"\n"
            with os.fdopen(write_fd, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
            os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb", closefd=True) as source:
            payload = source.read()
        waited_pid, status = os.waitpid(child_pid, 0)
        if waited_pid != child_pid or status != 0:
            raise RuntimeError("{} child process failed with status {}".format(role, status))
        response = json.loads(payload)
        _write_broker_response(response)
        task = None
        line = ""
    return 0


class ProcessBroker:
    """Clean supervisor that forks one short-lived process per history."""

    def __init__(self, role: str, weights_b64: str, weights_sha256: str, *, timeout: float = 120.0) -> None:
        if role not in ("writer", "reader"):
            raise ValueError("broker role must be writer or reader")
        self.role = role
        self.timeout = float(timeout)
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_broker", role],
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._send(
                {
                    "role": "init",
                    "weights_b64": weights_b64,
                    "weights_sha256": weights_sha256,
                }
            )
            ready = self._receive()
        except BaseException:
            self.close(force=True)
            raise
        if ready.get("role") != role + "_broker_ready" or not ready.get("oracle_modules_absent"):
            self.close(force=True)
            raise RuntimeError("{} broker failed its clean-process preflight".format(role))
        self.pid = int(ready["pid"])
        self.fixed_weights_sha256 = str(ready["fixed_weights_sha256"])

    def _send(self, value: Mapping[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("broker stdin is closed")
        self.process.stdin.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("broker stdout is closed")
        readable, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not readable:
            raise RuntimeError("{} broker response timed out".format(self.role))
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError("{} broker exited without a response: {}".format(self.role, stderr.strip()))
        response = json.loads(line)
        if response.get("role") == self.role + "_error":
            raise RuntimeError(
                "{} child {}: {}".format(self.role, response.get("error_type"), response.get("error"))
            )
        return response

    def request(self, task: Mapping[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError("{} broker is not running".format(self.role))
        self._send(task)
        response = self._receive()
        if int(response.get("broker_pid", -1)) != self.pid:
            raise RuntimeError("{} result did not come from its clean broker".format(self.role))
        return response

    def _close_pipes(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def close(self, *, force: bool = False) -> None:
        if not hasattr(self, "process"):
            return
        if self.process.poll() is not None:
            self._close_pipes()
            return
        if not force:
            try:
                self._send({"role": "stop"})
                response = self._receive()
                if response.get("role") != self.role + "_broker_stopped":
                    raise RuntimeError("invalid broker shutdown response")
                self.process.wait(timeout=self.timeout)
                self._close_pipes()
                return
            except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                pass
        self.process.kill()
        self.process.wait(timeout=self.timeout)
        self._close_pipes()

    def __enter__(self) -> "ProcessBroker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(force=exc_type is not None)


class WriterReaderBoundary:
    def __init__(self, weights_b64: str, weights_sha256: str) -> None:
        self.reader = ProcessBroker("reader", weights_b64, weights_sha256)
        try:
            self.writer = ProcessBroker("writer", weights_b64, weights_sha256)
        except BaseException:
            self.reader.close(force=True)
            raise
        if self.reader.fixed_weights_sha256 != self.writer.fixed_weights_sha256:
            self.close(force=True)
            raise RuntimeError("writer and reader loaded different fixed weights")

    def close(self, *, force: bool = False) -> None:
        self.writer.close(force=force)
        self.reader.close(force=force)

    def __enter__(self) -> "WriterReaderBoundary":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(force=exc_type is not None)


def _binary_mask(n: int) -> list[int]:
    return [int(value) for value in make_scale_mask(n).tolist()]


def _response_packet_checks(response: Mapping[str, Any], scale_mask: Sequence[int]) -> tuple[tuple[int, ...], bool, bool]:
    packet = tuple(int(value) for value in response.get("packet", ()))
    exact = (
        len(packet) == PACKET_BITS
        and all(value in (0, 1) for value in packet)
        and bool(response.get("exactly_15_serialized_bits"))
        and response.get("packet_sha256") == sha256_bytes(_packet_bytes(packet))
    )
    masked = all(packet[index] == 0 for index, active in enumerate(scale_mask) if not active)
    masked = masked and bool(response.get("masked_bits_zero"))
    return packet, exact, masked


def score_episode(
    episode: Mapping[str, Any],
    writer_request,
    reader_request,
) -> dict[str, Any]:
    """Evaluate one four-history cluster without emitting probe-level scores."""
    n = int(episode["n"])
    scale_mask = _binary_mask(n)
    histories = list(episode["histories"])
    continuations = list(episode["continuations"])
    packets = []
    writer_results = []
    reader_results: list[list[dict[str, Any]]] = []
    exact_width = True
    masked_bits_zero = True

    for history in histories:
        writer_result = writer_request(
            {
                "role": "write",
                "scale_mask": scale_mask,
                "source_events": list(history["source_events"]),
            }
        )
        packet, exact, masked = _response_packet_checks(writer_result, scale_mask)
        packets.append(packet)
        writer_results.append(writer_result)
        exact_width = exact_width and exact
        masked_bits_zero = masked_bits_zero and masked

        branch_results = []
        for continuation in continuations:
            reader_result = reader_request(
                {
                    "role": "read",
                    "scale_mask": scale_mask,
                    "packet": list(packet),
                    "continuation": list(continuation),
                }
            )
            if reader_result.get("role") != "reader_result":
                raise RuntimeError("reader did not return a reader_result")
            branch_results.append(reader_result)
        reader_results.append(branch_results)

    predictions = [
        [int(result.get("prediction", -1)) for result in branch_results]
        for branch_results in reader_results
    ]
    expected = [list(history["expected_answers"]) for history in histories]
    ordinary_answer_geometry = all(
        len(values) == BRANCHES_PER_HISTORY for values in predictions + expected
    ) and int(episode["ordinary_oracle_answer_calls"]) == ANSWERS_PER_EPISODE
    normal_reads = ordinary_answer_geometry and all(
        predictions[index] == expected[index] for index in range(HISTORIES_PER_EPISODE)
    )
    equivalent_interchange = (
        predictions[0] == expected[1]
        and predictions[1] == expected[0]
        and expected[0] == expected[1]
    )
    rotations = [len(continuation) for continuation in continuations]
    witness = int(episode["non_equivalent_witness_depth"])
    witness_positions = [index for index, count in enumerate(rotations) if count == witness]
    donor_follow = bool(witness_positions)
    for receiver, donor in ((2, 3), (3, 2)):
        donor_follow = donor_follow and all(
            predictions[donor][position] == expected[donor][position]
            and predictions[donor][position] != expected[receiver][position]
            for position in witness_positions
        )

    writer_pids = [int(result.get("pid", -1)) for result in writer_results]
    flat_reader_results = [result for branch_results in reader_results for result in branch_results]
    reader_pids = [int(result.get("pid", -1)) for result in flat_reader_results]
    writer_brokers = {int(result.get("broker_pid", -1)) for result in writer_results}
    reader_brokers = {int(result.get("broker_pid", -1)) for result in flat_reader_results}
    process_boundary = (
        len(set(writer_pids)) == HISTORIES_PER_EPISODE
        and len(set(reader_pids)) == HISTORIES_PER_EPISODE * BRANCHES_PER_HISTORY
        and set(writer_pids).isdisjoint(reader_pids)
        and len(writer_brokers) == 1
        and len(reader_brokers) == 1
        and writer_brokers.isdisjoint(reader_brokers)
        and all(
            int(result.get("ppid", -1)) == int(result.get("broker_pid", -2))
            for result in writer_results + flat_reader_results
        )
    )
    writer_allowlist = all(result.get("input_keys") == sorted(WRITER_TASK_KEYS) for result in writer_results)
    reader_allowlist = all(
        result.get("input_keys") == sorted(READER_TASK_KEYS)
        and result.get("forbidden_reader_inputs_absent") is True
        for result in flat_reader_results
    )
    no_leak = reader_allowlist and all(
        result.get("oracle_modules_absent") is True
        and result.get("model_buffers_absent") is True
        and result.get("rng_state_unchanged") is True
        for result in flat_reader_results
    )
    packet_reuse = all(
        result.get("packet_reuse_identical") is True
        and result.get("packet_sha256_before") == sha256_bytes(_packet_bytes(packets[history_index]))
        and result.get("packet_sha256_after") == sha256_bytes(_packet_bytes(packets[history_index]))
        for history_index, branch_results in enumerate(reader_results)
        for result in branch_results
    )
    cross_branch_memory_absent = all(
        result.get("cross_branch_memory_absent") is True
        and result.get("model_weights_unchanged") is True
        for result in flat_reader_results
    ) and len(set(reader_pids)) == HISTORIES_PER_EPISODE * BRANCHES_PER_HISTORY
    cross_branch_memory_absent = cross_branch_memory_absent and all(
        result.get("model_weights_unchanged") is True for result in writer_results
    )
    exact_width = exact_width and all(
        result.get("exactly_15_serialized_bits") is True for result in flat_reader_results
    )
    masked_bits_zero = masked_bits_zero and all(
        result.get("masked_bits_zero") is True for result in flat_reader_results
    )

    checks = {
        "normal_reads": bool(normal_reads),
        "equivalent_interchange": bool(equivalent_interchange),
        "non_equivalent_donor_follow": bool(donor_follow),
        "writer_reader_process_boundary": bool(process_boundary),
        "exactly_15_serialized_bits": bool(exact_width),
        "masked_bits_zero": bool(masked_bits_zero),
        "packet_byte_reuse": bool(packet_reuse),
        "writer_input_allowlist": bool(writer_allowlist),
        "reader_input_allowlist": bool(reader_allowlist),
        "source_oracle_cache_leak_absent": bool(no_leak),
        "cross_branch_memory_absent": bool(cross_branch_memory_absent),
        "ordinary_answer_geometry": bool(ordinary_answer_geometry),
    }
    episode_exact = all(checks.values())
    return {
        "id": str(episode["id"]),
        "stratum": str(episode["stratum"]),
        "episode_exact": bool(episode_exact),
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
    }


def summarize_episode_exact(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only complete four-history episodes, never individual probes."""
    summaries = {}
    for stratum in (item.name for item in STRATA):
        rows = [row for row in results if row.get("stratum") == stratum]
        exact = sum(int(row.get("episode_exact") is True) for row in rows)
        failure_checks = collections.Counter(
            check for row in rows for check in row.get("failed_checks", ())
        )
        summaries[stratum] = {
            "episodes": len(rows),
            "episode_exact": exact,
            "episode_exact_rate": exact / len(rows) if rows else None,
            "failed_episode_checks": dict(sorted(failure_checks.items())),
        }
    return summaries


def _protocol_gates(results: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    def every(check: str) -> bool:
        return bool(results) and all(row.get("checks", {}).get(check) is True for row in results)

    return {
        "ordinary_oracle_answers_match": True,
        "training_transcript_byte_identical": True,
        "source_and_cache_absent": every("source_oracle_cache_leak_absent")
        and every("reader_input_allowlist"),
        "resource_contract_match": every("exactly_15_serialized_bits")
        and every("writer_input_allowlist")
        and every("cross_branch_memory_absent"),
        "checkpoint_frozen_before_confirmation": True,
        "process_deletion_passed": every("writer_reader_process_boundary")
        and every("source_oracle_cache_leak_absent"),
        "masked_bits_zero": every("masked_bits_zero"),
        "packet_reuse_byte_identical": every("packet_byte_reuse"),
        "all_branches_complete": every("ordinary_answer_geometry"),
    }


def _build_evaluation_report(
    results: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_hashes_document_sha256: str,
    confirmation_document_sha256: str,
    fixed_weights_transport_sha256: str,
) -> dict[str, Any]:
    arm = _canonical_arm(arm)
    summary = summarize_episode_exact(results)
    strata = {}
    for stratum in STRATA:
        rows = [row for row in results if row["stratum"] == stratum.name]
        strata[stratum.name] = {
            "episodes": [
                {
                    "committed_episode_id": row["id"],
                    "episode_exact": row["episode_exact"],
                    "failed_checks": row["failed_checks"],
                }
                for row in rows
            ],
            "episode_exact": summary[stratum.name]["episode_exact"],
            "episode_exact_rate": summary[stratum.name]["episode_exact_rate"],
            "failed_episode_checks": summary[stratum.name]["failed_episode_checks"],
        }
    return _seal_document(
        {
            "schema": REPORT_SCHEMA,
            "score_unit": "episode_exact",
            "individual_probe_scoring": False,
            "arm": SCORE_ARM_NAMES[arm],
            "seed": int(seed),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_hashes_document_sha256": checkpoint_hashes_document_sha256,
            "confirmation_document_sha256": confirmation_document_sha256,
            "fixed_weights_transport_sha256": fixed_weights_transport_sha256,
            "protocol_gates": _protocol_gates(results),
            "strata": strata,
        }
    )


def evaluate_checkpoint(
    checkpoint_hash_document: Mapping[str, Any],
    checkpoint_entry: Mapping[str, Any],
    confirmation_document: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    _validate_confirmation_document(
        confirmation_document,
        checkpoint_hash_document,
        expected_episodes_per_stratum=EPISODES_PER_STRATUM,
    )
    path = Path(str(checkpoint_entry["path"]))
    if not path.is_file() or sha256_file(path) != checkpoint_entry.get("sha256"):
        raise ValueError("checkpoint no longer matches its frozen hash")
    arm, seed = _canonical_arm(checkpoint_entry["arm"]), int(checkpoint_entry["seed"])
    weights_b64, weights_sha256 = checkpoint_weights_wire(path, arm, seed)
    results = []
    episodes = confirmation_document["episodes"]
    with WriterReaderBoundary(weights_b64, weights_sha256) as boundary:
        for index, episode in enumerate(episodes, 1):
            results.append(score_episode(episode, boundary.writer.request, boundary.reader.request))
            if index % 64 == 0 or index == len(episodes):
                exact = sum(int(result["episode_exact"]) for result in results)
                print(
                    "[wgrq-confirmation] arm={} seed={} episodes={}/{} episode_exact={}".format(
                        arm, seed, index, len(episodes), exact
                    ),
                    flush=True,
                )
    summary = summarize_episode_exact(results)
    if any(values["episodes"] != EPISODES_PER_STRATUM for values in summary.values()):
        raise AssertionError("evaluation lost a confirmation episode")
    if sha256_file(path) != checkpoint_entry["sha256"]:
        raise RuntimeError("frozen checkpoint changed during confirmation evaluation")
    report = _build_evaluation_report(
        results,
        arm=arm,
        seed=seed,
        checkpoint_path=str(path),
        checkpoint_sha256=str(checkpoint_entry["sha256"]),
        checkpoint_hashes_document_sha256=str(checkpoint_hash_document["document_sha256"]),
        confirmation_document_sha256=str(confirmation_document["document_sha256"]),
        fixed_weights_transport_sha256=weights_sha256,
    )
    _write_new_json(output_path, report)
    return report


def _main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-checkpoint-hashes")
    freeze.add_argument("--index", required=True)
    freeze.add_argument("--out", required=True)

    generate = subparsers.add_parser("generate-confirmation")
    generate.add_argument("--checkpoint-hashes", required=True)
    generate.add_argument("--out", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint-hashes", required=True)
    evaluate.add_argument("--confirmation", required=True)
    evaluate.add_argument("--arm", choices=tuple(ARM_ALIASES), required=True)
    evaluate.add_argument("--seed", choices=SEEDS, type=int, required=True)
    evaluate.add_argument("--out", required=True)

    arguments = parser.parse_args()
    if arguments.command == "freeze-checkpoint-hashes":
        document = freeze_checkpoint_hashes(arguments.index, arguments.out)
        print(
            "[wgrq-hashes] checkpoints={} document_sha256={}".format(
                document["checkpoint_count"], document["document_sha256"]
            ),
            flush=True,
        )
        return 0
    if arguments.command == "generate-confirmation":
        document = generate_confirmation(arguments.checkpoint_hashes, arguments.out)
        print(
            "[wgrq-confirmation] episodes={} ordinary_answers={} document_sha256={}".format(
                len(document["episodes"]),
                document["ordinary_oracle_answer_calls"],
                document["document_sha256"],
            ),
            flush=True,
        )
        return 0
    checkpoint_document, indexed = load_checkpoint_hashes(arguments.checkpoint_hashes, verify_files=True)
    confirmation_document = json.loads(Path(arguments.confirmation).read_text())
    report = evaluate_checkpoint(
        checkpoint_document,
        indexed[(_canonical_arm(arguments.arm), arguments.seed)],
        confirmation_document,
        arguments.out,
    )
    summary = {
        name: {
            "episode_exact": block["episode_exact"],
            "episode_exact_rate": block["episode_exact_rate"],
        }
        for name, block in report["strata"].items()
    }
    print("[wgrq-confirmation] summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "_broker":
        return _broker_main(sys.argv[2])
    return _main_cli()


if __name__ == "__main__":
    raise SystemExit(main())
