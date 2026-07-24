"""Offline labels for EFC-C compiler qualification.

This module is intentionally outside the candidate preprocessing boundary.
It may inspect ``PilotRow`` mechanics to construct train-only labels, but its
output is never accepted by the learned compiler forward path.  Source hashes
are the only join key between candidate inputs and supervisor labels.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    ACTION_COUNT,
    OBSERVER_COUNT,
    PilotRow,
    STATE_COUNT,
    decode_source,
    solve_unique_completion,
)
from pipeline.episode_functor_qualification_batch import (  # noqa: E402
    QualificationSupervisorBatch,
    QualificationSupervisorError,
)
from episode_functor_pointer_compiler import (  # noqa: E402
    MAX_KEY_OCCURRENCES,
)
from episode_functor_witness_compiler import (  # noqa: E402
    MAX_RECORDS,
    RECORD_LAW,
    RECORD_OBSERVATION,
    RECORD_OTHER,
    RECORD_STATE,
    RECORD_TRANSITION,
    ROLE_ACTION,
    ROLE_IGNORE,
    ROLE_OBSERVATION_STATE,
    ROLE_OBSERVER,
    ROLE_STATE_DECLARATION,
    ROLE_TRANSITION_DESTINATION,
    ROLE_TRANSITION_SOURCE,
    scan_witness_source,
)


def _normalized_record(payload: bytes) -> bytes:
    record = payload.strip()
    if record.startswith(b"EFC{ "):
        record = record[len(b"EFC{ ") :]
    if record.endswith(b" }"):
        record = record[: -len(b" }")]
    return record.strip()


def _answer_from_record(record: bytes) -> int:
    fields = record.split()
    if record.startswith(b"O answer="):
        token = fields[1].split(b"=", 1)[1]
    else:
        token = fields[-1]
    if token not in (b"0", b"1", b"2", b"3"):
        raise QualificationSupervisorError(
            "observation answer leaves frozen alphabet"
        )
    return int(token)


def _record_labels(
    row: PilotRow,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
]:
    scanned = scan_witness_source(row.source)
    record_types = [RECORD_OTHER] * len(scanned.record_spans)
    occurrence_roles = [ROLE_IGNORE] * len(scanned.pointer.spans)
    record_answers = [0] * len(scanned.record_spans)
    answer_valid = [False] * len(scanned.record_spans)
    occurrences_by_record: list[list[int]] = [
        [] for _ in scanned.record_spans
    ]
    for occurrence, record in enumerate(scanned.occurrence_to_record):
        occurrences_by_record[record].append(occurrence)

    for record_index, (start, end) in enumerate(scanned.record_spans):
        record = _normalized_record(row.source[start:end])
        occurrences = occurrences_by_record[record_index]
        if record in (b"BEGIN-EFC", b"END-EFC"):
            if occurrences:
                raise QualificationSupervisorError(
                    "source wrapper contains an opaque key"
                )
            continue
        if record.startswith(b"LAW-"):
            if occurrences:
                raise QualificationSupervisorError(
                    "law record contains an opaque key"
                )
            record_types[record_index] = RECORD_LAW
            continue
        if record.startswith(b"S "):
            if len(occurrences) != 1:
                raise QualificationSupervisorError(
                    "state record occurrence count differs"
                )
            record_types[record_index] = RECORD_STATE
            occurrence_roles[occurrences[0]] = ROLE_STATE_DECLARATION
            continue
        if record.startswith(b"T "):
            if len(occurrences) != 3:
                raise QualificationSupervisorError(
                    "transition record occurrence count differs"
                )
            record_types[record_index] = RECORD_TRANSITION
            roles = (
                (
                    ROLE_ACTION,
                    ROLE_TRANSITION_SOURCE,
                    ROLE_TRANSITION_DESTINATION,
                )
                if row.factors.organization == 0
                else (
                    ROLE_TRANSITION_DESTINATION,
                    ROLE_ACTION,
                    ROLE_TRANSITION_SOURCE,
                )
            )
            for occurrence, role in zip(
                occurrences,
                roles,
                strict=True,
            ):
                occurrence_roles[occurrence] = role
            continue
        if record.startswith(b"O "):
            if len(occurrences) != 2:
                raise QualificationSupervisorError(
                    "observation record occurrence count differs"
                )
            record_types[record_index] = RECORD_OBSERVATION
            roles = (
                (ROLE_OBSERVER, ROLE_OBSERVATION_STATE)
                if row.factors.organization == 0
                else (ROLE_OBSERVATION_STATE, ROLE_OBSERVER)
            )
            for occurrence, role in zip(
                occurrences,
                roles,
                strict=True,
            ):
                occurrence_roles[occurrence] = role
            record_answers[record_index] = _answer_from_record(record)
            answer_valid[record_index] = True
            continue
        raise QualificationSupervisorError(
            "supervisor encountered an unknown source record"
        )
    return (
        tuple(record_types),
        tuple(occurrence_roles),
        tuple(record_answers),
        tuple(answer_valid),
    )


def collate_qualification_supervision(
    rows: Sequence[PilotRow],
    *,
    device: torch.device | str = "cpu",
) -> QualificationSupervisorBatch:
    """Construct EFC-C labels without creating a candidate input object."""

    frozen = tuple(rows)
    if not frozen or any(type(row) is not PilotRow for row in frozen):
        raise QualificationSupervisorError(
            "supervision requires exact PilotRow objects"
        )
    hashes = tuple(sha256(row.source).hexdigest() for row in frozen)
    if len(set(hashes)) != len(hashes):
        raise QualificationSupervisorError(
            "supervision sources are duplicated"
        )

    batch = len(frozen)
    key_slot_to_unique = torch.zeros(
        (batch, STATE_COUNT + ACTION_COUNT + OBSERVER_COUNT),
        dtype=torch.long,
        device=device,
    )
    record_type = torch.full(
        (batch, MAX_RECORDS),
        RECORD_OTHER,
        dtype=torch.long,
        device=device,
    )
    record_label_valid = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.bool,
        device=device,
    )
    occurrence_role = torch.full(
        (batch, MAX_KEY_OCCURRENCES),
        ROLE_IGNORE,
        dtype=torch.long,
        device=device,
    )
    occurrence_label_valid = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.bool,
        device=device,
    )
    record_answer = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.long,
        device=device,
    )
    answer_label_valid = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.bool,
        device=device,
    )
    transition_next = torch.zeros(
        (batch, ACTION_COUNT, STATE_COUNT),
        dtype=torch.long,
        device=device,
    )
    observer_answer = torch.zeros(
        (batch, OBSERVER_COUNT, STATE_COUNT),
        dtype=torch.long,
        device=device,
    )
    transition_exposed = torch.zeros(
        (batch, ACTION_COUNT, STATE_COUNT),
        dtype=torch.bool,
        device=device,
    )
    observer_exposed = torch.zeros(
        (batch, OBSERVER_COUNT, STATE_COUNT),
        dtype=torch.bool,
        device=device,
    )

    for row_index, row in enumerate(frozen):
        scanned = scan_witness_source(row.source)
        evidence = decode_source(row.source)
        machine = solve_unique_completion(evidence)
        slot_keys = (
            *machine.state_keys,
            *machine.action_keys,
            *machine.observer_keys,
        )
        unique_index = {
            int.from_bytes(key, "little"): index
            for index, key in enumerate(scanned.pointer.unique_keys)
        }
        try:
            assignment = tuple(unique_index[key] for key in slot_keys)
        except KeyError as exc:
            raise QualificationSupervisorError(
                "gold machine key is absent from candidate inventory"
            ) from exc
        key_slot_to_unique[row_index] = torch.tensor(
            assignment,
            dtype=torch.long,
            device=device,
        )
        labels = _record_labels(row)
        record_count = len(labels[0])
        occurrence_count = len(labels[1])
        record_type[row_index, :record_count] = torch.tensor(
            labels[0],
            dtype=torch.long,
            device=device,
        )
        record_label_valid[row_index, :record_count] = True
        occurrence_role[row_index, :occurrence_count] = torch.tensor(
            labels[1],
            dtype=torch.long,
            device=device,
        )
        occurrence_label_valid[row_index, :occurrence_count] = True
        record_answer[row_index, :record_count] = torch.tensor(
            labels[2],
            dtype=torch.long,
            device=device,
        )
        answer_label_valid[row_index, :record_count] = torch.tensor(
            labels[3],
            dtype=torch.bool,
            device=device,
        )
        transition_next[row_index] = torch.tensor(
            machine.transitions,
            dtype=torch.long,
            device=device,
        )
        observer_answer[row_index] = torch.tensor(
            machine.observations,
            dtype=torch.long,
            device=device,
        )
        state_index = {
            key: index for index, key in enumerate(machine.state_keys)
        }
        action_index = {
            key: index for index, key in enumerate(machine.action_keys)
        }
        observer_index = {
            key: index for index, key in enumerate(machine.observer_keys)
        }
        for action, source, _ in evidence.transition_events:
            transition_exposed[
                row_index,
                action_index[action],
                state_index[source],
            ] = True
        for observer, state, _ in evidence.observation_events:
            observer_exposed[
                row_index,
                observer_index[observer],
                state_index[state],
            ] = True

    return QualificationSupervisorBatch(
        source_sha256=hashes,
        key_slot_to_unique=key_slot_to_unique,
        record_type=record_type,
        record_label_valid=record_label_valid,
        occurrence_role=occurrence_role,
        occurrence_label_valid=occurrence_label_valid,
        record_answer=record_answer,
        answer_label_valid=answer_label_valid,
        transition_next=transition_next,
        transition_exposed=transition_exposed,
        observer_answer=observer_answer,
        observer_exposed=observer_exposed,
    )


__all__ = [
    "QualificationSupervisorBatch",
    "QualificationSupervisorError",
    "collate_qualification_supervision",
]
