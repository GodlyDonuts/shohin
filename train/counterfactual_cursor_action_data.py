"""Strict loader for the frozen counterfactual cursor-action canary.

The loader deliberately keeps gold data and model inputs in disjoint frozen
dataclasses.  Callers receive source/cell metadata for relation losses, but
must explicitly ask ``CanarySplit`` for sidecar or text-control examples before
passing anything to a model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "counterfactual_cursor_action_canary_v1"
AUDIT_SCHEMA = "counterfactual_cursor_action_canary_audit_v1"
CANARY_ID = "ccaa-neural-canary-v1"
CONTRACT_SHA256 = "d767e2bf405364d929d6aab0b1e202d643b974e62e71adbddde012a09255a49b"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
LABELS = OPERATIONS + ("DONE",)
LABEL_TOKEN_IDS = (820, 5498, 4307, 7486, 2165)
IMPLEMENTATION_PATHS = (
    "R12_COUNTERFACTUAL_CURSOR_ACTION_THEORY.md",
    "R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md",
    "pipeline/counterfactual_cursor_action_canary_contract_v1.json",
    "pipeline/generate_counterfactual_cursor_action_canary.py",
    "pipeline/audit_counterfactual_cursor_action_canary.py",
    "pipeline/test_counterfactual_cursor_action_canary.py",
    "train/model.py",
    "train/counterfactual_cursor_action.py",
    "train/counterfactual_cursor_action_data.py",
    "train/counterfactual_cursor_action_objectives.py",
    "train/counterfactual_cursor_action_training.py",
    "train/train_counterfactual_cursor_action.py",
    "train/eval_counterfactual_cursor_action.py",
    "train/score_counterfactual_cursor_action.py",
    "train/test_counterfactual_cursor_action.py",
    "train/test_counterfactual_cursor_action_data.py",
    "train/test_counterfactual_cursor_action_objectives.py",
    "train/test_counterfactual_cursor_action_training.py",
    "train/test_train_counterfactual_cursor_action.py",
    "train/test_eval_counterfactual_cursor_action.py",
    "train/jobs/counterfactual_cursor_action_canary.sbatch",
    "train/jobs/eval_counterfactual_cursor_action.sbatch",
)
SPLIT_NAMES = ("train", "development", "confirmation")
SPLIT_RENDERER_IDS = {
    "train": (0, 1, 2, 3, 4, 5),
    "development": (6, 7),
    "confirmation": (8, 9, 10, 11, 12),
}
SPLIT_PACK_COUNTS = {"train": 8, "development": 4, "confirmation": 8}
TEXT_CURSOR_SUFFIXES = (
    "\nCursor: first.\nNext action:",
    "\nCursor: second.\nNext action:",
    "\nCursor: third.\nNext action:",
    "\nCursor: fourth.\nNext action:",
    "\nCursor: complete.\nNext action:",
)
PROMPT_SUFFIX = "\nNext action:"
EXPOSURE_CONTRACT = {
    "sidecar_model_row_inputs": ["prompt_token_ids"],
    "sidecar_model_side_state": ["cursor"],
    "text_control_model_row_inputs": ["text_prompt_token_ids"],
    "gold_only_source_fields": [
        "source_id", "renderer_id", "pack_id", "permutation_id",
        "operation_order", "clause_spans",
    ],
    "gold_only_cell_fields": [
        "cell_id", "source_id", "target_action", "target_index", "target_token_id",
    ],
}
PERMUTATIONS = tuple(itertools.permutations(OPERATIONS))
PERMUTATION_INDEX = {order: index for index, order in enumerate(PERMUTATIONS)}


class ModelInputMode(str, Enum):
    """The two frozen model-exposure surfaces."""

    SIDECAR = "sidecar"
    TEXT_CONTROL = "text_control"


class CanaryArm(str, Enum):
    """The six preregistered learned arms in fixed order."""

    TREATMENT = "orbit_interchange"
    ORDINARY_LOSS = "ordinary_loss"
    RELATION_SHAM = "relation_sham"
    SOURCE_ONLY = "source_only"
    CURSOR_TABLE = "cursor_table"
    TEXT_CURSOR_LORA = "text_cursor_lora"


ALL_ARMS = tuple(CanaryArm)


@dataclass(frozen=True)
class SplitCounts:
    sources: int
    cells: int
    content_groups: int
    adjacent_pairs: int
    training_units: int


EXPECTED_SPLIT_COUNTS = MappingProxyType({
    "train": SplitCounts(1152, 5760, 192, 1728, 288),
    "development": SplitCounts(192, 960, 96, 288, 144),
    "confirmation": SplitCounts(960, 4800, 192, 1440, 288),
})


@dataclass(frozen=True)
class SidecarModelInput:
    """The entire sidecar model-input surface: tokens plus cursor only."""

    prompt_token_ids: tuple[int, ...]
    cursor: int

    def __post_init__(self) -> None:
        _require_token_ids(self.prompt_token_ids, "sidecar prompt_token_ids")
        _require_cursor(self.cursor)


@dataclass(frozen=True)
class TextControlModelInput:
    """The entire text-control model-input surface: text prompt tokens only."""

    text_prompt_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_token_ids(self.text_prompt_token_ids, "text prompt_token_ids")


ModelInput = SidecarModelInput | TextControlModelInput


@dataclass(frozen=True)
class TrainerLabel:
    """Gold action supervision, intentionally separate from ``ModelInput``."""

    target_index: int
    target_token_id: int

    def __post_init__(self) -> None:
        if type(self.target_index) is not int or not 0 <= self.target_index < len(LABELS):
            raise ValueError("trainer label target_index is invalid")
        if type(self.target_token_id) is not int or self.target_token_id < 0:
            raise ValueError("trainer label target_token_id is invalid")


@dataclass(frozen=True)
class SidecarExample:
    model_input: SidecarModelInput
    label: TrainerLabel


@dataclass(frozen=True)
class TextControlExample:
    model_input: TextControlModelInput
    label: TrainerLabel


@dataclass(frozen=True)
class ClauseSpan:
    operation: str
    operand: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CanarySource:
    """Frozen gold source metadata. This is never a model-input object."""

    source_id: str
    split: str
    renderer_id: int
    pack_id: int
    permutation_id: int
    source_text: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    operation_order: tuple[str, ...]
    clause_spans: tuple[ClauseSpan, ...]


@dataclass(frozen=True)
class CanaryCell:
    """Frozen gold cell metadata. Labels are projected separately for training."""

    cell_id: str
    source_id: str
    cursor: int
    text_prompt: str
    text_prompt_token_ids: tuple[int, ...]
    target_action: str
    target_index: int
    target_token_id: int

    @property
    def label(self) -> TrainerLabel:
        return TrainerLabel(self.target_index, self.target_token_id)


@dataclass(frozen=True)
class ContentGroup:
    pack_id: int
    permutation_id: int
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class CursorInterchangeGroup:
    source_id: str
    cells: tuple[CanaryCell, ...]


@dataclass(frozen=True)
class CanaryAdjacentPair:
    pair_id: str
    pack_id: int
    renderer_id: int
    swap_index: int
    left_source_id: str
    right_source_id: str
    left_cells: tuple[CanaryCell, ...]
    right_cells: tuple[CanaryCell, ...]


@dataclass(frozen=True)
class CanaryTrainingUnit:
    """One adjacent-transposition unit, with renderer pairs in canonical order."""

    unit_id: str
    pack_id: int
    swap_index: int
    left_permutation_id: int
    right_permutation_id: int
    adjacent_pairs: tuple[CanaryAdjacentPair, ...]


@dataclass(frozen=True)
class CellPair:
    """A relation-only pair of canonical cell indices, not model inputs."""

    receiver_index: int
    donor_index: int


@dataclass(frozen=True)
class RelationPairSets:
    cursor_interchange: tuple[CellPair, ...]
    renderer_invariance: tuple[CellPair, ...]
    adjacent_affected: tuple[CellPair, ...]
    adjacent_unaffected: tuple[CellPair, ...]


@dataclass(frozen=True)
class ShamRelationContract:
    """The fixed wrong local relation used by the neural sham arm."""

    cursor_target_rotation: int = 1
    adjacent_cursor_rotation: int = 1
    renderer_cursor_rotation: int = 1


@dataclass(frozen=True)
class RelationMetadata:
    """Trainer-only relation metadata with a deterministic deranged sham."""

    content_groups: tuple[ContentGroup, ...]
    cursor_interchange_groups: tuple[CursorInterchangeGroup, ...]
    adjacent_pairs: tuple[CanaryAdjacentPair, ...]
    training_units: tuple[CanaryTrainingUnit, ...]
    relations: RelationPairSets
    relation_sham: ShamRelationContract


@dataclass(frozen=True)
class ArmIndexMap:
    arm: CanaryArm
    input_mode: ModelInputMode
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class CanaryBatch:
    """A deterministic batch with model inputs and labels in separate fields."""

    arm: CanaryArm
    row_indices: tuple[int, ...]
    model_inputs: tuple[ModelInput, ...]
    labels: tuple[TrainerLabel, ...]


@dataclass(frozen=True)
class CanarySplit:
    """One split's frozen rows, lookup maps, relations, and arm index maps."""

    name: str
    counts: SplitCounts
    sources: tuple[CanarySource, ...]
    cells: tuple[CanaryCell, ...]
    training_units: tuple[CanaryTrainingUnit, ...]
    relations: RelationMetadata
    source_by_id: Mapping[str, CanarySource]
    cell_by_key: Mapping[tuple[str, int], CanaryCell]
    source_index_by_id: Mapping[str, int]
    cell_index_by_key: Mapping[tuple[str, int], int]
    arm_index_maps: Mapping[CanaryArm, ArmIndexMap]

    def sidecar_examples(self, indices: Iterable[int] | None = None) -> tuple[SidecarExample, ...]:
        """Return canonical sidecar examples for treatment-style arms."""
        rows = self._indices(indices)
        return tuple(
            SidecarExample(
                SidecarModelInput(self.source_by_id[self.cells[index].source_id].prompt_token_ids,
                                  self.cells[index].cursor),
                self.cells[index].label,
            )
            for index in rows
        )

    def text_examples(self, indices: Iterable[int] | None = None) -> tuple[TextControlExample, ...]:
        """Return canonical text-control examples without a sidecar cursor."""
        rows = self._indices(indices)
        return tuple(
            TextControlExample(
                TextControlModelInput(self.cells[index].text_prompt_token_ids),
                self.cells[index].label,
            )
            for index in rows
        )

    def examples_for_arm(
        self, arm: CanaryArm | str, indices: Iterable[int] | None = None,
    ) -> tuple[SidecarExample | TextControlExample, ...]:
        """Return examples for one preregistered arm without altering row order."""
        normalized_arm = _normalize_arm(arm)
        rows = self._indices(indices)
        if normalized_arm is CanaryArm.TEXT_CURSOR_LORA:
            return self.text_examples(rows)
        if normalized_arm is CanaryArm.SOURCE_ONLY:
            return tuple(
                SidecarExample(
                    SidecarModelInput(
                        self.source_by_id[self.cells[index].source_id].prompt_token_ids, 0,
                    ),
                    self.cells[index].label,
                )
                for index in rows
            )
        return self.sidecar_examples(rows)

    def batches(self, arm: CanaryArm | str, batch_size: int) -> tuple[CanaryBatch, ...]:
        """Chunk the arm's canonical index map without shuffle or dropped rows."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        normalized_arm = _normalize_arm(arm)
        index_map = self.arm_index_maps[normalized_arm]
        batches = []
        for start in range(0, len(index_map.row_indices), batch_size):
            rows = index_map.row_indices[start:start + batch_size]
            examples = self.examples_for_arm(normalized_arm, rows)
            batches.append(CanaryBatch(
                arm=normalized_arm,
                row_indices=rows,
                model_inputs=tuple(example.model_input for example in examples),
                labels=tuple(example.label for example in examples),
            ))
        return tuple(batches)

    def _indices(self, indices: Iterable[int] | None) -> tuple[int, ...]:
        if indices is None:
            return tuple(range(len(self.cells)))
        result = tuple(indices)
        for index in result:
            if type(index) is not int or not 0 <= index < len(self.cells):
                raise IndexError("canary cell index is outside this split")
        return result


@dataclass(frozen=True)
class CanaryDataset:
    """Validated frozen canary plus its bound audit and tokenizer identities."""

    canary_file_sha256: str
    payload_sha256: str
    tokenizer_sha256: str
    audit_file_sha256: str
    implementation_commit: str | None
    implementation_file_sha256: Mapping[str, str]
    splits: tuple[CanarySplit, ...]

    def split(self, name: str) -> CanarySplit:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown canary split: {name}")
        for split in self.splits:
            if split.name == name:
                return split
        raise AssertionError("validated split is missing")


def load_canary(
    canary_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    *,
    requested_model_fields: Mapping[ModelInputMode | str, Iterable[str]] | None = None,
) -> CanaryDataset:
    """Load and bind a frozen canary, independent audit, and tokenizer bytes.

    Every supplied file must be a non-symlink, regular, read-only file.  The
    optional request policy exists so a trainer configuration cannot request a
    gold field under the name of a model input.
    """
    if requested_model_fields is not None:
        if not isinstance(requested_model_fields, Mapping):
            raise TypeError("requested_model_fields must be a mapping")
        for mode, fields in requested_model_fields.items():
            validate_model_input_fields(mode, fields)

    canary_bytes = _read_regular_read_only(canary_path, "canary")
    audit_bytes = _read_regular_read_only(audit_path, "audit")
    tokenizer_bytes = _read_regular_read_only(tokenizer_path, "tokenizer")
    canary_file_sha256 = _sha256_bytes(canary_bytes)
    tokenizer_sha256 = _sha256_bytes(tokenizer_bytes)
    document = _load_json_strict(canary_bytes, "canary")
    report = _load_json_strict(audit_bytes, "audit")

    _validate_top_level(document)
    payload_sha256 = document["payload_sha256"]
    if payload_sha256 != _sha256_bytes(_canonical_json({
        key: value for key, value in document.items() if key != "payload_sha256"
    })):
        raise ValueError("canary payload hash mismatch")
    if tokenizer_sha256 != TOKENIZER_SHA256:
        raise ValueError("tokenizer hash does not match the frozen canary")
    _validate_audit_report(report, canary_file_sha256, payload_sha256, tokenizer_sha256, document)

    splits = tuple(_build_split(name, document["splits"][name]) for name in SPLIT_NAMES)
    return CanaryDataset(
        canary_file_sha256=canary_file_sha256,
        payload_sha256=payload_sha256,
        tokenizer_sha256=tokenizer_sha256,
        audit_file_sha256=_sha256_bytes(audit_bytes),
        implementation_commit=(
            document["implementation_identity"]["git_commit"]
            if document["implementation_identity"] is not None else None
        ),
        implementation_file_sha256=MappingProxyType(
            dict(document["implementation_identity"]["file_sha256"])
            if document["implementation_identity"] is not None else {}
        ),
        splits=splits,
    )


def load_counterfactual_cursor_action_data(
    canary_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    *,
    requested_model_fields: Mapping[ModelInputMode | str, Iterable[str]] | None = None,
) -> CanaryDataset:
    """Compatibility spelling for ``load_canary``."""
    return load_canary(
        canary_path,
        audit_path,
        tokenizer_path,
        requested_model_fields=requested_model_fields,
    )


def validate_model_input_fields(
    mode: ModelInputMode | str, fields: Iterable[str],
) -> None:
    """Reject any requested model field outside the exact frozen allowlist."""
    normalized_mode = _normalize_mode(mode)
    requested = tuple(fields)
    expected = (
        ("prompt_token_ids", "cursor")
        if normalized_mode is ModelInputMode.SIDECAR
        else ("text_prompt_token_ids",)
    )
    if len(requested) != len(expected) or set(requested) != set(expected):
        raise ValueError(
            f"forbidden model input fields for {normalized_mode.value}: {requested!r}"
        )
    if any(type(field) is not str for field in requested):
        raise ValueError("model input field names must be strings")


def _build_split(name: str, value: Any) -> CanarySplit:
    _require_exact_keys(value, {
        "geometry", "sources_sha256", "cells_sha256", "sources", "cells",
        "content_groups", "adjacent_pairs", "training_units",
    }, f"{name} split")
    expected_counts = EXPECTED_SPLIT_COUNTS[name]
    _validate_geometry(value["geometry"], expected_counts, name)
    _require_sha256(value["sources_sha256"], f"{name} sources_sha256")
    _require_sha256(value["cells_sha256"], f"{name} cells_sha256")
    if value["sources_sha256"] != _sha256_bytes(_canonical_json(value["sources"])):
        raise ValueError(f"{name} sources hash mismatch")
    if value["cells_sha256"] != _sha256_bytes(_canonical_json(value["cells"])):
        raise ValueError(f"{name} cells hash mismatch")
    if not isinstance(value["sources"], list) or not isinstance(value["cells"], list):
        raise ValueError(f"{name} sources and cells must be lists")
    if len(value["sources"]) != expected_counts.sources or len(value["cells"]) != expected_counts.cells:
        raise ValueError(f"{name} split counts mismatch")

    sources = _build_sources(name, value["sources"])
    source_by_id = {source.source_id: source for source in sources}
    source_index_by_id = {source.source_id: index for index, source in enumerate(sources)}
    cells = _build_cells(name, value["cells"], sources, source_by_id)
    cell_by_key = {(cell.source_id, cell.cursor): cell for cell in cells}
    cell_index_by_key = {
        (cell.source_id, cell.cursor): index for index, cell in enumerate(cells)
    }
    content_groups = _build_content_groups(name, value["content_groups"], sources)
    adjacent_pairs = _build_adjacent_pairs(
        name, value["adjacent_pairs"], sources, source_by_id, cell_by_key,
    )
    training_units = _build_training_units(name, value["training_units"], adjacent_pairs)
    relations = _build_relations(
        sources,
        cells,
        content_groups,
        adjacent_pairs,
        training_units,
        source_index_by_id,
        cell_index_by_key,
    )
    row_indices = tuple(range(len(cells)))
    arm_index_maps = MappingProxyType({
        arm: ArmIndexMap(
            arm=arm,
            input_mode=(
                ModelInputMode.TEXT_CONTROL
                if arm is CanaryArm.TEXT_CURSOR_LORA
                else ModelInputMode.SIDECAR
            ),
            row_indices=row_indices,
        )
        for arm in ALL_ARMS
    })
    return CanarySplit(
        name=name,
        counts=expected_counts,
        sources=sources,
        cells=cells,
        training_units=training_units,
        relations=relations,
        source_by_id=MappingProxyType(source_by_id),
        cell_by_key=MappingProxyType(cell_by_key),
        source_index_by_id=MappingProxyType(source_index_by_id),
        cell_index_by_key=MappingProxyType(cell_index_by_key),
        arm_index_maps=arm_index_maps,
    )


def _build_sources(name: str, raw_sources: list[Any]) -> tuple[CanarySource, ...]:
    renderer_ids = SPLIT_RENDERER_IDS[name]
    pack_count = SPLIT_PACK_COUNTS[name]
    expected_ids = tuple(
        f"{name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
        for renderer_id in renderer_ids
        for pack_id in range(pack_count)
        for permutation_id in range(len(PERMUTATIONS))
    )
    result = []
    for index, raw in enumerate(raw_sources):
        _require_exact_keys(raw, {
            "schema", "source_id", "split", "renderer_id", "pack_id",
            "permutation_id", "source_text", "prompt", "prompt_token_ids",
            "operation_order", "clause_spans",
        }, f"{name} source")
        if raw["schema"] != "counterfactual_cursor_action_source_v1":
            raise ValueError("source schema mismatch")
        source_id = raw["source_id"]
        if source_id != expected_ids[index]:
            raise ValueError(f"{name} source canonical ordering mismatch")
        renderer_id = _require_int(raw["renderer_id"], "source renderer_id")
        pack_id = _require_int(raw["pack_id"], "source pack_id")
        permutation_id = _require_int(raw["permutation_id"], "source permutation_id")
        if renderer_id not in renderer_ids or not 0 <= pack_id < pack_count:
            raise ValueError("source split geometry mismatch")
        if not 0 <= permutation_id < len(PERMUTATIONS):
            raise ValueError("source permutation_id is invalid")
        if raw["split"] != name:
            raise ValueError("source split mismatch")
        _require_ascii_string(raw["source_text"], "source_text")
        _require_ascii_string(raw["prompt"], "prompt")
        if raw["prompt"] != raw["source_text"] + PROMPT_SUFFIX:
            raise ValueError("source prompt mismatch")
        prompt_token_ids = _freeze_token_ids(raw["prompt_token_ids"], "source prompt_token_ids")
        operation_order = _freeze_string_tuple(raw["operation_order"], "source operation_order")
        if operation_order != PERMUTATIONS[permutation_id]:
            raise ValueError("source operation_order mismatch")
        spans = _build_clause_spans(raw["clause_spans"], operation_order)
        result.append(CanarySource(
            source_id=source_id,
            split=name,
            renderer_id=renderer_id,
            pack_id=pack_id,
            permutation_id=permutation_id,
            source_text=raw["source_text"],
            prompt=raw["prompt"],
            prompt_token_ids=prompt_token_ids,
            operation_order=operation_order,
            clause_spans=spans,
        ))
    return tuple(result)


def _build_clause_spans(raw_spans: Any, order: tuple[str, ...]) -> tuple[ClauseSpan, ...]:
    if not isinstance(raw_spans, list) or len(raw_spans) != len(OPERATIONS):
        raise ValueError("source clause_spans mismatch")
    spans = []
    for index, raw in enumerate(raw_spans):
        _require_exact_keys(raw, {"operation", "operand", "text", "start", "end"}, "clause span")
        operation = raw["operation"]
        if operation != order[index]:
            raise ValueError("clause span operation order mismatch")
        operand = _require_int(raw["operand"], "clause operand")
        start = _require_int(raw["start"], "clause start")
        end = _require_int(raw["end"], "clause end")
        _require_ascii_string(raw["text"], "clause text")
        if operand < 0 or start < 0 or end <= start:
            raise ValueError("clause span bounds mismatch")
        spans.append(ClauseSpan(operation, operand, raw["text"], start, end))
    return tuple(spans)


def _build_cells(
    name: str,
    raw_cells: list[Any],
    sources: tuple[CanarySource, ...],
    source_by_id: Mapping[str, CanarySource],
) -> tuple[CanaryCell, ...]:
    result = []
    for index, raw in enumerate(raw_cells):
        _require_exact_keys(raw, {
            "schema", "cell_id", "source_id", "cursor", "text_prompt",
            "text_prompt_token_ids", "target_action", "target_index", "target_token_id",
        }, f"{name} cell")
        if raw["schema"] != "counterfactual_cursor_action_cell_v1":
            raise ValueError("cell schema mismatch")
        source = sources[index // len(LABELS)]
        cursor = index % len(LABELS)
        if raw["source_id"] != source.source_id or raw["cursor"] != cursor:
            raise ValueError(f"{name} cell canonical ordering mismatch")
        if raw["cell_id"] != f"{source.source_id}-c{cursor}":
            raise ValueError("cell ID mismatch")
        _require_ascii_string(raw["text_prompt"], "text_prompt")
        if raw["text_prompt"] != source.source_text + TEXT_CURSOR_SUFFIXES[cursor]:
            raise ValueError("text-control prompt mismatch")
        text_prompt_token_ids = _freeze_token_ids(
            raw["text_prompt_token_ids"], "text_prompt_token_ids",
        )
        target_action = source.operation_order[cursor] if cursor < 4 else "DONE"
        target_index = LABELS.index(target_action)
        if raw["target_action"] != target_action:
            raise ValueError("cell target_action mismatch")
        if raw["target_index"] != target_index:
            raise ValueError("cell target_index mismatch")
        if raw["target_token_id"] != LABEL_TOKEN_IDS[target_index]:
            raise ValueError("cell target_token_id mismatch")
        result.append(CanaryCell(
            cell_id=raw["cell_id"],
            source_id=source.source_id,
            cursor=cursor,
            text_prompt=raw["text_prompt"],
            text_prompt_token_ids=text_prompt_token_ids,
            target_action=target_action,
            target_index=target_index,
            target_token_id=LABEL_TOKEN_IDS[target_index],
        ))
    if set(source_by_id) != {source.source_id for source in sources}:
        raise AssertionError("source map construction failed")
    return tuple(result)


def _build_content_groups(
    name: str, raw_groups: Any, sources: tuple[CanarySource, ...],
) -> tuple[ContentGroup, ...]:
    expected = []
    for pack_id in range(SPLIT_PACK_COUNTS[name]):
        for permutation_id in range(len(PERMUTATIONS)):
            expected.append({
                "pack_id": pack_id,
                "permutation_id": permutation_id,
                "source_ids": [
                    f"{name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                    for renderer_id in SPLIT_RENDERER_IDS[name]
                ],
            })
    if raw_groups != expected:
        raise ValueError(f"{name} content group map mismatch")
    source_ids = {source.source_id for source in sources}
    if any(not set(group["source_ids"]).issubset(source_ids) for group in raw_groups):
        raise ValueError("content group references unknown source")
    return tuple(ContentGroup(
        pack_id=group["pack_id"],
        permutation_id=group["permutation_id"],
        source_ids=tuple(group["source_ids"]),
    ) for group in raw_groups)


def _expected_adjacent_raw(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adjacent = []
    units = []
    for pack_id in range(SPLIT_PACK_COUNTS[name]):
        for permutation_id, order in enumerate(PERMUTATIONS):
            for swap_index in range(3):
                swapped = list(order)
                swapped[swap_index], swapped[swap_index + 1] = (
                    swapped[swap_index + 1], swapped[swap_index]
                )
                other_id = PERMUTATION_INDEX[tuple(swapped)]
                if permutation_id >= other_id:
                    continue
                pair_ids = []
                for renderer_id in SPLIT_RENDERER_IDS[name]:
                    pair_id = (
                        f"{name}-r{renderer_id:02d}-k{pack_id:02d}-"
                        f"p{permutation_id:02d}-p{other_id:02d}-s{swap_index}"
                    )
                    adjacent.append({
                        "pair_id": pair_id,
                        "pack_id": pack_id,
                        "renderer_id": renderer_id,
                        "swap_index": swap_index,
                        "left_source_id": (
                            f"{name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                        ),
                        "right_source_id": (
                            f"{name}-r{renderer_id:02d}-k{pack_id:02d}-p{other_id:02d}"
                        ),
                    })
                    pair_ids.append(pair_id)
                units.append({
                    "unit_id": (
                        f"{name}-k{pack_id:02d}-p{permutation_id:02d}-"
                        f"p{other_id:02d}-s{swap_index}"
                    ),
                    "pack_id": pack_id,
                    "swap_index": swap_index,
                    "left_permutation_id": permutation_id,
                    "right_permutation_id": other_id,
                    "adjacent_pair_ids": pair_ids,
                })
    return adjacent, units


def _build_adjacent_pairs(
    name: str,
    raw_pairs: Any,
    sources: tuple[CanarySource, ...],
    source_by_id: Mapping[str, CanarySource],
    cell_by_key: Mapping[tuple[str, int], CanaryCell],
) -> tuple[CanaryAdjacentPair, ...]:
    expected_pairs, _ = _expected_adjacent_raw(name)
    if raw_pairs != expected_pairs:
        raise ValueError(f"{name} adjacent pair map mismatch")
    source_ids = {source.source_id for source in sources}
    pairs = []
    for raw in raw_pairs:
        _require_exact_keys(raw, {
            "pair_id", "pack_id", "renderer_id", "swap_index", "left_source_id", "right_source_id",
        }, "adjacent pair")
        if raw["left_source_id"] not in source_ids or raw["right_source_id"] not in source_ids:
            raise ValueError("adjacent pair references unknown source")
        left_source = source_by_id[raw["left_source_id"]]
        right_source = source_by_id[raw["right_source_id"]]
        if left_source.renderer_id != right_source.renderer_id:
            raise ValueError("adjacent pair renderer mismatch")
        pairs.append(CanaryAdjacentPair(
            pair_id=raw["pair_id"],
            pack_id=raw["pack_id"],
            renderer_id=raw["renderer_id"],
            swap_index=raw["swap_index"],
            left_source_id=left_source.source_id,
            right_source_id=right_source.source_id,
            left_cells=tuple(cell_by_key[(left_source.source_id, cursor)] for cursor in range(5)),
            right_cells=tuple(cell_by_key[(right_source.source_id, cursor)] for cursor in range(5)),
        ))
    return tuple(pairs)


def _build_training_units(
    name: str, raw_units: Any, adjacent_pairs: tuple[CanaryAdjacentPair, ...],
) -> tuple[CanaryTrainingUnit, ...]:
    _, expected_units = _expected_adjacent_raw(name)
    if raw_units != expected_units:
        raise ValueError(f"{name} training unit map mismatch")
    pair_by_id = {pair.pair_id: pair for pair in adjacent_pairs}
    units = []
    for raw in raw_units:
        _require_exact_keys(raw, {
            "unit_id", "pack_id", "swap_index", "left_permutation_id",
            "right_permutation_id", "adjacent_pair_ids",
        }, "training unit")
        pair_ids = raw["adjacent_pair_ids"]
        if not isinstance(pair_ids, list) or any(pair_id not in pair_by_id for pair_id in pair_ids):
            raise ValueError("training unit references unknown adjacent pair")
        units.append(CanaryTrainingUnit(
            unit_id=raw["unit_id"],
            pack_id=raw["pack_id"],
            swap_index=raw["swap_index"],
            left_permutation_id=raw["left_permutation_id"],
            right_permutation_id=raw["right_permutation_id"],
            adjacent_pairs=tuple(pair_by_id[pair_id] for pair_id in pair_ids),
        ))
    return tuple(units)


def _build_relations(
    sources: tuple[CanarySource, ...],
    cells: tuple[CanaryCell, ...],
    content_groups: tuple[ContentGroup, ...],
    adjacent_pairs: tuple[CanaryAdjacentPair, ...],
    training_units: tuple[CanaryTrainingUnit, ...],
    source_index_by_id: Mapping[str, int],
    cell_index_by_key: Mapping[tuple[str, int], int],
) -> RelationMetadata:
    cursor_groups = tuple(CursorInterchangeGroup(
        source_id=source.source_id,
        cells=tuple(cells[source_index_by_id[source.source_id] * 5 + cursor] for cursor in range(5)),
    ) for source in sources)
    cursor_pairs = []
    renderer_pairs = []
    affected_pairs = []
    unaffected_pairs = []
    for source_index in range(len(sources)):
        for receiver_cursor in range(5):
            for donor_cursor in range(5):
                if receiver_cursor != donor_cursor:
                    cursor_pairs.append(CellPair(
                        source_index * 5 + receiver_cursor,
                        source_index * 5 + donor_cursor,
                    ))
    for group in content_groups:
        for left_offset, left_source_id in enumerate(group.source_ids):
            for right_source_id in group.source_ids[left_offset + 1:]:
                for cursor in range(5):
                    renderer_pairs.append(CellPair(
                        cell_index_by_key[(left_source_id, cursor)],
                        cell_index_by_key[(right_source_id, cursor)],
                    ))
    for pair in adjacent_pairs:
        for cursor in range(5):
            relation = CellPair(
                cell_index_by_key[(pair.left_source_id, cursor)],
                cell_index_by_key[(pair.right_source_id, cursor)],
            )
            if cursor in (pair.swap_index, pair.swap_index + 1):
                affected_pairs.append(relation)
            else:
                unaffected_pairs.append(relation)
    relations = RelationPairSets(
        cursor_interchange=tuple(cursor_pairs),
        renderer_invariance=tuple(renderer_pairs),
        adjacent_affected=tuple(affected_pairs),
        adjacent_unaffected=tuple(unaffected_pairs),
    )
    return RelationMetadata(
        content_groups=content_groups,
        cursor_interchange_groups=cursor_groups,
        adjacent_pairs=adjacent_pairs,
        training_units=training_units,
        relations=relations,
        relation_sham=ShamRelationContract(),
    )


def _validate_top_level(document: Any) -> None:
    _require_exact_keys(document, {
        "schema", "canary_id", "contract_sha256", "tokenizer_sha256", "generator_sha256",
        "implementation_identity", "exposure_contract", "label_order", "label_token_ids",
        "splits", "payload_sha256",
    }, "canary")
    if document["schema"] != SCHEMA:
        raise ValueError("canary schema mismatch")
    if document["canary_id"] != CANARY_ID:
        raise ValueError("canary ID mismatch")
    if document["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("canary contract hash mismatch")
    if document["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("canary tokenizer hash mismatch")
    _require_sha256(document["generator_sha256"], "canary generator_sha256")
    _validate_implementation_identity(document["implementation_identity"])
    if document["exposure_contract"] != EXPOSURE_CONTRACT:
        raise ValueError("canary exposure contract mismatch")
    if document["label_order"] != list(LABELS) or document["label_token_ids"] != list(LABEL_TOKEN_IDS):
        raise ValueError("canary label contract mismatch")
    _require_sha256(document["payload_sha256"], "canary payload_sha256")
    if not isinstance(document["splits"], dict) or set(document["splits"]) != set(SPLIT_NAMES):
        raise ValueError("canary split ordering mismatch")


def _validate_implementation_identity(value: Any) -> None:
    if value is None:
        return
    _require_exact_keys(value, {"git_commit", "file_sha256"}, "implementation identity")
    commit = value["git_commit"]
    if type(commit) is not str or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("implementation identity git commit mismatch")
    hashes = value["file_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(IMPLEMENTATION_PATHS):
        raise ValueError("implementation identity file hashes mismatch")
    for relative, digest in hashes.items():
        _require_ascii_string(relative, "implementation identity path")
        _require_sha256(digest, "implementation identity digest")


def _validate_audit_report(
    report: Any,
    canary_file_sha256: str,
    payload_sha256: str,
    tokenizer_sha256: str,
    document: Mapping[str, Any],
) -> None:
    _require_exact_keys(report, {
        "schema", "canary_id", "canary_file_sha256", "canary_payload_sha256",
        "contract_sha256", "tokenizer_sha256", "evalgrams_sha256", "auditor_sha256",
        "implementation_identity", "split_summary", "cross_split_13gram_counts",
        "pretraining_corpus_overlap", "all_checks_pass",
    }, "audit")
    if report["schema"] != AUDIT_SCHEMA or report["canary_id"] != CANARY_ID:
        raise ValueError("audit identity mismatch")
    if report["all_checks_pass"] is not True:
        raise ValueError("audit all_checks_pass must be true")
    if report["canary_file_sha256"] != canary_file_sha256:
        raise ValueError("audit canary file hash mismatch")
    if report["canary_payload_sha256"] != payload_sha256:
        raise ValueError("audit canary payload hash mismatch")
    if report["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("audit tokenizer hash mismatch")
    if report["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("audit contract hash mismatch")
    _require_sha256(report["evalgrams_sha256"], "audit evalgrams_sha256")
    _require_sha256(report["auditor_sha256"], "audit auditor_sha256")
    if report["implementation_identity"] != document["implementation_identity"]:
        raise ValueError("audit implementation identity mismatch")
    _validate_audit_split_summary(report["split_summary"])
    counts = report["cross_split_13gram_counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "train__development", "train__confirmation", "development__confirmation",
    } or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("audit cross-split ngram summary mismatch")
    if report["pretraining_corpus_overlap"] != {
        "status": "not_audited_packed_shards_lack_raw_row_boundaries",
        "claim_authorized": False,
        "consequence": "no_pretraining_novelty_or_memorization_exclusion_claim",
    }:
        raise ValueError("audit pretraining-corpus overlap boundary mismatch")


def _validate_audit_split_summary(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(SPLIT_NAMES):
        raise ValueError("audit split summary ordering mismatch")
    expected_keys = {
        "renderers", "packs", "permutations", "cursor_states", "sources", "cells",
        "content_groups", "adjacent_pairs", "training_units", "public_evalgram_hits",
        "source_only_ceiling", "cursor_only_ceiling", "target_counts", "max_prompt_tokens",
    }
    for name, summary in value.items():
        _require_exact_keys(summary, expected_keys, "audit split summary")
        counts = EXPECTED_SPLIT_COUNTS[name]
        if summary["sources"] != counts.sources or summary["cells"] != counts.cells:
            raise ValueError("audit split summary counts mismatch")
        if summary["content_groups"] != counts.content_groups:
            raise ValueError("audit content group count mismatch")
        if summary["adjacent_pairs"] != counts.adjacent_pairs:
            raise ValueError("audit adjacent pair count mismatch")
        if summary["training_units"] != counts.training_units:
            raise ValueError("audit training unit count mismatch")
        if summary["public_evalgram_hits"] != 0:
            raise ValueError("audit public evalgram overlap")
        if summary["target_counts"] != {label: counts.sources for label in LABELS}:
            raise ValueError("audit target count mismatch")


def _validate_geometry(value: Any, counts: SplitCounts, name: str) -> None:
    _require_exact_keys(value, {
        "renderers", "packs", "permutations", "cursor_states", "sources", "cells",
        "content_groups", "adjacent_pairs", "training_units",
    }, f"{name} geometry")
    expected = {
        "renderers": len(SPLIT_RENDERER_IDS[name]),
        "packs": SPLIT_PACK_COUNTS[name],
        "permutations": 24,
        "cursor_states": 5,
        "sources": counts.sources,
        "cells": counts.cells,
        "content_groups": counts.content_groups,
        "adjacent_pairs": counts.adjacent_pairs,
        "training_units": counts.training_units,
    }
    if value != expected:
        raise ValueError(f"{name} geometry mismatch")


def _read_regular_read_only(path: str | os.PathLike[str], label: str) -> bytes:
    candidate = Path(path)
    try:
        before = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{label} input is unavailable") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} input may not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"{label} input could not be opened without following links") from error
    try:
        opened = os.fstat(descriptor)
        _require_regular_read_only_stat(opened, label)
        if not _same_file(before, opened):
            raise ValueError(f"{label} input changed while opening")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.lstat(candidate)
        _require_regular_read_only_stat(after, label)
        if not _same_file(opened, after):
            raise ValueError(f"{label} input changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_regular_read_only_stat(info: os.stat_result, label: str) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} input may not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} input must be a regular file")
    if info.st_mode & 0o222:
        raise ValueError(f"{label} input must be read-only")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _load_json_strict(data: bytes, label: str) -> Any:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} JSON must be ASCII") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: _reject_json_constant(constant, label),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(constant: str, label: str) -> None:
    raise ValueError(f"{label} JSON constant is forbidden: {constant}")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("canary payload is not canonical-JSON serializable") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{context} fields mismatch")


def _require_sha256(value: Any, context: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA256")


def _require_ascii_string(value: Any, context: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must be ASCII") from error


def _require_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _freeze_token_ids(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    result = tuple(_require_int(token_id, context) for token_id in value)
    _require_token_ids(result, context)
    return result


def _freeze_string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    result = tuple(value)
    for item in result:
        _require_ascii_string(item, context)
    return result


def _require_token_ids(value: Sequence[int], context: str) -> None:
    if not value or any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{context} must contain nonnegative integer token IDs")


def _require_cursor(value: int) -> None:
    if type(value) is not int or not 0 <= value < 5:
        raise ValueError("cursor must be an integer in [0,4]")


def _normalize_mode(value: ModelInputMode | str) -> ModelInputMode:
    try:
        return value if isinstance(value, ModelInputMode) else ModelInputMode(value)
    except ValueError as error:
        raise ValueError(f"unknown model input mode: {value!r}") from error


def _normalize_arm(value: CanaryArm | str) -> CanaryArm:
    try:
        return value if isinstance(value, CanaryArm) else CanaryArm(value)
    except ValueError as error:
        raise ValueError(f"unknown canary arm: {value!r}") from error
