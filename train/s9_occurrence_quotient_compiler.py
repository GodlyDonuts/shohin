"""Neural bounded-span compiler for S9 occurrence-quotient relations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS, sha256_file
from s9_occurrence_quotient import compile_quotient, quotient_from_emitted_spans


MAX_SPAN_WIDTH = 4


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    text: str
    char_start: int
    char_end: int
    target: int


@dataclass(frozen=True)
class S9Example:
    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    candidates: tuple[SpanCandidate, ...]
    row: dict[str, object]


def _role_for_label(label: str) -> str:
    if label.startswith("entity.roster."):
        return "entity.roster"
    if label.startswith("position.roster."):
        return "position.roster"
    if label.startswith("state.entity."):
        return "state.entity"
    if label.startswith("card."):
        return "card." + label.rsplit(".", 1)[-1]
    if label == "entry.tag":
        return label
    if label.startswith("event."):
        return "event." + label.rsplit(".", 1)[-1]
    if label == "query.position":
        return label
    raise ValueError(f"unknown S9 span label {label}")


def _trimmed_source_span(
    question: str, offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[str, int, int]:
    char_start = int(offsets[start][0])
    char_end = int(offsets[end][1])
    while char_start < char_end and question[char_start].isspace():
        char_start += 1
    while char_end > char_start and question[char_end - 1].isspace():
        char_end -= 1
    return question[char_start:char_end], char_start, char_end


def compile_row(row: dict[str, object], tokenizer) -> S9Example:
    question = str(row["question"])
    encoding = tokenizer.encode(question)
    token_hash = hashlib.sha256(
        json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if token_hash != row["token_ids_sha256"]:
        raise ValueError(f"S9 tokenizer mismatch for {row.get('id')}")
    gold: dict[tuple[int, ...], int] = {}
    for label, span in row["spans"].items():
        positions = tuple(int(value) for value in span["token_positions"])
        if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError("S9 requires contiguous gold spans")
        if len(positions) > MAX_SPAN_WIDTH:
            raise ValueError("S9 gold span exceeds frozen proposal width")
        key = tuple(positions)
        if key in gold:
            raise ValueError("S9 gold spans collide")
        gold[key] = ROLE_INDEX[_role_for_label(str(label))]

    candidates = []
    for start in range(len(encoding.ids)):
        for width in range(1, MAX_SPAN_WIDTH + 1):
            end = start + width - 1
            if end >= len(encoding.ids):
                break
            text, char_start, char_end = _trimmed_source_span(
                question, encoding.offsets, start, end
            )
            if not text:
                continue
            positions = tuple(range(start, end + 1))
            candidates.append(SpanCandidate(
                start=start,
                end=end,
                text=text,
                char_start=char_start,
                char_end=char_end,
                target=gold.get(positions, ROLE_INDEX["none"]),
            ))
    covered = {
        tuple(range(candidate.start, candidate.end + 1))
        for candidate in candidates if candidate.target != ROLE_INDEX["none"]
    }
    if covered != set(gold):
        raise ValueError("S9 proposal enumeration missed a gold span")
    return S9Example(
        ids=tuple(int(value) for value in encoding.ids),
        offsets=tuple((int(start), int(end)) for start, end in encoding.offsets),
        candidates=tuple(candidates),
        row=row,
    )


def load_examples(path: Path, tokenizer, expected_split: str, seq_len: int) -> list[S9Example]:
    examples = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"S9 split mismatch at row {line_number}")
            example = compile_row(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError(f"S9 row {line_number} exceeds context")
            examples.append(example)
    if not examples:
        raise ValueError("no S9 examples")
    return examples


def shuffle_relation_supervision(examples: Sequence[S9Example], seed: int) -> list[S9Example]:
    rng = random.Random(seed)
    result = []
    for example in examples:
        labels = [
            candidate.target
            for candidate in example.candidates
            if candidate.target != ROLE_INDEX["none"]
        ]
        rng.shuffle(labels)
        cursor = 0
        candidates = []
        for candidate in example.candidates:
            if candidate.target == ROLE_INDEX["none"]:
                candidates.append(candidate)
            else:
                candidates.append(replace(candidate, target=labels[cursor]))
                cursor += 1
        result.append(replace(example, candidates=tuple(candidates)))
    return result


def make_batches(examples: Sequence[S9Example], batch_size: int, seed: int) -> list[list[int]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def _sample_candidates(
    example: S9Example, negative_limit: int | None, rng: random.Random
) -> tuple[SpanCandidate, ...]:
    if negative_limit is None:
        return example.candidates
    positive = [value for value in example.candidates if value.target != ROLE_INDEX["none"]]
    negative = [value for value in example.candidates if value.target == ROLE_INDEX["none"]]
    if len(negative) > negative_limit:
        negative = rng.sample(negative, negative_limit)
    return tuple(sorted(positive + negative, key=lambda value: (value.start, value.end)))


def pad_batch(
    examples: Sequence[S9Example],
    indices: Sequence[int],
    device,
    *,
    negative_limit: int | None,
    seed: int,
):
    selected = [examples[index] for index in indices]
    length = max(len(example.ids) for example in selected)
    ids = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    valid = torch.zeros((len(selected), length), dtype=torch.bool, device=device)
    batch_ids = []
    starts = []
    ends = []
    targets = []
    candidate_rows: list[tuple[SpanCandidate, ...]] = []
    class_keys: dict[tuple[int, str], int] = {}
    class_ids = []
    for row, example in enumerate(selected):
        width = len(example.ids)
        ids[row, :width] = torch.tensor(example.ids, dtype=torch.long, device=device)
        valid[row, :width] = True
        candidates = _sample_candidates(
            example, negative_limit, random.Random(seed ^ (indices[row] << 17))
        )
        candidate_rows.append(candidates)
        for candidate in candidates:
            batch_ids.append(row)
            starts.append(candidate.start)
            ends.append(candidate.end)
            targets.append(candidate.target)
            key = (row, candidate.text)
            if key not in class_keys:
                class_keys[key] = len(class_keys)
            class_ids.append(class_keys[key])
    tensors = {
        "batch": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "start": torch.tensor(starts, dtype=torch.long, device=device),
        "end": torch.tensor(ends, dtype=torch.long, device=device),
        "class": torch.tensor(class_ids, dtype=torch.long, device=device),
        "target": torch.tensor(targets, dtype=torch.long, device=device),
    }
    return selected, tuple(candidate_rows), ids, valid, tensors


class OccurrenceQuotientCompiler(nn.Module):
    def __init__(
        self,
        model,
        layer: int = 19,
        width: int = 384,
        heads: int = 8,
        encoder_layers: int = 5,
        ff: int = 1408,
    ) -> None:
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("S9 compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks) or width % heads:
            raise ValueError("invalid S9 compiler dimensions")
        self.model = model
        self.layer = int(layer)
        self.width = int(width)
        self.model.requires_grad_(False)
        self.memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.memory_projection = nn.Linear(model.cfg.d_model, width, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )
        self.span_projection = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
        )
        self.class_projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.relation_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, len(ROLE_LABELS)),
        )

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def initialize_memory_encoder(self, state: dict[str, torch.Tensor]) -> tuple[str, ...]:
        own = self.state_dict()
        prefixes = ("memory_norm.", "memory_projection.", "memory_encoder.")
        loaded = []
        for name, value in state.items():
            if name in own and own[name].shape == value.shape and name.startswith(prefixes):
                own[name].copy_(value)
                loaded.append(name)
        expected = [name for name in own if name.startswith(prefixes)]
        if set(loaded) != set(expected):
            raise ValueError("S9 memory initialization incomplete")
        return tuple(sorted(loaded))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.tok(ids)
            cos = self.model.cos[:ids.shape[1]].to(hidden.device)
            sin = self.model.sin[:ids.shape[1]].to(hidden.device)
            for block in self.model.blocks[:self.layer + 1]:
                hidden, _ = block(hidden, cos, sin)
        return hidden.detach()

    def forward(
        self,
        ids: torch.Tensor,
        valid: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        *,
        class_messages: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        batch = candidates["batch"]
        start = candidates["start"]
        end = candidates["end"]
        prefix = torch.cat((
            torch.zeros(
                memory.shape[0], 1, memory.shape[2],
                device=memory.device, dtype=memory.dtype,
            ),
            memory.cumsum(dim=1),
        ), dim=1)
        mean = (prefix[batch, end + 1] - prefix[batch, start]) / (
            (end - start + 1).to(memory.dtype).unsqueeze(-1)
        )
        span = self.span_projection(torch.cat((
            memory[batch, start], memory[batch, end], mean,
        ), dim=-1))
        class_index = candidates["class"]
        class_count = int(class_index.max().item()) + 1
        class_sum = torch.zeros(
            class_count, self.width, device=span.device, dtype=span.dtype
        ).index_add(0, class_index, span)
        counts = torch.zeros(
            class_count, device=span.device, dtype=span.dtype
        ).index_add(0, class_index, torch.ones_like(class_index, dtype=span.dtype))
        class_mean = class_sum / counts.clamp_min(1).unsqueeze(-1)
        class_context = self.class_projection(class_mean)[class_index]
        if not class_messages:
            class_context = torch.zeros_like(class_context)
        logits = self.relation_head(torch.cat((span, class_context), dim=-1)).float()
        return {"role_logits": logits, "span": span, "class_context": class_context}


def compiler_loss(outputs, targets):
    weights = torch.ones(
        len(ROLE_LABELS), device=outputs["role_logits"].device, dtype=torch.float32
    )
    weights[ROLE_INDEX["none"]] = 0.15
    return F.cross_entropy(outputs["role_logits"], targets, weight=weights)


def _one_inside(values, start, end, label):
    matches = [value for value in values if start <= value.start < end]
    if len(matches) != 1:
        raise ValueError(f"S9 region has {len(matches)} {label} islands")
    return matches[0]


def emitted_spans_from_logits(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
) -> dict[str, dict[str, object]]:
    if len(candidates) != logits.shape[0]:
        raise ValueError("S9 candidate/logit count mismatch")
    none = ROLE_INDEX["none"]
    non_none_score, non_none_role = logits[:, 1:].max(dim=-1)
    margins = non_none_score - logits[:, none]
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(margins[index].item()),
            candidates[index].end - candidates[index].start,
            candidates[index].start,
        ),
    )
    occupied: set[int] = set()
    selected: dict[str, list[SpanCandidate]] = {role: [] for role in ROLE_LABELS[1:]}
    for index in order:
        role_index = int(non_none_role[index].item()) + 1
        if int(logits[index].argmax().item()) == none:
            continue
        candidate = candidates[index]
        positions = set(range(candidate.start, candidate.end + 1))
        if occupied & positions:
            continue
        occupied.update(positions)
        selected[ROLE_LABELS[role_index]].append(candidate)
    for values in selected.values():
        values.sort(key=lambda value: value.start)

    labeled: dict[str, SpanCandidate] = {}
    for role, prefix in (
        ("entity.roster", "entity.roster"),
        ("position.roster", "position.roster"),
        ("state.entity", "state.entity"),
    ):
        for index, candidate in enumerate(selected[role]):
            labeled[f"{prefix}.{index}"] = candidate

    card_anchors = selected["card.operation"]
    for index, anchor in enumerate(card_anchors):
        end = card_anchors[index + 1].start if index + 1 < len(card_anchors) else len(example.ids)
        labeled[f"card.{index}.operation"] = anchor
        labeled[f"card.{index}.y0"] = _one_inside(selected["card.y0"], anchor.start, end, "card.y0")
        labeled[f"card.{index}.y1"] = _one_inside(selected["card.y1"], anchor.start, end, "card.y1")

    event_anchors = selected["event.tag"]
    for index, anchor in enumerate(event_anchors):
        end = event_anchors[index + 1].start if index + 1 < len(event_anchors) else len(example.ids)
        labeled[f"event.{index}.tag"] = anchor
        for role, suffix in (
            ("event.operation", "operation"),
            ("event.entity", "entity"),
        ):
            labeled[f"event.{index}.{suffix}"] = _one_inside(selected[role], anchor.start, end, role)
        next_values = [value for value in selected["event.next"] if anchor.start <= value.start < end]
        nil_values = [value for value in selected["event.nil"] if anchor.start <= value.start < end]
        if len(next_values) + len(nil_values) != 1:
            raise ValueError("S9 event has invalid next/nil count")
        if next_values:
            labeled[f"event.{index}.next"] = next_values[0]
        else:
            labeled[f"event.{index}.nil"] = nil_values[0]
    for role, label in (("entry.tag", "entry.tag"), ("query.position", "query.position")):
        if len(selected[role]) != 1:
            raise ValueError(f"S9 requires exactly one {role}")
        labeled[label] = selected[role][0]
    return {
        label: {
            "start": value.char_start,
            "end": value.char_end,
            "text": value.text,
        }
        for label, value in labeled.items()
    }


def decode_graph(example, candidates, logits):
    spans = emitted_spans_from_logits(example, candidates, logits)
    quotient = quotient_from_emitted_spans(str(example.row["question"]), spans)
    return compile_quotient(quotient), spans


def adapter_state(model):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("model.")
    }


def adapter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(adapter_state(model).items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_adapter_state(model, state):
    result = model.load_state_dict(state, strict=False)
    expected_missing = [name for name in model.state_dict() if name.startswith("model.")]
    if sorted(result.missing_keys) != sorted(expected_missing) or result.unexpected_keys:
        raise ValueError(
            f"S9 adapter mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )


__all__ = [
    "MAX_SPAN_WIDTH",
    "OccurrenceQuotientCompiler",
    "S9Example",
    "SpanCandidate",
    "adapter_hash",
    "adapter_state",
    "compile_row",
    "compiler_loss",
    "decode_graph",
    "emitted_spans_from_logits",
    "load_adapter_state",
    "load_examples",
    "make_batches",
    "pad_batch",
    "sha256_file",
    "shuffle_relation_supervision",
]
