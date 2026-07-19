"""Whole-source compiler for S8 nil-linked contextual law graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from s8_nil_linked_law_graph import (
    EventNode,
    LawCardNode,
    NIL,
    NilLinkedLawGraph,
    linked_path,
    rewire_path,
)


ROLE_LABELS = (
    "none",
    "entity.roster",
    "position.roster",
    "state.entity",
    "card.operation",
    "card.y0",
    "card.y1",
    "entry.tag",
    "event.tag",
    "event.operation",
    "event.entity",
    "event.next",
    "event.nil",
    "query.position",
)
ROLE_INDEX = {value: index for index, value in enumerate(ROLE_LABELS)}
MAX_DEPTH = 8
ADMITTED_MODULI = (5, 7, 11)


@dataclass(frozen=True)
class S8GraphExample:
    ids: tuple[int, ...]
    roles: tuple[int, ...]
    ranks: tuple[int, ...]
    role_positions: dict[str, tuple[tuple[int, ...], ...]]
    operation_positions: dict[str, tuple[tuple[int, ...], ...]]
    row: dict[str, object]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if not name.startswith("model.")
    }


def adapter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(module).items()):
        value = tensor.contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_adapter_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load every non-trunk tensor and reject silent adapter drift."""

    result = module.load_state_dict(state, strict=False)
    expected_missing = {
        name for name in module.state_dict() if name.startswith("model.")
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(
            "S8 adapter state mismatch: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )


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
    raise ValueError(f"unknown S8 span label {label}")


def _position_tuple(span: dict[str, object]) -> tuple[int, ...]:
    values = tuple(int(value) for value in span["token_positions"])
    if not values:
        raise ValueError("S8 span has no token positions")
    return values


def compile_row(row: dict[str, object], tokenizer) -> S8GraphExample:
    encoding = tokenizer.encode(str(row["question"]))
    token_hash = hashlib.sha256(
        json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if token_hash != row["token_ids_sha256"]:
        raise ValueError(f"S8 tokenizer mismatch for {row.get('id')}")
    roles = [ROLE_INDEX["none"]] * len(encoding.ids)
    ranks = [-100] * len(encoding.ids)
    grouped: dict[str, list[tuple[int, ...]]] = {role: [] for role in ROLE_LABELS[1:]}
    spans = row["spans"]
    for label, span in spans.items():
        role = _role_for_label(str(label))
        positions = _position_tuple(span)
        grouped[role].append(positions)
        for position in positions:
            if not 0 <= position < len(roles):
                raise ValueError("S8 role position outside token sequence")
            if roles[position] != ROLE_INDEX["none"]:
                raise ValueError("S8 role spans overlap")
            roles[position] = ROLE_INDEX[role]

    tag_to_rank = {
        str(tag): rank for rank, tag in enumerate(row["execution_tags"])
    }
    operation_positions: dict[str, list[tuple[int, ...]]] = {}
    for label, span in spans.items():
        if _role_for_label(str(label)) not in {
            "card.operation",
            "event.operation",
        }:
            continue
        operation_positions.setdefault(str(span["text"]), []).append(
            _position_tuple(span)
        )
    for node_index, node in enumerate(row["nodes"]):
        tag_positions = _position_tuple(spans[f"event.{node_index}.tag"])
        rank = tag_to_rank[str(node["tag"])]
        for position in tag_positions:
            ranks[position] = rank

    return S8GraphExample(
        ids=tuple(int(value) for value in encoding.ids),
        roles=tuple(roles),
        ranks=tuple(ranks),
        role_positions={
            role: tuple(sorted(values, key=lambda cell: cell[0]))
            for role, values in grouped.items()
        },
        operation_positions={
            name: tuple(values) for name, values in operation_positions.items()
        },
        row=row,
    )


def load_examples(path, tokenizer, expected_split: str, seq_len: int) -> list[S8GraphExample]:
    examples = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"S8 split mismatch at row {line_number}")
            example = compile_row(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError(f"S8 row {line_number} exceeds context")
            examples.append(example)
    if not examples:
        raise ValueError("no S8 graph examples")
    return examples


def shuffle_supervision(
    examples: Sequence[S8GraphExample], seed: int
) -> list[S8GraphExample]:
    rng = random.Random(seed)
    result = []
    for example in examples:
        permutation = list(range(len(example.ids)))
        rng.shuffle(permutation)
        roles = [ROLE_INDEX["none"]] * len(example.ids)
        ranks = [-100] * len(example.ids)
        for old_position, new_position in enumerate(permutation):
            roles[new_position] = example.roles[old_position]
            ranks[new_position] = example.ranks[old_position]
        result.append(replace(example, roles=tuple(roles), ranks=tuple(ranks)))
    return result


def make_batches(
    examples: Sequence[S8GraphExample], batch_size: int, seed: int
) -> list[list[int]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def pad_batch(examples, indices, device):
    selected = [examples[index] for index in indices]
    length = max(len(example.ids) for example in selected)
    ids = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    valid = torch.zeros((len(selected), length), dtype=torch.bool, device=device)
    roles = torch.full((len(selected), length), -100, dtype=torch.long, device=device)
    ranks = torch.full((len(selected), length), -100, dtype=torch.long, device=device)
    for row, example in enumerate(selected):
        width = len(example.ids)
        ids[row, :width] = torch.tensor(example.ids, dtype=torch.long, device=device)
        valid[row, :width] = True
        roles[row, :width] = torch.tensor(example.roles, dtype=torch.long, device=device)
        ranks[row, :width] = torch.tensor(example.ranks, dtype=torch.long, device=device)
    return selected, ids, valid, roles, ranks


class NilLinkedGraphCompiler(nn.Module):
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
            raise ValueError("S8 compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid S8 frozen layer")
        if width % heads or encoder_layers <= 0:
            raise ValueError("invalid S8 compiler dimensions")
        self.model = model
        self.layer = int(layer)
        self.width = int(width)
        self.encoder_layers = int(encoder_layers)
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
            encoder_layer,
            num_layers=self.encoder_layers,
            enable_nested_tensor=False,
        )
        self.role_head = nn.Linear(width, len(ROLE_LABELS))
        self.rank_head = nn.Linear(width, MAX_DEPTH)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def initialize_memory_encoder(self, state: dict[str, torch.Tensor]) -> tuple[str, ...]:
        own = self.state_dict()
        loaded = []
        prefixes = ("memory_norm.", "memory_projection.", "memory_encoder.")
        for name, value in state.items():
            if name.startswith("model.") or name not in own or own[name].shape != value.shape:
                continue
            if name.startswith(prefixes):
                own[name].copy_(value)
                loaded.append(name)
        expected = [name for name in own if name.startswith(prefixes)]
        if set(loaded) != set(expected):
            raise ValueError("S8 memory initialization incomplete")
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

    def forward(self, ids: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        return {
            "memory": memory,
            "role_logits": self.role_head(memory).float(),
            "rank_logits": self.rank_head(memory).float(),
        }


def compiler_loss(outputs, roles, ranks, role_weight=1.0, rank_weight=1.0):
    role_weights = torch.ones(
        len(ROLE_LABELS), device=outputs["role_logits"].device, dtype=torch.float32
    )
    role_weights[ROLE_INDEX["none"]] = 0.15
    role_loss = F.cross_entropy(
        outputs["role_logits"].reshape(-1, len(ROLE_LABELS)),
        roles.reshape(-1),
        ignore_index=-100,
        weight=role_weights,
    )
    selected = ranks >= 0
    if not bool(selected.any()):
        raise ValueError("S8 batch has no event-rank targets")
    rank_loss = F.cross_entropy(outputs["rank_logits"][selected], ranks[selected])
    total = role_weight * role_loss + rank_weight * rank_loss
    return total, {"role": role_loss, "rank": rank_loss}


def _islands(labels: Sequence[int], role: str) -> tuple[tuple[int, ...], ...]:
    target = ROLE_INDEX[role]
    result = []
    start = None
    for index, label in enumerate(labels):
        if label == target and start is None:
            start = index
        if label != target and start is not None:
            result.append(tuple(range(start, index)))
            start = None
    if start is not None:
        result.append(tuple(range(start, len(labels))))
    return tuple(result)


def _signature(ids: Sequence[int], positions: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(ids[position]) for position in positions)


def _unique_index(values: Sequence[tuple[int, ...]], target: tuple[int, ...]) -> int:
    matches = [index for index, value in enumerate(values) if value == target]
    if len(matches) != 1:
        raise ValueError("S8 carrier match is not unique")
    return matches[0]


def _one_inside(
    values: Sequence[tuple[int, ...]], start: int, end: int, label: str
) -> tuple[int, ...]:
    matches = [value for value in values if start <= value[0] < end]
    if len(matches) != 1:
        raise ValueError(f"S8 region has {len(matches)} {label} islands")
    return matches[0]


def decode_graph(
    example: S8GraphExample,
    role_logits: torch.Tensor,
    rank_logits: torch.Tensor,
) -> dict[str, object]:
    length = len(example.ids)
    labels = role_logits[:length].argmax(-1).detach().cpu().tolist()
    islands = {role: _islands(labels, role) for role in ROLE_LABELS[1:]}
    entity_roster = islands["entity.roster"]
    position_roster = islands["position.roster"]
    state_entities = islands["state.entity"]
    modulus = len(entity_roster)
    if modulus not in ADMITTED_MODULI or len(position_roster) != modulus:
        raise ValueError("S8 roster cardinality mismatch")
    if len(state_entities) != modulus:
        raise ValueError("S8 state cardinality mismatch")
    entity_signatures = [_signature(example.ids, value) for value in entity_roster]
    position_signatures = [_signature(example.ids, value) for value in position_roster]
    initial_state = tuple(
        _unique_index(entity_signatures, _signature(example.ids, value))
        for value in state_entities
    )

    card_anchors = islands["card.operation"]
    cards: list[LawCardNode] = []
    card_signatures: list[tuple[int, ...]] = []
    for index, anchor in enumerate(card_anchors):
        end = card_anchors[index + 1][0] if index + 1 < len(card_anchors) else length
        y0 = _one_inside(islands["card.y0"], anchor[0], end, "card.y0")
        y1 = _one_inside(islands["card.y1"], anchor[0], end, "card.y1")
        operation_signature = _signature(example.ids, anchor)
        card_signatures.append(operation_signature)
        cards.append(
            LawCardNode(
                operation=",".join(map(str, operation_signature)),
                y0=_unique_index(position_signatures, _signature(example.ids, y0)),
                y1=_unique_index(position_signatures, _signature(example.ids, y1)),
            )
        )
    if len(cards) < 2:
        raise ValueError("S8 decoder found fewer than two cards")

    event_anchors = islands["event.tag"]
    tag_signatures = [_signature(example.ids, value) for value in event_anchors]
    nodes: list[EventNode] = []
    predicted_ranks: list[int] = []
    all_links_valid = True
    for index, anchor in enumerate(event_anchors):
        end = event_anchors[index + 1][0] if index + 1 < len(event_anchors) else length
        operation = _one_inside(
            islands["event.operation"], anchor[0], end, "event.operation"
        )
        entity = _one_inside(islands["event.entity"], anchor[0], end, "event.entity")
        next_values = [
            value for value in islands["event.next"] if anchor[0] <= value[0] < end
        ]
        nil_values = [
            value for value in islands["event.nil"] if anchor[0] <= value[0] < end
        ]
        link_valid = len(next_values) + len(nil_values) == 1
        operation_signature = _signature(example.ids, operation)
        operation_id = _unique_index(card_signatures, operation_signature)
        try:
            next_node = (
                NIL
                if link_valid and nil_values
                else _unique_index(
                    tag_signatures,
                    _signature(example.ids, next_values[0]),
                )
                if link_valid
                else NIL
            )
        except ValueError:
            link_valid = False
            next_node = NIL
        all_links_valid = all_links_valid and link_valid
        nodes.append(
            EventNode(
                identity=_unique_index(
                    entity_signatures, _signature(example.ids, entity)
                ),
                operation=cards[operation_id].operation,
                next_node=next_node,
            )
        )
        index_tensor = torch.tensor(anchor, dtype=torch.long, device=rank_logits.device)
        predicted_ranks.append(
            int(rank_logits.index_select(0, index_tensor).mean(0).argmax().item())
        )
    if not nodes:
        raise ValueError("S8 decoder found no event nodes")
    entry_values = islands["entry.tag"]
    query_values = islands["query.position"]
    if len(query_values) != 1:
        raise ValueError("S8 query cardinality mismatch")
    entry_node = 0
    entry_valid = False
    if len(entry_values) == 1:
        try:
            entry_node = _unique_index(
                tag_signatures, _signature(example.ids, entry_values[0])
            )
            entry_valid = True
        except ValueError:
            pass
    graph = NilLinkedLawGraph(
        modulus=modulus,
        initial_state=initial_state,
        cards=tuple(cards),
        nodes=tuple(nodes),
        entry_node=entry_node,
        query_position=_unique_index(
            position_signatures, _signature(example.ids, query_values[0])
        ),
    )
    try:
        treatment_path = linked_path(graph) if all_links_valid and entry_valid else None
    except ValueError:
        treatment_path = None
    if set(predicted_ranks) == set(range(len(nodes))):
        ordinary_path = tuple(
            sorted(range(len(nodes)), key=predicted_ranks.__getitem__)
        )
        ordinary_graph = rewire_path(graph, ordinary_path)
    else:
        ordinary_path = None
        ordinary_graph = None
    return {
        "graph": graph,
        "treatment_path": treatment_path,
        "ordinary_graph": ordinary_graph,
        "ordinary_path": ordinary_path,
        "predicted_ranks": tuple(predicted_ranks),
        "role_labels": tuple(labels),
    }


def recode_operation_ids(example: S8GraphExample) -> S8GraphExample:
    names = sorted(example.operation_positions)
    if len(names) < 2:
        raise ValueError("S8 operation recoding requires two names")
    ids = list(example.ids)
    signatures = {
        name: tuple(ids[position] for position in positions[0])
        for name, positions in example.operation_positions.items()
    }
    rotated = names[1:] + names[:1]
    replacement = dict(zip(names, rotated, strict=True))
    for name, occurrences in example.operation_positions.items():
        target = signatures[replacement[name]]
        for positions in occurrences:
            if len(positions) != len(target):
                raise ValueError("S8 operation nonce widths differ")
            for position, token_id in zip(positions, target, strict=True):
                ids[position] = token_id
    return replace(example, ids=tuple(ids))


def gold_graph(example: S8GraphExample) -> NilLinkedLawGraph:
    row = example.row
    tag_to_node = {
        str(node["tag"]): index for index, node in enumerate(row["nodes"])
    }
    return NilLinkedLawGraph(
        modulus=int(row["modulus"]),
        initial_state=tuple(int(value) for value in row["initial_state"]),
        cards=tuple(
            LawCardNode(
                str(card["operation"]),
                int(card["y0"]),
                int(card["y1"]),
            )
            for card in row["cards"]
        ),
        nodes=tuple(
            EventNode(
                identity=int(node["identity"]),
                operation=str(node["operation"]),
                next_node=(
                    NIL
                    if node["next_tag"] is None
                    else tag_to_node[str(node["next_tag"])]
                ),
            )
            for node in row["nodes"]
        ),
        entry_node=int(row["entry_node"]),
        query_position=int(row["query_position"]),
    )


def semantic_graph_key(graph: NilLinkedLawGraph) -> tuple[object, ...]:
    cards = {card.operation: (card.y0, card.y1) for card in graph.cards}
    return (
        graph.modulus,
        graph.initial_state,
        tuple(sorted(cards.values())),
        tuple(
            (node.identity, cards[node.operation], node.next_node)
            for node in graph.nodes
        ),
        graph.entry_node,
        graph.query_position,
    )


def reindex_graph(
    graph: NilLinkedLawGraph, permutation: Sequence[int]
) -> NilLinkedLawGraph:
    """Rename graph node IDs without changing its linked computation."""

    mapping = tuple(int(value) for value in permutation)
    if set(mapping) != set(range(len(graph.nodes))) or len(mapping) != len(graph.nodes):
        raise ValueError("S8 node reindexing requires a complete permutation")
    nodes: list[EventNode | None] = [None] * len(graph.nodes)
    for old_id, node in enumerate(graph.nodes):
        new_id = mapping[old_id]
        next_node = NIL if node.next_node == NIL else mapping[node.next_node]
        nodes[new_id] = EventNode(node.identity, node.operation, next_node)
    result = NilLinkedLawGraph(
        modulus=graph.modulus,
        initial_state=graph.initial_state,
        cards=graph.cards,
        nodes=tuple(node for node in nodes if node is not None),
        entry_node=mapping[graph.entry_node],
        query_position=graph.query_position,
    )
    linked_path(result)
    return result
