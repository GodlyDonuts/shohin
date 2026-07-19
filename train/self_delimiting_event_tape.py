"""Model-owned variable-length event parser over a frozen Shohin transformer."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_LABELS = (
    "none",
    "intro.entity0",
    "intro.entity1",
    "intro.entity2",
    "event.kind",
    "event.entity",
    "event.literal",
    "query.position",
)
ROLE_INDEX = {label: index for index, label in enumerate(ROLE_LABELS)}
KIND_TO_ID = {"left": 0, "right": 1}
ID_TO_KIND = {value: key for key, value in KIND_TO_ID.items()}


@dataclass(frozen=True)
class EventTapeExample:
    ids: tuple
    roles: tuple
    intro_positions: tuple
    event_positions: tuple
    query_positions: tuple
    kind_targets: tuple
    amount_targets: tuple
    query_target: int
    program: tuple
    initial_ids: tuple
    answer_identity: int
    final_state: tuple
    row_id: str
    depth: int
    surface_type: str


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_state(module):
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if not name.startswith("model.")
    }


def adapter_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(module).items()):
        tensor = tensor.contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def execute_program(program, query_position):
    state = [0, 1, 2]
    for kind, identity, amount in program:
        location = state.index(int(identity))
        destination = (
            max(0, location - int(amount))
            if kind == "left" else
            min(2, location + int(amount))
        )
        state.insert(destination, state.pop(location))
    return tuple(state), state[int(query_position)]


def compile_row(row, tokenizer):
    encoding = tokenizer.encode(row["question"])
    token_hash = hashlib.sha256(
        json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if token_hash != row["token_ids_sha256"]:
        raise ValueError("tokenizer mismatch for {}".format(row.get("id")))
    roles = [ROLE_INDEX["none"]] * len(encoding.ids)

    def positions(label):
        values = tuple(map(int, row["spans"][label]["token_positions"]))
        if not values or min(values) < 0 or max(values) >= len(roles):
            raise ValueError("invalid {} span in {}".format(label, row.get("id")))
        return values

    intro_positions = tuple(positions("intro.entity{}".format(index)) for index in range(3))
    for index, values in enumerate(intro_positions):
        for position in values:
            roles[position] = ROLE_INDEX["intro.entity{}".format(index)]
    event_positions = []
    program = []
    kind_targets = []
    amount_targets = []
    initial = tuple(row["initial_order"])
    for index, operation in enumerate(row["program"]):
        current = {
            role: positions("op{}.{}".format(index, role))
            for role in ("kind", "entity", "literal")
        }
        for role, values in current.items():
            for position in values:
                if roles[position] != ROLE_INDEX["none"]:
                    raise ValueError("overlapping role targets in {}".format(row.get("id")))
                roles[position] = ROLE_INDEX["event.{}".format(role)]
        identity = initial.index(operation["entity"])
        program.append((str(operation["kind"]), identity, int(operation["amount"])))
        kind_targets.append(KIND_TO_ID[operation["kind"]])
        amount_targets.append(int(operation["amount"]) - 1)
        event_positions.append(current)
    query_positions = positions("query.position")
    for position in query_positions:
        if roles[position] != ROLE_INDEX["none"]:
            raise ValueError("overlapping query target")
        roles[position] = ROLE_INDEX["query.position"]
    final_state, answer_identity = execute_program(program, int(row["query"]["position"]))
    if row["initial_order"][answer_identity] != row["answer"]:
        raise ValueError("symbolic answer mismatch")
    return EventTapeExample(
        ids=tuple(encoding.ids),
        roles=tuple(roles),
        intro_positions=intro_positions,
        event_positions=tuple(event_positions),
        query_positions=query_positions,
        kind_targets=tuple(kind_targets),
        amount_targets=tuple(amount_targets),
        query_target=int(row["query"]["position"]),
        program=tuple(program),
        initial_ids=tuple(tuple(encoding.ids[position] for position in values)
                          for values in intro_positions),
        answer_identity=int(answer_identity),
        final_state=final_state,
        row_id=str(row["id"]),
        depth=int(row["depth"]),
        surface_type=str(row["surface_type"]),
    )


def load_examples(path, tokenizer, expected_split, seq_len):
    examples = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError("split mismatch at row {}".format(line_number))
            example = compile_row(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError("row {} exceeds model context".format(line_number))
            examples.append(example)
    if not examples:
        raise ValueError("no event-tape examples")
    return examples


def shuffle_supervision(examples, seed):
    """Destroy source-label alignment while preserving each row's target inventory."""
    rng = random.Random(seed)
    shuffled = []
    for example in examples:
        permutation = list(range(len(example.ids)))
        rng.shuffle(permutation)

        def remap(values):
            return tuple(sorted(permutation[position] for position in values))

        intro = tuple(remap(values) for values in example.intro_positions)
        events = tuple({role: remap(values) for role, values in event.items()}
                       for event in example.event_positions)
        query = remap(example.query_positions)
        roles = [ROLE_INDEX["none"]] * len(example.ids)
        for index, values in enumerate(intro):
            for position in values:
                roles[position] = ROLE_INDEX["intro.entity{}".format(index)]
        for event in events:
            for role, values in event.items():
                for position in values:
                    roles[position] = ROLE_INDEX["event.{}".format(role)]
        for position in query:
            roles[position] = ROLE_INDEX["query.position"]
        shuffled.append(replace(
            example,
            roles=tuple(roles),
            intro_positions=intro,
            event_positions=events,
            query_positions=query,
        ))
    return shuffled


def make_batches(examples, batch_size, seed, shuffle=True):
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def pad_batch(examples, indices, device):
    selected = [examples[index] for index in indices]
    length = max(len(example.ids) for example in selected)
    ids = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    valid = torch.zeros((len(selected), length), dtype=torch.bool, device=device)
    roles = torch.full(
        (len(selected), length), -100, dtype=torch.long, device=device,
    )
    for row, example in enumerate(selected):
        width = len(example.ids)
        ids[row, :width] = torch.tensor(example.ids, dtype=torch.long, device=device)
        valid[row, :width] = True
        roles[row, :width] = torch.tensor(example.roles, dtype=torch.long, device=device)
    return selected, ids, valid, roles


class SelfDelimitingEventTapeParser(nn.Module):
    def __init__(self, model, layer=19, width=384, heads=8, encoder_layers=5, ff=1408):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("event parser requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid frozen layer")
        if width % heads or encoder_layers <= 0:
            raise ValueError("invalid event parser dimensions")
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
        self.kind_head = nn.Linear(width, 2)
        self.amount_head = nn.Linear(width, 2)
        self.query_head = nn.Linear(width, 3)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self):
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def initialize_memory_encoder(self, state):
        own = self.state_dict()
        loaded = []
        for name, value in state.items():
            if name.startswith("model.") or name not in own or own[name].shape != value.shape:
                continue
            if name.startswith(("memory_norm.", "memory_projection.", "memory_encoder.")):
                own[name].copy_(value)
                loaded.append(name)
        expected = [
            name for name in own
            if name.startswith(("memory_norm.", "memory_projection.", "memory_encoder."))
        ]
        if set(loaded) != set(expected):
            raise ValueError("memory initialization incomplete")
        return tuple(sorted(loaded))

    def encode(self, ids):
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.tok(ids)
            cos = self.model.cos[:ids.shape[1]].to(hidden.device)
            sin = self.model.sin[:ids.shape[1]].to(hidden.device)
            for block in self.model.blocks[:self.layer + 1]:
                hidden, _ = block(hidden, cos, sin)
        return hidden.detach()

    def forward(self, ids, valid):
        hidden = self.encode(ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        return {
            "memory": memory,
            "role_logits": self.role_head(memory).float(),
            "kind_logits": self.kind_head(memory).float(),
            "amount_logits": self.amount_head(memory).float(),
            "query_logits": self.query_head(memory).float(),
        }


def mean_positions(logits, row, positions):
    index = torch.tensor(positions, dtype=torch.long, device=logits.device)
    return logits[row].index_select(0, index).mean(0)


def parser_loss(outputs, examples, roles, role_weight=1.0, semantic_weight=1.0):
    valid_targets = roles[roles >= 0]
    counts = torch.bincount(valid_targets, minlength=len(ROLE_LABELS)).float()
    weights = valid_targets.numel() / (len(ROLE_LABELS) * counts.clamp_min(1.0))
    weights = weights.clamp(0.1, 20.0).to(outputs["role_logits"].device)
    role = F.cross_entropy(
        outputs["role_logits"].reshape(-1, len(ROLE_LABELS)),
        roles.reshape(-1),
        weight=weights,
        ignore_index=-100,
    )
    kind_logits = []
    kind_targets = []
    amount_logits = []
    amount_targets = []
    query_logits = []
    query_targets = []
    for row, example in enumerate(examples):
        for index, event in enumerate(example.event_positions):
            kind_logits.append(mean_positions(outputs["kind_logits"], row, event["kind"]))
            kind_targets.append(example.kind_targets[index])
            amount_logits.append(mean_positions(outputs["amount_logits"], row, event["literal"]))
            amount_targets.append(example.amount_targets[index])
        query_logits.append(mean_positions(outputs["query_logits"], row, example.query_positions))
        query_targets.append(example.query_target)
    kind = F.cross_entropy(
        torch.stack(kind_logits), torch.tensor(kind_targets, device=roles.device),
    )
    amount = F.cross_entropy(
        torch.stack(amount_logits), torch.tensor(amount_targets, device=roles.device),
    )
    query = F.cross_entropy(
        torch.stack(query_logits), torch.tensor(query_targets, device=roles.device),
    )
    semantic = (kind + amount + query) / 3.0
    return float(role_weight) * role + float(semantic_weight) * semantic, {
        "role": role,
        "kind": kind,
        "amount": amount,
        "query": query,
    }


def contiguous_runs(labels, target):
    runs = []
    start = None
    for index, value in enumerate(labels):
        if int(value) == int(target) and start is None:
            start = index
        if int(value) != int(target) and start is not None:
            runs.append(tuple(range(start, index)))
            start = None
    if start is not None:
        runs.append(tuple(range(start, len(labels))))
    return tuple(runs)


def build_kind_lexicon(examples):
    patterns = {}
    references = 0
    for example in examples:
        for index, event in enumerate(example.event_positions):
            token_ids = tuple(example.ids[position] for position in event["kind"])
            kind = ID_TO_KIND[example.kind_targets[index]]
            if token_ids in patterns and patterns[token_ids] != kind:
                raise ValueError("kind lexicon collision")
            patterns[token_ids] = kind
            references += 1
    return {
        "schema": "r12_s4_training_kind_lexicon_v1",
        "patterns": [
            {"token_ids": list(token_ids), "kind": kind}
            for token_ids, kind in sorted(patterns.items())
        ],
        "references": references,
        "pattern_count": len(patterns),
        "development_access": 0,
        "confirmation_access": 0,
    }


def decode_example(example, outputs, row, lexicon, host_count=False):
    length = len(example.ids)
    labels = outputs["role_logits"][row, :length].argmax(-1).tolist()
    intro_runs = [
        contiguous_runs(labels, ROLE_INDEX["intro.entity{}".format(index)])
        for index in range(3)
    ]
    kind_runs = contiguous_runs(labels, ROLE_INDEX["event.kind"])
    entity_runs = contiguous_runs(labels, ROLE_INDEX["event.entity"])
    literal_runs = contiguous_runs(labels, ROLE_INDEX["event.literal"])
    query_runs = contiguous_runs(labels, ROLE_INDEX["query.position"])
    raw_counts = (len(kind_runs), len(entity_runs), len(literal_runs))
    required = example.depth if host_count else None
    if any(len(runs) != 1 for runs in intro_runs) or len(query_runs) != 1:
        return {"valid": False, "event_count": min(raw_counts), "raw_counts": raw_counts}
    if host_count:
        if any(count < required for count in raw_counts):
            return {"valid": False, "event_count": min(raw_counts), "raw_counts": raw_counts}
        kind_runs = kind_runs[:required]
        entity_runs = entity_runs[:required]
        literal_runs = literal_runs[:required]
    elif len(set(raw_counts)) != 1:
        return {"valid": False, "event_count": min(raw_counts), "raw_counts": raw_counts}
    intro_ids = [tuple(example.ids[position] for position in runs[0]) for runs in intro_runs]
    lexicon_map = {
        tuple(record["token_ids"]): KIND_TO_ID[record["kind"]]
        for record in lexicon["patterns"]
    }
    program = []
    lexical_matches = []
    for kind_span, entity_span, literal_span in zip(kind_runs, entity_runs, literal_runs):
        kind_ids = tuple(example.ids[position] for position in kind_span)
        if kind_ids in lexicon_map:
            kind_id = lexicon_map[kind_ids]
            lexical_matches.append(True)
        else:
            kind_id = int(mean_positions(outputs["kind_logits"], row, kind_span).argmax())
            lexical_matches.append(False)
        entity_ids = tuple(example.ids[position] for position in entity_span)
        identities = [index for index, values in enumerate(intro_ids) if values == entity_ids]
        if len(identities) != 1:
            return {"valid": False, "event_count": len(kind_runs), "raw_counts": raw_counts}
        amount = int(mean_positions(outputs["amount_logits"], row, literal_span).argmax()) + 1
        program.append((ID_TO_KIND[kind_id], identities[0], amount))
    query = int(mean_positions(outputs["query_logits"], row, query_runs[0]).argmax())
    final_state, answer = execute_program(program, query)
    return {
        "valid": True,
        "event_count": len(program),
        "raw_counts": raw_counts,
        "program": tuple(program),
        "query": query,
        "final_state": final_state,
        "answer_identity": answer,
        "lexical_matches": tuple(lexical_matches),
        "intro_ids": tuple(intro_ids),
    }
