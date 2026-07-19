"""Complete source-pointer compiler over a frozen Shohin transformer.

The neural input is only the tokenized source. Learned program-slot queries
must select all initial-state, operation, literal, and query spans. Structured
rows are labels and evaluation evidence; they never enter ``forward``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_LABELS = (
    "intro.entity0", "intro.entity1", "intro.entity2",
    "op0.kind", "op0.entity", "op0.literal",
    "op1.kind", "op1.entity", "op1.literal",
    "query.position",
)
TARGET_INDEX = {label: index for index, label in enumerate(TARGET_LABELS)}
SLOT_FOR_TARGET = {
    "intro.entity0": 0,
    "intro.entity1": 1,
    "intro.entity2": 2,
    "op0.kind": 3,
    "op0.entity": 3,
    "op0.literal": 3,
    "op1.kind": 4,
    "op1.entity": 4,
    "op1.literal": 4,
    "query.position": 5,
}
FAMILY_FOR_TARGET = {
    "intro.entity0": "intro",
    "intro.entity1": "intro",
    "intro.entity2": "intro",
    "op0.kind": "kind",
    "op0.entity": "entity",
    "op0.literal": "literal",
    "op1.kind": "kind",
    "op1.entity": "entity",
    "op1.literal": "literal",
    "query.position": "query",
}
KIND_TO_ID = {"left": 0, "right": 1}
ID_TO_KIND = {value: key for key, value in KIND_TO_ID.items()}
QUERY_POSITIONS = (0, 1, 2)
WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class CompilerExample:
    ids: tuple
    target_positions: dict
    kind_targets: tuple
    row_id: str
    group: int
    surface_type: str
    question: str | None = None
    initial_order: tuple = ()
    program: tuple = ()
    query_position: int = -1
    answer: str | None = None
    factors: tuple = ()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_state(compiler):
    return {
        name: value.detach().cpu()
        for name, value in compiler.state_dict().items()
        if not name.startswith("model.")
    }


def adapter_hash(compiler):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(compiler).items()):
        tensor = tensor.contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compile_row(row, tokenizer, keep_evidence=False):
    encoding = tokenizer.encode(row["question"])
    token_hash = hashlib.sha256(
        json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if token_hash != row["token_ids_sha256"]:
        raise ValueError("tokenizer output does not match frozen row {}".format(row.get("id")))
    positions = {}
    for label in TARGET_LABELS:
        target = row.get("spans", {}).get(label)
        if not target or not target.get("token_positions"):
            raise ValueError("row {} lacks target {}".format(row.get("id"), label))
        values = tuple(map(int, target["token_positions"]))
        if min(values) < 0 or max(values) >= len(encoding.ids):
            raise ValueError("row {} target {} is out of range".format(row.get("id"), label))
        positions[label] = values
    kinds = tuple(KIND_TO_ID[operation["kind"]] for operation in row["program"])
    return CompilerExample(
        ids=tuple(encoding.ids),
        target_positions=positions,
        kind_targets=kinds,
        row_id=str(row["id"]),
        group=int(row["group"]),
        surface_type=str(row["surface_type"]),
        question=str(row["question"]) if keep_evidence else None,
        initial_order=tuple(row["initial_order"]) if keep_evidence else (),
        program=tuple(
            (str(operation["kind"]), str(operation["entity"]), int(operation["amount"]))
            for operation in row["program"]
        ) if keep_evidence else (),
        query_position=int(row["query"]["position"]) if keep_evidence else -1,
        answer=str(row["answer"]) if keep_evidence else None,
        factors=tuple(sorted(row.get("factors", {}).items())) if keep_evidence else (),
    )


def load_examples(path, tokenizer, expected_split, seq_len, keep_evidence=False, limit=0):
    examples = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError("row {} split mismatch".format(line_number))
            example = compile_row(row, tokenizer, keep_evidence=keep_evidence)
            if len(example.ids) > seq_len:
                raise ValueError("row {} exceeds model sequence length".format(line_number))
            examples.append(example)
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("no compiler examples loaded")
    return examples


def make_batches(examples, batch_size, seed, shuffle=True):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        buckets[len(example.ids)].append(index)
    rng = random.Random(seed)
    batches = []
    for length in sorted(buckets):
        indices = buckets[length]
        if shuffle:
            rng.shuffle(indices)
        batches.extend(
            indices[offset:offset + batch_size]
            for offset in range(0, len(indices), batch_size)
        )
    if shuffle:
        rng.shuffle(batches)
    return batches


def pad_batch(examples, indices, device):
    selected = [examples[index] for index in indices]
    length = max(len(example.ids) for example in selected)
    ids = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    valid = torch.zeros((len(selected), length), dtype=torch.bool, device=device)
    for row, example in enumerate(selected):
        ids[row, :len(example.ids)] = torch.tensor(example.ids, dtype=torch.long, device=device)
        valid[row, :len(example.ids)] = True
    return selected, ids, valid


class CompletePointerCompiler(nn.Module):
    """Six learned program slots over frozen causal token states."""

    def __init__(self, model, layer=19, width=256, heads=8, decoder_layers=2, ff=1024,
                 encoder_layers=0, role_supervision=False, separate_kind_decoder=False):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("complete compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid frozen layer")
        if width % heads:
            raise ValueError("compiler width must divide attention heads")
        self.model = model
        self.layer = int(layer)
        self.width = int(width)
        self.model.requires_grad_(False)
        self.memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.memory_projection = nn.Linear(model.cfg.d_model, width, bias=False)
        self.encoder_layers = int(encoder_layers)
        self.role_supervision = bool(role_supervision)
        self.separate_kind_decoder = bool(separate_kind_decoder)
        if self.encoder_layers:
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
        else:
            self.memory_encoder = None
        self.role_head = (
            nn.Linear(width, len(TARGET_LABELS)) if self.role_supervision else None
        )
        self.slot_queries = nn.Parameter(torch.empty(6, width))
        nn.init.normal_(self.slot_queries, mean=0.0, std=width ** -0.5)
        layer_module = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer_module, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(width)
        if self.separate_kind_decoder:
            self.kind_memory_norm = nn.LayerNorm(model.cfg.d_model)
            self.kind_memory_projection = nn.Linear(model.cfg.d_model, width, bias=False)
            kind_layer = nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=ff,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.kind_decoder = nn.TransformerDecoder(kind_layer, num_layers=decoder_layers)
            self.kind_output_norm = nn.LayerNorm(width)
        else:
            self.kind_memory_norm = None
            self.kind_memory_projection = None
            self.kind_decoder = None
            self.kind_output_norm = None
        families = sorted(set(FAMILY_FOR_TARGET.values()))
        self.pointer_keys = nn.ModuleDict({
            family: nn.Linear(width, width, bias=False) for family in families
        })
        self.pointer_queries = nn.ModuleDict({
            label.replace(".", "__"): nn.Linear(width, width, bias=False)
            for label in TARGET_LABELS
        })
        self.kind_head = nn.Linear(width, len(KIND_TO_ID))
        self.log_pointer_scale = nn.Parameter(torch.tensor(math.log(width ** -0.5)))

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self):
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def encode(self, ids):
        self.model.eval()
        with torch.no_grad():
            x = self.model.tok(ids)
            cos = self.model.cos[:ids.shape[1]].to(x.device)
            sin = self.model.sin[:ids.shape[1]].to(x.device)
            for block in self.model.blocks[:self.layer + 1]:
                x, _ = block(x, cos, sin)
        return x.detach()

    def forward(self, ids, valid_mask):
        if ids.ndim != 2 or valid_mask.shape != ids.shape:
            raise ValueError("ids and valid mask must be matching rank-2 tensors")
        hidden = self.encode(ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        if self.memory_encoder is not None:
            memory = self.memory_encoder(memory, src_key_padding_mask=~valid_mask)
        role_logits = self.role_head(memory).float() if self.role_head is not None else None
        slots = self.slot_queries.unsqueeze(0).expand(ids.shape[0], -1, -1)
        decoded = self.output_norm(self.decoder(
            tgt=slots,
            memory=memory,
            memory_key_padding_mask=~valid_mask,
        ))
        scale = self.log_pointer_scale.float().clamp(-8.0, 2.0).exp()
        pointer_logits = {}
        for label in TARGET_LABELS:
            family = FAMILY_FOR_TARGET[label]
            slot = decoded[:, SLOT_FOR_TARGET[label]]
            query = self.pointer_queries[label.replace(".", "__")](slot)
            keys = self.pointer_keys[family](memory)
            logits = torch.einsum("bd,bld->bl", query.float(), keys.float()) * scale
            if role_logits is not None:
                logits = logits + role_logits[:, :, TARGET_INDEX[label]]
            pointer_logits[label] = logits.masked_fill(~valid_mask, -1e9)
        if self.kind_decoder is not None:
            kind_memory = self.kind_memory_projection(self.kind_memory_norm(hidden))
            kind_slots = self.kind_output_norm(self.kind_decoder(
                tgt=slots,
                memory=kind_memory,
                memory_key_padding_mask=~valid_mask,
            ))
            kind_logits = self.kind_head(kind_slots[:, 3:5]).float()
        else:
            kind_logits = self.kind_head(decoded[:, 3:5]).float()
        return {
            "pointer_logits": pointer_logits,
            "kind_logits": kind_logits,
            "role_logits": role_logits,
        }


class OrdinaryTokenTaggerCompiler(nn.Module):
    """Favorable conventional bidirectional sequence-tagger control."""

    def __init__(self, model, layer=19, width=384, heads=8, encoder_layers=5,
                 ff=1408):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("ordinary compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid frozen layer")
        if width % heads:
            raise ValueError("compiler width must divide attention heads")
        if encoder_layers <= 0:
            raise ValueError("ordinary compiler requires a positive encoder depth")
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
        self.role_head = nn.Linear(width, len(TARGET_LABELS))
        self.kind_token_head = nn.Linear(width, 2 * len(KIND_TO_ID))

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self):
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def encode(self, ids):
        self.model.eval()
        with torch.no_grad():
            x = self.model.tok(ids)
            cos = self.model.cos[:ids.shape[1]].to(x.device)
            sin = self.model.sin[:ids.shape[1]].to(x.device)
            for block in self.model.blocks[:self.layer + 1]:
                x, _ = block(x, cos, sin)
        return x.detach()

    def forward(self, ids, valid_mask):
        if ids.ndim != 2 or valid_mask.shape != ids.shape:
            raise ValueError("ids and valid mask must be matching rank-2 tensors")
        hidden = self.encode(ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid_mask)
        role_logits = self.role_head(memory).float()
        pointer_logits = {
            label: role_logits[:, :, TARGET_INDEX[label]].masked_fill(~valid_mask, -1e9)
            for label in TARGET_LABELS
        }
        token_kind_logits = self.kind_token_head(memory).float().reshape(
            ids.shape[0], ids.shape[1], 2, len(KIND_TO_ID),
        )
        kind_logits = []
        for operation_index in range(2):
            span_logits = pointer_logits["op{}.kind".format(operation_index)]
            span_log_probabilities = F.log_softmax(span_logits, dim=-1)
            kind_logits.append(torch.logsumexp(
                span_log_probabilities.unsqueeze(-1)
                + token_kind_logits[:, :, operation_index, :],
                dim=1,
            ))
        return {
            "pointer_logits": pointer_logits,
            "kind_logits": torch.stack(kind_logits, dim=1),
            "role_logits": role_logits,
        }


def pointer_mass_loss(logits, examples, label):
    losses = []
    for row, example in enumerate(examples):
        targets = torch.tensor(
            example.target_positions[label], dtype=torch.long, device=logits.device,
        )
        log_probabilities = F.log_softmax(logits[row, :len(example.ids)].float(), dim=-1)
        losses.append(-torch.logsumexp(log_probabilities.index_select(0, targets), dim=0))
    return torch.stack(losses).mean()


def compiler_loss(outputs, examples, kind_weight=1.0):
    pointer_losses = {
        label: pointer_mass_loss(outputs["pointer_logits"][label], examples, label)
        for label in TARGET_LABELS
    }
    pointer = torch.stack(list(pointer_losses.values())).mean()
    kind_targets = torch.tensor(
        [example.kind_targets for example in examples],
        dtype=torch.long,
        device=outputs["kind_logits"].device,
    )
    kind = F.cross_entropy(outputs["kind_logits"].reshape(-1, 2), kind_targets.reshape(-1))
    total = pointer + float(kind_weight) * kind
    return total, pointer, kind, pointer_losses


def role_supervision_loss(outputs, examples):
    """Balanced token-role loss over the same gold source spans as the pointers."""
    logits = outputs.get("role_logits")
    if logits is None:
        raise ValueError("compiler does not expose role logits")
    losses = []
    for label, column in TARGET_INDEX.items():
        for row, example in enumerate(examples):
            length = len(example.ids)
            targets = torch.zeros(length, dtype=torch.float32, device=logits.device)
            targets[list(example.target_positions[label])] = 1.0
            positives = float(targets.sum().item())
            pos_weight = torch.tensor(
                max(1.0, (length - positives) / max(1.0, positives)),
                dtype=torch.float32,
                device=logits.device,
            )
            losses.append(F.binary_cross_entropy_with_logits(
                logits[row, :length, column].float(),
                targets,
                pos_weight=pos_weight,
            ))
    return torch.stack(losses).mean()


def predictions_from_outputs(outputs):
    pointers = {
        label: outputs["pointer_logits"][label].argmax(dim=-1).tolist()
        for label in TARGET_LABELS
    }
    kinds = outputs["kind_logits"].argmax(dim=-1).tolist()
    return pointers, kinds


def word_at_token(question, encoding, token_position):
    if not 0 <= int(token_position) < len(encoding.offsets):
        return None
    left, right = encoding.offsets[int(token_position)]
    if right <= left:
        return None
    for match in WORD.finditer(question):
        if match.start() <= left < match.end() or match.start() < right <= match.end():
            return match.group(0)
    return None


def execute_prediction(example, encoding, pointer_predictions, kind_predictions):
    if example.question is None:
        raise ValueError("execution requires retained evidence text")
    initial = tuple(
        word_at_token(example.question, encoding, pointer_predictions["intro.entity{}".format(index)])
        for index in range(3)
    )
    if None in initial or len(set(initial)) != 3:
        return None, None
    program = []
    for index in range(2):
        entity = word_at_token(example.question, encoding, pointer_predictions["op{}.entity".format(index)])
        literal = word_at_token(example.question, encoding, pointer_predictions["op{}.literal".format(index)])
        try:
            amount = int(literal)
        except (TypeError, ValueError):
            return None, None
        if entity not in initial or amount not in (1, 2):
            return None, None
        program.append((ID_TO_KIND[int(kind_predictions[index])], entity, amount))
    query_literal = word_at_token(example.question, encoding, pointer_predictions["query.position"])
    try:
        query_position = int(query_literal) - 1
    except (TypeError, ValueError):
        return None, None
    if query_position not in QUERY_POSITIONS:
        return None, None
    state = list(initial)
    for direction, entity, amount in program:
        location = state.index(entity)
        destination = max(0, location - amount) if direction == "left" else min(2, location + amount)
        state.insert(destination, state.pop(location))
    semantic = {
        "initial_order": initial,
        "program": tuple(program),
        "query_position": query_position,
    }
    return state[query_position], semantic


def semantic_exact(example, semantic):
    if semantic is None:
        return False
    return (
        tuple(semantic["initial_order"]) == example.initial_order
        and tuple(semantic["program"]) == example.program
        and int(semantic["query_position"]) == example.query_position
    )
