"""Source-deleted recurrent executor for the referential list machine.

The frozen compiler first gathers a fixed packet from source-token states. The
executor accepts only that packet: it never receives source token IDs, a source
mask, pointer logits, or the full source memory. Its mutable state is a soft
permutation matrix over the three initial entities. One neural update cell is
reused for both operations and a separate query consumer reads the final state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from referential_literal_pointer_compiler import KIND_TO_ID


PACKET_POINTER_FIELDS = (
    "intro.entity0",
    "intro.entity1",
    "intro.entity2",
    "op0.kind",
    "op0.entity",
    "op0.literal",
    "op1.kind",
    "op1.entity",
    "op1.literal",
    "query.position",
)


@dataclass(frozen=True)
class ExecutionTargets:
    transition_sources: tuple[tuple[int, int, int], ...]
    entity_locations: tuple[int, ...]
    amounts: tuple[int, ...]
    query_position: int
    answer_identity: int
    final_identities: tuple[int, int, int]


def executor_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def execution_targets(example, operation_indices=(0, 1)):
    """Derive training-only permutation targets from one structured example."""
    if len(example.initial_order) != 3 or len(example.program) != 2:
        raise ValueError("executor examples require three entities and two operations")
    state = [0, 1, 2]
    transitions = []
    entity_locations = []
    amounts = []
    for operation_index in operation_indices:
        kind, entity, amount = example.program[int(operation_index)]
        identity = example.initial_order.index(entity)
        source = state.index(identity)
        destination = (
            max(0, source - int(amount))
            if kind == "left" else
            min(2, source + int(amount))
        )
        next_state = list(state)
        next_state.insert(destination, next_state.pop(source))
        transitions.append(tuple(state.index(identity) for identity in next_state))
        entity_locations.append(source)
        amounts.append(int(amount) - 1)
        state = next_state
    return ExecutionTargets(
        transition_sources=tuple(transitions),
        entity_locations=tuple(entity_locations),
        amounts=tuple(amounts),
        query_position=int(example.query_position),
        answer_identity=int(state[int(example.query_position)]),
        final_identities=tuple(state),
    )


def _gold_weights(examples, label, length, device, dtype):
    weights = torch.zeros((len(examples), length), device=device, dtype=dtype)
    for row, example in enumerate(examples):
        positions = tuple(map(int, example.target_positions[label]))
        if not positions:
            raise ValueError("empty gold packet span {}".format(label))
        weights[row, list(positions)] = 1.0 / len(positions)
    return weights


def _predicted_weights(logits, valid_mask, temperature):
    if temperature <= 0:
        raise ValueError("packet temperature must be positive")
    masked = logits.float().masked_fill(~valid_mask, -1e9)
    return F.softmax(masked / float(temperature), dim=-1)


def gather_source_deleted_packet(
    compiler_outputs,
    examples,
    valid_mask,
    oracle="none",
    temperature=1.0,
):
    """Gather a bounded packet and return no full-source tensor.

    ``oracle`` may replace structural pointers and/or operation-kind classes for
    diagnostic ceilings. The promotion path must use ``none``.
    """
    if oracle not in {"none", "lexical", "structure", "full"}:
        raise ValueError("unknown packet oracle {}".format(oracle))
    memory = compiler_outputs.get("memory")
    if memory is None or memory.ndim != 3:
        raise ValueError("compiler output lacks contextual memory")
    if valid_mask.shape != memory.shape[:2] or len(examples) != memory.shape[0]:
        raise ValueError("packet source shapes do not match")
    gathered = {}
    for label in PACKET_POINTER_FIELDS:
        if oracle in {"structure", "full"}:
            weights = _gold_weights(
                examples, label, memory.shape[1], memory.device, memory.dtype,
            )
        else:
            weights = _predicted_weights(
                compiler_outputs["pointer_logits"][label], valid_mask, temperature,
            ).to(memory.dtype)
        gathered[label] = torch.einsum("bl,bld->bd", weights, memory)
    if oracle in {"lexical", "full"}:
        kind_ids = torch.tensor(
            [example.kind_targets for example in examples],
            device=memory.device,
            dtype=torch.long,
        )
        kind_probabilities = F.one_hot(kind_ids, num_classes=len(KIND_TO_ID)).to(memory.dtype)
    else:
        kind_probabilities = F.softmax(
            compiler_outputs["kind_logits"].float(), dim=-1,
        ).to(memory.dtype)
    return {
        "initial_entities": torch.stack([
            gathered["intro.entity{}".format(index)] for index in range(3)
        ], dim=1),
        "operations": tuple({
            "kind_context": gathered["op{}.kind".format(index)],
            "entity": gathered["op{}.entity".format(index)],
            "literal": gathered["op{}.literal".format(index)],
            "kind_probabilities": kind_probabilities[:, index],
        } for index in range(2)),
        "query": gathered["query.position"],
        "oracle": oracle,
    }


def select_packet_operations(packet, operation_indices):
    indices = tuple(map(int, operation_indices))
    if any(index not in (0, 1) for index in indices):
        raise ValueError("operation packet index must be zero or one")
    return {
        "initial_entities": packet["initial_entities"],
        "operations": tuple(packet["operations"][index] for index in indices),
        "query": packet["query"],
        "oracle": packet.get("oracle", "none"),
    }


def shuffle_operation_packet(packet, permutation):
    """Apply a no-fit causal intervention to operation fields only."""
    permutation = torch.as_tensor(
        permutation, device=packet["initial_entities"].device, dtype=torch.long,
    )
    if permutation.ndim != 1 or permutation.numel() != packet["initial_entities"].shape[0]:
        raise ValueError("operation shuffle must be one index per batch row")
    return {
        "initial_entities": packet["initial_entities"],
        "operations": tuple({
            name: value.index_select(0, permutation)
            for name, value in operation.items()
        } for operation in packet["operations"]),
        "query": packet["query"],
        "oracle": packet.get("oracle", "none"),
    }


def shuffle_query_packet(packet, permutation):
    """Apply a no-fit causal intervention to the query field only."""
    permutation = torch.as_tensor(
        permutation, device=packet["initial_entities"].device, dtype=torch.long,
    )
    if permutation.ndim != 1 or permutation.numel() != packet["initial_entities"].shape[0]:
        raise ValueError("query shuffle must be one index per batch row")
    return {
        "initial_entities": packet["initial_entities"],
        "operations": packet["operations"],
        "query": packet["query"].index_select(0, permutation),
        "oracle": packet.get("oracle", "none"),
    }


def sinkhorn(logits, iterations=6):
    """Map transition logits to the interior of the Birkhoff polytope."""
    if logits.ndim != 3 or logits.shape[-2:] != (3, 3):
        raise ValueError("transition logits must have shape [batch,3,3]")
    log_probabilities = logits.float()
    for _ in range(int(iterations)):
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, dim=-1, keepdim=True,
        )
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, dim=-2, keepdim=True,
        )
    return log_probabilities.exp().to(logits.dtype)


class PermutationUpdateCell(nn.Module):
    """Predict one destination-to-source permutation from one operation packet."""

    def __init__(self, width):
        super().__init__()
        self.width = int(width)
        self.matcher = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 1),
        )
        self.transition = nn.Sequential(
            nn.Linear(6 * width + 5, 3 * width),
            nn.GELU(),
            nn.Linear(3 * width, 9),
        )

    def forward(self, state_entities, operation_context, operation_entity,
                literal_context, kind_probabilities, position_embeddings):
        operation_entity = operation_entity.unsqueeze(1).expand(-1, 3, -1)
        match_features = torch.cat((
            state_entities,
            operation_entity,
            (state_entities - operation_entity).abs(),
            state_entities * operation_entity,
        ), dim=-1)
        match_logits = self.matcher(match_features).squeeze(-1)
        positioned_state = state_entities + position_embeddings.unsqueeze(0)
        controller_input = torch.cat((
            positioned_state.reshape(state_entities.shape[0], -1),
            operation_context,
            operation_entity[:, 0],
            literal_context,
            kind_probabilities.float(),
            match_logits.float(),
        ), dim=-1)
        transition_logits = self.transition(controller_input).reshape(-1, 3, 3)
        return transition_logits, match_logits


class GatherDeletePermutationExecutor(nn.Module):
    """Tied model-owned updater plus independent final-state consumer."""

    def __init__(self, packet_width=384, width=192, tied=True, sinkhorn_iterations=6):
        super().__init__()
        self.packet_width = int(packet_width)
        self.width = int(width)
        self.tied = bool(tied)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.entity_encoder = nn.Sequential(
            nn.LayerNorm(packet_width),
            nn.Linear(packet_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.literal_encoder = nn.Sequential(
            nn.LayerNorm(packet_width),
            nn.Linear(packet_width, width),
            nn.GELU(),
        )
        self.operation_encoder = nn.Sequential(
            nn.LayerNorm(packet_width),
            nn.Linear(packet_width, width),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(packet_width),
            nn.Linear(packet_width, width),
            nn.GELU(),
        )
        self.kind_encoder = nn.Linear(len(KIND_TO_ID), width)
        self.operation_fusion = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
        )
        self.position_embeddings = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.position_embeddings, mean=0.0, std=width ** -0.5)
        self.cells = nn.ModuleList([
            PermutationUpdateCell(width) for _ in range(1 if tied else 2)
        ])
        self.amount_head = nn.Linear(width, 2)
        self.query_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, packet, cell_indices=None):
        initial = packet.get("initial_entities")
        operations = packet.get("operations")
        query = packet.get("query")
        if initial is None or initial.ndim != 3 or initial.shape[1] != 3:
            raise ValueError("executor requires three gathered initial entities")
        if query is None or query.ndim != 2 or not operations:
            raise ValueError("executor packet is incomplete")
        if initial.shape[-1] != self.packet_width or query.shape[-1] != self.packet_width:
            raise ValueError("executor packet width mismatch")
        if cell_indices is None:
            cell_indices = tuple(range(len(operations)))
        if len(cell_indices) != len(operations):
            raise ValueError("one cell index is required per operation")
        initial_entities = self.entity_encoder(initial)
        batch = initial.shape[0]
        assignment = torch.eye(
            3, dtype=initial_entities.dtype, device=initial_entities.device,
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        transition_logits = []
        transition_matrices = []
        match_logits = []
        amount_logits = []
        for operation, cell_index in zip(operations, cell_indices):
            state_entities = torch.bmm(assignment, initial_entities)
            entity = self.entity_encoder(operation["entity"])
            literal = self.literal_encoder(operation["literal"])
            kind = self.kind_encoder(operation["kind_probabilities"].float())
            operation_context = self.operation_fusion(torch.cat((
                self.operation_encoder(operation["kind_context"]), kind,
            ), dim=-1))
            cell = self.cells[0 if self.tied else int(cell_index)]
            logits, matches = cell(
                state_entities,
                operation_context,
                entity,
                literal,
                operation["kind_probabilities"],
                self.position_embeddings,
            )
            matrix = sinkhorn(logits, self.sinkhorn_iterations)
            assignment = torch.bmm(matrix, assignment)
            transition_logits.append(logits)
            transition_matrices.append(matrix)
            match_logits.append(matches)
            amount_logits.append(self.amount_head(literal))
        query_logits = self.query_head(self.query_encoder(query))
        query_probabilities = F.softmax(query_logits.float(), dim=-1).to(assignment.dtype)
        answer_probabilities = torch.bmm(
            query_probabilities.unsqueeze(1), assignment,
        ).squeeze(1).float()
        return {
            "transition_logits": tuple(transition_logits),
            "transition_matrices": tuple(transition_matrices),
            "entity_match_logits": tuple(match_logits),
            "amount_logits": tuple(amount_logits),
            "assignment": assignment,
            "query_logits": query_logits,
            "answer_probabilities": answer_probabilities,
        }


class SourceRetainedAnswerControl(nn.Module):
    """Favorable direct-answer control with unrestricted source-memory access."""

    def __init__(self, packet_width=384, width=192, heads=8, layers=2, ff=768):
        super().__init__()
        if width % heads:
            raise ValueError("source-retained width must divide attention heads")
        self.packet_width = int(packet_width)
        self.width = int(width)
        self.heads = int(heads)
        self.layers = int(layers)
        self.ff = int(ff)
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(packet_width),
            nn.Linear(packet_width, width),
        )
        self.answer_query = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.answer_query, mean=0.0, std=width ** -0.5)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(width)
        self.answer_head = nn.Linear(width, 3)

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, source_memory, valid_mask):
        if source_memory.ndim != 3 or valid_mask.shape != source_memory.shape[:2]:
            raise ValueError("source-retained control requires matching source memory and mask")
        if source_memory.shape[-1] != self.packet_width:
            raise ValueError("source-retained packet width mismatch")
        memory = self.memory_projection(source_memory)
        query = self.answer_query.unsqueeze(0).expand(source_memory.shape[0], -1, -1)
        decoded = self.output_norm(self.decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=~valid_mask,
        ))
        return {"answer_logits": self.answer_head(decoded[:, 0]).float()}


def executor_loss(outputs, targets, transition_weight=1.0, answer_weight=1.0,
                  query_weight=0.5, entity_weight=0.5, amount_weight=0.5):
    if not targets:
        raise ValueError("executor loss requires targets")
    device = outputs["query_logits"].device
    transition_losses = []
    entity_losses = []
    amount_losses = []
    for step, logits in enumerate(outputs["transition_logits"]):
        target = torch.tensor(
            [row.transition_sources[step] for row in targets],
            dtype=torch.long,
            device=device,
        )
        transition_losses.append(F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1)))
        entity_target = torch.tensor(
            [row.entity_locations[step] for row in targets],
            dtype=torch.long,
            device=device,
        )
        entity_losses.append(F.cross_entropy(outputs["entity_match_logits"][step], entity_target))
        amount_target = torch.tensor(
            [row.amounts[step] for row in targets],
            dtype=torch.long,
            device=device,
        )
        amount_losses.append(F.cross_entropy(outputs["amount_logits"][step], amount_target))
    query_target = torch.tensor(
        [row.query_position for row in targets], dtype=torch.long, device=device,
    )
    answer_target = torch.tensor(
        [row.answer_identity for row in targets], dtype=torch.long, device=device,
    )
    transition = torch.stack(transition_losses).mean()
    entity = torch.stack(entity_losses).mean()
    amount = torch.stack(amount_losses).mean()
    query = F.cross_entropy(outputs["query_logits"], query_target)
    answer = -torch.log(outputs["answer_probabilities"].clamp_min(1e-8)).gather(
        1, answer_target.unsqueeze(1),
    ).mean()
    total = (
        float(transition_weight) * transition
        + float(answer_weight) * answer
        + float(query_weight) * query
        + float(entity_weight) * entity
        + float(amount_weight) * amount
    )
    return {
        "total": total,
        "transition": transition,
        "answer": answer,
        "query": query,
        "entity": entity,
        "amount": amount,
    }
