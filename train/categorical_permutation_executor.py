"""Categorical S3 register for source-deleted referential execution."""

from __future__ import annotations

import hashlib
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from probe_rgde_relational_identity import ordered_sequence_scores, role_weights
from referential_gather_delete_executor import gather_source_deleted_packet
from referential_literal_pointer_compiler import KIND_TO_ID


PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_ID = {permutation: index for index, permutation in enumerate(PERMUTATIONS)}


def permutation_matrices(device=None, dtype=torch.float32):
    matrices = torch.zeros((len(PERMUTATIONS), 3, 3), device=device, dtype=dtype)
    for index, permutation in enumerate(PERMUTATIONS):
        for destination, source in enumerate(permutation):
            matrices[index, destination, source] = 1
    return matrices


def local_action_ids(device=None):
    """Return the closed pop-insert action table [location, kind, amount]."""
    actions = torch.empty((3, len(KIND_TO_ID), 2), dtype=torch.long, device=device)
    for source in range(3):
        for kind_name, kind_id in KIND_TO_ID.items():
            for amount_id in range(2):
                amount = amount_id + 1
                destination = (
                    max(0, source - amount)
                    if kind_name == "left" else
                    min(2, source + amount)
                )
                order = list(range(3))
                order.insert(destination, order.pop(source))
                actions[source, kind_id, amount_id] = PERMUTATION_TO_ID[tuple(order)]
    return actions


def module_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def straight_through_permutation(logits, matrices):
    probabilities = F.softmax(logits.float(), dim=-1).to(logits.dtype)
    soft = torch.einsum("bp,pij->bij", probabilities, matrices.to(logits.dtype))
    hard = matrices.index_select(0, logits.argmax(-1)).to(logits.dtype)
    return hard + (soft - soft.detach())


def categorical_identity_packet(
    compiler_outputs,
    examples,
    ids,
    valid,
    mode="mean",
):
    """Compile source roles into categorical identity plus non-identity fields."""
    if mode not in {"mean", "ordered", "gold"}:
        raise ValueError("unknown categorical identity mode {}".format(mode))
    base = gather_source_deleted_packet(
        compiler_outputs,
        examples,
        valid,
        oracle="none",
        packet_mode="lexical_sigmoid_span",
    )
    initial = F.normalize(base["initial_entities"].float(), dim=-1)
    ordered_weights = None
    if mode == "ordered":
        logits = compiler_outputs["pointer_logits"]
        ordered_weights = {
            label: role_weights(logits[label], valid, 0.5)
            for label in (
                "intro.entity0", "intro.entity1", "intro.entity2",
                "op0.entity", "op1.entity",
            )
        }
    operations = []
    for operation_index, operation in enumerate(base["operations"]):
        if mode == "mean":
            entity = F.normalize(operation["entity"].float(), dim=-1)
            scores = torch.einsum("bd,bid->bi", entity, initial)
            identities = scores.argmax(-1)
        elif mode == "ordered":
            scores = ordered_sequence_scores(
                ids,
                [ordered_weights["intro.entity{}".format(index)] for index in range(3)],
                ordered_weights["op{}.entity".format(operation_index)],
            )
            identities = scores.argmax(-1)
        else:
            identities = torch.tensor([
                example.initial_order.index(example.program[operation_index][1])
                for example in examples
            ], device=ids.device, dtype=torch.long)
        operations.append({
            "identity_probabilities": F.one_hot(identities, num_classes=3).float(),
            "kind_context": operation["kind_context"],
            "literal": operation["literal"],
            "kind_probabilities": operation["kind_probabilities"],
        })
    return {
        "operations": tuple(operations),
        "query": base["query"],
        "identity_mode": mode,
    }


def lexical_kind_predictions(ids, weights, lexicon, minimum_mass=0.5):
    """Decode a pointed kind span against training-only exact-token patterns."""
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("lexical kind decoder expects matching [batch,length] tensors")
    scores = torch.full(
        (ids.shape[0], len(KIND_TO_ID)), -1.0,
        device=ids.device, dtype=torch.float32,
    )
    for record in lexicon["patterns"]:
        pattern = torch.tensor(record["token_ids"], device=ids.device, dtype=ids.dtype)
        width = pattern.numel()
        if width < 1 or width > ids.shape[1]:
            continue
        matches = ids.unfold(1, width, 1).eq(pattern.view(1, 1, -1)).all(-1)
        mass = weights.float().unfold(1, width, 1).sum(-1)
        candidate = mass.masked_fill(~matches, -1.0).amax(-1)
        kind_id = KIND_TO_ID[record["kind"]]
        scores[:, kind_id] = torch.maximum(scores[:, kind_id], candidate)
    values, predictions = scores.max(-1)
    return predictions, values >= float(minimum_mass), scores


def pointer_anchored_kind_predictions(ids, pointer_logits, valid, lexicon):
    """Decode the exact lexicon pattern containing the global pointer anchor."""
    if ids.ndim != 2 or pointer_logits.shape != ids.shape or valid.shape != ids.shape:
        raise ValueError("pointer-anchor decoder expects matching [batch,length] tensors")
    if not valid.bool().any(-1).all():
        raise ValueError("pointer-anchor decoder requires a valid token in every row")
    masked_logits = pointer_logits.float().masked_fill(~valid.bool(), float("-inf"))
    anchors = masked_logits.argmax(-1)
    scores = torch.full(
        (ids.shape[0], len(KIND_TO_ID)), float("-inf"),
        device=ids.device, dtype=torch.float32,
    )
    for record in lexicon["patterns"]:
        pattern = torch.tensor(record["token_ids"], device=ids.device, dtype=ids.dtype)
        width = pattern.numel()
        if width < 1 or width > ids.shape[1]:
            continue
        token_windows = ids.unfold(1, width, 1)
        valid_windows = valid.bool().unfold(1, width, 1)
        matches = token_windows.eq(pattern.view(1, 1, -1)).all(-1)
        matches &= valid_windows.all(-1)
        starts = torch.arange(
            matches.shape[1], device=ids.device, dtype=anchors.dtype,
        ).view(1, -1)
        contains_anchor = matches & (starts <= anchors[:, None])
        contains_anchor &= anchors[:, None] < starts + width
        candidate = masked_logits.gather(1, anchors[:, None]).squeeze(1)
        candidate = candidate.masked_fill(~contains_anchor.any(-1), float("-inf"))
        kind_id = KIND_TO_ID[record["kind"]]
        scores[:, kind_id] = torch.maximum(scores[:, kind_id], candidate)
    matched_classes = torch.isfinite(scores)
    unambiguous = matched_classes.sum(-1) == 1
    predictions = scores.argmax(-1)
    return predictions, unambiguous, scores


def apply_lexical_kind_override(
    packet, compiler_outputs, ids, valid, lexicon, temperature=0.5, minimum_mass=0.5,
    decoder="mass",
):
    """Replace matched direction fields while retaining neural fallback."""
    if decoder not in {"mass", "pointer_anchor"}:
        raise ValueError("unknown lexical kind decoder {}".format(decoder))
    operations = []
    for index, operation in enumerate(packet["operations"]):
        pointer_logits = compiler_outputs["pointer_logits"]["op{}.kind".format(index)]
        if decoder == "mass":
            weights = role_weights(pointer_logits, valid, temperature)
            lexical, matched, scores = lexical_kind_predictions(
                ids, weights, lexicon, minimum_mass=minimum_mass,
            )
        else:
            lexical, matched, scores = pointer_anchored_kind_predictions(
                ids, pointer_logits, valid, lexicon,
            )
        neural = operation["kind_probabilities"].float().argmax(-1)
        selected = torch.where(matched, lexical, neural)
        updated = dict(operation)
        updated["kind_probabilities"] = F.one_hot(
            selected, num_classes=len(KIND_TO_ID),
        ).to(operation["kind_probabilities"].dtype)
        updated["kind_lexical_matched"] = matched
        updated["kind_lexical_scores"] = scores
        operations.append(updated)
    return {
        "operations": tuple(operations),
        "query": packet["query"],
        "identity_mode": packet.get("identity_mode", "unknown"),
    }


def select_categorical_operations(packet, indices):
    indices = tuple(map(int, indices))
    if any(index not in (0, 1) for index in indices):
        raise ValueError("categorical operation index must be zero or one")
    return {
        "operations": tuple(packet["operations"][index] for index in indices),
        "query": packet["query"],
        "identity_mode": packet.get("identity_mode", "unknown"),
    }


class CategoricalUpdateCell(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * width + 17, 3 * width),
            nn.GELU(),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, len(PERMUTATIONS)),
        )

    def forward(self, assignment, identity, location, operation, literal, kind):
        features = torch.cat((
            assignment.reshape(assignment.shape[0], -1),
            identity.float(),
            location.float(),
            kind.float(),
            operation,
            literal,
        ), dim=-1)
        return self.network(features)


class EquivariantCategoricalUpdateCell(nn.Module):
    """Predict a local S3 action without access to the global coordinate frame."""

    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * width + 5, 3 * width),
            nn.GELU(),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, len(PERMUTATIONS)),
        )

    def forward(self, assignment, identity, location, operation, literal, kind):
        del assignment, identity
        return self.network(torch.cat((
            location.float(), kind.float(), operation, literal,
        ), dim=-1))


class S3CategoricalPermutationExecutor(nn.Module):
    """Tied neural update over an exactly categorical S3 state register."""

    def __init__(self, identity_context_width, context_width, width=192):
        super().__init__()
        self.identity_context_width = int(identity_context_width)
        self.context_width = int(context_width)
        self.width = int(width)
        self.operation_encoder = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, width),
            nn.GELU(),
        )
        self.literal_encoder = nn.Sequential(
            nn.LayerNorm(identity_context_width),
            nn.Linear(identity_context_width, width),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, width),
            nn.GELU(),
        )
        self.kind_encoder = nn.Linear(len(KIND_TO_ID), width)
        self.operation_fusion = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
        )
        self.cell = CategoricalUpdateCell(width)
        self.amount_head = nn.Linear(width, 2)
        self.query_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )
        self.register_buffer("permutations", permutation_matrices(), persistent=True)

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, packet):
        operations = packet.get("operations")
        query = packet.get("query")
        if not operations or query is None or query.ndim != 2:
            raise ValueError("categorical executor packet is incomplete")
        batch = query.shape[0]
        assignment = torch.eye(
            3, device=query.device, dtype=query.dtype,
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        permutation_logits = []
        transition_matrices = []
        entity_match_logits = []
        amount_logits = []
        for operation in operations:
            identity = operation["identity_probabilities"].to(assignment.dtype)
            if identity.shape != (batch, 3):
                raise ValueError("categorical operation identity must have shape [batch,3]")
            location = torch.bmm(assignment, identity.unsqueeze(-1)).squeeze(-1)
            literal = self.literal_encoder(operation["literal"])
            kind = self.kind_encoder(operation["kind_probabilities"].float())
            operation_context = self.operation_fusion(torch.cat((
                self.operation_encoder(operation["kind_context"]), kind,
            ), dim=-1))
            logits = self.cell(
                assignment, identity, location, operation_context, literal,
                operation["kind_probabilities"],
            )
            matrix = straight_through_permutation(logits, self.permutations)
            assignment = torch.bmm(matrix, assignment)
            permutation_logits.append(logits)
            transition_matrices.append(matrix)
            entity_match_logits.append(torch.log(location.float().clamp_min(1e-8)))
            amount_logits.append(self.amount_head(literal))
        query_logits = self.query_head(self.query_encoder(query))
        query_probabilities = F.softmax(query_logits.float(), dim=-1).to(assignment.dtype)
        answer_probabilities = torch.bmm(
            query_probabilities.unsqueeze(1), assignment,
        ).squeeze(1).float()
        return {
            "permutation_logits": tuple(permutation_logits),
            "transition_matrices": tuple(transition_matrices),
            "entity_match_logits": tuple(entity_match_logits),
            "amount_logits": tuple(amount_logits),
            "assignment": assignment,
            "query_logits": query_logits,
            "answer_probabilities": answer_probabilities,
        }


class S3EquivariantPermutationExecutor(S3CategoricalPermutationExecutor):
    """S3 register whose transition law is invariant to global identity labels."""

    def __init__(self, identity_context_width, context_width, width=192):
        super().__init__(identity_context_width, context_width, width)
        self.cell = EquivariantCategoricalUpdateCell(width)


class S3ClosedActionPermutationExecutor(S3EquivariantPermutationExecutor):
    """Exact finite S3 action driven by model-predicted categorical fields."""

    def __init__(self, identity_context_width, context_width, width=192):
        super().__init__(identity_context_width, context_width, width)
        self.register_buffer("local_actions", local_action_ids(), persistent=False)

    def forward(self, packet):
        operations = packet.get("operations")
        query = packet.get("query")
        if not operations or query is None or query.ndim != 2:
            raise ValueError("categorical executor packet is incomplete")
        batch = query.shape[0]
        assignment = torch.eye(
            3, device=query.device, dtype=query.dtype,
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        permutation_logits = []
        transition_matrices = []
        entity_match_logits = []
        amount_logits = []
        kind_predictions = []
        for operation in operations:
            identity = operation["identity_probabilities"].to(assignment.dtype)
            if identity.shape != (batch, 3):
                raise ValueError("categorical operation identity must have shape [batch,3]")
            location = torch.bmm(assignment, identity.unsqueeze(-1)).squeeze(-1)
            literal = self.literal_encoder(operation["literal"])
            current_amount_logits = self.amount_head(literal)
            current_kind = operation["kind_probabilities"].float().argmax(-1)
            current_amount = current_amount_logits.argmax(-1)
            current_location = location.argmax(-1)
            action_ids = self.local_actions[
                current_location, current_kind, current_amount,
            ]
            matrix = self.permutations.index_select(0, action_ids).to(assignment.dtype)
            assignment = torch.bmm(matrix, assignment)
            logits = torch.full(
                (batch, len(PERMUTATIONS)), -1e4,
                dtype=assignment.dtype, device=assignment.device,
            )
            logits.scatter_(1, action_ids.unsqueeze(1), 0.0)
            permutation_logits.append(logits)
            transition_matrices.append(matrix)
            entity_match_logits.append(torch.log(location.float().clamp_min(1e-8)))
            amount_logits.append(current_amount_logits)
            kind_predictions.append(current_kind)
        query_logits = self.query_head(self.query_encoder(query))
        query_probabilities = F.softmax(query_logits.float(), dim=-1).to(assignment.dtype)
        answer_probabilities = torch.bmm(
            query_probabilities.unsqueeze(1), assignment,
        ).squeeze(1).float()
        return {
            "permutation_logits": tuple(permutation_logits),
            "transition_matrices": tuple(transition_matrices),
            "entity_match_logits": tuple(entity_match_logits),
            "amount_logits": tuple(amount_logits),
            "kind_predictions": tuple(kind_predictions),
            "assignment": assignment,
            "query_logits": query_logits,
            "answer_probabilities": answer_probabilities,
        }


def categorical_executor_loss(
    outputs,
    targets,
    transition_weight=1.0,
    answer_weight=1.0,
    query_weight=0.5,
    entity_weight=0.25,
    amount_weight=0.5,
):
    if not targets:
        raise ValueError("categorical executor loss requires targets")
    device = outputs["query_logits"].device
    transition_losses = []
    entity_losses = []
    amount_losses = []
    for step, logits in enumerate(outputs["permutation_logits"]):
        transition_target = torch.tensor([
            PERMUTATION_TO_ID[tuple(row.transition_sources[step])] for row in targets
        ], dtype=torch.long, device=device)
        transition_losses.append(F.cross_entropy(logits, transition_target))
        entity_target = torch.tensor([
            row.entity_locations[step] for row in targets
        ], dtype=torch.long, device=device)
        entity_losses.append(F.cross_entropy(outputs["entity_match_logits"][step], entity_target))
        amount_target = torch.tensor([
            row.amounts[step] for row in targets
        ], dtype=torch.long, device=device)
        amount_losses.append(F.cross_entropy(outputs["amount_logits"][step], amount_target))
    query_target = torch.tensor([
        row.query_position for row in targets
    ], dtype=torch.long, device=device)
    answer_target = torch.tensor([
        row.answer_identity for row in targets
    ], dtype=torch.long, device=device)
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
