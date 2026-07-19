"""Learned source-deleted S3 generator behind the frozen S4 v5 parser."""

from __future__ import annotations

import hashlib
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from s4_monotone_event_region import discover_kind_windows, monotone_event_regions
from s4_set_identity_event_bus import (
    carrier_logits,
    masked_distribution,
    roster_carriers,
    roster_distributions,
    vocabulary_carrier,
)
from self_delimiting_event_tape import ROLE_INDEX
from s4_hard_island_soft_interface import (
    select_role_island,
    uniform_island_distribution,
)


PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_ID = {value: index for index, value in enumerate(PERMUTATIONS)}


def permutation_matrices(device=None, dtype=torch.float32):
    matrices = torch.zeros((len(PERMUTATIONS), 3, 3), device=device, dtype=dtype)
    for index, permutation in enumerate(PERMUTATIONS):
        for destination, source in enumerate(permutation):
            matrices[index, destination, source] = 1
    return matrices


def module_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def unit_generator_examples(device=None):
    """Return the six supervised unit moves; amount two is intentionally absent."""
    locations = []
    directions = []
    targets = []
    for location in range(3):
        for direction in range(2):
            destination = (
                max(0, location - 1) if direction == 0 else min(2, location + 1)
            )
            order = list(range(3))
            order.insert(destination, order.pop(location))
            locations.append(location)
            directions.append(direction)
            targets.append(PERMUTATION_TO_ID[tuple(order)])
    return (
        torch.tensor(locations, dtype=torch.long, device=device),
        torch.tensor(directions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


class LearnedUnitGenerator(nn.Module):
    """Predict one local position permutation from location and direction only."""

    def __init__(self, width=64):
        super().__init__()
        self.width = int(width)
        self.network = nn.Sequential(
            nn.Linear(5, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, len(PERMUTATIONS)),
        )

    def forward(self, locations, directions):
        if locations.ndim != 1 or directions.shape != locations.shape:
            raise ValueError("generator inputs must be matching rank-one tensors")
        features = torch.cat((
            F.one_hot(locations.long(), num_classes=3).float(),
            F.one_hot(directions.long(), num_classes=2).float(),
        ), dim=-1)
        return self.network(features)

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())


class GeneratorFactoredS3Executor(nn.Module):
    """Compose a learned unit generator over an exact categorical state register."""

    def __init__(self, width=64):
        super().__init__()
        self.generator = LearnedUnitGenerator(width)
        self.register_buffer("permutations", permutation_matrices(), persistent=True)

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def primitive(self, assignment, identities, directions):
        identity = F.one_hot(identities.long(), num_classes=3).to(assignment.dtype)
        location = torch.bmm(assignment, identity.unsqueeze(-1)).squeeze(-1).argmax(-1)
        logits = self.generator(location, directions)
        action_ids = logits.argmax(-1)
        matrix = self.permutations.index_select(0, action_ids).to(assignment.dtype)
        return torch.bmm(matrix, assignment), logits, action_ids, location

    def forward(self, identities, directions, amounts, query_positions, reset_state=False):
        if identities.ndim != 2:
            raise ValueError("executor identities must have shape [batch,depth]")
        if directions.shape != identities.shape or amounts.shape != identities.shape:
            raise ValueError("executor operation tensors must have matching shapes")
        if query_positions.shape != identities.shape[:1]:
            raise ValueError("executor query must have shape [batch]")
        if not bool(((amounts == 1) | (amounts == 2)).all()):
            raise ValueError("executor amounts must be one or two")
        batch, depth = identities.shape
        eye = torch.eye(3, device=identities.device, dtype=torch.float32)
        assignment = eye.unsqueeze(0).expand(batch, -1, -1).clone()
        transitions = []
        micro_actions = []
        for step in range(depth):
            if reset_state:
                assignment = eye.unsqueeze(0).expand(batch, -1, -1).clone()
            step_actions = []
            for microstep in range(2):
                candidate, logits, action_ids, location = self.primitive(
                    assignment, identities[:, step], directions[:, step],
                )
                active = (amounts[:, step] > microstep).view(batch, 1, 1)
                assignment = torch.where(active, candidate, assignment)
                step_actions.append({
                    "logits": logits,
                    "action_ids": action_ids,
                    "locations": location,
                    "active": active[:, 0, 0],
                })
            transitions.append(assignment)
            micro_actions.append(tuple(step_actions))
        query = F.one_hot(query_positions.long(), num_classes=3).to(assignment.dtype)
        answer_probabilities = torch.bmm(query.unsqueeze(1), assignment).squeeze(1)
        return {
            "assignment": assignment,
            "transitions": tuple(transitions),
            "micro_actions": tuple(micro_actions),
            "answer_probabilities": answer_probabilities,
        }


def decode_v5_program(
    parser,
    example,
    outputs,
    row,
    valid,
    lexicon,
    roster_permutation=None,
    region_shift=0,
):
    """Extract only the frozen v5 categorical program; no execution occurs here."""
    length = len(example.ids)
    row_valid = valid[:length]
    ids = torch.tensor(example.ids, dtype=torch.long, device=row_valid.device)
    try:
        kinds = discover_kind_windows(example, outputs, row, lexicon)
    except ValueError:
        return {"valid": False, "failure_reason": "kind_overlap", "event_count": 0}
    if not kinds:
        return {"valid": False, "failure_reason": "no_kind_anchors", "event_count": 0}
    regions = monotone_event_regions(kinds, length)
    roster = roster_carriers(
        ids,
        roster_distributions(outputs, row, row_valid),
        parser.model.cfg.vocab_size,
    )
    if roster_permutation is not None:
        index = torch.tensor(roster_permutation, dtype=torch.long, device=roster.device)
        roster = roster.index_select(0, index)

    role_logits = outputs["role_logits"][row, :length]
    program = []
    island_counts = []
    for event_index, (_, _, kind) in enumerate(kinds):
        region_index = (event_index + int(region_shift)) % len(regions)
        start, end = regions[region_index]
        entity_island, entity_count = select_role_island(
            role_logits, "event.entity", start, end,
        )
        literal_island, literal_count = select_role_island(
            role_logits, "event.literal", start, end,
        )
        island_counts.append((entity_count, literal_count))
        if entity_island is None or literal_island is None:
            return {
                "valid": False,
                "failure_reason": (
                    "entity_island" if entity_island is None else "literal_island"
                ),
                "event_count": len(kinds),
                "island_counts": tuple(island_counts),
            }
        entity_weights = uniform_island_distribution(entity_island, length, ids.device)
        entity = vocabulary_carrier(ids, entity_weights, parser.model.cfg.vocab_size)
        identity = int(carrier_logits(entity, roster).argmax())
        literal_index = torch.tensor(literal_island, dtype=torch.long, device=ids.device)
        amount_logits = outputs["amount_logits"][row, :length].index_select(
            0, literal_index,
        ).float().mean(0)
        program.append((str(kind), identity, int(amount_logits.argmax()) + 1))

    query_weights = masked_distribution(
        outputs["role_logits"][row, :length, ROLE_INDEX["query.position"]], row_valid,
    )
    query_logits = torch.sum(
        query_weights.unsqueeze(-1) * outputs["query_logits"][row, :length].float(), dim=0,
    )
    return {
        "valid": True,
        "failure_reason": "none",
        "event_count": len(program),
        "program": tuple(program),
        "query": int(query_logits.argmax()),
        "island_counts": tuple(island_counts),
    }


def stack_programs(decoded, device, direction_rotation=False):
    if not decoded or any(not row.get("valid") for row in decoded):
        raise ValueError("cannot stack invalid decoded programs")
    depths = {len(row["program"]) for row in decoded}
    if len(depths) != 1:
        raise ValueError("program stack requires one depth")
    identities = torch.tensor(
        [[operation[1] for operation in row["program"]] for row in decoded],
        dtype=torch.long,
        device=device,
    )
    directions = torch.tensor(
        [[0 if operation[0] == "left" else 1 for operation in row["program"]]
         for row in decoded],
        dtype=torch.long,
        device=device,
    )
    if direction_rotation:
        directions = 1 - directions
    amounts = torch.tensor(
        [[operation[2] for operation in row["program"]] for row in decoded],
        dtype=torch.long,
        device=device,
    )
    queries = torch.tensor(
        [row["query"] for row in decoded], dtype=torch.long, device=device,
    )
    return identities, directions, amounts, queries
