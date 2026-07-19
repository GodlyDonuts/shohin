"""Hard event islands with soft roster/query interfaces over frozen S4 v1."""

from __future__ import annotations

import torch

from s4_monotone_event_region import discover_kind_windows, monotone_event_regions
from s4_set_identity_event_bus import (
    carrier_logits,
    masked_distribution,
    roster_carriers,
    roster_distributions,
    vocabulary_carrier,
)
from self_delimiting_event_tape import ROLE_INDEX, execute_program


def role_islands(role_logits, role, start, end):
    """Return complete contiguous target-role argmax islands inside one region."""
    target = ROLE_INDEX[role]
    labels = role_logits[start:end].argmax(-1).tolist()
    islands = []
    begin = None
    for offset, label in enumerate(labels):
        if label == target and begin is None:
            begin = start + offset
        if label != target and begin is not None:
            islands.append(tuple(range(begin, start + offset)))
            begin = None
    if begin is not None:
        islands.append(tuple(range(begin, end)))
    return tuple(islands)


def island_margin(role_logits, role, island):
    """Sum the frozen target-vs-best-other role margin over one complete island."""
    target = ROLE_INDEX[role]
    index = torch.tensor(island, dtype=torch.long, device=role_logits.device)
    selected = role_logits.index_select(0, index).float()
    target_logits = selected[:, target]
    alternatives = torch.cat((selected[:, :target], selected[:, target + 1:]), dim=-1)
    return float((target_logits - alternatives.max(-1).values).sum())


def select_role_island(role_logits, role, start, end):
    candidates = role_islands(role_logits, role, start, end)
    if not candidates:
        return None, 0
    ranked = [
        (island_margin(role_logits, role, island), len(island), -island[0], island)
        for island in candidates
    ]
    return max(ranked)[-1], len(candidates)


def uniform_island_distribution(island, length, device):
    weights = torch.zeros(length, dtype=torch.float32, device=device)
    index = torch.tensor(island, dtype=torch.long, device=device)
    weights.index_fill_(0, index, 1.0 / index.numel())
    return weights


def decode_hard_island_soft_interface(
    parser,
    example,
    outputs,
    row,
    valid,
    lexicon,
    roster_permutation=None,
    region_shift=0,
):
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
        if entity_island is None:
            return {
                "valid": False,
                "failure_reason": "entity_island",
                "event_count": len(kinds),
                "island_counts": tuple(island_counts),
            }
        if literal_island is None:
            return {
                "valid": False,
                "failure_reason": "literal_island",
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
        amount = int(amount_logits.argmax()) + 1
        program.append((str(kind), identity, amount))

    query_weights = masked_distribution(
        outputs["role_logits"][row, :length, ROLE_INDEX["query.position"]], row_valid,
    )
    query_logits = torch.sum(
        query_weights.unsqueeze(-1) * outputs["query_logits"][row, :length].float(), dim=0,
    )
    query = int(query_logits.argmax())
    final_state, answer = execute_program(program, query)
    return {
        "valid": True,
        "failure_reason": "none",
        "event_count": len(program),
        "program": tuple(program),
        "query": query,
        "final_state": final_state,
        "answer_identity": answer,
        "island_counts": tuple(island_counts),
    }
