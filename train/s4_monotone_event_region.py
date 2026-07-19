"""Zero-fit monotone event-local decoding over the frozen S4 v1 parser."""

from __future__ import annotations

import torch

from s4_set_identity_event_bus import (
    carrier_logits,
    masked_distribution,
    roster_carriers,
    roster_distributions,
    vocabulary_carrier,
)
from self_delimiting_event_tape import ROLE_INDEX, execute_program, pattern_windows


def discover_kind_windows(example, outputs, row, lexicon):
    """Return non-overlapping lexical kinds admitted by model-owned kind-role labels."""
    length = len(example.ids)
    labels = outputs["role_logits"][row, :length].argmax(-1).tolist()
    records = [
        {"token_ids": record["token_ids"], "value": record["kind"]}
        for record in lexicon["patterns"]
    ]
    kind_role = ROLE_INDEX["event.kind"]
    windows = tuple(
        window for window in pattern_windows(example.ids, records)
        if any(labels[position] == kind_role for position in range(window[0], window[1]))
    )
    if any(windows[index][0] < windows[index - 1][1] for index in range(1, len(windows))):
        raise ValueError("overlapping kind anchors")
    return windows


def monotone_event_regions(kind_windows, length):
    """Partition the sequence at gap midpoints between consecutive kind anchors."""
    if not kind_windows:
        return ()
    cuts = [0]
    for previous, current in zip(kind_windows, kind_windows[1:]):
        boundary = (int(previous[1]) + int(current[0])) // 2
        if boundary < cuts[-1] or boundary > length:
            raise ValueError("invalid monotone event boundary")
        cuts.append(boundary)
    cuts.append(int(length))
    regions = tuple((cuts[index], cuts[index + 1]) for index in range(len(kind_windows)))
    for (start, end), (kind_start, kind_end, _) in zip(regions, kind_windows):
        if not (start <= kind_start < kind_end <= end):
            raise ValueError("event region does not contain its kind anchor")
    return regions


def region_distribution(logits, valid, start, end):
    positions = torch.arange(valid.numel(), device=valid.device)
    region_valid = valid & (positions >= int(start)) & (positions < int(end))
    if not bool(region_valid.any()):
        raise ValueError("empty event region")
    return masked_distribution(logits, region_valid)


def decode_monotone_event_region(
    parser,
    example,
    outputs,
    row,
    valid,
    lexicon,
    roster_permutation=None,
    region_shift=0,
):
    """Decode event arguments from local regions without fitting another parser."""
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

    program = []
    assigned_regions = []
    for index, (_, _, kind) in enumerate(kinds):
        region_index = (index + int(region_shift)) % len(regions)
        start, end = regions[region_index]
        assigned_regions.append((start, end))
        entity_weights = region_distribution(
            outputs["role_logits"][row, :length, ROLE_INDEX["event.entity"]],
            row_valid,
            start,
            end,
        )
        entity = vocabulary_carrier(ids, entity_weights, parser.model.cfg.vocab_size)
        identity = int(carrier_logits(entity, roster).argmax())
        literal_weights = region_distribution(
            outputs["role_logits"][row, :length, ROLE_INDEX["event.literal"]],
            row_valid,
            start,
            end,
        )
        amount_logits = torch.sum(
            literal_weights.unsqueeze(-1) * outputs["amount_logits"][row, :length].float(),
            dim=0,
        )
        amount = int(amount_logits.argmax()) + 1
        program.append((str(kind), identity, amount))

    query_weights = masked_distribution(
        outputs["role_logits"][row, :length, ROLE_INDEX["query.position"]],
        row_valid,
    )
    query_logits = torch.sum(
        query_weights.unsqueeze(-1) * outputs["query_logits"][row, :length].float(),
        dim=0,
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
        "regions": regions,
        "assigned_regions": tuple(assigned_regions),
    }
