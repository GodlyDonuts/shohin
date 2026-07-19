"""Event-relative start/end pointers over the frozen S4 v1 parser."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_delimiting_event_tape import (
    ROLE_INDEX,
    SelfDelimitingEventTapeParser,
    execute_program,
    mean_positions,
    pattern_windows,
)


POINTER_PREFIXES = (
    "intro_start_head.",
    "intro_end_head.",
    "query_start_head.",
    "query_end_head.",
    "event_entity_start_query.",
    "event_entity_start_key.",
    "event_entity_end_query.",
    "event_entity_end_key.",
    "event_literal_start_query.",
    "event_literal_start_key.",
    "event_literal_end_query.",
    "event_literal_end_key.",
)


class EventRelativePointerParser(SelfDelimitingEventTapeParser):
    def __init__(self, model, layer=19, width=384, heads=8, encoder_layers=5, ff=1408):
        super().__init__(model, layer, width, heads, encoder_layers, ff)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(False)
        self.intro_start_head = nn.Linear(width, 3)
        self.intro_end_head = nn.Linear(width, 3)
        self.query_start_head = nn.Linear(width, 1)
        self.query_end_head = nn.Linear(width, 1)
        self.event_entity_start_query = nn.Linear(width, width, bias=False)
        self.event_entity_start_key = nn.Linear(width, width, bias=False)
        self.event_entity_end_query = nn.Linear(width, width, bias=False)
        self.event_entity_end_key = nn.Linear(width, width, bias=False)
        self.event_literal_start_query = nn.Linear(width, width, bias=False)
        self.event_literal_start_key = nn.Linear(width, width, bias=False)
        self.event_literal_end_query = nn.Linear(width, width, bias=False)
        self.event_literal_end_key = nn.Linear(width, width, bias=False)

    def initialize_v1(self, state):
        own = self.state_dict()
        expected = {
            name for name in own
            if not name.startswith("model.") and not name.startswith(POINTER_PREFIXES)
        }
        supplied = {name for name in state if not name.startswith("model.")}
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError("v1 adapter mismatch missing={} unexpected={}".format(missing, unexpected))
        with torch.no_grad():
            for name in expected:
                if own[name].shape != state[name].shape:
                    raise ValueError("v1 adapter shape mismatch for {}".format(name))
                own[name].copy_(state[name])
        return tuple(sorted(expected))

    def pointer_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(POINTER_PREFIXES):
                yield parameter

    def pointer_num_params(self):
        return sum(parameter.numel() for parameter in self.pointer_parameters())

    def forward(self, ids, valid):
        outputs = super().forward(ids, valid)
        memory = outputs["memory"]
        outputs.update({
            "intro_start_logits": self.intro_start_head(memory).float(),
            "intro_end_logits": self.intro_end_head(memory).float(),
            "query_start_logits": self.query_start_head(memory).squeeze(-1).float(),
            "query_end_logits": self.query_end_head(memory).squeeze(-1).float(),
            "event_entity_start_keys": self.event_entity_start_key(memory),
            "event_entity_end_keys": self.event_entity_end_key(memory),
            "event_literal_start_keys": self.event_literal_start_key(memory),
            "event_literal_end_keys": self.event_literal_end_key(memory),
        })
        return outputs

    def event_pointer_scores(self, outputs, row, anchor_positions):
        memory = outputs["memory"]
        index = torch.tensor(anchor_positions, dtype=torch.long, device=memory.device)
        anchor = memory[row].index_select(0, index).mean(0)

        def score(query_layer, key_name):
            query = query_layer(anchor)
            keys = outputs[key_name][row]
            return torch.mv(keys, query) / math.sqrt(self.width)

        return {
            "entity_start": score(self.event_entity_start_query, "event_entity_start_keys"),
            "entity_end": score(self.event_entity_end_query, "event_entity_end_keys"),
            "literal_start": score(self.event_literal_start_query, "event_literal_start_keys"),
            "literal_end": score(self.event_literal_end_query, "event_literal_end_keys"),
        }


def masked_pointer_loss(logits, target, valid):
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked.unsqueeze(0), torch.tensor([target], device=logits.device))


def event_relative_pointer_loss(parser, outputs, examples, valid):
    losses = []
    components = {"intro": [], "query": [], "event_entity": [], "event_literal": []}
    for row, example in enumerate(examples):
        row_valid = valid[row, :len(example.ids)]
        for identity, positions in enumerate(example.intro_positions):
            for name, logits, target in (
                ("intro", outputs["intro_start_logits"][row, :len(example.ids), identity], min(positions)),
                ("intro", outputs["intro_end_logits"][row, :len(example.ids), identity], max(positions)),
            ):
                value = masked_pointer_loss(logits, target, row_valid)
                losses.append(value)
                components[name].append(value)
        for name, logits, target in (
            ("query", outputs["query_start_logits"][row, :len(example.ids)], min(example.query_positions)),
            ("query", outputs["query_end_logits"][row, :len(example.ids)], max(example.query_positions)),
        ):
            value = masked_pointer_loss(logits, target, row_valid)
            losses.append(value)
            components[name].append(value)
        for event in example.event_positions:
            scores = parser.event_pointer_scores(outputs, row, event["kind"])
            for name, target in (
                ("entity_start", min(event["entity"])),
                ("entity_end", max(event["entity"])),
                ("literal_start", min(event["literal"])),
                ("literal_end", max(event["literal"])),
            ):
                group = "event_entity" if name.startswith("entity") else "event_literal"
                value = masked_pointer_loss(scores[name][:len(example.ids)], target, row_valid)
                losses.append(value)
                components[group].append(value)
    means = {
        name: torch.stack(values).mean()
        for name, values in components.items()
    }
    return torch.stack(losses).mean(), means


def pointer_span(start_logits, end_logits, length):
    start = int(start_logits[:length].argmax())
    end = int(end_logits[:length].argmax())
    if end < start:
        return None
    return tuple(range(start, end + 1))


def decode_event_relative_example(parser, example, outputs, row, lexicon):
    length = len(example.ids)
    ids = example.ids
    labels = outputs["role_logits"][row, :length].argmax(-1).tolist()
    intro_spans = []
    for identity in range(3):
        span = pointer_span(
            outputs["intro_start_logits"][row, :, identity],
            outputs["intro_end_logits"][row, :, identity],
            length,
        )
        if span is None:
            return {"valid": False, "failure_reason": "intro_pointer", "event_count": 0}
        intro_spans.append(span)
    if len(set(intro_spans)) != 3:
        return {"valid": False, "failure_reason": "intro_overlap", "event_count": 0}
    intro_ids = [tuple(ids[position] for position in span) for span in intro_spans]
    if len(set(intro_ids)) != 3:
        return {"valid": False, "failure_reason": "intro_identity_collision", "event_count": 0}

    kind_role = ROLE_INDEX["event.kind"]
    kind_records = [
        {"token_ids": record["token_ids"], "value": record["kind"]}
        for record in lexicon["patterns"]
    ]
    kind_windows = [
        window for window in pattern_windows(ids, kind_records)
        if any(labels[position] == kind_role for position in range(window[0], window[1]))
    ]
    if not kind_windows:
        return {"valid": False, "failure_reason": "no_kind_anchors", "event_count": 0}
    if any(kind_windows[index][0] < kind_windows[index - 1][1]
           for index in range(1, len(kind_windows))):
        return {"valid": False, "failure_reason": "kind_overlap", "event_count": len(kind_windows)}

    query_span = pointer_span(
        outputs["query_start_logits"][row], outputs["query_end_logits"][row], length,
    )
    if query_span is None:
        return {"valid": False, "failure_reason": "query_pointer", "event_count": len(kind_windows)}
    query = int(mean_positions(outputs["query_logits"], row, query_span).argmax())
    program = []
    for start, end, kind in kind_windows:
        scores = parser.event_pointer_scores(outputs, row, tuple(range(start, end)))
        entity_span = pointer_span(scores["entity_start"], scores["entity_end"], length)
        literal_span = pointer_span(scores["literal_start"], scores["literal_end"], length)
        if entity_span is None or literal_span is None:
            return {"valid": False, "failure_reason": "event_pointer", "event_count": len(kind_windows)}
        entity_ids = tuple(ids[position] for position in entity_span)
        matches = [identity for identity, values in enumerate(intro_ids) if values == entity_ids]
        if len(matches) != 1:
            return {"valid": False, "failure_reason": "event_identity", "event_count": len(kind_windows)}
        amount = int(mean_positions(outputs["amount_logits"], row, literal_span).argmax()) + 1
        program.append((str(kind), matches[0], amount))
    final_state, answer = execute_program(program, query)
    return {
        "valid": True,
        "failure_reason": "none",
        "event_count": len(program),
        "program": tuple(program),
        "query": query,
        "final_state": final_state,
        "answer_identity": answer,
        "intro_ids": tuple(intro_ids),
    }
