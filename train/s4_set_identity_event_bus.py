"""Set-valued lexical identity transport over the frozen S4 v1 parser."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_delimiting_event_tape import (
    ROLE_INDEX,
    SelfDelimitingEventTapeParser,
    execute_program,
    pattern_windows,
)


CARRIER_PREFIXES = (
    "event_entity_query.",
    "event_entity_key.",
    "event_literal_query.",
    "event_literal_key.",
)
IDENTITY_LOGIT_SCALE = 20.0


def masked_distribution(logits, valid):
    """Return one normalized soft token set over valid source positions."""
    masked = logits.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    return F.softmax(masked, dim=-1)


def uniform_target(positions, length, device):
    target = torch.zeros(length, dtype=torch.float32, device=device)
    index = torch.tensor(tuple(positions), dtype=torch.long, device=device)
    target.index_fill_(0, index, 1.0 / max(1, index.numel()))
    return target


def distribution_loss(logits, positions, valid):
    log_probs = F.log_softmax(
        logits.float().masked_fill(~valid, torch.finfo(torch.float32).min), dim=-1,
    )
    return -(uniform_target(positions, logits.numel(), logits.device) * log_probs).sum()


def vocabulary_carrier(ids, weights, vocab_size):
    """Aggregate a soft token set into an exact vocabulary-aligned histogram."""
    carrier = torch.zeros(vocab_size, dtype=weights.dtype, device=weights.device)
    return carrier.scatter_add(0, ids.long(), weights)


def carrier_logits(candidate, roster):
    candidate = F.normalize(candidate.float(), dim=-1)
    roster = F.normalize(roster.float(), dim=-1)
    return IDENTITY_LOGIT_SCALE * torch.mv(roster, candidate)


class SetIdentityEventBus(SelfDelimitingEventTapeParser):
    def __init__(self, model, layer=19, width=384, heads=8, encoder_layers=5, ff=1408):
        super().__init__(model, layer, width, heads, encoder_layers, ff)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(False)
        self.event_entity_query = nn.Linear(width, width, bias=False)
        self.event_entity_key = nn.Linear(width, width, bias=False)
        self.event_literal_query = nn.Linear(width, width, bias=False)
        self.event_literal_key = nn.Linear(width, width, bias=False)

    def initialize_v1(self, state):
        own = self.state_dict()
        expected = {
            name for name in own
            if not name.startswith("model.") and not name.startswith(CARRIER_PREFIXES)
        }
        supplied = {name for name in state if not name.startswith("model.")}
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError("v1 adapter mismatch missing={} unexpected={}".format(
                missing, unexpected,
            ))
        with torch.no_grad():
            for name in expected:
                if own[name].shape != state[name].shape:
                    raise ValueError("v1 adapter shape mismatch for {}".format(name))
                own[name].copy_(state[name])
        return tuple(sorted(expected))

    def carrier_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(CARRIER_PREFIXES):
                yield parameter

    def carrier_num_params(self):
        return sum(parameter.numel() for parameter in self.carrier_parameters())

    def forward(self, ids, valid):
        outputs = super().forward(ids, valid)
        memory = outputs["memory"]
        outputs.update({
            "event_entity_keys": self.event_entity_key(memory),
            "event_literal_keys": self.event_literal_key(memory),
        })
        return outputs

    def event_membership_scores(self, outputs, row, anchor_positions):
        memory = outputs["memory"]
        index = torch.tensor(anchor_positions, dtype=torch.long, device=memory.device)
        anchor = memory[row].index_select(0, index).mean(0)

        def score(query_layer, key_name, role):
            query = query_layer(anchor)
            contextual = torch.mv(outputs[key_name][row], query) / math.sqrt(self.width)
            role_prior = outputs["role_logits"][row, :, ROLE_INDEX[role]]
            return contextual.float() + role_prior.float()

        return {
            "entity": score(
                self.event_entity_query, "event_entity_keys", "event.entity",
            ),
            "literal": score(
                self.event_literal_query, "event_literal_keys", "event.literal",
            ),
        }


def roster_distributions(outputs, row, valid):
    length = valid.numel()
    return tuple(
        masked_distribution(
            outputs["role_logits"][
                row, :length, ROLE_INDEX["intro.entity{}".format(identity)]
            ],
            valid,
        )
        for identity in range(3)
    )


def roster_carriers(ids, distributions, vocab_size):
    return torch.stack([
        vocabulary_carrier(ids, weights, vocab_size) for weights in distributions
    ])


def set_identity_loss(parser, outputs, examples, ids, valid):
    membership_losses = []
    identity_losses = []
    amount_losses = []
    for row, example in enumerate(examples):
        length = len(example.ids)
        row_ids = ids[row, :length]
        row_valid = valid[row, :length]
        roster = roster_carriers(
            row_ids,
            roster_distributions(outputs, row, row_valid),
            parser.model.cfg.vocab_size,
        )
        for event_index, event in enumerate(example.event_positions):
            scores = parser.event_membership_scores(outputs, row, event["kind"])
            entity_scores = scores["entity"][:length]
            literal_scores = scores["literal"][:length]
            membership_losses.extend((
                distribution_loss(entity_scores, event["entity"], row_valid),
                distribution_loss(literal_scores, event["literal"], row_valid),
            ))
            entity_weights = masked_distribution(entity_scores, row_valid)
            entity = vocabulary_carrier(
                row_ids, entity_weights, parser.model.cfg.vocab_size,
            )
            identity_losses.append(F.cross_entropy(
                carrier_logits(entity, roster).unsqueeze(0),
                torch.tensor([example.program[event_index][1]], device=ids.device),
            ))
            literal_weights = masked_distribution(literal_scores, row_valid)
            amount = torch.sum(
                literal_weights.unsqueeze(-1)
                * outputs["amount_logits"][row, :length].float(),
                dim=0,
            )
            amount_losses.append(F.cross_entropy(
                amount.unsqueeze(0),
                torch.tensor([example.amount_targets[event_index]], device=ids.device),
            ))
    components = {
        "membership": torch.stack(membership_losses).mean(),
        "identity": torch.stack(identity_losses).mean(),
        "amount": torch.stack(amount_losses).mean(),
    }
    return sum(components.values()) / len(components), components


def roster_recovery_exact(example, outputs, row, valid, vocab_size):
    length = len(example.ids)
    ids = torch.tensor(example.ids, dtype=torch.long, device=valid.device)
    predicted = roster_carriers(
        ids,
        roster_distributions(outputs, row, valid[:length]),
        vocab_size,
    )
    gold = torch.stack([
        vocabulary_carrier(
            ids,
            uniform_target(positions, length, ids.device),
            vocab_size,
        )
        for positions in example.intro_positions
    ])
    matches = [int(carrier_logits(predicted[index], gold).argmax()) for index in range(3)]
    return tuple(matches) == (0, 1, 2)


def decode_set_identity_example(
    parser, example, outputs, row, valid, lexicon, roster_permutation=None,
):
    length = len(example.ids)
    row_valid = valid[:length]
    ids = torch.tensor(example.ids, dtype=torch.long, device=row_valid.device)
    labels = outputs["role_logits"][row, :length].argmax(-1).tolist()
    kind_records = [
        {"token_ids": record["token_ids"], "value": record["kind"]}
        for record in lexicon["patterns"]
    ]
    kind_role = ROLE_INDEX["event.kind"]
    kind_windows = [
        window for window in pattern_windows(example.ids, kind_records)
        if any(labels[position] == kind_role for position in range(window[0], window[1]))
    ]
    if not kind_windows:
        return {"valid": False, "failure_reason": "no_kind_anchors", "event_count": 0}
    if any(kind_windows[index][0] < kind_windows[index - 1][1]
           for index in range(1, len(kind_windows))):
        return {
            "valid": False,
            "failure_reason": "kind_overlap",
            "event_count": len(kind_windows),
        }

    roster = roster_carriers(
        ids,
        roster_distributions(outputs, row, row_valid),
        parser.model.cfg.vocab_size,
    )
    if roster_permutation is not None:
        index = torch.tensor(roster_permutation, dtype=torch.long, device=roster.device)
        roster = roster.index_select(0, index)
    program = []
    for start, end, kind in kind_windows:
        scores = parser.event_membership_scores(outputs, row, tuple(range(start, end)))
        entity_weights = masked_distribution(scores["entity"][:length], row_valid)
        entity = vocabulary_carrier(ids, entity_weights, parser.model.cfg.vocab_size)
        identity = int(carrier_logits(entity, roster).argmax())
        literal_weights = masked_distribution(scores["literal"][:length], row_valid)
        amount_logits = torch.sum(
            literal_weights.unsqueeze(-1)
            * outputs["amount_logits"][row, :length].float(),
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
    }
