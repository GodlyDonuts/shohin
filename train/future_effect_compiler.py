"""Probe-conditioned text compiler for active future-effect reasoning.

Unlike a direct opcode or operator-coordinate head, this module answers one
selected counterfactual at a time. The same small head receives cached event
text features, dynamic entity-slot evidence, a hypothetical input state, and a
future query. Active inference can therefore spend additional calls only on
probes that distinguish its remaining operator hypotheses.

Structured operations and numeric values are not accepted by inference. They
may be used outside this module to construct supervised scalar effects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from referential_slot_microcode import ReferentialSlotMicrocodeCompiler


def probe_descriptor(states, queries):
    """Build fixed state/query features without operator-label information."""
    if states.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError("states and queries must end in three coordinates")
    states, queries = torch.broadcast_tensors(states, queries)
    outer = torch.einsum("...i,...j->...ij", queries, states).reshape(*states.shape[:-1], 9)
    return torch.cat((states, queries, outer), dim=-1)


class ProbeConditionedEffectCompiler(ReferentialSlotMicrocodeCompiler):
    """R4 dynamic binding plus a shared scalar counterfactual-effect head."""

    def __init__(self, model, layer=19, hidden=256, effect_hidden=128):
        super().__init__(model, layer=layer, hidden=hidden, role_mode="pointer")
        effect_hidden = int(effect_hidden)
        if effect_hidden <= 0:
            raise ValueError("effect_hidden must be positive")
        self.effect_hidden = effect_hidden
        # The event representation contains two independently pooled text
        # contexts plus relational evidence from the dynamic slots.
        event_width = 2 * hidden + 4
        self.effect_text = nn.Sequential(
            nn.Linear(event_width, effect_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(effect_hidden, effect_hidden, bias=False),
        )
        self.effect_probe = nn.Sequential(
            nn.Linear(15, effect_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(effect_hidden, effect_hidden, bias=False),
        )
        self.effect_interaction = nn.Sequential(
            nn.Linear(4 * effect_hidden, effect_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(effect_hidden, 1),
        )

    @staticmethod
    def operation_effect_context(operation_output):
        """Extract text/slot evidence from one inference-only event result."""
        required = {
            "kind_context", "target_weights", "role_logits", "slot_presence_scores",
        }
        missing = required.difference(operation_output)
        if missing:
            raise ValueError("operation output is missing {}".format(sorted(missing)))
        role = torch.softmax(operation_output["role_logits"].float(), dim=-1)
        presence = operation_output["slot_presence_scores"].float()
        # target_context is retained by the R4 line compiler specifically for
        # semantic mention selection; add it to the public result if absent in
        # older adapters rather than reconstructing it from labels.
        if "target_context" not in operation_output:
            raise ValueError("operation output lacks target_context")
        return torch.cat((
            operation_output["kind_context"].float(),
            operation_output["target_context"].float(),
            role,
            presence,
        ), dim=-1)

    def predict_effect(self, operation_output, states, queries):
        """Predict scalar ``query @ event(state)`` for selected probes."""
        context = self.operation_effect_context(operation_output)
        descriptor = probe_descriptor(states.float(), queries.float())
        text = self.effect_text(context)
        text = text.expand(*descriptor.shape[:-1], text.shape[-1])
        probe = self.effect_probe(descriptor)
        joint = torch.cat((text, probe, text * probe, text - probe), dim=-1)
        return self.effect_interaction(joint).squeeze(-1).float()

    def predict_effect_bank(self, operation_output, states, queries):
        """Score a Cartesian probe bank for audits, not active inference."""
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("states must have shape [state_probes, 3]")
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape [query_probes, 3]")
        expanded_states = states.unsqueeze(0).expand(queries.shape[0], -1, -1)
        expanded_queries = queries.unsqueeze(1).expand(-1, states.shape[0], -1)
        return self.predict_effect(operation_output, expanded_states, expanded_queries)

