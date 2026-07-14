"""Proof-carrying context folding for the active-distinction experiment.

An event may leave source context only after counterfactual observations select
one lawful operator and an unused probe independently agrees with that claim.
Accepted operators compose into one fixed 3x3 state. This is an exact CPU
contract for a possible context mechanism, not evidence that a neural compiler
can produce valid certificates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from future_distinction_cell import (
    compatible_hypotheses,
    identify_with_oracle,
)
from future_effect_algebra import query_operator


@dataclass(frozen=True)
class ProbeObservation:
    probe: int
    effect: float


@dataclass(frozen=True)
class EventCertificate:
    accepted: bool
    selected: int | None
    remaining: tuple[int, ...]
    observations: tuple[ProbeObservation, ...]
    validation: ProbeObservation | None
    reason: str


@dataclass(frozen=True)
class FoldedContext:
    operator: torch.Tensor
    folded_events: int

    @property
    def scalar_payload(self):
        return int(self.operator.numel())


def empty_context(*, dtype=torch.float64):
    return FoldedContext(torch.eye(3, dtype=dtype), 0)


def verify_event_certificate(codes, observations, validation, *, atol=1e-9):
    """Admit a source-droppable event only when evidence is unique and checked."""
    observations = tuple(observations)
    candidates = tuple(range(codes.shape[0]))
    seen = set()
    for observation in observations:
        probe = int(observation.probe)
        if probe in seen or not 0 <= probe < codes.shape[1]:
            return EventCertificate(
                False, None, candidates, observations, validation, "invalid_selected_probe",
            )
        seen.add(probe)
        candidates = compatible_hypotheses(
            codes, candidates, probe, observation.effect, atol=atol,
        )
        if not candidates:
            return EventCertificate(
                False, None, (), observations, validation, "inconsistent_selected_effect",
            )
    if len(candidates) != 1:
        return EventCertificate(
            False, None, candidates, observations, validation, "ambiguous_operator",
        )
    selected = candidates[0]
    if validation is None or validation.probe in seen:
        return EventCertificate(
            False, selected, candidates, observations, validation, "missing_independent_validation",
        )
    expected = float(codes[selected, int(validation.probe)].item())
    if abs(expected - float(validation.effect)) > float(atol):
        return EventCertificate(
            False, selected, candidates, observations, validation, "validation_mismatch",
        )
    return EventCertificate(
        True, selected, candidates, observations, validation, "accepted",
    )


def oracle_event_certificate(codes, target, *, max_probes=3):
    """Construct an exact certificate for mechanics tests only."""
    trace = identify_with_oracle(codes, target, max_probes=max_probes, policy="active")
    observations = tuple(
        ProbeObservation(item["probe"], item["effect"])
        for item in trace["trace"]
    )
    used = {item.probe for item in observations}
    validation_probe = next(probe for probe in range(codes.shape[1]) if probe not in used)
    validation = ProbeObservation(
        validation_probe, float(codes[int(target), validation_probe].item()),
    )
    return verify_event_certificate(codes, observations, validation)


def fold_event(context, certificate, hypotheses):
    """Drop one certified source event into the fixed-size chronological state."""
    if not certificate.accepted or certificate.selected is None:
        raise ValueError("cannot fold an uncertified event")
    event = hypotheses[certificate.selected].operator.to(
        dtype=context.operator.dtype, device=context.operator.device,
    )
    return FoldedContext(event @ context.operator, context.folded_events + 1)


def merge_contexts(earlier, later):
    """Merge independently folded chronological chunks without their source."""
    if earlier.operator.dtype != later.operator.dtype:
        raise ValueError("folded contexts must share dtype")
    return FoldedContext(
        later.operator @ earlier.operator,
        earlier.folded_events + later.folded_events,
    )


def read_context(context, initial_values, query):
    if len(initial_values) != 2:
        raise ValueError("exactly two initial values are required")
    state = torch.tensor(
        [initial_values[0], initial_values[1], 1], dtype=context.operator.dtype,
    )
    readout = query_operator(query, dtype=context.operator.dtype)
    return int((readout @ context.operator @ state).item())
