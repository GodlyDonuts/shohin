"""Hard-bit recurrent learner for the delayed-witness edge-parity ring.

The public packet boundary in this module is deliberately small: a packet is
an immutable tuple of exactly 15 binary values.  Reader methods accept only
that packet, a public scale mask, and continuation events; source histories
and execution caches are not part of the reader interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PACKET_BITS = 15
EVENT_BITS = 2
HIDDEN_WIDTH = 64
TRANSITION_INPUT_BITS = PACKET_BITS * 2 + EVENT_BITS
READOUT_INPUT_BITS = PACKET_BITS * 2
EXPECTED_PARAMETER_COUNT = 5_136

ROTATE = 0
FLIP = 1
EVENT_NAMES = ("R", "F")

SerializedPacket = tuple[int, ...]


def make_scale_mask(n: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the public mask with the first ``n - 1`` packet bits active."""
    n = int(n)
    if n < 4 or n > PACKET_BITS + 1 or n % 2:
        raise ValueError("learner scales must be even and satisfy 4 <= n <= 16")
    mask = torch.zeros(PACKET_BITS, dtype=torch.float32, device=device)
    mask[: n - 1] = 1.0
    return mask


def _validate_scale_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim < 1 or mask.shape[-1] != PACKET_BITS:
        raise ValueError("scale_mask must end in exactly 15 values")
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError("scale_mask must be finite")
    binary = (mask == 0) | (mask == 1)
    if not bool(binary.all()):
        raise ValueError("scale_mask must be binary")
    flat = mask.reshape(-1, PACKET_BITS)
    active = flat.sum(dim=-1).to(torch.long)
    allowed = (active >= 3) & (active <= PACKET_BITS) & ((active + 1) % 2 == 0)
    if not bool(allowed.all()):
        raise ValueError("each scale mask must encode an even n in [4, 16]")
    positions = torch.arange(PACKET_BITS, device=mask.device).unsqueeze(0)
    expected = positions < active.unsqueeze(1)
    if not bool((flat.bool() == expected).all()):
        raise ValueError("active scale-mask values must be a contiguous prefix")
    return mask.to(dtype=torch.float32)


def _broadcast_mask(mask: torch.Tensor, packet: torch.Tensor) -> torch.Tensor:
    mask = _validate_scale_mask(mask).to(device=packet.device, dtype=packet.dtype)
    try:
        return torch.broadcast_to(mask, packet.shape)
    except RuntimeError as error:
        raise ValueError("scale_mask cannot broadcast to the packet shape") from error


def straight_through_hard_bits(logits: torch.Tensor, scale_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold sigmoid probabilities in forward mode and retain sigmoid gradients."""
    if logits.shape[-1:] != (PACKET_BITS,):
        raise ValueError("transition logits must end in 15 values")
    mask = _broadcast_mask(scale_mask, logits)
    probabilities = torch.sigmoid(logits)
    hard = (probabilities >= 0.5).to(probabilities.dtype)
    packet = probabilities + (hard - probabilities).detach()
    return packet * mask, probabilities


def _event_features(event: torch.Tensor | int, leading_shape: torch.Size, *, device: torch.device) -> torch.Tensor:
    event = torch.as_tensor(event, device=device)
    if event.shape == leading_shape:
        if event.is_floating_point() and not bool((event == event.round()).all()):
            raise ValueError("event codes must be integers")
        event = event.to(torch.long)
        if not bool(((event == ROTATE) | (event == FLIP)).all()):
            raise ValueError("event codes must be ROTATE=0 or FLIP=1")
        return F.one_hot(event, num_classes=EVENT_BITS).to(torch.float32)
    if event.shape != leading_shape + (EVENT_BITS,):
        raise ValueError("event must be one code or one two-bit code per packet")
    if not bool(((event == 0) | (event == 1)).all()) or not bool((event.sum(dim=-1) == 1).all()):
        raise ValueError("two-bit event codes must be one-hot")
    return event.to(torch.float32)


@dataclass(frozen=True)
class HardBitRollout:
    packet: torch.Tensor
    probabilities: torch.Tensor
    transition_mask: torch.Tensor


class HardBitDWEPRLearner(nn.Module):
    """The frozen 5,136-scalar tied-transition DWEPR learner."""

    def __init__(self) -> None:
        super().__init__()
        self.transition_mlp = nn.Sequential(
            nn.Linear(TRANSITION_INPUT_BITS, HIDDEN_WIDTH, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, PACKET_BITS, dtype=torch.float32),
        )
        self.readout_mlp = nn.Sequential(
            nn.Linear(READOUT_INPUT_BITS, HIDDEN_WIDTH, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, 1, dtype=torch.float32),
        )
        self.assert_contract()

    def assert_contract(self) -> None:
        parameters = list(self.parameters())
        count = sum(parameter.numel() for parameter in parameters)
        if count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(f"DWEPR learner has {count} parameters, expected 5,136")
        if any(parameter.dtype != torch.float32 for parameter in parameters):
            raise RuntimeError("every DWEPR learner parameter must be fp32")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_packet(self, scale_mask: torch.Tensor) -> torch.Tensor:
        mask = _validate_scale_mask(scale_mask)
        return torch.zeros_like(mask, dtype=torch.float32)

    def transition_logits(
        self,
        packet: torch.Tensor,
        event: torch.Tensor | int,
        scale_mask: torch.Tensor,
    ) -> torch.Tensor:
        if packet.shape[-1:] != (PACKET_BITS,):
            raise ValueError("packet must end in exactly 15 values")
        packet = packet.to(device=self.transition_mlp[0].weight.device, dtype=torch.float32)
        mask = _broadcast_mask(scale_mask, packet)
        event_bits = _event_features(event, packet.shape[:-1], device=packet.device)
        return self.transition_mlp(torch.cat((packet * mask, mask, event_bits), dim=-1))

    def transition(
        self,
        packet: torch.Tensor,
        event: torch.Tensor | int,
        scale_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.transition_logits(packet, event, scale_mask)
        return straight_through_hard_bits(logits, scale_mask)

    def rollout(
        self,
        packet: torch.Tensor,
        events: torch.Tensor,
        scale_mask: torch.Tensor,
        transition_mask: torch.Tensor | None = None,
    ) -> HardBitRollout:
        """Apply a padded event sequence while preserving hard packets at every step."""
        if packet.shape[-1:] != (PACKET_BITS,):
            raise ValueError("packet must end in exactly 15 values")
        packet = packet.to(device=self.transition_mlp[0].weight.device, dtype=torch.float32)
        leading = packet.shape[:-1]
        events = torch.as_tensor(events, device=packet.device)
        if events.ndim == len(leading) + 1 and events.shape[:-1] == leading:
            steps = events.shape[-1]
            encoded_events = events
            one_hot = False
        elif events.ndim == len(leading) + 2 and events.shape[:-2] == leading and events.shape[-1] == EVENT_BITS:
            steps = events.shape[-2]
            encoded_events = events
            one_hot = True
        else:
            raise ValueError("events must have shape [...,steps] or [...,steps,2]")
        if transition_mask is None:
            active = torch.ones(leading + (steps,), dtype=torch.bool, device=packet.device)
        else:
            active = torch.as_tensor(transition_mask, device=packet.device)
            if active.shape != leading + (steps,):
                raise ValueError("transition_mask must match the event sequence")
            if not bool(((active == 0) | (active == 1)).all()):
                raise ValueError("transition_mask must be binary")
            active = active.bool()

        probability_trace = []
        for step in range(steps):
            step_event = encoded_events[..., step, :] if one_hot else encoded_events[..., step]
            candidate, probabilities = self.transition(packet, step_event, scale_mask)
            step_active = active[..., step, None]
            packet = torch.where(step_active, candidate, packet)
            probability_trace.append(probabilities)
        if probability_trace:
            probabilities = torch.stack(probability_trace, dim=-2)
        else:
            probabilities = packet.new_empty(leading + (0, PACKET_BITS))
        packet = packet * _broadcast_mask(scale_mask, packet)
        return HardBitRollout(packet=packet, probabilities=probabilities, transition_mask=active)

    def encode(
        self,
        source_events: torch.Tensor,
        scale_mask: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> HardBitRollout:
        return self.rollout(self.initial_packet(scale_mask), source_events, scale_mask, source_mask)

    def read_logits(self, packet: torch.Tensor, scale_mask: torch.Tensor) -> torch.Tensor:
        if packet.shape[-1:] != (PACKET_BITS,):
            raise ValueError("packet must end in exactly 15 values")
        packet = packet.to(device=self.readout_mlp[0].weight.device, dtype=torch.float32)
        mask = _broadcast_mask(scale_mask, packet)
        return self.readout_mlp(torch.cat((packet * mask, mask), dim=-1)).squeeze(-1)


WGRQStateMachine = HardBitDWEPRLearner


def serialize_packet(packet: torch.Tensor, scale_mask: torch.Tensor) -> SerializedPacket:
    """Convert one hard packet to an immutable sequence of exactly 15 bits."""
    if packet.shape != (PACKET_BITS,):
        raise ValueError("only one exactly 15-bit packet may be serialized")
    mask = _validate_scale_mask(scale_mask)
    if mask.shape != (PACKET_BITS,):
        raise ValueError("serialization requires one scale mask")
    detached = packet.detach().to(device="cpu", dtype=torch.float32)
    cpu_mask = mask.detach().to(device="cpu", dtype=torch.float32)
    if not bool(((detached == 0) | (detached == 1)).all()):
        raise ValueError("serialized packet values must be hard bits")
    if not bool((detached[cpu_mask == 0] == 0).all()):
        raise ValueError("masked packet bits must be zero")
    return tuple(int(value) for value in detached.tolist())


def deserialize_packet(
    packet: Sequence[int],
    scale_mask: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Validate and materialize one immutable packet without retaining it."""
    frozen = tuple(packet)
    if len(frozen) != PACKET_BITS or any(value not in (0, 1) for value in frozen):
        raise ValueError("packet must contain exactly 15 binary values")
    mask = _validate_scale_mask(scale_mask)
    if mask.shape != (PACKET_BITS,):
        raise ValueError("deserialization requires one scale mask")
    tensor = torch.tensor(frozen, dtype=torch.float32, device=device)
    device_mask = mask.to(device=tensor.device)
    if not bool((tensor[device_mask == 0] == 0).all()):
        raise ValueError("masked packet bits must be zero")
    return tensor


class WGRQWriter:
    """Stateless source-history writer for one public scale."""

    def __init__(self, model: HardBitDWEPRLearner, scale_mask: torch.Tensor) -> None:
        self.model = model
        self.scale_mask = _validate_scale_mask(scale_mask).detach().clone()

    @torch.no_grad()
    def write(self, source_events: torch.Tensor, source_mask: torch.Tensor | None = None) -> SerializedPacket:
        rollout = self.model.encode(source_events, self.scale_mask, source_mask)
        return serialize_packet(rollout.packet, self.scale_mask)


class WGRQReader:
    """Fresh source-free reader; each call branches from a cloned packet."""

    def __init__(self, model: HardBitDWEPRLearner, scale_mask: torch.Tensor) -> None:
        self.model = model
        self.scale_mask = _validate_scale_mask(scale_mask).detach().clone()

    @torch.no_grad()
    def read_logits(
        self,
        packet: Sequence[int],
        continuation_events: torch.Tensor,
        continuation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frozen = tuple(packet)
        state = deserialize_packet(
            frozen,
            self.scale_mask,
            device=self.model.readout_mlp[0].weight.device,
        ).clone()
        branch = self.model.rollout(state, continuation_events, self.scale_mask, continuation_mask)
        result = self.model.read_logits(branch.packet, self.scale_mask).detach()
        if tuple(packet) != frozen:
            raise RuntimeError("reader mutated the serialized packet")
        return result

    @torch.no_grad()
    def read_probability(
        self,
        packet: Sequence[int],
        continuation_events: torch.Tensor,
        continuation_mask: torch.Tensor | None = None,
    ) -> float:
        logit = self.read_logits(packet, continuation_events, continuation_mask)
        if logit.numel() != 1:
            raise ValueError("reader probability requires one continuation")
        return float(torch.sigmoid(logit).item())
