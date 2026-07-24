"""Differentiable and source-deleted runtimes for an episodic functor machine.

The soft machine is the attached optimization object emitted by a future
source compiler.  The hard machine is the only semantic object accepted by
the detached evaluator.  Neither runtime reads source tokens, residuals, KV
state, compiler pointers, targets, or verifier feedback.

Opaque source/query key parsing belongs to a separate compiler/parser module.
The compact semantic payload deliberately omits key bytes.  ``HardFunctorKeys``
retains exact copied key bytes, and ``deployed_wire`` combines both objects
into the existing 1,536-byte EFC C/Rust wire with zero padding and SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import struct
from typing import Sequence

import torch
import torch.nn.functional as F


MAX_STATES = 16
MAX_ACTIONS = 8
MAX_OBSERVERS = 8
MAX_ANSWERS = 5
MAX_STEPS = 32
DEPLOYED_MACHINE_BYTES = 1_536
DEPLOYED_HASH_OFFSET = 1_504
DEPLOYED_LEGACY_INITIAL_OFFSET = 56
SEMANTIC_BYTES_PER_ROW = (
    MAX_STATES
    + MAX_ACTIONS
    + MAX_OBSERVERS
    + MAX_ACTIONS * MAX_STATES
    + MAX_OBSERVERS * MAX_STATES
)


class FunctorMachineError(ValueError):
    """A machine, query, execution, or intervention contract failed."""


@dataclass(frozen=True, slots=True)
class LearnedFunctorWireSpec:
    """Learned-board dimensions without the old identity-observer law."""

    state_count: int = 8
    action_count: int = 3
    observer_count: int = 2
    answer_count: int = 4
    renderer_count: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.state_count <= MAX_STATES:
            raise FunctorMachineError("learned wire state count differs")
        if not 1 <= self.action_count <= MAX_ACTIONS:
            raise FunctorMachineError("learned wire action count differs")
        if not 1 <= self.observer_count <= MAX_OBSERVERS:
            raise FunctorMachineError("learned wire observer count differs")
        if not 1 <= self.answer_count <= MAX_ANSWERS:
            raise FunctorMachineError("learned wire answer count differs")
        if self.renderer_count != 1:
            raise FunctorMachineError("learned wire renderer count differs")


def _finite_float(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise FunctorMachineError(f"{name} must be finite floating point {shape}")


def _byte_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != torch.uint8
    ):
        raise FunctorMachineError(f"{name} must be uint8 {shape}")


def _straight_through_active(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(-1)[..., 1]
    hard = logits.argmax(-1).eq(1).to(probabilities.dtype)
    if not bool(hard.bool().any(-1).all()):
        raise FunctorMachineError("attached machine has an empty hard support")
    return hard + probabilities - probabilities.detach()


def _supported_softmax(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != support.shape:
        raise FunctorMachineError("supported softmax geometry differs")
    weights = logits.float().softmax(-1) * support
    denominator = weights.sum(-1, keepdim=True)
    if not bool(denominator.detach().gt(0).all()):
        raise FunctorMachineError("supported softmax has an empty hard support")
    return weights / denominator


def _straight_through_distribution(
    logits: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = (
        logits.float().softmax(-1)
        if support is None
        else _supported_softmax(logits, support)
    )
    indices = probabilities.argmax(-1)
    hard = F.one_hot(indices, probabilities.shape[-1]).to(
        probabilities.dtype
    )
    return hard + probabilities - probabilities.detach()


@dataclass(frozen=True, slots=True)
class SoftFunctorMachine:
    """Attached anonymous categorical machine logits."""

    state_active: torch.Tensor
    action_active: torch.Tensor
    observer_active: torch.Tensor
    action_next: torch.Tensor
    observer_answer: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state_active, torch.Tensor) or self.state_active.ndim != 3:
            raise FunctorMachineError("state_active must be rank three")
        batch = int(self.state_active.shape[0])
        _finite_float(
            "state_active",
            self.state_active,
            (batch, MAX_STATES, 2),
        )
        _finite_float(
            "action_active",
            self.action_active,
            (batch, MAX_ACTIONS, 2),
        )
        _finite_float(
            "observer_active",
            self.observer_active,
            (batch, MAX_OBSERVERS, 2),
        )
        _finite_float(
            "action_next",
            self.action_next,
            (batch, MAX_ACTIONS, MAX_STATES, MAX_STATES),
        )
        _finite_float(
            "observer_answer",
            self.observer_answer,
            (batch, MAX_OBSERVERS, MAX_STATES, MAX_ANSWERS),
        )
        devices = {getattr(self, field.name).device for field in fields(self)}
        if len(devices) != 1:
            raise FunctorMachineError("soft machine tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.state_active.shape[0])

    def detached_clone(self) -> "SoftFunctorMachine":
        return SoftFunctorMachine(
            *(getattr(self, field.name).detach().clone() for field in fields(self))
        )

    @torch.no_grad()
    def harden(self) -> "HardFunctorMachine":
        state_active = self.state_active.argmax(-1).bool()
        action_active = self.action_active.argmax(-1).bool()
        observer_active = self.observer_active.argmax(-1).bool()
        if not bool(state_active.any(-1).all()):
            raise FunctorMachineError("hard machine has no active state")
        if not bool(action_active.any(-1).all()):
            raise FunctorMachineError("hard machine has no active action")
        if not bool(observer_active.any(-1).all()):
            raise FunctorMachineError("hard machine has no active observer")
        _prefix_count(state_active[0], "state")
        _prefix_count(action_active[0], "action")
        _prefix_count(observer_active[0], "observer")
        for row in range(1, self.batch_size):
            _prefix_count(state_active[row], "state")
            _prefix_count(action_active[row], "action")
            _prefix_count(observer_active[row], "observer")
        negative = torch.finfo(self.action_next.dtype).min
        destination_mask = state_active[:, None, None, :]
        action_next = self.action_next.masked_fill(
            ~destination_mask,
            negative,
        ).argmax(-1)
        valid_transition_row = action_active[:, :, None] & state_active[:, None, :]
        action_next = torch.where(
            valid_transition_row,
            action_next,
            torch.zeros_like(action_next),
        )
        observer_answer = self.observer_answer.argmax(-1)
        valid_observer_row = observer_active[:, :, None] & state_active[:, None, :]
        observer_answer = torch.where(
            valid_observer_row,
            observer_answer,
            torch.zeros_like(observer_answer),
        )
        return HardFunctorMachine(
            state_active=state_active.to(torch.uint8),
            action_active=action_active.to(torch.uint8),
            observer_active=observer_active.to(torch.uint8),
            action_next=action_next.to(torch.uint8),
            observer_answer=observer_answer.to(torch.uint8),
        )


@dataclass(frozen=True, slots=True)
class HardFunctorMachine:
    """Detached categorical machine accepted by the source-free evaluator."""

    state_active: torch.Tensor
    action_active: torch.Tensor
    observer_active: torch.Tensor
    action_next: torch.Tensor
    observer_answer: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state_active, torch.Tensor) or self.state_active.ndim != 2:
            raise FunctorMachineError("hard state_active must be rank two")
        batch = int(self.state_active.shape[0])
        _byte_tensor("state_active", self.state_active, (batch, MAX_STATES))
        _byte_tensor("action_active", self.action_active, (batch, MAX_ACTIONS))
        _byte_tensor(
            "observer_active",
            self.observer_active,
            (batch, MAX_OBSERVERS),
        )
        _byte_tensor(
            "action_next",
            self.action_next,
            (batch, MAX_ACTIONS, MAX_STATES),
        )
        _byte_tensor(
            "observer_answer",
            self.observer_answer,
            (batch, MAX_OBSERVERS, MAX_STATES),
        )
        devices = {getattr(self, field.name).device for field in fields(self)}
        if len(devices) != 1:
            raise FunctorMachineError("hard machine tensors must share one device")
        for name in ("state_active", "action_active", "observer_active"):
            value = getattr(self, name)
            if bool(value.gt(1).any()) or not bool(value.bool().any(-1).all()):
                raise FunctorMachineError(f"{name} is not a nonempty binary mask")
        if self.action_next.numel() and int(self.action_next.max()) >= MAX_STATES:
            raise FunctorMachineError("transition destination leaves the state domain")
        active_rows = (
            self.action_active.bool()[:, :, None]
            & self.state_active.bool()[:, None, :]
        )
        active_destinations = self.state_active.bool().gather(
            1,
            self.action_next.long().flatten(1),
        ).reshape_as(self.action_next)
        if bool((active_rows & ~active_destinations).any()):
            raise FunctorMachineError("active transition reaches an inactive state")
        if bool((~active_rows & self.action_next.ne(0)).any()):
            raise FunctorMachineError("inactive transition padding is nonzero")
        if self.observer_answer.numel() and int(self.observer_answer.max()) >= MAX_ANSWERS:
            raise FunctorMachineError("observer answer leaves the answer alphabet")
        active_observer_rows = (
            self.observer_active.bool()[:, :, None]
            & self.state_active.bool()[:, None, :]
        )
        if bool((~active_observer_rows & self.observer_answer.ne(0)).any()):
            raise FunctorMachineError("inactive observer padding is nonzero")

    @property
    def batch_size(self) -> int:
        return int(self.state_active.shape[0])

    @property
    def bytes_per_row(self) -> int:
        return SEMANTIC_BYTES_PER_ROW

    def row_bytes(self, row: int) -> bytes:
        if row not in range(self.batch_size):
            raise FunctorMachineError("semantic payload row is out of range")
        parts = (
            self.state_active[row],
            self.action_active[row],
            self.observer_active[row],
            self.action_next[row].flatten(),
            self.observer_answer[row].flatten(),
        )
        payload = b"".join(part.detach().cpu().contiguous().numpy().tobytes() for part in parts)
        if len(payload) != SEMANTIC_BYTES_PER_ROW:
            raise AssertionError("semantic payload byte accounting differs")
        return payload

    @classmethod
    def from_row_bytes(
        cls,
        payload: bytes,
        *,
        device: torch.device | str = "cpu",
    ) -> "HardFunctorMachine":
        if len(payload) != SEMANTIC_BYTES_PER_ROW:
            raise FunctorMachineError("semantic payload has the wrong byte count")
        data = torch.tensor(tuple(payload), dtype=torch.uint8, device=device)
        offset = 0

        def take(count: int) -> torch.Tensor:
            nonlocal offset
            value = data[offset : offset + count]
            offset += count
            return value

        state_active = take(MAX_STATES)[None]
        action_active = take(MAX_ACTIONS)[None]
        observer_active = take(MAX_OBSERVERS)[None]
        action_next = take(MAX_ACTIONS * MAX_STATES).reshape(
            1,
            MAX_ACTIONS,
            MAX_STATES,
        )
        observer_answer = take(MAX_OBSERVERS * MAX_STATES).reshape(
            1,
            MAX_OBSERVERS,
            MAX_STATES,
        )
        if offset != len(payload):
            raise AssertionError("semantic payload parser did not consume all bytes")
        return cls(
            state_active=state_active,
            action_active=action_active,
            observer_active=observer_active,
            action_next=action_next,
            observer_answer=observer_answer,
        )

    def deployed_wire(
        self,
        keys: "HardFunctorKeys",
        row: int,
    ) -> bytes:
        """Serialize one prefix-canonical row to the exact deployed wire."""

        if row not in range(self.batch_size) or keys.batch_size != self.batch_size:
            raise FunctorMachineError("deployed machine row or key batch differs")
        state_count = _prefix_count(self.state_active[row], "state")
        action_count = _prefix_count(self.action_active[row], "action")
        observer_count = _prefix_count(self.observer_active[row], "observer")
        keys.validate_masks(self, row)

        machine = bytearray(DEPLOYED_MACHINE_BYTES)
        machine[:8] = b"EFCMACH\0"
        struct.pack_into("<I", machine, 8, 1)
        struct.pack_into("<I", machine, 12, 64)
        struct.pack_into("<I", machine, 16, DEPLOYED_MACHINE_BYTES)
        struct.pack_into("<H", machine, 24, state_count)
        struct.pack_into("<H", machine, 26, action_count)
        struct.pack_into("<H", machine, 28, observer_count)
        struct.pack_into("<Q", machine, 32, (1 << state_count) - 1)
        struct.pack_into("<Q", machine, 40, (1 << action_count) - 1)
        struct.pack_into("<Q", machine, 48, (1 << observer_count) - 1)
        machine[DEPLOYED_LEGACY_INITIAL_OFFSET] = 0
        machine[64:192] = keys.state_keys[row].detach().cpu().numpy().tobytes()
        machine[192:256] = keys.action_keys[row].detach().cpu().numpy().tobytes()
        machine[256:320] = keys.observer_keys[row].detach().cpu().numpy().tobytes()
        for action in range(MAX_ACTIONS):
            for state in range(MAX_STATES):
                machine[320 + action * MAX_STATES + state] = int(
                    self.action_next[row, action, state]
                )
        for observer in range(MAX_OBSERVERS):
            for state in range(MAX_STATES):
                struct.pack_into(
                    "<Q",
                    machine,
                    448 + (observer * MAX_STATES + state) * 8,
                    int(self.observer_answer[row, observer, state]),
                )
        machine[DEPLOYED_HASH_OFFSET:] = sha256(
            machine[:DEPLOYED_HASH_OFFSET]
        ).digest()
        return bytes(machine)

    def permute_action_transitions(
        self,
        permutation: Sequence[int],
    ) -> "HardFunctorMachine":
        """Move each old action table to ``permutation[old]``."""

        checked = _permutation(permutation, MAX_ACTIONS, "action")
        old_for_new = _inverse_permutation(checked)
        index = torch.tensor(old_for_new, device=self.action_next.device)
        return replace(
            self,
            action_active=self.action_active.index_select(1, index),
            action_next=self.action_next.index_select(1, index),
        )

    def permute_observer_maps(
        self,
        permutation: Sequence[int],
    ) -> "HardFunctorMachine":
        """Move each old observer map to ``permutation[old]``."""

        checked = _permutation(permutation, MAX_OBSERVERS, "observer")
        old_for_new = _inverse_permutation(checked)
        index = torch.tensor(old_for_new, device=self.observer_answer.device)
        return replace(
            self,
            observer_active=self.observer_active.index_select(1, index),
            observer_answer=self.observer_answer.index_select(1, index),
        )

    def transplant_transition_cell(
        self,
        *,
        row: int,
        action: int,
        state: int,
        destination: int,
    ) -> "HardFunctorMachine":
        if row not in range(self.batch_size):
            raise FunctorMachineError("transition transplant row is out of range")
        if action not in range(MAX_ACTIONS) or state not in range(MAX_STATES):
            raise FunctorMachineError("transition transplant coordinate is out of range")
        if destination not in range(MAX_STATES):
            raise FunctorMachineError("transition transplant destination is out of range")
        if not bool(self.action_active[row, action]) or not bool(
            self.state_active[row, state]
        ):
            raise FunctorMachineError("transition transplant source is inactive")
        if not bool(self.state_active[row, destination]):
            raise FunctorMachineError("transition transplant destination is inactive")
        action_next = self.action_next.clone()
        action_next[row, action, state] = destination
        return replace(self, action_next=action_next)


@dataclass(frozen=True, slots=True)
class HardFunctorKeys:
    """Exact little-endian opaque key bytes retained across source deletion."""

    state_keys: torch.Tensor
    action_keys: torch.Tensor
    observer_keys: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state_keys, torch.Tensor) or self.state_keys.ndim != 3:
            raise FunctorMachineError("state key bytes must be rank three")
        batch = int(self.state_keys.shape[0])
        _byte_tensor("state key bytes", self.state_keys, (batch, MAX_STATES, 8))
        _byte_tensor("action key bytes", self.action_keys, (batch, MAX_ACTIONS, 8))
        _byte_tensor(
            "observer key bytes",
            self.observer_keys,
            (batch, MAX_OBSERVERS, 8),
        )
        if len({self.state_keys.device, self.action_keys.device, self.observer_keys.device}) != 1:
            raise FunctorMachineError("opaque key tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.state_keys.shape[0])

    def validate_masks(self, machine: HardFunctorMachine, row: int) -> None:
        if self.batch_size != machine.batch_size or row not in range(self.batch_size):
            raise FunctorMachineError("opaque key and machine rows differ")
        for label, values, active in (
            ("state", self.state_keys[row], machine.state_active[row]),
            ("action", self.action_keys[row], machine.action_active[row]),
            ("observer", self.observer_keys[row], machine.observer_active[row]),
        ):
            rows = values.detach().cpu().contiguous().numpy().tobytes()
            keys = tuple(rows[offset : offset + 8] for offset in range(0, len(rows), 8))
            live = tuple(key for key, keep in zip(keys, active.tolist(), strict=True) if keep)
            dead = tuple(key for key, keep in zip(keys, active.tolist(), strict=True) if not keep)
            if any(key == b"\0" * 8 for key in live) or len(set(live)) != len(live):
                raise FunctorMachineError(f"active {label} keys are zero or duplicated")
            if any(key != b"\0" * 8 for key in dead):
                raise FunctorMachineError(f"inactive {label} key padding is nonzero")


@dataclass(frozen=True, slots=True)
class SoftFunctorQuery:
    """Attached query-parser logits over start, action path, STOP, and observer."""

    start_state: torch.Tensor
    action_path: torch.Tensor
    stop_position: torch.Tensor
    observer: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.action_path, torch.Tensor) or self.action_path.ndim != 3:
            raise FunctorMachineError("soft action_path must be rank three")
        batch, steps, actions = self.action_path.shape
        if not 1 <= steps <= MAX_STEPS or actions != MAX_ACTIONS:
            raise FunctorMachineError("soft action_path geometry differs")
        _finite_float(
            "soft start_state",
            self.start_state,
            (batch, MAX_STATES),
        )
        _finite_float(
            "soft action_path",
            self.action_path,
            (batch, steps, MAX_ACTIONS),
        )
        _finite_float(
            "soft stop_position",
            self.stop_position,
            (batch, steps + 1),
        )
        _finite_float(
            "soft observer",
            self.observer,
            (batch, MAX_OBSERVERS),
        )
        if len(
            {
                self.start_state.device,
                self.action_path.device,
                self.stop_position.device,
                self.observer.device,
            }
        ) != 1:
            raise FunctorMachineError("soft query tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.action_path.shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.action_path.shape[1])

    @torch.no_grad()
    def harden(self, machine: HardFunctorMachine) -> "HardFunctorQuery":
        if self.batch_size != machine.batch_size:
            raise FunctorMachineError("soft query and hard machine batches differ")
        negative = torch.finfo(self.action_path.dtype).min
        action_path = self.action_path.masked_fill(
            ~machine.action_active.bool()[:, None],
            negative,
        ).argmax(-1)
        observer = self.observer.masked_fill(
            ~machine.observer_active.bool(),
            negative,
        ).argmax(-1)
        start_state = self.start_state.masked_fill(
            ~machine.state_active.bool(),
            negative,
        ).argmax(-1)
        return HardFunctorQuery(
            start_state=start_state.to(torch.uint8),
            action_path=action_path.to(torch.uint8),
            stop_position=self.stop_position.argmax(-1).to(torch.uint8),
            observer=observer.to(torch.uint8),
        )


@dataclass(frozen=True, slots=True)
class HardFunctorQuery:
    """Detached query parse consumed by the hard executor."""

    start_state: torch.Tensor
    action_path: torch.Tensor
    stop_position: torch.Tensor
    observer: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.action_path, torch.Tensor) or self.action_path.ndim != 2:
            raise FunctorMachineError("hard action_path must be rank two")
        batch, steps = self.action_path.shape
        if not 1 <= steps <= MAX_STEPS:
            raise FunctorMachineError("hard action_path length is out of range")
        _byte_tensor("hard start_state", self.start_state, (batch,))
        _byte_tensor("hard action_path", self.action_path, (batch, steps))
        _byte_tensor("hard stop_position", self.stop_position, (batch,))
        _byte_tensor("hard observer", self.observer, (batch,))
        if self.action_path.numel() and int(self.action_path.max()) >= MAX_ACTIONS:
            raise FunctorMachineError("hard action_path leaves the action domain")
        if self.start_state.numel() and int(self.start_state.max()) >= MAX_STATES:
            raise FunctorMachineError("hard start_state leaves the state domain")
        if self.stop_position.numel() and int(self.stop_position.max()) > steps:
            raise FunctorMachineError("hard STOP leaves the path domain")
        if self.observer.numel() and int(self.observer.max()) >= MAX_OBSERVERS:
            raise FunctorMachineError("hard observer leaves the observer domain")
        if len(
            {
                self.start_state.device,
                self.action_path.device,
                self.stop_position.device,
                self.observer.device,
            }
        ) != 1:
            raise FunctorMachineError("hard query tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.action_path.shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.action_path.shape[1])

    def remap_actions(self, mapping: Sequence[int]) -> "HardFunctorQuery":
        checked = _permutation(mapping, MAX_ACTIONS, "action mapping")
        lookup = torch.tensor(checked, dtype=torch.uint8, device=self.action_path.device)
        return replace(self, action_path=lookup[self.action_path.long()])

    def remap_observers(self, mapping: Sequence[int]) -> "HardFunctorQuery":
        checked = _permutation(mapping, MAX_OBSERVERS, "observer mapping")
        lookup = torch.tensor(checked, dtype=torch.uint8, device=self.observer.device)
        return replace(self, observer=lookup[self.observer.long()])


@dataclass(frozen=True, slots=True)
class FunctorRollout:
    """State trajectory and answer distribution or categorical answer."""

    states: torch.Tensor
    answer: torch.Tensor


def execute_soft(
    machine: SoftFunctorMachine,
    query: SoftFunctorQuery,
    *,
    straight_through: bool = False,
) -> FunctorRollout:
    """Execute attached probabilities without source or compiler evidence."""

    if machine.batch_size != query.batch_size:
        raise FunctorMachineError("soft machine and query batches differ")
    state_active = _straight_through_active(machine.state_active)
    action_active = _straight_through_active(machine.action_active)
    observer_active = _straight_through_active(machine.observer_active)
    distribution = (
        _straight_through_distribution
        if straight_through
        else _supported_softmax
    )
    initial = distribution(query.start_state, state_active)
    destination_support = state_active[:, None, None, :].expand_as(
        machine.action_next
    )
    transitions = distribution(
        machine.action_next,
        destination_support,
    )
    action_support = action_active[:, None, :].expand_as(query.action_path)
    action_path = distribution(
        query.action_path,
        action_support,
    )
    stop = (
        _straight_through_distribution(query.stop_position)
        if straight_through
        else query.stop_position.float().softmax(-1)
    )
    observer = distribution(
        query.observer,
        observer_active,
    )
    answer_map = (
        _straight_through_distribution(machine.observer_answer)
        if straight_through
        else machine.observer_answer.float().softmax(-1)
    )
    state = initial
    trajectory = [state]
    stop_indices = torch.arange(
        query.max_steps + 1,
        device=stop.device,
    )
    for step in range(query.max_steps):
        selected_transition = torch.einsum(
            "ba,bask->bsk",
            action_path[:, step],
            transitions,
        )
        updated = torch.einsum("bs,bsk->bk", state, selected_transition)
        alive = (
            stop
            * stop_indices.gt(step).to(stop.dtype)[None]
        ).sum(-1, keepdim=True)
        state = alive * updated + (1.0 - alive) * state
        trajectory.append(state)
    selected_observer = torch.einsum(
        "bp,bpsy->bsy",
        observer,
        answer_map,
    )
    answer = torch.einsum("bs,bsy->by", state, selected_observer)
    return FunctorRollout(states=torch.stack(trajectory, dim=1), answer=answer)


@torch.no_grad()
def execute_hard(
    machine: HardFunctorMachine,
    query: HardFunctorQuery,
    *,
    reset_each_step: bool = False,
) -> FunctorRollout:
    """Execute only detached categorical fields."""

    if machine.batch_size != query.batch_size:
        raise FunctorMachineError("hard machine and query batches differ")
    selected_start_is_active = machine.state_active.bool().gather(
        1,
        query.start_state.long()[:, None],
    ).squeeze(1)
    if bool((~selected_start_is_active).any()):
        raise FunctorMachineError("hard query selects an inactive start state")
    selected_actions_are_active = machine.action_active.bool().gather(
        1,
        query.action_path.long(),
    )
    live_steps = (
        torch.arange(query.max_steps, device=query.action_path.device)[None]
        < query.stop_position.long()[:, None]
    )
    if bool((~selected_actions_are_active & live_steps).any()):
        raise FunctorMachineError("hard query selects an inactive action")
    selected_observer_is_active = machine.observer_active.bool().gather(
        1,
        query.observer.long()[:, None],
    ).squeeze(1)
    if bool((~selected_observer_is_active).any()):
        raise FunctorMachineError("hard query selects an inactive observer")
    state = query.start_state.long()
    initial = state.clone()
    trajectory = [
        F.one_hot(state, MAX_STATES).to(torch.uint8)
    ]
    batch = torch.arange(machine.batch_size, device=state.device)
    for step in range(query.max_steps):
        action = query.action_path[:, step].long()
        destination = machine.action_next[batch, action, state].long()
        active = step < query.stop_position.long()
        state = torch.where(active, destination, state)
        if reset_each_step:
            state = torch.where(active, initial, state)
        trajectory.append(F.one_hot(state, MAX_STATES).to(torch.uint8))
    observer = query.observer.long()
    answer = machine.observer_answer[batch, observer, state].long()
    return FunctorRollout(
        states=torch.stack(trajectory, dim=1),
        answer=answer,
    )


def _permutation(
    values: Sequence[int],
    size: int,
    label: str,
) -> tuple[int, ...]:
    checked = tuple(int(value) for value in values)
    if len(checked) != size or sorted(checked) != list(range(size)):
        raise FunctorMachineError(f"{label} is not a complete permutation")
    return checked


def _inverse_permutation(values: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(values)
    for old, new in enumerate(values):
        inverse[new] = old
    return tuple(inverse)


def _prefix_count(mask: torch.Tensor, label: str) -> int:
    values = tuple(int(value) for value in mask.tolist())
    count = sum(values)
    if values != (1,) * count + (0,) * (len(values) - count):
        raise FunctorMachineError(f"{label} mask is not prefix-canonical")
    return count


__all__ = [
    "FunctorMachineError",
    "FunctorRollout",
    "HardFunctorKeys",
    "HardFunctorMachine",
    "HardFunctorQuery",
    "DEPLOYED_HASH_OFFSET",
    "DEPLOYED_LEGACY_INITIAL_OFFSET",
    "DEPLOYED_MACHINE_BYTES",
    "LearnedFunctorWireSpec",
    "MAX_ACTIONS",
    "MAX_ANSWERS",
    "MAX_OBSERVERS",
    "MAX_STATES",
    "MAX_STEPS",
    "SEMANTIC_BYTES_PER_ROW",
    "SoftFunctorMachine",
    "SoftFunctorQuery",
    "execute_hard",
    "execute_soft",
]
