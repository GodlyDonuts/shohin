"""Source-deleted categorical state transport (SD-CST) model core.

The module enforces a hard architectural boundary: source tokens are compiled
once into an initial-state category and three categorical event tensors. The payload is committed
before a late query is disclosed. Only hard integer categories may reach the
score-bearing recurrent state machine. The transition law and answer reader are learned;
there is no symbolic action table or host-side semantic execution.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


EVENT_STEPS = 8
EVENT_KIND_COUNT = 3
IDENTITY_COUNT = 3
AMOUNT_COUNT = 2
STATE_COUNT = 6
QUERY_COUNT = 3
ANSWER_COUNT = 3
STOP_KIND = 2
MAX_SYSTEM_PARAMETERS = 150_000_000


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
        actual = None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)
        raise ValueError(f"{name} must have shape {shape}, got {actual}")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _categorical(logits: torch.Tensor, *, hard: bool) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=-1).to(logits.dtype)
    if not hard:
        return probabilities
    selected = F.one_hot(
        probabilities.argmax(dim=-1), num_classes=probabilities.shape[-1],
    ).to(probabilities.dtype)
    return selected + probabilities - probabilities.detach()


def _validate_batch_permutation(
    permutation: torch.Tensor, batch: int, device: torch.device,
) -> torch.Tensor:
    if permutation.dtype != torch.long or permutation.shape != (batch,):
        raise ValueError("batch permutation must be a rank-1 torch.long tensor")
    permutation = permutation.to(device=device)
    expected = torch.arange(batch, device=device)
    if not torch.equal(permutation.sort().values, expected):
        raise ValueError("batch permutation must contain every batch index exactly once")
    return permutation


@dataclass(frozen=True, slots=True)
class DeletedProgramTape:
    """The complete and exclusive source-deletion boundary payload.

    Fields are training-time logits. No query, source IDs, masks, token memory,
    pointers, text, confidence side channel, or full-source tensors are retained.
    """

    initial_state: torch.Tensor
    event_kind: torch.Tensor
    event_identity: torch.Tensor
    amount: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, torch.Tensor) or self.event_kind.ndim != 3:
            raise ValueError("event_kind must have shape [batch,8,3]")
        batch = self.event_kind.shape[0]
        _require_shape("initial_state", self.initial_state, (batch, STATE_COUNT))
        _require_shape(
            "event_kind", self.event_kind,
            (batch, EVENT_STEPS, EVENT_KIND_COUNT),
        )
        _require_shape(
            "event_identity", self.event_identity,
            (batch, EVENT_STEPS, IDENTITY_COUNT),
        )
        _require_shape(
            "amount", self.amount, (batch, EVENT_STEPS, AMOUNT_COUNT),
        )
        devices = {
            self.initial_state.device,
            self.event_kind.device,
            self.event_identity.device,
            self.amount.device,
        }
        if len(devices) != 1:
            raise ValueError("all DeletedProgramTape fields must be on one device")

    @property
    def batch_size(self) -> int:
        return int(self.event_kind.shape[0])

    def detached_clone(self) -> "DeletedProgramTape":
        return DeletedProgramTape(*(
            getattr(self, field.name).detach().clone() for field in fields(self)
        ))

    def hard(self) -> "HardProgramTape":
        return HardProgramTape(
            self.initial_state.argmax(-1).to(torch.uint8),
            self.event_kind.argmax(-1).to(torch.uint8),
            self.event_identity.argmax(-1).to(torch.uint8),
            self.amount.argmax(-1).to(torch.uint8),
        )


@dataclass(frozen=True, slots=True)
class LateQuery:
    """A query compiled only after the program payload has been committed."""

    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 2:
            raise ValueError("late query logits must have shape [batch,3]")
        _require_shape("late_query", self.logits, (self.logits.shape[0], QUERY_COUNT))

    @property
    def batch_size(self) -> int:
        return int(self.logits.shape[0])

    def hard(self) -> "HardLateQuery":
        return HardLateQuery(self.logits.argmax(-1).to(torch.uint8))


@dataclass(frozen=True, slots=True)
class HardProgramTape:
    """The score-bearing channel: 25 categorical bytes per program row."""

    initial_state: torch.Tensor
    event_kind: torch.Tensor
    event_identity: torch.Tensor
    amount: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, torch.Tensor) or self.event_kind.ndim != 2:
            raise ValueError("hard event_kind must have shape [batch,8]")
        batch = self.event_kind.shape[0]
        if (
            self.initial_state.shape != (batch,)
            or self.initial_state.dtype != torch.uint8
        ):
            raise ValueError("hard initial_state must be rank-1 uint8")
        if self.initial_state.numel() and int(self.initial_state.max()) >= STATE_COUNT:
            raise ValueError("hard initial_state is outside its categorical range")
        shape = (batch, EVENT_STEPS)
        for name, value, upper in (
            ("event_kind", self.event_kind, EVENT_KIND_COUNT),
            ("event_identity", self.event_identity, IDENTITY_COUNT),
            ("amount", self.amount, AMOUNT_COUNT),
        ):
            if value.shape != shape or value.dtype != torch.uint8:
                raise ValueError(f"hard {name} must be uint8 with shape {shape}")
            if value.numel() and int(value.max()) >= upper:
                raise ValueError(f"hard {name} is outside its categorical range")
        stop_counts = self.event_kind.eq(STOP_KIND).sum(dim=1)
        if not torch.equal(stop_counts, torch.ones_like(stop_counts)):
            raise ValueError("hard program tape requires exactly one STOP per row")

    @property
    def batch_size(self) -> int:
        return int(self.event_kind.shape[0])


@dataclass(frozen=True, slots=True)
class HardLateQuery:
    position: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position, torch.Tensor)
            or self.position.ndim != 1
            or self.position.dtype != torch.uint8
        ):
            raise ValueError("hard late query must be rank-1 uint8")
        if self.position.numel() and int(self.position.max()) >= QUERY_COUNT:
            raise ValueError("hard late query is outside its categorical range")


@dataclass(frozen=True, slots=True)
class StateSwap:
    """Swap only the six-way register after one fixed rollout step."""

    after_step: int
    batch_permutation: torch.Tensor

    def __post_init__(self) -> None:
        if not 0 <= int(self.after_step) < EVENT_STEPS:
            raise ValueError(f"state swap step must be in [0,{EVENT_STEPS - 1}]")


@dataclass(frozen=True, slots=True)
class RolloutResult:
    final_state: torch.Tensor
    answer_logits: torch.Tensor
    state_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]
    motor_logits: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class HardRolloutResult:
    final_state: torch.Tensor
    answer_logits: torch.Tensor
    state_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class SourcePoisonResult:
    bit_identical: bool
    clean: RolloutResult
    poisoned: RolloutResult


class SDCSTSourceCompiler(nn.Module):
    """Compile program and late query in separate, stateless invocations."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        layer: int = 19,
        width: int = 384,
        heads: int = 8,
        encoder_layers: int = 5,
        ff: int = 1408,
    ) -> None:
        super().__init__()
        if base_model is None:
            raise ValueError("a frozen Shohin-compatible base model is required")
        if getattr(base_model.cfg, "n_loop", 1) != 1:
            raise ValueError("SD-CST source compilation requires n_loop=1")
        if not 0 <= int(layer) < len(base_model.blocks):
            raise ValueError("compiler layer is outside the frozen base")
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("compiler width must be positive and divide attention heads")
        if encoder_layers <= 0 or ff <= 0:
            raise ValueError("encoder depth and feed-forward width must be positive")

        self.base_model = base_model
        self.layer = int(layer)
        self.width = int(width)
        self.encoder_layers = int(encoder_layers)
        self.base_model.requires_grad_(False)
        self.base_model.eval()

        d_model = int(base_model.cfg.d_model)
        self.source_norm = nn.LayerNorm(d_model)
        self.source_projection = nn.Linear(d_model, width, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        self.initial_slot = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.initial_slot, mean=0.0, std=0.02)
        self.slot_queries = nn.Parameter(torch.empty(EVENT_STEPS, width))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.query_slot = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.query_slot, mean=0.0, std=0.02)
        self.slot_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(width)
        self.initial_head = nn.Linear(width, STATE_COUNT)
        self.kind_head = nn.Linear(width, EVENT_KIND_COUNT)
        self.identity_head = nn.Linear(width, IDENTITY_COUNT)
        self.amount_head = nn.Linear(width, AMOUNT_COUNT)
        self.query_head = nn.Linear(width, QUERY_COUNT)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("base_model."):
                yield parameter

    def adapter_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def _frozen_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        self.base_model.eval()
        with torch.no_grad():
            hidden = self.base_model.tok(ids)
            cosine = self.base_model.cos[:ids.shape[1]].to(hidden.device)
            sine = self.base_model.sin[:ids.shape[1]].to(hidden.device)
            for block in self.base_model.blocks[:self.layer + 1]:
                hidden, _ = block(hidden, cosine, sine)
        return hidden.detach()

    def _encode(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a rank-2 torch.long tensor")
        if valid_mask.shape != ids.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean and match ids")
        if ids.shape[1] < 1 or ids.shape[1] > self.base_model.cfg.seq_len:
            raise ValueError("source length is outside the base context window")
        if not bool(valid_mask.any(dim=-1).all()):
            raise ValueError("every source must contain at least one valid token")

        source_hidden = self._frozen_hidden(ids)
        memory = self.source_projection(self.source_norm(source_hidden))
        return self.encoder(memory, src_key_padding_mask=~valid_mask)

    def forward(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> DeletedProgramTape:
        memory = self._encode(ids, valid_mask)
        requested = torch.cat((self.initial_slot, self.slot_queries), dim=0)
        slots = requested.unsqueeze(0).expand(ids.shape[0], -1, -1)
        slots, _ = self.slot_attention(
            slots,
            memory,
            memory,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        slots = self.slot_norm(slots)
        initial = slots[:, 0]
        events = slots[:, 1:]

        # Returning the strict dataclass is the deletion event. Memory,
        # source IDs, masks, and base residuals become unreachable here.
        return DeletedProgramTape(
            initial_state=self.initial_head(initial).float(),
            event_kind=self.kind_head(events).float(),
            event_identity=self.identity_head(events).float(),
            amount=self.amount_head(events).float(),
        )

    def compile_late_query(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> LateQuery:
        """Compile a query without access to program text, tape, or state."""
        memory = self._encode(ids, valid_mask)
        query_slot = self.query_slot.unsqueeze(0).expand(ids.shape[0], -1, -1)
        query_slot, _ = self.slot_attention(
            query_slot,
            memory,
            memory,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        query_slot = self.slot_norm(query_slot[:, 0])
        return LateQuery(self.query_head(query_slot).float())


class TiedCategoricalMotor(nn.Module):
    """Learned one-step transition over a six-way categorical register."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("motor hidden width must be positive")
        self.hidden = int(hidden)
        input_width = STATE_COUNT + EVENT_KIND_COUNT + IDENTITY_COUNT + AMOUNT_COUNT
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, STATE_COUNT),
        )

    def forward(
        self,
        state: torch.Tensor,
        event_kind: torch.Tensor,
        event_identity: torch.Tensor,
        amount: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        _require_shape("state", state, (batch, STATE_COUNT))
        _require_shape("event_kind", event_kind, (batch, EVENT_KIND_COUNT))
        _require_shape("event_identity", event_identity, (batch, IDENTITY_COUNT))
        _require_shape("amount", amount, (batch, AMOUNT_COUNT))
        return self.network(torch.cat((state, event_kind, event_identity, amount), dim=-1))


class CategoricalStateReader(nn.Module):
    """Independent learned answer reader over final state and query only."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("reader hidden width must be positive")
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(STATE_COUNT + QUERY_COUNT, hidden),
            nn.GELU(),
            nn.Linear(hidden, ANSWER_COUNT),
        )

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        _require_shape("state", state, (batch, STATE_COUNT))
        _require_shape("query", query, (batch, QUERY_COUNT))
        return self.network(torch.cat((state, query), dim=-1))


@torch.no_grad()
def rollout_hard_categorical(
    motor: TiedCategoricalMotor,
    reader: CategoricalStateReader,
    tape: HardProgramTape,
    query_packet: HardLateQuery,
    *,
    control: Literal["normal", "reset", "freeze"] = "normal",
    state_swap: StateSwap | None = None,
    force_alive: bool = False,
) -> HardRolloutResult:
    """Execute the production hard-byte path without requiring a source compiler."""
    if not isinstance(tape, HardProgramTape):
        raise TypeError("hard rollout requires HardProgramTape")
    if not isinstance(query_packet, HardLateQuery):
        raise TypeError("hard rollout requires HardLateQuery")
    if tape.batch_size != query_packet.position.shape[0]:
        raise ValueError("hard program/query batch sizes differ")
    if control not in {"normal", "reset", "freeze"}:
        raise ValueError("control must be normal, reset, or freeze")
    device = next(motor.parameters()).device
    kind_ids = tape.event_kind.to(device=device, dtype=torch.long)
    identity_ids = tape.event_identity.to(device=device, dtype=torch.long)
    amount_ids = tape.amount.to(device=device, dtype=torch.long)
    query_ids = query_packet.position.to(device=device, dtype=torch.long)
    batch = tape.batch_size
    state_ids = tape.initial_state.to(device=device, dtype=torch.long)
    initial_ids = state_ids.clone()
    alive = torch.ones(batch, device=device, dtype=torch.bool)
    state_trajectory = []
    alive_trajectory = []
    for step in range(EVENT_STEPS):
        motor_state = initial_ids if control == "reset" else state_ids
        logits = motor(
            F.one_hot(motor_state, STATE_COUNT).float(),
            F.one_hot(kind_ids[:, step], EVENT_KIND_COUNT).float(),
            F.one_hot(identity_ids[:, step], IDENTITY_COUNT).float(),
            F.one_hot(amount_ids[:, step], AMOUNT_COUNT).float(),
        )
        proposal_ids = logits.argmax(-1)
        stop = kind_ids[:, step].eq(STOP_KIND)
        update = alive & ~stop
        if control != "freeze":
            state_ids = torch.where(update, proposal_ids, state_ids)
        if not force_alive:
            alive = alive & ~stop
        if state_swap is not None and step == state_swap.after_step:
            permutation = _validate_batch_permutation(
                state_swap.batch_permutation, batch, state_ids.device,
            )
            state_ids = state_ids.index_select(0, permutation)
        state_trajectory.append(state_ids.to(torch.uint8))
        alive_trajectory.append(alive.clone())
    answer_logits = reader(
        F.one_hot(state_ids, STATE_COUNT).float(),
        F.one_hot(query_ids, QUERY_COUNT).float(),
    )
    return HardRolloutResult(
        final_state=state_ids.to(torch.uint8),
        answer_logits=answer_logits,
        state_trajectory=tuple(state_trajectory),
        alive_trajectory=tuple(alive_trajectory),
    )


def swap_tape_suffix(
    tape: DeletedProgramTape,
    batch_permutation: torch.Tensor,
    *,
    start_step: int,
) -> DeletedProgramTape:
    """Keep each recipient prefix and replace only its remaining event tape."""
    if not 0 <= int(start_step) <= EVENT_STEPS:
        raise ValueError(f"suffix start must be in [0,{EVENT_STEPS}]")
    permutation = _validate_batch_permutation(
        batch_permutation, tape.batch_size, tape.event_kind.device,
    )

    def swapped(field: torch.Tensor) -> torch.Tensor:
        donor = field.index_select(0, permutation)
        return torch.cat((field[:, :start_step], donor[:, start_step:]), dim=1)

    return DeletedProgramTape(
        initial_state=tape.initial_state,
        event_kind=swapped(tape.event_kind),
        event_identity=swapped(tape.event_identity),
        amount=swapped(tape.amount),
    )


def swap_late_queries(
    query: LateQuery, batch_permutation: torch.Tensor,
) -> LateQuery:
    """Exchange only late queries; no program payload is available here."""
    permutation = _validate_batch_permutation(
        batch_permutation, query.batch_size, query.logits.device,
    )
    return LateQuery(query.logits.index_select(0, permutation))


class SDCSTSystem(nn.Module):
    """Frozen source compiler plus source-deleted recurrent execution core."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        compiler_layer: int = 19,
        compiler_width: int = 384,
        compiler_heads: int = 8,
        compiler_layers: int = 5,
        compiler_ff: int = 1408,
        motor_hidden: int = 128,
        reader_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.compiler = SDCSTSourceCompiler(
            base_model,
            layer=compiler_layer,
            width=compiler_width,
            heads=compiler_heads,
            encoder_layers=compiler_layers,
            ff=compiler_ff,
        )
        self.motor = TiedCategoricalMotor(motor_hidden)
        self.reader = CategoricalStateReader(reader_hidden)
        report = self.parameter_report()
        if report["complete_system"] >= MAX_SYSTEM_PARAMETERS:
            raise ValueError(
                "SD-CST complete system exceeds strict 150M cap: "
                f"{report['complete_system']}"
            )

    @property
    def base_model(self) -> nn.Module:
        return self.compiler.base_model

    def parameter_report(self) -> dict[str, int]:
        base_ids = {id(parameter) for parameter in self.base_model.parameters()}
        base = sum(parameter.numel() for parameter in self.base_model.parameters())
        complete_parameters = list(self.parameters())
        complete = sum(parameter.numel() for parameter in complete_parameters)
        compiler_added = sum(
            parameter.numel()
            for parameter in self.compiler.parameters()
            if id(parameter) not in base_ids
        )
        motor = sum(parameter.numel() for parameter in self.motor.parameters())
        reader = sum(parameter.numel() for parameter in self.reader.parameters())
        trainable = sum(
            parameter.numel() for parameter in complete_parameters
            if parameter.requires_grad
        )
        return {
            "base": int(base),
            "compiler_added": int(compiler_added),
            "motor": int(motor),
            "reader": int(reader),
            "added": int(compiler_added + motor + reader),
            "trainable": int(trainable),
            "complete_system": int(complete),
            "strict_cap": MAX_SYSTEM_PARAMETERS,
            "headroom": int(MAX_SYSTEM_PARAMETERS - complete),
        }

    def compile_program(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> DeletedProgramTape:
        return self.compiler(ids, valid_mask)

    def compile_late_query(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> LateQuery:
        return self.compiler.compile_late_query(ids, valid_mask)

    def rollout(
        self,
        tape: DeletedProgramTape,
        query_packet: LateQuery,
        *,
        hard: bool = False,
        control: Literal["normal", "reset", "freeze"] = "normal",
        state_swap: StateSwap | None = None,
    ) -> RolloutResult:
        if not isinstance(tape, DeletedProgramTape):
            raise TypeError("rollout accepts only a DeletedProgramTape")
        if not isinstance(query_packet, LateQuery):
            raise TypeError("rollout requires a separately compiled LateQuery")
        if tape.batch_size != query_packet.batch_size:
            raise ValueError("program tape and late query batch sizes differ")
        if control not in {"normal", "reset", "freeze"}:
            raise ValueError("control must be normal, reset, or freeze")
        batch = tape.batch_size
        state = _categorical(tape.initial_state, hard=hard)
        initial = state
        alive = torch.ones((batch, 1), device=state.device, dtype=state.dtype)
        kinds = _categorical(tape.event_kind, hard=hard)
        identities = _categorical(tape.event_identity, hard=hard)
        amounts = _categorical(tape.amount, hard=hard)
        query = _categorical(query_packet.logits, hard=hard)
        states: list[torch.Tensor] = []
        alive_states: list[torch.Tensor] = []
        motor_logits: list[torch.Tensor] = []

        # This loop has a fixed public trip count.  Every arm calls the same
        # tied motor exactly eight times; no semantic value controls Python.
        for step in range(EVENT_STEPS):
            motor_input = initial if control == "reset" else state
            proposal_logits = self.motor(
                motor_input,
                kinds[:, step],
                identities[:, step],
                amounts[:, step],
            )
            proposal = _categorical(proposal_logits, hard=hard)
            action_gate = 1.0 - kinds[:, step, STOP_KIND:STOP_KIND + 1]
            if control == "freeze":
                candidate = state + proposal.sum(dim=-1, keepdim=True) * 0.0
            else:
                candidate = proposal
            state = state + alive * action_gate * (candidate - state)
            alive = alive * action_gate
            if state_swap is not None and step == state_swap.after_step:
                permutation = _validate_batch_permutation(
                    state_swap.batch_permutation, batch, state.device,
                )
                state = state.index_select(0, permutation)
            states.append(state)
            alive_states.append(alive)
            motor_logits.append(proposal_logits)

        answer_logits = self.reader(state, query)
        return RolloutResult(
            final_state=state,
            answer_logits=answer_logits,
            state_trajectory=tuple(states),
            alive_trajectory=tuple(alive_states),
            motor_logits=tuple(motor_logits),
        )

    @torch.no_grad()
    def rollout_hard(
        self,
        tape: HardProgramTape,
        query_packet: HardLateQuery,
        *,
        control: Literal["normal", "reset", "freeze"] = "normal",
        state_swap: StateSwap | None = None,
        force_alive: bool = False,
    ) -> HardRolloutResult:
        """Score only integer categories; no logit or confidence survives a step."""
        return rollout_hard_categorical(
            self.motor,
            self.reader,
            tape,
            query_packet,
            control=control,
            state_swap=state_swap,
            force_alive=force_alive,
        )

    def source_poison_invariance(
        self,
        tape: DeletedProgramTape,
        query_packet: LateQuery,
        poison_source: Callable[[], None],
        *,
        hard: bool = True,
        control: Literal["normal", "reset", "freeze"] = "normal",
    ) -> SourcePoisonResult:
        """Poison external source storage after deletion and replay the tape."""
        if not callable(poison_source):
            raise TypeError("poison_source must be a zero-argument callable")
        sealed_tape = tape.detached_clone()
        sealed_query = LateQuery(query_packet.logits.detach().clone())
        clean = self.rollout(
            sealed_tape, sealed_query, hard=hard, control=control,
        )
        poison_source()
        poisoned = self.rollout(
            sealed_tape, sealed_query, hard=hard, control=control,
        )
        clean_tensors = (
            clean.final_state,
            clean.answer_logits,
            *clean.state_trajectory,
            *clean.alive_trajectory,
            *clean.motor_logits,
        )
        poisoned_tensors = (
            poisoned.final_state,
            poisoned.answer_logits,
            *poisoned.state_trajectory,
            *poisoned.alive_trajectory,
            *poisoned.motor_logits,
        )
        identical = all(
            torch.equal(left, right)
            for left, right in zip(clean_tensors, poisoned_tensors, strict=True)
        )
        return SourcePoisonResult(identical, clean, poisoned)


def compiler_field_losses(
    tape: DeletedProgramTape,
    *,
    initial_state_targets: torch.Tensor,
    event_kind_targets: torch.Tensor,
    event_identity_targets: torch.Tensor,
    amount_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Factorized source-compiler supervision; no episode outcome is used."""
    batch = tape.batch_size
    if initial_state_targets.shape != (batch,) or initial_state_targets.dtype != torch.long:
        raise ValueError("initial_state_targets must be torch.long with shape [batch]")
    expected_events = (batch, EVENT_STEPS)
    for name, target in (
        ("event_kind_targets", event_kind_targets),
        ("event_identity_targets", event_identity_targets),
        ("amount_targets", amount_targets),
    ):
        if target.shape != expected_events or target.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long with shape {expected_events}")
    initial_state = F.cross_entropy(tape.initial_state, initial_state_targets)
    kind = F.cross_entropy(
        tape.event_kind.reshape(-1, EVENT_KIND_COUNT),
        event_kind_targets.reshape(-1),
    )
    active = event_kind_targets.ne(STOP_KIND).reshape(-1)
    if not bool(active.any()):
        raise ValueError("compiler supervision requires at least one non-STOP event")
    identity = F.cross_entropy(
        tape.event_identity.reshape(-1, IDENTITY_COUNT)[active],
        event_identity_targets.reshape(-1)[active],
    )
    amount = F.cross_entropy(
        tape.amount.reshape(-1, AMOUNT_COUNT)[active],
        amount_targets.reshape(-1)[active],
    )
    total = initial_state + kind + identity + amount
    return {
        "total": total,
        "initial_state": initial_state,
        "event_kind": kind,
        "event_identity": identity,
        "amount": amount,
    }


def late_query_loss(
    query: LateQuery, *, query_targets: torch.Tensor,
) -> torch.Tensor:
    if (
        query_targets.shape != (query.batch_size,)
        or query_targets.dtype != torch.long
    ):
        raise ValueError("query_targets must be torch.long with shape [batch]")
    return F.cross_entropy(query.logits, query_targets)


def atomic_motor_loss(
    motor: TiedCategoricalMotor,
    *,
    state: torch.Tensor,
    event_kind: torch.Tensor,
    event_identity: torch.Tensor,
    amount: torch.Tensor,
    next_state_targets: torch.Tensor,
) -> torch.Tensor:
    """Train only independently sampled one-step transitions."""
    if next_state_targets.shape != (state.shape[0],) or next_state_targets.dtype != torch.long:
        raise ValueError("next_state_targets must be torch.long with shape [batch]")
    logits = motor(state, event_kind, event_identity, amount)
    return F.cross_entropy(logits, next_state_targets)


def reader_loss(
    reader: CategoricalStateReader,
    *,
    state: torch.Tensor,
    query: torch.Tensor,
    answer_targets: torch.Tensor,
) -> torch.Tensor:
    """Train the reader on independently sampled state/query pairs."""
    if answer_targets.shape != (state.shape[0],) or answer_targets.dtype != torch.long:
        raise ValueError("answer_targets must be torch.long with shape [batch]")
    return F.cross_entropy(reader(state, query), answer_targets)
