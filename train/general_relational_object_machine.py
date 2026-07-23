"""Source-deleted relational object machine for general-reasoning experiments.

The compiler reads source text once and emits a private object file containing
only categorical relation cards, an initial state, and an event tape. The
executor never receives source tokens, token residuals, pointer logits,
identity carriers, parser metadata, or verifier feedback. A separate late
query is compiled after the recurrent state has committed.

The recurrent core is shared across task families. A relation card is a
row-stochastic map over an episode-local object set, so the same tensor
composition can implement list permutation, graph/function traversal, and
finite-state transition programs without a task-specific host executor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctaa_trunk_compiler import TrunkCausalCTAACompiler, TrunkResidualBundle


MIN_OBJECTS = 2
MAX_OBJECTS = 8
MAX_RULES = 8
MAX_EVENTS = 32
MAX_RELATION_EDGES = 24
APPLY_KIND = 0
STOP_KIND = 1
NOOP_KIND = 2
EVENT_KIND_COUNT = 3
STRICT_PARAMETER_CAP = 200_000_000


class RelationalObjectError(ValueError):
    """Raised when an object-file or recurrence contract is violated."""


def _require_float(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise RelationalObjectError(f"{name} must be finite floating point {shape}")


def _one_stop_map(kind_logits: torch.Tensor) -> torch.Tensor:
    """Decode the maximum-score tape under an exactly-one-STOP grammar."""

    if kind_logits.ndim != 3 or kind_logits.shape[-1] != EVENT_KIND_COUNT:
        raise RelationalObjectError("event-kind logits differ")
    non_stop_logits = kind_logits[..., (APPLY_KIND, NOOP_KIND)]
    non_stop_score, non_stop_local = non_stop_logits.max(-1)
    non_stop_kind = torch.where(
        non_stop_local.eq(0),
        torch.full_like(non_stop_local, APPLY_KIND),
        torch.full_like(non_stop_local, NOOP_KIND),
    )
    stop_gain = kind_logits[..., STOP_KIND] - non_stop_score
    stop_slot = stop_gain.argmax(-1)
    decoded = non_stop_kind
    decoded = decoded.scatter(
        1,
        stop_slot[:, None],
        torch.full_like(stop_slot[:, None], STOP_KIND),
    )
    return decoded


def _cardinality_probabilities(
    logits: torch.Tensor,
    *,
    hard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.float().softmax(-1)
    if hard:
        selected = F.one_hot(
            probabilities.argmax(-1),
            probabilities.shape[-1],
        ).float()
        probabilities = selected + probabilities - probabilities.detach()
    cardinalities = torch.arange(
        MIN_OBJECTS,
        MAX_OBJECTS + 1,
        device=logits.device,
    )
    positions = torch.arange(MAX_OBJECTS, device=logits.device)
    active_by_cardinality = positions[None] < cardinalities[:, None]
    active = torch.einsum(
        "bc,cn->bn",
        probabilities,
        active_by_cardinality.float(),
    )
    return probabilities, active


def _masked_relation_probabilities(
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    hard: bool,
) -> torch.Tensor:
    """Decode independent relation edges inside the active object square."""

    if logits.ndim not in {3, 4} or logits.shape[-2:] != (
        MAX_OBJECTS,
        MAX_OBJECTS,
    ):
        raise RelationalObjectError("relation logits differ")
    if active.shape != (logits.shape[0], MAX_OBJECTS):
        raise RelationalObjectError("relation active mask differs")
    leading = (slice(None),) + (None,) * (logits.ndim - 3)
    input_active = active[leading + (None, slice(None))]
    output_active = active[leading + (slice(None), None)]
    edge_mask = input_active * output_active
    probabilities = logits.float().sigmoid() * edge_mask
    if hard:
        selected = probabilities.ge(0.5).float() * edge_mask
        empty = selected.sum(-1, keepdim=True).eq(0) & output_active.bool()
        fallback_logits = logits.float() + input_active.clamp_min(1e-30).log()
        fallback = F.one_hot(
            fallback_logits.argmax(-1),
            MAX_OBJECTS,
        ).float() * output_active
        selected = torch.where(empty, fallback, selected)
        probabilities = selected + probabilities - probabilities.detach()
    return probabilities


def _nonempty_relation_map(
    logits: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return binary relation edges with at least one input per active row."""

    if logits.ndim not in {3, 4} or logits.shape[-2:] != (
        MAX_OBJECTS,
        MAX_OBJECTS,
    ):
        raise RelationalObjectError("relation-map logits differ")
    leading = (slice(None),) + (None,) * (logits.ndim - 3)
    input_active = active[leading + (None, slice(None))]
    output_active = active[leading + (slice(None), None)]
    edge_mask = input_active & output_active
    selected = logits.ge(0) & edge_mask
    empty = selected.sum(-1, keepdim=True).eq(0) & output_active
    negative = torch.finfo(logits.dtype).min
    fallback = F.one_hot(
        logits.masked_fill(~input_active, negative).argmax(-1),
        MAX_OBJECTS,
    ).bool() & output_active
    return torch.where(empty, fallback, selected)


def probabilistic_relation_compose(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compose fuzzy relations with a differentiable Boolean semiring."""

    if (
        left.ndim != 3
        or right.ndim != 3
        or left.shape != right.shape
        or left.shape[-2:] != (MAX_OBJECTS, MAX_OBJECTS)
    ):
        raise RelationalObjectError("relation-composition geometry differs")
    path = left[:, :, :, None] * right[:, None, :, :]
    return 1.0 - (1.0 - path).prod(dim=2)


@dataclass(frozen=True, slots=True)
class DeletedRelationalProgram:
    """Exclusive differentiable payload crossing the source-deletion boundary."""

    cardinality: torch.Tensor
    initial_state: torch.Tensor
    rule_cards: torch.Tensor
    rule_active: torch.Tensor
    event_rule: torch.Tensor
    event_kind: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.cardinality, torch.Tensor) or self.cardinality.ndim != 2:
            raise RelationalObjectError("cardinality logits must be rank two")
        batch = self.cardinality.shape[0]
        _require_float(
            "cardinality",
            self.cardinality,
            (batch, MAX_OBJECTS - MIN_OBJECTS + 1),
        )
        _require_float(
            "initial_state",
            self.initial_state,
            (batch, MAX_OBJECTS, MAX_OBJECTS),
        )
        _require_float(
            "rule_cards",
            self.rule_cards,
            (batch, MAX_RULES, MAX_OBJECTS, MAX_OBJECTS),
        )
        _require_float(
            "rule_active",
            self.rule_active,
            (batch, MAX_RULES, 2),
        )
        _require_float(
            "event_rule",
            self.event_rule,
            (batch, MAX_EVENTS, MAX_RULES),
        )
        _require_float(
            "event_kind",
            self.event_kind,
            (batch, MAX_EVENTS, EVENT_KIND_COUNT),
        )
        devices = {getattr(self, field.name).device for field in fields(self)}
        if len(devices) != 1:
            raise RelationalObjectError("object-file tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.cardinality.shape[0])

    def detached_clone(self) -> "DeletedRelationalProgram":
        return DeletedRelationalProgram(
            *(getattr(self, field.name).detach().clone() for field in fields(self))
        )

    def seal(self) -> "HardDeletedRelationalProgram":
        """Materialize the categorical private object file and discard logits."""

        cardinality = self.cardinality.argmax(-1) + MIN_OBJECTS
        positions = torch.arange(MAX_OBJECTS, device=self.cardinality.device)
        active = positions[None] < cardinality[:, None]
        negative = torch.finfo(self.initial_state.dtype).min
        initial = _nonempty_relation_map(self.initial_state, active)
        cards = _nonempty_relation_map(self.rule_cards, active)
        rule_active = self.rule_active.argmax(-1).bool()
        missing_rule = ~rule_active.any(-1)
        if bool(missing_rule.any()):
            best_rule = (
                self.rule_active[..., 1] - self.rule_active[..., 0]
            ).argmax(-1)
            fallback = F.one_hot(best_rule, MAX_RULES).bool()
            rule_active = torch.where(
                missing_rule[:, None],
                fallback,
                rule_active,
            )
        event_scores = self.event_rule.masked_fill(
            ~rule_active[:, None],
            negative,
        )
        return HardDeletedRelationalProgram(
            cardinality=cardinality.to(torch.uint8),
            initial_edges=initial.to(torch.uint8),
            rule_edges=cards.to(torch.uint8),
            rule_active=rule_active,
            event_rule=event_scores.argmax(-1).to(torch.uint8),
            event_kind=_one_stop_map(self.event_kind).to(torch.uint8),
        )


@dataclass(frozen=True, slots=True)
class DeletedRelationalQuery:
    """Late query packet compiled without program source or recurrent state."""

    position: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position, torch.Tensor)
            or self.position.ndim != 2
            or self.position.shape[1] != MAX_OBJECTS
            or not self.position.is_floating_point()
            or not bool(torch.isfinite(self.position).all())
        ):
            raise RelationalObjectError("late-query logits differ")

    @property
    def batch_size(self) -> int:
        return int(self.position.shape[0])

    def seal(self, cardinality: torch.Tensor) -> "HardDeletedRelationalQuery":
        if cardinality.shape != (self.batch_size,):
            raise RelationalObjectError("late-query cardinality differs")
        positions = torch.arange(MAX_OBJECTS, device=self.position.device)
        active = positions[None] < cardinality.long()[:, None]
        negative = torch.finfo(self.position.dtype).min
        selected = self.position.masked_fill(~active, negative).argmax(-1)
        return HardDeletedRelationalQuery(selected.to(torch.uint8))


@dataclass(frozen=True, slots=True)
class HardDeletedRelationalProgram:
    """Source-free categorical object file used by score-bearing execution."""

    cardinality: torch.Tensor
    initial_edges: torch.Tensor
    rule_edges: torch.Tensor
    rule_active: torch.Tensor
    event_rule: torch.Tensor
    event_kind: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.cardinality, torch.Tensor) or self.cardinality.ndim != 1:
            raise RelationalObjectError("hard cardinality must be rank one")
        batch = self.cardinality.shape[0]
        expected = {
            "cardinality": (batch,),
            "initial_edges": (batch, MAX_OBJECTS, MAX_OBJECTS),
            "rule_edges": (batch, MAX_RULES, MAX_OBJECTS, MAX_OBJECTS),
            "rule_active": (batch, MAX_RULES),
            "event_rule": (batch, MAX_EVENTS),
            "event_kind": (batch, MAX_EVENTS),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise RelationalObjectError(f"hard object-file field differs: {name}")
        if any(
            getattr(self, name).dtype != torch.uint8
            for name in (
                "cardinality",
                "initial_edges",
                "rule_edges",
                "event_rule",
                "event_kind",
            )
        ) or self.rule_active.dtype != torch.bool:
            raise RelationalObjectError("hard object-file dtypes differ")
        if (
            self.cardinality.numel() == 0
            or int(self.cardinality.min()) < MIN_OBJECTS
            or int(self.cardinality.max()) > MAX_OBJECTS
            or int(self.initial_edges.max()) > 1
            or int(self.rule_edges.max()) > 1
            or int(self.event_rule.max()) >= MAX_RULES
            or int(self.event_kind.max()) >= EVENT_KIND_COUNT
        ):
            raise RelationalObjectError("hard object-file value leaves its domain")
        if not bool(self.event_kind.eq(STOP_KIND).sum(-1).eq(1).all()):
            raise RelationalObjectError("hard event tape requires exactly one STOP")
        if not bool(self.rule_active.any(-1).all()):
            raise RelationalObjectError("hard object file requires an active rule")
        positions = torch.arange(MAX_OBJECTS, device=self.cardinality.device)
        active = positions[None] < self.cardinality.long()[:, None]
        active_square = active[:, :, None] & active[:, None, :]
        if bool(self.initial_edges.bool().logical_and(~active_square).any()):
            raise RelationalObjectError(
                "hard initial relation stores state outside cardinality"
            )
        if bool(
            self.rule_edges.bool()
            .logical_and(~active_square[:, None])
            .any()
        ):
            raise RelationalObjectError(
                "hard rule relation stores state outside cardinality"
            )
        initial_rows = self.initial_edges.bool().sum(-1)
        rule_rows = self.rule_edges.bool().sum(-1)
        if not bool(initial_rows[active].ge(1).all()):
            raise RelationalObjectError("active initial row has no input edge")
        expanded_active = active[:, None].expand(-1, MAX_RULES, -1)
        if not bool(rule_rows[expanded_active].ge(1).all()):
            raise RelationalObjectError("active rule row has no input edge")
        active_event_rule = self.rule_active.gather(1, self.event_rule.long())
        alive = torch.ones(batch, dtype=torch.bool, device=self.cardinality.device)
        for step in range(MAX_EVENTS):
            applying = alive & self.event_kind[:, step].eq(APPLY_KIND)
            if bool((applying & ~active_event_rule[:, step]).any()):
                raise RelationalObjectError("live event references an inactive rule")
            alive &= ~self.event_kind[:, step].eq(STOP_KIND)

    @property
    def batch_size(self) -> int:
        return int(self.cardinality.shape[0])

    @property
    def bytes_per_row(self) -> int:
        return (
            1
            + MAX_OBJECTS * MAX_OBJECTS
            + MAX_RULES * MAX_OBJECTS * MAX_OBJECTS
            + MAX_RULES
            + 2 * MAX_EVENTS
        )


@dataclass(frozen=True, slots=True)
class HardDeletedRelationalQuery:
    position: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position, torch.Tensor)
            or self.position.ndim != 1
            or self.position.dtype != torch.uint8
            or (
                self.position.numel() > 0
                and int(self.position.max()) >= MAX_OBJECTS
            )
        ):
            raise RelationalObjectError("hard late query differs")


@dataclass(frozen=True, slots=True)
class RelationalStateTransplant:
    after_step: int
    batch_permutation: torch.Tensor

    def __post_init__(self) -> None:
        if not 0 <= int(self.after_step) < MAX_EVENTS:
            raise RelationalObjectError("state transplant step differs")


@dataclass(frozen=True, slots=True)
class RelationalRollout:
    final_state: torch.Tensor
    answer_distribution: torch.Tensor
    state_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]
    halted_trajectory: tuple[torch.Tensor, ...]


def _validate_transplant(
    transplant: RelationalStateTransplant | None,
    batch: int,
    device: torch.device,
) -> torch.Tensor | None:
    if transplant is None:
        return None
    permutation = transplant.batch_permutation
    if permutation.shape != (batch,) or permutation.dtype != torch.long:
        raise RelationalObjectError("state transplant permutation differs")
    permutation = permutation.to(device)
    expected = torch.arange(batch, device=device)
    if not torch.equal(permutation.sort().values, expected):
        raise RelationalObjectError("state transplant is not a permutation")
    return permutation


def _rollout_probabilities(
    *,
    active: torch.Tensor,
    initial_state: torch.Tensor,
    rule_cards: torch.Tensor,
    event_rule: torch.Tensor,
    event_kind: torch.Tensor,
    query: torch.Tensor,
    control: Literal["normal", "reset", "freeze"],
    transplant: RelationalStateTransplant | None,
) -> RelationalRollout:
    if control not in {"normal", "reset", "freeze"}:
        raise RelationalObjectError("relational control differs")
    batch = active.shape[0]
    permutation = _validate_transplant(transplant, batch, active.device)
    selected_cards = torch.einsum("btr,brij->btij", event_rule, rule_cards)
    alive_state = initial_state
    halted_state = torch.zeros_like(initial_state)
    alive_mass = torch.ones(batch, device=active.device)
    halted_mass = torch.zeros(batch, device=active.device)
    initial = initial_state
    state_trajectory: list[torch.Tensor] = []
    alive_trajectory: list[torch.Tensor] = []
    halted_trajectory: list[torch.Tensor] = []

    for step in range(MAX_EVENTS):
        state_input = initial if control == "reset" else alive_state
        if control == "freeze":
            proposal = alive_state
        else:
            proposal = probabilistic_relation_compose(
                selected_cards[:, step],
                state_input,
            )
        apply = event_kind[:, step, APPLY_KIND, None, None]
        stop = event_kind[:, step, STOP_KIND, None, None]
        noop = event_kind[:, step, NOOP_KIND, None, None]
        stop_mass = event_kind[:, step, STOP_KIND] * alive_mass
        continuation_mass = (
            event_kind[:, step, APPLY_KIND]
            + event_kind[:, step, NOOP_KIND]
        ) * alive_mass
        next_alive = apply * proposal + noop * alive_state
        next_halted = halted_state + stop * alive_state
        alive_state, halted_state = next_alive, next_halted
        alive_mass, halted_mass = continuation_mass, halted_mass + stop_mass
        if transplant is not None and step == transplant.after_step:
            if permutation is None:
                raise AssertionError("validated transplant permutation is absent")
            alive_state = alive_state.index_select(0, permutation)
            halted_state = halted_state.index_select(0, permutation)
            alive_mass = alive_mass.index_select(0, permutation)
            halted_mass = halted_mass.index_select(0, permutation)
        state_trajectory.append(alive_state + halted_state)
        alive_trajectory.append(alive_mass)
        halted_trajectory.append(halted_mass)

    final_state = alive_state + halted_state
    answer = torch.bmm(query[:, None], final_state).squeeze(1)
    answer = answer * active
    return RelationalRollout(
        final_state=final_state,
        answer_distribution=answer,
        state_trajectory=tuple(state_trajectory),
        alive_trajectory=tuple(alive_trajectory),
        halted_trajectory=tuple(halted_trajectory),
    )


def rollout_relational_program(
    program: DeletedRelationalProgram,
    query: DeletedRelationalQuery,
    *,
    hard: bool = False,
    control: Literal["normal", "reset", "freeze"] = "normal",
    transplant: RelationalStateTransplant | None = None,
) -> RelationalRollout:
    """Differentiably execute a private relational object file."""

    if program.batch_size != query.batch_size:
        raise RelationalObjectError("program and late-query batches differ")
    _, active = _cardinality_probabilities(program.cardinality, hard=hard)
    initial = _masked_relation_probabilities(
        program.initial_state,
        active,
        hard=hard,
    )
    cards = _masked_relation_probabilities(
        program.rule_cards,
        active,
        hard=hard,
    )
    rule_active = program.rule_active.float().softmax(-1)[..., 1]
    if hard:
        selected = F.one_hot(
            rule_active.ge(0.5).long(),
            2,
        ).float()[..., 1]
        rule_active = selected + rule_active - rule_active.detach()
    event_rule_logits = program.event_rule.float()
    event_rule_logits = event_rule_logits + rule_active[:, None].clamp_min(1e-30).log()
    event_rule = event_rule_logits.softmax(-1)
    if hard:
        selected = F.one_hot(event_rule.argmax(-1), MAX_RULES).float()
        event_rule = selected + event_rule - event_rule.detach()
    event_kind = program.event_kind.float().softmax(-1)
    if hard:
        selected = F.one_hot(
            _one_stop_map(program.event_kind),
            EVENT_KIND_COUNT,
        ).float()
        event_kind = selected + event_kind - event_kind.detach()
    query_logits = query.position.float() + active.clamp_min(1e-30).log()
    query_probability = query_logits.softmax(-1)
    if hard:
        selected = F.one_hot(
            query_probability.argmax(-1),
            MAX_OBJECTS,
        ).float()
        query_probability = selected + query_probability - query_probability.detach()
    return _rollout_probabilities(
        active=active,
        initial_state=initial,
        rule_cards=cards,
        event_rule=event_rule,
        event_kind=event_kind,
        query=query_probability,
        control=control,
        transplant=transplant,
    )


@torch.no_grad()
def rollout_hard_relational_program(
    program: HardDeletedRelationalProgram,
    query: HardDeletedRelationalQuery,
    *,
    control: Literal["normal", "reset", "freeze"] = "normal",
    transplant: RelationalStateTransplant | None = None,
) -> RelationalRollout:
    """Execute only categorical object-file bytes after source destruction."""

    if program.batch_size != query.position.shape[0]:
        raise RelationalObjectError("hard program and late-query batches differ")
    if bool(query.position.long().ge(program.cardinality.long()).any()):
        raise RelationalObjectError("hard late query leaves declared cardinality")
    device = program.cardinality.device
    positions = torch.arange(MAX_OBJECTS, device=device)
    active = positions[None] < program.cardinality.long()[:, None]
    initial = program.initial_edges.float()
    cards = program.rule_edges.float()
    event_rule = F.one_hot(program.event_rule.long(), MAX_RULES).float()
    event_kind = F.one_hot(program.event_kind.long(), EVENT_KIND_COUNT).float()
    query_probability = F.one_hot(query.position.long(), MAX_OBJECTS).float()
    return _rollout_probabilities(
        active=active.float(),
        initial_state=initial,
        rule_cards=cards,
        event_rule=event_rule,
        event_kind=event_kind,
        query=query_probability,
        control=control,
        transplant=transplant,
    )


@dataclass(frozen=True, slots=True)
class RelationalCompilerEvidence:
    """Train-only pointers; never accepted by the executor."""

    pointer_logits: torch.Tensor
    declaration_carriers: torch.Tensor


class TrunkRelationalObjectCompiler(nn.Module):
    """Compile episode-local relation cards from frozen Shohin residuals.

    Occurrence routing and identity transport are deliberately separate.
    Decoder slots choose source positions. Identity carriers are gathered only
    from source values, so learned slot embeddings cannot directly encode an
    episode's opaque names.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        compiler_width: int = 384,
        compiler_heads: int = 8,
        encoder_layers: int = 5,
        encoder_feedforward: int = 1408,
        decoder_layers: int = 2,
        decoder_feedforward: int = 1024,
        identity_width: int = 128,
        early_layer: int = 19,
        late_layer: int = 29,
        padding_id: int = 1,
    ) -> None:
        super().__init__()
        if identity_width < 1:
            raise RelationalObjectError("identity width must be positive")
        self.backbone = TrunkCausalCTAACompiler(
            model,
            compiler_width=compiler_width,
            heads=compiler_heads,
            encoder_layers=encoder_layers,
            encoder_feedforward=encoder_feedforward,
            decoder_layers=decoder_layers,
            decoder_feedforward=decoder_feedforward,
            early_layer=early_layer,
            late_layer=late_layer,
            padding_id=padding_id,
        )
        self.compiler_width = int(compiler_width)
        self.identity_width = int(identity_width)

        self.control_count = 1
        self.declaration_start = self.control_count
        self.initial_start = self.declaration_start + MAX_OBJECTS
        self.rule_start = self.initial_start + MAX_OBJECTS
        self.rule_stride = 1 + 2 * MAX_RELATION_EDGES
        self.event_start = self.rule_start + MAX_RULES * self.rule_stride
        self.object_slot_count = self.event_start + MAX_EVENTS

        self.object_queries = nn.Parameter(
            torch.empty(self.object_slot_count, compiler_width)
        )
        self.late_query = nn.Parameter(torch.empty(1, compiler_width))
        self.pointer_query = nn.Linear(compiler_width, compiler_width, bias=False)
        self.pointer_key = nn.Linear(compiler_width, compiler_width, bias=False)
        self.identity_value = nn.Linear(
            compiler_width,
            identity_width,
            bias=False,
        )
        self.identity_norm = nn.LayerNorm(identity_width)
        self.cardinality_head = nn.Linear(
            compiler_width,
            MAX_OBJECTS - MIN_OBJECTS + 1,
        )
        self.rule_active_head = nn.Linear(compiler_width, 2)
        self.edge_active_head = nn.Linear(compiler_width, 2)
        self.event_kind_head = nn.Linear(compiler_width, EVENT_KIND_COUNT)
        self.late_query_head = nn.Linear(compiler_width, MAX_OBJECTS)
        self.identity_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.object_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.late_query, mean=0.0, std=0.02)

    def _slots_and_carriers(
        self,
        bundle: TrunkResidualBundle,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RelationalCompilerEvidence]:
        memory, valid = self.backbone.memory_from_residuals(
            bundle,
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
        )
        slots = self.backbone._decode(memory, valid, self.object_queries)
        queries = self.pointer_query(slots)
        keys = self.pointer_key(memory)
        pointer_logits = torch.einsum("bsw,blw->bsl", queries, keys)
        pointer_logits = pointer_logits / math.sqrt(self.compiler_width)
        pointer_logits = pointer_logits.masked_fill(
            ~valid[:, None],
            torch.finfo(pointer_logits.dtype).min,
        ).float()
        weights = pointer_logits.softmax(-1).to(memory.dtype)
        values = self.identity_value(memory)
        carriers = self.identity_norm(
            torch.einsum("bsl,blw->bsw", weights, values)
        )
        declaration = carriers[
            :,
            self.declaration_start : self.initial_start,
        ]
        return slots, carriers, RelationalCompilerEvidence(
            pointer_logits=pointer_logits,
            declaration_carriers=declaration,
        )

    def _similarity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = F.normalize(left.float(), dim=-1)
        right = F.normalize(right.float(), dim=-1)
        scale = self.identity_log_scale.exp().clamp(max=100.0)
        return scale * torch.einsum("...iw,...jw->...ij", left, right)

    def compile_program_from_residuals(
        self,
        bundle: TrunkResidualBundle,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
        return_evidence: bool = False,
    ) -> DeletedRelationalProgram | tuple[
        DeletedRelationalProgram,
        RelationalCompilerEvidence,
    ]:
        slots, carriers, evidence = self._slots_and_carriers(
            bundle,
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
        )
        declaration = carriers[
            :,
            self.declaration_start : self.initial_start,
        ]
        initial = carriers[:, self.initial_start : self.rule_start]
        initial_logits = self._similarity(initial, declaration)

        rule_slots = []
        rule_cards = []
        for rule in range(MAX_RULES):
            start = self.rule_start + rule * self.rule_stride
            rule_slots.append(slots[:, start])
            source_slots = slots[
                :,
                start + 1 : start + 1 + MAX_RELATION_EDGES,
            ]
            sources = carriers[
                :,
                start + 1 : start + 1 + MAX_RELATION_EDGES,
            ]
            destinations = carriers[
                :,
                start + 1 + MAX_RELATION_EDGES :
                start + 1 + 2 * MAX_RELATION_EDGES,
            ]
            source_probability = self._similarity(
                sources,
                declaration,
            ).softmax(-1)
            destination_probability = self._similarity(
                destinations,
                declaration,
            ).softmax(-1)
            edge_probability = self.edge_active_head(
                source_slots,
            ).float().softmax(-1)[..., 1]
            path_probability = (
                edge_probability[:, :, None, None]
                * destination_probability[:, :, :, None]
                * source_probability[:, :, None, :]
            )
            card_probability = 1.0 - (1.0 - path_probability).prod(dim=1)
            card_probability = card_probability.clamp(1e-6, 1.0 - 1e-6)
            rule_cards.append(torch.logit(card_probability))
        rule_slots_tensor = torch.stack(rule_slots, dim=1)
        event_slots = slots[:, self.event_start :]
        event_carriers = carriers[:, self.event_start :]
        rule_opcode_carriers = torch.stack(
            [
                carriers[:, self.rule_start + rule * self.rule_stride]
                for rule in range(MAX_RULES)
            ],
            dim=1,
        )
        event_rule = self._similarity(event_carriers, rule_opcode_carriers)
        output = DeletedRelationalProgram(
            cardinality=self.cardinality_head(slots[:, 0]).float(),
            initial_state=initial_logits.float(),
            rule_cards=torch.stack(rule_cards, dim=1).float(),
            rule_active=self.rule_active_head(rule_slots_tensor).float(),
            event_rule=event_rule.float(),
            event_kind=self.event_kind_head(event_slots).float(),
        )
        return (output, evidence) if return_evidence else output

    def compile_program(
        self,
        ids: torch.Tensor,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
        return_evidence: bool = False,
    ) -> DeletedRelationalProgram | tuple[
        DeletedRelationalProgram,
        RelationalCompilerEvidence,
    ]:
        return self.compile_program_from_residuals(
            self.backbone.encode_source(ids),
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
            return_evidence=return_evidence,
        )

    def compile_late_query(self, ids: torch.Tensor) -> DeletedRelationalQuery:
        bundle = self.backbone.encode_source(ids)
        memory, valid = self.backbone.memory_from_residuals(bundle)
        decoded = self.backbone._decode(memory, valid, self.late_query)
        return DeletedRelationalQuery(
            self.late_query_head(decoded[:, 0]).float()
        )

    @torch.no_grad()
    def compile_and_seal_program(
        self,
        ids: torch.Tensor,
    ) -> HardDeletedRelationalProgram:
        output = self.compile_program(ids)
        if not isinstance(output, DeletedRelationalProgram):
            raise AssertionError("compiler unexpectedly returned evidence")
        return output.seal()

    def parameter_report(self) -> dict[str, int]:
        base_ids = {
            id(parameter) for parameter in self.backbone.model.parameters()
        }
        unique = {id(parameter): parameter for parameter in self.parameters()}
        base = sum(
            parameter.numel()
            for identifier, parameter in unique.items()
            if identifier in base_ids
        )
        complete = sum(parameter.numel() for parameter in unique.values())
        added = complete - base
        report = {
            "base": int(base),
            "added": int(added),
            "complete_system": int(complete),
            "trainable": int(
                sum(
                    parameter.numel()
                    for parameter in unique.values()
                    if parameter.requires_grad
                )
            ),
            "strict_cap": STRICT_PARAMETER_CAP,
            "headroom": int(STRICT_PARAMETER_CAP - complete),
        }
        if complete >= STRICT_PARAMETER_CAP:
            raise RelationalObjectError("relational object machine reaches 200M")
        return report


class GeneralRelationalObjectMachine(nn.Module):
    """Frozen-trunk compiler plus one shared source-deleted executor."""

    def __init__(self, base_model: nn.Module, **compiler_kwargs: int) -> None:
        super().__init__()
        self.compiler = TrunkRelationalObjectCompiler(
            base_model,
            **compiler_kwargs,
        )

    def forward(
        self,
        program_ids: torch.Tensor,
        query_ids: torch.Tensor,
        *,
        hard: bool = False,
    ) -> RelationalRollout:
        program = self.compiler.compile_program(program_ids)
        if not isinstance(program, DeletedRelationalProgram):
            raise AssertionError("compiler unexpectedly returned evidence")
        query = self.compiler.compile_late_query(query_ids)
        return rollout_relational_program(program, query, hard=hard)

    @torch.no_grad()
    def source_deleted_rollout(
        self,
        program_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> RelationalRollout:
        program = self.compiler.compile_and_seal_program(program_ids)
        query_logits = self.compiler.compile_late_query(query_ids)
        query = query_logits.seal(program.cardinality)
        del program_ids, query_ids, query_logits
        return rollout_hard_relational_program(program, query)
