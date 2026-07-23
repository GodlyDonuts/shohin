"""Raw-token mechanics board for episode-specific action binding.

The board is an offline identifiability and anti-shortcut test for the EPISODE
proposal. It is not a neural runtime and not a reasoning claim. Model-visible
packets contain only ordinary integer token IDs plus an attention mask.
Bindings, physical operators, targets, cyclic-orbit membership, and oracle
products remain assessor-side.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
import json
import random
from typing import Iterable, Mapping, Sequence


PAD = 0
BOS = 1
DEMO = 2
YIELDS = 3
SEP = 4
QUERY = 5
THEN = 6
ANSWER = 7
EOS = 8
ABSTAIN = 9
ERASED_ACTION = 10

RESERVED_TOKENS = frozenset(
    {PAD, BOS, DEMO, YIELDS, SEP, QUERY, THEN, ANSWER, EOS, ABSTAIN, ERASED_ACTION}
)
STATE_COUNT = 8
ACTION_COUNT = 3
MAX_QUERY_DEPTH = 6

Permutation = tuple[int, ...]
Transition = tuple[int, int, int]


class EpisodeBoardError(ValueError):
    """Base class for fail-closed mechanics errors."""


class MalformedEpisodeError(EpisodeBoardError):
    """A raw token stream violates the frozen grammar or typing contract."""


class InconsistentEpisodeError(EpisodeBoardError):
    """No physical action binding is consistent with the demonstrations."""


class GenerationError(EpisodeBoardError):
    """A deterministic board instance could not satisfy the frozen gates."""


@dataclass(frozen=True)
class ModelPacket:
    """The complete model-visible source packet."""

    tokens: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True)
class ParsedEpisode:
    """Typed parse used only by independent offline mechanics."""

    demonstrations: tuple[Transition, ...]
    query_start: int
    query_actions: tuple[int, ...]


@dataclass(frozen=True)
class AssessorSystem:
    """Offline system identity; never part of ``model_packet_payload``."""

    physical_operators: tuple[Permutation, ...]
    state_tokens: tuple[int, ...]
    action_tokens: tuple[int, ...]
    demonstration_order: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EpisodeCase:
    """One scored case with an assessor-side target."""

    packet: ModelPacket
    target_token: int
    binding_shift: int


@dataclass(frozen=True)
class CyclicBindingGroup:
    """Three action-recoding variants sharing one action-agnostic view."""

    system: AssessorSystem
    query_start_state: int
    query_action_indices: tuple[int, ...]
    variants: tuple[EpisodeCase, ...]
    group_digest: str


@dataclass(frozen=True)
class OrderTwin:
    """Two queries with the same action multiset and different order."""

    system: AssessorSystem
    binding_shift: int
    forward: EpisodeCase
    reverse: EpisodeCase


@dataclass(frozen=True)
class CyclicOrderCluster:
    """Two three-way recoding orbits whose query words share one token bag."""

    primary: CyclicBindingGroup
    reordered: CyclicBindingGroup


def model_packet_payload(packet: ModelPacket) -> dict[str, tuple[int, ...]]:
    """Return the only fields a model may receive."""

    validate_model_packet(packet)
    return {"tokens": packet.tokens, "attention_mask": packet.attention_mask}


def validate_model_packet(packet: ModelPacket) -> None:
    if not packet.tokens:
        raise MalformedEpisodeError("token stream must not be empty")
    if len(packet.tokens) != len(packet.attention_mask):
        raise MalformedEpisodeError("tokens and attention mask must have equal length")
    if any(not isinstance(token, int) or token < 0 for token in packet.tokens):
        raise MalformedEpisodeError("all token IDs must be nonnegative integers")
    if any(value not in (0, 1) for value in packet.attention_mask):
        raise MalformedEpisodeError("attention mask must be binary")
    seen_padding = False
    for token, active in zip(packet.tokens, packet.attention_mask, strict=True):
        if active == 0:
            seen_padding = True
            if token != PAD:
                raise MalformedEpisodeError("inactive positions must contain PAD")
        elif seen_padding:
            raise MalformedEpisodeError("active token follows right padding")
    if not any(packet.attention_mask):
        raise MalformedEpisodeError("packet contains no active source tokens")


def _active_tokens(packet: ModelPacket) -> tuple[int, ...]:
    validate_model_packet(packet)
    return tuple(
        token
        for token, active in zip(packet.tokens, packet.attention_mask, strict=True)
        if active
    )


def parse_episode(packet: ModelPacket) -> ParsedEpisode:
    """Parse one raw stream without consulting any assessor metadata."""

    tokens = _active_tokens(packet)
    if len(tokens) < 10 or tokens[0] != BOS:
        raise MalformedEpisodeError("episode must start with BOS")

    cursor = 1
    demonstrations: list[Transition] = []
    transition_table: dict[tuple[int, int], int] = {}
    while cursor < len(tokens) and tokens[cursor] == DEMO:
        if cursor + 5 >= len(tokens):
            raise MalformedEpisodeError("truncated demonstration")
        _, source, action, yields, target, separator = tokens[cursor : cursor + 6]
        if yields != YIELDS or separator != SEP:
            raise MalformedEpisodeError("malformed demonstration markers")
        if source in RESERVED_TOKENS or action in RESERVED_TOKENS:
            raise MalformedEpisodeError("reserved token used as a source or action")
        if target in RESERVED_TOKENS:
            raise MalformedEpisodeError("reserved token used as a target")
        key = (source, action)
        if key in transition_table:
            qualifier = (
                "duplicate" if transition_table[key] == target else "conflicting"
            )
            raise MalformedEpisodeError(f"{qualifier} transition for {key}")
        transition_table[key] = target
        demonstrations.append((source, action, target))
        cursor += 6

    if not demonstrations:
        raise MalformedEpisodeError("episode contains no demonstrations")
    if cursor + 4 >= len(tokens) or tokens[cursor] != QUERY:
        raise MalformedEpisodeError("missing QUERY clause")

    query_start = tokens[cursor + 1]
    if query_start in RESERVED_TOKENS:
        raise MalformedEpisodeError("reserved token used as query start")
    cursor += 2
    query_actions: list[int] = []
    expect_action = True
    while cursor < len(tokens):
        token = tokens[cursor]
        if expect_action:
            if token in RESERVED_TOKENS:
                raise MalformedEpisodeError("missing query action")
            query_actions.append(token)
            expect_action = False
            cursor += 1
            continue
        if token == THEN:
            expect_action = True
            cursor += 1
            continue
        if token == ANSWER:
            cursor += 1
            break
        raise MalformedEpisodeError("unexpected token in query word")

    if expect_action or not query_actions:
        raise MalformedEpisodeError("query word is empty or truncated")
    if cursor >= len(tokens) or tokens[cursor] != EOS:
        raise MalformedEpisodeError("episode must end with ANSWER EOS")
    if cursor + 1 != len(tokens):
        raise MalformedEpisodeError("trailing active tokens after EOS")
    if len(query_actions) > MAX_QUERY_DEPTH:
        raise MalformedEpisodeError("query exceeds frozen depth")

    state_tokens = {
        token for source, _, target in demonstrations for token in (source, target)
    }
    demonstrated_action_tokens = {action for _, action, _ in demonstrations}
    action_tokens = demonstrated_action_tokens | set(query_actions)
    if query_start not in state_tokens:
        raise MalformedEpisodeError("query references an unknown state token")
    if len(state_tokens) != STATE_COUNT:
        raise MalformedEpisodeError(
            f"episode must expose exactly {STATE_COUNT} state tokens"
        )
    if not action_tokens or len(action_tokens) > ACTION_COUNT:
        raise MalformedEpisodeError(
            "query introduces an unknown action token beyond the frozen domain"
        )
    if state_tokens & action_tokens:
        raise MalformedEpisodeError("state and action token types overlap")
    if any(target not in state_tokens for _, _, target in demonstrations):
        raise MalformedEpisodeError("transition target is outside the state domain")

    return ParsedEpisode(
        demonstrations=tuple(demonstrations),
        query_start=query_start,
        query_actions=tuple(query_actions),
    )


def visible_table_oracle(packet: ModelPacket) -> int:
    """Execute the ordered query using only parsed visible demonstrations."""

    parsed = parse_episode(packet)
    table = {
        (source, action): target for source, action, target in parsed.demonstrations
    }
    state = parsed.query_start
    for action in parsed.query_actions:
        target = table.get((state, action))
        if target is None:
            return ABSTAIN
        state = target
    return state


def binding_enumerator_oracle(packet: ModelPacket, system: AssessorSystem) -> int:
    """Enumerate all physical bindings consistent with every demonstration."""

    parsed = parse_episode(packet)
    _validate_assessor_system(system)
    state_by_token = {token: state for state, token in enumerate(system.state_tokens)}
    action_index = {token: index for index, token in enumerate(system.action_tokens)}
    try:
        demonstrations = tuple(
            (
                state_by_token[source],
                action_index[action],
                state_by_token[target],
            )
            for source, action, target in parsed.demonstrations
        )
        start_state = state_by_token[parsed.query_start]
        query_indices = tuple(action_index[action] for action in parsed.query_actions)
    except KeyError as exc:
        raise MalformedEpisodeError(
            "packet contains a nonce absent from the assessor system"
        ) from exc

    survivors: list[tuple[int, ...]] = []
    for binding in permutations(range(ACTION_COUNT)):
        if all(
            system.physical_operators[binding[action]][source] == target
            for source, action, target in demonstrations
        ):
            survivors.append(binding)
    if not survivors:
        raise InconsistentEpisodeError(
            "no physical action binding satisfies the demonstrations"
        )

    outcomes: set[int] = set()
    for binding in survivors:
        state = start_state
        for action in query_indices:
            state = system.physical_operators[binding[action]][state]
        outcomes.add(system.state_tokens[state])
    return outcomes.pop() if len(outcomes) == 1 else ABSTAIN


def pad_packet(packet: ModelPacket, length: int) -> ModelPacket:
    """Right-pad a packet without changing its active source."""

    validate_model_packet(packet)
    if length < len(packet.tokens):
        raise MalformedEpisodeError("requested padding length truncates the packet")
    padding = length - len(packet.tokens)
    return ModelPacket(
        tokens=packet.tokens + (PAD,) * padding,
        attention_mask=packet.attention_mask + (0,) * padding,
    )


def raw_token_histogram(packet: ModelPacket) -> Counter[int]:
    """Count active raw token IDs."""

    return Counter(_active_tokens(packet))


def erase_demonstration_actions(packet: ModelPacket) -> tuple[int, ...]:
    """Erase only action identity in demonstrations, preserving query tokens."""

    tokens = list(_active_tokens(packet))
    cursor = 1
    while cursor < len(tokens) and tokens[cursor] == DEMO:
        tokens[cursor + 2] = ERASED_ACTION
        cursor += 6
    return tuple(tokens)


def split_world_and_query(
    packet: ModelPacket,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the immutable world prefix and late-materialized query suffix."""

    tokens = _active_tokens(packet)
    query_positions = [index for index, token in enumerate(tokens) if token == QUERY]
    if len(query_positions) != 1:
        raise MalformedEpisodeError("episode must contain exactly one QUERY marker")
    boundary = query_positions[0]
    world = tokens[:boundary]
    query = tokens[boundary:]
    if not world or world[0] != BOS or not query:
        raise MalformedEpisodeError("world/query boundary is malformed")
    return world, query


def world_commitment(packet: ModelPacket) -> str:
    """Hash the world before the query is available to a future compiler."""

    world, _ = split_world_and_query(packet)
    payload = json.dumps(world, separators=(",", ":")).encode("ascii")
    return sha256(payload).hexdigest()


def action_agnostic_baseline(packet: ModelPacket) -> int:
    """Deterministic baseline constrained to the action-erased view."""

    parsed = parse_episode(packet)
    states = sorted(
        {
            token
            for source, _, target in parsed.demonstrations
            for token in (source, target)
        }
    )
    payload = json.dumps(
        erase_demonstration_actions(packet), separators=(",", ":")
    ).encode("ascii")
    return states[int.from_bytes(sha256(payload).digest()[:8], "big") % len(states)]


def all_actions_union_baseline(packet: ModelPacket) -> int:
    """Apply the unlabeled union relation and choose one deterministic state."""

    parsed = parse_episode(packet)
    union: dict[int, set[int]] = {}
    for source, _, target in parsed.demonstrations:
        union.setdefault(source, set()).add(target)
    frontier = {parsed.query_start}
    for _ in parsed.query_actions:
        frontier = {target for source in frontier for target in union.get(source, ())}
        if not frontier:
            return ABSTAIN
    return min(frontier)


def query_order_bagging_baseline(packet: ModelPacket) -> int:
    """Use the visible action table but replace the query by sorted action IDs."""

    parsed = parse_episode(packet)
    table = {
        (source, action): target for source, action, target in parsed.demonstrations
    }
    state = parsed.query_start
    for action in sorted(parsed.query_actions):
        target = table.get((state, action))
        if target is None:
            return ABSTAIN
        state = target
    return state


def make_underidentified(
    packet: ModelPacket,
    removed_action_tokens: Iterable[int],
) -> ModelPacket:
    """Remove all demonstrations for selected actions while preserving grammar."""

    removed = set(removed_action_tokens)
    parsed = parse_episode(packet)
    kept = [
        transition
        for transition in parsed.demonstrations
        if transition[1] not in removed
    ]
    if not kept:
        raise MalformedEpisodeError("underidentified packet needs one visible action")
    return _render_packet(
        kept,
        query_start=parsed.query_start,
        query_actions=parsed.query_actions,
    )


def permute_demonstrations(packet: ModelPacket, order: Sequence[int]) -> ModelPacket:
    """Reorder complete demonstration clauses without changing the query."""

    parsed = parse_episode(packet)
    if sorted(order) != list(range(len(parsed.demonstrations))):
        raise MalformedEpisodeError("demonstration order is not a permutation")
    demonstrations = [parsed.demonstrations[index] for index in order]
    return _render_packet(
        demonstrations,
        query_start=parsed.query_start,
        query_actions=parsed.query_actions,
    )


def rename_nonces(
    packet: ModelPacket,
    nonce_map: Mapping[int, int],
) -> ModelPacket:
    """Apply a typed opaque-nonce renaming to a raw stream."""

    if any(source in RESERVED_TOKENS for source in nonce_map):
        raise MalformedEpisodeError("reserved source token in nonce renaming")
    if any(target in RESERVED_TOKENS for target in nonce_map.values()):
        raise MalformedEpisodeError("reserved target token in nonce renaming")
    if len(set(nonce_map.values())) != len(nonce_map):
        raise MalformedEpisodeError("nonce renaming must be injective")
    active = _active_tokens(packet)
    renamed = tuple(nonce_map.get(token, token) for token in active)
    result = ModelPacket(renamed, (1,) * len(renamed))
    parse_episode(result)
    return result


def rename_assessor_system(
    system: AssessorSystem,
    nonce_map: Mapping[int, int],
) -> AssessorSystem:
    """Apply the same nonce renaming to offline state/action receipts."""

    renamed = AssessorSystem(
        physical_operators=system.physical_operators,
        state_tokens=tuple(
            nonce_map.get(token, token) for token in system.state_tokens
        ),
        action_tokens=tuple(
            nonce_map.get(token, token) for token in system.action_tokens
        ),
        demonstration_order=system.demonstration_order,
    )
    _validate_assessor_system(renamed)
    return renamed


def generate_cyclic_binding_group(
    seed: int,
    *,
    query_depth: int | None = None,
) -> CyclicBindingGroup:
    """Generate one deterministic adversarial cyclic binding triple."""

    rng = random.Random(seed)
    depth = query_depth if query_depth is not None else 1 + seed % MAX_QUERY_DEPTH
    if depth < 1 or depth > MAX_QUERY_DEPTH:
        raise GenerationError(f"query depth must be in [1, {MAX_QUERY_DEPTH}]")

    nonce_pool = rng.sample(
        range(1_000, 32_768),
        STATE_COUNT + ACTION_COUNT,
    )
    state_tokens = tuple(nonce_pool[:STATE_COUNT])
    action_tokens = tuple(nonce_pool[STATE_COUNT:])
    operator_pool = list(_affine_permutations())

    for _ in range(2_000):
        physical_operators = tuple(rng.sample(operator_pool, ACTION_COUNT))
        if not _has_noncommuting_pair(physical_operators):
            continue
        query_start = rng.randrange(STATE_COUNT)
        query_indices = tuple(rng.randrange(ACTION_COUNT) for _ in range(depth))
        outcomes = tuple(
            _execute_latent(
                physical_operators,
                shift,
                query_start,
                query_indices,
            )
            for shift in range(ACTION_COUNT)
        )
        if len(set(outcomes)) != ACTION_COUNT:
            continue

        demonstration_order = list(product(range(ACTION_COUNT), range(STATE_COUNT)))
        rng.shuffle(demonstration_order)
        system = AssessorSystem(
            physical_operators=physical_operators,
            state_tokens=state_tokens,
            action_tokens=action_tokens,
            demonstration_order=tuple(demonstration_order),
        )
        variants = tuple(
            _make_case(
                system,
                binding_shift=shift,
                query_start_state=query_start,
                query_action_indices=query_indices,
            )
            for shift in range(ACTION_COUNT)
        )
        group = CyclicBindingGroup(
            system=system,
            query_start_state=query_start,
            query_action_indices=query_indices,
            variants=variants,
            group_digest=_group_digest(variants),
        )
        validate_cyclic_group(group)
        return group
    raise GenerationError("failed to find a separating cyclic binding triple")


def generate_order_twin(
    system: AssessorSystem,
    *,
    binding_shift: int = 0,
) -> OrderTwin:
    """Find a deterministic noncommuting two-action query pair."""

    _validate_assessor_system(system)
    for start_state in range(STATE_COUNT):
        for left, right in permutations(range(ACTION_COUNT), 2):
            forward = _make_case(
                system,
                binding_shift=binding_shift,
                query_start_state=start_state,
                query_action_indices=(left, right),
            )
            reverse = _make_case(
                system,
                binding_shift=binding_shift,
                query_start_state=start_state,
                query_action_indices=(right, left),
            )
            if forward.target_token == reverse.target_token:
                continue
            twin = OrderTwin(
                system=system,
                binding_shift=binding_shift,
                forward=forward,
                reverse=reverse,
            )
            validate_order_twin(twin)
            return twin
    raise GenerationError("system contains no visible noncommuting order twin")


def generate_cyclic_order_cluster(
    seed: int,
    *,
    query_depth: int = 4,
) -> CyclicOrderCluster:
    """Generate one six-case cluster requiring binding and ordered execution."""

    if query_depth < 2 or query_depth > MAX_QUERY_DEPTH:
        raise GenerationError(f"cluster query depth must be in [2, {MAX_QUERY_DEPTH}]")
    for attempt in range(200):
        primary = generate_cyclic_binding_group(
            seed + attempt * 1_000_003,
            query_depth=query_depth,
        )
        alternatives = sorted(set(permutations(primary.query_action_indices)))
        for query_indices in alternatives:
            if query_indices == primary.query_action_indices:
                continue
            reordered_variants = tuple(
                _make_case(
                    primary.system,
                    binding_shift=shift,
                    query_start_state=primary.query_start_state,
                    query_action_indices=query_indices,
                )
                for shift in range(ACTION_COUNT)
            )
            if len({case.target_token for case in reordered_variants}) != ACTION_COUNT:
                continue
            if any(
                left.target_token == right.target_token
                for left, right in zip(
                    primary.variants,
                    reordered_variants,
                    strict=True,
                )
            ):
                continue
            reordered = CyclicBindingGroup(
                system=primary.system,
                query_start_state=primary.query_start_state,
                query_action_indices=tuple(query_indices),
                variants=reordered_variants,
                group_digest=_group_digest(reordered_variants),
            )
            cluster = CyclicOrderCluster(primary=primary, reordered=reordered)
            validate_cyclic_order_cluster(cluster)
            return cluster
    raise GenerationError("failed to generate a separating six-case cluster")


def validate_cyclic_group(group: CyclicBindingGroup) -> None:
    """Check all frozen anti-shortcut invariants for one triple."""

    if len(group.variants) != ACTION_COUNT:
        raise GenerationError("cyclic group must contain exactly three variants")
    targets = {case.target_token for case in group.variants}
    if len(targets) != ACTION_COUNT:
        raise GenerationError("cyclic group targets are not all distinct")

    reference_histogram = raw_token_histogram(group.variants[0].packet)
    reference_erased = erase_demonstration_actions(group.variants[0].packet)
    for expected_shift, case in enumerate(group.variants):
        if case.binding_shift != expected_shift:
            raise GenerationError("cyclic variants are not in shift order")
        if raw_token_histogram(case.packet) != reference_histogram:
            raise GenerationError("cyclic variants have different token histograms")
        if erase_demonstration_actions(case.packet) != reference_erased:
            raise GenerationError("action-erased transition streams differ")
        visible = visible_table_oracle(case.packet)
        enumerated = binding_enumerator_oracle(case.packet, group.system)
        if visible != case.target_token or enumerated != case.target_token:
            raise GenerationError("independent oracles disagree with target")

    for baseline in (action_agnostic_baseline, all_actions_union_baseline):
        answers = [baseline(case.packet) for case in group.variants]
        if len(set(answers)) != 1:
            raise GenerationError("action-agnostic baseline varies within a triple")
        if (
            sum(
                answer == case.target_token
                for answer, case in zip(answers, group.variants, strict=True)
            )
            > 1
        ):
            raise GenerationError("action-agnostic baseline exceeds one-third")


def validate_cyclic_order_cluster(cluster: CyclicOrderCluster) -> None:
    """Validate a six-case statistical family as one indivisible unit."""

    validate_cyclic_group(cluster.primary)
    validate_cyclic_group(cluster.reordered)
    if cluster.primary.system != cluster.reordered.system:
        raise GenerationError("cluster orbits use different physical systems")
    if cluster.primary.query_start_state != cluster.reordered.query_start_state:
        raise GenerationError("cluster orbits use different query starts")
    if Counter(cluster.primary.query_action_indices) != Counter(
        cluster.reordered.query_action_indices
    ):
        raise GenerationError("cluster query words have different action bags")
    if cluster.primary.query_action_indices == cluster.reordered.query_action_indices:
        raise GenerationError("cluster query words have identical order")
    for left, right in zip(
        cluster.primary.variants,
        cluster.reordered.variants,
        strict=True,
    ):
        if world_commitment(left.packet) != world_commitment(right.packet):
            raise GenerationError("cluster query pair has different committed worlds")
        if left.target_token == right.target_token:
            raise GenerationError("cluster query order does not change the answer")
        bagged = (
            query_order_bagging_baseline(left.packet),
            query_order_bagging_baseline(right.packet),
        )
        if bagged[0] != bagged[1]:
            raise GenerationError("query-order-bagging control varies within a pair")
        if (
            sum(
                answer == case.target_token
                for answer, case in zip(bagged, (left, right), strict=True)
            )
            > 1
        ):
            raise GenerationError("query-order-bagging control exceeds one-half")


def validate_order_twin(twin: OrderTwin) -> None:
    """Check the frozen order-bagging collision."""

    left = parse_episode(twin.forward.packet)
    right = parse_episode(twin.reverse.packet)
    if left.demonstrations != right.demonstrations:
        raise GenerationError("order twins have different demonstrations")
    if left.query_start != right.query_start:
        raise GenerationError("order twins have different start states")
    if world_commitment(twin.forward.packet) != world_commitment(twin.reverse.packet):
        raise GenerationError("order twins have different committed worlds")
    if Counter(left.query_actions) != Counter(right.query_actions):
        raise GenerationError("order twins have different action histograms")
    if twin.forward.target_token == twin.reverse.target_token:
        raise GenerationError("order twins have the same target")
    for case in (twin.forward, twin.reverse):
        if visible_table_oracle(case.packet) != case.target_token:
            raise GenerationError("visible oracle fails an order twin")
        if binding_enumerator_oracle(case.packet, twin.system) != case.target_token:
            raise GenerationError("binding oracle fails an order twin")


def _make_case(
    system: AssessorSystem,
    *,
    binding_shift: int,
    query_start_state: int,
    query_action_indices: Sequence[int],
) -> EpisodeCase:
    _validate_assessor_system(system)
    if binding_shift not in range(ACTION_COUNT):
        raise GenerationError("binding shift is outside the cyclic group")
    demonstrations: list[Transition] = []
    for physical_operator, source_state in system.demonstration_order:
        visible_action = (physical_operator - binding_shift) % ACTION_COUNT
        target_state = system.physical_operators[physical_operator][source_state]
        demonstrations.append(
            (
                system.state_tokens[source_state],
                system.action_tokens[visible_action],
                system.state_tokens[target_state],
            )
        )
    packet = _render_packet(
        demonstrations,
        query_start=system.state_tokens[query_start_state],
        query_actions=tuple(
            system.action_tokens[index] for index in query_action_indices
        ),
    )
    target_state = _execute_latent(
        system.physical_operators,
        binding_shift,
        query_start_state,
        query_action_indices,
    )
    return EpisodeCase(
        packet=packet,
        target_token=system.state_tokens[target_state],
        binding_shift=binding_shift,
    )


def _render_packet(
    demonstrations: Sequence[Transition],
    *,
    query_start: int,
    query_actions: Sequence[int],
) -> ModelPacket:
    tokens: list[int] = [BOS]
    for source, action, target in demonstrations:
        tokens.extend((DEMO, source, action, YIELDS, target, SEP))
    tokens.extend((QUERY, query_start))
    for index, action in enumerate(query_actions):
        if index:
            tokens.append(THEN)
        tokens.append(action)
    tokens.extend((ANSWER, EOS))
    packet = ModelPacket(tokens=tuple(tokens), attention_mask=(1,) * len(tokens))
    parse_episode(packet)
    return packet


def _execute_latent(
    physical_operators: Sequence[Permutation],
    binding_shift: int,
    start_state: int,
    query_action_indices: Sequence[int],
) -> int:
    state = start_state
    for visible_action in query_action_indices:
        physical_operator = (visible_action + binding_shift) % ACTION_COUNT
        state = physical_operators[physical_operator][state]
    return state


def _validate_assessor_system(system: AssessorSystem) -> None:
    if len(system.physical_operators) != ACTION_COUNT:
        raise GenerationError("assessor system must have three physical operators")
    if len(set(system.physical_operators)) != ACTION_COUNT:
        raise GenerationError("physical operators must be distinct")
    expected_domain = tuple(range(STATE_COUNT))
    if any(
        tuple(sorted(operator)) != expected_domain
        for operator in system.physical_operators
    ):
        raise GenerationError("physical operator is not a state permutation")
    if (
        len(system.state_tokens) != STATE_COUNT
        or len(set(system.state_tokens)) != STATE_COUNT
    ):
        raise GenerationError("state nonce domain is malformed")
    if (
        len(system.action_tokens) != ACTION_COUNT
        or len(set(system.action_tokens)) != ACTION_COUNT
    ):
        raise GenerationError("action nonce domain is malformed")
    if set(system.state_tokens) & set(system.action_tokens):
        raise GenerationError("state and action nonce domains overlap")
    if (set(system.state_tokens) | set(system.action_tokens)) & RESERVED_TOKENS:
        raise GenerationError("assessor nonces overlap reserved tokens")
    expected_order = set(product(range(ACTION_COUNT), range(STATE_COUNT)))
    if set(system.demonstration_order) != expected_order:
        raise GenerationError("demonstration order is not a complete transition table")
    if len(system.demonstration_order) != len(expected_order):
        raise GenerationError("demonstration order contains duplicates")


def _group_digest(variants: Sequence[EpisodeCase]) -> str:
    payload = [
        {
            "tokens": case.packet.tokens,
            "attention_mask": case.packet.attention_mask,
            "target": case.target_token,
            "shift": case.binding_shift,
        }
        for case in variants
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _has_noncommuting_pair(operators: Sequence[Permutation]) -> bool:
    for left, right in permutations(operators, 2):
        if any(
            left[right[state]] != right[left[state]] for state in range(STATE_COUNT)
        ):
            return True
    return False


@lru_cache(maxsize=1)
def _affine_permutations() -> tuple[Permutation, ...]:
    """Enumerate all invertible affine maps on three binary variables."""

    operators: set[Permutation] = set()
    for matrix_code in range(1 << 9):
        linear = tuple(
            _apply_binary_matrix(matrix_code, state) for state in range(STATE_COUNT)
        )
        if len(set(linear)) != STATE_COUNT:
            continue
        for offset in range(STATE_COUNT):
            operators.add(tuple(value ^ offset for value in linear))
    identity = tuple(range(STATE_COUNT))
    operators.discard(identity)
    return tuple(sorted(operators))


def _apply_binary_matrix(matrix_code: int, state: int) -> int:
    output = 0
    for row in range(3):
        parity = 0
        for column in range(3):
            coefficient = (matrix_code >> (row * 3 + column)) & 1
            parity ^= coefficient & ((state >> column) & 1)
        output |= parity << row
    return output
