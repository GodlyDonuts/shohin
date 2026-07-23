from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random

import pytest

from pipeline.episode_action_binding_board import (
    ABSTAIN,
    ACTION_COUNT,
    ANSWER,
    BOS,
    EOS,
    PAD,
    QUERY,
    GenerationError,
    MalformedEpisodeError,
    ModelPacket,
    action_agnostic_baseline,
    all_actions_union_baseline,
    binding_enumerator_oracle,
    erase_demonstration_actions,
    generate_cyclic_binding_group,
    generate_cyclic_order_cluster,
    generate_order_twin,
    make_underidentified,
    model_packet_payload,
    pad_packet,
    parse_episode,
    permute_demonstrations,
    query_order_bagging_baseline,
    raw_token_histogram,
    rename_assessor_system,
    rename_nonces,
    split_world_and_query,
    validate_cyclic_group,
    validate_cyclic_order_cluster,
    visible_table_oracle,
    world_commitment,
)


@pytest.mark.parametrize("depth", range(1, 7))
def test_cyclic_triples_pass_all_frozen_mechanics(depth: int) -> None:
    group = generate_cyclic_binding_group(2026072300 + depth, query_depth=depth)
    validate_cyclic_group(group)
    assert len({case.target_token for case in group.variants}) == ACTION_COUNT
    assert all(
        visible_table_oracle(case.packet) == case.target_token
        for case in group.variants
    )
    assert all(
        binding_enumerator_oracle(case.packet, group.system) == case.target_token
        for case in group.variants
    )


def test_cyclic_triple_views_are_identical_for_action_agnostic_methods() -> None:
    group = generate_cyclic_binding_group(2026072311, query_depth=6)
    histograms = [raw_token_histogram(case.packet) for case in group.variants]
    erased = [erase_demonstration_actions(case.packet) for case in group.variants]
    assert histograms[0] == histograms[1] == histograms[2]
    assert erased[0] == erased[1] == erased[2]

    for baseline in (action_agnostic_baseline, all_actions_union_baseline):
        answers = [baseline(case.packet) for case in group.variants]
        assert len(set(answers)) == 1
        assert (
            sum(
                answer == case.target_token
                for answer, case in zip(answers, group.variants, strict=True)
            )
            <= 1
        )


def test_order_twin_requires_noncommuting_query_order() -> None:
    group = generate_cyclic_binding_group(2026072312)
    twin = generate_order_twin(group.system, binding_shift=1)
    left = parse_episode(twin.forward.packet)
    right = parse_episode(twin.reverse.packet)
    assert left.demonstrations == right.demonstrations
    assert left.query_start == right.query_start
    assert Counter(left.query_actions) == Counter(right.query_actions)
    assert twin.forward.target_token != twin.reverse.target_token
    assert world_commitment(twin.forward.packet) == world_commitment(
        twin.reverse.packet
    )


@pytest.mark.parametrize("seed", (2026072330, 2026072331, 2026072332))
def test_six_case_cluster_requires_binding_and_order(seed: int) -> None:
    cluster = generate_cyclic_order_cluster(seed, query_depth=4)
    validate_cyclic_order_cluster(cluster)
    assert Counter(cluster.primary.query_action_indices) == Counter(
        cluster.reordered.query_action_indices
    )
    assert (
        cluster.primary.query_action_indices != cluster.reordered.query_action_indices
    )
    for left, right in zip(
        cluster.primary.variants,
        cluster.reordered.variants,
        strict=True,
    ):
        assert left.target_token != right.target_token
        assert world_commitment(left.packet) == world_commitment(right.packet)
        answers = (
            query_order_bagging_baseline(left.packet),
            query_order_bagging_baseline(right.packet),
        )
        assert answers[0] == answers[1]
        assert (
            sum(
                answer == case.target_token
                for answer, case in zip(answers, (left, right), strict=True)
            )
            <= 1
        )


def test_world_is_committed_before_query_materialization() -> None:
    group = generate_cyclic_binding_group(20260723125)
    twin = generate_order_twin(group.system)
    world, forward_query = split_world_and_query(twin.forward.packet)
    reverse_world, reverse_query = split_world_and_query(twin.reverse.packet)
    assert world == reverse_world
    assert forward_query != reverse_query
    assert QUERY not in world
    assert forward_query[0] == QUERY


def test_nonce_recoding_is_natural() -> None:
    group = generate_cyclic_binding_group(2026072313, query_depth=5)
    case = group.variants[2]
    nonces = group.system.state_tokens + group.system.action_tokens
    renamed_values = tuple(range(2_000_000, 2_000_000 + len(nonces)))
    nonce_map = dict(zip(nonces, renamed_values, strict=True))
    packet = rename_nonces(case.packet, nonce_map)
    system = rename_assessor_system(group.system, nonce_map)
    expected = nonce_map[case.target_token]
    assert visible_table_oracle(packet) == expected
    assert binding_enumerator_oracle(packet, system) == expected


def test_demonstration_permutation_and_padding_preserve_answer() -> None:
    group = generate_cyclic_binding_group(2026072314, query_depth=4)
    case = group.variants[1]
    parsed = parse_episode(case.packet)
    order = list(range(len(parsed.demonstrations)))
    random.Random(17).shuffle(order)
    permuted = permute_demonstrations(case.packet, order)
    padded = pad_packet(permuted, len(permuted.tokens) + 19)
    assert visible_table_oracle(padded) == case.target_token
    assert binding_enumerator_oracle(padded, group.system) == case.target_token


def test_underidentified_action_binding_abstains() -> None:
    group = generate_cyclic_binding_group(2026072315, query_depth=1)
    case = group.variants[0]
    parsed = parse_episode(case.packet)
    queried = parsed.query_actions[0]
    other = next(action for action in group.system.action_tokens if action != queried)
    packet = make_underidentified(case.packet, (queried, other))
    assert visible_table_oracle(packet) == ABSTAIN
    assert binding_enumerator_oracle(packet, group.system) == ABSTAIN


def test_payload_contains_no_assessor_fields() -> None:
    group = generate_cyclic_binding_group(2026072316)
    payload = model_packet_payload(group.variants[0].packet)
    assert set(payload) == {"tokens", "attention_mask"}
    serialized = repr(payload).lower()
    for forbidden in (
        "target",
        "binding",
        "operator",
        "physical",
        "latent",
        "orbit",
        "shift",
    ):
        assert forbidden not in serialized


def test_generation_is_deterministic() -> None:
    left = generate_cyclic_binding_group(2026072317, query_depth=6)
    right = generate_cyclic_binding_group(2026072317, query_depth=6)
    assert left == right
    assert left.group_digest == right.group_digest


def test_generated_token_ids_fit_shohin_vocabulary() -> None:
    group = generate_cyclic_binding_group(20260723175, query_depth=6)
    for case in group.variants:
        assert max(case.packet.tokens) < 32_768


def test_bad_depth_fails_closed() -> None:
    with pytest.raises(GenerationError):
        generate_cyclic_binding_group(1, query_depth=0)
    with pytest.raises(GenerationError):
        generate_cyclic_binding_group(1, query_depth=7)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda tokens: tokens[1:],
        lambda tokens: tokens[:-1],
        lambda tokens: tuple([*tokens[:-1], QUERY]),
        lambda tokens: tuple([*tokens, EOS]),
    ],
)
def test_truncated_or_malformed_streams_fail_closed(mutator) -> None:
    group = generate_cyclic_binding_group(2026072318)
    packet = group.variants[0].packet
    tokens = mutator(packet.tokens)
    malformed = ModelPacket(tokens=tokens, attention_mask=(1,) * len(tokens))
    with pytest.raises(MalformedEpisodeError):
        parse_episode(malformed)


def test_duplicate_transition_fails_closed() -> None:
    group = generate_cyclic_binding_group(2026072319)
    packet = group.variants[0].packet
    first_demo = packet.tokens[1:7]
    query_position = packet.tokens.index(QUERY)
    tokens = (
        packet.tokens[:query_position] + first_demo + packet.tokens[query_position:]
    )
    malformed = ModelPacket(tokens=tokens, attention_mask=(1,) * len(tokens))
    with pytest.raises(MalformedEpisodeError, match="duplicate"):
        parse_episode(malformed)


def test_conflicting_transition_fails_closed() -> None:
    group = generate_cyclic_binding_group(2026072320)
    packet = group.variants[0].packet
    first_demo = list(packet.tokens[1:7])
    first_demo[4] = next(
        token for token in group.system.state_tokens if token != first_demo[4]
    )
    query_position = packet.tokens.index(QUERY)
    tokens = (
        packet.tokens[:query_position]
        + tuple(first_demo)
        + packet.tokens[query_position:]
    )
    malformed = ModelPacket(tokens=tokens, attention_mask=(1,) * len(tokens))
    with pytest.raises(MalformedEpisodeError, match="conflicting"):
        parse_episode(malformed)


def test_unknown_query_action_fails_closed() -> None:
    group = generate_cyclic_binding_group(2026072321)
    packet = group.variants[0].packet
    tokens = list(packet.tokens)
    query_position = tokens.index(QUERY)
    tokens[query_position + 2] = 9_999_999
    malformed = ModelPacket(tuple(tokens), packet.attention_mask)
    with pytest.raises(MalformedEpisodeError, match="unknown action"):
        parse_episode(malformed)


def test_bad_padding_fails_closed() -> None:
    group = generate_cyclic_binding_group(2026072322)
    packet = group.variants[0].packet
    malformed = ModelPacket(
        tokens=packet.tokens + (PAD, BOS),
        attention_mask=packet.attention_mask + (0, 1),
    )
    with pytest.raises(MalformedEpisodeError, match="active token follows"):
        parse_episode(malformed)


def test_target_is_not_present_after_answer_marker() -> None:
    group = generate_cyclic_binding_group(2026072323)
    for case in group.variants:
        tokens = tuple(
            token
            for token, active in zip(
                case.packet.tokens,
                case.packet.attention_mask,
                strict=True,
            )
            if active
        )
        answer_position = tokens.index(ANSWER)
        assert tokens[answer_position:] == (ANSWER, EOS)


def test_assessor_tamper_is_detected() -> None:
    group = generate_cyclic_binding_group(2026072324)
    operators = list(group.system.physical_operators)
    operators[0] = operators[1]
    tampered = replace(group.system, physical_operators=tuple(operators))
    with pytest.raises(GenerationError, match="distinct"):
        binding_enumerator_oracle(group.variants[0].packet, tampered)
