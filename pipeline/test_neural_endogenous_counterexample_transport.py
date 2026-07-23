from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest
import torch

from pipeline import endogenous_congruence_board as board
from pipeline import neural_endogenous_congruence as neural
from pipeline import neural_endogenous_counterexample_transport as transport
from pipeline import tensorize_endogenous_congruence as boundary


def _small_model() -> transport.NeuralEndogenousCounterexampleTransport:
    torch.manual_seed(2026072311)
    model = transport.NeuralEndogenousCounterexampleTransport(
        transport.NeuralEndogenousCounterexampleTransportConfig(
            hidden_dim=32,
            dynamical_bits=3,
        )
    )
    model.eval()
    return model


def _prefix_mask(batch_size: int, active: int, width: int) -> torch.Tensor:
    return torch.arange(width)[None, :].expand(batch_size, -1) < active


def _restricted_growth_strings(size: int) -> list[tuple[int, ...]]:
    if size == 0:
        return [()]
    output: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], maximum: int) -> None:
        if len(prefix) == size:
            output.append(prefix)
            return
        for value in range(maximum + 2):
            visit((*prefix, value), max(maximum, value))

    visit((0,), 0)
    return output


def _observation_fibers(
    labels: torch.Tensor,
    *,
    active_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, active_records = labels.shape
    record_mask = _prefix_mask(batch_size, active_records, boundary.N)
    query_mask = _prefix_mask(batch_size, active_queries, boundary.Q)
    fibers = torch.zeros(
        (batch_size, boundary.N, boundary.Q, boundary.N),
        dtype=torch.bool,
    )
    relation = labels[:, :, None] == labels[:, None, :]
    for query_index in range(active_queries):
        fibers[
            :,
            :active_records,
            query_index,
            :active_records,
        ] = relation
    return fibers, record_mask, query_mask


def _masked_logits(
    active_logits: torch.Tensor,
    record_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, active, _, bits = active_logits.shape
    output = torch.full(
        (batch_size, boundary.N, boundary.N, bits),
        neural.MASKED_LOGIT,
    )
    output[:, :active, :active] = active_logits
    pair_mask = record_mask[:, :, None] & record_mask[:, None, :]
    return torch.where(
        pair_mask.unsqueeze(-1),
        output,
        torch.tensor(neural.MASKED_LOGIT),
    )


def _assert_equivalence_and_observation_guarantees(
    hard: transport.CounterexampleTransportHardDecoding,
) -> None:
    relation = hard.equivalence
    record_mask = hard.record_mask
    pair_mask = record_mask[:, :, None] & record_mask[:, None, :]
    identity = torch.eye(boundary.N, dtype=torch.bool)[None] & pair_mask
    assert torch.equal(relation & identity, identity)
    assert torch.equal(relation, relation.transpose(1, 2))
    composed = torch.einsum(
        "bij,bjk->bik",
        relation.to(torch.float32),
        relation.to(torch.float32),
    )
    assert not torch.any((composed > 0) & ~relation & pair_mask)
    assert not torch.any(relation & ~pair_mask)

    observation_relation = hard.observation_fibers.permute(0, 2, 1, 3)
    active_query_pairs = hard.query_mask[:, :, None, None] & pair_mask[:, None]
    assert not torch.any(relation[:, None] & ~observation_relation & active_query_pairs)

    projector = hard.projector
    torch.testing.assert_close(
        projector,
        projector.transpose(1, 2),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        projector @ projector,
        projector,
        rtol=0.0,
        atol=1e-6,
    )
    active_rows = projector.sum(dim=-1)
    torch.testing.assert_close(
        active_rows.masked_select(record_mask),
        torch.ones_like(active_rows.masked_select(record_mask)),
        rtol=0.0,
        atol=0.0,
    )


def _reordered_packet(
    packet: board.EndogenousCongruencePacket,
    *,
    record_order: tuple[str, ...],
    generator_order: tuple[str, ...],
    query_order: tuple[str, ...],
) -> board.EndogenousCongruencePacket:
    tables = board.validate_packet(packet)
    return board.EndogenousCongruencePacket(
        records=record_order,
        generators=generator_order,
        query_ports=query_order,
        transition_witnesses=tuple(
            board.TransitionWitness(
                source,
                generator,
                tables.transition[(source, generator)],
            )
            for source in record_order
            for generator in generator_order
        ),
        observation_witnesses=tuple(
            board.ObservationWitness(
                record,
                query,
                tables.observation[(record, query)],
            )
            for record in record_order
            for query in query_order
        ),
    )


def _alignment(
    source: tuple[str, ...],
    target: tuple[str, ...],
    width: int,
) -> torch.Tensor:
    active = [source.index(name) for name in target]
    return torch.tensor([*active, *range(len(active), width)], dtype=torch.long)


def _replace_observation_values(
    tensors: boundary.EndogenousCongruenceTensors,
    values: torch.Tensor,
) -> boundary.EndogenousCongruenceTensors:
    pair_mask = (
        tensors.observation_mask[:, :, :, None, None]
        & tensors.observation_mask[:, None, None, :, :]
    )
    observation_equal = values[:, :, :, None, None] == values[:, None, None, :, :]
    return replace(
        tensors,
        observation_value=values,
        observation_equal=observation_equal & pair_mask,
    )


def _arbitrary_per_query_recoding(
    tensors: boundary.EndogenousCongruenceTensors,
) -> boundary.EndogenousCongruenceTensors:
    values = torch.zeros_like(tensors.observation_value)
    for batch_index in range(values.shape[0]):
        for query_index in range(boundary.Q):
            active = tensors.observation_mask[batch_index, :, query_index]
            source = tensors.observation_value[batch_index, active, query_index]
            for rank, original in enumerate(torch.unique(source, sorted=True).tolist()):
                replacement = (-1 if rank % 2 else 1) * (
                    9_999_991 + 1_003 * query_index + 37 * rank
                )
                selected = active & (
                    tensors.observation_value[batch_index, :, query_index] == original
                )
                values[batch_index, selected, query_index] = replacement
    return _replace_observation_values(tensors, values)


def test_source_deleted_surface_and_fixed_reactor_contract() -> None:
    source = inspect.getsource(transport)
    tree = ast.parse(source)
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "pipeline.endogenous_congruence_board" not in imported_modules
    forbidden = {
        "CongruenceSolution",
        "DistinctionCertificate",
        "EndogenousCongruenceAxisReceipt",
        "PresentationMorphism",
        "compute_exhaustive_partition",
        "compute_refinement_partition",
        "solve_with_independent_crosscheck",
        "union_find",
    }
    assert not forbidden & {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    signature = inspect.signature(
        transport.NeuralEndogenousCounterexampleTransport.forward
    )
    assert tuple(signature.parameters) == ("self", "tensors")
    assert signature.parameters["tensors"].annotation == "EndogenousCongruenceTensors"
    assert transport.MCTFR_ROUNDS == 8
    assert "observation_value" not in inspect.getsource(
        transport.NeuralEndogenousCounterexampleTransport._initial_pair_state
    )


def test_exhaustive_small_signatures_guarantee_equivalence_and_observations() -> None:
    tested = 0
    for active in range(1, 4):
        partitions = _restricted_growth_strings(active)
        for partition in partitions:
            labels = torch.tensor((partition,), dtype=torch.int64)
            observation_fibers, record_mask, query_mask = _observation_fibers(
                labels,
                active_queries=2,
            )
            bit_count = active * active
            for code in range(1 << bit_count):
                bits = (
                    (
                        torch.tensor(code, dtype=torch.int64)
                        >> torch.arange(bit_count, dtype=torch.int64)
                    )
                    & 1
                ).to(torch.bool)
                active_logits = torch.where(
                    bits.reshape(1, active, active, 1),
                    torch.tensor(12.0),
                    torch.tensor(-12.0),
                )
                hard = transport.decode_counterexample_transport_fibers(
                    observation_fibers,
                    _masked_logits(active_logits, record_mask),
                    record_mask,
                    query_mask,
                )
                _assert_equivalence_and_observation_guarantees(hard)
                tested += 1
    assert tested == 2_594


def test_random_and_adversarial_logits_cannot_break_hard_guarantees() -> None:
    torch.manual_seed(2026072312)
    batch_size = 127
    labels = torch.randint(0, 4, (batch_size, boundary.N))
    observation_fibers, record_mask, query_mask = _observation_fibers(
        labels,
        active_queries=boundary.Q,
    )
    logits = torch.randn(batch_size, boundary.N, boundary.N, 7) * 100.0
    logits[0] = 0.0
    logits[1] = 1e6
    logits[2] = -1e6
    hard = transport.decode_counterexample_transport_fibers(
        dynamical_logits=logits,
        observation_fibers=observation_fibers,
        record_mask=record_mask,
        query_mask=query_mask,
    )
    _assert_equivalence_and_observation_guarantees(hard)


def test_aligned_successor_gather_matches_direct_indexing_exactly() -> None:
    packet = board.build_congruence_collision_orbit().noncommuting_twin
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    pair_state = torch.arange(
        boundary.N * boundary.N * 3,
        dtype=torch.float32,
    ).reshape(1, boundary.N, boundary.N, 3)
    gathered = transport.gather_aligned_successor_pairs(
        pair_state,
        tensors.transition_target,
    )
    targets = tensors.transition_target[0].to(torch.int64).argmax(dim=-1)
    for left in range(len(packet.records)):
        for right in range(len(packet.records)):
            for generator in range(len(packet.generators)):
                expected = pair_state[
                    0,
                    targets[left, generator],
                    targets[right, generator],
                ]
                assert torch.equal(gathered[0, left, right, generator], expected)


def test_distinction_channel_is_monotone_for_every_round_and_pair() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.base, orbit.noncommuting_twin, orbit.split_bisimilar)
    ).tensors
    output = _small_model()(tensors)
    assert len(output.pair_state_trace) == transport.MCTFR_ROUNDS + 1
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    for previous, current in zip(
        output.distinction_trace[:-1],
        output.distinction_trace[1:],
        strict=True,
    ):
        assert torch.all(
            current.masked_select(pair_mask) >= previous.masked_select(pair_mask)
        )
        assert torch.equal(
            current.masked_select(~pair_mask),
            torch.zeros_like(current.masked_select(~pair_mask)),
        )


@pytest.mark.parametrize("axis", ("record", "generator", "query"))
def test_forward_is_record_generator_and_query_permutation_equivariant(
    axis: str,
) -> None:
    packet = board.build_congruence_collision_orbit().base
    record_order = packet.records[::-1] if axis == "record" else packet.records
    generator_order = (
        packet.generators[::-1] if axis == "generator" else packet.generators
    )
    query_order = packet.query_ports[::-1] if axis == "query" else packet.query_ports
    reordered = _reordered_packet(
        packet,
        record_order=record_order,
        generator_order=generator_order,
        query_order=query_order,
    )
    original = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    transformed = boundary.tensorize_endogenous_congruence_packets((reordered,)).tensors
    record_alignment = _alignment(
        reordered.records,
        packet.records,
        boundary.N,
    )
    query_alignment = _alignment(
        reordered.query_ports,
        packet.query_ports,
        boundary.Q,
    )
    model = _small_model()
    with torch.no_grad():
        left = model(original)
        right = model(transformed)

    for left_state, right_state in zip(
        left.pair_state_trace,
        right.pair_state_trace,
        strict=True,
    ):
        aligned = right_state.index_select(1, record_alignment).index_select(
            2,
            record_alignment,
        )
        torch.testing.assert_close(aligned, left_state, rtol=0.0, atol=0.0)
    aligned_logits = right.dynamical_logits.index_select(
        1,
        record_alignment,
    ).index_select(2, record_alignment)
    torch.testing.assert_close(
        aligned_logits,
        left.dynamical_logits,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        right.hard.equivalence.index_select(
            1,
            record_alignment,
        ).index_select(2, record_alignment),
        left.hard.equivalence,
    )
    aligned_observations = (
        right.hard.observation_fibers.index_select(1, record_alignment)
        .index_select(2, query_alignment)
        .index_select(3, record_alignment)
    )
    assert torch.equal(aligned_observations, left.hard.observation_fibers)


def test_arbitrary_injective_observation_recoding_is_bit_exact() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.base, orbit.noncommuting_twin, orbit.split_bisimilar)
    ).tensors
    recoded = _arbitrary_per_query_recoding(tensors)
    assert not torch.equal(tensors.observation_value, recoded.observation_value)
    model = _small_model()
    with torch.no_grad():
        original = model(tensors)
        transformed = model(recoded)
    for left, right in zip(
        original.pair_state_trace,
        transformed.pair_state_trace,
        strict=True,
    ):
        assert torch.equal(left, right)
    assert torch.equal(original.dynamical_logits, transformed.dynamical_logits)
    assert torch.equal(
        original.hard.equivalence,
        transformed.hard.equivalence,
    )


def test_transition_transport_detects_noncommuting_physical_change() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.commuting_path_twin, orbit.noncommuting_twin)
    ).tensors
    assert torch.equal(tensors.observation_equal[0], tensors.observation_equal[1])
    assert not torch.equal(tensors.transition_target[0], tensors.transition_target[1])
    with torch.no_grad():
        output = _small_model()(tensors)
    assert torch.equal(
        output.pair_state_trace[0][0],
        output.pair_state_trace[0][1],
    )
    assert not torch.equal(
        output.pair_state_trace[-1][0],
        output.pair_state_trace[-1][1],
    )
    assert not torch.equal(
        output.dynamical_logits[0],
        output.dynamical_logits[1],
    )


def test_soft_fibers_have_finite_nonzero_gradient_through_all_reactor_parts() -> None:
    packet = board.build_congruence_collision_orbit().noncommuting_twin
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    model = _small_model()
    model.train()
    output = model(tensors)
    target = torch.eye(boundary.N, dtype=torch.float32)[None]
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    loss = (
        output.soft_fiber_relation.masked_select(pair_mask)
        - target.masked_select(pair_mask)
    ).square().mean() + 1e-4 * output.final_pair_state.square().mean()
    loss.backward()
    required_modules = (
        model.query_encoder,
        model.initial_auxiliary,
        model.successor_encoder,
        model.distinction_increment,
        model.auxiliary_update,
        model.dynamical_head,
    )
    for module in required_modules:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
        assert any(torch.any(gradient != 0) for gradient in gradients)


def test_negative_controls_fail_closed_and_physical_descent_is_not_repaired() -> None:
    labels = torch.tensor(((0, 0, 1),), dtype=torch.int64)
    observation_fibers, record_mask, query_mask = _observation_fibers(
        labels,
        active_queries=1,
    )
    logits = _masked_logits(torch.zeros((1, 3, 3, 1)), record_mask)

    nonfinite = logits.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(
        transport.NeuralEndogenousCounterexampleTransportError,
        match="non-finite",
    ):
        transport.decode_counterexample_transport_fibers(
            observation_fibers,
            nonfinite,
            record_mask,
            query_mask,
        )

    broken_observation = observation_fibers.clone()
    broken_observation[0, 0, 0, 0] = False
    with pytest.raises(
        transport.NeuralEndogenousCounterexampleTransportError,
        match="reflexive",
    ):
        transport.decode_counterexample_transport_fibers(
            broken_observation,
            logits,
            record_mask,
            query_mask,
        )

    packet = board.build_congruence_collision_orbit().minimal_noncongruent
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    output = _small_model()(tensors)
    assert output.hard_residuals.descent.shape == (1,)
    assert torch.all(torch.isfinite(output.hard_residuals.descent))


def test_parameter_ledger_stays_below_module_and_complete_system_caps() -> None:
    model = transport.NeuralEndogenousCounterexampleTransport()
    count = model.parameter_count()
    direct = sum(parameter.numel() for parameter in model.parameters())
    assert count.total == direct
    assert count.trainable == direct
    assert count.total < 24_000_000
    assert count.complete_system == neural.PROTECTED_BASE_PARAMETERS + direct
    assert count.complete_system < 200_000_000
    assert count.under_cap
    assert count.under_system_cap
