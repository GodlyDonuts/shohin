from __future__ import annotations

import ast
import dataclasses
import inspect
from dataclasses import replace

import pytest
import torch

from pipeline import endogenous_congruence_board as board
from pipeline import neural_endogenous_congruence as neural
from pipeline import neural_endogenous_record_fiber as fiber
from pipeline import tensorize_endogenous_congruence as boundary


def _small_model() -> fiber.NeuralEndogenousRecordFiber:
    torch.manual_seed(2026072305)
    model = fiber.NeuralEndogenousRecordFiber(
        fiber.NeuralEndogenousRecordFiberConfig(
            encoder_config=neural.NeuralEndogenousCongruenceConfig(
                hidden_dim=48,
                rounds=2,
            ),
            vote_hidden_dim=32,
        )
    )
    model.eval()
    return model


def _record_mask(batch_size: int, active: int) -> torch.Tensor:
    mask = torch.zeros((batch_size, boundary.N), dtype=torch.bool)
    mask[:, :active] = True
    return mask


def _vote_logits_from_signatures(
    signatures: torch.Tensor,
    record_mask: torch.Tensor,
) -> torch.Tensor:
    pair_mask = record_mask[:, :, None] & record_mask[:, None, :]
    active_logits = torch.where(
        signatures,
        torch.tensor(12.0),
        torch.tensor(-12.0),
    ).unsqueeze(-1)
    active_logits = active_logits.expand(
        -1,
        -1,
        -1,
        fiber.SIGNATURE_REPLICAS,
    )
    return torch.where(
        pair_mask.unsqueeze(-1),
        active_logits,
        torch.tensor(neural.MASKED_LOGIT),
    )


def _assert_equivalence_laws(
    relation: torch.Tensor,
    record_mask: torch.Tensor,
) -> None:
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


def _reference_relation(
    packet: board.EndogenousCongruencePacket,
) -> torch.Tensor:
    solution = board.solve_with_independent_crosscheck(packet)
    relation = torch.zeros(
        (boundary.N, boundary.N),
        dtype=torch.bool,
    )
    for left, left_class in enumerate(solution.record_class):
        for right, right_class in enumerate(solution.record_class):
            relation[left, right] = left_class == right_class
    return relation


def _restricted_growth_strings(size: int) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], maximum: int) -> None:
        if len(prefix) == size:
            output.append(prefix)
            return
        for value in range(maximum + 2):
            visit((*prefix, value), max(maximum, value))

    visit((0,), 0)
    return output


def _reordered_packet(
    packet: board.EndogenousCongruencePacket,
    record_order: tuple[str, ...],
) -> board.EndogenousCongruencePacket:
    tables = board.validate_packet(packet)
    return board.EndogenousCongruencePacket(
        records=record_order,
        generators=packet.generators,
        query_ports=packet.query_ports,
        transition_witnesses=tuple(
            board.TransitionWitness(
                source,
                generator,
                tables.transition[(source, generator)],
            )
            for source in record_order
            for generator in packet.generators
        ),
        observation_witnesses=tuple(
            board.ObservationWitness(
                record,
                query,
                tables.observation[(record, query)],
            )
            for record in record_order
            for query in packet.query_ports
        ),
    )


def _alignment(
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> torch.Tensor:
    active = [source.index(name) for name in target]
    return torch.tensor(
        [*active, *range(len(active), boundary.N)],
        dtype=torch.long,
    )


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
                sign = -1 if (batch_index + query_index + rank) % 2 else 1
                replacement = sign * (
                    1_000_003 * (batch_index + 1)
                    + 10_007 * (query_index + 1)
                    + 97 * (rank + 1)
                )
                selected = active & (
                    tensors.observation_value[
                        batch_index,
                        :,
                        query_index,
                    ]
                    == original
                )
                values[batch_index, selected, query_index] = replacement
    return _replace_observation_values(tensors, values)


def test_module_has_no_partition_or_assessor_custody() -> None:
    source = inspect.getsource(fiber)
    tree = ast.parse(source)
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "pipeline.endogenous_congruence_board" not in imported_modules
    forbidden_names = {
        "CongruenceSolution",
        "compute_exhaustive_partition",
        "compute_refinement_partition",
        "solve_with_independent_crosscheck",
        "union_find",
    }
    assert not forbidden_names & {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert fiber.SIGNATURE_REPLICAS == 5
    assert fiber.SIGNATURE_MAJORITY == 3


def test_exhaustive_binary_signature_matrices_are_equivalence_relations() -> None:
    tested = 0
    chunk_size = 1_024
    for active in range(1, 5):
        matrix_count = 1 << (active * active)
        bit_positions = torch.arange(active * active, dtype=torch.int64)
        for start in range(0, matrix_count, chunk_size):
            stop = min(start + chunk_size, matrix_count)
            indices = torch.arange(start, stop, dtype=torch.int64)
            active_signatures = ((indices[:, None] >> bit_positions[None, :]) & 1).to(
                torch.bool
            )
            active_signatures = active_signatures.reshape(
                stop - start,
                active,
                active,
            )
            signatures = torch.zeros(
                (stop - start, boundary.N, boundary.N),
                dtype=torch.bool,
            )
            signatures[:, :active, :active] = active_signatures
            record_mask = _record_mask(stop - start, active)
            hard = fiber.decode_record_fiber_vote_logits(
                _vote_logits_from_signatures(signatures, record_mask),
                record_mask,
            )
            expected_signatures = signatures.clone()
            diagonal = torch.arange(active)
            expected_signatures[:, diagonal, diagonal] = True
            assert torch.equal(hard.signatures, expected_signatures)
            _assert_equivalence_laws(hard.equivalence, record_mask)
            torch.testing.assert_close(
                hard.projector @ hard.projector,
                hard.projector,
                rtol=0.0,
                atol=1e-6,
            )
            tested += stop - start
    assert tested == 66_066


def test_every_partition_at_n8_is_represented_by_its_relation_rows() -> None:
    partitions = _restricted_growth_strings(boundary.N)
    assert len(partitions) == 4_140
    labels = torch.tensor(partitions, dtype=torch.int64)
    target = labels[:, :, None] == labels[:, None, :]
    record_mask = torch.ones(
        (len(partitions), boundary.N),
        dtype=torch.bool,
    )
    hard = fiber.decode_record_fiber_vote_logits(
        _vote_logits_from_signatures(target, record_mask),
        record_mask,
    )
    assert torch.equal(hard.signatures, target)
    assert torch.equal(hard.equivalence, target)
    expected_projector = target.to(torch.float32)
    expected_projector /= expected_projector.sum(
        dim=-1,
        keepdim=True,
    )
    assert torch.equal(hard.projector, expected_projector)


def test_random_and_adversarial_logits_remain_valid_and_fail_closed() -> None:
    torch.manual_seed(2026072306)
    batch_size = 97
    counts = torch.randint(1, boundary.N + 1, (batch_size,))
    record_mask = torch.arange(boundary.N)[None, :] < counts[:, None]
    pair_mask = record_mask[:, :, None] & record_mask[:, None, :]
    logits = torch.randn(
        batch_size,
        boundary.N,
        boundary.N,
        fiber.SIGNATURE_REPLICAS,
    )
    logits[0] = 0.0
    logits[1, : counts[1], : counts[1], :] = 10_000.0
    logits[2, : counts[2], : counts[2], :] = -10_000.0
    logits = torch.where(
        pair_mask.unsqueeze(-1),
        logits,
        torch.tensor(neural.MASKED_LOGIT),
    )
    hard = fiber.decode_record_fiber_vote_logits(logits, record_mask)
    _assert_equivalence_laws(hard.equivalence, record_mask)

    nonfinite = logits.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(
        fiber.NeuralEndogenousRecordFiberError,
        match="non-finite",
    ):
        fiber.decode_record_fiber_vote_logits(nonfinite, record_mask)

    wrong_padding = logits.clone()
    padded_location = torch.nonzero(
        ~pair_mask,
        as_tuple=False,
    )[0]
    wrong_padding[
        padded_location[0],
        padded_location[1],
        padded_location[2],
        0,
    ] = 0.0
    with pytest.raises(
        fiber.NeuralEndogenousRecordFiberError,
        match="padding",
    ):
        fiber.decode_record_fiber_vote_logits(wrong_padding, record_mask)

    with pytest.raises(
        fiber.NeuralEndogenousRecordFiberError,
        match="floating",
    ):
        fiber.decode_record_fiber_vote_logits(
            logits.to(torch.int64),
            record_mask,
        )


def test_record_permutation_equivariance_for_all_fiber_outputs() -> None:
    packet = board.build_congruence_collision_orbit().base
    reordered = _reordered_packet(packet, packet.records[::-1])
    original = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    transformed = boundary.tensorize_endogenous_congruence_packets((reordered,)).tensors
    permutation = _alignment(reordered.records, packet.records)
    model = _small_model()
    with torch.no_grad():
        left = model(original)
        right = model(transformed)

    aligned_logits = right.vote_logits.index_select(
        1,
        permutation,
    ).index_select(2, permutation)
    torch.testing.assert_close(
        aligned_logits,
        left.vote_logits,
        rtol=2e-5,
        atol=2e-6,
    )
    aligned_soft = right.soft_signatures.index_select(
        1,
        permutation,
    ).index_select(2, permutation)
    torch.testing.assert_close(
        aligned_soft,
        left.soft_signatures,
        rtol=2e-5,
        atol=2e-6,
    )
    aligned_relation = right.soft_fiber_relation.index_select(
        1,
        permutation,
    ).index_select(2, permutation)
    torch.testing.assert_close(
        aligned_relation,
        left.soft_fiber_relation,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.equal(
        right.hard.signatures.index_select(
            1,
            permutation,
        ).index_select(2, permutation),
        left.hard.signatures,
    )
    assert torch.equal(
        right.hard.equivalence.index_select(
            1,
            permutation,
        ).index_select(2, permutation),
        left.hard.equivalence,
    )


def test_arbitrary_observation_recoding_is_bit_exact() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.base, orbit.split_bisimilar, orbit.merged)
    ).tensors
    recoded = _arbitrary_per_query_recoding(tensors)
    assert not torch.equal(
        tensors.observation_value,
        recoded.observation_value,
    )
    model = _small_model()
    with torch.no_grad():
        original = model(tensors)
        transformed = model(recoded)
    for name in (
        "vote_logits",
        "vote_probabilities",
        "soft_signatures",
        "soft_fiber_relation",
    ):
        torch.testing.assert_close(
            getattr(original, name),
            getattr(transformed, name),
            rtol=0.0,
            atol=0.0,
        )
    assert torch.equal(
        original.hard.signatures,
        transformed.hard.signatures,
    )
    assert torch.equal(
        original.hard.equivalence,
        transformed.hard.equivalence,
    )


def test_observation_equality_pattern_changes_active_vote_logits() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    values = tensors.observation_value.clone()
    mutation: tuple[int, int] | None = None
    for query_index in range(boundary.Q):
        active_indices = torch.nonzero(
            tensors.observation_mask[0, :, query_index],
            as_tuple=False,
        ).flatten()
        for left_offset, left_index in enumerate(active_indices.tolist()):
            for right_index in active_indices[left_offset + 1 :].tolist():
                if (
                    values[0, left_index, query_index]
                    == values[0, right_index, query_index]
                ):
                    mutation = (right_index, query_index)
                    break
            if mutation is not None:
                break
        if mutation is not None:
            break
    assert mutation is not None
    record_index, query_index = mutation
    values[0, record_index, query_index] = values.abs().max() + 1_000_003
    mutated = _replace_observation_values(tensors, values)
    model = _small_model()
    with torch.no_grad():
        original = model(tensors)
        changed = model(mutated)
    active_pairs = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    difference = (original.vote_logits - changed.vote_logits).abs()
    expanded_active = active_pairs.unsqueeze(-1).expand_as(difference)
    assert torch.max(difference[expanded_active]).item() > 1e-7


def test_soft_fiber_objective_has_finite_nonzero_gradient_paths() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    target = _reference_relation(packet).unsqueeze(0)
    model = _small_model()
    model.train()
    output = model(tensors)
    output.vote_logits.retain_grad()
    loss = fiber.record_fiber_loss(output, target)
    loss.total.backward()

    assert torch.isfinite(loss.total)
    for component in (
        loss.code,
        loss.fiber,
        loss.distance,
        loss.margin,
    ):
        assert torch.isfinite(component)
    assert output.vote_logits.grad is not None
    active = (
        tensors.record_mask[:, :, None, None] & tensors.record_mask[:, None, :, None]
    ).expand_as(output.vote_logits)
    assert torch.any(output.vote_logits.grad[active] != 0)
    vote_gradients = [
        parameter.grad
        for parameter in model.vote_head.parameters()
        if parameter.grad is not None
    ]
    encoder_gradients = [
        parameter.grad
        for name, parameter in model.encoder.named_parameters()
        if not name.startswith("pair_head") and parameter.grad is not None
    ]
    assert vote_gradients
    assert encoder_gradients
    assert all(torch.all(torch.isfinite(item)) for item in vote_gradients)
    assert all(torch.all(torch.isfinite(item)) for item in encoder_gradients)
    assert any(torch.any(item != 0) for item in vote_gradients)
    assert any(torch.any(item != 0) for item in encoder_gradients)
    assert output.soft_signatures.grad_fn is not None
    assert output.soft_fiber_relation.grad_fn is not None


def test_identity_merge_all_and_collision_controls_are_explicit() -> None:
    active = 4
    record_mask = _record_mask(3, active)
    signatures = torch.zeros(
        (3, boundary.N, boundary.N),
        dtype=torch.bool,
    )
    signatures[0, :active, :active] = torch.eye(active, dtype=torch.bool)
    signatures[1, :active, :active] = True
    signatures[2, 0, :2] = True
    signatures[2, 1, :2] = True
    signatures[2, 2, 2] = True
    signatures[2, 3, 3] = True
    hard = fiber.decode_record_fiber_vote_logits(
        _vote_logits_from_signatures(signatures, record_mask),
        record_mask,
    )
    identity = torch.eye(active, dtype=torch.bool)
    assert torch.equal(hard.equivalence[0, :active, :active], identity)
    assert torch.all(hard.equivalence[1, :active, :active])
    expected_collision = identity.clone()
    expected_collision[:2, :2] = True
    assert torch.equal(
        hard.equivalence[2, :active, :active],
        expected_collision,
    )


def test_physical_validation_separates_validity_from_correct_congruence() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    reference = _reference_relation(packet).unsqueeze(0)
    hard_reference = fiber.decode_record_fiber_vote_logits(
        _vote_logits_from_signatures(reference, tensors.record_mask),
        tensors.record_mask,
    )
    validation = fiber.validate_record_fiber_physical_laws(
        tensors,
        hard_reference,
    )
    assert torch.all(validation.valid)
    assert validation.residuals.observation.item() == 0.0
    assert validation.residuals.descent.item() == 0.0

    identity = tensors.record_equal
    hard_identity = fiber.decode_record_fiber_vote_logits(
        _vote_logits_from_signatures(identity, tensors.record_mask),
        tensors.record_mask,
    )
    identity_validation = fiber.validate_record_fiber_physical_laws(
        tensors,
        hard_identity,
    )
    assert torch.all(identity_validation.valid)
    assert not torch.equal(identity, reference)

    merge_all = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    hard_merge = fiber.decode_record_fiber_vote_logits(
        _vote_logits_from_signatures(merge_all, tensors.record_mask),
        tensors.record_mask,
    )
    structural = fiber.measure_record_fiber_physical_laws(
        tensors,
        hard_merge,
    )
    _assert_equivalence_laws(hard_merge.equivalence, tensors.record_mask)
    assert not torch.all(structural.observation_valid)
    with pytest.raises(
        fiber.NeuralEndogenousRecordFiberError,
        match="observations",
    ):
        fiber.validate_record_fiber_physical_laws(tensors, hard_merge)


def test_forward_masks_padding_and_rejects_invalid_boundaries() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.base, orbit.merged)
    ).tensors
    model = _small_model()
    output = model(tensors)
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    expanded_padding = ~pair_mask.unsqueeze(-1).expand_as(output.vote_logits)
    assert torch.all(output.vote_logits[expanded_padding] == neural.MASKED_LOGIT)
    assert torch.all(output.vote_probabilities[expanded_padding] == 0)
    assert not torch.any(output.hard.signatures & ~pair_mask)
    assert not torch.any(output.hard.equivalence & ~pair_mask)

    corrupted = replace(
        tensors,
        transition_target=torch.zeros_like(tensors.transition_target),
    )
    with pytest.raises(neural.NeuralEndogenousCongruenceError):
        model(corrupted)

    bad_hard = dataclasses.replace(
        output.hard,
        equivalence=torch.zeros_like(output.hard.equivalence),
    )
    with pytest.raises(
        fiber.NeuralEndogenousRecordFiberError,
        match="row equality",
    ):
        fiber.measure_record_fiber_physical_laws(tensors, bad_hard)


def test_parameter_ledger_is_complete_and_below_200m() -> None:
    model = fiber.NeuralEndogenousRecordFiber()
    count = model.parameter_count()
    assert count.encoder == sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    assert count.vote_head == sum(
        parameter.numel() for parameter in model.vote_head.parameters()
    )
    assert count.total == count.encoder + count.vote_head
    assert count.total == count.trainable
    assert count.total <= count.cap
    assert count.under_cap
    assert count.protected_base == 125_081_664
    assert count.complete_system == count.protected_base + count.total
    assert count.complete_system < 200_000_000
    assert count.under_system_cap
    assert count.headroom == count.system_cap - count.complete_system
