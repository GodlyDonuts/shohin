from __future__ import annotations

import ast
import dataclasses
import inspect
from dataclasses import replace

import pytest
import torch

from pipeline import endogenous_congruence_board as board
from pipeline import neural_endogenous_congruence as neural
from pipeline import tensorize_endogenous_congruence as boundary


def _small_model() -> neural.NeuralEndogenousCongruence:
    torch.manual_seed(20260723)
    model = neural.NeuralEndogenousCongruence(
        neural.NeuralEndogenousCongruenceConfig(hidden_dim=48, rounds=2)
    )
    model.eval()
    return model


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


def _reference_relation(
    packet: board.EndogenousCongruencePacket,
) -> torch.Tensor:
    solution = board.solve_with_independent_crosscheck(packet)
    relation = torch.zeros((boundary.N, boundary.N), dtype=torch.bool)
    for left, left_class in enumerate(solution.record_class):
        for right, right_class in enumerate(solution.record_class):
            relation[left, right] = left_class == right_class
    return relation


def _relation_logits(relation: torch.Tensor) -> torch.Tensor:
    return torch.where(
        relation,
        torch.tensor(12.0),
        torch.tensor(-12.0),
    ).unsqueeze(0)


def test_source_boundary_has_no_assessor_or_coordinate_custody() -> None:
    source = inspect.getsource(neural)
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pipeline.tensorize_endogenous_congruence"
        for alias in node.names
    }
    assert imports == {"G", "N", "Q", "EndogenousCongruenceTensors"}
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "pipeline.endogenous_congruence_board" not in imported_modules
    forbidden = {
        "CongruenceSolution",
        "EndogenousCongruenceAxisReceipt",
        "TensorizedEndogenousCongruencePackets",
        "compute_refinement_partition",
        "compute_exhaustive_partition",
        "solve_with_independent_crosscheck",
        "validate_candidate_partition",
        "union_find",
    }
    assert not forbidden & {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    signature = inspect.signature(neural.NeuralEndogenousCongruence.forward)
    assert tuple(signature.parameters) == ("self", "tensors")
    assert signature.parameters["tensors"].annotation == "EndogenousCongruenceTensors"


@pytest.mark.parametrize("axis", ("record", "generator", "query"))
def test_forward_is_entity_reindex_equivariant(axis: str) -> None:
    packet = board.build_congruence_collision_orbit().base
    records = packet.records[::-1] if axis == "record" else packet.records
    generators = packet.generators[::-1] if axis == "generator" else packet.generators
    queries = packet.query_ports[::-1] if axis == "query" else packet.query_ports
    reordered = _reordered_packet(
        packet,
        record_order=records,
        generator_order=generators,
        query_order=queries,
    )
    original = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    transformed = boundary.tensorize_endogenous_congruence_packets((reordered,)).tensors
    model = _small_model()
    with torch.no_grad():
        left = model(original)
        right = model(transformed)

    record_permutation = _alignment(
        reordered.records,
        packet.records,
        boundary.N,
    )
    generator_permutation = _alignment(
        reordered.generators,
        packet.generators,
        boundary.G,
    )
    query_permutation = _alignment(
        reordered.query_ports,
        packet.query_ports,
        boundary.Q,
    )
    torch.testing.assert_close(
        right.record_features.index_select(1, record_permutation),
        left.record_features,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        right.generator_features.index_select(1, generator_permutation),
        left.generator_features,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        right.query_features.index_select(1, query_permutation),
        left.query_features,
        rtol=2e-5,
        atol=2e-6,
    )
    aligned_logits = right.same_class_logits.index_select(
        1,
        record_permutation,
    ).index_select(2, record_permutation)
    torch.testing.assert_close(
        aligned_logits,
        left.same_class_logits,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.equal(
        right.equivalence_mask.index_select(
            1,
            record_permutation,
        ).index_select(2, record_permutation),
        left.equivalence_mask,
    )


def test_hard_decoder_accepts_known_reference_relation_and_projector() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    relation = _reference_relation(packet)
    decoded = neural.decode_endogenous_congruence_logits(
        tensors,
        _relation_logits(relation),
    )
    assert torch.equal(decoded.equivalence[0], relation)
    expected = relation.to(torch.float32)
    expected /= expected.sum(dim=-1, keepdim=True).clamp_min(1)
    torch.testing.assert_close(decoded.projector[0], expected)
    torch.testing.assert_close(
        decoded.projector[0] @ decoded.projector[0],
        decoded.projector[0],
    )
    assert decoded.residuals.descent.item() == 0.0
    assert decoded.residuals.observation.item() == 0.0


def test_merge_all_rejects_but_identity_is_valid_without_coarseness_claim() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    active = tensors.record_mask[0]
    merge_all = active[:, None] & active[None, :]
    with pytest.raises(
        neural.NeuralEndogenousCongruenceError,
        match="observations",
    ):
        neural.decode_endogenous_congruence_logits(
            tensors,
            _relation_logits(merge_all),
        )

    identity = tensors.record_equal[0]
    decoded = neural.decode_endogenous_congruence_logits(
        tensors,
        _relation_logits(identity),
    )
    assert torch.equal(decoded.equivalence[0], identity)
    assert not torch.equal(identity, _reference_relation(packet))


def test_noncongruent_matched_corruption_rejects_generator_compatibility() -> None:
    orbit = board.build_congruence_collision_orbit()
    tensors = boundary.tensorize_endogenous_congruence_packets(
        (orbit.minimal_noncongruent,)
    ).tensors
    stale_relation = _reference_relation(orbit.base)
    with pytest.raises(
        neural.NeuralEndogenousCongruenceError,
        match="generators",
    ):
        neural.decode_endogenous_congruence_logits(
            tensors,
            _relation_logits(stale_relation),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("padding", "padding"),
        ("reflexive", "reflexive"),
        ("symmetric", "symmetric"),
        ("transitive", "transitive"),
    ),
)
def test_decoder_rejects_without_repair(mutation: str, message: str) -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    relation = _reference_relation(packet)
    logits = _relation_logits(relation)
    if mutation == "padding":
        logits[0, -1, -1] = 12.0
    elif mutation == "reflexive":
        logits[0, 0, 0] = -12.0
    elif mutation == "symmetric":
        logits[0, 0, 1] = -logits[0, 0, 1]
    else:
        chain = tensors.record_equal[0].clone()
        chain[0, 1] = chain[1, 0] = True
        chain[1, 2] = chain[2, 1] = True
        logits = _relation_logits(chain)
    with pytest.raises(neural.NeuralEndogenousCongruenceError, match=message):
        neural.decode_endogenous_congruence_logits(tensors, logits)


def test_wrong_pair_logits_receive_finite_nonzero_gradients() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    model = _small_model()
    model.train()
    output = model(tensors)
    output.same_class_logits.retain_grad()
    target = _reference_relation(packet).unsqueeze(0).to(torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.same_class_logits[output.equivalence_mask],
        target[output.equivalence_mask],
    )
    loss = loss + output.residuals.descent.mean()
    loss = loss + output.residuals.observation.mean()
    loss.backward()
    gradient = output.same_class_logits.grad
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient[output.equivalence_mask]))
    assert torch.any(gradient[output.equivalence_mask] != 0)
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.all(torch.isfinite(item)) for item in parameter_gradients)
    assert any(torch.any(item != 0) for item in parameter_gradients)


def test_split_and_merge_batches_preserve_variable_masks() -> None:
    orbit = board.build_congruence_collision_orbit()
    packets = (orbit.split_bisimilar, orbit.merged)
    tensors = boundary.tensorize_endogenous_congruence_packets(packets).tensors
    output = _small_model()(tensors)
    assert tensors.record_mask.sum(dim=1).tolist() == [
        len(orbit.split_bisimilar.records),
        len(orbit.merged.records),
    ]
    assert torch.equal(output.record_mask, tensors.record_mask)
    assert torch.equal(
        output.equivalence_mask,
        tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :],
    )
    for index, packet in enumerate(packets):
        relation = _reference_relation(packet)
        logits = torch.full((1, boundary.N, boundary.N), -12.0)
        logits[0][relation] = 12.0
        single = boundary.EndogenousCongruenceTensors(
            **{
                field.name: getattr(tensors, field.name)[index : index + 1]
                for field in dataclasses.fields(tensors)
            }
        )
        decoded = neural.decode_endogenous_congruence_logits(single, logits)
        assert torch.equal(decoded.equivalence[0], relation)


def test_parameter_ledger_keeps_complete_system_below_cap() -> None:
    model = neural.NeuralEndogenousCongruence()
    count = model.parameter_count()
    assert count.total == count.trainable
    assert count.total <= 8_000_000
    assert count.under_cap
    assert count.protected_base == 125_081_664
    assert count.complete_system == count.protected_base + count.total
    assert count.complete_system < 200_000_000
    assert count.under_system_cap
    assert count.headroom == count.system_cap - count.complete_system


def test_forward_output_is_symmetric_masked_and_differentiable() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    output = _small_model()(tensors)
    torch.testing.assert_close(
        output.same_class_logits,
        output.same_class_logits.transpose(1, 2),
    )
    assert torch.all(
        output.same_class_logits.masked_select(~output.equivalence_mask)
        == neural.MASKED_LOGIT
    )
    assert torch.all(
        output.soft_equivalence.masked_select(~output.equivalence_mask) == 0
    )
    assert torch.all(torch.isfinite(output.soft_projector))
    assert torch.all(torch.isfinite(output.residuals.descent))
    assert torch.all(torch.isfinite(output.residuals.observation))


def test_invalid_boundary_and_nonfinite_logits_fail_closed() -> None:
    packet = board.build_congruence_collision_orbit().base
    tensors = boundary.tensorize_endogenous_congruence_packets((packet,)).tensors
    corrupted = replace(
        tensors,
        transition_target=torch.zeros_like(tensors.transition_target),
    )
    with pytest.raises(neural.NeuralEndogenousCongruenceError):
        _small_model()(corrupted)
    logits = _relation_logits(_reference_relation(packet))
    logits[0, 0, 0] = float("nan")
    with pytest.raises(neural.NeuralEndogenousCongruenceError, match="non-finite"):
        neural.decode_endogenous_congruence_logits(tensors, logits)

    empty = boundary.EndogenousCongruenceTensors(
        **{
            field.name: torch.zeros_like(getattr(tensors, field.name))
            for field in dataclasses.fields(tensors)
        }
    )
    with pytest.raises(neural.NeuralEndogenousCongruenceError, match="two active"):
        _small_model()(empty)
