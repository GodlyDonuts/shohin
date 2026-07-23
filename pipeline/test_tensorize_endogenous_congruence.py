from __future__ import annotations

import ast
import dataclasses
import inspect
from dataclasses import fields

import pytest
import torch

from pipeline import endogenous_congruence_board as board
from pipeline import tensorize_endogenous_congruence as boundary


def _visible_facts(
    packet: board.EndogenousCongruencePacket,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    frozenset[tuple[str, str, str]],
    frozenset[tuple[str, str, int]],
]:
    return (
        packet.records,
        packet.generators,
        packet.query_ports,
        frozenset(
            (item.source, item.generator, item.target)
            for item in packet.transition_witnesses
        ),
        frozenset(
            (item.record, item.query_port, item.value)
            for item in packet.observation_witnesses
        ),
    )


def _canonical_reorder(
    packet: board.EndogenousCongruencePacket,
    *,
    records: tuple[str, ...],
    generators: tuple[str, ...],
    queries: tuple[str, ...],
) -> board.EndogenousCongruencePacket:
    tables = board.validate_packet(packet)
    value = board.EndogenousCongruencePacket(
        records=records,
        generators=generators,
        query_ports=queries,
        transition_witnesses=tuple(
            board.TransitionWitness(
                record,
                generator,
                tables.transition[(record, generator)],
            )
            for record in records
            for generator in generators
        ),
        observation_witnesses=tuple(
            board.ObservationWitness(
                record,
                query,
                tables.observation[(record, query)],
            )
            for record in records
            for query in queries
        ),
    )
    board.validate_packet(value)
    return value


_TENSOR_AXES: dict[str, tuple[str, ...]] = {
    "record_mask": ("N",),
    "generator_mask": ("G",),
    "query_mask": ("Q",),
    "record_equal": ("N", "N"),
    "generator_equal": ("G", "G"),
    "query_equal": ("Q", "Q"),
    "transition_mask": ("N", "G"),
    "transition_target": ("N", "G", "N"),
    "observation_mask": ("N", "Q"),
    "observation_value": ("N", "Q"),
    "observation_equal": ("N", "Q", "N", "Q"),
}


def _permutation(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    width: int,
    *,
    inverse_names: dict[str, str] | None = None,
) -> torch.Tensor:
    inverse_names = inverse_names or {}
    normalized_source = tuple(inverse_names.get(item, item) for item in source_ids)
    assert set(normalized_source) == set(target_ids)
    active = [normalized_source.index(item) for item in target_ids]
    return torch.tensor([*active, *range(len(active), width)], dtype=torch.long)


def _aligned_fields(
    value: boundary.TensorizedEndogenousCongruencePackets,
    *,
    target_receipt: boundary.EndogenousCongruenceAxisReceipt,
    inverse_names: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    receipt = value.receipts[0]
    permutations = {
        "N": _permutation(
            receipt.record_ids,
            target_receipt.record_ids,
            boundary.N,
            inverse_names=inverse_names,
        ),
        "G": _permutation(
            receipt.generator_ids,
            target_receipt.generator_ids,
            boundary.G,
            inverse_names=inverse_names,
        ),
        "Q": _permutation(
            receipt.query_ids,
            target_receipt.query_ids,
            boundary.Q,
            inverse_names=inverse_names,
        ),
    }
    output = {}
    for field, axes in _TENSOR_AXES.items():
        active = getattr(value.tensors, field)[0]
        for dimension, axis in enumerate(axes):
            active = active.index_select(dimension, permutations[axis])
        output[field] = active
    return output


def _assert_aligned(
    left: boundary.TensorizedEndogenousCongruencePackets,
    right: boundary.TensorizedEndogenousCongruencePackets,
    *,
    inverse_names: dict[str, str] | None = None,
) -> None:
    left_fields = _aligned_fields(left, target_receipt=left.receipts[0])
    right_fields = _aligned_fields(
        right,
        target_receipt=left.receipts[0],
        inverse_names=inverse_names,
    )
    for field in left_fields:
        assert torch.equal(left_fields[field], right_fields[field]), field


def test_model_tensor_surface_contains_only_physical_packet_channels() -> None:
    assert {field.name for field in fields(boundary.EndogenousCongruenceTensors)} == {
        "record_mask",
        "generator_mask",
        "query_mask",
        "record_equal",
        "generator_equal",
        "query_equal",
        "transition_mask",
        "transition_target",
        "observation_mask",
        "observation_value",
        "observation_equal",
    }
    assert {
        field.name for field in fields(boundary.EndogenousCongruenceAxisReceipt)
    } == {"record_ids", "generator_ids", "query_ids"}
    signature = inspect.signature(boundary.tensorize_endogenous_congruence_packets)
    assert tuple(signature.parameters) == ("packets", "device")


def test_module_imports_no_hidden_assessor_products_or_computations() -> None:
    source = inspect.getsource(boundary)
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pipeline.endogenous_congruence_board"
        for alias in node.names
    }
    assert imported_names == {
        "MAX_GENERATORS",
        "MAX_QUERY_PORTS",
        "MAX_RECORDS",
        "CongruenceBoardError",
        "EndogenousCongruencePacket",
        "ObservationWitness",
        "TransitionWitness",
        "validate_packet",
    }
    forbidden = {
        "CongruenceSolution",
        "MergeCertificate",
        "DistinctionCertificate",
        "PathCongruence",
        "PresentationMorphism",
        "NaturalityWitness",
        "compute_refinement_partition",
        "compute_exhaustive_partition",
        "solve_with_independent_crosscheck",
        "audit_collision_orbit",
    }
    assert not forbidden & {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }


def test_hidden_computation_mutations_cannot_affect_tensor_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = board.build_congruence_collision_orbit().base
    baseline = boundary.tensorize_endogenous_congruence_packets((packet,))

    def forbidden_call(*args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))

    for name in (
        "compute_refinement_partition",
        "compute_exhaustive_partition",
        "solve_with_independent_crosscheck",
        "audit_collision_orbit",
    ):
        monkeypatch.setattr(board, name, forbidden_call)
    repeated = boundary.tensorize_endogenous_congruence_packets((packet,))
    for field in _TENSOR_AXES:
        assert torch.equal(
            getattr(baseline.tensors, field),
            getattr(repeated.tensors, field),
        )


def test_packet_subclass_with_extra_hidden_output_is_rejected() -> None:
    @dataclasses.dataclass(frozen=True)
    class ExtendedPacket(board.EndogenousCongruencePacket):
        hidden_output: tuple[int, ...]

    base = board.build_congruence_collision_orbit().base
    extended = ExtendedPacket(
        records=base.records,
        generators=base.generators,
        query_ports=base.query_ports,
        transition_witnesses=base.transition_witnesses,
        observation_witnesses=base.observation_witnesses,
        hidden_output=(0, 1, 2),
    )
    with pytest.raises(
        boundary.EndogenousCongruenceTensorError,
        match="exact EndogenousCongruencePacket",
    ):
        boundary.tensorize_endogenous_congruence_packets((extended,))


def test_frozen_geometry_masks_and_dtypes_match_cpu_board() -> None:
    packet = board.build_congruence_collision_orbit().base
    value = boundary.tensorize_endogenous_congruence_packets((packet,))
    tensors = value.tensors
    assert (boundary.N, boundary.G, boundary.Q) == (8, 4, 4)
    assert tensors.record_mask.shape == (1, 8)
    assert tensors.generator_mask.shape == (1, 4)
    assert tensors.query_mask.shape == (1, 4)
    assert tensors.transition_target.shape == (1, 8, 4, 8)
    assert tensors.observation_value.shape == (1, 8, 4)
    assert tensors.observation_equal.shape == (1, 8, 4, 8, 4)
    assert tensors.observation_value.dtype == torch.int64
    for field in _TENSOR_AXES:
        if field != "observation_value":
            assert getattr(tensors, field).dtype == torch.bool
    assert int(tensors.record_mask.sum()) == len(packet.records)
    assert int(tensors.generator_mask.sum()) == len(packet.generators)
    assert int(tensors.query_mask.sum()) == len(packet.query_ports)
    assert int(tensors.transition_mask.sum()) == len(packet.transition_witnesses)
    assert int(tensors.observation_mask.sum()) == len(packet.observation_witnesses)


def test_round_trip_reconstructs_visible_facts_for_every_orbit_packet() -> None:
    orbit = board.build_congruence_collision_orbit()
    packets = (
        orbit.base,
        orbit.reindexed,
        orbit.split_bisimilar,
        orbit.merged,
        orbit.minimal_noncongruent,
        orbit.commuting_path_twin,
        orbit.noncommuting_twin,
    )
    value = boundary.tensorize_endogenous_congruence_packets(packets)
    for index, packet in enumerate(packets):
        reconstructed = boundary.detensorize_endogenous_congruence_packet(
            value,
            index,
        )
        assert _visible_facts(reconstructed) == _visible_facts(packet)


def test_record_generator_query_order_permutations_are_exactly_equivariant() -> None:
    packet = board.build_congruence_collision_orbit().base
    reordered = _canonical_reorder(
        packet,
        records=tuple(reversed(packet.records)),
        generators=(packet.generators[1], packet.generators[2], packet.generators[0]),
        queries=tuple(reversed(packet.query_ports)),
    )
    left = boundary.tensorize_endogenous_congruence_packets((packet,))
    right = boundary.tensorize_endogenous_congruence_packets((reordered,))
    _assert_aligned(left, right)


def test_opaque_renaming_and_axis_permutation_are_exactly_equivariant() -> None:
    orbit = board.build_congruence_collision_orbit()
    left = boundary.tensorize_endogenous_congruence_packets((orbit.base,))
    right = boundary.tensorize_endogenous_congruence_packets((orbit.reindexed,))
    record_map = dict(orbit.base_to_reindexed.record_map)
    generator_map = dict(orbit.base_to_reindexed.generator_map)
    query_map = dict(orbit.base_to_reindexed.query_map)
    inverse_names = {
        **{target: source for source, target in record_map.items()},
        **{target: source for source, target in generator_map.items()},
        **{target: source for source, target in query_map.items()},
    }
    _assert_aligned(left, right, inverse_names=inverse_names)


def test_split_and_merge_presentations_use_masks_not_shape_changes() -> None:
    orbit = board.build_congruence_collision_orbit()
    packets = (orbit.base, orbit.split_bisimilar, orbit.merged)
    value = boundary.tensorize_endogenous_congruence_packets(packets)
    assert value.tensors.transition_target.shape == (3, 8, 4, 8)
    assert value.tensors.observation_value.shape == (3, 8, 4)
    assert value.tensors.record_mask.sum(dim=1).tolist() == [6, 7, 5]
    assert value.tensors.transition_mask.sum(dim=(1, 2)).tolist() == [18, 21, 15]
    assert value.tensors.observation_mask.sum(dim=(1, 2)).tolist() == [12, 14, 10]
    for index, packet in enumerate(packets):
        reconstructed = boundary.detensorize_endogenous_congruence_packet(
            value,
            index,
        )
        assert _visible_facts(reconstructed) == _visible_facts(packet)


def test_out_of_range_observation_and_board_capacity_fail_closed() -> None:
    packet = board.build_congruence_collision_orbit().base
    first = packet.observation_witnesses[0]
    oversized = dataclasses.replace(
        packet,
        observation_witnesses=(
            dataclasses.replace(first, value=2**63),
            *packet.observation_witnesses[1:],
        ),
    )
    with pytest.raises(
        boundary.EndogenousCongruenceTensorError,
        match="signed-int64",
    ):
        boundary.tensorize_endogenous_congruence_packets((oversized,))

    records = tuple(f"r{index}" for index in range(9))
    over_capacity = board.EndogenousCongruencePacket(
        records=records,
        generators=("g0",),
        query_ports=("q0",),
        transition_witnesses=tuple(
            board.TransitionWitness(record, "g0", record) for record in records
        ),
        observation_witnesses=tuple(
            board.ObservationWitness(record, "q0", 0) for record in records
        ),
    )
    with pytest.raises(
        boundary.EndogenousCongruenceTensorError,
        match="record count exceeds 8",
    ):
        boundary.tensorize_endogenous_congruence_packets((over_capacity,))


def test_decoder_rejects_padding_and_non_one_hot_mutations() -> None:
    packet = board.build_congruence_collision_orbit().merged
    value = boundary.tensorize_endogenous_congruence_packets((packet,))

    changed_target = value.tensors.transition_target.clone()
    changed_target[0, 0, 0, 1] = True
    malformed_target = dataclasses.replace(
        value,
        tensors=dataclasses.replace(
            value.tensors,
            transition_target=changed_target,
        ),
    )
    with pytest.raises(
        boundary.EndogenousCongruenceTensorError,
        match="not one-hot",
    ):
        boundary.detensorize_endogenous_congruence_packet(malformed_target, 0)

    changed_values = value.tensors.observation_value.clone()
    changed_values[0, 7, 3] = 91
    malformed_padding = dataclasses.replace(
        value,
        tensors=dataclasses.replace(
            value.tensors,
            observation_value=changed_values,
        ),
    )
    with pytest.raises(
        boundary.EndogenousCongruenceTensorError,
        match="observation padding",
    ):
        boundary.detensorize_endogenous_congruence_packet(malformed_padding, 0)


def test_tensorization_is_deterministic_and_moves_to_requested_device() -> None:
    packet = board.build_congruence_collision_orbit().minimal_noncongruent
    first = boundary.tensorize_endogenous_congruence_packets((packet,))
    second = boundary.tensorize_endogenous_congruence_packets(
        (packet,),
        device=torch.device("cpu"),
    )
    for field in _TENSOR_AXES:
        left = getattr(first.tensors, field)
        right = getattr(second.tensors, field)
        assert left.device.type == "cpu"
        assert torch.equal(left, right)
