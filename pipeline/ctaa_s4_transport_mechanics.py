#!/usr/bin/env python3
"""Deterministic CPU mechanics audit for CTAA S4-tied particle transport.

This audit contains no learned result. It verifies finite component mechanics,
including transport covariance under conjugated cue coordinates. It cannot
authorize a neural board because it does not exercise byte-source compilation,
source/KV destruction, or a late-query neural reader.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ctaa_binding_completion import ACTION_COUNT, BINDINGS
from ctaa_s4_particle_transport import (
    ACTION_EVENT,
    BINDING_TO_INDEX,
    CUE_EVENT,
    FULL_STATES,
    PARTICLE_COUNT,
    S4_DELTA_INDEX,
    S4_GENERATORS,
    STOP_EVENT,
    Z24_DELTA_INDEX,
    S4TiedTransport,
    conjugate_rebinding_element,
    compose_permutations,
    execute_interleaved_particle_ctaa,
    s4_transport_resource_receipt,
    invert_permutation,
    transform_binding_coordinates,
)


SCHEMA = "r12_ctaa_s4_tied_particle_transport_cpu_mechanics_v2"


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def oracle_compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    if (
        len(left) != ACTION_COUNT
        or len(right) != ACTION_COUNT
        or sorted(left) != list(range(ACTION_COUNT))
        or sorted(right) != list(range(ACTION_COUNT))
    ):
        raise AssertionError("independent S4 oracle received invalid element")
    return tuple(left[right[index]] for index in range(ACTION_COUNT))


def _audit_interleaved_executor() -> dict[str, int | bool]:
    transport = S4TiedTransport()
    identity_card = (0, 1, 2)
    action_maps = tuple(itertools.product(range(3), repeat=3))
    total = 0
    exact = 0
    mass_checks = 0
    chunk_size = 2_048
    cases: list[
        tuple[int, tuple[int, int, int], tuple[int, int, int], int]
    ] = []

    def check_chunk(
        chunk: list[
            tuple[int, tuple[int, int, int], tuple[int, int, int], int]
        ],
    ) -> tuple[int, int, int]:
        batch = len(chunk)
        bindings = torch.tensor([row[0] for row in chunk], dtype=torch.long)
        particles = F.one_hot(bindings, PARTICLE_COUNT).float()
        initial = torch.tensor([row[1] for row in chunk], dtype=torch.long)
        cards = torch.tensor(
            [[identity_card] * ACTION_COUNT for _ in chunk],
            dtype=torch.long,
        )
        expected_indices = []
        opcodes = []
        for row_index, (binding_index, state, action, opcode) in enumerate(chunk):
            selected_card = BINDINGS[binding_index][opcode]
            cards[row_index, selected_card] = torch.tensor(action)
            expected = tuple(state[position] for position in action)
            expected_indices.append(expected[0] * 9 + expected[1] * 3 + expected[2])
            opcodes.append(opcode)
        kinds = torch.tensor(
            [[ACTION_EVENT, STOP_EVENT]] * batch,
            dtype=torch.long,
        )
        values = torch.stack(
            (
                torch.tensor(opcodes, dtype=torch.long),
                torch.zeros(batch, dtype=torch.long),
            ),
            dim=1,
        )
        result = execute_interleaved_particle_ctaa(
            transport,
            particles,
            cards,
            initial,
            kinds,
            values,
            torch.zeros(batch, dtype=torch.long),
        )
        expected_tensor = torch.tensor(expected_indices, dtype=torch.long)
        exact_count = int(
            result.full_state_marginals.argmax(-1).eq(expected_tensor).sum()
        )
        exact_count += int(
            result.binding_marginals.argmax(-1).eq(bindings).sum()
        )
        mass_count = int(
            torch.isclose(
                result.final_joint.sum((1, 2)),
                torch.ones(batch),
                rtol=1e-6,
                atol=1e-6,
            ).sum()
        )
        return batch, exact_count, mass_count

    for binding_index in range(len(BINDINGS)):
        for state in FULL_STATES:
            for action in action_maps:
                for opcode in range(ACTION_COUNT):
                    cases.append((binding_index, state, action, opcode))
                    if len(cases) == chunk_size:
                        checked, correct, conserved = check_chunk(cases)
                        total += checked
                        exact += correct
                        mass_checks += conserved
                        cases = []
    if cases:
        checked, correct, conserved = check_chunk(cases)
        total += checked
        exact += correct
        mass_checks += conserved

    torch.manual_seed(2_026_072_306)
    gradient_transport = S4TiedTransport()
    particle_logits = torch.randn(2, PARTICLE_COUNT, requires_grad=True)
    particles = particle_logits.softmax(-1)
    cards = torch.tensor(
        [
            [[0, 0, 0], [0, 1, 1], [2, 2, 1], [1, 0, 2]],
            [[2, 2, 2], [1, 0, 1], [0, 2, 0], [2, 1, 0]],
        ],
        dtype=torch.long,
    )
    initial = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    prefix_kinds = torch.tensor(
        [
            [CUE_EVENT, ACTION_EVENT, STOP_EVENT],
            [ACTION_EVENT, CUE_EVENT, STOP_EVENT],
        ],
        dtype=torch.long,
    )
    prefix_values = torch.tensor([[0, 1, 0], [2, 3, 0]], dtype=torch.long)
    queries = torch.tensor([0, 2], dtype=torch.long)
    prefix = execute_interleaved_particle_ctaa(
        gradient_transport,
        particles,
        cards,
        initial,
        prefix_kinds,
        prefix_values,
        queries,
    )
    suffix = execute_interleaved_particle_ctaa(
        gradient_transport,
        particles,
        cards,
        initial,
        torch.cat(
            (
                prefix_kinds,
                torch.tensor(
                    [
                        [CUE_EVENT, ACTION_EVENT],
                        [ACTION_EVENT, CUE_EVENT],
                    ],
                    dtype=torch.long,
                ),
            ),
            dim=1,
        ),
        torch.cat(
            (
                prefix_values,
                torch.tensor([[4, 0], [1, 5]], dtype=torch.long),
            ),
            dim=1,
        ),
        queries,
    )
    torch.testing.assert_close(
        suffix.final_joint,
        prefix.final_joint,
        rtol=0,
        atol=0,
    )
    probability_weights = torch.tensor([0.2, 0.5, 0.3])
    loss = -torch.log(
        (suffix.query_distribution * probability_weights).sum(-1)
    ).mean()
    loss.backward()
    particle_gradient_ok = bool(
        particle_logits.grad is not None
        and torch.isfinite(particle_logits.grad).all()
        and particle_logits.grad.abs().sum() > 0
    )
    kernel_gradient = gradient_transport.kernel_logits.grad
    kernel_gradient_ok = bool(
        kernel_gradient is not None
        and torch.isfinite(kernel_gradient).all()
        and kernel_gradient.abs().sum() > 0
    )
    trajectory_mass_ok = bool(
        torch.allclose(
            suffix.binding_trajectory.sum(-1),
            torch.ones_like(suffix.binding_trajectory[..., 0]),
            rtol=1e-5,
            atol=1e-6,
        )
        and torch.allclose(
            suffix.full_state_trajectory.sum(-1),
            torch.ones_like(suffix.full_state_trajectory[..., 0]),
            rtol=1e-5,
            atol=1e-6,
        )
    )
    return {
        "one_step_cases": total,
        "one_step_state_and_binding_exact_checks": exact,
        "one_step_mass_checks": mass_checks,
        "action_maps": len(action_maps),
        "post_stop_suffix_invariant": True,
        "mixed_trajectory_mass_conserved": trajectory_mass_ok,
        "particle_gradient_finite_nonzero": particle_gradient_ok,
        "cue_kernel_gradient_finite_nonzero": kernel_gradient_ok,
    }


def build_report() -> dict[str, object]:
    identity = tuple(range(ACTION_COUNT))

    inverse_checks = 0
    independent_composition_checks = 0
    for element in BINDINGS:
        inverse = invert_permutation(element)
        if (
            compose_permutations(element, inverse) != identity
            or compose_permutations(inverse, element) != identity
        ):
            raise AssertionError("S4 inverse audit failed")
        inverse_checks += 1
        for right in BINDINGS:
            if compose_permutations(element, right) != oracle_compose(element, right):
                raise AssertionError("S4 implementation disagrees with independent oracle")
            independent_composition_checks += 1

    associativity_checks = 0
    for left in BINDINGS:
        for middle in BINDINGS:
            for right in BINDINGS:
                lhs = compose_permutations(
                    compose_permutations(left, middle),
                    right,
                )
                rhs = compose_permutations(
                    left,
                    compose_permutations(middle, right),
                )
                if lhs != rhs:
                    raise AssertionError("S4 associativity audit failed")
                associativity_checks += 1

    generator_pairs = 0
    noncommuting_pairs = 0
    abelian_order_collapses = 0
    treatment_destinations: list[dict[str, object]] = []
    for first_index, first in enumerate(S4_GENERATORS):
        for second_index, second in enumerate(S4_GENERATORS):
            generator_pairs += 1
            forward = compose_permutations(first, second)
            reverse = compose_permutations(second, first)
            if forward != reverse:
                noncommuting_pairs += 1
                treatment_destinations.append(
                    {
                        "first_generator": first_index,
                        "second_generator": second_index,
                        "forward_particle": BINDING_TO_INDEX[forward],
                        "reverse_particle": BINDING_TO_INDEX[reverse],
                    }
                )
            if (
                BINDING_TO_INDEX[first] + BINDING_TO_INDEX[second]
            ) % len(BINDINGS) == (
                BINDING_TO_INDEX[second] + BINDING_TO_INDEX[first]
            ) % len(BINDINGS):
                abelian_order_collapses += 1

    coordinate_roundtrip_checks = 0
    transport_equivariance_checks = 0
    for binding in BINDINGS:
        for opcode_order in BINDINGS:
            opcode_inverse = invert_permutation(opcode_order)
            for card_order in BINDINGS:
                transformed = transform_binding_coordinates(
                    binding,
                    opcode_order,
                    card_order,
                )
                restored = transform_binding_coordinates(
                    transformed,
                    opcode_inverse,
                    invert_permutation(card_order),
                )
                if restored != binding:
                    raise AssertionError("S4 coordinate audit failed")
                coordinate_roundtrip_checks += 1
                for generator in S4_GENERATORS:
                    expected = transform_binding_coordinates(
                        oracle_compose(binding, generator),
                        opcode_order,
                        card_order,
                    )
                    transformed_binding = transform_binding_coordinates(
                        binding,
                        opcode_order,
                        card_order,
                    )
                    transformed_generator = conjugate_rebinding_element(
                        generator,
                        opcode_order,
                    )
                    actual = oracle_compose(
                        transformed_binding,
                        transformed_generator,
                    )
                    if actual != expected:
                        raise AssertionError("S4 transport equivariance audit failed")
                    transport_equivariance_checks += 1

    resources = s4_transport_resource_receipt()
    executor = _audit_interleaved_executor()
    s4_table_sha256 = hashlib.sha256(
        S4_DELTA_INDEX.numpy().tobytes()
    ).hexdigest()
    z24_table_sha256 = hashlib.sha256(
        Z24_DELTA_INDEX.numpy().tobytes()
    ).hexdigest()
    gates = {
        "s4_group_laws": (
            inverse_checks == len(BINDINGS)
            and associativity_checks == len(BINDINGS) ** 3
        ),
        "independent_composition_oracle": (
            independent_composition_checks == len(BINDINGS) ** 2
        ),
        "noncommuting_order_witness": noncommuting_pairs > 0,
        "matched_abelian_order_collapse": (
            abelian_order_collapses == generator_pairs
        ),
        "coordinate_roundtrip": (
            coordinate_roundtrip_checks == len(BINDINGS) ** 3
        ),
        "transport_equivariance_with_conjugated_cues": (
            transport_equivariance_checks
            == len(BINDINGS) ** 3 * len(S4_GENERATORS)
        ),
        "arm_multiplication_tables_are_distinct": (
            s4_table_sha256 != z24_table_sha256
        ),
        "interleaved_executor_exhaustive": (
            executor["one_step_cases"] == 69_984
            and executor["one_step_state_and_binding_exact_checks"] == 139_968
            and executor["one_step_mass_checks"] == 69_984
            and executor["action_maps"] == 27
        ),
        "interleaved_executor_dynamic_checks": (
            executor["post_stop_suffix_invariant"] is True
            and executor["mixed_trajectory_mass_conserved"] is True
            and executor["particle_gradient_finite_nonzero"] is True
            and executor["cue_kernel_gradient_finite_nonzero"] is True
        ),
        "parameter_match": resources["mechanistic_parameter_gap"] == 0,
        "transport_mac_match": (
            resources["treatment_transport_macs_per_cue"]
            == resources["abelian_control_transport_macs_per_cue"]
            == resources["dense_control_transport_macs_per_cue"]
        ),
        "dense_control_is_favorable": (
            int(resources["dense_favorable_control_parameters"])
            > int(resources["treatment_parameters"])
        ),
        "strictly_below_200m": (
            int(resources["complete_system_parameters"])
            < int(resources["strict_system_parameter_limit"])
        ),
    }
    decision = (
        "record_component_mechanics_only_no_neural_authorization"
        if all(gates.values())
        else "reject_s4_tied_particle_transport"
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "finite_group_transport_mechanics_only_no_reasoning_claim"
        ),
        "group": {
            "elements": len(BINDINGS),
            "generators": len(S4_GENERATORS),
            "inverse_checks": inverse_checks,
            "independent_composition_checks": independent_composition_checks,
            "associativity_checks": associativity_checks,
            "ordered_generator_pairs": generator_pairs,
            "noncommuting_ordered_pairs": noncommuting_pairs,
            "abelian_control_order_collapses": abelian_order_collapses,
            "coordinate_roundtrip_checks": coordinate_roundtrip_checks,
            "transport_equivariance_checks": transport_equivariance_checks,
            "s4_table_sha256": s4_table_sha256,
            "z24_table_sha256": z24_table_sha256,
        },
        "order_witnesses": treatment_destinations,
        "executor": executor,
        "resources": resources,
        "gates": gates,
        "decision": decision,
    }
    report["payload_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    write_exclusive(args.out, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
