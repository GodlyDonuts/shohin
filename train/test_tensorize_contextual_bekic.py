from __future__ import annotations

import torch

from contextual_bekic_graph_machine import (
    ContextualBekicGraphMachine,
    LateContextualProgramQuery,
)
from contextual_relation_primitive_compiler import (
    ContextualRelationPrimitiveCompiler,
)
from contextualize_bekic_program import contextualize_simultaneous_packet
from contrastive_bekic_program_orbits import (
    evaluate_simultaneous,
    generate_orbit,
    select_machine_input,
    transplant_constants,
    transplant_program,
)
from tensorize_contextual_bekic import (
    tensorize_contextual_packets,
    tensorize_target_environment,
)


def _execute(
    source_packets: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contextual = [
        contextualize_simultaneous_packet(
            packet,
            seed=seed + index,
        )
        for index, packet in enumerate(source_packets)
    ]
    tensors = tensorize_contextual_packets(contextual)
    compiler = ContextualRelationPrimitiveCompiler()
    compiled = compiler(
        tensors.witness_left,
        tensors.witness_right,
        tensors.witness_output,
        tensors.witness_mask,
        tensors.argument_mask,
        tensors.object_mask,
        hard=True,
    )
    active = tensors.packet.slot_arity.ge(0)
    assert bool(compiled.identifiable[active].all())
    assert not bool(compiled.identifiable[~active].any())
    machine = ContextualBekicGraphMachine()
    query = LateContextualProgramQuery(
        variable=torch.zeros(len(source_packets), dtype=torch.long),
        position=torch.zeros(len(source_packets), dtype=torch.long),
    )
    result = machine(tensors.packet, compiled.discrete_assignment, query)
    assert bool(result.converged.all())
    return result.terminal_variables, result.answer


def _expected(
    packet: dict[str, object],
) -> torch.Tensor:
    environment = evaluate_simultaneous(packet)
    return tensorize_target_environment(
        environment,
        [str(item) for item in packet["program"]["variables"]],
        cardinality=int(packet["cardinality"]),
    )


def test_p_pprime_and_equivalent_rewrite_execute_exactly() -> None:
    row = generate_orbit(split="train", seed=2026072313)
    packets = [
        select_machine_input(row, arm=arm, form="simultaneous")
        for arm in ("p", "p_prime", "p_eq")
    ]
    observed, answers = _execute(packets, seed=7300)
    expected = torch.stack([_expected(packet) for packet in packets])
    assert torch.equal(observed, expected)
    assert torch.equal(answers, expected[:, 0, 0])
    assert not torch.equal(observed[0], observed[1])


def test_program_and_constant_transplants_execute_donor_semantics() -> None:
    rows = [
        generate_orbit(split="development", seed=2026072314, index=index)
        for index in range(20)
    ]
    donor, recipient = next(
        (left, right)
        for left in rows
        for right in rows
        if left is not right
        and left["axes"]["cardinality"] == right["axes"]["cardinality"]
    )
    program = transplant_program(donor, recipient)["simultaneous"]
    constants = transplant_constants(donor, recipient)["simultaneous"]
    observed, _answers = _execute(
        [program, constants],
        seed=7400,
    )
    expected = torch.stack([_expected(program), _expected(constants)])
    assert torch.equal(observed, expected)


def test_fresh_contextualization_seed_does_not_change_execution() -> None:
    row = generate_orbit(split="development", seed=2026072315)
    packet = select_machine_input(row, arm="p", form="simultaneous")
    first, _ = _execute([packet], seed=7500)
    second, _ = _execute([packet], seed=7600)
    assert torch.equal(first, second)
