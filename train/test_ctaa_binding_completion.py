from __future__ import annotations

from dataclasses import replace
import itertools

import pytest
import torch

from ctaa_binding_completion import (
    ACTION_COUNT,
    BINDINGS,
    COMPILER_WIDTH,
    FACTORIZED_MACS,
    GLOBAL_MACS,
    RELATION_SLOT_COUNT,
    READOUT_PARAMETERS,
    BindingCompletionError,
    FactorizedBindingReadout,
    GlobalStructuredBindingReadout,
    SingleSlotFullBindingProbe,
    WholePermutationReadout,
    binding_class_targets,
    factorized_loss,
    materialize_factorized,
    materialize_whole,
    permutation_parity,
    readout_resource_receipt,
    split_a4_rows,
    whole_loss,
)
from ctaa_compiler_training import TokenizedCompilerRow
from finalize_ctaa_binding_completion import evaluate_seed


def compiler_row(
    binding: tuple[int, int, int, int],
    serial: int,
) -> TokenizedCompilerRow:
    opcode_schedule = (0, 1, 2, 3, 4, *([0] * 36))
    schedule = tuple(
        4 if event == 4 else binding[event] for event in opcode_schedule
    )
    return TokenizedCompilerRow(
        program_ids=(serial + 2, *binding),
        query_ids=(7,),
        action_cards=((0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1)),
        opcode_to_card=binding,
        initial_state=(0, 1, 2),
        opcode_schedule=opcode_schedule,
        schedule=schedule,
        query_position=0,
    )


def test_a4_row_split_is_exact_locally_balanced_and_disjoint() -> None:
    rows = [
        compiler_row(binding, repeat * len(BINDINGS) + index)
        for repeat in range(2)
        for index, binding in enumerate(BINDINGS)
    ]
    split = split_a4_rows(rows)
    assert len(split.train_even) == len(split.confirmation_odd) == 24
    assert split.audit["per_binding"] == 2
    assert split.audit["train_local_marginals"] == ((6, 6, 6, 6),) * 4
    assert split.audit["confirmation_local_marginals"] == ((6, 6, 6, 6),) * 4
    assert all(permutation_parity(row.opcode_to_card) == 0 for row in split.train_even)
    assert all(
        permutation_parity(row.opcode_to_card) == 1
        for row in split.confirmation_odd
    )


def test_a4_row_split_rejects_imbalance_and_conflicting_source() -> None:
    rows = [compiler_row(binding, index) for index, binding in enumerate(BINDINGS)]
    with pytest.raises(BindingCompletionError, match="multiplicity"):
        split_a4_rows(rows[:-1])
    conflict = replace(
        compiler_row(BINDINGS[1], len(BINDINGS)),
        program_ids=rows[0].program_ids,
    )
    with pytest.raises(BindingCompletionError, match="conflicting"):
        split_a4_rows([*rows, conflict])


def test_binding_class_targets_round_trip_all_s4() -> None:
    bindings = torch.tensor(BINDINGS)
    targets = binding_class_targets(bindings)
    assert targets.tolist() == list(range(len(BINDINGS)))
    assert [permutation_parity(binding) for binding in BINDINGS].count(0) == 12
    with pytest.raises(BindingCompletionError, match="leaves S4"):
        binding_class_targets(torch.tensor([[0, 0, 2, 3]]))


def test_readouts_are_exactly_parameter_matched_and_mac_matched() -> None:
    receipt = readout_resource_receipt()
    assert receipt["factorized_parameters"] == READOUT_PARAMETERS
    assert receipt["global_structured_parameters"] == READOUT_PARAMETERS
    assert receipt["parameter_gap"] == 0
    assert receipt["factorized_macs"] == FACTORIZED_MACS
    assert receipt["global_structured_macs"] == GLOBAL_MACS
    assert float(receipt["relative_mac_gap"]) < 0.002
    assert receipt["whole_control_role"] == "support_starved_lookup_negative_only"


def test_readouts_materialize_only_valid_permutations() -> None:
    slots = torch.randn(7, RELATION_SLOT_COUNT, COMPILER_WIDTH)
    factorized_logits = FactorizedBindingReadout()(slots)
    global_logits = GlobalStructuredBindingReadout()(slots)
    whole_logits = WholePermutationReadout()(slots)
    probe_logits = SingleSlotFullBindingProbe(2)(slots)
    factorized = materialize_factorized(factorized_logits)
    global_structured = materialize_factorized(global_logits)
    whole = materialize_whole(whole_logits)
    probe = materialize_factorized(probe_logits)
    expected = set(BINDINGS)
    assert all(tuple(row.tolist()) in expected for row in factorized)
    assert all(tuple(row.tolist()) in expected for row in global_structured)
    assert all(tuple(row.tolist()) in expected for row in whole)
    assert all(tuple(row.tolist()) in expected for row in probe)


def test_factorized_heads_have_no_cross_slot_input_and_losses_reach_weights() -> None:
    treatment = FactorizedBindingReadout()
    slots = torch.randn(
        3,
        RELATION_SLOT_COUNT,
        COMPILER_WIDTH,
        requires_grad=True,
    )
    targets = torch.tensor(BINDINGS[:3])
    logits = treatment(slots)
    baseline = logits.detach().clone()
    perturbed = slots.detach().clone()
    perturbed[:, 1] += 5.0
    changed = treatment(perturbed).detach()
    assert torch.equal(baseline[:, 0], changed[:, 0])
    assert not torch.equal(baseline[:, 1], changed[:, 1])
    loss = factorized_loss(logits, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in treatment.parameters())
    assert slots.grad is not None

    global_control = GlobalStructuredBindingReadout()
    global_slots = slots.detach().clone()
    global_baseline = global_control(global_slots).detach()
    global_slots[:, 1] += 5.0
    global_changed = global_control(global_slots).detach()
    assert not torch.equal(global_baseline[:, 0], global_changed[:, 0])
    global_loss = factorized_loss(global_control(slots.detach()), targets)
    global_loss.backward()
    assert all(
        parameter.grad is not None for parameter in global_control.parameters()
    )

    control = WholePermutationReadout()
    control_logits = control(slots.detach())
    control_loss = whole_loss(control_logits, targets)
    control_loss.backward()
    assert all(parameter.grad is not None for parameter in control.parameters())


def test_factorized_readout_is_exactly_biequivariant() -> None:
    model = FactorizedBindingReadout()
    slots = torch.randn(2, RELATION_SLOT_COUNT, COMPILER_WIDTH)
    baseline = model(slots)
    for opcode_order in itertools.permutations(range(ACTION_COUNT)):
        opcode_index = torch.tensor(opcode_order)
        for card_order in itertools.permutations(range(ACTION_COUNT)):
            card_index = torch.tensor(card_order)
            transformed = torch.cat(
                (
                    slots[:, opcode_index],
                    slots[:, ACTION_COUNT + card_index],
                ),
                dim=1,
            )
            expected = baseline[:, opcode_index][:, :, card_index]
            torch.testing.assert_close(
                model(transformed),
                expected,
                rtol=1e-6,
                atol=1e-7,
            )


def test_factorized_assignment_matches_exhaustive_permutation_score() -> None:
    logits = torch.randn(11, ACTION_COUNT, ACTION_COUNT)
    predicted = materialize_factorized(logits)
    for row_index, row_logits in enumerate(logits):
        scores = {
            binding: sum(
                float(row_logits[opcode, card])
                for opcode, card in enumerate(binding)
            )
            for binding in itertools.permutations(range(ACTION_COUNT))
        }
        assert tuple(predicted[row_index].tolist()) == max(scores, key=scores.get)


def test_final_decision_gate_is_fail_closed_per_seed() -> None:
    admission = {
        "minimum_confirmation_factorized_exact": 0.75,
        "minimum_factorized_advantage": 0.10,
        "maximum_single_slot_exact": 0.10,
        "minimum_chimera_exact": 0.75,
    }
    row = {
        "seed": 11,
        "confirmation_metrics": {
            "factorized": {
                "projected_binding_exact": 0.90,
                "raw_binding_exact": 0.88,
                "raw_assignment_valid": 0.95,
            },
            "global_structured": {"projected_binding_exact": 0.60},
        },
        "program_packet_metrics": {
            "factorized": {
                "program_exact": 0.85,
                "opcode_persistent_excitation": 1.0,
                "binding_counterfactual_effect": 1.0,
            }
        },
        "single_slot_probe_metrics": {
            f"single_slot_{index}": {"projected_binding_exact": 0.05}
            for index in range(4)
        },
        "two_slot_chimera_metrics": {
            "factorized": {"projected_binding_exact": 0.82}
        },
    }
    passed = evaluate_seed(row, {"all_arms_qualified": True}, admission)
    assert passed["seed_pass"]
    failed = evaluate_seed(
        {
            **row,
            "single_slot_probe_metrics": {
                **row["single_slot_probe_metrics"],
                "single_slot_2": {"projected_binding_exact": 0.20},
            },
        },
        {"all_arms_qualified": True},
        admission,
    )
    assert not failed["seed_pass"]
    assert not failed["gates"]["single_slot_leakage"]
