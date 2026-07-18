from __future__ import annotations

import copy

import pytest

import vamt_full_machine_falsifier as vamt


def canonical_machine() -> vamt.FullProgramMachine:
    return vamt.FullProgramMachine(
        vamt.canonical_candidate_executor_table(),
        vamt.canonical_candidate_serializer_table(),
    )


def test_frozen_tokenizer_digit_contract():
    assert vamt.TOKENIZER_SHA256 == (
        "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
    )
    assert dict(vamt.CANDIDATE_TOKEN_BY_DIGIT) == {
        0: 28,
        1: 29,
        2: 30,
        3: 31,
        4: 32,
        5: 33,
        6: 34,
        7: 35,
        8: 36,
        9: 37,
    }
    assert dict(vamt.CANDIDATE_DIGIT_BY_TOKEN) == {
        token: digit for digit, token in vamt.CANDIDATE_TOKEN_BY_DIGIT.items()
    }


def test_candidate_tables_have_complete_local_coverage():
    executor = vamt.local_executor_certificate(
        vamt.canonical_candidate_executor_table()
    )
    serializer = vamt.serializer_context_certificate(
        vamt.canonical_candidate_serializer_table()
    )
    assert executor == {
        "contexts": 400,
        "correct": 400,
        "mismatches": [],
        "reference_uses_candidate_table": False,
        "pass": True,
    }
    assert serializer == {
        "contexts": 40,
        "correct": 40,
        "mismatches": [],
        "reference_uses_candidate_table": False,
        "pass": True,
    }


def test_full_board_is_exact_and_charges_every_cycle():
    result = vamt.full_program_certificate(
        vamt.canonical_candidate_executor_table(),
        vamt.canonical_candidate_serializer_table(),
    )
    assert result["pass"]
    assert result["executions"] == 152
    assert result["category_counts"] == {
        "malformed_span": 5,
        "missing_halt": 1,
        "post_halt_masking": 16,
        "seven_add_bound": 1,
        "terminal_carry_reuse": 1,
        "width_sweep": 128,
    }
    assert result["negative_subtractions"] == 32
    assert result["negative_subtractions_execute_all_17_phases"]
    assert result["executor_cycles"] == 152 * 136
    assert result["serializer_cycles"] == 152 * 17
    assert not result["candidate_reference_tables_shared"]


def test_terminal_carry_is_consumed_and_reused_across_operations():
    case = next(
        case
        for case in vamt.build_full_program_board()
        if case.name == "terminal_carry_reuse"
    )
    result = canonical_machine().run(case.source_tokens, case.program)
    assert result.status == "ACCEPT"
    assert result.output_digits == (2, 7)
    assert result.state.accumulator[:3] == [7, 2, 0]
    assert result.state.carry_or_borrow == 0
    assert result.ledger.add_transition_lookups == 34
    assert result.ledger.executor_cycles == 136
    assert result.ledger.serializer_cycles == 17


def test_negative_subtraction_executes_all_cells_then_rejects():
    source = (vamt.REFERENCE_TOKEN_BY_DIGIT[1], vamt.REFERENCE_TOKEN_BY_DIGIT[2])
    program = vamt._padded_program(
        (
            vamt.Instruction("LOAD", 0, 0),
            vamt.Instruction("SUB", 1, 1),
            vamt.Instruction("HALT"),
        )
    )
    result = canonical_machine().run(source, program)
    assert result.status == "REJECT"
    assert result.output_token_ids == ()
    assert result.state.invalid
    assert result.ledger.sub_transition_lookups == 17
    assert result.ledger.executor_cycles == 136
    assert result.ledger.serializer_cycles == 17
    assert vamt.reference_execute(source, program).status == "REJECT"


def test_seven_add_maximum_bound_uses_the_seventeenth_digit():
    case = next(
        case
        for case in vamt.build_full_program_board()
        if case.name == "seven_add_maximum_bound"
    )
    result = canonical_machine().run(case.source_tokens, case.program)
    assert result.status == "ACCEPT"
    assert "".join(map(str, result.output_digits)) == "69999999999999993"
    assert result.ledger.add_transition_lookups == 7 * 17


def test_post_halt_program_bytes_are_masked_but_cycles_are_charged():
    cases = [
        case
        for case in vamt.build_full_program_board()
        if case.category == "post_halt_masking"
    ]
    results = [canonical_machine().run(case.source_tokens, case.program) for case in cases]
    assert len(cases) == 16
    assert {result.output_digits for result in results} == {(7,)}
    assert {result.status for result in results} == {"ACCEPT"}
    assert {result.ledger.executor_cycles for result in results} == {136}
    assert {result.ledger.active_executor_cycles for result in results} == {18}
    assert {result.ledger.masked_executor_cycles for result in results} == {118}


def test_all_malformed_spans_reject_without_host_repair():
    cases = [
        case
        for case in vamt.build_full_program_board()
        if case.category == "malformed_span"
    ]
    assert len(cases) == 5
    for case in cases:
        result = canonical_machine().run(case.source_tokens, case.program)
        assert result.status == "REJECT", case.name
        assert result.output_token_ids == ()
        assert result.state.invalid
        assert result.ledger.parser_repairs == 0
        assert vamt.reference_execute(case.source_tokens, case.program).status == "REJECT"


def test_corrupted_retained_cursor_sets_sticky_invalid():
    source = (vamt.REFERENCE_TOKEN_BY_DIGIT[4],)
    program = vamt._padded_program(
        (vamt.Instruction("LOAD", 0, 0), vamt.Instruction("HALT"))
    )
    machine = canonical_machine()
    state = vamt.MachineState(source_cursor=vamt.PAD_CURSOR)
    ledger = vamt.RuntimeLedger()
    machine._step_executor(state, source, program, ledger)
    assert state.invalid
    assert ledger.invalidations == 1
    assert ledger.active_executor_cycles == 1

    before = copy.deepcopy(state)
    machine._step_executor(state, source, program, ledger)
    assert state == before
    assert ledger.masked_executor_cycles == 1


def test_missing_halt_rejects_after_all_eight_slots():
    case = next(
        case for case in vamt.build_full_program_board() if case.name == "missing_halt"
    )
    result = canonical_machine().run(case.source_tokens, case.program)
    assert result.status == "REJECT"
    assert result.state.invalid
    assert result.state.halted
    assert result.ledger.active_executor_cycles == 136
    assert result.ledger.masked_executor_cycles == 0


def test_malformed_program_cardinality_and_source_bound_fail_closed():
    machine = canonical_machine()
    with pytest.raises(ValueError, match="exactly eight"):
        machine.run((28,), (vamt.Instruction("HALT"),))
    with pytest.raises(ValueError, match="256-token"):
        machine.run((28,) * 257, (vamt.Instruction("HALT"),) * 8)


def test_candidate_executor_poison_is_detected_by_independent_oracle():
    executor = vamt.canonical_candidate_executor_table()
    executor[("ADD", 9, 9, 0)] = (9, 0)
    assert not vamt.local_executor_certificate(executor)["pass"]
    assert not vamt.full_program_certificate(
        executor, vamt.canonical_candidate_serializer_table()
    )["pass"]


def test_candidate_serializer_poison_is_detected_by_independent_oracle():
    serializer = vamt.canonical_candidate_serializer_table()
    serializer[(0, 0, 0)] = (1, 0, 0, 9)
    assert not vamt.serializer_context_certificate(serializer)["pass"]
    assert not vamt.full_program_certificate(
        vamt.canonical_candidate_executor_table(), serializer
    )["pass"]


def test_joint_poison_cannot_mutate_or_cancel_the_reference():
    result = vamt.poison_independence_certificate()
    assert result == {
        "canonical_pass": True,
        "executor_poison_rejected": True,
        "serializer_poison_rejected": True,
        "joint_poison_rejected": True,
        "mutable_global_truth_table_exists": False,
        "pass": True,
    }
    with pytest.raises(TypeError):
        vamt.REFERENCE_DIGIT_BY_TOKEN[28] = 9
    assert not hasattr(vamt, "TRUTH_TABLE")
    assert not hasattr(vamt, "SERIALIZER_TABLE")


def test_parameter_target_resource_and_compute_ledgers_are_exact():
    parameters = vamt.parameter_ledger()
    assert parameters["pass"]
    assert parameters["additional_parameters"] == 187_332
    assert parameters["total_parameters"] == 125_268_996
    assert parameters["strict_headroom"] == 24_731_003
    assert not parameters["globally_minimal_claim"]

    targets = vamt.target_information_ledger()
    assert targets["pass"]
    assert targets["compiler_target_bits_per_program"] == 144
    assert targets["executor_serializer_target_bits"] == 2_280
    assert targets["context_and_target_bits"] == 6_520

    resources = vamt.bounded_resource_ledger()
    assert resources["pass"]
    assert resources["packed_program_private_bytes"] == 31
    assert resources["byte_addressed_program_private_bytes"] == 53
    assert resources["compiler_phase_including_source_codebook_bytes"] == 750_764
    assert resources["base_activation_allocation_peak"] == "UNKNOWN_MUST_BE_MEASURED"
    assert not resources["exact_full_peak_claim_allowed"]

    compute = vamt.compute_ledger()
    assert compute["pass"]
    assert compute["compiler_matrix_macs"] == 19_738_624
    assert compute["executor_serializer_dense_one_hot_equivalent_macs"] == 663_680
    assert compute["total_non_base_dense_equivalent_macs"] == 20_402_304


def test_runtime_accounting_is_candid_about_external_execution():
    result = vamt.symbolic_runtime_accounting()
    assert result["pass"]
    assert result["output_digits"] == [2, 7]
    assert all(result["forbidden_runtime_calls_zero"].values())
    assert result["all_executor_cycles_charged"]
    assert result["all_serializer_cycles_charged"]
    assert result["oracle_injected_external_execution"]
    assert not result["autonomous_capability"]


def test_report_is_deterministic_and_neural_authority_remains_false():
    first = vamt.build_report()
    second = vamt.build_report()
    assert first == second
    assert first["all_pass"]
    assert first["oracle_injected_external_execution"]
    assert not first["autonomous_capability"]
    assert not first["neural_preregistration_authorized"]
    assert not first["sections"]["finite_machine_and_control_boundary"][
        "novel_primitive_claim_allowed"
    ]
    without_hash = copy.deepcopy(first)
    digest = without_hash.pop("payload_sha256")
    assert digest == vamt.payload_sha256(without_hash)
