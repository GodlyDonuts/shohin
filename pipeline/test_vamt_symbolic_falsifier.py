from __future__ import annotations

import copy

import vamt_symbolic_falsifier as vamt


def test_truth_table_has_exact_complete_coverage():
    result = vamt.exact_local_coverage(vamt.TRUTH_TABLE)
    assert result == {
        "contexts": 400,
        "correct": 400,
        "mismatches": [],
        "pass": True,
    }


def test_local_poison_is_detected():
    poisoned = dict(vamt.TRUTH_TABLE)
    poisoned[("add", 9, 9, 1)] = (8, 1)
    result = vamt.exact_local_coverage(poisoned)
    assert not result["pass"]
    assert result["correct"] == 399


def test_tied_machine_replays_all_width_one_and_two_pairs():
    result = vamt.exhaustive_small_width_replay(
        vamt.TiedDigitMachine(vamt.TRUTH_TABLE)
    )
    assert result["pass"]
    assert result["admitted_checks"] == 15_205
    assert result["rejected_negative_subtractions"] == 4_995
    assert not result["signed_subtraction_supported"]


def test_induction_certificate_counts_every_position_context():
    result = vamt.induction_cell_certificate(vamt.TRUTH_TABLE, maximum_width=64)
    assert result["pass"]
    assert result["checks"] == 832_000
    assert result["checks"] == result["expected_checks"]


def test_induction_certificate_fails_on_poison():
    poisoned = dict(vamt.TRUTH_TABLE)
    poisoned[("sub", 0, 9, 1)] = (1, 1)
    result = vamt.induction_cell_certificate(poisoned, maximum_width=2)
    assert not result["pass"]


def test_untied_unseen_position_is_not_identified():
    result = vamt.untied_nonidentifiability_witness(observed_positions=4)
    assert result["pass"]
    assert result["observed_contexts"] == 1_600
    assert result["untied_context_parameters_at_confirmation_width"] == 2_000


def test_pointer_equivariance_certificate():
    result = vamt.pointer_equivariance_certificate()
    assert result["pass"]
    assert result["base_copy"] == ["3", "2", "1", "0", "0"]
    assert result["inclusive_span"] == {"start": 2, "end": 4}
    assert result["unequal_width_zero_padding"]


def test_pointer_rejects_out_of_range_address():
    try:
        vamt.addressed_copy(("a",), (1,))
    except ValueError as error:
        assert "outside source" in str(error)
    else:
        raise AssertionError("out-of-range pointer was accepted")


def test_span_reader_rejects_nondigit_and_bad_endpoints():
    for tokens, span in (
        (("1", "x"), vamt.OperandSpan(0, 1)),
        (("1",), vamt.OperandSpan(1, 1)),
    ):
        try:
            vamt.read_operand_lsd(tokens, span, width=2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid operand span was accepted")


def test_serializer_is_exhaustive_and_keeps_terminal_carry():
    result = vamt.serializer_certificate(vamt.TiedSerializer(vamt.SERIALIZER_TABLE))
    assert result["pass"]
    assert result["atomic_contexts"] == 40
    assert result["exhaustive_checks"] == 11_110
    assert result["terminal_carry_example"] == "107593"


def test_serializer_poison_is_detected():
    poisoned = dict(vamt.SERIALIZER_TABLE)
    poisoned[(0, 7, 0)] = (None, 0, 0)
    serializer = vamt.TiedSerializer(poisoned)
    result = vamt.serializer_certificate(serializer, maximum_exhaustive_width=2)
    assert not result["pass"]


def test_parameter_ledger_matches_theory_and_strict_limit():
    result = vamt.parameter_ledger()
    assert result["minimal"]["components"] == {
        "r4_style_compiler_and_transition_table": 300_493,
        "boundary_head": 771,
        "digit_key": 32_768,
        "slot_start_end_queries": 32_768,
        "event_start_end_queries": 65_536,
        "serializer": 1_741,
    }
    assert result["minimal"]["additional_parameters"] == 434_077
    assert result["minimal"]["total_parameters"] == 125_515_741
    assert result["minimal"]["strict_headroom"] == 24_484_258
    assert result["maximum"]["components"] == {
        "late_lora": 7_815_168,
        "compiler_front": 1_928_344,
        "compiler_decoder": 3_031_578,
        "executor": 5_394_720,
        "serializer": 2_007_362,
    }
    assert result["maximum"]["additional_parameters"] == 20_177_172
    assert result["maximum"]["total_parameters"] == 145_258_836
    assert result["maximum"]["strict_headroom"] == 4_741_163
    assert result["pass"]


def test_bounded_resource_ledger_is_exact_and_charges_targets():
    result = vamt.bounded_resource_ledger()
    assert result["pass"]
    assert result["retained_program_and_private_state_bytes"] == 67
    assert result["output_bytes"] == 51
    assert result["fixed_executor_cycles"] == 128
    assert result["minimal_serializer_matrix_macs"] == 28_288
    assert result["maximum_executor_matrix_macs"] == 679_477_248
    assert result["structured_target_bits_must_be_charged"]


def test_symbolic_runtime_is_candid_about_external_table_execution():
    result = vamt.symbolic_runtime_accounting(
        vamt.TiedDigitMachine(vamt.TRUTH_TABLE)
    )
    assert result["pass"]
    assert all(result["semantic_forbidden_zero"].values())
    assert all(result["fixed_runtime_nonzero"].values())
    assert result["external_symbolic_execution_counted"]
    assert not result["future_neural_host_boundary_proven"]


def test_report_is_deterministic_and_candid_about_collapse():
    first = vamt.build_report()
    second = vamt.build_report()
    assert first == second
    assert first["all_pass"]
    assert first["payload_sha256"] == second["payload_sha256"]
    assert not first["sections"]["mealy_collapse"]["novel_primitive_claim_allowed"]
    without_hash = copy.deepcopy(first)
    digest = without_hash.pop("payload_sha256")
    assert digest == vamt.payload_sha256(without_hash)
