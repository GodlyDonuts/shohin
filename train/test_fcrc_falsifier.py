#!/usr/bin/env python3
"""Deterministic tests for the R12 FCRC CPU structural falsifier."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import fcrc_falsifier as fcrc  # noqa: E402


_ADDRESS_MUTABLE_STATE = {"flip": 0}
_LATE_MUTABLE_STATE = {"flip": 0}
_TERMINAL_MUTABLE_STATE = {"flip": 0}


def mutable_global_address_source(
    source: fcrc.Source,
    cursor: int,
) -> fcrc.AddressRead:
    return fcrc.AddressRead(
        source.operation,
        (source.left_digits[cursor] + _ADDRESS_MUTABLE_STATE["flip"]) % 10,
        source.right_digits[cursor],
    )


def mutable_global_late_actuator(local_result: fcrc.LocalResult) -> fcrc.Emission:
    return fcrc.Emission(
        "digit",
        (local_result.digit + _LATE_MUTABLE_STATE["flip"]) % 10,
    )


def mutable_global_terminal_actuator(packet: fcrc.Packet) -> fcrc.Emission:
    return fcrc.Emission(
        "terminal_carry",
        (packet.carry + _TERMINAL_MUTABLE_STATE["flip"]) % 2,
    )


class PacketAndSurfaceTests(unittest.TestCase):
    def test_packet_is_exactly_three_fixed_slots(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(fcrc.Packet)),
            fcrc.PACKET_FIELDS,
        )
        self.assertEqual(fcrc.Packet.__slots__, fcrc.PACKET_FIELDS)
        self.assertTrue(fcrc.packet_schema_audit(fcrc.Packet)["valid"])
        self.assertFalse(fcrc.packet_schema_audit(fcrc.GrowingPacket)["valid"])

    def test_packet_capacity_accounts_for_cursor_growth(self) -> None:
        self.assertEqual(
            {width: fcrc.packet_bits(width) for width in (2, 4, 6, 8)},
            {2: 5, 4: 6, 6: 6, 8: 7},
        )
        with self.assertRaises(ValueError):
            fcrc.packet_bits(0)

    def test_forbidden_context_is_absent_from_local_and_step_surfaces(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(fcrc.address_source).parameters),
            fcrc.ADDRESS_SOURCE_PARAMETERS,
        )
        self.assertEqual(
            tuple(inspect.signature(fcrc.position_blind_local_operator).parameters),
            fcrc.LOCAL_OPERATOR_PARAMETERS,
        )
        self.assertEqual(
            tuple(inspect.signature(fcrc.late_residual_actuator).parameters),
            fcrc.LATE_ACTUATOR_PARAMETERS,
        )
        self.assertEqual(
            tuple(inspect.signature(fcrc.terminal_actuator).parameters),
            fcrc.TERMINAL_ACTUATOR_PARAMETERS,
        )
        step_parameters = set(inspect.signature(fcrc.fcrc_step).parameters)
        self.assertTrue(
            {
                "generated_history",
                "history",
                "kv",
                "result_prefix",
                "result_tape",
                "terminal",
                "width",
            }.isdisjoint(step_parameters)
        )

    def test_packet_validation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            fcrc.Packet(-1, 0, fcrc.PHASE_RUN)
        with self.assertRaises(ValueError):
            fcrc.Packet(0, 2, fcrc.PHASE_RUN)
        with self.assertRaises(ValueError):
            fcrc.Packet(0, 0, "write")

        class PayloadInt(int):
            payload = b"x" * 1024

        with self.assertRaises(TypeError):
            fcrc.Packet(0, PayloadInt(1), fcrc.PHASE_RUN)
        with self.assertRaises(TypeError):
            fcrc.Source("add", (PayloadInt(1), 0), (0, 0))
        with self.assertRaises(TypeError):
            fcrc.AddressRead("add", PayloadInt(1), 0)
        with self.assertRaises(TypeError):
            fcrc.LocalResult(0, PayloadInt(1))


class LocalFactorizationTests(unittest.TestCase):
    def test_all_400_local_cells_match_decimal_arithmetic(self) -> None:
        seen = 0
        for operation, left, right, carry in fcrc.iter_local_keys():
            observed = fcrc.position_blind_local_operator(operation, left, right, carry)
            if operation == "add":
                total = left + right + carry
                expected = fcrc.LocalResult(total % 10, total // 10)
            else:
                total = left - right - carry
                expected = fcrc.LocalResult((total + 10) % 10, int(total < 0))
            self.assertEqual(observed, expected)
            seen += 1
        self.assertEqual(seen, fcrc.LOCAL_TABLE_ENTRIES)

    def test_context_invariance_is_exact(self) -> None:
        audit = fcrc.context_invariance_audit(fcrc.reference_context_operator)
        self.assertEqual(audit["local_equivalence_groups"], 400)
        self.assertEqual(audit["total_violations"], 0)
        self.assertEqual(
            audit["violations_by_variant"],
            {
                "cursor": 0,
                "terminal": 0,
                "width_6": 0,
                "width_8": 0,
                "result_prefix": 0,
                "history": 0,
            },
        )

    def test_each_forbidden_dependency_has_a_detectable_negative(self) -> None:
        terminal = fcrc.context_invariance_audit(fcrc.terminal_zero_leak)
        cursor = fcrc.context_invariance_audit(fcrc.cursor_leak)
        width_6 = fcrc.context_invariance_audit(fcrc.width_6_leak)
        width_8 = fcrc.context_invariance_audit(fcrc.width_8_leak)
        prefix = fcrc.context_invariance_audit(fcrc.result_prefix_leak)
        history = fcrc.context_invariance_audit(fcrc.generated_history_leak)
        self.assertEqual(terminal["violations_by_variant"]["terminal"], 100)
        self.assertEqual(cursor["violations_by_variant"]["cursor"], 400)
        self.assertEqual(width_6["violations_by_variant"]["width_6"], 400)
        self.assertEqual(width_8["violations_by_variant"]["width_8"], 400)
        self.assertEqual(prefix["violations_by_variant"]["result_prefix"], 400)
        self.assertEqual(history["violations_by_variant"]["history"], 400)

    def test_stateful_or_alternate_callable_cannot_receive_structural_go(self) -> None:
        mutable = {"flip": 0}

        def global_state_leak(context: fcrc.ProbeContext) -> fcrc.LocalResult:
            result = fcrc.reference_context_operator(context)
            return fcrc.LocalResult(
                (result.digit + mutable["flip"]) % 10,
                result.next_carry,
            )

        behavior = fcrc.context_invariance_audit(global_state_leak)
        self.assertEqual(behavior["total_violations"], 0)
        static = fcrc.callable_dependency_audit(
            global_state_leak,
            expected=fcrc.reference_context_operator,
            allowed_globals={"reference_context_operator"},
        )
        self.assertFalse(static["valid"])
        self.assertFalse(static["checks"]["exact_callable_identity"])
        self.assertFalse(static["checks"]["no_closure_cells"])
        self.assertTrue(fcrc.fixed_operator_dependency_audit()["valid"])

    def test_address_and_actuator_mutable_global_exploits_fail_admission(self) -> None:
        source = fcrc.source_from_values("add", 95, 8, 2)
        cases = (
            (
                "address_source",
                "address",
                mutable_global_address_source,
                _ADDRESS_MUTABLE_STATE,
            ),
            (
                "late_residual_actuator",
                "late_actuator",
                mutable_global_late_actuator,
                _LATE_MUTABLE_STATE,
            ),
            (
                "terminal_actuator",
                "terminal_actuator",
                mutable_global_terminal_actuator,
                _TERMINAL_MUTABLE_STATE,
            ),
        )
        for public_name, audit_name, candidate, state in cases:
            with self.subTest(public_name=public_name):
                state["flip"] = 0
                with mock.patch.object(fcrc, public_name, candidate):
                    first = fcrc.emission_trace(fcrc.rollout(source))
                    audit = fcrc.fixed_operator_dependency_audit()[audit_name]
                    self.assertFalse(audit["valid"])
                    self.assertFalse(audit["checks"]["exact_callable_identity"])
                    self.assertFalse(audit["checks"]["no_mutable_globals"])
                    state["flip"] = 1
                    second = fcrc.emission_trace(fcrc.rollout(source))
                    self.assertNotEqual(first, second)
                state["flip"] = 0

    def test_every_address_and_actuator_state_channel_fails_closed(self) -> None:
        originals = {
            "address_source": fcrc.address_source,
            "late_residual_actuator": fcrc.late_residual_actuator,
            "terminal_actuator": fcrc.terminal_actuator,
        }
        audit_names = {
            "address_source": "address",
            "late_residual_actuator": "late_actuator",
            "terminal_actuator": "terminal_actuator",
        }
        expected_parameters = {
            "address_source": fcrc.ADDRESS_SOURCE_PARAMETERS,
            "late_residual_actuator": fcrc.LATE_ACTUATOR_PARAMETERS,
            "terminal_actuator": fcrc.TERMINAL_ACTUATOR_PARAMETERS,
        }

        def variants(public_name: str) -> dict[str, object]:
            original = originals[public_name]
            closure_value = 0
            nonlocal_value = 0
            if public_name == "address_source":

                def closure(source: fcrc.Source, cursor: int) -> fcrc.AddressRead:
                    if closure_value:
                        raise AssertionError
                    return original(source, cursor)

                def nonlocal_mutation(
                    source: fcrc.Source, cursor: int
                ) -> fcrc.AddressRead:
                    nonlocal nonlocal_value
                    nonlocal_value += 1
                    return original(source, cursor)

                def default(source: fcrc.Source, cursor: int = 0) -> fcrc.AddressRead:
                    return original(source, cursor)

                def attribute(source: fcrc.Source, cursor: int) -> fcrc.AddressRead:
                    return original(source, cursor)

            elif public_name == "late_residual_actuator":

                def closure(local_result: fcrc.LocalResult) -> fcrc.Emission:
                    if closure_value:
                        raise AssertionError
                    return original(local_result)

                def nonlocal_mutation(
                    local_result: fcrc.LocalResult,
                ) -> fcrc.Emission:
                    nonlocal nonlocal_value
                    nonlocal_value += 1
                    return original(local_result)

                def default(
                    local_result: fcrc.LocalResult = fcrc.LocalResult(0, 0),
                ) -> fcrc.Emission:
                    return original(local_result)

                def attribute(local_result: fcrc.LocalResult) -> fcrc.Emission:
                    return original(local_result)

            else:

                def closure(packet: fcrc.Packet) -> fcrc.Emission:
                    if closure_value:
                        raise AssertionError
                    return original(packet)

                def nonlocal_mutation(packet: fcrc.Packet) -> fcrc.Emission:
                    nonlocal nonlocal_value
                    nonlocal_value += 1
                    return original(packet)

                def default(
                    packet: fcrc.Packet = fcrc.Packet(0, 0, fcrc.PHASE_FINAL),
                ) -> fcrc.Emission:
                    return original(packet)

                def attribute(packet: fcrc.Packet) -> fcrc.Emission:
                    return original(packet)

            attribute.hidden_state = {"flip": 0}
            return {
                "closure": closure,
                "nonlocal": nonlocal_mutation,
                "default": default,
                "attribute": attribute,
            }

        failed_check = {
            "closure": "no_closure_cells",
            "nonlocal": "no_nonlocals",
            "default": "no_defaults",
            "attribute": "no_function_attributes",
        }
        for public_name in originals:
            for variant_name, candidate in variants(public_name).items():
                with self.subTest(
                    public_name=public_name,
                    variant_name=variant_name,
                ):
                    self.assertEqual(
                        tuple(inspect.signature(candidate).parameters),
                        expected_parameters[public_name],
                    )
                    with mock.patch.object(fcrc, public_name, candidate):
                        audit = fcrc.fixed_operator_dependency_audit()[
                            audit_names[public_name]
                        ]
                    self.assertFalse(audit["valid"])
                    self.assertFalse(audit["checks"][failed_check[variant_name]])

    def test_terminal_carry_no_go_has_exact_finite_witness(self) -> None:
        audit = fcrc.terminal_carry_no_go_audit()
        self.assertEqual(audit["training_terminal_zero_cells"], 100)
        self.assertEqual(audit["training_agreements"], 100)
        self.assertEqual(audit["omitted_terminal_one_cells"], 100)
        self.assertEqual(audit["omitted_disagreements"], 100)
        self.assertTrue(audit["observationally_indistinguishable_on_training_support"])
        self.assertTrue(audit["different_on_every_omitted_terminal_carry_cell"])

    def test_terminal_and_nonterminal_updates_share_the_same_law(self) -> None:
        audit = fcrc.terminal_update_audit()
        self.assertEqual(
            audit,
            {
                "cells": 400,
                "terminal_carry_updates_exact": 400,
                "nonterminal_carry_updates_exact": 400,
                "terminal_nonterminal_emissions_equal": 400,
            },
        )


class AutonomousCycleTests(unittest.TestCase):
    def test_one_rollout_has_no_result_tape_and_emits_terminal_carry(self) -> None:
        source = fcrc.source_from_values("add", 95, 8, 2)
        records = fcrc.rollout(source)
        self.assertEqual(
            fcrc.emission_trace(records),
            (("digit", 3), ("digit", 0), ("terminal_carry", 1)),
        )
        self.assertEqual(records[-1].after, fcrc.Packet(2, 1, fcrc.PHASE_HALT))
        self.assertEqual(
            tuple(field.name for field in fields(records[-1].after)),
            fcrc.PACKET_FIELDS,
        )

    def test_emission_observer_cannot_feed_back(self) -> None:
        source = fcrc.source_from_values("add", 9_876, 6_789, 4)
        side_a: list[object] = []
        side_b: list[object] = []

        def observer_a(index: int, emission: fcrc.Emission) -> object:
            side_a.append((index, emission))
            return ("fake-token", index)

        def observer_b(index: int, emission: fcrc.Emission) -> object:
            side_b.append({"fake_kv": [emission.value] * 20})
            return side_b[-1]

        first = fcrc.rollout(source, observer=observer_a)
        second = fcrc.rollout(source, observer=observer_b)
        self.assertEqual(fcrc.emission_trace(first), fcrc.emission_trace(second))
        self.assertNotEqual(side_a, side_b)

    def test_exhaustive_two_step_gate(self) -> None:
        audit = fcrc.autonomous_two_step_audit()
        self.assertEqual(audit["cases"], 20_000)
        self.assertEqual(audit["exact"], 20_000)
        self.assertEqual(audit["carry_boundary_cases"], 9_000)

    def test_width_and_value_ood_mechanics_are_exact(self) -> None:
        audit = fcrc.ood_rollout_audit()
        self.assertEqual(audit["cases"], 500)
        self.assertEqual(audit["exact"], 500)
        self.assertEqual(audit["history_independent"], 500)
        self.assertEqual(audit["packet_schema_stable"], 500)
        self.assertTrue(audit["balance"]["balanced"])
        for row in audit["balance"]["by_regime"].values():
            self.assertEqual(
                row,
                {
                    "add_carry_0": 25,
                    "add_carry_1": 25,
                    "sub_no_intermediate_borrow": 25,
                    "sub_with_intermediate_borrow": 25,
                },
            )
        self.assertEqual(set(audit["by_regime"]), set(fcrc.REGIME_SPECS))
        for row in audit["by_regime"].values():
            self.assertEqual(row, {"cases": 100, "exact": 100})

    def test_fit_and_value_ood_scalar_supports_are_strictly_disjoint(self) -> None:
        audit = fcrc.mechanics_scalar_support_audit()
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["declared_fit_value_ood_disjoint"])
        self.assertTrue(all(audit["declared_regime_supports_valid"].values()))
        self.assertTrue(
            all(not rows for rows in audit["declared_intersections"].values())
        )
        self.assertEqual(audit["observed_fit_value_ood_intersection"], [])
        self.assertEqual(audit["operand_membership_violations"], [])
        for regime, left, right, _source in fcrc.mechanics_cases():
            self.assertTrue(fcrc.scalar_in_declared_support(regime, left))
            self.assertTrue(fcrc.scalar_in_declared_support(regime, right))

    def test_invalid_phase_transitions_fail_closed(self) -> None:
        source = fcrc.source_from_values("add", 10, 20, 2)
        with self.assertRaises(fcrc.ContractError):
            fcrc.fcrc_step(source, fcrc.Packet(0, 0, fcrc.PHASE_FINAL))
        with self.assertRaises(fcrc.ContractError):
            fcrc.fcrc_step(source, fcrc.Packet(2, 0, fcrc.PHASE_RUN))
        with self.assertRaises(fcrc.ContractError):
            fcrc.fcrc_step(source, fcrc.Packet(2, 0, fcrc.PHASE_HALT))


class CollapseAndResourceTests(unittest.TestCase):
    def test_lookup_table_collapse_is_admitted(self) -> None:
        audit = fcrc.local_table_collapse_audit()
        self.assertEqual(audit["entries"], 400)
        self.assertEqual(audit["lookup_exact"], 400)
        self.assertTrue(audit["finite_lookup_table_extensionally_equivalent"])
        self.assertEqual(audit["minimum_raw_output_bits"], 2_000)

    def test_each_packet_field_has_a_behavioral_collision_witness(self) -> None:
        audit = fcrc.minimal_packet_witnesses()
        self.assertTrue(audit["carry_required"])
        self.assertTrue(audit["cursor_required"])
        self.assertTrue(audit["phase_required"])

    def test_resource_vector_makes_oracle_and_memory_costs_explicit(self) -> None:
        resources = fcrc.resource_accounting()
        self.assertTrue(resources["cpu_positive_is_hardcoded_oracle"])
        self.assertTrue(resources["cpu_positive_disqualified_as_learned_reasoning"])
        self.assertEqual(resources["result_tape_bits"], 0)
        self.assertEqual(resources["generated_token_kv_causal_bits"], 0)
        self.assertEqual(resources["packet_bits_by_width"]["8"], 7)
        self.assertEqual(
            resources["hardcoded_learned_module_substitutes_by_width"]["8"], 25
        )
        self.assertEqual(resources["external_execution_calls_by_width"]["8"], 34)
        self.assertEqual(resources["read_only_source_control_symbols"], 2)
        self.assertIn(
            "carry_only_rank8_writer_reader",
            resources["required_neural_controls"],
        )
        self.assertTrue(resources["ordinary_rnn_simulable"])
        self.assertTrue(resources["lookup_table_simulable"])


class FullContractTests(unittest.TestCase):
    def test_full_report_is_deterministic_and_every_gate_passes(self) -> None:
        first = fcrc.run_audit()
        second = fcrc.run_audit()
        self.assertEqual(
            fcrc.canonical_json_bytes(first),
            fcrc.canonical_json_bytes(second),
        )
        self.assertTrue(first["mechanics_contract_satisfied"])
        self.assertTrue(all(first["gates"].values()))
        self.assertEqual(first["r3_post_hoc_localization"]["digit_site_reached"], 16)
        self.assertEqual(
            first["r3_post_hoc_localization"][
                "digit_site_reached_and_full_state_exact"
            ],
            14,
        )
        self.assertIn("not a neural", first["go_boundary"])
        json.loads(fcrc.canonical_json_bytes(first))

    def test_module_has_no_accelerator_network_or_subprocess_import(self) -> None:
        tree = ast.parse(Path(fcrc.__file__).read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden = {
            "numpy",
            "requests",
            "socket",
            "subprocess",
            "tensorflow",
            "torch",
        }
        self.assertTrue(forbidden.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
