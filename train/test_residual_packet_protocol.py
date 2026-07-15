#!/usr/bin/env python3
"""Pure tests for the frozen source-deleted residual packet protocol."""

import unittest

try:
    import residual_packet_protocol as protocol
except ModuleNotFoundError:
    from train import residual_packet_protocol as protocol


class OperationGrammarTests(unittest.TestCase):
    def test_all_operations_round_trip(self):
        for operation in (("add", 2), ("multiply", 7), ("subtract", 25)):
            with self.subTest(operation=operation):
                rendered = protocol.render_operation(operation)
                self.assertEqual(protocol.parse_operation(rendered), operation)

    def test_operation_parser_rejects_every_noncanonical_form(self):
        malformed = (
            "add 0",
            "add 01",
            "add +1",
            "add -1",
            "Add 1",
            "plus 1",
            "multiply by 2",
            "subtract: 2",
            "add 2 ",
            " add 2",
            "add 2 # comment",
            "add\t2",
            "add 2\n",
            "",
            None,
            b"add 2",
        )
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(protocol.parse_operation(text))

    def test_structured_operation_validation_is_not_forgiving(self):
        malformed = (
            ("divide", 2),
            ("add", 0),
            ("add", -1),
            ("add", True),
            ("add", "2"),
            ("add",),
            ("add", 2, 3),
            "add 2",
            None,
        )
        for operation in malformed:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    protocol.normalize_operation(operation)


class PacketGrammarTests(unittest.TestCase):
    def setUp(self):
        self.plan = (("add", 2), ("multiply", 3), ("subtract", 4))
        self.packet = "State: 10\nPlan: add 2; multiply 3; subtract 4"

    def test_packet_round_trip_and_integer_states(self):
        self.assertEqual(protocol.canonical_packet(10, self.plan), self.packet)
        self.assertEqual(
            protocol.parse_packet(self.packet),
            {"state": 10, "plan": self.plan},
        )
        self.assertEqual(
            protocol.parse_packet("State: 0\nPlan: add 2"),
            {"state": 0, "plan": (("add", 2),)},
        )
        self.assertEqual(
            protocol.parse_packet("State: -17\nPlan: subtract 2"),
            {"state": -17, "plan": (("subtract", 2),)},
        )

    def test_only_surrounding_ascii_whitespace_is_ignored(self):
        self.assertEqual(
            protocol.parse_packet(" \t\n" + self.packet + "\r\n\v"),
            {"state": 10, "plan": self.plan},
        )
        self.assertEqual(
            protocol.parse_packet("State: 1\nPlan: add 2 "),
            {"state": 1, "plan": (("add", 2),)},
        )
        self.assertIsNone(protocol.parse_packet("\u00a0" + self.packet))
        self.assertIsNone(protocol.parse_packet(self.packet + "\u00a0"))

    def test_packet_parser_rejects_malformed_state_and_plan_grammar(self):
        malformed = (
            "State: 01\nPlan: add 2",
            "State: -0\nPlan: add 2",
            "State: +1\nPlan: add 2",
            "State: 1.0\nPlan: add 2",
            "State: 1,000\nPlan: add 2",
            "State: 1\nPlan:",
            "state: 1\nPlan: add 2",
            "State: 1\nplan: add 2",
            "State: 1\r\nPlan: add 2",
            "State: 1 \nPlan: add 2",
            "State: 1\nPlan: add 2;multiply 3",
            "State: 1\nPlan: add 2;  multiply 3",
            "State: 1\nPlan: add 2;; multiply 3",
            "State: 1\nPlan: add 0",
            "State: 1\nPlan: add 02",
            "State: 1\nPlan: add +2",
            "State: 1\nPlan: add -2",
            "State: 1\nPlan: plus 2",
            "State: 1\nPlan: multiply by 2",
            "State: 1\nPlan: add 2 # comment",
            "Packet:\nState: 1\nPlan: add 2",
            "State: 1\nPlan: add 2\nAnswer: 3",
            "State: 1\nPlan: add 2\nExtra: text",
            "explanation\nState: 1\nPlan: add 2",
            "State: 1\nState: 1\nPlan: add 2",
            "",
            "   ",
            None,
            b"State: 1\nPlan: add 2",
        )
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(protocol.parse_packet(text))

    def test_packet_rendering_rejects_invalid_structured_values(self):
        bad_calls = (
            (True, self.plan),
            ("10", self.plan),
            (10, ()),
            (10, "add 2"),
            (10, (("add", 0),)),
        )
        for state, plan in bad_calls:
            with self.subTest(state=state, plan=plan):
                with self.assertRaises(ValueError):
                    protocol.canonical_packet(state, plan)

    def test_repeated_operations_are_grammatical_programs(self):
        text = "State: 8\nPlan: add 2; add 2"
        self.assertEqual(
            protocol.parse_packet(text),
            {"state": 8, "plan": (("add", 2), ("add", 2))},
        )


class AnswerAndChannelTests(unittest.TestCase):
    def test_answer_round_trip(self):
        for value in (-19, 0, 42):
            with self.subTest(value=value):
                text = protocol.canonical_answer(value)
                self.assertEqual(protocol.parse_answer(text), value)
                self.assertEqual(
                    protocol.parse_controller_output(text),
                    {"channel": "answer", "answer": value},
                )

    def test_answer_rejects_malformed_or_mixed_channels(self):
        malformed = (
            "Answer: 01",
            "Answer: -0",
            "Answer: +1",
            "answer: 1",
            "Answer=1",
            "Answer: 1 extra",
            "Answer: 1\nState: 1\nPlan: add 2",
            "explanation\nAnswer: 1",
            "",
            None,
        )
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(protocol.parse_answer(text))
                self.assertIsNone(protocol.parse_controller_output(text))

    def test_packet_channel_is_tagged(self):
        self.assertEqual(
            protocol.parse_controller_output("State: 3\nPlan: multiply 4"),
            {"channel": "packet", "state": 3, "plan": (("multiply", 4),)},
        )


class PromptAndTransitionTests(unittest.TestCase):
    def setUp(self):
        self.source = "Begin with 10. Execute this sequence from left to right: add 2; multiply by 3."
        self.packet = {
            "state": 10,
            "plan": (("add", 2), ("multiply", 3), ("subtract", 4)),
        }

    def test_compiler_prompt_is_exact(self):
        self.assertEqual(
            protocol.compiler_prompt(self.source),
            "Problem: "
            + self.source
            + "\nCompile only the execution packet.\nPacket:",
        )

    def test_compiler_prompt_rejects_multiline_nonascii_and_untrimmed_sources(self):
        malformed = (
            "",
            " leading",
            "trailing ",
            "line one\nPacket:\nState: 9",
            "tab\tinside",
            "nonascii-\u00e9",
            None,
            10,
        )
        for source in malformed:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    protocol.compiler_prompt(source)

    def test_update_prompt_is_exact_and_source_free(self):
        prompt = protocol.update_prompt(self.packet, 999)
        self.assertEqual(
            prompt,
            "Packet:\n"
            "State: 10\n"
            "Plan: add 2; multiply 3; subtract 4\n"
            "Observed result: 999\n"
            "Next packet:",
        )
        self.assertNotIn(self.source, prompt)
        self.assertNotIn("Problem:", prompt)

    def test_update_prompt_reparses_raw_packet_instead_of_carrying_extra_text(self):
        canonical = protocol.canonical_packet(self.packet["state"], self.packet["plan"])
        self.assertEqual(
            protocol.update_prompt("\n" + canonical + "\n", 12),
            protocol.update_prompt(self.packet, 12),
        )
        with self.assertRaises(ValueError):
            protocol.update_prompt(canonical + "\nProblem: retained source", 12)
        with self.assertRaises(ValueError):
            protocol.update_prompt({"state": 10, "plan": self.packet["plan"], "source": self.source}, 12)
        with self.assertRaises(ValueError):
            protocol.update_prompt(self.packet, "12")

    def test_expected_update_copies_observation_and_deletes_only_the_head(self):
        self.assertEqual(
            protocol.expected_update(self.packet, 999),
            "State: 999\nPlan: multiply 3; subtract 4",
        )
        one_step = {"state": -5, "plan": (("add", 2),)}
        self.assertEqual(protocol.expected_update(one_step, -777), "Answer: -777")

    def test_exact_update_validation_rejects_wrong_state_deletion_and_channel(self):
        valid = "State: 12\nPlan: multiply 3; subtract 4"
        self.assertTrue(protocol.update_is_exact(self.packet, 12, valid))
        self.assertTrue(protocol.update_is_exact(self.packet, 12, "\n" + valid + "\n"))
        self.assertEqual(
            protocol.parse_exact_update(self.packet, 12, valid),
            {
                "channel": "packet",
                "state": 12,
                "plan": (("multiply", 3), ("subtract", 4)),
            },
        )
        malformed = (
            "State: 13\nPlan: multiply 3; subtract 4",
            "State: 12\nPlan: add 2; multiply 3; subtract 4",
            "State: 12\nPlan: subtract 4",
            "State: 12\nPlan: multiply 3; subtract 4; add 2",
            "Answer: 12",
            valid + "\nAnswer: 12",
            valid + "\ncomment",
        )
        for response in malformed:
            with self.subTest(response=response):
                self.assertFalse(protocol.update_is_exact(self.packet, 12, response))
                self.assertIsNone(protocol.parse_exact_update(self.packet, 12, response))

    def test_terminal_update_requires_only_the_exact_answer_channel(self):
        packet = {"state": 7, "plan": (("subtract", 2),)}
        self.assertTrue(protocol.update_is_exact(packet, 5, "Answer: 5"))
        for response in (
            "State: 5\nPlan: subtract 2",
            "State: 5\nPlan: add 1",
            "Answer: 7",
            "answer: 5",
            "Answer: 5 extra",
        ):
            with self.subTest(response=response):
                self.assertFalse(protocol.update_is_exact(packet, 5, response))


class ArithmeticAndTrajectoryTests(unittest.TestCase):
    def test_operations_and_trajectory(self):
        self.assertEqual(protocol.apply_operation(10, ("add", 2)), 12)
        self.assertEqual(protocol.apply_operation(12, "multiply", 3), 36)
        self.assertEqual(protocol.apply_operation(36, ("subtract", 5)), 31)
        self.assertEqual(protocol.apply_operation(3, ("subtract", 8)), -5)
        plan = [["add", 2], ["multiply", 3], ["subtract", 5]]
        self.assertEqual(protocol.trajectory(10, plan), (10, 12, 36, 31))
        self.assertEqual(protocol.compute_trajectory(10, plan), (10, 12, 36, 31))
        self.assertEqual(protocol.apply_plan(10, plan), 31)
        self.assertEqual(protocol.trajectory(10, ()), (10,))
        self.assertEqual(plan, [["add", 2], ["multiply", 3], ["subtract", 5]])

    def test_packet_trajectory_contains_every_residual_and_terminal_answer(self):
        plan = (("add", 2), ("multiply", 3), ("subtract", 5))
        self.assertEqual(
            protocol.packet_trajectory(10, plan),
            (
                "State: 10\nPlan: add 2; multiply 3; subtract 5",
                "State: 12\nPlan: multiply 3; subtract 5",
                "State: 36\nPlan: subtract 5",
                "Answer: 31",
            ),
        )

    def test_arithmetic_helpers_reject_noninteger_state(self):
        for state in (True, "10", 1.5, None):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    protocol.apply_operation(state, ("add", 2))


class SourceRenderingTests(unittest.TestCase):
    def setUp(self):
        self.plan = (("add", 3), ("multiply", 4), ("subtract", 5))
        self.clauses = "add 3; multiply by 4; subtract 5"

    def test_frozen_template_ids_and_clauses(self):
        self.assertEqual(
            protocol.TRAIN_SOURCE_TEMPLATE_IDS,
            ("train_0", "train_1", "train_2", "train_3"),
        )
        self.assertEqual(protocol.RESERVED_SOURCE_TEMPLATE_ID, "reserved")
        self.assertEqual(protocol.SOURCE_TEMPLATE_IDS[-1], "reserved")
        self.assertEqual(protocol.render_source_clauses(self.plan), self.clauses)

    def test_all_training_sources_are_exact_ascii_and_exclude_reserved_wording(self):
        expected = {
            "train_0": "Begin with 10. Execute this sequence from left to right: " + self.clauses + ".",
            "train_1": "Take 10 as the running value. Perform, in sequence: " + self.clauses + ".",
            "train_2": "The starting number is 10. Make these changes in order: " + self.clauses + ".",
            "train_3": "Set the running total to 10. Follow the listed commands: " + self.clauses + ".",
        }
        for template_id, source in expected.items():
            with self.subTest(template_id=template_id):
                self.assertEqual(protocol.render_source(10, self.plan, template_id), source)
                self.assertTrue(source.isascii())
                self.assertNotIn("Initialize the value", source)

    def test_reserved_renderer_is_the_preregistered_wording(self):
        self.assertEqual(
            protocol.render_source(10, self.plan, "reserved"),
            "Initialize the value to 10. Apply these instructions in order: "
            + self.clauses
            + ".",
        )

    def test_source_templates_are_immutable_and_validate_inputs(self):
        with self.assertRaises(TypeError):
            protocol.SOURCE_TEMPLATES["reserved"] = "changed"
        for initial_state in (0, -1, True, "10"):
            with self.subTest(initial_state=initial_state):
                with self.assertRaises(ValueError):
                    protocol.render_source(initial_state, self.plan, "train_0")
        with self.assertRaises(ValueError):
            protocol.render_source(10, self.plan, "unknown")
        with self.assertRaises(ValueError):
            protocol.render_source(10, (), "train_0")


class NativeExecutorTests(unittest.TestCase):
    def test_native_problem_work_prompts_are_exact(self):
        self.assertEqual(
            protocol.format_atomic_prompt(14, "add", 3),
            "Problem: Compute 14 plus 3.\nWork:",
        )
        self.assertEqual(
            protocol.atomic_prompt(14, ("multiply", 3)),
            "Problem: Compute 14 times 3.\nWork:",
        )
        self.assertEqual(
            protocol.format_atomic_prompt(-4, ("subtract", 3)),
            "Problem: Compute -4 minus 3.\nWork:",
        )
        with self.assertRaises(ValueError):
            protocol.format_atomic_prompt(14, "remainder", 3)

    def test_first_line_parser_uses_only_last_integer_on_first_nonempty_line(self):
        cases = (
            (" 12 * 7 = 84\nAnswer: 84", 84),
            ("\n  -11\nNext: 3", -11),
            ("6,999\n2", 6999),
            ("step 1 gives 4 and step 2 gives 9\n100", 9),
            ("Answer: 0", 0),
            ("No number here\n999", None),
            ("malformed 6,99\n999", None),
            ("value 1.5\n5", None),
            ("value x12\n5", None),
            ("value 12x\n5", None),
            ("", None),
            (None, None),
        )
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(protocol.parse_first_line_integer(response), expected)
                self.assertEqual(protocol.parse_first_line_final(response), expected)


if __name__ == "__main__":
    unittest.main()
