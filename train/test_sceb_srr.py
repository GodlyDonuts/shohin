#!/usr/bin/env python3
"""Unit tests for SCEB host-exec parsing and SRR digit codec."""

import unittest

import torch

from eval_typed_controller_host_exec import parse_model_step, parse_ops, parse_register
from generate_typed_controller_v1 import apply_op
from stateful_residual_register import digits_to_int, int_to_digits


class HostExecParseTest(unittest.TestCase):
    def test_register(self):
        state, ops, cursor = parse_register(
            "Problem: state=99; ops=multiply 13 | subtract 5; cursor=0\nWork:"
        )
        self.assertEqual(state, 99)
        self.assertEqual(ops, [("multiply", 13), ("subtract", 5)])
        self.assertEqual(cursor, 0)

    def test_step(self):
        p = parse_model_step("multiply 13 -> 1287; cursor=1; done=0\n")
        self.assertEqual(p["op"], "multiply")
        self.assertEqual(p["arg"], 13)
        self.assertEqual(p["next_claimed"], 1287)
        self.assertEqual(apply_op(99, p["op"], p["arg"]), 1287)

    def test_horner_ops(self):
        self.assertEqual(parse_ops("horner 10 3 | horner 10 4"), [("horner", 10003), ("horner", 10004)])


class SRRCodecTest(unittest.TestCase):
    def test_roundtrip(self):
        for v in (0, 7, 99, 1287, -42, 10**7 - 1):
            d = int_to_digits(v, 8)
            self.assertEqual(digits_to_int(d), v)


if __name__ == "__main__":
    unittest.main()
