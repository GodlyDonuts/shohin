#!/usr/bin/env python3
"""Focused CPU contracts for the hard-bit DWEPR learner."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


TRAIN = Path(__file__).resolve().parent
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from wgrq_state_machine import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    FLIP,
    PACKET_BITS,
    ROTATE,
    HardBitDWEPRLearner,
    WGRQReader,
    WGRQWriter,
    deserialize_packet,
    make_scale_mask,
    serialize_packet,
    straight_through_hard_bits,
)


class ArchitectureTests(unittest.TestCase):
    def test_exact_gelu_mlp_geometry_and_fp32_parameter_count(self):
        model = HardBitDWEPRLearner()
        transition_in, transition_out = model.transition_mlp[0], model.transition_mlp[2]
        readout_in, readout_out = model.readout_mlp[0], model.readout_mlp[2]
        self.assertEqual((transition_in.in_features, transition_in.out_features), (32, 64))
        self.assertEqual((transition_out.in_features, transition_out.out_features), (64, 15))
        self.assertEqual((readout_in.in_features, readout_in.out_features), (30, 64))
        self.assertEqual((readout_out.in_features, readout_out.out_features), (64, 1))
        self.assertIsInstance(model.transition_mlp[1], nn.GELU)
        self.assertIsInstance(model.readout_mlp[1], nn.GELU)
        self.assertEqual(model.parameter_count, EXPECTED_PARAMETER_COUNT)
        self.assertEqual(model.parameter_count, 5_136)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        self.assertEqual(
            set(model.state_dict()),
            {
                "transition_mlp.0.weight",
                "transition_mlp.0.bias",
                "transition_mlp.2.weight",
                "transition_mlp.2.bias",
                "readout_mlp.0.weight",
                "readout_mlp.0.bias",
                "readout_mlp.2.weight",
                "readout_mlp.2.bias",
            },
        )

    def test_public_masks_are_exact_prefixes(self):
        for n in (4, 6, 8, 16):
            mask = make_scale_mask(n)
            self.assertEqual(mask.shape, (PACKET_BITS,))
            self.assertTrue(torch.equal(mask[: n - 1], torch.ones(n - 1)))
            self.assertTrue(torch.equal(mask[n - 1 :], torch.zeros(PACKET_BITS - n + 1)))
        for n in (3, 5, 17):
            with self.assertRaises(ValueError):
                make_scale_mask(n)


class HardBitTests(unittest.TestCase):
    def test_forward_threshold_and_straight_through_gradient(self):
        logits = torch.tensor(
            [-1.0, 0.0, 1.0] + [3.0] * (PACKET_BITS - 3),
            requires_grad=True,
        )
        packet, probabilities = straight_through_hard_bits(logits, make_scale_mask(4))
        self.assertTrue(torch.equal(packet, torch.tensor([0.0, 1.0, 1.0] + [0.0] * 12)))
        self.assertTrue(torch.all((probabilities > 0) & (probabilities < 1)))
        packet.sum().backward()
        self.assertTrue(torch.all(logits.grad[:3] > 0))
        self.assertTrue(torch.equal(logits.grad[3:], torch.zeros(12)))

    def test_every_active_transition_is_hard_and_padding_preserves_state(self):
        torch.manual_seed(7)
        model = HardBitDWEPRLearner()
        scale_mask = make_scale_mask(8)
        packet = model.initial_packet(scale_mask)
        first, _ = model.transition(packet, ROTATE, scale_mask)
        second, _ = model.transition(first, FLIP, scale_mask)
        self.assertTrue(bool(((first == 0) | (first == 1)).all()))
        self.assertTrue(bool(((second == 0) | (second == 1)).all()))
        self.assertTrue(bool((second[scale_mask == 0] == 0).all()))
        padded = model.rollout(
            packet,
            torch.tensor([ROTATE, FLIP]),
            scale_mask,
            torch.tensor([1, 0]),
        )
        self.assertTrue(torch.equal(padded.packet, first))
        self.assertEqual(padded.probabilities.shape, (2, PACKET_BITS))
        self.assertTrue(torch.equal(padded.transition_mask, torch.tensor([True, False])))


class PacketBoundaryTests(unittest.TestCase):
    def test_serialization_is_exactly_15_immutable_binary_values(self):
        mask = make_scale_mask(6)
        tensor = torch.tensor([1, 0, 1, 1, 0] + [0] * 10, dtype=torch.float32)
        packet = serialize_packet(tensor, mask)
        self.assertIsInstance(packet, tuple)
        self.assertEqual(len(packet), PACKET_BITS)
        self.assertEqual(deserialize_packet(packet, mask).tolist(), list(packet))
        with self.assertRaises(ValueError):
            deserialize_packet(packet[:-1], mask)
        with self.assertRaises(ValueError):
            deserialize_packet(packet[:-1] + (2,), mask)
        leaking = list(packet)
        leaking[-1] = 1
        with self.assertRaisesRegex(ValueError, "masked"):
            deserialize_packet(leaking, mask)
        bad_tensor = tensor.clone()
        bad_tensor[-1] = 1
        with self.assertRaisesRegex(ValueError, "masked"):
            serialize_packet(bad_tensor, mask)

    def test_writer_reader_reuse_does_not_mutate_or_retain_source(self):
        torch.manual_seed(11)
        model = HardBitDWEPRLearner().eval()
        mask = make_scale_mask(8)
        writer = WGRQWriter(model, mask)
        packet = writer.write(torch.tensor([ROTATE, FLIP, ROTATE, ROTATE]))
        frozen = tuple(packet)
        reader = WGRQReader(model, mask)
        first = reader.read_probability(packet, torch.tensor([], dtype=torch.long))
        second = reader.read_probability(packet, torch.tensor([ROTATE, ROTATE]))
        self.assertTrue(0.0 <= first <= 1.0)
        self.assertTrue(0.0 <= second <= 1.0)
        self.assertEqual(packet, frozen)
        self.assertEqual(set(vars(reader)), {"model", "scale_mask"})
        self.assertEqual(set(vars(writer)), {"model", "scale_mask"})

    def test_reader_apis_have_no_source_cache_or_path_parameter(self):
        forbidden = {"source", "history", "cache", "path", "oracle", "simulator", "verifier"}
        for function in (WGRQReader.__init__, WGRQReader.read_logits, WGRQReader.read_probability):
            parameters = set(inspect.signature(function).parameters) - {"self"}
            for parameter in parameters:
                lowered = parameter.lower()
                self.assertTrue(
                    all(token not in lowered for token in forbidden),
                    (function.__qualname__, parameter),
                )


if __name__ == "__main__":
    unittest.main()
