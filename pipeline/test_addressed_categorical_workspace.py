import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.addressed_categorical_workspace import (
    FIELD_SIZE,
    AddressedCategoricalWorkspace,
    AddressedContinuousTrackSModel,
    AnswerMotorControl,
    AffineEvent,
    CategoricalTrackSModel,
    DenseCategoricalTrackSModel,
    GRUTrackSModel,
    PacketTokenTransformerTrackSModel,
    SourceRetainedTrackSModel,
    apply_affine_event,
    canonical_json_bytes,
    hard_categorical,
    literal_packet_update,
    literal_packet_update_array,
    payload_sha256,
    packet_to_symbols,
    read_packet,
    run_symbolic_dimension,
    run_symbolic_falsifier,
    symbols_to_packet,
    trainable_parameters,
    validate_literal_one_hot,
    write_report,
)


class SymbolicWorkspaceTests(unittest.TestCase):
    def test_affine_event_writes_exactly_one_register(self):
        state = (3, 5, 7)
        event = AffineEvent(destination=1, source=2, alpha=2, beta=4, gamma=6)
        observed = literal_packet_update(state, event)
        expected = apply_affine_event(state, event)
        self.assertEqual(observed, expected)
        self.assertEqual(observed[0], state[0])
        self.assertEqual(observed[2], state[2])
        self.assertEqual(read_packet(observed, 1), expected[1])

    def test_dimension_two_exhaustive_control(self):
        report = run_symbolic_dimension(2, coefficient_values=(0, 1, 16))
        self.assertFalse(report["pass"])
        self.assertFalse(report["gates"]["full_coefficient_field_exhausted"])
        self.assertEqual(report["states_checked"], FIELD_SIZE**2)
        self.assertTrue(report["narrow_collision"]["collision"])
        self.assertTrue(report["narrow_collision"]["separated"])

    def test_full_symbolic_report_is_deterministic_and_hash_bound(self):
        first = run_symbolic_falsifier(coefficient_values=(0, 1, 16))
        second = run_symbolic_falsifier(coefficient_values=(0, 1, 16))
        self.assertEqual(first, second)
        self.assertFalse(first["pass"])
        self.assertEqual(first["payload_sha256"], payload_sha256(first))
        self.assertEqual([item["dimension"] for item in first["dimensions"]], [2, 3])

    def test_report_write_round_trip(self):
        report = run_symbolic_falsifier(coefficient_values=(0, 1, 16))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_report(report, path)
            observed = json.loads(path.read_text())
            self.assertEqual(canonical_json_bytes(observed), canonical_json_bytes(report))
            self.assertEqual(observed["payload_sha256"], payload_sha256(observed))

    def test_overcomplete_literal_update_preserves_sentinel(self):
        packets = torch.tensor([[1, 2, 3, 6], [4, 5, 6, 15]]).numpy()
        event = AffineEvent(destination=1, source=2, alpha=2, beta=3, gamma=4)
        updated = literal_packet_update_array(packets, event)
        self.assertTrue((updated[:, 0] == packets[:, 0]).all())
        self.assertTrue((updated[:, 2:] == packets[:, 2:]).all())

    def test_report_writer_rejects_stale_hash(self):
        report = run_symbolic_falsifier(coefficient_values=(0, 1))
        report["pass"] = not report["pass"]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_report(report, Path(temporary) / "stale.json")


class NeuralWorkspaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AddressedCategoricalWorkspace()

    def test_frozen_shohin_parameter_ledger(self):
        ledger = self.model.parameter_ledger()
        self.assertEqual(
            ledger["components"],
            {
                "source_projector": 39_236,
                "event_projector": 39_236,
                "updater": 10_129,
                "bridge": 4_352,
            },
        )
        self.assertEqual(ledger["total"], 92_953)

    def test_hard_codes_are_literal_one_hot(self):
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
        observed = hard_categorical(logits, straight_through=False)
        self.assertTrue(torch.equal(observed, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])))

    def test_straight_through_forward_is_exact_in_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(2048, 17, dtype=dtype, requires_grad=True)
            observed = hard_categorical(logits, straight_through=True)
            validate_literal_one_hot(observed)
            self.assertTrue(torch.equal(observed.sum(dim=-1), torch.ones(2048, dtype=dtype)))
            self.assertTrue(torch.all((observed == 0) | (observed == 1)))
            (observed * torch.randn_like(observed)).sum().backward()
            self.assertIsNotNone(logits.grad)

    def test_uint8_persistence_round_trip_and_rejection(self):
        packet = self.model.encode_source(torch.randn(4, 576), straight_through=False)
        symbols = packet_to_symbols(packet)
        self.assertEqual(symbols.dtype, torch.uint8)
        self.assertEqual(symbols.nelement() * symbols.element_size(), 16)
        reconstructed = symbols_to_packet(
            symbols, 17, dtype=packet.dtype, device=packet.device,
        )
        self.assertTrue(torch.equal(packet, reconstructed))
        corrupted = packet.clone()
        corrupted[0, 0, 0] = 0.25
        with self.assertRaises(ValueError):
            packet_to_symbols(corrupted)

    def test_update_cannot_modify_unaddressed_registers(self):
        batch = 5
        source = torch.randn(batch, 576)
        event = torch.randn(batch, 576)
        packet = self.model.encode_source(source, straight_through=False)
        event_code = self.model.encode_event(event, straight_through=False)
        address = torch.tensor([0, 1, 2, 3, 1])
        updated = self.model.update(
            packet, event_code, address, straight_through=False,
        )
        for row in range(batch):
            for register in range(4):
                if register != int(address[row]):
                    self.assertTrue(torch.equal(updated[row, register], packet[row, register]))
        self.assertTrue(torch.equal(updated.sum(dim=-1), torch.ones(batch, 4)))

    def test_symbol_update_persists_only_uint8_state(self):
        batch = 5
        packet = self.model.encode_source_symbols(torch.randn(batch, 576))
        event = self.model.encode_event_symbols(torch.randn(batch, 576))
        address = torch.tensor([0, 1, 2, 3, 1])
        updated = self.model.update_symbols(packet, event, address)
        self.assertEqual(updated.dtype, torch.uint8)
        self.assertEqual(updated.nelement() * updated.element_size(), batch * 4)
        for row in range(batch):
            for register in range(4):
                if register != int(address[row]):
                    self.assertEqual(int(updated[row, register]), int(packet[row, register]))
        self.assertEqual(self.model.packet_delta_symbols(updated).shape, (batch, 64))

    def test_shapes_and_validation(self):
        packet = self.model.encode_source(torch.randn(2, 576), straight_through=False)
        event = self.model.encode_event(torch.randn(2, 576), straight_through=False)
        updated = self.model.update(packet, event, torch.tensor([1, 3]), straight_through=False)
        self.assertEqual(updated.shape, (2, 4, 17))
        self.assertEqual(self.model.packet_delta(updated).shape, (2, 64))
        with self.assertRaises(ValueError):
            self.model.update(packet, event, torch.tensor([1, 4]), straight_through=False)

    def test_straight_through_path_carries_gradient(self):
        self.model.train()
        source = torch.randn(3, 576)
        event = torch.randn(3, 576)
        packet = self.model.encode_source(source)
        event_code = self.model.encode_event(event)
        updated = self.model.update(packet, event_code, torch.tensor([0, 1, 2]))
        loss = self.model.packet_delta(updated).square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.source_projector.weight.grad)
        self.assertIsNotNone(self.model.event_projector.weight.grad)
        self.assertIsNotNone(self.model.updater[0].weight.grad)

    def test_unaddressed_loss_has_no_updater_or_event_gradient(self):
        self.model.train()
        source = torch.randn(3, 576)
        event = torch.randn(3, 576)
        packet = self.model.encode_source(source)
        event_code = self.model.encode_event(event)
        updated = self.model.update(packet, event_code, torch.zeros(3, dtype=torch.int64))
        weights = torch.randn_like(updated[:, 1:])
        (updated[:, 1:] * weights).sum().backward()
        self.assertGreater(float(self.model.source_projector.weight.grad.abs().sum()), 0)
        event_gradient = self.model.event_projector.weight.grad
        updater_gradient = self.model.updater[0].weight.grad
        self.assertTrue(event_gradient is None or torch.count_nonzero(event_gradient) == 0)
        self.assertTrue(updater_gradient is None or torch.count_nonzero(updater_gradient) == 0)

    def test_addressed_loss_reaches_updater_and_event_code(self):
        self.model.train()
        source = torch.randn(3, 576)
        event = torch.randn(3, 576)
        packet = self.model.encode_source(source)
        event_code = self.model.encode_event(event)
        updated = self.model.update(packet, event_code, torch.zeros(3, dtype=torch.int64))
        (updated[:, 0] * torch.randn_like(updated[:, 0])).sum().backward()
        self.assertGreater(float(self.model.event_projector.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.updater[0].weight.grad.abs().sum()), 0)

    def test_cpu_controls_match_frozen_parameter_ledger(self):
        observed = {
            "acw": trainable_parameters(CategoricalTrackSModel()),
            "dense_categorical": trainable_parameters(DenseCategoricalTrackSModel()),
            "addressed_continuous": trainable_parameters(AddressedContinuousTrackSModel()),
            "gru": trainable_parameters(GRUTrackSModel()),
            "packet_token_transformer": trainable_parameters(
                PacketTokenTransformerTrackSModel(),
            ),
            "answer_motor": trainable_parameters(AnswerMotorControl()),
            "source_retained": trainable_parameters(SourceRetainedTrackSModel()),
        }
        self.assertEqual(
            observed,
            {
                "acw": 26_008,
                "dense_categorical": 26_250,
                "addressed_continuous": 26_008,
                "gru": 26_036,
                "packet_token_transformer": 25_872,
                "answer_motor": 25_939,
                "source_retained": 166_801,
            },
        )

    def test_parameter_matched_controls_execute(self):
        batch = 4
        source = torch.randn(batch, 96)
        event = torch.randn(batch, 96)
        query = torch.tensor([0, 1, 2, 3])
        address = torch.tensor([0, 1, 2, 0])

        acw = CategoricalTrackSModel()
        packet = acw.workspace.encode_source(source, straight_through=False)
        event_code = acw.workspace.encode_event(event, straight_through=False)
        packet = acw.workspace.update(
            packet, event_code, address, straight_through=False,
        )
        self.assertEqual(acw.read(packet, query).shape, (batch, FIELD_SIZE))

        dense = DenseCategoricalTrackSModel()
        packet = dense.encode_source(source, straight_through=False)
        event_code = dense.encode_event(event, straight_through=False)
        packet = dense.update(
            packet, event_code, address, straight_through=False,
        )
        self.assertEqual(dense.read(packet, query).shape, (batch, FIELD_SIZE))

        continuous = AddressedContinuousTrackSModel()
        state = continuous.encode_source(source)
        event_code = continuous.encode_event(event, straight_through=False)
        updated = continuous.update(state, event_code, address)
        for row in range(batch):
            for register in range(3):
                if register != int(address[row]):
                    self.assertTrue(torch.equal(updated[row, register], state[row, register]))
        self.assertEqual(continuous.read(updated, query).shape, (batch, FIELD_SIZE))

        gru = GRUTrackSModel()
        state = gru.update(gru.encode_source(source), event, address)
        self.assertEqual(gru.read(state, query).shape, (batch, FIELD_SIZE))

        packet_transformer = PacketTokenTransformerTrackSModel()
        packet = packet_transformer.encode_source(source, straight_through=False)
        event_code = packet_transformer.encode_event(event, straight_through=False)
        packet = packet_transformer.update(
            packet, event_code, address, straight_through=False,
        )
        self.assertEqual(packet_transformer.read(packet, query).shape, (batch, FIELD_SIZE))

        motor = AnswerMotorControl()
        self.assertEqual(motor(source, event, query).shape, (batch, FIELD_SIZE))

        retained = SourceRetainedTrackSModel()
        state = retained.update(retained.encode_source(source), event, address)
        self.assertEqual(retained.read(state, source, query).shape, (batch, FIELD_SIZE))


if __name__ == "__main__":
    unittest.main()
