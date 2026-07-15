import unittest

import torch

from counterfactual_cursor_action_training import (
    ARMS,
    adapter_state_payload,
    build_adapter,
    encode_before_final_block,
    freeze_base,
    logits_from_final_block_cache,
)
from model import GPT, GPTConfig


class CounterfactualCursorActionTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.cfg = GPTConfig(
            vocab_size=8192, n_layer=3, n_head=9, n_kv_head=3,
            d_model=576, d_ff=64, seq_len=32, qk_norm=True,
        )
        self.model = GPT(self.cfg).eval()
        freeze_base(self.model)
        self.ids = torch.randint(1, 8192, (4, 11))
        self.positions = torch.tensor([6, 8, 9, 10])
        self.cursors = torch.tensor([0, 1, 3, 4])
        self.labels = torch.tensor([820, 5498, 4307, 7486, 2165])

    def test_adapter_shapes_and_zero_initialization(self):
        expected = {
            "orbit_interchange": 192,
            "ordinary_loss": 192,
            "relation_sham": 192,
            "source_only": 192,
            "cursor_table": 512,
            "text_cursor_lora": 640,
        }
        for arm in ARMS:
            adapter, spec = build_adapter(arm, self.cfg, 17)
            self.assertEqual(sum(p.numel() for p in adapter.parameters()), expected[arm])
            payload = adapter_state_payload(adapter, spec)
            self.assertEqual(payload["adapter_spec"]["arm"], arm)

    def test_cached_sidecar_path_matches_full_model(self):
        adapter, _ = build_adapter("orbit_interchange", self.cfg, 17)
        with torch.no_grad():
            adapter.projection.weight.normal_(0.0, 0.2)
        grid = self.cursors[:, None].expand(-1, self.ids.shape[1]).clone()
        mask = torch.zeros_like(grid, dtype=torch.bool)
        mask.scatter_(1, self.positions[:, None], True)
        delta = adapter(grid, mask)
        expected, _ = self.model(
            self.ids, q_delta=delta, q_delta_layer=-1, q_delta_head=0,
        )
        prefix = encode_before_final_block(self.model, self.ids)
        full, _ = logits_from_final_block_cache(
            self.model, prefix, self.positions, adapter, "orbit_interchange",
            self.cursors, self.labels,
        )
        selected = expected[torch.arange(4), self.positions]
        self.assertTrue(torch.allclose(full, selected, atol=1e-6, rtol=1e-6))

    def test_cached_text_lora_path_matches_full_model(self):
        adapter, _ = build_adapter("text_cursor_lora", self.cfg, 17)
        with torch.no_grad():
            adapter.up.weight.normal_(0.0, 0.1)
        expected, _ = self.model(
            self.ids, q_adapter=adapter, q_delta_layer=-1, q_delta_head=0,
        )
        prefix = encode_before_final_block(self.model, self.ids)
        full, _ = logits_from_final_block_cache(
            self.model, prefix, self.positions, adapter, "text_cursor_lora",
            None, self.labels,
        )
        selected = expected[torch.arange(4), self.positions]
        self.assertTrue(torch.allclose(full, selected, atol=1e-6, rtol=1e-6))

    def test_only_adapter_receives_gradient(self):
        adapter, _ = build_adapter("ordinary_loss", self.cfg, 17)
        prefix = encode_before_final_block(self.model, self.ids)
        full, _ = logits_from_final_block_cache(
            self.model, prefix, self.positions, adapter, "ordinary_loss",
            self.cursors, self.labels,
        )
        full.sum().backward()
        self.assertIsNotNone(adapter.projection.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_text_control_rejects_hidden_cursor(self):
        adapter, _ = build_adapter("text_cursor_lora", self.cfg, 17)
        prefix = encode_before_final_block(self.model, self.ids)
        with self.assertRaisesRegex(ValueError, "may not receive"):
            logits_from_final_block_cache(
                self.model, prefix, self.positions, adapter, "text_cursor_lora",
                self.cursors, self.labels,
            )


if __name__ == "__main__":
    unittest.main()
