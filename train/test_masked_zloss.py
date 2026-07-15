#!/usr/bin/env python3
"""Exact contracts for completion-masked z-loss."""

import unittest

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, _supervised_lm_loss


class MaskedZLossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[
                [0.2, -0.4, 0.7, 1.1],
                [-0.3, 0.8, 0.1, -0.6],
                [0.9, -0.2, 0.4, -0.7],
            ]],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([[-1, 1, 2]])

    def test_masked_prompt_logits_do_not_affect_loss_or_zloss(self):
        original = self.logits.clone().requires_grad_(True)
        changed_data = self.logits.clone()
        changed_data[:, 0, :] = torch.tensor([100.0, -80.0, 45.0, 3.0])
        changed = changed_data.requires_grad_(True)
        before = _supervised_lm_loss(original, self.targets, 0.25)
        after = _supervised_lm_loss(changed, self.targets, 0.25)
        self.assertTrue(torch.equal(before, after))
        before.backward()
        after.backward()
        self.assertTrue(torch.equal(original.grad[:, 0, :], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(changed.grad[:, 0, :], torch.zeros(1, 4)))

    def test_supervised_logit_shift_affects_only_zloss(self):
        changed = self.logits.clone()
        changed[:, 1, :] += 3.0
        ce_before = _supervised_lm_loss(self.logits, self.targets, 0.0)
        ce_after = _supervised_lm_loss(changed, self.targets, 0.0)
        with_z_before = _supervised_lm_loss(self.logits, self.targets, 0.25)
        with_z_after = _supervised_lm_loss(changed, self.targets, 0.25)
        self.assertTrue(torch.equal(ce_before, ce_after))
        self.assertFalse(torch.equal(with_z_before, with_z_after))

    def test_no_mask_is_numerically_identical_to_previous_formula(self):
        targets = torch.tensor([[0, 1, 2]])
        weight = 0.125
        old_logits = self.logits.clone().requires_grad_(True)
        new_logits = self.logits.clone().requires_grad_(True)
        lf = old_logits.float()
        previous = F.cross_entropy(
            lf.reshape(-1, lf.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        ) + weight * torch.logsumexp(lf, dim=-1).pow(2).mean()
        current = _supervised_lm_loss(new_logits, targets, weight)
        self.assertTrue(torch.equal(previous, current))
        previous.backward()
        current.backward()
        self.assertTrue(torch.equal(old_logits.grad, new_logits.grad))

    def test_forward_and_forward_embeds_have_exact_masked_loss_parity(self):
        torch.manual_seed(17)
        cfg = GPTConfig(
            vocab_size=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=16,
            zloss=0.125,
        )
        model = GPT(cfg).eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[-1, 2, 3], [-1, -1, 6]])
        with torch.no_grad():
            token_logits, token_loss = model(ids, targets)
            embed_logits, embed_loss = model.forward_embeds(model.tok(ids), targets)
        self.assertTrue(torch.equal(token_logits, embed_logits))
        self.assertTrue(torch.equal(token_loss, embed_loss))

    def test_all_ignored_targets_fail_clearly_on_both_paths(self):
        ignored = torch.full((1, 3), -1)
        with self.assertRaisesRegex(ValueError, "no supervised positions"):
            _supervised_lm_loss(self.logits, ignored, 0.25)

        cfg = GPTConfig(
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=1,
            d_model=16,
            d_ff=32,
            seq_len=8,
        )
        model = GPT(cfg).eval()
        ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "no supervised positions"):
            model(ids, ignored)
        with self.assertRaisesRegex(ValueError, "no supervised positions"):
            model.forward_embeds(model.tok(ids), ignored)

    def test_compiled_loss_keeps_the_fail_closed_contract(self):
        compiled_loss = torch.compile(_supervised_lm_loss, backend="eager", fullgraph=True)
        value = compiled_loss(self.logits, self.targets, 0.25)
        self.assertTrue(torch.isfinite(value))
        with self.assertRaisesRegex(RuntimeError, "no supervised positions"):
            compiled_loss(self.logits, torch.full((1, 3), -1), 0.25)


if __name__ == "__main__":
    unittest.main()
