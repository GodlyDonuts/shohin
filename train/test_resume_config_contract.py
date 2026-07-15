#!/usr/bin/env python3
"""Fail-closed checkpoint configuration contracts for pretraining resume."""

import unittest
from dataclasses import asdict, replace

from model import GPTConfig
from train import validate_resume_config


class ResumeConfigContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = GPTConfig()

    def test_exact_matching_checkpoint_config_is_allowed(self):
        validate_resume_config(self.cfg, asdict(self.cfg))

    def test_every_behavior_field_mismatch_is_rejected(self):
        mutations = {
            "vocab_size": self.cfg.vocab_size + 1,
            "n_layer": self.cfg.n_layer + 1,
            "n_head": self.cfg.n_head + 1,
            "n_kv_head": self.cfg.n_kv_head + 1,
            "d_model": self.cfg.d_model + 1,
            "d_ff": self.cfg.d_ff + 1,
            "seq_len": self.cfg.seq_len + 1,
            "rope_theta": self.cfg.rope_theta + 1.0,
            "qk_norm": not self.cfg.qk_norm,
            "tie_embeddings": not self.cfg.tie_embeddings,
            "zloss": self.cfg.zloss * 2.0,
            "n_loop": self.cfg.n_loop + 1,
        }
        for field_name, requested_value in mutations.items():
            with self.subTest(field=field_name):
                requested = replace(self.cfg, **{field_name: requested_value})
                with self.assertRaisesRegex(ValueError, field_name):
                    validate_resume_config(requested, asdict(self.cfg))

    def test_legacy_checkpoint_without_config_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy checkpoints cannot be resumed safely"):
            validate_resume_config(self.cfg, None)

    def test_incomplete_or_future_checkpoint_schema_is_rejected(self):
        missing = asdict(self.cfg)
        del missing["n_loop"]
        with self.assertRaisesRegex(ValueError, "missing fields: n_loop"):
            validate_resume_config(self.cfg, missing)

        unknown = asdict(self.cfg)
        unknown["future_behavior"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields: future_behavior"):
            validate_resume_config(self.cfg, unknown)


if __name__ == "__main__":
    unittest.main()
