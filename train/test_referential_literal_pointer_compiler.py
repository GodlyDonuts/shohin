#!/usr/bin/env python3

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import referential_literal_pointer_compiler as compiler_module  # noqa: E402


class IdentityBlock(nn.Module):
    def forward(self, x, cos, sin):
        return x, None


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(n_loop=1, d_model=32)
        self.tok = nn.Embedding(32768, 32)
        self.blocks = nn.ModuleList([IdentityBlock(), IdentityBlock()])
        self.register_buffer("cos", torch.zeros(256, 1))
        self.register_buffer("sin", torch.zeros(256, 1))


class ReferentialLiteralPointerCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts" / "shohin-tok-32k.json"))
        with (ROOT / "artifacts" / "r12" / "referential_literal_pointer_corpus_v1" / "development.jsonl").open() as source:
            cls.row = json.loads(next(source))

    def test_compile_row_requires_all_ten_targets(self):
        example = compiler_module.compile_row(self.row, self.tokenizer, keep_evidence=True)
        self.assertEqual(set(example.target_positions), set(compiler_module.TARGET_LABELS))
        self.assertEqual(len(example.target_positions), 10)

    def test_compile_row_retains_factor_evidence_only_when_requested(self):
        row = dict(self.row)
        row["factors"] = {"argument_order": 2, "lexicon": "known"}
        retained = compiler_module.compile_row(row, self.tokenizer, keep_evidence=True)
        hidden = compiler_module.compile_row(row, self.tokenizer, keep_evidence=False)
        self.assertEqual(dict(retained.factors), row["factors"])
        self.assertEqual(hidden.factors, ())

    def test_model_forward_and_loss_are_finite(self):
        example = compiler_module.compile_row(self.row, self.tokenizer, keep_evidence=True)
        model = compiler_module.CompletePointerCompiler(
            DummyBase(), layer=1, width=32, heads=4, decoder_layers=1, ff=64,
        )
        ids = torch.tensor([example.ids], dtype=torch.long)
        valid = torch.ones_like(ids, dtype=torch.bool)
        outputs = model(ids, valid)
        self.assertEqual(set(outputs["pointer_logits"]), set(compiler_module.TARGET_LABELS))
        self.assertEqual(tuple(outputs["kind_logits"].shape), (1, 2, 2))
        loss, pointer, kind, losses = compiler_module.compiler_loss(outputs, [example])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pointer))
        self.assertTrue(torch.isfinite(kind))
        self.assertEqual(len(losses), 10)
        loss.backward()

    def test_gold_pointer_dereference_executes_without_structured_values(self):
        example = compiler_module.compile_row(self.row, self.tokenizer, keep_evidence=True)
        encoding = self.tokenizer.encode(example.question)
        pointers = {label: positions[0] for label, positions in example.target_positions.items()}
        answer, semantic = compiler_module.execute_prediction(
            example, encoding, pointers, example.kind_targets,
        )
        self.assertEqual(answer, example.answer)
        self.assertTrue(compiler_module.semantic_exact(example, semantic))

    def test_bidirectional_role_parser_has_finite_auxiliary_loss(self):
        example = compiler_module.compile_row(self.row, self.tokenizer, keep_evidence=True)
        model = compiler_module.CompletePointerCompiler(
            DummyBase(), layer=1, width=32, heads=4, decoder_layers=1, ff=64,
            encoder_layers=2, role_supervision=True,
        )
        ids = torch.tensor([example.ids], dtype=torch.long)
        valid = torch.ones_like(ids, dtype=torch.bool)
        outputs = model(ids, valid)
        self.assertEqual(tuple(outputs["role_logits"].shape), (1, len(example.ids), 10))
        role = compiler_module.role_supervision_loss(outputs, [example])
        self.assertTrue(torch.isfinite(role))
        role.backward()

    def test_separate_kind_island_has_an_independent_memory_path(self):
        example = compiler_module.compile_row(self.row, self.tokenizer, keep_evidence=True)
        model = compiler_module.CompletePointerCompiler(
            DummyBase(), layer=1, width=32, heads=4, decoder_layers=1, ff=64,
            encoder_layers=2, role_supervision=True, separate_kind_decoder=True,
        )
        self.assertIsNot(model.memory_projection, model.kind_memory_projection)
        ids = torch.tensor([example.ids], dtype=torch.long)
        valid = torch.ones_like(ids, dtype=torch.bool)
        outputs = model(ids, valid)
        loss, _, kind, _ = compiler_module.compiler_loss(outputs, [example])
        self.assertTrue(torch.isfinite(kind))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
