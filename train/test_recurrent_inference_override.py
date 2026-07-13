#!/usr/bin/env python3
"""Verify the evaluator-only latent-depth override and its KV-cache shape."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from capability_matrix import build_cases, load_model as load_matrix_model, run_checkpoint
from generalization_interview import load_model as load_interview_model
from model import GPT, GPTConfig


class TinyTokenizer:
    def encode(self, _text):
        return SimpleNamespace(ids=[1, 2, 3])

    def token_to_id(self, _name):
        return 0

    def decode(self, _ids):
        return ""

    def decode_batch(self, rows):
        return [self.decode(row) for row in rows]


def tiny_checkpoint(path):
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        d_model=32,
        d_ff=64,
        seq_len=16,
        zloss=0.0,
        n_loop=1,
    )
    torch.manual_seed(7)
    model = GPT(cfg)
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "step": 123}, path)


def check_loader(loader, path):
    checkpoint, model = loader(path, "cpu", n_loop=2)
    assert checkpoint["step"] == 123
    assert model.cfg.n_loop == 2
    with torch.inference_mode():
        logits, cache = model(torch.tensor([[1, 2, 3]]), return_cache=True)
        assert logits.shape == (1, 3, 64)
        assert len(cache) == 4
        next_logits, next_cache = model(torch.tensor([[4]]), cache=cache, pos=3, return_cache=True)
        assert next_logits.shape == (1, 1, 64)
        assert len(next_cache) == 4


def check_matrix_metadata(path):
    result = run_checkpoint(path, TinyTokenizer(), "cpu", build_cases(1, 7), ["qa"], 1, n_loop=2)
    assert result["n_loop"] == 2


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.pt"
        tiny_checkpoint(path)
        check_loader(load_matrix_model, path)
        check_loader(load_interview_model, path)
        check_matrix_metadata(path)
    print("recurrent inference override tests passed")


if __name__ == "__main__":
    main()
