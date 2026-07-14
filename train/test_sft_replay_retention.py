#!/usr/bin/env python3
"""CPU contracts for prompt-only retention helpers in ``sft.py``."""
from __future__ import annotations

import json
import tempfile

import numpy as np
import torch

from sft import load_replay_prompts, make_replay_batch, replay_kl


class Tokenizer:
    def encode(self, text):
        class Encoded:
            ids = [len(part) for part in text.split()]
        return Encoded()


with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as source:
    source.write(json.dumps({"prompt": "one two three"}) + "\n")
    source.write(json.dumps({"prompt": "four five"}) + "\n")
    source.write(json.dumps({"prompt": ""}) + "\n")
    source.flush()
    prompts, skipped = load_replay_prompts(source.name, Tokenizer(), 4)

assert len(prompts) == 2
assert skipped == {"invalid": 1}
ids, lengths = make_replay_batch(prompts, batch_size=3, max_tokens=4, pad_id=0,
                                  rng=np.random.default_rng(4), device="cpu")
assert ids.shape == (3, 4)
assert lengths.min().item() >= 2
teacher = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [2.0, 2.0], [0.0, 0.0]]])
assert replay_kl(teacher, teacher, torch.tensor([3])).abs().item() < 1e-6
student = teacher.clone()
student[0, 0] = torch.tensor([0.0, 4.0])
assert replay_kl(student, teacher, torch.tensor([3])).item() > 0
print("SFT replay retention checks: passed")
