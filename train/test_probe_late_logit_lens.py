#!/usr/bin/env python3
"""Unit contracts for the read-only late logit-lens screen."""
from __future__ import annotations

import torch

from probe_late_logit_lens import rank_for_ids


logits = torch.tensor([0.1, 2.0, 1.0, 3.0])
rank, token_id = rank_for_ids(logits, [1, 2])
assert token_id == 1
assert rank == 2
rank, token_id = rank_for_ids(logits, [0, 2])
assert token_id == 2
assert rank == 3
print("Late logit-lens screen checks: passed")
