#!/usr/bin/env python3
"""Static component-boundary checks for FQRB checkpoint probes."""
from __future__ import annotations

import torch

from make_fqrb_checkpoint_probes import decoder_keys, make_payload


state = {
    "tok.weight": torch.ones(2, 2),
    "head.weight": torch.ones(2, 2),
    "norm.w": torch.ones(2),
    "blocks.19.attn.q.weight": torch.ones(2, 2),
    "blocks.20.attn.q.weight": torch.ones(2, 2),
    "blocks.29.n2.w": torch.ones(2),
}
keys = decoder_keys(state, 19)
assert "blocks.19.attn.q.weight" not in keys
assert {"tok.weight", "head.weight", "norm.w", "blocks.20.attn.q.weight", "blocks.29.n2.w"} <= keys
raw = {"cfg": {"n_layer": 30}, "model": state}
fqrb = {"cfg": {"n_layer": 30}, "model": {key: value * 3 for key, value in state.items()}, "step": "cra_ep1"}
payload = make_payload(raw, fqrb, {"tok.weight", "head.weight"}, "raw_lexicon", 19, "raw", "fqrb")
assert torch.equal(payload["model"]["tok.weight"], raw["model"]["tok.weight"])
assert torch.equal(payload["model"]["blocks.20.attn.q.weight"], fqrb["model"]["blocks.20.attn.q.weight"])
assert payload["fqrb_checkpoint_probe"]["replaced_key_count"] == 2
print("FQRB checkpoint probe checks: passed")
