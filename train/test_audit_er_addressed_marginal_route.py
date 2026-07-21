from __future__ import annotations

import inspect

from audit_er_addressed_marginal_route import ORDINAL_SCALES, SCHEMA, main


def test_addressed_audit_is_scale_only_and_read_only() -> None:
    assert SCHEMA == "r12_er_addressed_marginal_route_read_only_audit_v1"
    assert ORDINAL_SCALES == (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    source = inspect.getsource(main)
    assert 'filename="train.jsonl"' in source
    assert 'filename="development.jsonl"' not in source
    assert 'filename="confirmation.jsonl"' not in source
    assert "optimizer" not in source
    assert "load_trainable_state" in source
    assert "closed-checkpoint-sha256" in source
