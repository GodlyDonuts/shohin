from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pilot_er_cst_rule_card_adapter import (
    CONFIRMATION_ASSESSMENT_SHA256,
    CONFIRMED_CHECKPOINT_SHA256,
    _validate_confirmation,
    state_dict_digest,
)


def test_state_dict_digest_is_order_invariant_and_value_sensitive() -> None:
    left = {
        "a": torch.tensor([1, 2], dtype=torch.int64),
        "b": torch.tensor([[3.0]], dtype=torch.float32),
    }
    right = {"b": left["b"].clone(), "a": left["a"].clone()}
    assert state_dict_digest(left) == state_dict_digest(right)
    right["b"][0, 0] = 4.0
    assert state_dict_digest(left) != state_dict_digest(right)


def test_confirmation_validator_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "assessment.json"
    path.write_text(json.dumps({"decision": "confirm_complete_physical_fresh_v1_3"}))
    with pytest.raises(ValueError, match="hash"):
        _validate_confirmation(path)


def test_exact_parent_hash_constants_are_full_sha256() -> None:
    assert len(CONFIRMED_CHECKPOINT_SHA256) == 64
    assert len(CONFIRMATION_ASSESSMENT_SHA256) == 64
