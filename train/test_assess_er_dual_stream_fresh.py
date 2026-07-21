from __future__ import annotations

import torch

from assess_er_dual_stream_fresh import recompute_invariance
from er_dual_stream_fresh_scoring import SEMANTIC_KEYS


def _evidence() -> dict[str, object]:
    hard = {"field": torch.ones(2_048, 2, dtype=torch.int16)}
    semantic = {
        key: torch.ones(2_048, 1, dtype=torch.int16) for key in SEMANTIC_KEYS
    }
    return {
        "canonical_hard": hard,
        "alpha_hard": {name: value.clone() for name, value in hard.items()},
        "distractor_hard": {name: value.clone() for name, value in hard.items()},
        "canonical_semantic": semantic,
        "rule_reindex": {name: value.clone() for name, value in semantic.items()},
        "physical_reindex": {name: value.clone() for name, value in semantic.items()},
    }


def test_independent_invariance_recomputation_detects_one_changed_row() -> None:
    raw = _evidence()
    exact = recompute_invariance(raw)
    assert all(value["exact"] == 2_048 for value in exact.values())
    raw["alpha_hard"]["field"][17, 0] = 0
    changed = recompute_invariance(raw)
    assert changed["alpha"]["exact"] == 2_047
    assert changed["distractor_rotation"]["exact"] == 2_048
