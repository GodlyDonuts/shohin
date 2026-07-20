from __future__ import annotations

import torch

from audit_sd_cst_complete_physical_record_bus_v1_1 import classify_top_indices


def test_classify_top_indices_assigns_six_spans_and_other() -> None:
    masks = torch.zeros(2, 6, 10, dtype=torch.bool)
    for batch in range(2):
        for category in range(6):
            masks[batch, category, category] = True
    top = torch.tensor([[0, 1, 2, 3, 4, 9], [5, 4, 3, 2, 1, 0]])
    actual = classify_top_indices(masks, top)
    assert actual.tolist() == [[0, 1, 2, 3, 4, 6], [5, 4, 3, 2, 1, 0]]
