from __future__ import annotations

import pytest

from audit_sd_cst_renderer_native_joint import _rates_from_counts


def test_rates_from_counts_uses_slot_specific_denominators() -> None:
    counts = {
        "line_slots": 90,
        "kind_slots": 80,
        "active_slots": 70,
        "line_slot": 45,
        "kind_slot": 60,
        "amount_active": 56,
        "identity_active": 49,
        "event_pointer_active": 42,
        "gold_line_kind_slot": 72,
        "gold_line_amount_active": 63,
        "gold_event_identity_active": 70,
    }
    rates = _rates_from_counts(counts)
    assert rates == {
        "line_slot": 0.5,
        "kind_slot": 0.75,
        "amount_active": 0.8,
        "identity_active": 0.7,
        "event_pointer_active": 0.6,
        "gold_line_kind_slot": 0.9,
        "gold_line_amount_active": 0.9,
        "gold_event_identity_active": 1.0,
    }


def test_rates_from_counts_rejects_empty_denominator() -> None:
    with pytest.raises(ValueError, match="no line_slots"):
        _rates_from_counts({"line_slots": 0})
