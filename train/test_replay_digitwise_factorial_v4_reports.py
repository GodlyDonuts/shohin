from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import replay_digitwise_factorial_v4_reports as replay  # noqa: E402


REPORT_ROOT = (
    ROOT
    / "artifacts"
    / "evals"
    / "digitwise_factorial_v4_full_de45ace58b5cf1f1490adb11fcbf18524aeb0cb7"
)
KNOWN_REPORT_SHA256 = {
    "iid": "8fa437564cacc2b14c81659d63932426fb4c2abaad86ad2e6c6d5233f409bd01",
    "width": "5ca53f82af9d4ae4649dcb6380e33940003f20d19d97cbb323767a107ca61c59",
    "term_width": "45a4728a115e4532a5ad1d3a45bfe35389206e2954020f8d1ce4934ac70422d2",
}


def test_decimal_oracle_reconstructs_terminal_carry_and_answer() -> None:
    initial = replay.parse_state("dws:op=add;w=4;p=0;c=0;a=9999;b=2200;r=0000;z=0")
    assert initial is not None
    path = replay.oracle_path(initial)
    assert path == [
        {
            "op": "add",
            "w": 4,
            "p": 1,
            "c": 1,
            "a": "9999",
            "b": "2200",
            "r": "1000",
            "z": 0,
        },
        {
            "op": "add",
            "w": 4,
            "p": 2,
            "c": 1,
            "a": "9999",
            "b": "2200",
            "r": "1200",
            "z": 0,
        },
        {
            "op": "add",
            "w": 4,
            "p": 3,
            "c": 1,
            "a": "9999",
            "b": "2200",
            "r": "1200",
            "z": 0,
        },
        {
            "op": "add",
            "w": 4,
            "p": 4,
            "c": 1,
            "a": "9999",
            "b": "2200",
            "r": "1200",
            "z": 1,
        },
    ]
    assert replay.expected_terminal_class(path, initial) == "11"
    assert replay.state_answer(path[-1]) == 10021


def test_exact_mcnemar_matches_locked_paired_result() -> None:
    assert replay.exact_mcnemar(252, 343) == pytest.approx(
        0.00021883968181106602, rel=0.0, abs=1e-18
    )
    assert replay.exact_mcnemar(0, 0) == 1.0


def test_serializer_error_helpers_are_exact() -> None:
    assert replay.decimal_edit_distance("372", "273") == 2
    assert replay.decimal_edit_distance("727", "277") == 2
    assert replay.decimal_edit_distance("9991", "4991") == 1
    assert replay.decimal_edit_distance("123", "1234") == 1
    assert replay.is_adjacent_transposition("727", "277")
    assert not replay.is_adjacent_transposition("372", "273")
    assert not replay.is_adjacent_transposition("9991", "4991")


@pytest.mark.parametrize("arm", tuple(KNOWN_REPORT_SHA256))
def test_existing_sealed_report_replays_exactly_when_present(arm: str) -> None:
    path = REPORT_ROOT / arm / "report.json"
    if not path.exists():
        pytest.skip("large immutable report is not installed")
    result = replay.load_and_replay(path)
    assert result["arm"] == arm
    assert result["report_sha256"] == KNOWN_REPORT_SHA256[arm]
    assert result["accounting"]["pairs"] == 1500
    assert result["metrics"]["branches"]["overall"]["counts"]["branches"] == 3000


def test_canonical_metric_tamper_is_rejected_when_report_present(
    tmp_path: Path,
) -> None:
    source = REPORT_ROOT / "iid" / "report.json"
    if not source.exists():
        pytest.skip("large immutable report is not installed")
    report = json.loads(source.read_text(encoding="ascii"))
    report["metrics"]["branches"]["overall"]["counts"]["closed_loop_success"] += 1
    target = tmp_path / "report.json"
    target.write_bytes(replay.canonical_json_bytes(report))
    os.chmod(target, 0o400)
    with pytest.raises(replay.ReplayError, match="metrics do not replay"):
        replay.load_and_replay(target)


def test_locked_width_to_term_width_paired_contrast_when_present() -> None:
    width_path = REPORT_ROOT / "width" / "report.json"
    term_width_path = REPORT_ROOT / "term_width" / "report.json"
    if not width_path.exists() or not term_width_path.exists():
        pytest.skip("large immutable reports are not installed")
    comparison = replay.compare_arms(
        replay.load_and_replay(width_path),
        replay.load_and_replay(term_width_path),
    )
    terminal = comparison["branches"]["terminal_transition_exact"]
    assert terminal["left_success"] == 390
    assert terminal["right_success"] == 481
    assert terminal["left_only_losses"] == 252
    assert terminal["right_only_gains"] == 343
    assert terminal["mcnemar_exact_two_sided_p"] == pytest.approx(
        0.00021883968181106602, rel=0.0, abs=1e-18
    )


def test_locked_width_to_term_width_carry_tradeoff_when_present() -> None:
    width_path = REPORT_ROOT / "width" / "report.json"
    term_width_path = REPORT_ROOT / "term_width" / "report.json"
    if not width_path.exists() or not term_width_path.exists():
        pytest.skip("large immutable reports are not installed")
    tradeoff = replay.carry_conditioned_branch_tradeoff(
        replay.load_and_replay(width_path),
        replay.load_and_replay(term_width_path),
    )
    concentration = tradeoff["state_change_concentration"]
    assert concentration == {
        "total_right_only_gains": 249,
        "total_left_only_losses": 211,
        "class_10_right_only_gains": 179,
        "class_10_gain_share": pytest.approx(179 / 249),
        "carry_field_involved_right_only_gains": 247,
        "carry_field_involved_gain_share": pytest.approx(247 / 249),
        "class_10_carry_field_involved_right_only_gains": 177,
        "class_10_carry_field_involved_gain_share": pytest.approx(177 / 179),
        "class_00_left_only_losses": 200,
        "class_00_loss_share": pytest.approx(200 / 211),
        "sub_class_00_left_only_losses": 158,
        "sub_class_00_loss_share": pytest.approx(158 / 211),
        "carry_field_involved_left_only_losses": 207,
        "carry_field_involved_loss_share": pytest.approx(207 / 211),
        "class_00_carry_field_involved_left_only_losses": 196,
        "class_00_carry_field_involved_loss_share": pytest.approx(196 / 200),
        "sub_class_00_carry_field_involved_left_only_losses": 155,
        "sub_class_00_carry_field_involved_loss_share": pytest.approx(155 / 158),
    }
    assert (
        tradeoff["groups"]["add|w4|10"]["state_closed_loop_exact"]["right_only_gains"]
        == 65
    )
    assert (
        tradeoff["groups"]["add|w4|10"]["state_closed_loop_exact"]["left_only_losses"]
        == 4
    )
    assert (
        tradeoff["groups"]["sub|w4|00"]["state_closed_loop_exact"]["right_only_gains"]
        == 36
    )
    assert (
        tradeoff["groups"]["sub|w4|00"]["state_closed_loop_exact"]["left_only_losses"]
        == 107
    )
    assert tradeoff["groups"]["sub|w4|00"]["state_left_only_source_first_mismatch"] == {
        "count": 107,
        "carry_field_involved": 106,
        "carry_field_involved_rate": pytest.approx(106 / 107),
        "by_field_set": {"c": 99, "c+r": 7, "r": 1},
        "by_position": {"0": 47, "1": 35, "2": 24, "3": 1},
    }
    assert tradeoff["groups"]["add|w4|00"]["state_left_only_source_first_mismatch"] == {
        "count": 24,
        "carry_field_involved": 23,
        "carry_field_involved_rate": pytest.approx(23 / 24),
        "by_field_set": {"c": 23, "r": 1},
        "by_position": {"0": 1, "1": 4, "2": 19},
    }
    assert tradeoff["groups"]["add|w4|10"][
        "state_right_only_source_first_mismatch"
    ] == {
        "count": 65,
        "carry_field_involved": 65,
        "carry_field_involved_rate": 1.0,
        "by_field_set": {"c": 65},
        "by_position": {"0": 27, "1": 6, "2": 32},
    }
    assert tradeoff["groups"]["sub|w4|10"][
        "state_right_only_source_first_mismatch"
    ] == {
        "count": 94,
        "carry_field_involved": 92,
        "carry_field_involved_rate": pytest.approx(92 / 94),
        "by_field_set": {"c": 90, "c+r": 2, "r": 2},
        "by_position": {"0": 43, "1": 29, "2": 22},
    }


def test_locked_serializer_error_profile_when_reports_present() -> None:
    expected_profiles = {
        "iid": {
            "count": 17,
            "exclusive_class_counts": {
                "unparseable": 0,
                "omitted_final_carry": 0,
                "extra_final_carry": 0,
                "exact_digit_reversal": 1,
                "stored_tape_order_value": 0,
                "adjacent_transposition": 3,
                "one_digit_substitution": 6,
                "same_digit_multiset_other": 0,
                "single_edit_other": 6,
                "other": 1,
            },
        },
        "term": {
            "count": 21,
            "exclusive_class_counts": {
                "unparseable": 0,
                "omitted_final_carry": 0,
                "extra_final_carry": 0,
                "exact_digit_reversal": 6,
                "stored_tape_order_value": 2,
                "adjacent_transposition": 4,
                "one_digit_substitution": 6,
                "same_digit_multiset_other": 0,
                "single_edit_other": 2,
                "other": 1,
            },
        },
        "width": {
            "count": 57,
            "exclusive_class_counts": {
                "unparseable": 0,
                "omitted_final_carry": 0,
                "extra_final_carry": 0,
                "exact_digit_reversal": 4,
                "stored_tape_order_value": 1,
                "adjacent_transposition": 11,
                "one_digit_substitution": 11,
                "same_digit_multiset_other": 5,
                "single_edit_other": 11,
                "other": 14,
            },
        },
        "term_width": {
            "count": 84,
            "exclusive_class_counts": {
                "unparseable": 0,
                "omitted_final_carry": 0,
                "extra_final_carry": 0,
                "exact_digit_reversal": 7,
                "stored_tape_order_value": 0,
                "adjacent_transposition": 20,
                "one_digit_substitution": 30,
                "same_digit_multiset_other": 3,
                "single_edit_other": 11,
                "other": 13,
            },
        },
    }
    for arm, expected in expected_profiles.items():
        path = REPORT_ROOT / arm / "report.json"
        if not path.exists():
            pytest.skip("large immutable reports are not installed")
        profile = replay.load_and_replay(path)["diagnostics"][
            "exact_state_wrong_final"
        ]["error_profile"]
        assert profile["count"] == expected["count"]
        assert profile["exclusive_class_counts"] == expected["exclusive_class_counts"]
        assert sum(profile["exclusive_class_counts"].values()) == expected["count"]
