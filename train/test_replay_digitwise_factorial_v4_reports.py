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
