"""Deterministic construction tests for the paired-board DRS factorial."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "pipeline" / "generate_digitwise_factorial_v4.py"
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_digitwise_factorial_v4 import (  # noqa: E402
    allocations_for_arm,
    pair_distribution_contract,
    structural_counts,
)
from generate_digitwise_recurrent_v1 import (  # noqa: E402
    counterfactual_episode,
    episode_from_operands,
)


ARMS = ("iid", "term", "width", "term_width")
PAIRS = (("iid", "term"), ("width", "term_width"))
TERM_COUNTS = {
    "add": {"00": 50, "01": 50, "10": 50, "11": 50},
    "sub": {"00": 100, "10": 100},
}
CONTROL_COUNTS = {
    "add": {"00": 100, "10": 100},
    "sub": {"00": 100, "10": 100},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def heldout_episode(
    episode_id: str,
    operation: str,
    left: int,
    right: int,
    width: int = 8,
) -> dict:
    episode = episode_from_operands(
        episode_id,
        "fixture_w{}".format(width),
        width,
        operation,
        left,
        right,
        "heldout",
    )
    episode["counterfactual"] = counterfactual_episode(episode)
    return episode


def make_heldout_fixture(path: Path, extra: dict | None = None) -> Path:
    episodes = [
        heldout_episode("fixture-add-0", "add", 12_345_670, 1_111_111),
        heldout_episode("fixture-add-1", "add", 23_456_780, 2_222_222),
        heldout_episode("fixture-sub-0", "sub", 87_654_320, 12_345_678),
        heldout_episode("fixture-sub-1", "sub", 76_543_210, 23_456_789),
    ]
    if extra is not None:
        episodes.append(extra)
    write_jsonl(path, episodes)
    return path


def generate(
    root: Path,
    arm: str,
    heldout: Path,
    label: str = "run",
) -> tuple[Path, Path, Path, dict]:
    directory = root / "{}-{}".format(arm, label)
    directory.mkdir()
    data = directory / "data.jsonl"
    episodes = directory / "episodes.jsonl"
    report = directory / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--mode",
            "test",
            "--arm",
            arm,
            "--heldout",
            str(heldout),
            "--data-out",
            str(data),
            "--episodes-out",
            str(episodes),
            "--report",
            str(report),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return data, episodes, report, json.loads(report.read_text())


def visible_features(data: Path, episodes: Path) -> Counter:
    board = {episode["id"]: episode for episode in read_jsonl(episodes)}
    return Counter(
        (
            row["kind"],
            row["width"],
            row["operation"],
            board[row["episode_id"]]["left"],
            board[row["episode_id"]]["right"],
        )
        for row in read_jsonl(data)
    )


def test_production_structural_contracts() -> None:
    expected_allocations = {
        "iid": {4: 19_985, 6: 20_000},
        "term": {4: 19_985, 6: 20_000},
        "width": {3: 7_982, 4: 8_012, 5: 7_997, 6: 7_997, 7: 7_997},
        "term_width": {3: 7_982, 4: 8_012, 5: 7_997, 6: 7_997, 7: 7_997},
    }
    for arm in ARMS:
        allocations = allocations_for_arm(arm)
        assert allocations == expected_allocations[arm]
        assert structural_counts(allocations) == {
            "episodes": 39_985,
            "transitions": 199_940,
            "rows": 439_865,
        }


def test_small_arms_share_frozen_boards_and_exact_visible_budgets(
    tmp_path: Path,
) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = {arm: generate(tmp_path, arm, heldout) for arm in ARMS}

    for arm, (data, episodes_path, _, report) in generated.items():
        rows = read_jsonl(data)
        episodes = read_jsonl(episodes_path)
        assert report["mode"] == "test"
        assert report["production_eligible"] is False
        assert report["production_admission"] is False
        assert report["structural_target"] == {
            "episodes": 400,
            "transitions": 2_000,
            "rows": 4_400,
        }
        assert len(rows) == report["unique_normalized_prompts"] == 4_400
        assert len(episodes) == report["unique_complete_episode_signatures"] == 400
        assert len({(e["width"], e["left"], e["right"]) for e in episodes}) == 400
        assert report["required_arithmetic_classes"] == 400
        assert report["covered_arithmetic_classes"] == 400
        assert report["paired_position_budget_exact"] is True
        assert report["pair_distribution_contract"]["all_cells_pass"] is True
        assert report["terminal_transition_classes"] == (
            TERM_COUNTS if arm in ("term", "term_width") else CONTROL_COUNTS
        )
        assert {row["arm"] for row in rows} == {arm}
        assert {row["kind"] for row in rows} == {"transition", "digit", "final"}
        assert all(row["question"] == row["completion_prompt"] for row in rows)
        assert sha256(data) == report["data_sha256"]
        assert sha256(episodes_path) == report["episodes_sha256"]
        assert report["heldout_binding"]["sha256"] == sha256(heldout)
        assert report["train_heldout_reserved_signature_hits"] == 0

    for control, treatment in PAIRS:
        control_data, control_episodes, _, control_report = generated[control]
        term_data, term_episodes, _, term_report = generated[treatment]
        assert control_episodes.read_bytes() == term_episodes.read_bytes()
        assert visible_features(control_data, control_episodes) == visible_features(
            term_data, term_episodes
        )
        assert (
            control_report["paired_board"]["literal_jsonl_sha256"]
            == term_report["paired_board"]["literal_jsonl_sha256"]
        )
        assert (
            control_report["paired_board"]["episode_ids_sha256"]
            == term_report["paired_board"]["episode_ids_sha256"]
        )
        assert (
            control_report["paired_board"]["full_visible_feature_multiset_sha256"]
            == term_report["paired_board"]["full_visible_feature_multiset_sha256"]
        )

    first_data, first_episodes, _, _ = generated["term_width"]
    second_data, second_episodes, _, _ = generate(
        tmp_path, "term_width", heldout, "repeat"
    )
    assert first_data.read_bytes() == second_data.read_bytes()
    assert first_episodes.read_bytes() == second_episodes.read_bytes()


def test_pair_distribution_guard_rejects_deterministic_shortcuts() -> None:
    assert not pair_distribution_contract(Counter({(0, 0): 100}))
    assert not pair_distribution_contract(Counter({(9, 9): 50, (0, 0): 50}))
    assert pair_distribution_contract(Counter({(i, i): 2 for i in range(10)}))


def test_heldout_reservation_corruption_modes_and_path_aliases(tmp_path: Path) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    _, baseline_episodes_path, _, _ = generate(tmp_path, "iid", heldout, "base")
    baseline = read_jsonl(baseline_episodes_path)[0]
    collision = heldout_episode(
        "fixture-collision",
        baseline["operation"],
        baseline["left"],
        baseline["right"],
        baseline["width"],
    )
    collision_heldout = make_heldout_fixture(
        tmp_path / "collision-heldout.jsonl", collision
    )
    _, reserved_episodes, _, report = generate(
        tmp_path, "iid", collision_heldout, "reserved"
    )
    reserved_signatures = {
        (branch["width"], branch["left"], branch["right"])
        for episode in read_jsonl(collision_heldout)
        for branch in (episode, episode["counterfactual"])
    }
    generated_signatures = {
        (episode["width"], episode["left"], episode["right"])
        for episode in read_jsonl(reserved_episodes)
    }
    assert not (generated_signatures & reserved_signatures)
    assert report["reserved_signature_draw_collisions_rejected"] >= 1

    corrupted = read_jsonl(heldout)
    corrupted[0]["expected_states"][0] = "not-a-canonical-state"
    corrupt_heldout = tmp_path / "corrupt-heldout.jsonl"
    write_jsonl(corrupt_heldout, corrupted)
    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    corrupt = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--mode",
            "test",
            "--arm",
            "iid",
            "--heldout",
            str(corrupt_heldout),
            "--data-out",
            str(corrupt_root / "data.jsonl"),
            "--episodes-out",
            str(corrupt_root / "episodes.jsonl"),
            "--report",
            str(corrupt_root / "report.json"),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert corrupt.returncode != 0
    assert "invalid heldout episode" in corrupt.stderr
    assert not any(corrupt_root.iterdir())

    mode_root = tmp_path / "bad-mode"
    mode_root.mkdir()
    bad_mode = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--mode",
            "production",
            "--arm",
            "iid",
            "--heldout",
            str(heldout),
            "--data-out",
            str(mode_root / "data.jsonl"),
            "--episodes-out",
            str(mode_root / "episodes.jsonl"),
            "--report",
            str(mode_root / "report.json"),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad_mode.returncode != 0
    assert "production mode forbids --test-scale" in bad_mode.stderr
    assert not any(mode_root.iterdir())

    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    final = alias_root / "X"
    alias = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--mode",
            "test",
            "--arm",
            "iid",
            "--heldout",
            str(heldout),
            "--data-out",
            str(final),
            "--episodes-out",
            str(alias_root / "X.partial"),
            "--report",
            str(alias_root / "report.json"),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert alias.returncode != 0
    assert "final and .partial paths must be pairwise distinct" in alias.stderr
    assert not any(alias_root.iterdir())
