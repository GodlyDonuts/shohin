"""Independent admission and corruption tests for paired-board factorial v4."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "pipeline" / "generate_digitwise_factorial_v4.py"
AUDITOR = ROOT / "pipeline" / "audit_digitwise_factorial_v4.py"
TOKENIZER = ROOT / "artifacts" / "shohin-tok-32k.json"
ARMS = ("iid", "term", "width", "term_width")
PAIRED = {
    "iid": "term",
    "term": "iid",
    "width": "term_width",
    "term_width": "width",
}
CANONICAL_TEST_PACKING = {
    "iid": {
        "encoded_tokens": 535_063,
        "encoded_supervised_tokens": 106_463,
        "packed_sequences": 261,
        "packed_supervised_tokens": 106_391,
        "packing_sha256": (
            "4691a670e9d897f6bacb7f0c3eaa1dd25d4c89ddf752f7432f30935086c5b1b5"
        ),
    },
    "term": {
        "encoded_tokens": 535_063,
        "encoded_supervised_tokens": 106_463,
        "packed_sequences": 261,
        "packed_supervised_tokens": 106_345,
        "packing_sha256": (
            "ebc89055c69bd354ebb663afff999bc810eb6b111ee96d543ff0856981277cac"
        ),
    },
    "width": {
        "encoded_tokens": 538_667,
        "encoded_supervised_tokens": 107_667,
        "packed_sequences": 263,
        "packed_supervised_tokens": 107_657,
        "packing_sha256": (
            "dc76fdf2d9d2f39a731a16bc927e0809025d1545bb8a067222949e2458879781"
        ),
    },
    "term_width": {
        "encoded_tokens": 538_667,
        "encoded_supervised_tokens": 107_667,
        "packed_sequences": 263,
        "packed_supervised_tokens": 107_657,
        "packing_sha256": (
            "9d50360422f500b22243b925261507e8b59949e84258c4bbfe645fa385e11dc9"
        ),
    },
}

sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from digitwise_protocol import (  # noqa: E402
    canonical_state,
    microstep_prompt,
    parse_state,
)
from test_generate_digitwise_factorial_v4 import (  # noqa: E402
    heldout_episode,
    make_heldout_fixture,
    read_jsonl,
    write_jsonl,
)


def generate(
    root: Path, arm: str, heldout: Path, label: str = "run"
) -> tuple[Path, Path]:
    directory = root / "{}-{}".format(arm, label)
    directory.mkdir()
    data = directory / "data.jsonl"
    episodes = directory / "episodes.jsonl"
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
            str(directory / "generated.json"),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return data, episodes


def generate_all(root: Path, heldout: Path) -> dict[str, tuple[Path, Path]]:
    return {arm: generate(root, arm, heldout) for arm in ARMS}


def run_audit(
    generated: dict[str, tuple[Path, Path]],
    heldout: Path,
    out: Path,
    arm: str,
    *,
    data: Path | None = None,
    episodes: Path | None = None,
    paired_data: Path | None = None,
    paired_episodes: Path | None = None,
    tokenizer: bool = False,
    pack_length: int = 2048,
    mode: str = "test",
) -> tuple[subprocess.CompletedProcess, dict]:
    primary_data, primary_episodes = generated[arm]
    counterpart_data, counterpart_episodes = generated[PAIRED[arm]]
    command = [
        sys.executable,
        str(AUDITOR),
        "--mode",
        mode,
        "--arm",
        arm,
        "--data",
        str(data or primary_data),
        "--episodes",
        str(episodes or primary_episodes),
        "--paired-data",
        str(paired_data or counterpart_data),
        "--paired-episodes",
        str(paired_episodes or counterpart_episodes),
        "--heldout",
        str(heldout),
        "--out",
        str(out),
        "--pack-length",
        str(pack_length),
    ]
    if mode == "test":
        command.extend(("--test-scale", "1"))
    if tokenizer:
        command.extend(("--tokenizer", str(TOKENIZER)))
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert out.is_file()
    return process, json.loads(out.read_text())


def test_all_declared_arms_pass_mechanics_but_test_mode_never_admits(
    tmp_path: Path,
) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = generate_all(tmp_path, heldout)
    assert TOKENIZER.is_file()
    for arm in ARMS:
        process, report = run_audit(
            generated,
            heldout,
            tmp_path / "audit-{}.json".format(arm),
            arm,
            tokenizer=arm == "iid",
        )
        assert process.returncode != 0
        assert "without production admission" in process.stderr
        assert report["mode"] == "test"
        assert report["mechanical_pass"] is True
        assert report["test_mechanics_pass"] is True
        assert report["production_admission"] is False
        assert report["admission_pass"] is False
        assert report["failures"] == {}
        assert all(report["checks"].values())
        assert report["target"] == {
            "episodes": 400,
            "transitions": 2_000,
            "rows": 4_400,
        }
        paired = report["paired_board"]
        assert paired["literal_jsonl_equal"] is True
        assert paired["full_visible_feature_equality"] is True
        assert paired["position_counts_equal"] is True
        assert report["pair_distribution"]["all_cells_nonconcentrated"] is True
        assert report["contamination"]["train_heldout_reserved_signature_hits"] == 0
        assert (
            report["contamination"]["train_heldout_exact_normalized_prompt_hits"] == 0
        )
        assert report["contamination"]["train_heldout_literal_13gram_hits"] == 0
        assert report["residual_bundle_confound"]


def test_token_accounting_matches_exact_production_build_packed(tmp_path: Path) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = generate_all(tmp_path, heldout)

    from sft import DEFAULT_Q_FIELDS, DEFAULT_R_FIELDS, build_packed
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None
    for arm in ARMS:
        _, report = run_audit(
            generated,
            heldout,
            tmp_path / "token-audit-{}.json".format(arm),
            arm,
            tokenizer=True,
        )
        _, _, _, production = build_packed(
            [generated[arm][0]],
            tokenizer,
            2048,
            DEFAULT_Q_FIELDS,
            DEFAULT_R_FIELDS,
            eos_id,
            return_stats=True,
        )
        audited = report["tokenizer_accounting"]
        assert "without a prompt override" in audited["encoding_boundary"]
        assert audited["production_build_packed"] == production
        for field, expected in CANONICAL_TEST_PACKING[arm].items():
            assert production[field] == expected

    _, short_report = run_audit(
        generated,
        heldout,
        tmp_path / "overlength-audit.json",
        "iid",
        tokenizer=True,
        pack_length=2,
    )
    assert short_report["mechanical_pass"] is False
    assert not short_report["checks"]["tokenizer_no_overlength_rows"]
    assert not short_report["checks"]["tokenizer_no_zero_pack"]
    assert not short_report["checks"]["tokenizer_all_rows_accepted"]


def test_wrong_arm_terminal_labels_and_paired_board_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = generate_all(tmp_path, heldout)

    iid_copy = tmp_path / "iid-copy.jsonl"
    iid_copy.write_bytes(generated["iid"][0].read_bytes())
    wrong_process, wrong_report = run_audit(
        generated,
        heldout,
        tmp_path / "wrong-arm.json",
        "term",
        data=generated["iid"][0],
        paired_data=iid_copy,
    )
    assert wrong_process.returncode != 0
    assert wrong_report["mechanical_pass"] is False
    assert any("row_arm" in key for key in wrong_report["failures"])

    episodes = read_jsonl(generated["term"][1])
    target_episode = next(e for e in episodes if e["operation"] == "add")
    target_episode["terminal_class"] = (
        "11" if target_episode["terminal_class"] != "11" else "00"
    )
    bad_episodes = tmp_path / "bad-terminal-class-episodes.jsonl"
    write_jsonl(bad_episodes, episodes)
    _, class_report = run_audit(
        generated,
        heldout,
        tmp_path / "bad-terminal-class-audit.json",
        "term",
        episodes=bad_episodes,
    )
    assert class_report["mechanical_pass"] is False
    assert any(
        "episode_terminal_class_label" in key for key in class_report["failures"]
    )

    rows = read_jsonl(generated["term"][0])
    terminal_row = next(
        row
        for row in rows
        if row["kind"] == "transition"
        and row["operation"] == "add"
        and row["transition_index"] == row["width"] - 1
    )
    corrupted_state = parse_state(terminal_row["expected_state"])
    assert corrupted_state is not None and corrupted_state["z"] == 1
    corrupted_state["c"] = 1 - corrupted_state["c"]
    corrupted_line = canonical_state(corrupted_state)
    terminal_row["expected_state"] = corrupted_line
    terminal_row["response"] = corrupted_line
    bad_data = tmp_path / "bad-terminal-label-data.jsonl"
    write_jsonl(bad_data, rows)
    _, label_report = run_audit(
        generated,
        heldout,
        tmp_path / "bad-terminal-label-audit.json",
        "term",
        data=bad_data,
    )
    assert label_report["mechanical_pass"] is False
    assert any("transition_witness" in key for key in label_report["failures"])

    corrupt_pair = tmp_path / "paired-episodes-corrupt.jsonl"
    corrupt_pair.write_bytes(generated["iid"][1].read_bytes() + b"\n")
    _, pair_report = run_audit(
        generated,
        heldout,
        tmp_path / "paired-board-audit.json",
        "term",
        paired_episodes=corrupt_pair,
    )
    assert pair_report["mechanical_pass"] is False
    assert not pair_report["checks"]["paired_board_literal_bytes"]


def test_heldout_signature_prompt_and_solver_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = generate_all(tmp_path, heldout)

    train_episode = read_jsonl(generated["term"][1])[0]
    collision = heldout_episode(
        "fixture-train-collision",
        train_episode["operation"],
        train_episode["left"],
        train_episode["right"],
        train_episode["width"],
    )
    contaminated_heldout = make_heldout_fixture(
        tmp_path / "contaminated-heldout.jsonl", collision
    )
    _, contaminated_report = run_audit(
        generated,
        contaminated_heldout,
        tmp_path / "contaminated-audit.json",
        "term",
    )
    assert contaminated_report["mechanical_pass"] is False
    assert (
        contaminated_report["contamination"]["train_heldout_reserved_signature_hits"]
        >= 1
    )

    heldout_document = read_jsonl(heldout)[0]
    heldout_state = parse_state(heldout_document["initial_state"])
    assert heldout_state is not None
    heldout_prompt = microstep_prompt(heldout_state, style="heldout")
    prompt_rows = read_jsonl(generated["term"][0])
    prompt_rows[0]["question"] = heldout_prompt
    prompt_rows[0]["completion_prompt"] = heldout_prompt
    prompt_contaminated = tmp_path / "prompt-contaminated.jsonl"
    write_jsonl(prompt_contaminated, prompt_rows)
    _, prompt_report = run_audit(
        generated,
        heldout,
        tmp_path / "prompt-contaminated-audit.json",
        "term",
        data=prompt_contaminated,
    )
    assert prompt_report["mechanical_pass"] is False
    assert (
        prompt_report["contamination"]["train_heldout_exact_normalized_prompt_hits"]
        >= 1
    )

    corrupted = read_jsonl(heldout)
    corrupted[0]["counterfactual"]["expected_states"][0] = (
        "not-a-canonical-heldout-state"
    )
    corrupt_heldout = tmp_path / "corrupt-heldout.jsonl"
    write_jsonl(corrupt_heldout, corrupted)
    _, corrupt_report = run_audit(
        generated,
        corrupt_heldout,
        tmp_path / "corrupt-heldout-audit.json",
        "term",
    )
    assert corrupt_report["mechanical_pass"] is False
    assert any("heldout_transition_target" in key for key in corrupt_report["failures"])
    assert not corrupt_report["checks"]["heldout_solver_valid"]


def test_production_mode_and_all_final_partial_aliases_fail_closed(
    tmp_path: Path,
) -> None:
    heldout = make_heldout_fixture(tmp_path / "heldout.jsonl")
    generated = generate_all(tmp_path, heldout)
    process, report = run_audit(
        generated,
        heldout,
        tmp_path / "unfrozen-production-audit.json",
        "term",
        tokenizer=True,
        mode="production",
    )
    assert process.returncode != 0
    assert report["production_contract"] is True
    assert report["production_admission"] is False
    assert report["admission_pass"] is False
    assert not report["checks"]["heldout_frozen_sha256"]

    data, episodes = generated["term"]
    paired_data, paired_episodes = generated["iid"]
    alias_out = data.with_name(data.name + ".partial")
    alias = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--mode",
            "test",
            "--arm",
            "term",
            "--data",
            str(data),
            "--episodes",
            str(episodes),
            "--paired-data",
            str(paired_data),
            "--paired-episodes",
            str(paired_episodes),
            "--heldout",
            str(heldout),
            "--out",
            str(alias_out),
            "--test-scale",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert alias.returncode != 0
    assert "final and .partial paths must be pairwise distinct" in alias.stderr
    assert not alias_out.exists()
