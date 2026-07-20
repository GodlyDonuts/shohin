from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile

import pytest
import torch

from model import GPT, GPTConfig
from sd_cst import (
    EVENT_STEPS,
    HardLateQuery,
    HardProgramTape,
    HardRolloutResult,
    SDCSTSystem,
    STOP_KIND,
)
from train_sd_cst import (
    BOARD_SCHEMA,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    EncodedRow,
    board_manifest,
    deterministic_batches,
    fit_compiler,
    fit_motor_certificate,
    fit_reader_certificate,
    hard_score_payload,
    load_rows,
    motor_certificate,
    reader_certificate,
    seal_hard_payload,
    sha256_bytes,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class CharacterTokenizer:
    def encode(self, text):
        return _Encoding([ord(character) % 47 + 1 for character in text])


def tiny_base() -> GPT:
    return GPT(GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
        seq_len=256,
    ))


def tiny_system() -> SDCSTSystem:
    return SDCSTSystem(
        tiny_base(),
        compiler_layer=0,
        compiler_width=24,
        compiler_heads=4,
        compiler_layers=1,
        compiler_ff=48,
        motor_hidden=96,
        reader_hidden=48,
    )


def fixture_row(split: str, *, oracle: bool = False) -> dict[str, object]:
    slots = []
    for ordinal in range(1, EVENT_STEPS + 1):
        stop = ordinal == 3
        slots.append({
            "semantic_ordinal": ordinal,
            "kind": "stop" if stop else "right",
            "kind_id": STOP_KIND if stop else 1,
            "entity_role": 0 if stop else (ordinal - 1) % 3,
            "amount_id": 0,
            "identity_and_amount_scored": not stop,
        })
    row: dict[str, object] = {
        "id": "fixture-0",
        "split": split,
        "variant": "canonical" if oracle else "compiler_train",
        "family_id": "fixture-family" if oracle else None,
        "program_text": "program-only tokens",
        "late_query_text": "late-query tokens",
        "compiler_targets": {
            "initial_order_roles": [0, 1, 2],
            "initial_state_id": 0,
            "event_slots": slots,
            "halt_after": 2,
        },
        "late_query_target": {"position": 0},
    }
    if oracle:
        row["oracle"] = {
            "final_state_roles": [1, 2, 0],
            "answer_role": 1,
        }
    return row


def write_board(root: Path, split: str, row: dict[str, object]) -> None:
    filename = "train.jsonl" if split == TRAIN_SPLIT else "development.jsonl"
    payload = (json.dumps(row, sort_keys=True) + "\n").encode()
    (root / filename).write_bytes(payload)
    report = {
        "schema": BOARD_SCHEMA,
        "all_gates_pass": True,
        "seed": 17,
        "source_commit": "a" * 40,
        "confirmation_accesses": 0,
        "files": {filename: {"sha256": sha256_bytes(payload)}},
    }
    (root / "report.json").write_text(json.dumps(report))


def valid_hard_payload(batch: int = 1) -> tuple[HardProgramTape, HardLateQuery]:
    kinds = torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8)
    kinds[:, 2] = STOP_KIND
    return HardProgramTape(
        torch.zeros(batch, dtype=torch.uint8),
        kinds,
        torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8),
        torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8),
    ), HardLateQuery(torch.zeros(batch, dtype=torch.uint8))


def development_row() -> EncodedRow:
    return EncodedRow(
        row_id="dev-0",
        split=DEVELOPMENT_SPLIT,
        variant="canonical",
        family_id="family-0",
        program_text="program",
        late_query_text="query",
        program_ids=(1, 2),
        query_ids=(3,),
        initial_state=0,
        event_kind=(0, 0, 2, 0, 0, 0, 0, 0),
        event_identity=(0,) * EVENT_STEPS,
        amount=(0,) * EVENT_STEPS,
        query_position=0,
        halt_after=2,
        final_state=0,
        answer_role=0,
        full_suffix_state=0,
    )


def test_rows_tokenize_program_and_late_query_separately_and_exclude_train_outcomes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        row = fixture_row(TRAIN_SPLIT)
        write_board(root, TRAIN_SPLIT, row)
        rows, manifest = load_rows(root, CharacterTokenizer(), 256, TRAIN_SPLIT)
        assert manifest["opened_file"] == "train.jsonl"
        assert len(rows) == 1
        assert rows[0].program_ids != rows[0].query_ids
        assert rows[0].final_state is None
        leaked = fixture_row(TRAIN_SPLIT)
        leaked["oracle"] = {"answer_role": 0}
        write_board(root, TRAIN_SPLIT, leaked)
        with pytest.raises(ValueError, match="forbidden outcome"):
            load_rows(root, CharacterTokenizer(), 256, TRAIN_SPLIT)


def test_development_loader_requires_oracle_and_no_third_split_is_addressable():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        row = fixture_row(DEVELOPMENT_SPLIT, oracle=True)
        write_board(root, DEVELOPMENT_SPLIT, row)
        rows, manifest = load_rows(
            root,
            CharacterTokenizer(),
            256,
            DEVELOPMENT_SPLIT,
            source_commit="a" * 40,
            access_ledger_dir=root / "ledgers",
        )
        assert manifest["opened_file"] == "development.jsonl"
        assert rows[0].final_state is not None
        assert manifest["access_ledger"]["sha256"]
        with pytest.raises(FileExistsError):
            load_rows(
                root,
                CharacterTokenizer(),
                256,
                DEVELOPMENT_SPLIT,
                source_commit="a" * 40,
                access_ledger_dir=root / "ledgers",
            )
        with pytest.raises(ValueError, match="train or development"):
            board_manifest(root, "sealed")


def test_atomic_certificate_tables_are_complete_and_exhaustive():
    motor = motor_certificate(torch.device("cpu"))
    reader = reader_certificate(torch.device("cpu"))
    assert motor["targets"].shape == (78,)
    assert int((~motor["is_stop"]).sum()) == 72
    assert int(motor["is_stop"].sum()) == 6
    assert reader["targets"].shape == (18,)
    action_keys = torch.cat((
        motor["state"].argmax(-1, keepdim=True),
        motor["event_kind"].argmax(-1, keepdim=True),
        motor["event_identity"].argmax(-1, keepdim=True),
        motor["amount"].argmax(-1, keepdim=True),
    ), dim=1)
    assert len({tuple(row) for row in action_keys[:72].tolist()}) == 72
    reader_keys = torch.cat((
        reader["state"].argmax(-1, keepdim=True),
        reader["query"].argmax(-1, keepdim=True),
    ), dim=1)
    assert len({tuple(row) for row in reader_keys.tolist()}) == 18


def test_atomic_fit_must_reach_all_78_motor_and_18_reader_certificates():
    system = tiny_system()
    motor = fit_motor_certificate(system, seed=31, lr=0.03, max_updates=1800)
    reader = fit_reader_certificate(system, seed=37, lr=0.04, max_updates=900)
    assert motor["state_action_correct"] == 72
    assert motor["stop_correct"] == 6
    assert motor["exact"] is True
    assert motor["updates"] == 1800
    assert reader["correct"] == 18
    assert reader["exact"] is True
    assert reader["updates"] == 900


def test_base_is_frozen_and_complete_parameter_count_is_strictly_below_cap():
    system = tiny_system()
    report = system.parameter_report()
    assert not any(parameter.requires_grad for parameter in system.base_model.parameters())
    assert report["complete_system"] < report["strict_cap"] == 150_000_000
    assert report["headroom"] > 0
    assert report["complete_system"] == sum(parameter.numel() for parameter in system.parameters())


def test_tiny_compiler_phase_uses_only_field_targets_and_separate_sources():
    system = tiny_system()
    row = development_row()
    train_row = replace(
        row,
        row_id="train-0",
        split=TRAIN_SPLIT,
        variant="compiler_train",
        family_id=None,
        program_ids=(1, 2, 3, 4, 5),
        query_ids=(6, 7),
        final_state=None,
        answer_role=None,
        full_suffix_state=None,
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in system.compiler.named_parameters()
        if not name.startswith("base_model.")
    }
    report = fit_compiler(
        system,
        [train_row, replace(train_row, row_id="train-1", query_position=1)],
        seed=43,
        batch_size=2,
        epochs=2,
        lr=1e-3,
        warmup=1,
        clip=1.0,
    )
    assert report["charged_rows"] == 4
    assert report["updates"] == 2
    assert "no state, answer, trajectory" in report["objective"]
    assert any(
        not torch.equal(initial[name], parameter.detach())
        for name, parameter in system.compiler.named_parameters()
        if name in initial
    )
    assert not any(parameter.requires_grad for parameter in system.base_model.parameters())
    assert not any(parameter.requires_grad for parameter in system.motor.parameters())
    assert not any(parameter.requires_grad for parameter in system.reader.parameters())


def test_sealed_payload_is_cpu_uint8_and_exactly_one_stop():
    tape, query = valid_hard_payload(2)
    sealed_tape, sealed_query = seal_hard_payload(tape, query)
    assert sealed_tape.event_kind.device.type == "cpu"
    assert sealed_query.position.device.type == "cpu"
    assert sealed_tape.event_kind.dtype == torch.uint8
    assert sealed_query.position.dtype == torch.uint8
    assert sealed_tape.event_kind.eq(STOP_KIND).sum(dim=1).tolist() == [1, 1]
    invalid = tape.event_kind.clone()
    invalid[:, 4] = STOP_KIND
    with pytest.raises(ValueError, match="exactly one STOP"):
        HardProgramTape(tape.initial_state, invalid, tape.event_identity, tape.amount)


class HardOnlySystem:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def rollout(self, *args, **kwargs):
        raise AssertionError("soft rollout must never be score-bearing")

    def rollout_hard(self, tape, query, **kwargs):
        assert isinstance(tape, HardProgramTape)
        assert isinstance(query, HardLateQuery)
        assert tape.event_kind.dtype == torch.uint8
        assert query.position.dtype == torch.uint8
        self.calls.append(kwargs)
        batch = tape.batch_size
        answer = torch.full((batch, 3), -10.0)
        answer[:, 0] = 10.0
        state = torch.zeros(batch, dtype=torch.uint8)
        return HardRolloutResult(
            final_state=state,
            answer_logits=answer,
            state_trajectory=tuple(state.clone() for _ in range(EVENT_STEPS)),
            alive_trajectory=tuple(torch.ones(batch, dtype=torch.bool) for _ in range(EVENT_STEPS)),
        )


def test_score_path_invokes_rollout_hard_only_for_normal_and_causal_controls():
    system = HardOnlySystem()
    tape, query = valid_hard_payload()
    records, totals = hard_score_payload(system, tape, query, [development_row()])
    assert totals["joint"] == 1
    assert records[0]["joint_correct"] is True
    assert len(system.calls) == 4
    assert system.calls == [
        {},
        {"force_alive": True},
        {"control": "reset"},
        {"control": "freeze"},
    ]
    source = inspect.getsource(hard_score_payload)
    assert ".rollout(" not in source
    assert source.count("rollout_hard") == 4


def test_batch_schedule_is_seeded_complete_and_deterministic():
    first = deterministic_batches(17, 4, 99, 2)
    second = deterministic_batches(17, 4, 99, 2)
    assert first == second
    flat = [value for batch in first for value in batch]
    assert sorted(flat) == list(range(17))
    assert max(map(len, first)) == 4


def test_reader_certificate_targets_are_the_state_query_truth_table():
    board = reader_certificate(torch.device("cpu"))
    states = board["state"].argmax(-1).tolist()
    queries = board["query"].argmax(-1).tolist()
    expected = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    for state, query, target in zip(states, queries, board["targets"].tolist(), strict=True):
        assert target == expected[state][query]


def test_hard_score_rejects_training_rows_without_outcomes():
    system = HardOnlySystem()
    tape, query = valid_hard_payload()
    row = replace(development_row(), split=TRAIN_SPLIT, final_state=None, answer_role=None)
    with pytest.raises(ValueError, match="development oracles"):
        hard_score_payload(system, tape, query, [row])
