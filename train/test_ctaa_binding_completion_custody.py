from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from assess_ctaa_binding_completion import (
    assess,
    claim_oracle_access,
    open_oracle_once,
    packet_metrics,
)
from predict_ctaa_binding_completion import load_source_rows, predict
from train_ctaa_binding_completion import (
    safe_torch_load,
    sha256_file,
    train,
    write_once,
)


TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/tokenizer/tokenizer.json"
)


def test_three_stages_have_physically_separate_inputs() -> None:
    train_parameters = set(inspect.signature(train).parameters)
    predict_parameters = set(inspect.signature(predict).parameters)
    assess_parameters = set(inspect.signature(assess).parameters)
    assert "confirmation_source_path" not in train_parameters
    assert "confirmation_oracle_path" not in train_parameters
    assert "confirmation_oracle_path" not in predict_parameters
    assert "confirmation_source_path" not in assess_parameters
    assert "confirmation_source_path" in predict_parameters
    assert "confirmation_oracle_path" in assess_parameters


def test_source_loader_rejects_any_oracle_field(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    clean = tmp_path / "source.jsonl"
    clean.write_text(
        json.dumps(
            {
                "row_id": "r0",
                "family_id": "f0",
                "program_source": "state",
            }
        )
        + "\n"
    )
    row_ids, family_ids, rows, hashes = load_source_rows(
        clean,
        tokenizer,
        2_048,
    )
    assert row_ids == ["r0"]
    assert family_ids == ["f0"]
    assert len(rows) == len(hashes) == 1
    poisoned = tmp_path / "poisoned.jsonl"
    poisoned.write_text(
        json.dumps(
            {
                "row_id": "r0",
                "family_id": "f0",
                "program_source": "state",
                "opcode_to_card": [0, 1, 2, 3],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="schema"):
        load_source_rows(poisoned, tokenizer, 2_048)


def test_oracle_loader_reads_one_committed_blob_and_rejects_source(
    tmp_path: Path,
) -> None:
    row = {
        "row_id": "r0",
        "family_id": "f0",
        "query_source": "query",
        "action_cards": [[0, 1, 2]] * 4,
        "opcode_to_card": [0, 1, 2, 3],
        "initial_state": [0, 1, 2],
        "opcode_schedule": [4, *([0] * 40)],
        "schedule": [4, *([0] * 40)],
        "query_position": 0,
        "renderer": 0,
    }
    path = tmp_path / "oracle.jsonl"
    encoded = (json.dumps(row) + "\n").encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    loaded, observed = open_oracle_once(path, digest)
    assert loaded == [row]
    assert observed == digest
    poisoned = {**row, "program_source": "forbidden"}
    poisoned_path = tmp_path / "poisoned-oracle.jsonl"
    poisoned_encoded = (json.dumps(poisoned) + "\n").encode()
    poisoned_path.write_bytes(poisoned_encoded)
    with pytest.raises(ValueError, match="schema"):
        open_oracle_once(
            poisoned_path,
            hashlib.sha256(poisoned_encoded).hexdigest(),
        )


def test_packet_metrics_isolate_binding_causality() -> None:
    cards = torch.tensor(
        [[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]]],
        dtype=torch.long,
    )
    initial = torch.tensor([[2, 1, 0]], dtype=torch.long)
    opcode_schedule = torch.tensor(
        [[0, 1, 2, 3, 0, 2, 4, *([0] * 34)]],
        dtype=torch.long,
    )
    binding = torch.tensor([[2, 0, 3, 1]], dtype=torch.long)
    resolved = binding.gather(1, opcode_schedule.clamp_max(3))
    resolved = torch.where(opcode_schedule.eq(4), opcode_schedule, resolved)
    common_logits = {
        "action_cards": F.one_hot(cards, 3).float() * 10.0,
        "initial_state": F.one_hot(initial, 3).float() * 10.0,
        "opcode_schedule": F.one_hot(opcode_schedule, 5).float() * 10.0,
    }
    oracle = [
        {
            "row_id": "r0",
            "family_id": "f0",
            "query_source": "query",
            "action_cards": cards[0].tolist(),
            "opcode_to_card": binding[0].tolist(),
            "initial_state": initial[0].tolist(),
            "opcode_schedule": opcode_schedule[0].tolist(),
            "schedule": resolved[0].tolist(),
            "query_position": 0,
            "renderer": 0,
        }
    ]
    correct_logits = F.one_hot(binding, 4).float() * 10.0
    correct = packet_metrics(
        common_logits,
        correct_logits,
        oracle,
        arm="factorized",
    )
    assert all(
        value == 1.0
        for key, value in correct.items()
        if key != "packet_bytes_per_valid_row"
    )
    assert correct["packet_bytes_per_valid_row"] == 60.0

    corrupted = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    corrupted_metrics = packet_metrics(
        common_logits,
        F.one_hot(corrupted, 4).float() * 10.0,
        oracle,
        arm="factorized",
    )
    assert corrupted_metrics["packet_valid"] == 1.0
    assert corrupted_metrics["cards_exact"] == 1.0
    assert corrupted_metrics["initial_exact"] == 1.0
    assert corrupted_metrics["opcode_schedule_exact"] == 1.0
    assert corrupted_metrics["binding_exact"] == 0.0
    assert corrupted_metrics["resolved_schedule_exact"] == 0.0
    assert corrupted_metrics["program_exact"] == 0.0


def test_safe_artifact_loader_rejects_hash_before_deserialization(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "untrusted.pt"
    artifact.write_bytes(b"not a torch artifact")
    with pytest.raises(ValueError, match="hash differs before load"):
        safe_torch_load(artifact, expected_sha256="0" * 64)
    assert sha256_file(artifact) != "0" * 64


def test_artifact_io_is_exclusive_and_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    digest = write_once(artifact, {"schema": "test", "tensor": torch.arange(3)})
    loaded, observed = safe_torch_load(
        artifact,
        expected_sha256=digest,
    )
    assert observed == digest
    assert loaded["schema"] == "test"
    with pytest.raises(FileExistsError):
        write_once(artifact, {"schema": "replacement"})
    symlink = tmp_path / "artifact-link.pt"
    symlink.symlink_to(artifact)
    with pytest.raises(OSError):
        safe_torch_load(symlink)


def test_oracle_access_claim_is_global_and_atomic(tmp_path: Path) -> None:
    ledger = tmp_path / "oracle-access.json"
    digest = claim_oracle_access(
        ledger,
        admission_sha256="1" * 64,
        oracle_sha256="2" * 64,
        prediction_sha256="3" * 64,
        assessment_output=tmp_path / "assessment.pt",
        code_commit="4" * 40,
        protocol_source_sha256="5" * 64,
    )
    assert digest == hashlib.sha256(ledger.read_bytes()).hexdigest()
    payload = json.loads(ledger.read_text())
    assert payload["access_number"] == 1
    with pytest.raises(FileExistsError):
        claim_oracle_access(
            ledger,
            admission_sha256="1" * 64,
            oracle_sha256="2" * 64,
            prediction_sha256="3" * 64,
            assessment_output=tmp_path / "assessment-elsewhere.pt",
            code_commit="4" * 40,
            protocol_source_sha256="5" * 64,
        )
