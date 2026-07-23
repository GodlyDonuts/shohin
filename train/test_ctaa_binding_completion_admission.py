from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import pytest

from ctaa_binding_completion_admission import (
    PROTOCOL_SOURCE_PATHS,
    SCHEMA,
    protocol_source_sha256,
    require_admitted_protocol_source,
    validate_admission,
)


def admission() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "code_commit": "a" * 40,
        "protocol_source_sha256": "0" * 64,
        "custody_root": str(Path("/tmp/ctaa-completion-custody").resolve()),
        "base_sha256": "1" * 64,
        "qualified_compiler_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "board_manifest_sha256": "4" * 64,
        "train_even_sha256": "5" * 64,
        "confirmation_source_sha256": "6" * 64,
        "confirmation_oracle_sha256": "7" * 64,
        "seeds": [11, 13, 17, 19, 23],
        "seed_artifact_names": [f"seed-{index}.pt" for index in range(5)],
        "prediction_artifact_name": "predictions.pt",
        "assessment_artifact_name": "assessment.pt",
        "capacity_artifact_name": "capacity.pt",
        "seed_freeze_manifest_name": "frozen-seeds.json",
        "oracle_access_ledger_name": "oracle-access.json",
        "decision_artifact_name": "decision.json",
        "resource_artifact_name": "resources.json",
        "qualifier_updates": 2000,
        "readout_updates": 2000,
        "capacity_updates": 2000,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "minimum_train_exact": 0.99,
        "minimum_confirmation_factorized_exact": 0.75,
        "minimum_factorized_advantage": 0.10,
        "maximum_single_slot_exact": 0.10,
        "minimum_chimera_exact": 0.75,
        "minimum_seed_passes": 5,
        "maximum_resource_relative_gap": 0.05,
    }


def test_admission_binds_five_seeds_hyperparameters_and_artifacts() -> None:
    checked = validate_admission(admission())
    assert checked["seeds"] == (11, 13, 17, 19, 23)
    assert checked["seed_artifact_names"][0] == "seed-0.pt"


def test_admission_rejects_duplicate_seed_and_path_escape() -> None:
    duplicate = admission()
    duplicate["seeds"] = [11, 11, 17, 19, 23]
    with pytest.raises(ValueError, match="seeds"):
        validate_admission(duplicate)
    escaped = admission()
    escaped["prediction_artifact_name"] = "../prediction.pt"
    with pytest.raises(ValueError, match="artifact name"):
        validate_admission(escaped)
    relaxed = admission()
    relaxed["minimum_seed_passes"] = 1
    with pytest.raises(ValueError, match="canonical decision"):
        validate_admission(relaxed)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("batch_size", True, "integer"),
        ("readout_updates", 2.0, "integer"),
        ("learning_rate", math.inf, "float"),
        ("minimum_train_exact", 1, "float"),
        ("minimum_seed_passes", 6, "integer"),
    ),
)
def test_admission_rejects_coercible_or_nonfinite_numbers(
    key: str,
    value: object,
    message: str,
) -> None:
    changed = admission()
    changed[key] = value
    with pytest.raises(ValueError, match=message):
        validate_admission(changed)


def test_protocol_bundle_requires_tracked_clean_exact_sources(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repo"
    for relative in PROTOCOL_SOURCE_PATHS:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ctaa@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CTAA Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "protocol"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    receipt = {
        "code_commit": commit,
        "protocol_source_sha256": protocol_source_sha256(repository),
    }
    require_admitted_protocol_source(receipt, repository)
    first = repository / PROTOCOL_SOURCE_PATHS[0]
    first.write_text(first.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        require_admitted_protocol_source(receipt, repository)
