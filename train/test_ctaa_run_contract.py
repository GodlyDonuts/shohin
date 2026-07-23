from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

import ctaa_bootstrap_seed_receipt as bootstrap
import ctaa_run_contract as contract
from ctaa_bootstrap_seed_receipt import build_receipt
from ctaa_evaluation_io import sha256_file
from ctaa_run_contract import (
    ARMS,
    CORE_TRAINING_SCHEMA,
    RUN_CONTRACT_SCHEMA,
    RUN_INPUT_SCHEMA,
    RUN_PLAN_SCHEMA,
    RunContractError,
    canonical_json,
    create_run_contract,
    inspect_sealed_manifest,
    validate_run_contract,
)
from test_ctaa_bootstrap_seed_receipt import (
    TEST_ROOT_PUBLIC,
    _beacon as _signed_test_beacon,
)


bootstrap.CUSTODY_ROOT_PUBLIC_KEY_HEX = TEST_ROOT_PUBLIC


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(mode)


def _edit_json(path: Path, change: Callable[[dict[str, object]], None]) -> None:
    parent_mode = path.parent.stat().st_mode & 0o777
    path.parent.chmod(parent_mode | 0o200)
    path.chmod(0o644)
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o444)
    path.parent.chmod(parent_mode)


def _core(seed: int, arm: str) -> dict[str, object]:
    return {
        "core_sha256": _hash(f"core:{seed}:{arm}"),
        "core_kind": "outer_product_control"
        if arm == "oprc_closure"
        else "closure_feature",
        "training_schema": CORE_TRAINING_SCHEMA,
        "training_seed": seed,
        "training_arm": arm,
        "atomic_sha256": _hash("atomic"),
        "closure_sha256": _hash("closure"),
        "updates": 2000,
        "batch_size": 256,
        "learning_rate": 0.001,
    }


def _raw_receipt(
    *,
    seed: int,
    arm: str,
    dataset: str,
    files: dict[str, str],
) -> dict[str, object]:
    prefix = f"development{'_intervention' if dataset == 'intervention' else ''}"
    core = _core(seed, arm)
    identity = f"{seed}:{arm}:{dataset}"
    return {
        "schema": "r12_ctaa_v2_raw_evidence_receipt_v1",
        "rows": 100,
        "valid_packets": 97,
        "executed_rows": 97,
        "queried_rows": 97,
        "answered_rows": 97,
        "program_predictions_sha256": _hash(f"program-predictions:{seed}:{dataset}"),
        "compiler_sha256": _hash(f"compiler:{seed}"),
        "program_source_sha256": files[f"{prefix}_program.jsonl"],
        "query_source_sha256": files[f"{prefix}_query.jsonl"],
        "packet_index_sha256": _hash(f"packet-index:{seed}:{dataset}"),
        "execution_sha256": _hash(f"execution:{identity}"),
        "core_sha256": core["core_sha256"],
        "core_kind": core["core_kind"],
        "core_training": core,
        "query_predictions_sha256": _hash(f"query-predictions:{identity}"),
        "query_positions_sha256": _hash(f"query-positions:{identity}"),
        "answers_sha256": _hash(f"answers:{identity}"),
        "evidence_sha256": _hash(f"evidence:{identity}"),
        "oracle_access": 0,
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    board = tmp_path / "board"
    board.mkdir(parents=True)
    files = {
        "development_program.jsonl": _hash("base-program"),
        "development_query.jsonl": _hash("base-query"),
        "development_oracle.jsonl": _hash("base-oracle"),
        "development_intervention_program.jsonl": _hash("intervention-program"),
        "development_intervention_query.jsonl": _hash("intervention-query"),
        "development_intervention_oracle.jsonl": _hash("intervention-oracle"),
        "confirmation_program.jsonl": _hash("confirmation-program"),
        "confirmation_query.jsonl": _hash("confirmation-query"),
        "confirmation_oracle.jsonl": _hash("confirmation-oracle"),
        "access_ledger.json": _hash("initial-ledger"),
    }
    manifest = board / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": "r12_ctaa_v2_manifest_v1",
            "seed": 918273,
            "files": files,
        },
    )
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    receipts: dict[tuple[int, str, str], Path] = {}
    seeds = [101, 202, 303, 404, 505]
    for seed in seeds:
        for arm in ARMS:
            for dataset in ("base", "intervention"):
                directory = evidence_root / f"{seed}-{arm}-{dataset}"
                receipt = directory / "receipt.json"
                _write_json(
                    receipt,
                    _raw_receipt(seed=seed, arm=arm, dataset=dataset, files=files),
                )
                directory.chmod(0o555)
                receipts[(seed, arm, dataset)] = receipt
    anchors = inspect_sealed_manifest(manifest)
    bootstrap_receipt = tmp_path / "bootstrap.json"
    _write_json(
        bootstrap_receipt,
        build_receipt(
            source_commit="a" * 40,
            manifest_sha256=anchors["manifest_sha256"],
            gate_source_sha256=sha256_file(
                Path(contract.__file__).with_name("evaluate_ctaa_advancement_gates.py")
            ),
            statistics_source_sha256=sha256_file(
                Path(contract.__file__).with_name("ctaa_gate_statistics.py")
            ),
            beacon=_signed_test_beacon(),
        ),
    )
    runs = []
    for seed in reversed(seeds):
        for arm in reversed(ARMS):
            for dataset in reversed(("base", "intervention")):
                runs.append(
                    {
                        "schema": RUN_INPUT_SCHEMA,
                        "seed": seed,
                        "arm": arm,
                        "dataset": dataset,
                        "evidence_receipt_path": str(receipts[(seed, arm, dataset)]),
                        "parent_evidence_receipt_path": (
                            str(receipts[(seed, arm, "base")])
                            if dataset == "intervention"
                            else None
                        ),
                        "core_training": _core(seed, arm),
                    }
                )
    plan = tmp_path / "run_plan.json"
    _write_json(
        plan,
        {
            "schema": RUN_PLAN_SCHEMA,
            "partition": "development",
            "expected_manifest_sha256": anchors["manifest_sha256"],
            "expected_board_sha256": anchors["board_sha256"],
            "runs": runs,
        },
    )
    return {
        "manifest": manifest,
        "bootstrap": bootstrap_receipt,
        "plan": plan,
        "receipts": receipts,
        "files": files,
        "output": tmp_path / "run_contract.json",
    }


def _create(paths: dict[str, object]) -> dict[str, object]:
    return create_run_contract(
        manifest_path=Path(paths["manifest"]),
        run_plan_path=Path(paths["plan"]),
        bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
        output_path=Path(paths["output"]),
    )


def test_contract_is_canonical_outcome_free_and_recomputable(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _create(paths)
    output = Path(paths["output"])
    validated = validate_run_contract(
        contract_path=output,
        manifest_path=Path(paths["manifest"]),
        run_plan_path=Path(paths["plan"]),
        bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
    )

    assert validated == result
    assert result["schema"] == RUN_CONTRACT_SCHEMA
    assert result["run_count"] == 40
    assert result["training_seeds"] == [101, 202, 303, 404, 505]
    assert len(result["runs"]) == 40
    assert (
        result["manifest_sha256"]
        == inspect_sealed_manifest(Path(paths["manifest"]))["manifest_sha256"]
    )
    assert (
        result["board_sha256"]
        == inspect_sealed_manifest(Path(paths["manifest"]))["board_sha256"]
    )
    assert output.stat().st_mode & 0o222 == 0
    assert output.read_text() == canonical_json(result) + "\n"
    encoded = canonical_json(result).lower()
    for forbidden in (
        "accuracy",
        "score",
        "threshold",
        "capability",
        "family_id",
        "oracle_value",
        "all_gates_pass",
    ):
        assert forbidden not in encoded
    intervention = next(
        row
        for row in result["runs"]
        if row["seed"] == 101
        and row["arm"] == "ctaa_closure"
        and row["dataset"] == "intervention"
    )
    base = next(
        row
        for row in result["runs"]
        if row["seed"] == 101
        and row["arm"] == "ctaa_closure"
        and row["dataset"] == "base"
    )
    assert (
        intervention["parent_evidence_receipt_sha256"]
        == base["raw_evidence_receipt_sha256"]
    )
    assert (
        intervention["parent_evidence_sha256"]
        == base["evidence_artifacts"]["evidence_sha256"]
    )


@pytest.mark.parametrize("mode", ["omitted", "duplicate"])
def test_exactly_forty_complete_unique_runs_are_required(
    tmp_path: Path, mode: str
) -> None:
    paths = _fixture(tmp_path)

    def change(plan: dict[str, object]) -> None:
        runs = plan["runs"]
        assert isinstance(runs, list)
        if mode == "omitted":
            runs.pop()
        else:
            runs[-1] = dict(runs[0])

    _edit_json(Path(paths["plan"]), change)
    with pytest.raises(RunContractError, match="exactly 40|identities repeat"):
        _create(paths)


def test_swapped_receipt_and_core_metadata_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def swap_receipts(plan: dict[str, object]) -> None:
        runs = plan["runs"]
        assert isinstance(runs, list)
        left = next(
            row
            for row in runs
            if row["seed"] == 101 and row["arm"] == ARMS[0] and row["dataset"] == "base"
        )
        right = next(
            row
            for row in runs
            if row["seed"] == 101 and row["arm"] == ARMS[1] and row["dataset"] == "base"
        )
        left["evidence_receipt_path"], right["evidence_receipt_path"] = (
            right["evidence_receipt_path"],
            left["evidence_receipt_path"],
        )

    _edit_json(Path(paths["plan"]), swap_receipts)
    with pytest.raises(RunContractError, match="core-training metadata was swapped"):
        _create(paths)

    paths = _fixture(tmp_path / "core")

    def swap_core(plan: dict[str, object]) -> None:
        runs = plan["runs"]
        assert isinstance(runs, list)
        row = runs[0]
        row["core_training"] = _core(101, "ctaa_closure")

    _edit_json(Path(paths["plan"]), swap_core)
    with pytest.raises(
        RunContractError, match="run-plan core metadata|metadata was swapped"
    ):
        _create(paths)


def test_oracle_files_are_never_opened_or_required_to_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    original = contract.os.open

    def guarded_open(path, flags, *args, **kwargs):
        if Path(path).name.endswith("_oracle.jsonl"):
            raise AssertionError("run contract attempted to open an oracle file")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(contract.os, "open", guarded_open)
    result = _create(paths)
    assert result["oracle_files"]["base"] == {
        "filename": "development_oracle.jsonl",
        "sha256": _hash("base-oracle"),
    }
    assert not (Path(paths["manifest"]).parent / "development_oracle.jsonl").exists()


@pytest.mark.parametrize(
    "target", ["manifest", "plan", "bootstrap", "receipt", "contract"]
)
def test_mutable_inputs_and_receipt_are_rejected(tmp_path: Path, target: str) -> None:
    paths = _fixture(tmp_path)
    if target == "contract":
        _create(paths)
        path = Path(paths["output"])
        path.chmod(0o644)
        with pytest.raises(RunContractError, match="read-only"):
            validate_run_contract(
                contract_path=path,
                manifest_path=Path(paths["manifest"]),
                run_plan_path=Path(paths["plan"]),
                bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
            )
        return
    if target == "receipt":
        path = next(iter(paths["receipts"].values()))
    else:
        path = Path(paths[target])
    path.chmod(0o644)
    with pytest.raises(RunContractError, match="read-only"):
        _create(paths)


@pytest.mark.parametrize("target", ["manifest", "plan", "run", "receipt", "contract"])
def test_unknown_keys_are_rejected(tmp_path: Path, target: str) -> None:
    paths = _fixture(tmp_path)
    if target == "manifest":
        _edit_json(Path(paths["manifest"]), lambda value: value.__setitem__("extra", 1))
    elif target == "plan":
        _edit_json(Path(paths["plan"]), lambda value: value.__setitem__("extra", 1))
    elif target == "run":

        def add_run_key(value: dict[str, object]) -> None:
            value["runs"][0]["extra"] = 1

        _edit_json(Path(paths["plan"]), add_run_key)
    elif target == "receipt":
        receipt = next(iter(paths["receipts"].values()))
        _edit_json(receipt, lambda value: value.__setitem__("extra", 1))
    else:
        _create(paths)
        output = Path(paths["output"])
        value = json.loads(output.read_text())
        value["extra"] = 1
        output.chmod(0o644)
        output.write_text(canonical_json(value) + "\n")
        output.chmod(0o444)
        with pytest.raises(RunContractError, match="schema differs"):
            validate_run_contract(
                contract_path=Path(paths["output"]),
                manifest_path=Path(paths["manifest"]),
                run_plan_path=Path(paths["plan"]),
                bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
            )
        return
    with pytest.raises(RunContractError, match="schema differs"):
        _create(paths)


def test_manifest_and_file_hash_substitution_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def substitute_oracle(value: dict[str, object]) -> None:
        value["files"]["development_oracle.jsonl"] = _hash("substituted-oracle")

    _edit_json(Path(paths["manifest"]), substitute_oracle)
    with pytest.raises(RunContractError, match="substituted"):
        _create(paths)

    paths = _fixture(tmp_path / "source")

    def substitute_source(value: dict[str, object]) -> None:
        value["files"]["development_program.jsonl"] = _hash("substituted-program")

    _edit_json(Path(paths["manifest"]), substitute_source)
    anchors = inspect_sealed_manifest(Path(paths["manifest"]))

    def reanchor(value: dict[str, object]) -> None:
        value["expected_manifest_sha256"] = anchors["manifest_sha256"]
        value["expected_board_sha256"] = anchors["board_sha256"]

    _edit_json(Path(paths["plan"]), reanchor)
    with pytest.raises(RunContractError, match="sealed sources|bootstrap receipt"):
        _create(paths)


def test_published_contract_detects_later_manifest_substitution(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _create(paths)

    def substitute(value: dict[str, object]) -> None:
        value["files"]["access_ledger.json"] = _hash("replacement-ledger")

    _edit_json(Path(paths["manifest"]), substitute)
    with pytest.raises(RunContractError, match="substituted"):
        validate_run_contract(
            contract_path=Path(paths["output"]),
            manifest_path=Path(paths["manifest"]),
            run_plan_path=Path(paths["plan"]),
            bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
        )


def test_published_contract_detects_bootstrap_receipt_substitution(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _create(paths)
    replacement = tmp_path / "replacement-bootstrap.json"
    anchors = inspect_sealed_manifest(Path(paths["manifest"]))
    _write_json(
        replacement,
        build_receipt(
            source_commit="a" * 40,
            manifest_sha256=anchors["manifest_sha256"],
            gate_source_sha256=sha256_file(
                Path(contract.__file__).with_name("evaluate_ctaa_advancement_gates.py")
            ),
            statistics_source_sha256=sha256_file(
                Path(contract.__file__).with_name("ctaa_gate_statistics.py")
            ),
            beacon=_signed_test_beacon(output_value="cd" * 64),
        ),
    )
    with pytest.raises(RunContractError, match="differs from pre-access inputs"):
        validate_run_contract(
            contract_path=Path(paths["output"]),
            manifest_path=Path(paths["manifest"]),
            run_plan_path=Path(paths["plan"]),
            bootstrap_seed_receipt_path=replacement,
        )


def test_intervention_parent_must_be_exact_paired_base_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def mismatch(value: dict[str, object]) -> None:
        runs = value["runs"]
        child = next(
            row
            for row in runs
            if row["seed"] == 101
            and row["arm"] == ARMS[0]
            and row["dataset"] == "intervention"
        )
        child["parent_evidence_receipt_path"] = str(
            paths["receipts"][(101, ARMS[1], "base")]
        )

    _edit_json(Path(paths["plan"]), mismatch)
    with pytest.raises(RunContractError, match="parent evidence differs"):
        _create(paths)


@pytest.mark.parametrize(
    "field", ["accuracy", "score", "threshold", "family_id", "capability_pass"]
)
def test_caller_authored_outcome_and_family_fields_are_rejected(
    tmp_path: Path, field: str
) -> None:
    paths = _fixture(tmp_path)

    def inject(value: dict[str, object]) -> None:
        value["runs"][0][field] = 1

    _edit_json(Path(paths["plan"]), inject)
    with pytest.raises(RunContractError, match="run-plan entry schema differs"):
        _create(paths)


def test_outcome_field_in_raw_receipt_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = next(iter(paths["receipts"].values()))
    _edit_json(receipt, lambda value: value.__setitem__("answer_accuracy", 0.99))
    with pytest.raises(RunContractError, match="schema differs"):
        _create(paths)


def test_repeated_raw_evidence_and_core_identities_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = paths["receipts"][(101, ARMS[0], "base")]
    second = paths["receipts"][(101, ARMS[1], "base")]
    first_value = json.loads(first.read_text())
    second.parent.chmod(0o755)
    second.chmod(0o644)
    second.write_text(json.dumps(first_value, sort_keys=True, indent=2) + "\n")
    second.chmod(0o444)
    second.parent.chmod(0o555)
    with pytest.raises(
        RunContractError, match="metadata was swapped|identities repeat"
    ):
        _create(paths)


def test_write_once_refuses_existing_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = Path(paths["output"])
    output.write_text("occupied")
    with pytest.raises(FileExistsError):
        _create(paths)


def test_noncanonical_published_encoding_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _create(paths)
    output = Path(paths["output"])
    value = json.loads(output.read_text())
    output.chmod(0o644)
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    output.chmod(0o444)
    with pytest.raises(RunContractError, match="not canonical JSON"):
        validate_run_contract(
            contract_path=output,
            manifest_path=Path(paths["manifest"]),
            run_plan_path=Path(paths["plan"]),
            bootstrap_seed_receipt_path=Path(paths["bootstrap"]),
        )


def test_committed_reader_rejects_symlinked_intermediate_parent(tmp_path: Path) -> None:
    real = tmp_path / "real" / "nested"
    real.mkdir(parents=True)
    target = real / "value.json"
    target.write_bytes(b"authentic\n")
    target.chmod(0o444)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(RunContractError, match="cannot be opened safely"):
        contract._read_committed_bytes(alias / "nested" / target.name, "test input")


def test_committed_reader_holds_parent_across_symlink_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visible = tmp_path / "visible"
    authentic_parent = visible / "nested"
    authentic_parent.mkdir(parents=True)
    authentic = authentic_parent / "value.json"
    authentic.write_bytes(b"authentic\n")
    authentic.chmod(0o444)

    decoy = tmp_path / "decoy"
    decoy_parent = decoy / "nested"
    decoy_parent.mkdir(parents=True)
    decoy_file = decoy_parent / authentic.name
    decoy_file.write_bytes(b"decoy\n")
    decoy_file.chmod(0o444)
    held = tmp_path / "held"
    original = contract.os.open
    mutated = False

    def mutate_after_open(path, flags, *args, **kwargs):
        nonlocal mutated
        descriptor = original(path, flags, *args, **kwargs)
        if path == visible.name and kwargs.get("dir_fd") is not None and not mutated:
            visible.rename(held)
            visible.symlink_to(decoy, target_is_directory=True)
            mutated = True
        return descriptor

    monkeypatch.setattr(contract.os, "open", mutate_after_open)
    assert contract._read_committed_bytes(authentic, "test input") == b"authentic\n"
    assert mutated
    assert authentic.read_bytes() == b"decoy\n"
