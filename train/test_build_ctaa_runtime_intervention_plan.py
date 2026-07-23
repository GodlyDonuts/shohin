from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from build_ctaa_runtime_intervention_plan import (
    RUNTIME_IMPLEMENTATION_SOURCES,
    RuntimePlanBuildError,
    _decode_json,
    build_runtime_intervention_plan,
)
from ctaa_bootstrap_seed_receipt import build_receipt
from ctaa_intervention_protocol import (
    ANCHORS_PER_CLASS_DEPTH,
    ANCHORS_PER_RENDERER,
    ANCHORS_PER_QUERY_STATE_CELL,
    RUNTIME_PANEL_SIZE,
    MANDATORY_OPERATIONS,
    ProtocolValidationError,
    plan_to_dict,
    validate_runtime_intervention_plan,
)
from ctaa_run_contract import (
    ARMS,
    BOARD_TREE_SCHEMA,
    CORE_TRAINING_SCHEMA,
    RUN_BINDING_SCHEMA,
    RUN_CONTRACT_SCHEMA,
    canonical_json,
    canonical_sha256,
)
from pipeline.ctaa_board_v2 import INITIAL_STATES, PROGRAM_CLASSES
from pipeline.generate_ctaa_board import MAX_STEPS, RENDERERS, SCORED_DEPTHS
from test_ctaa_bootstrap_seed_receipt import TEST_ROOT_PUBLIC, _beacon


COMPILER_SHA256 = hashlib.sha256(b"compiler").hexdigest()
BASE_SHA256 = hashlib.sha256(b"base").hexdigest()


def test_runtime_implementation_commitment_covers_the_execution_chain() -> None:
    assert set(RUNTIME_IMPLEMENTATION_SOURCES) == {
        "build_ctaa_runtime_intervention_plan.py",
        "ctaa_intervention_protocol.py",
        "ctaa_runtime_interventions.py",
        "ctaa_trunk_compiler.py",
        "ctaa_neural_core.py",
        "ctaa_packet_io.py",
        "ctaa_runtime_plan_replay.py",
        "ctaa_runtime_execution_projection.py",
        "ctaa_runtime_program_artifacts.py",
        "ctaa_runtime_execution_engine.py",
        "ctaa_runtime_execution_artifact.py",
        "ctaa_runtime_execution_receipt.py",
        "ctaa_runtime_evidence_finalizer.py",
        "ctaa_runtime_execution_set.py",
        "ctaa_runtime_evidence.py",
        "ctaa_runtime_bundle.py",
    }


def _digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    payload = (
        canonical_json(value) + "\n"
        if canonical
        else json.dumps(value, sort_keys=True, indent=2) + "\n"
    )
    _write(path, payload.encode("ascii"))


def _tuple_text(values: tuple[str, str, str], style: int) -> str:
    return (
        "[" + ",".join(values) + "]" if style == 0 else "(" + " | ".join(values) + ")"
    )


def _cards(program_class: str) -> tuple[tuple[int, int, int], ...]:
    if program_class == "explicit_final_collapse":
        return ((0, 0, 1), (0, 1, 2), (1, 0, 2), (0, 0, 0))
    return ((0, 0, 1), (0, 1, 2), (1, 0, 2), (2, 1, 0))


def _source_row(
    *,
    renderer_index: int,
    query_state_cell: int,
    replica: int,
    program_class: str,
    depth: int,
) -> tuple[str, str]:
    renderer = RENDERERS["development"][renderer_index]
    bits = tuple((renderer >> index) & 1 for index in range(6))
    symbols = ("A", "B", "C")
    opcodes = ("OP0", "OP1", "OP2", "OP3")
    cards = _cards(program_class)
    lines = [
        ("SYMBOL ORDER :: " if bits[0] == 0 else "REGISTER ALPHABET = ")
        + _tuple_text(symbols, bits[0])
    ]
    order = tuple(range(4)) if replica == 0 else (2, 0, 3, 1)
    before = _tuple_text(symbols, bits[1])
    for slot in order:
        after = _tuple_text(tuple(symbols[index] for index in cards[slot]), bits[1])
        if bits[1] == 0:
            lines.append(
                f"CARD W{slot + 1}; CODE {opcodes[slot]}; BEFORE {before}; AFTER {after}"
            )
        else:
            lines.append(f"W{slot + 1} binds {opcodes[slot]}: {before} => {after}")
    initial_index = query_state_cell % len(INITIAL_STATES)
    query_position = query_state_cell // len(INITIAL_STATES)
    initial = tuple(symbols[index] for index in INITIAL_STATES[initial_index])
    lines.append(
        ("INITIAL STATE :: " if bits[2] == 0 else "LOAD REGISTERS = ")
        + _tuple_text(initial, bits[2])
    )
    if program_class == "stable_rank_two":
        active = [0, *([1] * (depth - 1))]
    elif program_class == "implicit_final_collapse":
        active = [0, 0, *([1] * (depth - 2))]
    else:
        active = [1] * (depth - 1) + [3]
    suffix = [
        (replica + renderer_index + query_state_cell + offset) % 4
        for offset in range(MAX_STEPS - depth - 1)
    ]
    stop = "STOP" if bits[4] == 0 else "HALT_NOW"
    event_names = (
        [opcodes[index] for index in active]
        + [stop]
        + [opcodes[index] for index in suffix]
    )
    separator = " ; " if bits[4] == 0 else " / "
    lines.append(
        ("EVENT TAPE :: " if bits[3] == 0 else "RUN QUEUE = ")
        + separator.join(event_names)
    )
    positions = (
        ("FIRST", "SECOND", "THIRD") if bits[5] == 0 else ("LEFT", "MIDDLE", "RIGHT")
    )
    query = (
        f"READ THE {positions[query_position]} CELL.\n"
        if bits[5] == 0
        else f"REPORT VALUE AT {positions[query_position]}.\n"
    )
    return "\n".join(lines) + "\n", query


def _core(seed: int, arm: str) -> dict[str, object]:
    return {
        "core_sha256": _digest(f"core:{seed}:{arm}"),
        "core_kind": "outer_product_control"
        if arm == "oprc_closure"
        else "closure_feature",
        "training_schema": CORE_TRAINING_SCHEMA,
        "training_seed": seed,
        "training_arm": arm,
        "atomic_sha256": _digest("atomic"),
        "closure_sha256": _digest("closure"),
        "updates": 20,
        "batch_size": 8,
        "learning_rate": 0.001,
    }


def _run_contract(
    manifest_sha: str,
    board_sha: str,
    files: dict[str, str],
    bootstrap_receipt_sha: str,
    bootstrap_seed: int,
) -> dict[str, object]:
    seeds = [11, 22, 33, 44, 55]
    runs = []
    for seed in seeds:
        for arm in ARMS:
            for dataset in ("base", "intervention"):
                infix = "" if dataset == "base" else "_intervention"
                prefix = f"development{infix}"
                identity = f"seed-{seed}:{arm}:{dataset}"
                evidence = {
                    key: _digest(f"{identity}:{key}")
                    for key in (
                        "program_predictions_sha256",
                        "packet_index_sha256",
                        "execution_sha256",
                        "query_predictions_sha256",
                        "query_positions_sha256",
                        "answers_sha256",
                        "evidence_sha256",
                    )
                }
                runs.append(
                    {
                        "schema": RUN_BINDING_SCHEMA,
                        "run_id": identity,
                        "seed": seed,
                        "arm": arm,
                        "dataset": dataset,
                        "raw_evidence_receipt_sha256": _digest(f"receipt:{identity}"),
                        "compiler_sha256": COMPILER_SHA256,
                        "sealed_sources": {
                            "program_filename": f"{prefix}_program.jsonl",
                            "program_sha256": files[f"{prefix}_program.jsonl"],
                            "query_filename": f"{prefix}_query.jsonl",
                            "query_sha256": files[f"{prefix}_query.jsonl"],
                            "oracle_filename": f"{prefix}_oracle.jsonl",
                            "oracle_sha256": files[f"{prefix}_oracle.jsonl"],
                        },
                        "evidence_artifacts": evidence,
                        "core_training": _core(seed, arm),
                        "parent_run_id": (
                            f"seed-{seed}:{arm}:base"
                            if dataset == "intervention"
                            else None
                        ),
                        "parent_evidence_receipt_sha256": (
                            _digest(f"receipt:seed-{seed}:{arm}:base")
                            if dataset == "intervention"
                            else None
                        ),
                        "parent_evidence_sha256": (
                            _digest(f"seed-{seed}:{arm}:base:evidence_sha256")
                            if dataset == "intervention"
                            else None
                        ),
                    }
                )
    payload = {
        "schema": RUN_CONTRACT_SCHEMA,
        "partition": "development",
        "manifest_sha256": manifest_sha,
        "board_sha256": board_sha,
        "run_plan_sha256": _digest("run-plan"),
        "bootstrap_seed_receipt_sha256": bootstrap_receipt_sha,
        "bootstrap_seed": bootstrap_seed,
        "training_seeds": seeds,
        "arms": list(ARMS),
        "datasets": ["base", "intervention"],
        "run_count": 40,
        "oracle_files": {
            "base": {
                "filename": "development_oracle.jsonl",
                "sha256": files["development_oracle.jsonl"],
            },
            "intervention": {
                "filename": "development_intervention_oracle.jsonl",
                "sha256": files["development_intervention_oracle.jsonl"],
            },
        },
        "runs": runs,
    }
    return {**payload, "run_contract_sha256": canonical_sha256(payload)}


def _board_tree(files: dict[str, str]) -> str:
    return canonical_sha256(
        {
            "schema": BOARD_TREE_SCHEMA,
            "files": [{"name": name, "sha256": files[name]} for name in sorted(files)],
        }
    )


@pytest.fixture(scope="module")
def frozen_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("ctaa-runtime-plan")
    board = root / "board"
    board.mkdir()
    program_rows = []
    query_rows = []
    serial = 0
    for program_class in PROGRAM_CLASSES:
        for depth in SCORED_DEPTHS:
            for renderer_index in range(16):
                for query_state_cell in range(18):
                    for replica in range(2):
                        family_id = f"Dhhh{serial:08d}"
                        serial += 1
                        program, query = _source_row(
                            renderer_index=renderer_index,
                            query_state_cell=query_state_cell,
                            replica=replica,
                            program_class=program_class,
                            depth=depth,
                        )
                        program_rows.append(
                            canonical_json(
                                {"family_id": family_id, "program_source": program}
                            )
                        )
                        query_rows.append(
                            canonical_json(
                                {"family_id": family_id, "query_source": query}
                            )
                        )
    assert serial == 3456
    program_path = board / "development_program.jsonl"
    query_path = board / "development_query.jsonl"
    _write(program_path, ("\n".join(program_rows) + "\n").encode("ascii"))
    _write(query_path, ("\n".join(reversed(query_rows)) + "\n").encode("ascii"))

    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = root / "tokenizer.json"
    _write(tokenizer_path, tokenizer.to_str().encode("utf-8"))

    files = {
        "development_program.jsonl": _digest(program_path.read_bytes()),
        "development_query.jsonl": _digest(query_path.read_bytes()),
        "development_oracle.jsonl": _digest("sealed base oracle"),
        "development_intervention_program.jsonl": _digest("intervention program"),
        "development_intervention_query.jsonl": _digest("intervention query"),
        "development_intervention_oracle.jsonl": _digest("sealed intervention oracle"),
    }
    manifest_path = board / "manifest.json"
    _write_json(
        manifest_path,
        {"schema": "r12_ctaa_v2_manifest_v2", "seed": 1729, "files": files},
    )
    manifest_sha = _digest(manifest_path.read_bytes())
    seed_receipt = build_receipt(
        source_commit="a" * 40,
        manifest_sha256=manifest_sha,
        gate_source_sha256=_digest("gate"),
        statistics_source_sha256=_digest("statistics"),
        beacon=_beacon(),
        custody_root_public_key_hex=TEST_ROOT_PUBLIC,
    )
    seed_path = root / "selection_seed_receipt.json"
    _write_json(seed_path, seed_receipt)
    contract_path = root / "run_contract.json"
    _write_json(
        contract_path,
        _run_contract(
            manifest_sha,
            _board_tree(files),
            files,
            _digest(seed_path.read_bytes()),
            int(seed_receipt["bootstrap_seed"]),
        ),
        canonical=True,
    )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "program_path": program_path,
        "query_path": query_path,
        "tokenizer_path": tokenizer_path,
        "run_contract_path": contract_path,
        "selection_seed_receipt_path": seed_path,
    }


def _build(inputs: dict[str, object], output: Path):
    return build_runtime_intervention_plan(
        manifest_path=Path(inputs["manifest_path"]),
        program_path=Path(inputs["program_path"]),
        query_path=Path(inputs["query_path"]),
        tokenizer_path=Path(inputs["tokenizer_path"]),
        run_contract_path=Path(inputs["run_contract_path"]),
        selection_seed_receipt_path=Path(inputs["selection_seed_receipt_path"]),
        output_path=output,
        compiler_sha256=COMPILER_SHA256,
        base_checkpoint_sha256=BASE_SHA256,
        arm_id="ctaa_closure",
        training_seed=11,
        core_sha256=_digest("core:11:ctaa_closure"),
        core_kind="closure_feature",
        base_raw_evidence_receipt_sha256=_digest("receipt:seed-11:ctaa_closure:base"),
        partition="development",
        custody_root_public_key_hex=TEST_ROOT_PUBLIC,
    )


@pytest.fixture(scope="module")
def built_plan(frozen_inputs: dict[str, object]):
    output = Path(frozen_inputs["root"]) / "runtime_plan_a.json"
    return _build(frozen_inputs, output), output


def test_balanced_864_hhh_panel_and_all_commitments_validate(built_plan) -> None:
    plan, output = built_plan
    assert len(plan.anchors) == RUNTIME_PANEL_SIZE
    assert output.stat().st_mode & 0o777 == 0o444
    assert output.stat().st_nlink == 1
    assert validate_runtime_intervention_plan(plan) == plan
    by_stratum = Counter((row.class_id, row.depth) for row in plan.anchors)
    assert set(by_stratum.values()) == {ANCHORS_PER_CLASS_DEPTH}
    for stratum in by_stratum:
        rows = [row for row in plan.anchors if (row.class_id, row.depth) == stratum]
        assert set(Counter(row.renderer_index for row in rows).values()) == {
            ANCHORS_PER_RENDERER
        }
        assert set(Counter(row.query_state_cell_id for row in rows).values()) == {
            ANCHORS_PER_QUERY_STATE_CELL
        }
    assert all(row.shift_cell == "hhh" for row in plan.anchors)
    assert plan.bindings.run_contract_sha256
    assert plan.bindings.selection_seed_receipt_sha256
    assert len(plan.attempts) == RUNTIME_PANEL_SIZE * len(MANDATORY_OPERATIONS)
    assert len({(item.operation, item.anchor_id) for item in plan.attempts}) == len(
        plan.attempts
    )
    assert [item.attempt_index for item in plan.attempts] == list(
        range(len(plan.attempts))
    )
    by_operation = Counter(item.operation for item in plan.attempts)
    assert set(by_operation) == set(MANDATORY_OPERATIONS)
    assert set(by_operation.values()) == {RUNTIME_PANEL_SIZE}


def test_concrete_attempt_payloads_bind_nonidentity_mutations(built_plan) -> None:
    plan, _ = built_plan
    anchors = {item.anchor_id: item for item in plan.anchors}
    result_program = {
        "entity_recode",
        "witness_recode",
        "opcode_recode",
        "renderer_substitution",
        "rule_line_shuffle",
        "witness_corruption",
        "paired_shuffled_law",
        "schedule_order_twin",
        "source_poison",
        "stop_relocation",
    }
    result_packet = {
        "card_only_counterfactual",
        "binding_only_counterfactual",
        "compensated_opcode_relabel",
        "card_storage_reindex",
        "witness_corruption",
        "paired_shuffled_law",
        "schedule_order_twin",
        "future_mask",
        "stop_relocation",
        "post_stop_poison",
        "packet_transplant",
    }
    for attempt in plan.attempts:
        payload = json.loads(attempt.mutation_payload_json)
        assert payload["schema"] == "r12_ctaa_v2_concrete_mutation_v2"
        assert payload["operation"] == attempt.operation
        assert payload["anchor_id"] == attempt.anchor_id
        assert payload["timing"]
        parent = anchors[attempt.anchor_id]
        if attempt.operation in result_program:
            assert attempt.resulting_program_source_sha256 is not None
            assert (
                attempt.resulting_program_source_sha256 != parent.program_source_sha256
            )
        if attempt.operation in result_packet:
            assert attempt.resulting_packet_sha256 is not None
            assert attempt.resulting_packet_sha256 != parent.packet_sha256
        if attempt.operation == "late_query_swap":
            assert attempt.resulting_query_source_sha256 is not None
            assert attempt.resulting_query_source_sha256 != parent.query_source_sha256
            assert payload["parent_query_position"] != payload["donor_query_position"]
        if attempt.operation in {"midpoint_donor_state", "midpoint_donor_action"}:
            assert payload["midpoint_step"] == parent.depth // 2
        if attempt.operation == "future_mask":
            assert 0 < payload["first_exposure_step"] < parent.depth
        if attempt.operation == "post_stop_poison":
            assert payload["stop_index"] == parent.depth


def test_builder_never_opens_oracle_and_rebuild_is_byte_deterministic(
    frozen_inputs: dict[str, object], built_plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, first_output = built_plan
    original_open = os.open

    def guarded_open(path, *args, **kwargs):
        if "oracle" in os.fspath(path).lower():
            raise AssertionError("runtime-plan builder attempted to open an oracle")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    second_output = Path(frozen_inputs["root"]) / "runtime_plan_b.json"
    second = _build(frozen_inputs, second_output)
    assert first_output.read_bytes() == second_output.read_bytes()
    assert second.plan_sha256 == built_plan[0].plan_sha256


def test_donor_substitution_is_rejected(built_plan) -> None:
    plan, _ = built_plan
    payload = plan_to_dict(plan)
    pair = payload["donor_derangements"][0]["pairs"][0]
    pair["donor_anchor_id"] = pair["anchor_id"]
    with pytest.raises(ProtocolValidationError, match="fixed point"):
        validate_runtime_intervention_plan(payload)


def test_manifest_detects_read_only_input_mutation(
    frozen_inputs: dict[str, object], tmp_path: Path
) -> None:
    hostile = tmp_path / "development_program.jsonl"
    shutil.copyfile(Path(frozen_inputs["program_path"]), hostile)
    hostile.write_bytes(hostile.read_bytes().replace(b"OP0", b"XP0", 1))
    hostile.chmod(0o444)
    changed = dict(frozen_inputs)
    changed["program_path"] = hostile
    with pytest.raises(RuntimePlanBuildError, match="manifest binding"):
        _build(changed, tmp_path / "plan.json")


@pytest.mark.parametrize("kind", ["writable", "symlink", "hardlink"])
def test_unsafe_input_inode_is_rejected(
    frozen_inputs: dict[str, object], tmp_path: Path, kind: str
) -> None:
    source = Path(frozen_inputs["query_path"])
    hostile = tmp_path / "development_query.jsonl"
    if kind == "writable":
        shutil.copyfile(source, hostile)
        hostile.chmod(0o644)
    elif kind == "symlink":
        hostile.symlink_to(source)
    else:
        os.link(source, hostile)
    changed = dict(frozen_inputs)
    changed["query_path"] = hostile
    try:
        with pytest.raises(RuntimePlanBuildError, match="single-link read-only"):
            _build(changed, tmp_path / "plan.json")
    finally:
        if kind == "hardlink" and hostile.exists():
            hostile.unlink()


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{"x":1,"x":2}', "duplicate"),
        (b'{"x":NaN}', "non-finite"),
        (b'{"x":Infinity}', "non-finite"),
    ],
)
def test_duplicate_and_nonfinite_json_are_rejected(raw: bytes, match: str) -> None:
    with pytest.raises(RuntimePlanBuildError, match=match):
        _decode_json(raw, "hostile")


def test_existing_output_is_never_replaced(
    frozen_inputs: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "plan.json"
    _write(output, b"do not replace\n")
    with pytest.raises(FileExistsError):
        _build(frozen_inputs, output)
    assert output.read_bytes() == b"do not replace\n"
