from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import orchestrate_ctaa_evaluation as orchestration
from commit_ctaa_raw_evidence import commit_raw_evidence
from ctaa_neural_core import ClosureFeatureTransitionCore
from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    packet_valid_mask,
    sha256_file,
    write_json_once,
    write_torch_once,
)
from ctaa_packet_io import read_packet_file, read_query_file
from run_ctaa_packet_executor import EXECUTION_SCHEMA, load_core, write_execution_once
from seal_ctaa_late_queries import seal_queries
from seal_ctaa_program_packets import seal_predictions
from prepare_ctaa_program_packets import SCHEMA as PREPARED_SCHEMA


def _arguments(command: list[str]) -> dict[str, Path]:
    return {
        command[index][2:].replace("-", "_"): Path(command[index + 1])
        for index in range(2, len(command) - 1, 2)
        if command[index].startswith("--")
    }


def test_orchestrator_has_no_oracle_surface_and_opens_query_after_execution(tmp_path, monkeypatch) -> None:
    assert "oracle" not in inspect.signature(orchestration.orchestrate).parameters
    inputs = {}
    for name in ("base", "qualified", "tokenizer", "compiler", "core"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = path
    inputs["core"].unlink()
    core_module = ClosureFeatureTransitionCore().eval()
    with torch.no_grad():
        for parameter in core_module.parameters():
            parameter.zero_()
    write_torch_once(
        inputs["core"],
        {
            "schema": "ctaa_recurrent_core_v1",
            "kind": "closure_feature",
            "state": core_module.state_dict(),
            "training": {
                "schema": "r12_ctaa_v2_core_training_v1",
                "arm": "ctaa_closure",
                "seed": 31,
                "atomic_sha256": "1" * 64,
                "closure_sha256": "2" * 64,
            },
        },
    )
    program_source = tmp_path / "program.jsonl"
    program_source.write_text(json.dumps({"family_id": "f0", "program_source": "program"}) + "\n")
    query_source = tmp_path / "query.jsonl"
    query_source.write_text(json.dumps({"family_id": "f0", "query_source": "query"}) + "\n")
    query_source.chmod(0o600)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    prepared_predictions = prepared / "program_predictions.pt"
    prepared_packet = prepared / "program_packets.bin"
    prepared_index = prepared / "packet_index.json"
    schedule = torch.zeros((1, 41), dtype=torch.uint8)
    schedule[:, 1] = 4
    write_torch_once(
        prepared_predictions,
        {
            "schema": PROGRAM_PREDICTION_SCHEMA,
            "family_ids": ["f0"],
            "program_source_sha256": sha256_file(program_source),
            "compiler_sha256": sha256_file(inputs["compiler"]),
            "action_cards": torch.zeros((1, 4, 3), dtype=torch.uint8),
            "initial_state": torch.zeros((1, 3), dtype=torch.uint8),
            "schedule": schedule,
            "packet_valid": packet_valid_mask(schedule),
        },
    )
    sealed = seal_predictions(prepared_predictions, prepared_packet, prepared_index)
    write_json_once(
        prepared / "preparation_receipt.json",
        {
            "schema": PREPARED_SCHEMA,
            "program_source_sha256": sha256_file(program_source),
            "compiler_sha256": sha256_file(inputs["compiler"]),
            "program_predictions_sha256": sha256_file(prepared_predictions),
            "packet_index_sha256": sha256_file(prepared_index),
            "packet_sha256": sha256_file(prepared_packet),
            "valid_rows": sealed["valid_rows"],
            "invalid_rows": sealed["invalid_rows"],
            "stages": [],
            "oracle_access": 0,
        },
    )
    stage_names = []

    def fake_run(
        command: list[str],
        *,
        root: Path,
        hidden_board_root: Path | None = None,
    ) -> dict[str, object]:
        del root
        assert hidden_board_root == query_source.parent
        script = Path(command[1]).name
        stage_names.append(script)
        args = _arguments(command)
        if script == "run_ctaa_packet_executor.py":
            packet_sha = json.loads((args["packet"].parent / "packet_index.json").read_text())["packet_sha256"]
            replay_core, _ = load_core(inputs["core"])
            trace = read_packet_file(args["packet"]).execute_dual(replay_core)
            write_execution_once(
                args["output"],
                {
                    "schema": EXECUTION_SCHEMA,
                    "core_kind": "closure_feature",
                    "packet_sha256": packet_sha,
                    "core_sha256": sha256_file(inputs["core"]),
                    "state_route": trace.state_route.states.to(torch.uint8),
                    "halted": trace.state_route.halted,
                    "composed_cards": trace.composed_cards.to(torch.uint8),
                    "composed_states": trace.composed_states.to(torch.uint8),
                },
            )
        elif script == "run_ctaa_query_compiler.py":
            assert args["execution"].exists()
            write_torch_once(
                args["output"],
                {
                    "schema": QUERY_PREDICTION_SCHEMA,
                    "family_ids": ["f0"],
                    "query_source_sha256": sha256_file(query_source),
                    "compiler_sha256": sha256_file(inputs["compiler"]),
                    "execution_sha256": sha256_file(args["execution"]),
                    "positions": torch.tensor([0], dtype=torch.uint8),
                },
            )
        elif script == "seal_ctaa_late_queries.py":
            seal_queries(
                args["predictions"],
                args["packet_index"],
                args["execution"],
                args["output"],
            )
        elif script == "run_ctaa_late_query.py":
            execution_value = torch.load(
                args["execution"], map_location="cpu", weights_only=True
            )
            position = read_query_file(args["query"]).position
            answer = int(
                execution_value["state_route"][0, -1, int(position[0])]
            )
            write_json_once(
                args["output"],
                {
                    "schema": "ctaa_late_query_answer_v1",
                    "execution_sha256": sha256_file(args["execution"]),
                    "query_sha256": sha256_file(args["query"]),
                    "answers": [answer],
                },
            )
        elif script == "commit_ctaa_raw_evidence.py":
            commit_raw_evidence(
                program_predictions_path=args["program_predictions"],
                packet_index_path=args["packet_index"],
                execution_path=args.get("execution"),
                query_predictions_path=args.get("query_predictions"),
                answers_path=args.get("answers"),
                query_source_path=args.get("query_source_commitment"),
                core_checkpoint_path=args["core_checkpoint_commitment"],
                packet_path=args.get("packet_commitment"),
                hard_query_path=args.get("hard_query_commitment"),
                output_dir=args["output_dir"],
            )
        else:
            raise AssertionError(script)
        return {"argv": command, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(orchestration, "_run", fake_run)
    output = tmp_path / "evaluation"
    report = orchestration.orchestrate(
        base=inputs["base"],
        qualified_compiler=inputs["qualified"],
        tokenizer=inputs["tokenizer"],
        compiler=inputs["compiler"],
        core=inputs["core"],
        prepared_program_root=prepared,
        query_source=query_source,
        output_root=output,
        device="cpu",
        batch_size=1,
        python="python3",
    )
    assert stage_names.index("run_ctaa_packet_executor.py") < stage_names.index(
        "run_ctaa_query_compiler.py"
    )
    assert "run_ctaa_program_compiler.py" not in stage_names
    assert report["oracle_access"] == 0
    assert (output / "raw_evidence" / "evidence.jsonl").exists()

    malicious_stage_names = []

    def fake_successful_malformed_executor(
        command: list[str],
        *,
        root: Path,
        hidden_board_root: Path | None = None,
    ) -> dict[str, object]:
        del root
        assert hidden_board_root == query_source.parent
        script = Path(command[1]).name
        malicious_stage_names.append(script)
        args = _arguments(command)
        if script == "run_ctaa_packet_executor.py":
            write_torch_once(args["output"], {"schema": EXECUTION_SCHEMA})
        elif script == "run_ctaa_query_compiler.py":
            raise AssertionError("query compiler ran after invalid execution")
        elif script == "commit_ctaa_raw_evidence.py":
            commit_raw_evidence(
                program_predictions_path=args["program_predictions"],
                packet_index_path=args["packet_index"],
                execution_path=args.get("execution"),
                query_predictions_path=args.get("query_predictions"),
                answers_path=args.get("answers"),
                query_source_path=args.get("query_source_commitment"),
                core_checkpoint_path=args["core_checkpoint_commitment"],
                packet_path=args.get("packet_commitment"),
                hard_query_path=args.get("hard_query_commitment"),
                output_dir=args["output_dir"],
            )
        else:
            raise AssertionError(script)
        return {"argv": command, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(orchestration, "_run", fake_successful_malformed_executor)
    malformed_output = tmp_path / "malformed_execution_evaluation"
    malformed = orchestration.orchestrate(
        base=inputs["base"],
        qualified_compiler=inputs["qualified"],
        tokenizer=inputs["tokenizer"],
        compiler=inputs["compiler"],
        core=inputs["core"],
        prepared_program_root=prepared,
        query_source=query_source,
        output_root=malformed_output,
        device="cpu",
        batch_size=1,
        python="python3",
    )
    assert "run_ctaa_query_compiler.py" not in malicious_stage_names
    assert not (malformed_output / "disclosed_query.jsonl").exists()
    assert any(
        stage.get("argv") == ["validate_execution_artifact"]
        and stage.get("succeeded") is False
        for stage in malformed["stages"]
    )
    malformed_row = json.loads(
        (malformed_output / "raw_evidence" / "evidence.jsonl").read_text()
    )
    assert malformed_row["packet_valid"] is True
    assert malformed_row["state_route"] is None
    assert malformed_row["answer"] is None

    def fail_late_query(
        command: list[str],
        *,
        root: Path,
        hidden_board_root: Path | None = None,
    ) -> dict[str, object]:
        if Path(command[1]).name == "run_ctaa_late_query.py":
            raise RuntimeError("injected late-query failure")
        return fake_run(
            command,
            root=root,
            hidden_board_root=hidden_board_root,
        )

    monkeypatch.setattr(orchestration, "_run", fail_late_query)
    partial_output = tmp_path / "partial_evaluation"
    partial = orchestration.orchestrate(
        base=inputs["base"],
        qualified_compiler=inputs["qualified"],
        tokenizer=inputs["tokenizer"],
        compiler=inputs["compiler"],
        core=inputs["core"],
        prepared_program_root=prepared,
        query_source=query_source,
        output_root=partial_output,
        device="cpu",
        batch_size=1,
        python="python3",
    )
    partial_row = json.loads(
        (partial_output / "raw_evidence" / "evidence.jsonl").read_text()
    )
    assert partial_row["packet_valid"] is True
    assert partial_row["answer"] is None
    assert any(stage.get("succeeded") is False for stage in partial["stages"])

    receipt_path = prepared / "preparation_receipt.json"
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["program_predictions_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    receipt_path.chmod(0o444)
    with pytest.raises(ValueError, match="prepared"):
        orchestration.orchestrate(
            base=inputs["base"],
            qualified_compiler=inputs["qualified"],
            tokenizer=inputs["tokenizer"],
            compiler=inputs["compiler"],
            core=inputs["core"],
            prepared_program_root=prepared,
            query_source=query_source,
            output_root=tmp_path / "tampered_evaluation",
            device="cpu",
            batch_size=1,
            python="python3",
        )
