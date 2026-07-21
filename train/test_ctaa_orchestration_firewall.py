from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

import orchestrate_ctaa_evaluation as orchestration
from commit_ctaa_raw_evidence import commit_raw_evidence
from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    packet_valid_mask,
    sha256_file,
    write_json_once,
    write_torch_once,
)
from run_ctaa_packet_executor import EXECUTION_SCHEMA, write_execution_once
from seal_ctaa_late_queries import seal_queries
from seal_ctaa_program_packets import seal_predictions


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
    program_source = tmp_path / "program.jsonl"
    program_source.write_text(json.dumps({"family_id": "f0", "program_source": "program"}) + "\n")
    query_source = tmp_path / "query.jsonl"
    query_source.write_text(json.dumps({"family_id": "f0", "query_source": "query"}) + "\n")
    query_source.chmod(0o600)
    stage_names = []

    def fake_run(command: list[str], *, root: Path) -> dict[str, object]:
        del root
        script = Path(command[1]).name
        stage_names.append(script)
        args = _arguments(command)
        if script == "run_ctaa_program_compiler.py":
            schedule = torch.zeros((1, 41), dtype=torch.uint8)
            schedule[:, 1] = 4
            write_torch_once(
                args["output"],
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
        elif script == "seal_ctaa_program_packets.py":
            seal_predictions(args["predictions"], args["packet"], args["index"])
        elif script == "run_ctaa_packet_executor.py":
            packet_sha = json.loads((args["packet"].parent / "packet_index.json").read_text())["packet_sha256"]
            state = torch.zeros((1, 42, 3), dtype=torch.uint8)
            halted = torch.zeros((1, 42), dtype=torch.bool)
            halted[:, 2:] = True
            write_execution_once(
                args["output"],
                {
                    "schema": EXECUTION_SCHEMA,
                    "core_kind": "closure_feature",
                    "packet_sha256": packet_sha,
                    "core_sha256": sha256_file(inputs["core"]),
                    "state_route": state,
                    "halted": halted,
                    "composed_cards": state.clone(),
                    "composed_states": state.clone(),
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
            write_json_once(
                args["output"],
                {
                    "schema": "ctaa_late_query_answer_v1",
                    "execution_sha256": sha256_file(args["execution"]),
                    "query_sha256": sha256_file(args["query"]),
                    "answers": [0],
                },
            )
        elif script == "commit_ctaa_raw_evidence.py":
            commit_raw_evidence(
                program_predictions_path=args["program_predictions"],
                packet_index_path=args["packet_index"],
                execution_path=args["execution"],
                query_predictions_path=args["query_predictions"],
                answers_path=args["answers"],
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
        program_source=program_source,
        query_source=query_source,
        output_root=output,
        device="cpu",
        batch_size=1,
        python="python3",
    )
    assert stage_names.index("run_ctaa_packet_executor.py") < stage_names.index(
        "run_ctaa_query_compiler.py"
    )
    assert report["oracle_access"] == 0
    assert (output / "raw_evidence" / "evidence.jsonl").exists()

