#!/usr/bin/env python3
"""Run the oracle-blind CTAA evaluation stages in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ctaa_evaluation_io import (
    read_packet_index,
    read_program_predictions,
    sha256_file,
    write_json_once,
)
from ctaa_process_sandbox import hidden_board_command
from prepare_ctaa_program_packets import SCHEMA as PREPARED_SCHEMA
from run_ctaa_packet_executor import validate_execution_artifact


SCHEMA = "r12_ctaa_v2_orchestration_v1"


def _load_prepared_program_packets(
    prepared_root: Path,
    compiler: Path,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    predictions = prepared_root / "program_predictions.pt"
    packet = prepared_root / "program_packets.bin"
    packet_index = prepared_root / "packet_index.json"
    receipt_path = prepared_root / "preparation_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    index = read_packet_index(packet_index)
    program = read_program_predictions(predictions)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != PREPARED_SCHEMA
        or receipt.get("compiler_sha256") != sha256_file(compiler)
        or receipt.get("compiler_sha256") != program["compiler_sha256"]
        or receipt.get("program_source_sha256") != program["program_source_sha256"]
        or receipt.get("program_predictions_sha256") != sha256_file(predictions)
        or receipt.get("packet_index_sha256") != sha256_file(packet_index)
        or index["program_predictions_sha256"] != sha256_file(predictions)
        or receipt.get("valid_rows") != len(index["valid_family_ids"])
        or receipt.get("invalid_rows") != len(index["invalid_family_ids"])
        or any(path.stat().st_mode & 0o222 for path in (predictions, packet_index, receipt_path))
    ):
        raise ValueError("CTAA prepared program-packet receipt differs")
    if index["valid_family_ids"]:
        if (
            not packet.exists()
            or packet.stat().st_mode & 0o222
            or receipt.get("packet_sha256") != sha256_file(packet)
            or index["packet_sha256"] != sha256_file(packet)
        ):
            raise ValueError("CTAA prepared program packet differs")
    elif packet.exists() or receipt.get("packet_sha256") is not None:
        raise ValueError("CTAA empty prepared packet differs")
    return predictions, packet, packet_index, index, receipt


def _run(
    command: list[str],
    *,
    root: Path,
    hidden_board_root: Path | None = None,
) -> dict[str, object]:
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[1]
    python_path = os.pathsep.join((str(repository / "train"), str(repository)))
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else python_path + os.pathsep + environment["PYTHONPATH"]
    )
    actual_command = (
        hidden_board_command(
            command,
            writable_root=root,
            board_root=hidden_board_root,
        )
        if hidden_board_root is not None
        else command
    )
    result = subprocess.run(
        actual_command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "CTAA evaluation stage failed: "
            + json.dumps(
                {
                    "argv": command,
                    "sandbox_argv": actual_command,
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-4000:],
                    "stderr_tail": result.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    return {
        "argv": command,
        "sandboxed": hidden_board_root is not None,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def orchestrate(
    *,
    base: Path,
    qualified_compiler: Path,
    tokenizer: Path,
    compiler: Path,
    core: Path,
    prepared_program_root: Path,
    query_source: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    python: str,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing existing CTAA evaluation root: {output_root}")
    board_root = query_source.resolve().parent
    predictions, packet, packet_index, index, preparation = (
        _load_prepared_program_packets(prepared_program_root, compiler)
    )
    output_root.mkdir(parents=True, mode=0o700)
    scripts = Path(__file__).resolve().parent
    execution = output_root / "execution.pt"
    query_predictions = output_root / "query_predictions.pt"
    disclosed_query = output_root / "disclosed_query.jsonl"
    hard_query = output_root / "late_query.bin"
    answers = output_root / "answers.json"
    evidence = output_root / "raw_evidence"
    stages = []
    def try_stage(command: list[str]) -> bool:
        try:
            stages.append(
                {
                    **_run(
                        command,
                        root=output_root,
                        hidden_board_root=board_root,
                    ),
                    "succeeded": True,
                }
            )
            return True
        except RuntimeError as error:
            stages.append(
                {
                    "argv": command,
                    "succeeded": False,
                    "error": str(error),
                }
            )
            return False

    if index["valid_family_ids"]:
        execution_ok = try_stage(
            [
                python,
                str(scripts / "run_ctaa_packet_executor.py"),
                "--packet",
                str(packet),
                "--core",
                str(core),
                "--output",
                str(execution),
            ]
        )
        if execution_ok:
            try:
                validate_execution_artifact(execution, packet, core)
                stages.append(
                    {
                        "argv": ["validate_execution_artifact"],
                        "execution_sha256": sha256_file(execution),
                        "succeeded": True,
                    }
                )
            except (OSError, PermissionError, ValueError) as error:
                stages.append(
                    {
                        "argv": ["validate_execution_artifact"],
                        "succeeded": False,
                        "error": str(error),
                    }
                )
                execution_ok = False
        if execution_ok:
            if query_source.stat().st_mode & 0o077:
                raise PermissionError("CTAA sealed query source permissions differ")
            temporary_query = disclosed_query.with_name(disclosed_query.name + ".tmp")
            if temporary_query.exists():
                raise FileExistsError("refusing existing CTAA query disclosure temporary")
            shutil.copyfile(query_source, temporary_query)
            temporary_query.chmod(0o444)
            temporary_query.replace(disclosed_query)
        query_ok = execution_ok and try_stage(
            [
                python,
                str(scripts / "run_ctaa_query_compiler.py"),
                "--base",
                str(base),
                "--qualified-compiler",
                str(qualified_compiler),
                "--tokenizer",
                str(tokenizer),
                "--compiler",
                str(compiler),
                "--packet-index",
                str(packet_index),
                "--execution",
                str(execution),
                "--query-source",
                str(disclosed_query),
                "--output",
                str(query_predictions),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
            ]
        )
        seal_ok = query_ok and try_stage(
            [
                python,
                str(scripts / "seal_ctaa_late_queries.py"),
                "--predictions",
                str(query_predictions),
                "--packet-index",
                str(packet_index),
                "--execution",
                str(execution),
                "--output",
                str(hard_query),
            ]
        )
        if seal_ok:
            try_stage(
                [
                    python,
                    str(scripts / "run_ctaa_late_query.py"),
                    "--execution",
                    str(execution),
                    "--query",
                    str(hard_query),
                    "--output",
                    str(answers),
                ]
            )
    commit_command = [
        python,
        str(scripts / "commit_ctaa_raw_evidence.py"),
        "--program-predictions",
        str(predictions),
        "--packet-index",
        str(packet_index),
        "--output-dir",
        str(evidence),
        "--core-checkpoint-commitment",
        str(core),
    ]
    if disclosed_query.exists():
        commit_command.extend(("--query-source-commitment", str(disclosed_query)))
    optional_paths = (
        ("--packet-commitment", packet if index["valid_family_ids"] else None),
        ("--execution", execution if execution_ok else None),
        ("--query-predictions", query_predictions),
        ("--hard-query-commitment", hard_query),
        ("--answers", answers),
    )
    for flag, path in optional_paths:
        if path is not None and path.exists():
            commit_command.extend((flag, str(path)))
    stages.append(
        _run(
            commit_command,
            root=output_root,
            hidden_board_root=board_root,
        )
    )
    report = {
        "schema": SCHEMA,
        "valid_rows": len(index["valid_family_ids"]),
        "invalid_rows": len(index["invalid_family_ids"]),
        "program_source_sha256": preparation["program_source_sha256"],
        "program_predictions_sha256": sha256_file(predictions),
        "packet_index_sha256": sha256_file(packet_index),
        "packet_sha256": index["packet_sha256"],
        "query_source_sha256": (
            sha256_file(disclosed_query) if disclosed_query.exists() else None
        ),
        "compiler_sha256": sha256_file(compiler),
        "core_sha256": sha256_file(core),
        "raw_evidence_receipt_sha256": sha256_file(evidence / "receipt.json"),
        "stages": stages,
        "oracle_access": 0,
    }
    report_sha = write_json_once(output_root / "orchestration_receipt.json", report)
    return {**report, "report_sha256": report_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--prepared-program-root", type=Path, required=True)
    parser.add_argument("--query-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(
        json.dumps(
            orchestrate(
                base=args.base,
                qualified_compiler=args.qualified_compiler,
                tokenizer=args.tokenizer,
                compiler=args.compiler,
                core=args.core,
                prepared_program_root=args.prepared_program_root,
                query_source=args.query_source,
                output_root=args.output_root,
                device=args.device,
                batch_size=args.batch_size,
                python=args.python,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
