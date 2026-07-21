#!/usr/bin/env python3
"""Compile and seal one immutable CTAA program packet set for paired arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ctaa_evaluation_io import read_packet_index, sha256_file, write_json_once


SCHEMA = "r12_ctaa_v2_prepared_program_packets_v1"


def _run(command: list[str], *, root: Path) -> dict[str, object]:
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[1]
    python_path = os.pathsep.join((str(repository / "train"), str(repository)))
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else python_path + os.pathsep + environment["PYTHONPATH"]
    )
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "CTAA program preparation failed: "
            + json.dumps(
                {
                    "argv": command,
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-4000:],
                    "stderr_tail": result.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    return {
        "argv": command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def prepare(
    *,
    base: Path,
    qualified_compiler: Path,
    tokenizer: Path,
    compiler: Path,
    program_source: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    python: str,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing existing CTAA prepared packet root: {output_root}")
    output_root.mkdir(parents=True, mode=0o700)
    scripts = Path(__file__).resolve().parent
    predictions = output_root / "program_predictions.pt"
    packet = output_root / "program_packets.bin"
    packet_index = output_root / "packet_index.json"
    stages = [
        _run(
            [
                python,
                str(scripts / "run_ctaa_program_compiler.py"),
                "--base",
                str(base),
                "--qualified-compiler",
                str(qualified_compiler),
                "--tokenizer",
                str(tokenizer),
                "--compiler",
                str(compiler),
                "--program-source",
                str(program_source),
                "--output",
                str(predictions),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
            ],
            root=output_root,
        ),
        _run(
            [
                python,
                str(scripts / "seal_ctaa_program_packets.py"),
                "--predictions",
                str(predictions),
                "--packet",
                str(packet),
                "--index",
                str(packet_index),
            ],
            root=output_root,
        ),
    ]
    index = read_packet_index(packet_index)
    report = {
        "schema": SCHEMA,
        "program_source_sha256": sha256_file(program_source),
        "compiler_sha256": sha256_file(compiler),
        "program_predictions_sha256": sha256_file(predictions),
        "packet_index_sha256": sha256_file(packet_index),
        "packet_sha256": sha256_file(packet) if packet.exists() else None,
        "valid_rows": len(index["valid_family_ids"]),
        "invalid_rows": len(index["invalid_family_ids"]),
        "stages": stages,
        "oracle_access": 0,
    }
    write_json_once(output_root / "preparation_receipt.json", report)
    output_root.chmod(0o555)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--program-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                base=args.base,
                qualified_compiler=args.qualified_compiler,
                tokenizer=args.tokenizer,
                compiler=args.compiler,
                program_source=args.program_source,
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
