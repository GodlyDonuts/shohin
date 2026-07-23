#!/usr/bin/env python3
"""Commit the five source-free CTAA seed artifacts before odd-source access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from train_ctaa_binding_completion import (
    safe_torch_load,
    sha256_file,
    tensor_mapping_sha256,
    validate_frozen_seed,
)


SCHEMA = "r12_ctaa_a4_binding_completion_seed_freeze_v1"


def write_json_once(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o444)
    try:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.chmod(0o600)
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def freeze_seeds(
    *,
    admission_path: Path,
    frozen_seed_paths: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    admission = load_admission(admission_path)
    require_admitted_protocol_source(admission)
    require_admitted_artifact_path(
        output,
        admission,
        "seed_freeze_manifest_name",
    )
    if len(frozen_seed_paths) != 5:
        raise ValueError("CTAA completion seed freeze lattice differs")
    admission_sha256 = sha256_file(admission_path)
    records = []
    for index, (path, seed, name) in enumerate(
        zip(
            frozen_seed_paths,
            admission["seeds"],
            admission["seed_artifact_names"],
            strict=True,
        )
    ):
        if path.resolve().parent != Path(str(admission["custody_root"])):
            raise ValueError("CTAA completion seed custody root differs")
        if path.name != name:
            raise ValueError("CTAA completion seed freeze order differs")
        value, artifact_sha256 = safe_torch_load(path)
        validate_frozen_seed(
            value,
            admission=admission,
            admission_sha256=admission_sha256,
            expected_seed=int(seed),
        )
        records.append(
            {
                "index": index,
                "seed": seed,
                "artifact_name": name,
                "artifact_sha256": artifact_sha256,
                "train_cache_bundle_sha256": value["training"][
                    "train_cache_bundle_sha256"
                ],
                "common_compiler_state_sha256": tensor_mapping_sha256(
                    value["common_compiler_state"]
                ),
            }
        )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "admission_sha256": admission_sha256,
        "code_commit": admission["code_commit"],
        "protocol_source_sha256": admission["protocol_source_sha256"],
        "seed_records": records,
        "confirmation_source_access": 0,
        "confirmation_oracle_access": 0,
    }
    digest = write_json_once(output, payload)
    return {
        "seed_freeze_sha256": digest,
        "seeds": [record["seed"] for record in records],
        "confirmation_source_access": 0,
        "confirmation_oracle_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument(
        "--frozen-seed",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_seeds(
        admission_path=args.admission,
        frozen_seed_paths=args.frozen_seed,
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
