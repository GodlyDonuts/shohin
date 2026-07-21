#!/usr/bin/env python3
"""Write a single-use immutable CTAA artifact and parameter receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctaa_artifact_loader import (
    load_qualified_memory_state,
    load_raw_trunk,
    sha256_file,
    verify_complete_system_parameters,
)
from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl
from ctaa_trunk_compiler import TrunkCausalCTAACompiler


SCHEMA = "r12_ctaa_v2_immutable_artifact_preflight_v1"


def write_once(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA artifact receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA receipt temporary: {temporary}")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trunk, base_receipt = load_raw_trunk(args.base)
    qualified = load_qualified_memory_state(args.qualified_compiler)
    compiler = TrunkCausalCTAACompiler(trunk)
    loaded = compiler.initialize_qualified_memory(qualified)
    treatment = ClosureFeatureTransitionCore()
    control = OuterProductTransitionControl()
    if treatment.unique_parameters != control.unique_parameters:
        raise ValueError("CTAA matched core parameter count differs")
    ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        treatment.unique_parameters,
    )
    receipt = {
        "schema": SCHEMA,
        "base": {
            "path": str(args.base),
            "sha256": base_receipt.sha256,
            "step": base_receipt.step,
            "strict_missing_keys": list(base_receipt.missing_keys),
            "strict_unexpected_keys": list(base_receipt.unexpected_keys),
        },
        "qualified_compiler": {
            "path": str(args.qualified_compiler),
            "sha256": sha256_file(args.qualified_compiler),
            "memory_tensors_present": len(qualified),
            "memory_tensors_loaded": len(loaded),
        },
        "parameter_ledger": ledger,
        "core_match": {
            "treatment_parameters": treatment.unique_parameters,
            "control_parameters": control.unique_parameters,
            "treatment_flops": treatment.analytic_inference_flops,
            "control_flops": control.analytic_inference_flops,
        },
        "production_seed_generated": False,
        "board_artifact_written": False,
        "jobs_launched": False,
        "all_gates_pass": True,
    }
    write_once(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
