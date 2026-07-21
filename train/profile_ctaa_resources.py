#!/usr/bin/env python3
"""Scoreless parameter, state, FLOP, and optional runtime CTAA profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ctaa_artifact_loader import (
    load_qualified_memory_state,
    load_raw_trunk,
    verify_complete_system_parameters,
)
from ctaa_evaluation_io import sha256_file, write_json_once
from ctaa_neural_core import (
    CTAA_ACTION_COUNT,
    CTAA_MAX_STEPS,
    ClosureFeatureTransitionCore,
    OuterProductTransitionControl,
    execute_streamed_dual,
)
from ctaa_trunk_compiler import TrunkCausalCTAACompiler


SCHEMA = "r12_ctaa_v2_resource_profile_v1"
DUAL_ROUTE_CALLS_PER_ROW = CTAA_MAX_STEPS * 3


def _runtime_profile(core, device: torch.device, batch_size: int, repeats: int) -> dict[str, object]:
    if batch_size < 1 or repeats < 1:
        raise ValueError("CTAA runtime profile configuration differs")
    cards = torch.tensor(
        [[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 0, 1]]],
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1, -1)
    initial = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device).expand(batch_size, -1)
    schedule = torch.zeros((batch_size, CTAA_MAX_STEPS), dtype=torch.long, device=device)
    schedule[:, 32] = CTAA_ACTION_COUNT
    schedule[:, 33:] = 3
    core = core.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            execute_streamed_dual(core, 3, cards, schedule, initial)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            execute_streamed_dual(core, 3, cards, schedule, initial)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "device": str(device),
        "batch_size": batch_size,
        "repeats": repeats,
        "milliseconds_per_batch": 1000.0 * elapsed / repeats,
        "rows_per_second": batch_size * repeats / elapsed,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def profile(
    *,
    base_path: Path,
    qualified_path: Path,
    output_path: Path,
    runtime_device: str | None,
    batch_size: int,
    repeats: int,
) -> dict[str, object]:
    trunk, base_receipt = load_raw_trunk(base_path)
    qualified = load_qualified_memory_state(qualified_path)
    compiler = TrunkCausalCTAACompiler(trunk)
    loaded = compiler.initialize_qualified_memory(qualified)
    treatment = ClosureFeatureTransitionCore()
    control = OuterProductTransitionControl()
    ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        treatment.unique_parameters,
    )
    charged_transition_flops = max(
        treatment.analytic_inference_flops,
        control.analytic_inference_flops,
    )
    runtime = None
    if runtime_device is not None:
        device = torch.device(runtime_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CTAA runtime profile requires available CUDA")
        runtime = {
            "closure_feature": _runtime_profile(treatment, device, batch_size, repeats),
            "outer_product_control": _runtime_profile(control, device, batch_size, repeats),
        }
    report = {
        "schema": SCHEMA,
        "base_sha256": base_receipt.sha256,
        "base_step": base_receipt.step,
        "qualified_compiler_sha256": sha256_file(qualified_path),
        "qualified_memory_tensors": len(loaded),
        "parameter_ledger": ledger,
        "core_parameters": {
            "closure_feature": treatment.unique_parameters,
            "outer_product_control": control.unique_parameters,
            "exactly_matched": treatment.unique_parameters == control.unique_parameters,
        },
        "transition_flops": {
            "closure_feature_analytic": treatment.analytic_inference_flops,
            "outer_product_control_analytic": control.analytic_inference_flops,
            "charged_per_call": charged_transition_flops,
            "treatment_padding_charge": charged_transition_flops
            - treatment.analytic_inference_flops,
            "control_padding_charge": charged_transition_flops
            - control.analytic_inference_flops,
        },
        "state_contract": {
            "hard_packet_bytes_per_row": 56,
            "semantic_recurrent_state_bytes": 3,
            "implementation_recurrent_state_int64_bytes": 24,
            "halt_state_bytes": 1,
            "matched_across_arms": True,
        },
        "evaluation_charge": {
            "dual_route_core_calls_per_row": DUAL_ROUTE_CALLS_PER_ROW,
            "charged_core_flops_per_row": DUAL_ROUTE_CALLS_PER_ROW
            * charged_transition_flops,
            "note": "evaluation-only route agreement executes one state call plus two composition-route calls per fixed event",
        },
        "runtime": runtime,
        "board_seed_generated": False,
        "oracle_access": 0,
        "all_static_gates_pass": (
            treatment.unique_parameters == control.unique_parameters == 107_753
            and ledger["total"] < 150_000_000
            and len(loaded) == 63
        ),
    }
    report_sha = write_json_once(output_path, report)
    return {**report, "report_sha256": report_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            profile(
                base_path=args.base,
                qualified_path=args.qualified_compiler,
                output_path=args.output,
                runtime_device=args.runtime_device,
                batch_size=args.batch_size,
                repeats=args.repeats,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

