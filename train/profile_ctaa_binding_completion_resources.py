#!/usr/bin/env python3
"""Measure capability-time parity for the two decisive CTAA readouts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import time

import torch

from ctaa_binding_completion import (
    FactorizedBindingReadout,
    GlobalStructuredBindingReadout,
    factorized_loss,
    readout_resource_receipt,
)
from ctaa_binding_completion_admission import (
    load_admission,
    require_admitted_artifact_path,
    require_admitted_protocol_source,
)
from freeze_ctaa_binding_completion_seeds import write_json_once
from predict_ctaa_binding_completion import load_seed_freeze
from train_ctaa_binding_completion import (
    fixed_schedule,
    load_state,
    safe_torch_load,
    sha256_file,
    tagged_seed,
    tensor_sha256,
    validate_frozen_seed,
)


SCHEMA = "r12_ctaa_a4_binding_completion_measured_resources_v1"
WARMUP_UPDATES = 20
MEASURED_UPDATES = 100


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def profile_resources(
    *,
    admission_path: Path,
    seed_freeze_manifest_path: Path,
    frozen_seed_path: Path,
    output: Path,
    device_name: str,
) -> dict[str, object]:
    admission = load_admission(admission_path)
    require_admitted_protocol_source(admission)
    require_admitted_artifact_path(
        output,
        admission,
        "resource_artifact_name",
    )
    require_admitted_artifact_path(
        seed_freeze_manifest_path,
        admission,
        "seed_freeze_manifest_name",
    )
    if (
        frozen_seed_path.resolve().parent
        != Path(str(admission["custody_root"]))
        or frozen_seed_path.name != admission["seed_artifact_names"][0]
    ):
        raise ValueError("CTAA completion resource seed identity differs")
    admission_sha256 = sha256_file(admission_path)
    freeze_records, freeze_sha256 = load_seed_freeze(
        seed_freeze_manifest_path,
        admission_sha256=admission_sha256,
    )
    frozen, frozen_sha256 = safe_torch_load(
        frozen_seed_path,
        expected_sha256=str(freeze_records[0]["artifact_sha256"]),
    )
    seed = int(admission["seeds"][0])
    validate_frozen_seed(
        frozen,
        admission=admission,
        admission_sha256=admission_sha256,
        expected_seed=seed,
    )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CTAA completion resource profiler requires CUDA")
    batch_size = int(admission["batch_size"])
    learning_rate = float(admission["learning_rate"])
    slots = frozen["train_slot_cache"].float()
    bindings = frozen["train_bindings"].long()
    total = WARMUP_UPDATES + MEASURED_UPDATES
    schedule = fixed_schedule(
        slots.shape[0],
        total,
        batch_size,
        tagged_seed(seed, "measured-resource-batches"),
    )
    models = {
        "factorized": FactorizedBindingReadout().to(device),
        "global_structured": GlobalStructuredBindingReadout().to(device),
    }
    for name, model in models.items():
        load_state(model, frozen["arm_states"][name])
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )
        for name, model in models.items()
    }
    measurements = {name: [] for name in models}
    peak_memory = {name: [] for name in models}
    for update in range(total):
        order = (
            ("factorized", "global_structured")
            if update % 2 == 0
            else ("global_structured", "factorized")
        )
        for name in order:
            model = models[name]
            optimizer = optimizers[name]
            indices = schedule[update]
            selected_slots = slots.index_select(0, indices).to(device)
            selected_bindings = bindings.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = factorized_loss(
                    model(selected_slots),
                    selected_bindings,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if update >= WARMUP_UPDATES:
                measurements[name].append(elapsed_ms)
                peak_memory[name].append(
                    int(torch.cuda.max_memory_allocated(device))
                )
    receipt = readout_resource_receipt()
    arms = {
        name: {
            "parameters": int(receipt[f"{name}_parameters"]),
            "analytic_macs_per_row": int(receipt[f"{name}_macs"]),
            "median_update_ms": statistics.median(values),
            "p95_update_ms": percentile(values, 0.95),
            "measured_updates": len(values),
            "update_ms": values,
            "peak_memory_bytes": peak_memory[name],
        }
        for name, values in measurements.items()
    }
    medians = [float(value["median_update_ms"]) for value in arms.values()]
    relative_runtime_gap = abs(medians[0] - medians[1]) / max(medians)
    memory_peaks = [
        max(value["peak_memory_bytes"])
        for value in arms.values()
    ]
    relative_peak_memory_gap = (
        abs(memory_peaks[0] - memory_peaks[1]) / max(memory_peaks)
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "admission_sha256": admission_sha256,
        "code_commit": admission["code_commit"],
        "protocol_source_sha256": admission["protocol_source_sha256"],
        "seed_freeze_sha256": freeze_sha256,
        "frozen_seed_sha256": frozen_sha256,
        "seed": seed,
        "device_name": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "slots_sha256": tensor_sha256(slots),
        "bindings_sha256": tensor_sha256(bindings),
        "schedule_sha256": tensor_sha256(schedule),
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "counterbalanced_order": True,
        "arms": arms,
        "analytic_relative_mac_gap": receipt["relative_mac_gap"],
        "measured_relative_runtime_gap": relative_runtime_gap,
        "measured_relative_peak_memory_gap": relative_peak_memory_gap,
        "resource_gate": (
            receipt["factorized_parameters"]
            == receipt["global_structured_parameters"]
            and receipt["relative_mac_gap"]
            <= float(admission["maximum_resource_relative_gap"])
            and relative_runtime_gap
            <= float(admission["maximum_resource_relative_gap"])
            and relative_peak_memory_gap
            <= float(admission["maximum_resource_relative_gap"])
        ),
    }
    digest = write_json_once(output, payload)
    return {
        "resource_sha256": digest,
        "measured_relative_runtime_gap": relative_runtime_gap,
        "measured_relative_peak_memory_gap": relative_peak_memory_gap,
        "resource_gate": payload["resource_gate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--seed-freeze-manifest", type=Path, required=True)
    parser.add_argument("--frozen-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = profile_resources(
        admission_path=args.admission,
        seed_freeze_manifest_path=args.seed_freeze_manifest,
        frozen_seed_path=args.frozen_seed,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
