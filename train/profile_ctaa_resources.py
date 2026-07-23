#!/usr/bin/env python3
"""Produce a fail-closed, artifact-bound CTAA resource profile.

The static ledger is useful, but it is not an admission receipt.  A complete
receipt also contains measurements for both matched recurrent-core arms, every
required execution depth, and every required training/inference phase.  Each
measurement repeats the immutable artifact bindings so that observations from
another checkpoint or source corpus cannot be substituted after profiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ctaa_artifact_loader import (
    TOKENIZER_SHA256,
    load_qualified_memory_state,
    load_raw_trunk,
    require_sha256,
    verify_complete_system_parameters,
)
from ctaa_compiler_training import (
    TokenizedCompilerRow,
    collate_compiler_rows,
    compiler_loss,
    parse_train_row,
)
from ctaa_evaluation_io import sha256_file, write_json_once
from ctaa_neural_core import (
    CTAA_ACTION_COUNT,
    CTAA_MAX_STEPS,
    execute_streamed_dual,
)
from ctaa_trunk_compiler import TrunkCausalCTAACompiler
from run_ctaa_packet_executor import load_core


SCHEMA = "r12_ctaa_v2_resource_profile_v2"
OBSERVATION_SCHEMA = "r12_ctaa_v2_resource_observation_v1"
COMPARISON_SCHEMA = "r12_ctaa_v2_matched_resource_comparison_v1"
DUAL_ROUTE_CALLS_PER_ROW = CTAA_MAX_STEPS * 3
PROFILE_DEPTHS = (1, 16, 32, 39)
PROFILE_ARMS = ("closure_feature", "outer_product_control")
PROFILE_PHASES = (
    "curriculum_selection",
    "forward",
    "backward",
    "optimizer_step",
    "compiler_training",
    "inference",
)
SHARED_BINDING_KEYS = (
    "trunk_checkpoint_sha256",
    "qualified_compiler_checkpoint_sha256",
    "compiler_initial_adapter_sha256",
    "tokenizer_sha256",
    "compiler_training_source_sha256",
    "atomic_training_source_sha256",
    "closure_training_source_sha256",
    "curriculum_selection_plan_sha256",
)
CONTEXT_BINDING_KEYS = ("admission_device",)
OBSERVATION_KEYS = {
    "schema",
    "arm",
    "phase",
    "active_depth",
    "device",
    "batch_size",
    "repeats",
    "warmup_count",
    "elapsed_ns",
    "milliseconds_per_iteration",
    "rows_per_second",
    "peak_allocated_bytes",
    "work_units_per_iteration",
    "bindings",
    "observation_sha256",
}
COMPARISON_KEYS = {
    "schema",
    "phase",
    "active_depth",
    "treatment_observation_sha256",
    "control_observation_sha256",
    "shared_bindings_exact",
    "work_units_exact",
    "batch_size_exact",
    "repeats_exact",
    "warmup_count_exact",
    "elapsed_ratio_control_over_treatment",
    "peak_bytes_ratio_control_over_treatment",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _restore_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    own = module.state_dict()
    if not set(state).issubset(own):
        raise ValueError("CTAA resource measurement restore state differs")
    with torch.no_grad():
        for name, value in state.items():
            if own[name].shape != value.shape or own[name].dtype != value.dtype:
                raise ValueError("CTAA resource measurement restore tensor differs")
            own[name].copy_(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _observation_digest(observation: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            key: value
            for key, value in observation.items()
            if key != "observation_sha256"
        }
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    operation_factory: Callable[[], Callable[[], None]],
    *,
    arm: str,
    phase: str,
    depth: int,
    device: torch.device,
    batch_size: int,
    repeats: int,
    warmup_count: int,
    work_units_per_iteration: int,
    bindings: Mapping[str, str],
) -> dict[str, object]:
    if arm not in PROFILE_ARMS or phase not in PROFILE_PHASES:
        raise ValueError("CTAA resource observation arm/phase differs")
    if depth not in PROFILE_DEPTHS or batch_size < 1 or repeats < 1 or warmup_count < 1:
        raise ValueError("CTAA resource observation geometry differs")
    if work_units_per_iteration < 1:
        raise ValueError("CTAA resource observation work units differ")
    warmup_operation = operation_factory()
    for _ in range(warmup_count):
        warmup_operation()
    _synchronize(device)
    operation = operation_factory()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter_ns()
    for _ in range(repeats):
        operation()
    _synchronize(device)
    elapsed_ns = time.perf_counter_ns() - start
    if elapsed_ns <= 0:
        raise RuntimeError("CTAA resource observation clock differs")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    observation: dict[str, object] = {
        "schema": OBSERVATION_SCHEMA,
        "arm": arm,
        "phase": phase,
        "active_depth": depth,
        "device": str(device),
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup_count": warmup_count,
        "elapsed_ns": elapsed_ns,
        "milliseconds_per_iteration": elapsed_ns / repeats / 1_000_000.0,
        "rows_per_second": batch_size * repeats * 1_000_000_000.0 / elapsed_ns,
        "peak_allocated_bytes": peak,
        "work_units_per_iteration": work_units_per_iteration,
        "bindings": dict(bindings),
    }
    observation["observation_sha256"] = _observation_digest(observation)
    return observation


def _expected_binding_keys() -> set[str]:
    return {
        *SHARED_BINDING_KEYS,
        *CONTEXT_BINDING_KEYS,
        "core_checkpoint_sha256",
        "core_kind",
    }


def _validate_binding_map(
    bindings: object,
    *,
    arm: str,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    if not isinstance(bindings, dict) or set(bindings) != _expected_binding_keys():
        raise ValueError("CTAA resource observation binding schema differs")
    if bindings != expected_bindings[arm]:
        raise ValueError("CTAA resource observation artifact binding differs")
    if bindings.get("core_kind") != arm:
        raise ValueError("CTAA resource observation core kind differs")
    admission_device = bindings.get("admission_device")
    if not isinstance(admission_device, str) or not admission_device:
        raise ValueError("CTAA resource admission device differs")
    for key, value in bindings.items():
        if key.endswith("_sha256") and not _is_sha256(value):
            raise ValueError("CTAA resource observation digest differs")
    return bindings


def validate_measurement_matrix(
    observations: object,
    *,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> dict[tuple[str, str, int], dict[str, object]]:
    """Validate the exact 2 x 6 x 4 matrix and return it by identity."""
    if set(expected_bindings) != set(PROFILE_ARMS):
        raise ValueError("CTAA resource expected arm bindings differ")
    if not isinstance(observations, list):
        raise ValueError("CTAA resource observations schema differs")
    expected_identities = {
        (arm, phase, depth)
        for arm in PROFILE_ARMS
        for phase in PROFILE_PHASES
        for depth in PROFILE_DEPTHS
    }
    indexed: dict[tuple[str, str, int], dict[str, object]] = {}
    for value in observations:
        if not isinstance(value, dict) or set(value) != OBSERVATION_KEYS:
            raise ValueError("CTAA resource observation schema differs")
        arm = value.get("arm")
        phase = value.get("phase")
        depth = value.get("active_depth")
        if (
            not isinstance(arm, str)
            or not isinstance(phase, str)
            or not isinstance(depth, int)
        ):
            raise ValueError("CTAA resource observation identity differs")
        identity = (arm, phase, depth)
        if identity not in expected_identities or identity in indexed:
            raise ValueError("CTAA resource observation identity differs")
        if value.get("schema") != OBSERVATION_SCHEMA:
            raise ValueError("CTAA resource observation version differs")
        _validate_binding_map(
            value.get("bindings"), arm=arm, expected_bindings=expected_bindings
        )
        if value.get("observation_sha256") != _observation_digest(value):
            raise ValueError("CTAA resource observation receipt hash differs")
        for key in (
            "batch_size",
            "repeats",
            "warmup_count",
            "elapsed_ns",
            "work_units_per_iteration",
        ):
            item = value.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError("CTAA resource observation measurement differs")
        for key in ("milliseconds_per_iteration", "rows_per_second"):
            item = value.get(key)
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or item <= 0
            ):
                raise ValueError("CTAA resource observation timing differs")
        elapsed_ns = int(value["elapsed_ns"])
        repeats = int(value["repeats"])
        batch_size = int(value["batch_size"])
        expected_ms = elapsed_ns / repeats / 1_000_000.0
        expected_rows = batch_size * repeats * 1_000_000_000.0 / elapsed_ns
        if not math.isclose(
            float(value["milliseconds_per_iteration"]),
            expected_ms,
            rel_tol=1e-12,
            abs_tol=0.0,
        ) or not math.isclose(
            float(value["rows_per_second"]),
            expected_rows,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("CTAA resource observation derived timing differs")
        peak = value.get("peak_allocated_bytes")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
            raise ValueError("CTAA resource observation memory differs")
        device = value.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError("CTAA resource observation device differs")
        admission_device = value["bindings"]["admission_device"]
        if phase == "curriculum_selection":
            if device not in {"cpu", admission_device}:
                raise ValueError("CTAA curriculum selection device differs")
        elif device != admission_device:
            raise ValueError("CTAA resource observation admission device differs")
        if device.startswith("cuda") and peak <= 0:
            raise ValueError("CTAA CUDA resource observation memory is absent")
        indexed[identity] = value
    if set(indexed) != expected_identities:
        raise ValueError("CTAA resource measurement matrix is incomplete")
    return indexed


def build_matched_arm_comparisons(
    observations: object,
    *,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    indexed = validate_measurement_matrix(
        observations, expected_bindings=expected_bindings
    )
    comparisons = []
    for phase in PROFILE_PHASES:
        for depth in PROFILE_DEPTHS:
            treatment = indexed[(PROFILE_ARMS[0], phase, depth)]
            control = indexed[(PROFILE_ARMS[1], phase, depth)]
            treatment_peak = int(treatment["peak_allocated_bytes"])
            control_peak = int(control["peak_allocated_bytes"])
            comparisons.append(
                {
                    "schema": COMPARISON_SCHEMA,
                    "phase": phase,
                    "active_depth": depth,
                    "treatment_observation_sha256": treatment["observation_sha256"],
                    "control_observation_sha256": control["observation_sha256"],
                    "shared_bindings_exact": all(
                        treatment["bindings"][key] == control["bindings"][key]  # type: ignore[index]
                        for key in (*SHARED_BINDING_KEYS, *CONTEXT_BINDING_KEYS)
                    ),
                    "work_units_exact": treatment["work_units_per_iteration"]
                    == control["work_units_per_iteration"],
                    "batch_size_exact": treatment["batch_size"]
                    == control["batch_size"],
                    "repeats_exact": treatment["repeats"] == control["repeats"],
                    "warmup_count_exact": treatment["warmup_count"]
                    == control["warmup_count"],
                    "elapsed_ratio_control_over_treatment": float(control["elapsed_ns"])
                    / float(treatment["elapsed_ns"]),
                    "peak_bytes_ratio_control_over_treatment": (
                        float(control_peak) / treatment_peak
                        if treatment_peak
                        else (1.0 if control_peak == 0 else None)
                    ),
                }
            )
    return comparisons


def validate_matched_arm_comparisons(
    comparisons: object,
    *,
    observations: object,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> None:
    expected = build_matched_arm_comparisons(
        observations,
        expected_bindings=expected_bindings,
    )
    if not isinstance(comparisons, list) or len(comparisons) != len(expected):
        raise ValueError("CTAA matched-arm comparison matrix is incomplete")
    for observed, derived in zip(comparisons, expected, strict=True):
        if not isinstance(observed, dict) or set(observed) != COMPARISON_KEYS:
            raise ValueError("CTAA matched-arm comparison schema differs")
        if observed != derived:
            raise ValueError("CTAA matched-arm comparison differs")
        if not all(
            bool(observed[key])
            for key in (
                "shared_bindings_exact",
                "work_units_exact",
                "batch_size_exact",
                "repeats_exact",
                "warmup_count_exact",
            )
        ):
            raise ValueError("CTAA matched-arm resource workload differs")


def validate_resource_receipt(
    *,
    observations: object,
    comparisons: object,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> None:
    validate_measurement_matrix(observations, expected_bindings=expected_bindings)
    validate_matched_arm_comparisons(
        comparisons,
        observations=observations,
        expected_bindings=expected_bindings,
    )


def resource_gates_pass(
    *,
    observations: object,
    comparisons: object,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> bool:
    """Return true only after the complete measured receipt validates."""
    validate_resource_receipt(
        observations=observations,
        comparisons=comparisons,
        expected_bindings=expected_bindings,
    )
    return True


def _runtime_profile(
    core,
    device: torch.device,
    batch_size: int,
    repeats: int,
    depth: int,
) -> dict[str, object]:
    """Compatibility helper for the original inference-only profile."""
    if batch_size < 1 or repeats < 1:
        raise ValueError("CTAA runtime profile configuration differs")
    if depth not in PROFILE_DEPTHS:
        raise ValueError("CTAA runtime profile depth differs")
    cards, schedule, initial = _hard_workload(device, batch_size, depth)
    core = core.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            execute_streamed_dual(core, 3, cards, schedule, initial)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter_ns()
        for _ in range(repeats):
            execute_streamed_dual(core, 3, cards, schedule, initial)
        _synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start
    return {
        "device": str(device),
        "batch_size": batch_size,
        "repeats": repeats,
        "active_depth": depth,
        "milliseconds_per_batch": elapsed_ns / repeats / 1_000_000.0,
        "rows_per_second": batch_size * repeats * 1_000_000_000.0 / elapsed_ns,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def _hard_workload(
    device: torch.device,
    batch_size: int,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cards = torch.tensor(
        [[[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 0, 1]]],
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1, -1)
    initial = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device).expand(
        batch_size, -1
    )
    schedule = torch.full(
        (batch_size, CTAA_MAX_STEPS),
        3,
        dtype=torch.long,
        device=device,
    )
    schedule[:, :depth] = torch.arange(depth, device=device).remainder(
        CTAA_ACTION_COUNT
    )
    schedule[:, depth] = CTAA_ACTION_COUNT
    return cards, schedule, initial


def _core_training_operation(
    core: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    depth: int,
    phase: str,
) -> Callable[[], None]:
    left = (
        torch.arange(batch_size * 3, device=device).reshape(batch_size, 3).remainder(3)
    )
    right = left.roll(1, dims=1)
    optimizer = (
        torch.optim.AdamW(core.parameters(), lr=1e-4, weight_decay=0.0)
        if phase == "optimizer_step"
        else None
    )

    def operation() -> None:
        core.train()
        core.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        state = right
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            for index in range(depth):
                logits = core(left.roll(index % 3, dims=1), state)
                target = state.gather(1, left.roll(index % 3, dims=1))
                total = total + F.cross_entropy(
                    logits.reshape(-1, 3), target.reshape(-1)
                )
                state = target
        if phase in {"backward", "optimizer_step"}:
            total.backward()
        if optimizer is not None:
            optimizer.step()
        # Materialize a scalar so an eager backend cannot discard the graph.
        _ = float(total.detach())

    return operation


def _restored_core_factory(
    core: torch.nn.Module,
    baseline: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    depth: int,
    phase: str,
) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        _restore_state(core, baseline)
        return _core_training_operation(
            core,
            device=device,
            batch_size=batch_size,
            depth=depth,
            phase=phase,
        )

    return factory


def _inference_operation(
    core: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    depth: int,
) -> Callable[[], None]:
    cards, schedule, initial = _hard_workload(device, batch_size, depth)
    core.eval()

    def operation() -> None:
        with torch.inference_mode():
            execute_streamed_dual(core, 3, cards, schedule, initial)

    return operation


def _restored_inference_factory(
    core: torch.nn.Module,
    baseline: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    depth: int,
) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        _restore_state(core, baseline)
        return _inference_operation(
            core,
            device=device,
            batch_size=batch_size,
            depth=depth,
        )

    return factory


def _load_compiler_rows_by_depth(
    source_path: Path,
    tokenizer_path: Path,
    max_length: int,
) -> dict[int, list[TokenizedCompilerRow]]:
    require_sha256(tokenizer_path, TOKENIZER_SHA256, "tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = {depth: [] for depth in PROFILE_DEPTHS}
    with source_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = parse_train_row(json.loads(line), tokenizer, max_length)
            except Exception as error:
                raise ValueError(
                    f"CTAA resource compiler row {line_number} failed"
                ) from error
            stop = row.schedule.index(CTAA_ACTION_COUNT)
            if stop in result:
                result[stop].append(row)
    if any(not result[depth] for depth in PROFILE_DEPTHS):
        raise ValueError("CTAA resource compiler source omits a profiled depth")
    return result


def _canonical_jsonl_row_digests(path: Path) -> list[tuple[dict[str, object], str]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"CTAA curriculum source row {line_number} is malformed"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"CTAA curriculum source row {line_number} differs")
            rows.append((value, _canonical_sha256(value)))
    if not rows:
        raise ValueError("CTAA curriculum source is empty")
    return rows


def _load_curriculum_source_pools(
    *,
    compiler_train_path: Path,
    atomic_train_path: Path,
    closure_train_path: Path,
) -> dict[str, torch.Tensor]:
    """Load source-derived row identities used by deterministic selection."""

    def tensor_from_digests(digests: Sequence[str]) -> torch.Tensor:
        if not digests:
            raise ValueError("CTAA curriculum source pool is empty")
        # Sixty bits fit in signed int64 while retaining a source-derived identity.
        return torch.tensor(
            [int(digest[:15], 16) for digest in digests], dtype=torch.int64
        )

    compiler_by_depth = {depth: [] for depth in PROFILE_DEPTHS}
    for value, digest in _canonical_jsonl_row_digests(compiler_train_path):
        schedule = value.get("schedule")
        if not isinstance(schedule, list) or schedule.count(CTAA_ACTION_COUNT) != 1:
            raise ValueError("CTAA curriculum compiler schedule differs")
        stop = schedule.index(CTAA_ACTION_COUNT)
        if stop in compiler_by_depth:
            compiler_by_depth[stop].append(digest)
    if any(not compiler_by_depth[depth] for depth in PROFILE_DEPTHS):
        raise ValueError("CTAA curriculum compiler source omits a profiled depth")
    atomic = [
        digest for _value, digest in _canonical_jsonl_row_digests(atomic_train_path)
    ]
    closure = [
        digest for _value, digest in _canonical_jsonl_row_digests(closure_train_path)
    ]
    return {
        "atomic": tensor_from_digests(atomic),
        "closure": tensor_from_digests(closure),
        **{
            f"compiler_depth_{depth}": tensor_from_digests(compiler_by_depth[depth])
            for depth in PROFILE_DEPTHS
        },
    }


def _curriculum_selection_factory(
    pools: Mapping[str, torch.Tensor],
    *,
    depth: int,
    batch_size: int,
    repeats: int,
    warmup_count: int,
    source_hashes: Mapping[str, str],
) -> tuple[Callable[[], Callable[[], None]], str]:
    selected_pools = {
        "atomic": pools["atomic"],
        "closure": pools["closure"],
        "compiler": pools[f"compiler_depth_{depth}"],
    }
    pool_bindings = {
        name: {
            "rows": int(values.numel()),
            "identity_sha256": _canonical_sha256(values.tolist()),
        }
        for name, values in selected_pools.items()
    }
    plan = {
        "schema": "r12_ctaa_v2_curriculum_selection_plan_v1",
        "active_depth": depth,
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup_count": warmup_count,
        "sources": dict(source_hashes),
        "pools": pool_bindings,
    }
    plan_sha256 = _canonical_sha256(plan)
    seed = int(plan_sha256[:16], 16) % (2**63 - 1)

    def factory() -> Callable[[], None]:
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def operation() -> None:
            checksum = torch.zeros((), dtype=torch.int64)
            for values in selected_pools.values():
                indices = torch.randint(
                    values.numel(),
                    (batch_size,),
                    generator=generator,
                )
                checksum = torch.bitwise_xor(
                    checksum, values.index_select(0, indices).sum()
                )
            # Force the selected source-derived identities to be consumed.
            if checksum.numel() != 1:
                raise RuntimeError("CTAA curriculum selection checksum differs")

        return operation

    return factory, plan_sha256


def _compiler_training_operation(
    compiler: TrunkCausalCTAACompiler,
    rows: Sequence[TokenizedCompilerRow],
    *,
    device: torch.device,
    batch_size: int,
) -> Callable[[], None]:
    selected = [rows[index % len(rows)] for index in range(batch_size)]
    batch = collate_compiler_rows(selected, device=device)
    parameters = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)

    def operation() -> None:
        compiler.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            receipt = compiler_loss(compiler, batch)
        receipt.total.backward()
        optimizer.step()

    return operation


def _restored_compiler_factory(
    compiler: TrunkCausalCTAACompiler,
    baseline: Mapping[str, torch.Tensor],
    rows: Sequence[TokenizedCompilerRow],
    *,
    device: torch.device,
    batch_size: int,
) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        _restore_state(compiler, baseline)
        return _compiler_training_operation(
            compiler,
            rows,
            device=device,
            batch_size=batch_size,
        )

    return factory


def _load_bound_core(
    path: Path,
    *,
    expected_kind: str,
    atomic_sha256: str,
    closure_sha256: str,
) -> torch.nn.Module:
    core, kind = load_core(path)
    if kind != expected_kind:
        raise ValueError("CTAA resource core arm checkpoint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    training = payload.get("training") if isinstance(payload, dict) else None
    if not isinstance(training, dict):
        raise ValueError("CTAA resource core training receipt is absent")
    if training.get("atomic_sha256") != atomic_sha256:
        raise ValueError("CTAA resource atomic training source binding differs")
    if training.get("closure_sha256") != closure_sha256:
        raise ValueError("CTAA resource closure training source binding differs")
    return core


def profile(
    *,
    base_path: Path,
    qualified_path: Path,
    tokenizer_path: Path,
    compiler_train_path: Path,
    atomic_train_path: Path,
    closure_train_path: Path,
    treatment_core_path: Path,
    control_core_path: Path,
    output_path: Path,
    runtime_device: str | None,
    batch_size: int,
    repeats: int,
    warmup_count: int,
) -> dict[str, object]:
    if runtime_device is None:
        raise ValueError("CTAA admission resource profile requires measured runtime")
    if batch_size < 1 or repeats < 1 or warmup_count < 1:
        raise ValueError("CTAA resource profile configuration differs")
    device = torch.device(runtime_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA runtime profile requires available CUDA")

    source_hashes = {
        "tokenizer_sha256": require_sha256(
            tokenizer_path, TOKENIZER_SHA256, "tokenizer"
        ),
        "compiler_training_source_sha256": sha256_file(compiler_train_path),
        "atomic_training_source_sha256": sha256_file(atomic_train_path),
        "closure_training_source_sha256": sha256_file(closure_train_path),
    }
    trunk, base_receipt = load_raw_trunk(base_path)
    qualified = load_qualified_memory_state(qualified_path)
    compiler = TrunkCausalCTAACompiler(trunk)
    loaded = compiler.initialize_qualified_memory(qualified)
    qualified_sha256 = sha256_file(qualified_path)
    compiler_baseline = {
        name: value.detach().cpu().clone()
        for name, value in compiler.state_dict().items()
        if not name.startswith("model.")
    }
    compiler_initial_adapter_sha256 = _state_dict_sha256(compiler_baseline)
    compiler.to(device)
    rows_by_depth = _load_compiler_rows_by_depth(
        compiler_train_path,
        tokenizer_path,
        trunk.cfg.seq_len,
    )
    curriculum_pools = _load_curriculum_source_pools(
        compiler_train_path=compiler_train_path,
        atomic_train_path=atomic_train_path,
        closure_train_path=closure_train_path,
    )

    core_paths = {
        "closure_feature": treatment_core_path,
        "outer_product_control": control_core_path,
    }
    core_hashes = {arm: sha256_file(path) for arm, path in core_paths.items()}
    cores = {
        arm: _load_bound_core(
            core_paths[arm],
            expected_kind=arm,
            atomic_sha256=source_hashes["atomic_training_source_sha256"],
            closure_sha256=source_hashes["closure_training_source_sha256"],
        ).to(device)
        for arm in PROFILE_ARMS
    }
    core_baselines = {
        arm: {
            name: value.detach().cpu().clone()
            for name, value in cores[arm].state_dict().items()
        }
        for arm in PROFILE_ARMS
    }
    curriculum_factories: dict[tuple[str, int], Callable[[], Callable[[], None]]] = {}
    curriculum_depth_plans: dict[str, dict[int, str]] = {
        arm: {} for arm in PROFILE_ARMS
    }
    for arm in PROFILE_ARMS:
        for depth in PROFILE_DEPTHS:
            factory, plan_sha256 = _curriculum_selection_factory(
                curriculum_pools,
                depth=depth,
                batch_size=batch_size,
                repeats=repeats,
                warmup_count=warmup_count,
                source_hashes=source_hashes,
            )
            curriculum_factories[(arm, depth)] = factory
            curriculum_depth_plans[arm][depth] = plan_sha256
    expected_bindings = {
        arm: {
            "trunk_checkpoint_sha256": base_receipt.sha256,
            "qualified_compiler_checkpoint_sha256": qualified_sha256,
            "compiler_initial_adapter_sha256": compiler_initial_adapter_sha256,
            **source_hashes,
            "curriculum_selection_plan_sha256": _canonical_sha256(
                {
                    str(depth): curriculum_depth_plans[arm][depth]
                    for depth in PROFILE_DEPTHS
                }
            ),
            "admission_device": str(device),
            "core_checkpoint_sha256": core_hashes[arm],
            "core_kind": arm,
        }
        for arm in PROFILE_ARMS
    }

    treatment = cores["closure_feature"]
    control = cores["outer_product_control"]
    ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        treatment.unique_parameters,  # type: ignore[attr-defined]
    )
    charged_transition_flops = max(
        treatment.analytic_inference_flops,  # type: ignore[attr-defined]
        control.analytic_inference_flops,  # type: ignore[attr-defined]
    )
    observations: list[dict[str, object]] = []
    for arm in PROFILE_ARMS:
        core = cores[arm]
        for depth in PROFILE_DEPTHS:
            observations.append(
                _measure(
                    curriculum_factories[(arm, depth)],
                    arm=arm,
                    phase="curriculum_selection",
                    depth=depth,
                    device=torch.device("cpu"),
                    batch_size=batch_size,
                    repeats=repeats,
                    warmup_count=warmup_count,
                    work_units_per_iteration=batch_size * 3,
                    bindings=expected_bindings[arm],
                )
            )
            for phase in ("forward", "backward", "optimizer_step"):
                observations.append(
                    _measure(
                        _restored_core_factory(
                            core,
                            core_baselines[arm],
                            device=device,
                            batch_size=batch_size,
                            depth=depth,
                            phase=phase,
                        ),
                        arm=arm,
                        phase=phase,
                        depth=depth,
                        device=device,
                        batch_size=batch_size,
                        repeats=repeats,
                        warmup_count=warmup_count,
                        work_units_per_iteration=depth,
                        bindings=expected_bindings[arm],
                    )
                )
            observations.append(
                _measure(
                    _restored_compiler_factory(
                        compiler,
                        compiler_baseline,
                        rows_by_depth[depth],
                        device=device,
                        batch_size=batch_size,
                    ),
                    arm=arm,
                    phase="compiler_training",
                    depth=depth,
                    device=device,
                    batch_size=batch_size,
                    repeats=repeats,
                    warmup_count=warmup_count,
                    work_units_per_iteration=batch_size,
                    bindings=expected_bindings[arm],
                )
            )
            observations.append(
                _measure(
                    _restored_inference_factory(
                        core,
                        core_baselines[arm],
                        device=device,
                        batch_size=batch_size,
                        depth=depth,
                    ),
                    arm=arm,
                    phase="inference",
                    depth=depth,
                    device=device,
                    batch_size=batch_size,
                    repeats=repeats,
                    warmup_count=warmup_count,
                    work_units_per_iteration=DUAL_ROUTE_CALLS_PER_ROW,
                    bindings=expected_bindings[arm],
                )
            )
    comparisons = build_matched_arm_comparisons(
        observations,
        expected_bindings=expected_bindings,
    )
    resource_pass = resource_gates_pass(
        observations=observations,
        comparisons=comparisons,
        expected_bindings=expected_bindings,
    )

    runtime = {
        arm: {
            str(depth): next(
                observation
                for observation in observations
                if observation["arm"] == arm
                and observation["phase"] == "inference"
                and observation["active_depth"] == depth
            )
            for depth in PROFILE_DEPTHS
        }
        for arm in PROFILE_ARMS
    }
    static_pass = (
        treatment.unique_parameters  # type: ignore[attr-defined]
        == control.unique_parameters  # type: ignore[attr-defined]
        == 107_753
        and ledger["total"] < 150_000_000
        and len(loaded) == 63
    )
    report = {
        "schema": SCHEMA,
        "base_sha256": base_receipt.sha256,
        "base_step": base_receipt.step,
        "qualified_compiler_sha256": qualified_sha256,
        "qualified_memory_tensors": len(loaded),
        "artifact_bindings": expected_bindings,
        "parameter_ledger": ledger,
        "core_parameters": {
            "closure_feature": treatment.unique_parameters,  # type: ignore[attr-defined]
            "outer_product_control": control.unique_parameters,  # type: ignore[attr-defined]
            "exactly_matched": treatment.unique_parameters  # type: ignore[attr-defined]
            == control.unique_parameters,  # type: ignore[attr-defined]
        },
        "transition_flops": {
            "closure_feature_analytic": treatment.analytic_inference_flops,  # type: ignore[attr-defined]
            "outer_product_control_analytic": control.analytic_inference_flops,  # type: ignore[attr-defined]
            "charged_per_call": charged_transition_flops,
            "treatment_padding_charge": charged_transition_flops
            - treatment.analytic_inference_flops,  # type: ignore[attr-defined]
            "control_padding_charge": charged_transition_flops
            - control.analytic_inference_flops,  # type: ignore[attr-defined]
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
        "measurements": observations,
        "matched_arm_comparisons": comparisons,
        "required_phases": list(PROFILE_PHASES),
        "profile_depths": list(PROFILE_DEPTHS),
        "board_seed_generated": False,
        "oracle_access": 0,
        "all_static_gates_pass": static_pass,
        "all_resource_gates_pass": resource_pass,
        "all_gates_pass": static_pass and resource_pass,
    }
    report_sha = write_json_once(output_path, report)
    return {**report, "report_sha256": report_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compiler-train", type=Path, required=True)
    parser.add_argument("--atomic-train", type=Path, required=True)
    parser.add_argument("--closure-train", type=Path, required=True)
    parser.add_argument("--treatment-core", type=Path, required=True)
    parser.add_argument("--control-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-device", choices=("cpu", "cuda", "mps"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            profile(
                base_path=args.base,
                qualified_path=args.qualified_compiler,
                tokenizer_path=args.tokenizer,
                compiler_train_path=args.compiler_train,
                atomic_train_path=args.atomic_train,
                closure_train_path=args.closure_train,
                treatment_core_path=args.treatment_core,
                control_core_path=args.control_core,
                output_path=args.output,
                runtime_device=args.runtime_device,
                batch_size=args.batch_size,
                repeats=args.repeats,
                warmup_count=args.warmups,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
