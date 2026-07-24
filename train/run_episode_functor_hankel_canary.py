#!/usr/bin/env python3
"""Isolated two-update HSC measurement canary.

This role discards trained weights.  It can measure memory, timing, finite
gradients, optimizer state, and receipt reconstruction, but cannot authorize a
qualification fit or make a reasoning claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_hankel_experiment import (  # noqa: E402
    create_hankel_experiment_receipt,
    create_hankel_initialization_receipt,
    create_hankel_optimizer_contract,
    create_hankel_schedule_contract,
    create_hankel_target_accounting,
    tensor_mapping_sha256,
)
from pipeline.episode_functor_hankel_train_package import (  # noqa: E402
    load_hankel_train_package,
)
from pipeline.episode_functor_runtime_custody import (  # noqa: E402
    abort_atomic_bundle,
    atomic_bundle_directory,
    finish_atomic_bundle,
    verify_landlock_stage,
    write_json_fsync,
)
from episode_functor_capacity_lanes import (  # noqa: E402
    build_hankel_shift_capacity_lane,
)
from episode_functor_hankel_arm import (  # noqa: E402
    create_hankel_arm_receipt,
)
from episode_functor_learned_system import LearnedEFCSystem  # noqa: E402
from episode_functor_qualification_loss import (  # noqa: E402
    EFCHankelQualificationLoss,
)
from episode_functor_qualification_trainer import (  # noqa: E402
    EFCQualificationTrainer,
    QualificationTrainerConfig,
)
from episode_functor_shohin_trunk import FrozenShohinTrunk  # noqa: E402


CANARY_REPORT_SCHEMA = "efc-hankel-measurement-canary/v1"
CANARY_COMPLETE_SCHEMA = "efc-hankel-measurement-complete/v1"
INITIALIZATION_SEED = 2_026_072_401
CONTROL_SEED = "efc-hsc-qualification-controls-v1"
ORDER_SEED = 2_026_072_411


class HankelCanaryError(ValueError):
    """The isolated HSC canary left its frozen authorization."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_complete_receipt(staging: Path) -> str:
    report_sha256 = _file_sha256(staging / "canary_report.json")
    payload = {
        "schema": CANARY_COMPLETE_SCHEMA,
        "files_sha256": [
            {
                "name": "canary_report.json",
                "sha256": report_sha256,
            }
        ],
    }
    payload["completion_sha256"] = sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    write_json_fsync(staging / "COMPLETE.json", payload)
    return report_sha256


def verify_complete_output(
    output: Path,
    *,
    expected_report_sha256: str | None = None,
) -> dict[str, object]:
    """Verify exact output closure and its prepublication completion receipt."""

    try:
        root = output.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HankelCanaryError("HSC canary output is absent") from exc
    entries = tuple(root.iterdir())
    if (
        not root.is_dir()
        or {path.name for path in entries}
        != {"canary_report.json", "COMPLETE.json"}
        or any(
            path.is_symlink() or not path.is_file()
            for path in entries
        )
    ):
        raise HankelCanaryError("HSC canary output closure differs")
    try:
        complete = json.loads(
            (root / "COMPLETE.json").read_text(encoding="ascii")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HankelCanaryError(
            "HSC canary completion receipt is invalid"
        ) from exc
    if (
        not isinstance(complete, dict)
        or set(complete)
        != {"schema", "files_sha256", "completion_sha256"}
        or complete["schema"] != CANARY_COMPLETE_SCHEMA
        or not isinstance(complete["files_sha256"], list)
        or complete["files_sha256"]
        != [
            {
                "name": "canary_report.json",
                "sha256": _file_sha256(root / "canary_report.json"),
            }
        ]
    ):
        raise HankelCanaryError("HSC canary completion receipt differs")
    receipt = dict(complete)
    observed_completion = receipt.pop("completion_sha256")
    if (
        not isinstance(observed_completion, str)
        or observed_completion
        != sha256(_canonical_json_bytes(receipt)).hexdigest()
    ):
        raise HankelCanaryError("HSC canary completion hash differs")
    report_sha256 = complete["files_sha256"][0]["sha256"]
    if (
        expected_report_sha256 is not None
        and report_sha256 != expected_report_sha256
    ):
        raise HankelCanaryError("HSC canary report hash differs")
    return complete


def _verify_source_snapshot(
    source_root: Path,
    manifest_path: Path,
    expected_sha256: str,
) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    with manifest_path.open("r", encoding="ascii") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"manifest_sha256", "sources"}
        or manifest["manifest_sha256"] != expected_sha256
        or not isinstance(manifest["sources"], list)
        or sha256(
            _canonical_json_bytes(tuple(manifest["sources"]))
        ).hexdigest()
        != expected_sha256
    ):
        raise HankelCanaryError("HSC source manifest differs")
    expected_paths: set[str] = set()
    for row in manifest["sources"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not isinstance(row["sha256"], str)
        ):
            raise HankelCanaryError("HSC source manifest row differs")
        path = source_root / row["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or _file_sha256(path) != row["sha256"]
        ):
            raise HankelCanaryError(
                f"HSC source snapshot differs: {row['path']}"
            )
        expected_paths.add(row["path"])
    observed_paths = {
        str(path.relative_to(source_root))
        for directory in (source_root / "pipeline", source_root / "train")
        for path in directory.glob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise HankelCanaryError(
            "HSC source snapshot closure differs"
        )
    return manifest


def _assert_network_namespace() -> dict[str, object]:
    if os.environ.get("SHOHIN_NETWORK_NAMESPACE_ISOLATED") != "1":
        raise HankelCanaryError(
            "HSC canary requires an isolated network namespace"
        )
    interfaces = tuple(name for _, name in socket.if_nameindex())
    if any(name != "lo" for name in interfaces):
        raise HankelCanaryError(
            "HSC canary network namespace exposes a non-loopback interface"
        )
    return {
        "environment_receipt": True,
        "interfaces": list(interfaces),
        "non_loopback_interfaces": 0,
    }


def _gpu_receipt() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise HankelCanaryError("HSC canary requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    if "H100" not in name.upper():
        raise HankelCanaryError(
            f"HSC canary requires an H100, observed {name}"
        )
    allocation = torch.empty((1024,), device="cuda")
    del allocation
    return {
        "device_count": torch.cuda.device_count(),
        "device_index": 0,
        "name": name,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    landlock = verify_landlock_stage(
        "efc-hsc-canary",
        args.deny_probe,
    )
    network = _assert_network_namespace()
    source_manifest = _verify_source_snapshot(
        args.source_root,
        args.source_manifest,
        args.expected_source_manifest_sha256,
    )
    if _file_sha256(args.tokenizer) != args.expected_tokenizer_sha256:
        raise HankelCanaryError("HSC tokenizer artifact differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    gpu = _gpu_receipt()

    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)

    candidate, supervisor, custody, package = (
        load_hankel_train_package(
            args.package,
            tokenizer=tokenizer,
            expected_package_sha256=args.expected_package_sha256,
            device="cuda",
        )
    )
    if package.row_count != 4:
        raise HankelCanaryError("HSC canary package row count differs")
    targets = create_hankel_target_accounting(supervisor)
    optimizer_contract = create_hankel_optimizer_contract()
    schedule = create_hankel_schedule_contract(
        updates=2,
        microbatch_size=4,
        gradient_accumulation=1,
        order_seed=ORDER_SEED,
        checkpoint_interval=2,
        metric_interval=1,
    )
    trunk = FrozenShohinTrunk.from_checkpoint(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        block_indices=(9, 19, 29),
        device="cpu",
    )
    # Checkpoint reconstruction allocates modules and advances CPU RNG. The
    # compiler initialization is a separately bound experiment object.
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    compiler, query, _ = build_hankel_shift_capacity_lane(
        external_feature_width=trunk.feature_width,
        incidence_mode=args.incidence_mode,
        random_seed=CONTROL_SEED,
        decode_mode=args.decode_mode,
    )
    objective = EFCHankelQualificationLoss()
    arm = create_hankel_arm_receipt(
        compiler=compiler,
        query_parser=query,
        objective=objective,
    )
    if arm.receipt_sha256 != args.expected_arm_receipt_sha256:
        raise HankelCanaryError("HSC arm receipt differs")
    initialization = create_hankel_initialization_receipt(
        compiler,
        query,
        seed=INITIALIZATION_SEED,
        arm_receipt=arm,
    )
    if (
        initialization.receipt_sha256
        != args.expected_initialization_receipt_sha256
    ):
        raise HankelCanaryError("HSC initialization receipt differs")
    experiment = create_hankel_experiment_receipt(
        phase="measurement-canary",
        run_id=f"{arm.arm_name}-measurement-canary",
        arm_receipt=arm,
        initialization=initialization,
        optimizer=optimizer_contract,
        schedule=schedule,
        targets=targets,
        train_custody=custody,
        train_source_bytes=package.source_bytes,
        tokenizer_sha256=args.expected_tokenizer_sha256,
        runtime_source_manifest_sha256=(
            args.expected_source_manifest_sha256
        ),
    )
    if experiment.receipt_sha256 != args.expected_canary_receipt_sha256:
        raise HankelCanaryError("HSC canary authorization differs")

    system = LearnedEFCSystem(
        source_compiler=compiler,
        query_parser=query,
        frozen_trunk=trunk,
    )
    system.to("cuda")
    initial_trainable_sha256 = tensor_mapping_sha256(
        dict(system.source_compiler.named_parameters())
    )
    if initial_trainable_sha256 != initialization.trainable_state_sha256:
        raise HankelCanaryError(
            "HSC CUDA trainable initialization differs"
        )
    trainer = EFCQualificationTrainer(
        system,
        objective=objective,
        config=QualificationTrainerConfig(
            learning_rate=float.fromhex(
                optimizer_contract.learning_rate_hex
            ),
            weight_decay=float.fromhex(
                optimizer_contract.weight_decay_hex
            ),
            maximum_gradient_norm=float.fromhex(
                optimizer_contract.maximum_gradient_norm_hex
            ),
            beta1=float.fromhex(optimizer_contract.beta1_hex),
            beta2=float.fromhex(optimizer_contract.beta2_hex),
            epsilon=float.fromhex(optimizer_contract.epsilon_hex),
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=True,
            maximum_updates=schedule.updates,
            autocast_dtype="bfloat16",
            tf32=True,
            deterministic_algorithms=False,
        ),
        training_custody=custody,
        require_verified_trunk=True,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    steps = [
        trainer.train_step(candidate, supervisor)
        for _ in range(schedule.updates)
    ]
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    trainer.seal_training()
    final_trainable_sha256 = tensor_mapping_sha256(
        dict(system.source_compiler.named_parameters())
    )
    if final_trainable_sha256 == initial_trainable_sha256:
        raise HankelCanaryError(
            "HSC canary completed without changing trainable state"
        )
    report = {
        "schema": CANARY_REPORT_SCHEMA,
        "decision": "measurement_only_qualification_fit_no_go",
        "claim_scope": (
            "two-update train-only H100 memory/timing/gradient canary; "
            "no capability, reasoning, fit, or pretraining claim"
        ),
        "process_id": os.getpid(),
        "landlock_receipt": landlock,
        "network_namespace": network,
        "gpu": gpu,
        "source_manifest_sha256": (
            args.expected_source_manifest_sha256
        ),
        "source_manifest_rows": len(source_manifest["sources"]),
        "package_sha256": package.package_sha256,
        "arm_receipt_sha256": arm.receipt_sha256,
        "initialization_receipt_sha256": initialization.receipt_sha256,
        "canary_receipt_sha256": experiment.receipt_sha256,
        "initial_trainable_state_sha256": initial_trainable_sha256,
        "final_trainable_state_sha256": final_trainable_sha256,
        "unique_rows": package.row_count,
        "cumulative_presentations": (
            package.row_count * schedule.updates
        ),
        "unique_source_bytes": package.source_bytes,
        "cumulative_source_bytes_presented": (
            package.source_bytes * schedule.updates
        ),
        "independent_target_bits_per_presentation": (
            targets.independent_target_bits
        ),
        "supplied_target_bits_per_presentation": (
            targets.supplied_target_bits
        ),
        "cumulative_supplied_target_bits": (
            targets.supplied_target_bits * schedule.updates
        ),
        "updates": schedule.updates,
        "elapsed_ns": elapsed_ns,
        "nanoseconds_per_update": elapsed_ns // schedule.updates,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "optimizer_state_bytes": steps[-1].optimizer_state_bytes,
        "steps": [asdict(step) for step in steps],
        "trainer_phase": trainer.phase,
        "weights_persisted": False,
        "optimizer_persisted": False,
        "development_visible": False,
        "confirmation_visible": False,
        "fit_authorized": False,
        "pretraining_authorized": False,
    }
    staging, lock = atomic_bundle_directory(args.output)
    try:
        write_json_fsync(staging / "canary_report.json", report)
        report_sha256 = _write_complete_receipt(staging)
        finish_atomic_bundle(staging, args.output, lock)
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise
    verify_complete_output(
        args.output,
        expected_report_sha256=report_sha256,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--incidence-mode", required=True)
    parser.add_argument("--decode-mode", required=True)
    parser.add_argument("--expected-arm-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-initialization-receipt-sha256",
        required=True,
    )
    parser.add_argument("--expected-canary-receipt-sha256", required=True)
    parser.add_argument("--deny-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
