#!/usr/bin/env python3
"""Freeze HSC canary and train-only qualification inputs.

This preparation role may generate all board splits in memory, but publishes
only train sources, train supervisor tensors, architecture receipts, and a
measurement-canary authorization.  It never emits a qualification-fit
authorization because measured H100 resources do not exist yet.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

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
)
from pipeline.episode_functor_hankel_train_package import (  # noqa: E402
    build_hankel_train_package,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    generate_pilot_rows,
    project_candidate_sources,
)
from pipeline.episode_functor_qualification_boundary import (  # noqa: E402
    collate_candidate_sources,
    tokenizer_runtime_sha256,
)
from pipeline.episode_functor_qualification_custody import (  # noqa: E402
    create_qualification_split_custody,
)
from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    collate_qualification_supervision,
)
from pipeline.episode_workspace_custody import (  # noqa: E402
    abort_atomic_bundle,
    atomic_bundle_directory,
    finish_atomic_bundle,
    fsync_directory,
    write_json_fsync,
)
from episode_functor_capacity_lanes import (  # noqa: E402
    build_hankel_shift_capacity_lane,
)
from episode_functor_hankel_arm import (  # noqa: E402
    create_hankel_arm_receipt,
)
from episode_functor_qualification_loss import (  # noqa: E402
    EFCHankelQualificationLoss,
)


PREPARATION_SCHEMA = "efc-hankel-qualification-preparation/v4"
BOARD_SEED = "efc-identifiable-pilot-20260724"
INITIALIZATION_SEED = 2_026_072_401
CONTROL_SEED = "efc-hsc-qualification-controls-v1"
BOARD_COUNTS = {
    "train": 96,
    "mechanics": 48,
    "development": 32,
    "confirmation": 24,
}
ARMS = (
    ("prefix", "hankel-shift"),
    ("random", "hankel-shift"),
    ("commutative", "hankel-shift"),
    ("prefix", "direct-base"),
)
class HankelPreparationError(ValueError):
    """Qualification preparation failed before atomic publication."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_source_paths() -> tuple[Path, ...]:
    """Trace only modules imported by the isolated worker entry point."""

    script = r"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path[:0] = [str(root), str(root / "train")]
import run_episode_functor_hankel_canary  # noqa: F401

paths = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if not source:
        continue
    try:
        path = Path(source).resolve(strict=True)
        relative = path.relative_to(root)
    except (OSError, ValueError):
        continue
    if relative.parts[0] in {"pipeline", "train"} and path.suffix == ".py":
        paths.add(str(relative))
print(json.dumps(sorted(paths), separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        relative_paths = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HankelPreparationError(
            "HSC runtime import trace is invalid"
        ) from exc
    if (
        not isinstance(relative_paths, list)
        or not relative_paths
        or any(not isinstance(path, str) for path in relative_paths)
    ):
        raise HankelPreparationError(
            "HSC runtime import trace is empty or malformed"
        )
    forbidden = {
        "pipeline/episode_functor_identifiable_board.py",
        "pipeline/episode_functor_qualification_supervisor.py",
        "pipeline/prepare_episode_functor_hankel_qualification.py",
    }
    if forbidden.intersection(relative_paths):
        raise HankelPreparationError(
            "HSC runtime source closure exposes an offline board or oracle"
        )
    entries = {
        "train/landlock_stage_exec.py",
        "train/run_episode_functor_hankel_canary.py",
    }
    if "train/run_episode_functor_hankel_canary.py" not in relative_paths:
        raise HankelPreparationError(
            "HSC runtime source closure omits its worker entry point"
        )
    return tuple(ROOT / path for path in sorted(set(relative_paths) | entries))


def _source_manifest() -> tuple[dict[str, str], ...]:
    result = []
    paths = _runtime_source_paths()
    if not paths:
        raise HankelPreparationError("HSC source closure is empty")
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if not path.is_file() or path.is_symlink():
            raise HankelPreparationError(
                f"HSC source path differs: {relative}"
            )
        result.append(
            {"path": relative, "sha256": _file_sha256(path)}
        )
    return tuple(result)


def _manifest_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _write_receipt(path: Path, receipt) -> None:
    write_json_fsync(path, receipt.to_mapping())


def prepare(output: Path, tokenizer_path: Path) -> dict[str, object]:
    tokenizer_path = tokenizer_path.resolve(strict=True)
    tokenizer_artifact_sha256 = _file_sha256(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_runtime = tokenizer_runtime_sha256(tokenizer)
    rows = generate_pilot_rows(
        seed=BOARD_SEED,
        counts=BOARD_COUNTS,
    )
    train_rows = tuple(row for row in rows if row.split == "train")
    if len(train_rows) != 384:
        raise HankelPreparationError(
            "HSC frozen train cardinality differs"
        )
    canary_rows = train_rows[:4]
    if (
        len({row.world_id for row in canary_rows}) != 1
        or len({row.source for row in canary_rows}) != 4
    ):
        raise HankelPreparationError(
            "HSC canary must contain four renderers of one train world"
        )

    def collate(selected):
        return collate_candidate_sources(
            project_candidate_sources(selected, split="train"),
            tokenizer=tokenizer,
            tokenizer_artifact_sha256=tokenizer_artifact_sha256,
            expected_tokenizer_runtime_sha256=tokenizer_runtime,
        )

    train_candidate = collate(train_rows)
    train_supervisor = collate_qualification_supervision(train_rows)
    train_custody = create_qualification_split_custody(
        train_rows,
        split="train",
        candidate=train_candidate,
    )
    canary_candidate = collate(canary_rows)
    canary_supervisor = collate_qualification_supervision(canary_rows)
    canary_custody = create_qualification_split_custody(
        canary_rows,
        split="train",
        candidate=canary_candidate,
    )
    train_targets = create_hankel_target_accounting(train_supervisor)
    canary_targets = create_hankel_target_accounting(canary_supervisor)
    optimizer = create_hankel_optimizer_contract()
    canary_schedule = create_hankel_schedule_contract(
        updates=2,
        microbatch_size=4,
        gradient_accumulation=1,
        order_seed=2_026_072_411,
        checkpoint_interval=2,
        metric_interval=1,
    )
    source_manifest = _source_manifest()
    source_manifest_sha256 = _manifest_sha256(source_manifest)

    staging, lock = atomic_bundle_directory(output)
    try:
        train_package = build_hankel_train_package(
            staging / "train_package",
            sources=tuple(row.source for row in train_rows),
            candidate=train_candidate,
            supervisor=train_supervisor,
            custody=train_custody,
        )
        canary_package = build_hankel_train_package(
            staging / "canary_package",
            sources=tuple(row.source for row in canary_rows),
            candidate=canary_candidate,
            supervisor=canary_supervisor,
            custody=canary_custody,
        )
        _write_receipt(staging / "optimizer.json", optimizer)
        _write_receipt(
            staging / "canary_schedule.json",
            canary_schedule,
        )
        _write_receipt(
            staging / "train_targets.json",
            train_targets,
        )
        _write_receipt(
            staging / "canary_targets.json",
            canary_targets,
        )
        write_json_fsync(
            staging / "source_manifest.json",
            {
                "manifest_sha256": source_manifest_sha256,
                "sources": list(source_manifest),
            },
        )

        arm_rows: list[dict[str, object]] = []
        reference_initialization = None
        objective = EFCHankelQualificationLoss()
        for incidence_mode, decode_mode in ARMS:
            torch.manual_seed(INITIALIZATION_SEED)
            compiler, query, _ = build_hankel_shift_capacity_lane(
                external_feature_width=1_728,
                incidence_mode=incidence_mode,
                random_seed=CONTROL_SEED,
                decode_mode=decode_mode,
            )
            arm = create_hankel_arm_receipt(
                compiler=compiler,
                query_parser=query,
                objective=objective,
            )
            initialization = create_hankel_initialization_receipt(
                compiler,
                query,
                seed=INITIALIZATION_SEED,
                arm_receipt=arm,
            )
            if reference_initialization is None:
                reference_initialization = initialization
            else:
                reference_initialization.assert_matched_trainable_initialization(
                    initialization
                )
            run_id = f"{arm.arm_name}-measurement-canary"
            experiment = create_hankel_experiment_receipt(
                phase="measurement-canary",
                run_id=run_id,
                arm_receipt=arm,
                initialization=initialization,
                optimizer=optimizer,
                schedule=canary_schedule,
                targets=canary_targets,
                train_custody=canary_custody,
                train_source_bytes=canary_package.source_bytes,
                tokenizer_sha256=tokenizer_artifact_sha256,
                runtime_source_manifest_sha256=(
                    source_manifest_sha256
                ),
            )
            arm_path = staging / f"{arm.arm_name}.arm.json"
            initialization_path = (
                staging / f"{arm.arm_name}.initialization.json"
            )
            experiment_path = (
                staging / f"{arm.arm_name}.canary.json"
            )
            _write_receipt(arm_path, arm)
            _write_receipt(initialization_path, initialization)
            _write_receipt(experiment_path, experiment)
            arm_rows.append(
                {
                    "arm": arm.arm_name,
                    "arm_receipt_sha256": arm.receipt_sha256,
                    "canary_receipt_sha256": (
                        experiment.receipt_sha256
                    ),
                    "compiler_state_sha256": (
                        initialization.compiler_state_sha256
                    ),
                    "initialization_receipt_sha256": (
                        initialization.receipt_sha256
                    ),
                    "trainable_state_sha256": (
                        initialization.trainable_state_sha256
                    ),
                }
            )
            del compiler, query, arm, initialization, experiment
            gc.collect()

        if reference_initialization is None:
            raise HankelPreparationError("HSC arm set is empty")
        published_files = tuple(
            sorted(
                str(path.relative_to(staging))
                for path in staging.rglob("*")
                if path.is_file()
            )
        )
        file_manifest = tuple(
            {
                "path": relative,
                "sha256": _file_sha256(staging / relative),
            }
            for relative in published_files
        )
        report = {
            "schema": PREPARATION_SCHEMA,
            "decision": (
                "measurement_canary_authorized_qualification_fit_no_go"
            ),
            "board": {
                "counts": BOARD_COUNTS,
                "seed": BOARD_SEED,
                "total_rows": len(rows),
                "train_rows": len(train_rows),
            },
            "canary": {
                "rows": len(canary_rows),
                "package_sha256": canary_package.package_sha256,
                "updates": canary_schedule.updates,
            },
            "train": {
                "independent_target_bits": (
                    train_targets.independent_target_bits
                ),
                "package_sha256": train_package.package_sha256,
                "rows": train_package.row_count,
                "source_bytes": train_package.source_bytes,
                "supplied_target_bits": (
                    train_targets.supplied_target_bits
                ),
            },
            "initialization_seed": INITIALIZATION_SEED,
            "shared_trainable_state_sha256": (
                reference_initialization.trainable_state_sha256
            ),
            "arms": arm_rows,
            "source_manifest_sha256": source_manifest_sha256,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "tokenizer_runtime_sha256": tokenizer_runtime,
            "fit_authorized": False,
            "pretraining_authorized": False,
            "files": list(file_manifest),
        }
        write_json_fsync(staging / "preparation_report.json", report)
        fsync_directory(staging)
        finish_atomic_bundle(staging, output, lock)
        return report
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/r12/"
            "episode_functor_hankel_qualification_v6_20260724"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare(arguments.output, arguments.tokenizer),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
