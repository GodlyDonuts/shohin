#!/usr/bin/env python3
"""Build and verify custody artifacts for the ACW development matrix.

The committed development plan is the only source of executable argv, paths,
Slurm allocations, step identities, and attempt ordering. Producer and verifier
batch jobs call this module instead of reconstructing those values in shell.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pipeline.acw_hidden_basis_training import (
    DEVELOPMENT_PLAN_RAW_SHA256,
    canonical_json_bytes,
    file_sha256,
)
from pipeline.adjudicate_acw_hidden_basis import (
    DEVELOPMENT_MANIFEST_PROTOCOL,
    DEVELOPMENT_MANIFEST_SCHEMA,
    DEVELOPMENT_SEEDS,
    DIRECT_STATE_ARM,
    DIRECT_STATE_MANIFEST_PROTOCOL,
    DIRECT_STATE_MANIFEST_SCHEMA,
    SCORED_ARMS,
    _validate_frozen_development_baseline,
    _validate_stage_receipts,
    freeze_development_baseline,
    write_immutable_development_baseline,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BASE = Path("/lustre/fs1/home/sa305415/shohin_acw")
PYTHON = Path("/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13")
PILOT = BASE / "artifacts/r12/acw_cgbr_pilot_v6"
ATTEMPT_ROOT = BASE / "artifacts/r12/acw_development_g1"
DIRECT_PRIVATE_ROOT = BASE / "artifacts/r12/acw_development_g1_direct_verifier"
FINAL_PRIVATE_ROOT = BASE / "artifacts/r12/acw_development_g1_final_verifier"
TERMINAL_ACCOUNTING_PATH = (
    BASE / "artifacts/r12/acw_development_g1_terminal_accounting.json"
)
MONITOR_ANCHOR_PATH = BASE / "artifacts/r12/acw_development_g1_monitor_anchor.json"
PLAN_PATH = REPOSITORY / "R12_ACW_DEVELOPMENT_PLAN_V1.json"
PLAN_COPY_NAME = "development_plan.json"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")

ROLE_PHASE1 = "phase1_producer"
ROLE_DIRECT_VERIFIER = "phase1_verifier"
ROLE_PHASE2 = "phase2_producer"
ROLE_FINAL_VERIFIER = "phase2_verifier"
ROLES = (ROLE_PHASE1, ROLE_DIRECT_VERIFIER, ROLE_PHASE2, ROLE_FINAL_VERIFIER)
STAGES = ("phase1", "direct_verified", "phase2", "final")

ROLE_ROOTS = {
    ROLE_PHASE1: ATTEMPT_ROOT,
    ROLE_DIRECT_VERIFIER: DIRECT_PRIVATE_ROOT,
    ROLE_PHASE2: ATTEMPT_ROOT,
    ROLE_FINAL_VERIFIER: FINAL_PRIVATE_ROOT,
}
ROLE_JOB_NAMES = {
    ROLE_PHASE1: "shohin-acw-dev-p1",
    ROLE_DIRECT_VERIFIER: "shohin-acw-dev-v1",
    ROLE_PHASE2: "shohin-acw-dev-p2",
    ROLE_FINAL_VERIFIER: "shohin-acw-dev-v2",
}
ROLE_NODES = {
    ROLE_PHASE1: "ec51",
    ROLE_DIRECT_VERIFIER: "ec52",
    ROLE_PHASE2: "ec51",
    ROLE_FINAL_VERIFIER: "ec52",
}
ROLE_SCRIPTS = {
    ROLE_PHASE1: "pipeline/jobs/run_acw_development_stokes.sbatch",
    ROLE_DIRECT_VERIFIER: "pipeline/jobs/run_acw_development_stokes.sbatch",
    ROLE_PHASE2: "pipeline/jobs/run_acw_development_stokes.sbatch",
    ROLE_FINAL_VERIFIER: "pipeline/jobs/run_acw_development_stokes.sbatch",
}
MONITOR_JOB_NAME = "shohin-acw-dev-monitor"
MONITOR_NODE = "ec51"
MONITOR_SCRIPT = "pipeline/jobs/run_acw_terminal_monitor_stokes.sbatch"
ROLE_START_FILES = {role: f"custody/stages/{role}_start.json" for role in ROLES}
ROLE_COMPLETION_FILES = {
    role: f"custody/stages/{role}_completion.json" for role in ROLES
}
ROLE_ACCOUNTING_FILES = {
    role: f"custody/stages/{role}_accounting.json" for role in ROLES
}
STAGE_ROLES = {
    "phase1": ROLE_PHASE1,
    "direct_verified": ROLE_DIRECT_VERIFIER,
    "phase2": ROLE_PHASE2,
    "final": ROLE_FINAL_VERIFIER,
}
TERMINAL_OUTPUT_NAMES = (
    "development_baseline.json",
    "best_development_checkpoint.pt",
)


def _hash_bound(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be an object")
    if raw != canonical_json_bytes(parsed) + b"\n":
        raise ValueError(f"{label} is not canonical newline-framed JSON")
    recorded = parsed.get("payload_sha256")
    payload = dict(parsed)
    payload.pop("payload_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if recorded != observed:
        raise ValueError(f"{label} payload hash mismatch")
    return parsed, raw


def _reference(path: Path, base: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
        raise ValueError(f"required immutable file is invalid: {path}")
    resolved = path.resolve(strict=True)
    try:
        rendered = resolved.relative_to(base.resolve(strict=True)).as_posix()
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": file_sha256(path)}


def _rooted_reference(root: Path, base: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink() or root.stat().st_mode & 0o222:
        raise ValueError(f"required immutable root is invalid: {root}")
    resolved = root.resolve(strict=True)
    try:
        rendered = resolved.relative_to(base.resolve(strict=True)).as_posix()
    except ValueError:
        rendered = str(resolved)
    return {
        "root": rendered,
        "manifest": _reference(root / "manifest.json", base),
    }


def _plan_reference(root: Path) -> dict[str, str]:
    plan = root / PLAN_COPY_NAME
    reference = _reference(plan, root)
    if reference["sha256"] != DEVELOPMENT_PLAN_RAW_SHA256:
        raise ValueError("development plan copy differs from committed bytes")
    _load_canonical_json(plan, "development plan copy")
    return reference


def _plan_for_root(root: Path) -> dict[str, Any]:
    plan_path = root / PLAN_COPY_NAME
    plan, _ = _load_canonical_json(plan_path, "development plan copy")
    if file_sha256(plan_path) != DEVELOPMENT_PLAN_RAW_SHA256:
        raise ValueError("development plan copy differs from committed bytes")
    return plan


def _stage_by_role(plan: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        stage
        for stage in plan.get("custody_stages", [])
        if isinstance(stage, dict) and stage.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"development plan does not uniquely bind role {role}")
    return matches[0]


def _monitor_stage(plan: dict[str, Any]) -> dict[str, Any]:
    accounting = plan.get("accounting")
    if not isinstance(accounting, dict):
        raise ValueError("development plan accounting is missing")
    monitor = accounting.get("monitor_stage")
    if not isinstance(monitor, dict):
        raise ValueError("development plan monitor stage is missing")
    return monitor


def _resolved(path: str, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts or "\x00" in path:
        raise ValueError(f"{label} is not a safe absolute path")
    return candidate


def _python_argv(script: str, *arguments: str) -> list[str]:
    return [str(PYTHON), "-S", "-P", str(BASE / script), *arguments]


def _input_paths(role: str, index: int) -> dict[str, Path]:
    root = ROLE_ROOTS[role]
    return {
        "dataset": root / "inputs/datasets" / f"development_{index}",
        "cgb": root / "inputs/bundles" / f"development_{index}_cgb",
        "uniform": root / "inputs/bundles" / f"development_{index}_uniform",
    }


def expected_input_table(plan: dict[str, Any]) -> list[dict[str, Any]]:
    del plan
    records: list[dict[str, Any]] = []
    for role in (ROLE_PHASE1, ROLE_DIRECT_VERIFIER, ROLE_FINAL_VERIFIER):
        for index, seed in enumerate(DEVELOPMENT_SEEDS):
            paths = _input_paths(role, index)
            records.append(
                {
                    "role": role,
                    "index": index,
                    "seed": seed,
                    "worker": {
                        "wave": 0,
                        "slot": index,
                        "max_parallel_workers": 3,
                        "threads": 1,
                    },
                    "paths": {key: str(value) for key, value in paths.items()},
                    "generator_argv": _python_argv(
                        "pipeline/generate_acw_hidden_basis.py",
                        "--development-seed",
                        str(seed),
                        "--out",
                        str(paths["dataset"]),
                    ),
                    "cgb_bundle_argv": _python_argv(
                        "pipeline/freeze_acw_curriculum.py",
                        "bundle",
                        "--dataset",
                        str(paths["dataset"]),
                        "--schedule",
                        str(PILOT / "cgb_schedule.jsonl"),
                        "--pilot-report",
                        str(PILOT / "report.json"),
                        "--out",
                        str(paths["cgb"]),
                    ),
                    "uniform_bundle_argv": _python_argv(
                        "pipeline/freeze_acw_curriculum.py",
                        "bundle",
                        "--dataset",
                        str(paths["dataset"]),
                        "--schedule",
                        str(PILOT / "uniform_schedule.jsonl"),
                        "--pilot-report",
                        str(PILOT / "report.json"),
                        "--out",
                        str(paths["uniform"]),
                    ),
                }
            )
    return records


def _attempt_side(
    *,
    role: str,
    index: int,
    seed: int,
    logical_arm: str,
    trainer_arm: str,
    family: str,
    wave: int,
    task_root: Path,
) -> dict[str, Any]:
    inputs = _input_paths(role, index)
    bundle = inputs[family]
    task = task_root / f"{index:02d}_{logical_arm}"
    checkpoint = task / "checkpoint.pt"
    evaluation = task / "evaluation.json"
    replay = task / "replay.json"
    training = _python_argv(
        "pipeline/acw_hidden_basis_training.py",
        "--bundle",
        str(bundle),
        "--curriculum",
        str(bundle / "curriculum.jsonl"),
        "--arm",
        trainer_arm,
        "--seed",
        str(seed),
        "--attempt-id",
        f"{logical_arm}__{seed}",
        "--out",
        str(checkpoint),
    )
    if logical_arm == DIRECT_STATE_ARM:
        training.extend(("--oracle-dataset", str(inputs["dataset"])))
    return {
        "job_role": role,
        "worker": {
            "wave": wave,
            "slot": index,
            "max_parallel_workers": 3,
            "threads": 1,
        },
        "paths": {
            "dataset": str(inputs["dataset"]),
            "bundle": str(bundle),
            "curriculum": str(bundle / "curriculum.jsonl"),
            "task_root": str(task),
            "checkpoint": str(checkpoint),
            "evaluation": str(evaluation),
            "replay": str(replay),
        },
        "train_argv": training,
        "evaluation_argv": _python_argv(
            "pipeline/evaluate_acw_hidden_basis.py",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(inputs["dataset"]),
            "--out",
            str(evaluation),
        ),
        "replay_argv": _python_argv(
            "pipeline/evaluate_acw_hidden_basis.py",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(inputs["dataset"]),
            "--out",
            str(replay),
        ),
    }


def expected_attempt_table(plan: dict[str, Any]) -> list[dict[str, Any]]:
    stages = plan.get("custody_stages")
    if not isinstance(stages, list):
        raise ValueError("development plan custody stages are missing")
    jobs = {
        stage.get("role"): stage.get("held_slurm_job_id")
        for stage in stages
        if isinstance(stage, dict)
    }
    records: list[dict[str, Any]] = []
    for ordinal, (logical_arm, index) in enumerate(
        (arm, index) for arm in (DIRECT_STATE_ARM, *SCORED_ARMS) for index in range(3)
    ):
        seed = DEVELOPMENT_SEEDS[index]
        family = "uniform" if logical_arm == "uniform_query_acw" else "cgb"
        trainer_arm = "acw" if logical_arm == "uniform_query_acw" else logical_arm
        if logical_arm == DIRECT_STATE_ARM:
            producer_role = ROLE_PHASE1
            verifier_role = ROLE_DIRECT_VERIFIER
            producer_wave = 0
            verifier_wave = 0
        else:
            arm_index = SCORED_ARMS.index(logical_arm)
            producer_role = ROLE_PHASE2
            verifier_role = ROLE_FINAL_VERIFIER
            producer_wave = arm_index
            verifier_wave = arm_index
        producer = _attempt_side(
            role=producer_role,
            index=index,
            seed=seed,
            logical_arm=logical_arm,
            trainer_arm=trainer_arm,
            family=family,
            wave=producer_wave,
            task_root=ATTEMPT_ROOT / "runs",
        )
        verifier = _attempt_side(
            role=verifier_role,
            index=index,
            seed=seed,
            logical_arm=logical_arm,
            trainer_arm=trainer_arm,
            family=family,
            wave=verifier_wave,
            task_root=ROLE_ROOTS[verifier_role] / "runs",
        )
        for side in (producer, verifier):
            role = side["job_role"]
            side["job_id"] = jobs.get(role)
            side["node"] = ROLE_NODES[role]
        records.append(
            {
                "ordinal": ordinal,
                "attempt_id": f"{logical_arm}__{seed}",
                "index": index,
                "seed": seed,
                "logical_arm": logical_arm,
                "trainer_arm": trainer_arm,
                "schedule_family": family,
                "phase": "phase1" if logical_arm == DIRECT_STATE_ARM else "phase2",
                "producer": producer,
                "verifier": verifier,
            }
        )
    return records


def _expected_stage_work(role: str) -> list[dict[str, Any]]:
    if role == ROLE_PHASE1:
        values = (("generate_inputs", 0), (DIRECT_STATE_ARM, 0))
    elif role == ROLE_DIRECT_VERIFIER:
        values = (
            ("regenerate_inputs", 0),
            (DIRECT_STATE_ARM, 0),
            ("verify_and_qualify", 1),
        )
    elif role == ROLE_PHASE2:
        values = tuple((arm, index) for index, arm in enumerate(SCORED_ARMS))
    else:
        values = (
            (("regenerate_inputs", 0),)
            + tuple((arm, index) for index, arm in enumerate(SCORED_ARMS))
            + (("verify_and_seal_evidence", len(SCORED_ARMS)),)
        )
    return [
        {
            "operation": operation,
            "wave": wave,
            "max_parallel_workers": 1 if operation.startswith("verify_and_") else 3,
            "worker_threads": 1,
        }
        for operation, wave in values
    ]


def validate_plan(plan: dict[str, Any], *, require_ready: bool) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "protocol",
        "execution_parent_commit",
        "artifact_root",
        "private_roots",
        "attempt_registry",
        "ready_for_g_commit",
        "custody_stages",
        "input_table",
        "attempt_table",
        "claim_records",
        "closed_world",
        "accounting",
        "confirmation",
        "claim_boundary",
        "development_seeds",
        "optimizer",
        "direct_state",
        "scored_phase",
        "retry_policy",
        "runtime",
        "stored_evaluations_per_checkpoint",
        "payload_sha256",
    }
    if set(plan) != expected_keys:
        raise ValueError("development plan has the wrong exact schema")
    payload = dict(plan)
    recorded_payload_sha256 = payload.pop("payload_sha256", None)
    if (
        recorded_payload_sha256
        != hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    ):
        raise ValueError("development plan payload hash differs")
    if (
        plan["schema"] != "r12_acw_development_plan_v1"
        or plan["protocol"] != "R12-ACW-DEVELOPMENT-PLAN-v1"
        or plan["execution_parent_commit"] != "7433062211c4ad0371a975019c37625f7d811b27"
        or plan["artifact_root"] != str(ATTEMPT_ROOT.relative_to(BASE))
        or plan["development_seeds"] != list(DEVELOPMENT_SEEDS)
        or plan["stored_evaluations_per_checkpoint"] != 2
        or not isinstance(plan["ready_for_g_commit"], bool)
        or plan["confirmation"].get("authorized") is not False
    ):
        raise ValueError("development plan fixed contract differs")
    private_roots = plan["private_roots"]
    if private_roots != {
        ROLE_DIRECT_VERIFIER: str(DIRECT_PRIVATE_ROOT),
        ROLE_FINAL_VERIFIER: str(FINAL_PRIVATE_ROOT),
    }:
        raise ValueError("development plan private roots differ")
    stages = plan["custody_stages"]
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("development plan must bind exactly four custody stages")
    job_ids: list[str] = []
    prior: str | None = None
    for ordinal, role in enumerate(ROLES):
        record = stages[ordinal]
        expected_stage_keys = {
            "ordinal",
            "role",
            "held_slurm_job_id",
            "job_name",
            "expected_node",
            "script",
            "dependency",
            "submitted_held_before_g_commit",
            "release_only_after_g_is_pushed",
            "allocation",
            "work",
        }
        if not isinstance(record, dict) or set(record) != expected_stage_keys:
            raise ValueError(f"development plan job schema differs: {role}")
        script = record["script"]
        script_path = REPOSITORY / ROLE_SCRIPTS[role]
        expected_dependency = (
            None if prior is None else {"type": "afterok", "held_slurm_job_id": prior}
        )
        if (
            record["ordinal"] != ordinal
            or record["role"] != role
            or record["job_name"] != ROLE_JOB_NAMES[role]
            or record["expected_node"] != ROLE_NODES[role]
            or record["dependency"] != expected_dependency
            or record["submitted_held_before_g_commit"] is not True
            or record["release_only_after_g_is_pushed"] is not True
            or record["allocation"]
            != {
                "partition": "normal",
                "constraint": "skylake",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 4,
                "memory_gib": 96,
                "batch_cgroup_suffix": "step_batch/task_0",
                "normal_slurm_steps_forbidden": True,
                "max_parallel_workers": 3,
                "worker_threads": 1,
            }
            or record["work"] != _expected_stage_work(role)
            or not isinstance(script, dict)
            or set(script) != {"path", "sha256"}
            or script["path"] != ROLE_SCRIPTS[role]
            or not script_path.is_file()
            or script_path.is_symlink()
            or script["sha256"] != file_sha256(script_path)
        ):
            raise ValueError(f"development plan job contract differs: {role}")
        job_id = str(record["held_slurm_job_id"])
        if require_ready and JOB_ID_RE.fullmatch(job_id) is None:
            raise ValueError(f"development plan held job ID is unresolved: {role}")
        job_ids.append(job_id)
        prior = job_id
    if require_ready and len(set(job_ids)) != len(job_ids):
        raise ValueError("development plan held job IDs must be distinct")
    if require_ready and plan["ready_for_g_commit"] is not True:
        raise ValueError("development plan is not ready for G commit")
    attempt_registry = plan["attempt_registry"]
    if attempt_registry != {
        "held_slurm_job_id": stages[0]["held_slurm_job_id"],
        "job_name": stages[0]["job_name"],
        "must_be_allocated_held_before_g_commit": True,
        "release_only_after_g_is_pushed": True,
        "attempt_ids": [
            f"{arm}__{seed}"
            for arm in (DIRECT_STATE_ARM, *SCORED_ARMS)
            for seed in DEVELOPMENT_SEEDS
        ],
    }:
        raise ValueError("development plan compatibility attempt registry differs")
    if plan["input_table"] != expected_input_table(plan):
        raise ValueError("development plan input table differs")
    attempts = plan["attempt_table"]
    if attempts != expected_attempt_table(plan) or len(attempts) != 27:
        raise ValueError("development plan exact 27-entry attempt table differs")
    if len({record["attempt_id"] for record in attempts}) != 27:
        raise ValueError("development plan attempt IDs are not unique")
    monitor = _monitor_stage(plan)
    monitor_script = REPOSITORY / MONITOR_SCRIPT
    expected_monitor_dependency = {
        "type": "afterok",
        "held_slurm_job_id": stages[-1]["held_slurm_job_id"],
    }
    if monitor != {
        "held_slurm_job_id": monitor.get("held_slurm_job_id"),
        "job_name": MONITOR_JOB_NAME,
        "expected_node": MONITOR_NODE,
        "script": {"path": MONITOR_SCRIPT, "sha256": file_sha256(monitor_script)},
        "dependency": expected_monitor_dependency,
        "submitted_held_before_g_commit": True,
        "release_only_after_g_is_pushed": True,
        "allocation": {
            "partition": "normal",
            "constraint": "skylake",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 4,
            "memory_gib": 96,
            "batch_cgroup_suffix": "step_batch/task_0",
            "normal_slurm_steps_forbidden": True,
        },
        "runtime_identity_sha256": (
            "0e91de0e3dbca24ea4f04b9b03398a91486b93b31eff5a3ba4574dd43eaa677f"
        ),
    }:
        raise ValueError("development plan monitor stage differs")
    monitor_job_id = str(monitor["held_slurm_job_id"])
    if require_ready and (
        JOB_ID_RE.fullmatch(monitor_job_id) is None or monitor_job_id in job_ids
    ):
        raise ValueError("development plan monitor job ID is unresolved or reused")
    expected_accounting = {
        "terminal_receipt": str(
            BASE / "artifacts/r12/acw_development_g1_terminal_accounting.json"
        ),
        "required_before_any_performance_claim": True,
        "external_sha256_anchor_required_before_performance_claim": True,
        "performance_claim_ready_at_g_launch": False,
        "batch_step_required": True,
        "normal_steps_forbidden": True,
        "poll_timeout_seconds": 180,
        "monitor_terminal_accounting_required_before_external_anchor": True,
        "monitor_anchor_ready_envelope": str(MONITOR_ANCHOR_PATH),
        "monitor_must_be_completed_before_anchor_ready_envelope": True,
        "monitor_stage": monitor,
    }
    if plan["accounting"] != expected_accounting:
        raise ValueError("development plan accounting contract differs")
    if plan["claim_records"] != {
        "attempt_claim": "attempt_claim.json",
        "attempt_start": "attempt_start.json",
        "direct_producer_manifest": "direct_state_producer_manifest.json",
        "direct_manifest": "direct_state_manifest.json",
        "phase2_authorization": "phase2_authorization.json",
        "development_producer_manifest": "development_producer_manifest.json",
        "development_manifest": "development_manifest.json",
        "development_baseline": "development_baseline.json",
        "baseline_checkpoint": "best_development_checkpoint.pt",
    }:
        raise ValueError("development plan claim-record contract differs")
    if plan["closed_world"] != {
        "attempt_count": 27,
        "attempt_files": [
            "attempt.json",
            "checkpoint.pt",
            "evaluation.json",
            "replay.json",
        ],
        "exact_registered_root_required": True,
        "no_additional_slurm_steps_required": True,
        "same_uid_external_compute_not_excluded": True,
        "ordinary_batch_children_not_independently_attested": True,
        "claim_limited_to_exact_final_rooted_files": True,
        "symlinks_forbidden": True,
        "special_files_forbidden": True,
    }:
        raise ValueError("development plan closed-world contract differs")
    fixed_contract = {
        "claim_boundary": (
            "One fixed public development falsifier for synthetic state transport. "
            "Custody covers exact final rooted files and exact Slurm accounting rows; "
            "it does not exclude same-UID external compute or independently attest "
            "ordinary child processes inside a batch allocation. It is not a language, "
            "autonomous reasoning, generalization, or confirmation claim."
        ),
        "confirmation": {
            "authorized": False,
            "reason": (
                "Confirmation remains closed until a future NIST Beacon pulse is "
                "commit-bound and independently verified."
            ),
        },
        "optimizer": {
            "batch_size": 256,
            "final_scalar_labels": 57344,
            "optimizer_updates": 3400,
        },
        "direct_state": {
            "all_seeds_must_pass": True,
            "arm": DIRECT_STATE_ARM,
            "depth": 8,
            "scalar_accuracy_floor": 0.99,
            "state_exactness_floor": 0.95,
        },
        "scored_phase": {
            "arms": list(SCORED_ARMS),
            "requires_direct_state_authorization": True,
        },
        "retry_policy": {
            "checkpoint_publication_is_terminal": True,
            "in_place_resume": False,
            "one_attempt": True,
            "overwrite": False,
            "retry_after_checkpoint": False,
        },
        "runtime": {
            "cpu_model": "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz",
            "runtime_identity_sha256": (
                "0e91de0e3dbca24ea4f04b9b03398a91486b93b31eff5a3ba4574dd43eaa677f"
            ),
            "threads_per_fit": 1,
            "fits_in_parallel": 3,
            "normal_slurm_steps_forbidden": True,
        },
    }
    for key, expected in fixed_contract.items():
        if plan[key] != expected:
            raise ValueError(f"development plan fixed field differs: {key}")
    return plan


def load_committed_plan(*, require_ready: bool) -> dict[str, Any]:
    plan, raw = _load_canonical_json(PLAN_PATH, "committed development plan")
    if hashlib.sha256(raw).hexdigest() != DEVELOPMENT_PLAN_RAW_SHA256:
        raise ValueError("committed development plan raw hash differs from G pin")
    return validate_plan(plan, require_ready=require_ready)


def _attempt_id(index: int, arm: str) -> str:
    return f"{arm}__{DEVELOPMENT_SEEDS[index]}"


def _validate_attempt_inventory(root: Path, arms: tuple[str, ...]) -> None:
    runs = root / "runs"
    if not runs.is_dir() or runs.is_symlink():
        raise ValueError("attempt inventory root is unavailable")
    expected_directories = {
        runs / f"{index:02d}_{arm}" for arm in arms for index in range(3)
    }
    actual_directories: set[Path] = set()
    actual_files: set[Path] = set()
    for path in runs.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"attempt inventory contains a symlink: {path}")
        if path.is_dir():
            actual_directories.add(path)
        elif path.is_file():
            actual_files.add(path)
        else:
            raise ValueError(f"attempt inventory contains a special entry: {path}")
    expected_files = {
        task / name
        for task in expected_directories
        for name in ("attempt.json", "checkpoint.pt", "evaluation.json", "replay.json")
    }
    if actual_directories != expected_directories or actual_files != expected_files:
        raise ValueError("attempt inventory differs from the exact claimed matrix")

    for task in expected_directories:
        receipt, _ = _load_canonical_json(task / "attempt.json", "attempt receipt")
        relative = task.relative_to(root).as_posix()
        expected_id = _attempt_id(
            int(task.name.split("_", 1)[0]), task.name.split("_", 1)[1]
        )
        if (
            receipt.get("schema") != "r12_acw_development_attempt_receipt_v1"
            or receipt.get("protocol") != "R12-ACW-DEVELOPMENT-ATTEMPT-v1"
            or receipt.get("attempt_id") != expected_id
            or receipt.get("artifact_root") != str(root.resolve(strict=True))
            or receipt.get("task_root") != relative
            or receipt.get("completed_once") is not True
        ):
            raise ValueError(f"attempt receipt differs or is off-root: {task}")


def _run_record(
    root: Path,
    index: int,
    arm: str,
) -> dict[str, object]:
    dataset = root / "inputs" / "datasets" / f"development_{index}"
    family = "uniform" if arm == "uniform_query_acw" else "cgb"
    bundle = root / "inputs" / "bundles" / f"development_{index}_{family}"
    task = root / "runs" / f"{index:02d}_{arm}"
    return {
        "attempt_id": _attempt_id(index, arm),
        "arm": arm,
        "attempt_receipt": _reference(task / "attempt.json", root),
        "checkpoint": _reference(task / "checkpoint.pt", root),
        "dataset": _rooted_reference(dataset, root),
        "trainer_bundle": _rooted_reference(bundle, root),
        "evaluation_report": _reference(task / "evaluation.json", root),
        "replay_report": _reference(task / "replay.json", root),
    }


def _stage_receipts(
    root: Path,
    roles: tuple[str, ...],
    *,
    open_role: str | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for role in roles:
        record = {"start": _reference(root / ROLE_START_FILES[role], root)}
        if role != open_role:
            record["completion"] = _reference(root / ROLE_COMPLETION_FILES[role], root)
            record["terminal_accounting"] = _reference(
                root / ROLE_ACCOUNTING_FILES[role], root
            )
        records[role] = record
    return records


def build_direct_producer_manifest(root: Path) -> dict[str, object]:
    _validate_attempt_inventory(root, (DIRECT_STATE_ARM,))
    return _hash_bound(
        {
            "schema": "r12_acw_direct_state_producer_manifest_v1",
            "protocol": "R12-ACW-DIRECT-STATE-PRODUCER-MANIFEST-v1",
            "development_plan": _plan_reference(root),
            "attempt_claim": _reference(root / "attempt_claim.json", root),
            "attempt_start": _reference(root / "attempt_start.json", root),
            "stage_receipts": _stage_receipts(
                root, (ROLE_PHASE1,), open_role=ROLE_PHASE1
            ),
            "reports": [
                _run_record(
                    root,
                    index,
                    DIRECT_STATE_ARM,
                )
                for index in range(3)
            ],
        }
    )


def build_direct_manifest(root: Path) -> dict[str, object]:
    _validate_attempt_inventory(root, (DIRECT_STATE_ARM,))
    return _hash_bound(
        {
            "schema": DIRECT_STATE_MANIFEST_SCHEMA,
            "protocol": DIRECT_STATE_MANIFEST_PROTOCOL,
            "development_plan": _plan_reference(root),
            "attempt_claim": _reference(root / "attempt_claim.json", root),
            "attempt_start": _reference(root / "attempt_start.json", root),
            "private_refit_verification": _reference(
                root / "direct_refit_verification.json", root
            ),
            "stage_receipts": _stage_receipts(
                root,
                (ROLE_PHASE1, ROLE_DIRECT_VERIFIER),
                open_role=ROLE_DIRECT_VERIFIER,
            ),
            "reports": [
                _run_record(
                    root,
                    index,
                    DIRECT_STATE_ARM,
                )
                for index in range(3)
            ],
        }
    )


def build_development_producer_manifest(root: Path) -> dict[str, object]:
    arms = (DIRECT_STATE_ARM, *SCORED_ARMS)
    _validate_attempt_inventory(root, arms)
    return _hash_bound(
        {
            "schema": "r12_acw_development_producer_manifest_v1",
            "protocol": "R12-ACW-DEVELOPMENT-PRODUCER-MANIFEST-v1",
            "development_plan": _plan_reference(root),
            "attempt_claim": _reference(root / "attempt_claim.json", root),
            "attempt_start": _reference(root / "attempt_start.json", root),
            "phase2_authorization": _reference(
                root / "phase2_authorization.json", root
            ),
            "direct_refit_verification": _reference(
                root / "direct_refit_verification.json", root
            ),
            "stage_receipts": _stage_receipts(
                root,
                (ROLE_PHASE1, ROLE_DIRECT_VERIFIER, ROLE_PHASE2),
                open_role=ROLE_PHASE2,
            ),
            "reports": [
                _run_record(
                    root,
                    index,
                    arm,
                )
                for arm in arms
                for index in range(3)
            ],
        }
    )


def build_development_manifest(root: Path) -> dict[str, object]:
    arms = (DIRECT_STATE_ARM, *SCORED_ARMS)
    _validate_attempt_inventory(root, arms)
    return _hash_bound(
        {
            "schema": DEVELOPMENT_MANIFEST_SCHEMA,
            "protocol": DEVELOPMENT_MANIFEST_PROTOCOL,
            "development_plan": _plan_reference(root),
            "attempt_claim": _reference(root / "attempt_claim.json", root),
            "attempt_start": _reference(root / "attempt_start.json", root),
            "phase2_authorization": _reference(
                root / "phase2_authorization.json", root
            ),
            "direct_refit_verification": _reference(
                root / "direct_refit_verification.json", root
            ),
            "private_refit_verification": _reference(
                root / "final_refit_verification.json", root
            ),
            "stage_receipts": _stage_receipts(
                root,
                ROLES,
                open_role=ROLE_FINAL_VERIFIER,
            ),
            "reports": [
                _run_record(
                    root,
                    index,
                    arm,
                )
                for arm in arms
                for index in range(3)
            ],
        }
    )


def _git_commit(repository: Path) -> str:
    commit = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_commit = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "rev-parse",
            "refs/remotes/origin/main",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != remote_commit:
        raise ValueError("attempt start requires pushed origin/main")
    return commit


def _validated_g_commit(repository: Path, plan: dict[str, Any]) -> str:
    commit = _git_commit(repository)
    lineage = (
        subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "rev-list",
                "--parents",
                "-n",
                "1",
                commit,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()
    )
    if lineage != [commit, plan["execution_parent_commit"]]:
        raise ValueError("development G must be the sole direct child of custody F")
    remote = (
        subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "ls-remote",
                "origin",
                "refs/heads/main",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()
    )
    if len(remote) != 2 or remote[0] != commit:
        raise ValueError("development G is not the pushed origin/main commit")
    status = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("development execution requires a clean G worktree")
    return commit


def build_attempt_start(root: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    runs = root / "runs"
    if runs.exists() and any(runs.iterdir()):
        raise ValueError("attempt start must precede every fit and checkpoint")
    expected_root = repository.parent / "shohin_acw/artifacts/r12/acw_development_g1"
    if root != expected_root:
        raise ValueError("attempt root differs from the committed fixed path")
    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    commit = _validated_g_commit(repository, plan)
    stage = _stage_by_role(plan, ROLE_PHASE1)
    held_job_id = str(stage["held_slurm_job_id"])
    if (
        not str(slurm["job_id"] or "").isdigit()
        or str(slurm["job_id"]) != held_job_id
        or slurm["job_name"] != stage["job_name"]
        or slurm["node_list"] != ROLE_NODES[ROLE_PHASE1]
        or slurm["cpus_per_task"] != "4"
    ):
        raise ValueError("attempt start has a noncanonical Slurm identity")
    return _hash_bound(
        {
            "schema": "r12_acw_development_attempt_start_v1",
            "protocol": "R12-ACW-DEVELOPMENT-ATTEMPT-START-v1",
            "scientific_commit": commit,
            "development_plan": _plan_reference(root),
            "artifact_root": str(root),
            "slurm": slurm,
            "created_before_scoring": True,
            "checkpoint_count_at_creation": 0,
            "one_attempt": True,
            "overwrite": False,
            "attempt_ids": plan["attempt_registry"]["attempt_ids"],
        }
    )


def build_attempt_claim(root: Path) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    if list(root.rglob("checkpoint.pt")):
        raise ValueError("attempt claim must precede every checkpoint")
    attempts = plan["attempt_table"]
    return _hash_bound(
        {
            "schema": "r12_acw_development_attempt_claim_v1",
            "protocol": "R12-ACW-DEVELOPMENT-ATTEMPT-CLAIM-v1",
            "development_plan": _plan_reference(root),
            "artifact_root": str(root),
            "attempt_count": 27,
            "attempt_table_sha256": hashlib.sha256(
                canonical_json_bytes(attempts)
            ).hexdigest(),
            "jobs_sha256": hashlib.sha256(
                canonical_json_bytes(plan["custody_stages"])
            ).hexdigest(),
            "all_argv_and_paths_claimed_before_scoring": True,
            "checkpoint_count_at_creation": 0,
            "confirmation_authorized": False,
            "claim_boundary": plan["claim_boundary"],
        }
    )


def _protocol_publish_temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.r12-acw-publish.tmp")


def _discard_protocol_publish_temp(path: Path) -> None:
    temporary = _protocol_publish_temp(path)
    if not os.path.lexists(temporary):
        return
    if temporary.is_symlink() or not temporary.is_file():
        raise ValueError(f"protocol publication temporary is unsafe: {temporary}")
    temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_publish_bytes(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _discard_protocol_publish_temp(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = _protocol_publish_temp(path)
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while staging immutable publication")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if (
            temporary.is_symlink()
            or not temporary.is_file()
            or stat.S_IMODE(temporary.stat().st_mode) != 0o444
            or temporary.read_bytes() != raw
        ):
            raise ValueError("staged immutable publication differs")
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            if temporary.is_symlink() or not temporary.is_file():
                raise ValueError(
                    f"protocol publication temporary became unsafe: {temporary}"
                )
            temporary.unlink()
            _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def write_exclusive(path: Path, payload: dict[str, object]) -> str:
    return _atomic_publish_bytes(path, canonical_json_bytes(payload) + b"\n")


def _write_immutable_bytes(path: Path, raw: bytes) -> str:
    return _atomic_publish_bytes(path, raw)


def _job_snapshot(job_id: str) -> tuple[str, str]:
    completed = subprocess.run(
        ["/apps/slurm/current/bin/scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout.strip()
    if not raw or f"JobId={job_id}" not in raw:
        raise ValueError("Slurm job snapshot is unavailable")
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _current_planned_job_binding(job: dict[str, Any], *, label: str) -> dict[str, Any]:
    from pipeline.freeze_acw_curriculum import (
        CANONICAL_PILOT_UID,
        _validate_canonical_pilot_process_membership,
    )

    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    node = str(os.environ.get("SLURM_JOB_NODELIST", ""))
    cpus = str(os.environ.get("SLURM_CPUS_PER_TASK", ""))
    if (
        job_id != str(job["held_slurm_job_id"])
        or node != job["expected_node"]
        or cpus != "4"
        or os.environ.get("SLURM_JOB_NAME") != job["job_name"]
    ):
        raise ValueError(f"live Slurm allocation differs from plan {label}")
    snapshot, snapshot_sha256 = _job_snapshot(job_id)
    if (
        f"JobName={job['job_name']}" not in snapshot
        or f"NodeList={job['expected_node']}" not in snapshot
    ):
        raise ValueError("live Slurm job snapshot differs from plan")
    expected_dependency = job["dependency"]
    if expected_dependency is not None:
        token = f"afterok:{expected_dependency['held_slurm_job_id']}"
        if (
            f"Dependency={token}" not in snapshot
            and "Dependency=(null)" not in snapshot
        ):
            raise ValueError("live Slurm dependency differs from precommitted chain")
    script_path = REPOSITORY / job["script"]["path"]
    spool_path = Path(f"/var/spool/slurmd/job{job_id}/slurm_script")
    if not spool_path.is_file() or spool_path.is_symlink():
        raise ValueError("literal Slurm spool script is unavailable")
    spool_sha256 = file_sha256(spool_path)
    if (
        spool_sha256 != file_sha256(script_path)
        or spool_sha256 != job["script"]["sha256"]
    ):
        raise ValueError("Slurm spool script differs from committed plan bytes")
    membership = _validate_canonical_pilot_process_membership(
        Path("/proc/self/cgroup").read_text(errors="strict"),
        Path("/proc/self/status").read_text(errors="strict"),
        job_id=job_id,
        user_id=CANONICAL_PILOT_UID,
    )
    expected_cgroup = f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{job_id}/step_batch/task_0"
    if membership.get("task_cgroup") != expected_cgroup:
        raise ValueError("custody stage is not running in its top-level batch cgroup")
    return {
        "job_id": job_id,
        "job_name": job["job_name"],
        "node": node,
        "cpus_per_task": cpus,
        "dependency": expected_dependency,
        "script": dict(job["script"]),
        "spool_script_sha256": spool_sha256,
        "scontrol_snapshot_sha256": snapshot_sha256,
        "process_membership": membership,
    }


def _current_job_binding(plan: dict[str, Any], role: str) -> dict[str, Any]:
    return _current_planned_job_binding(_stage_by_role(plan, role), label=role)


def _current_monitor_binding(plan: dict[str, Any]) -> dict[str, Any]:
    return _current_planned_job_binding(_monitor_stage(plan), label="terminal monitor")


def _require_canonical_monitor_runtime(plan: dict[str, Any]) -> dict[str, Any]:
    import torch

    from pipeline.freeze_acw_curriculum import (
        CANONICAL_PILOT_RUNTIME,
        CANONICAL_PILOT_STATIC_ENV,
        pilot_runtime_identity,
    )

    dynamic_keys = {
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_NODELIST",
        "SLURM_SUBMIT_DIR",
    }
    if set(os.environ) != set(CANONICAL_PILOT_STATIC_ENV) | dynamic_keys:
        raise ValueError("terminal monitor environment allowlist differs")
    observed_static = {
        key: os.environ.get(key) for key in sorted(CANONICAL_PILOT_STATIC_ENV)
    }
    if observed_static != CANONICAL_PILOT_STATIC_ENV:
        raise ValueError("terminal monitor static environment differs")
    binding = _current_monitor_binding(plan)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observed = pilot_runtime_identity()
    runtime_sha256 = hashlib.sha256(canonical_json_bytes(observed)).hexdigest()
    if (
        observed != CANONICAL_PILOT_RUNTIME
        or runtime_sha256 != _monitor_stage(plan)["runtime_identity_sha256"]
    ):
        raise ValueError("terminal monitor numerical runtime differs")
    return {**binding, "runtime_identity_sha256": runtime_sha256}


def _build_predecessor_handoff(root: Path, role: str) -> dict[str, Any] | None:
    ordinal = ROLES.index(role)
    if ordinal == 0:
        return None
    predecessor = ROLES[ordinal - 1]
    predecessor_stage = STAGES[ordinal - 1]
    scans = {
        "main": closed_world_scan(
            root,
            predecessor_stage,
            include_current_accounting=True,
        )
    }
    if ordinal >= 2:
        scans["direct_verifier"] = closed_world_scan(
            root,
            predecessor_stage,
            scan_root=DIRECT_PRIVATE_ROOT,
        )
    return {
        "predecessor_role": predecessor,
        "predecessor_stage": predecessor_stage,
        "predecessor_completion": _reference(
            root / ROLE_COMPLETION_FILES[predecessor], root
        ),
        "predecessor_terminal_accounting": _reference(
            root / ROLE_ACCOUNTING_FILES[predecessor], root
        ),
        "live_closed_world_before_consumer_scoring": scans,
        "consumer_observed_before_role_scoring": True,
    }


def build_stage_start(root: Path, role: str) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    scientific_commit = _validated_g_commit(REPOSITORY, plan)
    if role == ROLE_PHASE1:
        if list(root.rglob("checkpoint.pt")):
            raise ValueError("phase-1 start must precede every checkpoint")
    elif role == ROLE_PHASE2:
        for record in plan["attempt_table"]:
            if (
                record["phase"] == "phase2"
                and Path(record["producer"]["paths"]["checkpoint"]).exists()
            ):
                raise ValueError("phase-2 start must precede every scored checkpoint")
        for required in (
            root / "phase2_authorization.json",
            root / "direct_refit_verification.json",
            root / ROLE_COMPLETION_FILES[ROLE_DIRECT_VERIFIER],
        ):
            _reference(required, root)
    else:
        private_root = ROLE_ROOTS[role]
        if private_root.exists() and list(private_root.rglob("checkpoint.pt")):
            raise ValueError("verifier start must precede every private refit")
    return _hash_bound(
        {
            "schema": "r12_acw_development_stage_start_v1",
            "protocol": "R12-ACW-DEVELOPMENT-STAGE-START-v1",
            "role": role,
            "development_plan": _plan_reference(root),
            "attempt_claim": _reference(root / "attempt_claim.json", root),
            "scientific_commit": scientific_commit,
            "slurm": _current_job_binding(plan, role),
            "planned_work": _stage_by_role(plan, role)["work"],
            "predecessor_handoff": _build_predecessor_handoff(root, role),
            "created_before_role_scoring": True,
            "confirmation_authorized": False,
        }
    )


def _run_argv(argv: Any, *, label: str) -> None:
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or argv[:3] != [str(PYTHON), "-S", "-P"]
    ):
        raise ValueError(f"{label} is not canonical argv")
    subprocess.run(argv, cwd=BASE, env=dict(os.environ), check=True)


def run_inputs(root: Path, role: str, index: int) -> None:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    records = [
        record
        for record in plan["input_table"]
        if record["role"] == role and record["index"] == index
    ]
    if len(records) != 1:
        raise ValueError("input execution is not uniquely precommitted")
    record = records[0]
    if _stage_by_role(plan, role)["held_slurm_job_id"] != os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise ValueError("input execution job differs from the committed stage")
    paths = {
        key: _resolved(value, label=f"input {key}")
        for key, value in record["paths"].items()
    }
    if any(os.path.lexists(path) for path in paths.values()):
        raise FileExistsError("input output path already exists")
    _run_argv(record["generator_argv"], label="generator argv")
    _run_argv(record["cgb_bundle_argv"], label="CGB bundle argv")
    _run_argv(record["uniform_bundle_argv"], label="uniform bundle argv")
    for tree in paths.values():
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"input tree contains a symlink: {path}")
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            tree.rglob("*"), key=lambda value: len(value.parts), reverse=True
        ):
            if path.is_dir():
                path.chmod(0o555)
        tree.chmod(0o555)


def _attempt(
    plan: dict[str, Any], role: str, index: int, arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        record
        for record in plan["attempt_table"]
        if record["index"] == index
        and record["logical_arm"] == arm
        and role in {record["producer"]["job_role"], record["verifier"]["job_role"]}
    ]
    if len(matches) != 1:
        raise ValueError("attempt execution is not uniquely precommitted")
    attempt = matches[0]
    side_name = "producer" if attempt["producer"]["job_role"] == role else "verifier"
    return attempt, attempt[side_name]


def run_attempt(root: Path, role: str, index: int, arm: str) -> None:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    attempt, side = _attempt(plan, role, index, arm)
    stage = _stage_by_role(plan, role)
    if (
        str(side["job_id"]) != str(stage["held_slurm_job_id"])
        or side["node"] != ROLE_NODES[role]
        or os.environ.get("SLURM_JOB_ID") != str(stage["held_slurm_job_id"])
    ):
        raise ValueError("attempt allocation binding differs")
    paths = {
        key: _resolved(value, label=f"attempt {key}")
        for key, value in side["paths"].items()
    }
    for key in ("dataset", "bundle", "curriculum"):
        path = paths[key]
        if not path.exists() or path.is_symlink():
            raise ValueError(f"attempt input is unavailable: {path}")
    task = paths["task_root"]
    if os.path.lexists(task):
        raise FileExistsError(task)
    task.mkdir(parents=True, mode=0o700)
    try:
        _run_argv(side["train_argv"], label="training argv")
        _run_argv(side["evaluation_argv"], label="evaluation argv")
        _run_argv(side["replay_argv"], label="replay argv")
        if paths["evaluation"].read_bytes() != paths["replay"].read_bytes():
            raise ValueError("two evaluator artifacts are not byte-identical")
        for name in ("checkpoint", "evaluation", "replay"):
            path = paths[name]
            if (
                not path.is_file()
                or path.is_symlink()
                or stat.S_IMODE(path.stat().st_mode) != 0o444
            ):
                raise ValueError(f"attempt output is not immutable: {path}")
        side_root = ROLE_ROOTS[role]
        receipt = _hash_bound(
            {
                "schema": "r12_acw_development_attempt_receipt_v1",
                "protocol": "R12-ACW-DEVELOPMENT-ATTEMPT-v1",
                "attempt_id": attempt["attempt_id"],
                "role": role,
                "logical_arm": attempt["logical_arm"],
                "trainer_arm": attempt["trainer_arm"],
                "seed": attempt["seed"],
                "development_plan_sha256": DEVELOPMENT_PLAN_RAW_SHA256,
                "artifact_root": str(side_root.resolve(strict=True)),
                "task_root": task.relative_to(side_root).as_posix(),
                "slurm": _current_job_binding(plan, role),
                "outputs": {
                    name: {
                        "path": paths[name].relative_to(side_root).as_posix(),
                        "sha256": file_sha256(paths[name]),
                    }
                    for name in ("checkpoint", "evaluation", "replay")
                },
                "completed_once": True,
                "confirmation_authorized": False,
            }
        )
        write_exclusive(task / "attempt.json", receipt)
        task.chmod(0o555)
    except BaseException:
        for path in task.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        task.chmod(0o555)
        raise
    print(
        f"[acw-development-attempt] id={attempt['attempt_id']} role={role} "
        f"job={stage['held_slurm_job_id']} slot={index}"
    )


def _tensor_state_sha256(state: Any) -> str:
    import torch

    if not isinstance(state, dict) or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("checkpoint tensor state has the wrong shape")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_reproducibility(path: Path) -> dict[str, Any]:
    import torch

    with path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload is not an object")
    training = copy.deepcopy(checkpoint.get("training_report"))
    if not isinstance(training, dict):
        raise ValueError("checkpoint training report is missing")
    for key in ("wall_seconds", "resource_measurements", "execution_receipt"):
        training.pop(key, None)
    snapshots = checkpoint.get("label_efficiency_models")
    if snapshots is None:
        snapshot_hashes: list[str] = []
    elif isinstance(snapshots, list):
        snapshot_hashes = [_tensor_state_sha256(state) for state in snapshots]
    else:
        raise ValueError("checkpoint label-efficiency states are malformed")
    stable = {
        key: checkpoint.get(key)
        for key in (
            "protocol",
            "arm",
            "seed",
            "dataset_manifest_payload_sha256",
            "source_manifest_payload_sha256",
            "curriculum_sha256",
            "query_schedule_sha256",
            "query_schedule_kind",
            "pilot_report_payload_sha256",
            "parameters",
            "scientific_identity",
        )
    }
    stable["training_report"] = training
    stable["model_tensor_sha256"] = _tensor_state_sha256(checkpoint.get("model"))
    stable["label_efficiency_model_sha256"] = snapshot_hashes
    return {
        "raw_sha256": file_sha256(path),
        "model_tensor_sha256": stable["model_tensor_sha256"],
        "stable_payload_sha256": hashlib.sha256(
            canonical_json_bytes(stable)
        ).hexdigest(),
        "stable_payload": stable,
        "execution_receipt": checkpoint["training_report"].get("execution_receipt"),
    }


def _normalized_evaluation(path: Path) -> tuple[str, dict[str, Any]]:
    report, raw = _load_canonical_json(path, "evaluation report")
    normalized = copy.deepcopy(report)
    normalized.pop("checkpoint_sha256", None)
    normalized.pop("payload_sha256", None)
    training = normalized.get("training_evidence")
    if isinstance(training, dict):
        training.pop("resource_measurements", None)
        training.pop("execution_receipt", None)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest(), {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_payload_sha256": hashlib.sha256(
            canonical_json_bytes(normalized)
        ).hexdigest(),
    }


def _validate_receipt_stage(
    receipt: Any,
    expected: dict[str, Any],
    plan: dict[str, Any],
    *,
    attempt_id: str,
) -> None:
    from pipeline.freeze_acw_curriculum import CANONICAL_PILOT_UID

    if not isinstance(receipt, dict) or not isinstance(receipt.get("slurm"), dict):
        raise ValueError("checkpoint execution receipt is missing")
    slurm = receipt["slurm"]
    role = expected["job_role"]
    stage = _stage_by_role(plan, role)
    required = {
        "job_id": stage["held_slurm_job_id"],
        "job_name": stage["job_name"],
        "node_list": expected["node"],
        "cpus_per_task": "4",
    }
    membership = receipt.get("process_membership")
    task_cgroup = (
        membership.get("task_cgroup") if isinstance(membership, dict) else None
    )
    expected_cgroup = (
        f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{stage['held_slurm_job_id']}"
        "/step_batch/task_0"
    )
    if (
        any(str(slurm.get(key)) != str(value) for key, value in required.items())
        or receipt.get("role") != role
        or receipt.get("attempt_id") != attempt_id
        or task_cgroup != expected_cgroup
    ):
        raise ValueError("checkpoint top-level stage receipt differs from plan")


def _require_registered_tree_bytes_equal(
    producer_root: Path, verifier_root: Path, *, kind: str
) -> None:
    producer_files = _registered_tree_files(producer_root, kind=kind)
    verifier_files = _registered_tree_files(verifier_root, kind=kind)
    producer_relatives = {path.relative_to(producer_root) for path in producer_files}
    verifier_relatives = {path.relative_to(verifier_root) for path in verifier_files}
    if producer_relatives != verifier_relatives:
        raise ValueError(f"private {kind} file registry differs")
    for relative in sorted(producer_relatives, key=lambda path: path.as_posix()):
        if (producer_root / relative).read_bytes() != (
            verifier_root / relative
        ).read_bytes():
            raise ValueError(f"private {kind} regeneration differs: {relative}")


def verify_refits(root: Path, scope: str) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    if scope == "direct":
        _validate_attempt_inventory(root, (DIRECT_STATE_ARM,))
        _validate_attempt_inventory(DIRECT_PRIVATE_ROOT, (DIRECT_STATE_ARM,))
    else:
        _validate_attempt_inventory(root, (DIRECT_STATE_ARM, *SCORED_ARMS))
        _validate_attempt_inventory(FINAL_PRIVATE_ROOT, SCORED_ARMS)
    attempts = [
        record
        for record in plan["attempt_table"]
        if (scope == "direct" and record["logical_arm"] == DIRECT_STATE_ARM)
        or (scope == "final" and record["logical_arm"] in SCORED_ARMS)
    ]
    expected_count = 3 if scope == "direct" else 24
    if len(attempts) != expected_count:
        raise ValueError("refit scope has the wrong attempt count")
    comparisons = []
    for attempt in attempts:
        producer = attempt["producer"]
        verifier = attempt["verifier"]
        producer_checkpoint = _checkpoint_reproducibility(
            Path(producer["paths"]["checkpoint"])
        )
        verifier_checkpoint = _checkpoint_reproducibility(
            Path(verifier["paths"]["checkpoint"])
        )
        _validate_receipt_stage(
            producer_checkpoint["execution_receipt"],
            producer,
            plan,
            attempt_id=attempt["attempt_id"],
        )
        _validate_receipt_stage(
            verifier_checkpoint["execution_receipt"],
            verifier,
            plan,
            attempt_id=attempt["attempt_id"],
        )
        if (
            producer_checkpoint["stable_payload"]
            != verifier_checkpoint["stable_payload"]
        ):
            raise ValueError(f"private refit differs: {attempt['attempt_id']}")
        producer_eval_hash, producer_eval = _normalized_evaluation(
            Path(producer["paths"]["evaluation"])
        )
        verifier_eval_hash, verifier_eval = _normalized_evaluation(
            Path(verifier["paths"]["evaluation"])
        )
        if producer_eval_hash != verifier_eval_hash:
            raise ValueError(
                f"private refit evaluation differs: {attempt['attempt_id']}"
            )
        if (
            Path(producer["paths"]["evaluation"]).read_bytes()
            != Path(producer["paths"]["replay"]).read_bytes()
            or Path(verifier["paths"]["evaluation"]).read_bytes()
            != Path(verifier["paths"]["replay"]).read_bytes()
        ):
            raise ValueError("stored evaluator replay differs")
        for family in ("dataset", "bundle"):
            _require_registered_tree_bytes_equal(
                Path(producer["paths"][family]),
                Path(verifier["paths"][family]),
                kind=family,
            )
        comparisons.append(
            {
                "attempt_id": attempt["attempt_id"],
                "model_tensor_sha256": producer_checkpoint["model_tensor_sha256"],
                "stable_checkpoint_payload_sha256": producer_checkpoint[
                    "stable_payload_sha256"
                ],
                "producer_checkpoint_sha256": producer_checkpoint["raw_sha256"],
                "verifier_checkpoint_sha256": verifier_checkpoint["raw_sha256"],
                "producer_evaluation": producer_eval,
                "verifier_evaluation": verifier_eval,
                "producer_stage": producer["job_role"],
                "verifier_stage": verifier["job_role"],
            }
        )
    return _hash_bound(
        {
            "schema": "r12_acw_development_private_refit_verification_v1",
            "protocol": "R12-ACW-DEVELOPMENT-PRIVATE-REFIT-VERIFICATION-v1",
            "scope": scope,
            "development_plan": _plan_reference(root),
            "attempt_count": expected_count,
            "comparisons": comparisons,
            "datasets_regenerated_privately": True,
            "curricula_regenerated_privately": True,
            "models_refit_from_private_copies": True,
            "model_tensors_byte_identical": True,
            "normalized_evaluations_identical": True,
            "confirmation_authorized": False,
        }
    )


def _registered_tree_files(tree: Path, *, kind: str) -> set[Path]:
    manifest, _ = _load_canonical_json(tree / "manifest.json", f"{kind} manifest")
    relatives = {"manifest.json"}
    if kind == "dataset":
        registries = (manifest.get("arrays"),)
    else:
        registries = (
            manifest.get("arrays"),
            manifest.get("files"),
            manifest.get("pilot_artifacts"),
        )
    for registry in registries:
        if not isinstance(registry, dict):
            raise ValueError(f"{kind} manifest registry is missing")
        relatives.update(str(relative) for relative in registry)
    expected = {tree / relative for relative in relatives}
    actual = {
        path for path in tree.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise ValueError(f"{kind} tree differs from its exact manifest registry")
    return expected


def _role_output_files(root: Path, role: str) -> set[Path]:
    names = {
        ROLE_PHASE1: ("direct_state_producer_manifest.json",),
        ROLE_DIRECT_VERIFIER: (
            "direct_refit_verification.json",
            "direct_state_manifest.json",
            "direct_state_decision.json",
            "phase2_authorization.json",
        ),
        ROLE_PHASE2: ("development_producer_manifest.json",),
        ROLE_FINAL_VERIFIER: (
            "final_refit_verification.json",
            "development_manifest.json",
        ),
    }[role]
    return {root / name for name in names}


def _expected_scan_files(
    root: Path,
    stage: str,
    scan_root: Path,
    *,
    pending_completion: Path | None,
    include_current_accounting: bool,
    include_terminal_outputs: bool,
) -> set[Path]:
    if stage not in STAGES:
        raise ValueError(f"unknown closed-world stage: {stage}")
    plan = _plan_for_root(root)
    current_role = STAGE_ROLES[stage]
    current_ordinal = ROLES.index(current_role)
    expected: set[Path] = set()

    if scan_root == root:
        expected.update(
            {
                root / PLAN_COPY_NAME,
                root / "attempt_claim.json",
                root / "attempt_start.json",
            }
        )
        for ordinal, role in enumerate(ROLES[: current_ordinal + 1]):
            expected.add(root / ROLE_START_FILES[role])
            if ordinal < current_ordinal or pending_completion is None:
                expected.add(root / ROLE_COMPLETION_FILES[role])
            if ordinal < current_ordinal or include_current_accounting:
                expected.add(root / ROLE_ACCOUNTING_FILES[role])
            expected.update(_role_output_files(root, role))
        if include_terminal_outputs and stage == "final":
            expected.update(root / name for name in TERMINAL_OUTPUT_NAMES)

    for record in plan["input_table"]:
        if ROLES.index(record["role"]) > current_ordinal:
            continue
        for name, raw_path in record["paths"].items():
            tree = Path(raw_path)
            if tree.is_relative_to(scan_root):
                expected.update(
                    _registered_tree_files(
                        tree,
                        kind="dataset" if name == "dataset" else "bundle",
                    )
                )

    for attempt in plan["attempt_table"]:
        for side_name in ("producer", "verifier"):
            side = attempt[side_name]
            if ROLES.index(side["job_role"]) > current_ordinal:
                continue
            task = Path(side["paths"]["task_root"])
            if task.is_relative_to(scan_root):
                expected.update(
                    task / name
                    for name in (
                        "attempt.json",
                        "checkpoint.pt",
                        "evaluation.json",
                        "replay.json",
                    )
                )
    return expected


def closed_world_scan(
    root: Path,
    stage: str,
    *,
    scan_root: Path | None = None,
    pending_completion: Path | None = None,
    include_current_accounting: bool = False,
    include_terminal_outputs: bool = False,
    require_frozen_directories: bool = False,
) -> dict[str, Any]:
    scan_root = root if scan_root is None else scan_root
    expected = _expected_scan_files(
        root,
        stage,
        scan_root,
        pending_completion=pending_completion,
        include_current_accounting=include_current_accounting,
        include_terminal_outputs=include_terminal_outputs,
    )
    if pending_completion is not None:
        expected.discard(pending_completion)
    actual_files: set[Path] = set()
    actual_directories = {scan_root}
    for path in scan_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"closed-world root contains a symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            if stat.S_IMODE(mode) != 0o444:
                raise ValueError(f"closed-world file is not mode 0444: {path}")
            actual_files.add(path)
        elif stat.S_ISDIR(mode):
            if require_frozen_directories and stat.S_IMODE(mode) != 0o555:
                raise ValueError(f"closed-world directory is not mode 0555: {path}")
            actual_directories.add(path)
        else:
            raise ValueError(f"closed-world root contains a special file: {path}")
    if actual_files != expected:
        missing = sorted(
            str(path.relative_to(scan_root)) for path in expected - actual_files
        )
        extra = sorted(
            str(path.relative_to(scan_root)) for path in actual_files - expected
        )
        raise ValueError(
            f"closed-world file set differs; missing={missing}, extra={extra}"
        )
    expected_directories = {scan_root}
    for path in expected:
        parent = path.parent
        while parent != scan_root:
            expected_directories.add(parent)
            parent = parent.parent
    if actual_directories != expected_directories:
        missing = sorted(
            str(path.relative_to(scan_root))
            for path in expected_directories - actual_directories
        )
        extra = sorted(
            str(path.relative_to(scan_root))
            for path in actual_directories - expected_directories
        )
        raise ValueError(
            f"closed-world directory set differs; missing={missing}, extra={extra}"
        )
    records = []
    digest = hashlib.sha256()
    for path in sorted(
        actual_files, key=lambda value: value.relative_to(scan_root).as_posix()
    ):
        relative = path.relative_to(scan_root).as_posix()
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
            "sha256": file_sha256(path),
        }
        records.append(record)
        digest.update(canonical_json_bytes(record) + b"\n")
    return {
        "stage": stage,
        "root": str(scan_root),
        "file_count": len(records),
        "directory_count": len(actual_directories),
        "files": records,
        "tree_sha256": digest.hexdigest(),
        "exact_file_set": True,
        "exact_directory_set": True,
        "symlinks": 0,
        "special_files": 0,
    }


def _stage_output_references(root: Path, role: str) -> dict[str, dict[str, str]]:
    names = {
        ROLE_PHASE1: ("direct_state_producer_manifest.json",),
        ROLE_DIRECT_VERIFIER: (
            "direct_refit_verification.json",
            "direct_state_manifest.json",
            "direct_state_decision.json",
            "phase2_authorization.json",
        ),
        ROLE_PHASE2: ("development_producer_manifest.json",),
        ROLE_FINAL_VERIFIER: (
            "final_refit_verification.json",
            "development_manifest.json",
        ),
    }[role]
    return {name: _reference(root / name, root) for name in names}


def build_stage_completion(root: Path, stage: str, output: Path) -> dict[str, Any]:
    role = STAGE_ROLES[stage]
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    closed_world = {
        "main": closed_world_scan(
            root,
            stage,
            pending_completion=output,
        )
    }
    if role in {ROLE_DIRECT_VERIFIER, ROLE_FINAL_VERIFIER}:
        closed_world["private"] = closed_world_scan(
            root,
            stage,
            scan_root=ROLE_ROOTS[role],
        )
    return _hash_bound(
        {
            "schema": "r12_acw_development_stage_completion_v1",
            "protocol": "R12-ACW-DEVELOPMENT-STAGE-COMPLETION-v1",
            "role": role,
            "development_plan": _plan_reference(root),
            "stage_start": _reference(root / ROLE_START_FILES[role], root),
            "slurm": _current_job_binding(plan, role),
            "completed_work": _stage_by_role(plan, role)["work"],
            "outputs": _stage_output_references(root, role),
            "closed_world": closed_world,
            "all_outputs_immutable": True,
            "normal_slurm_steps_used": 0,
            "confirmation_authorized": False,
        }
    )


def _validated_terminal_rows_for_stage(
    stage: dict[str, Any], raw_output: str
) -> list[dict[str, str]]:
    job_id = str(stage["held_slurm_job_id"])
    rows: dict[str, dict[str, str]] = {}
    normal_steps: list[str] = []
    for raw in raw_output.splitlines():
        if not raw:
            continue
        fields = raw.split("|")
        if len(fields) != 8 or not fields[0]:
            raise ValueError("terminal accounting row is malformed")
        record = {
            "job_id_raw": fields[0],
            "job_name": fields[1],
            "state": fields[2].split()[0],
            "exit_code": fields[3],
            "node_list": fields[4],
            "cpus": fields[5],
            "elapsed_raw": fields[6],
            "max_rss": fields[7],
        }
        if record["job_id_raw"] in rows:
            raise ValueError("terminal accounting contains a duplicate row")
        rows[record["job_id_raw"]] = record
        if re.fullmatch(rf"{re.escape(job_id)}\.[0-9]+", record["job_id_raw"]):
            normal_steps.append(record["job_id_raw"])

    expected_names = {
        job_id: stage["job_name"],
        f"{job_id}.batch": "batch",
        f"{job_id}.extern": "extern",
    }
    if set(rows) != set(expected_names) or normal_steps:
        raise ValueError("terminal accounting row set is not exact and step-free")
    normalized = []
    for row_id, expected_name in expected_names.items():
        record = rows[row_id]
        if (
            record["job_name"] != expected_name
            or record["state"] != "COMPLETED"
            or record["exit_code"] != "0:0"
            or record["node_list"] != stage["expected_node"]
            or record["cpus"] != "4"
        ):
            raise ValueError("terminal accounting allocation binding differs")
        normalized.append(record)
    return normalized


def _validated_terminal_rows(
    plan: dict[str, Any], role: str, raw_output: str
) -> list[dict[str, str]]:
    return _validated_terminal_rows_for_stage(_stage_by_role(plan, role), raw_output)


def _query_terminal_rows(plan: dict[str, Any], role: str) -> list[dict[str, str]]:
    return _query_terminal_rows_for_stage(plan, _stage_by_role(plan, role))


def _query_monitor_terminal_rows(plan: dict[str, Any]) -> list[dict[str, str]]:
    return _query_terminal_rows_for_stage(plan, _monitor_stage(plan))


def _query_terminal_rows_for_stage(
    plan: dict[str, Any], stage: dict[str, Any]
) -> list[dict[str, str]]:
    job_id = str(stage["held_slurm_job_id"])
    timeout = float(plan["accounting"]["poll_timeout_seconds"])
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while True:
        try:
            completed = subprocess.run(
                [
                    "/apps/slurm/current/bin/sacct",
                    "-n",
                    "-P",
                    "-j",
                    job_id,
                    "--format=JobIDRaw,JobName,State,ExitCode,NodeList,NCPUS,ElapsedRaw,MaxRSS",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return _validated_terminal_rows_for_stage(stage, completed.stdout)
        except (subprocess.CalledProcessError, ValueError) as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"terminal accounting did not stabilize for job {job_id}"
            ) from last_error
        time.sleep(min(2.0, remaining))


def build_stage_accounting(root: Path, role: str) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    ordinal = ROLES.index(role)
    if ordinal == len(ROLES) - 1:
        observer = dict(_require_canonical_monitor_runtime(plan))
        observer.pop("runtime_identity_sha256")
    else:
        observer = _current_job_binding(plan, ROLES[ordinal + 1])
    rows = _query_terminal_rows(plan, role)
    return _hash_bound(
        {
            "schema": "r12_acw_development_stage_accounting_v1",
            "protocol": "R12-ACW-DEVELOPMENT-STAGE-ACCOUNTING-v1",
            "role": role,
            "development_plan": _plan_reference(root),
            "observed_by": observer,
            "terminal_rows": rows,
            "normal_slurm_steps": [],
            "terminal_completed": True,
            "resource_values_are_diagnostic_only": True,
            "confirmation_authorized": False,
        }
    )


def _freeze_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"custody tree is unavailable: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"custody tree contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif not path.is_dir():
            raise ValueError(f"custody tree contains a special file: {path}")
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    root.chmod(0o555)


def _validate_all_stage_receipts_and_accounting(
    root: Path, plan: dict[str, Any]
) -> None:
    scientific_commit = _validated_g_commit(REPOSITORY, plan)
    receipts = _stage_receipts(root, ROLES)
    _validate_stage_receipts(
        receipts,
        root,
        scope="development",
        expected_plan=plan,
        require_all_closed=True,
    )
    for role in ROLES:
        start, _ = _load_canonical_json(
            root / ROLE_START_FILES[role], f"{role} stage start"
        )
        if start.get("scientific_commit") != scientific_commit:
            raise ValueError(f"{role} stage commit differs from installed G")
        accounting_path = root / ROLE_ACCOUNTING_FILES[role]
        accounting, _ = _load_canonical_json(
            accounting_path, f"{role} terminal accounting"
        )
        if accounting.get("terminal_rows") != _query_terminal_rows(plan, role):
            raise ValueError(f"{role} terminal accounting differs from fresh sacct")


def validate_terminal_prerequisites(root: Path) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    monitor = _require_canonical_monitor_runtime(plan)
    _validate_all_stage_receipts_and_accounting(root, plan)
    scans = {
        "main": closed_world_scan(
            root,
            "final",
            include_current_accounting=True,
        ),
        "direct_verifier": closed_world_scan(
            root,
            "final",
            scan_root=DIRECT_PRIVATE_ROOT,
        ),
        "final_verifier": closed_world_scan(
            root,
            "final",
            scan_root=FINAL_PRIVATE_ROOT,
        ),
    }
    return {"plan": plan, "monitor": monitor, "closed_world": scans}


def publish_development_baseline_after_terminal_validation(
    root: Path,
) -> dict[str, str]:
    validate_terminal_prerequisites(root)
    return _prepare_development_baseline_outputs(root)


def _prepare_development_baseline_outputs(root: Path) -> dict[str, str]:
    checkpoint_path = root / "best_development_checkpoint.pt"
    baseline_path = root / "development_baseline.json"
    _discard_protocol_publish_temp(checkpoint_path)
    _discard_protocol_publish_temp(baseline_path)
    checkpoint_exists = os.path.lexists(checkpoint_path)
    baseline_exists = os.path.lexists(baseline_path)
    if baseline_exists:
        if not checkpoint_exists:
            raise ValueError("development baseline exists without its checkpoint")
        validated = _validate_frozen_development_baseline(baseline_path)
        copied = validated["copied_checkpoint"]
        if Path(str(copied["path"])).resolve(strict=True) != checkpoint_path.resolve(
            strict=True
        ):
            raise ValueError("development baseline checkpoint path differs")
        return {
            "baseline_sha256": str(validated["record"]["sha256"]),
            "baseline_payload_sha256": str(validated["record"]["payload_sha256"]),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        }

    with tempfile.TemporaryDirectory(prefix="shohin-acw-baseline-") as temporary:
        staged_checkpoint = Path(temporary) / "best_development_checkpoint.pt"
        baseline = freeze_development_baseline(
            root / "development_manifest.json", staged_checkpoint
        )
        staged_raw = staged_checkpoint.read_bytes()
        staged_sha256 = hashlib.sha256(staged_raw).hexdigest()
        copied = dict(baseline["copied_checkpoint"])
        if copied.get("sha256") != staged_sha256 or copied.get("bytes") != len(
            staged_raw
        ):
            raise ValueError("staged development checkpoint binding differs")
        if checkpoint_exists:
            if (
                not checkpoint_path.is_file()
                or checkpoint_path.is_symlink()
                or stat.S_IMODE(checkpoint_path.stat().st_mode) != 0o444
                or checkpoint_path.read_bytes() != staged_raw
            ):
                raise ValueError("existing development checkpoint differs")
        else:
            observed_sha256 = _write_immutable_bytes(checkpoint_path, staged_raw)
            if observed_sha256 != staged_sha256:
                raise ValueError("preserved development checkpoint hash differs")
        copied.update(
            {
                "path": str(checkpoint_path.resolve(strict=True)),
                "sha256": staged_sha256,
                "bytes": len(staged_raw),
                "mode": "0444",
            }
        )
        baseline = _hash_bound({**baseline, "copied_checkpoint": copied})
        staged_baseline = Path(temporary) / "development_baseline.json"
        staged_baseline_sha256 = write_immutable_development_baseline(
            staged_baseline, baseline
        )
        staged_baseline_raw = staged_baseline.read_bytes()
        if hashlib.sha256(staged_baseline_raw).hexdigest() != staged_baseline_sha256:
            raise ValueError("staged development baseline hash differs")
        baseline_sha256 = _atomic_publish_bytes(baseline_path, staged_baseline_raw)
    validated = _validate_frozen_development_baseline(baseline_path)
    if validated["record"]["sha256"] != baseline_sha256:
        raise ValueError("development baseline validation hash differs")
    return {
        "baseline_sha256": baseline_sha256,
        "baseline_payload_sha256": str(baseline["payload_sha256"]),
        "checkpoint_sha256": staged_sha256,
    }


def build_terminal_accounting(root: Path) -> dict[str, Any]:
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    monitor = _require_canonical_monitor_runtime(plan)
    _validate_all_stage_receipts_and_accounting(root, plan)
    for role in ROLES:
        _reference(root / ROLE_START_FILES[role], root)
        _reference(root / ROLE_COMPLETION_FILES[role], root)
        _reference(root / ROLE_ACCOUNTING_FILES[role], root)
    for required in (
        "development_manifest.json",
        "development_baseline.json",
        "best_development_checkpoint.pt",
    ):
        _reference(root / required, root)
    validated_baseline = _validate_frozen_development_baseline(
        root / "development_baseline.json"
    )
    semantic_baseline = {
        "record": validated_baseline["record"],
        "development_manifest": validated_baseline["development_manifest"],
        "source_checkpoint": validated_baseline["source_checkpoint"],
        "copied_checkpoint": validated_baseline["copied_checkpoint"],
        "selection_sha256": hashlib.sha256(
            canonical_json_bytes(validated_baseline["selection"])
        ).hexdigest(),
        "selection_rederived_from_registered_manifest": True,
        "copied_checkpoint_matches_selected_source_bytes": True,
    }

    for tree in (root, DIRECT_PRIVATE_ROOT, FINAL_PRIVATE_ROOT):
        _freeze_tree(tree)
    scans = {
        "main": closed_world_scan(
            root,
            "final",
            include_current_accounting=True,
            include_terminal_outputs=True,
            require_frozen_directories=True,
        ),
        "direct_verifier": closed_world_scan(
            root,
            "final",
            scan_root=DIRECT_PRIVATE_ROOT,
            require_frozen_directories=True,
        ),
        "final_verifier": closed_world_scan(
            root,
            "final",
            scan_root=FINAL_PRIVATE_ROOT,
            require_frozen_directories=True,
        ),
    }
    return _hash_bound(
        {
            "schema": "r12_acw_development_terminal_accounting_v1",
            "protocol": "R12-ACW-DEVELOPMENT-TERMINAL-ACCOUNTING-v1",
            "development_plan": _plan_reference(root),
            "scientific_commit": _validated_g_commit(REPOSITORY, plan),
            "monitor": monitor,
            "stages": {
                role: {
                    "start": _reference(root / ROLE_START_FILES[role], root),
                    "completion": _reference(root / ROLE_COMPLETION_FILES[role], root),
                    "terminal_accounting": _reference(
                        root / ROLE_ACCOUNTING_FILES[role], root
                    ),
                }
                for role in ROLES
            },
            "development_manifest": _reference(
                root / "development_manifest.json", root
            ),
            "development_baseline": _reference(
                root / "development_baseline.json", root
            ),
            "baseline_checkpoint": _reference(
                root / "best_development_checkpoint.pt", root
            ),
            "semantic_baseline_validation": semantic_baseline,
            "closed_world": scans,
            "all_four_jobs_terminal_and_step_free": True,
            "exact_registered_root_verified": True,
            "same_uid_external_compute_excluded": False,
            "ordinary_batch_children_independently_attested": False,
            "claim_limited_to_exact_final_rooted_files_and_slurm_rows": True,
            "resource_values_are_diagnostic_only": True,
            "required_before_any_performance_claim": True,
            "external_sha256_anchor_required_before_performance_claim": True,
            "performance_claim_ready": False,
            "confirmation_authorized": False,
            "promotion_authorized": False,
        }
    )


def validate_terminal_accounting(
    root: Path, receipt_path: Path = TERMINAL_ACCOUNTING_PATH
) -> dict[str, Any]:
    if receipt_path.resolve(strict=True) != TERMINAL_ACCOUNTING_PATH:
        raise ValueError("terminal accounting path differs from the committed plan")
    if receipt_path.is_symlink() or stat.S_IMODE(receipt_path.stat().st_mode) != 0o444:
        raise ValueError("terminal accounting receipt is not immutable")
    recorded, _ = _load_canonical_json(receipt_path, "terminal accounting receipt")
    fresh = build_terminal_accounting(root)
    expected_keys = set(fresh)
    if set(recorded) != expected_keys:
        raise ValueError("terminal accounting receipt schema differs")
    monitor = recorded.get("monitor")
    fresh_monitor = fresh["monitor"]
    if (
        not isinstance(monitor, dict)
        or set(monitor) != set(fresh_monitor)
        or {
            key: value
            for key, value in monitor.items()
            if key != "scontrol_snapshot_sha256"
        }
        != {
            key: value
            for key, value in fresh_monitor.items()
            if key != "scontrol_snapshot_sha256"
        }
        or HASH_RE.fullmatch(str(monitor.get("scontrol_snapshot_sha256"))) is None
    ):
        raise ValueError("terminal accounting monitor identity differs")
    for key in expected_keys - {"monitor", "payload_sha256"}:
        if recorded[key] != fresh[key]:
            raise ValueError(f"terminal accounting receipt differs: {key}")
    if recorded.get("performance_claim_ready") is not False:
        raise ValueError(
            "terminal accounting cannot self-authorize a performance claim"
        )
    return recorded


def _validate_recorded_monitor_binding(
    plan: dict[str, Any], value: Any
) -> dict[str, Any]:
    from pipeline.freeze_acw_curriculum import CANONICAL_PILOT_UID, _cpu_list_members

    if not isinstance(value, dict):
        raise ValueError("terminal monitor binding is not an object")
    expected_keys = {
        "job_id",
        "job_name",
        "node",
        "cpus_per_task",
        "dependency",
        "script",
        "spool_script_sha256",
        "scontrol_snapshot_sha256",
        "process_membership",
        "runtime_identity_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("terminal monitor binding schema differs")
    monitor = _monitor_stage(plan)
    membership = value.get("process_membership")
    if not isinstance(membership, dict) or set(membership) != {
        "cpu_list",
        "memory_list",
        "task_cgroup",
    }:
        raise ValueError("terminal monitor process membership differs")
    try:
        cpu_members = _cpu_list_members(str(membership["cpu_list"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal monitor CPU membership is malformed") from exc
    job_id = str(monitor["held_slurm_job_id"])
    if (
        value["job_id"] != job_id
        or value["job_name"] != monitor["job_name"]
        or value["node"] != monitor["expected_node"]
        or value["cpus_per_task"] != "4"
        or value["dependency"] != monitor["dependency"]
        or value["script"] != monitor["script"]
        or value["spool_script_sha256"] != monitor["script"]["sha256"]
        or HASH_RE.fullmatch(str(value["scontrol_snapshot_sha256"])) is None
        or value["runtime_identity_sha256"] != monitor["runtime_identity_sha256"]
        or len(cpu_members) != 4
        or not str(membership["memory_list"])
        or membership["task_cgroup"]
        != f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{job_id}/step_batch/task_0"
    ):
        raise ValueError("terminal monitor binding differs from the plan")
    return dict(value)


def build_monitor_anchor_ready_envelope(root: Path) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID") is not None:
        raise ValueError("monitor anchor finalizer must run outside Slurm")
    plan = _plan_for_root(root)
    validate_plan(plan, require_ready=True)
    scientific_commit = _validated_g_commit(REPOSITORY, plan)
    receipt_path = Path(plan["accounting"]["terminal_receipt"])
    if receipt_path != TERMINAL_ACCOUNTING_PATH:
        raise ValueError("terminal receipt path differs")
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or stat.S_IMODE(receipt_path.stat().st_mode) != 0o444
    ):
        raise ValueError("terminal accounting receipt is not immutable")
    receipt, _ = _load_canonical_json(
        receipt_path, "terminal accounting receipt for monitor anchor"
    )
    expected_receipt_keys = {
        "schema",
        "protocol",
        "development_plan",
        "scientific_commit",
        "monitor",
        "stages",
        "development_manifest",
        "development_baseline",
        "baseline_checkpoint",
        "semantic_baseline_validation",
        "closed_world",
        "all_four_jobs_terminal_and_step_free",
        "exact_registered_root_verified",
        "same_uid_external_compute_excluded",
        "ordinary_batch_children_independently_attested",
        "claim_limited_to_exact_final_rooted_files_and_slurm_rows",
        "resource_values_are_diagnostic_only",
        "required_before_any_performance_claim",
        "external_sha256_anchor_required_before_performance_claim",
        "performance_claim_ready",
        "confirmation_authorized",
        "promotion_authorized",
        "payload_sha256",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt["schema"] != "r12_acw_development_terminal_accounting_v1"
        or receipt["protocol"] != "R12-ACW-DEVELOPMENT-TERMINAL-ACCOUNTING-v1"
        or receipt["development_plan"] != _plan_reference(root)
        or receipt["scientific_commit"] != scientific_commit
        or receipt["all_four_jobs_terminal_and_step_free"] is not True
        or receipt["exact_registered_root_verified"] is not True
        or receipt["required_before_any_performance_claim"] is not True
        or receipt["external_sha256_anchor_required_before_performance_claim"]
        is not True
        or receipt["performance_claim_ready"] is not False
        or receipt["confirmation_authorized"] is not False
        or receipt["promotion_authorized"] is not False
    ):
        raise ValueError("terminal accounting receipt contract differs")
    monitor = _validate_recorded_monitor_binding(plan, receipt["monitor"])
    terminal_rows = _query_monitor_terminal_rows(plan)
    monitor_job_id = str(_monitor_stage(plan)["held_slurm_job_id"])
    log_path = BASE / "logs" / f"acw_development_monitor_{monitor_job_id}.out"
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError("terminal monitor log is unavailable")
    log_raw = log_path.read_bytes()
    completion_marker = (
        f"[acw-development-monitor] complete root={root} performance_claim_ready=0\n"
    ).encode("ascii")
    if not log_raw.endswith(completion_marker):
        raise ValueError("terminal monitor completion marker is missing")
    return _hash_bound(
        {
            "schema": "r12_acw_development_monitor_anchor_v1",
            "protocol": "R12-ACW-DEVELOPMENT-MONITOR-ANCHOR-v1",
            "development_plan": _plan_reference(root),
            "scientific_commit": scientific_commit,
            "monitor": monitor,
            "monitor_terminal_rows": terminal_rows,
            "terminal_accounting": _reference(receipt_path, root),
            "monitor_log": {
                "path": str(log_path),
                "bytes": len(log_raw),
                "sha256": hashlib.sha256(log_raw).hexdigest(),
            },
            "monitor_completed_zero_exit_and_step_free": True,
            "monitor_receipt_verified_before_completed_exit": True,
            "exact_final_rooted_files_and_slurm_rows_only": True,
            "same_uid_external_compute_excluded": False,
            "external_sha256_anchor_required_before_performance_claim": True,
            "performance_claim_ready": False,
            "confirmation_authorized": False,
            "promotion_authorized": False,
        }
    )


def _output_for_kind(root: Path, kind: str, role: str | None) -> Path:
    names = {
        "attempt-start": "attempt_start.json",
        "attempt-claim": "attempt_claim.json",
        "direct-producer": "direct_state_producer_manifest.json",
        "direct": "direct_state_manifest.json",
        "development-producer": "development_producer_manifest.json",
        "development": "development_manifest.json",
    }
    if kind == "stage-start":
        if role is None:
            raise ValueError("stage-start requires --role")
        return root / ROLE_START_FILES[role]
    if kind == "stage-complete":
        if role is None:
            raise ValueError("stage-complete requires --role")
        return root / ROLE_COMPLETION_FILES[role]
    if kind == "stage-accounting":
        if role is None:
            raise ValueError("stage-accounting requires --role")
        return root / ROLE_ACCOUNTING_FILES[role]
    if kind == "terminal-accounting":
        return TERMINAL_ACCOUNTING_PATH
    if kind in names:
        return root / names[kind]
    raise ValueError(f"kind has no fixed output: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=(
            "validate-plan",
            "attempt-start",
            "attempt-claim",
            "stage-start",
            "run-inputs",
            "run-attempt",
            "verify-refits",
            "direct-producer",
            "direct",
            "development-producer",
            "development",
            "stage-complete",
            "stage-accounting",
            "terminal-accounting",
            "verify-terminal-accounting",
            "monitor-anchor",
            "closed-world",
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--scope", choices=("direct", "final"))
    parser.add_argument("--index", type=int, choices=(0, 1, 2))
    parser.add_argument("--arm", choices=(*SCORED_ARMS, DIRECT_STATE_ARM))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    if args.kind == "validate-plan":
        plan = load_committed_plan(require_ready=args.require_ready)
        print(
            "[acw-development-plan] "
            f"ready={int(plan['ready_for_g_commit'])} attempts={len(plan['attempt_table'])}"
        )
        return
    if args.root is None:
        parser.error("--root is required")
    root = args.root.expanduser().absolute().resolve(strict=True)
    if root != ATTEMPT_ROOT:
        raise ValueError("canonical attempt root differs")

    if args.kind == "monitor-anchor":
        _discard_protocol_publish_temp(MONITOR_ANCHOR_PATH)
        payload = build_monitor_anchor_ready_envelope(root)
        if os.path.lexists(MONITOR_ANCHOR_PATH):
            if (
                MONITOR_ANCHOR_PATH.is_symlink()
                or not MONITOR_ANCHOR_PATH.is_file()
                or stat.S_IMODE(MONITOR_ANCHOR_PATH.stat().st_mode) != 0o444
            ):
                raise ValueError("monitor anchor envelope is not immutable")
            recorded, _ = _load_canonical_json(
                MONITOR_ANCHOR_PATH, "monitor anchor envelope"
            )
            if recorded != payload:
                raise ValueError("monitor anchor envelope differs from fresh evidence")
            digest = file_sha256(MONITOR_ANCHOR_PATH)
        else:
            digest = write_exclusive(MONITOR_ANCHOR_PATH, payload)
        print(
            "[acw-development-monitor-anchor] "
            f"job_id={payload['monitor']['job_id']} sha256={digest} "
            f"payload_sha256={payload['payload_sha256']} "
            "external_anchor_required=1 performance_claim_ready=0"
        )
        return

    if args.kind == "run-inputs":
        if args.role is None or args.index is None:
            parser.error("run-inputs requires --role and --index")
        run_inputs(root, args.role, args.index)
        return
    if args.kind == "run-attempt":
        if args.role is None or args.index is None or args.arm is None:
            parser.error("run-attempt requires --role, --index, and --arm")
        run_attempt(root, args.role, args.index, args.arm)
        return
    if args.kind == "closed-world":
        if args.stage is None or args.out is None:
            parser.error("closed-world requires --stage and --out")
        payload = _hash_bound(
            {
                "schema": "r12_acw_development_closed_world_receipt_v1",
                "protocol": "R12-ACW-DEVELOPMENT-CLOSED-WORLD-v1",
                "development_plan": _plan_reference(root),
                "scan": closed_world_scan(
                    root,
                    args.stage,
                    require_frozen_directories=args.stage == "final",
                ),
                "confirmation_authorized": False,
            }
        )
        digest = write_exclusive(args.out.expanduser().absolute(), payload)
        print(f"[acw-development-closed-world] stage={args.stage} sha256={digest}")
        return
    if args.kind == "verify-refits":
        if args.scope is None or args.out is None:
            parser.error("verify-refits requires --scope and --out")
        payload = verify_refits(root, args.scope)
        digest = write_exclusive(args.out.expanduser().absolute(), payload)
        print(f"[acw-development-refits] scope={args.scope} sha256={digest}")
        return
    if args.kind == "terminal-accounting":
        final_accounting_path = root / ROLE_ACCOUNTING_FILES[ROLE_FINAL_VERIFIER]
        _discard_protocol_publish_temp(final_accounting_path)
        if os.path.lexists(final_accounting_path):
            _reference(final_accounting_path, root)
            _load_canonical_json(
                final_accounting_path, "phase2 verifier terminal accounting"
            )
            accounting_sha256 = file_sha256(final_accounting_path)
        else:
            final_accounting = build_stage_accounting(root, ROLE_FINAL_VERIFIER)
            accounting_sha256 = write_exclusive(final_accounting_path, final_accounting)
        baseline = publish_development_baseline_after_terminal_validation(root)
        _discard_protocol_publish_temp(TERMINAL_ACCOUNTING_PATH)
        if os.path.lexists(TERMINAL_ACCOUNTING_PATH):
            payload = validate_terminal_accounting(root)
            digest = file_sha256(TERMINAL_ACCOUNTING_PATH)
        else:
            payload = build_terminal_accounting(root)
            digest = write_exclusive(TERMINAL_ACCOUNTING_PATH, payload)
        print(
            "[acw-development-terminal] "
            f"final_accounting_sha256={accounting_sha256} "
            f"baseline_sha256={baseline['baseline_sha256']} "
            f"checkpoint_sha256={baseline['checkpoint_sha256']} "
            f"sha256={digest} performance_claim_ready=0"
        )
        return
    if args.kind == "verify-terminal-accounting":
        payload = validate_terminal_accounting(root)
        print(
            "[acw-development-terminal-verify] "
            f"sha256={file_sha256(TERMINAL_ACCOUNTING_PATH)} "
            f"payload_sha256={payload['payload_sha256']} "
            "external_anchor_required=1 performance_claim_ready=0"
        )
        return

    if args.kind == "attempt-start":
        payload = build_attempt_start(root)
    elif args.kind == "attempt-claim":
        payload = build_attempt_claim(root)
    elif args.kind == "stage-start":
        if args.role is None:
            parser.error("stage-start requires --role")
        payload = build_stage_start(root, args.role)
    elif args.kind == "direct-producer":
        payload = build_direct_producer_manifest(root)
    elif args.kind == "direct":
        payload = build_direct_manifest(root)
    elif args.kind == "development-producer":
        payload = build_development_producer_manifest(root)
    elif args.kind == "development":
        payload = build_development_manifest(root)
    elif args.kind == "stage-accounting":
        if args.role is None:
            parser.error("stage-accounting requires --role")
        payload = build_stage_accounting(root, args.role)
    else:
        if args.stage is None or args.role is None:
            parser.error("stage-complete requires --stage and --role")
        if STAGE_ROLES[args.stage] != args.role:
            raise ValueError("stage and role differ")
        fixed = root / ROLE_COMPLETION_FILES[args.role]
        if args.out is not None and args.out.expanduser().absolute() != fixed:
            raise ValueError("stage completion output differs from fixed path")
        payload = build_stage_completion(root, args.stage, fixed)

    fixed_output = _output_for_kind(root, args.kind, args.role)
    if args.out is not None and args.out.expanduser().absolute() != fixed_output:
        raise ValueError("artifact output differs from its fixed path")
    digest = write_exclusive(fixed_output, payload)
    print(f"[acw-development-manifest] kind={args.kind} sha256={digest}")


if __name__ == "__main__":
    main()
