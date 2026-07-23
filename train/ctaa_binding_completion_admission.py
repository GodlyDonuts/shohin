"""Immutable admission contract for the five-seed A4 completion experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


SCHEMA = "r12_ctaa_a4_binding_completion_admission_v1"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
PROTOCOL_SOURCE_PATHS = (
    "pipeline/build_ctaa_binding_completion_board.py",
    "pipeline/ctaa_binding_identification.py",
    "pipeline/ctaa_board_v2.py",
    "pipeline/ctaa_name_pool.py",
    "pipeline/generate_ctaa_board.py",
    "train/assess_ctaa_binding_completion.py",
    "train/capacity_ctaa_binding_completion.py",
    "train/ctaa_artifact_loader.py",
    "train/ctaa_binding_completion.py",
    "train/ctaa_binding_completion_admission.py",
    "train/ctaa_compiler_training.py",
    "train/ctaa_neural_core.py",
    "train/ctaa_trunk_compiler.py",
    "train/finalize_ctaa_binding_completion.py",
    "train/freeze_ctaa_binding_completion_seeds.py",
    "train/model.py",
    "train/profile_ctaa_binding_completion_resources.py",
    "train/predict_ctaa_binding_completion.py",
    "train/train_ctaa_binding_completion.py",
)
KEYS = {
    "schema",
    "code_commit",
    "protocol_source_sha256",
    "custody_root",
    "base_sha256",
    "qualified_compiler_sha256",
    "tokenizer_sha256",
    "board_manifest_sha256",
    "train_even_sha256",
    "confirmation_source_sha256",
    "confirmation_oracle_sha256",
    "seeds",
    "seed_artifact_names",
    "prediction_artifact_name",
    "assessment_artifact_name",
    "capacity_artifact_name",
    "seed_freeze_manifest_name",
    "oracle_access_ledger_name",
    "decision_artifact_name",
    "resource_artifact_name",
    "qualifier_updates",
    "readout_updates",
    "capacity_updates",
    "batch_size",
    "learning_rate",
    "minimum_train_exact",
    "minimum_confirmation_factorized_exact",
    "minimum_factorized_advantage",
    "maximum_single_slot_exact",
    "minimum_chimera_exact",
    "minimum_seed_passes",
    "maximum_resource_relative_gap",
}
HASH_KEYS = {
    "protocol_source_sha256",
    "base_sha256",
    "qualified_compiler_sha256",
    "tokenizer_sha256",
    "board_manifest_sha256",
    "train_even_sha256",
    "confirmation_source_sha256",
    "confirmation_oracle_sha256",
}
NAME_KEYS = {
    "prediction_artifact_name",
    "assessment_artifact_name",
    "capacity_artifact_name",
    "seed_freeze_manifest_name",
    "oracle_access_ledger_name",
    "decision_artifact_name",
    "resource_artifact_name",
}
INTEGER_LIMITS = {
    "qualifier_updates": (1, 10_000_000),
    "readout_updates": (1, 10_000_000),
    "capacity_updates": (1, 10_000_000),
    "batch_size": (1, 1_000_000),
    "minimum_seed_passes": (1, 5),
}
FLOAT_LIMITS = {
    "learning_rate": (0.0, 1.0),
    "minimum_train_exact": (0.0, 1.0),
    "minimum_confirmation_factorized_exact": (0.0, 1.0),
    "minimum_factorized_advantage": (0.0, 1.0),
    "maximum_single_slot_exact": (0.0, 1.0),
    "minimum_chimera_exact": (0.0, 1.0),
    "maximum_resource_relative_gap": (0.0, 1.0),
}
CANONICAL_DECISION_VALUES = {
    "minimum_train_exact": 0.99,
    "minimum_confirmation_factorized_exact": 0.75,
    "minimum_factorized_advantage": 0.10,
    "maximum_single_slot_exact": 0.10,
    "minimum_chimera_exact": 0.75,
    "minimum_seed_passes": 5,
    "maximum_resource_relative_gap": 0.05,
}


def _artifact_name(value: object) -> str:
    if type(value) is not str:
        raise ValueError("CTAA completion admission artifact name differs")
    name = value
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("CTAA completion admission artifact name differs")
    return name


def _exact_integer(value: object, key: str) -> int:
    if type(value) is not int:
        raise ValueError(f"CTAA completion admission integer differs: {key}")
    lower, upper = INTEGER_LIMITS[key]
    if not lower <= value <= upper:
        raise ValueError(f"CTAA completion admission integer differs: {key}")
    return value


def _exact_float(value: object, key: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"CTAA completion admission float differs: {key}")
    lower, upper = FLOAT_LIMITS[key]
    lower_ok = value > lower if key in {
        "learning_rate",
        "minimum_train_exact",
        "minimum_confirmation_factorized_exact",
        "minimum_chimera_exact",
    } else value >= lower
    if not lower_ok or value > upper:
        raise ValueError(f"CTAA completion admission float differs: {key}")
    return value


def protocol_source_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in PROTOCOL_SOURCE_PATHS:
        path = repo_root / relative
        encoded = path.read_bytes()
        digest.update(relative.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def require_admitted_protocol_source(
    admission: dict[str, object],
    repo_root: Path | None = None,
) -> None:
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root.resolve()
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    if commit != admission["code_commit"]:
        raise ValueError("CTAA completion admission code commit differs")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *PROTOCOL_SOURCE_PATHS],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *PROTOCOL_SOURCE_PATHS],
        cwd=root,
        check=False,
    )
    if dirty.returncode != 0:
        raise ValueError("CTAA completion admitted protocol source is dirty")
    if protocol_source_sha256(root) != admission["protocol_source_sha256"]:
        raise ValueError("CTAA completion admitted protocol source differs")


def require_admitted_artifact_path(
    path: Path,
    admission: dict[str, object],
    name_key: str,
) -> None:
    expected_root = Path(str(admission["custody_root"]))
    if path.resolve().parent != expected_root:
        raise ValueError("CTAA completion artifact custody root differs")
    if path.name != admission[name_key]:
        raise ValueError("CTAA completion artifact identity differs")


def validate_admission(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("CTAA completion admission schema keys differ")
    if value["schema"] != SCHEMA:
        raise ValueError("CTAA completion admission schema differs")
    if (
        type(value["code_commit"]) is not str
        or not COMMIT_PATTERN.fullmatch(value["code_commit"])
    ):
        raise ValueError("CTAA completion admission code commit differs")
    for key in HASH_KEYS:
        if type(value[key]) is not str or not HASH_PATTERN.fullmatch(value[key]):
            raise ValueError(f"CTAA completion admission hash differs: {key}")
    custody_root_value = value["custody_root"]
    if type(custody_root_value) is not str:
        raise ValueError("CTAA completion admission custody root differs")
    custody_root = Path(custody_root_value)
    if not custody_root.is_absolute() or custody_root != custody_root.resolve():
        raise ValueError("CTAA completion admission custody root differs")
    if not isinstance(value["seeds"], list):
        raise ValueError("CTAA completion admission seeds differ")
    seeds = tuple(value["seeds"])
    if any(
        type(seed) is not int or not 0 <= seed < 2**63
        for seed in seeds
    ):
        raise ValueError("CTAA completion admission seeds differ")
    if (
        len(seeds) != 5
        or len(set(seeds)) != 5
        or tuple(sorted(seeds)) != seeds
    ):
        raise ValueError("CTAA completion admission seeds differ")
    if not isinstance(value["seed_artifact_names"], list):
        raise ValueError("CTAA completion admission seed artifacts differ")
    seed_names = tuple(
        _artifact_name(name) for name in value["seed_artifact_names"]
    )
    if len(seed_names) != 5 or len(set(seed_names)) != 5:
        raise ValueError("CTAA completion admission seed artifacts differ")
    names = tuple(_artifact_name(value[key]) for key in sorted(NAME_KEYS))
    if len(set((*seed_names, *names))) != len(seed_names) + len(names):
        raise ValueError("CTAA completion admission artifact identities overlap")
    integers = {key: _exact_integer(value[key], key) for key in INTEGER_LIMITS}
    floats = {key: _exact_float(value[key], key) for key in FLOAT_LIMITS}
    for key, expected in CANONICAL_DECISION_VALUES.items():
        observed = integers.get(key, floats.get(key))
        if observed != expected:
            raise ValueError(
                f"CTAA completion canonical decision value differs: {key}"
            )
    return {
        **value,
        **integers,
        **floats,
        "custody_root": str(custody_root),
        "seeds": seeds,
        "seed_artifact_names": seed_names,
    }


def load_admission(path: Path) -> dict[str, object]:
    return validate_admission(json.loads(path.read_text(encoding="ascii")))
