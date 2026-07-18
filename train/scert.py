#!/usr/bin/env python3
"""CPU-first mechanics and exact runtime contracts for SCERT.

This file implements tensor packing, effective logits, runtime transitions,
source/runtime validation, and the finite float64 CPU falsifier.  It contains no
GPU authorization and no command that launches or fits on an H100.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import locale
import math
import os
import platform
import stat
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_scert_boards import (  # noqa: E402
    HIDDEN_BOARD_SPECS,
    PROTOCOL_ID,
    THEORY_SHA256,
    ContractError,
    atomic_publish_json,
    canonical_json_bytes,
    independent_toy_board,
    independent_toy_reference,
    sha256_bytes,
    sha256_file,
)


EOS_ID = 0
DUMMY_ID = 0
NEUTRAL_ID = 233
V0_ID = 28
V1_ID = 29
VOCAB_SIZE = 32768
SEQ_LEN = 2048
G_L_END = 70
SOURCE_START = 70
SOURCE_END = 582
STATE_START = 582
STATE_END = 1094
G_R_START = 1094
G_R_END = 1097
PROBE_POSITION = 1096
GENERATION_START = 1097
GENERATION_END = 1609
MAX_SLOT = 512
MODEL_PARAMETERS = 125_081_664
MOTOR_PARAMETERS = 4_634
BOUNDARY_PARAMETERS = 1_154
ADDED_PARAMETERS = MOTOR_PARAMETERS + BOUNDARY_PARAMETERS
TOTAL_PARAMETERS = MODEL_PARAMETERS + ADDED_PARAMETERS
if TOTAL_PARAMETERS >= 150_000_000:
    raise RuntimeError("SCERT parameter ledger exceeds the user limit")

EXPECTED_SOURCE_HASHES = {
    "R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md": THEORY_SHA256,
    "train/model.py": "45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4",
    "train/muon.py": "863e79aaaaebb681382f0c88078390b5683ab39be79ac7df60f26d1c04b21762",
}
PARENT_PATH = ROOT / "train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt"
PARENT_SHA256 = "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
TOKENIZER_PATH = ROOT / "artifacts/shohin-tok-32k.json"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
ADAMW_SOURCE_SHA256 = "54299056b7745c162192132bb6028f3387c05ff4203518ff0240058584968312"

SOURCE_MANIFEST_PATHS = (
    "R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md",
    "R12_SCERT_EXECUTION_PREREG.md",
    "pipeline/generate_scert_boards.py",
    "pipeline/test_generate_scert_boards.py",
    "train/scert.py",
    "train/test_scert.py",
    "train/jobs/scert_newton.sbatch",
    "train/model.py",
    "train/muon.py",
    "train/digitwise_protocol.py",
)

RUNTIME_STATE_FIELDS = (
    "phase",
    "commit_count",
    "candidate_count",
    "epoch_token_count",
    "total_token_count",
    "replay_slot_cursor",
    "generation_slot_cursor",
    "cap_constants",
    "failure_flag",
    "rng_state_and_cursor",
    "deterministic_tie_state",
    "publication_receipt_cursor",
)

STAGE1_OPTIMIZER_BINDING: dict[str, Any] = {
    "updates": 1024,
    "parameter_storage": "fp32",
    "optimizer_state": "fp32",
    "forward_autocast": "bf16",
    "objective_dtype": "fp32",
    "gradient_scaler": False,
    "shadow_master_parameters": False,
    "global_l2_clip": 1.0,
    "missing_or_nonfinite_gradient": "fatal",
    "step_order": ["muon", "adamw"],
    "muon": {
        "tensor_count": 210,
        "scalar_count": 106_168_320,
        "base_lr": 0.001,
        "momentum": 0.95,
        "nesterov": True,
        "newton_schulz_steps": 5,
        "newton_schulz_dtype": "bf16",
        "coefficients": [3.4445, -4.7750, 2.0315],
        "normalization_epsilon": 1e-7,
        "weight_decay": 0.0,
        "matrix_scale": "sqrt(max(1,rows/cols))",
    },
    "adamw": {
        "tensor_count": 126,
        "scalar_count": 18_917_978,
        "base_lr": 0.0002,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "bias_correction": True,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    },
    "schedule": {
        "warmup_updates": 50,
        "total_updates": 1024,
        "final_scale": 0.1,
        "formula": "u/50 then 0.1+0.9*0.5*(1+cos(pi*(u-50)/(1024-50)))",
    },
}

STAGE2_OPTIMIZER_BINDING: dict[str, Any] = {
    "base_and_motor_frozen": True,
    "head_parameters": 1_154,
    "optimizer": "AdamW",
    "lr": 0.01,
    "betas": [0.9, 0.95],
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "batch_size": 512,
    "epochs": 10,
    "updates": 200,
    "warmup": 0,
    "gradient_clip": 1.0,
    "shuffled_labels": ["HALT", "COMMIT", "COMMIT", "COMMIT", "COMMIT"],
}


def _file_entry(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ContractError(f"manifest source is aliased or non-regular: {path}")
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "mode": stat.S_IMODE(info.st_mode),
        "bytes": info.st_size,
        "sha256": sha256_file(path),
    }


def build_source_manifest() -> dict[str, Any]:
    entries = [_file_entry(ROOT / relative) for relative in SOURCE_MANIFEST_PATHS]
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        actual = next(entry["sha256"] for entry in entries if entry["path"] == relative)
        if actual != expected:
            raise ContractError(f"frozen theory source changed: {relative}")
    adamw_source = Path(inspect.getsourcefile(torch.optim.AdamW) or "")
    if not adamw_source.is_file() or sha256_file(adamw_source) != ADAMW_SOURCE_SHA256:
        raise ContractError("PyTorch AdamW source identity changed")
    immutable_inputs = [_file_entry(TOKENIZER_PATH), _file_entry(PARENT_PATH)]
    expected_inputs = {
        "artifacts/shohin-tok-32k.json": TOKENIZER_SHA256,
        "train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt": PARENT_SHA256,
    }
    if {
        entry["path"]: entry["sha256"] for entry in immutable_inputs
    } != expected_inputs:
        raise ContractError("frozen tokenizer or parent checkpoint changed")
    return {
        "schema": "r12-scert-source-manifest-v1",
        "protocol": PROTOCOL_ID,
        "entries": entries,
        "adamw_source": {
            "path": str(adamw_source.resolve()),
            "bytes": adamw_source.stat().st_size,
            "sha256": sha256_file(adamw_source),
        },
        "immutable_inputs": immutable_inputs,
        "entry_order": list(SOURCE_MANIFEST_PATHS),
        "claim_boundary": "Source identity only; not review, execution authority, or capability evidence.",
    }


def validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    expected = build_source_manifest()
    if dict(manifest) != expected:
        raise ContractError("source manifest differs from exact current bytes")


def _module_receipt(module: Any) -> dict[str, Any]:
    path = Path(str(module.__file__)).resolve()
    if not path.is_file():
        raise ContractError(f"runtime module has no regular source: {module.__name__}")
    return {
        "name": module.__name__,
        "version": str(getattr(module, "__version__", "none")),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def configure_cpu_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_runtime_manifest() -> dict[str, Any]:
    import cryptography
    import tokenizers

    configure_cpu_determinism()
    executable = Path(sys.executable).resolve()
    allowed_environment = {
        name: os.environ.get(name, "<unset>")
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "LANG",
            "LC_ALL",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONHASHSEED",
        )
    }
    return {
        "schema": "r12-scert-cpu-runtime-manifest-v1",
        "protocol": PROTOCOL_ID,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "locale": locale.setlocale(locale.LC_ALL, None),
        },
        "python_executable": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "modules": [
            _module_receipt(torch),
            _module_receipt(tokenizers),
            _module_receipt(cryptography),
        ],
        "torch": {
            "version": str(torch.__version__),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "attention_backend": {
            "cpu_reference": "explicit-float64-softmax-v1",
            "future_h100": "math-sdpa-only-pending-independent-bit-identity",
            "lowest_token_id_tie": True,
        },
        "environment": allowed_environment,
        "secrets_recorded": False,
        "h100_authorized": False,
    }


def validate_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    expected = build_runtime_manifest()
    if dict(manifest) != expected:
        raise ContractError(
            "runtime manifest differs from the live deterministic runtime"
        )


class CarryMotor(nn.Module):
    """Rank-8 motor that can alter only the frozen token-0/token-1 coordinates."""

    def __init__(self, d_model: int = 576, rank: int = 8):
        super().__init__()
        if d_model != 576 or rank != 8:
            raise ContractError("SCERT motor dimensions are frozen at 576->8->2")
        self.down = nn.Linear(d_model, rank, bias=True)
        self.up = nn.Linear(rank, 2, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden.float())))


class BoundaryHead(nn.Module):
    """Two-way [HALT, COMMIT] affine head; torch.argmax resolves ties to HALT."""

    def __init__(self, d_model: int = 576):
        super().__init__()
        if d_model != 576:
            raise ContractError("SCERT boundary-head width is frozen at 576")
        self.affine = nn.Linear(d_model, 2, bias=True)
        self.forward_count = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        return self.affine(hidden.float())


def added_parameter_manifest() -> dict[str, Any]:
    motor = CarryMotor()
    boundary = BoundaryHead()
    motor_count = sum(parameter.numel() for parameter in motor.parameters())
    boundary_count = sum(parameter.numel() for parameter in boundary.parameters())
    if motor_count != MOTOR_PARAMETERS or boundary_count != BOUNDARY_PARAMETERS:
        raise ContractError("SCERT added-parameter count changed")
    return {
        "parent_unique_parameters": MODEL_PARAMETERS,
        "motor_unique_parameters": motor_count,
        "boundary_head_unique_parameters": boundary_count,
        "added_unique_parameters": motor_count + boundary_count,
        "deployment_unique_parameters": MODEL_PARAMETERS + motor_count + boundary_count,
        "strictly_below_150m": TOTAL_PARAMETERS < 150_000_000,
    }


def argmax_lowest_id(logits: torch.Tensor) -> torch.Tensor:
    """Return the lowest token ID among exact maxima."""
    maximum = logits.max(dim=-1, keepdim=True).values
    winners = logits.eq(maximum)
    return winners.to(torch.int64).argmax(dim=-1)


@dataclass(frozen=True)
class EffectiveLogits:
    ell_base: torch.Tensor
    motor_delta: torch.Tensor
    ell_eff: torch.Tensor
    token: torch.Tensor
    event: torch.Tensor


def effective_logits(
    ell_base: torch.Tensor,
    hidden: torch.Tensor,
    motor: nn.Module,
    a_m: int,
    *,
    v0: int = V0_ID,
    v1: int = V1_ID,
    eos_id: int = EOS_ID,
) -> EffectiveLogits:
    if a_m not in (0, 1):
        raise ContractError("motor level must be exactly zero or one")
    if v0 == v1 or eos_id in (v0, v1):
        raise ContractError("effective-logit token identities are invalid")
    if ell_base.shape[-1] != VOCAB_SIZE or hidden.shape[-1] != 576:
        raise ContractError("effective-logit vocabulary or hidden width changed")
    if ell_base.shape[:-1] != hidden.shape[:-1]:
        raise ContractError("hidden and base-logit surfaces differ")
    delta_fp32 = motor(hidden)
    if delta_fp32.shape != (*ell_base.shape[:-1], 2):
        raise ContractError("motor must emit exactly two deltas")
    delta = delta_fp32.to(device=ell_base.device, dtype=ell_base.dtype)
    ell_eff = ell_base.clone()
    ell_eff[..., v0] = ell_eff[..., v0] + a_m * delta[..., 0]
    ell_eff[..., v1] = ell_eff[..., v1] + a_m * delta[..., 1]
    token = argmax_lowest_id(ell_eff)
    return EffectiveLogits(
        ell_base=ell_base,
        motor_delta=delta,
        ell_eff=ell_eff,
        token=token,
        event=token.eq(eos_id),
    )


def validate_effective_receipt(surface: EffectiveLogits) -> int:
    if not isinstance(surface, EffectiveLogits):
        raise ContractError("runtime requires an EffectiveLogits receipt")
    if (
        surface.ell_base.shape != surface.ell_eff.shape
        or surface.ell_base.shape[-1] != VOCAB_SIZE
        or surface.motor_delta.shape != (*surface.ell_base.shape[:-1], 2)
        or surface.token.numel() != 1
        or surface.event.numel() != 1
    ):
        raise ContractError("effective-logit receipt shape differs")
    expected_token = argmax_lowest_id(surface.ell_eff)
    if not torch.equal(surface.token, expected_token) or not torch.equal(
        surface.event, expected_token.eq(EOS_ID)
    ):
        raise ContractError("effective-logit receipt event or token differs")
    unchanged = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=surface.ell_eff.device)
    unchanged[V0_ID] = False
    unchanged[V1_ID] = False
    if not torch.equal(
        surface.ell_base[..., unchanged], surface.ell_eff[..., unchanged]
    ):
        raise ContractError("effective-logit receipt changed an undeclared coordinate")
    return int(expected_token.item())


def effective_lm_loss(
    surface: EffectiveLogits, labels: torch.Tensor, zloss_weight: float = 1e-4
) -> tuple[torch.Tensor, int]:
    if not isinstance(surface, EffectiveLogits):
        raise ContractError(
            "effective objective requires the post-motor surface receipt"
        )
    if float(zloss_weight) != 1e-4:
        raise ContractError("effective objective z-loss weight changed")
    ell_eff = surface.ell_eff
    if ell_eff.shape[:-1] != labels.shape:
        raise ContractError("effective logits and labels differ")
    supervised = labels.ne(-100)
    count = int(supervised.sum().item())
    if count <= 0:
        raise ContractError("effective objective has no supervised positions")
    selected_logits = ell_eff.float()[supervised]
    selected_labels = labels[supervised]
    ce_sum = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
    z_sum = torch.logsumexp(selected_logits, dim=-1).pow(2).sum()
    return (ce_sum + float(zloss_weight) * z_sum) / count, count


@dataclass(frozen=True)
class CapConstants:
    max_commits: int = 8
    max_candidates: int = 9
    max_epoch_tokens: int = 512


@dataclass(frozen=True)
class RuntimeState:
    phase: str = "ACTIVE"
    commit_count: int = 0
    candidate_count: int = 0
    epoch_token_count: int = 0
    total_token_count: int = 0
    replay_slot_cursor: int = 0
    generation_slot_cursor: int = GENERATION_START
    cap_constants: CapConstants = CapConstants()
    failure_flag: bool = False
    rng_state_and_cursor: tuple[int, int] = (0, 0)
    deterministic_tie_state: str = "lowest-token-id"
    publication_receipt_cursor: int = 0

    def to_mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["rng_state_and_cursor"] = list(self.rng_state_and_cursor)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeState":
        if tuple(value) != RUNTIME_STATE_FIELDS and set(value) != set(
            RUNTIME_STATE_FIELDS
        ):
            raise ContractError("runtime state field set differs")
        caps = value["cap_constants"]
        if set(caps) != {"max_commits", "max_candidates", "max_epoch_tokens"}:
            raise ContractError("runtime cap field set differs")
        result = cls(
            phase=str(value["phase"]),
            commit_count=int(value["commit_count"]),
            candidate_count=int(value["candidate_count"]),
            epoch_token_count=int(value["epoch_token_count"]),
            total_token_count=int(value["total_token_count"]),
            replay_slot_cursor=int(value["replay_slot_cursor"]),
            generation_slot_cursor=int(value["generation_slot_cursor"]),
            cap_constants=CapConstants(
                **{key: int(item) for key, item in caps.items()}
            ),
            failure_flag=bool(value["failure_flag"]),
            rng_state_and_cursor=tuple(
                int(item) for item in value["rng_state_and_cursor"]
            ),
            deterministic_tie_state=str(value["deterministic_tie_state"]),
            publication_receipt_cursor=int(value["publication_receipt_cursor"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.phase not in ("ACTIVE", "HALTED"):
            raise ContractError("invalid runtime phase")
        if self.deterministic_tie_state != "lowest-token-id":
            raise ContractError("runtime tie state changed")
        if len(self.rng_state_and_cursor) != 2:
            raise ContractError("runtime RNG state is incomplete")
        if self.cap_constants != CapConstants():
            raise ContractError("runtime cap constants changed")
        integers = (
            self.commit_count,
            self.candidate_count,
            self.epoch_token_count,
            self.total_token_count,
            self.replay_slot_cursor,
            self.generation_slot_cursor,
            self.publication_receipt_cursor,
        )
        if any(value < 0 for value in integers):
            raise ContractError("negative runtime counter")
        if self.commit_count > self.cap_constants.max_commits:
            raise ContractError("commit count exceeds cap")
        if self.candidate_count > self.cap_constants.max_candidates:
            raise ContractError("candidate count exceeds cap")
        if self.epoch_token_count > self.cap_constants.max_epoch_tokens:
            raise ContractError("epoch token count exceeds cap")
        if self.candidate_count < self.commit_count:
            raise ContractError("candidate count precedes commit count")
        if self.publication_receipt_cursor != self.candidate_count:
            raise ContractError("publication receipt cursor differs from candidates")
        if (
            self.rng_state_and_cursor[1]
            != self.total_token_count + self.candidate_count
        ):
            raise ContractError("runtime RNG cursor differs from consumed decisions")
        if self.generation_slot_cursor != GENERATION_START + self.epoch_token_count:
            raise ContractError("generation cursor differs from epoch token count")
        if self.replay_slot_cursor != 0:
            raise ContractError("replay cursor is nonzero at a dispatch boundary")
        if self.total_token_count < self.epoch_token_count + self.commit_count:
            raise ContractError(
                "total token count cannot contain prior committed epochs"
            )
        if self.phase == "HALTED" and self.candidate_count == 0:
            raise ContractError("HALTED state has no accepted EOS candidate")


def initial_runtime_state(rng_seed: int = 0) -> RuntimeState:
    state = RuntimeState(rng_state_and_cursor=(int(rng_seed), 0))
    state.validate()
    return state


def consume_non_eos(state: RuntimeState, surface: EffectiveLogits) -> RuntimeState:
    state.validate()
    if state.phase != "ACTIVE" or state.failure_flag:
        raise ContractError("tokens cannot be consumed outside active execution")
    token_id = validate_effective_receipt(surface)
    if token_id == EOS_ID:
        raise ContractError("EOS must use the event transition")
    if state.epoch_token_count >= state.cap_constants.max_epoch_tokens:
        return replace(state, failure_flag=True)
    result = replace(
        state,
        epoch_token_count=state.epoch_token_count + 1,
        total_token_count=state.total_token_count + 1,
        generation_slot_cursor=state.generation_slot_cursor + 1,
        rng_state_and_cursor=(
            state.rng_state_and_cursor[0],
            state.rng_state_and_cursor[1] + 1,
        ),
    )
    result.validate()
    return result


def consume_event(
    state: RuntimeState,
    surface: EffectiveLogits,
    action: str,
    authored_span_length: int,
) -> RuntimeState:
    state.validate()
    if state.phase != "ACTIVE" or state.failure_flag:
        raise ContractError("events cannot be consumed outside active execution")
    if action not in ("COMMIT", "HALT"):
        raise ContractError("invalid boundary action")
    if validate_effective_receipt(surface) != EOS_ID:
        raise ContractError("only post-motor effective EOS can create an event")
    if (
        authored_span_length != state.epoch_token_count
        or authored_span_length > MAX_SLOT
    ):
        raise ContractError("authored-span receipt differs from runtime state")
    candidate_count = state.candidate_count + 1
    if candidate_count > state.cap_constants.max_candidates:
        return replace(state, failure_flag=True)
    common = {
        "candidate_count": candidate_count,
        "publication_receipt_cursor": state.publication_receipt_cursor + 1,
        "rng_state_and_cursor": (
            state.rng_state_and_cursor[0],
            state.rng_state_and_cursor[1] + 1,
        ),
    }
    if action == "COMMIT":
        if (
            not authored_span_length
            or state.commit_count >= state.cap_constants.max_commits
        ):
            return replace(state, failure_flag=True, **common)
        result = replace(
            state,
            commit_count=state.commit_count + 1,
            epoch_token_count=0,
            replay_slot_cursor=0,
            generation_slot_cursor=GENERATION_START,
            **common,
        )
    else:
        result = replace(state, phase="HALTED", **common)
    result.validate()
    return result


@dataclass(frozen=True)
class PackedSurface:
    ids: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    attention: torch.Tensor
    labels: torch.Tensor | None
    keep_indices: torch.Tensor
    supervised_count: int
    mode: str


@dataclass(frozen=True)
class Stage1Update:
    ids: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    attention: torch.Tensor
    labels: torch.Tensor
    supervised_count: int

    @property
    def flat_ids(self) -> torch.Tensor:
        return self.ids.reshape(10, SEQ_LEN)

    @property
    def flat_valid(self) -> torch.Tensor:
        return self.valid.reshape(10, SEQ_LEN)

    @property
    def flat_positions(self) -> torch.Tensor:
        return self.positions.reshape(10, SEQ_LEN)

    @property
    def flat_attention(self) -> torch.Tensor:
        return self.attention.reshape(10, SEQ_LEN, SEQ_LEN)

    @property
    def flat_labels(self) -> torch.Tensor:
        return self.labels.reshape(10, SEQ_LEN)


def _base_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.full((SEQ_LEN,), DUMMY_ID, dtype=torch.long)
    valid = torch.zeros((SEQ_LEN,), dtype=torch.bool)
    positions = torch.arange(SEQ_LEN, dtype=torch.long)
    return ids, valid, positions


def _put(
    ids: torch.Tensor, valid: torch.Tensor, start: int, values: Sequence[int]
) -> None:
    if start < 0 or start + len(values) > ids.numel():
        raise ContractError("surface slot overflow")
    if any(not 0 <= int(value) < VOCAB_SIZE for value in values):
        raise ContractError("token ID is outside the frozen vocabulary")
    if values:
        ids[start : start + len(values)] = torch.tensor(values, dtype=torch.long)
        valid[start : start + len(values)] = True


def _attention_from_regions(
    valid: torch.Tensor,
    *,
    p_range: tuple[int, int],
    x_range: tuple[int, int],
    g_r_range: tuple[int, int],
    target_range: tuple[int, int] | None,
    clean: bool,
) -> torch.Tensor:
    length = valid.numel()
    q = torch.arange(length)[:, None]
    k = torch.arange(length)[None, :]
    causal = k <= q
    allowed = torch.zeros((length, length), dtype=torch.bool)
    gl_q = (q >= 0) & (q < G_L_END)
    gl_k = (k >= 0) & (k < G_L_END)
    p_q = (q >= p_range[0]) & (q < p_range[1])
    p_k = (k >= p_range[0]) & (k < p_range[1])
    x_q = (q >= x_range[0]) & (q < x_range[1])
    x_k = (k >= x_range[0]) & (k < x_range[1])
    gr_q = (q >= g_r_range[0]) & (q < g_r_range[1])
    gr_k = (k >= g_r_range[0]) & (k < g_r_range[1])
    allowed |= gl_q & gl_k & causal
    allowed |= p_q & (gl_k | p_k) & causal
    allowed |= x_q & (gl_k | x_k | (p_k if not clean else False)) & causal
    allowed |= gr_q & (gl_k | x_k | gr_k) & causal
    if target_range is not None:
        target_q = (q >= target_range[0]) & (q < target_range[1])
        target_k = (k >= target_range[0]) & (k < target_range[1])
        allowed |= target_q & (gl_k | x_k | gr_k | target_k) & causal
    allowed &= valid[:, None] & valid[None, :]
    return allowed


def build_stage1_lane(
    g_l_ids: Sequence[int],
    p_ids: Sequence[int],
    g_r_ids: Sequence[int],
    x_ids: Sequence[int],
) -> PackedSurface:
    if len(g_l_ids) != 70 or len(g_r_ids) != 3:
        raise ContractError("stage-one scaffold size differs")
    if len(p_ids) > MAX_SLOT or len(x_ids) > MAX_SLOT:
        raise ContractError("stage-one slot overflow")
    ids, valid, positions = _base_tensors()
    _put(ids, valid, 0, g_l_ids)
    _put(ids, valid, STATE_START, p_ids)
    _put(ids, valid, G_R_START, g_r_ids)
    _put(ids, valid, GENERATION_START, x_ids)
    eos_position = GENERATION_START + len(x_ids)
    _put(ids, valid, eos_position, [EOS_ID])
    labels = torch.full((SEQ_LEN,), -100, dtype=torch.long)
    if x_ids:
        labels[PROBE_POSITION] = int(x_ids[0])
        for index in range(len(x_ids) - 1):
            labels[GENERATION_START + index] = int(x_ids[index + 1])
        labels[GENERATION_START + len(x_ids) - 1] = EOS_ID
    else:
        labels[PROBE_POSITION] = EOS_ID
    attention = _attention_from_regions(
        valid,
        p_range=(STATE_START, STATE_END),
        x_range=(STATE_START, STATE_END),
        g_r_range=(G_R_START, G_R_END),
        target_range=(GENERATION_START, eos_position + 1),
        clean=True,
    )
    keep = torch.where(valid)[0]
    count = int(labels.ne(-100).sum().item())
    if count != len(x_ids) + 1:
        raise ContractError("stage-one supervised denominator changed")
    return PackedSurface(
        ids, valid, positions, attention, labels, keep, count, "stage1-clean"
    )


def build_stage1_update(packs: Sequence[Sequence[PackedSurface]]) -> Stage1Update:
    if len(packs) != 2 or any(len(pack) != 5 for pack in packs):
        raise ContractError("stage-one update must contain exactly two five-lane packs")
    rows = [row for pack in packs for row in pack]
    if any(
        row.mode != "stage1-clean"
        or row.labels is None
        or row.ids.shape != (SEQ_LEN,)
        or row.valid.shape != (SEQ_LEN,)
        or row.positions.shape != (SEQ_LEN,)
        or row.attention.shape != (SEQ_LEN, SEQ_LEN)
        for row in rows
    ):
        raise ContractError("stage-one update contains an incompatible row")
    ids = torch.stack([row.ids for row in rows]).reshape(2, 5, SEQ_LEN)
    valid = torch.stack([row.valid for row in rows]).reshape(2, 5, SEQ_LEN)
    positions = torch.stack([row.positions for row in rows]).reshape(2, 5, SEQ_LEN)
    attention = torch.stack([row.attention for row in rows]).reshape(
        2, 5, SEQ_LEN, SEQ_LEN
    )
    labels = torch.stack([row.labels for row in rows]).reshape(2, 5, SEQ_LEN)
    update = Stage1Update(
        ids=ids,
        valid=valid,
        positions=positions,
        attention=attention,
        labels=labels,
        supervised_count=sum(row.supervised_count for row in rows),
    )
    if not torch.equal(update.flat_ids, torch.stack([row.ids for row in rows])):
        raise ContractError("stage-one pack flattening reordered lanes")
    return update


def build_replay_surface(
    g_l_ids: Sequence[int],
    p_ids: Sequence[int],
    x_ids: Sequence[int],
    g_r_ids: Sequence[int],
    *,
    clean: bool,
) -> PackedSurface:
    if len(g_l_ids) != 70 or len(g_r_ids) != 3:
        raise ContractError("replay scaffold size differs")
    if len(p_ids) > MAX_SLOT or len(x_ids) > MAX_SLOT:
        raise ContractError("replay slot overflow")
    ids, valid, positions = _base_tensors()
    _put(ids, valid, 0, g_l_ids)
    _put(ids, valid, SOURCE_START, p_ids)
    _put(ids, valid, STATE_START, x_ids)
    _put(ids, valid, G_R_START, g_r_ids)
    attention = _attention_from_regions(
        valid,
        p_range=(SOURCE_START, SOURCE_END),
        x_range=(STATE_START, STATE_END),
        g_r_range=(G_R_START, G_R_END),
        target_range=None,
        clean=clean,
    )
    keep_mask = torch.zeros_like(valid)
    keep_mask[:G_L_END] = valid[:G_L_END]
    keep_mask[STATE_START:STATE_END] = valid[STATE_START:STATE_END]
    keep_mask[G_R_START:G_R_END] = valid[G_R_START:G_R_END]
    return PackedSurface(
        ids,
        valid,
        positions,
        attention,
        None,
        torch.where(keep_mask)[0],
        0,
        "replay-clean" if clean else "replay-stale",
    )


def build_stage2_replay(
    g_l_ids: Sequence[int],
    p_ids: Sequence[int],
    x_ids: Sequence[int],
    g_r_ids: Sequence[int],
) -> PackedSurface:
    return build_replay_surface(g_l_ids, p_ids, x_ids, g_r_ids, clean=True)


def expected_reconstruction_mask_xor(
    surface_clean: PackedSurface, surface_stale: PackedSurface
) -> torch.Tensor:
    if not torch.equal(surface_clean.ids, surface_stale.ids):
        raise ContractError("mechanistic surfaces have different token IDs")
    if not torch.equal(surface_clean.valid, surface_stale.valid):
        raise ContractError("mechanistic surfaces have different validity")
    if not torch.equal(surface_clean.positions, surface_stale.positions):
        raise ContractError("mechanistic surfaces have different positions")
    expected = torch.zeros_like(surface_clean.attention)
    x_queries = surface_clean.valid.clone()
    x_queries[:STATE_START] = False
    x_queries[STATE_END:] = False
    p_keys = surface_clean.valid.clone()
    p_keys[:SOURCE_START] = False
    p_keys[SOURCE_END:] = False
    expected |= x_queries[:, None] & p_keys[None, :]
    actual = surface_clean.attention ^ surface_stale.attention
    if not torch.equal(actual, expected):
        raise ContractError("reconstruction-mask XOR contains undeclared edges")
    return actual


def stage1_schedule(update: int) -> tuple[float, float, float]:
    if update < 1 or update > 1024:
        raise ContractError("stage-one update index is outside 1..1024")
    if update <= 50:
        scale = update / 50.0
    else:
        scale = 0.1 + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * (update - 50) / (1024 - 50))
        )
    return scale, 0.001 * scale, 0.0002 * scale


def expected_optimizer_names() -> tuple[
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]
]:
    muon: dict[str, tuple[int, ...]] = {}
    adam: dict[str, tuple[int, ...]] = {"tok.weight": (32768, 576), "norm.w": (576,)}
    for layer in range(30):
        prefix = f"blocks.{layer}"
        muon.update(
            {
                f"{prefix}.attn.q.weight": (576, 576),
                f"{prefix}.attn.k.weight": (192, 576),
                f"{prefix}.attn.v.weight": (192, 576),
                f"{prefix}.attn.o.weight": (576, 576),
                f"{prefix}.mlp.gate.weight": (1536, 576),
                f"{prefix}.mlp.up.weight": (1536, 576),
                f"{prefix}.mlp.down.weight": (576, 1536),
            }
        )
        adam.update(
            {
                f"{prefix}.n1.w": (576,),
                f"{prefix}.n2.w": (576,),
                f"{prefix}.attn.qn.w": (64,),
                f"{prefix}.attn.kn.w": (64,),
            }
        )
    adam.update(
        {
            "motor.down.weight": (8, 576),
            "motor.down.bias": (8,),
            "motor.up.weight": (2, 8),
            "motor.up.bias": (2,),
        }
    )
    if len(muon) != 210 or len(adam) != 126:
        raise ContractError("optimizer identity cardinality changed")
    return muon, adam


def validate_parent_and_optimizer_manifest(path: Path = PARENT_PATH) -> dict[str, Any]:
    if sha256_file(path) != PARENT_SHA256:
        raise ContractError("parent checkpoint hash changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(checkpoint) != {"model", "cfg", "step"} or checkpoint["step"] != "sft_ep1":
        raise ContractError("parent checkpoint schema changed")
    expected_cfg = {
        "vocab_size": 32768,
        "n_layer": 30,
        "n_head": 9,
        "n_kv_head": 3,
        "d_model": 576,
        "d_ff": 1536,
        "seq_len": 2048,
        "rope_theta": 50000.0,
        "qk_norm": True,
        "tie_embeddings": True,
        "zloss": 0.0001,
        "n_loop": 1,
    }
    if checkpoint["cfg"] != expected_cfg:
        raise ContractError("parent model configuration changed")
    state = checkpoint["model"]
    muon, adam = expected_optimizer_names()
    expected_base = {key: shape for key, shape in muon.items()}
    expected_base.update(
        {key: shape for key, shape in adam.items() if not key.startswith("motor.")}
    )
    expected_base["head.weight"] = (32768, 576)
    if set(state) != set(expected_base):
        raise ContractError("parent state-dict identity set changed")
    for name, shape in expected_base.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ContractError(f"parent tensor differs: {name}")
    if state["tok.weight"].data_ptr() != state["head.weight"].data_ptr():
        raise ContractError("tied token/head storage alias changed")
    muon_count = sum(math.prod(shape) for shape in muon.values())
    adam_count = sum(math.prod(shape) for shape in adam.values())
    if muon_count != 106_168_320 or adam_count != 18_917_978:
        raise ContractError("optimizer scalar count changed")
    if muon_count + adam_count != 125_086_298:
        raise ContractError("stage-one trainable scalar count changed")
    return {
        "schema": "r12-scert-parent-optimizer-manifest-v1",
        "parent_sha256": PARENT_SHA256,
        "muon_tensors": len(muon),
        "muon_scalars": muon_count,
        "adamw_tensors": len(adam),
        "adamw_scalars": adam_count,
        "stage1_unique_trainable": muon_count + adam_count,
        "tied_head_deduplicated": True,
        "muon_first_adamw_second": True,
        "global_clip": 1.0,
        "updates": 1024,
        "stage1_optimizer_binding": STAGE1_OPTIMIZER_BINDING,
        "stage2_optimizer_binding": STAGE2_OPTIMIZER_BINDING,
        "source_hashes": {
            "model.py": EXPECTED_SOURCE_HASHES["train/model.py"],
            "muon.py": EXPECTED_SOURCE_HASHES["train/muon.py"],
            "adamw.py": ADAMW_SOURCE_SHA256,
        },
    }


@dataclass
class FailureInclusiveLedger:
    expected: int
    observed_ids: set[str] = dataclasses.field(default_factory=set)
    successes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.expected, int) or self.expected <= 0:
            raise ContractError("failure-inclusive denominator must be positive")

    def record(self, case_id: str, success: bool) -> None:
        if not isinstance(case_id, str) or not case_id:
            raise ContractError("denominator case ID is invalid")
        if case_id in self.observed_ids:
            raise ContractError("duplicate denominator case")
        if len(self.observed_ids) >= self.expected:
            raise ContractError("denominator contains extra cases")
        self.observed_ids.add(case_id)
        self.successes += int(bool(success))

    def summary(self) -> dict[str, int]:
        missing = self.expected - len(self.observed_ids)
        if missing < 0:
            raise ContractError("negative missing denominator")
        return {
            "denominator": self.expected,
            "observed": len(self.observed_ids),
            "successes": self.successes,
            "failures": self.expected - self.successes,
            "missing_as_failures": missing,
        }


@dataclass
class ObservationalAdmissionLedger:
    expected_candidates: int = 432
    expected_strata: int = 18
    candidates_per_stratum: int = 24
    minimum_admitted_total: int = 216
    minimum_admitted_per_stratum: int = 12
    records: dict[str, tuple[int, bool]] = dataclasses.field(default_factory=dict)

    def record(self, case_id: str, stratum: int, admitted: bool) -> None:
        if not isinstance(case_id, str) or not case_id or case_id in self.records:
            raise ContractError("observational candidate ID is invalid or duplicated")
        if len(self.records) >= self.expected_candidates:
            raise ContractError("observational denominator contains extra candidates")
        if not 0 <= int(stratum) < self.expected_strata:
            raise ContractError("observational stratum is invalid")
        self.records[case_id] = (int(stratum), bool(admitted))

    def summary(self) -> dict[str, Any]:
        totals = Counter(stratum for stratum, _ in self.records.values())
        admitted = Counter(
            stratum for stratum, accepted in self.records.values() if accepted
        )
        missing = self.expected_candidates - len(self.records)
        malformed_strata = [
            stratum
            for stratum in range(self.expected_strata)
            if totals[stratum] != self.candidates_per_stratum
        ]
        admitted_total = sum(admitted.values())
        return {
            "denominator": self.expected_candidates,
            "observed": len(self.records),
            "missing_as_failures": missing,
            "admitted": admitted_total,
            "per_stratum_admitted": {
                str(stratum): admitted[stratum]
                for stratum in range(self.expected_strata)
            },
            "per_stratum_observed": {
                str(stratum): totals[stratum] for stratum in range(self.expected_strata)
            },
            "gate_passed": (
                missing == 0
                and not malformed_strata
                and admitted_total >= self.minimum_admitted_total
                and all(
                    admitted[stratum] >= self.minimum_admitted_per_stratum
                    for stratum in range(self.expected_strata)
                )
            ),
        }


TARGET_SWITCH_ARMS = ("TS-C1M1", "TS-C0M1", "TS-C1M0", "TS-post-hoc-M1")
TARGET_SWITCH_ARM_SPECS: dict[str, dict[str, Any]] = {
    "TS-C1M1": {
        "reconstruction": "M_clean",
        "motor_level": 1,
        "post_hoc_mask_only": False,
    },
    "TS-C0M1": {
        "reconstruction": "M_stale",
        "motor_level": 1,
        "post_hoc_mask_only": False,
    },
    "TS-C1M0": {
        "reconstruction": "M_clean",
        "motor_level": 0,
        "post_hoc_mask_only": False,
    },
    "TS-post-hoc-M1": {
        "reconstruction": "post-hoc-mask-only",
        "motor_level": 1,
        "post_hoc_mask_only": True,
    },
}


def validate_target_switch_arm_specs(specs: Mapping[str, Any]) -> None:
    if tuple(specs) != TARGET_SWITCH_ARMS or dict(specs) != TARGET_SWITCH_ARM_SPECS:
        raise ContractError("target-switch arm set or semantics changed")


def validate_target_switch_case(case: Mapping[str, Any]) -> None:
    required = {
        "case_id",
        "P_0",
        "X_nom",
        "X_carry",
        "X_result",
        "carry_edit_index",
        "carry_replacement_id",
        "result_edit_index",
        "result_replacement_id",
        "Y_nom",
        "Y_carry",
        "Y_result",
        "E_1",
        "D_1",
        "Q_1",
    }
    if set(case) != required:
        raise ContractError("target-switch case schema differs")
    spans = [tuple(case[key]) for key in ("X_nom", "X_carry", "X_result")]
    if (
        len({len(value) for value in spans}) != 1
        or not spans[0]
        or len(spans[0]) > MAX_SLOT
    ):
        raise ContractError("target-switch spans differ in length or are empty")
    token_sequences = [tuple(case["P_0"]), *spans]
    token_sequences.extend(tuple(case[key]) for key in ("Y_nom", "Y_carry", "Y_result"))
    if any(
        not values
        or len(values) > MAX_SLOT
        or any(not 0 <= int(token) < VOCAB_SIZE for token in values)
        for values in token_sequences
    ):
        raise ContractError("target-switch token sequence is invalid")
    for arm_index, (index_key, replacement_key) in enumerate(
        (
            ("carry_edit_index", "carry_replacement_id"),
            ("result_edit_index", "result_replacement_id"),
        ),
        start=1,
    ):
        index = int(case[index_key])
        if not 0 <= index < len(spans[0]):
            raise ContractError("target-switch edit index is invalid")
        changed = [
            offset
            for offset, values in enumerate(zip(spans[0], spans[arm_index]))
            if values[0] != values[1]
        ]
        if changed != [index] or spans[arm_index][index] != int(case[replacement_key]):
            raise ContractError("target-switch arm is not the frozen one-token edit")
    if case["E_1"] is not True or case["D_1"] != "COMMIT":
        raise ContractError("target-switch nominal event/action differs")
    runtime = RuntimeState.from_mapping(case["Q_1"])
    if (
        runtime.phase != "ACTIVE"
        or runtime.failure_flag
        or runtime.epoch_token_count != len(spans[0])
        or runtime.generation_slot_cursor != GENERATION_START + len(spans[0])
    ):
        raise ContractError("target-switch nominal runtime state is inconsistent")
    if tuple(case["Y_nom"]) in (tuple(case["Y_carry"]), tuple(case["Y_result"])):
        raise ContractError("carry target-switch targets do not differ")


def score_target_switch_pair(
    case: Mapping[str, Any],
    nominal: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
) -> dict[str, bool]:
    validate_target_switch_case(case)
    for result in (nominal, counterfactual):
        if set(result) != {"event", "observed_action", "output_ids"}:
            raise ContractError("target-switch result schema differs")
    boundary_action_changed = (
        nominal["observed_action"] != "COMMIT"
        or counterfactual["observed_action"] != "COMMIT"
    )
    receipts_valid = nominal["event"] is True and counterfactual["event"] is True
    nominal_exact = tuple(nominal["output_ids"]) == tuple(case["Y_nom"])
    carry_exact = tuple(counterfactual["output_ids"]) == tuple(case["Y_carry"])
    output_switch = tuple(nominal["output_ids"]) != tuple(counterfactual["output_ids"])
    paired = (
        receipts_valid
        and not boundary_action_changed
        and nominal_exact
        and carry_exact
        and output_switch
    )
    return {
        "receipts_valid": receipts_valid,
        "boundary_action_changed": boundary_action_changed,
        "nominal_exact": nominal_exact,
        "carry_exact": carry_exact,
        "output_switch": output_switch,
        "paired_target_switch": paired,
    }


def validate_hidden_board_receipt_counts(board_specs: Mapping[str, Any]) -> None:
    if dict(board_specs) != HIDDEN_BOARD_SPECS:
        raise ContractError("hidden-board count specification differs")
    hb = board_specs["H_B"]
    expected_cells = {
        "4-add": {"COMMIT": 256, "HALT": 64, "rows": 320},
        "4-sub": {"COMMIT": 256, "HALT": 64, "rows": 320},
        "6-add": {"COMMIT": 384, "HALT": 64, "rows": 448},
        "6-sub": {"COMMIT": 384, "HALT": 64, "rows": 448},
        "8-add": {"COMMIT": 512, "HALT": 64, "rows": 576},
        "8-sub": {"COMMIT": 512, "HALT": 64, "rows": 576},
    }
    if (
        hb["rows"] != 2688
        or sum(hb["width_row_counts"].values()) != 2688
        or sum(hb["action_row_counts"].values()) != 2688
        or hb["action_row_counts"] != {"COMMIT": 2304, "HALT": 384}
        or hb["cell_row_counts"] != expected_cells
        or sum(cell["rows"] for cell in hb["cell_row_counts"].values()) != 2688
    ):
        raise ContractError("H_B failure-inclusive denominator differs")


# ---------------------------------------------------------------------------
# Independent finite float64 mechanics falsifier.

TOY_VOCAB = 32
TOY_WIDTH = 16
TOY_HEADS = 2
TOY_HEAD_DIM = 8
TOY_NEUTRAL = 31
TOY_BASELINE_TOKEN = 7
TOY_POSITIVE_TOKEN = 8
TOY_NEGATIVE_TOKEN = 9


class _ZeroMotor(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*hidden.shape[:-1], 2), dtype=torch.float32)


def _mechanics_surface(token_id: int) -> EffectiveLogits:
    if not 0 <= int(token_id) < VOCAB_SIZE:
        raise ContractError("mechanics token is outside the frozen vocabulary")
    base = torch.full((1, VOCAB_SIZE), -1.0, dtype=torch.float64)
    base[0, int(token_id)] = 1.0
    return effective_logits(
        base,
        torch.zeros((1, 576), dtype=torch.float64),
        _ZeroMotor(),
        1,
    )


def _toy_weights() -> dict[str, torch.Tensor]:
    """Explicit non-random weights for a two-layer causal Transformer."""
    dtype = torch.float64
    embedding = torch.zeros((TOY_VOCAB, TOY_WIDTH), dtype=dtype)
    embedding[:, 1] = 1.0
    positive = (2, 4, 6, 8, 10, 12, 14, 16, 18)
    negative = (3, 5, 9, 11, 13, 15, 17, 19, 21)
    embedding[list(positive), 0] = 1.0
    embedding[list(negative), 0] = -1.0
    embedding[TOY_NEUTRAL, 0] = 0.0
    positions = torch.zeros((8, TOY_WIDTH), dtype=dtype)
    positions[:, 1] = 0.0
    positions[:, 2] = -8.0
    positions[2, 2] = 8.0
    positions[3, 2] = 4.0
    positions[4:6, 3] = 8.0
    matrices = {}
    for layer in range(2):
        for name in ("q", "k", "v", "o"):
            matrices[f"l{layer}_{name}"] = torch.zeros(
                (TOY_WIDTH, TOY_WIDTH), dtype=dtype
            )
    matrices["l0_q"][1, 0] = 1.0
    matrices["l0_k"][2, 0] = 1.0
    matrices["l0_v"][0, 0] = 1.0
    matrices["l0_o"][0, 4] = 1.0
    matrices["l1_q"][1, 0] = 1.0
    matrices["l1_k"][3, 0] = 1.0
    matrices["l1_v"][4, 0] = 1.0
    matrices["l1_o"][0, 5] = 1.0
    unembedding = torch.zeros((TOY_WIDTH, TOY_VOCAB), dtype=dtype)
    unembedding[1, TOY_BASELINE_TOKEN] = 0.5
    unembedding[5, TOY_POSITIVE_TOKEN] = 8.0
    unembedding[5, TOY_NEGATIVE_TOKEN] = -8.0
    return {
        "embedding": embedding,
        "positions": positions,
        "unembedding": unembedding,
        **matrices,
    }


def _toy_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    scores = (q @ k.T) / math.sqrt(TOY_HEAD_DIM)
    scores = torch.where(mask, scores, torch.full_like(scores, -torch.inf))
    rows = mask.any(dim=-1)
    probabilities = torch.zeros_like(scores)
    probabilities[rows] = torch.softmax(scores[rows], dim=-1)
    return probabilities @ v


def _toy_encode(
    ids: Sequence[int], allowed: torch.Tensor
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor]:
    weights = _toy_weights()
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    x = weights["embedding"][ids_tensor] + weights["positions"][: len(ids)]
    caches = []
    for layer in range(2):
        q = x @ weights[f"l{layer}_q"]
        k = x @ weights[f"l{layer}_k"]
        v = x @ weights[f"l{layer}_v"]
        joined = []
        for head in range(TOY_HEADS):
            start = head * TOY_HEAD_DIM
            stop = start + TOY_HEAD_DIM
            joined.append(
                _toy_attention(
                    q[:, start:stop], k[:, start:stop], v[:, start:stop], allowed
                )
            )
        x = x + torch.cat(joined, dim=-1) @ weights[f"l{layer}_o"]
        caches.append((k.clone(), v.clone()))
    logits = x @ weights["unembedding"]
    return x, tuple(caches), logits


def _toy_decode_one(
    kept_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    kept_positions: Sequence[int],
) -> tuple[torch.Tensor, int]:
    weights = _toy_weights()
    x = weights["embedding"][torch.tensor([1])] + weights["positions"][[7]]
    for layer in range(2):
        q = x @ weights[f"l{layer}_q"]
        k_current = x @ weights[f"l{layer}_k"]
        v_current = x @ weights[f"l{layer}_v"]
        cached_k, cached_v = kept_cache[layer]
        all_k = torch.cat((cached_k[list(kept_positions)], k_current), dim=0)
        all_v = torch.cat((cached_v[list(kept_positions)], v_current), dim=0)
        joined = []
        full_mask = torch.ones((1, all_k.shape[0]), dtype=torch.bool)
        for head in range(TOY_HEADS):
            start = head * TOY_HEAD_DIM
            stop = start + TOY_HEAD_DIM
            joined.append(
                _toy_attention(
                    q[:, start:stop],
                    all_k[:, start:stop],
                    all_v[:, start:stop],
                    full_mask,
                )
            )
        x = x + torch.cat(joined, dim=-1) @ weights[f"l{layer}_o"]
    logits = (x @ weights["unembedding"])[0]
    return logits, int(argmax_lowest_id(logits).item())


def _toy_masks(clean: bool) -> tuple[torch.Tensor, tuple[int, ...]]:
    # GL=[0,2), P=[2,4), X=[4,6), GR=[6,7).  GR never reads P.
    mask = torch.zeros((7, 7), dtype=torch.bool)
    for query in range(7):
        for key in range(query + 1):
            if query < 2:
                allowed = key < 2
            elif query < 4:
                allowed = key < 4
            elif query < 6:
                allowed = key < 2 or 4 <= key <= query or (not clean and 2 <= key < 4)
            else:
                allowed = key < 2 or 4 <= key <= 6
            mask[query, key] = allowed
    return mask, (0, 1, 4, 5, 6)


def toy_board() -> list[dict[str, Any]]:
    positive = (2, 4, 6, 10, 12, 14, 16, 18)
    negative = (3, 5, 11, 13, 15, 17, 19, 21)
    latest = tuple(range(20, 30)) + (0, 1, 22, 23, 24, 25)
    rows = []
    for wrapper in range(2):
        for pair in range(8):
            for span in range(16):
                rows.append(
                    {
                        "id": f"toy-{wrapper}-{pair}-{span}",
                        "wrapper": (0, 1 + wrapper),
                        "p_positive": (positive[pair], negative[pair]),
                        "p_negative": (negative[pair], positive[pair]),
                        "x": (latest[span], 30),
                    }
                )
    if len(rows) != 256:
        raise ContractError("toy board size changed")
    return rows


def _toy_run(p: Sequence[int], x: Sequence[int], clean: bool) -> dict[str, Any]:
    ids = (0, 1, *p, *x, 1)
    mask, keep = _toy_masks(clean)
    hidden, cache, logits = _toy_encode(ids, mask)
    next_logits, token = _toy_decode_one(cache, keep)
    return {
        "ids": ids,
        "mask": mask,
        "keep": keep,
        "hidden": hidden,
        "cache": cache,
        "probe": hidden[6].clone(),
        "head_logits": torch.tensor([hidden[6, 5], -hidden[6, 5]], dtype=torch.float64),
        "action": "HALT" if hidden[6, 5] >= 0 else "COMMIT",
        "head_forward_count": 1,
        "next_logits": next_logits,
        "next_token": token,
        "surface_logits": logits,
    }


def _validate_toy_position_receipt(
    ids: Sequence[int], mask: torch.Tensor, positions: Sequence[int]
) -> None:
    if len(ids) != 7 or mask.shape != (7, 7) or tuple(positions) != tuple(range(7)):
        raise ContractError("toy position or physical-surface receipt differs")


def _fixed_policy_trace(policy: str, events: int) -> tuple[str, ...]:
    if policy == "always-halt":
        return ("HALT",)
    if policy == "always-commit":
        return tuple("COMMIT" for _ in range(events))
    if policy.startswith("K"):
        count = int(policy[1:])
        return tuple(
            ["COMMIT"] * min(count, events - 1) + (["HALT"] if events > count else [])
        )
    raise ContractError("unknown fixed policy")


def run_cpu_reference_gates() -> dict[str, Any]:
    configure_cpu_determinism()
    for path, expected in (
        (ROOT / "R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md", THEORY_SHA256),
        (TOKENIZER_PATH, TOKENIZER_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ContractError(f"CPU reference input changed: {path}")
    board = toy_board()
    reference_board = independent_toy_board()
    if canonical_json_bytes(board) != canonical_json_bytes(reference_board):
        raise ContractError("independently generated toy board differs")
    counts = Counter()
    expected_outputs: list[dict[str, Any]] = []
    clean_mask, keep = _toy_masks(True)
    stale_mask, _ = _toy_masks(False)
    expected_xor = torch.zeros_like(clean_mask)
    expected_xor[4:6, 2:4] = True
    if not torch.equal(clean_mask ^ stale_mask, expected_xor):
        raise ContractError("toy mask XOR changed")

    for row in board:
        positive = _toy_run(row["p_positive"], row["x"], clean=False)
        negative = _toy_run(row["p_negative"], row["x"], clean=False)
        clean_positive = _toy_run(row["p_positive"], row["x"], clean=True)
        clean_negative = _toy_run(row["p_negative"], row["x"], clean=True)
        neutral = _toy_run((TOY_NEUTRAL, TOY_NEUTRAL), row["x"], clean=False)
        shuffled_positive = _toy_run(
            tuple(reversed(row["p_positive"])), row["x"], clean=False
        )
        if any(
            arm["head_forward_count"] != 1
            for arm in (
                positive,
                negative,
                clean_positive,
                clean_negative,
                neutral,
                shuffled_positive,
            )
        ):
            raise ContractError("a toy arm did not execute exactly one head forward")
        if not (
            torch.equal(positive["mask"], negative["mask"])
            and torch.equal(positive["mask"], neutral["mask"])
            and torch.equal(positive["mask"], shuffled_positive["mask"])
        ):
            raise ContractError("stale source-content controls changed open edges")
        _validate_toy_position_receipt(
            clean_positive["ids"], clean_positive["mask"], range(7)
        )
        reference = independent_toy_reference(row["p_positive"], row["x"], True)
        if (
            not torch.equal(reference["probe"], clean_positive["probe"])
            or not torch.equal(reference["next_logits"], clean_positive["next_logits"])
            or reference["next_token"] != clean_positive["next_token"]
        ):
            raise ContractError("independent float64 reference differs")
        for layer, (reference_k, reference_v) in enumerate(reference["kept_cache"]):
            actual_k, actual_v = clean_positive["cache"][layer]
            if not torch.equal(reference_k, actual_k[list(keep)]) or not torch.equal(
                reference_v, actual_v[list(keep)]
            ):
                raise ContractError("independent retained cache differs")
        if not torch.equal(clean_positive["probe"], clean_negative["probe"]):
            raise ContractError("clean classifier depends on source content")
        if clean_positive["action"] != clean_negative["action"]:
            raise ContractError("clean head action depends on source content")
        for layer in range(2):
            left_k, left_v = clean_positive["cache"][layer]
            right_k, right_v = clean_negative["cache"][layer]
            if not torch.equal(
                left_k[list(keep)], right_k[list(keep)]
            ) or not torch.equal(left_v[list(keep)], right_v[list(keep)]):
                raise ContractError("clean retained cache depends on source content")
        if not torch.equal(
            clean_positive["next_logits"], clean_negative["next_logits"]
        ):
            raise ContractError("clean do(P) next logits differ")
        if positive["next_token"] != TOY_POSITIVE_TOKEN:
            raise ContractError("planted positive stale path missed its target")
        if negative["next_token"] != TOY_NEGATIVE_TOKEN:
            raise ContractError("planted negative stale path missed its target")
        if neutral["next_token"] != TOY_BASELINE_TOKEN:
            raise ContractError("neutral stale control reached a directed target")
        if shuffled_positive["next_token"] != TOY_NEGATIVE_TOKEN:
            raise ContractError("shuffled stale control did not destroy direction")
        expected_outputs.append(
            {
                "id": row["id"],
                "clean": clean_positive["next_token"],
                "stale_positive": positive["next_token"],
                "stale_negative": negative["next_token"],
                "neutral": neutral["next_token"],
                "shuffled": shuffled_positive["next_token"],
            }
        )
        poisoned_cache = []
        for keys, values in clean_positive["cache"]:
            poisoned_keys = keys.clone()
            poisoned_values = values.clone()
            poisoned_keys[2:4] = torch.nan
            poisoned_values[2:4] = torch.nan
            poisoned_cache.append((poisoned_keys, poisoned_values))
        poisoned_logits, poisoned_token = _toy_decode_one(tuple(poisoned_cache), keep)
        if not torch.equal(poisoned_logits, clean_positive["next_logits"]) or (
            poisoned_token != clean_positive["next_token"]
        ):
            raise ContractError("dropped-source poisoning changed clean continuation")
        counts["paired_cases"] += 1
        counts["clone_surface_equal"] += 1
        counts["clean_reference_equal"] += 1
        counts["shared_dispatch_equal"] += 1
        counts["source_intervention_equal"] += 1
        counts["source_directed_switch"] += 1
        counts["poison_invariant"] += 1

    # FSM exhaustiveness over every legal boundary and cap edge.
    initial = initial_runtime_state(17)
    eos_surface = _mechanics_surface(EOS_ID)
    non_eos = consume_non_eos(initial, _mechanics_surface(5))
    committed = consume_event(non_eos, eos_surface, "COMMIT", 1)
    halted = consume_event(
        consume_non_eos(committed, _mechanics_surface(6)), eos_surface, "HALT", 1
    )
    if (
        halted.phase != "HALTED"
        or halted.commit_count != 1
        or halted.candidate_count != 2
    ):
        raise ContractError("runtime FSM trace differs")
    try:
        consume_non_eos(halted, _mechanics_surface(7))
    except ContractError:
        pass
    else:
        raise ContractError("post-HALT token was accepted")
    empty_commit = consume_event(initial, eos_surface, "COMMIT", 0)
    if not empty_commit.failure_flag:
        raise ContractError("empty COMMIT did not fail")
    cap_state = replace(
        initial,
        epoch_token_count=512,
        total_token_count=512,
        generation_slot_cursor=GENERATION_START + 512,
        rng_state_and_cursor=(17, 512),
    )
    if not consume_non_eos(cap_state, _mechanics_surface(2)).failure_flag:
        raise ContractError("epoch cap did not fail")
    consecutive = consume_event(committed, eos_surface, "COMMIT", 0)
    if not consecutive.failure_flag:
        raise ContractError("consecutive empty COMMIT did not fail")
    empty_halt = consume_event(initial, eos_surface, "HALT", 0)
    if empty_halt.phase != "HALTED" or empty_halt.failure_flag:
        raise ContractError("empty HALT event was not mechanically classified")
    commit_cap = initial
    for token in range(8):
        commit_cap = consume_non_eos(commit_cap, _mechanics_surface(2 + token))
        commit_cap = consume_event(commit_cap, eos_surface, "COMMIT", 1)
    final_candidate = consume_event(commit_cap, eos_surface, "HALT", 0)
    if final_candidate.phase != "HALTED" or final_candidate.candidate_count != 9:
        raise ContractError("ninth and final candidate did not halt exactly")
    over_commit = consume_non_eos(commit_cap, _mechanics_surface(17))
    if not consume_event(over_commit, eos_surface, "COMMIT", 1).failure_flag:
        raise ContractError("commit cap did not fail")

    try:
        _validate_toy_position_receipt((0, 1, 2, 3, 4, 5, 6), clean_mask, range(1, 8))
    except ContractError:
        pass
    else:
        raise ContractError("shifted toy positions were accepted")

    # Runtime API has no text, field, schedule, or gold arguments.
    forbidden = {
        "text",
        "operation",
        "width",
        "gold",
        "schedule",
        "answer",
        "state_line",
    }
    api_names = set(inspect.signature(consume_non_eos).parameters) | set(
        inspect.signature(consume_event).parameters
    )
    if api_names & forbidden:
        raise ContractError("runtime API exposes a forbidden semantic resource")

    # Exactly one head forward per event and motor-off still computes delta.
    head = BoundaryHead()
    head(torch.zeros((1, 576)))
    if head.forward_count != 1:
        raise ContractError("boundary head forward count differs")
    motor = CarryMotor()
    with torch.no_grad():
        for parameter in motor.parameters():
            parameter.fill_(0.125)
    base = torch.linspace(-1.0, 1.0, 32768).reshape(1, -1)
    hidden = torch.ones((1, 576))
    off = effective_logits(base, hidden, motor, 0)
    on = effective_logits(base, hidden, motor, 1)
    changed = torch.where(off.ell_eff.ne(on.ell_eff))[1].tolist()
    if changed != [V0_ID, V1_ID] or not torch.equal(off.motor_delta, on.motor_delta):
        raise ContractError("motor-off/on coordinate contract differs")

    policies = {
        "always-halt": _fixed_policy_trace("always-halt", 9),
        "always-commit": _fixed_policy_trace("always-commit", 9),
        "K4": _fixed_policy_trace("K4", 9),
        "K6": _fixed_policy_trace("K6", 9),
        "K8": _fixed_policy_trace("K8", 9),
    }
    if policies["always-halt"] != ("HALT",) or policies["K4"] != (
        "COMMIT",
        "COMMIT",
        "COMMIT",
        "COMMIT",
        "HALT",
    ):
        raise ContractError("fixed policy traces differ")
    fixed_head_counts = {}
    for policy in policies:
        control_head = BoundaryHead()
        control_head(torch.zeros((1, 576)))
        fixed_head_counts[policy] = control_head.forward_count
    if set(fixed_head_counts.values()) != {1}:
        raise ContractError(
            "fixed policy did not execute true-and-discard head exactly once"
        )

    # A bounded finite-state enumerator reproduces the explicit two-event trace.
    enumerated = []
    state = initial_runtime_state(17)
    for token, action in ((5, "COMMIT"), (6, "HALT")):
        state = consume_non_eos(state, _mechanics_surface(token))
        state = consume_event(state, eos_surface, action, 1)
        enumerated.append((state.phase, state.commit_count, state.candidate_count))
    if enumerated != [("ACTIVE", 1, 1), ("HALTED", 1, 2)]:
        raise ContractError("finite-state collapse enumerator differs")

    # Arm-local divergence is permitted. Each second event consumes only the
    # token authored by its own first reconstruction.
    first_stale = _toy_run((2, 3), (20, 30), clean=False)["next_token"]
    first_clean = _toy_run((2, 3), (20, 30), clean=True)["next_token"]
    second_stale_arm = _toy_run((2, 3), (first_stale, 30), clean=True)["next_token"]
    second_clean_arm = _toy_run((2, 3), (first_clean, 30), clean=True)["next_token"]
    tokens = (first_stale, first_clean, second_stale_arm, second_clean_arm)
    if tokens != (8, 7, 8, 7):
        raise ContractError(
            "autonomous two-event toy did not preserve arm-local divergence"
        )

    gates = {
        "01_cloned_dispatch_and_mask_xor": counts["clone_surface_equal"] == 256,
        "02_clean_reference_kv": counts["clean_reference_equal"] == 256,
        "03_common_classifier_and_endpoint": counts["shared_dispatch_equal"] == 256,
        "04_do_p_and_structured_stale_control": (
            counts["source_intervention_equal"] == 256
            and counts["source_directed_switch"] == 256
        ),
        "05_dropped_source_poison_invariance": counts["poison_invariant"] == 256,
        "06_fixed_positions_and_surface": True,
        "07_complete_runtime_fsm": True,
        "08_token_only_runtime_api": True,
        "09_one_head_forward_per_event": head.forward_count == 1,
        "10_autonomous_arm_local_divergence": tokens == (8, 7, 8, 7),
        "11_fixed_policy_traces": True,
        "12_motor_single_surface": changed == [V0_ID, V1_ID],
        "13_finite_state_collapse": True,
    }
    if not all(gates.values()):
        raise ContractError("one or more CPU mechanics gates failed")
    return {
        "schema": "r12-scert-cpu-reference-gates-v1",
        "protocol": PROTOCOL_ID,
        "theory_sha256": THEORY_SHA256,
        "dtype": "torch.float64",
        "device": "cpu",
        "toy": {
            "layers": 2,
            "width": 16,
            "heads": 2,
            "vocabulary": 32,
            "paired_cases": 256,
            "board_sha256": sha256_bytes(canonical_json_bytes(board)),
            "positions_sha256": sha256_bytes(
                torch.arange(7, dtype=torch.int64).numpy().tobytes()
            ),
            "mask_pair_sha256": sha256_bytes(
                clean_mask.numpy().tobytes() + stale_mask.numpy().tobytes()
            ),
            "expected_outputs_sha256": sha256_bytes(
                canonical_json_bytes(expected_outputs)
            ),
            "weights_sha256": sha256_bytes(
                b"".join(
                    value.contiguous().numpy().tobytes()
                    for _, value in sorted(_toy_weights().items())
                )
            ),
        },
        "gates": gates,
        "all_passed": True,
        "added_parameters": added_parameter_manifest(),
        "hidden_content_generated": False,
        "capability_claim": False,
        "h100_authorized": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ContractError("JSON artifact must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifests = subparsers.add_parser("write-cpu-manifests")
    manifests.add_argument("--source", type=Path, required=True)
    manifests.add_argument("--runtime", type=Path, required=True)
    reference = subparsers.add_parser("cpu-reference")
    reference.add_argument("--report", type=Path, required=True)
    validate = subparsers.add_parser("validate-cpu-package")
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--runtime", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "write-cpu-manifests":
        source_receipt = atomic_publish_json(
            args.source, build_source_manifest(), "r12-scert-source-manifest-v1"
        )
        runtime_receipt = atomic_publish_json(
            args.runtime, build_runtime_manifest(), "r12-scert-cpu-runtime-manifest-v1"
        )
        print(
            json.dumps(
                {"source": source_receipt, "runtime": runtime_receipt}, sort_keys=True
            )
        )
        return
    if args.command == "cpu-reference":
        receipt = atomic_publish_json(
            args.report, run_cpu_reference_gates(), "r12-scert-cpu-reference-gates-v1"
        )
        print(
            json.dumps(
                {"cpu_reference": receipt, "h100_authorized": False}, sort_keys=True
            )
        )
        return
    validate_source_manifest(_load_json(args.source))
    validate_runtime_manifest(_load_json(args.runtime))
    parent = validate_parent_and_optimizer_manifest()
    validate_hidden_board_receipt_counts(HIDDEN_BOARD_SPECS)
    validate_target_switch_arm_specs(TARGET_SWITCH_ARM_SPECS)
    report = {
        "schema": "r12-scert-cpu-package-validation-v1",
        "protocol": PROTOCOL_ID,
        "source_manifest_sha256": sha256_file(args.source),
        "runtime_manifest_sha256": sha256_file(args.runtime),
        "parent_optimizer": parent,
        "target_switch_arm_specs": TARGET_SWITCH_ARM_SPECS,
        "hidden_board_specs_sha256": sha256_bytes(
            canonical_json_bytes(HIDDEN_BOARD_SPECS)
        ),
        "cpu_reference": run_cpu_reference_gates(),
        "hidden_content_generated": False,
        "h100_authorized": False,
    }
    receipt = atomic_publish_json(
        args.report, report, "r12-scert-cpu-package-validation-v1"
    )
    print(json.dumps({"validation": receipt, "h100_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
