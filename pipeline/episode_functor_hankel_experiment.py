"""Fail-closed experiment receipts for neural Hankel-shift qualification.

The HSC arm receipt binds architecture and objective.  This module binds the
remaining facts needed to run an auditable neural experiment: exact initialized
tensors, optimizer and precision semantics, update order, target-bit accounting,
the train-only split, and the measured-resource gate.

It deliberately does not launch work, read environment variables, or generate
any board split.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Literal, Mapping

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_qualification_custody import (  # noqa: E402
    QualificationSplitCustody,
)
from pipeline.episode_functor_qualification_batch import (  # noqa: E402
    QualificationSupervisorBatch,
)
from pipeline.episode_functor_resource_receipt import (  # noqa: E402
    QualificationResourceReceipt,
)
from episode_functor_constrained_transport import (  # noqa: E402
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_hankel_arm import HankelArmReceipt  # noqa: E402
from episode_functor_hankel_completion import (  # noqa: E402
    HankelShiftCompletionProjector,
)
from episode_functor_learned_system import PROTECTED_SHOHIN_SHA256  # noqa: E402
from episode_functor_pointer_compiler import MAX_UNIQUE_KEYS  # noqa: E402
from episode_functor_query_parser import NeuralOpaqueQueryParser  # noqa: E402
from episode_functor_witness_compiler import (  # noqa: E402
    ProofCarryingWitnessCompiler,
    RECORD_TYPES,
    ROLE_IGNORE,
)


HANKEL_INITIALIZATION_SCHEMA = "efc-hankel-initialization/v1"
HANKEL_TARGET_SCHEMA = "efc-hankel-target-accounting/v1"
HANKEL_OPTIMIZER_SCHEMA = "efc-hankel-optimizer/v1"
HANKEL_SCHEDULE_SCHEMA = "efc-hankel-update-schedule/v1"
HANKEL_EXPERIMENT_SCHEMA = "efc-hankel-neural-experiment/v1"

ExperimentPhase = Literal["measurement-canary", "qualification-fit"]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class HankelExperimentError(ValueError):
    """An HSC neural experiment fact is absent, ambiguous, or inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HankelExperimentError(f"{label} SHA-256 differs")
    return value


def _float_hex(value: float, *, label: str) -> str:
    if not isinstance(value, float) or not math.isfinite(value):
        raise HankelExperimentError(f"{label} must be a finite float")
    return value.hex()


def _bits_for_classes(classes: int) -> int:
    if not isinstance(classes, int) or classes < 2:
        raise HankelExperimentError("target class cardinality differs")
    return int(math.ceil(math.log2(classes)))


def tensor_mapping_sha256(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """Hash exact tensor names, dtypes, shapes, and raw contiguous bytes."""

    if not isinstance(tensors, Mapping) or not tensors:
        raise HankelExperimentError("tensor mapping is empty")
    digest = sha256(b"EFC-HSC-TENSOR-MAPPING-V1\0")
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise HankelExperimentError("tensor mapping entry differs")
        canonical = value.detach().to(device="cpu", copy=True).contiguous()
        header = _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "name": name,
                "shape": list(canonical.shape),
            }
        )
        raw = canonical.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _module_parameter_mapping(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
    }


@dataclass(frozen=True, slots=True)
class HankelInitializationReceipt:
    """Exact initialized state, including the independent code branches."""

    seed: int
    arm_receipt_sha256: str
    compiler_state_sha256: str
    trainable_state_sha256: str
    query_state_sha256: str
    base_branch_state_sha256: str
    derivative_branch_state_sha256: str
    trainable_parameters: int
    compiler_buffers: int
    independent_noncollapsed_branches: bool
    receipt_sha256: str
    schema: str = HANKEL_INITIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_INITIALIZATION_SCHEMA
            or not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
            or self.trainable_parameters < 1
            or self.compiler_buffers < 1
            or self.independent_noncollapsed_branches is not True
        ):
            raise HankelExperimentError("HSC initialization identity differs")
        for field_name in (
            "arm_receipt_sha256",
            "compiler_state_sha256",
            "trainable_state_sha256",
            "query_state_sha256",
            "base_branch_state_sha256",
            "derivative_branch_state_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        if self.base_branch_state_sha256 == self.derivative_branch_state_sha256:
            raise HankelExperimentError(
                "HSC base and derivative branches are initialization-collapsed"
            )
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise HankelExperimentError("HSC initialization receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"

    def assert_matched_trainable_initialization(
        self,
        other: "HankelInitializationReceipt",
    ) -> None:
        """Require controls to start from identical trainable parameters."""

        if (
            not isinstance(other, HankelInitializationReceipt)
            or self.seed != other.seed
            or self.trainable_parameters != other.trainable_parameters
            or self.trainable_state_sha256 != other.trainable_state_sha256
            or self.query_state_sha256 != other.query_state_sha256
        ):
            raise HankelExperimentError(
                "HSC control trainable initialization differs"
            )


def create_hankel_initialization_receipt(
    compiler: ProofCarryingWitnessCompiler,
    query_parser: NeuralOpaqueQueryParser,
    *,
    seed: int,
    arm_receipt: HankelArmReceipt,
) -> HankelInitializationReceipt:
    """Bind one already-constructed maximum-lane system before any update."""

    if (
        not isinstance(compiler, ProofCarryingWitnessCompiler)
        or not isinstance(query_parser, NeuralOpaqueQueryParser)
        or not isinstance(arm_receipt, HankelArmReceipt)
        or not isinstance(compiler.projector, HankelShiftCompletionProjector)
    ):
        raise HankelExperimentError("HSC initialized system differs")
    projector = compiler.projector
    if (
        compiler.parameter_count() != arm_receipt.source_compiler_parameters
        or query_parser.parameter_count()
        != arm_receipt.query_parser_parameters
    ):
        raise HankelExperimentError(
            "HSC initialized system leaves its arm receipt"
        )
    trainable = _module_parameter_mapping(compiler)
    if any(not parameter.requires_grad for parameter in trainable.values()):
        raise HankelExperimentError("HSC compiler contains a frozen parameter")
    compiler_buffers = sum(buffer.numel() for buffer in compiler.buffers())
    unsigned = {
        "seed": seed,
        "arm_receipt_sha256": arm_receipt.receipt_sha256,
        "compiler_state_sha256": tensor_mapping_sha256(compiler.state_dict()),
        "trainable_state_sha256": tensor_mapping_sha256(trainable),
        "query_state_sha256": tensor_mapping_sha256(
            query_parser.state_dict()
        ),
        "base_branch_state_sha256": tensor_mapping_sha256(
            projector.base.state_dict()
        ),
        "derivative_branch_state_sha256": tensor_mapping_sha256(
            projector.derivative.state_dict()
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "compiler_buffers": compiler_buffers,
        "independent_noncollapsed_branches": True,
        "schema": HANKEL_INITIALIZATION_SCHEMA,
    }
    if (
        unsigned["base_branch_state_sha256"]
        == unsigned["derivative_branch_state_sha256"]
    ):
        raise HankelExperimentError(
            "HSC exact uniform branch initialization is a NO-GO"
        )
    return HankelInitializationReceipt(
        **unsigned,
        receipt_sha256=_digest(unsigned),
    )


@dataclass(frozen=True, slots=True)
class HankelTargetAccounting:
    """Supplied bits, with deterministic signature labels not double-credited."""

    rows: int
    key_assignment_bits: int
    record_type_bits: int
    occurrence_role_bits: int
    record_answer_bits: int
    transition_bits: int
    observer_bits: int
    independent_machine_target_bits: int
    derived_signature_target_bits: int
    supplied_target_bits: int
    independent_target_bits: int
    receipt_sha256: str
    schema: str = HANKEL_TARGET_SCHEMA

    def __post_init__(self) -> None:
        components = (
            self.key_assignment_bits,
            self.record_type_bits,
            self.occurrence_role_bits,
            self.record_answer_bits,
            self.transition_bits,
            self.observer_bits,
        )
        if (
            self.schema != HANKEL_TARGET_SCHEMA
            or self.rows < 1
            or any(value < 1 for value in components)
            or self.independent_machine_target_bits != sum(components)
            or self.derived_signature_target_bits != self.rows * 5_120
            or self.supplied_target_bits
            != self.independent_machine_target_bits
            + self.derived_signature_target_bits
            or self.independent_target_bits
            != self.independent_machine_target_bits
        ):
            raise HankelExperimentError("HSC target accounting differs")
        _require_sha256(self.receipt_sha256, label="target receipt")
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise HankelExperimentError("HSC target receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"


def create_hankel_target_accounting(
    supervisor: QualificationSupervisorBatch,
) -> HankelTargetAccounting:
    """Count the exact frozen labels visible to the optimizer."""

    if not isinstance(supervisor, QualificationSupervisorBatch):
        raise HankelExperimentError("HSC target supervisor differs")
    rows = supervisor.batch_size
    key_bits = (
        rows
        * (PRIMARY_STATES + PRIMARY_ACTIONS + PRIMARY_OBSERVERS)
        * _bits_for_classes(MAX_UNIQUE_KEYS)
    )
    record_type_bits = int(supervisor.record_label_valid.sum()) * (
        _bits_for_classes(RECORD_TYPES)
    )
    occurrence_role_bits = int(supervisor.occurrence_label_valid.sum()) * (
        _bits_for_classes(ROLE_IGNORE + 1)
    )
    record_answer_bits = int(supervisor.answer_label_valid.sum()) * (
        _bits_for_classes(PRIMARY_ANSWERS)
    )
    transition_bits = (
        rows
        * PRIMARY_ACTIONS
        * PRIMARY_STATES
        * _bits_for_classes(PRIMARY_STATES)
    )
    observer_bits = (
        rows
        * PRIMARY_OBSERVERS
        * PRIMARY_STATES
        * _bits_for_classes(PRIMARY_ANSWERS)
    )
    independent = sum(
        (
            key_bits,
            record_type_bits,
            occurrence_role_bits,
            record_answer_bits,
            transition_bits,
            observer_bits,
        )
    )
    derived = rows * 5_120
    unsigned = {
        "rows": rows,
        "key_assignment_bits": key_bits,
        "record_type_bits": record_type_bits,
        "occurrence_role_bits": occurrence_role_bits,
        "record_answer_bits": record_answer_bits,
        "transition_bits": transition_bits,
        "observer_bits": observer_bits,
        "independent_machine_target_bits": independent,
        "derived_signature_target_bits": derived,
        "supplied_target_bits": independent + derived,
        "independent_target_bits": independent,
        "schema": HANKEL_TARGET_SCHEMA,
    }
    return HankelTargetAccounting(
        **unsigned,
        receipt_sha256=_digest(unsigned),
    )


@dataclass(frozen=True, slots=True)
class HankelOptimizerContract:
    """Exact optimizer and mixed-precision semantics for every arm."""

    optimizer: str = "torch.optim.AdamW"
    learning_rate_hex: str = float(3e-4).hex()
    beta1_hex: str = float(0.9).hex()
    beta2_hex: str = float(0.999).hex()
    epsilon_hex: str = float(1e-8).hex()
    weight_decay_hex: str = float(0.01).hex()
    maximum_gradient_norm_hex: str = float(1.0).hex()
    parameter_dtype: str = "float32"
    autocast_dtype: str = "bfloat16"
    optimizer_state_dtype: str = "float32"
    gradient_scaler: bool = False
    fused: bool = True
    foreach: bool = False
    capturable: bool = False
    tf32: bool = True
    deterministic_algorithms: bool = False
    compile_mode: str = "eager"
    receipt_sha256: str = ""
    schema: str = HANKEL_OPTIMIZER_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_OPTIMIZER_SCHEMA
            or self.optimizer != "torch.optim.AdamW"
            or self.parameter_dtype != "float32"
            or self.autocast_dtype != "bfloat16"
            or self.optimizer_state_dtype != "float32"
            or type(self.gradient_scaler) is not bool
            or type(self.fused) is not bool
            or type(self.foreach) is not bool
            or type(self.capturable) is not bool
            or type(self.tf32) is not bool
            or type(self.deterministic_algorithms) is not bool
            or self.gradient_scaler
            or not self.fused
            or self.foreach
            or self.capturable
            or not self.tf32
            or self.deterministic_algorithms
            or self.compile_mode != "eager"
        ):
            raise HankelExperimentError("HSC optimizer semantics differ")
        for name in (
            "learning_rate_hex",
            "beta1_hex",
            "beta2_hex",
            "epsilon_hex",
            "weight_decay_hex",
            "maximum_gradient_norm_hex",
        ):
            try:
                value = float.fromhex(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise HankelExperimentError(
                    f"HSC {name} differs"
                ) from exc
            if not math.isfinite(value):
                raise HankelExperimentError(f"HSC {name} is nonfinite")
        _require_sha256(self.receipt_sha256, label="optimizer receipt")
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise HankelExperimentError("HSC optimizer receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"


def create_hankel_optimizer_contract(
    *,
    learning_rate: float = 3e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
    maximum_gradient_norm: float = 1.0,
) -> HankelOptimizerContract:
    unsigned = {
        "optimizer": "torch.optim.AdamW",
        "learning_rate_hex": _float_hex(learning_rate, label="learning rate"),
        "beta1_hex": _float_hex(beta1, label="beta1"),
        "beta2_hex": _float_hex(beta2, label="beta2"),
        "epsilon_hex": _float_hex(epsilon, label="epsilon"),
        "weight_decay_hex": _float_hex(weight_decay, label="weight decay"),
        "maximum_gradient_norm_hex": _float_hex(
            maximum_gradient_norm,
            label="maximum gradient norm",
        ),
        "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "optimizer_state_dtype": "float32",
        "gradient_scaler": False,
        "fused": True,
        "foreach": False,
        "capturable": False,
        "tf32": True,
        "deterministic_algorithms": False,
        "compile_mode": "eager",
        "schema": HANKEL_OPTIMIZER_SCHEMA,
    }
    return HankelOptimizerContract(
        **unsigned,
        receipt_sha256=_digest(unsigned),
    )


@dataclass(frozen=True, slots=True)
class HankelScheduleContract:
    """Finite, equal-budget optimizer order with no adaptive early stopping."""

    updates: int
    microbatch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    order_seed: int
    order_seed_sha256: str
    checkpoint_interval: int
    metric_interval: int
    scheduler: str
    early_stopping: bool
    drop_last: bool
    receipt_sha256: str
    schema: str = HANKEL_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_SCHEDULE_SCHEMA
            or self.updates < 1
            or self.microbatch_size < 1
            or self.gradient_accumulation < 1
            or self.effective_batch_size
            != self.microbatch_size * self.gradient_accumulation
            or self.order_seed < 0
            or self.checkpoint_interval < 1
            or self.metric_interval < 1
            or self.checkpoint_interval > self.updates
            or self.metric_interval > self.updates
            or self.scheduler != "constant"
            or self.early_stopping is not False
            or self.drop_last is not False
        ):
            raise HankelExperimentError("HSC update schedule differs")
        _require_sha256(self.order_seed_sha256, label="order seed")
        expected_seed = sha256(
            f"EFC-HSC-ORDER-SEED-V1:{self.order_seed}".encode("ascii")
        ).hexdigest()
        if self.order_seed_sha256 != expected_seed:
            raise HankelExperimentError("HSC order seed commitment differs")
        _require_sha256(self.receipt_sha256, label="schedule receipt")
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise HankelExperimentError("HSC schedule receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def create_hankel_schedule_contract(
    *,
    updates: int,
    microbatch_size: int,
    gradient_accumulation: int,
    order_seed: int,
    checkpoint_interval: int,
    metric_interval: int,
) -> HankelScheduleContract:
    unsigned = {
        "updates": updates,
        "microbatch_size": microbatch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": microbatch_size * gradient_accumulation,
        "order_seed": order_seed,
        "order_seed_sha256": sha256(
            f"EFC-HSC-ORDER-SEED-V1:{order_seed}".encode("ascii")
        ).hexdigest(),
        "checkpoint_interval": checkpoint_interval,
        "metric_interval": metric_interval,
        "scheduler": "constant",
        "early_stopping": False,
        "drop_last": False,
        "schema": HANKEL_SCHEDULE_SCHEMA,
    }
    return HankelScheduleContract(
        **unsigned,
        receipt_sha256=_digest(unsigned),
    )


@dataclass(frozen=True, slots=True)
class HankelExperimentReceipt:
    """Complete canary or qualification-fit authorization receipt."""

    phase: ExperimentPhase
    run_id: str
    arm_receipt_sha256: str
    initialization_receipt_sha256: str
    optimizer_receipt_sha256: str
    schedule_receipt_sha256: str
    target_receipt_sha256: str
    train_custody_receipt_sha256: str
    resource_receipt_sha256: str | None
    protected_checkpoint_sha256: str
    tokenizer_sha256: str
    runtime_source_manifest_sha256: str
    train_rows: int
    train_source_bytes: int
    train_supplied_target_bits: int
    allowed_optimizer_split: str
    development_visible: bool
    confirmation_visible: bool
    output_mode: str
    network_mode: str
    pretraining_authorized: bool
    receipt_sha256: str
    schema: str = HANKEL_EXPERIMENT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_EXPERIMENT_SCHEMA
            or self.phase not in ("measurement-canary", "qualification-fit")
            or not self.run_id
            or not self.run_id.isascii()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in self.run_id)
            or self.train_rows < 1
            or self.train_source_bytes < 1
            or self.train_supplied_target_bits < 1
            or self.allowed_optimizer_split != "train"
            or self.development_visible is not False
            or self.confirmation_visible is not False
            or self.output_mode != "exclusive-atomic-single-writer"
            or self.network_mode != "deny"
            or self.pretraining_authorized is not False
            or self.protected_checkpoint_sha256 != PROTECTED_SHOHIN_SHA256
        ):
            raise HankelExperimentError("HSC experiment identity differs")
        for field_name in (
            "arm_receipt_sha256",
            "initialization_receipt_sha256",
            "optimizer_receipt_sha256",
            "schedule_receipt_sha256",
            "target_receipt_sha256",
            "train_custody_receipt_sha256",
            "protected_checkpoint_sha256",
            "tokenizer_sha256",
            "runtime_source_manifest_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        if self.phase == "qualification-fit":
            _require_sha256(
                self.resource_receipt_sha256,
                label="resource receipt",
            )
        elif self.resource_receipt_sha256 is not None:
            raise HankelExperimentError(
                "measurement canary cannot claim a measured resource receipt"
            )
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise HankelExperimentError("HSC experiment receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"

    def assert_matched_experiment(self, other: "HankelExperimentReceipt") -> None:
        """Require equal noncausal resources before comparing two arms."""

        if not isinstance(other, HankelExperimentReceipt):
            raise HankelExperimentError("HSC comparison receipt differs")
        excluded = {
            "arm_receipt_sha256",
            "initialization_receipt_sha256",
            "receipt_sha256",
            "run_id",
        }
        left = {
            key: value
            for key, value in self._unsigned_mapping().items()
            if key not in excluded
        }
        right = {
            key: value
            for key, value in other._unsigned_mapping().items()
            if key not in excluded
        }
        if left != right:
            raise HankelExperimentError(
                "HSC arm comparison changes noncausal resources"
            )


def create_hankel_experiment_receipt(
    *,
    phase: ExperimentPhase,
    run_id: str,
    arm_receipt: HankelArmReceipt,
    initialization: HankelInitializationReceipt,
    optimizer: HankelOptimizerContract,
    schedule: HankelScheduleContract,
    targets: HankelTargetAccounting,
    train_custody: QualificationSplitCustody,
    train_source_bytes: int,
    tokenizer_sha256: str,
    runtime_source_manifest_sha256: str,
    resource_receipt: QualificationResourceReceipt | None = None,
) -> HankelExperimentReceipt:
    """Create a canary receipt or admit a fit after measured resources exist."""

    if (
        not isinstance(arm_receipt, HankelArmReceipt)
        or not isinstance(initialization, HankelInitializationReceipt)
        or not isinstance(optimizer, HankelOptimizerContract)
        or not isinstance(schedule, HankelScheduleContract)
        or not isinstance(targets, HankelTargetAccounting)
        or not isinstance(train_custody, QualificationSplitCustody)
    ):
        raise HankelExperimentError("HSC experiment component type differs")
    train_custody.assert_training_split()
    if (
        initialization.arm_receipt_sha256 != arm_receipt.receipt_sha256
        or targets.rows != train_custody.row_count
        or train_source_bytes < train_custody.row_count
    ):
        raise HankelExperimentError("HSC experiment component binding differs")
    if phase == "measurement-canary":
        if resource_receipt is not None:
            raise HankelExperimentError(
                "HSC canary resource receipt must be measured afterward"
            )
        resource_sha256 = None
    elif phase == "qualification-fit":
        if (
            not isinstance(resource_receipt, QualificationResourceReceipt)
            or resource_receipt.receipt_kind == "forecast"
            or resource_receipt.bindings.board_sha256
            != train_custody.board_manifest_sha256
            or resource_receipt.bindings.source_sha256
            != train_custody.receipt_sha256
            or resource_receipt.resources.examples.value
            != train_custody.row_count
            or resource_receipt.resources.target_bits.value
            != targets.supplied_target_bits
            or resource_receipt.resources.source_bytes.value
            != train_source_bytes
            or resource_receipt.resources.updates.value != schedule.updates
            or resource_receipt.resources.trainable_parameters.value
            != initialization.trainable_parameters
            or resource_receipt.resources.total_parameters.value
            != arm_receipt.complete_parameters
        ):
            raise HankelExperimentError(
                "HSC measured resource receipt does not bind the fit"
            )
        resource_sha256 = resource_receipt.receipt_sha256
    else:
        raise HankelExperimentError("HSC experiment phase differs")
    unsigned = {
        "phase": phase,
        "run_id": run_id,
        "arm_receipt_sha256": arm_receipt.receipt_sha256,
        "initialization_receipt_sha256": initialization.receipt_sha256,
        "optimizer_receipt_sha256": optimizer.receipt_sha256,
        "schedule_receipt_sha256": schedule.receipt_sha256,
        "target_receipt_sha256": targets.receipt_sha256,
        "train_custody_receipt_sha256": train_custody.receipt_sha256,
        "resource_receipt_sha256": resource_sha256,
        "protected_checkpoint_sha256": PROTECTED_SHOHIN_SHA256,
        "tokenizer_sha256": _require_sha256(
            tokenizer_sha256,
            label="tokenizer",
        ),
        "runtime_source_manifest_sha256": _require_sha256(
            runtime_source_manifest_sha256,
            label="runtime source manifest",
        ),
        "train_rows": train_custody.row_count,
        "train_source_bytes": train_source_bytes,
        "train_supplied_target_bits": targets.supplied_target_bits,
        "allowed_optimizer_split": "train",
        "development_visible": False,
        "confirmation_visible": False,
        "output_mode": "exclusive-atomic-single-writer",
        "network_mode": "deny",
        "pretraining_authorized": False,
        "schema": HANKEL_EXPERIMENT_SCHEMA,
    }
    return HankelExperimentReceipt(
        **unsigned,
        receipt_sha256=_digest(unsigned),
    )


__all__ = [
    "HANKEL_EXPERIMENT_SCHEMA",
    "HANKEL_INITIALIZATION_SCHEMA",
    "HANKEL_OPTIMIZER_SCHEMA",
    "HANKEL_SCHEDULE_SCHEMA",
    "HANKEL_TARGET_SCHEMA",
    "HankelExperimentError",
    "HankelExperimentReceipt",
    "HankelInitializationReceipt",
    "HankelOptimizerContract",
    "HankelScheduleContract",
    "HankelTargetAccounting",
    "create_hankel_experiment_receipt",
    "create_hankel_initialization_receipt",
    "create_hankel_optimizer_contract",
    "create_hankel_schedule_contract",
    "create_hankel_target_accounting",
    "tensor_mapping_sha256",
]
