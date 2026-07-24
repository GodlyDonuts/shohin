from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_hankel_experiment import (  # noqa: E402
    HankelExperimentError,
    HankelInitializationReceipt,
    create_hankel_experiment_receipt,
    create_hankel_optimizer_contract,
    create_hankel_schedule_contract,
    create_hankel_target_accounting,
    tensor_mapping_sha256,
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
from episode_functor_hankel_arm import (  # noqa: E402
    HANKEL_ARM_SCHEMA,
    SOURCE_BOUNDARY,
    SUPERVISION_CONTRACT,
    HankelArmReceipt,
)


def _rows():
    return generate_pilot_rows(
        seed="efc-hankel-experiment-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )


class _Encoded:
    def __init__(self, payload: str) -> None:
        self.ids = list(payload.encode("ascii"))
        self.offsets = [
            (index, index + 1)
            for index in range(len(self.ids))
        ]


class _ByteTokenizer:
    def to_str(self) -> str:
        return '{"kind":"test-byte-tokenizer"}'

    def encode(self, payload: str) -> _Encoded:
        return _Encoded(payload)


def _candidate(rows, split):
    selected = tuple(row for row in rows if row.split == split)
    tokenizer = _ByteTokenizer()
    return collate_candidate_sources(
        project_candidate_sources(selected, split=split),
        tokenizer=tokenizer,
        tokenizer_artifact_sha256="a" * 64,
        expected_tokenizer_runtime_sha256=tokenizer_runtime_sha256(
            tokenizer
        ),
    )


def _arm() -> HankelArmReceipt:
    unsigned = {
        "arm_name": "hsc-prefix-treatment",
        "decode_mode": "hankel-shift",
        "incidence_mode": "prefix",
        "incidence_sha256": "1" * 64,
        "random_seed_sha256": "2" * 64,
        "max_depth": 3,
        "word_count": 40,
        "base_signature_cells_per_example": 640,
        "derivative_signature_cells_per_example": 1_920,
        "signature_target_bits_per_example": 5_120,
        "temperature_hex": float(0.05).hex(),
        "objective_sha256": "3" * 64,
        "source_compiler_parameters": 64_407_956,
        "projector_parameters": 19_717_124,
        "query_parser_parameters": 6_003_489,
        "qualification_trainable_parameters": 64_407_956,
        "complete_parameters": 195_493_109,
        "headroom": 4_506_891,
        "persistent_machine_bytes": 1_536,
        "protected_checkpoint_sha256": (
            "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
        ),
        "source_boundary": SOURCE_BOUNDARY,
        "supervision_contract": SUPERVISION_CONTRACT,
        "schema": HANKEL_ARM_SCHEMA,
    }
    return HankelArmReceipt(
        **unsigned,
        receipt_sha256=hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    )


def _initialization(
    arm: HankelArmReceipt,
    *,
    trainable_state_sha256: str = "5" * 64,
) -> HankelInitializationReceipt:
    unsigned = {
        "seed": 91,
        "arm_receipt_sha256": arm.receipt_sha256,
        "compiler_state_sha256": "4" * 64,
        "trainable_state_sha256": trainable_state_sha256,
        "query_state_sha256": "6" * 64,
        "base_branch_state_sha256": "7" * 64,
        "derivative_branch_state_sha256": "8" * 64,
        "trainable_parameters": 64_407_956,
        "compiler_buffers": 6_400,
        "independent_noncollapsed_branches": True,
        "schema": "efc-hankel-initialization/v1",
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return HankelInitializationReceipt(
        **unsigned,
        receipt_sha256=digest,
    )


def test_tensor_mapping_hash_binds_name_dtype_shape_and_values() -> None:
    left = {
        "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "b": torch.tensor([3], dtype=torch.int64),
    }
    assert tensor_mapping_sha256(left) == tensor_mapping_sha256(
        {"b": left["b"].clone(), "a": left["a"].clone()}
    )
    assert tensor_mapping_sha256(left) != tensor_mapping_sha256(
        {"a": left["a"] + 1, "b": left["b"]}
    )
    assert tensor_mapping_sha256(left) != tensor_mapping_sha256(
        {"x": left["a"], "b": left["b"]}
    )


def test_target_accounting_separates_derived_signature_credit() -> None:
    rows = _rows()
    train = tuple(row for row in rows if row.split == "train")
    receipt = create_hankel_target_accounting(
        collate_qualification_supervision(train)
    )
    assert receipt.rows == 4
    assert receipt.derived_signature_target_bits == 4 * 5_120
    assert receipt.independent_target_bits == receipt.independent_machine_target_bits
    assert receipt.supplied_target_bits > receipt.independent_target_bits


def test_optimizer_and_schedule_contracts_fail_closed() -> None:
    optimizer = create_hankel_optimizer_contract()
    assert optimizer.fused
    assert optimizer.autocast_dtype == "bfloat16"
    with pytest.raises(HankelExperimentError, match="optimizer semantics"):
        replace(optimizer, fused=False)

    schedule = create_hankel_schedule_contract(
        updates=32,
        microbatch_size=2,
        gradient_accumulation=4,
        order_seed=17,
        checkpoint_interval=16,
        metric_interval=8,
    )
    assert schedule.effective_batch_size == 8
    with pytest.raises(HankelExperimentError, match="schedule"):
        replace(schedule, effective_batch_size=7)


def test_initialization_rejects_collapsed_or_unmatched_controls() -> None:
    arm = _arm()
    receipt = _initialization(arm)
    with pytest.raises(HankelExperimentError, match="collapsed"):
        replace(
            receipt,
            derivative_branch_state_sha256=receipt.base_branch_state_sha256,
        )
    changed = _initialization(
        arm,
        trainable_state_sha256="9" * 64,
    )
    with pytest.raises(HankelExperimentError, match="initialization differs"):
        changed.assert_matched_trainable_initialization(receipt)


def test_canary_receipt_binds_train_only_and_rejects_resource_claim() -> None:
    rows = _rows()
    train = tuple(row for row in rows if row.split == "train")
    custody = create_qualification_split_custody(
        rows,
        split="train",
        candidate=_candidate(rows, "train"),
    )
    targets = create_hankel_target_accounting(
        collate_qualification_supervision(train)
    )
    arm = _arm()
    receipt = create_hankel_experiment_receipt(
        phase="measurement-canary",
        run_id="hsc-prefix-canary-test",
        arm_receipt=arm,
        initialization=_initialization(arm),
        optimizer=create_hankel_optimizer_contract(),
        schedule=create_hankel_schedule_contract(
            updates=2,
            microbatch_size=1,
            gradient_accumulation=1,
            order_seed=23,
            checkpoint_interval=1,
            metric_interval=1,
        ),
        targets=targets,
        train_custody=custody,
        train_source_bytes=sum(len(row.source) for row in train),
        tokenizer_sha256="b" * 64,
        runtime_source_manifest_sha256="c" * 64,
    )
    assert receipt.phase == "measurement-canary"
    assert receipt.resource_receipt_sha256 is None
    assert not receipt.development_visible
    assert not receipt.confirmation_visible
    assert not receipt.pretraining_authorized
    with pytest.raises(HankelExperimentError, match="resource receipt"):
        replace(receipt, phase="qualification-fit")
