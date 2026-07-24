from __future__ import annotations

from dataclasses import replace
import gc

import pytest

from episode_functor_capacity_lanes import build_hankel_shift_capacity_lane
from episode_functor_hankel_arm import (
    HankelArmReceiptError,
    create_hankel_arm_receipt,
)
from episode_functor_qualification_loss import (
    EFCHankelQualificationLoss,
    HankelQualificationLossWeights,
)


def _receipt(mode: str):
    compiler, query, _ = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        incidence_mode=mode,
        random_seed="hsc-arm-receipt-control-v1",
    )
    receipt = create_hankel_arm_receipt(
        compiler=compiler,
        query_parser=query,
        objective=EFCHankelQualificationLoss(),
    )
    del compiler, query
    gc.collect()
    return receipt


def _receipt_with_seed(mode: str, seed: str):
    compiler, query, _ = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        incidence_mode=mode,
        random_seed=seed,
    )
    receipt = create_hankel_arm_receipt(
        compiler=compiler,
        query_parser=query,
        objective=EFCHankelQualificationLoss(),
    )
    del compiler, query
    gc.collect()
    return receipt


def _direct_receipt():
    compiler, query, _ = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        incidence_mode="prefix",
        random_seed="hsc-arm-receipt-control-v1",
        decode_mode="direct-base",
    )
    receipt = create_hankel_arm_receipt(
        compiler=compiler,
        query_parser=query,
        objective=EFCHankelQualificationLoss(),
    )
    del compiler, query
    gc.collect()
    return receipt


def test_hankel_arm_receipt_binds_exact_architecture_and_objective() -> None:
    receipt = _receipt("prefix")
    assert receipt.arm_name == "hsc-prefix-treatment"
    assert receipt.word_count == 40
    assert receipt.base_signature_cells_per_example == 640
    assert receipt.derivative_signature_cells_per_example == 1_920
    assert receipt.signature_target_bits_per_example == 5_120
    assert receipt.source_compiler_parameters == 64_407_956
    assert receipt.projector_parameters == 19_717_124
    assert receipt.query_parser_parameters == 6_003_489
    assert receipt.complete_parameters == 195_493_109
    assert receipt.headroom == 4_506_891
    assert receipt.to_json_bytes().endswith(b"\n")


def test_hankel_controls_are_isoparametric_but_transform_distinct() -> None:
    prefix = _receipt("prefix")
    random = _receipt("random")
    commutative = _receipt("commutative")
    prefix.assert_incidence_control(random)
    prefix.assert_incidence_control(commutative)
    direct = _direct_receipt()
    prefix.assert_decode_control(direct)
    assert direct.arm_name == "hsc-dual-direct-decode-control"
    assert len(
        {
            prefix.incidence_sha256,
            random.incidence_sha256,
            commutative.incidence_sha256,
        }
    ) == 3


def test_hankel_control_pairing_rejects_two_factor_change() -> None:
    random = _receipt("random")
    direct_prefix = _direct_receipt()
    with pytest.raises(
        HankelArmReceiptError,
        match="exactly one causal factor",
    ):
        random.assert_isoparametric(direct_prefix)
    with pytest.raises(
        HankelArmReceiptError,
        match="exactly one causal factor",
    ):
        random.assert_incidence_control(direct_prefix)


def test_hankel_incidence_control_rejects_seed_only_noop() -> None:
    prefix_a = _receipt_with_seed("prefix", "unused-prefix-seed-a")
    prefix_b = _receipt_with_seed("prefix", "unused-prefix-seed-b")
    assert prefix_a.incidence_sha256 == prefix_b.incidence_sha256
    assert prefix_a.random_seed_sha256 != prefix_b.random_seed_sha256
    with pytest.raises(
        HankelArmReceiptError,
        match="exactly one causal factor",
    ):
        prefix_a.assert_incidence_control(prefix_b)


def test_hankel_receipt_rejects_hash_or_objective_drift() -> None:
    receipt = _receipt("prefix")
    with pytest.raises(HankelArmReceiptError, match="receipt hash"):
        replace(receipt, receipt_sha256="0" * 64)

    compiler, query, _ = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        incidence_mode="prefix",
    )
    changed = create_hankel_arm_receipt(
        compiler=compiler,
        query_parser=query,
        objective=EFCHankelQualificationLoss(
            hankel_weights=HankelQualificationLossWeights(
                state_separation=0.5,
            )
        ),
    )
    del compiler, query
    gc.collect()
    assert changed.objective_sha256 != receipt.objective_sha256
    with pytest.raises(HankelArmReceiptError, match="resources differ"):
        receipt.assert_isoparametric(changed)
