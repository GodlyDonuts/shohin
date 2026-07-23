from __future__ import annotations

import itertools

import pytest

from pipeline.ctaa_binding_identification import (
    ADJACENT_TRANSPOSITIONS,
    BINDINGS,
    DELAYS,
    IDENTITY,
    PROBE_TAPE,
    BindingIdentificationError,
    alternating_group_audit,
    apply_rebinding,
    build_report,
    compose,
    delayed_read,
    permutation_parity,
    rebinding_dynamics_audit,
    resolve,
    write_delete_delay_read_audit,
)


def test_a4_odd_split_is_exact_and_locally_balanced() -> None:
    audit = alternating_group_audit()
    assert audit["train_even_count"] == 12
    assert audit["confirmation_odd_count"] == 12
    assert audit["train_local_marginals"] == ((3, 3, 3, 3),) * 4
    assert audit["confirmation_local_marginals"] == ((3, 3, 3, 3),) * 4


def test_permutation_composition_and_parity_are_consistent() -> None:
    for left, right in itertools.product(BINDINGS, repeat=2):
        product = compose(left, right)
        assert permutation_parity(product) == (
            permutation_parity(left) ^ permutation_parity(right)
        )
        assert resolve(product, PROBE_TAPE) == tuple(
            left[right[opcode]] for opcode in PROBE_TAPE
        )


def test_write_delete_delay_read_is_delay_invariant_and_causal() -> None:
    audit = write_delete_delay_read_audit()
    assert audit["receipt_count"] == len(BINDINGS) * len(DELAYS)
    assert audit["identity_reset_differences"] == 23
    assert audit["donor_following"] == 24
    binding = BINDINGS[17]
    receipts = [
        delayed_read(binding, delay, distractor_seed=123) for delay in DELAYS
    ]
    assert {receipt.resolved_tape for receipt in receipts} == {
        resolve(binding, PROBE_TAPE)
    }
    assert len({receipt.distractor_checksum for receipt in receipts}) == len(DELAYS)


def test_adjacent_transpositions_generate_s4_without_future_leakage() -> None:
    audit = rebinding_dynamics_audit(max_cues=6)
    assert audit["reachable_binding_count"] == 24
    assert audit["sequence_count"] == sum(3**length for length in range(1, 7))
    state = IDENTITY
    for cue in ADJACENT_TRANSPOSITIONS:
        state = apply_rebinding(state, cue)
    assert state != IDENTITY


def test_invalid_binding_and_unregistered_delay_fail_closed() -> None:
    with pytest.raises(BindingIdentificationError, match="permutation"):
        resolve((0, 0, 2, 3), PROBE_TAPE)
    with pytest.raises(BindingIdentificationError, match="preregistered"):
        delayed_read(IDENTITY, 7, distractor_seed=1)


def test_report_is_deterministic_and_claim_bounded() -> None:
    left = build_report()
    right = build_report()
    assert left == right
    assert left["schema"] == "r12_ctaa_binding_identification_mechanics_v1"
    assert left["claim_boundary"] == "finite_mechanics_only_no_neural_capability_claim"
    assert len(left["payload_sha256"]) == 64
