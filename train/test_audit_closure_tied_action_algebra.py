from __future__ import annotations

import pytest

from audit_closure_tied_action_algebra import SCHEMA, audit
from closure_tied_action_algebra import ExecutionTrace, compose_actions, execute_packet


def test_exhaustive_ctaa_cpu_audit_passes() -> None:
    report = audit()
    assert report["schema"] == SCHEMA
    assert report["all_gates_pass"] is True
    assert report["counts"] == {
        "actions": 27,
        "states": 27,
        "atomic_action_state_cells": 729,
        "action_pair_state_closure_checks": 19_683,
        "action_triple_state_associativity_checks": 531_441,
        "ordered_noncommuting_action_pairs": report["counts"][
            "ordered_noncommuting_action_pairs"
        ],
        "value_alpha_checks": 4_374,
        "storage_reindex_checks": 4_374,
        "behavioral_signatures": 27,
        "behavioral_signature_length": 26,
        "minimum_signature_hamming": report["counts"]["minimum_signature_hamming"],
        "executable_atomic_action_checks": 729,
        "halt_trace_checks": 6_561,
        "donor_changed_terminal_checks": 648,
        "donor_same_terminal_shams": 54,
    }
    assert report["counts"]["ordered_noncommuting_action_pairs"] == 588
    assert report["counts"]["minimum_signature_hamming"] > 0


def test_ctaa_cpu_audit_is_deterministic() -> None:
    assert audit() == audit()


def test_audit_kills_identity_executor_mutation() -> None:
    with pytest.raises(AssertionError, match="atomic action"):
        audit(apply_impl=lambda _action, state: state)


def test_audit_kills_reversed_composition_mutation() -> None:
    with pytest.raises(AssertionError, match="closure card"):
        audit(
            compose_impl=lambda after, before: compose_actions(before, after),
        )


def test_audit_kills_direct_state_signature_mutation() -> None:
    with pytest.raises(AssertionError, match="behavioral signature"):
        audit(
            signature_impl=lambda state, _continuations, **_kwargs: state,
        )


def test_audit_kills_single_action_executor_mutation() -> None:
    def corrupted_executor(packet, initial, *, suffix=()):
        trace = execute_packet(packet, initial, suffix=suffix)
        if packet.actions == ((2, 2, 2),) and packet.halt_at == 1 and not suffix:
            return ExecutionTrace(
                states=(initial, initial),
                halted=trace.halted,
            )
        return trace

    with pytest.raises(AssertionError, match="executable atomic action"):
        audit(execute_impl=corrupted_executor)
