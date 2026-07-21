from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from pipeline.audit_ctaa_board import (
    AuditFailure,
    audit_board,
    candidate_subject,
    canonical_json,
)


def test_full_seedless_read_only_audit_passes() -> None:
    report = audit_board()
    checks = report["checks"]
    assert report["status"] == "pass"
    assert report["scope"] == {
        "production_seed_generated": False,
        "board_artifact_written": False,
        "jobs_launched": False,
        "audit_render_sentinel": 0,
        "audit_render_sentinel_is_production_seed": False,
    }
    assert checks["semantic_split"]["split_sizes"] == {
        "train": 9,
        "development": 9,
        "confirmation": 9,
    }
    assert checks["semantic_split"]["coordinate_balance_checks"] == 9
    assert checks["copy_algebra"] == {
        "atomic_execution_checks": 729,
        "composition_card_checks": 729,
        "closure_execution_checks": 19_683,
    }
    assert checks["stop_suffix"]["halt_boundaries"] == 41
    assert checks["stop_suffix"]["halt_trace_checks"] == 1_107
    assert checks["causal_depth"]["class_checks"] == 6
    assert checks["renderer_cosets"] == {
        "split_sizes": {
            "train": 16,
            "development": 16,
            "confirmation": 16,
        },
        "reserved_size": 16,
        "low_order_marginal_checks": 164,
    }
    assert checks["training_and_query"] == {
        "training_renderer_rows": 16,
        "scored_training_rejections": 1,
        "late_query_pair_checks": 1,
    }
    assert len(report["mutation_kills"]) == 9
    assert all(result["killed"] for result in report["mutation_kills"].values())
    preimage = dict(report)
    digest = preimage.pop("report_sha256")
    assert hashlib.sha256(canonical_json(preimage).encode()).hexdigest() == digest


def test_identity_copy_executor_is_rejected() -> None:
    subject = candidate_subject()
    mutant = replace(subject, apply_copy=lambda _action, state: state)
    with pytest.raises(AuditFailure, match="copy executor differs"):
        audit_board(mutant, include_mutation_kills=False)


def test_reversed_composition_is_rejected() -> None:
    subject = candidate_subject()

    def reversed_compose(
        after: tuple[int, ...], before: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(after[index] for index in before)

    mutant = replace(subject, compose_maps=reversed_compose)
    with pytest.raises(AuditFailure, match="copy composition differs"):
        audit_board(mutant, include_mutation_kills=False)


def test_nonabsorbing_stop_is_rejected() -> None:
    subject = candidate_subject()

    def execute_after_stop(family: object, initial: tuple[int, int, int]):
        state = initial
        states = [state]
        for event in family.schedule:
            if event != subject.stop_id:
                action = family.action_cards[event]
                state = tuple(state[index] for index in action)
            states.append(state)
        return tuple(states)

    mutant = replace(subject, execute_family=execute_after_stop)
    with pytest.raises(AuditFailure, match="STOP/suffix execution differs"):
        audit_board(mutant, include_mutation_kills=False)


def test_raw_depth_substitution_is_rejected() -> None:
    subject = candidate_subject()
    mutant = replace(subject, causal_depth=lambda family: family.depth)
    with pytest.raises(AuditFailure, match="family causal depth differs"):
        audit_board(mutant, include_mutation_kills=False)


def test_renderer_coset_leak_is_rejected() -> None:
    subject = candidate_subject()
    renderers = {split: tuple(values) for split, values in subject.renderers.items()}
    renderers["confirmation"] = (
        subject.renderers["train"][0],
        *subject.renderers["confirmation"][1:],
    )
    mutant = replace(subject, renderers=renderers)
    with pytest.raises(AuditFailure, match="confirmation renderer coset differs"):
        audit_board(mutant, include_mutation_kills=False)


def test_training_outcome_leak_is_rejected() -> None:
    subject = candidate_subject()

    def leak_outcome(row: object) -> dict[str, object]:
        record = dict(row.training_record())
        record["answer"] = 0
        return record

    mutant = replace(subject, training_record=leak_outcome)
    with pytest.raises(AuditFailure, match="training record fields differ"):
        audit_board(mutant, include_mutation_kills=False)


def test_query_leak_into_program_is_rejected() -> None:
    subject = candidate_subject()

    def leak_query(*args: object, **kwargs: object) -> object:
        row = subject.render_row(*args, **kwargs)
        return replace(row, program_source=row.program_source + row.query_source)

    mutant = replace(subject, render_row=leak_query)
    with pytest.raises(AuditFailure, match="query source leaked into program"):
        audit_board(mutant, include_mutation_kills=False)
