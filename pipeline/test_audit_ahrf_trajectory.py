from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from audit_ahrf_trajectory import (
    AHRFTrajectoryAuditError,
    _atomic_publish_json,
    _validate_rollout,
    aggregate_trajectory_records,
    classify_trajectory_history,
    summarize_trajectory_records,
)
from autocatalytic_hysteretic_relation_field import AHRFRollout


def _history(*values: tuple[int, int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool).reshape(len(values), 1, 1, 2)


def _classify(
    history: torch.Tensor,
    *,
    halt_step: int = -1,
    safety_exhausted: bool = True,
) -> dict[str, object]:
    return classify_trajectory_history(
        history,
        torch.tensor([[[True, False]]]),
        torch.ones(1, 1, 2, dtype=torch.bool),
        halt_step=halt_step,
        safety_exhausted=safety_exhausted,
        milestone_step=4,
    )


def test_classifies_late_exact_stable_trajectory() -> None:
    result = _classify(
        _history(
            (0, 0),
            (0, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
        )
    )
    assert result == {
        "first_exact_step": 2,
        "last_exact_step": 5,
        "ever_exact": True,
        "exact_at_milestone": True,
        "final_exact": True,
        "first_false_write_after_first_exact": None,
        "halt_step": -1,
        "halt_exactness": None,
        "safety_exhausted": True,
    }


def test_detects_first_false_write_after_transient_exactness() -> None:
    result = _classify(
        _history(
            (0, 0),
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 1),
        )
    )
    assert result["first_exact_step"] == 1
    assert result["last_exact_step"] == 2
    assert result["first_false_write_after_first_exact"] == 3
    assert result["exact_at_milestone"] is False
    assert result["final_exact"] is False


def test_classifies_exact_and_inexact_learned_halts() -> None:
    exact = _classify(
        _history(
            (0, 0),
            (0, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        halt_step=2,
        safety_exhausted=False,
    )
    inexact = _classify(
        _history(
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ),
        halt_step=2,
        safety_exhausted=False,
    )
    assert exact["halt_exactness"] is True
    assert exact["final_exact"] is True
    assert inexact["halt_exactness"] is False
    assert inexact["ever_exact"] is False


@pytest.mark.parametrize(
    ("halt_step", "safety_exhausted", "match"),
    [
        (-1, False, "halt and safety"),
        (2, True, "halt and safety"),
        (0, False, "halt step"),
        (6, False, "halt step"),
    ],
)
def test_trajectory_classification_fails_closed_on_status_inconsistency(
    halt_step: int,
    safety_exhausted: bool,
    match: str,
) -> None:
    with pytest.raises(AHRFTrajectoryAuditError, match=match):
        _classify(
            _history(
                (0, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
            ),
            halt_step=halt_step,
            safety_exhausted=safety_exhausted,
        )


def test_aggregate_reports_halt_precision_recall_and_group_rates() -> None:
    records = [
        {
            "split": "train",
            "cell": "a",
            "arm": "p",
            "first_exact_step": 2,
            "last_exact_step": 4,
            "ever_exact": True,
            "exact_at_step64": True,
            "final_exact": True,
            "first_false_write_after_first_exact": None,
            "halt_step": 4,
            "halt_exactness": True,
            "safety_exhausted": False,
        },
        {
            "split": "development",
            "cell": "a",
            "arm": "p",
            "first_exact_step": None,
            "last_exact_step": None,
            "ever_exact": False,
            "exact_at_step64": False,
            "final_exact": False,
            "first_false_write_after_first_exact": None,
            "halt_step": 3,
            "halt_exactness": False,
            "safety_exhausted": False,
        },
        {
            "split": "development",
            "cell": "b",
            "arm": "p_eq",
            "first_exact_step": 1,
            "last_exact_step": 2,
            "ever_exact": True,
            "exact_at_step64": False,
            "final_exact": False,
            "first_false_write_after_first_exact": 3,
            "halt_step": -1,
            "halt_exactness": None,
            "safety_exhausted": True,
        },
    ]
    aggregate = aggregate_trajectory_records(records)
    assert aggregate["halt_precision"] == 0.5
    assert aggregate["halt_recall"] == 0.5
    assert aggregate["rates"]["ever_exact"] == pytest.approx(2 / 3)
    assert aggregate["rates"]["false_write_after_first_exact"] == pytest.approx(1 / 3)
    summary = summarize_trajectory_records(records)
    assert set(summary["by_cell"]) == {"a", "b"}
    assert set(summary["by_arm"]) == {"p", "p_eq"}
    assert summary["by_cell_arm"]["a:p"]["count"] == 2
    assert summary["by_split_cell_arm"]["development:b:p_eq"]["count"] == 1


def test_atomic_publication_is_no_clobber(tmp_path) -> None:
    output = tmp_path / "audit.json"
    payload = {"protocol": "test", "value": 7}
    digest = _atomic_publish_json(output, payload)
    assert json.loads(output.read_text(encoding="ascii")) == payload
    assert len(digest) == 64
    with pytest.raises(AHRFTrajectoryAuditError, match="already exists"):
        _atomic_publish_json(output, {"value": 8})
    assert json.loads(output.read_text(encoding="ascii")) == payload


def _valid_rollout_fixture() -> tuple[AHRFRollout, SimpleNamespace]:
    fact_history = torch.zeros(1, 3, 1, 1, 1)
    membrane_history = torch.zeros(1, 3, 1, 1, 1, 2)
    evidence_history = torch.zeros_like(fact_history)
    halted_history = torch.zeros(1, 3, dtype=torch.bool)
    rollout = AHRFRollout(
        terminal_facts=fact_history[:, -1],
        terminal_readout=torch.zeros(1, 1, 1),
        terminal_membrane=membrane_history[:, -1],
        terminal_evidence=evidence_history[:, -1],
        halt_step=torch.tensor([-1]),
        learned_halted=torch.tensor([False]),
        safety_exhausted=torch.tensor([True]),
        halt_logits=torch.zeros(1, 2),
        halt_probabilities=torch.full((1, 2), 0.5),
        write_probabilities=torch.full((1, 2, 1, 1, 1), 0.5),
        fact_history=fact_history,
        membrane_history=membrane_history,
        evidence_history=evidence_history,
        halted_history=halted_history,
    )
    graph = SimpleNamespace(
        node_features=torch.zeros(1, 1, 1),
        object_mask=torch.ones(1, 1, dtype=torch.bool),
        seed_facts=torch.zeros(1, 1, 1, 1),
    )
    return rollout, graph


def test_rollout_validation_rejects_absent_history() -> None:
    rollout, graph = _valid_rollout_fixture()
    with pytest.raises(AHRFTrajectoryAuditError, match="omitted required histories"):
        _validate_rollout(
            replace(rollout, fact_history=None),
            graph=graph,
            hard_events=True,
            enable_halt=False,
            max_steps=2,
        )


def test_rollout_validation_rejects_inconsistent_halt_history() -> None:
    rollout, graph = _valid_rollout_fixture()
    with pytest.raises(AHRFTrajectoryAuditError, match="recorded halt step"):
        _validate_rollout(
            replace(rollout, halt_step=torch.tensor([1])),
            graph=graph,
            hard_events=True,
            enable_halt=True,
            max_steps=2,
        )


def test_rollout_validation_rejects_state_change_after_halt() -> None:
    rollout, graph = _valid_rollout_fixture()
    fact_history = rollout.fact_history.clone()
    fact_history[0, 1, 0, 0, 0] = 1
    halted_history = torch.tensor([[False, True, True]])
    changed = replace(
        rollout,
        fact_history=fact_history,
        terminal_facts=fact_history[:, -1],
        halt_step=torch.tensor([1]),
        learned_halted=torch.tensor([True]),
        safety_exhausted=torch.tensor([False]),
        halted_history=halted_history,
    )
    with pytest.raises(AHRFTrajectoryAuditError, match="facts changed"):
        _validate_rollout(
            changed,
            graph=graph,
            hard_events=True,
            enable_halt=True,
            max_steps=2,
        )
