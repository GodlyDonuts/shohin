from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from build_ctaa_runtime_intervention_plan import _parse_program
from ctaa_intervention_protocol import (
    MANDATORY_OPERATIONS,
    AnchorOperationCommitment,
    make_anchor_operation_commitment,
    make_runtime_intervention_plan,
)
from ctaa_runtime_plan_replay import (
    ReplayValidationError,
    load_runtime_replay_rows,
    replay_attempt,
    replay_runtime_plan,
)
from test_build_ctaa_runtime_intervention_plan import (
    _build,
    frozen_inputs as _frozen_inputs_fixture,  # noqa: F401
)


@pytest.fixture(scope="module")
def replay_inputs(request: pytest.FixtureRequest):
    frozen_inputs = request.getfixturevalue("_frozen_inputs_fixture")
    output = Path(frozen_inputs["root"]) / "replay-plan.json"
    plan = _build(frozen_inputs, output)
    tokenizer = Tokenizer.from_file(str(frozen_inputs["tokenizer_path"]))
    programs = {
        row["family_id"]: row["program_source"]
        for row in (
            json.loads(line)
            for line in Path(frozen_inputs["program_path"]).read_text().splitlines()
        )
    }
    queries = {
        row["family_id"]: row["query_source"]
        for row in (
            json.loads(line)
            for line in Path(frozen_inputs["query_path"]).read_text().splitlines()
        )
    }
    rows = {
        anchor.anchor_id: _parse_program(
            anchor.family_id,
            programs[anchor.family_id],
            queries[anchor.family_id],
            tokenizer,
            plan.bindings.partition.value,
        )
        for anchor in plan.anchors
    }
    return plan, rows


@pytest.fixture(scope="module")
def replayed(replay_inputs):
    plan, rows = replay_inputs
    return replay_runtime_plan(plan, rows)


@pytest.mark.parametrize("operation", MANDATORY_OPERATIONS)
def test_every_mandatory_operation_replays(operation: str, replayed) -> None:
    counts = Counter(result.operation for result in replayed.results)
    assert counts[operation] == 864
    assert replayed.attempt_count == 25_056


def test_immutable_source_loader_reconstructs_selected_panel(
    replay_inputs, request: pytest.FixtureRequest
) -> None:
    plan, expected = replay_inputs
    frozen_inputs = request.getfixturevalue("_frozen_inputs_fixture")
    rows = load_runtime_replay_rows(
        plan=plan,
        program_path=Path(frozen_inputs["program_path"]),
        query_path=Path(frozen_inputs["query_path"]),
        tokenizer_path=Path(frozen_inputs["tokenizer_path"]),
    )
    assert rows == expected


def _replace_attempt(plan, index: int, payload: dict[str, object], **results):
    old = plan.attempts[index]
    replacement = make_anchor_operation_commitment(
        attempt_index=old.attempt_index,
        operation=old.operation,
        operation_sha256=old.operation_sha256,
        anchor_id=old.anchor_id,
        donor_anchor_id=old.donor_anchor_id,
        mutation_payload=payload,
        resulting_program_source_sha256=results.get(
            "program", old.resulting_program_source_sha256
        ),
        resulting_query_source_sha256=results.get(
            "query", old.resulting_query_source_sha256
        ),
        resulting_packet_sha256=results.get("packet", old.resulting_packet_sha256),
    )
    attempts = list(plan.attempts)
    attempts[index] = replacement
    return make_runtime_intervention_plan(
        bindings=plan.bindings,
        anchors=plan.anchors,
        donor_derangements=plan.donor_derangements,
        attempts=attempts,
    )


def _payload(attempt: AnchorOperationCommitment) -> dict[str, object]:
    return json.loads(attempt.mutation_payload_json)


def _index(plan, operation: str) -> int:
    return next(
        attempt.attempt_index
        for attempt in plan.attempts
        if attempt.operation == operation
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_payload_field_mutations_fail_closed(replay_inputs, mutation: str) -> None:
    plan, rows = replay_inputs
    index = _index(plan, "entity_recode")
    payload = _payload(plan.attempts[index])
    if mutation == "missing":
        del payload["old_to_new"]
    else:
        payload["uncommitted"] = 1
    changed = _replace_attempt(plan, index, payload)
    with pytest.raises(ReplayValidationError, match="payload schema"):
        replay_attempt(changed, rows, index)


def test_rehashed_noop_recipe_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    index = _index(plan, "entity_recode")
    attempt = plan.attempts[index]
    payload = _payload(attempt)
    row = rows[attempt.anchor_id]
    payload["old_to_new"] = dict(zip(row.symbols, row.symbols, strict=True))
    changed = _replace_attempt(
        plan,
        index,
        payload,
        program=plan.anchors[
            plan.bindings.batch_order.index(attempt.anchor_id)
        ].program_source_sha256,
    )
    with pytest.raises(ReplayValidationError, match="entity recode"):
        replay_attempt(changed, rows, index)


def test_rehashed_result_hash_substitution_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    index = _index(plan, "post_stop_poison")
    changed = _replace_attempt(
        plan,
        index,
        _payload(plan.attempts[index]),
        packet="0" * 64,
    )
    with pytest.raises(ReplayValidationError, match="resulting artifact hash"):
        replay_attempt(changed, rows, index)


def test_payload_hash_substitution_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    index = _index(plan, "source_deletion")
    attempts = list(plan.attempts)
    attempts[index] = replace(attempts[index], mutation_payload_sha256="0" * 64)
    corrupt = replace(plan, attempts=tuple(attempts))
    with pytest.raises(ReplayValidationError, match="plan validation"):
        replay_attempt(corrupt, rows, index)


def test_donor_substitution_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    index = _index(plan, "packet_transplant")
    attempts = list(plan.attempts)
    attempts[index] = replace(
        attempts[index], donor_anchor_id=attempts[index].anchor_id
    )
    corrupt = replace(plan, attempts=tuple(attempts))
    with pytest.raises(ReplayValidationError, match="plan validation"):
        replay_attempt(corrupt, rows, index)


def test_source_registry_substitution_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    first, second = plan.bindings.batch_order[:2]
    substituted = dict(rows)
    substituted[first] = rows[second]
    with pytest.raises(ReplayValidationError, match="anchor .* differs"):
        replay_attempt(plan, substituted, 0)


def test_unknown_operation_fails_closed(replay_inputs) -> None:
    plan, rows = replay_inputs
    attempts = list(plan.attempts)
    attempts[0] = replace(attempts[0], operation="unknown_operation")
    corrupt = replace(plan, attempts=tuple(attempts))
    with pytest.raises(ReplayValidationError, match="plan validation"):
        replay_attempt(corrupt, rows, 0)
