from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import pytest

import pipeline.audit_episode_functor_quotient_fisher_retraction as qfcr
from pipeline.acw_nist_beacon import verify_pulse
from pipeline.audit_episode_functor_quotient_fisher_retraction import (
    ARMS,
    Entry,
    MARGINS,
    _arm_receipt,
    _canonical_json_bytes,
    _decision,
    _fresh_board,
    _recoded_entries,
    _run_arm,
)
from pipeline.audit_episode_functor_acso_oracle_recovery import (
    _board,
    _publish,
    _reserve_output,
    deep_fault_inventory,
)


def _receipt(
    exact: float,
    *,
    median: float = 4.0,
) -> dict[str, object]:
    return {
        "exact_recovery": exact,
        "row_recovery": exact,
        "final_ties": 0,
        "monotonic_violations": 0,
        "recoded_monotonic_violations": 0,
        "recoding_decision_mismatches": 0,
        "maximum_recoding_innovation_delta": 0.0,
        "median_first_exact_cycle": median,
        "minimum_world_exact_recovery": exact,
        "world_exact_recovery": {"fixture": exact},
        "valid": True,
    }


def _margins(
    treatment: float,
    euclidean: float,
    *,
    treatment_median: float = 4.0,
    euclidean_median: float = 4.0,
) -> list[dict[str, object]]:
    return [
        {
            "margin": margin,
            "arms": {
                "qf_causal": _receipt(
                    treatment,
                    median=treatment_median,
                ),
                "euclidean_equal_step": _receipt(
                    euclidean,
                    median=euclidean_median,
                ),
                "qf_small_step": _receipt(0.0, median=5.0),
                "qf_one_step": _receipt(0.0, median=5.0),
            },
        }
        for margin in MARGINS
    ]


def test_arm_contract_is_exact() -> None:
    assert [arm.name for arm in ARMS] == [
        "qf_causal",
        "euclidean_equal_step",
        "qf_small_step",
        "qf_one_step",
    ]
    assert ARMS[0].step == ARMS[1].step == 1.0
    assert ARMS[2].step == 0.1


def test_equal_endpoint_is_explicit_non_attribution() -> None:
    decision, mechanics, attributed = _decision(
        _margins(1.0, 1.0),
        bindings_pass=True,
    )
    assert decision == "step_scale_sufficient_qfcr_not_attributed"
    assert mechanics
    assert not attributed


def test_geometry_requires_gap_and_earlier_recovery() -> None:
    receipts = _margins(
        1.0,
        0.90,
        treatment_median=3.0,
        euclidean_median=4.0,
    )
    receipts[0]["arms"]["euclidean_equal_step"] = _receipt(
        1.0,
        median=4.0,
    )
    decision, mechanics, attributed = _decision(
        receipts,
        bindings_pass=True,
    )
    assert decision == "qfcr_geometry_attributed"
    assert mechanics
    assert attributed


def test_binding_failure_is_mechanics_no_go() -> None:
    decision, mechanics, attributed = _decision(
        _margins(1.0, 0.0),
        bindings_pass=False,
    )
    assert decision == "qfcr_mechanics_no_go"
    assert not mechanics
    assert not attributed


def test_real_single_fault_arm_and_recoding_receipt() -> None:
    world_id, machine = next(
        item
        for item in sorted(_board().items())
        if deep_fault_inventory(item[1])
    )
    entry = Entry(
        world_id=world_id,
        split="fixture",
        machine=machine,
        fault=deep_fault_inventory(machine)[0],
    )
    original = _run_arm(
        [entry],
        margin=0.2,
        arm=ARMS[0],
    )
    recoded = _run_arm(
        _recoded_entries([entry]),
        margin=0.2,
        arm=ARMS[0],
    )
    receipt = _arm_receipt(original, recoded, [entry])
    assert receipt["exact_recovery"] == 1.0
    assert receipt["minimum_world_exact_recovery"] == 1.0
    assert receipt["recoding_decision_mismatches"] == 0
    assert receipt["maximum_recoding_innovation_delta"] <= 1e-5


def test_renderer_rows_are_deduplicated_to_unique_worlds() -> None:
    entries, manifest, eligible, observed = _fresh_board(
        "qfcr-unit-dedup-v1"
    )
    assert len(manifest) == 200
    assert observed == {
        "confirmation": 24,
        "development": 32,
        "mechanics": 48,
        "train": 96,
    }
    assert sum(int(row["source_count"]) for row in manifest) > 200
    identities = {
        (
            entry.world_id,
            entry.fault.action,
            entry.fault.state,
            entry.fault.wrong,
        )
        for entry in entries
    }
    assert len(identities) == len(entries)
    assert sum(eligible.values()) > 0


def test_source_failure_precedes_board_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qfcr, "_reservation_valid", lambda _: True)
    monkeypatch.setattr(
        qfcr,
        "_runtime_source_paths",
        lambda: ("fixture.py",),
    )
    monkeypatch.setattr(
        qfcr,
        "_source_receipt",
        lambda _: ("fixture", [], False),
    )

    def forbidden(_: str):
        raise AssertionError("board generated after source failure")

    monkeypatch.setattr(qfcr, "_fresh_board", forbidden)
    with pytest.raises(qfcr.QFCRAuditError):
        qfcr.audit(
            object(),
            authorization_path=Path("unused"),
            beacon_snapshot_path=Path("unused"),
        )


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        _canonical_json_bytes({"not_finite": math.nan})


def test_trusted_nist_pin_authenticates_both_fixture_pulses() -> None:
    snapshot = json.loads(
        (
            qfcr.ROOT
            / "pipeline/testdata/acw_nist_beacon_snapshot.json"
        ).read_text("ascii")
    )
    certificate = snapshot["certificate_pem"].encode("ascii")
    previous = snapshot["previous_pulse"]
    pulse = snapshot["pulse"]
    previous_receipt = verify_pulse(
        previous,
        certificate,
        expected_chain_index=int(previous["chainIndex"]),
        expected_pulse_index=int(previous["pulseIndex"]),
    )
    pulse_receipt = verify_pulse(
        pulse,
        certificate,
        previous_pulse=previous,
        expected_chain_index=int(pulse["chainIndex"]),
        expected_pulse_index=int(pulse["pulseIndex"]),
    )
    assert (
        previous_receipt["certificate_der_sha512"]
        == qfcr.TRUSTED_NIST_CERTIFICATE_DER_SHA512
    )
    assert (
        pulse_receipt["certificate_der_sha512"]
        == qfcr.TRUSTED_NIST_CERTIFICATE_DER_SHA512
    )
    assert pulse_receipt["previous_link"] is not None


class _FakeGitHubResponse:
    def __init__(self, payload: bytes) -> None:
        self.status = 200
        self._payload = payload
        self.headers = {"Date": "Fri, 24 Jul 2026 07:00:00 GMT"}

    def __enter__(self) -> "_FakeGitHubResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, _: int) -> bytes:
        return self._payload


def test_github_receipt_binds_both_public_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def event(
        identity: str,
        head: str,
        created_at: str,
    ) -> dict[str, object]:
        return {
            "id": identity,
            "type": "PushEvent",
            "public": True,
            "created_at": created_at,
            "repo": {"name": qfcr.GITHUB_REPOSITORY},
            "payload": {
                "ref": "refs/heads/main",
                "before": (
                    "source-head"
                    if head == "authorization-head"
                    else (
                        "anchor-head"
                        if head == "source-head"
                        else qfcr.SOURCE_ANCHOR_PARENT_COMMIT
                    )
                ),
                "size": 1,
                "distinct_size": 1,
                "head": head,
            },
        }

    payload = _canonical_json_bytes(
        [
            event("2", "authorization-head", "2026-07-24T00:01:00Z"),
            event("1", "source-head", "2026-07-24T00:00:00Z"),
            event("0", "anchor-head", "2026-07-23T23:59:00Z"),
        ]
    )
    monkeypatch.setattr(
        qfcr.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeGitHubResponse(payload),
    )
    receipt = qfcr._github_push_receipts(
        anchor_head="anchor-head",
        source_head="source-head",
        authorization_head="authorization-head",
        branch="main",
    )
    assert receipt["source"]["github_push_event_head"] == "source-head"
    assert (
        receipt["authorization"]["github_push_event_head"]
        == "authorization-head"
    )
    for label in ("anchor", "source", "authorization"):
        assert (
            receipt[label]["github_push_event_repository"]
            == qfcr.GITHUB_REPOSITORY
        )
        assert len(
            receipt[label]["github_push_event_payload_sha256"]
        ) == 64


def test_github_receipt_requires_distinct_source_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _canonical_json_bytes(
        [
            {
                "id": "2",
                "type": "PushEvent",
                "public": True,
                "created_at": "2026-07-24T00:01:00Z",
                "repo": {"name": qfcr.GITHUB_REPOSITORY},
                "payload": {
                    "ref": "refs/heads/main",
                    "before": "source-head",
                    "size": 1,
                    "distinct_size": 1,
                    "head": "authorization-head",
                },
            }
        ]
    )
    monkeypatch.setattr(
        qfcr.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeGitHubResponse(payload),
    )
    with pytest.raises(qfcr.QFCRAuditError):
        qfcr._github_push_receipts(
            anchor_head="anchor-head",
            source_head="source-head",
            authorization_head="authorization-head",
            branch="main",
        )


def test_github_receipt_rejects_authorization_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _canonical_json_bytes(
        [
            {
                "id": "3",
                "type": "PushEvent",
                "public": True,
                "created_at": "2026-07-24T00:02:00Z",
                "repo": {"name": qfcr.GITHUB_REPOSITORY},
                "payload": {
                    "ref": "refs/heads/main",
                    "before": "source-head",
                    "head": "authorization-head",
                    "size": 1,
                    "distinct_size": 1,
                },
            },
            {
                "id": "2",
                "type": "PushEvent",
                "public": True,
                "created_at": "2026-07-24T00:01:00Z",
                "repo": {"name": qfcr.GITHUB_REPOSITORY},
                "payload": {
                    "ref": "refs/heads/main",
                    "before": "source-head",
                    "head": "retired-authorization",
                    "size": 1,
                    "distinct_size": 1,
                },
            },
            {
                "id": "1",
                "type": "PushEvent",
                "public": True,
                "created_at": "2026-07-24T00:00:00Z",
                "repo": {"name": qfcr.GITHUB_REPOSITORY},
                "payload": {
                    "ref": "refs/heads/main",
                    "before": "anchor-head",
                    "head": "source-head",
                    "size": 1,
                    "distinct_size": 1,
                },
            },
            {
                "id": "0",
                "type": "PushEvent",
                "public": True,
                "created_at": "2026-07-23T23:59:00Z",
                "repo": {"name": qfcr.GITHUB_REPOSITORY},
                "payload": {
                    "ref": "refs/heads/main",
                    "before": qfcr.SOURCE_ANCHOR_PARENT_COMMIT,
                    "head": "anchor-head",
                    "size": 1,
                    "distinct_size": 1,
                },
            },
        ]
    )
    monkeypatch.setattr(
        qfcr.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeGitHubResponse(payload),
    )
    with pytest.raises(qfcr.QFCRAuditError):
        qfcr._github_push_receipts(
            anchor_head="anchor-head",
            source_head="source-head",
            authorization_head="authorization-head",
            branch="main",
        )


def test_runtime_source_closure_contains_only_real_files() -> None:
    paths = qfcr._runtime_source_paths()
    assert qfcr.PROTOCOL_FILE in paths
    assert all((qfcr.ROOT / path).is_file() for path in paths)


def test_missing_arm_fails_closed() -> None:
    receipts = _margins(1.0, 1.0)
    del receipts[0]["arms"]["qf_one_step"]
    with pytest.raises(qfcr.QFCRAuditError):
        _decision(receipts, bindings_pass=True)


def test_shared_no_clobber_publication_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qfcr-fixture.json"
    reservation = _reserve_output(output)
    _publish(
        reservation,
        {"schema": "qfcr-publication-fixture/v1", "value": 1},
    )
    assert json.loads(output.read_text("ascii")) == {
        "schema": "qfcr-publication-fixture/v1",
        "value": 1,
    }
    assert output.with_suffix(".json.reserve").is_file()
