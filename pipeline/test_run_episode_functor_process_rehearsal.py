from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import platform
import subprocess

import pytest

from pipeline.acw_nist_beacon import (
    PULSE_PROTOCOL,
    canonical_json_bytes as beacon_json_bytes,
    sha256_bytes,
    verify_pulse,
)
from pipeline.episode_functor_process_custody import canonical_json_bytes
from pipeline.run_episode_functor_process_rehearsal import (
    AUTHORIZATION_SCHEMA,
    RehearsalError,
    _closed_tree_receipt,
    _git_receipt,
    _github_push_receipt,
    _load_beacon_snapshot,
    _new_output_root,
    frozen_source_receipt,
    run_rehearsal,
)
import pipeline.run_episode_functor_process_rehearsal as rehearsal


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "pipeline" / "testdata" / "acw_nist_beacon_snapshot.json"
UNMOCKED_CUSTODY_AVAILABLE = platform.system() == "Darwin" or (
    platform.system() == "Linux"
    and Path("/usr/bin/bwrap").is_file()
    and Path("/usr/bin/python3").is_file()
)


def _authorization(
    path: Path,
    *,
    source_root: str,
    target_pulse_index: int,
    certificate_der_sha512: str,
) -> Path:
    value = {
        "candidate_source_root": source_root,
        "claim_boundary": "cpu-custody-mechanics-only",
        "machine_bytes": 1_536,
        "minimum_pulse_delay_seconds": 60,
        "nist_certificate_der_sha512": certificate_der_sha512,
        "nist_chain_index": 2,
        "pulse_selection_rule": "exact-precommitted-chain-2-pulse-index",
        "schema": AUTHORIZATION_SCHEMA,
        "source_languages": [
            "canonical-json-events",
            "strict-line-events",
            "cycle-program",
        ],
        "target_pulse_index": target_pulse_index,
    }
    path.write_bytes(canonical_json_bytes(value))
    return path


def _verified_snapshot(path: Path) -> Path:
    raw = json.loads(SNAPSHOT.read_bytes())
    previous_verification = verify_pulse(
        raw["previous_pulse"],
        raw["certificate_pem"].encode("ascii"),
        expected_chain_index=raw["previous_pulse"]["chainIndex"],
        expected_pulse_index=raw["previous_pulse"]["pulseIndex"],
        expected_timestamp=raw["previous_pulse"]["timeStamp"],
    )
    verification = verify_pulse(
        raw["pulse"],
        raw["certificate_pem"].encode("ascii"),
        previous_pulse=raw["previous_pulse"],
        expected_chain_index=raw["pulse"]["chainIndex"],
        expected_pulse_index=raw["pulse"]["pulseIndex"],
    )
    value = {
        "certificate_pem": raw["certificate_pem"],
        "previous_pulse": raw["previous_pulse"],
        "previous_verification": previous_verification,
        "protocol": PULSE_PROTOCOL,
        "pulse": raw["pulse"],
        "verification": verification,
    }
    value["payload_sha256"] = sha256_bytes(beacon_json_bytes(value))
    path.write_bytes(beacon_json_bytes(value) + b"\n")
    return path


def _mock_official_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.loads(SNAPSHOT.read_bytes())
    certificate = raw["certificate_pem"].encode("ascii")
    monkeypatch.setattr(rehearsal, "fetch_certificate", lambda _identifier: certificate)


def _fixture_certificate_der_sha512() -> str:
    raw = json.loads(SNAPSHOT.read_bytes())
    return str(
        verify_pulse(
            raw["pulse"],
            raw["certificate_pem"].encode("ascii"),
            previous_pulse=raw["previous_pulse"],
            expected_chain_index=raw["pulse"]["chainIndex"],
            expected_pulse_index=raw["pulse"]["pulseIndex"],
        )["certificate_der_sha512"]
    )


@pytest.mark.skipif(
    not UNMOCKED_CUSTODY_AVAILABLE,
    reason="exact macOS Seatbelt or Linux Bubblewrap/Python prerequisites are absent",
)
def test_signed_process_rehearsal_runs_three_blind_compilers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = json.loads(SNAPSHOT.read_bytes())
    pulse_time = datetime.fromisoformat(
        snapshot["pulse"]["timeStamp"].replace("Z", "+00:00")
    )
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(rehearsal, "AUTHORIZATION_PATH", authorization)
    _mock_official_certificate(monkeypatch)
    monkeypatch.setattr(
        rehearsal,
        "_git_receipt",
        lambda **_kwargs: {
            "branch": "test",
            "github_push_event_created_at": (
                pulse_time - timedelta(seconds=120)
            ).isoformat(),
            "head": "ab" * 20,
            "origin_head_verified": True,
        },
    )
    source = frozen_source_receipt()
    authorization = _authorization(
        authorization,
        source_root=str(source["source_root"]),
        target_pulse_index=snapshot["pulse"]["pulseIndex"],
        certificate_der_sha512=_fixture_certificate_der_sha512(),
    )
    verified_snapshot = _verified_snapshot(tmp_path / "snapshot.json")
    output = tmp_path / "artifact"
    report = run_rehearsal(
        authorization_path=authorization,
        beacon_snapshot_path=verified_snapshot,
        output_root=output,
    )
    assert report["future_pulse_after_publication"]
    assert report["candidate_process_count"] == 3
    assert report["exact_matches"] == 3
    assert report["source_language_count"] == 3
    assert report["blindness_probe"]["all_gates_pass"]
    assert len(report["machine_sha256"]) == 64
    persisted = json.loads((output / "final_report.json").read_bytes())
    assert persisted == report


def test_process_rehearsal_rejects_wrong_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    monkeypatch.setattr(rehearsal, "AUTHORIZATION_PATH", authorization_path)
    authorization = _authorization(
        authorization_path,
        source_root="00" * 32,
        target_pulse_index=1,
        certificate_der_sha512=_fixture_certificate_der_sha512(),
    )
    with pytest.raises(RehearsalError, match="authorization values"):
        run_rehearsal(
            authorization_path=authorization,
            beacon_snapshot_path=SNAPSHOT,
            output_root=tmp_path / "artifact",
        )


def test_process_rehearsal_rejects_valid_uncommitted_pulse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = json.loads(SNAPSHOT.read_bytes())
    pulse_time = datetime.fromisoformat(
        snapshot["pulse"]["timeStamp"].replace("Z", "+00:00")
    )
    authorization_path = tmp_path / "authorization.json"
    monkeypatch.setattr(rehearsal, "AUTHORIZATION_PATH", authorization_path)
    _mock_official_certificate(monkeypatch)
    monkeypatch.setattr(
        rehearsal,
        "_git_receipt",
        lambda **_kwargs: {
            "branch": "test",
            "github_push_event_created_at": (
                pulse_time - timedelta(seconds=120)
            ).isoformat(),
            "head": "cd" * 20,
            "origin_head_verified": True,
        },
    )
    source = frozen_source_receipt()
    authorization = _authorization(
        authorization_path,
        source_root=str(source["source_root"]),
        target_pulse_index=snapshot["pulse"]["pulseIndex"] + 1,
        certificate_der_sha512=_fixture_certificate_der_sha512(),
    )
    verified_snapshot = _verified_snapshot(tmp_path / "snapshot.json")
    with pytest.raises(RehearsalError, match="future rule"):
        run_rehearsal(
            authorization_path=authorization,
            beacon_snapshot_path=verified_snapshot,
            output_root=tmp_path / "artifact",
        )


def test_beacon_snapshot_rejects_unpinned_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_official_certificate(monkeypatch)
    verified_snapshot = _verified_snapshot(tmp_path / "snapshot.json")
    with pytest.raises(RehearsalError, match="certificate differs"):
        _load_beacon_snapshot(
            verified_snapshot,
            expected_certificate_der_sha512="00" * 64,
        )


def test_beacon_snapshot_rejects_certificate_not_served_by_nist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_snapshot = _verified_snapshot(tmp_path / "snapshot.json")
    monkeypatch.setattr(
        rehearsal,
        "fetch_certificate",
        lambda _identifier: b"not a certificate",
    )
    with pytest.raises(ValueError, match="NIST certificate"):
        _load_beacon_snapshot(
            verified_snapshot,
            expected_certificate_der_sha512=_fixture_certificate_der_sha512(),
        )


def test_github_push_receipt_requires_exact_public_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "12" * 20
    event = {
        "created_at": "2026-07-23T20:00:00Z",
        "id": "123456789",
        "payload": {"head": head, "ref": "refs/heads/main"},
        "public": True,
        "repo": {"name": "GodlyDonuts/shohin"},
        "type": "PushEvent",
    }

    class Response:
        status = 200
        headers = {"Date": "Thu, 23 Jul 2026 20:00:01 GMT"}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _maximum: int) -> bytes:
            return json.dumps([event]).encode("utf-8")

    monkeypatch.setattr(
        rehearsal.urllib.request, "urlopen", lambda *_a, **_k: Response()
    )
    receipt = _github_push_receipt(head=head, branch="main")
    assert receipt["github_push_event_id"] == event["id"]
    with pytest.raises(RehearsalError, match="matching public"):
        _github_push_receipt(head="34" * 20, branch="main")


def test_git_receipt_binds_authorization_and_sources_to_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    origin = tmp_path / "origin.git"
    repository.mkdir()
    subprocess.run(("git", "init", "--bare", str(origin)), check=True)
    subprocess.run(("git", "init", "-b", "main"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.name", "EFC Test"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "efc@example.invalid"),
        cwd=repository,
        check=True,
    )
    authorization = (
        repository / "artifacts" / "r12" / "efc_process_custody_authorization_v1.json"
    )
    source = repository / "pipeline" / "candidate.py"
    authorization.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    authorization.write_bytes(b'{"schema":"test"}\n')
    source.write_bytes(b"source\n")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-m", "freeze"),
        cwd=repository,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ("git", "remote", "add", "origin", str(origin)),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "push", "-u", "origin", "main"),
        cwd=repository,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    monkeypatch.setattr(rehearsal, "ROOT", repository)
    monkeypatch.setattr(
        rehearsal,
        "_github_push_receipt",
        lambda **_kwargs: {
            "github_push_event_created_at": "2026-07-23T20:00:00Z",
            "github_push_event_id": "test",
        },
    )
    monkeypatch.chdir(repository)
    receipt = _git_receipt(
        authorization_path=authorization.relative_to(repository),
        source_receipt={
            "files": {"pipeline/candidate.py": rehearsal.sha256_bytes(b"source\n")}
        },
    )
    assert receipt["authorization_in_head_verified"]
    assert receipt["origin_head_verified"]
    assert receipt["source_blobs_in_head_verified"]

    source.write_bytes(b"mutated\n")
    with pytest.raises(RehearsalError, match="worktree must be clean"):
        _git_receipt(
            authorization_path=authorization.relative_to(repository),
            source_receipt={
                "files": {"pipeline/candidate.py": rehearsal.sha256_bytes(b"source\n")}
            },
        )


def test_closed_tree_receipt_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"target\n")
    (tmp_path / "linked.txt").symlink_to(target)
    with pytest.raises(RehearsalError, match="nonregular"):
        _closed_tree_receipt(tmp_path)


def test_new_output_root_rejects_dangling_symlink(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(RehearsalError, match="already exists or is a symlink"):
        _new_output_root(output)
