#!/usr/bin/env python3
"""Run a signed-beacon, process-separated EFC CPU rehearsal.

The committed source authorization must predate the verified NIST pulse. The
same independently generated world is rendered in three source languages.
Fresh default-deny candidate processes compile each source, and later assessor
processes compare the sealed outputs with an independent expected machine.
This is a custody/mechanics rehearsal, not a neural or reasoning result.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Mapping
import urllib.request

from pipeline.acw_nist_beacon import (
    PULSE_PROTOCOL,
    canonical_json_bytes as beacon_json_bytes,
    fetch_certificate,
    verify_pulse,
)
from pipeline.episode_functor_cycle_language import encode_cycle_language
from pipeline.episode_functor_independent_world import (
    generate_independent_world,
)
from pipeline.episode_functor_process_custody import (
    ASSESSOR_ROLE,
    CANDIDATE_ROLE,
    LANDLOCK_LAUNCHER,
    PROBE_ROLE,
    ProcessCustodyError,
    canonical_json_bytes,
    run_process_custody,
    run_sandbox_blindness_probe,
    sha256_bytes,
)
from pipeline.episode_functor_source_renderers import encode_line_events
from pipeline.episode_functor_wire_protocol import (
    MACHINE_SIZE,
    WireProtocolSpec,
    encode_deployed_machine,
)


REHEARSAL_SCHEMA = "efc-signed-process-rehearsal-v1"
AUTHORIZATION_SCHEMA = "efc-process-custody-authorization-v1"
ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    ROOT / "artifacts" / "r12" / "efc_process_custody_authorization_v1.json"
)
GITHUB_EVENTS_URL = "https://api.github.com/repos/GodlyDonuts/shohin/events"
FROZEN_SOURCE_PATHS = (
    ROOT / "R12_EFC_PROCESS_CUSTODY_PREREG.md",
    ASSESSOR_ROLE,
    CANDIDATE_ROLE,
    ROOT / "pipeline" / "acw_nist_beacon.py",
    ROOT / "pipeline" / "episode_functor_cycle_language.py",
    ROOT / "pipeline" / "episode_functor_independent_world.py",
    ROOT / "pipeline" / "episode_functor_process_custody.py",
    ROOT / "pipeline" / "episode_functor_seal_protocol.py",
    PROBE_ROLE,
    ROOT / "pipeline" / "episode_functor_source_renderers.py",
    ROOT / "pipeline" / "episode_functor_wire_protocol.py",
    LANDLOCK_LAUNCHER,
    Path(__file__).resolve(),
)


class RehearsalError(RuntimeError):
    """The authorization, beacon, source tree, or rehearsal result is invalid."""


def _read_plain_file(
    path: Path,
    label: str,
    *,
    maximum: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RehearsalError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RehearsalError(f"{label} must be a nonsymlink regular file")
        if maximum is not None and metadata.st_size > maximum:
            raise RehearsalError(f"{label} exceeds its byte bound")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        if len(payload) != metadata.st_size:
            raise RehearsalError(f"{label} changed size during read")
        return payload
    finally:
        os.close(descriptor)


def frozen_source_receipt() -> dict[str, object]:
    files = {
        str(path.relative_to(ROOT)): sha256_bytes(
            _read_plain_file(path, f"frozen source {path.name}")
        )
        for path in FROZEN_SOURCE_PATHS
    }
    return {
        "files": files,
        "source_root": sha256_bytes(canonical_json_bytes(files)),
    }


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = _read_plain_file(path, path.name, maximum=1024 * 1024)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RehearsalError(f"{path.name} is malformed JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise RehearsalError(f"{path.name} is not canonical JSON")
    return value


def _load_beacon_snapshot(
    path: Path,
    *,
    expected_certificate_der_sha512: str,
) -> tuple[dict[str, object], str]:
    payload = _read_plain_file(path, "beacon snapshot", maximum=8 * 1024 * 1024)
    try:
        snapshot = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RehearsalError("beacon snapshot is malformed") from exc
    if (
        not isinstance(snapshot, dict)
        or beacon_json_bytes(snapshot) + b"\n" != payload
        or set(snapshot)
        != {
            "certificate_pem",
            "payload_sha256",
            "previous_pulse",
            "previous_verification",
            "protocol",
            "pulse",
            "verification",
        }
        or snapshot["protocol"] != PULSE_PROTOCOL
    ):
        raise RehearsalError("beacon snapshot schema or encoding differs")
    hash_input = dict(snapshot)
    claimed_hash = hash_input.pop("payload_sha256")
    if not isinstance(claimed_hash, str) or claimed_hash != sha256_bytes(
        beacon_json_bytes(hash_input)
    ):
        raise RehearsalError("beacon snapshot payload hash differs")
    pulse = snapshot["pulse"]
    previous = snapshot["previous_pulse"]
    certificate = snapshot["certificate_pem"]
    if (
        not isinstance(pulse, dict)
        or not isinstance(previous, dict)
        or not isinstance(certificate, str)
    ):
        raise RehearsalError("beacon snapshot payload types differ")
    official_certificate = fetch_certificate(str(pulse["certificateId"]))
    previous_verification = verify_pulse(
        previous,
        official_certificate,
        expected_chain_index=previous["chainIndex"],
        expected_pulse_index=previous["pulseIndex"],
        expected_timestamp=previous["timeStamp"],
    )
    verification = verify_pulse(
        pulse,
        official_certificate,
        previous_pulse=previous,
        expected_chain_index=pulse["chainIndex"],
        expected_pulse_index=pulse["pulseIndex"],
        expected_timestamp=pulse["timeStamp"],
    )
    snapshot_certificate_verification = verify_pulse(
        pulse,
        certificate.encode("ascii"),
        previous_pulse=previous,
        expected_chain_index=pulse["chainIndex"],
        expected_pulse_index=pulse["pulseIndex"],
        expected_timestamp=pulse["timeStamp"],
    )
    if verification["certificate_der_sha512"] != expected_certificate_der_sha512:
        raise RehearsalError("beacon certificate differs from authorization")
    if (
        verification != snapshot_certificate_verification
        or verification != snapshot["verification"]
        or previous_verification != snapshot["previous_verification"]
    ):
        raise RehearsalError("beacon verification receipt differs on replay")
    return snapshot, sha256_bytes(payload)


def _git_blob(revision: str, path: Path) -> bytes:
    try:
        relative = str(path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise RehearsalError("Git blob path is outside the repository") from exc
    completed = subprocess.run(
        ("git", "show", f"{revision}:{relative}"),
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RehearsalError(f"{relative} is absent from {revision}")
    return completed.stdout


def _github_push_receipt(*, head: str, branch: str) -> dict[str, object]:
    request = urllib.request.Request(
        GITHUB_EVENTS_URL + "?per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "shohin-efc-custody/1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RehearsalError(f"GitHub Events returned HTTP {response.status}")
        raw = response.read(4 * 1024 * 1024 + 1)
        response_date = response.headers.get("Date")
    if len(raw) > 4 * 1024 * 1024:
        raise RehearsalError("GitHub Events response exceeds four MiB")
    try:
        events = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RehearsalError("GitHub Events response is malformed") from exc
    if not isinstance(events, list):
        raise RehearsalError("GitHub Events response is not a list")
    target_ref = f"refs/heads/{branch}"
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "PushEvent":
            continue
        payload = event.get("payload")
        repository = event.get("repo")
        if (
            not isinstance(payload, dict)
            or not isinstance(repository, dict)
            or repository.get("name") != "GodlyDonuts/shohin"
            or payload.get("ref") != target_ref
            or payload.get("head") != head
            or event.get("public") is not True
            or not isinstance(event.get("id"), str)
            or not isinstance(event.get("created_at"), str)
        ):
            continue
        created_at = str(event["created_at"])
        try:
            parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RehearsalError("GitHub push event timestamp is malformed") from exc
        if parsed.tzinfo is None:
            raise RehearsalError("GitHub push event timestamp lacks a timezone")
        return {
            "github_events_response_date": response_date,
            "github_push_event_created_at": created_at,
            "github_push_event_id": event["id"],
            "github_push_event_public": True,
            "github_push_event_ref": target_ref,
        }
    raise RehearsalError("published HEAD lacks a matching public GitHub PushEvent")


def _git_receipt(
    *,
    authorization_path: Path,
    source_receipt: Mapping[str, object],
) -> dict[str, object]:
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RehearsalError("source worktree must be clean before beacon consumption")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    branch = _git_branch()
    remote = subprocess.run(
        ("git", "ls-remote", "origin", f"refs/heads/{branch}"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not remote or remote.split()[0] != head:
        raise RehearsalError("candidate HEAD is not published at origin")
    authorization_payload = _read_plain_file(
        authorization_path,
        "fixed authorization",
        maximum=1024 * 1024,
    )
    if _git_blob("HEAD", authorization_path) != authorization_payload:
        raise RehearsalError("authorization bytes differ from published HEAD")
    files = source_receipt.get("files")
    if not isinstance(files, Mapping):
        raise RehearsalError("source receipt files are malformed")
    for relative, expected_sha256 in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or sha256_bytes(_git_blob("HEAD", ROOT / relative)) != expected_sha256
        ):
            raise RehearsalError(
                f"frozen source differs from published HEAD: {relative}"
            )
    github = _github_push_receipt(head=head, branch=branch)
    return {
        "authorization_blob_sha256": sha256_bytes(authorization_payload),
        "authorization_in_head_verified": True,
        "branch": branch,
        **github,
        "head": head,
        "origin_head_verified": True,
        "source_blobs_in_head_verified": True,
    }


def _git_branch() -> str:
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not branch:
        raise RehearsalError("candidate source is not on a named branch")
    return branch


def _write_immutable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RehearsalError("immutable write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _closed_tree_receipt(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not path.is_symlink():
            continue
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RehearsalError(f"artifact tree contains a nonregular entry: {path}")
        files[str(path.relative_to(root))] = sha256_bytes(
            _read_plain_file(path, f"artifact file {path.name}")
        )
    return files


def _new_output_root(path: Path) -> Path:
    candidate = Path(os.path.abspath(path.expanduser()))
    parent = candidate.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise RehearsalError("rehearsal output parent is unavailable") from exc
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or Path(os.path.realpath(parent)) != parent
    ):
        raise RehearsalError(
            "rehearsal output parent must be an existing nonsymlink directory"
        )
    try:
        candidate.lstat()
    except FileNotFoundError:
        return candidate
    except OSError as exc:
        raise RehearsalError("rehearsal output root cannot be inspected") from exc
    raise RehearsalError("rehearsal output root already exists or is a symlink")


def _validate_authorization(
    authorization: Mapping[str, object],
    source_receipt: Mapping[str, object],
) -> None:
    if set(authorization) != {
        "candidate_source_root",
        "claim_boundary",
        "machine_bytes",
        "minimum_pulse_delay_seconds",
        "nist_certificate_der_sha512",
        "nist_chain_index",
        "pulse_selection_rule",
        "schema",
        "source_languages",
        "target_pulse_index",
    }:
        raise RehearsalError("authorization fields differ")
    if (
        authorization["schema"] != AUTHORIZATION_SCHEMA
        or authorization["candidate_source_root"] != source_receipt["source_root"]
        or authorization["claim_boundary"] != "cpu-custody-mechanics-only"
        or authorization["machine_bytes"] != MACHINE_SIZE
        or authorization["minimum_pulse_delay_seconds"] != 60
        or not isinstance(authorization["nist_certificate_der_sha512"], str)
        or len(authorization["nist_certificate_der_sha512"]) != 128
        or authorization["nist_certificate_der_sha512"].lower()
        != authorization["nist_certificate_der_sha512"]
        or any(
            character not in "0123456789abcdef"
            for character in authorization["nist_certificate_der_sha512"]
        )
        or authorization["nist_chain_index"] != 2
        or authorization["pulse_selection_rule"]
        != "exact-precommitted-chain-2-pulse-index"
        or authorization["source_languages"]
        != ["canonical-json-events", "strict-line-events", "cycle-program"]
        or isinstance(authorization["target_pulse_index"], bool)
        or not isinstance(authorization["target_pulse_index"], int)
        or authorization["target_pulse_index"] <= 0
    ):
        raise RehearsalError("authorization values differ")


def run_rehearsal(
    *,
    authorization_path: Path,
    beacon_snapshot_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Consume one future pulse and atomically publish a CPU custody artifact."""

    authorization_path = authorization_path.resolve()
    if authorization_path != AUTHORIZATION_PATH.resolve():
        raise RehearsalError("authorization must use the fixed repository path")
    source_receipt = frozen_source_receipt()
    authorization = _load_canonical_json(authorization_path)
    _validate_authorization(authorization, source_receipt)
    git = _git_receipt(
        authorization_path=authorization_path,
        source_receipt=source_receipt,
    )
    snapshot, snapshot_sha256 = _load_beacon_snapshot(
        beacon_snapshot_path,
        expected_certificate_der_sha512=str(
            authorization["nist_certificate_der_sha512"]
        ),
    )
    pulse = snapshot["pulse"]
    assert isinstance(pulse, dict)
    pulse_timestamp = datetime.fromisoformat(
        str(pulse["timeStamp"]).replace("Z", "+00:00")
    )
    publication_timestamp = datetime.fromisoformat(
        str(git["github_push_event_created_at"]).replace("Z", "+00:00")
    )
    if (
        pulse["chainIndex"] != authorization["nist_chain_index"]
        or pulse["pulseIndex"] != authorization["target_pulse_index"]
        or (pulse_timestamp - publication_timestamp).total_seconds()
        < authorization["minimum_pulse_delay_seconds"]
    ):
        raise RehearsalError("NIST pulse does not satisfy the frozen future rule")
    output_root = _new_output_root(output_root)

    with tempfile.TemporaryDirectory(
        prefix=output_root.name + ".tmp.",
        dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary)
        protocol_root = str(source_receipt["source_root"])
        world = generate_independent_world(
            protocol_root=protocol_root,
            beacon_round=int(pulse["pulseIndex"]),
            beacon_value=str(pulse["outputValue"]),
            state_count=5,
            action_count=3,
            observer_count=2,
            answer_count=5,
            renderer_count=1,
        )
        normalized_row = json.loads(world.evidence)
        sources = {
            "canonical-json-events": world.evidence,
            "cycle-program": encode_cycle_language(normalized_row),
            "strict-line-events": encode_line_events(world.evidence),
        }
        wire_spec = WireProtocolSpec(source_renderer_count=3)
        process_reports: dict[str, object] = {}
        machine_hashes: dict[str, str] = {}
        first_expected: Path | None = None
        for language in sorted(sources):
            language_root = staging / language
            public = language_root / "public"
            secret = language_root / "secret"
            candidate = language_root / "candidate"
            assessor = language_root / "assessor"
            for directory in (public, secret, candidate, assessor):
                directory.mkdir(parents=True)
            evidence = public / "evidence.bin"
            _write_immutable(evidence, sources[language])
            expected = secret / "expected_machine.bin"
            _write_immutable(
                expected,
                encode_deployed_machine(sources[language], wire_spec),
            )
            if first_expected is None:
                first_expected = expected
            process_report = run_process_custody(
                public_evidence=evidence,
                expected_machine=expected,
                candidate_root=candidate,
                assessor_root=assessor,
            )
            if not process_report.exact_machine_match:
                raise RehearsalError(f"{language} candidate machine differs")
            source_files = source_receipt["files"]
            assert isinstance(source_files, Mapping)
            if (
                process_report.candidate_run.role_source_sha256
                != source_files["pipeline/episode_functor_candidate_role.py"]
                or process_report.assessor_run.role_source_sha256
                != source_files["pipeline/episode_functor_assessor_role.py"]
            ):
                raise RehearsalError(
                    f"{language} isolated role source differs from freeze"
                )
            if (
                process_report.candidate_run.sandbox_launcher_source_sha256 is not None
                and process_report.candidate_run.sandbox_launcher_source_sha256
                != source_files["train/landlock_stage_exec.py"]
                or process_report.assessor_run.sandbox_launcher_source_sha256
                is not None
                and process_report.assessor_run.sandbox_launcher_source_sha256
                != source_files["train/landlock_stage_exec.py"]
            ):
                raise RehearsalError(f"{language} Linux launcher differs from freeze")
            process_reports[language] = process_report.canonical_dict()
            machine_hashes[language] = process_report.candidate_machine_sha256

        if len(set(machine_hashes.values())) != 1:
            raise RehearsalError("source languages compiled to different machines")
        assert first_expected is not None
        probe_public = staging / "blindness-probe" / "public"
        probe_root = staging / "blindness-probe" / "output"
        probe_public.mkdir(parents=True)
        probe_root.mkdir()
        public_sentinel = probe_public / "allowed.txt"
        _write_immutable(public_sentinel, b"public\n")
        probe, probe_run = run_sandbox_blindness_probe(
            public_input=public_sentinel,
            forbidden_secret=first_expected,
            probe_root=probe_root,
        )
        source_files = source_receipt["files"]
        assert isinstance(source_files, Mapping)
        if probe_run.role_source_sha256 != source_files[
            "pipeline/episode_functor_sandbox_probe_role.py"
        ] or (
            probe_run.sandbox_launcher_source_sha256 is not None
            and probe_run.sandbox_launcher_source_sha256
            != source_files["train/landlock_stage_exec.py"]
        ):
            raise RehearsalError("isolated blindness-probe source differs from freeze")
        report: dict[str, object] = {
            "authorization": authorization,
            "authorization_sha256": sha256_bytes(
                _read_plain_file(authorization_path, "authorization")
            ),
            "beacon_snapshot_sha256": snapshot_sha256,
            "blindness_probe": probe,
            "blindness_probe_run": probe_run.canonical_dict(),
            "candidate_assessor_temporal_order": "candidate-exit-before-assessor-start",
            "candidate_process_count": len(process_reports),
            "candidate_source_root": source_receipt["source_root"],
            "exact_matches": sum(
                bool(value["exact_machine_match"]) for value in process_reports.values()
            ),
            "future_pulse_after_publication": True,
            "git_publication": git,
            "machine_bytes": MACHINE_SIZE,
            "machine_sha256": next(iter(machine_hashes.values())),
            "nist_pulse": snapshot["verification"],
            "process_reports": process_reports,
            "schema": REHEARSAL_SCHEMA,
            "source_language_count": len(sources),
            "source_languages": sorted(sources),
            "source_receipt": source_receipt,
            "world_admissibility": world.admissibility_receipt,
            "world_seed_commitment": world.world_seed_commitment,
        }
        if (
            report["candidate_process_count"] != 3
            or report["exact_matches"] != 3
            or not world.admissibility_receipt["admitted"]
        ):
            raise RehearsalError("rehearsal gates did not all pass")
        files_before_report = _closed_tree_receipt(staging)
        report["files_before_report"] = files_before_report
        report["files_before_report_root"] = sha256_bytes(
            canonical_json_bytes(files_before_report)
        )
        report = json.loads(canonical_json_bytes(report))
        _write_immutable(
            staging / "final_report.json",
            canonical_json_bytes(report),
        )
        os.rename(staging, output_root)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--beacon-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = run_rehearsal(
            authorization_path=arguments.authorization,
            beacon_snapshot_path=arguments.beacon_snapshot,
            output_root=arguments.output_root,
        )
    except (OSError, ProcessCustodyError, RehearsalError, ValueError) as exc:
        print(f"efc-process-rehearsal: {exc}", flush=True)
        return 125
    print(
        json.dumps(
            {
                "machine_sha256": report["machine_sha256"],
                "schema": report["schema"],
                "source_language_count": report["source_language_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
