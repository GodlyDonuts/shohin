from __future__ import annotations

import ast
import json
from pathlib import Path
import platform
import subprocess
import sys

import pytest

from pipeline.episode_functor_candidate_role import (
    CandidateRoleError,
    _compile_machine,
)
from pipeline.episode_functor_cycle_language import encode_cycle_language
from pipeline.episode_functor_independent_world import generate_independent_world
from pipeline.episode_functor_process_custody import (
    CANDIDATE_ROLE,
    ProcessCustodyError,
    normalized_sandbox_profile,
    run_process_custody,
    run_sandbox_blindness_probe,
)
from pipeline.episode_functor_source_renderers import encode_line_events
from pipeline.episode_functor_wire_protocol import (
    WireProtocolSpec,
    encode_deployed_machine,
)


UNMOCKED_CUSTODY_AVAILABLE = platform.system() == "Darwin" or (
    platform.system() == "Linux"
    and Path("/usr/bin/bwrap").is_file()
    and Path("/usr/bin/python3").is_file()
)


def _world(tag: str, round_value: int) -> bytes:
    return generate_independent_world(
        protocol_root=("42" if round_value % 2 else "24") * 32,
        beacon_round=round_value,
        beacon_value=tag,
        state_count=5,
        action_count=3,
        observer_count=2,
        answer_count=5,
        renderer_count=1,
    ).evidence


def _roots(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name in ("public", "secret", "candidate", "assessor", "probe"):
        path = tmp_path / name
        path.mkdir()
        result[name] = path
    return result


def _rendered_evidence(
    renderer: str,
    evidence: bytes,
) -> tuple[bytes, WireProtocolSpec]:
    if renderer == "lines":
        return encode_line_events(evidence), WireProtocolSpec()
    if renderer == "cycle":
        return (
            encode_cycle_language(json.loads(evidence)),
            WireProtocolSpec(source_renderer_count=3),
        )
    return evidence, WireProtocolSpec()


@pytest.mark.parametrize("renderer", ("json", "lines", "cycle"))
def test_standalone_candidate_matches_deployed_compiler(renderer: str) -> None:
    evidence = _world(f"candidate-{renderer}", 11)
    evidence, spec = _rendered_evidence(renderer, evidence)
    assert _compile_machine(evidence) == encode_deployed_machine(evidence, spec)


def test_candidate_matches_reference_across_generated_worlds_and_languages() -> None:
    for index in range(16):
        evidence = _world(f"candidate-cross-world-{index}", 100 + index)
        expected = encode_deployed_machine(evidence, WireProtocolSpec())
        rendered = (
            evidence,
            encode_line_events(evidence),
            encode_cycle_language(json.loads(evidence)),
        )
        for payload in rendered:
            assert _compile_machine(payload) == expected


def test_candidate_rejects_boolean_renderer_choice() -> None:
    row = json.loads(_world("bool-renderer", 101))
    row["renderer_choice"] = False
    payload = (
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    with pytest.raises(CandidateRoleError, match="plain integer"):
        _compile_machine(payload)


def test_candidate_role_has_only_standard_library_imports() -> None:
    tree = ast.parse(CANDIDATE_ROLE.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported <= {
        "__future__",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "re",
        "struct",
        "sys",
    }


def test_candidate_role_refuses_unsandboxed_execution(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(_world("unsandboxed", 12))
    completed = subprocess.run(
        (
            sys.executable,
            str(CANDIDATE_ROLE),
            "evidence.bin",
            "machine.bin",
            "receipt.json",
        ),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 125
    assert not (tmp_path / "machine.bin").exists()


@pytest.mark.skipif(
    not UNMOCKED_CUSTODY_AVAILABLE,
    reason="exact macOS Seatbelt or Linux Bubblewrap/Python prerequisites are absent",
)
@pytest.mark.parametrize("renderer", ("json", "lines", "cycle"))
def test_unmocked_process_custody_is_exact_and_blind(
    tmp_path: Path,
    renderer: str,
) -> None:
    roots = _roots(tmp_path)
    evidence = _world(f"process-{renderer}", 13)
    evidence, spec = _rendered_evidence(renderer, evidence)
    public = roots["public"] / "evidence.bin"
    public.write_bytes(evidence)
    expected = roots["secret"] / "expected.bin"
    expected.write_bytes(encode_deployed_machine(evidence, spec))
    report = run_process_custody(
        public_evidence=public,
        expected_machine=expected,
        candidate_root=roots["candidate"],
        assessor_root=roots["assessor"],
    )
    assert report.exact_machine_match
    assert report.candidate_never_received_expected_machine
    assert report.assessor_started_after_candidate_exit
    assert report.candidate_run.exit_code == 0
    assert report.assessor_run.exit_code == 0
    assert report.candidate_run.cwd_regular_files_before == ("evidence.bin",)
    assert report.candidate_root_files == (
        "candidate_receipt.json",
        "evidence.bin",
        "machine.bin",
    )
    candidate_receipt = json.loads(
        (roots["candidate"] / "candidate_receipt.json").read_bytes()
    )
    assert candidate_receipt["machine_sha256"] == (report.candidate_machine_sha256)
    assert (
        candidate_receipt["candidate_source_sha256"]
        == report.candidate_run.role_source_sha256
    )

    allowed = roots["public"] / "allowed.txt"
    allowed.write_bytes(b"public\n")
    probe, probe_run = run_sandbox_blindness_probe(
        public_input=allowed,
        forbidden_secret=expected,
        probe_root=roots["probe"],
    )
    assert probe["all_gates_pass"]
    assert probe_run.exit_code == 0


@pytest.mark.skipif(
    not UNMOCKED_CUSTODY_AVAILABLE,
    reason="exact macOS Seatbelt or Linux Bubblewrap/Python prerequisites are absent",
)
def test_assessor_reports_valid_mismatched_machine(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    evidence = _world("candidate-world", 15)
    other = _world("different-valid-world", 16)
    public = roots["public"] / "evidence.bin"
    public.write_bytes(evidence)
    expected = roots["secret"] / "expected.bin"
    expected.write_bytes(encode_deployed_machine(other, WireProtocolSpec()))
    report = run_process_custody(
        public_evidence=public,
        expected_machine=expected,
        candidate_root=roots["candidate"],
        assessor_root=roots["assessor"],
    )
    assert not report.exact_machine_match
    assert report.assessor_run.exit_code == 2


def test_process_custody_rejects_symlinked_public_input(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    target = roots["public"] / "target.bin"
    target.write_bytes(_world("symlink", 17))
    linked = roots["public"] / "linked.bin"
    linked.symlink_to(target)
    expected = roots["secret"] / "expected.bin"
    expected.write_bytes(
        encode_deployed_machine(target.read_bytes(), WireProtocolSpec())
    )
    with pytest.raises(ProcessCustodyError, match="nonsymlink"):
        run_process_custody(
            public_evidence=linked,
            expected_machine=expected,
            candidate_root=roots["candidate"],
            assessor_root=roots["assessor"],
        )


def test_process_custody_rejects_oversized_public_input(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    public = roots["public"] / "evidence.bin"
    public.write_bytes(b"x" * (1024 * 1024 + 1))
    expected = roots["secret"] / "expected.bin"
    expected.write_bytes(b"\0" * 1_536)
    with pytest.raises(ProcessCustodyError, match="byte bound"):
        run_process_custody(
            public_evidence=public,
            expected_machine=expected,
            candidate_root=roots["candidate"],
            assessor_root=roots["assessor"],
        )


def test_normalized_profile_is_default_deny_and_not_repo_readable() -> None:
    profile = normalized_sandbox_profile(CANDIDATE_ROLE)
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert '(allow file-read* (literal "<ROLE_SCRIPT>"))' in profile
    assert '(allow file-read* (subpath "<ROLE_CWD>"))' in profile
    assert '(allow file-write* (subpath "<ROLE_CWD>"))' in profile
    assert '(allow file-read* (subpath "<REPOSITORY_ROOT>"))' not in profile
