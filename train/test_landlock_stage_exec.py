from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest

import landlock_stage_exec as launcher


def test_abi_rights_are_added_only_when_supported() -> None:
    abi3 = launcher.handled_access_fs(3)
    assert abi3 & launcher.ACCESS_FS_READ_FILE
    assert abi3 & launcher.ACCESS_FS_REFER
    assert abi3 & launcher.ACCESS_FS_TRUNCATE
    assert not abi3 & launcher.ACCESS_FS_IOCTL_DEV
    assert launcher.handled_access_fs(5) & launcher.ACCESS_FS_IOCTL_DEV
    assert launcher.handled_access_fs(6) == launcher.handled_access_fs(5)


@pytest.mark.parametrize("abi", [0, 1, 2, 11, True, "5"])
def test_unknown_or_invalid_abi_fails_closed(abi) -> None:
    with pytest.raises(launcher.LandlockUnsupportedError):
        launcher.validate_abi(abi)


@pytest.mark.parametrize(
    "stage",
    ["", "has space", "slash/name", "nonascii-\N{SNOWMAN}", "x" * 129],
)
def test_invalid_stage_fails_closed(stage: str) -> None:
    with pytest.raises(launcher.LandlockPolicyError):
        launcher.validate_stage(stage)


def test_policy_canonicalizes_deduplicates_and_merges_rights(
    tmp_path: Path,
) -> None:
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    readable.mkdir()
    writable.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(readable, target_is_directory=True)

    first = launcher.canonicalize_policy(
        stage="compiler",
        abi=5,
        ro=[alias, writable],
        rw=[writable],
        list_dir=[readable],
        cwd=tmp_path,
    )
    second = launcher.canonicalize_policy(
        stage="compiler",
        abi=5,
        ro=[writable, readable],
        rw=[writable],
        list_dir=[alias],
        cwd=tmp_path,
    )
    assert first == second
    assert launcher.policy_sha256(first) == launcher.policy_sha256(second)
    assert [row["path"] for row in first["rules"]] == sorted(
        {str(readable.resolve()), str(writable.resolve())}
    )

    by_path = {row["path"]: row for row in first["rules"]}
    readable_rule = by_path[str(readable.resolve())]
    writable_rule = by_path[str(writable.resolve())]
    assert readable_rule["allowed_access_fs"] == (
        launcher.ACCESS_FS_EXECUTE
        | launcher.ACCESS_FS_READ_FILE
        | launcher.ACCESS_FS_READ_DIR
    )
    assert writable_rule["allowed_access_fs"] == (launcher.handled_access_fs(5))


def test_policy_canonical_bytes_have_a_stable_exact_encoding() -> None:
    policy = launcher.canonicalize_policy(stage="executor", abi=3, ro=["/"])
    encoded = launcher.canonical_policy_bytes(policy)
    assert encoded == json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert b"\n" not in encoded
    assert launcher.policy_sha256(policy) == hashlib.sha256(encoded).hexdigest()


def test_policy_serialization_rejects_noncanonical_rule_order(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    policy = launcher.canonicalize_policy(
        stage="assessor",
        abi=5,
        ro=[left, right],
    )
    policy["rules"] = list(reversed(policy["rules"]))
    with pytest.raises(launcher.LandlockPolicyError, match="path-sorted"):
        launcher.canonical_policy_bytes(policy)


def test_policy_serialization_rejects_type_coercion_and_unnormalized_paths(
    tmp_path: Path,
) -> None:
    policy = launcher.canonicalize_policy(stage="assessor", abi=5, ro=[tmp_path])
    policy["landlock_abi"] = "5"
    with pytest.raises(launcher.LandlockUnsupportedError, match="integer"):
        launcher.canonical_policy_bytes(policy)

    policy = launcher.canonicalize_policy(stage="assessor", abi=5, ro=[tmp_path])
    policy["rules"][0]["path"] = f"{tmp_path}/../{tmp_path.name}"
    with pytest.raises(launcher.LandlockPolicyError, match="not absolute"):
        launcher.canonical_policy_bytes(policy)


def test_nonexistent_paths_and_list_dir_files_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(launcher.LandlockPolicyError, match="canonicalized"):
        launcher.canonicalize_policy(
            stage="compiler",
            abi=5,
            ro=[tmp_path / "missing"],
        )
    regular = tmp_path / "regular"
    regular.write_text("content")
    with pytest.raises(launcher.LandlockPolicyError, match="requires a directory"):
        launcher.canonicalize_policy(
            stage="compiler",
            abi=5,
            list_dir=[regular],
        )


def test_file_and_directory_rights_are_object_aware(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = tmp_path / "regular"
    regular.write_text("content")
    policy = launcher.canonicalize_policy(
        stage="compiler",
        abi=5,
        ro=[regular],
        rw=[directory],
    )
    by_path = {row["path"]: row for row in policy["rules"]}
    assert by_path[str(regular)]["object_type"] == "file"
    assert by_path[str(regular)]["allowed_access_fs"] == (
        launcher.ACCESS_FS_EXECUTE | launcher.ACCESS_FS_READ_FILE
    )
    assert by_path[str(directory)]["object_type"] == "directory"
    assert by_path[str(directory)]["allowed_access_fs"] == (
        launcher.handled_access_fs(5)
    )


def test_policy_object_identity_detects_path_replacement(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("original")
    policy = launcher.canonicalize_policy(stage="compiler", abi=5, ro=[target])
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement")
    replacement.replace(target)
    descriptor = os.open(target, os.O_RDONLY)
    try:
        with pytest.raises(launcher.LandlockStageError, match="changed"):
            launcher._validate_rule_object_identity(policy["rules"][0], descriptor)
    finally:
        os.close(descriptor)


def test_cli_requires_separator_stage_and_command() -> None:
    with pytest.raises(launcher.LandlockPolicyError, match="separator"):
        launcher.parse_cli(
            ["--stage", "compiler", "--policy-receipt", "/tmp/policy", "python"]
        )
    with pytest.raises(launcher.LandlockPolicyError, match="no command"):
        launcher.parse_cli(
            ["--stage", "compiler", "--policy-receipt", "/tmp/policy", "--"]
        )
    with pytest.raises(SystemExit):
        launcher.parse_cli(["--", "python"])
    args, command = launcher.parse_cli(
        [
            "--stage",
            "compiler",
            "--policy-receipt",
            "/tmp/policy",
            "--ro",
            "/",
            "--",
            "python",
            "-V",
        ]
    )
    assert args.stage == "compiler"
    assert args.ro == ["/"]
    assert command == ["python", "-V"]


def test_non_linux_abi_query_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with pytest.raises(launcher.LandlockUnsupportedError, match="requires Linux"):
        launcher.query_landlock_abi()


def test_launch_exports_exact_canonical_policy_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    policy = launcher.canonicalize_policy(
        stage="compiler",
        abi=5,
        ro=[tmp_path],
    )
    monkeypatch.setattr(launcher, "query_landlock_abi", lambda: 5)
    monkeypatch.setattr(
        launcher,
        "enforce_policy",
        lambda value: captured.setdefault("policy", value),
    )

    stage_script = tmp_path / "stage.py"
    stage_script.write_text("raise AssertionError('must be mocked')\n")
    policy_receipt = tmp_path / "policy.json"

    def fake_run_path(path: str, *, run_name: str) -> None:
        captured["script"] = path
        captured["run_name"] = run_name
        captured["command"] = list(sys.argv)
        captured["environment"] = dict(os.environ)

    monkeypatch.setattr(launcher.runpy, "run_path", fake_run_path)
    original_environment = dict(os.environ)
    original_argv = list(sys.argv)
    try:
        launcher.launch(
            [
                "--stage",
                "compiler",
                "--ro",
                str(tmp_path),
                "--policy-receipt",
                str(policy_receipt),
                "--",
                sys.executable,
                str(stage_script),
                "--flag",
            ]
        )
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
        sys.argv = original_argv

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert captured["policy"] == policy
    assert captured["script"] == str(stage_script)
    assert captured["run_name"] == "__main__"
    assert captured["command"] == [str(stage_script), "--flag"]
    assert {
        key: environment[key]
        for key in (
            "SHOHIN_LANDLOCK_ENFORCED",
            "SHOHIN_LANDLOCK_ABI",
            "SHOHIN_LANDLOCK_STAGE",
            "SHOHIN_LANDLOCK_POLICY_SHA256",
        )
    } == {
        "SHOHIN_LANDLOCK_ENFORCED": "1",
        "SHOHIN_LANDLOCK_ABI": "5",
        "SHOHIN_LANDLOCK_STAGE": "compiler",
        "SHOHIN_LANDLOCK_POLICY_SHA256": launcher.policy_sha256(policy),
    }
    assert policy_receipt.read_bytes() == launcher.canonical_policy_bytes(policy)
    assert environment["SHOHIN_LANDLOCK_POLICY_PATH"] == str(policy_receipt)


def _supported_linux_abi() -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        return launcher.query_landlock_abi()
    except launcher.LandlockStageError:
        return None


@pytest.mark.skipif(
    _supported_linux_abi() is None,
    reason="requires supported Linux Landlock",
)
def test_supported_linux_launcher_enforces_writes_and_exports_policy(
    tmp_path: Path,
) -> None:
    abi = _supported_linux_abi()
    assert abi is not None
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    denied_secret = denied / "secret.txt"
    denied_secret.write_text("secret", encoding="ascii")
    child = (
        "import json, os, pathlib; "
        f"pathlib.Path({str(allowed / 'ok.txt')!r}).write_text('ok'); "
        "write_denied=False; read_denied=False; "
        "\ntry:\n"
        f" pathlib.Path({str(denied / 'blocked.txt')!r}).write_text('bad')\n"
        "except PermissionError:\n write_denied=True\n"
        "try:\n"
        f" pathlib.Path({str(denied_secret)!r}).read_text()\n"
        "except PermissionError:\n read_denied=True\n"
        "print(json.dumps({'write_denied':write_denied,"
        "'read_denied':read_denied,"
        "'enforced':os.environ.get('SHOHIN_LANDLOCK_ENFORCED'),"
        "'abi':os.environ.get('SHOHIN_LANDLOCK_ABI'),"
        "'stage':os.environ.get('SHOHIN_LANDLOCK_STAGE'),"
        "'policy':os.environ.get('SHOHIN_LANDLOCK_POLICY_SHA256')}))"
    )
    child_script = tmp_path / "child.py"
    child_script.write_text(child)
    policy = launcher.canonicalize_policy(
        stage="integration",
        abi=abi,
        ro=[child_script],
        rw=[allowed],
    )
    expected_digest = launcher.policy_sha256(policy)
    script = Path(launcher.__file__).resolve()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stage",
            "integration",
            "--ro",
            str(child_script),
            "--rw",
            str(allowed),
            "--policy-receipt",
            str(tmp_path / "policy.json"),
            "--",
            sys.executable,
            str(child_script),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    exported = json.loads(result.stdout)
    assert exported == {
        "write_denied": True,
        "read_denied": True,
        "enforced": "1",
        "abi": str(abi),
        "stage": "integration",
        "policy": expected_digest,
    }
    assert (allowed / "ok.txt").read_text() == "ok"
    assert not (denied / "blocked.txt").exists()
    assert denied_secret.read_text() == "secret"
