from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile

import pytest

from ctaa_assessment_source_bundle import (
    ARCHIVE_MEMBERS,
    ASSESSMENT_SOURCE_MEMBERS,
    AssessmentSourceBundleError,
    MAIN_MEMBER,
    MAIN_SOURCE,
    build_assessment_source_bundle,
    canonical_manifest_bytes,
    load_sealed_assessment_bundle_memfd,
    validate_assessment_source_bundle,
    validate_sealed_assessment_bundle_fd,
)

TRAIN_ROOT = Path(__file__).resolve().parent
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR = (stat.S_IFREG | 0o444) << 16


def _write_executable(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o555)
    return path


def _copy_source_tree(destination: Path) -> Path:
    destination.mkdir()
    for member in ASSESSMENT_SOURCE_MEMBERS:
        shutil.copyfile(TRAIN_ROOT / member, destination / member)
        (destination / member).chmod(0o444)
    return destination


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_root = _copy_source_tree(tmp_path / "source")
    python = _write_executable(tmp_path / "python", b"python-interpreter-v1\n")
    bwrap = _write_executable(tmp_path / "bwrap", b"bubblewrap-v1\n")
    return {
        "source_root": source_root,
        "python": python,
        "bwrap": bwrap,
        "bundle": tmp_path / "assessor.pyz",
        "manifest": tmp_path / "assessor.manifest.json",
    }


def _build(tmp_path: Path) -> dict[str, Path]:
    paths = _fixture_paths(tmp_path)
    build_assessment_source_bundle(
        source_root=paths["source_root"],
        bundle_path=paths["bundle"],
        manifest_path=paths["manifest"],
        python_executable=paths["python"],
        bwrap_executable=paths["bwrap"],
    )
    return paths


def _validate(paths: dict[str, Path]) -> dict[str, object]:
    return validate_assessment_source_bundle(
        source_root=paths["source_root"],
        bundle_path=paths["bundle"],
        manifest_path=paths["manifest"],
        python_executable=paths["python"],
        bwrap_executable=paths["bwrap"],
    )


def _fixed_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = FIXED_EXTERNAL_ATTR
    info.extra = b""
    info.comment = b""
    return info


def _archive_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in entries:
            archive.writestr(_fixed_info(name), payload)
    return stream.getvalue()


def _read_entries(bundle_path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(bundle_path.read_bytes()), "r") as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _replace_read_only(path: Path, payload: bytes) -> None:
    path.chmod(0o600)
    path.unlink()
    path.write_bytes(payload)
    path.chmod(0o444)


def _rewrite_bundle_and_manifest(
    paths: dict[str, Path],
    bundle: bytes,
    *,
    member_updates: dict[str, bytes] | None = None,
) -> None:
    manifest = json.loads(paths["manifest"].read_text("ascii"))
    manifest["bundle_sha256"] = hashlib.sha256(bundle).hexdigest()
    manifest["bundle_size"] = len(bundle)
    for name, payload in (member_updates or {}).items():
        manifest["members"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    _replace_read_only(paths["bundle"], bundle)
    _replace_read_only(paths["manifest"], canonical_manifest_bytes(manifest))


def test_deterministic_zipapp_and_manifest_bytes(tmp_path: Path) -> None:
    assert "ctaa_statistical_gate_spec.py" in ASSESSMENT_SOURCE_MEMBERS
    first = _fixture_paths(tmp_path / "first")
    second = _fixture_paths(tmp_path / "second")
    # Bind identical canonical executable paths for both builds.
    first["python"] = second["python"] = _write_executable(
        tmp_path / "shared-python", b"shared-python\n"
    )
    first["bwrap"] = second["bwrap"] = _write_executable(
        tmp_path / "shared-bwrap", b"shared-bwrap\n"
    )
    for paths in (first, second):
        build_assessment_source_bundle(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=paths["manifest"],
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
        )
    assert first["bundle"].read_bytes() == second["bundle"].read_bytes()
    assert first["manifest"].read_bytes() == second["manifest"].read_bytes()
    manifest = _validate(first)
    assert tuple(sorted(manifest["members"])) == ARCHIVE_MEMBERS
    assert (
        manifest["bundle_sha256"]
        == hashlib.sha256(first["bundle"].read_bytes()).hexdigest()
    )
    with zipfile.ZipFile(first["bundle"], "r") as archive:
        assert [info.filename for info in archive.infolist()] == list(ARCHIVE_MEMBERS)
        assert archive.read(MAIN_MEMBER) == MAIN_SOURCE
        for info in archive.infolist():
            assert info.date_time == FIXED_TIME
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.external_attr == FIXED_EXTERNAL_ATTR


def test_member_rewrite_fails_even_with_recomputed_manifest_hashes(
    tmp_path: Path,
) -> None:
    paths = _build(tmp_path)
    entries = _read_entries(paths["bundle"])
    target = "ctaa_assessment.py"
    mutated = dict(entries)[target] + b"\n# unauthorized mutation\n"
    rewritten = _archive_bytes(
        [(name, mutated if name == target else payload) for name, payload in entries]
    )
    _rewrite_bundle_and_manifest(paths, rewritten, member_updates={target: mutated})
    with pytest.raises(AssessmentSourceBundleError, match="current source"):
        _validate(paths)


def test_source_mutation_after_build_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    source = paths["source_root"] / "ctaa_assessment.py"
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b"\n# changed source\n")
    source.chmod(0o444)
    with pytest.raises(AssessmentSourceBundleError, match="current source"):
        _validate(paths)


def test_missing_source_module_is_rejected(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    missing = paths["source_root"] / ASSESSMENT_SOURCE_MEMBERS[-1]
    missing.unlink()
    with pytest.raises(AssessmentSourceBundleError, match="unavailable"):
        build_assessment_source_bundle(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=paths["manifest"],
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
        )


@pytest.mark.parametrize(
    "members",
    [
        ASSESSMENT_SOURCE_MEMBERS[:-1],
        (*ASSESSMENT_SOURCE_MEMBERS, "extra_assessor_module.py"),
    ],
)
def test_changed_source_allowlist_is_rejected(
    tmp_path: Path,
    members: tuple[str, ...],
) -> None:
    paths = _fixture_paths(tmp_path)
    with pytest.raises(AssessmentSourceBundleError, match="allowlist differs"):
        build_assessment_source_bundle(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=paths["manifest"],
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
            source_members=members,
        )


def test_new_local_import_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    extra = paths["source_root"] / "ctaa_unreviewed.py"
    extra.write_text("VALUE = 1\n", encoding="ascii")
    extra.chmod(0o444)
    entrypoint = paths["source_root"] / "assess_ctaa_evidence.py"
    entrypoint.chmod(0o644)
    entrypoint.write_bytes(entrypoint.read_bytes() + b"\nimport ctaa_unreviewed\n")
    entrypoint.chmod(0o444)
    with pytest.raises(AssessmentSourceBundleError, match="undeclared local modules"):
        build_assessment_source_bundle(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=paths["manifest"],
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
        )


@pytest.mark.parametrize(
    "attack", ["extra", "traversal", "duplicate", "missing", "reordered"]
)
def test_archive_member_set_traversal_and_duplicates_are_rejected(
    tmp_path: Path,
    attack: str,
) -> None:
    paths = _build(tmp_path)
    entries = _read_entries(paths["bundle"])
    if attack == "extra":
        entries.append(("unreviewed.py", b"pass\n"))
    elif attack == "traversal":
        entries.append(("../escape.py", b"pass\n"))
    elif attack == "duplicate":
        entries.append(entries[0])
    elif attack == "missing":
        entries.pop()
    else:
        entries[0], entries[1] = entries[1], entries[0]
    if attack == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            rewritten = _archive_bytes(entries)
    else:
        rewritten = _archive_bytes(entries)
    _rewrite_bundle_and_manifest(paths, rewritten)
    with pytest.raises(AssessmentSourceBundleError):
        _validate(paths)


def test_writable_bundle_path_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    paths["bundle"].chmod(0o644)
    with pytest.raises(AssessmentSourceBundleError, match="single-link immutable"):
        _validate(paths)


def test_symlink_bundle_path_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    link = tmp_path / "bundle-link.pyz"
    link.symlink_to(paths["bundle"])
    paths["bundle"] = link
    with pytest.raises(AssessmentSourceBundleError, match="single-link immutable"):
        _validate(paths)


def test_hardlink_bundle_path_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    os.link(paths["bundle"], tmp_path / "bundle-alias.pyz")
    with pytest.raises(AssessmentSourceBundleError, match="single-link immutable"):
        _validate(paths)


@pytest.mark.parametrize("field", ["python", "bwrap"])
def test_interpreter_and_bwrap_substitution_are_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    paths = _build(tmp_path)
    # The replacement deliberately has identical bytes. The canonical path is
    # part of the binding, so substitution still fails.
    replacement = _write_executable(
        tmp_path / f"replacement-{field}", paths[field].read_bytes()
    )
    paths[field] = replacement
    expected = "Python interpreter" if field == "python" else "bwrap executable"
    with pytest.raises(AssessmentSourceBundleError, match=expected):
        _validate(paths)


def test_executable_byte_mutation_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    paths["python"].chmod(0o755)
    paths["python"].write_bytes(b"different-python\n")
    paths["python"].chmod(0o555)
    with pytest.raises(AssessmentSourceBundleError, match="Python interpreter"):
        _validate(paths)


def _linux_memfd_available() -> bool:
    names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    )
    return (
        sys.platform == "linux"
        and hasattr(os, "memfd_create")
        and hasattr(os, "MFD_ALLOW_SEALING")
        and all(hasattr(fcntl, name) for name in names)
    )


@pytest.mark.skipif(
    not _linux_memfd_available(), reason="Linux memfd seals unavailable"
)
def test_linux_loader_returns_inheritable_read_only_fully_sealed_fd(
    tmp_path: Path,
) -> None:
    paths = _build(tmp_path)
    expected = hashlib.sha256(paths["bundle"].read_bytes()).hexdigest()
    descriptor = load_sealed_assessment_bundle_memfd(
        source_root=paths["source_root"],
        bundle_path=paths["bundle"],
        manifest_path=paths["manifest"],
        python_executable=paths["python"],
        bwrap_executable=paths["bwrap"],
    )
    try:
        payload = validate_sealed_assessment_bundle_fd(descriptor, expected)
        assert payload == paths["bundle"].read_bytes()
        assert os.get_inheritable(descriptor)
        assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        expected_seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == expected_seals
        with pytest.raises(OSError):
            os.write(descriptor, b"x")
        writable = os.open(f"/proc/self/fd/{descriptor}", os.O_RDWR)
        try:
            with pytest.raises(OSError):
                os.write(writable, b"x")
        finally:
            os.close(writable)
        with zipfile.ZipFile(f"/proc/self/fd/{descriptor}", "r") as archive:
            assert archive.read(MAIN_MEMBER) == MAIN_SOURCE
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    not _linux_memfd_available(), reason="Linux memfd seals unavailable"
)
def test_linux_unsealed_memfd_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    payload = paths["bundle"].read_bytes()
    writer = os.memfd_create("unsealed-ctaa-assessor", os.MFD_ALLOW_SEALING)
    reader = -1
    try:
        os.write(writer, payload)
        os.fchmod(writer, 0o400)
        reader = os.open(f"/proc/self/fd/{writer}", os.O_RDONLY)
        os.set_inheritable(reader, True)
        with pytest.raises(AssessmentSourceBundleError, match="custody differs"):
            validate_sealed_assessment_bundle_fd(
                reader, hashlib.sha256(payload).hexdigest()
            )
    finally:
        if reader >= 0:
            os.close(reader)
        os.close(writer)


def test_non_linux_memfd_request_fails_closed_when_unavailable(tmp_path: Path) -> None:
    if _linux_memfd_available():
        pytest.skip("sealing is available on this platform")
    paths = _build(tmp_path)
    with pytest.raises(AssessmentSourceBundleError, match="sealing is unavailable"):
        load_sealed_assessment_bundle_memfd(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=paths["manifest"],
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
        )


def test_manifest_and_bundle_are_single_link_read_only_files(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    for name in ("bundle", "manifest"):
        metadata = paths[name].stat()
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o444


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    original_bundle = paths["bundle"].read_bytes()
    with pytest.raises(AssessmentSourceBundleError, match="created exclusively"):
        build_assessment_source_bundle(
            source_root=paths["source_root"],
            bundle_path=paths["bundle"],
            manifest_path=tmp_path / "second-manifest.json",
            python_executable=paths["python"],
            bwrap_executable=paths["bwrap"],
        )
    assert paths["bundle"].read_bytes() == original_bundle
