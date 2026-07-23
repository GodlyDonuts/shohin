#!/usr/bin/env python3
"""Deterministic, immutable source bundles for the CTAA assessor.

The manifest is deliberately a sidecar. Embedding the final zip SHA-256 in the
zip itself would require a circular hash commitment.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import zipfile

BUNDLE_SCHEMA = "r12_ctaa_assessment_source_bundle_v1"
ARCHIVE_FORMAT = "deterministic_python_zipapp_stored_v1"
MAIN_MEMBER = "__main__.py"
MAIN_SOURCE = (
    b'from assess_ctaa_evidence import main\n\nif __name__ == "__main__":\n    main()\n'
)

# This is the exact transitive local import closure rooted at
# assess_ctaa_evidence.py. It is intentionally not discovered at build time:
# changing the executable source set must be an explicit reviewed code change.
ASSESSMENT_SOURCE_MEMBERS = (
    "assess_ctaa_evidence.py",
    "build_ctaa_runtime_intervention_plan.py",
    "commit_ctaa_raw_evidence.py",
    "ctaa_access_registry.py",
    "ctaa_assessment.py",
    "ctaa_bootstrap_seed_receipt.py",
    "ctaa_core_training.py",
    "ctaa_evaluation_io.py",
    "ctaa_intervention_protocol.py",
    "ctaa_neural_core.py",
    "ctaa_packet_io.py",
    "ctaa_run_contract.py",
    "ctaa_runtime_bundle.py",
    "ctaa_runtime_evidence.py",
    "ctaa_runtime_evidence_finalizer.py",
    "ctaa_runtime_execution_artifact.py",
    "ctaa_runtime_execution_engine.py",
    "ctaa_runtime_execution_projection.py",
    "ctaa_runtime_execution_receipt.py",
    "ctaa_runtime_execution_set.py",
    "ctaa_runtime_interventions.py",
    "ctaa_runtime_plan_replay.py",
    "ctaa_statistical_gate_spec.py",
    "ctaa_trunk_compiler.py",
    "run_ctaa_packet_executor.py",
)
ARCHIVE_MEMBERS = tuple(sorted((*ASSESSMENT_SOURCE_MEMBERS, MAIN_MEMBER)))

_ENTRYPOINT_MODULE = "assess_ctaa_evidence"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = (stat.S_IFREG | 0o444) << 16
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_EXECUTABLE_BYTES = 1024 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_SHA256_HEX_LENGTH = 64


class AssessmentSourceBundleError(ValueError):
    """The assessor source bundle or its custody metadata is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_directory(path: Path | str, label: str) -> Path:
    absolute = _absolute(path)
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise AssessmentSourceBundleError(f"{label} must be a non-symlink directory")
    return absolute


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_file(
    path: Path | str,
    label: str,
    *,
    require_read_only: bool,
    maximum_bytes: int,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise AssessmentSourceBundleError(f"{label} byte limit differs")
    if not hasattr(os, "O_NOFOLLOW"):
        raise AssessmentSourceBundleError("O_NOFOLLOW is required")
    absolute = _absolute(path)
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size < 0
        or metadata.st_size > maximum_bytes
        or (require_read_only and metadata.st_mode & 0o222)
    ):
        qualifier = (
            "single-link immutable" if require_read_only else "single-link regular"
        )
        raise AssessmentSourceBundleError(f"{label} must be a {qualifier} file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(metadata):
            raise AssessmentSourceBundleError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(_READ_CHUNK, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise AssessmentSourceBundleError(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _metadata_identity(after) != _metadata_identity(before)
        or after.st_nlink != 1
        or (require_read_only and after.st_mode & 0o222)
        or observed != before.st_size
    ):
        raise AssessmentSourceBundleError(f"{label} changed while being read")
    return b"".join(chunks)


def read_immutable_file(
    path: Path | str,
    label: str = "immutable file",
    *,
    maximum_bytes: int = _MAX_BUNDLE_BYTES,
) -> bytes:
    """Read a read-only, single-link regular file without following symlinks."""

    return _read_regular_file(
        path,
        label,
        require_read_only=True,
        maximum_bytes=maximum_bytes,
    )


def write_immutable_file_once(
    path: Path | str,
    payload: bytes,
    label: str = "immutable file",
) -> str:
    """Create a single-link mode-0444 file exactly once via an open parent fd."""

    if not isinstance(payload, bytes):
        raise TypeError("immutable payload must be bytes")
    if not hasattr(os, "O_NOFOLLOW"):
        raise AssessmentSourceBundleError("O_NOFOLLOW is required")
    absolute = _absolute(path)
    parent = _require_directory(absolute.parent, f"{label} parent")
    if absolute.name in {"", ".", ".."}:
        raise AssessmentSourceBundleError(f"{label} filename differs")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_descriptor = os.open(parent, directory_flags)
    except OSError as error:
        raise AssessmentSourceBundleError(
            f"{label} parent cannot be opened safely"
        ) from error
    descriptor = -1
    created = False
    published_identity: tuple[int, ...] | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                absolute.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            created = True
        except OSError as error:
            raise AssessmentSourceBundleError(
                f"{label} cannot be created exclusively"
            ) from error
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise AssessmentSourceBundleError(f"{label} write made no progress")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o444
            or final.st_size != len(payload)
        ):
            raise AssessmentSourceBundleError(f"{label} publication differs")
        published_identity = _metadata_identity(final)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                os.unlink(absolute.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    try:
        published = absolute.lstat()
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} publication disappeared") from error
    if (
        published_identity is None
        or _metadata_identity(published) != published_identity
    ):
        raise AssessmentSourceBundleError(f"{label} publication identity differs")
    observed = read_immutable_file(
        absolute,
        label,
        maximum_bytes=max(len(payload), 1),
    )
    if observed != payload:
        raise AssessmentSourceBundleError(f"{label} publication bytes differ")
    return _sha256(payload)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssessmentSourceBundleError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AssessmentSourceBundleError(
            "manifest is not canonical JSON data"
        ) from error


def _parse_canonical_manifest(raw: bytes) -> dict[str, object]:
    try:
        manifest = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssessmentSourceBundleError(f"non-finite manifest value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssessmentSourceBundleError("bundle manifest JSON differs") from error
    if not isinstance(manifest, dict) or canonical_manifest_bytes(manifest) != raw:
        raise AssessmentSourceBundleError("bundle manifest is not canonical")
    return manifest


def _validate_requested_allowlist(source_members: Sequence[str]) -> tuple[str, ...]:
    observed = tuple(source_members)
    if observed != ASSESSMENT_SOURCE_MEMBERS:
        raise AssessmentSourceBundleError("assessment source allowlist differs")
    return observed


def _local_imports(source: bytes, member: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=member)
    except (SyntaxError, ValueError) as error:
        raise AssessmentSourceBundleError(
            f"assessment source {member} is not valid Python"
        ) from error
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def _validate_import_closure(source_root: Path, sources: Mapping[str, bytes]) -> None:
    allowed_modules = {Path(member).stem for member in ASSESSMENT_SOURCE_MEMBERS}
    try:
        local_modules = {
            Path(entry.name).stem
            for entry in os.scandir(source_root)
            if entry.name.endswith(".py") and entry.is_file(follow_symlinks=False)
        }
    except OSError as error:
        raise AssessmentSourceBundleError(
            "source root cannot be enumerated safely"
        ) from error
    graph: dict[str, set[str]] = {}
    for member, source in sources.items():
        module = Path(member).stem
        dependencies = _local_imports(source, member)
        undeclared = {
            dependency
            for dependency in dependencies
            if (
                dependency in local_modules
                or dependency.startswith("ctaa_")
                or dependency
                in {
                    "build_ctaa_runtime_intervention_plan",
                    "commit_ctaa_raw_evidence",
                    "run_ctaa_packet_executor",
                }
            )
            and dependency not in allowed_modules
        }
        if undeclared:
            raise AssessmentSourceBundleError(
                f"assessment source {member} imports undeclared local modules: "
                + ", ".join(sorted(undeclared))
            )
        graph[module] = dependencies & allowed_modules
    reachable: set[str] = set()
    pending = [_ENTRYPOINT_MODULE]
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(graph.get(module, set()) - reachable)
    if reachable != allowed_modules:
        missing = sorted(allowed_modules - reachable)
        extra = sorted(reachable - allowed_modules)
        raise AssessmentSourceBundleError(
            f"assessment local import closure differs; missing={missing}, extra={extra}"
        )


def _read_sources(
    source_root: Path | str,
    source_members: Sequence[str],
) -> tuple[Path, dict[str, bytes]]:
    root = _require_directory(source_root, "assessment source root")
    members = _validate_requested_allowlist(source_members)
    sources = {
        member: _read_regular_file(
            root / member,
            f"assessment source {member}",
            require_read_only=False,
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
        for member in members
    }
    _validate_import_closure(root, sources)
    return root, sources


def _canonical_executable(path: Path | str, label: str) -> tuple[Path, bytes]:
    try:
        resolved = _absolute(path).resolve(strict=True)
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} cannot be resolved") from error
    payload = _read_regular_file(
        resolved,
        label,
        require_read_only=False,
        maximum_bytes=_MAX_EXECUTABLE_BYTES,
    )
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise AssessmentSourceBundleError(f"{label} is unavailable") from error
    if not payload or not mode & 0o111:
        raise AssessmentSourceBundleError(f"{label} must be a nonempty executable")
    return resolved, payload


def _executable_record(path: Path | str, label: str) -> dict[str, object]:
    resolved, payload = _canonical_executable(path, label)
    return {
        "path": str(resolved),
        "sha256": _sha256(payload),
        "size": len(payload),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = _ZIP_EXTERNAL_ATTR
    info.extra = b""
    info.comment = b""
    return info


def _build_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    if tuple(sorted(members)) != ARCHIVE_MEMBERS:
        raise AssessmentSourceBundleError("archive member set differs from allowlist")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        archive.comment = b""
        for name in ARCHIVE_MEMBERS:
            archive.writestr(_zip_info(name), members[name])
    payload = stream.getvalue()
    if not payload or len(payload) > _MAX_BUNDLE_BYTES:
        raise AssessmentSourceBundleError("assessment source bundle size differs")
    return payload


def _member_record(payload: bytes) -> dict[str, object]:
    return {"sha256": _sha256(payload), "size": len(payload)}


def build_assessment_source_bundle(
    *,
    source_root: Path | str,
    bundle_path: Path | str,
    manifest_path: Path | str,
    python_executable: Path | str,
    bwrap_executable: Path | str,
    source_members: Sequence[str] = ASSESSMENT_SOURCE_MEMBERS,
) -> dict[str, object]:
    """Build and publish a deterministic assessor zipapp and sidecar manifest."""

    if _absolute(bundle_path) == _absolute(manifest_path):
        raise AssessmentSourceBundleError("bundle and manifest paths must differ")
    _root, sources = _read_sources(source_root, source_members)
    archive_members = {**sources, MAIN_MEMBER: MAIN_SOURCE}
    bundle = _build_zip_bytes(archive_members)
    manifest: dict[str, object] = {
        "archive_format": ARCHIVE_FORMAT,
        "bwrap_executable": _executable_record(bwrap_executable, "bwrap executable"),
        "bundle_sha256": _sha256(bundle),
        "bundle_size": len(bundle),
        "entrypoint": MAIN_MEMBER,
        "members": {
            name: _member_record(archive_members[name]) for name in ARCHIVE_MEMBERS
        },
        "python_interpreter": _executable_record(
            python_executable, "Python interpreter"
        ),
        "schema": BUNDLE_SCHEMA,
        "source_allowlist": list(ASSESSMENT_SOURCE_MEMBERS),
    }
    manifest_payload = canonical_manifest_bytes(manifest)
    write_immutable_file_once(bundle_path, bundle, "assessment source bundle")
    write_immutable_file_once(
        manifest_path, manifest_payload, "assessment source manifest"
    )
    return manifest


def _validate_manifest_shape(manifest: Mapping[str, object]) -> None:
    expected_keys = {
        "archive_format",
        "bwrap_executable",
        "bundle_sha256",
        "bundle_size",
        "entrypoint",
        "members",
        "python_interpreter",
        "schema",
        "source_allowlist",
    }
    if set(manifest) != expected_keys:
        raise AssessmentSourceBundleError("bundle manifest keys differ")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("archive_format") != ARCHIVE_FORMAT
        or manifest.get("entrypoint") != MAIN_MEMBER
        or manifest.get("source_allowlist") != list(ASSESSMENT_SOURCE_MEMBERS)
        or not _is_sha256(manifest.get("bundle_sha256"))
        or type(manifest.get("bundle_size")) is not int
        or not 0 < manifest["bundle_size"] <= _MAX_BUNDLE_BYTES
        or not isinstance(manifest.get("members"), dict)
    ):
        raise AssessmentSourceBundleError("bundle manifest contract differs")
    members = manifest["members"]
    assert isinstance(members, dict)
    if tuple(sorted(members)) != ARCHIVE_MEMBERS:
        raise AssessmentSourceBundleError("bundle manifest member set differs")
    for name in ARCHIVE_MEMBERS:
        record = members[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or not _is_sha256(record.get("sha256"))
            or type(record.get("size")) is not int
            or not 0 < record["size"] <= _MAX_SOURCE_BYTES
        ):
            raise AssessmentSourceBundleError(f"bundle manifest member {name} differs")
    for field in ("python_interpreter", "bwrap_executable"):
        record = manifest.get(field)
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size"}
            or not isinstance(record.get("path"), str)
            or not os.path.isabs(record["path"])
            or not _is_sha256(record.get("sha256"))
            or type(record.get("size")) is not int
            or not 0 < record["size"] <= _MAX_EXECUTABLE_BYTES
        ):
            raise AssessmentSourceBundleError(f"bundle manifest {field} differs")


def _safe_archive_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        return False
    parts = name.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _validate_archive(
    bundle: bytes,
    member_records: Mapping[str, object],
) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(bundle), mode="r") as archive:
            if archive.comment != b"":
                raise AssessmentSourceBundleError("archive comment differs")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if any(not _safe_archive_name(name) for name in names):
                raise AssessmentSourceBundleError(
                    "archive contains an unsafe member name"
                )
            if len(names) != len(set(names)):
                raise AssessmentSourceBundleError("archive contains duplicate members")
            if tuple(names) != ARCHIVE_MEMBERS:
                raise AssessmentSourceBundleError("archive member set differs")
            result: dict[str, bytes] = {}
            for info in infos:
                if (
                    info.date_time != _FIXED_ZIP_TIME
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.create_system != 3
                    or info.create_version != 20
                    or info.extract_version != 20
                    or info.flag_bits != 0
                    or info.internal_attr != 0
                    or info.external_attr != _ZIP_EXTERNAL_ATTR
                    or info.extra != b""
                    or info.comment != b""
                    or info.is_dir()
                ):
                    raise AssessmentSourceBundleError(
                        f"archive member {info.filename} metadata differs"
                    )
                record = member_records[info.filename]
                assert isinstance(record, dict)
                if (
                    info.file_size != record["size"]
                    or info.file_size > _MAX_SOURCE_BYTES
                ):
                    raise AssessmentSourceBundleError(
                        f"archive member {info.filename} size differs"
                    )
                payload = archive.read(info)
                if (
                    len(payload) != record["size"]
                    or _sha256(payload) != record["sha256"]
                ):
                    raise AssessmentSourceBundleError(
                        f"archive member {info.filename} hash differs"
                    )
                result[info.filename] = payload
            if archive.testzip() is not None:
                raise AssessmentSourceBundleError("archive CRC validation failed")
            if _build_zip_bytes(result) != bundle:
                raise AssessmentSourceBundleError("archive bytes are not canonical")
            return result
    except AssessmentSourceBundleError:
        raise
    except Exception as error:
        raise AssessmentSourceBundleError(
            "assessment source bundle archive differs"
        ) from error


def _validate_assessment_source_bundle_material(
    *,
    source_root: Path | str,
    bundle_path: Path | str,
    manifest_path: Path | str,
    python_executable: Path | str,
    bwrap_executable: Path | str,
    source_members: Sequence[str] = ASSESSMENT_SOURCE_MEMBERS,
) -> tuple[dict[str, object], bytes]:
    """Validate once and retain the exact bundle bytes that passed."""

    _root, sources = _read_sources(source_root, source_members)
    bundle = read_immutable_file(
        bundle_path,
        "assessment source bundle",
        maximum_bytes=_MAX_BUNDLE_BYTES,
    )
    manifest_raw = read_immutable_file(
        manifest_path,
        "assessment source manifest",
        maximum_bytes=_MAX_SOURCE_BYTES,
    )
    manifest = _parse_canonical_manifest(manifest_raw)
    _validate_manifest_shape(manifest)
    if manifest["bundle_size"] != len(bundle) or manifest["bundle_sha256"] != _sha256(
        bundle
    ):
        raise AssessmentSourceBundleError(
            "assessment source bundle aggregate hash differs"
        )
    expected_python = _executable_record(python_executable, "Python interpreter")
    expected_bwrap = _executable_record(bwrap_executable, "bwrap executable")
    if manifest["python_interpreter"] != expected_python:
        raise AssessmentSourceBundleError("Python interpreter binding differs")
    if manifest["bwrap_executable"] != expected_bwrap:
        raise AssessmentSourceBundleError("bwrap executable binding differs")
    records = manifest["members"]
    assert isinstance(records, dict)
    archived = _validate_archive(bundle, records)
    expected_members = {**sources, MAIN_MEMBER: MAIN_SOURCE}
    for name in ARCHIVE_MEMBERS:
        if archived[name] != expected_members[name]:
            raise AssessmentSourceBundleError(
                f"archive member {name} differs from its current source"
            )
    return manifest, bundle


def validate_assessment_source_bundle(
    *,
    source_root: Path | str,
    bundle_path: Path | str,
    manifest_path: Path | str,
    python_executable: Path | str,
    bwrap_executable: Path | str,
    source_members: Sequence[str] = ASSESSMENT_SOURCE_MEMBERS,
) -> dict[str, object]:
    """Validate custody, bytes, sources, executables, and archive metadata."""

    manifest, _ = _validate_assessment_source_bundle_material(
        source_root=source_root,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        python_executable=python_executable,
        bwrap_executable=bwrap_executable,
        source_members=source_members,
    )
    return manifest


def _linux_seal_constants() -> tuple[int, int, int]:
    required_names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    )
    if (
        sys.platform != "linux"
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
        or any(not hasattr(fcntl, name) for name in required_names)
    ):
        raise AssessmentSourceBundleError("Linux memfd sealing is unavailable")
    seals = (
        fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    )
    return fcntl.F_ADD_SEALS, fcntl.F_GET_SEALS, seals


def _pread_all(descriptor: int, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < byte_count:
        chunk = os.pread(descriptor, min(_READ_CHUNK, byte_count - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def validate_sealed_assessment_bundle_fd(
    descriptor: int,
    expected_sha256: str,
) -> bytes:
    """Validate and return bytes from an inheritable, read-only sealed memfd."""

    _add_seals, get_seals, required_seals = _linux_seal_constants()
    if type(descriptor) is not int or descriptor < 0 or not _is_sha256(expected_sha256):
        raise AssessmentSourceBundleError("sealed assessor descriptor arguments differ")
    try:
        metadata = os.fstat(descriptor)
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        observed_seals = fcntl.fcntl(descriptor, get_seals)
        inheritable = os.get_inheritable(descriptor)
    except OSError as error:
        raise AssessmentSourceBundleError(
            "sealed assessor descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or not 0 < metadata.st_size <= _MAX_BUNDLE_BYTES
        or access_mode != os.O_RDONLY
        or observed_seals != required_seals
        or not inheritable
    ):
        raise AssessmentSourceBundleError("sealed assessor descriptor custody differs")
    payload = _pread_all(descriptor, metadata.st_size)
    after = os.fstat(descriptor)
    if (
        len(payload) != metadata.st_size
        or _metadata_identity(after) != _metadata_identity(metadata)
        or _sha256(payload) != expected_sha256
    ):
        raise AssessmentSourceBundleError("sealed assessor descriptor bytes differ")
    return payload


def load_sealed_assessment_bundle_memfd(
    *,
    source_root: Path | str,
    bundle_path: Path | str,
    manifest_path: Path | str,
    python_executable: Path | str,
    bwrap_executable: Path | str,
    source_members: Sequence[str] = ASSESSMENT_SOURCE_MEMBERS,
) -> int:
    """Load a validated bundle into an inheritable O_RDONLY sealed Linux memfd."""

    manifest, bundle = _validate_assessment_source_bundle_material(
        source_root=source_root,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        python_executable=python_executable,
        bwrap_executable=bwrap_executable,
        source_members=source_members,
    )
    expected_sha256 = manifest["bundle_sha256"]
    assert isinstance(expected_sha256, str)
    add_seals, _get_seals, required_seals = _linux_seal_constants()
    writer = -1
    reader = -1
    try:
        writer = os.memfd_create(
            "r12-ctaa-assessor-source-bundle",
            os.MFD_ALLOW_SEALING,
        )
        written = 0
        while written < len(bundle):
            count = os.write(writer, bundle[written:])
            if count <= 0:
                raise AssessmentSourceBundleError(
                    "sealed bundle write made no progress"
                )
            written += count
        os.fchmod(writer, 0o400)
        os.lseek(writer, 0, os.SEEK_SET)
        fcntl.fcntl(writer, add_seals, required_seals)
        reader = os.open(
            f"/proc/self/fd/{writer}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(reader, True)
        os.close(writer)
        writer = -1
        validate_sealed_assessment_bundle_fd(reader, expected_sha256)
        return reader
    except (OSError, AssessmentSourceBundleError) as error:
        if reader >= 0:
            os.close(reader)
        if writer >= 0:
            os.close(writer)
        if isinstance(error, AssessmentSourceBundleError):
            raise
        raise AssessmentSourceBundleError(
            "sealed assessor bundle creation failed"
        ) from error


__all__ = [
    "ARCHIVE_MEMBERS",
    "ASSESSMENT_SOURCE_MEMBERS",
    "AssessmentSourceBundleError",
    "BUNDLE_SCHEMA",
    "MAIN_MEMBER",
    "MAIN_SOURCE",
    "build_assessment_source_bundle",
    "canonical_manifest_bytes",
    "load_sealed_assessment_bundle_memfd",
    "read_immutable_file",
    "validate_assessment_source_bundle",
    "validate_sealed_assessment_bundle_fd",
    "write_immutable_file_once",
]
