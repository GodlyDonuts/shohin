from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Callable

import pytest

import ctaa_score_snapshot as snapshot


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _immutable(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return {
        "schema": snapshot.MEMBER_SCHEMA,
        "path": str(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _run_id(seed: int, arm: str, dataset: str) -> str:
    return f"seed-{seed}:{arm}:{dataset}"


def _fixture(tmp_path: Path) -> tuple[dict[str, object], bytes]:
    seeds = (101, 202, 303, 404, 505)
    root = tmp_path / "inputs"
    members: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for dataset in snapshot.DATASETS:
        member_id = f"oracle:{dataset}"
        members.append(
            {
                **_immutable(
                    root / "oracles" / f"{dataset}.jsonl",
                    f'{{"answer":"{dataset}"}}\n'.encode(),
                ),
                "member_id": member_id,
                "role": "oracle",
            }
        )
    for seed in seeds:
        for arm in snapshot.ARMS:
            for dataset in snapshot.DATASETS:
                run_id = _run_id(seed, arm, dataset)
                receipt_id = f"receipt:{run_id}"
                evidence_id = f"evidence:{run_id}"
                members.extend(
                    (
                        {
                            **_immutable(
                                root / "receipts" / f"{run_id}.json",
                                f'{{"run_id":"{run_id}"}}\n'.encode(),
                            ),
                            "member_id": receipt_id,
                            "role": "raw_evidence_receipt",
                        },
                        {
                            **_immutable(
                                root / "evidence" / f"{run_id}.jsonl",
                                f'{{"prediction":"{run_id}"}}\n'.encode(),
                            ),
                            "member_id": evidence_id,
                            "role": "evidence",
                        },
                    )
                )
                runs.append(
                    {
                        "schema": snapshot.RUN_INPUT_SCHEMA,
                        "run_id": run_id,
                        "seed": seed,
                        "arm": arm,
                        "dataset": dataset,
                        "receipt_member_id": receipt_id,
                        "evidence_member_id": evidence_id,
                        "oracle_member_id": f"oracle:{dataset}",
                        "parent_evidence_member_id": (
                            None
                            if dataset == "base"
                            else f"evidence:{_run_id(seed, arm, 'base')}"
                        ),
                    }
                )
    members.sort(key=lambda member: str(member["member_id"]))
    unsigned = {
        "schema": snapshot.INVENTORY_SCHEMA,
        "partition": "development",
        "manifest_sha256": _sha256(b"manifest"),
        "board_sha256": _sha256(b"board"),
        "run_contract_sha256": _sha256(b"contract"),
        "runtime_execution_set_sha256": _sha256(b"execution-set"),
        "runs": runs,
        "members": members,
    }
    inventory = snapshot.finalize_score_input_inventory(unsigned)
    return inventory, snapshot.encode_score_input_inventory(inventory)


def _edit_inventory(
    inventory: dict[str, object],
    change: Callable[[dict[str, object]], None],
) -> bytes:
    unsigned = json.loads(snapshot.canonical_json_bytes(inventory).decode())
    unsigned.pop("inventory_sha256")
    change(unsigned)
    finalized = snapshot.finalize_score_input_inventory(unsigned)
    return snapshot.encode_score_input_inventory(finalized)


def _repack_snapshot(
    value: bytes,
    change: Callable[[dict[str, object]], None],
    *,
    canonical: bool = True,
) -> bytes:
    (header_length,) = struct.unpack(">Q", value[:8])
    header = json.loads(value[8 : 8 + header_length])
    content = value[8 + header_length :]
    change(header)
    if canonical:
        header_raw = snapshot.canonical_json_bytes(header)
    else:
        header_raw = json.dumps(header, indent=2, sort_keys=True).encode()
    return struct.pack(">Q", len(header_raw)) + header_raw + content


def test_inventory_and_snapshot_are_deterministic_and_lossless(
    tmp_path: Path,
) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    first = snapshot.build_score_snapshot(inventory_raw)
    second = snapshot.build_score_snapshot(inventory_raw)
    assert first == second
    header, members = snapshot.verify_score_snapshot(
        first,
        expected_inventory_sha256=str(inventory["inventory_sha256"]),
    )
    assert header["run_count"] == 40
    assert len(header["runs"]) == 40
    assert len(header["members"]) == 82
    assert len(members) == 82
    for member in inventory["members"]:
        assert members[member["member_id"]] == Path(member["path"]).read_bytes()


def test_each_source_is_read_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inventory, inventory_raw = _fixture(tmp_path)
    observed: list[str] = []
    original = snapshot._read_immutable_member_once

    def counted(member: dict[str, object]) -> bytes:
        observed.append(str(member["member_id"]))
        return original(member)

    monkeypatch.setattr(snapshot, "_read_immutable_member_once", counted)
    snapshot.build_score_snapshot(inventory_raw)
    assert len(observed) == 82
    assert len(set(observed)) == 82


@pytest.mark.parametrize(
    "change,pattern",
    (
        (lambda value: value["runs"].pop(), "exactly 40"),
        (
            lambda value: value["runs"].append(value["runs"][0]),
            "exactly 40|repeat",
        ),
        (
            lambda value: value["runs"].__setitem__(
                slice(0, 2), reversed(value["runs"][:2])
            ),
            "order",
        ),
        (lambda value: value["members"].pop(), "member set"),
        (
            lambda value: value["members"].append(value["members"][0]),
            "identity|member set",
        ),
        (
            lambda value: value["members"].__setitem__(
                slice(0, 2), reversed(value["members"][:2])
            ),
            "order",
        ),
    ),
)
def test_missing_duplicate_or_swapped_inventory_fails_before_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict[str, object]], None],
    pattern: str,
) -> None:
    inventory, _inventory_raw = _fixture(tmp_path)
    unsigned = json.loads(snapshot.canonical_json_bytes(inventory).decode())
    unsigned.pop("inventory_sha256")
    change(unsigned)
    raw = snapshot.canonical_json_bytes(
        {
            **unsigned,
            "inventory_sha256": _sha256(snapshot.canonical_json_bytes(unsigned)),
        }
    )
    opened = 0

    def forbidden(_member: object) -> bytes:
        nonlocal opened
        opened += 1
        raise AssertionError("source open preceded full inventory validation")

    monkeypatch.setattr(snapshot, "_read_immutable_member_once", forbidden)
    with pytest.raises(snapshot.ScoreSnapshotError, match=pattern):
        snapshot.build_score_snapshot(raw)
    assert opened == 0


def test_cross_wired_evidence_member_fails_before_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, _inventory_raw = _fixture(tmp_path)
    unsigned = json.loads(snapshot.canonical_json_bytes(inventory).decode())
    unsigned.pop("inventory_sha256")
    first, second = unsigned["runs"][:2]
    first["evidence_member_id"], second["evidence_member_id"] = (
        second["evidence_member_id"],
        first["evidence_member_id"],
    )
    raw = snapshot.canonical_json_bytes(
        {
            **unsigned,
            "inventory_sha256": _sha256(snapshot.canonical_json_bytes(unsigned)),
        }
    )
    opened = 0

    def forbidden(_member: object) -> bytes:
        nonlocal opened
        opened += 1
        raise AssertionError

    monkeypatch.setattr(snapshot, "_read_immutable_member_once", forbidden)
    with pytest.raises(snapshot.ScoreSnapshotError, match="run/member binding"):
        snapshot.build_score_snapshot(raw)
    assert opened == 0


def test_extra_keys_and_noncanonical_inventory_are_rejected_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    opened = 0

    def forbidden(_member: object) -> bytes:
        nonlocal opened
        opened += 1
        raise AssertionError

    monkeypatch.setattr(snapshot, "_read_immutable_member_once", forbidden)
    extra = json.loads(inventory_raw)
    extra["unexpected"] = True
    with pytest.raises(snapshot.ScoreSnapshotError, match="schema"):
        snapshot.build_score_snapshot(snapshot.canonical_json_bytes(extra))
    pretty = json.dumps(inventory, indent=2, sort_keys=True).encode()
    with pytest.raises(snapshot.ScoreSnapshotError, match="not canonical"):
        snapshot.build_score_snapshot(pretty)
    assert opened == 0


def test_duplicate_inventory_json_key_is_rejected(tmp_path: Path) -> None:
    _inventory, inventory_raw = _fixture(tmp_path)
    duplicate = inventory_raw.replace(
        b'{"board_sha256":',
        b'{"board_sha256":"' + b"0" * 64 + b'","board_sha256":',
        1,
    )
    with pytest.raises(snapshot.ScoreSnapshotError, match="duplicate key"):
        snapshot.decode_score_input_inventory(duplicate)


def test_symlinked_intermediate_parent_is_rejected(tmp_path: Path) -> None:
    inventory, _inventory_raw = _fixture(tmp_path)
    target = Path(inventory["members"][0]["path"])
    alias = tmp_path / "aliased-inputs"
    alias.symlink_to(target.parent)

    def change(value: dict[str, object]) -> None:
        value["members"][0]["path"] = str(alias / target.name)

    raw = _edit_inventory(inventory, change)
    with pytest.raises(snapshot.ScoreSnapshotError, match="parent cannot be opened"):
        snapshot.build_score_snapshot(raw)


def test_source_mutation_after_inventory_is_rejected(tmp_path: Path) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    target = Path(inventory["members"][10]["path"])
    original = target.read_bytes()
    target.chmod(0o644)
    target.write_bytes(b"x" * len(original))
    target.chmod(0o444)
    with pytest.raises(snapshot.ScoreSnapshotError, match="changed or differs"):
        snapshot.build_score_snapshot(inventory_raw)


def test_writable_or_hardlinked_source_is_rejected(tmp_path: Path) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    writable = Path(inventory["members"][4]["path"])
    writable.chmod(0o644)
    with pytest.raises(snapshot.ScoreSnapshotError, match="immutable file"):
        snapshot.build_score_snapshot(inventory_raw)
    writable.chmod(0o444)
    sibling = writable.with_suffix(".alias")
    os.link(writable, sibling)
    with pytest.raises(snapshot.ScoreSnapshotError, match="immutable file"):
        snapshot.build_score_snapshot(inventory_raw)


def test_snapshot_member_mutation_and_swap_are_rejected(tmp_path: Path) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    encoded = snapshot.build_score_snapshot(inventory_raw)
    mutated = bytearray(encoded)
    mutated[-1] ^= 1
    with pytest.raises(snapshot.ScoreSnapshotError, match="content differs"):
        snapshot.verify_score_snapshot(
            bytes(mutated),
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )

    def swap_members(header: dict[str, object]) -> None:
        header["members"][0], header["members"][1] = (
            header["members"][1],
            header["members"][0],
        )

    swapped = _repack_snapshot(encoded, swap_members)
    with pytest.raises(snapshot.ScoreSnapshotError, match="identity|set"):
        snapshot.verify_score_snapshot(
            swapped,
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )


def test_snapshot_extra_key_noncanonical_header_and_trailing_bytes_fail(
    tmp_path: Path,
) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    encoded = snapshot.build_score_snapshot(inventory_raw)

    with pytest.raises(snapshot.ScoreSnapshotError, match="schema"):
        snapshot.verify_score_snapshot(
            _repack_snapshot(encoded, lambda header: header.update(extra=True)),
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )
    with pytest.raises(snapshot.ScoreSnapshotError, match="not canonical"):
        snapshot.verify_score_snapshot(
            _repack_snapshot(encoded, lambda _header: None, canonical=False),
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )
    with pytest.raises(snapshot.ScoreSnapshotError, match="content differs"):
        snapshot.verify_score_snapshot(
            encoded + b"x",
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )


def test_snapshot_inventory_binding_is_required(tmp_path: Path) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    encoded = snapshot.build_score_snapshot(inventory_raw)
    with pytest.raises(snapshot.ScoreSnapshotError, match="identity differs"):
        snapshot.verify_score_snapshot(
            encoded,
            expected_inventory_sha256="f" * 64,
        )


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
    not _linux_memfd_available(),
    reason="Linux memfd sealing unavailable",
)
def test_linux_memfd_is_fully_sealed_and_has_independent_read_offsets(
    tmp_path: Path,
) -> None:
    inventory, inventory_raw = _fixture(tmp_path)
    encoded = snapshot.build_score_snapshot(inventory_raw)
    first, second = snapshot.create_sealed_score_snapshot_fds(
        encoded,
        expected_inventory_sha256=str(inventory["inventory_sha256"]),
    )
    try:
        first_stat = os.fstat(first)
        second_stat = os.fstat(second)
        assert (first_stat.st_dev, first_stat.st_ino) == (
            second_stat.st_dev,
            second_stat.st_ino,
        )
        assert first_stat.st_nlink == 0
        assert stat.S_IMODE(first_stat.st_mode) == 0o400
        assert fcntl.fcntl(first, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert fcntl.fcntl(second, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        expected_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        assert fcntl.fcntl(first, fcntl.F_GET_SEALS) == expected_seals
        assert fcntl.fcntl(second, fcntl.F_GET_SEALS) == expected_seals
        assert os.get_inheritable(first)
        assert os.get_inheritable(second)

        assert os.lseek(first, 0, os.SEEK_CUR) == 0
        assert os.lseek(second, 0, os.SEEK_CUR) == 0
        assert os.read(first, 13) == encoded[:13]
        assert os.lseek(first, 0, os.SEEK_CUR) == 13
        assert os.lseek(second, 0, os.SEEK_CUR) == 0
        assert os.read(second, 7) == encoded[:7]
        assert os.lseek(first, 0, os.SEEK_CUR) == 13

        for descriptor in (first, second):
            assert (
                snapshot.validate_sealed_score_snapshot_fd(
                    descriptor,
                    expected_snapshot_sha256=_sha256(encoded),
                    expected_inventory_sha256=str(inventory["inventory_sha256"]),
                )
                == encoded
            )
        writable = os.open(f"/proc/self/fd/{first}", os.O_RDWR)
        try:
            with pytest.raises(OSError):
                os.write(writable, b"x")
            with pytest.raises(OSError):
                os.ftruncate(writable, 0)
            with pytest.raises(OSError):
                os.ftruncate(writable, len(encoded) + 1)
        finally:
            os.close(writable)
    finally:
        os.close(first)
        os.close(second)


def test_non_linux_memfd_request_fails_closed(tmp_path: Path) -> None:
    if _linux_memfd_available():
        pytest.skip("Linux memfd sealing is available")
    inventory, inventory_raw = _fixture(tmp_path)
    encoded = snapshot.build_score_snapshot(inventory_raw)
    with pytest.raises(snapshot.ScoreSnapshotError, match="sealing is unavailable"):
        snapshot.create_sealed_score_snapshot_fds(
            encoded,
            expected_inventory_sha256=str(inventory["inventory_sha256"]),
        )
