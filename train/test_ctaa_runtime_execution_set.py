from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import ctaa_runtime_execution_set as execution_set


SEEDS = (101, 202, 303, 404, 505)


def _digest(label: object) -> str:
    return hashlib.sha256(repr(label).encode("ascii")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _publish(path: Path, value: object | bytes) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


def _replace(path: Path, value: object) -> None:
    path.chmod(0o600)
    path.unlink()
    _publish(path, value)


def _rehash(value: dict[str, object]) -> None:
    rows = value.get("entries")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "member_sha256" in row:
                row["member_sha256"] = execution_set._canonical_hash(
                    {key: item for key, item in row.items() if key != "member_sha256"}
                )
    value["execution_set_sha256"] = execution_set._canonical_hash(
        {key: item for key, item in value.items() if key != "execution_set_sha256"}
    )


class SyntheticCustody:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.receipt_calls: list[int] = []
        self.evidence_calls: list[int] = []
        self.finalizer_calls: list[int] = []
        self.wrong_plan_seed: int | None = None
        self.unbound_seeds: set[int] = set()
        self.mismatched_evidence_seed: int | None = None
        self.run_contract = {
            "partition": "development",
            "run_contract_sha256": _digest("run-contract"),
        }
        self.plans: dict[int, SimpleNamespace] = {}
        self.evidence: dict[int, dict[str, object]] = {}
        self.plan_seed_by_name: dict[str, int] = {}
        self.evidence_seed_by_name: dict[str, int] = {}
        self.execution_seed_by_name: dict[str, int] = {}
        self.projection_file_sha_by_seed: dict[int, str] = {}
        self.aggregate_file_sha_by_seed: dict[int, str] = {}
        bundle_entries: list[dict[str, object]] = []
        self.sources: list[execution_set.RuntimeExecutionSetSource] = []

        for seed in SEEDS:
            plan_name = f"plan-{seed}.json"
            evidence_name = f"evidence-{seed}.json"
            projection_name = f"projection-{seed}.json"
            aggregate_name = f"aggregate-{seed}.json"
            receipt_name = f"receipt-{seed}.json"
            artifact_name = f"objects-{seed}"
            plan_file_sha = _publish(root / plan_name, {"seed": seed, "kind": "plan"})
            evidence = {
                "training_seed": seed,
                "evidence_sha256": _digest((seed, "evidence")),
                "attempts": [
                    {
                        "attempt_index": 0,
                        "custody_receipts": {
                            "execution_receipt_sha256": _digest((seed, "receipt"))
                        },
                    }
                ],
            }
            evidence_file_sha = _publish(root / evidence_name, evidence)
            self.projection_file_sha_by_seed[seed] = _publish(
                root / projection_name, {"seed": seed, "kind": "projection"}
            )
            self.aggregate_file_sha_by_seed[seed] = _publish(
                root / aggregate_name, {"seed": seed, "kind": "aggregate"}
            )
            _publish(root / receipt_name, {"seed": seed, "kind": "receipt"})
            (root / artifact_name).mkdir()
            plan = SimpleNamespace(
                plan_sha256=_digest((seed, "plan")),
                bindings=SimpleNamespace(training_seed=seed),
            )
            self.plans[seed] = plan
            self.evidence[seed] = evidence
            self.plan_seed_by_name[plan_name] = seed
            self.evidence_seed_by_name[evidence_name] = seed
            for name in (projection_name, aggregate_name, receipt_name, artifact_name):
                self.execution_seed_by_name[name] = seed
            bundle_entries.append(
                {
                    "training_seed": seed,
                    "runtime_plan_filename": plan_name,
                    "runtime_plan_file_sha256": plan_file_sha,
                    "runtime_plan_sha256": plan.plan_sha256,
                    "runtime_evidence_filename": evidence_name,
                    "runtime_evidence_file_sha256": evidence_file_sha,
                    "runtime_evidence_sha256": evidence["evidence_sha256"],
                }
            )
            self.sources.append(
                execution_set.RuntimeExecutionSetSource(
                    training_seed=seed,
                    execution_projection_filename=projection_name,
                    execution_aggregate_filename=aggregate_name,
                    execution_artifact_directory=artifact_name,
                    execution_receipt_filename=receipt_name,
                )
            )

        self.bundle = {
            "schema": "synthetic-runtime-bundle",
            "partition": "development",
            "run_contract_sha256": self.run_contract["run_contract_sha256"],
            "entries": bundle_entries,
        }
        self.bundle_path = root / "runtime-bundle.json"
        _publish(self.bundle_path, self.bundle)

        def validate_bundle(
            value: object, *, run_contract: object
        ) -> dict[str, object]:
            assert run_contract is self.run_contract
            assert isinstance(value, dict)
            return dict(value)

        def read_plan(path: Path) -> tuple[SimpleNamespace, str]:
            seed = self.plan_seed_by_name[path.name]
            raw = execution_set._read_immutable_bytes(path, "test plan", 4096)
            return self.plans[seed], hashlib.sha256(raw).hexdigest()

        def read_evidence(
            path: Path,
            plan: SimpleNamespace,
            *,
            expected_file_sha256: str | None = None,
        ) -> dict[str, object]:
            seed = self.evidence_seed_by_name[path.name]
            self.evidence_calls.append(seed)
            assert plan.bindings.training_seed == seed
            raw = execution_set._read_immutable_bytes(path, "test evidence", 4096)
            assert hashlib.sha256(raw).hexdigest() == expected_file_sha256
            return deepcopy(self.evidence[seed])

        def read_receipt_envelope(
            path: Path,
            *,
            verification_key: object,
        ) -> tuple[dict[str, object], str]:
            seed = self.execution_seed_by_name[path.name]
            self.receipt_calls.append(seed)
            assert verification_key == b"v" * 32
            bound_seed = SEEDS[0] if seed in self.unbound_seeds else seed
            plan_sha = (
                _digest((seed, "wrong-plan"))
                if seed == self.wrong_plan_seed
                else self.plans[seed].plan_sha256
            )
            payload = {
                "training_seed": bound_seed,
                "plan_sha256": plan_sha,
                "run_contract_sha256": self.run_contract["run_contract_sha256"],
                "partition": "development",
                "execution_projection_file_sha256": self.projection_file_sha_by_seed[
                    seed
                ],
                "execution_projection_sha256": _digest((seed, "projection-logical")),
                "execution_aggregate_sha256": self.aggregate_file_sha_by_seed[seed],
                "execution_sha256": _digest((seed, "execution")),
            }
            receipt = {
                "payload": payload,
                "receipt_sha256": _digest((seed, "receipt")),
            }
            raw = execution_set._read_immutable_bytes(path, "test receipt", 4096)
            return receipt, hashlib.sha256(raw).hexdigest()

        def validate_receipt(
            value: dict[str, object],
            *,
            execution_projection_path: Path,
            plan: SimpleNamespace,
            execution_aggregate_path: Path,
            execution_artifact_directory: Path,
            execution_aggregate_sha256: str,
            verification_key: object,
        ) -> dict[str, object]:
            seed = plan.bindings.training_seed
            assert self.execution_seed_by_name[execution_projection_path.name] == seed
            assert self.execution_seed_by_name[execution_aggregate_path.name] == seed
            assert (
                self.execution_seed_by_name[execution_artifact_directory.name] == seed
            )
            assert execution_aggregate_sha256 == self.aggregate_file_sha_by_seed[seed]
            assert verification_key == b"v" * 32
            return deepcopy(value)

        def finalize(*, plan: SimpleNamespace, **kwargs: object) -> dict[str, object]:
            seed = plan.bindings.training_seed
            self.finalizer_calls.append(seed)
            assert kwargs["receipt_verification_key"] == b"v" * 32
            result = deepcopy(self.evidence[seed])
            if seed == self.mismatched_evidence_seed:
                result["evidence_sha256"] = _digest((seed, "mismatch"))
            return result

        monkeypatch.setattr(execution_set, "validate_runtime_bundle", validate_bundle)
        monkeypatch.setattr(execution_set, "read_runtime_plan", read_plan)
        monkeypatch.setattr(execution_set, "read_runtime_evidence", read_evidence)
        monkeypatch.setattr(
            execution_set,
            "read_runtime_execution_receipt_envelope_with_sha",
            read_receipt_envelope,
        )
        monkeypatch.setattr(
            execution_set, "validate_runtime_execution_receipt", validate_receipt
        )
        monkeypatch.setattr(execution_set, "make_finalized_runtime_evidence", finalize)

        self.set_path = root / "execution-set.json"
        execution_set.write_runtime_execution_set(
            self.set_path,
            runtime_bundle_path=self.bundle_path,
            run_contract=self.run_contract,
            members=self.sources,
            verification_key=b"v" * 32,
        )
        self.value = json.loads(self.set_path.read_text())
        self.receipt_calls.clear()
        self.evidence_calls.clear()
        self.finalizer_calls.clear()

    def read(self) -> tuple[dict[str, object], str]:
        return execution_set.read_runtime_execution_set_with_replay(
            self.set_path,
            runtime_bundle_path=self.bundle_path,
            run_contract=self.run_contract,
            verification_key=b"v" * 32,
        )

    def install(self, value: dict[str, object]) -> None:
        _replace(self.set_path, value)


@pytest.fixture
def custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticCustody:
    return SyntheticCustody(tmp_path, monkeypatch)


def test_valid_five_member_set_replays_every_receipt_and_finalizer(
    custody: SyntheticCustody,
) -> None:
    value, file_sha = custody.read()
    assert value["schema"] == execution_set.EXECUTION_SET_SCHEMA
    assert value["seed_count"] == 5
    assert value["execution_set_sha256"] == execution_set._canonical_hash(
        {key: item for key, item in value.items() if key != "execution_set_sha256"}
    )
    assert [row["training_seed"] for row in value["entries"]] == list(SEEDS)
    assert custody.receipt_calls == list(SEEDS)
    assert custody.evidence_calls == list(SEEDS)
    assert custody.finalizer_calls == list(SEEDS)
    assert file_sha == hashlib.sha256(custody.set_path.read_bytes()).hexdigest()
    assert set(value) == execution_set._SET_KEYS
    assert all(set(row) == execution_set._MEMBER_KEYS for row in value["entries"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_missing_duplicate_or_reordered_member_fails_closed(
    custody: SyntheticCustody, mutation: str
) -> None:
    changed = deepcopy(custody.value)
    rows = changed["entries"]
    assert isinstance(rows, list)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[0], rows[1] = rows[1], rows[0]
    _rehash(changed)
    custody.install(changed)
    with pytest.raises(execution_set.RuntimeExecutionSetError):
        custody.read()
    assert custody.receipt_calls == []
    assert custody.finalizer_calls == []


def test_cross_seed_swapped_member_fails_closed(custody: SyntheticCustody) -> None:
    changed = deepcopy(custody.value)
    first, second = changed["entries"][:2]
    for key in execution_set._MEMBER_PATH_KEYS:
        first[key], second[key] = second[key], first[key]
    _rehash(changed)
    custody.install(changed)
    with pytest.raises(
        execution_set.RuntimeExecutionSetError, match="signed receipt binding differs"
    ):
        custody.read()


def test_receipt_for_wrong_plan_fails_closed(custody: SyntheticCustody) -> None:
    custody.wrong_plan_seed = SEEDS[2]
    with pytest.raises(
        execution_set.RuntimeExecutionSetError, match="signed receipt binding differs"
    ):
        custody.read()
    assert custody.receipt_calls == list(SEEDS[:3])
    assert custody.evidence_calls == []
    assert custody.finalizer_calls == []


def test_four_unbound_members_plus_one_bound_member_fails_closed(
    custody: SyntheticCustody,
) -> None:
    custody.unbound_seeds = set(SEEDS[1:])
    with pytest.raises(
        execution_set.RuntimeExecutionSetError, match="signed receipt binding differs"
    ):
        custody.read()
    assert custody.receipt_calls == list(SEEDS[:2])
    assert custody.evidence_calls == []
    assert custody.finalizer_calls == []


def test_invalid_fifth_receipt_blocks_all_query_aware_evidence(
    custody: SyntheticCustody,
) -> None:
    custody.wrong_plan_seed = SEEDS[-1]
    with pytest.raises(
        execution_set.RuntimeExecutionSetError, match="signed receipt binding differs"
    ):
        custody.read()
    assert custody.receipt_calls == list(SEEDS)
    assert custody.evidence_calls == []
    assert custody.finalizer_calls == []


def test_finalized_evidence_must_exactly_equal_bundle_evidence(
    custody: SyntheticCustody,
) -> None:
    custody.mismatched_evidence_seed = SEEDS[-1]
    with pytest.raises(
        execution_set.RuntimeExecutionSetError,
        match="finalized evidence differs from bundle evidence",
    ):
        custody.read()
    assert custody.receipt_calls == list(SEEDS)
    assert custody.finalizer_calls == list(SEEDS)


def test_unsafe_member_component_fails_before_receipt(
    custody: SyntheticCustody,
) -> None:
    changed = deepcopy(custody.value)
    changed["entries"][0]["execution_projection_filename"] = "../projection.json"
    _rehash(changed)
    custody.install(changed)
    with pytest.raises(execution_set.RuntimeExecutionSetError, match="unsafe"):
        custody.read()
    assert custody.receipt_calls == []


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlink_member_fails_closed(custody: SyntheticCustody, kind: str) -> None:
    changed = deepcopy(custody.value)
    if kind == "file":
        link_name = "projection-link.json"
        os.symlink("projection-101.json", custody.root / link_name)
        changed["entries"][0]["execution_projection_filename"] = link_name
    else:
        link_name = "objects-link"
        os.symlink("objects-101", custody.root / link_name)
        changed["entries"][0]["execution_artifact_directory"] = link_name
    _rehash(changed)
    custody.install(changed)
    with pytest.raises(execution_set.RuntimeExecutionSetError, match="symlink"):
        custody.read()


@pytest.mark.parametrize("level", ["member", "set"])
def test_canonical_hash_tamper_fails_before_any_replay(
    custody: SyntheticCustody, level: str
) -> None:
    changed = deepcopy(custody.value)
    if level == "member":
        changed["entries"][0]["member_sha256"] = _digest("tampered-member")
        changed["execution_set_sha256"] = execution_set._canonical_hash(
            {
                key: item
                for key, item in changed.items()
                if key != "execution_set_sha256"
            }
        )
    else:
        changed["execution_set_sha256"] = _digest("tampered-set")
    custody.install(changed)
    with pytest.raises(execution_set.RuntimeExecutionSetError, match="commitment"):
        custody.read()
    assert custody.receipt_calls == []
    assert custody.finalizer_calls == []


def test_runtime_bundle_file_hash_is_exact_and_held(custody: SyntheticCustody) -> None:
    changed_bundle = deepcopy(custody.bundle)
    changed_bundle["uncommitted_note"] = "replacement"
    _replace(custody.bundle_path, changed_bundle)
    with pytest.raises(
        execution_set.RuntimeExecutionSetError,
        match="top-level custody binding differs",
    ):
        custody.read()
    assert custody.receipt_calls == []
