"""Audit the bounded source-deleted N-TCRR local-transition board."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import neural_tcrr_board as board


SOURCE_PATHS = (
    "R12_NEURAL_TCRR_PREREG.md",
    "pipeline/neural_tcrr_board.py",
    "pipeline/test_neural_tcrr_board.py",
    "pipeline/audit_neural_tcrr_board.py",
    "pipeline/test_audit_neural_tcrr_board.py",
    "pipeline/typed_critical_pair_rewrite_board.py",
    "pipeline/test_typed_critical_pair_rewrite_board.py",
)
DEFAULT_SEED = 2026072301


class NeuralTcrrBoardAuditError(RuntimeError):
    """Raised when the audited board or source custody fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _git(
    root: Path,
    *arguments: str,
) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_receipt(root: Path) -> dict[str, str]:
    missing = [relative for relative in SOURCE_PATHS if not (root / relative).is_file()]
    if missing:
        raise NeuralTcrrBoardAuditError(f"missing audit source files: {missing}")
    dirty = _git(root, "status", "--short", "--", *SOURCE_PATHS)
    if dirty:
        raise NeuralTcrrBoardAuditError(
            f"refusing to audit dirty source files:\n{dirty}"
        )
    return {
        relative: _sha256_path(root / relative)
        for relative in SOURCE_PATHS
    }


def _packet_identifiers(packet: board.SourceDeletedPacket) -> set[str]:
    identifiers = set(packet.graph.reservoir)
    for constructor in packet.constructors:
        identifiers.add(constructor.identifier)
        identifiers.add(constructor.result_type)
        identifiers.update(constructor.argument_types)

    def collect_rule_term(term: board.RuleTermRecord | None) -> None:
        if term is None:
            return
        identifiers.add(term.type_id)
        if term.constructor_id is not None:
            identifiers.add(term.constructor_id)
        if term.variable_id is not None:
            identifiers.add(term.variable_id)
        for child in term.children:
            collect_rule_term(child)

    for rule in packet.rules:
        identifiers.add(rule.identifier)
        collect_rule_term(rule.lhs)
        collect_rule_term(rule.rhs)
    for node in packet.graph.nodes:
        identifiers.add(node.storage_id)
        identifiers.add(node.type_id)
        if node.constructor_id is not None:
            identifiers.add(node.constructor_id)
        if node.variable_id is not None:
            identifiers.add(node.variable_id)
    return identifiers


def _max_path_depth(
    expected_records: tuple[board.ExpectedTransitionRecord, ...],
) -> int:
    return max(
        (
            len(transition.occurrence_path)
            for record in expected_records
            for transition in record.transitions
        ),
        default=0,
    )


def _ledger_alignment(value: board.LocalTransitionSlice) -> bool:
    packet_digests = {board.packet_sha256(packet) for packet in value.packets}
    return (
        packet_digests
        == {item.packet_sha256 for item in value.expected_records}
        == {item.packet_sha256 for item in value.split_assignments}
        == {item.packet_sha256 for item in value.fingerprints}
    )


def _identity_namespaces_disjoint(value: board.LocalTransitionSlice) -> bool:
    seen: set[str] = set()
    for packet in value.packets:
        active = _packet_identifiers(packet)
        if seen & active:
            return False
        seen.update(active)
    return True


def _model_packets_exclude_offline_ledger(
    value: board.LocalTransitionSlice,
) -> bool:
    forbidden_strings = (
        "occurrence_path",
        "successor",
        "packet_sha256",
        "partition",
        "fingerprint",
        "twin",
    )
    for packet in value.packets:
        serialized = board.serialize_model_packet(packet)
        if any(item in serialized for item in forbidden_strings):
            return False
    return True


def _split_counts(
    value: board.LocalTransitionSlice,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split_assignments:
        result[item.partition] = result.get(item.partition, 0) + 1
    return dict(sorted(result.items()))


def build_audit_report(
    value: board.LocalTransitionSlice,
    *,
    seed: int,
    source_commit: str,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    """Build a fail-closed audit report from one frozen board value."""

    board.validate_local_transition_slice(value)
    serialized = dataclasses.asdict(value)
    board_sha256 = _sha256_bytes(_canonical_bytes(serialized))
    packet_sha256 = tuple(
        board.packet_sha256(packet) for packet in value.packets
    )
    transition_count = sum(
        len(item.transitions) for item in value.expected_records
    )
    identity_disjoint = _identity_namespaces_disjoint(value)
    ledger_aligned = _ledger_alignment(value)
    packet_custody = _model_packets_exclude_offline_ledger(value)
    split_counts = _split_counts(value)
    twin_kinds = tuple(sorted(item.kind for item in value.twins))
    required_twins = (
        "capacity",
        "rhs_pointer",
        "rule_reindex",
        "shared_occurrence",
        "storage_reindex",
    )
    admitted = (
        len(value.packets) == 21
        and transition_count == 25
        and split_counts
        == {
            "local_transition_development": 6,
            "local_transition_train": 15,
        }
        and identity_disjoint
        and ledger_aligned
        and packet_custody
        and twin_kinds == required_twins
        and _max_path_depth(value.expected_records) <= board.MAX_PATH_DEPTH
        and max(len(packet.graph.reservoir) for packet in value.packets)
        <= board.MAX_CAPACITY
    )
    return {
        "protocol": "neural_tcrr_local_transition_board_audit_v1",
        "decision": (
            "admit_source_deleted_local_board_only"
            if admitted
            else "reject"
        ),
        "claim_boundary": (
            "This audit admits a deterministic, source-deleted local-transition "
            "board with separated offline labels and leakage fingerprints. It "
            "does not admit an independent successor oracle, a neural matcher, "
            "autonomous graph rewriting, Shohin integration, or general reasoning."
        ),
        "seed": seed,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "board_sha256": board_sha256,
        "packet_count": len(value.packets),
        "packet_sha256": packet_sha256,
        "split_counts": split_counts,
        "transition_count": transition_count,
        "twin_kinds": twin_kinds,
        "identity_namespaces_disjoint": identity_disjoint,
        "ledger_alignment": ledger_aligned,
        "model_packets_exclude_offline_ledger": packet_custody,
        "max_occurrence_path_depth": _max_path_depth(value.expected_records),
        "max_capacity": max(
            len(packet.graph.reservoir) for packet in value.packets
        ),
        "independent_successor_oracle": False,
        "neural_runtime_present": False,
    }


def audit(
    *,
    root: Path,
    output: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Audit frozen source and write one deterministic JSON receipt."""

    source_commit = _git(root, "rev-parse", "HEAD")
    source_sha256 = _source_receipt(root)
    value = board.build_local_transition_slice(seed)
    report = build_audit_report(
        value,
        seed=seed,
        source_commit=source_commit,
        source_sha256=source_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    report = audit(
        root=arguments.root.resolve(),
        output=arguments.output,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "board_sha256": report["board_sha256"],
                "packet_count": report["packet_count"],
                "transition_count": report["transition_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
