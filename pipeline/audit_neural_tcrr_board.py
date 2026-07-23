"""Audit the bounded source-deleted N-TCRR local-transition board."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
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
    return {relative: _sha256_path(root / relative) for relative in SOURCE_PATHS}


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
        == {item.packet_sha256 for item in value.oracle_agreements}
        == {item.packet_sha256 for item in value.primitive_coverage}
    )


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


def _controlled_no_redex_count(value: board.LocalTransitionSlice) -> int:
    expected = {item.packet_sha256: item for item in value.expected_records}
    return sum(
        item.kind
        in {
            "repeated_variable_equality",
            "partial_nested_match",
            "type_mismatch",
            "capacity",
        }
        and bool(expected[item.left_packet_sha256].transitions)
        and not expected[item.right_packet_sha256].transitions
        for item in value.twins
    )


def _oracle_receipt(value: board.LocalTransitionSlice) -> dict[str, object]:
    return {
        "packet_agreement": sum(
            item.exact_agreement for item in value.oracle_agreements
        ),
        "packet_count": len(value.oracle_agreements),
        "state_count": sum(item.state_count for item in value.oracle_agreements),
        "transition_count": sum(
            item.transition_count for item in value.oracle_agreements
        ),
        "normal_form_count": sum(
            item.normal_form_count for item in value.oracle_agreements
        ),
        "cyclic_component_count": sum(
            item.cyclic_component_count for item in value.oracle_agreements
        ),
    }


def _export_custody_receipt(
    value: board.LocalTransitionSlice,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="neural-tcrr-audit-") as temporary:
        root = Path(temporary)
        packet_root = root / "packets"
        train_label_root = root / "train-labels"
        development_assessment_root = root / "development-assessor"
        receipt = board.export_packet_only_corpus(
            value,
            packet_root=packet_root,
            train_label_root=train_label_root,
            development_assessment_root=development_assessment_root,
        )
        train_packets = board.load_packet_only_partition(packet_root, "train")
        development_packets = board.load_packet_only_partition(
            packet_root,
            "development",
        )
        assessor = board.load_sealed_development_assessment(
            development_assessment_root / "sealed_development_assessment.json",
            expected_sha256=receipt.sealed_development_artifact_sha256,
        )
        records = assessor.get("records")
        if not isinstance(records, list):
            raise NeuralTcrrBoardAuditError(
                "sealed development assessor records are malformed"
            )
        return {
            "packet_manifest_sha256": receipt.packet_manifest_sha256,
            "train_label_manifest_sha256": (receipt.train_label_manifest_sha256),
            "development_assessment_manifest_sha256": (
                receipt.development_assessment_manifest_sha256
            ),
            "sealed_development_artifact_sha256": (
                receipt.sealed_development_artifact_sha256
            ),
            "train_packet_count": len(train_packets),
            "development_packet_count": len(development_packets),
            "sealed_development_record_count": len(records),
        }


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
    packet_sha256 = tuple(board.packet_sha256(packet) for packet in value.packets)
    transition_count = sum(len(item.transitions) for item in value.expected_records)
    ledger_aligned = _ledger_alignment(value)
    packet_custody = _model_packets_exclude_offline_ledger(value)
    split_counts = _split_counts(value)
    twin_kinds = tuple(sorted(item.kind for item in value.twins))
    required_twins = (
        "capacity",
        "constructor_reindex",
        "partial_nested_match",
        "repeated_variable_equality",
        "rhs_pointer",
        "rule_reindex",
        "shared_occurrence",
        "storage_reindex",
        "type_mismatch",
        "type_reindex",
    )
    no_redex_count = _controlled_no_redex_count(value)
    rule_pair_count = sum(
        len(item.normalized_rule_pairs) for item in value.fingerprints
    )
    composition_count = sum(
        len(item.reachable_two_rule_compositions) for item in value.fingerprints
    )
    oracle = _oracle_receipt(value)
    custody = _export_custody_receipt(value)
    admitted = (
        len(value.packets) == 22
        and transition_count == 24
        and split_counts
        == {
            "local_transition_development": 6,
            "local_transition_train": 16,
        }
        and ledger_aligned
        and packet_custody
        and twin_kinds == required_twins
        and no_redex_count == 4
        and rule_pair_count == 51
        and composition_count == 8
        and oracle["packet_agreement"] == oracle["packet_count"] == 22
        and custody["train_packet_count"] == 16
        and custody["development_packet_count"] == 6
        and custody["sealed_development_record_count"] == 6
        and _max_path_depth(value.expected_records) <= board.MAX_PATH_DEPTH
        and max(len(packet.graph.reservoir) for packet in value.packets)
        <= board.MAX_CAPACITY
    )
    return {
        "protocol": "neural_tcrr_local_transition_board_audit_v2",
        "decision": ("admit_source_deleted_local_board_only" if admitted else "reject"),
        "claim_boundary": (
            "This audit admits a deterministic, source-deleted local-transition "
            "board with exact-axis causal twins, separated offline labels, "
            "independent state-graph agreement, and split-leakage fingerprints. "
            "It does not admit a neural matcher, autonomous graph rewriting, "
            "Shohin integration, or general reasoning."
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
        "controlled_no_redex_count": no_redex_count,
        "rule_pair_fingerprint_count": rule_pair_count,
        "reachable_two_rule_composition_count": composition_count,
        "ledger_alignment": ledger_aligned,
        "model_packets_exclude_offline_ledger": packet_custody,
        "exact_axis_twins_recomputed": True,
        "max_occurrence_path_depth": _max_path_depth(value.expected_records),
        "max_capacity": max(len(packet.graph.reservoir) for packet in value.packets),
        "independent_oracle": oracle,
        "export_custody": custody,
        "independent_successor_oracle": True,
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
