#!/usr/bin/env python3
"""Atomic CPU qualification audit for the EFC identifiable-board pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    FAMILIES,
    GrammarFactors,
    LateQuery,
    canonical_action_bytes,
    decode_query,
    decode_source,
    encode_query,
    execute_query,
    generate_pilot_rows,
    project_candidate_sources,
    resource_receipt,
    solve_unique_completion,
)
from pipeline.episode_functor_identifiable_reference import (  # noqa: E402
    enumerate_consistent_machines,
    version_space_receipt,
)
from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    collate_qualification_supervision,
)


DEFAULT_COUNTS = {
    "confirmation": 24,
    "development": 32,
    "mechanics": 48,
    "train": 96,
}


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _machine_tuple(machine) -> tuple[object, ...]:
    return (
        machine.state_keys,
        machine.action_keys,
        machine.observer_keys,
        machine.transitions,
        machine.observations,
    )


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return sha256(payload).hexdigest()


def _action_overlap_receipt(rows) -> dict[str, object]:
    action_orbits: dict[str, set[str]] = {
        family: set() for family in FAMILIES
    }
    for row in rows:
        action_orbits[row.family].add(
            sha256(canonical_action_bytes(row.machine.transitions)).hexdigest()
        )
    missing = tuple(
        family for family, values in action_orbits.items() if not values
    )
    if missing:
        raise ValueError(
            f"action-only overlap audit omits families: {', '.join(missing)}"
        )

    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                action_orbits[left] & action_orbits[right]
            )
    if any(overlaps.values()):
        raise ValueError("action-only canonical orbit crosses families")
    return {
        "canonical_orbit_count_by_family": {
            family: len(action_orbits[family]) for family in FAMILIES
        },
        "cross_family_overlap_counts": overlaps,
        "family_pair_count": len(overlaps),
    }


def _candidate_projection_receipt(
    rows,
    *,
    splits: tuple[str, ...],
) -> dict[str, object]:
    expected_fields = ("source",)
    forbidden = ("target", "family", "split", "factors", "machine")
    counts: dict[str, int] = {}
    payload_receipts: dict[str, str] = {}
    for split in splits:
        projected = project_candidate_sources(rows, split=split)
        expected_sources = tuple(
            row.source for row in rows if row.split == split
        )
        if len(projected) != len(expected_sources):
            raise ValueError("candidate projection row count differs")
        for candidate, expected_source in zip(
            projected,
            expected_sources,
            strict=True,
        ):
            try:
                field_names = tuple(field.name for field in fields(candidate))
            except TypeError as exc:
                raise ValueError(
                    "candidate projection is not a frozen data object"
                ) from exc
            if field_names != expected_fields:
                raise ValueError(
                    "candidate projection exposes fields other than source"
                )
            if type(candidate.source) is not bytes:
                raise ValueError("candidate projection source is not bytes")
            if candidate.source != expected_source:
                raise ValueError("candidate projection changes source bytes")
            if any(hasattr(candidate, name) for name in forbidden):
                raise ValueError(
                    "candidate projection exposes forbidden metadata"
                )
        counts[split] = len(projected)
        payload_receipts[split] = _canonical_json_sha256(
            [sha256(candidate.source).hexdigest() for candidate in projected]
        )
    return {
        "counts_by_split": counts,
        "fields": list(expected_fields),
        "forbidden_attribute_hits": 0,
        "payload_manifest_sha256_by_split": payload_receipts,
    }


def _qualification_supervisor_receipt(rows) -> dict[str, object]:
    supervisor = collate_qualification_supervision(rows)
    expected_hashes = tuple(
        sha256(row.source).hexdigest() for row in rows
    )
    supervisor.assert_candidate_alignment(expected_hashes)
    manifest: list[dict[str, object]] = []
    for row_index, source_hash in enumerate(supervisor.source_sha256):
        record_valid = supervisor.record_label_valid[row_index]
        occurrence_valid = supervisor.occurrence_label_valid[row_index]
        answer_valid = supervisor.answer_label_valid[row_index]
        manifest.append(
            {
                "answer_labels": supervisor.record_answer[
                    row_index,
                    answer_valid,
                ].tolist(),
                "key_slot_to_unique": supervisor.key_slot_to_unique[
                    row_index
                ].tolist(),
                "observer_answer": supervisor.observer_answer[
                    row_index
                ].tolist(),
                "observer_exposed": supervisor.observer_exposed[
                    row_index
                ].tolist(),
                "occurrence_roles": supervisor.occurrence_role[
                    row_index,
                    occurrence_valid,
                ].tolist(),
                "record_types": supervisor.record_type[
                    row_index,
                    record_valid,
                ].tolist(),
                "source_sha256": source_hash,
                "transition_next": supervisor.transition_next[
                    row_index
                ].tolist(),
                "transition_exposed": supervisor.transition_exposed[
                    row_index
                ].tolist(),
            }
        )
    return {
        "answer_label_count": int(
            supervisor.answer_label_valid.sum().item()
        ),
        "candidate_join_key": "source_sha256",
        "label_manifest_sha256": _canonical_json_sha256(manifest),
        "occurrence_label_count": int(
            supervisor.occurrence_label_valid.sum().item()
        ),
        "record_label_count": int(
            supervisor.record_label_valid.sum().item()
        ),
        "row_count": supervisor.batch_size,
        "supervisor_fields_exposed_to_candidate": 0,
    }


def _trunk_tokenizer_receipt(rows) -> dict[str, object]:
    from tokenizers import Tokenizer

    tokenizer_path = ROOT / "artifacts" / "shohin-tok-32k.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    counts: list[int] = []
    exact_coverage = 0
    for row in rows:
        encoded = tokenizer.encode(row.source.decode("ascii"))
        coverage = [0] * len(row.source)
        for start, end in encoded.offsets:
            if not 0 <= start < end <= len(row.source):
                raise ValueError("tokenizer offset leaves source payload")
            for index in range(start, end):
                coverage[index] += 1
        if any(value != 1 for value in coverage):
            raise ValueError(
                "tokenizer offsets do not exactly partition source bytes"
            )
        exact_coverage += 1
        counts.append(len(encoded.ids))
    return {
        "disconnected_window_semantics": True,
        "exact_byte_coverage_rows": exact_coverage,
        "maximum_tokens": max(counts),
        "minimum_tokens": min(counts),
        "parent_context_tokens": 2_048,
        "rows_exceeding_parent_context": sum(
            count > 2_048 for count in counts
        ),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
    }


def _row_structural_receipt(rows) -> dict[str, object]:
    manifest: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        entry: dict[str, object] = {
            "action_only_sha256": sha256(
                canonical_action_bytes(row.machine.transitions)
            ).hexdigest(),
            "factor_values": list(row.factors.values),
            "family": row.family,
            "row_index": row_index,
            "source_bytes": len(row.source),
            "source_sha256": sha256(row.source).hexdigest(),
            "split": row.split,
            "structural_sha256": row.canonical_sha256,
            "world_id": row.world_id,
        }
        entry["row_receipt_sha256"] = _canonical_json_sha256(entry)
        manifest.append(entry)

    by_split: dict[str, list[str]] = defaultdict(list)
    by_world: dict[str, list[str]] = defaultdict(list)
    for entry in manifest:
        receipt = str(entry["row_receipt_sha256"])
        by_split[str(entry["split"])].append(receipt)
        by_world[str(entry["world_id"])].append(receipt)
    return {
        "manifest": manifest,
        "manifest_sha256": _canonical_json_sha256(manifest),
        "source_manifest_sha256": _canonical_json_sha256(
            [
                {
                    "row_index": entry["row_index"],
                    "source_sha256": entry["source_sha256"],
                }
                for entry in manifest
            ]
        ),
        "split_manifest_sha256": {
            split: _canonical_json_sha256(receipts)
            for split, receipts in sorted(by_split.items())
        },
        "world_manifest_sha256": {
            world_id: _canonical_json_sha256(receipts)
            for world_id, receipts in sorted(by_world.items())
        },
    }


def audit_identifiable_pilot(
    *,
    seed: str,
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    frozen_counts = dict(DEFAULT_COUNTS if counts is None else counts)
    rows = generate_pilot_rows(seed=seed, counts=frozen_counts)
    source_hashes: set[str] = set()
    canonical_by_split: dict[str, set[str]] = defaultdict(set)
    family_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    factors_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_bytes_by_split: Counter[str] = Counter()
    direct_reference_exact = 0
    query_renderer_exact = 0
    maximum_version_space = 0
    maximum_behavior_classes = 0
    exhaustive_coordinates = 0
    changed_same_bag_rows = 0

    for row in rows:
        source_hash = sha256(row.source).hexdigest()
        if source_hash in source_hashes:
            raise ValueError("pilot source bytes are duplicated")
        source_hashes.add(source_hash)
        if any(marker in row.source.lower() for marker in (b"renderer", b"family", b"split")):
            raise ValueError("candidate-visible source contains forbidden metadata")
        canonical_by_split[row.split].add(row.canonical_sha256)
        family_by_split[row.split][row.family] += 1
        factors_by_split[row.split][str(row.factors.values)] += 1
        source_bytes_by_split[row.split] += len(row.source)

        direct = solve_unique_completion(decode_source(row.source))
        reference = enumerate_consistent_machines(row.source)
        if len(reference) != 1 or _machine_tuple(direct) != _machine_tuple(reference[0]):
            raise ValueError("production and reference completion differ")
        if direct.canonical_structural_bytes() != row.machine.canonical_structural_bytes():
            raise ValueError("completed machine differs from latent semantics")
        direct_reference_exact += 1
        receipt = version_space_receipt(row.source, max_depth=4)
        maximum_version_space = max(maximum_version_space, receipt["version_space"])
        maximum_behavior_classes = max(
            maximum_behavior_classes,
            receipt["behavior_classes"],
        )
        exhaustive_coordinates += receipt["coordinates"]

        query = LateQuery(
            start_key=row.machine.state_keys[3],
            action_keys=(
                row.machine.action_keys[2],
                row.machine.action_keys[0],
                row.machine.action_keys[2],
                row.machine.action_keys[1],
            ),
            observer_key=row.machine.observer_keys[1],
        )
        expected = execute_query(row.machine, query)
        for values in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ):
            rendered = encode_query(query, GrammarFactors(*values))
            parsed = decode_query(rendered)
            if parsed != query or execute_query(direct, parsed) != expected:
                raise ValueError("query metagrammar changes semantics")
            query_renderer_exact += 1

        changed = False
        for left in range(3):
            for right in range(left + 1, 3):
                for start in range(8):
                    for observer in range(2):
                        first = LateQuery(
                            row.machine.state_keys[start],
                            (
                                row.machine.action_keys[left],
                                row.machine.action_keys[right],
                            ),
                            row.machine.observer_keys[observer],
                        )
                        second = LateQuery(
                            row.machine.state_keys[start],
                            (
                                row.machine.action_keys[right],
                                row.machine.action_keys[left],
                            ),
                            row.machine.observer_keys[observer],
                        )
                        if execute_query(row.machine, first) != execute_query(
                            row.machine,
                            second,
                        ):
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        changed_same_bag_rows += int(changed)

    split_names = tuple(sorted(canonical_by_split))
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                canonical_by_split[left] & canonical_by_split[right]
            )
    if any(overlaps.values()):
        raise ValueError("pilot structural orbit crosses splits")
    if maximum_version_space != 1 or maximum_behavior_classes != 1:
        raise ValueError("pilot source is not uniquely identifying")
    if changed_same_bag_rows != len(rows):
        raise ValueError("pilot contains a world without a changed order twin")

    action_overlap_receipt = _action_overlap_receipt(rows)
    candidate_projection_receipt = _candidate_projection_receipt(
        rows,
        splits=split_names,
    )
    qualification_supervisor_receipt = (
        _qualification_supervisor_receipt(rows)
    )
    trunk_tokenizer_receipt = _trunk_tokenizer_receipt(rows)
    row_structural_receipt = _row_structural_receipt(rows)

    source_files = (
        ROOT / "pipeline" / "audit_episode_functor_identifiable_board.py",
        ROOT / "pipeline" / "episode_functor_identifiable_board.py",
        ROOT / "pipeline" / "episode_functor_qualification_boundary.py",
        ROOT
        / "pipeline"
        / "episode_functor_qualification_supervisor.py",
        ROOT / "pipeline" / "episode_functor_resource_receipt.py",
        ROOT / "pipeline" / "episode_functor_identifiable_reference.py",
        ROOT / "train" / "episode_functor_constrained_transport.py",
        ROOT / "train" / "episode_functor_capacity_lanes.py",
        ROOT / "train" / "episode_functor_learned_completion.py",
        ROOT / "train" / "episode_functor_learned_system.py",
        ROOT / "train" / "episode_functor_machine.py",
        ROOT / "train" / "model.py",
        ROOT / "train" / "episode_functor_pointer_compiler.py",
        ROOT / "train" / "episode_functor_query_parser.py",
        ROOT / "train" / "episode_functor_qualification_loss.py",
        ROOT / "train" / "episode_functor_qualification_trainer.py",
        ROOT / "train" / "episode_functor_shohin_trunk.py",
        ROOT / "train" / "episode_functor_witness_compiler.py",
        ROOT / "artifacts" / "shohin-tok-32k.json",
        ROOT / "train" / "flagship_out" / "ckpt_0300000.pt",
        ROOT / "R12_EPISODE_FUNCTOR_COMPILER_THEORY_DRAFT.md",
        ROOT / "R12_EFC_LEARNED_COMPILER_PREREG.md",
    )
    report: dict[str, object] = {
        "claim_boundary": (
            "CPU mechanics and identifiability pilot only; no neural learning "
            "or reasoning claim"
        ),
        "action_only_family_audit": action_overlap_receipt,
        "candidate_projection": candidate_projection_receipt,
        "counts": dict(sorted(frozen_counts.items())),
        "decision": "cpu_qualification_candidate_neural_fit_no_go",
        "direct_reference_exact": direct_reference_exact,
        "exhaustive_depth_0_4_coordinates": exhaustive_coordinates,
        "family_by_split": {
            split: dict(sorted(values.items()))
            for split, values in sorted(family_by_split.items())
        },
        "factor_combinations_by_split": {
            split: dict(sorted(values.items()))
            for split, values in sorted(factors_by_split.items())
        },
        "forbidden_candidate_metadata_hits": 0,
        "latent_world_count": sum(frozen_counts.values()),
        "maximum_behavior_classes": maximum_behavior_classes,
        "maximum_version_space": maximum_version_space,
        "query_renderer_exact": query_renderer_exact,
        "qualification_supervisor": qualification_supervisor_receipt,
        "resource_depth_12": resource_receipt(12),
        "row_count": len(rows),
        "row_structural_receipt": row_structural_receipt,
        "same_bag_changed_order_rows": changed_same_bag_rows,
        "schema": "efc-identifiable-cpu-audit-v3",
        "seed": seed,
        "source_bytes_by_split": dict(sorted(source_bytes_by_split.items())),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in source_files
        },
        "split_structural_overlaps": overlaps,
        "trunk_tokenizer": trunk_tokenizer_receipt,
        "unique_canonical_worlds": len(
            set().union(*canonical_by_split.values())
        ),
        "unique_source_payloads": len(source_hashes),
    }
    canonical = (
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    report["report_payload_sha256"] = sha256(canonical).hexdigest()
    return report


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="efc-identifiable-pilot-20260724")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "r12" / "efc_identifiable_pilot_report.json",
    )
    arguments = parser.parse_args()
    report = audit_identifiable_pilot(seed=arguments.seed)
    write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
