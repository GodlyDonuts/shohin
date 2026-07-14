#!/usr/bin/env python3
"""Build a gated source-free event-history curriculum for CRCS.

Each event is an ordinary FQRB source triple compiled to one native residual
state.  A history exposes only a fixed signed CountSketch of those states to a
later suffix.  The suffix supplies a fresh ECLI-style codebook and an event
ordinal/query, never an event source or a precomputed answer.  This script is
CPU-only and solver-derived; it writes no model artifact and requires a
passing ECLI assessment before it can write durable data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path

from generate_ephemeral_codebook_fqrb_v1 import (
    CANONICAL_LABELS,
    codebook_key,
    mapping_for_group,
    render_codebook,
    swapped_mapping,
)
from generate_finite_query_residual_basis_v1 import (
    QUERY_KINDS,
    TWO_DIGIT_VALUES,
    build as build_fqrb,
    ngrams,
)


EVENT_SOURCE_FIELDS = (
    "base_source", "edited_source", "donor_source",
    "paraphrase_base_source", "paraphrase_edited_source", "paraphrase_donor_source",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def group_fqrb_rows(rows: list[dict]) -> list[list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        groups.setdefault(row["basis_id"], []).append(row)
    result = []
    for basis_id, group in groups.items():
        indexed = {row["query_kind"]: row for row in group}
        if len(group) != len(QUERY_KINDS) or set(indexed) != set(QUERY_KINDS):
            raise ValueError("FQRB event group {} is incomplete".format(basis_id))
        first = group[0]
        if any(any(row[field] != first[field] for field in EVENT_SOURCE_FIELDS) for row in group[1:]):
            raise ValueError("FQRB event group {} does not share source state".format(basis_id))
        result.append([indexed[kind] for kind in QUERY_KINDS])
    return result


def event_from_group(group: list[dict]) -> dict:
    first = group[0]
    return {
        "event_id": first["basis_id"],
        **{field: first[field] for field in EVENT_SOURCE_FIELDS},
        "counterfactual_edited_by_query": {
            row["query_kind"]: row["counterfactual_edited_source"] for row in group
        },
        "semantic_by_query": {row["query_kind"]: row["response"] for row in group},
        "counterfactual_semantic_by_query": {
            row["query_kind"]: row["counterfactual_response"] for row in group
        },
        "state": first["state"],
    }


def render_suffix(event_ordinal: int, query_kind: str, mapping: dict[str, str]) -> str:
    return (
        "Binding table:\n{}\n"
        "History query: inspect event ordinal {} and return its {} class.\n"
        "Return exactly the matching code.\nAnswer:"
    ).format(render_codebook(mapping), event_ordinal, query_kind)


def validate_row(row: dict) -> None:
    if row.get("mechanism") != "causal_residual_count_sketch_v1":
        raise ValueError("not a CRCS curriculum row")
    events = row.get("events")
    ordinal, kind = row.get("event_ordinal"), row.get("query_kind")
    if not isinstance(events, list) or not isinstance(ordinal, int) or not 0 <= ordinal < len(events):
        raise ValueError("invalid CRCS event selection")
    if kind not in QUERY_KINDS:
        raise ValueError("invalid CRCS query kind")
    event = events[ordinal]
    semantic = event["semantic_by_query"][kind]
    counter = event["counterfactual_semantic_by_query"][kind]
    if semantic == counter:
        raise ValueError("selected CRCS counterfactual is answer invariant")
    mapping, swap = row["codebook"], row["codebook_swap"]
    codebook_key(mapping)
    codebook_key(swap)
    if row["response"] != "code=" + mapping[semantic]:
        raise ValueError("CRCS normal code is not solver-derived")
    if row["counterfactual_response"] != "code=" + mapping[counter]:
        raise ValueError("CRCS counterfactual code is not solver-derived")
    if row["codebook_swap_response"] != "code=" + swap[semantic]:
        raise ValueError("CRCS codebook intervention is not solver-derived")
    if row["response"] == row["counterfactual_response"] or row["response"] == row["codebook_swap_response"]:
        raise ValueError("CRCS interventions must change the response")
    if any(source in row["suffix_prompt"] for event in events for source in (event["base_source"], event["edited_source"], event["donor_source"])):
        raise ValueError("CRCS suffix leaked an event source")


def build_histories(
    history_count: int,
    event_counts: tuple[int, ...],
    seed: int,
    split: str,
    language_heldout: bool,
    forbidden_codebooks: set[tuple[tuple[str, str], ...]],
) -> tuple[list[dict], set[tuple[tuple[str, str], ...]]]:
    if history_count <= 0 or not event_counts or any(count <= 0 for count in event_counts):
        raise ValueError("history count and event counts must be positive")
    rng = random.Random(seed)
    counts = [event_counts[index % len(event_counts)] for index in range(history_count)]
    fqrb_groups = group_fqrb_rows(build_fqrb(sum(counts), seed, split, TWO_DIGIT_VALUES, 90, language_heldout))
    cursor, rows, used = 0, [], set(forbidden_codebooks)
    for history_index, event_count in enumerate(counts):
        events = [event_from_group(group) for group in fqrb_groups[cursor:cursor + event_count]]
        cursor += event_count
        mapping = mapping_for_group(rng, used)
        used.add(codebook_key(mapping))
        for query_index, kind in enumerate(QUERY_KINDS):
            ordinal = rng.randrange(event_count)
            event = events[ordinal]
            semantic = event["semantic_by_query"][kind]
            swap, decoy = swapped_mapping(mapping, semantic, history_index + query_index)
            row = {
                "schema": "causal_residual_count_sketch_v1",
                "mechanism": "causal_residual_count_sketch_v1",
                "split": split,
                "history_id": "{}-{:06d}".format(split, history_index),
                "episode_id": "{}-{:06d}:{}".format(split, history_index, kind),
                "event_count": event_count,
                "events": events,
                "event_ordinal": ordinal,
                "query_kind": kind,
                "sketch_rows": 4,
                "sketch_buckets": 4,
                "sketch_seed": 20260714,
                "codebook": mapping,
                "codebook_swap": swap,
                "codebook_swap_decoy": decoy,
                "suffix_prompt": render_suffix(ordinal, kind, mapping),
                "codebook_swap_suffix_prompt": render_suffix(ordinal, kind, swap),
                "response": "code=" + mapping[semantic],
                "counterfactual_response": "code=" + mapping[event["counterfactual_semantic_by_query"][kind]],
                "codebook_swap_response": "code=" + swap[semantic],
                "axes": {
                    "event_count": event_count,
                    "language_heldout": language_heldout,
                    "codebook_heldout": bool(forbidden_codebooks),
                    "source_free_events": True,
                },
            }
            validate_row(row)
            rows.append(row)
    return rows, used - forbidden_codebooks


def history_key(row: dict) -> tuple:
    return tuple(
        (event["base_source"], event["edited_source"], event["donor_source"])
        for event in row["events"]
    )


def audit(train: list[dict], heldout: list[dict]) -> dict:
    train_histories = Counter(row["history_id"] for row in train)
    heldout_histories = Counter(row["history_id"] for row in heldout)
    train_keys = {history_key(row) for row in train}
    heldout_keys = {history_key(row) for row in heldout}
    train_codebooks = {codebook_key(row["codebook"]) for row in train}
    heldout_codebooks = {codebook_key(row["codebook"]) for row in heldout}
    train_surfaces = {
        "\n".join((str(row["event_ordinal"]), row["query_kind"], row["response"], *(
            event["base_source"] + "\n" + event["edited_source"] + "\n" + event["donor_source"] for event in row["events"]
        ))) for row in train
    }
    heldout_surfaces = {
        "\n".join((str(row["event_ordinal"]), row["query_kind"], row["response"], *(
            event["base_source"] + "\n" + event["edited_source"] + "\n" + event["donor_source"] for event in row["events"]
        ))) for row in heldout
    }
    train_grams = set().union(*(ngrams(surface) for surface in train_surfaces))
    heldout_grams = set().union(*(ngrams(surface) for surface in heldout_surfaces))
    return {
        "train_rows": len(train), "heldout_rows": len(heldout),
        "train_histories": len(train_histories), "heldout_histories": len(heldout_histories),
        "bad_train_history_cardinality": sum(count != len(QUERY_KINDS) for count in train_histories.values()),
        "bad_heldout_history_cardinality": sum(count != len(QUERY_KINDS) for count in heldout_histories.values()),
        "train_heldout_exact_history_hits": len(train_keys & heldout_keys),
        "train_heldout_codebook_hits": len(train_codebooks & heldout_codebooks),
        "train_heldout_semantic_13gram_hits": len(train_grams & heldout_grams),
        "train_event_counts": sorted({row["event_count"] for row in train}),
        "heldout_event_counts": sorted({row["event_count"] for row in heldout}),
    }


def admitted_parent(path: Path) -> dict:
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit("ECLI assessment is missing or empty")
    assessment = json.loads(path.read_text())
    if assessment.get("decision") != "bounded_ecli_late_binding_candidate":
        raise SystemExit("ECLI assessment does not admit CRCS data")
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ecli-assessment", required=True)
    parser.add_argument("--train-histories", type=int, default=12_000)
    parser.add_argument("--heldout-histories", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071419)
    args = parser.parse_args()
    paths = tuple(Path(item) for item in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in paths):
        raise SystemExit("all CRCS output paths must be fresh")
    assessment_path = Path(args.ecli_assessment)
    assessment = admitted_parent(assessment_path)
    train, train_codebooks = build_histories(args.train_histories, (4,), args.seed, "train", False, set())
    heldout, _ = build_histories(args.heldout_histories, (8, 16), args.seed + 1, "heldout", True, train_codebooks)
    report = audit(train, heldout)
    required_zero = (
        "bad_train_history_cardinality", "bad_heldout_history_cardinality", "train_heldout_exact_history_hits",
        "train_heldout_codebook_hits", "train_heldout_semantic_13gram_hits",
    )
    if any(report[key] for key in required_zero):
        raise SystemExit("CRCS split audit failed: {}".format(report))
    write_jsonl(paths[0], train)
    write_jsonl(paths[1], heldout)
    report.update({
        "audit": "causal_residual_count_sketch_v1",
        "claim_boundary": "CPU-only CRCS data admission. A later result could establish only bounded source-free event-history sketching, not general reasoning.",
        "train_sha256": sha256_file(paths[0]), "heldout_sha256": sha256_file(paths[1]),
        "ecli_parent_assessment": str(assessment_path),
        "ecli_parent_assessment_sha256": sha256_file(assessment_path),
        "ecli_parent_decision": assessment["decision"],
        "sketch": {"rows": 4, "buckets": 4, "seed": 20260714},
    })
    paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
