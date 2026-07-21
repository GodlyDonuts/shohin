#!/usr/bin/env python3
"""Build the neutral-namespace ordinal-route fresh board."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from build_er_relation_tensor_board import (
    BOARD_SCHEMA,
    CONFIRMATION_SPLIT,
    DEFAULT_FAMILIES,
    DEVELOPMENT_SPLIT,
    FAMILY_STRIDE,
    MAX_LINE_BYTES,
    MAX_SOURCE_BYTES,
    PROTOCOL,
    SPLIT_STRIDE,
    TRAIN_SPLIT,
    VIEWS_PER_FAMILY,
    CompactNames,
    _control_rates,
    _latent_family,
    _semantic_state,
    _word_ngrams,
    canonical_json,
    derived_seed,
    sha256_bytes,
)
from er_dual_stream_fresh_renderers import (
    EVENT_SLOTS,
    MAX_RULES,
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    independently_execute,
    parse_rendered_row,
    render_row,
)


BOARD_VARIANT = "ordinal_route_neutral_distractor_v1"
NEUTRAL_TOKEN = re.compile(r"(?<!\S)z[0-9a-z]{5}(?!\S)")


class NeutralCompactNames(CompactNames):
    """Allocate every semantic and distractor token from one namespace."""

    def family(self, split: str, family: int) -> str:
        return "f" + self._code(split, family, 63)

    def entity(self, split: str, family: int, role: int) -> str:
        return "z" + self._code(split, family, role)

    def opcode(self, split: str, family: int, rule: int) -> str:
        return "z" + self._code(split, family, 8 + rule)

    def witness(self, split: str, family: int, rule: int, position: int) -> str:
        return "z" + self._code(split, family, 16 + rule * 6 + position)

    def rule_distractor(self, split: str, family: int, slot: int) -> str:
        return "z" + self._code(split, family, 40 + slot)

    def event_distractor(self, split: str, family: int, slot: int) -> str:
        return "z" + self._code(split, family, 44 + slot)

    def query_distractor(self, split: str, family: int) -> str:
        return "z" + self._code(split, family, 57)


def _span_text(row: Mapping[str, object], span: Sequence[int]) -> str:
    start, end = map(int, span)
    return str(row["program_text"])[start:end]


def _source_names(row: Mapping[str, object]) -> set[str]:
    source = f"{row['program_text']}\n{row['late_query_text']}"
    return set(NEUTRAL_TOKEN.findall(source))


def validate_row(row: Mapping[str, object]) -> None:
    parsed = independently_execute(row)
    target = row.get("compiler_targets")
    if not isinstance(target, Mapping):
        raise ValueError("dual-stream row target differs")
    cardinality = int(target["cardinality"])
    bindings = tuple(map(str, target["entity_bindings"]))
    initial = tuple(bindings[int(index)] for index in target["initial_order"])
    if (
        parsed["cardinality"] != cardinality
        or tuple(parsed["bindings"]) != bindings
        or tuple(parsed["initial"]) != initial
        or int(parsed["query_position"]) != int(target["query_position"])
    ):
        raise ValueError("dual-stream declaration/query differs")
    active = [item for item in target["rule_cards"] if item["active"]]
    expected_relations = {
        int(item["slot"]): tuple(map(int, item["relation"])) for item in active
    }
    if parsed["rule_relations"] != expected_relations:
        raise ValueError("dual-stream relation inference differs")
    expected_events = tuple(
        None if item["halt"] else str(item["opcode"]) for item in target["events"]
    )
    if tuple(parsed["events"]) != expected_events:
        raise ValueError("dual-stream event parse differs")
    expected_final = _semantic_state(
        target["initial_order"],
        [
            item["relation"] if item["active"] else list(range(cardinality))
            for item in target["rule_cards"]
        ],
        target["events"],
    )
    parsed_final = tuple(bindings.index(str(value)) for value in parsed["final_state"])
    if parsed_final != expected_final:
        raise ValueError("dual-stream execution differs")
    if int(parsed["answer_role"]) != expected_final[int(target["query_position"])]:
        raise ValueError("dual-stream answer differs")
    oracle = row.get("oracle")
    if row["split"] == TRAIN_SPLIT:
        if oracle is not None or row.get("supervision") != "compiler_fields_only":
            raise ValueError("dual-stream training row exposes an outcome oracle")
    elif not isinstance(oracle, Mapping) or (
        tuple(map(int, oracle["final_state"])) != expected_final
        or int(oracle["answer_role"]) != int(parsed["answer_role"])
    ):
        raise ValueError("dual-stream scored oracle differs")

    if len(target["line_ranges"]) != 18 or sorted(target["physical_roles"]) != list(range(18)):
        raise ValueError("dual-stream physical record targets differ")
    if len(target["binding_ranges"]) != cardinality or len(target["initial_ranges"]) != cardinality:
        raise ValueError("dual-stream declaration pointer targets differ")
    if tuple(_span_text(row, span) for span in target["binding_ranges"]) != bindings:
        raise ValueError("dual-stream binding spans differ")
    if tuple(_span_text(row, span) for span in target["initial_ranges"]) != initial:
        raise ValueError("dual-stream initial spans differ")
    for item, before_ranges, after_ranges in zip(
        target["rule_cards"],
        target["witness_before_ranges"],
        target["witness_after_ranges"],
        strict=True,
    ):
        if not item["active"]:
            if before_ranges or after_ranges:
                raise ValueError("dual-stream inactive rule has witness spans")
            continue
        if tuple(_span_text(row, span) for span in before_ranges) != tuple(item["before"]):
            raise ValueError("dual-stream before-witness spans differ")
        if tuple(_span_text(row, span) for span in after_ranges) != tuple(item["after"]):
            raise ValueError("dual-stream after-witness spans differ")
    query_start, query_end = map(int, target["query_range"])
    if str(row["late_query_text"])[query_start:query_end] != str(
        int(target["query_position"]) + 1
    ):
        raise ValueError("dual-stream query span differs")
    expected_distractors = 2 + (MAX_RULES - int(target["rule_count"]))
    if len(parsed["distractors"]) != expected_distractors:
        raise ValueError("dual-stream distractor count differs")
    names = _source_names(row)
    if not names or any(not value.startswith("z") for value in names):
        raise ValueError("dual-stream source leaves one neutral namespace")


def _swap_distractors(row: Mapping[str, object]) -> dict[str, object]:
    parsed = parse_rendered_row(row)
    distractors = tuple(map(str, parsed["distractors"]))
    if len(distractors) < 2 or len(set(distractors)) != len(distractors):
        raise ValueError("dual-stream distractor swap requires unique tokens")
    rotated = distractors[1:] + distractors[:1]
    mapping = dict(zip(distractors, rotated, strict=True))
    result = json.loads(json.dumps(dict(row)))

    def replace(text: str) -> str:
        return NEUTRAL_TOKEN.sub(
            lambda match: mapping.get(match.group(0), match.group(0)), text
        )

    result["program_text"] = replace(str(result["program_text"]))
    result["late_query_text"] = replace(str(result["late_query_text"]))
    return result


def build_board(
    *,
    seed: int,
    families: Mapping[str, int] = DEFAULT_FAMILIES,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    expected_keys = {TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT}
    if set(families) != expected_keys or any(int(value) <= 0 for value in families.values()):
        raise ValueError("dual-stream family counts differ")
    if int(families[TRAIN_SPLIT]) > SPLIT_STRIDE:
        raise ValueError("dual-stream family count exceeds compact namespace")
    if FAMILY_STRIDE <= 63:
        raise ValueError("dual-stream compact family stride differs")
    names = NeutralCompactNames(seed)
    splits: dict[str, list[dict[str, object]]] = {key: [] for key in expected_keys}
    signatures: set[str] = set()
    distractor_swap_exact = 0
    distractor_swap_rows = 0
    for split in (TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT):
        renderers = TRAIN_RENDERERS if split == TRAIN_SPLIT else SCORED_RENDERERS
        for family_index in range(int(families[split])):
            base = _latent_family(
                seed=seed,
                split=split,
                family_index=family_index,
                names=names,
                forbidden_signatures=signatures,
            )
            base["board_variant"] = BOARD_VARIANT
            signature = str(base["semantic_family_sha256"])
            signatures.add(signature)
            family_id = names.family(split, family_index)
            private_final = list(base["oracle"]["final_state"])
            rule_noise = [
                names.rule_distractor(split, family_index, slot)
                for slot in range(MAX_RULES)
            ]
            query_noise = names.query_distractor(split, family_index)
            for view, renderer in enumerate(renderers):
                storage = list(range(18))
                random.Random(
                    derived_seed(seed, f"{split}:{family_index}:storage-v2:{view}")
                ).shuffle(storage)
                noise_slot = derived_seed(
                    seed, f"{split}:{family_index}:event-noise:{view}"
                ) % EVENT_SLOTS
                row = render_row(
                    base,
                    renderer,
                    storage_order=storage,
                    row_id=f"{split}-{family_id}-v{view}",
                    family_id=family_id,
                    rule_distractors=rule_noise,
                    event_distractor=names.event_distractor(
                        split, family_index, int(noise_slot)
                    ),
                    event_distractor_slot=int(noise_slot),
                    query_distractor=query_noise,
                )
                row["oracle_private_final_state"] = private_final
                if split == TRAIN_SPLIT:
                    row["oracle"] = None
                    row["supervision"] = "compiler_fields_only"
                validate_row(row)
                swapped = _swap_distractors(row)
                swapped_result = independently_execute(swapped)
                distractor_swap_rows += 1
                distractor_swap_exact += int(
                    tuple(swapped_result["final_state"])
                    == tuple(independently_execute(row)["final_state"])
                    and int(swapped_result["answer_role"])
                    == int(independently_execute(row)["answer_role"])
                )
                splits[split].append(row)

    split_names = {
        split: set().union(*(_source_names(row) for row in rows))
        for split, rows in splits.items()
    }
    split_signatures = {
        split: {str(row["semantic_family_sha256"]) for row in rows}
        for split, rows in splits.items()
    }
    split_prompts = {
        split: {
            sha256_bytes(
                (str(row["program_text"]) + "\0" + str(row["late_query_text"])).encode()
            )
            for row in rows
        }
        for split, rows in splits.items()
    }
    split_ngrams = {
        split: set().union(*(_word_ngrams(str(row["program_text"])) for row in rows))
        for split, rows in splits.items()
    }
    pairs = (
        (TRAIN_SPLIT, DEVELOPMENT_SPLIT),
        (TRAIN_SPLIT, CONFIRMATION_SPLIT),
        (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT),
    )
    overlap = {
        f"{left}__{right}": {
            "names": len(split_names[left] & split_names[right]),
            "semantic_families": len(split_signatures[left] & split_signatures[right]),
            "exact_prompts": len(split_prompts[left] & split_prompts[right]),
            "word_13grams": len(split_ngrams[left] & split_ngrams[right]),
        }
        for left, right in pairs
    }
    all_rows = [row for rows in splits.values() for row in rows]
    cardinality_counts = Counter(
        int(row["compiler_targets"]["cardinality"]) for row in all_rows
    )
    rule_counts = Counter(int(row["compiler_targets"]["rule_count"]) for row in all_rows)
    depth_counts = Counter(int(row["compiler_targets"]["depth"]) for row in all_rows)
    family_rows = [row for row in all_rows if str(row["id"]).endswith("-v0")]
    non_bijective = sum(
        any(
            item["active"]
            and len(set(map(int, item["relation"])))
            < int(row["compiler_targets"]["cardinality"])
            for item in row["compiler_targets"]["rule_cards"]
        )
        for row in family_rows
    )
    max_source = max(int(row["source_shape"]["program_bytes"]) for row in all_rows)
    max_line = max(int(row["source_shape"]["max_line_bytes"]) for row in all_rows)
    controls = _control_rates(all_rows, seed)
    controls["distractor_swap_exact"] = distractor_swap_exact
    controls["distractor_swap_rows"] = distractor_swap_rows
    controls["distractor_swap_rate"] = distractor_swap_exact / distractor_swap_rows
    renderer_train = {renderer.name for renderer in TRAIN_RENDERERS}
    renderer_scored = {renderer.name for renderer in SCORED_RENDERERS}
    gates = {
        "row_counts_exact": all(
            len(splits[split]) == int(families[split]) * VIEWS_PER_FAMILY
            for split in expected_keys
        ),
        "semantic_validation_exact": True,
        "neutral_namespace_exact": all(
            all(value.startswith("z") for value in _source_names(row))
            for row in all_rows
        ),
        "distractor_swap_semantics_exact": distractor_swap_exact == distractor_swap_rows,
        "global_names_unique_by_family_role": sum(len(value) for value in split_names.values())
        == len(set().union(*split_names.values())),
        "cross_split_names_zero": all(value["names"] == 0 for value in overlap.values()),
        "cross_split_semantic_families_zero": all(
            value["semantic_families"] == 0 for value in overlap.values()
        ),
        "cross_split_exact_prompts_zero": all(
            value["exact_prompts"] == 0 for value in overlap.values()
        ),
        "cross_split_word_13grams_zero": all(
            value["word_13grams"] == 0 for value in overlap.values()
        ),
        "train_scored_renderers_disjoint": not bool(renderer_train & renderer_scored),
        "cardinality_balanced": max(cardinality_counts.values())
        - min(cardinality_counts.values())
        <= VIEWS_PER_FAMILY,
        "rule_count_balanced": max(rule_counts.values()) - min(rule_counts.values())
        <= 2 * VIEWS_PER_FAMILY,
        "depth_balanced": max(depth_counts.values()) - min(depth_counts.values())
        <= 2 * VIEWS_PER_FAMILY,
        "non_bijective_families_at_least_90pct": non_bijective / len(family_rows)
        >= 0.90,
        "program_within_640_bytes": max_source <= MAX_SOURCE_BYTES,
        "line_within_144_bytes": max_line <= MAX_LINE_BYTES,
        "deranged_state_below_40pct": controls["family_deranged_state_rate"] < 0.40,
        "equality_ablated_state_below_40pct": controls["equality_ablated_state_rate"]
        < 0.40,
    }
    report: dict[str, object] = {
        "schema": BOARD_SCHEMA,
        "protocol": PROTOCOL,
        "board_variant": BOARD_VARIANT,
        "seed": seed,
        "families": {key: int(value) for key, value in families.items()},
        "views_per_family": VIEWS_PER_FAMILY,
        "rows": {key: len(value) for key, value in splits.items()},
        "cardinality_counts": dict(sorted(cardinality_counts.items())),
        "rule_count_counts": dict(sorted(rule_counts.items())),
        "depth_counts": dict(sorted(depth_counts.items())),
        "non_bijective_families": non_bijective,
        "non_bijective_family_rate": non_bijective / len(family_rows),
        "maximum_program_bytes": max_source,
        "maximum_line_bytes": max_line,
        "overlap": overlap,
        "renderers": {
            "train": sorted(renderer_train),
            "scored": sorted(renderer_scored),
        },
        "controls": controls,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "claim_boundary": (
            "Fresh neutral-namespace distractor board only; no neural score or "
            "broad-reasoning claim."
        ),
    }
    for rows in splits.values():
        for row in rows:
            row.pop("oracle_private_final_state", None)
    return splits, report


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode()


def write_board(
    output: Path,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    report: Mapping[str, object],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing existing dual-stream board: {output}")
    output.mkdir(parents=True)
    filenames = {
        TRAIN_SPLIT: "train.jsonl",
        DEVELOPMENT_SPLIT: "development.jsonl",
        CONFIRMATION_SPLIT: "confirmation.jsonl",
    }
    files = {}
    for split, filename in filenames.items():
        payload = _jsonl_bytes(splits[split])
        path = output / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if split != CONFIRMATION_SPLIT:
            path.chmod(0o644)
        files[filename] = {
            "bytes": len(payload),
            "rows": len(splits[split]),
            "sha256": sha256_bytes(payload),
        }
    final = {**dict(report), "files": files}
    report_path = output / "report.json"
    payload = (json.dumps(final, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits, report = build_board(seed=args.seed)
    if not report["all_gates_pass"]:
        raise SystemExit("dual-stream fresh board gates failed")
    final = write_board(args.output, splits, report)
    print(
        canonical_json(
            {"all_gates_pass": True, "files": final["files"], "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
