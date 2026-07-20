#!/usr/bin/env python3
"""Build the fresh variable-cardinality ER-TT compiler board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from er_relation_tensor_renderers import (
    EVENT_SLOTS,
    MAX_CARDINALITY,
    MAX_RULES,
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    independently_execute,
    render_row,
)


PROTOCOL = "R12-ER-TT-v1"
BOARD_SCHEMA = "r12_er_relation_tensor_fresh_board_v1"
TRAIN_SPLIT = "er_tt_train"
DEVELOPMENT_SPLIT = "er_tt_development"
CONFIRMATION_SPLIT = "er_tt_confirmation"
DEFAULT_FAMILIES = {
    TRAIN_SPLIT: 12_000,
    DEVELOPMENT_SPLIT: 512,
    CONFIRMATION_SPLIT: 512,
}
VIEWS_PER_FAMILY = 4
MAX_SOURCE_BYTES = 640
MAX_LINE_BYTES = 144
NAME_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
NAME_WIDTH = 5
NAME_MODULUS = len(NAME_ALPHABET) ** NAME_WIDTH
FAMILY_STRIDE = 64
SPLIT_STRIDE = 20_000


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _base36(value: int) -> str:
    if not 0 <= value < NAME_MODULUS:
        raise ValueError("ER-TT compact name value exceeds its fixed width")
    chars = []
    for _ in range(NAME_WIDTH):
        value, digit = divmod(value, len(NAME_ALPHABET))
        chars.append(NAME_ALPHABET[digit])
    return "".join(reversed(chars))


class CompactNames:
    def __init__(self, seed: int) -> None:
        rng = random.Random(derived_seed(seed, "er-tt-name-bijection"))
        multiplier = rng.randrange(1, NAME_MODULUS)
        while math.gcd(multiplier, NAME_MODULUS) != 1:
            multiplier = rng.randrange(1, NAME_MODULUS)
        self.multiplier = multiplier
        self.offset = rng.randrange(NAME_MODULUS)

    def _code(self, split: str, family: int, local: int) -> str:
        split_index = (TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT).index(split)
        raw = (split_index * SPLIT_STRIDE + family) * FAMILY_STRIDE + local
        if raw >= NAME_MODULUS:
            raise ValueError("ER-TT compact-name namespace is exhausted")
        return _base36((raw * self.multiplier + self.offset) % NAME_MODULUS)

    def family(self, split: str, family: int) -> str:
        return "f" + self._code(split, family, 63)

    def entity(self, split: str, family: int, role: int) -> str:
        return "e" + self._code(split, family, role)

    def opcode(self, split: str, family: int, rule: int) -> str:
        return "o" + self._code(split, family, 8 + rule)

    def witness(self, split: str, family: int, rule: int, position: int) -> str:
        return "w" + self._code(split, family, 16 + rule * MAX_CARDINALITY + position)


def _semantic_state(
    initial_order: Sequence[int],
    relations: Sequence[Sequence[int]],
    events: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    state = tuple(map(int, initial_order))
    alive = True
    for item in events:
        if not alive:
            continue
        if bool(item["halt"]):
            alive = False
            continue
        relation = tuple(map(int, relations[int(item["card_slot"])]))
        state = tuple(state[index] for index in relation)
    return state


def _family_signature(target: Mapping[str, object]) -> str:
    value = {
        "cardinality": target["cardinality"],
        "rule_count": target["rule_count"],
        "relations": [
            item["relation"] for item in target["rule_cards"] if item["active"]
        ],
        "initial_order": target["initial_order"],
        "events": [
            [item["halt"], item["card_slot"]] for item in target["events"]
        ],
        "query_position": target["query_position"],
    }
    return sha256_bytes(canonical_json(value).encode())


def _latent_family(
    *,
    seed: int,
    split: str,
    family_index: int,
    names: CompactNames,
    forbidden_signatures: set[str],
) -> dict[str, object]:
    cardinality = 3 + family_index % 4
    rule_count = 2 + family_index % 3
    block, combination = divmod(family_index, 12)
    depth = 1 + (block + combination * 5) % 12
    for retry in range(10_000):
        rng = random.Random(
            derived_seed(seed, f"{split}:{family_index}:semantic:{retry}")
        )
        bindings = [names.entity(split, family_index, role) for role in range(cardinality)]
        initial_order = list(range(cardinality))
        rng.shuffle(initial_order)
        rules = []
        relations = []
        opcodes = []
        for slot in range(MAX_RULES):
            active = slot < rule_count
            if not active:
                rules.append(
                    {
                        "slot": slot,
                        "active": False,
                        "opcode": None,
                        "before": [],
                        "after": [],
                        "relation": [],
                    }
                )
                relations.append(tuple(range(cardinality)))
                continue
            opcode = names.opcode(split, family_index, slot)
            before = [
                names.witness(split, family_index, slot, position)
                for position in range(cardinality)
            ]
            rng.shuffle(before)
            relation = tuple(rng.randrange(cardinality) for _ in range(cardinality))
            after = [before[index] for index in relation]
            rules.append(
                {
                    "slot": slot,
                    "active": True,
                    "opcode": opcode,
                    "before": before,
                    "after": after,
                    "relation": list(relation),
                }
            )
            relations.append(relation)
            opcodes.append(opcode)
        if all(len(set(value)) == cardinality for value in relations[:rule_count]):
            relation = list(relations[0])
            relation[0] = relation[1]
            relations[0] = tuple(relation)
            rules[0]["relation"] = relation
            rules[0]["after"] = [rules[0]["before"][index] for index in relation]

        active_events = [rng.randrange(rule_count) for _ in range(depth)]
        suffix_events = [
            rng.randrange(rule_count) for _ in range(EVENT_SLOTS - depth - 1)
        ]
        card_slots = active_events + [0] + suffix_events
        events = []
        for slot, card_slot in enumerate(card_slots):
            halt = slot == depth
            events.append(
                {
                    "slot": slot,
                    "halt": halt,
                    "card_slot": card_slot,
                    "opcode": None if halt else opcodes[card_slot],
                }
            )
        query_position = rng.randrange(cardinality)
        final_state = _semantic_state(initial_order, relations, events)
        target = {
            "cardinality": cardinality,
            "rule_count": rule_count,
            "depth": depth,
            "entity_bindings": bindings,
            "initial_order": initial_order,
            "rule_cards": rules,
            "events": events,
            "query_position": query_position,
        }
        signature = _family_signature(target)
        if signature in forbidden_signatures:
            continue
        return {
            "protocol": PROTOCOL,
            "split": split,
            "compiler_targets": target,
            "semantic_family_sha256": signature,
            "oracle": {
                "final_state": list(final_state),
                "answer_role": int(final_state[query_position]),
            },
        }
    raise RuntimeError("ER-TT could not allocate a fresh semantic family")


def _span_text(row: Mapping[str, object], span: Sequence[int]) -> str:
    start, end = map(int, span)
    return str(row["program_text"])[start:end]


def validate_row(row: Mapping[str, object]) -> None:
    parsed = independently_execute(row)
    target = row["compiler_targets"]
    if not isinstance(target, Mapping):
        raise ValueError("ER-TT row target differs")
    cardinality = int(target["cardinality"])
    bindings = tuple(map(str, target["entity_bindings"]))
    initial = tuple(bindings[int(index)] for index in target["initial_order"])
    if (
        parsed["cardinality"] != cardinality
        or tuple(parsed["bindings"]) != bindings
        or tuple(parsed["initial"]) != initial
        or int(parsed["query_position"]) != int(target["query_position"])
    ):
        raise ValueError("ER-TT independent declaration/query differs")
    active = [item for item in target["rule_cards"] if item["active"]]
    expected_relations = {
        int(item["slot"]): tuple(map(int, item["relation"])) for item in active
    }
    if parsed["rule_relations"] != expected_relations:
        raise ValueError("ER-TT independent relation inference differs")
    expected_events = tuple(
        None if item["halt"] else str(item["opcode"]) for item in target["events"]
    )
    if tuple(parsed["events"]) != expected_events:
        raise ValueError("ER-TT independent event parse differs")
    expected_final = _semantic_state(
        target["initial_order"],
        [item["relation"] if item["active"] else list(range(cardinality)) for item in target["rule_cards"]],
        target["events"],
    )
    parsed_final = tuple(bindings.index(str(value)) for value in parsed["final_state"])
    if parsed_final != expected_final:
        raise ValueError("ER-TT independent execution differs")
    if int(parsed["answer_role"]) != expected_final[int(target["query_position"])]:
        raise ValueError("ER-TT independent answer differs")
    oracle = row.get("oracle")
    if row["split"] == TRAIN_SPLIT:
        if oracle is not None or row.get("supervision") != "compiler_fields_only":
            raise ValueError("ER-TT training row exposes an outcome oracle")
    elif not isinstance(oracle, Mapping) or (
        tuple(map(int, oracle["final_state"])) != expected_final
        or int(oracle["answer_role"]) != int(parsed["answer_role"])
    ):
        raise ValueError("ER-TT scored oracle differs")

    if len(target["line_ranges"]) != 18 or sorted(target["physical_roles"]) != list(range(18)):
        raise ValueError("ER-TT physical record targets differ")
    if len(target["binding_ranges"]) != cardinality or len(target["initial_ranges"]) != cardinality:
        raise ValueError("ER-TT declaration pointer targets differ")
    if tuple(_span_text(row, span) for span in target["binding_ranges"]) != bindings:
        raise ValueError("ER-TT binding spans differ")
    if tuple(_span_text(row, span) for span in target["initial_ranges"]) != initial:
        raise ValueError("ER-TT initial spans differ")
    for item, before_ranges, after_ranges in zip(
        target["rule_cards"],
        target["witness_before_ranges"],
        target["witness_after_ranges"],
        strict=True,
    ):
        if not item["active"]:
            if before_ranges or after_ranges:
                raise ValueError("ER-TT inactive rule has witness spans")
            continue
        if tuple(_span_text(row, span) for span in before_ranges) != tuple(item["before"]):
            raise ValueError("ER-TT before-witness spans differ")
        if tuple(_span_text(row, span) for span in after_ranges) != tuple(item["after"]):
            raise ValueError("ER-TT after-witness spans differ")
    query_start, query_end = map(int, target["query_range"])
    query_value = str(row["late_query_text"])[query_start:query_end]
    if query_value != str(int(target["query_position"]) + 1):
        raise ValueError("ER-TT query span differs")


def _row_names(row: Mapping[str, object]) -> set[str]:
    target = row["compiler_targets"]
    names = set(map(str, target["entity_bindings"]))
    for item in target["rule_cards"]:
        if item["active"]:
            names.add(str(item["opcode"]))
            names.update(map(str, item["before"]))
    return names


def _word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"\S+", text)
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def _control_rates(rows: Sequence[Mapping[str, object]], seed: int) -> dict[str, object]:
    first_views = [row for row in rows if str(row["id"]).endswith("-v0")]
    buckets: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in first_views:
        target = row["compiler_targets"]
        buckets[(int(target["cardinality"]), int(target["rule_count"]))].append(row)
    deranged_exact = 0
    ablated_exact = 0
    for bucket, values in buckets.items():
        cardinality, rule_count = bucket
        for index, row in enumerate(values):
            target = row["compiler_targets"]
            expected = tuple(map(int, row["oracle_private_final_state"]))
            donor = values[(index + 1) % len(values)]
            donor_relations = [
                item["relation"] for item in donor["compiler_targets"]["rule_cards"]
            ]
            deranged = _semantic_state(target["initial_order"], donor_relations, target["events"])
            deranged_exact += int(deranged == expected)
            rng = random.Random(derived_seed(seed, f"ablate:{row['family_id']}"))
            ablated_relations = [
                [rng.randrange(cardinality) for _ in range(cardinality)]
                if slot < rule_count
                else list(range(cardinality))
                for slot in range(MAX_RULES)
            ]
            ablated = _semantic_state(target["initial_order"], ablated_relations, target["events"])
            ablated_exact += int(ablated == expected)
    total = len(first_views)
    return {
        "families": total,
        "family_deranged_state_exact": deranged_exact,
        "family_deranged_state_rate": deranged_exact / total,
        "equality_ablated_state_exact": ablated_exact,
        "equality_ablated_state_rate": ablated_exact / total,
    }


def build_board(
    *,
    seed: int,
    families: Mapping[str, int] = DEFAULT_FAMILIES,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    expected_keys = {TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT}
    if set(families) != expected_keys or any(int(value) <= 0 for value in families.values()):
        raise ValueError("ER-TT family counts differ")
    if int(families[TRAIN_SPLIT]) > SPLIT_STRIDE:
        raise ValueError("ER-TT train family count exceeds the compact namespace")
    names = CompactNames(seed)
    splits: dict[str, list[dict[str, object]]] = {key: [] for key in expected_keys}
    signatures: set[str] = set()
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
            signature = str(base["semantic_family_sha256"])
            signatures.add(signature)
            family_id = names.family(split, family_index)
            private_final = list(base["oracle"]["final_state"])
            for view, renderer in enumerate(renderers):
                storage = list(range(18))
                random.Random(
                    derived_seed(seed, f"{split}:{family_index}:storage:{view}")
                ).shuffle(storage)
                row = render_row(
                    base,
                    renderer,
                    storage_order=storage,
                    row_id=f"{split}-{family_id}-v{view}",
                    family_id=family_id,
                )
                row["oracle_private_final_state"] = private_final
                if split == TRAIN_SPLIT:
                    row["oracle"] = None
                    row["supervision"] = "compiler_fields_only"
                validate_row(row)
                splits[split].append(row)

    split_names = {
        split: set().union(*(_row_names(row) for row in rows))
        for split, rows in splits.items()
    }
    split_signatures = {
        split: {str(row["semantic_family_sha256"]) for row in rows}
        for split, rows in splits.items()
    }
    split_prompts = {
        split: {
            sha256_bytes((str(row["program_text"]) + "\0" + str(row["late_query_text"])).encode())
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
            and len(set(map(int, item["relation"]))) < int(row["compiler_targets"]["cardinality"])
            for item in row["compiler_targets"]["rule_cards"]
        )
        for row in family_rows
    )
    max_source = max(int(row["source_shape"]["program_bytes"]) for row in all_rows)
    max_line = max(int(row["source_shape"]["max_line_bytes"]) for row in all_rows)
    controls = _control_rates(all_rows, seed)
    renderer_train = {renderer.name for renderer in TRAIN_RENDERERS}
    renderer_scored = {renderer.name for renderer in SCORED_RENDERERS}
    gates = {
        "row_counts_exact": all(
            len(splits[split]) == int(families[split]) * VIEWS_PER_FAMILY
            for split in expected_keys
        ),
        "semantic_validation_exact": True,
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
        "non_bijective_families_at_least_90pct": non_bijective / len(family_rows) >= 0.90,
        "program_within_640_bytes": max_source <= MAX_SOURCE_BYTES,
        "line_within_144_bytes": max_line <= MAX_LINE_BYTES,
        "deranged_state_below_40pct": controls["family_deranged_state_rate"] < 0.40,
        "equality_ablated_state_below_40pct": controls["equality_ablated_state_rate"] < 0.40,
    }
    report: dict[str, object] = {
        "schema": BOARD_SCHEMA,
        "protocol": PROTOCOL,
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
            "Fresh variable-cardinality finite-relation compiler board only; no neural score "
            "or broad-reasoning claim."
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
        raise FileExistsError(f"refusing existing ER-TT board: {output}")
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
        path.write_bytes(payload)
        if split == CONFIRMATION_SPLIT:
            path.chmod(0o600)
        files[filename] = {"bytes": len(payload), "rows": len(splits[split]), "sha256": sha256_bytes(payload)}
    final = {**dict(report), "files": files}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits, report = build_board(seed=args.seed)
    if not report["all_gates_pass"]:
        raise SystemExit("ER-TT fresh board gates failed")
    final = write_board(args.output, splits, report)
    print(canonical_json({"all_gates_pass": True, "files": final["files"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
