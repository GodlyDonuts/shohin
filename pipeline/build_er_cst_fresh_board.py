#!/usr/bin/env python3
"""Build and audit the fresh ER-CST episodic rule-card board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
from typing import Iterable, Mapping, Sequence

from er_cst_fresh_renderers import (
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    parse_rendered_row,
    render_row,
)
from er_cst_rule_cards import (
    PERMUTATIONS,
    infer_position_permutation,
)


BOARD_SCHEMA = "r12_er_cst_fresh_board_report_v1"
ROW_SCHEMA = "r12_er_cst_fresh_row_v1"
PROTOCOL = "R12-ER-CST-v1.2"
TRAIN_SPLIT = "er_cst_train"
DEVELOPMENT_SPLIT = "er_cst_development"
CONFIRMATION_SPLIT = "er_cst_confirmation"
FAMILY_SIZE = 4
DEFAULT_FAMILIES = {
    TRAIN_SPLIT: 12_000,
    DEVELOPMENT_SPLIT: 512,
    CONFIRMATION_SPLIT: 512,
}
NAME_RE = re.compile(r"\b[ewox][0-9a-f]{8}\b")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_int(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _opaque(seed: int, split: str, family: int, kind: str, slot: int) -> str:
    if kind not in {"e", "w", "o", "x"}:
        raise ValueError("ER-CST opaque-name kind differs")
    digest = hashlib.sha256(
        f"{seed}:{split}:{family}:{kind}:{slot}".encode("utf-8")
    ).hexdigest()
    return f"{kind}{digest[:8]}"


def _family_id(seed: int, split: str, index: int) -> str:
    return f"er-{split}-{sha256_bytes(f'{seed}:{split}:{index}'.encode())[:16]}"


def _state_id(state: Sequence[int]) -> int:
    value = tuple(map(int, state))
    if value not in PERMUTATIONS:
        raise ValueError("ER-CST state is not a permutation")
    return PERMUTATIONS.index(value)


def _make_family(seed: int, split: str, index: int) -> dict[str, object]:
    rng = random.Random(_derived_int(seed, split, index, "family"))
    entities = tuple(_opaque(seed, split, index, "e", slot) for slot in range(3))
    opcodes = tuple(_opaque(seed, split, index, "o", slot) for slot in range(3))
    card_offset = (
        index + _derived_int(seed, split, "card-offset")
    ) % len(PERMUTATIONS)
    card_ids = tuple((card_offset + slot) % len(PERMUTATIONS) for slot in range(3))
    rules = []
    for slot, (opcode, card_id) in enumerate(zip(opcodes, card_ids, strict=True)):
        symbols = [
            _opaque(seed, split, index, "w", slot * 3 + witness)
            for witness in range(3)
        ]
        rng.shuffle(symbols)
        before = tuple(symbols)
        permutation = PERMUTATIONS[card_id]
        after = tuple(before[position] for position in permutation)
        if infer_position_permutation(before, after) != permutation:
            raise RuntimeError("ER-CST generated witness does not identify its card")
        rules.append(
            {
                "slot": slot,
                "opcode": opcode,
                "permutation_id": card_id,
                "permutation": list(permutation),
                "before": list(before),
                "after": list(after),
            }
        )

    initial = list(range(3))
    rng.shuffle(initial)
    depth = 1 + (index % 8)
    program = [rng.randrange(3) for _ in range(depth)]
    events = []
    for slot in range(9):
        halt = slot == depth
        card_slot = 0 if halt else (
            program[slot] if slot < depth else rng.randrange(3)
        )
        events.append(
            {
                "slot": slot,
                "halt": halt,
                "card_slot": card_slot,
                "opcode": None if halt else opcodes[card_slot],
            }
        )

    state: tuple[int, int, int] = tuple(initial)  # type: ignore[assignment]
    trajectory = [state]
    for card_slot in program:
        state = tuple(state[position] for position in PERMUTATIONS[card_ids[card_slot]])
        trajectory.append(state)
    query_position = index % 3
    answer_role = state[query_position]
    base: dict[str, object] = {
        "schema": ROW_SCHEMA,
        "protocol": PROTOCOL,
        "split": split,
        "supervision": "compiler_fields_only" if split == TRAIN_SPLIT else "scorer_sealed",
        "compiler_targets": {
            "entity_bindings": [
                {"role": role, "name": name} for role, name in enumerate(entities)
            ],
            "initial_order": initial,
            "initial_state_id": _state_id(initial),
            "rule_cards": rules,
            "events": events,
            "depth": depth,
            "query_position": query_position,
        },
    }
    if split != TRAIN_SPLIT:
        base["oracle"] = {
            "final_state_roles": list(state),
            "final_state_id": _state_id(state),
            "trajectory_roles": [list(value) for value in trajectory],
            "answer_role": answer_role,
            "answer_entity": entities[answer_role],
        }
    return base


def _storage_order(seed: int, split: str, family: int, renderer: str) -> list[int]:
    values = list(range(13))
    random.Random(_derived_int(seed, split, family, renderer, "storage")).shuffle(values)
    return values


def build_split(
    *, seed: int, split: str, families: int
) -> list[dict[str, object]]:
    if families <= 0:
        raise ValueError("ER-CST family count must be positive")
    renderers = TRAIN_RENDERERS if split == TRAIN_SPLIT else SCORED_RENDERERS
    rows = []
    for index in range(families):
        family_id = _family_id(seed, split, index)
        base = _make_family(seed, split, index)
        for renderer in renderers:
            rows.append(
                render_row(
                    base,
                    renderer,
                    storage_order=_storage_order(seed, split, index, renderer.name),
                    row_id=f"{family_id}::{renderer.name}",
                    family_id=family_id,
                )
            )
    return rows


def _names(row: Mapping[str, object]) -> set[str]:
    target = row["compiler_targets"]
    values = {str(item["name"]) for item in target["entity_bindings"]}
    for rule in target["rule_cards"]:
        values.add(str(rule["opcode"]))
        values.update(map(str, rule["before"]))
        values.update(map(str, rule["after"]))
    return values


def _word_ngrams(row: Mapping[str, object], size: int = 13) -> set[tuple[str, ...]]:
    words = (
        str(row["program_text"]) + "\n" + str(row["late_query_text"])
    ).lower().split()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _execute_targets(target: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    cards = {
        int(item["slot"]): tuple(map(int, item["permutation"]))
        for item in target["rule_cards"]
    }
    state = tuple(map(int, target["initial_order"]))
    trajectory = [state]
    alive = True
    for event in sorted(target["events"], key=lambda item: int(item["slot"])):
        if not alive:
            continue
        if bool(event["halt"]):
            alive = False
            continue
        state = tuple(
            state[position] for position in cards[int(event["card_slot"])]
        )
        trajectory.append(state)
    return state, tuple(trajectory)


def _row_exact(row: Mapping[str, object]) -> bool:
    try:
        parsed = parse_rendered_row(row)
        target = row["compiler_targets"]
        bindings = tuple(
            str(item["name"])
            for item in sorted(target["entity_bindings"], key=lambda item: int(item["role"]))
        )
        rules = sorted(target["rule_cards"], key=lambda item: int(item["slot"]))
        cards = {
            str(item["opcode"]): tuple(map(int, item["permutation"]))
            for item in rules
        }
        if parsed["bindings"] != bindings:
            return False
        if parsed["initial"] != tuple(bindings[int(role)] for role in target["initial_order"]):
            return False
        if set(parsed["rules"]) != set(cards):
            return False
        for opcode, (before, after) in parsed["rules"].items():
            if infer_position_permutation(before, after) != cards[opcode]:
                return False
        expected_events = tuple(
            None if bool(item["halt"]) else str(item["opcode"])
            for item in sorted(target["events"], key=lambda item: int(item["slot"]))
        )
        if parsed["events"] != expected_events or parsed["query_position"] != int(target["query_position"]):
            return False
        if sum(value is None for value in parsed["events"]) != 1:
            return False
        if parsed["events"].index(None) != int(target["depth"]):
            return False
        state, trajectory = _execute_targets(target)
        if row["split"] == TRAIN_SPLIT:
            if "oracle" in row or row.get("supervision") != "compiler_fields_only":
                return False
        else:
            oracle = row.get("oracle")
            if not isinstance(oracle, Mapping):
                return False
            answer_role = state[int(target["query_position"])]
            if (
                tuple(map(int, oracle["final_state_roles"])) != state
                or tuple(tuple(map(int, item)) for item in oracle["trajectory_roles"]) != trajectory
                or int(oracle["answer_role"]) != answer_role
                or str(oracle["answer_entity"]) != bindings[answer_role]
            ):
                return False
        source = str(row["program_text"]).encode("utf-8")
        if len(source) > 512 or len(source.splitlines()) != 13:
            return False
        if max(map(len, source.splitlines())) > 144:
            return False
        query = str(row["late_query_text"]).encode("utf-8")
        if len(query) > 144 or b"\n" in query:
            return False
        for ranges in (target["binding_ranges"], target["initial_ranges"]):
            if any(source[int(start):int(end)].decode() not in bindings for start, end in ranges):
                return False
        qstart, qend = map(int, target["query_range"])
        if query[qstart:qend].decode() != str(int(target["query_position"]) + 1):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _family_control(rows: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    canonical = rows[0]
    target = canonical["compiler_targets"]
    state, _ = _execute_targets(target)
    changed = json.loads(json.dumps(target))
    cards = [item["permutation"] for item in changed["rule_cards"]]
    rotated = cards[1:] + cards[:1]
    for item, card in zip(changed["rule_cards"], rotated, strict=True):
        item["permutation"] = card
    deranged, _ = _execute_targets(changed)
    return int(deranged == state), 1


def audit_board(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    *, expected_families: Mapping[str, int],
) -> dict[str, object]:
    expected_splits = {TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT}
    if set(splits) != expected_splits:
        raise ValueError("ER-CST board splits differ")
    summaries: dict[str, object] = {}
    names_by_split: dict[str, set[str]] = {}
    prompts_by_split: dict[str, set[str]] = {}
    ngrams_by_split: dict[str, set[tuple[str, ...]]] = {}
    families_by_split: dict[str, set[str]] = {}
    all_rows_exact = True
    names_by_family: dict[str, set[str]] = {}
    deranged_correct = 0
    deranged_total = 0

    for split, rows in splits.items():
        families: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        depth_counts: Counter[int] = Counter()
        query_counts: Counter[int] = Counter()
        card_counts: Counter[int] = Counter()
        renderer_counts: Counter[str] = Counter()
        exact = 0
        max_program_bytes = 0
        max_line_bytes = 0
        for row in rows:
            families[str(row["family_id"])].append(row)
            target = row["compiler_targets"]
            depth_counts[int(target["depth"])] += 1
            query_counts[int(target["query_position"])] += 1
            card_counts.update(int(item["permutation_id"]) for item in target["rule_cards"])
            renderer_counts[str(row["template_id"])] += 1
            exact += int(_row_exact(row))
            max_program_bytes = max(max_program_bytes, len(str(row["program_text"]).encode()))
            max_line_bytes = max(
                max_line_bytes,
                max(len(line.encode()) for line in str(row["program_text"]).splitlines()),
            )
        for family_rows in families.values():
            correct, total = _family_control(family_rows)
            deranged_correct += correct
            deranged_total += total
        family_complete = all(
            len(value) == FAMILY_SIZE
            and len({str(row["template_id"]) for row in value}) == FAMILY_SIZE
            for value in families.values()
        )
        expected_renderers = TRAIN_RENDERERS if split == TRAIN_SPLIT else SCORED_RENDERERS
        summaries[split] = {
            "rows": len(rows),
            "families": len(families),
            "family_complete": family_complete,
            "semantic_exact": exact,
            "depth_counts": dict(sorted(depth_counts.items())),
            "query_counts": dict(sorted(query_counts.items())),
            "card_counts": dict(sorted(card_counts.items())),
            "renderer_counts": dict(sorted(renderer_counts.items())),
            "max_program_bytes": max_program_bytes,
            "max_line_bytes": max_line_bytes,
            "expected_renderer_names": sorted(item.name for item in expected_renderers),
        }
        all_rows_exact &= exact == len(rows)
        for family_id, family_rows in families.items():
            family_name_sets = {frozenset(_names(row)) for row in family_rows}
            if len(family_name_sets) != 1:
                raise ValueError("ER-CST family views change opaque names")
            names_by_family[family_id] = set(next(iter(family_name_sets)))
        names_by_split[split] = set().union(
            *(names_by_family[family_id] for family_id in families)
        )
        prompts_by_split[split] = {
            str(row["program_text"]) + "\n<QUERY>\n" + str(row["late_query_text"])
            for row in rows
        }
        ngrams_by_split[split] = set().union(*(_word_ngrams(row) for row in rows))
        families_by_split[split] = set(families)

    pairs = (
        (TRAIN_SPLIT, DEVELOPMENT_SPLIT),
        (TRAIN_SPLIT, CONFIRMATION_SPLIT),
        (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT),
    )
    overlaps = {
        f"{left}__{right}": {
            "names": len(names_by_split[left] & names_by_split[right]),
            "prompts": len(prompts_by_split[left] & prompts_by_split[right]),
            "word_13grams": len(ngrams_by_split[left] & ngrams_by_split[right]),
            "families": len(families_by_split[left] & families_by_split[right]),
        }
        for left, right in pairs
    }
    expected_rows = {
        split: expected_families[split] * FAMILY_SIZE for split in expected_splits
    }
    distribution_exact = True
    for split, summary in summaries.items():
        distribution_exact &= (
            set(map(int, summary["depth_counts"])) == set(range(1, 9))
            and max(summary["depth_counts"].values()) - min(summary["depth_counts"].values()) == 0
            and max(summary["query_counts"].values()) - min(summary["query_counts"].values()) <= FAMILY_SIZE
            and max(summary["card_counts"].values()) - min(summary["card_counts"].values()) <= FAMILY_SIZE * 3
        )
    gates = {
        "exact_row_counts": all(len(splits[split]) == expected_rows[split] for split in expected_splits),
        "exact_family_counts": all(summaries[split]["families"] == expected_families[split] for split in expected_splits),
        "complete_renderer_families": all(bool(summaries[split]["family_complete"]) for split in expected_splits),
        "all_rows_semantic_exact": all_rows_exact,
        "train_has_no_oracle": all("oracle" not in row for row in splits[TRAIN_SPLIT]),
        "scored_rows_have_oracle": all("oracle" in row for split in (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT) for row in splits[split]),
        "program_within_512_bytes": all(int(summaries[split]["max_program_bytes"]) <= 512 for split in expected_splits),
        "lines_within_144_bytes": all(int(summaries[split]["max_line_bytes"]) <= 144 for split in expected_splits),
        "balanced_depth_card_query": distribution_exact,
        "globally_unique_opaque_names": sum(map(len, names_by_family.values()))
        == len(set().union(*names_by_family.values()))
        and all(
            NAME_RE.fullmatch(name) is not None
            for values in names_by_family.values()
            for name in values
        ),
        "zero_cross_split_overlap": all(all(value == 0 for value in overlap.values()) for overlap in overlaps.values()),
        "renderer_compositions_disjoint": not ({item.name for item in TRAIN_RENDERERS} & {item.name for item in SCORED_RENDERERS}),
        "deranged_card_state_below_40pct": deranged_correct / deranged_total < 0.40,
    }
    return {
        "schema": BOARD_SCHEMA,
        "protocol": PROTOCOL,
        "splits": summaries,
        "overlap": overlaps,
        "controls": {
            "family_deranged_state_exact": deranged_correct,
            "families": deranged_total,
            "rate": deranged_correct / deranged_total,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }


def build_board(
    *, seed: int, families: Mapping[str, int] = DEFAULT_FAMILIES
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    splits = {
        split: build_split(seed=seed, split=split, families=int(families[split]))
        for split in (TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
    }
    report = audit_board(splits, expected_families=families)
    report["board_seed"] = int(seed)
    return splits, report


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _verify_source_commit(source_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("ER-CST source commit must be a full lowercase Git SHA")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != source_commit:
        raise ValueError("ER-CST source commit is not current HEAD")
    if subprocess.run(["git", "diff", "--quiet", source_commit, "--"], cwd=root).returncode:
        raise ValueError("ER-CST tracked worktree differs from source commit")
    return source_commit


def write_board(
    *, output: Path, source_commit: str, splits: Mapping[str, Sequence[Mapping[str, object]]], report: Mapping[str, object]
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing existing ER-CST output: {output}")
    output.mkdir(parents=True)
    files = {}
    names = {
        TRAIN_SPLIT: "train.jsonl",
        DEVELOPMENT_SPLIT: "development.jsonl",
        CONFIRMATION_SPLIT: "confirmation.jsonl",
    }
    for split, filename in names.items():
        path = output / filename
        payload = _jsonl(splits[split])
        with path.open("xb") as destination:
            destination.write(payload)
        if split == CONFIRMATION_SPLIT:
            os.chmod(path, 0o600)
        files[filename] = {"bytes": len(payload), "rows": len(splits[split]), "sha256": sha256_bytes(payload)}
    value = dict(report)
    value["source_commit"] = source_commit
    value["files"] = files
    report_path = output / "report.json"
    with report_path.open("x") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_commit = _verify_source_commit(args.source_commit)
    splits, report = build_board(seed=args.seed)
    if report["all_gates_pass"] is not True:
        raise SystemExit("ER-CST board audit failed before write")
    value = write_board(output=args.output, source_commit=source_commit, splits=splits, report=report)
    print(canonical_json({"all_gates_pass": value["all_gates_pass"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
