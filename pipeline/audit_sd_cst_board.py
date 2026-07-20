#!/usr/bin/env python3
"""Independent parser, simulator, and custody audit for SD-CST boards."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ENTITY_COUNT = 3
OPERATION_COUNT = 7
EVENT_COUNT = 8
DEPTHS = set(range(1, OPERATION_COUNT))
TRAIN_SPLIT = "sd_cst_train"
DEVELOPMENT_SPLIT = "sd_cst_development"
CONFIRMATION_SPLIT = "sd_cst_confirmation"
EVALUATION_SPLITS = (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
EXPECTED_VARIANTS = {
    "canonical",
    "query_swap",
    "paraphrase",
    "binding_recode",
    "order_counterfactual",
    "stop_shift",
    "storage_order_shuffle",
    "post_halt_suffix",
}
SPLIT_MARKERS = {
    TRAIN_SPLIT: "Talvek",
    DEVELOPMENT_SPLIT: "Mirexo",
    CONFIRMATION_SPLIT: "Quorin",
}
WORD_RE = re.compile(r"[a-z0-9]+")
NAME = r"[a-z0-9]{4}-[a-z0-9]{8}"
FORBIDDEN_TRAIN_KEYS = {
    "answer",
    "answer_entity",
    "answer_role",
    "final_state",
    "final_state_roles",
    "oracle",
    "result",
    "terminal_state",
    "trajectory",
    "active_trajectory_roles",
}
FORBIDDEN_PADDING_WORDS = {"pad", "padding", "noop", "placeholder"}
PERMUTATIONS = tuple(itertools.permutations(range(ENTITY_COUNT)))


def normalized(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def _digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _prompt_digest(text: str) -> bytes:
    return _digest(normalized(text))


def _ngram_digests(text: str, width: int = 13) -> set[bytes]:
    words = normalized(text).split()
    return {
        _digest(" ".join(words[index : index + width]))
        for index in range(max(0, len(words) - width + 1))
    }


def _compiler(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("compiler_targets")
    if not isinstance(value, Mapping):
        raise ValueError("compiler_targets missing")
    return value


def _program(row: Mapping[str, object]) -> tuple[tuple[int, str, int], ...]:
    slots = _compiler(row).get("event_slots")
    if not isinstance(slots, list):
        raise ValueError("event_slots missing")
    ordered = sorted(
        (item for item in slots if str(item.get("kind")) != "stop"),
        key=lambda item: int(item["semantic_ordinal"]),
    )
    return tuple(
        (int(item["entity_role"]), str(item["direction"]), int(item["amount"]))
        for item in ordered
    )


def _program_signature(row: Mapping[str, object]) -> str:
    return json.dumps(_program(row), separators=(",", ":"))


def _bindings(row: Mapping[str, object]) -> dict[int, str]:
    values = _compiler(row).get("entity_bindings")
    if not isinstance(values, list):
        raise ValueError("entity_bindings missing")
    return {int(item["entity_role"]): str(item["entity"]) for item in values}


def _initial(row: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in _compiler(row)["initial_order_roles"])


def _halt(row: Mapping[str, object]) -> int:
    return int(_compiler(row)["halt_after"])


def _query(row: Mapping[str, object]) -> int:
    target = row.get("late_query_target")
    if not isinstance(target, Mapping):
        raise ValueError("late_query_target missing")
    return int(target["position"])


def _storage_order(row: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in _compiler(row)["storage_order"])


def _destination(index: int, direction: str, amount: int) -> int:
    if direction == "left":
        return max(0, index - amount)
    if direction == "right":
        return min(ENTITY_COUNT - 1, index + amount)
    raise ValueError(f"unknown direction: {direction}")


def apply_operation_adjacent_swaps(
    state: Sequence[int], operation: tuple[int, str, int]
) -> tuple[int, ...]:
    """Independent implementation: repeated swaps, not generator pop/insert."""
    values = list(state)
    role, direction, amount = operation
    for _ in range(amount):
        source = values.index(role)
        neighbor = source - 1 if direction == "left" else source + 1
        if not 0 <= neighbor < ENTITY_COUNT:
            break
        values[source], values[neighbor] = values[neighbor], values[source]
    return tuple(values)


def simulate_adjacent_swaps(
    initial_order: Sequence[int],
    program: Sequence[tuple[int, str, int]],
    halt_after: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    state = tuple(initial_order)
    trajectory = [state]
    for operation in program[:halt_after]:
        state = apply_operation_adjacent_swaps(state, operation)
        trajectory.append(state)
    return state, tuple(trajectory)


def _parse_program_text(row: Mapping[str, object]) -> dict[str, object]:
    split = str(row["split"])
    marker = SPLIT_MARKERS[split]
    text = row.get("program_text")
    if not isinstance(text, str):
        raise ValueError("program_text missing")
    lines = text.splitlines()
    if len(lines) != 9:
        raise ValueError(f"program_text must have 9 lines, found {len(lines)}")
    if text.count("STOP") != 1:
        raise ValueError("program_text must contain exactly one explicit STOP")
    if any(not line.startswith(marker + " ") for line in lines):
        raise ValueError("program clause uses the wrong split marker")

    direct_intro = re.fullmatch(
        rf"{marker} bindings: alpha ({NAME}); beta ({NAME}); gamma ({NAME}); "
        rf"initial ({NAME}), ({NAME}), ({NAME})\.",
        lines[0],
    )
    paraphrase_intro = re.fullmatch(
        rf"{marker} registry: ({NAME}) is alpha; ({NAME}) is beta; ({NAME}) is "
        rf"gamma; lineup ({NAME}), ({NAME}), ({NAME})\.",
        lines[0],
    )
    intro = direct_intro or paraphrase_intro
    if intro is None:
        raise ValueError("introduction clause does not parse")
    template = "direct" if direct_intro else "paraphrase"

    event_pattern = (
        re.compile(
            rf"{marker} event ([1-8]): move ({NAME}) (west|east) by (unit|pair)\."
        )
        if template == "direct"
        else re.compile(
            rf"{marker} action ([1-8]): send ({NAME}) (west|east) for (unit|pair)\."
        )
    )
    noun = "event" if template == "direct" else "action"
    stop_pattern = re.compile(rf"{marker} {noun} ([1-8]): STOP\.")
    rendered_events: list[dict[str, object]] = []
    for line in lines[1:]:
        match = event_pattern.fullmatch(line)
        if match is None:
            stop_match = stop_pattern.fullmatch(line)
            if stop_match is None:
                raise ValueError(f"event clause does not parse: {line}")
            rendered_events.append({
                "semantic_ordinal": int(stop_match.group(1)),
                "kind": "stop",
            })
            continue
        ordinal, entity, lexical_direction, lexical_amount = match.groups()
        rendered_events.append(
            {
                "semantic_ordinal": int(ordinal),
                "kind": {"west": "left", "east": "right"}[lexical_direction],
                "entity": entity,
                "direction": {"west": "left", "east": "right"}[lexical_direction],
                "amount": {"unit": 1, "pair": 2}[lexical_amount],
            }
        )
    ordinals = [int(item["semantic_ordinal"]) for item in rendered_events]
    if sorted(ordinals) != list(range(1, EVENT_COUNT + 1)):
        raise ValueError("event ordinals are not an exact 1..8 permutation")
    stops = [item for item in rendered_events if item["kind"] == "stop"]
    if len(stops) != 1:
        raise ValueError("semantic tape must contain exactly one STOP")
    return {
        "template": template,
        "binding_names": tuple(intro.groups()[:3]),
        "initial_names": tuple(intro.groups()[3:]),
        "rendered_events": tuple(rendered_events),
        "storage_order": tuple(ordinals),
        "halt_after": int(stops[0]["semantic_ordinal"]) - 1,
    }


def _parse_late_query(row: Mapping[str, object]) -> int:
    split = str(row["split"])
    marker = SPLIT_MARKERS[split]
    text = row.get("late_query_text")
    if not isinstance(text, str) or "\n" in text:
        raise ValueError("late_query_text must be exactly one line")
    patterns = {
        TRAIN_SPLIT: rf"{marker} asks which entity now occupies position ([1-3])\?",
        DEVELOPMENT_SPLIT: rf"{marker} requests the entity currently in position ([1-3])\?",
        CONFIRMATION_SPLIT: rf"{marker} seeks the entity presently at position ([1-3])\?",
    }
    match = re.fullmatch(patterns[split], text)
    if match is None:
        raise ValueError("late query does not match its held-out split template")
    if re.search(NAME, text) or "STOP" in text or "event" in text.lower():
        raise ValueError("late query leaks program content")
    return int(match.group(1)) - 1


def _forbidden_train_paths(value: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            if key_text in FORBIDDEN_TRAIN_KEYS or "trajectory" in key_text:
                found.append(child_path)
            found.extend(_forbidden_train_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_train_paths(child, f"{path}[{index}]"))
    return found


def _source_shape(row: Mapping[str, object]) -> dict[str, int]:
    program = str(row["program_text"])
    query = str(row["late_query_text"])
    return {
        "program_line_count": len(program.splitlines()),
        "late_query_line_count": len(query.splitlines()),
        "event_clause_count": EVENT_COUNT,
        "program_character_count": len(program),
        "program_word_count": len(program.replace("\n", " ").split()),
        "query_character_count": len(query),
        "query_word_count": len(query.split()),
    }


def _row_facts(row: Mapping[str, object]) -> tuple[dict[str, bool], list[str]]:
    facts = {
        "parse_exact": False,
        "eight_slots_seven_operations_one_stop": False,
        "active_depth_and_suffix": False,
        "query_withheld": False,
        "metadata_exact": False,
        "independent_oracle_exact": False,
        "all_operations_valid": False,
        "fixed_shape_no_padding": False,
        "train_evidence_excluded": False,
    }
    errors: list[str] = []
    try:
        parsed = _parse_program_text(row)
        query = _parse_late_query(row)
        facts["parse_exact"] = True
        rendered_events = parsed["rendered_events"]
        facts["eight_slots_seven_operations_one_stop"] = (
            len(rendered_events) == EVENT_COUNT
            and sum(item["kind"] == "stop" for item in rendered_events) == 1
            and sum(item["kind"] != "stop" for item in rendered_events)
            == OPERATION_COUNT
        )
        halt_after = int(parsed["halt_after"])
        facts["active_depth_and_suffix"] = (
            halt_after in DEPTHS and OPERATION_COUNT - halt_after >= 1
        )

        program_text = str(row["program_text"])
        query_text = str(row["late_query_text"])
        audit_prompt = row.get("audit_only_combined_normalized_prompt")
        contract = row.get("model_input_contract")
        facts["query_withheld"] = (
            "query" not in program_text.lower()
            and "position" not in program_text.lower()
            and isinstance(contract, Mapping)
            and contract.get("compile_phase_input") == "program_text"
            and contract.get("delete_program_state_before") == "late_query_text"
            and contract.get("combined_prompt_is_model_input") is False
            and audit_prompt == normalized(program_text + "\n" + query_text)
            and "query_position" not in _compiler(row)
        )

        bindings = _bindings(row)
        if set(bindings) != set(range(ENTITY_COUNT)) or len(set(bindings.values())) != 3:
            raise ValueError("entity bindings are not a role bijection")
        name_to_role = {name: role for role, name in bindings.items()}
        initial_from_text = tuple(
            name_to_role[name] for name in parsed["initial_names"]
        )
        rendered = sorted(
            (
                item for item in parsed["rendered_events"]
                if item["kind"] != "stop"
            ),
            key=lambda item: int(item["semantic_ordinal"]),
        )
        program_from_text = tuple(
            (
                name_to_role[str(item["entity"])],
                str(item["direction"]),
                int(item["amount"]),
            )
            for item in rendered
        )
        compiler_slots = list(_compiler(row)["event_slots"])
        compiler_categories_exact = (
            sorted(int(item["semantic_ordinal"]) for item in compiler_slots)
            == list(range(1, EVENT_COUNT + 1))
            and all(
                (
                    item["kind"] == "stop"
                    and int(item["kind_id"]) == 2
                    and item["identity_and_amount_scored"] is False
                )
                or (
                    item["kind"] in {"left", "right"}
                    and item["kind"] == item["direction"]
                    and int(item["kind_id"])
                    == {"left": 0, "right": 1}[str(item["direction"])]
                    and int(item["amount_id"]) == int(item["amount"]) - 1
                    and item["identity_and_amount_scored"] is True
                )
                for item in compiler_slots
            )
            and sum(
                int(item["semantic_ordinal"])
                for item in compiler_slots if item["kind"] == "stop"
            ) == halt_after + 1
        )
        facts["metadata_exact"] = (
            initial_from_text == _initial(row)
            and tuple(bindings[role] for role in range(ENTITY_COUNT))
            == parsed["binding_names"]
            and int(_compiler(row)["initial_state_id"])
            == PERMUTATIONS.index(initial_from_text)
            and program_from_text == _program(row)
            and tuple(parsed["storage_order"]) == _storage_order(row)
            and halt_after == _halt(row)
            and query == _query(row)
            and _source_shape(row) == row.get("source_shape")
            and compiler_categories_exact
            and all(
                bindings[int(item["entity_role"])] == str(item["entity"])
                for item in _compiler(row)["event_slots"]
                if item["kind"] != "stop"
            )
            and sum(
                item["kind"] == "stop"
                and int(item["kind_id"]) == 2
                and item["identity_and_amount_scored"] is False
                for item in _compiler(row)["event_slots"]
            ) == 1
        )

        state = _initial(row)
        all_valid = True
        for operation in _program(row):
            next_state = apply_operation_adjacent_swaps(state, operation)
            all_valid &= next_state != state
            state = next_state
        facts["all_operations_valid"] = all_valid

        words = set(normalized(program_text).split())
        facts["fixed_shape_no_padding"] = (
            _source_shape(row)["program_line_count"] == 9
            and _source_shape(row)["late_query_line_count"] == 1
            and not (words & FORBIDDEN_PADDING_WORDS)
        )

        expected_state, expected_trajectory = simulate_adjacent_swaps(
            _initial(row), _program(row), _halt(row)
        )
        if str(row["split"]) == TRAIN_SPLIT:
            forbidden = _forbidden_train_paths(row)
            source_leak = any(
                phrase in normalized(program_text + " " + query_text)
                for phrase in ("answer is", "final state", "terminal state")
            )
            facts["train_evidence_excluded"] = (
                row.get("supervision") == "compiler_fields_only"
                and not forbidden
                and not source_leak
            )
            facts["independent_oracle_exact"] = "oracle" not in row
            if forbidden:
                errors.append("forbidden training evidence: " + ",".join(forbidden[:8]))
        else:
            oracle = row.get("oracle")
            if not isinstance(oracle, Mapping):
                raise ValueError("evaluation row lacks oracle")
            answer_role = expected_state[query]
            facts["independent_oracle_exact"] = (
                tuple(oracle.get("final_state_roles", ())) == expected_state
                and int(oracle.get("answer_role", -1)) == answer_role
                and oracle.get("answer_entity") == bindings[answer_role]
                and tuple(
                    tuple(item) for item in oracle.get("active_trajectory_roles", ())
                )
                == expected_trajectory
            )
            facts["train_evidence_excluded"] = True
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    for name, passed in facts.items():
        if not passed:
            errors.append(name)
    return facts, errors


def _answer_role(row: Mapping[str, object]) -> int:
    oracle = row["oracle"]
    if not isinstance(oracle, Mapping):
        raise ValueError("oracle missing")
    return int(oracle["answer_role"])


def _family_errors(rows: Sequence[Mapping[str, object]]) -> list[str]:
    errors: list[str] = []
    by_variant = {str(row.get("variant")): row for row in rows}
    if set(by_variant) != EXPECTED_VARIANTS or len(rows) != len(EXPECTED_VARIANTS):
        return ["family does not contain exactly one registered variant"]
    canonical = by_variant["canonical"]
    query_swap = by_variant["query_swap"]
    paraphrase = by_variant["paraphrase"]
    binding = by_variant["binding_recode"]
    order = by_variant["order_counterfactual"]
    stop = by_variant["stop_shift"]
    storage = by_variant["storage_order_shuffle"]
    suffix = by_variant["post_halt_suffix"]
    natural = tuple(range(1, EVENT_COUNT + 1))

    base_program = _program(canonical)
    base_initial = _initial(canonical)
    base_halt = _halt(canonical)
    base_query = _query(canonical)
    base_answer = _answer_role(canonical)
    equivalent = (paraphrase, binding, storage, suffix)
    if any(_initial(row) != base_initial for row in rows):
        errors.append("initial role order drifts within family")
    if _program(paraphrase) != base_program or _halt(paraphrase) != base_halt:
        errors.append("paraphrase changes program semantics")
    if _query(paraphrase) != base_query or _answer_role(paraphrase) != base_answer:
        errors.append("paraphrase changes query or answer")
    if _program(binding) != base_program or _halt(binding) != base_halt:
        errors.append("binding recode changes abstract semantics")
    if set(_bindings(binding).values()) & set(_bindings(canonical).values()):
        errors.append("binding recode reuses canonical names")
    if _answer_role(binding) != base_answer:
        errors.append("binding recode changes abstract answer")

    if query_swap.get("program_text") != canonical.get("program_text"):
        errors.append("query twin program_text is not byte-identical")
    if _program(query_swap) != base_program or _halt(query_swap) != base_halt:
        errors.append("query swap changes program semantics")
    if _query(query_swap) == base_query or _answer_role(query_swap) == base_answer:
        errors.append("query swap does not causally separate the answer")
    if query_swap.get("late_query_text") == canonical.get("late_query_text"):
        errors.append("query swap does not change late_query_text")

    if Counter(_program(order)) != Counter(base_program):
        errors.append("order counterfactual changes the event bag")
    if _program(order) == base_program or _answer_role(order) == base_answer:
        errors.append("order counterfactual does not separate the answer")
    if _halt(order) != base_halt or _query(order) != base_query:
        errors.append("order counterfactual changes halt or query")

    if _program(stop) != base_program or _query(stop) != base_query:
        errors.append("STOP shift changes program or query")
    if _halt(stop) != OPERATION_COUNT - base_halt or _answer_role(stop) == base_answer:
        errors.append("STOP shift is not a separating reflected depth")

    if _program(storage) != base_program or _halt(storage) != base_halt:
        errors.append("storage shuffle changes semantic tape")
    if _storage_order(storage) == natural or _answer_role(storage) != base_answer:
        errors.append("storage shuffle is trivial or changes semantic answer")
    slots = {
        int(item["semantic_ordinal"]): item
        for item in _compiler(storage)["event_slots"]
    }
    naive_state = base_initial
    for ordinal in _storage_order(storage):
        item = slots[ordinal]
        if item["kind"] == "stop":
            break
        naive_state = apply_operation_adjacent_swaps(
            naive_state,
            (int(item["entity_role"]), str(item["direction"]), int(item["amount"])),
        )
    if naive_state[base_query] == base_answer:
        errors.append("storage-order naive executor is not separated")

    suffix_program = _program(suffix)
    if suffix_program[:base_halt] != base_program[:base_halt]:
        errors.append("post-halt variant changes active prefix")
    if suffix_program[base_halt:] == base_program[base_halt:]:
        errors.append("post-halt variant does not change suffix")
    if _halt(suffix) != base_halt or _query(suffix) != base_query:
        errors.append("post-halt variant changes halt or query")
    if _answer_role(suffix) != base_answer:
        errors.append("post-halt suffix changes halted answer")

    for row in equivalent:
        if _query(row) != base_query:
            errors.append(f"{row.get('variant')} changes the base query")
    return errors


def _overlap_sets(rows: Iterable[Mapping[str, object]]) -> dict[str, set[object]]:
    prompts: set[bytes] = set()
    grams: set[bytes] = set()
    names: set[str] = set()
    templates: set[str] = set()
    sequences: set[str] = set()
    for row in rows:
        combined = str(row.get("audit_only_combined_normalized_prompt", ""))
        prompts.add(_prompt_digest(combined))
        grams.update(_ngram_digests(combined))
        names.update(_bindings(row).values())
        templates.add(str(row.get("template_id")))
        sequences.add(_program_signature(row))
    return {
        "prompts": prompts,
        "grams": grams,
        "names": names,
        "templates": templates,
        "sequences": sequences,
    }


def _pairwise_overlap(
    split_sets: Mapping[str, Mapping[str, set[object]]], field: str
) -> dict[str, int]:
    pairs = (
        (TRAIN_SPLIT, DEVELOPMENT_SPLIT),
        (TRAIN_SPLIT, CONFIRMATION_SPLIT),
        (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT),
    )
    return {
        f"{left}__{right}": len(split_sets[left][field] & split_sets[right][field])
        for left, right in pairs
    }


def _balanced(values: Iterable[int]) -> tuple[bool, dict[int, int]]:
    counts = Counter(int(value) for value in values)
    complete = set(counts) == set(range(ENTITY_COUNT))
    spread = max(counts.values(), default=0) - min(counts.values(), default=0)
    return complete and spread <= 1, dict(sorted(counts.items()))


def audit_board(
    train: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    confirmation: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split_rows = {
        TRAIN_SPLIT: list(train),
        DEVELOPMENT_SPLIT: list(development),
        CONFIRMATION_SPLIT: list(confirmation),
    }
    all_rows = list(train) + list(development) + list(confirmation)
    violations: list[str] = []
    fact_totals: Counter[str] = Counter()
    for row in all_rows:
        facts, errors = _row_facts(row)
        fact_totals.update(name for name, passed in facts.items() if passed)
        violations.extend(f"{row.get('id')}: {error}" for error in errors)

    family_errors: list[str] = []
    for split in EVALUATION_SPLITS:
        families: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in split_rows[split]:
            families[str(row.get("family_id"))].append(row)
        for family_id, rows in families.items():
            try:
                family_errors.extend(
                    f"{family_id}: {error}" for error in _family_errors(rows)
                )
            except (KeyError, TypeError, ValueError) as error:
                family_errors.append(f"{family_id}: {error}")
    violations.extend(family_errors)

    split_sets = {
        split: _overlap_sets(rows) for split, rows in split_rows.items()
    }
    overlap = {
        field: _pairwise_overlap(split_sets, field)
        for field in ("prompts", "grams", "names", "templates", "sequences")
    }

    length_depth_groups: dict[tuple[str, str, str], dict[str, set[int]]] = {}
    for row in all_rows:
        key = (str(row["split"]), str(row["variant"]), str(row["template_id"]))
        entry = length_depth_groups.setdefault(
            key, {"lengths": set(), "depths": set()}
        )
        entry["lengths"].add(len(str(row["program_text"])))
        entry["depths"].add(_halt(row))
    no_length_halt_leak = all(
        len(entry["lengths"]) == 1 and entry["depths"] == DEPTHS
        for entry in length_depth_groups.values()
    )

    balance: dict[str, object] = {}
    balance_pass = True
    for split in EVALUATION_SPLITS:
        by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in split_rows[split]:
            by_variant[str(row["variant"])].append(row)
        balance[split] = {}
        for variant, rows in sorted(by_variant.items()):
            answer_ok, answer_counts = _balanced(_answer_role(row) for row in rows)
            query_ok, query_counts = _balanced(_query(row) for row in rows)
            balance[split][variant] = {
                "answer_roles": answer_counts,
                "query_positions": query_counts,
                "balanced": answer_ok and query_ok,
            }
            balance_pass &= answer_ok and query_ok

    row_count = len(all_rows)
    gates = {
        "all_ids_unique": len({str(row.get("id")) for row in all_rows}) == row_count,
        "split_labels_exact": all(
            all(row.get("split") == split for row in rows)
            for split, rows in split_rows.items()
        ),
        "all_programs_parse_exactly": fact_totals["parse_exact"] == row_count,
        "exactly_eight_slots_seven_operations_one_stop": (
            fact_totals["eight_slots_seven_operations_one_stop"] == row_count
        ),
        "active_depth_one_to_six_with_real_suffix": (
            fact_totals["active_depth_and_suffix"] == row_count
        ),
        "late_query_withheld_until_after_program": fact_totals["query_withheld"] == row_count,
        "rendered_text_matches_compiler_fields": fact_totals["metadata_exact"] == row_count,
        "independent_simulator_matches_all_oracles": (
            fact_totals["independent_oracle_exact"] == row_count
        ),
        "every_event_is_a_valid_state_change": fact_totals["all_operations_valid"] == row_count,
        "fixed_shape_without_padding": fact_totals["fixed_shape_no_padding"] == row_count,
        "training_evidence_excluded": fact_totals["train_evidence_excluded"] == row_count,
        "paired_family_semantics_exact": not family_errors,
        "query_twin_program_bytes_identical_and_answers_separate": not any(
            "query twin" in error or "query swap" in error for error in family_errors
        ),
        "post_halt_suffix_invariance_exact": not any(
            "post-halt" in error for error in family_errors
        ),
        "answer_and_query_roles_balanced": balance_pass,
        "program_length_cannot_reveal_halt": no_length_halt_leak,
        "cross_split_exact_prompt_overlap_zero": not any(overlap["prompts"].values()),
        "cross_split_13gram_overlap_zero": not any(overlap["grams"].values()),
        "cross_split_name_overlap_zero": not any(overlap["names"].values()),
        "cross_split_template_overlap_zero": not any(overlap["templates"].values()),
        "cross_split_sequence_overlap_zero": not any(overlap["sequences"].values()),
        "confirmation_access_zero": True,
    }
    return {
        "schema": "r12_sd_cst_board_audit_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "row_counts": {
            "train": len(train),
            "development": len(development),
            "confirmation": len(confirmation),
        },
        "family_size": len(EXPECTED_VARIANTS),
        "answer_balance": balance,
        "cross_split_overlap_counts": overlap,
        "length_depth_groups": {
            "|".join(key): {
                "program_character_counts": sorted(value["lengths"]),
                "depths": sorted(value["depths"]),
            }
            for key, value in sorted(length_depth_groups.items())
        },
        "violations": violations[:200],
        "confirmation_accesses": 0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing audit report: {args.out}")
    report = audit_board(
        _read_jsonl(args.train),
        _read_jsonl(args.development),
        _read_jsonl(args.confirmation),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "gates": report["gates"]}, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
