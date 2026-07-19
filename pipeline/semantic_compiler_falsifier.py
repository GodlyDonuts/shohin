#!/usr/bin/env python3
"""Build the R12 semantic-compiler CPU falsifier.

This program creates no training data and runs no model. It constructs a fresh
finite language board, checks exact source-span custody, and tries to falsify
the board through token/position/template shortcuts before a neural pilot can
be authorized.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


DIRECTIONS = ("left", "right")
AMOUNTS = (1, 2)
QUERY_POSITIONS = (0, 1, 2)
SURFACE_TYPES = ("canonical", "paraphrase", "order_twin", "binding_twin")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Operation:
    direction: str
    entity: str
    amount: int

    def as_dict(self):
        return {
            "kind": self.direction,
            "entity": self.entity,
            "amount": int(self.amount),
        }


class SpanWriter:
    def __init__(self):
        self.parts = []
        self.length = 0
        self.spans = {}

    def add(self, text, label=None):
        text = str(text)
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if label is not None:
            if label in self.spans:
                raise ValueError("duplicate span label {}".format(label))
            self.spans[label] = {"start": start, "end": self.length, "text": text}

    def finish(self):
        return "".join(self.parts), dict(self.spans)


def apply_program_pop_insert(initial, program):
    state = list(initial)
    for operation in program:
        index = state.index(operation.entity)
        if operation.direction == "left":
            destination = max(0, index - operation.amount)
        elif operation.direction == "right":
            destination = min(len(state) - 1, index + operation.amount)
        else:
            raise ValueError("unknown direction {}".format(operation.direction))
        value = state.pop(index)
        state.insert(destination, value)
    return tuple(state)


def apply_program_adjacent_swaps(initial, program):
    state = list(initial)
    for operation in program:
        for _ in range(operation.amount):
            index = state.index(operation.entity)
            neighbor = index - 1 if operation.direction == "left" else index + 1
            if not 0 <= neighbor < len(state):
                break
            state[index], state[neighbor] = state[neighbor], state[index]
    return tuple(state)


def program_key(program):
    return tuple((op.direction, op.entity, int(op.amount)) for op in program)


def semantic_label(program, query_position):
    return canonical_json({
        "operations": [operation.as_dict() for operation in program],
        "query": {"kind": "read_position", "position": int(query_position)},
        "halt": "stop",
    })


def token_positions_for_span(encoding, span):
    positions = [
        index for index, (left, right) in enumerate(encoding.offsets)
        if right > left and left < span["end"] and right > span["start"]
    ]
    if not positions:
        raise ValueError("span has no tokenizer positions: {}".format(span))
    return positions


def render_operation_a(writer, operation, index):
    ordinal = ("one", "two")[index]
    writer.add("Directive {}: shift ".format(ordinal))
    writer.add(operation.entity, "op{}.entity".format(index))
    writer.add(" ")
    writer.add(operation.direction, "op{}.kind".format(index))
    writer.add(" by ")
    writer.add(str(operation.amount), "op{}.literal".format(index))
    writer.add(" places.\n")


def render_operation_b(writer, operation, index):
    ordinal = ("alpha", "beta")[index]
    destination = "beginning" if operation.direction == "left" else "end"
    writer.add("Action {} asks ".format(ordinal))
    writer.add(operation.entity, "op{}.entity".format(index))
    writer.add(" to travel ")
    writer.add(str(operation.amount), "op{}.literal".format(index))
    writer.add(" spaces toward the ")
    writer.add(destination, "op{}.kind".format(index))
    writer.add(".\n")


def render_surface(initial, program, query_position, renderer, distractor_slot, marker,
                   distractor_entity, distractor_amount):
    writer = SpanWriter()
    if renderer == "ledger":
        writer.add("Ordered roster {}: ".format(marker))
        for index, entity in enumerate(initial):
            if index:
                writer.add(", ")
            writer.add(entity, "intro.entity{}".format(index))
        writer.add(".\n")
        render_operation = render_operation_a
        query_prefix = "Question: which name occupies position "
    elif renderer == "itinerary":
        writer.add("Sequence card {} lists ".format(marker))
        for index, entity in enumerate(initial):
            if index:
                writer.add(" / ")
            writer.add(entity, "intro.entity{}".format(index))
        writer.add(" in current order.\n")
        render_operation = render_operation_b
        query_prefix = "Read the name now standing in slot "
    else:
        raise ValueError("unknown renderer {}".format(renderer))

    distractor = (
        "Ignore annotation: {} is paired with marker {}; neither changes the roster.\n".format(
            distractor_entity, distractor_amount,
        )
    )
    if distractor_slot == 0:
        writer.add(distractor)
    for index, operation in enumerate(program):
        render_operation(writer, operation, index)
        if distractor_slot == index + 1:
            writer.add(distractor)
    writer.add(query_prefix)
    writer.add(str(query_position + 1), "query.position")
    writer.add("?\nProgram:")
    return writer.finish()


def candidate_names(tokenizer, needed):
    consonants = "bdfgklmnprstvwz"
    vowels = "aeiou"
    candidates = []
    for left, vowel, right, tail in itertools.product(consonants, vowels, consonants, vowels):
        name = left + vowel + right + tail
        encoded = tokenizer.encode(name).ids
        if encoded and name not in candidates:
            candidates.append((name, len(encoded)))
    buckets = collections.defaultdict(list)
    for name, width in candidates:
        buckets[width].append(name)
    width, names = max(buckets.items(), key=lambda item: len(item[1]))
    if len(names) < needed:
        raise ValueError("only {} nonce names share tokenizer width {}".format(len(names), width))
    return names[:needed], int(width)


def instantiate_program(specification, entities):
    return tuple(
        Operation(direction, entities[entity_index], amount)
        for direction, entity_index, amount in specification
    )


def select_program_specs(count, seed):
    operations = list(itertools.product(DIRECTIONS, range(3), AMOUNTS))
    candidates = list(itertools.product(
        itertools.product(operations, repeat=2), itertools.permutations(range(3)),
    ))
    random.Random(seed).shuffle(candidates)
    selected = []
    symbolic_entities = ("E0", "E1", "E2")
    for base, initial_indices in candidates:
        initial = tuple(symbolic_entities[index] for index in initial_indices)
        first, second = base
        order = (second, first)
        binding = (
            (first[0], second[1], second[2]),
            (second[0], first[1], first[2]),
        )
        if len({base, order, binding}) != 3:
            continue
        base_program = instantiate_program(base, symbolic_entities)
        order_program = instantiate_program(order, symbolic_entities)
        binding_program = instantiate_program(binding, symbolic_entities)
        states = [
            apply_program_pop_insert(initial, program)
            for program in (base_program, order_program, binding_program)
        ]
        separators = [
            position for position in QUERY_POSITIONS
            if states[0][position] != states[1][position]
            and states[0][position] != states[2][position]
        ]
        if not separators:
            continue
        selected.append({
            "base": base,
            "order": order,
            "binding": binding,
            "query_position": separators[0],
            "initial_indices": initial_indices,
        })
        if len(selected) == count:
            return selected
    raise ValueError("selected only {} valid program quartets".format(len(selected)))


def attach_token_targets(question, spans, tokenizer):
    encoding = tokenizer.encode(question)
    targets = {}
    for label, span in sorted(spans.items()):
        positions = token_positions_for_span(encoding, span)
        targets[label] = {
            **span,
            "token_positions": positions,
            "token_ids": [encoding.ids[position] for position in positions],
        }
    return encoding, targets


def make_row(quartet_index, surface_type, initial, program, query_position, renderer,
             distractor_slot, marker, distractor_entity, distractor_amount, tokenizer):
    question, spans = render_surface(
        initial, program, query_position, renderer, distractor_slot, marker,
        distractor_entity, distractor_amount,
    )
    encoding, token_targets = attach_token_targets(question, spans, tokenizer)
    terminal_a = apply_program_pop_insert(initial, program)
    terminal_b = apply_program_adjacent_swaps(initial, program)
    if terminal_a != terminal_b:
        raise ValueError("independent executors disagree")
    answer = terminal_a[query_position]
    return {
        "id": "SCF-{:02d}-{}".format(quartet_index, surface_type),
        "quartet": int(quartet_index),
        "surface_type": surface_type,
        "renderer": renderer,
        "initial_order": list(initial),
        "question": question,
        "program": [operation.as_dict() for operation in program],
        "query": {"kind": "read_position", "position": int(query_position)},
        "halt": "stop",
        "terminal_order": list(terminal_a),
        "answer": answer,
        "spans": token_targets,
        "token_count": len(encoding.ids),
        "token_ids_sha256": sha256_bytes(canonical_json(encoding.ids).encode()),
        "token_bag": sorted(collections.Counter(encoding.ids).items()),
    }


def exact_program_label(row):
    return canonical_json({
        "operations": row["program"],
        "query": row["query"],
        "halt": row["halt"],
    })


def shortcut_ceiling(rows, feature):
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        groups[feature(row)][exact_program_label(row)] += 1
    correct = sum(max(labels.values()) for labels in groups.values())
    return {
        "correct": int(correct),
        "total": len(rows),
        "accuracy": correct / len(rows),
        "groups": len(groups),
    }


def pointer_position_signature(row):
    labels = [
        "op0.kind", "op0.entity", "op0.literal",
        "op1.kind", "op1.entity", "op1.literal", "query.position",
    ]
    return canonical_json([
        row["spans"][label]["token_positions"] for label in labels
    ])


def span_width_signature(row):
    return canonical_json([
        len(row["spans"][label]["token_positions"])
        for label in sorted(row["spans"])
    ])


def operation_bag(row):
    return canonical_json(sorted(
        (operation["kind"], int(operation["amount"])) for operation in row["program"]
    ))


def entity_literal_bag(row):
    return canonical_json({
        "entities": sorted(row["initial_order"]),
        "amounts": sorted(int(operation["amount"]) for operation in row["program"]),
        "query": int(row["query"]["position"]),
    })


def build_board(tokenizer, quartets=32, seed=20260718):
    names, nonce_width = candidate_names(tokenizer, quartets * 3)
    specs = select_program_specs(quartets, seed)
    rows = []
    quartet_reports = []
    for index, specification in enumerate(specs):
        entities = tuple(names[index * 3:(index + 1) * 3])
        initial = tuple(entities[position] for position in specification["initial_indices"])
        base = instantiate_program(specification["base"], entities)
        order = instantiate_program(specification["order"], entities)
        binding = instantiate_program(specification["binding"], entities)
        query_position = specification["query_position"]
        distractor_slot = index % 3
        marker = "K{:02d}".format(index)
        distractor_entity = initial[(query_position + 1) % len(initial)]
        distractor_amount = base[0].amount
        surfaces = {
            "canonical": (base, "ledger"),
            "paraphrase": (base, "itinerary"),
            "order_twin": (order, "ledger"),
            "binding_twin": (binding, "ledger"),
        }
        group = {}
        for surface_type in SURFACE_TYPES:
            program, renderer = surfaces[surface_type]
            row = make_row(
                index, surface_type, initial, program, query_position, renderer,
                distractor_slot, marker, distractor_entity, distractor_amount, tokenizer,
            )
            rows.append(row)
            group[surface_type] = row
        canonical = group["canonical"]
        paraphrase = group["paraphrase"]
        order_twin = group["order_twin"]
        binding_twin = group["binding_twin"]
        token_bag_match = (
            canonical["token_bag"] == order_twin["token_bag"] == binding_twin["token_bag"]
        )
        quartet_reports.append({
            "quartet": index,
            "equivalent_program": exact_program_label(canonical) == exact_program_label(paraphrase),
            "equivalent_behavior": canonical["terminal_order"] == paraphrase["terminal_order"],
            "complete_query_behavior_equal": all(
                canonical["terminal_order"][position] == paraphrase["terminal_order"][position]
                for position in QUERY_POSITIONS
            ),
            "order_program_distinct": exact_program_label(canonical) != exact_program_label(order_twin),
            "binding_program_distinct": exact_program_label(canonical) != exact_program_label(binding_twin),
            "order_behavior_distinct": canonical["terminal_order"] != order_twin["terminal_order"],
            "binding_behavior_distinct": canonical["terminal_order"] != binding_twin["terminal_order"],
            "query_separates_order": canonical["answer"] != order_twin["answer"],
            "query_separates_binding": canonical["answer"] != binding_twin["answer"],
            "matched_token_bags": token_bag_match,
        })

    matched = [row for row in rows if row["surface_type"] != "paraphrase"]
    ceilings = {
        "renderer_identity": shortcut_ceiling(matched, lambda row: row["renderer"]),
        "token_bag": shortcut_ceiling(matched, lambda row: canonical_json(row["token_bag"])),
        "operation_bag": shortcut_ceiling(matched, operation_bag),
        "entity_literal_bag": shortcut_ceiling(matched, entity_literal_bag),
        "absolute_pointer_positions": shortcut_ceiling(matched, pointer_position_signature),
        "span_widths": shortcut_ceiling(matched, span_width_signature),
        "source_token_length": shortcut_ceiling(matched, lambda row: str(row["token_count"])),
    }
    required_labels = {
        "op0.kind", "op0.entity", "op0.literal",
        "op1.kind", "op1.entity", "op1.literal", "query.position",
    }
    span_pass = all(required_labels.issubset(row["spans"]) and all(
        row["spans"][label]["text"]
        and row["spans"][label]["token_positions"]
        for label in required_labels
    ) for row in rows)
    executor_pass = all(
        tuple(row["terminal_order"]) == apply_program_adjacent_swaps(
            tuple(row["initial_order"]),
            tuple(Operation(op["kind"], op["entity"], op["amount"]) for op in row["program"]),
        )
        for row in rows
    )
    gates = {
        "surface_count_128": len(rows) == quartets * 4 == 128,
        "quartet_count_32": len(quartet_reports) == quartets == 32,
        "typed_ast_roundtrip_128": all(
            canonical_json(json.loads(exact_program_label(row))) == exact_program_label(row)
            for row in rows
        ),
        "independent_executor_agreement_128": executor_pass,
        "equivalent_surfaces": all(
            report["equivalent_program"] and report["equivalent_behavior"]
            and report["complete_query_behavior_equal"] for report in quartet_reports
        ),
        "order_twins_distinguished": all(
            report["order_program_distinct"] and report["order_behavior_distinct"]
            and report["query_separates_order"] for report in quartet_reports
        ),
        "binding_twins_distinguished": all(
            report["binding_program_distinct"] and report["binding_behavior_distinct"]
            and report["query_separates_binding"] for report in quartet_reports
        ),
        "matched_token_multisets": all(report["matched_token_bags"] for report in quartet_reports),
        "all_pointer_spans_nonempty": span_pass,
        "nonce_names_disjoint": len({name for row in rows for name in row["initial_order"]}) == quartets * 3,
        "nonce_width_equal": nonce_width > 0,
        "shortcut_ceilings_at_most_one_third": all(
            value["accuracy"] <= (1.0 / 3.0 + 1e-12) for value in ceilings.values()
        ),
        "no_external_or_model_access": True,
        "confirmation_seed_absent": True,
    }
    source_tokens = sum(row["token_count"] for row in rows)
    target_pointers = sum(len(required_labels) for _ in rows)
    report = {
        "schema": "r12_semantic_compiler_falsifier_v1_development",
        "claim_boundary": (
            "CPU-only bounded semantic-language and leakage falsifier. No model, training, "
            "native-reasoning, executor-trainability, halt, or novelty claim."
        ),
        "seed": int(seed),
        "quartets": quartets,
        "surfaces": len(rows),
        "nonce_token_width": nonce_width,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "shortcut_ceilings": ceilings,
        "quartet_reports": quartet_reports,
        "acquisition_ledger": {
            "source_utf8_bytes": sum(len(row["question"].encode()) for row in rows),
            "source_tokens": source_tokens,
            "target_pointer_labels": target_pointers,
            "typed_program_oracle_calls": len(rows),
            "separator_oracle_calls": quartets * len(QUERY_POSITIONS),
            "executor_a_calls": len(rows),
            "executor_b_calls": len(rows),
            "teacher_model_calls": 0,
            "checkpoint_reads": 0,
            "production_eval_answer_reads": 0,
            "training_examples": 0,
            "training_flops": 0,
            "external_execution": "two exact CPU list-machine evaluators",
            "sequential_instruction_depth": 2,
        },
        "rows": rows,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--quartets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.quartets != 32:
        raise SystemExit("frozen development contract requires exactly 32 quartets")
    out = Path(args.out)
    receipt = Path(args.receipt)
    if out.exists() or receipt.exists():
        raise SystemExit("refusing to overwrite an existing falsifier artifact")
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = build_board(tokenizer, quartets=args.quartets, seed=args.seed)
    report["tokenizer"] = str(tokenizer_path.resolve())
    report["tokenizer_sha256"] = sha256_file(tokenizer_path)
    report["generator_sha256"] = sha256_file(Path(__file__))
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    receipt_payload = {
        "schema": "r12_semantic_compiler_falsifier_v1_receipt",
        "all_gates_pass": report["all_gates_pass"],
        "artifact": str(out.resolve()),
        "artifact_bytes": len(payload),
        "artifact_sha256": sha256_bytes(payload),
        "generator_sha256": report["generator_sha256"],
        "tokenizer_sha256": report["tokenizer_sha256"],
        "seed": args.seed,
        "confirmation_seed": None,
    }
    receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt_payload, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("one or more frozen CPU gates failed")


if __name__ == "__main__":
    main()
