#!/usr/bin/env python3
"""Build split-frozen data for the complete referential compiler pilot.

The generator consumes no model and no evaluation answer. Structured machine
programs are used only to construct gold pointer targets and exact audits. The
confirmation seed must be supplied only after this source is committed.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
from pathlib import Path

from tokenizers import Tokenizer

from semantic_compiler_falsifier import (
    AMOUNTS,
    DIRECTIONS,
    QUERY_POSITIONS,
    Operation,
    SpanWriter,
    apply_program_adjacent_swaps,
    apply_program_pop_insert,
    attach_token_targets,
    candidate_names,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


WORD = re.compile(r"\w+")
SURFACE_TYPES = ("canonical", "paraphrase", "order_twin", "binding_twin")
SPLIT_RENDERERS = {
    "train": ("forge", "route"),
    "development": ("archive", "tableau"),
    "confirmation": ("docket", "procession"),
}
SPLIT_MARKERS = {
    "train": ("amber", "cobalt", "ivory", "saffron"),
    "development": ("lilac", "ochre", "teal", "umber"),
    "confirmation": ("coral", "indigo", "pearl", "vermilion"),
}


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {
        " ".join(words[index:index + width])
        for index in range(max(0, len(words) - width + 1))
    }


def instantiate(specification, entities):
    return tuple(
        Operation(direction, entities[entity_index], int(amount))
        for direction, entity_index, amount in specification
    )


def valid_configurations():
    operations = list(itertools.product(DIRECTIONS, range(3), AMOUNTS))
    configurations = []
    symbolic = ("E0", "E1", "E2")
    for base in itertools.product(operations, repeat=2):
        first, second = base
        order = (second, first)
        binding = (
            (first[0], second[1], second[2]),
            (second[0], first[1], first[2]),
        )
        if len({base, order, binding}) != 3:
            continue
        for initial_indices in itertools.permutations(range(3)):
            initial = tuple(symbolic[index] for index in initial_indices)
            programs = [instantiate(spec, symbolic) for spec in (base, order, binding)]
            states = [apply_program_pop_insert(initial, program) for program in programs]
            for query_position in QUERY_POSITIONS:
                if states[0][query_position] == states[1][query_position]:
                    continue
                if states[0][query_position] == states[2][query_position]:
                    continue
                configurations.append({
                    "base": base,
                    "order": order,
                    "binding": binding,
                    "initial_indices": initial_indices,
                    "query_position": query_position,
                })
    if not configurations:
        raise ValueError("no valid semantic configurations")
    return tuple(configurations)


def add_intro(writer, renderer, marker, initial):
    if renderer == "forge":
        writer.add("Forge roster {} begins in this order: ".format(marker))
        separator, suffix = ", ", ".\n"
    elif renderer == "route":
        writer.add("Route card {} records the current sequence as ".format(marker))
        separator, suffix = " / ", ".\n"
    elif renderer == "archive":
        writer.add("Archive strip {} starts with ".format(marker))
        separator, suffix = " then ", " in that exact order.\n"
    elif renderer == "tableau":
        writer.add("Tableau {} has the lineup ".format(marker))
        separator, suffix = " | ", " before any changes.\n"
    elif renderer == "docket":
        writer.add("Docket {} declares the opening procession ".format(marker))
        separator, suffix = " followed by ", ".\n"
    elif renderer == "procession":
        writer.add("Procession note {} gives this initial arrangement: ".format(marker))
        separator, suffix = " ; ", ".\n"
    else:
        raise ValueError("unknown renderer {}".format(renderer))
    for index, entity in enumerate(initial):
        if index:
            writer.add(separator)
        writer.add(entity, "intro.entity{}".format(index))
    writer.add(suffix)


def add_operation(writer, renderer, operation, index):
    ordinal = ("one", "two")[index]
    greek = ("alpha", "beta")[index]
    if renderer == "forge":
        writer.add("Instruction {}: move ".format(ordinal))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" toward the ")
        writer.add("left edge" if operation.direction == "left" else "right edge",
                   "op{}.kind".format(index))
        writer.add(" by ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" positions.\n")
    elif renderer == "route":
        writer.add("Change {} sends ".format(greek))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" slots toward the ")
        writer.add("beginning" if operation.direction == "left" else "end",
                   "op{}.kind".format(index))
        writer.add(".\n")
    elif renderer == "archive":
        writer.add("Edit {}: slide ".format(ordinal))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" ")
        writer.add("earlier" if operation.direction == "left" else "later",
                   "op{}.kind".format(index))
        writer.add(" through ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" places.\n")
    elif renderer == "tableau":
        writer.add("Adjustment {} requires ".format(greek))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" to travel ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" cells toward the ")
        writer.add("front" if operation.direction == "left" else "rear",
                   "op{}.kind".format(index))
        writer.add(".\n")
    elif renderer == "docket":
        writer.add("Mandate {} shifts ".format(ordinal))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" in the ")
        writer.add("backward" if operation.direction == "left" else "forward",
                   "op{}.kind".format(index))
        writer.add(" index direction for ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" steps.\n")
    elif renderer == "procession":
        writer.add("Motion {} tells ".format(greek))
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" to cross ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" neighbors toward the ")
        writer.add("head" if operation.direction == "left" else "tail",
                   "op{}.kind".format(index))
        writer.add(".\n")
    else:
        raise ValueError("unknown renderer {}".format(renderer))


def distractor_text(renderer, entity, amount):
    templates = {
        "forge": "Side note: {} wears badge {}; this note changes no position.\n",
        "route": "Decoration: {} carries route tag {}; ignore the decoration.\n",
        "archive": "Marginalia pairs {} with index {}; it is not an edit.\n",
        "tableau": "Caption only: {} has plaque {}; the lineup is unaffected.\n",
        "docket": "Clerk note: {} is stamped {}; the stamp has no ordering force.\n",
        "procession": "Banner text links {} to numeral {}; do not move it for that reason.\n",
    }
    return templates[renderer].format(entity, amount)


def add_query(writer, renderer, query_position):
    prefixes = {
        "forge": "Read the name finally occupying roster position ",
        "route": "Which name ends in sequence slot ",
        "archive": "Return the entry now found at archive place ",
        "tableau": "Identify the member standing in tableau cell ",
        "docket": "Which name holds final docket rank ",
        "procession": "Report the participant now at procession location ",
    }
    writer.add(prefixes[renderer])
    writer.add(str(query_position + 1), "query.position")
    writer.add("?\nProgram:")


def render_source(initial, program, query_position, renderer, distractor_slot,
                  marker, distractor_entity, distractor_amount):
    writer = SpanWriter()
    add_intro(writer, renderer, marker, initial)
    distractor = distractor_text(renderer, distractor_entity, distractor_amount)
    if distractor_slot == 0:
        writer.add(distractor)
    for index, operation in enumerate(program):
        add_operation(writer, renderer, operation, index)
        if distractor_slot == index + 1:
            writer.add(distractor)
    add_query(writer, renderer, query_position)
    return writer.finish()


def row_label(row):
    return canonical_json({
        "operations": row["program"],
        "query": row["query"],
        "halt": row["halt"],
    })


def make_row(split, group_index, surface_type, initial, program, query_position,
             renderer, distractor_slot, marker, distractor_entity,
             distractor_amount, tokenizer):
    question, spans = render_source(
        initial, program, query_position, renderer, distractor_slot, marker,
        distractor_entity, distractor_amount,
    )
    encoding, token_targets = attach_token_targets(question, spans, tokenizer)
    terminal_a = apply_program_pop_insert(initial, program)
    terminal_b = apply_program_adjacent_swaps(initial, program)
    if terminal_a != terminal_b:
        raise ValueError("executor disagreement")
    return {
        "id": "RLPC-{}-{:06d}-{}".format(split, group_index, surface_type),
        "split": split,
        "group": int(group_index),
        "surface_type": surface_type,
        "renderer": renderer,
        "question": question,
        "initial_order": list(initial),
        "program": [operation.as_dict() for operation in program],
        "query": {"kind": "read_position", "position": int(query_position)},
        "halt": "stop",
        "terminal_order": list(terminal_a),
        "answer": terminal_a[query_position],
        "spans": token_targets,
        "token_count": len(encoding.ids),
        "token_ids_sha256": sha256_bytes(canonical_json(encoding.ids).encode()),
        "token_bag": sorted(collections.Counter(encoding.ids).items()),
    }


def build_split(split, groups, seed, tokenizer, name_pool):
    if split not in SPLIT_RENDERERS:
        raise ValueError("unknown split {}".format(split))
    if groups <= 0 or len(name_pool) < 3:
        raise ValueError("invalid groups or name pool")
    rng = random.Random(seed)
    configurations = valid_configurations()
    primary_renderer, paraphrase_renderer = SPLIT_RENDERERS[split]
    rows = []
    seen_questions = set()
    attempts = 0
    while len(rows) < groups * 4:
        attempts += 1
        if attempts > groups * 100:
            raise RuntimeError("could not construct enough unique groups")
        group_index = len(rows) // 4
        configuration = rng.choice(configurations)
        entities = tuple(rng.sample(name_pool, 3))
        initial = tuple(entities[index] for index in configuration["initial_indices"])
        base = instantiate(configuration["base"], entities)
        order = instantiate(configuration["order"], entities)
        binding = instantiate(configuration["binding"], entities)
        query_position = configuration["query_position"]
        distractor_slot = rng.randrange(3)
        marker = rng.choice(SPLIT_MARKERS[split])
        distractor_entity = initial[(query_position + 1) % 3]
        distractor_amount = base[0].amount
        surfaces = {
            "canonical": (base, primary_renderer),
            "paraphrase": (base, paraphrase_renderer),
            "order_twin": (order, primary_renderer),
            "binding_twin": (binding, primary_renderer),
        }
        candidate = [
            make_row(
                split, group_index, surface_type, initial, surfaces[surface_type][0],
                query_position, surfaces[surface_type][1], distractor_slot, marker,
                distractor_entity, distractor_amount, tokenizer,
            )
            for surface_type in SURFACE_TYPES
        ]
        normalized_questions = [normalized(row["question"]) for row in candidate]
        if len(set(normalized_questions)) != 4:
            continue
        if any(question in seen_questions for question in normalized_questions):
            continue
        canonical, paraphrase, order_twin, binding_twin = candidate
        if row_label(canonical) != row_label(paraphrase):
            raise ValueError("paraphrase program drift")
        if canonical["terminal_order"] != paraphrase["terminal_order"]:
            raise ValueError("paraphrase behavior drift")
        if canonical["answer"] in {order_twin["answer"], binding_twin["answer"]}:
            raise ValueError("frozen query is not a twin separator")
        if not canonical["token_bag"] == order_twin["token_bag"] == binding_twin["token_bag"]:
            raise ValueError("matched token bags diverged")
        rows.extend(candidate)
        seen_questions.update(normalized_questions)
    return rows


def shortcut_ceiling(rows, feature):
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        groups[feature(row)][row_label(row)] += 1
    correct = sum(max(labels.values()) for labels in groups.values())
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def pointer_positions(row):
    labels = (
        "op0.kind", "op0.entity", "op0.literal",
        "op1.kind", "op1.entity", "op1.literal", "query.position",
    )
    return canonical_json([row["spans"][label]["token_positions"] for label in labels])


def audit_splits(split_rows, tokenizer_path, generator_path):
    expected_labels = {
        "op0.kind", "op0.entity", "op0.literal",
        "op1.kind", "op1.entity", "op1.literal", "query.position",
    }
    split_names = {
        split: {name for row in rows for name in row["initial_order"]}
        for split, rows in split_rows.items()
    }
    split_renderers = {
        split: {row["renderer"] for row in rows} for split, rows in split_rows.items()
    }
    pairwise = {}
    for left, right in itertools.combinations(split_rows, 2):
        left_questions = {normalized(row["question"]) for row in split_rows[left]}
        right_questions = {normalized(row["question"]) for row in split_rows[right]}
        left_ngrams = set().union(*(ngrams(row["question"]) for row in split_rows[left]))
        right_ngrams = set().union(*(ngrams(row["question"]) for row in split_rows[right]))
        pairwise["{}__{}".format(left, right)] = {
            "exact_prompt_overlap": len(left_questions & right_questions),
            "word_13gram_overlap": len(left_ngrams & right_ngrams),
            "entity_name_overlap": len(split_names[left] & split_names[right]),
            "renderer_overlap": len(split_renderers[left] & split_renderers[right]),
        }
    per_split = {}
    all_rows = [row for rows in split_rows.values() for row in rows]
    for split, rows in split_rows.items():
        grouped = collections.defaultdict(dict)
        for row in rows:
            grouped[row["group"]][row["surface_type"]] = row
        matched = [row for row in rows if row["surface_type"] != "paraphrase"]
        group_pass = 0
        for group in grouped.values():
            canonical = group["canonical"]
            paraphrase = group["paraphrase"]
            order_twin = group["order_twin"]
            binding_twin = group["binding_twin"]
            passed = (
                row_label(canonical) == row_label(paraphrase)
                and canonical["terminal_order"] == paraphrase["terminal_order"]
                and canonical["answer"] != order_twin["answer"]
                and canonical["answer"] != binding_twin["answer"]
                and canonical["token_bag"] == order_twin["token_bag"]
                and canonical["token_bag"] == binding_twin["token_bag"]
            )
            group_pass += int(passed)
        ceilings = {
            "token_bag": shortcut_ceiling(matched, lambda row: canonical_json(row["token_bag"])),
            "absolute_pointer_positions": shortcut_ceiling(matched, pointer_positions),
            "source_token_length": shortcut_ceiling(matched, lambda row: str(row["token_count"])),
            "renderer": shortcut_ceiling(matched, lambda row: row["renderer"]),
        }
        per_split[split] = {
            "rows": len(rows),
            "groups": len(grouped),
            "group_gates_passed": group_pass,
            "duplicate_questions": len(rows) - len({normalized(row["question"]) for row in rows}),
            "renderers": sorted(split_renderers[split]),
            "entity_names": len(split_names[split]),
            "shortcut_ceilings": ceilings,
            "operation_kinds": dict(sorted(collections.Counter(
                operation["kind"] for row in rows for operation in row["program"]
            ).items())),
            "amounts": dict(sorted(collections.Counter(
                str(operation["amount"]) for row in rows for operation in row["program"]
            ).items())),
            "queries": dict(sorted(collections.Counter(
                str(row["query"]["position"]) for row in rows
            ).items())),
        }
    structural = {
        "all_ids_unique": len({row["id"] for row in all_rows}) == len(all_rows),
        "all_spans_present": all(expected_labels.issubset(row["spans"]) for row in all_rows),
        "all_spans_nonempty": all(
            row["spans"][label]["token_positions"]
            for row in all_rows for label in expected_labels
        ),
        "all_group_gates_pass": all(
            report["groups"] == report["group_gates_passed"] for report in per_split.values()
        ),
        "all_shortcut_ceilings_at_most_one_third": all(
            ceiling["accuracy"] <= 1.0 / 3.0 + 1e-12
            for report in per_split.values()
            for ceiling in report["shortcut_ceilings"].values()
        ),
        "cross_split_exact_overlap_zero": all(
            report["exact_prompt_overlap"] == 0 for report in pairwise.values()
        ),
        "cross_split_13gram_overlap_zero": all(
            report["word_13gram_overlap"] == 0 for report in pairwise.values()
        ),
        "cross_split_entity_overlap_zero": all(
            report["entity_name_overlap"] == 0 for report in pairwise.values()
        ),
        "cross_split_renderer_overlap_zero": all(
            report["renderer_overlap"] == 0 for report in pairwise.values()
        ),
    }
    return {
        "schema": "r12_referential_literal_pointer_corpus_v1",
        "claim_boundary": (
            "Frozen synthetic semantic-compiler supervision. Structured programs and answers "
            "are labels/audit data only and are forbidden at inference."
        ),
        "structural_gates": structural,
        "all_gates_pass": all(structural.values()),
        "pairwise_split_audits": pairwise,
        "splits": per_split,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "generator_sha256": sha256_file(generator_path),
        "acquisition_ledger": {
            "rows": len(all_rows),
            "source_utf8_bytes": sum(len(row["question"].encode()) for row in all_rows),
            "source_tokens": sum(row["token_count"] for row in all_rows),
            "target_pointer_labels": len(all_rows) * len(expected_labels),
            "teacher_model_calls": 0,
            "checkpoint_reads": 0,
            "production_eval_answer_reads": 0,
            "external_execution": "two exact CPU list-machine evaluators during generation",
        },
    }


def write_jsonl(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    Path(path).write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-groups", type=int, default=24000)
    parser.add_argument("--development-groups", type=int, default=512)
    parser.add_argument("--confirmation-groups", type=int, default=1024)
    parser.add_argument("--train-seed", type=int, default=2026071801)
    parser.add_argument("--development-seed", type=int, default=2026071802)
    parser.add_argument("--confirmation-seed", type=int, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    targets = {
        split: out_dir / "{}.jsonl" for split in SPLIT_RENDERERS
    }
    report_path = out_dir / "report.json"
    if any(path.exists() for path in (*targets.values(), report_path)):
        raise SystemExit("refusing to overwrite corpus output")
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    required_names = 4100
    names, nonce_width = candidate_names(tokenizer, required_names)
    name_pools = {
        "train": names[:3100],
        "development": names[3100:3500],
        "confirmation": names[3500:4100],
    }
    seeds = {
        "train": args.train_seed,
        "development": args.development_seed,
        "confirmation": args.confirmation_seed,
    }
    counts = {
        "train": args.train_groups,
        "development": args.development_groups,
        "confirmation": args.confirmation_groups,
    }
    split_rows = {
        split: build_split(split, counts[split], seeds[split], tokenizer, name_pools[split])
        for split in SPLIT_RENDERERS
    }
    report = audit_splits(split_rows, tokenizer_path, Path(__file__))
    report["seeds"] = seeds
    report["groups"] = counts
    report["nonce_token_width"] = nonce_width
    out_dir.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {
        split: write_jsonl(targets[split], rows) for split, rows in split_rows.items()
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": report["all_gates_pass"],
        "artifacts": report["artifacts"],
        "report": str(report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("corpus audit failed")


if __name__ == "__main__":
    main()
