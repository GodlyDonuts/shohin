#!/usr/bin/env python3
"""Build public S4 whole-source variable-depth event-tape data."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import (
    ARGUMENT_ORDERS,
    DISTRACTOR_FRAMES,
    INTRO_FRAMES,
    LIST_STYLES,
    QUERY_FRAMES,
    STYLE_FACTORS,
    direction_pairs,
    expected_factor_atoms,
    factor_atoms,
    factor_signature,
    normalized,
    ngrams,
    split_factor_candidates,
    styled,
)
from semantic_compiler_falsifier import (
    AMOUNTS,
    DIRECTIONS,
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


TRAIN_SPLIT = "s4_event_tape_train"
DEVELOPMENT_SPLIT = "s4_event_tape_development"
SURFACES = ("canonical", "paraphrase", "order_twin", "binding_twin")
EVENT_FRAMES = (
    "Apply this roster update: ",
    "Movement command: ",
    "Modify the lineup as follows: ",
    "Execute this positional change: ",
)
GENERIC_ROLES = (
    "intro.entity0",
    "intro.entity1",
    "intro.entity2",
    "event.kind",
    "event.entity",
    "event.literal",
    "query.position",
)


def source_rows(row):
    if row.get("question"):
        yield row
    yield from row.get("chunks", ())


def read_public(paths):
    questions = set()
    grams = set()
    names = set()
    factors = set()
    rows = 0
    for path in paths:
        if "confirmation" in Path(path).name.lower():
            raise ValueError("confirmation input is forbidden")
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                names.update(row.get("initial_order", ()))
                if row.get("neutral_anchor"):
                    names.add(row["neutral_anchor"])
                for item in source_rows(row):
                    questions.add(normalized(item["question"]))
                    grams.update(ngrams(item["question"]))
                    if item.get("factor_signature"):
                        factors.add(item["factor_signature"])
                    names.update(item.get("initial_order", ()))
                    if item.get("neutral_anchor"):
                        names.add(item["neutral_anchor"])
    return {
        "rows": rows,
        "questions": questions,
        "grams": grams,
        "names": names,
        "factors": factors,
    }


def fresh_paired_names(tokenizer, count, seed, excluded):
    atoms, _ = candidate_names(tokenizer, 4100)
    rng = random.Random(seed)
    selected = []
    seen = set(excluded)
    while len(selected) < count:
        left, right = rng.sample(atoms, 2)
        name = "{}-{}".format(left, right)
        if name in seen or not tokenizer.encode(name).ids:
            continue
        seen.add(name)
        selected.append(name)
    return selected


def select_factors(split, count, seed, excluded):
    candidates = [
        factors for factors in split_factor_candidates(split)
        if factor_signature(factors) not in excluded
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    selected_signatures = set()
    uncovered = expected_factor_atoms("known")
    for factors in candidates:
        if factor_atoms(factors) & uncovered:
            selected.append(factors)
            selected_signatures.add(factor_signature(factors))
            uncovered -= factor_atoms(factors)
        if not uncovered:
            break
    if uncovered:
        raise RuntimeError("fresh factors do not cover known atoms: {}".format(sorted(uncovered)))
    for factors in candidates:
        signature = factor_signature(factors)
        if signature in selected_signatures:
            continue
        selected.append(factors)
        selected_signatures.add(signature)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("could not select {} fresh factor combinations".format(count))
    rng.shuffle(selected)
    return selected


def add_intro(writer, initial, factors):
    prefix, _ = INTRO_FRAMES[factors["intro_frame"]]
    separator, final_separator = LIST_STYLES[factors["list_style"]]
    _, lower_lead, sentence_end, _ = STYLE_FACTORS[factors["style"]]
    writer.add(styled(prefix, lower_lead))
    for index, entity in enumerate(initial):
        if index:
            writer.add(final_separator if index == 2 else separator)
        writer.add(entity, "intro.entity{}".format(index))
    writer.add(sentence_end)


def add_event(writer, operation, index, factors):
    _, lower_lead, sentence_end, _ = STYLE_FACTORS[factors["style"]]
    direction_pair = direction_pairs(factors["lexicon"])[factors["direction_pair"]]
    direction_text = direction_pair[0 if operation.direction == "left" else 1]
    writer.add(styled(EVENT_FRAMES[factors["op_frame"]], lower_lead))
    order = ARGUMENT_ORDERS[factors["argument_order"]]
    if order == "entity_kind_literal":
        writer.add("move ")
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" ")
        writer.add(direction_text, "op{}.kind".format(index))
        writer.add(" by ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" positions")
    elif order == "kind_entity_literal":
        writer.add("send ")
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" by ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" positions toward ")
        writer.add(direction_text, "op{}.kind".format(index))
    elif order == "literal_entity_kind":
        writer.add("for ")
        writer.add(str(operation.amount), "op{}.literal".format(index))
        writer.add(" positions, shift ")
        writer.add(operation.entity, "op{}.entity".format(index))
        writer.add(" ")
        writer.add(direction_text, "op{}.kind".format(index))
    else:
        raise ValueError("unknown argument order")
    writer.add(sentence_end)


def distractor_boundary(depth, category):
    if int(category) == 0:
        return 0
    if int(category) == 1:
        return (int(depth) + 1) // 2
    if int(category) == 2:
        return int(depth)
    raise ValueError("invalid distractor category")


def render_source(initial, program, query_position, factors, distractor_entity,
                  distractor_amount, neutral_anchor):
    writer = SpanWriter()
    add_intro(writer, initial, factors)
    writer.add("Context anchor {}.\n".format(neutral_anchor))
    boundary = distractor_boundary(len(program), factors["distractor_location"])
    distractor = DISTRACTOR_FRAMES[factors["distractor_frame"]].format(
        entity=distractor_entity, amount=distractor_amount,
    )
    _, lower_lead, _, query_end = STYLE_FACTORS[factors["style"]]
    if boundary == 0:
        writer.add(styled(distractor, lower_lead))
    for index, operation in enumerate(program):
        add_event(writer, operation, index, factors)
        writer.add("Context anchor {}.\n".format(neutral_anchor))
        if boundary == index + 1:
            writer.add(styled(distractor, lower_lead))
    writer.add(styled(QUERY_FRAMES[factors["query_frame"]], lower_lead))
    writer.add(str(query_position + 1), "query.position")
    writer.add(query_end)
    return writer.finish()


def gold_events(row):
    events = []
    previous_end = -1
    for index, operation in enumerate(row["program"]):
        labels = ["op{}.{}".format(index, role) for role in ("kind", "entity", "literal")]
        targets = [row["spans"][label] for label in labels]
        start = min(min(target["token_positions"]) for target in targets)
        end = max(max(target["token_positions"]) for target in targets)
        if start <= previous_end:
            raise ValueError("event spans are not monotonically ordered")
        previous_end = end
        events.append({
            "kind": operation["kind"],
            "entity": operation["entity"],
            "amount": int(operation["amount"]),
            "start_token": start,
            "end_token": end,
        })
    return events


def make_row(split, row_id, group, surface, initial, program, query_position, factors,
             distractor_entity, distractor_amount, neutral_anchor, tokenizer):
    question, spans = render_source(
        initial, program, query_position, factors, distractor_entity, distractor_amount,
        neutral_anchor,
    )
    encoding, token_targets = attach_token_targets(question, spans, tokenizer)
    terminal_a = apply_program_pop_insert(initial, program)
    terminal_b = apply_program_adjacent_swaps(initial, program)
    if terminal_a != terminal_b:
        raise ValueError("independent executors disagree")
    row = {
        "id": row_id,
        "schema": "r12_s4_self_delimiting_event_tape_row_v1",
        "split": split,
        "group": int(group),
        "surface_type": surface,
        "renderer": "s4_whole_source_unpadded",
        "factors": dict(factors),
        "factor_signature": factor_signature(factors),
        "question": question,
        "neutral_anchor": neutral_anchor,
        "initial_order": list(initial),
        "program": [operation.as_dict() for operation in program],
        "depth": len(program),
        "query": {"kind": "read_position", "position": int(query_position)},
        "halt": "end_of_complete_event_tape",
        "terminal_order": list(terminal_a),
        "answer": terminal_a[query_position],
        "spans": token_targets,
        "token_count": len(encoding.ids),
        "token_ids_sha256": sha256_bytes(canonical_json(encoding.ids).encode()),
        "token_bag": sorted(collections.Counter(encoding.ids).items()),
        "executor_agreement": True,
    }
    row["gold_events"] = gold_events(row)
    return row


def random_program(rng, initial, depth):
    choices = tuple(itertools.product(DIRECTIONS, initial, AMOUNTS))
    return tuple(Operation(*rng.choice(choices)) for _ in range(depth))


def choose_twins(rng, initial, depth):
    choices = tuple(itertools.product(DIRECTIONS, range(3), AMOUNTS))
    for _ in range(10000):
        base = tuple(rng.choice(choices) for _ in range(depth))
        shift = rng.randrange(1, depth)
        order = base[shift:] + base[:shift]
        slots = [operation[1] for operation in base]
        slot_shift = rng.randrange(1, depth)
        rotated = slots[slot_shift:] + slots[:slot_shift]
        binding = tuple(
            (operation[0], rotated[index], operation[2])
            for index, operation in enumerate(base)
        )
        if len({base, order, binding}) != 3:
            continue
        programs = [
            tuple(Operation(kind, initial[slot], amount) for kind, slot, amount in spec)
            for spec in (base, order, binding)
        ]
        states = [apply_program_pop_insert(initial, program) for program in programs]
        separators = [
            position for position in range(3)
            if states[0][position] != states[1][position]
            and states[0][position] != states[2][position]
        ]
        if separators:
            return programs, rng.choice(separators)
    raise RuntimeError("could not sample depth-{} matched twins".format(depth))


def build_train(count, seed, tokenizer, names, factors):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        depth = 1 + index % 4
        initial = tuple(rng.sample(names, 3))
        neutral_anchor = rng.choice([name for name in names if name not in initial])
        program = random_program(rng, initial, depth)
        query = rng.randrange(3)
        rows.append(make_row(
            TRAIN_SPLIT, "S4-TRAIN-{:06d}".format(index), index, "train", initial,
            program, query, factors[index], initial[(query + 1) % 3],
            program[0].amount, neutral_anchor, tokenizer,
        ))
    return rows


def build_development(groups, seed, tokenizer, names, factors):
    rng = random.Random(seed)
    rows = []
    for group in range(groups):
        depth = 3 + group % 6
        initial = tuple(rng.sample(names, 3))
        neutral_anchor = rng.choice([name for name in names if name not in initial])
        programs, query = choose_twins(rng, initial, depth)
        canonical_factors = factors[group * 2]
        paraphrase_factors = factors[group * 2 + 1]
        definitions = {
            "canonical": (programs[0], canonical_factors),
            "paraphrase": (programs[0], paraphrase_factors),
            "order_twin": (programs[1], canonical_factors),
            "binding_twin": (programs[2], canonical_factors),
        }
        candidate = []
        for surface in SURFACES:
            program, selected_factors = definitions[surface]
            candidate.append(make_row(
                DEVELOPMENT_SPLIT,
                "S4-DEV-{:06d}-{}".format(group, surface),
                group,
                surface,
                initial,
                program,
                query,
                selected_factors,
                initial[(query + 1) % 3],
                programs[0][0].amount,
                neutral_anchor,
                tokenizer,
            ))
        canonical, paraphrase, order_twin, binding_twin = candidate
        if canonical["program"] != paraphrase["program"]:
            raise ValueError("paraphrase semantic drift")
        if canonical["answer"] in {order_twin["answer"], binding_twin["answer"]}:
            raise ValueError("twins do not separate the query")
        if not canonical["token_bag"] == order_twin["token_bag"] == binding_twin["token_bag"]:
            raise ValueError("matched twin token bags diverged")
        rows.extend(candidate)
    return rows


def audit(train, development, public, tokenizer_path, generator_path):
    all_rows = train + development
    train_questions = {normalized(row["question"]) for row in train}
    development_questions = {normalized(row["question"]) for row in development}
    train_grams = set().union(*(ngrams(row["question"]) for row in train))
    development_grams = set().union(*(ngrams(row["question"]) for row in development))
    train_names = {
        name for row in train for name in (*row["initial_order"], row["neutral_anchor"])
    }
    development_names = {
        name for row in development
        for name in (*row["initial_order"], row["neutral_anchor"])
    }
    train_factors = {row["factor_signature"] for row in train}
    development_factors = {row["factor_signature"] for row in development}
    depth_counts = {
        split: dict(sorted(collections.Counter(row["depth"] for row in rows).items()))
        for split, rows in (("train", train), ("development", development))
    }
    gates = {
        "all_ids_unique": len({row["id"] for row in all_rows}) == len(all_rows),
        "no_external_active_operation_field": all(
            "active_operations" not in row for row in all_rows
        ),
        "all_unpadded_whole_source": all(
            row["renderer"] == "s4_whole_source_unpadded" for row in all_rows
        ),
        "event_count_equals_depth": all(
            len(row["gold_events"]) == row["depth"] == len(row["program"])
            for row in all_rows
        ),
        "all_event_spans_present": all(
            all(
                "op{}.{}".format(index, role) in row["spans"]
                for index in range(row["depth"])
                for role in ("kind", "entity", "literal")
            )
            for row in all_rows
        ),
        "all_event_spans_nonempty": all(
            target["token_positions"]
            for row in all_rows for target in row["spans"].values()
        ),
        "two_cpu_executors_agree": all(row["executor_agreement"] for row in all_rows),
        "all_sources_fit_2048": max(row["token_count"] for row in all_rows) <= 2048,
        "train_depths_one_through_four": set(depth_counts["train"]) == set(range(1, 5)),
        "development_depths_three_through_eight": (
            set(depth_counts["development"]) == set(range(3, 9))
        ),
        "train_development_exact_overlap_zero": not (train_questions & development_questions),
        "train_development_13gram_overlap_zero": not (train_grams & development_grams),
        "train_development_name_overlap_zero": not (train_names & development_names),
        "train_development_factor_overlap_zero": not (train_factors & development_factors),
        "public_exact_overlap_zero": not ((train_questions | development_questions) & public["questions"]),
        "public_13gram_overlap_zero": not ((train_grams | development_grams) & public["grams"]),
        "public_name_overlap_zero": not ((train_names | development_names) & public["names"]),
        "public_factor_overlap_zero": not ((train_factors | development_factors) & public["factors"]),
        "confirmation_access_zero": True,
    }
    return {
        "schema": "r12_s4_self_delimiting_event_tape_corpus_report_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "splits": {
            "train": {
                "rows": len(train),
                "depth_counts": depth_counts["train"],
                "source_tokens": sum(row["token_count"] for row in train),
            },
            "development": {
                "rows": len(development),
                "depth_counts": depth_counts["development"],
                "source_tokens": sum(row["token_count"] for row in development),
            },
        },
        "maximum_source_tokens": max(row["token_count"] for row in all_rows),
        "generic_roles": list(GENERIC_ROLES),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "generator_sha256": sha256_file(generator_path),
        "public_rows_audited": public["rows"],
        "confirmation_access": 0,
        "claim_boundary": (
            "Public self-delimiting parser development only. Programs and spans are "
            "supervision/audit labels and are forbidden at inference."
        ),
    }


def write_jsonl(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    Path(path).write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-rows", type=int, default=48000)
    parser.add_argument("--development-groups", type=int, default=512)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--development-seed", type=int, required=True)
    args = parser.parse_args()
    if args.train_rows < 48 or args.development_groups < 24:
        raise SystemExit("insufficient S4 corpus size")
    out_dir = Path(args.out_dir)
    paths = {
        "train": out_dir / "train.jsonl",
        "development": out_dir / "development.jsonl",
        "report": out_dir / "report.json",
    }
    if any(path.exists() for path in paths.values()):
        raise SystemExit("refusing existing S4 output")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    public = read_public(args.public_data)
    names = fresh_paired_names(
        tokenizer, 3600, args.train_seed ^ args.development_seed, public["names"],
    )
    train_factors = select_factors(
        "train", args.train_rows, args.train_seed ^ 0xA51CE, public["factors"],
    )
    development_factors = select_factors(
        "development_compositional", args.development_groups * 2,
        args.development_seed ^ 0xD3E10, public["factors"] | {
            factor_signature(factors) for factors in train_factors
        },
    )
    train = build_train(
        args.train_rows, args.train_seed, tokenizer, names[:2800], train_factors,
    )
    development = build_development(
        args.development_groups, args.development_seed, tokenizer, names[2800:],
        development_factors,
    )
    report = audit(train, development, public, args.tokenizer, __file__)
    report["seeds"] = {
        "train": args.train_seed,
        "development": args.development_seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {
        "train": write_jsonl(paths["train"], train),
        "development": write_jsonl(paths["development"], development),
    }
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": report["all_gates_pass"],
        "artifacts": report["artifacts"],
        "maximum_source_tokens": report["maximum_source_tokens"],
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("S4 corpus gates failed")


if __name__ == "__main__":
    main()
