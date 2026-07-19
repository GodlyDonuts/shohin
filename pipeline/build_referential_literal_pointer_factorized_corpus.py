#!/usr/bin/env python3
"""Build the frozen factorized-language complete-compiler corpus.

All source roles are model-owned. Structured programs are used only to create
gold pointer labels and to audit the two exact CPU executors. Production seeds
have no defaults and must be chosen only after this source is committed.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
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
SPLITS = (
    "train",
    "development_compositional",
    "development_lexical_ood",
    "confirmation",
)
REQUIRED_SPANS = (
    "intro.entity0",
    "intro.entity1",
    "intro.entity2",
    "op0.kind",
    "op0.entity",
    "op0.literal",
    "op1.kind",
    "op1.entity",
    "op1.literal",
    "query.position",
)

INTRO_FRAMES = (
    ("The opening order is ", ".\n"),
    ("Begin with this sequence: ", ".\n"),
    ("The current lineup reads ", ".\n"),
    ("Initially arrange the names as ", ".\n"),
)
LIST_STYLES = (
    (", ", ", and "),
    (" / ", " / "),
    (" then ", " then "),
)
ORDINAL_VOCABS = (
    ("one", "two"),
    ("first", "second"),
    ("alpha", "beta"),
    ("I", "II"),
)
OP_FRAMES = (
    ("Instruction {ordinal}: ",),
    ("Step {ordinal} says to ",),
    ("For update {ordinal}, ",),
    ("Change {ordinal}: ",),
)
ARGUMENT_ORDERS = (
    "entity_kind_literal",
    "kind_entity_literal",
    "literal_entity_kind",
)
KNOWN_DIRECTION_PAIRS = (
    ("left", "right"),
    ("earlier", "later"),
    ("frontward", "rearward"),
    ("beginning-ward", "end-ward"),
    ("headward", "tailward"),
    ("lower-index", "higher-index"),
)
LEXICAL_OOD_DIRECTION_PAIRS = (
    ("preceding-side", "following-side"),
    ("start-facing", "finish-facing"),
    ("front-side", "back-side"),
    ("minus-index", "plus-index"),
)
DISTRACTOR_FRAMES = (
    "Ignore the badge pairing {entity} with {amount}; it changes no position.\n",
    "A margin note links {entity} to {amount}, but it is not an update.\n",
    "Decoration only: {entity} carries tag {amount}; keep the order unchanged.\n",
    "The label {amount} beside {entity} has no effect on the sequence.\n",
)
QUERY_FRAMES = (
    "Which name finally occupies position ",
    "Return the name now found in slot ",
    "Read the entry at final place ",
    "Identify who ends at sequence location ",
)
STYLE_FACTORS = (
    ("standard_period", False, ".\n", "?\nProgram:"),
    ("lower_semicolon", True, ";\n", "; answer with Program:"),
)
FACTOR_FIELDS = (
    "lexicon",
    "intro_frame",
    "list_style",
    "ordinal_vocab",
    "op_frame",
    "argument_order",
    "direction_pair",
    "distractor_frame",
    "distractor_location",
    "query_frame",
    "style",
)


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


@functools.lru_cache(maxsize=1)
def valid_configurations():
    operations = list(itertools.product(DIRECTIONS, range(3), AMOUNTS))
    symbolic = ("E0", "E1", "E2")
    configurations = []
    for base in itertools.product(operations, repeat=2):
        first, second = base
        order = (second, first)
        binding = (
            (first[0], second[1], second[2]),
            (second[0], first[1], first[2]),
        )
        if len({base, order, binding}) != 3:
            continue
        programs = [instantiate(spec, symbolic) for spec in (base, order, binding)]
        for initial_indices in itertools.permutations(range(3)):
            initial = tuple(symbolic[index] for index in initial_indices)
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


def direction_pairs(lexicon):
    if lexicon == "known":
        return KNOWN_DIRECTION_PAIRS
    if lexicon == "lexical_ood":
        return LEXICAL_OOD_DIRECTION_PAIRS
    raise ValueError("unknown lexicon {}".format(lexicon))


def factor_signature(factors):
    return canonical_json({field: factors[field] for field in FACTOR_FIELDS})


def combination_partition(factors):
    digest = hashlib.sha256(factor_signature(factors).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 16


@functools.lru_cache(maxsize=2)
def factor_catalogue(lexicon):
    directions = direction_pairs(lexicon)
    catalogue = []
    dimensions = (
        range(len(INTRO_FRAMES)),
        range(len(LIST_STYLES)),
        range(len(ORDINAL_VOCABS)),
        range(len(OP_FRAMES)),
        range(len(ARGUMENT_ORDERS)),
        range(len(directions)),
        range(len(DISTRACTOR_FRAMES)),
        range(3),
        range(len(QUERY_FRAMES)),
        range(len(STYLE_FACTORS)),
    )
    for values in itertools.product(*dimensions):
        factors = dict(zip(FACTOR_FIELDS[1:], values))
        factors["lexicon"] = lexicon
        catalogue.append(factors)
    return tuple(catalogue)


def split_factor_candidates(split):
    if split == "train":
        return tuple(
            factors for factors in factor_catalogue("known")
            if combination_partition(factors) < 12
        )
    if split == "development_compositional":
        return tuple(
            factors for factors in factor_catalogue("known")
            if combination_partition(factors) in {12, 13}
        )
    if split == "confirmation":
        return tuple(
            factors for factors in factor_catalogue("known")
            if combination_partition(factors) in {14, 15}
        )
    if split == "development_lexical_ood":
        return factor_catalogue("lexical_ood")
    raise ValueError("unknown split {}".format(split))


def factor_atoms(factors):
    return {(field, str(factors[field])) for field in FACTOR_FIELDS}


def expected_factor_atoms(lexicon):
    catalogue = factor_catalogue(lexicon)
    return set().union(*(factor_atoms(factors) for factors in catalogue))


def select_factor_specs(split, count, seed):
    candidates = list(split_factor_candidates(split))
    if count > len(candidates):
        raise ValueError("{} requests {} factor combinations from {}".format(
            split, count, len(candidates),
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    selected_signatures = set()
    if split == "train":
        uncovered = expected_factor_atoms("known")
        for factors in candidates:
            if factor_atoms(factors) & uncovered:
                selected.append(factors)
                selected_signatures.add(factor_signature(factors))
                uncovered -= factor_atoms(factors)
                if not uncovered:
                    break
        if uncovered:
            raise RuntimeError("training factor coverage failed: {}".format(sorted(uncovered)))
    for factors in candidates:
        signature = factor_signature(factors)
        if signature in selected_signatures:
            continue
        selected.append(factors)
        selected_signatures.add(signature)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("could not select {} unique factor combinations".format(count))
    rng.shuffle(selected)
    return selected


def styled(text, lower_lead):
    if not lower_lead or not text:
        return text
    return text[0].lower() + text[1:]


def add_intro(writer, initial, factors):
    prefix, _ = INTRO_FRAMES[factors["intro_frame"]]
    separator, final_separator = LIST_STYLES[factors["list_style"]]
    _, lower_lead, sentence_end, _ = STYLE_FACTORS[factors["style"]]
    writer.add(styled(prefix, lower_lead))
    for index, entity in enumerate(initial):
        if index:
            writer.add(final_separator if index == len(initial) - 1 else separator)
        writer.add(entity, "intro.entity{}".format(index))
    writer.add(sentence_end)


def add_operation(writer, operation, index, factors):
    ordinal = ORDINAL_VOCABS[factors["ordinal_vocab"]][index]
    prefix = OP_FRAMES[factors["op_frame"]][0].format(ordinal=ordinal)
    _, lower_lead, sentence_end, _ = STYLE_FACTORS[factors["style"]]
    direction_pair = direction_pairs(factors["lexicon"])[factors["direction_pair"]]
    direction_text = direction_pair[0 if operation.direction == "left" else 1]
    writer.add(styled(prefix, lower_lead))
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
        raise ValueError("unknown argument order {}".format(order))
    writer.add(sentence_end)


def add_distractor(writer, entity, amount, factors):
    template = DISTRACTOR_FRAMES[factors["distractor_frame"]]
    _, lower_lead, _, _ = STYLE_FACTORS[factors["style"]]
    writer.add(styled(template.format(entity=entity, amount=amount), lower_lead))


def add_query(writer, query_position, factors):
    prefix = QUERY_FRAMES[factors["query_frame"]]
    _, lower_lead, _, query_end = STYLE_FACTORS[factors["style"]]
    writer.add(styled(prefix, lower_lead))
    writer.add(str(query_position + 1), "query.position")
    writer.add(query_end)


def add_anchor(writer, neutral_anchor):
    writer.add("Context anchor ")
    writer.add(neutral_anchor)
    writer.add(".\n")


def render_source(initial, program, query_position, factors, distractor_entity,
                  distractor_amount, neutral_anchor):
    writer = SpanWriter()
    add_intro(writer, initial, factors)
    add_anchor(writer, neutral_anchor)
    location = factors["distractor_location"]
    if location == 0:
        add_distractor(writer, distractor_entity, distractor_amount, factors)
        add_anchor(writer, neutral_anchor)
    for index, operation in enumerate(program):
        add_operation(writer, operation, index, factors)
        add_anchor(writer, neutral_anchor)
        if location == index + 1:
            add_distractor(writer, distractor_entity, distractor_amount, factors)
            add_anchor(writer, neutral_anchor)
    add_query(writer, query_position, factors)
    return writer.finish()


def row_label(row):
    return canonical_json({
        "operations": row["program"],
        "query": row["query"],
        "halt": row["halt"],
    })


def make_row(split, group_index, surface_type, initial, program, query_position,
             factors, distractor_entity, distractor_amount, neutral_anchor, tokenizer):
    question, spans = render_source(
        initial, program, query_position, factors, distractor_entity, distractor_amount,
        neutral_anchor,
    )
    encoding, token_targets = attach_token_targets(question, spans, tokenizer)
    terminal_a = apply_program_pop_insert(initial, program)
    terminal_b = apply_program_adjacent_swaps(initial, program)
    if terminal_a != terminal_b:
        raise ValueError("executor disagreement")
    return {
        "id": "RLPCF-{}-{:06d}-{}".format(split, group_index, surface_type),
        "schema": "r12_referential_literal_pointer_factorized_row_v1",
        "split": split,
        "group": int(group_index),
        "surface_type": surface_type,
        "renderer": "factorized",
        "factors": dict(factors),
        "factor_signature": factor_signature(factors),
        "question": question,
        "neutral_anchor": neutral_anchor,
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
        "executor_agreement": True,
    }


def build_split(split, groups, seed, tokenizer, name_pool):
    if split not in SPLITS:
        raise ValueError("unknown split {}".format(split))
    if groups <= 0 or len(name_pool) < 3:
        raise ValueError("invalid groups or name pool")
    rng = random.Random(seed)
    configurations = valid_configurations()
    factor_specs = select_factor_specs(split, groups * 2, seed ^ 0x5F3759DF)
    rows = []
    seen_questions = set()
    for group_index in range(groups):
        configuration = rng.choice(configurations)
        entities_and_anchor = tuple(rng.sample(name_pool, 4))
        entities = entities_and_anchor[:3]
        neutral_anchor = entities_and_anchor[3]
        initial = tuple(entities[index] for index in configuration["initial_indices"])
        base = instantiate(configuration["base"], entities)
        order = instantiate(configuration["order"], entities)
        binding = instantiate(configuration["binding"], entities)
        query_position = configuration["query_position"]
        distractor_entity = initial[(query_position + 1) % 3]
        distractor_amount = base[0].amount
        canonical_factors = factor_specs[group_index * 2]
        paraphrase_factors = factor_specs[group_index * 2 + 1]
        surfaces = {
            "canonical": (base, canonical_factors),
            "paraphrase": (base, paraphrase_factors),
            "order_twin": (order, canonical_factors),
            "binding_twin": (binding, canonical_factors),
        }
        candidate = [
            make_row(
                split, group_index, surface_type, initial, surfaces[surface_type][0],
                query_position, surfaces[surface_type][1], distractor_entity,
                distractor_amount, neutral_anchor, tokenizer,
            )
            for surface_type in SURFACE_TYPES
        ]
        questions = [normalized(row["question"]) for row in candidate]
        if len(set(questions)) != 4 or any(question in seen_questions for question in questions):
            raise RuntimeError("factorized source collision in {} group {}".format(
                split, group_index,
            ))
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
        seen_questions.update(questions)
    return rows


def shortcut_ceiling(rows, feature):
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        groups[feature(row)][row_label(row)] += 1
    correct = sum(max(labels.values()) for labels in groups.values())
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def pointer_positions(row):
    labels = REQUIRED_SPANS[3:]
    return canonical_json([row["spans"][label]["token_positions"] for label in labels])


def split_factor_vocab(rows):
    vocabulary = collections.defaultdict(set)
    for row in rows:
        for field in FACTOR_FIELDS:
            vocabulary[field].add(str(row["factors"][field]))
    return {field: sorted(values) for field, values in sorted(vocabulary.items())}


def audit_splits(split_rows, tokenizer_path, generator_path):
    split_names = {
        split: {
            name
            for row in rows
            for name in (*row["initial_order"], row["neutral_anchor"])
        }
        for split, rows in split_rows.items()
    }
    split_signatures = {
        split: {row["factor_signature"] for row in rows}
        for split, rows in split_rows.items()
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
            "factor_combination_overlap": len(split_signatures[left] & split_signatures[right]),
        }
    per_split = {}
    all_rows = [row for rows in split_rows.values() for row in rows]
    for split, rows in split_rows.items():
        grouped = collections.defaultdict(dict)
        for row in rows:
            grouped[row["group"]][row["surface_type"]] = row
        group_pass = 0
        for group in grouped.values():
            canonical = group["canonical"]
            paraphrase = group["paraphrase"]
            order_twin = group["order_twin"]
            binding_twin = group["binding_twin"]
            group_pass += int(
                row_label(canonical) == row_label(paraphrase)
                and canonical["terminal_order"] == paraphrase["terminal_order"]
                and canonical["answer"] != order_twin["answer"]
                and canonical["answer"] != binding_twin["answer"]
                and canonical["token_bag"] == order_twin["token_bag"]
                and canonical["token_bag"] == binding_twin["token_bag"]
            )
        matched = [row for row in rows if row["surface_type"] != "paraphrase"]
        per_split[split] = {
            "rows": len(rows),
            "groups": len(grouped),
            "group_gates_passed": group_pass,
            "duplicate_questions": len(rows) - len({normalized(row["question"]) for row in rows}),
            "entity_names": len(split_names[split]),
            "factor_combinations": len(split_signatures[split]),
            "factor_vocabulary": split_factor_vocab(rows),
            "shortcut_ceilings": {
                "token_bag": shortcut_ceiling(
                    matched, lambda row: canonical_json(row["token_bag"]),
                ),
                "absolute_pointer_positions": shortcut_ceiling(matched, pointer_positions),
                "source_token_length": shortcut_ceiling(
                    matched, lambda row: str(row["token_count"]),
                ),
            },
        }
    train_vocab = per_split["train"]["factor_vocabulary"]
    comp_vocab = per_split["development_compositional"]["factor_vocabulary"]
    known_direction_vocabulary = {
        word for pair in KNOWN_DIRECTION_PAIRS for word in pair
    }
    lexical_direction_vocabulary = {
        word for pair in LEXICAL_OOD_DIRECTION_PAIRS for word in pair
    }
    structural = {
        "all_ids_unique": len({row["id"] for row in all_rows}) == len(all_rows),
        "all_spans_present": all(set(REQUIRED_SPANS).issubset(row["spans"]) for row in all_rows),
        "all_spans_nonempty": all(
            row["spans"][label]["token_positions"]
            for row in all_rows for label in REQUIRED_SPANS
        ),
        "two_executors_agree": all(row["executor_agreement"] for row in all_rows),
        "all_group_gates_pass": all(
            report["groups"] == report["group_gates_passed"]
            for report in per_split.values()
        ),
        "all_shortcut_ceilings_at_chance_plus_one_example": all(
            ceiling["accuracy"] <= 1.0 / 3.0 + 1.0 / ceiling["total"] + 1e-12
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
        "cross_split_factor_combination_overlap_zero": all(
            report["factor_combination_overlap"] == 0 for report in pairwise.values()
        ),
        "compositional_atoms_seen_in_train": all(
            set(comp_vocab[field]).issubset(train_vocab[field]) for field in FACTOR_FIELDS
        ),
        "training_covers_all_known_atoms": all(
            {value for atom_field, value in expected_factor_atoms("known") if atom_field == field}
            == set(train_vocab[field])
            for field in FACTOR_FIELDS
        ),
        "lexical_direction_vocabulary_disjoint": (
            not known_direction_vocabulary & lexical_direction_vocabulary
            and all(row["factors"]["lexicon"] == "lexical_ood"
                    for row in split_rows["development_lexical_ood"])
            and all(row["factors"]["lexicon"] == "known"
                    for split in ("train", "development_compositional", "confirmation")
                    for row in split_rows[split])
        ),
    }
    return {
        "schema": "r12_referential_literal_pointer_factorized_corpus_v1",
        "claim_boundary": (
            "Frozen synthetic complete-compiler supervision. Programs and answers are "
            "labels/audit data only and are forbidden at inference."
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
            "target_pointer_labels": len(all_rows) * len(REQUIRED_SPANS),
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


def artifact_paths(out_dir):
    out_dir = Path(out_dir)
    return {split: out_dir / "{}.jsonl".format(split) for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-groups", type=int, default=24000)
    parser.add_argument("--development-compositional-groups", type=int, default=512)
    parser.add_argument("--development-lexical-ood-groups", type=int, default=512)
    parser.add_argument("--confirmation-groups", type=int, default=2048)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--development-compositional-seed", type=int, required=True)
    parser.add_argument("--development-lexical-ood-seed", type=int, required=True)
    parser.add_argument("--confirmation-seed", type=int, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    targets = artifact_paths(out_dir)
    report_path = out_dir / "report.json"
    if any(path.exists() for path in (*targets.values(), report_path)):
        raise SystemExit("refusing to overwrite corpus output")
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    names, nonce_width = candidate_names(tokenizer, 4100)
    name_pools = {
        "train": names[:2600],
        "development_compositional": names[2600:3050],
        "development_lexical_ood": names[3050:3500],
        "confirmation": names[3500:4100],
    }
    seeds = {
        "train": args.train_seed,
        "development_compositional": args.development_compositional_seed,
        "development_lexical_ood": args.development_lexical_ood_seed,
        "confirmation": args.confirmation_seed,
    }
    counts = {
        "train": args.train_groups,
        "development_compositional": args.development_compositional_groups,
        "development_lexical_ood": args.development_lexical_ood_groups,
        "confirmation": args.confirmation_groups,
    }
    split_rows = {
        split: build_split(split, counts[split], seeds[split], tokenizer, name_pools[split])
        for split in SPLITS
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
