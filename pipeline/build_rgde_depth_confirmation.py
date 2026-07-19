#!/usr/bin/env python3
"""Build a fresh three-to-eight-step RGDE packet-stream confirmation board."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import (
    factor_atoms,
    factor_catalogue,
    factor_signature,
    make_row,
)
from semantic_compiler_falsifier import (
    AMOUNTS,
    DIRECTIONS,
    Operation,
    apply_program_adjacent_swaps,
    apply_program_pop_insert,
    candidate_names,
    sha256_bytes,
    sha256_file,
)


WORD = re.compile(r"\w+")
SURFACES = ("canonical", "paraphrase", "order_twin", "binding_twin")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngram_hashes(text, width=13):
    words = normalized(text).split()
    return {
        hashlib.blake2b(
            " ".join(words[index:index + width]).encode(), digest_size=16,
        ).digest()
        for index in range(max(0, len(words) - width + 1))
    }


def read_public(paths):
    names = set()
    questions = set()
    grams = set()
    factor_signatures = set()
    rows = 0
    for path in paths:
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                names.update(row["initial_order"])
                names.add(row["neutral_anchor"])
                questions.add(normalized(row["question"]))
                grams.update(ngram_hashes(row["question"]))
                factor_signatures.add(row["factor_signature"])
    return {
        "rows": rows,
        "names": names,
        "questions": questions,
        "grams": grams,
        "factor_signatures": factor_signatures,
    }


def fresh_factor_specs(public_signatures, count, seed):
    candidates = [
        factors for factors in factor_catalogue("known")
        if factor_signature(factors) not in public_signatures
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    uncovered = set().union(*(factor_atoms(factors) for factors in factor_catalogue("known")))
    selected = []
    for factors in candidates:
        if factor_atoms(factors) & uncovered:
            selected.append(factors)
            uncovered -= factor_atoms(factors)
        if not uncovered:
            break
    for factors in candidates:
        if len(selected) == count:
            break
        if factors not in selected:
            selected.append(factors)
    if uncovered or len(selected) != count:
        raise RuntimeError("could not construct fresh factor coverage")
    return selected


def paired_names(tokenizer, count, seed, public_names):
    atoms, _ = candidate_names(tokenizer, 4100)
    rng = random.Random(seed)
    selected = []
    seen = set()
    while len(selected) < count:
        left, right = rng.sample(atoms, 2)
        name = "{}-{}".format(left, right)
        if name in seen:
            continue
        seen.add(name)
        if name in public_names or not tokenizer.encode(name).ids:
            continue
        selected.append(name)
    return selected


def instantiate(specification, entities):
    return tuple(
        Operation(direction, entities[entity], amount)
        for direction, entity, amount in specification
    )


def choose_semantics(depth, entities, rng, required_query_position=None):
    operations = tuple(itertools.product(DIRECTIONS, range(3), AMOUNTS))
    for _ in range(10000):
        base = tuple(rng.choice(operations) for _ in range(depth))
        shift = rng.randrange(1, depth)
        order = base[shift:] + base[:shift]
        entity_slots = [operation[1] for operation in base]
        entity_shift = rng.randrange(1, depth)
        rotated_entities = entity_slots[entity_shift:] + entity_slots[:entity_shift]
        binding = tuple(
            (operation[0], rotated_entities[index], operation[2])
            for index, operation in enumerate(base)
        )
        if len({base, order, binding}) != 3:
            continue
        programs = [instantiate(spec, entities) for spec in (base, order, binding)]
        terminals = [apply_program_pop_insert(entities, program) for program in programs]
        if any(
            apply_program_adjacent_swaps(entities, program) != terminal
            for program, terminal in zip(programs, terminals)
        ):
            raise AssertionError("long executors disagree")
        separators = [
            position for position in range(3)
            if terminals[0][position] != terminals[1][position]
            and terminals[0][position] != terminals[2][position]
        ]
        if required_query_position is not None:
            if int(required_query_position) in separators:
                return programs, terminals, int(required_query_position)
        elif separators:
            return programs, terminals, rng.choice(separators)
    raise RuntimeError("could not sample separated long semantics")


def combine_token_bag(chunks):
    bag = collections.Counter()
    for chunk in chunks:
        bag.update(dict(chunk["token_bag"]))
    return sorted(bag.items())


def combine_word_bag(chunks):
    bag = collections.Counter()
    for chunk in chunks:
        bag.update(normalized(chunk["question"]).split())
    return sorted(bag.items())


def render_chunks(group, surface, initial, program, query_position, factors,
                  neutral_anchor, tokenizer):
    filler = Operation("left", initial[0], 1)
    chunks = []
    for chunk_index in range((len(program) + 1) // 2):
        start = 2 * chunk_index
        active = tuple(program[start:start + 2])
        rendered = active if len(active) == 2 else active + (filler,)
        chunk = make_row(
            "confirmation_depth",
            group * 8 + chunk_index,
            surface,
            initial,
            rendered,
            query_position,
            factors[chunk_index],
            initial[(query_position + 1) % 3],
            1,
            neutral_anchor,
            tokenizer,
        )
        chunk["active_operations"] = len(active)
        chunk["chunk_index"] = chunk_index
        chunks.append(chunk)
    return chunks


def build_board(groups, seed, tokenizer, public, balanced_queries=False):
    rng = random.Random(seed)
    max_chunks = 4
    factor_specs = fresh_factor_specs(
        public["factor_signatures"], groups * max_chunks * 2, seed ^ 0xA5A5A5A5,
    )
    names = paired_names(tokenizer, groups * 4, seed ^ 0x6C8E9CF5, public["names"])
    rows = []
    seen_streams = set()
    depth_counts = collections.Counter()
    factor_cursor = 0
    for group in range(groups):
        depth = 3 + group % 6
        depth_counts[depth] += 4
        entities = tuple(names[group * 4:group * 4 + 3])
        neutral_anchor = names[group * 4 + 3]
        initial = tuple(rng.sample(entities, 3))
        required_query = (group // 6) % 3 if balanced_queries else None
        programs, terminals, query_position = choose_semantics(
            depth, initial, rng, required_query_position=required_query,
        )
        chunk_count = (depth + 1) // 2
        canonical_factors = [factor_specs[factor_cursor]] * chunk_count
        factor_cursor += 1
        paraphrase_factors = [factor_specs[factor_cursor]] * chunk_count
        factor_cursor += 1
        surface_programs = {
            "canonical": programs[0],
            "paraphrase": programs[0],
            "order_twin": programs[1],
            "binding_twin": programs[2],
        }
        surface_terminals = {
            "canonical": terminals[0],
            "paraphrase": terminals[0],
            "order_twin": terminals[1],
            "binding_twin": terminals[2],
        }
        candidate = []
        for surface in SURFACES:
            factors = (
                paraphrase_factors if surface == "paraphrase" else canonical_factors
            )
            program = surface_programs[surface]
            chunks = render_chunks(
                group, surface, initial, program, query_position, factors,
                neutral_anchor, tokenizer,
            )
            questions = [normalized(chunk["question"]) for chunk in chunks]
            stream_signature = tuple(questions)
            grams = set().union(*(ngram_hashes(chunk["question"]) for chunk in chunks))
            if any(question in public["questions"] for question in questions):
                raise RuntimeError("exact source collision")
            if stream_signature in seen_streams:
                raise RuntimeError("packet-stream source collision")
            if grams & public["grams"]:
                raise RuntimeError("word-13-gram source collision")
            terminal = surface_terminals[surface]
            candidate.append({
                "id": "RGDE-DEPTH-{:06d}-{}".format(group, surface),
                "schema": "r12_rgde_depth_confirmation_row_v1",
                "split": "confirmation_depth",
                "group": group,
                "surface_type": surface,
                "depth": depth,
                "initial_order": list(initial),
                "program": [operation.as_dict() for operation in program],
                "query": {"kind": "read_position", "position": query_position},
                "terminal_order": list(terminal),
                "answer": terminal[query_position],
                "chunks": chunks,
                "source_tokens": sum(chunk["token_count"] for chunk in chunks),
                "token_bag": combine_token_bag(chunks),
                "word_bag": combine_word_bag(chunks),
                "executor_agreement": True,
                "external_schedule": {
                    "active_operations": depth,
                    "halt_after": depth,
                },
            })
            seen_streams.add(stream_signature)
        by_surface = {row["surface_type"]: row for row in candidate}
        canonical = by_surface["canonical"]
        if canonical["answer"] in {
            by_surface["order_twin"]["answer"], by_surface["binding_twin"]["answer"],
        }:
            raise AssertionError("long twin answer separation failed")
        if canonical["terminal_order"] != by_surface["paraphrase"]["terminal_order"]:
            raise AssertionError("long paraphrase behavior drift")
        if not (
            canonical["word_bag"] == by_surface["order_twin"]["word_bag"]
            == by_surface["binding_twin"]["word_bag"]
        ):
            raise AssertionError("long matched word bags diverged")
        rows.extend(candidate)
    return rows, dict(sorted(depth_counts.items()))


def write_board(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    Path(path).write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    board_path = out_dir / "confirmation_depth.jsonl"
    report_path = out_dir / "report.json"
    if board_path.exists() or report_path.exists():
        raise SystemExit("refusing existing depth-confirmation output")
    if args.groups < 6:
        raise SystemExit("depth confirmation requires at least six groups")
    if any("confirmation" in Path(path).name.lower() for path in args.public_data):
        raise SystemExit("old confirmation input is forbidden")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    public = read_public(args.public_data)
    rows, depth_counts = build_board(args.groups, args.seed, tokenizer, public)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_board(board_path, rows)
    all_chunks = [chunk for row in rows for chunk in row["chunks"]]
    used_factor_atoms = set().union(*(
        factor_atoms(chunk["factors"]) for chunk in all_chunks
    ))
    expected_atoms = set().union(*(
        factor_atoms(factors) for factors in factor_catalogue("known")
    ))
    gates = {
        "public_exact_prompt_overlap_zero": True,
        "public_word_13gram_overlap_zero": True,
        "public_entity_name_overlap_zero": True,
        "public_factor_combination_overlap_zero": True,
        "all_chunk_spans_present": all(len(chunk["spans"]) == 10 for chunk in all_chunks),
        "all_chunk_spans_nonempty": all(
            target["token_positions"]
            for chunk in all_chunks for target in chunk["spans"].values()
        ),
        "two_cpu_executors_agree": all(row["executor_agreement"] for row in rows),
        "all_quartets_complete": len(rows) == 4 * args.groups,
        "depths_three_through_eight_present": set(depth_counts) == set(range(3, 9)),
        "known_factor_atoms_covered": used_factor_atoms == expected_atoms,
        "old_confirmation_access_zero": True,
    }
    report = {
        "schema": "r12_rgde_depth_confirmation_report_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "seed": args.seed,
        "groups": args.groups,
        "rows": len(rows),
        "depth_counts": depth_counts,
        "chunks": len(all_chunks),
        "source_tokens": sum(row["source_tokens"] for row in rows),
        "unique_entity_names": len({name for row in rows for name in row["initial_order"]}),
        "public_rows_audited": public["rows"],
        "public_exact_prompt_overlap": 0,
        "public_word_13gram_overlap": 0,
        "public_entity_name_overlap": 0,
        "public_factor_combination_overlap": 0,
        "external_schedule": True,
        "old_confirmation_access": 0,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "generator_sha256": sha256_file(__file__),
        "public_data_sha256": {
            str(Path(path).name): sha256_file(path) for path in args.public_data
        },
        "artifact": artifact,
        "claim_boundary": (
            "Fresh source-deleted recurrent-depth component confirmation with an external "
            "operation count and halt. Not autonomous language reasoning or learned halting."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact": artifact,
        "depth_counts": depth_counts,
        "report": str(report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("depth-confirmation corpus gate failed")


if __name__ == "__main__":
    main()
