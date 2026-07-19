#!/usr/bin/env python3
"""Build the frozen whole-source S8 nil-linked law-graph board."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import normalized, ngrams
from semantic_compiler_falsifier import (
    SpanWriter,
    attach_token_targets,
    candidate_names,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from s6_contextual_affine_law import AffineLaw, pop_insert
from s7_learned_cayley_law import PRIMARY_MODULI, SymbolBinding, stride_two_successor
from s8_nil_linked_graph_compiler import (
    compile_row,
    recode_operation_ids,
)
from s8_nil_linked_law_graph import execute_graph, graph_from_ordered_events


TRAIN_SPLIT = "s8_nil_graph_train"
DEVELOPMENT_SPLIT = "s8_nil_graph_development"
CONFIRMATION_SPLIT = "s8_nil_graph_confirmation"
SPLIT_PERSON = "s8-nil-linked-natural-law-v1"
TRAIN_RENDERERS = ("registry", "ledger", "routing")
DEVELOPMENT_RENDERERS = ("dossier", "relay")
CONFIRMATION_RENDERERS = ("archive", "dispatch")
NAME_POOL_SIZE = 1100


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _law_order(law: AffineLaw) -> bytes:
    return hashlib.sha256(f"{SPLIT_PERSON}|{law.key}".encode("ascii")).digest()


def law_pools(modulus: int) -> dict[str, tuple[AffineLaw, ...]]:
    laws = sorted(
        (
            AffineLaw(modulus, slope, intercept)
            for slope in range(1, modulus)
            for intercept in range(modulus)
        ),
        key=_law_order,
    )
    train_end = max(2, int(len(laws) * 0.60))
    development_end = train_end + max(2, int(len(laws) * 0.20))
    pools = {
        "train": tuple(sorted(laws[:train_end])),
        "development": tuple(sorted(laws[train_end:development_end])),
        "confirmation": tuple(sorted(laws[development_end:])),
    }
    if min(map(len, pools.values())) < 2:
        raise ValueError("S8 law split has fewer than two laws")
    if set(pools["train"]) & set(pools["development"]):
        raise ValueError("S8 train/development law overlap")
    if set(pools["train"]) & set(pools["confirmation"]):
        raise ValueError("S8 train/confirmation law overlap")
    if set(pools["development"]) & set(pools["confirmation"]):
        raise ValueError("S8 development/confirmation law overlap")
    return pools


def _binding(rng: random.Random, modulus: int) -> SymbolBinding:
    values = list(range(modulus))
    rng.shuffle(values)
    if values == list(range(modulus)):
        values = values[1:] + values[:1]
    return SymbolBinding(modulus, tuple(values))


def _split_name_pools(tokenizer: Tokenizer, seed: int) -> dict[str, tuple[str, ...]]:
    values, width = candidate_names(tokenizer, NAME_POOL_SIZE * 3)
    rng = random.Random(seed)
    rng.shuffle(values)
    pools = {
        "train": tuple(values[:NAME_POOL_SIZE]),
        "development": tuple(values[NAME_POOL_SIZE:2 * NAME_POOL_SIZE]),
        "confirmation": tuple(values[2 * NAME_POOL_SIZE:3 * NAME_POOL_SIZE]),
    }
    if any(set(left) & set(right) for left, right in (
        (pools["train"], pools["development"]),
        (pools["train"], pools["confirmation"]),
        (pools["development"], pools["confirmation"]),
    )):
        raise ValueError("S8 name pools overlap")
    if width <= 0:
        raise ValueError("S8 nonce tokenizer width is invalid")
    return pools


def _add_list(
    writer: SpanWriter,
    values: Sequence[str],
    label_prefix: str,
    separator: str,
) -> None:
    for index, value in enumerate(values):
        if index:
            writer.add(separator)
        writer.add(value, f"{label_prefix}.{index}")


def _render_registry(
    writer: SpanWriter,
    entities: Sequence[str],
    positions: Sequence[str],
    state_entities: Sequence[str],
    renderer: str,
) -> None:
    if renderer in {"registry", "dossier", "archive"}:
        writer.add("Entity registry in identity order: ")
        _add_list(writer, entities, "entity.roster", ", ")
        writer.add(".\nPosition registry in display order: ")
        _add_list(writer, positions, "position.roster", ", ")
        writer.add(".\nInitial occupants by display position: ")
        _add_list(writer, state_entities, "state.entity", ", ")
        writer.add(".\n")
    elif renderer in {"ledger", "relay", "dispatch"}:
        writer.add("Identity ledger lists ")
        _add_list(writer, entities, "entity.roster", " / ")
        writer.add(".\nVisible markers are indexed as ")
        _add_list(writer, positions, "position.roster", " / ")
        writer.add(".\nThe marker-indexed lineup currently reads ")
        _add_list(writer, state_entities, "state.entity", " / ")
        writer.add(".\n")
    elif renderer == "routing":
        writer.add("Known identities, by registry rank: ")
        _add_list(writer, entities, "entity.roster", "; ")
        writer.add(".\nLocation codes, by visible index: ")
        _add_list(writer, positions, "position.roster", "; ")
        writer.add(".\nOccupancy across those visible indices: ")
        _add_list(writer, state_entities, "state.entity", "; ")
        writer.add(".\n")
    else:
        raise ValueError(f"unknown S8 renderer {renderer}")


def _render_card(
    writer: SpanWriter,
    index: int,
    operation: str,
    y0: str,
    y1: str,
    renderer: str,
) -> None:
    if renderer in {"registry", "relay", "archive"}:
        writer.add("Law card ")
        writer.add(operation, f"card.{index}.operation")
        writer.add(" sends the hidden-zero reference to ")
        writer.add(y0, f"card.{index}.y0")
        writer.add(" and the hidden-successor reference to ")
        writer.add(y1, f"card.{index}.y1")
        writer.add(".\n")
    elif renderer in {"ledger", "dossier", "dispatch"}:
        writer.add("For operation ")
        writer.add(operation, f"card.{index}.operation")
        writer.add(", the witnessed zero output is ")
        writer.add(y0, f"card.{index}.y0")
        writer.add("; its witnessed successor output is ")
        writer.add(y1, f"card.{index}.y1")
        writer.add(".\n")
    elif renderer == "routing":
        writer.add("Witness record for ")
        writer.add(operation, f"card.{index}.operation")
        writer.add(": zero maps to ")
        writer.add(y0, f"card.{index}.y0")
        writer.add(", while successor maps to ")
        writer.add(y1, f"card.{index}.y1")
        writer.add(".\n")
    else:
        raise ValueError(f"unknown S8 renderer {renderer}")


def _render_event(
    writer: SpanWriter,
    node_id: int,
    tag: str,
    operation: str,
    entity: str,
    next_tag: str | None,
    renderer: str,
) -> None:
    if renderer in {"registry", "dossier", "dispatch"}:
        writer.add("Event node ")
        writer.add(tag, f"event.{node_id}.tag")
        writer.add(" applies ")
        writer.add(operation, f"event.{node_id}.operation")
        writer.add(" to ")
        writer.add(entity, f"event.{node_id}.entity")
        writer.add("; control then goes to ")
    elif renderer in {"ledger", "relay", "archive"}:
        writer.add("At tagged step ")
        writer.add(tag, f"event.{node_id}.tag")
        writer.add(", use law ")
        writer.add(operation, f"event.{node_id}.operation")
        writer.add(" on identity ")
        writer.add(entity, f"event.{node_id}.entity")
        writer.add(", then hand off to ")
    elif renderer == "routing":
        writer.add("Route ")
        writer.add(tag, f"event.{node_id}.tag")
        writer.add(": execute ")
        writer.add(operation, f"event.{node_id}.operation")
        writer.add(" for ")
        writer.add(entity, f"event.{node_id}.entity")
        writer.add("; the outgoing route is ")
    else:
        raise ValueError(f"unknown S8 renderer {renderer}")
    if next_tag is None:
        writer.add("nil", f"event.{node_id}.nil")
    else:
        writer.add(next_tag, f"event.{node_id}.next")
    writer.add(".\n")


def render_source(
    *,
    entities: Sequence[str],
    positions: Sequence[str],
    initial_state: Sequence[int],
    cards: Sequence[tuple[str, int, int]],
    nodes: Sequence[tuple[str, str, int, str | None]],
    entry_tag: str,
    query_position: int,
    renderer: str,
    card_order: Sequence[int],
) -> tuple[str, dict[str, object]]:
    writer = SpanWriter()
    _render_registry(
        writer,
        entities,
        positions,
        [entities[index] for index in initial_state],
        renderer,
    )
    writer.add("Contextual operation witnesses follow.\n")
    for output_index, card_index in enumerate(card_order):
        operation, y0, y1 = cards[card_index]
        _render_card(
            writer,
            output_index,
            operation,
            positions[y0],
            positions[y1],
            renderer,
        )
    writer.add("Begin execution at tag ")
    writer.add(entry_tag, "entry.tag")
    writer.add(". Event definitions are intentionally unordered.\n")
    for node_id, (tag, operation, identity, next_tag) in enumerate(nodes):
        _render_event(
            writer,
            node_id,
            tag,
            operation,
            entities[identity],
            next_tag,
            renderer,
        )
    query_frames = {
        "registry": "Registry closure asks for the occupant at marker ",
        "ledger": "Ledger terminal review requests the identity at position ",
        "routing": "Routing completion must return the occupant of location code ",
        "dossier": "Dossier closeout asks which registered identity is at marker ",
        "relay": "Relay termination requests the final occupant of position ",
        "archive": "Archive completion records the identity found at marker ",
        "dispatch": "Dispatch shutdown reports the occupant assigned to position ",
    }
    writer.add(query_frames[renderer])
    writer.add(positions[query_position], "query.position")
    writer.add("?")
    return writer.finish()


def _make_row(
    *,
    rng: random.Random,
    tokenizer: Tokenizer,
    split: str,
    row_index: int,
    modulus: int,
    binding: SymbolBinding,
    laws: Sequence[AffineLaw],
    names: Sequence[str],
    renderer: str,
    depth: int,
    include_outputs: bool,
) -> dict[str, object]:
    required_names = 2 * modulus + 12
    chosen = rng.sample(list(names), required_names)
    entities = tuple(chosen[:modulus])
    positions = tuple(chosen[modulus:2 * modulus])
    cursor = 2 * modulus
    law_count = min(len(laws), max(2, 2 + rng.randrange(3)))
    selected_laws = rng.sample(list(laws), law_count)
    operations = tuple(chosen[cursor:cursor + law_count])
    cursor += law_count
    tags = tuple(chosen[cursor:cursor + depth])
    law_by_operation = dict(zip(operations, selected_laws, strict=True))
    cards = tuple(
        (operation, *binding.card(law_by_operation[operation]))
        for operation in operations
    )
    required_operation_count = min(2, depth)
    execution_operations = list(operations[:required_operation_count]) + [
        rng.choice(operations) for _ in range(depth - required_operation_count)
    ]
    rng.shuffle(execution_operations)
    execution_events = tuple(
        (rng.randrange(modulus), operation) for operation in execution_operations
    )
    initial_state = list(range(modulus))
    rng.shuffle(initial_state)
    query_position = rng.randrange(modulus)

    storage_order = list(range(depth))
    rng.shuffle(storage_order)
    if storage_order == list(range(depth)):
        storage_order = storage_order[1:] + storage_order[:1]
    execution_to_storage = {
        execution_index: storage_id
        for storage_id, execution_index in enumerate(storage_order)
    }
    stored_nodes: list[tuple[str, str, int, str | None]] = []
    for execution_index in storage_order:
        identity, operation = execution_events[execution_index]
        next_tag = (
            tags[execution_index + 1]
            if execution_index + 1 < depth
            else None
        )
        stored_nodes.append((tags[execution_index], operation, identity, next_tag))
    entry_node = execution_to_storage[0]
    graph_cards = {operation: (y0, y1) for operation, y0, y1 in cards}
    graph = graph_from_ordered_events(
        modulus=modulus,
        initial_state=initial_state,
        cards=graph_cards,
        events=execution_events,
        storage_ids=tuple(execution_to_storage[index] for index in range(depth)),
        query_position=query_position,
    )
    expected_state = tuple(initial_state)
    for identity, operation in execution_events:
        source = expected_state.index(identity)
        destination = binding.destination(law_by_operation[operation], source)
        expected_state = pop_insert(expected_state, identity, destination)
    graph_output = execute_graph(graph, binding.successor, binding.zero_symbol)
    if graph_output[:2] != (expected_state, expected_state[query_position]):
        raise ValueError("S8 independent graph executors disagree")

    card_order = list(range(law_count))
    rng.shuffle(card_order)
    question, spans = render_source(
        entities=entities,
        positions=positions,
        initial_state=initial_state,
        cards=cards,
        nodes=stored_nodes,
        entry_tag=tags[0],
        query_position=query_position,
        renderer=renderer,
        card_order=card_order,
    )
    encoding, token_targets = attach_token_targets(question, spans, tokenizer)
    row: dict[str, object] = {
        "id": f"S8-{split.upper()}-{row_index:06d}",
        "schema": "r12_s8_nil_linked_law_graph_row_v1",
        "split": split,
        "renderer": renderer,
        "question": question,
        "modulus": modulus,
        "depth": depth,
        "entities": list(entities),
        "positions": list(positions),
        "initial_state": initial_state,
        "cards": [
            {"operation": operation, "y0": y0, "y1": y1}
            for operation, y0, y1 in cards
        ],
        "nodes": [
            {
                "tag": tag,
                "operation": operation,
                "identity": identity,
                "next_tag": next_tag,
            }
            for tag, operation, identity, next_tag in stored_nodes
        ],
        "entry_tag": tags[0],
        "entry_node": entry_node,
        "query_position": query_position,
        "execution_tags": list(tags),
        "law_keys": [law_by_operation[operation].key for operation in operations],
        "spans": token_targets,
        "token_count": len(encoding.ids),
        "token_ids_sha256": sha256_bytes(canonical_json(encoding.ids).encode()),
        "token_bag": sorted(collections.Counter(encoding.ids).items()),
        "node_storage_is_noncanonical": (
            depth == 1 or storage_order != list(range(depth))
        ),
        "executor_agreement": True,
        "supervision": "graph_fields_only",
    }
    if include_outputs:
        row["final_state"] = list(expected_state)
        row["answer"] = int(expected_state[query_position])
    return row


def _build_rows(
    *,
    count: int,
    rng: random.Random,
    tokenizer: Tokenizer,
    split: str,
    bindings: dict[int, SymbolBinding],
    names: Sequence[str],
    renderers: Sequence[str],
    include_outputs: bool,
) -> list[dict[str, object]]:
    cells = [
        (modulus, depth, renderer)
        for modulus in PRIMARY_MODULI
        for depth in range(3, 9)
        for renderer in renderers
    ]
    rng.shuffle(cells)
    rows = []
    for index in range(count):
        modulus, depth, renderer = cells[index % len(cells)]
        rows.append(
            _make_row(
                rng=rng,
                tokenizer=tokenizer,
                split=split,
                row_index=index,
                modulus=modulus,
                binding=bindings[modulus],
                laws=law_pools(modulus)[
                    "train" if split == TRAIN_SPLIT else
                    "development" if split == DEVELOPMENT_SPLIT else
                    "confirmation"
                ],
                names=names,
                renderer=renderer,
                depth=depth,
                include_outputs=include_outputs,
            )
        )
    return rows


def _build_train_rows(
    *,
    count: int,
    rng: random.Random,
    tokenizer: Tokenizer,
    bindings: dict[int, SymbolBinding],
    names: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    cells = [
        (modulus, depth, renderer)
        for modulus in PRIMARY_MODULI
        for depth in range(1, 9)
        for renderer in TRAIN_RENDERERS
    ]
    rng.shuffle(cells)
    for index in range(count):
        modulus, depth, renderer = cells[index % len(cells)]
        rows.append(
            _make_row(
                rng=rng,
                tokenizer=tokenizer,
                split=TRAIN_SPLIT,
                row_index=index,
                modulus=modulus,
                binding=bindings[modulus],
                laws=law_pools(modulus)["train"],
                names=names,
                renderer=renderer,
                depth=depth,
                include_outputs=False,
            )
        )
    return rows


def _names(rows: Sequence[dict[str, object]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        result.update(map(str, row["entities"]))
        result.update(map(str, row["positions"]))
        result.update(str(card["operation"]) for card in row["cards"])
        result.update(str(node["tag"]) for node in row["nodes"])
    return result


def _prompt_sets(rows: Sequence[dict[str, object]]) -> tuple[set[str], set[str]]:
    prompts = {normalized(row["question"]) for row in rows}
    grams: set[str] = set()
    for row in rows:
        grams.update(ngrams(row["question"]))
    return prompts, grams


def _audit(
    train: list[dict[str, object]],
    development: list[dict[str, object]],
    confirmation: list[dict[str, object]],
    tokenizer: Tokenizer,
) -> dict[str, object]:
    if any("final_state" in row or "answer" in row for row in train):
        raise ValueError("S8 training rows leak state or answer")
    if not all(row["supervision"] == "graph_fields_only" for row in train):
        raise ValueError("S8 training supervision contract mismatch")
    if not all(row["executor_agreement"] for row in train + development + confirmation):
        raise ValueError("S8 executor disagreement")
    if not all(row["node_storage_is_noncanonical"] for row in train + development + confirmation):
        raise ValueError("S8 canonical node storage leaked")
    name_sets = [_names(rows) for rows in (train, development, confirmation)]
    if any(left & right for left, right in (
        (name_sets[0], name_sets[1]),
        (name_sets[0], name_sets[2]),
        (name_sets[1], name_sets[2]),
    )):
        raise ValueError("S8 split name overlap")
    prompt_sets = [_prompt_sets(rows) for rows in (train, development, confirmation)]
    exact_overlaps = {
        "train_development": len(prompt_sets[0][0] & prompt_sets[1][0]),
        "train_confirmation": len(prompt_sets[0][0] & prompt_sets[2][0]),
        "development_confirmation": len(prompt_sets[1][0] & prompt_sets[2][0]),
    }
    gram_overlaps = {
        "train_development": len(prompt_sets[0][1] & prompt_sets[1][1]),
        "train_confirmation": len(prompt_sets[0][1] & prompt_sets[2][1]),
        "development_confirmation": len(prompt_sets[1][1] & prompt_sets[2][1]),
    }
    if any(exact_overlaps.values()) or any(gram_overlaps.values()):
        gram_samples = {
            "train_development": sorted(prompt_sets[0][1] & prompt_sets[1][1])[:3],
            "train_confirmation": sorted(prompt_sets[0][1] & prompt_sets[2][1])[:3],
            "development_confirmation": sorted(prompt_sets[1][1] & prompt_sets[2][1])[:3],
        }
        raise ValueError(
            "S8 cross-split prompt overlap "
            f"exact={exact_overlaps} grams={gram_overlaps} samples={gram_samples}"
        )
    dev_cells = collections.Counter(
        (row["modulus"], row["depth"], row["renderer"]) for row in development
    )
    confirmation_cells = collections.Counter(
        (row["modulus"], row["depth"], row["renderer"]) for row in confirmation
    )
    for label, counts in (("development", dev_cells), ("confirmation", confirmation_cells)):
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"S8 {label} cells are imbalanced")
    recoded_maximum = 0
    recoded_changed_width = 0
    for row in train + development + confirmation:
        original = compile_row(row, tokenizer)
        recoded = recode_operation_ids(original, tokenizer)
        recoded_maximum = max(recoded_maximum, len(recoded.ids))
        recoded_changed_width += int(len(recoded.ids) != len(original.ids))
        card_names = {str(card["operation"]) for card in recoded.row["cards"]}
        event_names = {str(node["operation"]) for node in recoded.row["nodes"]}
        if event_names - card_names:
            raise ValueError("S8 nonce recoding broke event/card binding")
        if len(recoded.ids) > 512:
            raise ValueError("S8 nonce-recoded source exceeds context")
    return {
        "train_rows": len(train),
        "development_rows": len(development),
        "confirmation_rows": len(confirmation),
        "train_has_state_or_answer": False,
        "executor_agreement_rows": len(train) + len(development) + len(confirmation),
        "noncanonical_storage_rows": len(train) + len(development) + len(confirmation),
        "split_name_overlap": 0,
        "exact_prompt_overlap": exact_overlaps,
        "thirteen_gram_overlap": gram_overlaps,
        "development_cell_counts": {
            "|".join(map(str, key)): value for key, value in sorted(dev_cells.items())
        },
        "confirmation_cell_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(confirmation_cells.items())
        },
        "maximum_token_count": max(
            int(row["token_count"]) for row in train + development + confirmation
        ),
        "nonce_recode_rows": len(train) + len(development) + len(confirmation),
        "nonce_recode_maximum_token_count": recoded_maximum,
        "nonce_recode_changed_width_rows": recoded_changed_width,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--train-rows", type=int, default=48000)
    parser.add_argument("--development-rows", type=int, default=2048)
    parser.add_argument("--confirmation-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing S8 board: {args.out_dir}")
    if min(args.train_rows, args.development_rows, args.confirmation_rows) <= 0:
        raise SystemExit("S8 row counts must be positive")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    names = _split_name_pools(tokenizer, args.seed ^ 0x51A8)
    rng = random.Random(args.seed)
    bindings = {modulus: _binding(rng, modulus) for modulus in PRIMARY_MODULI}
    train = _build_train_rows(
        count=args.train_rows,
        rng=rng,
        tokenizer=tokenizer,
        bindings=bindings,
        names=names["train"],
    )
    development = _build_rows(
        count=args.development_rows,
        rng=rng,
        tokenizer=tokenizer,
        split=DEVELOPMENT_SPLIT,
        bindings=bindings,
        names=names["development"],
        renderers=DEVELOPMENT_RENDERERS,
        include_outputs=True,
    )
    confirmation = _build_rows(
        count=args.confirmation_rows,
        rng=rng,
        tokenizer=tokenizer,
        split=CONFIRMATION_SPLIT,
        bindings=bindings,
        names=names["confirmation"],
        renderers=CONFIRMATION_RENDERERS,
        include_outputs=True,
    )
    audit = _audit(train, development, confirmation, tokenizer)
    generator_rows = []
    for modulus in PRIMARY_MODULI:
        binding = bindings[modulus]
        false_successor = stride_two_successor(binding.successor, binding.zero_symbol)
        for symbol in range(modulus):
            generator_rows.append({
                "schema": "r12_s8_generator_cell_v1",
                "modulus": modulus,
                "current_symbol": symbol,
                "next_symbol": binding.successor[symbol],
                "false_next_symbol": false_successor[symbol],
                "zero_symbol": binding.zero_symbol,
                "supervision": "successor_and_zero_only",
            })
    payloads = {
        "generator_train.jsonl": _jsonl(generator_rows),
        "train.jsonl": _jsonl(train),
        "development.jsonl": _jsonl(development),
        "confirmation.sealed.jsonl": _jsonl(confirmation),
    }
    report = {
        "schema": "r12_s8_nil_linked_law_graph_board_report_v1",
        "decision": "admit_s8_nil_linked_law_graph_board",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "name_pool_size_per_split": NAME_POOL_SIZE,
        "renderers": {
            "train": TRAIN_RENDERERS,
            "development": DEVELOPMENT_RENDERERS,
            "confirmation": CONFIRMATION_RENDERERS,
        },
        "law_counts": {
            str(modulus): {
                split: len(values) for split, values in law_pools(modulus).items()
            }
            for modulus in PRIMARY_MODULI
        },
        "binding_hashes": {
            str(modulus): _sha256(
                json.dumps(
                    bindings[modulus].observed_to_latent,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            for modulus in PRIMARY_MODULI
        },
        "audit": audit,
        "files": {
            name: {"sha256": _sha256(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    args.out_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        (args.out_dir / name).write_bytes(payload)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "report_sha256": sha256_file(args.out_dir / "report.json"),
        "audit": audit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
