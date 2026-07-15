#!/usr/bin/env python3
"""Independently reconstruct and audit the R12 cursor-action neural canary."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import re
import stat
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


SCHEMA = "counterfactual_cursor_action_canary_v1"
CANARY_ID = "ccaa-neural-canary-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
LABELS = OPERATIONS + ("DONE",)
SPLITS = ("train", "development", "confirmation")
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pipeline/counterfactual_cursor_action_canary_contract_v1.json"
GENERATOR = ROOT / "pipeline/generate_counterfactual_cursor_action_canary.py"
CONTRACT_SHA256 = "d767e2bf405364d929d6aab0b1e202d643b974e62e71adbddde012a09255a49b"
IMPLEMENTATION_PATHS = (
    "R12_COUNTERFACTUAL_CURSOR_ACTION_THEORY.md",
    "R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md",
    "pipeline/counterfactual_cursor_action_canary_contract_v1.json",
    "pipeline/generate_counterfactual_cursor_action_canary.py",
    "pipeline/audit_counterfactual_cursor_action_canary.py",
    "pipeline/test_counterfactual_cursor_action_canary.py",
    "train/model.py",
    "train/counterfactual_cursor_action.py",
    "train/counterfactual_cursor_action_data.py",
    "train/counterfactual_cursor_action_objectives.py",
    "train/counterfactual_cursor_action_training.py",
    "train/train_counterfactual_cursor_action.py",
    "train/eval_counterfactual_cursor_action.py",
    "train/score_counterfactual_cursor_action.py",
    "train/test_counterfactual_cursor_action.py",
    "train/test_counterfactual_cursor_action_data.py",
    "train/test_counterfactual_cursor_action_objectives.py",
    "train/test_counterfactual_cursor_action_training.py",
    "train/test_train_counterfactual_cursor_action.py",
    "train/test_eval_counterfactual_cursor_action.py",
    "train/jobs/counterfactual_cursor_action_canary.sbatch",
    "train/jobs/eval_counterfactual_cursor_action.sbatch",
)
EXPECTED_EXPOSURE = {
    "sidecar_model_row_inputs": ["prompt_token_ids"],
    "sidecar_model_side_state": ["cursor"],
    "text_control_model_row_inputs": ["text_prompt_token_ids"],
    "gold_only_source_fields": [
        "source_id", "renderer_id", "pack_id", "permutation_id",
        "operation_order", "clause_spans",
    ],
    "gold_only_cell_fields": [
        "cell_id", "source_id", "target_action", "target_index", "target_token_id",
    ],
}
WORD = re.compile(r"\w+")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for alias, target in {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
        Path("/etc"): Path("/private/etc"),
    }.items():
        if (
            alias.is_symlink()
            and alias.resolve() == target
            and (absolute == alias or alias in absolute.parents)
        ):
            absolute = target / absolute.relative_to(alias)
            break
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
    return absolute


def read_regular_file(path: Path, *, require_read_only: bool) -> bytes:
    absolute = reject_symlink_components(path)
    metadata = os.lstat(absolute)
    require(stat.S_ISREG(metadata.st_mode), f"input is not a regular file: {absolute}")
    if require_read_only:
        require(metadata.st_mode & 0o222 == 0, f"input is writable: {absolute}")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    return json.loads(
        read_regular_file(path, require_read_only=False).decode("ascii"),
        object_pairs_hook=reject_duplicate_keys,
    )


def load_contract() -> dict[str, Any]:
    require(file_sha256(CONTRACT) == CONTRACT_SHA256, "contract hash mismatch")
    contract = load_json_strict(CONTRACT)
    require(
        contract.get("schema") == "counterfactual_cursor_action_canary_contract_v1",
        "contract schema mismatch",
    )
    require(tuple(item.get("name") for item in contract.get("operations", ())) == OPERATIONS,
            "contract operations mismatch")
    require(contract.get("done", {}).get("name") == "DONE", "contract DONE mismatch")
    require(tuple(contract.get("splits", {})) == SPLITS, "contract split ordering mismatch")
    require(contract.get("evalgrams_n") == 13, "contract evalgram width mismatch")
    return contract


def verify_implementation_identity(identity: dict[str, Any]) -> None:
    require(isinstance(identity, dict) and set(identity) == {"git_commit", "file_sha256"},
            "implementation identity fields mismatch")
    commit = identity["git_commit"]
    hashes = identity["file_sha256"]
    require(isinstance(commit, str) and len(commit) == 40, "implementation commit mismatch")
    require(isinstance(hashes, dict) and set(hashes) == set(IMPLEMENTATION_PATHS),
            "implementation ledger mismatch")
    for relative in IMPLEMENTATION_PATHS:
        expected = hashes[relative]
        require(expected == file_sha256(ROOT / relative), f"live implementation mismatch: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True,
            capture_output=True,
        ).stdout
        require(expected == sha256_bytes(committed), f"committed implementation mismatch: {relative}")


def target_contract(contract: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    token_ids = {item["name"]: item["target_token_id"] for item in contract["operations"]}
    token_ids["DONE"] = contract["done"]["target_token_id"]
    texts = {item["name"]: item["target_text"] for item in contract["operations"]}
    texts["DONE"] = contract["done"]["target_text"]
    return token_ids, texts


def expected_source(
    renderer: dict[str, Any], pack: dict[str, int], order: tuple[str, ...],
) -> tuple[str, list[dict[str, Any]]]:
    prefix = renderer["prefix"].format(**pack)
    suffix = renderer["suffix"].format(**pack)
    texts = [renderer["clauses"][operation].format(**pack) for operation in order]
    source = prefix + renderer["joiner"].join(texts) + suffix
    spans = []
    scan = len(prefix)
    for operation, text in zip(order, texts):
        start = source.find(text, scan)
        require(start >= 0, "could not reconstruct source clause")
        end = start + len(text)
        spans.append({
            "operation": operation,
            "operand": pack[operation],
            "text": text,
            "start": start,
            "end": end,
        })
        scan = end
    return source, spans


def expected_geometry(renderers: int, packs: int) -> dict[str, int]:
    return {
        "renderers": renderers,
        "packs": packs,
        "permutations": 24,
        "cursor_states": 5,
        "sources": renderers * packs * 24,
        "cells": renderers * packs * 24 * 5,
        "content_groups": packs * 24,
        "adjacent_pairs": renderers * packs * 36,
        "training_units": packs * 36,
    }


def audit_latin_packs(packs: list[dict[str, int]], split_name: str) -> None:
    require(len(packs) % 4 == 0, f"{split_name} pack count is not Latin-block aligned")
    marginals = {operation: Counter(pack[operation] for pack in packs) for operation in OPERATIONS}
    require(
        all(marginals[operation] == marginals[OPERATIONS[0]] for operation in OPERATIONS[1:]),
        f"{split_name} operand marginals leak operation identity",
    )
    for block_start in range(0, len(packs), 4):
        block = packs[block_start:block_start + 4]
        starts = {pack["start"] for pack in block}
        require(len(starts) == 1, f"{split_name} Latin block start values mismatch")
        operand_values = {pack[operation] for pack in block for operation in OPERATIONS}
        require(len(operand_values) == 4, f"{split_name} Latin block operand alphabet mismatch")
        for pack in block:
            require(
                {pack[operation] for operation in OPERATIONS} == operand_values,
                f"{split_name} Latin pack is not a row permutation",
            )
        for operation in OPERATIONS:
            require(
                Counter(pack[operation] for pack in block)
                == Counter({value: 1 for value in operand_values}),
                f"{split_name} Latin column is not balanced",
            )


def word_grams(text: str, width: int) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[index:index + width]) for index in range(len(words) - width + 1)}


def audit_document(
    document: dict[str, Any], tokenizer_path: Path, evalgrams_path: Path,
    canary_file_sha256: str | None = None, verify_commit: bool = False,
) -> dict[str, Any]:
    contract = load_contract()
    tokenizer_payload = read_regular_file(
        tokenizer_path, require_read_only=verify_commit,
    )
    evalgrams_payload = read_regular_file(
        evalgrams_path, require_read_only=verify_commit,
    )
    require(sha256_bytes(tokenizer_payload) == contract["tokenizer_sha256"],
            "tokenizer hash mismatch")
    require(sha256_bytes(evalgrams_payload) == contract["evalgrams_sha256"],
            "evalgrams hash mismatch")
    tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    evalgrams = pickle.loads(evalgrams_payload)
    require(
        isinstance(evalgrams, dict)
        and evalgrams.get("n") == contract["evalgrams_n"]
        and isinstance(evalgrams.get("grams"), set),
        "evalgrams payload mismatch",
    )

    expected_top = {
        "schema", "canary_id", "contract_sha256", "tokenizer_sha256",
        "generator_sha256", "implementation_identity", "exposure_contract",
        "label_order", "label_token_ids", "splits", "payload_sha256",
    }
    require(isinstance(document, dict) and set(document) == expected_top, "top-level fields mismatch")
    require(document["schema"] == SCHEMA, "canary schema mismatch")
    require(document["canary_id"] == CANARY_ID, "canary ID mismatch")
    require(document["contract_sha256"] == CONTRACT_SHA256, "canary contract hash mismatch")
    require(document["tokenizer_sha256"] == contract["tokenizer_sha256"],
            "canary tokenizer hash mismatch")
    require(document["generator_sha256"] == file_sha256(GENERATOR), "generator hash mismatch")
    require(document["exposure_contract"] == EXPECTED_EXPOSURE, "exposure contract mismatch")
    require(tuple(document["label_order"]) == LABELS, "label ordering mismatch")
    target_ids, target_texts = target_contract(contract)
    require(document["label_token_ids"] == [target_ids[label] for label in LABELS],
            "label token IDs mismatch")
    require(len(set(document["label_token_ids"])) == 5, "label token IDs are not distinct")
    for label in LABELS:
        encoding = tokenizer.encode(target_texts[label])
        require(encoding.ids == [target_ids[label]], f"label is not one token: {label}")
    require(
        document["payload_sha256"] == sha256_bytes(canonical_json({
            key: value for key, value in document.items() if key != "payload_sha256"
        })),
        "payload hash mismatch",
    )
    if verify_commit:
        verify_implementation_identity(document["implementation_identity"])

    renderers = {item["renderer_id"]: item for item in contract["renderers"]}
    require(set(renderers) == set(range(13)), "renderer IDs mismatch")
    all_renderer_ids = []
    split_numbers: dict[str, set[int]] = {}
    split_sources: dict[str, set[str]] = {}
    split_source_grams: dict[str, set[str]] = {}
    summary = {}
    permutations = list(itertools.permutations(OPERATIONS))
    permutation_index = {order: index for index, order in enumerate(permutations)}

    for split_name in SPLITS:
        split_contract = contract["splits"][split_name]
        split_document = document["splits"].get(split_name)
        require(isinstance(split_document, dict) and set(split_document) == {
            "geometry", "sources_sha256", "cells_sha256", "sources", "cells",
            "content_groups", "adjacent_pairs", "training_units",
        }, f"{split_name} fields mismatch")
        renderer_ids = split_contract["renderer_ids"]
        packs = split_contract["packs"]
        require(all(type(value) is int for value in renderer_ids), "renderer ID type mismatch")
        require(all(renderers[value]["split"] == split_name for value in renderer_ids),
                "renderer split assignment mismatch")
        all_renderer_ids.extend(renderer_ids)
        scalar_values = {value for pack in packs for value in pack.values()}
        require(all(type(value) is int for value in scalar_values), "pack scalar type mismatch")
        audit_latin_packs(packs, split_name)
        split_numbers[split_name] = scalar_values
        require(split_document["geometry"] == expected_geometry(len(renderer_ids), len(packs)),
                f"{split_name} geometry mismatch")

        sources = split_document["sources"]
        cells = split_document["cells"]
        require(split_document["sources_sha256"] == sha256_bytes(canonical_json(sources)),
                f"{split_name} source hash mismatch")
        require(split_document["cells_sha256"] == sha256_bytes(canonical_json(cells)),
                f"{split_name} cell hash mismatch")
        require(len(sources) == split_document["geometry"]["sources"],
                f"{split_name} source count mismatch")
        require(len(cells) == split_document["geometry"]["cells"],
                f"{split_name} cell count mismatch")

        expected_source_fields = {
            "schema", "source_id", "split", "renderer_id", "pack_id",
            "permutation_id", "source_text", "prompt", "prompt_token_ids",
            "operation_order", "clause_spans",
        }
        source_by_id = {}
        observed_source_order = []
        public_ngram_hits = 0
        split_grams = set()
        for source in sources:
            require(isinstance(source, dict) and set(source) == expected_source_fields,
                    f"{split_name} source fields mismatch")
            require(source["schema"] == "counterfactual_cursor_action_source_v1",
                    "source schema mismatch")
            renderer_id = source["renderer_id"]
            pack_id = source["pack_id"]
            permutation_id = source["permutation_id"]
            require(type(renderer_id) is int and renderer_id in renderer_ids, "invalid source renderer")
            require(type(pack_id) is int and 0 <= pack_id < len(packs), "invalid source pack")
            require(type(permutation_id) is int and 0 <= permutation_id < 24,
                    "invalid source permutation")
            source_id = f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
            require(source["source_id"] == source_id, "source ID mismatch")
            require(source_id not in source_by_id, "duplicate source ID")
            observed_source_order.append(source_id)
            expected_text, spans = expected_source(
                renderers[renderer_id], packs[pack_id], permutations[permutation_id],
            )
            require(source["source_text"] == expected_text, "source text mismatch")
            require(source["operation_order"] == list(permutations[permutation_id]),
                    "source operation order mismatch")
            require(source["clause_spans"] == spans, "source clause spans mismatch")
            expected_prompt = expected_text + contract["prompt_suffix"]
            require(source["prompt"] == expected_prompt, "source prompt mismatch")
            require(source["prompt_token_ids"] == tokenizer.encode(expected_prompt).ids,
                    "source prompt tokenization mismatch")
            require(source["prompt_token_ids"], "empty source prompt tokenization")
            require(len(source["prompt_token_ids"]) <= 256, "source prompt exceeds canary limit")
            grams = word_grams(expected_text, contract["evalgrams_n"])
            public_ngram_hits += sum(gram in evalgrams["grams"] for gram in grams)
            split_grams.update(grams)
            source_by_id[source_id] = source

        expected_source_order = [
            f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
            for renderer_id in renderer_ids
            for pack_id in range(len(packs))
            for permutation_id in range(24)
        ]
        require(observed_source_order == expected_source_order, "canonical source ordering mismatch")
        require(public_ngram_hits == 0, f"{split_name} public evalgram overlap")
        split_sources[split_name] = {source["source_text"] for source in sources}
        require(len(split_sources[split_name]) == len(sources), "duplicate rendered source")
        split_source_grams[split_name] = split_grams

        expected_cell_fields = {
            "schema", "cell_id", "source_id", "cursor", "text_prompt",
            "text_prompt_token_ids", "target_action", "target_index", "target_token_id",
        }
        observed_cell_order = []
        target_counts = Counter()
        cell_by_key = {}
        text_length_deltas: dict[str, set[int]] = defaultdict(set)
        for cell in cells:
            require(isinstance(cell, dict) and set(cell) == expected_cell_fields,
                    f"{split_name} cell fields mismatch")
            require(cell["schema"] == "counterfactual_cursor_action_cell_v1",
                    "cell schema mismatch")
            source_id = cell["source_id"]
            cursor = cell["cursor"]
            require(source_id in source_by_id, "cell source is unknown")
            require(type(cursor) is int and 0 <= cursor < 5, "cell cursor mismatch")
            cell_id = f"{source_id}-c{cursor}"
            require(cell["cell_id"] == cell_id, "cell ID mismatch")
            require((source_id, cursor) not in cell_by_key, "duplicate source/cursor cell")
            source = source_by_id[source_id]
            order = tuple(source["operation_order"])
            target = order[cursor] if cursor < 4 else "DONE"
            require(cell["target_action"] == target, "cell target action mismatch")
            require(cell["target_index"] == LABELS.index(target), "cell target index mismatch")
            require(cell["target_token_id"] == target_ids[target], "cell target token mismatch")
            expected_text_prompt = source["source_text"] + contract["text_cursor_suffixes"][cursor]
            require(cell["text_prompt"] == expected_text_prompt, "text-control prompt mismatch")
            expected_text_ids = tokenizer.encode(expected_text_prompt).ids
            require(cell["text_prompt_token_ids"] == expected_text_ids,
                    "text-control tokenization mismatch")
            require(
                tokenizer.encode(source["prompt"] + target_texts[target]).ids
                == source["prompt_token_ids"] + [target_ids[target]],
                "sidecar target append mismatch",
            )
            require(
                tokenizer.encode(expected_text_prompt + target_texts[target]).ids
                == expected_text_ids + [target_ids[target]],
                "text-control target append mismatch",
            )
            text_length_deltas[source_id].add(len(expected_text_ids) - len(source["prompt_token_ids"]))
            observed_cell_order.append(cell_id)
            target_counts[target] += 1
            cell_by_key[(source_id, cursor)] = cell

        expected_cell_order = [
            f"{source_id}-c{cursor}" for source_id in expected_source_order for cursor in range(5)
        ]
        require(observed_cell_order == expected_cell_order, "canonical cell ordering mismatch")
        require(target_counts == Counter({label: len(sources) for label in LABELS}),
                "cell targets are unbalanced")
        require(all(values == {5} for values in text_length_deltas.values()),
                "text cursor suffixes are not token-length matched")

        expected_content_groups = [
            {
                "pack_id": pack_id,
                "permutation_id": permutation_id,
                "source_ids": [
                    f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                    for renderer_id in renderer_ids
                ],
            }
            for pack_id in range(len(packs)) for permutation_id in range(24)
        ]
        require(split_document["content_groups"] == expected_content_groups,
                "content group map mismatch")

        expected_adjacent = []
        expected_units = []
        for pack_id in range(len(packs)):
            for permutation_id, order in enumerate(permutations):
                for swap_index in range(3):
                    swapped = list(order)
                    swapped[swap_index], swapped[swap_index + 1] = (
                        swapped[swap_index + 1], swapped[swap_index]
                    )
                    other_id = permutation_index[tuple(swapped)]
                    if permutation_id >= other_id:
                        continue
                    pair_ids = []
                    for renderer_id in renderer_ids:
                        pair_id = (
                            f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-"
                            f"p{permutation_id:02d}-p{other_id:02d}-s{swap_index}"
                        )
                        expected_adjacent.append({
                            "pair_id": pair_id,
                            "pack_id": pack_id,
                            "renderer_id": renderer_id,
                            "swap_index": swap_index,
                            "left_source_id": (
                                f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-"
                                f"p{permutation_id:02d}"
                            ),
                            "right_source_id": (
                                f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{other_id:02d}"
                            ),
                        })
                        pair_ids.append(pair_id)
                    expected_units.append({
                        "unit_id": (
                            f"{split_name}-k{pack_id:02d}-p{permutation_id:02d}-"
                            f"p{other_id:02d}-s{swap_index}"
                        ),
                        "pack_id": pack_id,
                        "swap_index": swap_index,
                        "left_permutation_id": permutation_id,
                        "right_permutation_id": other_id,
                        "adjacent_pair_ids": pair_ids,
                    })
        require(split_document["adjacent_pairs"] == expected_adjacent,
                "adjacent pair map mismatch")
        require(split_document["training_units"] == expected_units,
                "training unit map mismatch")

        source_only = sum(1 for source_id in source_by_id for cursor in range(5)
                          if cell_by_key[(source_id, 0)]["target_action"]
                          == cell_by_key[(source_id, cursor)]["target_action"])
        cursor_counts = defaultdict(Counter)
        for cell in cells:
            cursor_counts[cell["cursor"]][cell["target_action"]] += 1
        cursor_only = sum(max(counts.values()) for counts in cursor_counts.values())
        require(source_only == len(sources), "source-only ceiling mismatch")
        require(cursor_only == 2 * len(sources), "cursor-only ceiling mismatch")
        summary[split_name] = {
            **split_document["geometry"],
            "public_evalgram_hits": public_ngram_hits,
            "source_only_ceiling": source_only,
            "cursor_only_ceiling": cursor_only,
            "target_counts": dict(target_counts),
            "max_prompt_tokens": max(len(source["prompt_token_ids"]) for source in sources),
        }

    require(sorted(all_renderer_ids) == list(range(13)), "renderer split disjointness mismatch")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            require(not (split_numbers[left] & split_numbers[right]), "numeric packs overlap across splits")
            require(not (split_sources[left] & split_sources[right]), "exact sources overlap across splits")
    # Exact 13-gram separation is deliberately not required across synthetic
    # splits because shared task syntax and operation names are the invariant.
    cross_split_13gram_counts = {
        f"{left}__{right}": len(split_source_grams[left] & split_source_grams[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1:]
    }

    return {
        "schema": "counterfactual_cursor_action_canary_audit_v1",
        "canary_id": CANARY_ID,
        "canary_file_sha256": canary_file_sha256,
        "canary_payload_sha256": document["payload_sha256"],
        "contract_sha256": CONTRACT_SHA256,
        "tokenizer_sha256": contract["tokenizer_sha256"],
        "evalgrams_sha256": contract["evalgrams_sha256"],
        "auditor_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_identity": document["implementation_identity"],
        "split_summary": summary,
        "cross_split_13gram_counts": cross_split_13gram_counts,
        "all_checks_pass": True,
    }


def write_exclusive_read_only(path: Path, document: dict[str, Any]) -> None:
    path = Path(os.path.abspath(path))
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = reject_symlink_components(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    payload = json.dumps(document, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def require_regular_read_only(path: Path) -> None:
    absolute = reject_symlink_components(path)
    mode = os.lstat(absolute).st_mode
    require(stat.S_ISREG(mode), "canary input is not a regular file")
    require(mode & 0o222 == 0, "canary input is not read-only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--evalgrams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-commit", action="store_true")
    arguments = parser.parse_args()
    require_regular_read_only(arguments.canary)
    require_regular_read_only(arguments.tokenizer)
    require_regular_read_only(arguments.evalgrams)
    canary_payload = read_regular_file(arguments.canary, require_read_only=True)
    document = json.loads(
        canary_payload.decode("ascii"), object_pairs_hook=reject_duplicate_keys,
    )
    report = audit_document(
        document, arguments.tokenizer, arguments.evalgrams,
        canary_file_sha256=sha256_bytes(canary_payload),
        verify_commit=arguments.verify_commit,
    )
    write_exclusive_read_only(arguments.out, report)
    print(
        f"[ccaa-canary-audit] wrote {arguments.out} "
        f"canary_sha256={report['canary_file_sha256']} all_checks_pass=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
