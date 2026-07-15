#!/usr/bin/env python3
"""Generate the immutable R12 cursor-action neural-canary data document."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


SCHEMA = "counterfactual_cursor_action_canary_v1"
CANARY_ID = "ccaa-neural-canary-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
LABELS = OPERATIONS + ("DONE",)
SPLITS = ("train", "development", "confirmation")
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path(__file__).resolve().with_name(
    "counterfactual_cursor_action_canary_contract_v1.json"
)
CONTRACT_SHA256 = "d767e2bf405364d929d6aab0b1e202d643b974e62e71adbddde012a09255a49b"
IMPLEMENTATION_PATHS = (
    "R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md",
    "pipeline/counterfactual_cursor_action_canary_contract_v1.json",
    "pipeline/generate_counterfactual_cursor_action_canary.py",
    "pipeline/audit_counterfactual_cursor_action_canary.py",
    "pipeline/test_counterfactual_cursor_action_canary.py",
)
EXPOSURE_CONTRACT = {
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


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_contract() -> dict[str, Any]:
    if file_sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("canary contract hash mismatch")
    contract = json.loads(
        CONTRACT_PATH.read_text(encoding="ascii"), object_pairs_hook=reject_duplicate_keys,
    )
    if contract.get("schema") != "counterfactual_cursor_action_canary_contract_v1":
        raise ValueError("canary contract schema mismatch")
    if tuple(item.get("name") for item in contract.get("operations", ())) != OPERATIONS:
        raise ValueError("canary operation alphabet mismatch")
    if contract.get("done", {}).get("name") != "DONE":
        raise ValueError("canary DONE label mismatch")
    if tuple(contract.get("splits", {})) != SPLITS:
        raise ValueError("canary split ordering mismatch")
    return contract


def require_clean_implementation() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *IMPLEMENTATION_PATHS],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("refusing persistent canary data from a dirty implementation surface")


def implementation_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "file_sha256": {
            relative: file_sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
    }


def target_contract(contract: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    token_ids = {item["name"]: item["target_token_id"] for item in contract["operations"]}
    token_ids["DONE"] = contract["done"]["target_token_id"]
    texts = {item["name"]: item["target_text"] for item in contract["operations"]}
    texts["DONE"] = contract["done"]["target_text"]
    return token_ids, texts


def render_source(
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
        if start < 0:
            raise AssertionError("rendered clause is absent")
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


def encode_exact(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text).ids
    if not ids:
        raise ValueError("tokenizer produced an empty sequence")
    return ids


def generate_split(
    split_name: str, contract: dict[str, Any], tokenizer: Tokenizer,
) -> dict[str, Any]:
    split = contract["splits"][split_name]
    renderers = {item["renderer_id"]: item for item in contract["renderers"]}
    permutations = list(itertools.permutations(OPERATIONS))
    permutation_index = {order: index for index, order in enumerate(permutations)}
    target_ids, target_texts = target_contract(contract)
    prompt_suffix = contract["prompt_suffix"]
    text_suffixes = contract["text_cursor_suffixes"]
    sources = []
    cells = []

    for renderer_id in split["renderer_ids"]:
        renderer = renderers[renderer_id]
        for pack_id, pack in enumerate(split["packs"]):
            for permutation_id, order in enumerate(permutations):
                source_id = f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                source_text, spans = render_source(renderer, pack, order)
                prompt = source_text + prompt_suffix
                prompt_ids = encode_exact(tokenizer, prompt)
                sources.append({
                    "schema": "counterfactual_cursor_action_source_v1",
                    "source_id": source_id,
                    "split": split_name,
                    "renderer_id": renderer_id,
                    "pack_id": pack_id,
                    "permutation_id": permutation_id,
                    "source_text": source_text,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "operation_order": list(order),
                    "clause_spans": spans,
                })
                for cursor in range(5):
                    target = order[cursor] if cursor < 4 else "DONE"
                    text_prompt = source_text + text_suffixes[cursor]
                    text_prompt_ids = encode_exact(tokenizer, text_prompt)
                    target_id = target_ids[target]
                    for candidate_prompt, candidate_ids in (
                        (prompt, prompt_ids), (text_prompt, text_prompt_ids),
                    ):
                        extended = encode_exact(tokenizer, candidate_prompt + target_texts[target])
                        if extended != candidate_ids + [target_id]:
                            raise ValueError(
                                f"target token is not an append-stable singleton: {source_id} c={cursor}"
                            )
                    cells.append({
                        "schema": "counterfactual_cursor_action_cell_v1",
                        "cell_id": f"{source_id}-c{cursor}",
                        "source_id": source_id,
                        "cursor": cursor,
                        "text_prompt": text_prompt,
                        "text_prompt_token_ids": text_prompt_ids,
                        "target_action": target,
                        "target_index": LABELS.index(target),
                        "target_token_id": target_id,
                    })

    content_groups = []
    for pack_id in range(len(split["packs"])):
        for permutation_id in range(len(permutations)):
            content_groups.append({
                "pack_id": pack_id,
                "permutation_id": permutation_id,
                "source_ids": [
                    f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                    for renderer_id in split["renderer_ids"]
                ],
            })

    adjacent_pairs = []
    training_units = []
    for pack_id in range(len(split["packs"])):
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
                for renderer_id in split["renderer_ids"]:
                    left = f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{permutation_id:02d}"
                    right = f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-p{other_id:02d}"
                    pair_id = (
                        f"{split_name}-r{renderer_id:02d}-k{pack_id:02d}-"
                        f"p{permutation_id:02d}-p{other_id:02d}-s{swap_index}"
                    )
                    adjacent_pairs.append({
                        "pair_id": pair_id,
                        "pack_id": pack_id,
                        "renderer_id": renderer_id,
                        "swap_index": swap_index,
                        "left_source_id": left,
                        "right_source_id": right,
                    })
                    pair_ids.append(pair_id)
                training_units.append({
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

    return {
        "geometry": {
            "renderers": len(split["renderer_ids"]),
            "packs": len(split["packs"]),
            "permutations": 24,
            "cursor_states": 5,
            "sources": len(sources),
            "cells": len(cells),
            "content_groups": len(content_groups),
            "adjacent_pairs": len(adjacent_pairs),
            "training_units": len(training_units),
        },
        "sources_sha256": sha256_bytes(canonical_json(sources)),
        "cells_sha256": sha256_bytes(canonical_json(cells)),
        "sources": sources,
        "cells": cells,
        "content_groups": content_groups,
        "adjacent_pairs": adjacent_pairs,
        "training_units": training_units,
    }


def generate_document(tokenizer_path: Path, bind_identity: bool = True) -> dict[str, Any]:
    contract = load_contract()
    if file_sha256(tokenizer_path) != contract["tokenizer_sha256"]:
        raise ValueError("canary tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    document = {
        "schema": SCHEMA,
        "canary_id": CANARY_ID,
        "contract_sha256": CONTRACT_SHA256,
        "tokenizer_sha256": contract["tokenizer_sha256"],
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_identity": implementation_identity() if bind_identity else None,
        "exposure_contract": EXPOSURE_CONTRACT,
        "label_order": list(LABELS),
        "label_token_ids": [
            *[item["target_token_id"] for item in contract["operations"]],
            contract["done"]["target_token_id"],
        ],
        "splits": {
            split_name: generate_split(split_name, contract, tokenizer)
            for split_name in SPLITS
        },
    }
    document["payload_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in document.items() if key != "payload_sha256"
    }))
    return document


def write_exclusive_read_only(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    payload = json.dumps(document, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require_clean_implementation()
    document = generate_document(arguments.tokenizer)
    write_exclusive_read_only(arguments.out, document)
    counts = {name: split["geometry"]["cells"] for name, split in document["splits"].items()}
    print(
        f"[ccaa-canary] wrote {arguments.out} cells={counts} "
        f"payload_sha256={document['payload_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
