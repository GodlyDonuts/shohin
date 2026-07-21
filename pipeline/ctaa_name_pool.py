"""Deterministic fixed-token-width opaque name pools for CTAA boards."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from tokenizers import Tokenizer


SPLITS = ("train", "development", "confirmation")
BASE_PATTERN = re.compile(r"[a-z]{4,10}")
RESERVED = {
    "action",
    "after",
    "answer",
    "before",
    "card",
    "code",
    "confirmation",
    "control",
    "development",
    "event",
    "first",
    "initial",
    "left",
    "middle",
    "query",
    "register",
    "right",
    "schedule",
    "second",
    "state",
    "stop",
    "third",
    "train",
    "value",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_context_width(tokenizer: Tokenizer, value: str) -> tuple[int, ...]:
    contexts = (
        value,
        f" {value}",
        f"[{value}]",
        f"({value})",
        f"={value};",
        f",{value},",
        f"|{value}|",
    )
    decorations = (0, 0, 2, 2, 2, 2, 2)
    return tuple(
        len(tokenizer.encode(context).ids) - decoration
        for context, decoration in zip(contexts, decorations, strict=True)
    )


def build_name_pools(
    tokenizer_path: Path,
    *,
    per_split: int = 256,
) -> dict[str, tuple[str, ...]]:
    if per_split < 7:
        raise ValueError("CTAA name pool is too small")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha = sha256_file(tokenizer_path)
    bases = [
        value
        for value in tokenizer.get_vocab()
        if BASE_PATTERN.fullmatch(value)
        and value not in RESERVED
        and _fixed_context_width(tokenizer, value) == (1,) * 7
    ]
    bases.sort(key=lambda value: hashlib.sha256(
        f"{tokenizer_sha}|base|{value}".encode()
    ).digest())
    base_count = math.ceil((1 + math.sqrt(1 + 8 * per_split)) / 2) + 4
    required_bases = base_count * len(SPLITS)
    if len(bases) < required_bases:
        raise ValueError("CTAA tokenizer has insufficient opaque base tokens")
    pools = {}
    for split_index, split in enumerate(SPLITS):
        split_bases = bases[
            split_index * base_count : (split_index + 1) * base_count
        ]
        compounds = [
            f"{first}-{second}"
            for first_index, first in enumerate(split_bases)
            for second in split_bases[first_index + 1 :]
            if _fixed_context_width(tokenizer, f"{first}-{second}") == (3,) * 7
        ]
        compounds.sort(key=lambda value: hashlib.sha256(
            f"{tokenizer_sha}|compound|{split}|{value}".encode()
        ).digest())
        if len(compounds) < per_split:
            raise ValueError("CTAA tokenizer has insufficient opaque compounds")
        pools[split] = tuple(compounds[:per_split])
    return pools


def audit_name_pools(
    tokenizer_path: Path,
    pools: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    widths = {
        split: sorted({
            _fixed_context_width(tokenizer, value)
            for value in values
        })
        for split, values in pools.items()
    }
    sets = {split: set(values) for split, values in pools.items()}
    overlaps = {
        f"{left}_{right}": len(sets[left] & sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    gates = {
        "all_splits_present": set(pools) == set(SPLITS),
        "every_name_has_three_tokens_in_every_context": all(
            values == [(3,) * 7] for values in widths.values()
        ),
        "split_pools_are_disjoint": all(value == 0 for value in overlaps.values()),
        "names_are_unique_within_split": all(
            len(values) == len(set(values)) for values in pools.values()
        ),
    }
    return {
        "schema": "ctaa_fixed_width_name_pool_audit_v1",
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "pool_sizes": {split: len(values) for split, values in pools.items()},
        "context_widths": {
            split: [list(value) for value in values]
            for split, values in widths.items()
        },
        "overlaps": overlaps,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
