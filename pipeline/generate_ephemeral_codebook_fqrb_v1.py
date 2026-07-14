#!/usr/bin/env python3
"""Build a late-binding finite-query residual-basis curriculum.

This is a conditional successor to FQRB, not a generic SFT corpus.  The
source-free residual tape still carries ``donor + edited - base``.  What
changes is the readout contract: the suffix contains a fresh, per-world
arbitrary codebook mapping semantic FQRB labels to opaque code words.  A
successful model must recover the semantic answer from the tape and then use
the current codebook, rather than associate a fixed question template with a
fixed output token.

The builder is CPU-only and solver-derived.  It never loads a model or writes
to a training directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path

from generate_finite_query_residual_basis_v1 import (
    QUERY_KINDS,
    TWO_DIGIT_VALUES,
    build as build_fqrb,
    label,
    ngrams,
    render_bundle,
    source_bundle_key,
)


# Common, nonnumeric words deliberately avoid a semantic relationship to the
# categories they encode.  Sixteen choices yield a large disjoint-permutation
# space while keeping every code word in the normal tokenizer vocabulary.
CODE_TAGS = (
    "amber", "birch", "cedar", "dingo", "ember", "fable", "garnet", "harbor",
    "indigo", "juniper", "kestrel", "linden", "marble", "nectar", "onyx", "piper",
)
CANONICAL_LABELS = tuple(sorted({
    label(primary, secondary, kind)
    for primary, secondary in ((-11, -11), (-10, -9), (-9, 0), (-1, 1), (0, 0), (1, -1), (9, 10), (10, 9), (11, 11))
    for kind in QUERY_KINDS
}))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def codebook_key(mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    if set(mapping) != set(CANONICAL_LABELS) or len(set(mapping.values())) != len(CANONICAL_LABELS):
        raise ValueError("invalid codebook domain or duplicate code words")
    return tuple((semantic, mapping[semantic]) for semantic in CANONICAL_LABELS)


def render_codebook(mapping: dict[str, str]) -> str:
    return "\n".join("{} -> code={}".format(semantic, mapping[semantic]) for semantic in CANONICAL_LABELS)


def render_suffix(query: str, mapping: dict[str, str]) -> str:
    return "Binding table:\n{}\nQuestion: {}\nReturn exactly the matching code.\nAnswer:".format(
        render_codebook(mapping), query
    )


def mapping_for_group(rng: random.Random, forbidden: set[tuple[tuple[str, str], ...]]) -> dict[str, str]:
    for _ in range(10_000):
        tags = rng.sample(CODE_TAGS, len(CANONICAL_LABELS))
        mapping = dict(zip(CANONICAL_LABELS, tags))
        if codebook_key(mapping) not in forbidden:
            return mapping
    raise RuntimeError("could not sample a disjoint ephemeral codebook")


def swapped_mapping(mapping: dict[str, str], semantic: str, index: int) -> tuple[dict[str, str], str]:
    """Swap the active semantic label with a deterministic different label."""
    if semantic not in mapping:
        raise ValueError("semantic label absent from codebook")
    alternatives = [candidate for candidate in CANONICAL_LABELS if candidate != semantic]
    other = alternatives[index % len(alternatives)]
    result = dict(mapping)
    result[semantic], result[other] = result[other], result[semantic]
    return result, other


def validate_row(row: dict) -> None:
    required = (
        "semantic_response", "semantic_counterfactual_response", "codebook", "codebook_swap",
        "codebook_swap_suffix_prompt", "codebook_swap_response", "query_kind",
    )
    if row.get("mechanism") != "ephemeral_codebook_fqrb_v1" or any(field not in row for field in required):
        raise ValueError("not an ephemeral-codebook FQRB row")
    state = row["state"]
    semantic = label(state["target"]["primary"], state["target"]["secondary"], row["query_kind"])
    counter = label(state["counterfactual_target"]["primary"], state["counterfactual_target"]["secondary"], row["query_kind"])
    if row["semantic_response"] != semantic or row["semantic_counterfactual_response"] != counter:
        raise ValueError("semantic target is not solver-derived")
    mapping, swap = row["codebook"], row["codebook_swap"]
    codebook_key(mapping)
    codebook_key(swap)
    if row["response"] != "code=" + mapping[semantic]:
        raise ValueError("normal code does not bind the semantic answer")
    if row["counterfactual_response"] != "code=" + mapping[counter] or row["response"] == row["counterfactual_response"]:
        raise ValueError("counterfactual code contract is invalid")
    if row["codebook_swap_response"] != "code=" + swap[semantic] or row["codebook_swap_response"] == row["response"]:
        raise ValueError("codebook intervention did not alter the answer")
    if render_codebook(mapping) not in row["suffix_prompt"] or render_codebook(swap) not in row["codebook_swap_suffix_prompt"]:
        raise ValueError("suffix does not bind its codebook")
    source_text = "\n".join(row[field] for field in ("base_source", "edited_source", "donor_source"))
    for token in source_text.replace("-", " -").split():
        if token.lstrip("-").isdigit() and token in row["suffix_prompt"]:
            raise ValueError("suffix leaked a source number")


def wrap_groups(rows: list[dict], rng: random.Random, forbidden_codebooks: set[tuple[tuple[str, str], ...]]) -> tuple[list[dict], set[tuple[tuple[str, str], ...]]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["basis_id"], []).append(row)
    result, used = [], set(forbidden_codebooks)
    for group_index, (basis_id, group) in enumerate(grouped.items()):
        if len(group) != len(QUERY_KINDS) or {row["query_kind"] for row in group} != set(QUERY_KINDS):
            raise ValueError("FQRB group {} is incomplete".format(basis_id))
        mapping = mapping_for_group(rng, used)
        fingerprint = codebook_key(mapping)
        used.add(fingerprint)
        for query_index, source_row in enumerate(sorted(group, key=lambda item: QUERY_KINDS.index(item["query_kind"]))):
            row = dict(source_row)
            semantic, counter = source_row["response"], source_row["counterfactual_response"]
            swap, decoy = swapped_mapping(mapping, semantic, group_index + query_index)
            row.update({
                "mechanism": "ephemeral_codebook_fqrb_v1",
                "basis_mode": "multi_consumer_ephemeral_codebook",
                "semantic_response": semantic,
                "semantic_counterfactual_response": counter,
                "codebook": mapping,
                "codebook_swap": swap,
                "codebook_swap_decoy": decoy,
                "suffix_prompt": render_suffix(source_row["suffix_prompt"].split("Question: ", 1)[1].removesuffix("\nAnswer:"), mapping),
                "codebook_swap_suffix_prompt": render_suffix(source_row["suffix_prompt"].split("Question: ", 1)[1].removesuffix("\nAnswer:"), swap),
                "response": "code=" + mapping[semantic],
                "counterfactual_response": "code=" + mapping[counter],
                "codebook_swap_response": "code=" + swap[semantic],
                "axes": {**source_row["axes"], "ephemeral_codebook": True, "codebook_heldout": bool(forbidden_codebooks)},
            })
            validate_row(row)
            result.append(row)
    return result, used - forbidden_codebooks


def audit(train: list[dict], heldout: list[dict]) -> dict:
    train_prompts = {render_bundle(row) for row in train}
    heldout_prompts = {render_bundle(row) for row in heldout}
    train_groups = Counter(row["basis_id"] for row in train)
    heldout_groups = Counter(row["basis_id"] for row in heldout)
    train_codebooks = {codebook_key(row["codebook"]) for row in train}
    heldout_codebooks = {codebook_key(row["codebook"]) for row in heldout}
    train_source_bundles = {source_bundle_key(row) for row in train}
    heldout_source_bundles = {source_bundle_key(row) for row in heldout}
    # Full prompt 13-grams necessarily contain repeated binding-table syntax.
    # Report that structural overlap honestly, and separately require that the
    # source-and-question surface (without the arbitrary table) has no hit.
    def semantic_surface(row: dict) -> str:
        return "\n".join((row["base_source"], row["edited_source"], row["donor_source"], row["semantic_response"], row["query_kind"]))
    train_semantic_grams = set().union(*(ngrams(semantic_surface(row)) for row in train))
    heldout_semantic_grams = set().union(*(ngrams(semantic_surface(row)) for row in heldout))
    full_train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    full_heldout_grams = set().union(*(ngrams(prompt) for prompt in heldout_prompts))
    return {
        "duplicate_train_prompts": len(train) - len(train_prompts),
        "duplicate_heldout_prompts": len(heldout) - len(heldout_prompts),
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_exact_source_bundle_hits": len(train_source_bundles & heldout_source_bundles),
        "train_heldout_codebook_hits": len(train_codebooks & heldout_codebooks),
        "train_heldout_semantic_13gram_hits": len(train_semantic_grams & heldout_semantic_grams),
        "expected_binding_syntax_13gram_hits": len(full_train_grams & full_heldout_grams),
        "train_groups": len(train_groups),
        "heldout_groups": len(heldout_groups),
        "bad_train_group_cardinality": sum(count != len(QUERY_KINDS) for count in train_groups.values()),
        "bad_heldout_group_cardinality": sum(count != len(QUERY_KINDS) for count in heldout_groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-groups", type=int, default=12_000)
    parser.add_argument("--heldout-groups", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071418)
    args = parser.parse_args()
    paths = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if args.train_groups <= 0 or args.heldout_groups <= 0 or any(path.exists() for path in paths):
        raise SystemExit("group counts must be positive and all output paths fresh")
    train_base = build_fqrb(args.train_groups, args.seed, "train", TWO_DIGIT_VALUES, 90)
    heldout_base = build_fqrb(args.heldout_groups, args.seed + 1, "heldout", TWO_DIGIT_VALUES, 90, language_heldout=True)
    train, train_codebooks = wrap_groups(train_base, random.Random(args.seed + 2), set())
    heldout, _ = wrap_groups(heldout_base, random.Random(args.seed + 3), train_codebooks)
    report = audit(train, heldout)
    required_zero = (
        "duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits",
        "train_heldout_exact_source_bundle_hits", "train_heldout_codebook_hits",
        "train_heldout_semantic_13gram_hits", "bad_train_group_cardinality", "bad_heldout_group_cardinality",
    )
    if any(report[key] for key in required_zero):
        raise SystemExit("ephemeral-codebook split audit failed: {}".format(report))
    train_codes = {row["response"] for row in train}
    heldout_codes = {row[field] for row in heldout for field in ("response", "counterfactual_response", "codebook_swap_response")}
    if not heldout_codes <= train_codes:
        raise SystemExit("held-out code words are not supported by training")
    write_jsonl(paths[0], train)
    write_jsonl(paths[1], heldout)
    report.update({
        "audit": "ephemeral_codebook_fqrb_v1",
        "mechanism": "ephemeral_codebook_fqrb_v1",
        "claim_boundary": "CPU-only conditional data admission. A later pass could establish only late-bound finite latent interrogation, not general reasoning.",
        "train_rows": len(train), "heldout_rows": len(heldout),
        "train_sha256": sha256_file(paths[0]), "heldout_sha256": sha256_file(paths[1]),
        "query_kinds": list(QUERY_KINDS), "code_tags": list(CODE_TAGS),
        "combined_heldout_axes": ["source_tuple", "language", "ephemeral_codebook"],
    })
    paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
