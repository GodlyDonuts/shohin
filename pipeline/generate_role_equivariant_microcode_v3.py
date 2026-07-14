#!/usr/bin/env python3
"""Build anchor-preserving semantic and register-permutation microcode views."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path

from generate_categorical_microcode_equivalence_v2 import make_view
from generate_latent_operator_v1 import TRAIN_DOMAINS, render_question


SEMANTIC_VIEWS = ("anchor", "paraphrase_a", "paraphrase_b")
WORD = re.compile(r"\w+")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(text):
    return " ".join(WORD.findall(str(text).lower()))


def swap_key(value, keys):
    return keys[1] if value == keys[0] else keys[0] if value == keys[1] else value


def render_train_query(query, keys, item):
    kind = query["kind"]
    if kind == "read":
        return "What is the final {} total?".format(query["key"])
    if kind == "sum":
        return "What is the combined number of {} in {} and {}?".format(item, keys[0], keys[1])
    if kind == "difference":
        return "How many more {} are in {} than in {}?".format(item, query["high"], query["low"])
    raise ValueError("unknown query kind {}".format(kind))


def permute_source(source):
    """Apply the exact two-register automorphism while keeping key order fixed."""
    row = copy.deepcopy(source)
    keys = list(row["keys"])
    if len(keys) != 2:
        raise ValueError("role permutation requires exactly two keys")
    row["initial"] = {keys[0]: int(source["initial"][keys[1]]), keys[1]: int(source["initial"][keys[0]])}
    operations = []
    for operation in source["operations"]:
        transformed = dict(operation)
        for field in ("target", "source", "left", "right"):
            if field in transformed:
                transformed[field] = swap_key(transformed[field], keys)
        operations.append(transformed)
    row["operations"] = operations
    query = {key: value for key, value in source["query"].items() if key not in {"text"}}
    for field in ("key", "high", "low"):
        if field in query:
            query[field] = swap_key(query[field], keys)
    query["text"] = render_train_query(query, keys, next(domain[2] for domain in TRAIN_DOMAINS if domain[0] == source["family"]))
    row["query"] = query
    # The state-machine automorphism preserves the selected scalar answer.
    row["answer"] = str(source["answer"])
    return row


def anchor_view(source, index, permutation):
    domain = next(domain for domain in TRAIN_DOMAINS if domain[0] == source["family"])
    query = dict(source["query"])
    query["text"] = render_train_query(query, list(source["keys"]), domain[2])
    question = render_question(
        domain, source["initial"], source["operations"], query, False, source["reference"],
    )
    if not permutation and question != source["question"]:
        raise ValueError("anchor renderer did not reproduce immutable source row {}".format(index))
    return {
        **copy.deepcopy(source),
        "question": question,
        "query": query,
        "response": "The answer is {}.".format(source["answer"]),
        "answer": str(source["answer"]),
    }


def decorate(row, index, semantic_view, permutation, source_index=None):
    output = dict(row)
    output.update({
        "source": "role_equivariant_microcode_v3",
        "training_group": "role_equivariant_microcode",
        "equivalence_id": "crec-{:06d}".format(index),
        "semantic_view": semantic_view,
        "register_permutation": int(permutation),
        "reference": "CREC-{:06d}-{}-{}".format(index, semantic_view, permutation),
        "source_index": int(index if source_index is None else source_index),
        "heldout": False,
    })
    return output


def make_rows(source, index, source_index=None):
    rows = []
    for permutation in (0, 1):
        transformed = permute_source(source) if permutation else copy.deepcopy(source)
        rows.append(decorate(
            anchor_view(transformed, index, permutation), index, "anchor", permutation, source_index,
        ))
        rows.append(decorate(
            make_view(transformed, index, 0), index, "paraphrase_a", permutation, source_index,
        ))
        rows.append(decorate(
            make_view(transformed, index, 1), index, "paraphrase_b", permutation, source_index,
        ))
    return rows


def select_rows(source_rows, programs, eval_questions):
    """Select complete unique groups without changing any rendered example."""
    rows = []
    seen_questions = set()
    selected_source_indices = []
    skipped = collections.Counter()
    for source_index, source in enumerate(source_rows):
        selected_index = len(selected_source_indices)
        candidate_rows = make_rows(source, selected_index, source_index)
        candidate_questions = [normalized_question(row["question"]) for row in candidate_rows]
        candidate_set = set(candidate_questions)
        if len(candidate_set) != len(candidate_questions):
            skipped["duplicate_within_group"] += 1
            continue
        if candidate_set & eval_questions:
            skipped["exact_eval_prompt"] += 1
            continue
        if candidate_set & seen_questions:
            skipped["duplicate_prior_group"] += 1
            continue
        rows.extend(candidate_rows)
        seen_questions.update(candidate_set)
        selected_source_indices.append(source_index)
        if len(selected_source_indices) == programs:
            break
    if len(selected_source_indices) != programs:
        raise ValueError(
            "only selected {} unique groups from {} source rows".format(
                len(selected_source_indices), len(source_rows),
            )
        )
    return rows, selected_source_indices, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--eval", required=True, help="held-out prompt JSONL; answers are never read")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--programs", type=int, default=48000)
    args = parser.parse_args()
    if args.programs <= 0:
        raise SystemExit("programs must be positive")
    if Path(args.out).exists() or Path(args.report).exists():
        raise SystemExit("refusing existing output")
    source_rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()]
    eval_rows = [json.loads(line) for line in Path(args.eval).read_text().splitlines() if line.strip()]
    eval_questions = {
        normalized_question(row.get("question", "")) for row in eval_rows
        if normalized_question(row.get("question", ""))
    }
    if len(source_rows) < args.programs:
        raise SystemExit("source has fewer rows than requested programs")
    try:
        rows, selected_source_indices, skipped = select_rows(source_rows, args.programs, eval_questions)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "build": "role_equivariant_microcode_v3",
        "source": str(Path(args.source).resolve()),
        "source_sha256": sha256(args.source),
        "eval": str(Path(args.eval).resolve()),
        "eval_sha256": sha256(args.eval),
        "eval_prompts_loaded": len(eval_questions),
        "programs": args.programs,
        "source_rows_available": len(source_rows),
        "source_rows_scanned": selected_source_indices[-1] + 1,
        "selected_source_indices_sha256": hashlib.sha256(
            json.dumps(selected_source_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "skipped_groups": dict(sorted(skipped.items())),
        "rows": len(rows),
        "semantic_views": dict(collections.Counter(row["semantic_view"] for row in rows)),
        "register_permutations": dict(collections.Counter(str(row["register_permutation"]) for row in rows)),
        "depths": dict(collections.Counter(str(row["depth"]) for row in rows)),
        "data_sha256": sha256(args.out),
        "claim_boundary": (
            "Training-only anchor, paraphrase, and exact register-automorphism views. Held-out "
            "prompt text is read only to reject exact collisions; no evaluation answer is consumed."
        ),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
