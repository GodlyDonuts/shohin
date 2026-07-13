#!/usr/bin/env python3
"""Independent admission audit for Counterfactual Workspace Induction data.

The builder derives reflection examples from an admitted static-tape transition
corpus. This auditor intentionally does not import that builder: it reparses
the serialized state, rederives the legal update and semantic verdict, binds
every response-control field, checks train/held-out separation, and confirms
that held-out base/counterfactual worlds remain paired.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from counterfactual_workspace_protocol import (
    FOIL_KINDS,
    reflection_prompt,
    reflection_response,
    validate_reflection_record,
)
from digitwise_factor_protocol import (
    canonical_register,
    canonical_tape,
    local_context,
    parse_register,
    parse_tape,
)


WORD = re.compile(r"\w+")
LABEL_PERMUTATION = {
    "carry": "result_digit",
    "result_digit": "program_counter",
    "program_counter": "tape",
    "tape": "carry",
    "none": "none",
}
SYNTAX_ONLY_RESPONSE = "syntax=valid;shape=tape_register;slots=complete;mode=local"
RESPONSE_FIELDS = ("response", "response_label_permuted", "response_syntax_only")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_contexts():
    """Independently enumerate all reachable decimal one-step contexts."""

    result = set()
    for width in (4, 6):
        for operation in ("add", "sub"):
            for position in range(width):
                for carry in ((0,) if position == 0 else (0, 1)):
                    for left in range(10):
                        for right in range(10):
                            if operation == "sub" and position == width - 1:
                                if left < right or (left == right and carry):
                                    continue
                            result.add((width, operation, position, carry, left, right))
    return result


def _parse_row_record(row):
    fixed_tape = parse_tape(row.get("fixed_tape", ""))
    candidate_tape = parse_tape(row.get("candidate_tape", ""))
    if fixed_tape is None or candidate_tape is None:
        raise ValueError("invalid serialized tape")
    if row["fixed_tape"] != canonical_tape(fixed_tape):
        raise ValueError("fixed tape is not canonical")
    if row["candidate_tape"] != canonical_tape(candidate_tape):
        raise ValueError("candidate tape is not canonical")
    previous = parse_register(row.get("previous_register", ""), fixed_tape)
    legal = parse_register(row.get("legal_register", ""), fixed_tape)
    candidate = parse_register(row.get("candidate_register", ""), candidate_tape)
    if previous is None or legal is None or candidate is None:
        raise ValueError("invalid serialized register")
    if row["previous_register"] != canonical_register(fixed_tape, previous):
        raise ValueError("previous register is not canonical")
    if row["legal_register"] != canonical_register(fixed_tape, legal):
        raise ValueError("legal register is not canonical")
    if row["candidate_register"] != canonical_register(candidate_tape, candidate):
        raise ValueError("candidate register is not canonical")
    return {
        "fixed_tape": fixed_tape,
        "previous_register": previous,
        "legal_register": legal,
        "candidate_tape": candidate_tape,
        "candidate_register": candidate,
        "foil_kind": row.get("foil_kind"),
    }


def expected_response_variants(record):
    semantic = reflection_response(record)
    fields = dict(part.split("=", 1) for part in semantic.split(";"))
    fields["field"] = LABEL_PERMUTATION[fields["field"]]
    permuted = ";".join(
        "{}={}".format(key, fields[key])
        for key in ("verdict", "field", "at_p", "expected", "observed")
    )
    return semantic, permuted, SYNTAX_ONLY_RESPONSE


def audit_row(row, *, expected_partition):
    """Validate one serialized reflection row and return its semantic record."""

    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    if row.get("kind") != "reflection" or row.get("training_group") != "counterfactual_workspace":
        raise ValueError("invalid CWI row metadata")
    if row.get("question") != row.get("completion_prompt") or not row.get("question"):
        raise ValueError("invalid completion boundary")
    record = _parse_row_record(row)
    validate_reflection_record(record)
    prompt_style = row.get("prompt_style")
    if prompt_style not in ("core", "heldout"):
        raise ValueError("invalid CWI prompt style")
    if row["question"] != reflection_prompt(record, style=prompt_style):
        raise ValueError("reflection prompt mismatch")
    for field, expected in zip(RESPONSE_FIELDS, expected_response_variants(record)):
        if row.get(field) != expected:
            raise ValueError("{} does not match its control target".format(field))
    if expected_partition == "train":
        if row.get("split") != "train" or prompt_style != "core" or "world" in row:
            raise ValueError("invalid train reflection partition")
        if row.get("source") != "counterfactual_workspace_v1_train":
            raise ValueError("invalid train reflection source")
    elif expected_partition == "heldout":
        if row.get("split") == "train" or prompt_style != "heldout":
            raise ValueError("invalid held-out reflection partition")
        if row.get("world") not in {"base", "counterfactual"}:
            raise ValueError("missing held-out world binding")
        if row.get("source") != "counterfactual_workspace_v1_{}".format(row["split"]):
            raise ValueError("invalid held-out reflection source")
    else:
        raise ValueError("unknown expected partition")
    if not str(row.get("episode_id", "")) or int(row.get("transition_index")) < 0:
        raise ValueError("invalid transition identity")
    return record


def read_rows(path, *, partition):
    rows = []
    invalid = 0
    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record = audit_row(row, expected_partition=partition)
                rows.append((row, record))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    return rows, invalid


def duplicate_count(rows, *, include_world):
    identities = set()
    duplicates = 0
    for row, _ in rows:
        identity = (
            row["split"],
            row["episode_id"],
            int(row["transition_index"]),
            row["foil_kind"],
        )
        if include_world:
            identity += (row["world"],)
        duplicates += int(identity in identities)
        identities.add(identity)
    return duplicates


def prompt_sets(rows, *, include_grams):
    prompts, grams, duplicate_prompts = set(), set(), 0
    for row, _ in rows:
        prompt = normalized(row["completion_prompt"])
        duplicate_prompts += int(prompt in prompts)
        prompts.add(prompt)
        if include_grams:
            grams.update(ngrams(row["completion_prompt"]))
    return prompts, grams, duplicate_prompts


def train_heldout_ngram_hits(rows, heldout_grams):
    """Stream train grams against the held-out set without retaining all train grams."""

    hits = set()
    for row, _ in rows:
        hits.update(ngrams(row["completion_prompt"]) & heldout_grams)
    return hits


def heldout_pair_failures(rows):
    pairs = defaultdict(list)
    for row, record in rows:
        key = (row["split"], row["episode_id"], int(row["transition_index"]), row["foil_kind"])
        pairs[key].append((row, record))
    failures = 0
    for pair in pairs.values():
        worlds = {row["world"] for row, _ in pair}
        if len(pair) != 2 or worlds != {"base", "counterfactual"}:
            failures += 1
            continue
        by_world = {row["world"]: record for row, record in pair}
        if canonical_tape(by_world["base"]["fixed_tape"]) == canonical_tape(by_world["counterfactual"]["fixed_tape"]):
            failures += 1
    return len(pairs), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))

    train_rows, invalid_train = read_rows(args.train, partition="train")
    heldout_rows, invalid_heldout = read_rows(args.heldout, partition="heldout")
    train_prompts, _, duplicate_train_prompts = prompt_sets(train_rows, include_grams=False)
    heldout_prompts, heldout_grams, duplicate_heldout_prompts = prompt_sets(heldout_rows, include_grams=True)
    overlapping_grams = train_heldout_ngram_hits(train_rows, heldout_grams)
    train_foil_counts = Counter(row["foil_kind"] for row, _ in train_rows)
    heldout_foil_counts = Counter(row["foil_kind"] for row, _ in heldout_rows)
    observed_contexts = {
        local_context(record["fixed_tape"], record["previous_register"])
        for row, record in train_rows
        if row["foil_kind"] == "legal"
    }
    required = required_contexts()
    pair_count, heldout_pair_failures_count = heldout_pair_failures(heldout_rows)
    missing_foils = sorted(set(("legal",) + FOIL_KINDS) - set(train_foil_counts))
    missing_contexts = sorted(required - observed_contexts)
    result = {
        "audit": "counterfactual_workspace_v1_admission",
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256_file(args.train),
        "heldout": str(Path(args.heldout).resolve()),
        "heldout_sha256": sha256_file(args.heldout),
        "valid_train_rows": len(train_rows),
        "invalid_train_rows": invalid_train,
        "duplicate_train_identities": duplicate_count(train_rows, include_world=False),
        "duplicate_normalized_train_prompts": duplicate_train_prompts,
        "valid_heldout_rows": len(heldout_rows),
        "invalid_heldout_rows": invalid_heldout,
        "duplicate_heldout_identities": duplicate_count(heldout_rows, include_world=True),
        "duplicate_normalized_heldout_prompts": duplicate_heldout_prompts,
        "train_foil_counts": dict(sorted(train_foil_counts.items())),
        "heldout_foil_counts": dict(sorted(heldout_foil_counts.items())),
        "missing_train_foil_kinds": missing_foils,
        "required_local_contexts": len(required),
        "covered_legal_local_contexts": len(required & observed_contexts),
        "missing_legal_local_contexts": len(missing_contexts),
        "missing_legal_local_context_examples": [list(context) for context in missing_contexts[:12]],
        "heldout_world_pairs": pair_count,
        "invalid_heldout_world_pairs": heldout_pair_failures_count,
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(overlapping_grams),
        "response_fields": list(RESPONSE_FIELDS),
        "claim_boundary": (
            "Data admission only. Reflection targets and their control fields do not establish a model-authored "
            "workspace, direct-transition ability, reasoning score, or context-scaling result."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    failures = (
        result["invalid_train_rows"]
        or result["duplicate_train_identities"]
        or result["duplicate_normalized_train_prompts"]
        or result["invalid_heldout_rows"]
        or result["duplicate_heldout_identities"]
        or result["duplicate_normalized_heldout_prompts"]
        or result["missing_train_foil_kinds"]
        or result["missing_legal_local_contexts"]
        or result["invalid_heldout_world_pairs"]
        or result["train_heldout_exact_prompt_hits"]
        or result["train_heldout_13gram_hits"]
    )
    if failures:
        raise SystemExit("counterfactual workspace v1 admission failed")


if __name__ == "__main__":
    main()
