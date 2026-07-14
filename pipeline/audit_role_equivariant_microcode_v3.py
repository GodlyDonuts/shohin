#!/usr/bin/env python3
"""Independent admission for role-factorized register-equivariant microcode data."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from audit_categorical_microcode_v1 import audit_file, exact_table  # noqa: E402
from categorical_microcode import compile_example, sha256_file  # noqa: E402
from role_equivariant_microcode import permute_opcode, permute_query  # noqa: E402


WORD = re.compile(r"\w+")
EXPECTED_VIEWS = ("anchor", "paraphrase_a", "paraphrase_b")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def signature(example):
    return (
        example.operation_targets, example.operation_values, example.query_target,
        example.initial_values, example.answer,
    )


def permuted_signature(original):
    operations, values, query, initial, answer = original
    return (
        tuple(permute_opcode(opcode) for opcode in operations),
        values,
        permute_query(query),
        tuple(reversed(initial)),
        answer,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--programs", type=int, default=48000)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing report")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.train).read_text().splitlines() if line.strip()]
    eval_rows = [json.loads(line) for line in Path(args.eval).read_text().splitlines() if line.strip()]
    groups = collections.defaultdict(dict)
    malformed_group_rows = 0
    questions = []
    for row in rows:
        group = row.get("equivalence_id")
        key = (row.get("semantic_view"), row.get("register_permutation"))
        questions.append(normalized(row.get("question", "")))
        if not isinstance(group, str) or key[0] not in EXPECTED_VIEWS or key[1] not in (0, 1) or key in groups[group]:
            malformed_group_rows += 1
            continue
        groups[group][key] = compile_example(row, tokenizer)

    incomplete_groups = semantic_mismatches = permutation_mismatches = 0
    expected_keys = {(view, permutation) for view in EXPECTED_VIEWS for permutation in (0, 1)}
    for views in groups.values():
        if set(views) != expected_keys:
            incomplete_groups += 1
            continue
        original = signature(views[("anchor", 0)])
        if any(signature(views[(view, 0)]) != original for view in EXPECTED_VIEWS):
            semantic_mismatches += 1
        transformed = signature(views[("anchor", 1)])
        if any(signature(views[(view, 1)]) != transformed for view in EXPECTED_VIEWS):
            semantic_mismatches += 1
        if transformed != permuted_signature(original):
            permutation_mismatches += 1

    heldout_grams = set().union(*(
        ngrams(row["question"]) for row in eval_rows
        if row.get("eval_regime") in {"language_ood", "full_ood"}
    ))
    eval_questions = {normalized(row["question"]) for row in eval_rows}
    report = {
        "audit": "role_equivariant_microcode_v3",
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256_file(args.train),
        "eval": str(Path(args.eval).resolve()),
        "eval_sha256": sha256_file(args.eval),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "rows": len(rows),
        "programs": len(groups),
        "malformed_group_rows": malformed_group_rows,
        "incomplete_groups": incomplete_groups,
        "semantic_mismatches": semantic_mismatches,
        "permutation_mismatches": permutation_mismatches,
        "duplicate_train_questions": len(questions) - len(set(questions)),
        "exact_eval_prompt_hits": len(set(questions) & eval_questions),
        "heldout_language_ngram13_rows": sum(bool(ngrams(row["question"]) & heldout_grams) for row in rows),
        "train_report": audit_file(args.train, tokenizer, exact_table()),
        "eval_report": audit_file(args.eval, tokenizer, exact_table()),
    }
    report["all_checks_pass"] = (
        report["rows"] == args.programs * 6
        and report["programs"] == args.programs
        and not report["malformed_group_rows"]
        and not report["incomplete_groups"]
        and not report["semantic_mismatches"]
        and not report["permutation_mismatches"]
        and not report["duplicate_train_questions"]
        and not report["exact_eval_prompt_hits"]
        and not report["heldout_language_ngram13_rows"]
        and report["train_report"]["oracle_answer_errors"] == 0
        and report["eval_report"]["oracle_answer_errors"] == 0
        and report["train_report"]["executor_width_violations"] == 0
        and report["eval_report"]["executor_width_violations"] == 0
    )
    report["claim_boundary"] = (
        "This proves group integrity, exact register automorphisms, lexical/executor integrity, and "
        "held-out-language exclusion only. It does not establish learned equivariance or reasoning."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["all_checks_pass"]:
        raise SystemExit("role-equivariant admission failed: " + json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
