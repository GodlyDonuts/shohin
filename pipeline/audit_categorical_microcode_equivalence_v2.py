#!/usr/bin/env python3
"""Independent pair and held-out-language admission for microcode equivalence v2."""

from __future__ import annotations

import argparse
import collections
import hashlib
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


WORD = re.compile(r"\w+")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pair_signature(example):
    return (
        example.operation_targets,
        example.operation_values,
        example.query_target,
        example.initial_values,
        example.answer,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing report: {}".format(out))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    train_rows, eval_rows = load_rows(args.train), load_rows(args.eval)
    train_questions = [normalized(row.get("question", "")) for row in train_rows]
    eval_questions = [normalized(row.get("question", "")) for row in eval_rows]
    heldout_language_grams = set().union(*(
        ngrams(row["question"]) for row in eval_rows
        if row.get("eval_regime") in {"language_ood", "full_ood"}
    ))

    pairs = collections.defaultdict(dict)
    family_by_view = collections.Counter()
    invalid_pair_rows = 0
    for row in train_rows:
        equivalence_id = row.get("equivalence_id")
        view = row.get("equivalence_view")
        if not isinstance(equivalence_id, str) or view not in (0, 1) or view in pairs[equivalence_id]:
            invalid_pair_rows += 1
            continue
        pairs[equivalence_id][view] = compile_example(row, tokenizer)
        family_by_view[(int(view), str(row.get("family", "")))] += 1
    pair_mismatches = 0
    incomplete_pairs = 0
    for views in pairs.values():
        if set(views) != {0, 1}:
            incomplete_pairs += 1
            continue
        if pair_signature(views[0]) != pair_signature(views[1]):
            pair_mismatches += 1

    report = {
        "audit": "categorical_microcode_equivalence_v2",
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256_file(args.train),
        "eval": str(Path(args.eval).resolve()),
        "eval_sha256": sha256_file(args.eval),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "rows": len(train_rows),
        "pairs": len(pairs),
        "invalid_pair_rows": invalid_pair_rows,
        "incomplete_pairs": incomplete_pairs,
        "pair_mismatches": pair_mismatches,
        "family_by_view": {
            str(view): dict(sorted(
                (family, count) for (candidate_view, family), count in family_by_view.items()
                if candidate_view == view
            )) for view in (0, 1)
        },
        "duplicate_train_questions": len(train_questions) - len(set(train_questions)),
        "exact_eval_prompt_hits": len(set(train_questions) & set(eval_questions)),
        "heldout_language_ngram13_rows": sum(
            bool(ngrams(row["question"]) & heldout_language_grams) for row in train_rows
        ),
        "train_report": audit_file(args.train, tokenizer, exact_table()),
        "eval_report": audit_file(args.eval, tokenizer, exact_table()),
    }
    report["all_checks_pass"] = (
        report["rows"] == 96000
        and report["pairs"] == 48000
        and not report["invalid_pair_rows"]
        and not report["incomplete_pairs"]
        and not report["pair_mismatches"]
        and set(report["family_by_view"]["0"]) == set(report["family_by_view"]["1"])
        and set(report["family_by_view"]["0"].values()) == {3000}
        and set(report["family_by_view"]["1"].values()) == {3000}
        and not report["duplicate_train_questions"]
        and not report["exact_eval_prompt_hits"]
        and not report["heldout_language_ngram13_rows"]
        and report["train_report"]["oracle_answer_errors"] == 0
        and report["eval_report"]["oracle_answer_errors"] == 0
        and report["train_report"]["executor_width_violations"] == 0
        and report["eval_report"]["executor_width_violations"] == 0
    )
    report["claim_boundary"] = (
        "This admits matched semantic views and lexical/executor integrity only; it does not show "
        "that equivalence loss improves compilation or reasoning."
    )
    if not report["all_checks_pass"]:
        raise SystemExit("equivalence admission failed: " + json.dumps(report, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
