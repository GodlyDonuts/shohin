#!/usr/bin/env python3
"""CPU admission for the Causal Microcode Bottleneck lexical contract."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from categorical_microcode import (  # noqa: E402
    OPCODES,
    QUERIES,
    alu_basis_accuracy,
    compile_example,
    execute_program,
    sha256_file,
    transition_basis_targets,
)


EXECUTOR_WIDTH = 8
EXECUTOR_LIMIT = 10 ** EXECUTOR_WIDTH


def exact_table():
    targets = transition_basis_targets()
    logits = torch.full((*targets.shape, 20), -10.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 10.0)
    return logits


def structured_register_trace(row):
    """Independently replay structured rows to audit every register boundary."""
    keys = row["keys"]
    values = {key: int(row["initial"][key]) for key in keys}
    trace = [tuple(values[key] for key in keys)]
    for operation in row["operations"]:
        kind = operation["kind"]
        if kind == "add":
            values[operation["target"]] += int(operation["value"])
        elif kind == "sub":
            values[operation["target"]] -= int(operation["value"])
        elif kind == "move":
            value = int(operation["value"])
            values[operation["source"]] -= value
            values[operation["target"]] += value
        elif kind == "merge":
            values[operation["target"]] += values[operation["source"]]
        elif kind == "swap":
            left, right = operation["left"], operation["right"]
            values[left], values[right] = values[right], values[left]
        else:
            raise ValueError("unknown structured operation {}".format(kind))
        trace.append(tuple(values[key] for key in keys))
    return trace


def audit_file(path, tokenizer, table):
    report = {
        "rows": 0,
        "oracle_answer_errors": 0,
        "depths": collections.Counter(),
        "regimes": collections.Counter(),
        "opcodes": collections.Counter(),
        "queries": collections.Counter(),
        "max_initial": 0,
        "max_register": 0,
        "max_answer": 0,
        "max_tokens": 0,
        "executor_width_violations": 0,
    }
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                example = compile_example(row, tokenizer)
                oracle = execute_program(
                    example.initial_values, example.operation_targets,
                    example.operation_values, example.query_target, table,
                )
            except Exception as exc:
                raise ValueError("{} row {} failed lexical/executor admission: {}".format(
                    path, line_number, exc,
                )) from exc
            report["rows"] += 1
            report["oracle_answer_errors"] += int(oracle != example.answer)
            states = structured_register_trace(row)
            flat_states = [value for state in states for value in state]
            report["executor_width_violations"] += int(any(
                value < 0 or value >= EXECUTOR_LIMIT for value in flat_states
            ))
            report["max_register"] = max(report["max_register"], *flat_states)
            report["depths"][str(len(example.operation_targets))] += 1
            report["regimes"][example.regime] += 1
            report["opcodes"].update(OPCODES[target] for target in example.operation_targets)
            report["queries"][QUERIES[example.query_target]] += 1
            report["max_initial"] = max(report["max_initial"], *example.initial_values)
            report["max_answer"] = max(report["max_answer"], example.answer)
            report["max_tokens"] = max(report["max_tokens"], len(example.ids))
    for key in ("depths", "regimes", "opcodes", "queries"):
        report[key] = dict(sorted(report[key].items()))
    return report


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
    table = exact_table()
    basis_correct, basis_total = alu_basis_accuracy(table)
    train_report = audit_file(args.train, tokenizer, table)
    eval_report = audit_file(args.eval, tokenizer, table)
    result = {
        "audit": "causal_microcode_bottleneck_lexical_admission_v1",
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256_file(args.train),
        "eval": str(Path(args.eval).resolve()),
        "eval_sha256": sha256_file(args.eval),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "executor_width": EXECUTOR_WIDTH,
        "alu_basis": {"correct": basis_correct, "total": basis_total},
        "train_report": train_report,
        "eval_report": eval_report,
        "all_checks_pass": (
            train_report["rows"] == 96000
            and eval_report["rows"] == 896
            and train_report["oracle_answer_errors"] == 0
            and eval_report["oracle_answer_errors"] == 0
            and train_report["executor_width_violations"] == 0
            and eval_report["executor_width_violations"] == 0
            and max(train_report["max_answer"], eval_report["max_answer"]) < EXECUTOR_LIMIT
            and basis_correct == basis_total == 400
        ),
        "claim_boundary": (
            "This admits deterministic lexical extraction and the categorical executor contract only; "
            "it provides no evidence that the neural compiler predicts correct programs."
        ),
    }
    if not result["all_checks_pass"]:
        raise SystemExit("microcode lexical admission failed: " + json.dumps(result, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
