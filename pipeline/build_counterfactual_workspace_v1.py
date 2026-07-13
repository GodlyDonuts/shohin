#!/usr/bin/env python3
"""Derive CWI reflection data from an admitted static-tape register corpus.

This CPU-only builder does not create a model checkpoint.  It transforms each
already-admitted local transition into a legal candidate plus grammar-valid
single-invariant foils.  The reflection continuation is a training-only target;
future evaluation must omit this prompt and use the ordinary direct controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from counterfactual_workspace_protocol import (
    FOIL_KINDS,
    make_semantic_foil,
    reflection_prompt,
    reflection_response,
)
from digitwise_factor_protocol import apply_microstep, canonical_register, canonical_tape, parse_register, parse_tape


LABEL_PERMUTATION = {
    "carry": "result_digit",
    "result_digit": "program_counter",
    "program_counter": "tape",
    "tape": "carry",
    "none": "none",
}
RESPONSE_FIELDS = ("response", "response_label_permuted", "response_syntax_only")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def legal_record(tape, register):
    legal = apply_microstep(tape, register)
    return {
        "fixed_tape": dict(tape),
        "previous_register": dict(register),
        "legal_register": dict(legal),
        "candidate_tape": dict(tape),
        "candidate_register": dict(legal),
        "foil_kind": "legal",
    }


def response_variants(record):
    semantic = reflection_response(record)
    fields = dict(part.split("=", 1) for part in semantic.split(";"))
    fields["field"] = LABEL_PERMUTATION[fields["field"]]
    permuted = ";".join("{}={}".format(key, fields[key]) for key in ("verdict", "field", "at_p", "expected", "observed"))
    syntax = "syntax=valid;shape=tape_register;slots=complete;mode=local"
    return semantic, permuted, syntax


def record_row(record, *, split, episode_id, transition_index, prompt_style, world=None):
    prompt = reflection_prompt(record, style=prompt_style)
    semantic, permuted, syntax = response_variants(record)
    row = {
        "question": prompt,
        "completion_prompt": prompt,
        "response": semantic,
        "response_label_permuted": permuted,
        "response_syntax_only": syntax,
        "source": "counterfactual_workspace_v1_{}".format(split),
        "training_group": "counterfactual_workspace",
        "kind": "reflection",
        "split": split,
        "episode_id": str(episode_id),
        "transition_index": int(transition_index),
        "prompt_style": prompt_style,
        "foil_kind": record["foil_kind"],
        "fixed_tape": canonical_tape(record["fixed_tape"]),
        "previous_register": canonical_register(record["fixed_tape"], record["previous_register"]),
        "legal_register": canonical_register(record["fixed_tape"], record["legal_register"]),
        "candidate_tape": canonical_tape(record["candidate_tape"]),
        "candidate_register": canonical_register(record["candidate_tape"], record["candidate_register"]),
    }
    if world is not None:
        row["world"] = str(world)
    return row


def rows_from_transition(tape, register, *, split, episode_id, transition_index, prompt_style, world=None):
    records = [legal_record(tape, register)]
    for kind in FOIL_KINDS:
        try:
            records.append(make_semantic_foil(tape, register, kind))
        except ValueError as exc:
            if kind != "program_counter" or "terminal" not in str(exc):
                raise
    return [record_row(
        record,
        split=split,
        episode_id=episode_id,
        transition_index=transition_index,
        prompt_style=prompt_style,
        world=world,
    ) for record in records]


def train_rows(path):
    rows = []
    seen = set()
    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") != "transition":
                continue
            tape = parse_tape(item.get("tape", ""))
            register = parse_register(item.get("register", ""), tape) if tape is not None else None
            if tape is None or register is None:
                raise ValueError("input factor transition is malformed")
            key = (item.get("episode_id"), int(item.get("transition_index")))
            if key in seen:
                raise ValueError("input factor transitions are not unique")
            seen.add(key)
            rows.extend(rows_from_transition(
                tape,
                register,
                split="train",
                episode_id=item["episode_id"],
                transition_index=item["transition_index"],
                prompt_style="core",
            ))
    if not rows:
        raise ValueError("input factor corpus contains no transition rows")
    return rows


def heldout_rows(path):
    rows = []
    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            episode = json.loads(line)
            for world, item in (("base", episode), ("counterfactual", episode.get("counterfactual"))):
                if not isinstance(item, dict):
                    raise ValueError("heldout factor episode lacks counterfactual world")
                tape = parse_tape(item.get("tape", ""))
                register = parse_register(item.get("initial_register", ""), tape) if tape is not None else None
                if tape is None or register is None:
                    raise ValueError("heldout factor state is malformed")
                for index, expected_line in enumerate(item.get("expected_registers", ())):
                    expected = parse_register(expected_line, tape)
                    if expected is None or expected != apply_microstep(tape, register):
                        raise ValueError("heldout factor successor is malformed")
                    rows.extend(rows_from_transition(
                        tape,
                        register,
                        split=str(item["split"]),
                        episode_id=episode["id"],
                        transition_index=index,
                        prompt_style="heldout",
                        world=world,
                    ))
                    register = expected
    if not rows:
        raise ValueError("input heldout corpus contains no transitions")
    return rows


def summarize(rows):
    result = {}
    for row in rows:
        result[row["foil_kind"]] = result.get(row["foil_kind"], 0) + 1
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-train", required=True)
    parser.add_argument("--factor-heldout", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    outputs = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite existing CWI output")
    train = train_rows(args.factor_train)
    heldout = heldout_rows(args.factor_heldout)
    write_jsonl(args.train_out, train)
    write_jsonl(args.heldout_out, heldout)
    report = {
        "build": "counterfactual_workspace_v1",
        "factor_train": str(Path(args.factor_train).resolve()),
        "factor_train_sha256": sha256_file(args.factor_train),
        "factor_heldout": str(Path(args.factor_heldout).resolve()),
        "factor_heldout_sha256": sha256_file(args.factor_heldout),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_foil_counts": summarize(train),
        "heldout_foil_counts": summarize(heldout),
        "response_fields": list(RESPONSE_FIELDS),
        "claim_boundary": (
            "CPU data build only. Reflection continuations are training-time labels; no direct transition, "
            "checkpoint, H100 job, or reasoning result is created."
        ),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
