#!/usr/bin/env python3
"""Independent structural and split audit for append-ledger v1."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from append_ledger_protocol import (compact_prompt, expected_answer, expected_block, expected_delta,
                                    final_prompt, initial_base, parse_block, parse_delta, transition_prompt)
from generate_append_ledger_v1 import normalized, prompts_for_episode


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def audit_row(row):
    required = {"question", "completion_prompt", "response", "training_group", "kind", "base", "blocks", "live", "block_size", "operation", "width", "left", "right", "expected_answer", "prompt_style"}
    if any(field not in row or row[field] in (None, "") for field in required):
        raise ValueError("missing row field")
    if row["question"] != row["completion_prompt"] or row["training_group"] != "append_ledger":
        raise ValueError("invalid row surface")
    base = initial_base(row["operation"], int(row["left"]), int(row["right"]), int(row["width"]))
    if row["base"] != "adl:op={op};w={w};a={a};b={b}".format(**base):
        raise ValueError("base mismatch")
    if int(row["expected_answer"]) != expected_answer(base):
        raise ValueError("answer mismatch")
    blocks, live = row["blocks"], row["live"]
    if row["kind"] == "delta":
        step = int(row.get("step", -1))
        carry = parse_delta(live[-1])["c"] if live else (parse_block(blocks[-1])["c"] if blocks else 0)
        expected = "adl:step={step};d={d};c={c}".format(**expected_delta(base, step, carry))
        if row["question"] != transition_prompt(base, blocks, live, step, row["prompt_style"]) or row["response"] != expected:
            raise ValueError("invalid delta row")
        return
    if row["kind"] == "block":
        index = int(row.get("block", -1))
        parsed = [parse_delta(item) for item in live]
        if not live or any(item is None for item in parsed):
            raise ValueError("invalid live block")
        expected = "adl:block={block};digits={digits};c={c}".format(**expected_block(index, parsed))
        if row["question"] != compact_prompt(base, blocks, live, index, row["prompt_style"]) or row["response"] != expected:
            raise ValueError("invalid compaction row")
        return
    if row["kind"] == "final":
        if row["question"] != final_prompt(base, blocks, row["prompt_style"]) or row["response"] != "answer={}".format(row["expected_answer"]):
            raise ValueError("invalid final row")
        return
    raise ValueError("unknown row kind")


def audit_branch(branch):
    required = {"base", "expected_deltas", "expected_blocks", "expected_answer", "block_size", "operation", "width", "left", "right", "prompt_style"}
    if any(field not in branch for field in required):
        raise ValueError("missing episode field")
    base = initial_base(branch["operation"], int(branch["left"]), int(branch["right"]), int(branch["width"]))
    if branch["base"] != "adl:op={op};w={w};a={a};b={b}".format(**base) or int(branch["expected_answer"]) != expected_answer(base):
        raise ValueError("invalid episode base")
    carry, live, expected_blocks = 0, [], []
    for step, line in enumerate(branch["expected_deltas"]):
        expected = "adl:step={step};d={d};c={c}".format(**expected_delta(base, step, carry))
        if line != expected:
            raise ValueError("invalid expected delta")
        carry = parse_delta(line)["c"]
        live.append(line)
        if len(live) == branch["block_size"] or step + 1 == len(branch["expected_deltas"]):
            expected_blocks.append("adl:block={block};digits={digits};c={c}".format(**expected_block(len(expected_blocks), live)))
            live = []
    if expected_blocks != branch["expected_blocks"]:
        raise ValueError("invalid expected blocks")


def audit_episode(episode):
    if "counterfactual" not in episode or not isinstance(episode["counterfactual"], dict):
        raise ValueError("missing counterfactual")
    audit_branch(episode)
    cf = episode["counterfactual"]
    audit_branch(cf)
    if cf["operation"] != episode["operation"] or cf["width"] != episode["width"] or cf["block_size"] != episode["block_size"]:
        raise ValueError("counterfactual changed protocol")
    changed = int(cf["left"] != episode["left"]) + int(cf["right"] != episode["right"])
    if changed != 1 or int(cf["expected_answer"]) == int(episode["expected_answer"]):
        raise ValueError("invalid counterfactual")
    return prompts_for_episode(episode) + prompts_for_episode(cf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data, episodes_path, out = Path(args.data), Path(args.episodes), Path(args.out)
    if out.exists() or not data.is_file() or not episodes_path.is_file():
        raise SystemExit("missing input or existing output")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    failures, examples, seen = collections.Counter(), [], set()
    duplicates = 0
    for index, row in enumerate(rows, 1):
        key = normalized(row.get("question", ""))
        duplicates += int(key in seen)
        seen.add(key)
        try:
            audit_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            failures[str(exc)] += 1
            if len(examples) < 8:
                examples.append({"kind": "row", "line": index, "error": str(exc)})
    heldout_prompts, regimes = [], collections.Counter()
    for index, episode in enumerate(episodes, 1):
        regimes[episode.get("split", "missing")] += 1
        try:
            heldout_prompts.extend(audit_episode(episode))
        except (KeyError, TypeError, ValueError) as exc:
            failures[str(exc)] += 1
            if len(examples) < 8:
                examples.append({"kind": "episode", "line": index, "error": str(exc)})
    train_prompts = {normalized(row.get("question", "")) for row in rows}
    heldout_norm = {normalized(prompt) for prompt in heldout_prompts}
    train_grams = {gram for row in rows for gram in ngrams(row.get("question", ""))}
    heldout_grams = {gram for prompt in heldout_prompts for gram in ngrams(prompt)}
    result = {"audit": "append-ledger-v1", "data_sha256": digest(data), "episodes_sha256": digest(episodes_path),
              "train_rows": len(rows), "heldout_episodes": len(episodes), "counterfactual_pairs": len(episodes),
              "heldout_controller_prompts": len(heldout_prompts), "invalid_rows_or_episodes": sum(failures.values()),
              "failures": dict(sorted(failures.items())), "examples": examples,
              "duplicate_normalized_train_questions": duplicates, "regimes": dict(sorted(regimes.items())),
              "overlap": {"exact_prompt_hits": len(train_prompts & heldout_norm), "ngram13_hits": len(train_grams & heldout_grams)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["invalid_rows_or_episodes"] or duplicates or result["overlap"]["exact_prompt_hits"] or result["overlap"]["ngram13_hits"]:
        raise SystemExit("append-ledger audit failed")


if __name__ == "__main__":
    main()
