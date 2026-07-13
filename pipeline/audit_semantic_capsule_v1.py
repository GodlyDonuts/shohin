#!/usr/bin/env python3
"""Independently verify semantic-capsule data and controller-prompt separation."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from generate_semantic_capsule_v1 import apply_operation, canonical_capsule, query_prompt, update_prompt


WORD = re.compile(r"\w+")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def verify_query(query, values, keys):
    kind = query["kind"]
    if kind == "read":
        return int(query["answer"]) == int(values[query["key"]])
    if kind == "sum":
        return int(query["answer"]) == int(values[keys[0]] + values[keys[1]])
    if kind == "difference":
        return int(query["answer"]) == int(values[query["high"]] - values[query["low"]])
    return False


def heldout_prompts(episode):
    keys = tuple(episode["keys"])
    prompts = [episode["initial"]["prompt"]]
    current = dict(episode["initial"]["values"])
    for step in episode["operations"]:
        prompts.append(update_prompt(canonical_capsule(current, keys), step["instruction"], keys,
                                    heldout=True, reference=episode["reference"], revision=step["revision"]))
        current = dict(step["expected"])
    prompts.append(query_prompt(canonical_capsule(current, keys), episode["query"], heldout=True,
                                reference=episode["reference"]))
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
    malformed_rows = 0
    invalid_completion_prompts = 0
    duplicate_questions = 0
    seen_questions = set()
    for row in train_rows:
        if not all(str(row.get(field, "")).strip() for field in ("question", "response", "answer", "training_group")):
            malformed_rows += 1
            continue
        key = normalized(row["question"])
        duplicate_questions += int(key in seen_questions)
        seen_questions.add(key)
        if row.get("training_group") != "semantic_capsule" or not str(row["response"]).startswith("<think>"):
            malformed_rows += 1
        if row.get("completion_prompt") != row.get("question"):
            invalid_completion_prompts += 1

    invalid_episodes, prompt_rows = 0, []
    regimes = collections.Counter()
    for episode in episodes:
        required = ("heldout", "reference", "keys", "initial", "operations", "query", "regime")
        if any(field not in episode for field in required) or not episode["heldout"]:
            invalid_episodes += 1
            continue
        keys = tuple(episode["keys"])
        current = dict(episode["initial"].get("values", {}))
        if set(current) != set(keys):
            invalid_episodes += 1
            continue
        regimes[episode["regime"]] += 1
        for step in episode["operations"]:
            if step.get("before") != current or "operation" not in step or int(step.get("revision", 0)) <= 0:
                invalid_episodes += 1
                break
            expected = apply_operation(current, step["operation"])
            if expected != step.get("expected"):
                invalid_episodes += 1
                break
            current = expected
        else:
            if not verify_query(episode["query"], current, keys):
                invalid_episodes += 1
                continue
            prompt_rows.extend(heldout_prompts(episode))

    train_normalized = {normalized(row["question"]) for row in train_rows if row.get("question")}
    heldout_normalized = {normalized(prompt) for prompt in prompt_rows}
    train_grams, heldout_grams = set(), set()
    for row in train_rows:
        if row.get("question"):
            train_grams.update(ngrams(row["question"]))
    for prompt in prompt_rows:
        heldout_grams.update(ngrams(prompt))
    result = {
        "audit": "semantic_capsule_v1_protocol_audit",
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "episodes": str(Path(args.episodes).resolve()),
        "episodes_sha256": sha256_file(args.episodes),
        "train_rows": len(train_rows),
        "heldout_episodes": len(episodes),
        "heldout_controller_prompts": len(prompt_rows),
        "malformed_train_rows": malformed_rows,
        "duplicate_normalized_train_questions": duplicate_questions,
        "invalid_completion_prompts": invalid_completion_prompts,
        "invalid_heldout_episodes": invalid_episodes,
        "regimes": dict(sorted(regimes.items())),
        "overlap": {
            "exact_prompt_hits": len(train_normalized & heldout_normalized),
            "ngram13_hits": len(train_grams & heldout_grams),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
