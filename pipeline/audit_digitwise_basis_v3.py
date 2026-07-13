#!/usr/bin/env python3
"""Independent admission audit for a coverage-complete DRS v3 candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from digitwise_basis_protocol import local_context
from digitwise_protocol import parse_state
from audit_digitwise_recurrent_v1 import audit_episode, audit_train_row, ngrams, normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_contexts():
    """Independently enumerate reachable decimal contexts used by v3."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))

    train_prompts, train_grams, observed = set(), set(), Counter()
    invalid_rows, duplicate_rows = 0, 0
    with open(args.data) as source:
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audit_train_row(row)
                prompt = normalized(row["completion_prompt"])
                if prompt in train_prompts:
                    duplicate_rows += 1
                train_prompts.add(prompt)
                train_grams.update(ngrams(row["completion_prompt"]))
                if row["kind"] == "transition":
                    context = local_context(parse_state(row["state"]))
                    if context is None:
                        raise ValueError("transition state is terminal")
                    observed[context] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_rows += 1

    heldout_prompts, heldout_grams, regimes = set(), set(), Counter()
    invalid_episodes = 0
    with open(args.episodes) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                prompts = audit_episode(episode)
                regimes[episode["split"]] += 1
                for prompt in prompts:
                    normalized_prompt = normalized(prompt)
                    if normalized_prompt in heldout_prompts:
                        raise ValueError("duplicate heldout controller prompt")
                    heldout_prompts.add(normalized_prompt)
                    heldout_grams.update(ngrams(prompt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_episodes += 1

    required = required_contexts()
    missing = sorted(required - set(observed))
    result = {
        "audit": "digitwise_basis_v3_admission",
        "data": str(Path(args.data).resolve()),
        "episodes": str(Path(args.episodes).resolve()),
        "data_sha256": sha256_file(args.data),
        "episodes_sha256": sha256_file(args.episodes),
        "valid_train_rows": len(train_prompts),
        "invalid_train_rows": invalid_rows,
        "duplicate_normalized_train_prompts": duplicate_rows,
        "valid_heldout_episodes": sum(regimes.values()),
        "invalid_heldout_episodes": invalid_episodes,
        "heldout_regimes": dict(sorted(regimes.items())),
        "required_local_contexts": len(required),
        "covered_local_contexts": len(required & set(observed)),
        "missing_local_contexts": len(missing),
        "missing_local_context_examples": [list(context) for context in missing[:12]],
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(train_grams & heldout_grams),
        "claim_boundary": "Data admission only; basis coverage is not a model capability or reasoning result.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if (
        invalid_rows or duplicate_rows or invalid_episodes or missing or
        result["train_heldout_exact_prompt_hits"] or result["train_heldout_13gram_hits"]
    ):
        raise SystemExit("digitwise basis v3 admission failed")


if __name__ == "__main__":
    main()
