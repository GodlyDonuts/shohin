#!/usr/bin/env python3
"""Build a matched auxiliary counterfactual-reflection curriculum.

This is a *conditional* follow-up to the direct-only operator-trace baseline.
It does not ask the model to answer the original task.  Instead, a training
turn interrupts an ordinary three-operation story and asks for the state after
one explicitly counterfactual operation.  A token-surface-matched neutral arm
keeps the operation names but replaces every task-derived state with zeros.

The intended test is narrow and causal: when both arms receive the same direct
anchor examples and update budget, does counterfactual numeric state (rather
than reflection grammar or operation-label exposure) improve an *unreflected*
direct prompt?  The generator only creates immutable data; it never submits an
SFT or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from random import Random

from generate_operator_trace_contrast_v1 import (
    TRAIN_TEMPLATES,
    Episode,
    apply,
    episode_for,
    operator_name,
    render_question,
)


CONTRACT_REFLECTION = "counterfactual_reflection"
CONTRACT_NEUTRAL = "counterfactual_neutral"
STATE_WIDTH = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def padded(value: int) -> str:
    """Use a fixed decimal surface so the neutral arm matches state width."""
    if value < 0 or value >= 10 ** STATE_WIDTH:
        raise ValueError("counterfactual state must fit the fixed positive decimal surface")
    return f"{value:0{STATE_WIDTH}d}"


def edit_for(episode: Episode) -> tuple[int, str]:
    """Return the single operation index and deterministic reversed operation."""
    if episode.family == "add_multiply_subtract":
        return 0, "subtract"
    if episode.family == "subtract_multiply_add":
        return 0, "add"
    if episode.family == "double_add_divide":
        return 1, "subtract"
    raise ValueError(f"unsupported family: {episode.family}")


def operands(episode: Episode) -> tuple[int, int, int]:
    # ``double_add_divide`` stores c=2 but the actual first operand is 2,
    # second is ``a``, and third is divisor ``b``.
    if episode.family == "double_add_divide":
        return 2, episode.a, episode.b
    return episode.a, episode.b, episode.c


def counterfactual_state(episode: Episode) -> dict:
    index, replacement = edit_for(episode)
    values = [episode.start]
    for operation, operand in zip(episode.operations[:index], operands(episode)[:index]):
        values.append(apply(values[-1], operation, operand))
    before = values[-1]
    after = apply(before, replacement, operands(episode)[index])
    return {
        "index": index,
        "old_operation": episode.operations[index],
        "new_operation": replacement,
        "operand": operands(episode)[index],
        "state_before": before,
        "counterfactual_after": after,
    }


def prompt_for(question: str, state: dict, *, neutral: bool) -> str:
    ordinal = ("first", "second", "third")[state["index"]]
    if neutral:
        instruction = (
            f"Before solving the original task, an external interruption asks for a structural marker for "
            f"the {ordinal} operation. Report the original and reversed operation names, but do not "
            "calculate any task state."
        )
    else:
        instruction = (
            f"Before solving the original task, an external interruption reverses only the {ordinal} "
            "operation while preserving its operand. Do not answer the original task."
        )
    return (
        f"Question: {question} {instruction} Inside <reflect>, emit exactly old_op=OP;new_op=OP;"
        f"state_before={STATE_WIDTH}digits;counterfactual_after={STATE_WIDTH}digits.\nAnswer:"
    )


def response_for(state: dict, *, neutral: bool) -> str:
    before = after = "0" * STATE_WIDTH if neutral else None
    if not neutral:
        before = padded(state["state_before"])
        after = padded(state["counterfactual_after"])
    return (
        "<reflect>old_op={};new_op={};state_before={};counterfactual_after={}</reflect>".format(
            operator_name(state["old_operation"]), operator_name(state["new_operation"]), before, after
        )
    )


def row_for(episode: Episode, question: str, *, neutral: bool) -> dict:
    state = counterfactual_state(episode)
    prompt = prompt_for(question, state, neutral=neutral)
    response = response_for(state, neutral=neutral)
    return {
        "question": prompt,
        "completion_prompt": prompt,
        "response": response,
        "source": "operator_counterfactual_reflection_v1",
        "training_group": "operator_counterfactual_aux",
        "family": episode.family,
        "contract": CONTRACT_NEUTRAL if neutral else CONTRACT_REFLECTION,
        "operations": list(episode.operations),
        "counterfactual": state,
        "neutral_states": neutral,
    }


def build_rows(per_family: int, seed: int) -> tuple[list[dict], list[dict]]:
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    rng = Random(seed)
    reflection, neutral = [], []
    seen = set()
    for family in TRAIN_TEMPLATES:
        count = attempts = 0
        while count < per_family:
            attempts += 1
            if attempts > per_family * 200:
                raise RuntimeError("could not generate enough unique counterfactual rows")
            episode = episode_for(family, rng, heldout_values=False)
            state = counterfactual_state(episode)
            # Keep the target state positive and fixed-width. This is a data
            # construction constraint, not controller-side recovery.
            if state["state_before"] < 0 or state["counterfactual_after"] < 0:
                continue
            for variant in range(len(TRAIN_TEMPLATES[family])):
                question = render_question(episode, TRAIN_TEMPLATES, variant)
                reflected = row_for(episode, question, neutral=False)
                key = " ".join(reflected["completion_prompt"].lower().split())
                if key in seen:
                    continue
                seen.add(key)
                reflection.append(reflected)
                neutral.append(row_for(episode, question, neutral=True))
            count += 1
    rng.shuffle(reflection)
    rng.shuffle(neutral)
    if len(reflection) != len(neutral):
        raise RuntimeError("matched arms diverged")
    return reflection, neutral


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflection-out", required=True)
    parser.add_argument("--neutral-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--per-family", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    reflection_path, neutral_path, report_path = map(
        Path, (args.reflection_out, args.neutral_out, args.report_out)
    )
    if any(path.exists() for path in (reflection_path, neutral_path, report_path)):
        raise SystemExit("refusing to overwrite a counterfactual reflection candidate")
    reflection, neutral = build_rows(args.per_family, args.seed)
    write_jsonl(reflection_path, reflection)
    write_jsonl(neutral_path, neutral)
    report = {
        "schema": "operator_counterfactual_reflection_v1",
        "seed": args.seed,
        "rows_per_arm": len(reflection),
        "reflection_sha256": sha256(reflection_path),
        "neutral_sha256": sha256(neutral_path),
        "reflection_by_family": dict(sorted(collections.Counter(row["family"] for row in reflection).items())),
        "neutral_by_family": dict(sorted(collections.Counter(row["family"] for row in neutral).items())),
        "reflection_contracts": dict(sorted(collections.Counter(row["contract"] for row in reflection).items())),
        "neutral_contracts": dict(sorted(collections.Counter(row["contract"] for row in neutral).items())),
        "claim_boundary": (
            "Data-only matched auxiliary arms. Any later comparison must score ordinary direct prompts "
            "without a reflection request and compare numeric reflection against the neutral arm."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
