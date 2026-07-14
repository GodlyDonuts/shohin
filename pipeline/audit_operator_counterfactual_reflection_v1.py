#!/usr/bin/env python3
"""Audit matched operator counterfactual-reflection auxiliary arms.

The audit is deliberately independent from the generator's row counts.  It
replays each stored source episode, verifies the counterfactual target, checks
that the neutral arm preserves the response grammar and token budget without
task-derived states, and records the remaining prompt-token difference rather
than pretending the two task instructions are identical.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from generate_operator_counterfactual_reflection_v1 import (
    CONTRACT_NEUTRAL,
    CONTRACT_REFLECTION,
    STATE_WIDTH,
    apply,
    edit_for,
    operator_name,
    padded,
)
from generate_operator_trace_contrast_v1 import Episode


PAIR_MARKERS = ("problem a:", "problem b:", "the answers are a=")
RESPONSE = re.compile(
    r"^<reflect>old_op=([A-Z]+);new_op=([A-Z]+);state_before=(\d{%d});"
    r"counterfactual_after=(\d{%d})</reflect>$" % (STATE_WIDTH, STATE_WIDTH)
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as source:
        for number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: malformed JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def key(row: dict) -> str:
    return json.dumps(
        {field: row.get(field) for field in ("family", "variant", "episode", "counterfactual")},
        sort_keys=True,
        separators=(",", ":"),
    )


def replay(row: dict) -> dict:
    episode = row.get("episode")
    counterfactual = row.get("counterfactual")
    if not isinstance(episode, dict) or not isinstance(counterfactual, dict):
        raise ValueError("missing episode or counterfactual metadata")
    family = row.get("family")
    operations = tuple(episode.get("operations") or ())
    operands = tuple(episode.get("operands") or ())
    states = tuple(episode.get("states") or ())
    start = episode.get("start")
    if not isinstance(family, str) or len(operations) != 3 or len(operands) != 3 or len(states) != 4:
        raise ValueError("malformed episode shape")
    if not all(isinstance(value, int) for value in (*operands, *states, start)):
        raise ValueError("episode values must be integers")
    values = [start]
    for operation, operand in zip(operations, operands):
        values.append(apply(values[-1], operation, operand))
    if tuple(values) != states:
        raise ValueError("stored episode states do not replay")
    # ``edit_for`` only needs the family/operation/state layout. For its
    # legacy Episode fields, operands a/b/c are irrelevant here because the
    # explicit operand tuple above is the audit authority.
    proxy = Episode(family, start, operands[0], operands[1], operands[2], operations, states)
    index, replacement = edit_for(proxy)
    before = states[index]
    after = apply(before, replacement, operands[index])
    expected = {
        "index": index,
        "old_operation": operations[index],
        "new_operation": replacement,
        "operand": operands[index],
        "state_before": before,
        "counterfactual_after": after,
    }
    if counterfactual != expected:
        raise ValueError("stored counterfactual does not replay")
    return expected


def parse_response(row: dict) -> tuple[str, str, str, str]:
    response = str(row.get("response") or "")
    match = RESPONSE.fullmatch(response)
    if not match:
        raise ValueError("response lacks the exact reflection grammar")
    return match.groups()


def validate_consumed_fields(row: dict) -> None:
    for field in ("question", "completion_prompt", "response"):
        value = str(row.get(field) or "")
        if not value:
            raise ValueError(f"missing consumed field {field}")
        normalized = value.lower()
        if any(marker in normalized for marker in PAIR_MARKERS):
            raise ValueError("paired-answer response grammar leaked into auxiliary data")


def audit(reflection_path: str, neutral_path: str, tokenizer_path: str) -> dict:
    reflection = load_rows(reflection_path)
    neutral = load_rows(neutral_path)
    r_rows = {key(row): row for row in reflection}
    n_rows = {key(row): row for row in neutral}
    if len(r_rows) != len(reflection) or len(n_rows) != len(neutral):
        raise ValueError("duplicate episode keys")
    if r_rows.keys() != n_rows.keys():
        raise ValueError("reflection and neutral arms do not contain the same episodes")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    prompt_deltas = collections.Counter()
    by_family = collections.Counter()
    for episode_key, reflected in r_rows.items():
        control = n_rows[episode_key]
        if reflected.get("contract") != CONTRACT_REFLECTION or control.get("contract") != CONTRACT_NEUTRAL:
            raise ValueError("contract labels do not match their arms")
        if reflected.get("training_group") != "operator_counterfactual_aux" or control.get("training_group") != "operator_counterfactual_aux":
            raise ValueError("unexpected training group")
        if reflected.get("source") != "operator_counterfactual_reflection_v1" or control.get("source") != "operator_counterfactual_reflection_v1":
            raise ValueError("unexpected auxiliary source")
        if reflected.get("neutral_states") is not False or control.get("neutral_states") is not True:
            raise ValueError("neutral-state labels do not match their arms")
        if reflected.get("question") != reflected.get("completion_prompt") or control.get("question") != control.get("completion_prompt"):
            raise ValueError("question and completion prompt must be identical")
        validate_consumed_fields(reflected)
        validate_consumed_fields(control)
        expected = replay(reflected)
        if replay(control) != expected:
            raise ValueError("neutral arm episode differs from reflection arm")
        old, new, before, after = parse_response(reflected)
        n_old, n_new, n_before, n_after = parse_response(control)
        if (old, new) != (operator_name(expected["old_operation"]), operator_name(expected["new_operation"])):
            raise ValueError("reflection response operation labels are wrong")
        if (before, after) != (padded(expected["state_before"]), padded(expected["counterfactual_after"])):
            raise ValueError("reflection response states are wrong")
        if (n_old, n_new) != (old, new) or n_before != "0" * STATE_WIDTH or n_after != "0" * STATE_WIDTH:
            raise ValueError("neutral response does not preserve labels with zero states")
        response_tokens = len(tokenizer.encode(reflected["response"]).ids)
        if response_tokens != len(tokenizer.encode(control["response"]).ids):
            raise ValueError("arms differ in supervised response token count")
        prompt_deltas[len(tokenizer.encode(reflected["completion_prompt"]).ids) - len(tokenizer.encode(control["completion_prompt"]).ids)] += 1
        by_family[str(reflected["family"])] += 1
    return {
        "schema": "operator_counterfactual_reflection_audit_v1",
        "reflection": str(Path(reflection_path)),
        "neutral": str(Path(neutral_path)),
        "reflection_sha256": sha256(reflection_path),
        "neutral_sha256": sha256(neutral_path),
        "tokenizer": str(Path(tokenizer_path)),
        "tokenizer_sha256": sha256(tokenizer_path),
        "matched_pairs": len(r_rows),
        "by_family": dict(sorted(by_family.items())),
        "supervised_response_token_delta": {"all_pairs_equal": True},
        "prompt_token_delta_histogram": dict(sorted(prompt_deltas.items())),
        "claim_boundary": (
            "Data admission only. The response targets are token-matched; prompt differences are measured, "
            "not hidden. A later SFT comparison must still use identical direct anchors and score unreflected prompts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflection", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report = audit(args.reflection, args.neutral, args.tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
