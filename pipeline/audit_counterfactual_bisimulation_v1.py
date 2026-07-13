#!/usr/bin/env python3
"""Independently audit Counterfactual Bisimulation Compiler curriculum data.

The generator is not trusted: this script reconstructs every state, delta,
prompt, and counterfactual relation from its serialized records before any CBC
corpus can be admitted.  It intentionally does not create controller outputs
or make any capability claim.
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
sys.path.insert(0, str(ROOT / "pipeline"))
from bisimulation_compiler_protocol import (
    apply_operation,
    canonical_delta,
    canonical_state,
    compile_prompt,
    copy_values,
    delta_prompt,
    render_event,
    sum_query_prompt,
    update_prompt,
)
from generate_counterfactual_bisimulation_v1 import (
    HELDOUT_DOMAINS,
    ROW_FIELDS,
    TRAIN_DOMAINS,
    controller_prompts,
)


WORD = re.compile(r"\w+")
EXPECTED_EPISODE_FIELDS = {
    "schema", "id", "split", "heldout", "style", "domain", "item", "keys", "reference", "regime",
    "normal", "counterfactual",
}
EXPECTED_WORLD_FIELDS = {"initial_values", "initial_state", "compile_prompts", "steps", "query"}
EXPECTED_STEP_FIELDS = {
    "revision", "operation", "event", "before", "after", "before_state", "after_state", "delta",
    "update_prompt", "delta_prompt",
}
EXPECTED_QUERY_FIELDS = {"kind", "answer", "prompt"}

TRAIN_DOMAIN_ITEMS = {name: item for name, _keys, item in TRAIN_DOMAINS}
HELDOUT_DOMAIN_ITEMS = {name: item for name, _keys, item in HELDOUT_DOMAINS}


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def grams(text: str, width: int = 13):
    tokens = normalized(text).split()
    return {tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(value):
    if not isinstance(value, list) or len(value) != 2 or len(set(value)) != 2:
        raise ValueError("declared CBC keys must be a two-element list")
    return tuple(str(key) for key in value)


def _values(value, keys):
    if not isinstance(value, dict):
        raise ValueError("CBC values must be an object")
    return copy_values(value, keys)


def _domain_item(domain: str, style: str) -> str:
    mapping = TRAIN_DOMAIN_ITEMS if style == "train" else HELDOUT_DOMAIN_ITEMS
    if domain not in mapping:
        raise ValueError("domain is not valid for declared split/style")
    return mapping[domain]


def audit_row(row):
    if set(row) != ROW_FIELDS:
        raise ValueError("invalid CBC train row keys")
    if row["question"] != row["completion_prompt"] or not isinstance(row["response"], str) or not row["response"]:
        raise ValueError("malformed CBC train question or response")
    if row["source"] != "counterfactual_bisimulation_v1_train" or row["training_group"] != "counterfactual_bisimulation":
        raise ValueError("incorrect CBC train provenance")
    if row["style"] != "train":
        raise ValueError("CBC train rows must use train wording")
    keys = _keys(row["keys"])
    if _domain_item(str(row["domain"]), str(row["style"])) != row["item"]:
        raise ValueError("CBC row item does not match domain")
    if row["world"] not in {"normal", "counterfactual"} or not str(row["episode_id"]) or not str(row["reference"]):
        raise ValueError("malformed CBC row identity")
    initial = _values(row["initial_values"], keys)
    before = _values(row["before_values"], keys)
    after = _values(row["after_values"], keys)
    kind, revision = str(row["kind"]), int(row["revision"])
    if revision < 0:
        raise ValueError("negative CBC revision")
    if kind in {"compile_a", "compile_b"}:
        variant = kind[-1]
        if revision != 0 or row["operation"] is not None or row["answer"] is not None or before != initial or after != initial:
            raise ValueError("malformed CBC compile row")
        expected_prompt = compile_prompt(initial, keys, row["domain"], row["item"], row["reference"], "train", variant)
        expected_response = canonical_state(initial, keys)
    elif kind in {"update", "inverse_delta"}:
        if not isinstance(row["operation"], dict) or row["answer"] is not None or revision <= 0:
            raise ValueError("malformed CBC transition row")
        expected_after = apply_operation(before, row["operation"], keys)
        if after != expected_after:
            raise ValueError("CBC transition after-state is semantically wrong")
        before_state, after_state = canonical_state(before, keys), canonical_state(after, keys)
        event = render_event(row["operation"], row["item"], "train")
        if kind == "update":
            expected_prompt = update_prompt(before_state, event, row["reference"], revision, "train")
            expected_response = after_state
        else:
            expected_prompt = delta_prompt(before_state, after_state, row["reference"], revision, "train")
            expected_response = canonical_delta(row["operation"], keys)
    elif kind == "readout_sum":
        if row["operation"] is not None or not isinstance(row["answer"], int) or before != after:
            raise ValueError("malformed CBC readout row")
        expected_answer = int(before[keys[0]] + before[keys[1]])
        if row["answer"] != expected_answer:
            raise ValueError("CBC readout answer is semantically wrong")
        expected_prompt = sum_query_prompt(canonical_state(before, keys), keys, row["reference"], "train")
        expected_response = "answer={}".format(expected_answer)
    else:
        raise ValueError("unknown CBC train row kind")
    if row["completion_prompt"] != expected_prompt or row["response"] != expected_response:
        raise ValueError("CBC row prompt or target is not independently reproducible")


def audit_train_episode(rows):
    if not rows:
        raise ValueError("empty CBC train episode")
    by_world = defaultdict(list)
    for row in rows:
        by_world[row["world"]].append(row)
    if set(by_world) != {"normal", "counterfactual"}:
        raise ValueError("CBC train episode must include both worlds")
    world_initials, world_operations, world_answers = {}, {}, {}
    for world_name, world_rows in by_world.items():
        keys = _keys(world_rows[0]["keys"])
        initial = _values(world_rows[0]["initial_values"], keys)
        if any(_keys(row["keys"]) != keys or _values(row["initial_values"], keys) != initial for row in world_rows):
            raise ValueError("CBC train world has inconsistent metadata")
        compiles = [row for row in world_rows if str(row["kind"]).startswith("compile_")]
        if len(compiles) != 2 or {row["kind"] for row in compiles} != {"compile_a", "compile_b"}:
            raise ValueError("CBC train world must have exactly two compilation rows")
        updates = sorted((row for row in world_rows if row["kind"] == "update"), key=lambda row: int(row["revision"]))
        deltas = sorted((row for row in world_rows if row["kind"] == "inverse_delta"), key=lambda row: int(row["revision"]))
        readouts = [row for row in world_rows if row["kind"] == "readout_sum"]
        if len(updates) != len(deltas) or len(readouts) != 1:
            raise ValueError("CBC train world transition/readout count mismatch")
        current = initial
        for revision, (update, delta) in enumerate(zip(updates, deltas), 1):
            if int(update["revision"]) != revision or int(delta["revision"]) != revision:
                raise ValueError("CBC train revisions are not contiguous")
            if update["operation"] != delta["operation"] or _values(update["before_values"], keys) != current:
                raise ValueError("CBC train update/delta pair does not continue state")
            if _values(delta["before_values"], keys) != current or _values(update["after_values"], keys) != _values(delta["after_values"], keys):
                raise ValueError("CBC train paired transitions differ")
            current = _values(update["after_values"], keys)
        readout = readouts[0]
        if int(readout["revision"]) != len(updates) + 1 or _values(readout["before_values"], keys) != current:
            raise ValueError("CBC train readout is not terminal")
        world_initials[world_name] = initial
        world_operations[world_name] = tuple(row["operation"] for row in updates)
        world_answers[world_name] = int(readout["answer"])
    keys = _keys(by_world["normal"][0]["keys"])
    normal_initial, counter_initial = world_initials["normal"], world_initials["counterfactual"]
    if counter_initial[keys[0]] != normal_initial[keys[0]] + 1 or counter_initial[keys[1]] != normal_initial[keys[1]]:
        raise ValueError("CBC train counterfactual does not change exactly one initial fact")
    if world_operations["normal"] != world_operations["counterfactual"]:
        raise ValueError("CBC train worlds must share the same operation sequence")
    if world_answers["normal"] == world_answers["counterfactual"]:
        raise ValueError("CBC train counterfactual does not change final answer")


def audit_world(world, episode):
    if set(world) != EXPECTED_WORLD_FIELDS:
        raise ValueError("invalid CBC heldout world keys")
    keys = _keys(episode["keys"])
    initial = _values(world["initial_values"], keys)
    if world["initial_state"] != canonical_state(initial, keys):
        raise ValueError("CBC heldout initial state is wrong")
    if set(world["compile_prompts"]) != {"a", "b"}:
        raise ValueError("CBC heldout compile variants are incomplete")
    for variant, prompt in world["compile_prompts"].items():
        expected = compile_prompt(initial, keys, episode["domain"], episode["item"], episode["reference"], "heldout", variant)
        if prompt != expected:
            raise ValueError("CBC heldout compile prompt is wrong")
    current = initial
    if not isinstance(world["steps"], list) or not world["steps"]:
        raise ValueError("CBC heldout world must have transitions")
    for revision, step in enumerate(world["steps"], 1):
        if set(step) != EXPECTED_STEP_FIELDS or int(step["revision"]) != revision:
            raise ValueError("malformed CBC heldout transition")
        before = _values(step["before"], keys)
        after = _values(step["after"], keys)
        if before != current or after != apply_operation(current, step["operation"], keys):
            raise ValueError("CBC heldout transition is semantically wrong")
        before_state, after_state = canonical_state(before, keys), canonical_state(after, keys)
        if step["before_state"] != before_state or step["after_state"] != after_state:
            raise ValueError("CBC heldout serialized state is wrong")
        event = render_event(step["operation"], episode["item"], "heldout")
        if step["event"] != event or step["delta"] != canonical_delta(step["operation"], keys):
            raise ValueError("CBC heldout event or delta is wrong")
        if step["update_prompt"] != update_prompt(before_state, event, episode["reference"], revision, "heldout"):
            raise ValueError("CBC heldout update prompt is wrong")
        if step["delta_prompt"] != delta_prompt(before_state, after_state, episode["reference"], revision, "heldout"):
            raise ValueError("CBC heldout delta prompt is wrong")
        current = after
    if set(world["query"]) != EXPECTED_QUERY_FIELDS or world["query"]["kind"] != "sum":
        raise ValueError("malformed CBC heldout query")
    answer = int(current[keys[0]] + current[keys[1]])
    if world["query"]["answer"] != answer:
        raise ValueError("CBC heldout query answer is wrong")
    expected_prompt = sum_query_prompt(canonical_state(current, keys), keys, episode["reference"], "heldout")
    if world["query"]["prompt"] != expected_prompt:
        raise ValueError("CBC heldout query prompt is wrong")
    return initial, tuple(step["operation"] for step in world["steps"]), answer


def audit_episode(episode):
    if set(episode) != EXPECTED_EPISODE_FIELDS:
        raise ValueError("invalid CBC heldout episode keys")
    if episode["schema"] != "counterfactual_bisimulation_v1" or episode["split"] != "heldout" or episode["heldout"] is not True or episode["style"] != "heldout":
        raise ValueError("invalid CBC heldout episode metadata")
    if _domain_item(str(episode["domain"]), "heldout") != episode["item"]:
        raise ValueError("CBC heldout item does not match domain")
    if episode["regime"] not in {"cbc_len4", "cbc_len8", "cbc_len12"} or not str(episode["id"]) or not str(episode["reference"]):
        raise ValueError("invalid CBC heldout identity")
    normal_initial, normal_operations, normal_answer = audit_world(episode["normal"], episode)
    counter_initial, counter_operations, counter_answer = audit_world(episode["counterfactual"], episode)
    keys = _keys(episode["keys"])
    expected_length = int(str(episode["regime"]).removeprefix("cbc_len"))
    if len(normal_operations) != expected_length or normal_operations != counter_operations:
        raise ValueError("CBC heldout worlds must share the stated operation sequence")
    if counter_initial[keys[0]] != normal_initial[keys[0]] + 1 or counter_initial[keys[1]] != normal_initial[keys[1]]:
        raise ValueError("CBC heldout counterfactual does not change exactly one source fact")
    if counter_answer == normal_answer:
        raise ValueError("CBC heldout counterfactual does not change final answer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit("refusing to overwrite existing report: {}".format(report_path))

    train_prompts, heldout_prompts, heldout_grams = set(), set(), set()
    train_groups = defaultdict(list)
    train_kinds = Counter()
    invalid_rows = 0
    with open(args.train) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audit_row(row)
                prompt = normalized(row["completion_prompt"])
                if prompt in train_prompts:
                    raise ValueError("duplicate normalized CBC train prompt")
                train_prompts.add(prompt)
                train_groups[(row["episode_id"], row["reference"])].append(row)
                train_kinds[str(row["kind"])] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_rows += 1
    if invalid_rows == 0:
        for rows in train_groups.values():
            try:
                audit_train_episode(rows)
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1

    invalid_episodes, heldout_count = 0, 0
    heldout_regimes = Counter()
    with open(args.heldout) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                audit_episode(episode)
                heldout_count += 1
                heldout_regimes[str(episode["regime"])] += 1
                for prompt in controller_prompts(episode):
                    normalized_prompt = normalized(prompt)
                    if normalized_prompt in heldout_prompts:
                        raise ValueError("duplicate normalized CBC heldout controller prompt")
                    heldout_prompts.add(normalized_prompt)
                    heldout_grams.update(grams(prompt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_episodes += 1

    exact_hits = len(train_prompts & heldout_prompts)
    gram_hits = 0
    with open(args.train) as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                gram_hits += sum(1 for gram in grams(row["completion_prompt"]) if gram in heldout_grams)

    report = {
        "audit": "counterfactual_bisimulation_v1",
        "train": args.train,
        "heldout": args.heldout,
        "train_sha256": sha256_file(args.train),
        "heldout_sha256": sha256_file(args.heldout),
        "valid_train_rows": len(train_prompts),
        "valid_train_episodes": len(train_groups) if invalid_rows == 0 else 0,
        "valid_heldout_episodes": heldout_count,
        "invalid_train_rows": invalid_rows,
        "invalid_heldout_episodes": invalid_episodes,
        "duplicate_train_prompts": 0,
        "duplicate_heldout_prompts": 0,
        "train_heldout_exact_prompt_hits": exact_hits,
        "train_heldout_13gram_hits": gram_hits,
        "train_kinds": dict(sorted(train_kinds.items())),
        "heldout_regimes": dict(sorted(heldout_regimes.items())),
        "claim_boundary": "Data admission only; no model, compiler, or reasoning result is implied.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
