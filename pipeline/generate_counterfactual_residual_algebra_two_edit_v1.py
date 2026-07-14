#!/usr/bin/env python3
"""Build the pre-registered two-edit commutativity factor for CRA.

The evaluation target is a source-free composition of four native tapes:
``Z(donor) + Z(primary_edit) + Z(secondary_edit) - 2*Z(base)``.  It is never
used for training the first CRA arm.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from generate_counterfactual_residual_algebra_v1 import (
    TRAIN_PARAPHRASES,
    TRAIN_QUERIES,
    TRAIN_SOURCES,
    audit_sets,
    ngrams,
    readout,
    render_bundle,
    sha256_file,
    write_jsonl,
)


def choose(rng):
    values = tuple(range(-3, 4))
    deltas = (-1, 1)
    while True:
        primary, secondary = rng.choice(values), rng.choice(values)
        primary_delta, secondary_delta = rng.choice(deltas), rng.choice(deltas)
        donor_primary, donor_secondary = rng.choice(values), rng.choice(values)
        if all(value in values for value in (
            primary + primary_delta, secondary + secondary_delta,
            donor_primary + primary_delta, donor_secondary + secondary_delta,
        )) and (donor_primary, donor_secondary) != (primary, secondary):
            counter_delta = -primary_delta
            if donor_primary + counter_delta in values and primary + counter_delta in values:
                return primary, secondary, donor_primary, donor_secondary, primary_delta, secondary_delta, counter_delta


def source(template, primary, secondary):
    return template.format(primary=primary, secondary=secondary)


def render_two_edit_bundle(row, paraphrase=False, counterfactual=False):
    prefix = "paraphrase_" if paraphrase else ""
    primary = "counterfactual_primary_edited_source" if counterfactual else "primary_edited_source"
    return "Base world:\n{}\nPrimary edit:\n{}\nSecondary edit:\n{}\nDonor world:\n{}\n{}".format(
        row[prefix + "base_source"], row[prefix + primary], row[prefix + "secondary_edited_source"],
        row[prefix + "donor_source"], row["suffix_prompt"],
    )


def make_row(rng, index):
    p, q, r, s, dp, dq, counter_dp = choose(rng)
    kind, question = TRAIN_QUERIES[index % len(TRAIN_QUERIES)]
    template_index = index % len(TRAIN_SOURCES)
    paraphrase_index = (index // len(TRAIN_QUERIES)) % len(TRAIN_PARAPHRASES)
    target = (r + dp, s + dq)
    counter_target = (r + counter_dp, s + dq)
    row = {
        "schema": "counterfactual_residual_algebra_v1",
        "mode": "two_edit",
        "split": "two_edit",
        "episode_id": "two-edit-{:06d}".format(index),
        "base_source": source(TRAIN_SOURCES[template_index], p, q),
        "primary_edited_source": source(TRAIN_SOURCES[(template_index + 1) % len(TRAIN_SOURCES)], p + dp, q),
        "counterfactual_primary_edited_source": source(TRAIN_SOURCES[(template_index + 1) % len(TRAIN_SOURCES)], p + counter_dp, q),
        "secondary_edited_source": source(TRAIN_SOURCES[template_index], p, q + dq),
        "donor_source": source(TRAIN_SOURCES[(template_index + 1) % len(TRAIN_SOURCES)], r, s),
        "paraphrase_base_source": source(TRAIN_PARAPHRASES[paraphrase_index], p, q),
        "paraphrase_primary_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % len(TRAIN_PARAPHRASES)], p + dp, q),
        "paraphrase_counterfactual_primary_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % len(TRAIN_PARAPHRASES)], p + counter_dp, q),
        "paraphrase_secondary_edited_source": source(TRAIN_PARAPHRASES[paraphrase_index], p, q + dq),
        "paraphrase_donor_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % len(TRAIN_PARAPHRASES)], r, s),
        "suffix_prompt": "Question: {}\nAnswer:".format(question),
        "response": "answer={}".format(readout(target[0], target[1], kind)),
        "counterfactual_response": "answer={}".format(readout(counter_target[0], counter_target[1], kind)),
        "query_kind": kind,
        "primary_delta": dp,
        "secondary_delta": dq,
        "counterfactual_primary_delta": counter_dp,
        "state": {"base": {"primary": p, "secondary": q}, "donor": {"primary": r, "secondary": s},
                  "target": {"primary": target[0], "secondary": target[1]},
                  "counterfactual_target": {"primary": counter_target[0], "secondary": counter_target[1]}},
    }
    validate_row(row)
    return row


def validate_row(row):
    state = row["state"]
    donor, target, counter = state["donor"], state["target"], state["counterfactual_target"]
    if target["primary"] != donor["primary"] + row["primary_delta"] or target["secondary"] != donor["secondary"] + row["secondary_delta"]:
        raise ValueError("two-edit target is not solver-derived")
    if counter["primary"] != donor["primary"] + row["counterfactual_primary_delta"]:
        raise ValueError("counterfactual primary edit is not solver-derived")
    expected = "answer={}".format(readout(target["primary"], target["secondary"], row["query_kind"]))
    counter_expected = "answer={}".format(readout(counter["primary"], counter["secondary"], row["query_kind"]))
    if row["response"] != expected or row["counterfactual_response"] != counter_expected or expected == counter_expected:
        raise ValueError("two-edit response contract invalid")
    if "".join(str(value) for value in state["base"].values()) in row["suffix_prompt"]:
        raise ValueError("suffix leaked source state")


def state_key(row):
    state = row["state"]
    return (
        state["base"]["primary"], state["base"]["secondary"], state["donor"]["primary"], state["donor"]["secondary"],
        row["primary_delta"], row["secondary_delta"], row["counterfactual_primary_delta"], row["query_kind"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    out, audit_path = Path(args.out), Path(args.audit)
    if args.count <= 0 or out.exists() or audit_path.exists() or not Path(args.existing_train).is_file():
        raise SystemExit("count must be positive, outputs fresh, and frozen train data present")
    train = [json.loads(line) for line in open(args.existing_train) if line.strip()]
    if not train or any(row.get("schema") != "counterfactual_residual_algebra_v1" or row.get("split") != "train" for row in train):
        raise SystemExit("invalid frozen CRA train file")
    rng, rows, prompts = random.Random(args.seed), [], set()
    while len(rows) < args.count:
        row = make_row(rng, len(rows))
        prompt = render_two_edit_bundle(row)
        if prompt not in prompts:
            prompts.add(prompt)
            rows.append(row)
    train_prompts = {render_bundle(row) for row in train}
    train_states = {(
        row["state"]["base"]["primary"], row["state"]["base"]["secondary"], row["state"]["donor"]["primary"],
        row["state"]["donor"]["secondary"], row["delta"], row["counterfactual_delta"], row["query_kind"],
    ) for row in train}
    audit = {
        "audit": "counterfactual_residual_algebra_v1_two_edit",
        "train_sha256": sha256_file(Path(args.existing_train)),
        "rows": len(rows),
        "train_exact_bundle_hits": len(train_prompts & prompts),
        "train_exact_state_hits": len(train_states & {state_key(row) for row in rows}),
        "train_surface_13gram_hits": len(set().union(*(ngrams(prompt) for prompt in train_prompts)) & set().union(*(ngrams(prompt) for prompt in prompts))),
        "claim_boundary": "Pre-registered held-out two-edit commutativity control; no model result is created.",
    }
    if audit["train_exact_bundle_hits"] or audit["train_exact_state_hits"]:
        raise SystemExit("two-edit exact overlap audit failed: {}".format(audit))
    write_jsonl(out, rows)
    audit["sha256"] = sha256_file(out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
