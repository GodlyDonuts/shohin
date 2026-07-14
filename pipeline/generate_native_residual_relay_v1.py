#!/usr/bin/env python3
"""Build an audited counterfactual corpus for Native Residual Relay (NRR).

Rows supervise an answer only after a native residual relay crosses a hard
source-to-suffix cut.  Each source world has an independently worded paraphrase
and a one-fact counterfactual; the downstream event/query text contains neither
source values.  This builder is CPU-only and creates no model artifact.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path


TRAIN_SOURCES = (
    "The amber reserve contains {p} units, while the cobalt reserve contains {q} units.",
    "In a storehouse ledger, amber is recorded as {p} and cobalt as {q}.",
)
TRAIN_PARAPHRASES = (
    "A clerk counted {p} amber pieces and {q} cobalt pieces in the two bins.",
    "The two inventory entries read amber={p}; cobalt={q}.",
)
HELDOUT_SOURCES = (
    "North vault holds {p} tokens; south vault holds {q} tokens.",
    "The archive lists a north balance of {p} and a south balance of {q}.",
)
HELDOUT_PARAPHRASES = (
    "Records say the northern cache has {p} marks and the southern cache has {q} marks.",
    "For two vaults, the recorded amounts are north {p}, south {q}.",
)
TRAIN_EVENTS = (
    "An audit changes only the amber reserve by {delta:+d} units.",
    "Apply a {delta:+d} adjustment to amber and leave cobalt unchanged.",
)
HELDOUT_EVENTS = (
    "A transfer shifts the north balance by {delta:+d}; the south balance is untouched.",
    "Modify north by {delta:+d} marks and preserve south exactly.",
)
TRAIN_QUERIES = (
    ("difference", "What is amber minus cobalt?"),
    ("sum", "What is the combined amber and cobalt total?"),
)
HELDOUT_QUERIES = (
    ("difference", "Report north less south."),
    ("sum", "What total is stored across north and south?"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def answer(primary: int, secondary: int, delta: int, kind: str) -> int:
    primary += delta
    if kind == "difference":
        return primary - secondary
    if kind == "sum":
        return primary + secondary
    raise ValueError("unknown query kind")


def make_row(rng: random.Random, split: str, index: int) -> dict:
    heldout = split == "heldout"
    sources = HELDOUT_SOURCES if heldout else TRAIN_SOURCES
    paraphrases = HELDOUT_PARAPHRASES if heldout else TRAIN_PARAPHRASES
    events = HELDOUT_EVENTS if heldout else TRAIN_EVENTS
    queries = HELDOUT_QUERIES if heldout else TRAIN_QUERIES
    low, high = (1000, 8999) if heldout else (20, 899)
    delta_low, delta_high = (25, 99) if heldout else (1, 19)
    primary, secondary = rng.randint(low, high), rng.randint(low, high)
    delta = rng.randint(delta_low, delta_high) * (-1 if rng.randrange(2) else 1)
    counter_delta = rng.choice((-1, 1)) * rng.randint(delta_low, delta_high)
    kind, query = queries[index % len(queries)]
    source = sources[index % len(sources)].format(p=primary, q=secondary)
    paraphrase = paraphrases[index % len(paraphrases)].format(p=primary, q=secondary)
    counter_source = sources[index % len(sources)].format(p=primary + counter_delta, q=secondary)
    event = events[(index // len(queries)) % len(events)].format(delta=delta)
    suffix = "Event: {}\nQuestion: {}\nAnswer:".format(event, query)
    return {
        "schema": "native_residual_relay_v1",
        "split": split,
        "episode_id": "{}-{:06d}".format(split, index),
        "source": source,
        "paraphrase_source": paraphrase,
        "counterfactual_source": counter_source,
        "suffix_prompt": suffix,
        "response": "answer={}".format(answer(primary, secondary, delta, kind)),
        "counterfactual_response": "answer={}".format(answer(primary + counter_delta, secondary, delta, kind)),
        "query_kind": kind,
        "event_delta": delta,
        "counterfactual_primary_delta": counter_delta,
        "state": {"primary": primary, "secondary": secondary},
    }


def validate_row(row: dict) -> None:
    if row.get("schema") != "native_residual_relay_v1" or row.get("query_kind") not in {"sum", "difference"}:
        raise ValueError("row schema/query kind invalid")
    state = row.get("state")
    if not isinstance(state, dict):
        raise ValueError("row state missing")
    primary, secondary = int(state["primary"]), int(state["secondary"])
    expected = "answer={}".format(answer(primary, secondary, int(row["event_delta"]), row["query_kind"]))
    counter = "answer={}".format(answer(
        primary + int(row["counterfactual_primary_delta"]), secondary, int(row["event_delta"]), row["query_kind"],
    ))
    if row.get("response") != expected or row.get("counterfactual_response") != counter:
        raise ValueError("row answer is not solver-derived")
    if any(str(value) in row["suffix_prompt"] for value in (primary, secondary)):
        raise ValueError("suffix leaked a source value")
    if row["source"] == row["paraphrase_source"] or row["source"] == row["counterfactual_source"]:
        raise ValueError("paired sources are not distinct")


def render_prompt(row: dict, source_key: str = "source") -> str:
    return "Source: {}\n{}".format(row[source_key], row["suffix_prompt"])


def ngrams(text: str, width: int = 13):
    words = text.split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def build(split: str, count: int, seed: int):
    rng = random.Random(seed)
    rows, prompts = [], set()
    while len(rows) < count:
        row = make_row(rng, split, len(rows))
        validate_row(row)
        prompt = render_prompt(row)
        if prompt in prompts:
            continue
        prompts.add(prompt)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-count", type=int, default=30000)
    parser.add_argument("--heldout-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    paths = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if args.train_count <= 0 or args.heldout_count <= 0 or any(path.exists() for path in paths):
        raise SystemExit("counts must be positive and output paths must be fresh")
    train = build("train", args.train_count, args.seed)
    heldout = build("heldout", args.heldout_count, args.seed + 1)
    train_prompts = {render_prompt(row) for row in train}
    heldout_prompts = {render_prompt(row) for row in heldout}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    heldout_grams = set().union(*(ngrams(prompt) for prompt in heldout_prompts))
    write_jsonl(paths[0], train)
    write_jsonl(paths[1], heldout)
    report = {
        "audit": "native_residual_relay_v1",
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_sha256": sha256_file(paths[0]),
        "heldout_sha256": sha256_file(paths[1]),
        "duplicate_train_prompts": len(train) - len(train_prompts),
        "duplicate_heldout_prompts": len(heldout) - len(heldout_prompts),
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(train_grams & heldout_grams),
        "claim_boundary": "CPU-only corpus construction; no checkpoint, H100 run, or reasoning result is created.",
    }
    paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
