#!/usr/bin/env python3
"""Read-only intake probe for a prospective reasoning-training dataset.

This is deliberately not a curator. It streams a small fixed sample, records
schema/license metadata and checks every sampled string against the live eval
prompts. A source may only get a separate, decontaminated builder after this
report is reviewed; this command never writes model or training-data outputs.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


WORD_RE = re.compile(r"\w+")
PROMPT_FIELDS = ("question", "problem", "prompt", "task", "text")


def normalize(text):
    return " ".join(WORD_RE.findall(str(text).lower()))


def grams(text, n=13):
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index:index + n]) for index in range(len(words) - n + 1)}


def iter_strings(value):
    """Yield textual leaves without serializing a dataset row into the report."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from iter_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from iter_strings(nested)


def load_eval_index(evals_dir):
    exact = set()
    ngrams = set()
    paths = sorted(Path(evals_dir).glob("*.jsonl"))
    for path in paths:
        with path.open(errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = next((row[field] for field in PROMPT_FIELDS if row.get(field)), "")
                clean = normalize(prompt)
                if clean:
                    exact.add(clean)
                    ngrams.update(grams(clean))
    return {"exact": exact, "ngrams": ngrams, "files": [str(path) for path in paths]}


def parse_card_frontmatter(text):
    """Extract the flat card declarations we must preserve for provenance."""
    match = re.match(r"^---\s*\n(.*?)\n---(?:\s*\n|$)", text, flags=re.DOTALL)
    if not match:
        return {}
    metadata = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"license", "language", "pretty_name"}:
            metadata[key.strip()] = value.strip().strip('"\'')
    return metadata


def card_summary(dataset):
    """Read the public dataset card into the HF cache without persisting card text."""
    try:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=dataset, filename="README.md", repo_type="dataset"))
        text = path.read_text(errors="replace")
        return {
            "available": True,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text),
            "frontmatter": parse_card_frontmatter(text),
        }
    except Exception as exc:  # The report must reveal, not hide, missing provenance.
        return {"available": False, "error": type(exc).__name__}


def describe_rows(rows, eval_index):
    field_types = defaultdict(Counter)
    field_chars = defaultdict(list)
    field_nonempty = Counter()
    all_fields = set()
    exact_rows = ngram_rows = 0
    exact_fields = Counter()
    ngram_fields = Counter()

    for row in rows:
        exact_hit = ngram_hit = False
        all_fields.update(row)
        for field, value in row.items():
            field_types[field][type(value).__name__] += 1
            values = list(iter_strings(value))
            if values:
                field_nonempty[field] += 1
                field_chars[field].append(sum(len(item) for item in values))
            for item in values:
                clean = normalize(item)
                if not clean:
                    continue
                if clean in eval_index["exact"]:
                    exact_hit = True
                    exact_fields[field] += 1
                if grams(clean).intersection(eval_index["ngrams"]):
                    ngram_hit = True
                    ngram_fields[field] += 1
        exact_rows += int(exact_hit)
        ngram_rows += int(ngram_hit)

    fields = {}
    for field in sorted(all_fields):
        sizes = sorted(field_chars[field])
        fields[field] = {
            "types": dict(sorted(field_types[field].items())),
            "nonempty_rows": field_nonempty[field],
            "total_chars_min": sizes[0] if sizes else 0,
            "total_chars_median": sizes[len(sizes) // 2] if sizes else 0,
            "total_chars_max": sizes[-1] if sizes else 0,
        }
    return {
        "sample_rows": len(rows),
        "top_level_fields": fields,
        "sample_eval_overlap": {
            "exact_prompt_rows": exact_rows,
            "eval_13gram_rows": ngram_rows,
            "exact_hits_by_field": dict(sorted(exact_fields.items())),
            "eval_13gram_hits_by_field": dict(sorted(ngram_fields.items())),
        },
    }


def builder_summary(builder):
    info = builder.info
    features = getattr(info, "features", None)
    return {
        "builder_name": getattr(builder, "builder_name", None),
        "config_name": getattr(builder, "config_name", None),
        "license": getattr(info, "license", None),
        "homepage": getattr(info, "homepage", None),
        "description_chars": len(getattr(info, "description", "") or ""),
        "citation_chars": len(getattr(info, "citation", "") or ""),
        "features": str(features) if features is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--evals-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    # Keep imports here so the deterministic helper tests do not need datasets.
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset, load_dataset_builder

    configs = get_dataset_config_names(args.dataset)
    if args.config and args.config not in configs:
        raise SystemExit(f"requested config {args.config!r} not in discovered configs")
    config = args.config
    if config is None and len(configs) == 1:
        config = configs[0]
    if config is None:
        raise SystemExit(
            "dataset exposes multiple configs; resubmit with --config after reviewing: "
            + ", ".join(configs)
        )
    splits = get_dataset_split_names(args.dataset, config)
    if args.split not in splits:
        raise SystemExit(f"requested split {args.split!r} not in discovered splits {splits}")

    builder = load_dataset_builder(args.dataset, name=config)
    kwargs = {"split": args.split, "streaming": True}
    kwargs["name"] = config
    rows = []
    for row in load_dataset(args.dataset, **kwargs):
        rows.append(dict(row))
        if len(rows) >= args.max_rows:
            break
    if not rows:
        raise SystemExit("stream returned zero rows")

    eval_index = load_eval_index(args.evals_dir)
    report = {
        "schema": "shohin-external-reasoning-source-probe-v1",
        "purpose": "read-only schema, provenance, and sampled contamination intake",
        "dataset": args.dataset,
        "requested_config": args.config,
        "selected_config": config,
        "selected_split": args.split,
        "discovered_configs": configs,
        "discovered_splits": splits,
        "builder": builder_summary(builder),
        "dataset_card": card_summary(args.dataset),
        "eval_prompt_files": eval_index["files"],
        "eval_prompt_count": len(eval_index["exact"]),
        "eval_13gram_count": len(eval_index["ngrams"]),
        "sample": describe_rows(rows, eval_index),
        "admission_status": "inspection_only_not_training_data",
        "execution_performed": False,
    }
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    partial.replace(out)
    print(json.dumps({
        "dataset": args.dataset,
        "config": config,
        "split": args.split,
        "rows": len(rows),
        "exact_eval_rows": report["sample"]["sample_eval_overlap"]["exact_prompt_rows"],
        "eval_13gram_rows": report["sample"]["sample_eval_overlap"]["eval_13gram_rows"],
        "license": report["builder"]["license"],
        "card_license": report["dataset_card"].get("frontmatter", {}).get("license"),
        "out": str(out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
