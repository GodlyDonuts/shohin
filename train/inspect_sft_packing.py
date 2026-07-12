#!/usr/bin/env python3
"""Audit packed SFT group capacity before choosing oversampling weights."""
import argparse
import collections
import json
from pathlib import Path

from tokenizers import Tokenizer

from sft import build_packed


def parse_weights(items):
    result = {}
    for item in items:
        group, sep, value = item.partition("=")
        if not sep or not group:
            raise ValueError(f"invalid weight {item!r}; expected group=value")
        result[group] = float(value)
    if result and sum(result.values()) <= 0:
        raise ValueError("weights must sum to a positive value")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pack-len", type=int, default=2048)
    parser.add_argument("--group-field", default="training_group")
    parser.add_argument("--prompt-override-field", default="completion_prompt")
    parser.add_argument("--weights", nargs="*", default=[])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")
    _, _, groups = build_packed(
        args.data, tokenizer, args.pack_len,
        ["question", "problem", "prompt", "instruction"],
        ["response", "answer", "solution", "completion", "output"],
        eos_id,
        group_field=args.group_field,
        prompt_override_field=args.prompt_override_field,
    )
    counts = collections.Counter(map(str, groups))
    total = len(groups)
    weights = parse_weights(args.weights)
    weight_sum = sum(weights.values())
    report = {
        "data": args.data,
        "pack_len": args.pack_len,
        "packed_sequences": total,
        "group_counts": dict(sorted(counts.items())),
        "natural_group_share": {
            group: count / max(total, 1) for group, count in sorted(counts.items())
        },
    }
    if weights:
        report["requested_weights"] = weights
        report["requested_samples_per_epoch"] = {
            group: total * weight / weight_sum for group, weight in weights.items()
        }
        report["repeat_factor_per_epoch"] = {
            group: (total * weight / weight_sum) / max(counts.get(group, 0), 1)
            for group, weight in weights.items()
        }
        report["missing_weight_groups"] = sorted(set(weights) - set(counts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
