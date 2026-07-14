#!/usr/bin/env python3
"""Build factorized NRR held-out evaluations without altering the frozen train mix.

Each family shifts exactly one of natural wording, numeric values, or event
delta magnitude relative to train; ``combined`` shifts all three.  The
admission report compares every factor family against the immutable train
prompts and against the other factor families.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from generate_native_residual_relay_v1 import make_row_axes, ngrams, render_prompt, sha256_file, validate_row, write_jsonl


REGIMES = {
    "language": (True, False, False),
    "values": (False, True, False),
    "delta": (False, False, True),
    "combined": (True, True, True),
}


def build_regime(name, count, seed):
    rng, rows, prompts = random.Random(seed), [], set()
    axes = REGIMES[name]
    while len(rows) < count:
        template_family = None if name == "language" else "factor"
        row = make_row_axes(rng, "factor_" + name, len(rows), *axes, template_family=template_family)
        validate_row(row)
        prompt = render_prompt(row)
        if prompt in prompts:
            continue
        prompts.add(prompt)
        rows.append(row)
    return rows, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    train, out_dir, report_path = Path(args.train), Path(args.out_dir), Path(args.report)
    if args.count <= 0 or not train.is_file() or report_path.exists() or out_dir.exists():
        raise SystemExit("train must exist, count positive, and factor outputs fresh")
    train_prompts = {render_prompt(json.loads(line)) for line in train.read_text().splitlines() if line.strip()}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    prompts, grams, paths = {}, {}, {}
    for offset, name in enumerate(sorted(REGIMES)):
        rows, prompts[name] = build_regime(name, args.count, args.seed + offset)
        paths[name] = out_dir / "native_residual_relay_v1_factor_{}.jsonl".format(name)
        write_jsonl(paths[name], rows)
        grams[name] = set().union(*(ngrams(prompt) for prompt in prompts[name]))
    report = {
        "audit": "native_residual_relay_v1_factor_admission",
        "train_sha256": sha256_file(train),
        "count_per_regime": args.count,
        "regimes": {name: {"sha256": sha256_file(paths[name]), "rows": args.count,
                           "train_exact_prompt_hits": len(train_prompts & prompts[name]),
                           "train_13gram_hits": len(train_grams & grams[name])} for name in sorted(REGIMES)},
        "cross_regime_exact_prompt_hits": {
            "{}_{}".format(left, right): len(prompts[left] & prompts[right])
            for index, left in enumerate(sorted(REGIMES)) for right in sorted(REGIMES)[index + 1:]
        },
        "claim_boundary": "CPU-only factorized evaluation data; no model artifact or capability result is created.",
    }
    if any(value["train_exact_prompt_hits"] or value["train_13gram_hits"] for value in report["regimes"].values()):
        raise SystemExit("factor overlap with frozen train prompts")
    if any(report["cross_regime_exact_prompt_hits"].values()):
        raise SystemExit("factor regime exact-prompt overlap")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
