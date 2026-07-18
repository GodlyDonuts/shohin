#!/usr/bin/env python3
"""Generate typed-controller curriculum with easy-multiply upsampling."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from generate_typed_controller_v1 import build_corpus

PROTOCOL = "R12-TYPED-HOST-CORRECTED-TARGETS-v1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026071712)
    ap.add_argument("--n-train-cases", type=int, default=8000)
    ap.add_argument("--n-heldout-cases", type=int, default=256)
    ap.add_argument("--max-multiply", type=int, default=12)
    args = ap.parse_args()

    train, held = build_corpus(
        seed=args.seed, n_train=args.n_train_cases, n_heldout=args.n_heldout_cases
    )

    def is_easy(row: dict) -> bool:
        m = re.search(r"multiply (\d+)", row.get("completion_prompt", ""))
        if not m:
            return True
        return int(m.group(1)) <= args.max_multiply

    easy = [r for r in train if is_easy(r)]
    train = train + easy

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    dump(args.out_dir / "train.jsonl", train)
    dump(args.out_dir / "heldout.jsonl", held)
    audit = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "n_train_rows": len(train),
        "n_heldout_rows": len(held),
        "max_multiply_curriculum": args.max_multiply,
        "note": "v1 labels are already host-correct; upsamples easy multiplies.",
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
