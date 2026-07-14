#!/usr/bin/env python3
"""Regime-aware full-text overlap audit for anchor-replay compiler data.

The exact r1 anchor syntax intentionally matches the surface family used by
fit-IID and depth-only diagnostic slices.  This audit permits only that narrow
relationship, records it explicitly, and rejects overlap with language/full
OOD, the frozen manual board, or any other evaluation file.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


WORD = re.compile(r"\w+")
EVAL_FIELDS = ("question", "problem", "prompt", "instruction", "task", "text")
TRAIN_FIELDS = ("question", "response")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def grams(text, width):
    words = WORD.findall(str(text).lower())
    if not words:
        return ()
    if len(words) < width:
        return (" ".join(words),)
    return tuple(" ".join(words[index:index + width]) for index in range(len(words) - width + 1))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_label(path, row):
    return "{}:{}".format(path.name, str(row.get("eval_regime") or "unspecified"))


def build_eval_index(evals, width):
    exact_sources = collections.defaultdict(set)
    gram_sources = collections.defaultdict(set)
    files = []
    for path in sorted(Path(evals).glob("*.jsonl")):
        files.append(str(path))
        with path.open(errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = next((row[field] for field in EVAL_FIELDS if row.get(field)), "")
                clean = normalized(prompt)
                if not clean:
                    continue
                label = eval_label(path, row)
                exact_sources[clean].add(label)
                for gram in grams(prompt, width):
                    gram_sources[gram].add(label)
    return exact_sources, gram_sources, files


def audit_rows(rows, exact_sources, gram_sources, width, max_examples=24):
    allowed_labels = {
        "latent_operator_eval_slices_v2_64.jsonl:fit_iid",
        "latent_operator_eval_slices_v2_64.jsonl:depth_ood",
    }
    valid = malformed = allowed_rows = forbidden_rows = exact_rows = 0
    allowed_by_view = collections.Counter()
    semantic_view_rows = collections.Counter()
    forbidden_by_label = collections.Counter()
    examples = []
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            malformed += 1
            continue
        valid += 1
        view = str(row.get("semantic_view") or "missing")
        semantic_view_rows[view] += 1
        row_sources = set()
        row_exact = False
        for field in TRAIN_FIELDS:
            value = row.get(field)
            if value is None or not str(value).strip():
                row_sources.add("missing:{}".format(field))
                continue
            clean = normalized(value)
            direct = exact_sources.get(clean, set())
            row_sources.update(direct)
            row_exact = row_exact or bool(direct)
            for gram in grams(value, width):
                row_sources.update(gram_sources.get(gram, set()))
        exact_rows += int(row_exact)
        if not row_sources:
            continue
        permitted = view == "anchor" and row_sources <= allowed_labels
        if permitted:
            allowed_rows += 1
            allowed_by_view[view] += 1
        else:
            forbidden_rows += 1
            for label in row_sources:
                forbidden_by_label[label] += 1
            if len(examples) < max_examples:
                examples.append({
                    "line": line_number,
                    "reference": str(row.get("reference") or ""),
                    "semantic_view": view,
                    "sources": sorted(row_sources),
                })
    return {
        "valid_rows": valid,
        "malformed_rows": malformed,
        "exact_rows": exact_rows,
        "allowed_same_surface_rows": allowed_rows,
        "allowed_by_view": dict(sorted(allowed_by_view.items())),
        "semantic_view_rows": dict(sorted(semantic_view_rows.items())),
        "forbidden_rows": forbidden_rows,
        "forbidden_by_label": dict(sorted(forbidden_by_label.items())),
        "forbidden_examples": examples,
        "allowed_labels": sorted(allowed_labels),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--programs", type=int, required=True)
    args = parser.parse_args()
    if args.ngram <= 0 or args.programs <= 0:
        raise SystemExit("ngram and programs must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing report")
    exact_sources, gram_sources, files = build_eval_index(args.evals, args.ngram)
    malformed_json = [0]

    def rows():
        with open(args.data, errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    malformed_json[0] += 1

    report = {
        "schema": "role-equivariant-full-text-v3",
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256(args.data),
        "eval_files": files,
        "eval_exact_prompts": len(exact_sources),
        "eval_ngrams": len(gram_sources),
        "ngram": args.ngram,
        "programs": args.programs,
        **audit_rows(rows(), exact_sources, gram_sources, args.ngram),
        "malformed_json_rows": malformed_json[0],
    }
    report["all_checks_pass"] = (
        report["valid_rows"] == args.programs * 6
        and report["semantic_view_rows"] == {
            "anchor": args.programs * 2,
            "paraphrase_a": args.programs * 2,
            "paraphrase_b": args.programs * 2,
        }
        and not report["malformed_rows"]
        and not report["malformed_json_rows"]
        and not report["exact_rows"]
        and report["allowed_same_surface_rows"] <= args.programs * 2
        and report["allowed_by_view"] == (
            {"anchor": report["allowed_same_surface_rows"]}
            if report["allowed_same_surface_rows"] else {}
        )
        and not report["forbidden_rows"]
    )
    report["claim_boundary"] = (
        "Exactly two anchor rows per program are present. Any overlap may come only from anchor "
        "boilerplate shared with fit-IID/depth-only diagnostics; language/full OOD, manual, and "
        "every other evaluation source must remain zero. Clean anchors need not overlap."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_checks_pass": report["all_checks_pass"],
        "allowed_same_surface_rows": report["allowed_same_surface_rows"],
        "forbidden_rows": report["forbidden_rows"],
        "out": str(out),
    }, sort_keys=True))
    if not report["all_checks_pass"]:
        raise SystemExit("role-equivariant full-text gate failed")


if __name__ == "__main__":
    main()
