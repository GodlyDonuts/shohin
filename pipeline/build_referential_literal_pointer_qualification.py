#!/usr/bin/env python3
"""Build a fresh one-shot qualification board for the selected compiler.

The board uses only known language atoms but factor combinations and names that
do not occur in any public factorized split. The sealed factorized confirmation
split is intentionally not accepted as an input.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

import build_referential_literal_pointer_factorized_corpus as corpus
from semantic_compiler_falsifier import candidate_names, canonical_json, sha256_bytes, sha256_file


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_factor_specs(count, seed, excluded_signatures):
    candidates = [
        factors
        for factors in corpus.split_factor_candidates("development_compositional")
        if corpus.factor_signature(factors) not in excluded_signatures
    ]
    if count > len(candidates):
        raise ValueError("qualification factor request exceeds unused catalogue")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    selected_signatures = set()
    uncovered = corpus.expected_factor_atoms("known")
    for factors in candidates:
        if corpus.factor_atoms(factors) & uncovered:
            selected.append(factors)
            selected_signatures.add(corpus.factor_signature(factors))
            uncovered -= corpus.factor_atoms(factors)
            if not uncovered:
                break
    if uncovered:
        raise RuntimeError("qualification factor coverage failed: {}".format(sorted(uncovered)))
    for factors in candidates:
        signature = corpus.factor_signature(factors)
        if signature in selected_signatures:
            continue
        selected.append(factors)
        selected_signatures.add(signature)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("could not select enough qualification factors")
    rng.shuffle(selected)
    return selected


def build_qualification(groups, seed, tokenizer, name_pool, excluded_signatures):
    if groups <= 0 or len(name_pool) < 4:
        raise ValueError("invalid groups or qualification name pool")
    rng = random.Random(seed)
    configurations = corpus.valid_configurations()
    factor_specs = select_factor_specs(groups * 2, seed ^ 0x9E3779B9, excluded_signatures)
    rows = []
    seen_questions = set()
    for group_index in range(groups):
        configuration = rng.choice(configurations)
        entities_and_anchor = tuple(rng.sample(name_pool, 4))
        entities = entities_and_anchor[:3]
        neutral_anchor = entities_and_anchor[3]
        initial = tuple(entities[index] for index in configuration["initial_indices"])
        base = corpus.instantiate(configuration["base"], entities)
        order = corpus.instantiate(configuration["order"], entities)
        binding = corpus.instantiate(configuration["binding"], entities)
        query_position = configuration["query_position"]
        distractor_entity = initial[(query_position + 1) % 3]
        distractor_amount = base[0].amount
        canonical_factors = factor_specs[group_index * 2]
        paraphrase_factors = factor_specs[group_index * 2 + 1]
        surfaces = {
            "canonical": (base, canonical_factors),
            "paraphrase": (base, paraphrase_factors),
            "order_twin": (order, canonical_factors),
            "binding_twin": (binding, canonical_factors),
        }
        candidate = []
        for surface_type in corpus.SURFACE_TYPES:
            row = corpus.make_row(
                "development_compositional",
                group_index,
                surface_type,
                initial,
                surfaces[surface_type][0],
                query_position,
                surfaces[surface_type][1],
                distractor_entity,
                distractor_amount,
                neutral_anchor,
                tokenizer,
            )
            row["id"] = "RLPCQ-{:06d}-{}".format(group_index, surface_type)
            row["schema"] = "r12_referential_literal_pointer_qualification_row_v1"
            candidate.append(row)
        questions = [corpus.normalized(row["question"]) for row in candidate]
        if len(set(questions)) != 4 or any(question in seen_questions for question in questions):
            raise RuntimeError("qualification source collision in group {}".format(group_index))
        canonical, paraphrase, order_twin, binding_twin = candidate
        if corpus.row_label(canonical) != corpus.row_label(paraphrase):
            raise ValueError("qualification paraphrase program drift")
        if canonical["terminal_order"] != paraphrase["terminal_order"]:
            raise ValueError("qualification paraphrase behavior drift")
        if canonical["answer"] in {order_twin["answer"], binding_twin["answer"]}:
            raise ValueError("qualification twin separator failure")
        if not canonical["token_bag"] == order_twin["token_bag"] == binding_twin["token_bag"]:
            raise ValueError("qualification token bags diverged")
        rows.extend(candidate)
        seen_questions.update(questions)
    return rows


def source_names(rows):
    return {
        name
        for row in rows
        for name in (*row["initial_order"], row["neutral_anchor"])
    }


def audit_qualification(rows, reference_rows, tokenizer_path, generator_path):
    grouped = collections.defaultdict(dict)
    for row in rows:
        grouped[row["group"]][row["surface_type"]] = row
    group_gates = 0
    for group in grouped.values():
        canonical = group["canonical"]
        paraphrase = group["paraphrase"]
        order_twin = group["order_twin"]
        binding_twin = group["binding_twin"]
        group_gates += int(
            corpus.row_label(canonical) == corpus.row_label(paraphrase)
            and canonical["terminal_order"] == paraphrase["terminal_order"]
            and canonical["answer"] != order_twin["answer"]
            and canonical["answer"] != binding_twin["answer"]
            and canonical["token_bag"] == order_twin["token_bag"]
            and canonical["token_bag"] == binding_twin["token_bag"]
        )
    reference_questions = {corpus.normalized(row["question"]) for row in reference_rows}
    qualification_questions = {corpus.normalized(row["question"]) for row in rows}
    reference_ngrams = set().union(*(corpus.ngrams(row["question"]) for row in reference_rows))
    qualification_ngrams = set().union(*(corpus.ngrams(row["question"]) for row in rows))
    reference_signatures = {row["factor_signature"] for row in reference_rows}
    qualification_signatures = {row["factor_signature"] for row in rows}
    matched = [row for row in rows if row["surface_type"] != "paraphrase"]
    shortcuts = {
        "token_bag": corpus.shortcut_ceiling(
            matched, lambda row: canonical_json(row["token_bag"]),
        ),
        "absolute_pointer_positions": corpus.shortcut_ceiling(
            matched, corpus.pointer_positions,
        ),
        "source_token_length": corpus.shortcut_ceiling(
            matched, lambda row: str(row["token_count"]),
        ),
    }
    qualification_atoms = set().union(*(corpus.factor_atoms(row["factors"]) for row in rows))
    gates = {
        "all_ids_unique": len({row["id"] for row in rows}) == len(rows),
        "all_questions_unique": len(qualification_questions) == len(rows),
        "all_spans_present": all(
            set(corpus.REQUIRED_SPANS).issubset(row["spans"]) for row in rows
        ),
        "all_spans_nonempty": all(
            row["spans"][label]["token_positions"]
            for row in rows for label in corpus.REQUIRED_SPANS
        ),
        "two_executors_agree": all(row["executor_agreement"] for row in rows),
        "all_group_gates_pass": group_gates == len(grouped),
        "all_known_atoms_covered": qualification_atoms == corpus.expected_factor_atoms("known"),
        "public_exact_prompt_overlap_zero": not qualification_questions & reference_questions,
        "public_word_13gram_overlap_zero": not qualification_ngrams & reference_ngrams,
        "public_entity_name_overlap_zero": not source_names(rows) & source_names(reference_rows),
        "public_factor_combination_overlap_zero": not qualification_signatures & reference_signatures,
        "all_shortcut_ceilings_at_chance_plus_one_example": all(
            value["accuracy"] <= 1.0 / 3.0 + 1.0 / value["total"] + 1e-12
            for value in shortcuts.values()
        ),
    }
    return {
        "schema": "r12_referential_literal_pointer_qualification_report_v1",
        "claim_boundary": (
            "Fresh one-shot known-atom compiler qualification. No sealed factorized "
            "confirmation, executor, halt, autonomous rollout, or reasoning claim."
        ),
        "all_gates_pass": all(gates.values()),
        "structural_gates": gates,
        "rows": len(rows),
        "groups": len(grouped),
        "factor_combinations": len(qualification_signatures),
        "entity_names": len(source_names(rows)),
        "shortcut_ceilings": shortcuts,
        "public_overlap": {
            "exact_prompts": len(qualification_questions & reference_questions),
            "word_13grams": len(qualification_ngrams & reference_ngrams),
            "entity_names": len(source_names(rows) & source_names(reference_rows)),
            "factor_combinations": len(qualification_signatures & reference_signatures),
        },
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "generator_sha256": sha256_file(generator_path),
        "confirmation_access": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--reference-train", required=True)
    parser.add_argument("--reference-compositional", required=True)
    parser.add_argument("--reference-lexical-ood", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=2048)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    data_path = out_dir / "development_compositional.jsonl"
    report_path = out_dir / "report.json"
    if data_path.exists() or report_path.exists():
        raise SystemExit("refusing existing qualification output")
    reference_paths = [
        Path(args.reference_train),
        Path(args.reference_compositional),
        Path(args.reference_lexical_ood),
    ]
    reference_rows = [row for path in reference_paths for row in load_jsonl(path)]
    excluded_signatures = {row["factor_signature"] for row in reference_rows}
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    names, nonce_width = candidate_names(tokenizer, 4135)
    used_names = source_names(reference_rows)
    name_pool = [name for name in names if name not in used_names]
    rows = build_qualification(
        args.groups, args.seed, tokenizer, name_pool, excluded_signatures,
    )
    report = audit_qualification(rows, reference_rows, tokenizer_path, Path(__file__))
    report["seed"] = args.seed
    report["nonce_token_width"] = nonce_width
    report["reference_artifacts"] = {
        path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in reference_paths
    }
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    report["artifacts"] = {
        "development_compositional": {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(payload)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": report["all_gates_pass"],
        "data": str(data_path.resolve()),
        "data_sha256": report["artifacts"]["development_compositional"]["sha256"],
        "report": str(report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("qualification audit failed")


if __name__ == "__main__":
    main()
