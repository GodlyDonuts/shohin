#!/usr/bin/env python3
"""Assess the frozen factorized complete-compiler development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ARMS = ("free_slots", "structured", "islands", "ordinary", "shuffled_islands")
PRIMARY_GATES = {
    "answer_accuracy": 0.85,
    "semantic_program_exact": 0.75,
    "full_pointer_exact": 0.65,
    "kind_accuracy": 0.95,
    "initial_joint": 0.80,
}
EXPECTED_PROTOCOLS = {
    "free_slots": "r12_referential_literal_pointer_compiler_v1_1_factorized_development",
    "structured": (
        "r12_referential_literal_pointer_compiler_v1_2_structured_factorized_development"
    ),
    "islands": "r12_referential_literal_pointer_compiler_v1_3_islands_factorized_development",
    "ordinary": (
        "r12_referential_literal_pointer_compiler_ordinary_tagger_factorized_development"
    ),
    "shuffled_islands": (
        "r12_referential_literal_pointer_compiler_shuffled_islands_factorized_development"
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(path, expected_split, expected_oracle="none"):
    result = json.loads(Path(path).read_text())
    if result.get("split") != expected_split:
        raise ValueError("{} split mismatch".format(path))
    if result.get("confirmation_access") != 0:
        raise ValueError("{} accessed confirmation".format(path))
    if result.get("oracle", "none") != expected_oracle:
        raise ValueError("{} oracle mismatch".format(path))
    return result


def primary_gate_results(islands, ordinary):
    overall = islands["overall"]
    groups = islands["group_summary"]
    gates = {
        name: {
            "value": float(overall[name]),
            "floor": floor,
            "pass": float(overall[name]) >= floor,
        }
        for name, floor in PRIMARY_GATES.items()
    }
    gates["canonical_paraphrase_both_exact"] = {
        "value": int(groups["canonical_paraphrase_both_exact"]),
        "floor": 192,
        "pass": int(groups["canonical_paraphrase_both_exact"]) >= 192,
    }
    gates["all_four_exact"] = {
        "value": int(groups["all_four_full_pointer_exact"]),
        "floor": 96,
        "pass": int(groups["all_four_full_pointer_exact"]) >= 96,
    }
    advantage = (
        float(islands["overall"]["semantic_program_exact"])
        - float(ordinary["overall"]["semantic_program_exact"])
    )
    gates["islands_program_advantage_over_ordinary"] = {
        "value": advantage,
        "floor": 0.05,
        "pass": advantage >= 0.05,
    }
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="train")
    parser.add_argument("--seed", type=int, default=2026071810)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output_path = Path(args.out)
    if output_path.exists():
        raise SystemExit("refusing existing assessment output")

    train_root = Path(args.train_root)
    arm_data = {}
    shared_metadata = {}
    artifact_hashes = {}
    for arm in ARMS:
        arm_dir = train_root / "referential_literal_pointer_factorized_{}_{}".format(
            arm, args.seed,
        )
        adapter_path = arm_dir / "compiler.pt"
        compositional_path = arm_dir / "development_compositional.json"
        lexical_path = arm_dir / "development_lexical_ood.json"
        bundle = torch.load(adapter_path, map_location="cpu")
        metadata = bundle.get("compiler", {})
        if metadata.get("protocol") != EXPECTED_PROTOCOLS[arm]:
            raise SystemExit("{} protocol mismatch".format(arm))
        if metadata.get("confirmation_access") != 0:
            raise SystemExit("{} metadata accessed confirmation".format(arm))
        if metadata.get("examples") != 96000 or metadata.get("updates") != 1517:
            raise SystemExit("{} training budget mismatch".format(arm))
        if metadata.get("seed") != args.seed:
            raise SystemExit("{} seed mismatch".format(arm))
        for field in (
            "base_sha256", "data_sha256", "report_sha256", "tokenizer_sha256",
            "examples", "updates", "seed",
        ):
            value = metadata.get(field)
            if field in shared_metadata and shared_metadata[field] != value:
                raise SystemExit("shared metadata mismatch for {}".format(field))
            shared_metadata[field] = value
        arm_data[arm] = {
            "metadata": metadata,
            "compositional": load_result(
                compositional_path, "development_compositional",
            ),
            "lexical_ood": load_result(lexical_path, "development_lexical_ood"),
        }
        artifact_hashes[arm] = {
            "adapter": sha256_file(adapter_path),
            "compositional": sha256_file(compositional_path),
            "lexical_ood": sha256_file(lexical_path),
        }

    oracle_results = {}
    for arm in ("islands", "ordinary"):
        arm_dir = train_root / "referential_literal_pointer_factorized_{}_{}".format(
            arm, args.seed,
        )
        oracle_results[arm] = {}
        for oracle in ("lexical", "structure", "full"):
            path = arm_dir / "development_lexical_ood_{}_oracle.json".format(oracle)
            result = load_result(path, "development_lexical_ood", oracle)
            oracle_results[arm][oracle] = {
                "overall": result["overall"],
                "group_summary": result["group_summary"],
                "sha256": sha256_file(path),
            }

    gates = primary_gate_results(
        arm_data["islands"]["compositional"],
        arm_data["ordinary"]["compositional"],
    )
    absolute_gates_pass = all(
        gate["pass"] for name, gate in gates.items()
        if name != "islands_program_advantage_over_ordinary"
    )
    attribution_pass = gates["islands_program_advantage_over_ordinary"]["pass"]
    confirmation_authorized = absolute_gates_pass and attribution_pass
    assessment = {
        "schema": "r12_referential_literal_pointer_factorized_assessment_v1",
        "shared_metadata": shared_metadata,
        "arms": {
            arm: {
                "adapter_parameters": data["metadata"]["adapter_parameters"],
                "total_parameters": data["metadata"]["total_parameters"],
                "protocol": data["metadata"]["protocol"],
                "compositional_overall": data["compositional"]["overall"],
                "compositional_groups": data["compositional"]["group_summary"],
                "lexical_ood_overall": data["lexical_ood"]["overall"],
                "lexical_ood_groups": data["lexical_ood"]["group_summary"],
            }
            for arm, data in arm_data.items()
        },
        "oracles": oracle_results,
        "primary_gates": gates,
        "absolute_gates_pass": absolute_gates_pass,
        "attribution_pass": attribution_pass,
        "confirmation_authorized": confirmation_authorized,
        "decision": (
            "run_sealed_confirmation_once"
            if confirmation_authorized else
            "retain_as_conventional_compiler_baseline_confirmation_sealed"
            if absolute_gates_pass else
            "reject_factorized_curriculum_architecture_pair"
        ),
        "artifact_hashes": artifact_hashes,
        "claim_boundary": (
            "Development-only source compiler assessment. Oracle rows are explicit host ceilings. "
            "No confirmation, executor, halt, autonomous-rollout, or native-reasoning claim."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(output_path.resolve()),
        "decision": assessment["decision"],
        "absolute_gates_pass": absolute_gates_pass,
        "attribution_pass": attribution_pass,
        "confirmation_authorized": confirmation_authorized,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
