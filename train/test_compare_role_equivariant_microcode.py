#!/usr/bin/env python3
"""Contracts for the r3 matched comparator."""

from compare_role_equivariant_microcode import compare


def report(semantic, permutation, fit, depth, language, full, program, advance):
    metadata = {
        "base_sha256": "b", "data_sha256": "d", "admission_sha256": "a", "seed": 7,
        "layer": 19, "hidden": 256, "batch_groups": 4, "selected_groups": 100,
        "selected_examples": 600, "updates": 25, "learning_rate": 0.001,
        "warmup_updates": 50, "gradient_clip": 1.0, "basis_weight": 1.0,
        "role_factor_contract": "signed-z2-feature-v1", "initial_adapter_sha256": "i",
        "semantic_weight": semantic, "permutation_weight": permutation,
    }
    summary = {}
    for name, accuracy in (("fit_iid", fit), ("depth_ood", depth), ("language_ood", language), ("full_ood", full)):
        summary[name] = {"answer_accuracy": accuracy, "answer_correct": round(accuracy * 100), "cases": 100}
    summary["all"] = {"program_exact_accuracy": program}
    return {"adapter_metadata": metadata, "summary": summary, "advance_to_decoder_bridge": advance}


def main():
    control = report(0.0, 0.0, 0.9, 0.8, 0.5, 0.4, 0.55, True)
    candidate = report(0.5, 1.0, 0.89, 0.79, 0.58, 0.5, 0.62, True)
    result = compare(control, candidate)
    assert result["constraints_attributed"]
    assert result["decision"] == "advance_role_equivariant_compiler_to_manual_bridge_gate"
    weak = report(0.5, 1.0, 0.89, 0.79, 0.51, 0.41, 0.56, True)
    result = compare(control, weak)
    assert not result["constraints_attributed"]
    assert result["decision"] == "advance_role_factorized_control_only_constraints_not_attributed"
    print("role-equivariant comparator tests passed")


if __name__ == "__main__":
    main()
