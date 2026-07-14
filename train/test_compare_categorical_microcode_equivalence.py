#!/usr/bin/env python3
"""Contracts for the matched semantic-equivalence comparator."""

from compare_categorical_microcode_equivalence import compare


def report(weight, fit, depth, language, full, program, advance):
    meta = {
        "base_sha256": "b", "data_sha256": "d", "admission_sha256": "a", "seed": 7,
        "layer": 19, "hidden": 256, "batch_pairs": 8, "selected_pairs": 100,
        "selected_examples": 200, "initial_adapter_sha256": "i", "equivalence_weight": weight,
    }
    summary = {}
    for name, accuracy, cases in (
        ("fit_iid", fit, 100), ("depth_ood", depth, 100),
        ("language_ood", language, 100), ("full_ood", full, 100),
    ):
        summary[name] = {"answer_accuracy": accuracy, "answer_correct": round(accuracy * cases), "cases": cases}
    summary["all"] = {"program_exact_accuracy": program}
    return {"adapter_metadata": meta, "summary": summary, "advance_to_decoder_bridge": advance}


def main():
    control = report(0.0, 0.9, 0.8, 0.5, 0.4, 0.55, True)
    equivalence = report(0.2, 0.89, 0.79, 0.58, 0.5, 0.62, True)
    result = compare(control, equivalence)
    assert result["equivalence_attributed"]
    assert result["decision"] == "advance_equivalence_compiler_to_manual_bridge_gate"
    weak = report(0.2, 0.89, 0.79, 0.51, 0.41, 0.56, True)
    result = compare(control, weak)
    assert not result["equivalence_attributed"]
    assert result["decision"] == "advance_diverse_control_only_equivalence_not_attributed"
    print("categorical microcode equivalence comparator tests passed")


if __name__ == "__main__":
    main()
