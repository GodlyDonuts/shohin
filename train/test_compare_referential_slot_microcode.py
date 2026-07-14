#!/usr/bin/env python3
"""Contracts for the locked referential slot comparator."""

import copy
import json
import subprocess
import tempfile
from pathlib import Path


def report(role_mode, language, full, exact, role_accuracy):
    metadata = {
        "protocol": "causal_microcode_referential_slots_v4", "base_sha256": "b",
        "data_sha256": "d", "admission_sha256": "a", "label_admission_sha256": "l",
        "seed": 1, "layer": 1, "hidden": 8, "batch_groups": 1, "selected_groups": 1,
        "selected_examples": 6, "updates": 1, "learning_rate": 1e-3,
        "warmup_updates": 1, "gradient_clip": 1.0, "basis_weight": 1.0,
        "mention_weight": 1.0, "adapter_parameters": 10, "base_parameters_trainable": 0,
        "initial_adapter_sha256": "i", "view_contract": [],
        "inference_inputs": "token states plus formatting-derived intro/event/query spans only",
        "role_mode": role_mode,
    }
    summary = {
        "language_ood": {"answer_correct": language, "cases": 100, "answer_accuracy": language / 100},
        "full_ood": {"answer_correct": full, "cases": 100, "answer_accuracy": full / 100},
        "all": {"program_exact_accuracy": exact},
        "fit_iid": {"answer_accuracy": 0.9}, "depth_ood": {"answer_accuracy": 0.8},
    }
    records = [{
        "regime": "language_ood", "operation_kind_targets": [0] * 100,
        "operation_kind_predictions": [0] * 100, "operation_role_targets": [0] * 100,
        "operation_role_predictions": [0] * int(role_accuracy * 100) + [1] * (100 - int(role_accuracy * 100)),
    }]
    return {
        "audit": "referential_slot_microcode_eval_v4", "adapter_metadata": metadata,
        "summary": summary, "records": records, "gates": {"absolute": True},
    }


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        control = report("absolute", 20, 10, 0.40, 0.50)
        candidate = report("pointer", 30, 20, 0.50, 0.70)
        control_path, candidate_path, out = directory / "c.json", directory / "p.json", directory / "out.json"
        control_path.write_text(json.dumps(control))
        candidate_path.write_text(json.dumps(candidate))
        subprocess.run([
            "python3", str(root / "compare_referential_slot_microcode.py"),
            "--control", str(control_path), "--candidate", str(candidate_path), "--out", str(out),
        ], check=True, capture_output=True, text=True)
        result = json.loads(out.read_text())
        assert result["binding_attributed"]
        assert result["decision"] == "advance_referential_slot_compiler_r4"
        failed = copy.deepcopy(candidate)
        failed["adapter_metadata"]["initial_adapter_sha256"] = "wrong"
        candidate_path.write_text(json.dumps(failed))
        out.unlink()
        subprocess.run([
            "python3", str(root / "compare_referential_slot_microcode.py"),
            "--control", str(control_path), "--candidate", str(candidate_path), "--out", str(out),
        ], check=True, capture_output=True, text=True)
        assert not json.loads(out.read_text())["binding_attributed"]
    print("referential slot comparator tests passed")


if __name__ == "__main__":
    main()
