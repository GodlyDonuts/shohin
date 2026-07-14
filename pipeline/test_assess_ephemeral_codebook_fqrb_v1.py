#!/usr/bin/env python3
"""CPU-only decision contracts for the ECLI evidence gate."""
from __future__ import annotations

from assess_ephemeral_codebook_fqrb_v1 import QUERY_KINDS, assess


audit = {
    "audit": "ephemeral_codebook_fqrb_v1", "mechanism": "ephemeral_codebook_fqrb_v1",
    "train_rows": 60000, "heldout_rows": 2500, "heldout_sha256": "data",
}
consumer = {kind: {
    "normal_correct": 350, "paraphrase_correct": 350, "counterfactual_correct": 350, "codebook_swap_correct": 350,
} for kind in QUERY_KINDS}
report = {
    "groups": 500, "rows": 2500, "data_sha256": "data", "consumer_summary": consumer,
    "checkpoint_metadata": {
        "source_present_at_suffix": False, "extra_trainable_parameters": 0, "composition": "donor + edited - base",
    },
    "basis_summary": {
        "joint_strict": 300, "joint_codebook_swap": 300,
        "any_zero_recreates_normal": 25, "any_shuffle_recreates_normal": 25,
        "any_wrong_query_recreates_normal": 25, "any_codebook_swap_recreates_normal": 25,
    },
}
assert assess(report, audit)["decision"] == "bounded_ecli_late_binding_candidate"
report["consumer_summary"]["sign"]["codebook_swap_correct"] = 349
failed = assess(report, audit)
assert failed["decision"] == "reject_ecli_late_binding"
assert "sign_codebook_swap_correct_below_350" in failed["reasons"]
print("ECLI assessment checks: passed")
