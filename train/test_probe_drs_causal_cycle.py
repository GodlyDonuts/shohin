#!/usr/bin/env python3
"""CPU contracts for the DRS causal-cycle probe."""
import json
import argparse
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from digitwise_protocol import apply_microstep, initial_state  # noqa: E402
from eval_suite import generate  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from probe_drs_causal_cycle import (  # noqa: E402
    EXPECTED_REGIMES,
    active_digit,
    build_decision,
    cached_prefix_logits,
    canonical_preflight,
    carry_counterfactual,
    collect_cached_residuals,
    greedy_state,
    hybrid_next,
    irrelevant_history_counterfactual,
    publish_json_exclusive,
    scientific_source_hashes,
    select_cases,
    summarize,
)


state = initial_state("add", 95, 18, 4)
state = apply_microstep(state)
flipped = carry_counterfactual(state)
assert flipped["c"] == 1 - state["c"]
assert flipped["a"] == state["a"] and flipped["b"] == state["b"]
assert apply_microstep(flipped)["r"][state["p"]] != apply_microstep(state)["r"][state["p"]]

irrelevant = irrelevant_history_counterfactual(state)
assert irrelevant["r"][0] != state["r"][0]
base_next = apply_microstep(state)
irrelevant_next = apply_microstep(irrelevant)
assert active_digit(irrelevant_next) == active_digit(base_next)
assert irrelevant_next["c"] == base_next["c"]

counterfactual_next = apply_microstep(flipped)
carry_hybrid = hybrid_next(base_next, counterfactual_next, {"carry"})
digit_hybrid = hybrid_next(base_next, counterfactual_next, {"digit"})
assert carry_hybrid["c"] == counterfactual_next["c"]
assert active_digit(carry_hybrid) == active_digit(base_next)
assert digit_hybrid["c"] == base_next["c"]
assert active_digit(digit_hybrid) == active_digit(counterfactual_next)

cases, inventory = select_cases(
    ROOT / "artifacts" / "evals" / "digitwise_recurrent_v2_heldout.jsonl", 2, 10
)
assert len(cases) == 50
assert set(inventory) == set(EXPECTED_REGIMES)
assert all(inventory[regime] >= 10 for regime in EXPECTED_REGIMES)
assert all(case["boundary"] for case in cases)
assert len({case["episode"]["id"] for case in cases}) == 50
assert all(case["same_target_donor"]["episode"]["id"] != case["episode"]["id"] for case in cases)

assert scientific_source_hashes() == {
    "probe_drs_causal_cycle.py": "d81ff28db221e706d0b283fc933d718331a05a4a36a505d8b68898340679d82d",
    "digitwise_protocol.py": "708489d61c212c402e1533a1483e77bf3fd2d1a057ce924321bb19e4888461f6",
    "eval_suite.py": "d6f70b8828c967d7f59fae842f3320c6378ae42d5d8fa7b16e0e82ff5620e5e6",
    "model.py": "45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4",
    "probe_digitwise_workspace.py": "fb545450a93bbc04aac1549efd0a70b863f50e458fd95993cf9935bbe4a53ace",
}
try:
    canonical_preflight(
        argparse.Namespace(
            transition_index=2,
            per_regime=10,
            layer=29,
            max_new=96,
            out="/lustre/fs1/home/sa305415/shohin/artifacts/evals/drs_causal_cycle_post_drs_r3.json",
            ckpt="checkpoint",
            tokenizer="tokenizer",
            episodes="episodes",
        ),
        "mps",
    )
except SystemExit as error:
    assert "requires CUDA" in str(error)
else:
    raise AssertionError("canonical mode accepted a non-CUDA device")


class TinyEncoding:
    def __init__(self, ids):
        self.ids = ids


class TinyTokenizer:
    def encode(self, _text):
        return TinyEncoding([1, 2])

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(value) for value in ids)

    def token_to_id(self, _token):
        return None


torch.manual_seed(7)
tiny = GPT(GPTConfig(
    vocab_size=32, n_layer=2, n_head=2, n_kv_head=1, d_model=16,
    d_ff=32, seq_len=16, qk_norm=True,
)).eval()
tiny_tokenizer = TinyTokenizer()
stable = generate(tiny, tiny_tokenizer, "prompt", "cpu", max_new=3, temp=0.0, skip_special_tokens=False)
plain = greedy_state(tiny, tiny_tokenizer, "prompt", "cpu", 3)
assert plain["response"] == stable
first_id = plain["token_ids"][0]
prefix_ids = [1, 2, first_id]
residuals, cached_logits = collect_cached_residuals(tiny, [1, 2], prefix_ids, "cpu")
site = {
    "field": "digit",
    "layer": 0,
    "prefix_ids": tuple(prefix_ids),
    "source_residual": residuals[0],
    "target_id": 0,
    "target_digit": "0",
    "mode": "residual",
}
identity_logits = cached_prefix_logits(tiny, [1, 2], prefix_ids, "cpu", site)
assert torch.equal(identity_logits, cached_logits)
identity = greedy_state(tiny, tiny_tokenizer, "prompt", "cpu", 3, [site])
assert identity["token_ids"] == plain["token_ids"]
assert identity["fired"] == [{"field": "digit", "mode": "residual", "prefix_length": 3}]

deep_prefix_ids = [1, 2, *plain["token_ids"][:2]]
deep_residuals, deep_logits = collect_cached_residuals(tiny, [1, 2], deep_prefix_ids, "cpu")
deep_site = {
    **site,
    "field": "carry",
    "prefix_ids": tuple(deep_prefix_ids),
    "source_residual": deep_residuals[0],
}
deep_identity_logits = cached_prefix_logits(tiny, [1, 2], deep_prefix_ids, "cpu", deep_site)
assert torch.equal(deep_identity_logits, deep_logits)
deep_identity = greedy_state(tiny, tiny_tokenizer, "prompt", "cpu", 3, [deep_site])
assert deep_identity["token_ids"] == plain["token_ids"]
assert deep_identity["fired"] == [
    {"field": "carry", "mode": "residual", "prefix_length": len(deep_prefix_ids)}
]


def arm(exact=True, token=1, fields=("carry", "digit")):
    return {
        "exact": exact,
        "token_ids": [token],
        "fired": [{"field": field} for field in fields],
        "next_prompt_token_transport_exact": exact,
    }


def record(regime, baseline=True, same=True, both=True, identity_token=1):
    return {
        "regime": regime,
        "boundary_condition": True,
        "teacher_forced_identity_exact": True,
        "fields": {
            field: {
                "irrelevant_history": {"argmax_invariant": True},
                "irrelevant_transplant_into_base": {"argmax_invariant": True},
            }
            for field in ("carry", "digit")
        },
        "arms": {
            "baseline": arm(baseline, 1, ()),
            "identity": arm(baseline, identity_token),
            "same_target": arm(same),
            "carry_only": arm(True, 1, ("carry",)),
            "digit_only": arm(True, 1, ("digit",)),
            "both": arm(both),
            "token_ceiling": arm(True),
            "irrelevant_sham": arm(baseline),
        },
        "consumer": {
            "paired_active_digit_switch_correct": True,
            "base": {
                "greedy": {"exact": True},
                "fields": {field: {"correct": True} for field in ("carry", "digit")},
            },
            "counterfactual": {
                "greedy": {"exact": True},
                "fields": {field: {"correct": True} for field in ("carry", "digit")},
            },
        },
        "integrated_cycle_exact": both,
    }


rows = [record(regime) for regime in EXPECTED_REGIMES for _ in range(10)]
summary = summarize(rows)
assert summary["valid"] is True
assert summary["records"] == 50
assert summary["paired_second_call_active_digit_switch_correct_rate"] == 1.0
assert set(summary["by_regime"]) == set(EXPECTED_REGIMES)
assert all(summary["by_regime"][regime]["records"] == 10 for regime in EXPECTED_REGIMES)
assert build_decision(summary, True)["consumer_pass"] is True
invalid_decision = build_decision(summary, False)
assert invalid_decision["mechanically_valid"] is False
assert all(value is None for key, value in invalid_decision.items() if key != "mechanically_valid")
regime_failure = [record(regime, both=(regime != EXPECTED_REGIMES[0]))
                  for regime in EXPECTED_REGIMES for _ in range(10)]
failed_summary = summarize(regime_failure)
assert failed_summary["counterfactual_first_state_exact_rate"] == 0.8
assert build_decision(failed_summary, True)["write_serialization_pass"] is False
regime_failure[0]["fields"]["carry"]["irrelevant_transplant_into_base"]["argmax_invariant"] = False
regime_failure[0]["arms"]["irrelevant_sham"]["token_ids"] = [9]
causal_sham_summary = summarize(regime_failure)
assert causal_sham_summary["irrelevant_transplant_argmax_invariant"] == 49
assert causal_sham_summary["irrelevant_sham_token_equal_to_baseline"] == 49
rows[0] = record(EXPECTED_REGIMES[0], identity_token=2)
assert summarize(rows)["valid"] is False

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "result.json"
    publish_json_exclusive(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    try:
        publish_json_exclusive(path, {"ok": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("exclusive publisher overwrote an existing result")

print("DRS causal-cycle CPU contracts: passed")
