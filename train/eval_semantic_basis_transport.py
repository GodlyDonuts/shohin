#!/usr/bin/env python3
"""Evaluate exact model-authored semantic-basis transport with causal controls.

Normal episodes require two independent source descriptions to emit the same
exact ledger, then route the compile emission through an update and route the
update emission through both difference and sum consumers.  Cross-episode
interchange forwards the donor's exact raw update emission into another
episode's consumers.  Zero and mismatch carriers are evaluator-created
counterfactuals and are reported separately; they are never presented as
model-authored state.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig
from semantic_basis_transport_controller import consumers_correct, phase_rows, rollout_episode, run_consumers


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def model_prompt(protocol_prompt, mode):
    """Use the same inference surface that completion-masked SFT sees."""
    if mode == "qa":
        return "Question: {}\nAnswer:".format(protocol_prompt)
    if mode == "direct":
        return protocol_prompt
    raise ValueError("unknown prompt mode: {}".format(mode))


def load_episodes(path, split):
    episodes = collections.defaultdict(list)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "semantic_basis_transport_v2" or row.get("split") != split:
                raise ValueError("invalid {} semantic-basis row at line {}".format(split, line_number))
            episodes[row["episode_id"]].append(row)
    checked = []
    for episode_id, rows in episodes.items():
        indexed = phase_rows(rows)
        if indexed["compile"]["episode_id"] != episode_id:
            raise ValueError("inconsistent episode id")
        checked.append(rows)
    if not checked:
        raise ValueError("no held-out semantic-basis episodes")
    return checked


def updated_values(rows):
    update = phase_rows(rows)["update"]
    return int(update["primary_value"]) + int(update["delta"]), int(update["secondary_value"])


def select_pairs(episodes, pairs, seed):
    """Select deterministic distinct-carrier pairs for observable interventions."""
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    rng = random.Random(seed)
    candidates = list(episodes)
    rng.shuffle(candidates)
    selected, used = [], set()
    for left_index, left in enumerate(candidates):
        left_id = phase_rows(left)["compile"]["episode_id"]
        if left_id in used:
            continue
        left_p, left_q = updated_values(left)
        for right in candidates[left_index + 1:]:
            right_id = phase_rows(right)["compile"]["episode_id"]
            if right_id in used:
                continue
            right_p, right_q = updated_values(right)
            if left_p == right_p or left_q == right_q:
                continue
            selected.append((left, right))
            used.update((left_id, right_id))
            break
        if len(selected) == pairs:
            break
    if len(selected) != pairs:
        raise ValueError("could not select {} distinct semantic-basis pairs".format(pairs))
    return selected


def mismatch_spec(receiver_rows, donor_rows):
    """Build an explicitly evaluator-created P/Q recombination control."""
    donor_p, _ = updated_values(donor_rows)
    _, receiver_q = updated_values(receiver_rows)
    return {
        "kind": "evaluator_created_mismatch",
        "carrier": "ledger:P={};Q={}".format(donor_p, receiver_q),
        "difference": "answer={}".format(donor_p - receiver_q),
        "sum": "answer={}".format(donor_p + receiver_q),
    }


def rejects_original(consumers, receiver_rows):
    phases = phase_rows(receiver_rows)
    return bool(
        consumers["difference"]["exact_response"] != phases["difference"]["response"]
        and consumers["sum"]["exact_response"] != phases["sum"]["response"]
    )


def evaluate_pair(left_rows, right_rows, ask):
    """Score one model-authored interchange pair plus guarded controls."""
    left, right = rollout_episode(left_rows, ask), rollout_episode(right_rows, ask)
    result = {
        "left_episode_id": left["episode_id"],
        "right_episode_id": right["episode_id"],
        "left_normal": left,
        "right_normal": right,
        "normal_both_strict": bool(left["strict_success"] and right["strict_success"]),
        "left_receives_right": None,
        "right_receives_left": None,
        "left_zero": None,
        "right_zero": None,
        "left_mismatch": None,
        "right_mismatch": None,
        "model_authored_interchange_success": False,
        "zero_recreates_original": False,
        "mismatch_success_and_rejects_original": False,
        "strict_causal_pass": False,
    }
    if not result["normal_both_strict"]:
        return result

    left_phases, right_phases = phase_rows(left_rows), phase_rows(right_rows)
    left_carrier = left["update"]["exact_response"]
    right_carrier = right["update"]["exact_response"]
    result["left_receives_right"] = run_consumers(
        left_rows, ask, right_carrier,
        right_phases["difference"]["response"], right_phases["sum"]["response"],
    )
    result["right_receives_left"] = run_consumers(
        right_rows, ask, left_carrier,
        left_phases["difference"]["response"], left_phases["sum"]["response"],
    )
    result["model_authored_interchange_success"] = bool(
        consumers_correct(result["left_receives_right"])
        and consumers_correct(result["right_receives_left"])
    )

    zero = "ledger:P=0;Q=0"
    result["left_zero"] = run_consumers(
        left_rows, ask, zero, left_phases["difference"]["response"], left_phases["sum"]["response"],
    )
    result["right_zero"] = run_consumers(
        right_rows, ask, zero, right_phases["difference"]["response"], right_phases["sum"]["response"],
    )
    result["zero_recreates_original"] = bool(
        consumers_correct(result["left_zero"]) or consumers_correct(result["right_zero"])
    )

    left_spec, right_spec = mismatch_spec(left_rows, right_rows), mismatch_spec(right_rows, left_rows)
    result["left_mismatch"] = run_consumers(
        left_rows, ask, left_spec["carrier"], left_spec["difference"], left_spec["sum"],
    )
    result["right_mismatch"] = run_consumers(
        right_rows, ask, right_spec["carrier"], right_spec["difference"], right_spec["sum"],
    )
    mismatch_success = bool(
        consumers_correct(result["left_mismatch"])
        and consumers_correct(result["right_mismatch"])
        and rejects_original(result["left_mismatch"], left_rows)
        and rejects_original(result["right_mismatch"], right_rows)
    )
    result["mismatch_success_and_rejects_original"] = mismatch_success
    result["strict_causal_pass"] = bool(
        result["model_authored_interchange_success"]
        and not result["zero_recreates_original"]
        and mismatch_success
    )
    return result


def summarize(results):
    episodes = [item[key] for item in results for key in ("left_normal", "right_normal")]
    return {
        "pairs": len(results),
        "normal_episodes": len(episodes),
        "compile_correct": sum(bool(item["compile"]["correct"]) for item in episodes),
        "reflect_correct": sum(bool(item["reflect"]["correct"]) for item in episodes),
        "reportability_equal": sum(bool(item["reportability_equal"]) for item in episodes),
        "update_correct": sum(bool(item["update"] and item["update"]["correct"]) for item in episodes),
        "normal_strict_success": sum(bool(item["strict_success"]) for item in episodes),
        "normal_both_strict": sum(bool(item["normal_both_strict"]) for item in results),
        "model_authored_interchange_success": sum(bool(item["model_authored_interchange_success"]) for item in results),
        "zero_recreates_original": sum(bool(item["zero_recreates_original"]) for item in results),
        "mismatch_success_and_rejects_original": sum(bool(item["mismatch_success_and_rejects_original"]) for item in results),
        "strict_causal_pass": sum(bool(item["strict_causal_pass"]) for item in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--split", choices=("train", "heldout", "factor_language", "factor_values", "factor_delta"),
                        default="heldout", help="data split to score; train and factor splits are diagnostic-only")
    parser.add_argument("--prompt-mode", choices=("qa", "direct"), default="qa",
                        help="qa matches train/sft.py's standard Question/Answer surface")
    args = parser.parse_args()
    if args.examples < 0:
        raise SystemExit("--examples must be nonnegative")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pairs = select_pairs(load_episodes(args.data, args.split), args.pairs, args.seed)

    def ask(prompt, max_new):
        return generate(
            model, tokenizer, model_prompt(prompt, args.prompt_mode), device,
            max_new=max_new, temp=0.0, skip_special_tokens=False,
        )

    results = []
    for index, (left, right) in enumerate(pairs, 1):
        result = evaluate_pair(left, right, ask)
        results.append(result)
        print(
            "[semantic-basis] {}/{} normal_both={} interchange={} causal={}".format(
                index, len(pairs), result["normal_both_strict"], result["model_authored_interchange_success"],
                result["strict_causal_pass"],
            ), flush=True,
        )
    report = {
        "audit": "semantic_basis_transport_exact_carrier_v2",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "split": args.split,
        "pairs": args.pairs,
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "inference_prompt_template": "Question: {protocol_prompt}\\nAnswer:" if args.prompt_mode == "qa" else "{protocol_prompt}",
        "summary": summarize(results),
        "results": results,
        "examples": results[:args.examples],
        "claim_boundary": (
            "A strict pass establishes only exact model-authored two-value carrier transport over this "
            "synthetic source-deleted task. Train and factor-split results are diagnostic-only. No result establishes "
            "general language reasoning, latent thought, autonomous context compression, or a global workspace."
        ),
        "control_boundary": (
            "Cross-episode interchanges use an exact raw model emission. Zero and mismatch carriers are "
            "evaluator-created interventions and are reported separately from model-authored transport."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[semantic-basis] summary=" + json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
