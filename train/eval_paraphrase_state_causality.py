#!/usr/bin/env python3
"""Causally probe prompt-boundary state exchange for semantic-basis carriers.

This is an isolated intervention evaluation, not a trainer.  At one selected
block and the final prompt position, it exchanges the residual written by an
independently worded equivalent source prompt or by a different-state prompt.
Each decode step is a full replay so the intervention remains present at the
original answer boundary; it intentionally does not reuse a KV cache whose
pre-patch keys would make the causal interpretation ambiguous.

The only claim this can support is that a selected residual can causally affect
the immediate exact-ledger report on this synthetic task.  It cannot establish
general reasoning, a reusable workspace, or context scaling.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from semantic_basis_transport_controller import exact_ledger, phase_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_prompt(protocol_prompt):
    return "Question: {}\nAnswer:".format(protocol_prompt)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


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
    for episode_id, rows in sorted(episodes.items()):
        indexed = phase_rows(rows)
        if indexed["compile"]["episode_id"] != episode_id:
            raise ValueError("inconsistent episode id")
        if indexed["compile"]["response"] != indexed["reflect"]["response"]:
            raise ValueError("compile and reflect targets differ for {}".format(episode_id))
        checked.append(rows)
    if not checked:
        raise ValueError("no semantic-basis episodes in requested split")
    return checked


def select_pairs(episodes, pairs, seed):
    """Choose disjoint target/donor pairs with distinct exact source states."""
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    selected, used = [], set()
    for left_index, target in enumerate(shuffled):
        target_id = phase_rows(target)["compile"]["episode_id"]
        target_state = phase_rows(target)["compile"]["response"]
        if target_id in used:
            continue
        for donor in shuffled[left_index + 1:]:
            donor_id = phase_rows(donor)["compile"]["episode_id"]
            donor_state = phase_rows(donor)["compile"]["response"]
            if donor_id in used or donor_state == target_state:
                continue
            selected.append((target, donor))
            used.update((target_id, donor_id))
            break
        if len(selected) == pairs:
            break
    if len(selected) != pairs:
        raise ValueError("could not select {} disjoint state-distinct pairs".format(pairs))
    return selected


def encoded_prompt(tokenizer, protocol_prompt, cap):
    ids = tokenizer.encode(model_prompt(protocol_prompt)).ids
    if not ids or len(ids) >= cap:
        raise ValueError("prompt does not fit model context")
    return ids


@torch.no_grad()
def capture_boundary(model, ids, layer, device):
    captured = {}

    def hook(_module, _inputs, output):
        hidden, cache = output
        captured["state"] = hidden[:, len(ids) - 1, :].detach().clone()
        return hidden, cache

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        model(torch.tensor([ids], device=device, dtype=torch.long))
    finally:
        handle.remove()
    if "state" not in captured:
        raise RuntimeError("state capture hook did not run")
    return captured["state"]


@torch.no_grad()
def patched_logits(model, ids, layer, anchor_position, device, source=None, alpha=1.0):
    """Return next-token logits after an optional final-prompt residual replacement."""
    if source is None:
        logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
        return logits[:, -1, :]
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0 <= anchor_position < len(ids):
        raise ValueError("anchor position is outside current prefix")

    def hook(_module, _inputs, output):
        hidden, cache = output
        replacement = source.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != hidden[:, anchor_position, :].shape:
            raise ValueError("source residual shape does not match target state")
        patched = hidden.clone()
        current = hidden[:, anchor_position, :]
        patched[:, anchor_position, :] = current + alpha * (replacement - current)
        return patched, cache

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
    finally:
        handle.remove()
    return logits[:, -1, :]


@torch.no_grad()
def greedy_full_replay(model, tokenizer, prompt_ids, layer, source, alpha, device, max_new):
    """Decode with a fixed intervention at the original prompt boundary every step."""
    ids = list(prompt_ids)
    generated = []
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    for _ in range(max_new):
        logits = patched_logits(model, ids, layer, len(prompt_ids) - 1, device, source, alpha)
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)
        if len(ids) >= model.cfg.seq_len:
            break
    return tokenizer.decode(generated, skip_special_tokens=False)


@torch.no_grad()
def completion_logprob(model, prompt_ids, completion_ids, layer, source, alpha, device):
    """Teacher-forced log probability of a completion under the same intervention."""
    if not completion_ids:
        raise ValueError("completion is empty")
    ids = prompt_ids + completion_ids
    if len(ids) > model.cfg.seq_len:
        raise ValueError("prompt plus completion exceeds model context")
    anchor = len(prompt_ids) - 1
    if source is None:
        logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
        logits = logits[:, anchor:anchor + len(completion_ids), :]
    else:
        collected = []

        def hook(_module, _inputs, output):
            hidden, cache = output
            replacement = source.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            current = hidden[:, anchor, :]
            patched[:, anchor, :] = current + alpha * (replacement - current)
            return patched, cache

        handle = model.blocks[layer].register_forward_hook(hook)
        try:
            logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
        finally:
            handle.remove()
        logits = logits[:, anchor:anchor + len(completion_ids), :]
    target = torch.tensor(completion_ids, device=device, dtype=torch.long)
    return float(F.log_softmax(logits.float(), dim=-1)[0, torch.arange(len(completion_ids), device=device), target].sum())


def cosine(left, right):
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).item())


def evaluate_direction(model, tokenizer, target_rows, donor_rows, layer, alpha, device, max_new):
    target = phase_rows(target_rows)
    donor = phase_rows(donor_rows)
    target_prompt = encoded_prompt(tokenizer, target["compile"]["question"], model.cfg.seq_len)
    same_prompt = encoded_prompt(tokenizer, target["reflect"]["question"], model.cfg.seq_len)
    mismatch_prompt = encoded_prompt(tokenizer, donor["reflect"]["question"], model.cfg.seq_len)
    target_state = capture_boundary(model, target_prompt, layer, device)
    same_state = capture_boundary(model, same_prompt, layer, device)
    mismatch_state = capture_boundary(model, mismatch_prompt, layer, device)
    target_completion = tokenizer.encode(target["compile"]["response"]).ids
    donor_completion = tokenizer.encode(donor["compile"]["response"]).ids

    baseline = greedy_full_replay(model, tokenizer, target_prompt, layer, None, alpha, device, max_new)
    identity = greedy_full_replay(model, tokenizer, target_prompt, layer, target_state, alpha, device, max_new)
    same = greedy_full_replay(model, tokenizer, target_prompt, layer, same_state, alpha, device, max_new)
    mismatch = greedy_full_replay(model, tokenizer, target_prompt, layer, mismatch_state, alpha, device, max_new)
    baseline_lp = completion_logprob(model, target_prompt, target_completion, layer, None, alpha, device)
    identity_lp = completion_logprob(model, target_prompt, target_completion, layer, target_state, alpha, device)
    same_lp = completion_logprob(model, target_prompt, target_completion, layer, same_state, alpha, device)
    mismatch_target_lp = completion_logprob(model, target_prompt, target_completion, layer, mismatch_state, alpha, device)
    mismatch_donor_lp = completion_logprob(model, target_prompt, donor_completion, layer, mismatch_state, alpha, device)
    return {
        "target_episode_id": target["compile"]["episode_id"],
        "donor_episode_id": donor["compile"]["episode_id"],
        "target_ledger": target["compile"]["response"],
        "donor_ledger": donor["compile"]["response"],
        "target_same_cosine": cosine(target_state, same_state),
        "target_mismatch_cosine": cosine(target_state, mismatch_state),
        "baseline": {"response": baseline, "exact": exact_ledger(baseline), "target_logprob": baseline_lp},
        "identity": {"response": identity, "exact": exact_ledger(identity), "target_logprob": identity_lp},
        "same_state": {"response": same, "exact": exact_ledger(same), "target_logprob": same_lp},
        "mismatch_state": {
            "response": mismatch,
            "exact": exact_ledger(mismatch),
            "target_logprob": mismatch_target_lp,
            "donor_logprob": mismatch_donor_lp,
            "donor_minus_target_logprob": mismatch_donor_lp - mismatch_target_lp,
        },
    }


def summarize(results):
    return {
        "directions": len(results),
        "baseline_exact_target": sum(item["baseline"]["exact"] == item["target_ledger"] for item in results),
        "identity_exact_target": sum(item["identity"]["exact"] == item["target_ledger"] for item in results),
        "same_state_exact_target": sum(item["same_state"]["exact"] == item["target_ledger"] for item in results),
        "mismatch_state_exact_donor": sum(item["mismatch_state"]["exact"] == item["donor_ledger"] for item in results),
        "mismatch_state_donor_margin_positive": sum(item["mismatch_state"]["donor_minus_target_logprob"] > 0.0 for item in results),
        "mean_target_same_cosine": sum(item["target_same_cosine"] for item in results) / len(results),
        "mean_target_mismatch_cosine": sum(item["target_mismatch_cosine"] for item in results) / len(results),
        "mean_same_minus_baseline_logprob": sum(item["same_state"]["target_logprob"] - item["baseline"]["target_logprob"] for item in results) / len(results),
        "mean_mismatch_donor_minus_target_logprob": sum(item["mismatch_state"]["donor_minus_target_logprob"] for item in results) / len(results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="factor_language")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    if args.examples < 0 or args.max_new <= 0:
        raise SystemExit("examples must be nonnegative and max-new must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    if not 0 <= args.layer < len(model.blocks):
        raise SystemExit("layer is outside model depth")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pairs = select_pairs(load_episodes(args.data, args.split), args.pairs, args.seed)
    results = []
    for index, (left, right) in enumerate(pairs, 1):
        for target, donor in ((left, right), (right, left)):
            result = evaluate_direction(model, tokenizer, target, donor, args.layer, args.alpha, device, args.max_new)
            results.append(result)
        print("[psa-causal] pair={}/{} directions={}".format(index, len(pairs), len(results)), flush=True)
    report = {
        "audit": "paraphrase_state_causality_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "split": args.split,
        "pairs": args.pairs,
        "layer": args.layer,
        "alpha": args.alpha,
        "seed": args.seed,
        "summary": summarize(results),
        "results": results,
        "examples": results[:args.examples],
        "claim_boundary": (
            "This measures an activation-exchange effect on one synthetic exact-ledger report. "
            "Same-state preservation plus mismatch-state redirection would be causal evidence for a prompt-boundary "
            "representation, not evidence of flexible downstream reuse, general reasoning, or a global workspace."
        ),
        "control_boundary": (
            "Identity replacement checks hook fidelity. Same-state sources are independent compile/reflect prompts with "
            "the same frozen ledger. Mismatch sources are model states from a different frozen episode; no controller "
            "parses, rewrites, or selects the ledger. Every generated token is full-replayed with the intervention at the "
            "original prompt boundary, so no pre-patch KV cache is reused."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[psa-causal] summary=" + json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
