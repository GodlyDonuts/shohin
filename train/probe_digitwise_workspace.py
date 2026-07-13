#!/usr/bin/env python3
"""Probe actionable recurrent state with late-layer residual patching.

This is a small-model analogue of the causal part of a workspace study, not a
Jacobian lens implementation and not evidence of a global workspace by itself.
For matched held-out DRS transitions, it captures the residual at the final
teacher-forced completion-prefix token. It then replaces that residual with
the corresponding residual from a state whose correct next carry or result
digit differs. A useful local representation should move the target log-odds
toward the source state's answer after the patch.

The probe neither trains a model nor supplies a solver state during generation.
All arithmetic is used only to select and label already solver-verified held-out
episodes. Results are limited to the patched token position and cannot prove
broad reasoning, language compilation, or long-context capability.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from digitwise_protocol import canonical_state, microstep_prompt, parse_state
from model import GPT, GPTConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_layers(value, n_layer):
    layers = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 or layer >= int(n_layer):
            raise ValueError("layer {} is outside [0, {})".format(layer, n_layer))
        if layer not in layers:
            layers.append(layer)
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def field_prefix(prompt, next_state, field):
    """Return a teacher-forced prefix ending immediately before one digit field."""
    response = canonical_state(next_state)
    if field == "carry":
        offset = response.index(";c=") + len(";c=")
    elif field == "digit":
        position = int(next_state["p"]) - 1
        offset = response.index(";r=") + len(";r=") + position
    else:
        raise ValueError("unknown local field: {}".format(field))
    target = response[offset]
    if target not in "0123456789":
        raise ValueError("field target is not a decimal digit")
    return prompt + response[:offset], target


def token_for_digit(tokenizer, prefix, digit):
    """Require the target digit to be a prefix-stable single-token continuation."""
    prefix_ids = tokenizer.encode(prefix).ids
    full_ids = tokenizer.encode(prefix + digit).ids
    if full_ids[:len(prefix_ids)] != prefix_ids or len(full_ids) != len(prefix_ids) + 1:
        raise ValueError("target digit is not a prefix-stable single token")
    return prefix_ids, int(full_ids[-1])


def transition_examples(path, transition_index):
    """Extract one solver-verified state transition from each held-out episode."""
    examples = []
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        episode = json.loads(raw)
        state = parse_state(episode["initial_state"])
        if state is None:
            raise ValueError("invalid initial state in {}".format(episode["id"]))
        for index, expected_line in enumerate(episode["expected_states"]):
            next_state = parse_state(expected_line)
            if next_state is None:
                raise ValueError("invalid expected state in {}".format(episode["id"]))
            if index == transition_index:
                examples.append({
                    "id": episode["id"],
                    "regime": episode["split"],
                    "state": state,
                    "next_state": next_state,
                    "prompt_style": episode["prompt_style"],
                })
                break
            state = next_state
        else:
            raise ValueError("transition {} missing from {}".format(transition_index, episode["id"]))
    return examples


def target_digit(example, field):
    if field == "carry":
        return str(example["next_state"]["c"])
    if field == "digit":
        return example["next_state"]["r"][int(example["state"]["p"])]
    raise ValueError("unknown local field: {}".format(field))


def select_pairs(examples, field, pairs_per_regime):
    """Pair different next digits while retaining operation, width, position, and input carry."""
    if pairs_per_regime <= 0:
        raise ValueError("pairs_per_regime must be positive")
    by_regime = collections.defaultdict(list)
    for example in examples:
        by_regime[example["regime"]].append(example)
    selected = []
    for regime in sorted(by_regime):
        candidates = sorted(by_regime[regime], key=lambda item: item["id"])
        used, count = set(), 0
        for left_index, left in enumerate(candidates):
            if left["id"] in used:
                continue
            left_key = (
                left["state"]["op"], left["state"]["w"], left["state"]["p"], left["state"]["c"],
            )
            for right in candidates[left_index + 1:]:
                if right["id"] in used:
                    continue
                right_key = (
                    right["state"]["op"], right["state"]["w"], right["state"]["p"], right["state"]["c"],
                )
                if left_key != right_key or target_digit(left, field) == target_digit(right, field):
                    continue
                selected.append({"regime": regime, "field": field, "a": left, "b": right})
                used.update((left["id"], right["id"]))
                count += 1
                break
            if count >= pairs_per_regime:
                break
        if count != pairs_per_regime:
            raise ValueError("could not select {} {} pairs for {}".format(pairs_per_regime, field, regime))
    return selected


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    if int(model.cfg.n_loop) != 1:
        raise ValueError("workspace probe requires n_loop=1 for unambiguous block indices")
    return checkpoint, model


@torch.no_grad()
def collect_residuals(model, input_ids):
    """Capture each block's last-position residual plus the ordinary final logits."""
    captured, handles = {}, []
    for index, block in enumerate(model.blocks):
        def hook(_module, _inputs, output, index=index):
            hidden, _ = output
            captured[index] = hidden[:, -1, :].detach().clone()
        handles.append(block.register_forward_hook(hook))
    try:
        logits, _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(model.blocks))):
        raise RuntimeError("did not capture every transformer block")
    return captured, logits[:, -1, :].detach()


@torch.no_grad()
def patched_logits(model, input_ids, layer, source_residual, alpha):
    """Replace only the last-position residual after one block and continue normally."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    def hook(_module, _inputs, output):
        hidden, past = output
        replacement = source_residual.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != hidden[:, -1, :].shape:
            raise ValueError("source residual has the wrong shape")
        patched = hidden.clone()
        patched[:, -1, :] = hidden[:, -1, :] + alpha * (replacement - hidden[:, -1, :])
        return patched, past

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        logits, _ = model(input_ids)
    finally:
        handle.remove()
    return logits[:, -1, :].detach()


def rank(logits, token_id):
    return int((logits > logits[:, token_id:token_id + 1]).sum().item() + 1)


def stats(logits, own_id, other_id):
    own, other = float(logits[0, own_id]), float(logits[0, other_id])
    return {
        "own_logit": own,
        "other_logit": other,
        "toward_other_logodds": other - own,
        "own_rank": rank(logits, own_id),
        "other_rank": rank(logits, other_id),
    }


@torch.no_grad()
def run_pair(model, tokenizer, pair, layers, alpha, device):
    field = pair["field"]
    rendered = []
    for side in ("a", "b"):
        example = pair[side]
        prompt = microstep_prompt(example["state"], style=example["prompt_style"])
        prefix, target = field_prefix(prompt, example["next_state"], field)
        ids, target_id = token_for_digit(tokenizer, prefix, target)
        rendered.append({
            "example": example,
            "prompt": prompt,
            "prefix": prefix,
            "target": target,
            "target_id": target_id,
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        })
    a, b = rendered
    if a["target_id"] == b["target_id"]:
        raise ValueError("selected pair does not differ in its target token")
    residual_a, logits_a = collect_residuals(model, a["input_ids"])
    residual_b, logits_b = collect_residuals(model, b["input_ids"])
    layer_rows = []
    for layer in layers:
        lens_a = model.head(model.norm(residual_a[layer].unsqueeze(1)))[:, 0, :]
        lens_b = model.head(model.norm(residual_b[layer].unsqueeze(1)))[:, 0, :]
        a_to_b = patched_logits(model, a["input_ids"], layer, residual_b[layer], alpha)
        b_to_a = patched_logits(model, b["input_ids"], layer, residual_a[layer], alpha)
        a_base = stats(logits_a, a["target_id"], b["target_id"])
        b_base = stats(logits_b, b["target_id"], a["target_id"])
        a_patch = stats(a_to_b, a["target_id"], b["target_id"])
        b_patch = stats(b_to_a, b["target_id"], a["target_id"])
        layer_rows.append({
            "layer": layer,
            "a_logit_lens": stats(lens_a, a["target_id"], b["target_id"]),
            "b_logit_lens": stats(lens_b, b["target_id"], a["target_id"]),
            "a_to_b": {
                "baseline": a_base,
                "patched": a_patch,
                "toward_source_logodds_delta": a_patch["toward_other_logodds"] - a_base["toward_other_logodds"],
            },
            "b_to_a": {
                "baseline": b_base,
                "patched": b_patch,
                "toward_source_logodds_delta": b_patch["toward_other_logodds"] - b_base["toward_other_logodds"],
            },
        })
    return {
        "regime": pair["regime"],
        "field": field,
        "a": {
            "id": a["example"]["id"],
            "state": canonical_state(a["example"]["state"]),
            "next_state": canonical_state(a["example"]["next_state"]),
            "target": a["target"],
            "prefix_sha256": hashlib.sha256(a["prefix"].encode()).hexdigest(),
        },
        "b": {
            "id": b["example"]["id"],
            "state": canonical_state(b["example"]["state"]),
            "next_state": canonical_state(b["example"]["next_state"]),
            "target": b["target"],
            "prefix_sha256": hashlib.sha256(b["prefix"].encode()).hexdigest(),
        },
        "layers": layer_rows,
    }


def aggregate(records):
    buckets = collections.defaultdict(list)
    for record in records:
        for row in record["layers"]:
            bucket = buckets[(record["field"], row["layer"])]
            bucket.append(row)
    result = {}
    for (field, layer), rows in sorted(buckets.items()):
        shifts = [
            direction["toward_source_logodds_delta"]
            for row in rows
            for direction in (row["a_to_b"], row["b_to_a"])
        ]
        lens = [
            view["toward_other_logodds"]
            for row in rows
            for view in (row["a_logit_lens"], row["b_logit_lens"])
        ]
        key = "{}:layer_{}".format(field, layer)
        result[key] = {
            "field": field,
            "layer": layer,
            "directions": len(shifts),
            "mean_toward_source_logodds_delta": sum(shifts) / len(shifts),
            "positive_direction_count": sum(value > 0.0 for value in shifts),
            "mean_logit_lens_other_minus_own": sum(lens) / len(lens),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transition-index", type=int, default=2)
    parser.add_argument("--pairs-per-regime", type=int, default=1)
    parser.add_argument("--layers", default="5,9,13,17,21,25,29")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    if args.transition_index < 0 or args.pairs_per_regime <= 0:
        raise SystemExit("transition index must be nonnegative and pairs per regime must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    layers = parse_layers(args.layers, model.cfg.n_layer)
    examples = transition_examples(args.episodes, args.transition_index)
    pairs = []
    for field in ("carry", "digit"):
        pairs.extend(select_pairs(examples, field, args.pairs_per_regime))
    records = [run_pair(model, tokenizer, pair, layers, args.alpha, device) for pair in pairs]
    result = {
        "audit": "digitwise_residual_patch_workspace_proxy_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "transition_index": args.transition_index,
        "pairs_per_regime": args.pairs_per_regime,
        "layers": layers,
        "alpha": args.alpha,
        "records": records,
        "aggregate": aggregate(records),
        "claim_boundary": (
            "This probes whether a last-position residual causally changes local next-token carry or digit scores "
            "on matched recurrent states. It is a late-layer logit-lens proxy, not a Jacobian lens or evidence "
            "of a general workspace, broad reasoning, language compilation, or long-context capability."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
