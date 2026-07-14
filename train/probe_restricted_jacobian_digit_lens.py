#!/usr/bin/env python3
"""Test a restricted, causal digit lens on recurrent-state prompts.

This is deliberately narrower than the Jacobian lens in the global-workspace
paper.  It averages only the gradient from a selected block's final prompt
position to one *next-token digit* logit, on held-out recurrent-state contexts.
The learned directions are then evaluated on disjoint held-out contexts by
readout and by a two-coordinate swap.  A successful result would establish a
small, token-specific causal diagnostic, not a global workspace, broad
reasoning, or a semantic representation.

The probe does not train a model or write a corpus.  Arithmetic is used only
to label solver-verified episodes and choose matched A/B interventions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from digitwise_protocol import microstep_prompt
from model import GPT, GPTConfig
from probe_digitwise_workspace import (
    field_prefix,
    parse_layers,
    sha256_file,
    target_digit,
    token_for_digit,
    transition_examples,
)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_key(example: dict) -> tuple:
    state = example["state"]
    return (state["op"], int(state["w"]), int(state["p"]), int(state["c"]))


def build_records(examples, tokenizer):
    """Render digit-prediction prefixes and require stable one-token digits."""
    by_digit = collections.defaultdict(set)
    records = []
    for example in examples:
        prompt = microstep_prompt(example["state"], style=example["prompt_style"])
        prefix, digit = field_prefix(prompt, example["next_state"], "digit")
        ids, token_id = token_for_digit(tokenizer, prefix, digit)
        by_digit[digit].add(token_id)
        records.append({
            "id": example["id"],
            "regime": example["regime"],
            "example": example,
            "digit": digit,
            "token_id": token_id,
            "input_ids": ids,
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        })
    token_ids = {}
    for digit in sorted(by_digit):
        if len(by_digit[digit]) != 1:
            raise ValueError("digit {} does not have a prefix-stable token id".format(digit))
        token_ids[digit] = next(iter(by_digit[digit]))
    required = {str(value) for value in range(10)}
    missing = sorted(required - set(token_ids))
    if missing:
        raise ValueError("missing target digits in source episodes: {}".format(",".join(missing)))
    return records, token_ids


def split_discovery_and_eval(records, discovery_per_digit):
    """Split by stable episode hash before any gradients are computed."""
    if discovery_per_digit <= 0:
        raise ValueError("discovery_per_digit must be positive")
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["digit"]].append(record)
    discovery, evaluation = [], []
    for digit, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: (stable_key(item["id"]), item["id"]))
        if len(ordered) <= discovery_per_digit:
            raise ValueError("digit {} has no disjoint evaluation records".format(digit))
        discovery.extend(ordered[:discovery_per_digit])
        evaluation.extend(ordered[discovery_per_digit:])
    discovery_ids = {row["id"] for row in discovery}
    if discovery_ids.intersection(row["id"] for row in evaluation):
        raise RuntimeError("discovery and evaluation episode IDs overlap")
    return discovery, evaluation


def select_readout_records(records, evaluation_per_digit):
    """Optionally bound readout work without changing the causal-pair population."""
    if evaluation_per_digit < 0:
        raise ValueError("evaluation_per_digit cannot be negative")
    if evaluation_per_digit == 0:
        return list(records)
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["digit"]].append(record)
    selected = []
    for digit, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: (stable_key(item["id"]), item["id"]))
        if len(ordered) < evaluation_per_digit:
            raise ValueError("digit {} lacks {} readout records".format(digit, evaluation_per_digit))
        selected.extend(ordered[:evaluation_per_digit])
    return selected


def select_eval_pairs(records, pairs_per_regime):
    """Choose disjoint target-digit pairs with matched local recurrent context."""
    if pairs_per_regime <= 0:
        raise ValueError("pairs_per_regime must be positive")
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["regime"]].append(record)
    pairs = []
    for regime, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: (stable_key(item["id"]), item["id"]))
        used, selected = set(), 0
        for index, left in enumerate(ordered):
            if left["id"] in used:
                continue
            for right in ordered[index + 1:]:
                if right["id"] in used:
                    continue
                if local_key(left["example"]) != local_key(right["example"]):
                    continue
                if left["digit"] == right["digit"]:
                    continue
                pairs.append({"regime": regime, "a": left, "b": right})
                used.update((left["id"], right["id"]))
                selected += 1
                break
            if selected >= pairs_per_regime:
                break
        if selected != pairs_per_regime:
            raise ValueError("could not select {} matched pairs for {}".format(pairs_per_regime, regime))
    return pairs


def pair_directions(pair):
    """Return the two symmetric target/source records for one matched pair."""

    return (("a", pair["a"], pair["b"]), ("b", pair["b"], pair["a"]))


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    if int(model.cfg.n_loop) != 1:
        raise ValueError("restricted lens requires n_loop=1 for unambiguous block indices")
    return checkpoint, model


def normalized(vector):
    return F.normalize(vector.float(), dim=0, eps=1e-12)


def gradient_at_block(model, input_ids, layer, token_id):
    """Return d(next-token logit)/d(block output) at the final prompt token."""
    captured = {}

    def hook(_module, _inputs, output):
        hidden, _ = output
        hidden.retain_grad()
        captured["hidden"] = hidden

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("block hook did not capture an activation")
        gradient = torch.autograd.grad(logits[0, -1, token_id], hidden, retain_graph=False)[0][0, -1]
    finally:
        handle.remove()
    return normalized(gradient.detach())


@torch.no_grad()
def hidden_and_logits(model, input_ids, layer):
    captured = {}

    def hook(_module, _inputs, output):
        hidden, _ = output
        captured["hidden"] = hidden[:, -1, :].detach().clone()

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        logits, _ = model(input_ids)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("block hook did not capture an activation")
    return captured["hidden"][0], logits[:, -1, :].detach()


@torch.no_grad()
def swap_logits(model, input_ids, layer, left_direction, right_direction, alpha):
    """Swap only two normalized lens coordinates at one final prompt position."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    # MPS lacks some linalg kernels and this is a fixed 2-column calculation.
    # Compute it once on CPU; the intervention itself still runs on the model device.
    vectors = torch.stack((left_direction.float().cpu(), right_direction.float().cpu()), dim=1)
    pinv = torch.linalg.pinv(vectors)

    def hook(_module, _inputs, output):
        hidden, cache = output
        current = hidden[:, -1, :]
        local_vectors = vectors.to(device=current.device, dtype=current.dtype)
        local_pinv = pinv.to(device=current.device, dtype=current.dtype)
        coords = current @ local_pinv.T
        swapped = coords[:, [1, 0]]
        delta = (swapped - coords) @ local_vectors.T
        patched = hidden.clone()
        patched[:, -1, :] = current + alpha * delta
        return patched, cache

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        logits, _ = model(input_ids)
    finally:
        handle.remove()
    return logits[:, -1, :].detach()


def logodds(logits, own_id, other_id):
    return float(logits[0, other_id] - logits[0, own_id])


def rank(scores, target_index):
    return int((scores > scores[target_index]).sum().item() + 1)


def paired_effect_summary(signal, control):
    """Report descriptive paired effects without treating directional samples as IID trials."""

    if len(signal) != len(control) or not signal:
        raise ValueError("paired signal/control samples are required")
    differences = [left - right for left, right in zip(signal, control)]
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        sample_std, sem = 0.0, 0.0
    else:
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in differences) / (len(differences) - 1))
        sem = sample_std / math.sqrt(len(differences))
    return {
        "mean": mean,
        "sample_std": sample_std,
        "sem": sem,
        "signal_exceeds_control_count": sum(value > 0.0 for value in differences),
    }


def permutation(digit):
    """Fixed, non-identity label permutation for the matched causal control."""
    return str((int(digit) + 3) % 10)


def discover_directions(model, records, layers, device):
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["digit"]].append(record)
    directions, diagnostics = {}, {}
    for layer in layers:
        layer_directions, layer_diagnostics = {}, {}
        for digit, rows in sorted(grouped.items()):
            gradients = []
            for record in rows:
                ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
                gradients.append(gradient_at_block(model, ids, layer, record["token_id"]))
            stack = torch.stack(gradients)
            mean = normalized(stack.mean(dim=0))
            layer_directions[digit] = mean
            layer_diagnostics[digit] = {
                "examples": len(rows),
                "mean_cosine_to_direction": float((stack @ mean).mean()),
                "min_cosine_to_direction": float((stack @ mean).min()),
            }
        directions[layer] = layer_directions
        diagnostics[layer] = layer_diagnostics
    return directions, diagnostics


@torch.no_grad()
def readout_scores(hidden, directions, digits):
    raw = torch.stack([hidden.float() @ directions[digit].float() for digit in digits])
    return raw - raw.mean()


def evaluate_readout(model, records, directions, layers, token_ids, device):
    digits = sorted(token_ids)
    result = {}
    for layer in layers:
        rows = []
        for index, record in enumerate(records):
            ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
            hidden, _ = hidden_and_logits(model, ids, layer)
            scores = readout_scores(hidden, directions[layer], digits)
            target_index = digits.index(record["digit"])
            rows.append(rank(scores, target_index))
            if device == "mps" and index and index % 32 == 0:
                torch.mps.empty_cache()
        result[str(layer)] = {
            "examples": len(rows),
            "top1": sum(value == 1 for value in rows),
            "mean_rank": sum(rows) / len(rows),
            "chance_top1": 1.0 / len(digits),
            "chance_mean_rank": (len(digits) + 1) / 2.0,
        }
    return result


def evaluate_pairs(model, pairs, directions, layers, token_ids, alpha, device):
    records = []
    aggregate = collections.defaultdict(lambda: {"signal": [], "control": []})
    for pair in pairs:
        per_layer = []
        for layer in layers:
            row = {"layer": layer}
            for side, target, source in pair_directions(pair):
                own, other = target["digit"], source["digit"]
                ids = torch.tensor([target["input_ids"]], dtype=torch.long, device=device)
                _, base_logits = hidden_and_logits(model, ids, layer)
                patched_logits = swap_logits(
                    model, ids, layer, directions[layer][own], directions[layer][other], alpha,
                )
                control_logits = swap_logits(
                    model,
                    ids,
                    layer,
                    directions[layer][permutation(own)],
                    directions[layer][permutation(other)],
                    alpha,
                )
                baseline = logodds(base_logits, token_ids[own], token_ids[other])
                signal = logodds(patched_logits, token_ids[own], token_ids[other]) - baseline
                control = logodds(control_logits, token_ids[own], token_ids[other]) - baseline
                row["{}_to_{}".format(side, source["id"])] = {
                    "own_digit": own,
                    "source_digit": other,
                    "baseline_toward_source_logodds": baseline,
                    "signal_delta": signal,
                    "shuffled_label_control_delta": control,
                }
                aggregate[layer]["signal"].append(signal)
                aggregate[layer]["control"].append(control)
            per_layer.append(row)
        records.append({
            "regime": pair["regime"],
            "a": {"id": pair["a"]["id"], "digit": pair["a"]["digit"], "prefix_sha256": pair["a"]["prefix_sha256"]},
            "b": {"id": pair["b"]["id"], "digit": pair["b"]["digit"], "prefix_sha256": pair["b"]["prefix_sha256"]},
            "layers": per_layer,
        })
    summary = {}
    for layer, values in sorted(aggregate.items()):
        signal, control = values["signal"], values["control"]
        paired = paired_effect_summary(signal, control)
        summary[str(layer)] = {
            "directions": len(signal),
            "mean_signal_delta": sum(signal) / len(signal),
            "positive_signal_count": sum(value > 0.0 for value in signal),
            "mean_shuffled_label_control_delta": sum(control) / len(control),
            "mean_signal_minus_control": paired["mean"],
            "signal_minus_control_sample_std": paired["sample_std"],
            "signal_minus_control_sem": paired["sem"],
            "signal_exceeds_control_count": paired["signal_exceeds_control_count"],
        }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transition-index", type=int, default=2)
    parser.add_argument("--layers", default="13,17,21,25")
    parser.add_argument("--discovery-per-digit", type=int, default=8)
    parser.add_argument(
        "--evaluation-per-digit", type=int, default=0,
        help="per-digit held-out readout cap; zero evaluates every disjoint record",
    )
    parser.add_argument("--pairs-per-regime", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--progress", action="store_true", help="emit bounded phase messages for long diagnostics")
    args = parser.parse_args()
    if args.transition_index < 0:
        raise SystemExit("transition index must be nonnegative")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    if args.progress:
        print("[rjdl] loaded step={} device={}".format(checkpoint.get("step"), device), flush=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    layers = parse_layers(args.layers, model.cfg.n_layer)
    source = transition_examples(args.episodes, args.transition_index)
    records, token_ids = build_records(source, tokenizer)
    discovery, evaluation = split_discovery_and_eval(records, args.discovery_per_digit)
    readout_evaluation = select_readout_records(evaluation, args.evaluation_per_digit)
    pairs = select_eval_pairs(evaluation, args.pairs_per_regime)
    if args.progress:
        print("[rjdl] records={} discovery={} evaluation={} readout={} pairs={} layers={}".format(
            len(records), len(discovery), len(evaluation), len(readout_evaluation), len(pairs), layers
        ), flush=True)
    directions, direction_diagnostics = discover_directions(model, discovery, layers, device)
    if args.progress:
        print("[rjdl] discovery gradients complete", flush=True)
    readout = evaluate_readout(model, readout_evaluation, directions, layers, token_ids, device)
    if args.progress:
        print("[rjdl] held-out readout complete", flush=True)
    pair_records, pair_summary = evaluate_pairs(
        model, pairs, directions, layers, token_ids, args.alpha, device,
    )
    if args.progress:
        print("[rjdl] matched causal swaps complete", flush=True)
    result = {
        "audit": "restricted_jacobian_digit_lens_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "transition_index": args.transition_index,
        "layers": layers,
        "discovery_per_digit": args.discovery_per_digit,
        "evaluation_per_digit": args.evaluation_per_digit,
        "pairs_per_regime": args.pairs_per_regime,
        "alpha": args.alpha,
        "token_ids": token_ids,
        "discovery_episode_ids_sha256": hashlib.sha256("\n".join(sorted(row["id"] for row in discovery)).encode()).hexdigest(),
        "evaluation_episode_ids_sha256": hashlib.sha256("\n".join(sorted(row["id"] for row in evaluation)).encode()).hexdigest(),
        "readout_episode_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["id"] for row in readout_evaluation)).encode()
        ).hexdigest(),
        "direction_diagnostics": direction_diagnostics,
        "readout": readout,
        "pair_records": pair_records,
        "pair_summary": pair_summary,
        "claim_boundary": (
            "This is a restricted next-token digit Jacobian diagnostic. It does not compute the paper's full "
            "cross-position J-lens, does not establish a global workspace, and cannot establish broad reasoning, "
            "language-level concepts, or context scaling. A positive result requires both disjoint readout and "
            "a signal-over-shuffled-label causal swap; otherwise it is rejected."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "readout": readout, "pair_summary": pair_summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
