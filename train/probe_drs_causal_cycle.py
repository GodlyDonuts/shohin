#!/usr/bin/env python3
"""Causally decompose DRS write, serialization, transport, and consumption."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
from tokenizers import Tokenizer

from digitwise_protocol import apply_microstep, canonical_state, microstep_prompt, parse_state
from eval_suite import has_complete_final_answer
from model import GPT, GPTConfig
from probe_digitwise_workspace import collect_residuals, field_prefix, patched_logits, token_for_digit


EXPECTED_CHECKPOINT_SHA256 = "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
EXPECTED_EPISODES_SHA256 = "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
EXPECTED_TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
EXPECTED_REGIMES = ("fit_w4", "fit_w6", "value_ood_w4", "value_ood_w6", "width_ood_w8")
EXPECTED_CANONICAL_OUTPUT = Path(
    "/lustre/fs1/home/sa305415/shohin/artifacts/evals/drs_causal_cycle_post_drs_r2.json"
)
SCIENTIFIC_SOURCE_FILES = (
    "probe_drs_causal_cycle.py",
    "digitwise_protocol.py",
    "eval_suite.py",
    "model.py",
    "probe_digitwise_workspace.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(token_ids):
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def scientific_source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in SCIENTIFIC_SOURCE_FILES}


def verify_frozen_inputs(ckpt, episodes, tokenizer):
    observed = {
        "checkpoint": sha256_file(ckpt),
        "episodes": sha256_file(episodes),
        "tokenizer": sha256_file(tokenizer),
    }
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "episodes": EXPECTED_EPISODES_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
    }
    if observed != expected:
        raise ValueError("frozen input hash mismatch: {}".format(observed))
    return observed


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    if int(model.cfg.n_loop) != 1:
        raise ValueError("causal-cycle probe requires n_loop=1")
    return checkpoint, model


def state_before_transition(episode, transition_index):
    if transition_index <= 0 or transition_index >= len(episode["expected_states"]):
        return None
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("invalid initial state in {}".format(episode["id"]))
    for index in range(transition_index):
        state = parse_state(episode["expected_states"][index])
        if state is None:
            raise ValueError("invalid expected state in {}".format(episode["id"]))
    expected = parse_state(episode["expected_states"][transition_index])
    if expected is None or apply_microstep(state) != expected:
        raise ValueError("solver-inconsistent transition in {}".format(episode["id"]))
    return state


def carry_counterfactual(state):
    if int(state["p"]) <= 0 or int(state["z"]):
        raise ValueError("carry counterfactual requires an interior state")
    changed = dict(state)
    changed["c"] = 1 - int(state["c"])
    canonical_state(changed)
    return changed


def irrelevant_history_counterfactual(state):
    if int(state["p"]) <= 0 or int(state["z"]):
        raise ValueError("history counterfactual requires a written result digit")
    changed = dict(state)
    result = list(changed["r"])
    result[0] = str((int(result[0]) + 1) % 10)
    changed["r"] = "".join(result)
    canonical_state(changed)
    return changed


def active_digit(state):
    position = int(state["p"]) - 1
    if position < 0:
        raise ValueError("state has no newly written digit")
    return int(state["r"][position])


def case_from_episode(episode, transition_index):
    base = state_before_transition(episode, transition_index)
    if base is None:
        return None
    counterfactual = carry_counterfactual(base)
    base_next = apply_microstep(base)
    counterfactual_next = apply_microstep(counterfactual)
    return {
        "episode": episode,
        "base": base,
        "counterfactual": counterfactual,
        "base_next": base_next,
        "counterfactual_next": counterfactual_next,
        "boundary": int(base_next["c"]) != int(counterfactual_next["c"]),
    }


def same_target_signature(case):
    state, next_state = case["base"], case["base_next"]
    return (
        case["episode"]["split"], state["op"], int(state["w"]), int(state["p"]),
        int(state["c"]), int(next_state["c"]), active_digit(next_state),
    )


def select_cases(path, transition_index, per_regime):
    all_cases = []
    for raw in Path(path).read_text().splitlines():
        if raw.strip():
            case = case_from_episode(json.loads(raw), transition_index)
            if case is not None:
                all_cases.append(case)
    by_regime = collections.defaultdict(list)
    for case in sorted(all_cases, key=lambda item: item["episode"]["id"]):
        if case["boundary"]:
            by_regime[case["episode"]["split"]].append(case)
    inventory = {regime: len(by_regime[regime]) for regime in EXPECTED_REGIMES}
    if any(inventory[regime] < per_regime for regime in EXPECTED_REGIMES):
        raise ValueError("insufficient boundary-conditioned inventory: {}".format(inventory))
    selected = [case for regime in EXPECTED_REGIMES for case in by_regime[regime][:per_regime]]
    target_buckets = collections.defaultdict(list)
    for candidate in all_cases:
        target_buckets[same_target_signature(candidate)].append(candidate)
    for case in selected:
        donors = [
            candidate for candidate in target_buckets[same_target_signature(case)]
            if candidate["episode"]["id"] != case["episode"]["id"]
        ]
        if not donors:
            raise ValueError("no independent same-target donor for {}".format(case["episode"]["id"]))
        case["same_target_donor"] = sorted(donors, key=lambda item: item["episode"]["id"])[0]
    return selected, inventory


def hybrid_next(base_next, counterfactual_next, fields):
    fields = set(fields)
    result = dict(base_next)
    if "carry" in fields:
        result["c"] = int(counterfactual_next["c"])
    if "digit" in fields:
        position = int(result["p"]) - 1
        tape = list(result["r"])
        tape[position] = counterfactual_next["r"][position]
        result["r"] = "".join(tape)
    canonical_state(result)
    return result


def render_field(model, tokenizer, prompt, next_state, field, device):
    prefix, target = field_prefix(prompt, next_state, field)
    ids, target_id = token_for_digit(tokenizer, prefix, target)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    residuals, logits = collect_residuals(model, tensor)
    return {
        "prefix": prefix,
        "ids": ids,
        "target": target,
        "target_id": target_id,
        "residuals": residuals,
        "logits": logits,
    }


def digit_token_ids(tokenizer, prefix):
    return [token_for_digit(tokenizer, prefix, digit)[1] for digit in "0123456789"]


def field_metrics(record, tokenizer, identity_logits=None):
    ids = digit_token_ids(tokenizer, record["prefix"])
    local = record["logits"][0, ids]
    result = {
        "argmax": int(local.argmax().item()),
        "target": int(record["target"]),
        "correct": int(local.argmax().item()) == int(record["target"]),
        "digit_logits": [float(value) for value in local],
    }
    if identity_logits is not None:
        result["identity_max_abs_logit_delta"] = float(
            (record["logits"] - identity_logits).abs().max().item()
        )
    return result


def intervention_site(target_prompt, desired_state, field, layer, tokenizer, mode, source_residual=None):
    if mode not in {"residual", "token"}:
        raise ValueError("unknown intervention mode")
    prefix, desired = field_prefix(target_prompt, desired_state, field)
    ids, target_id = token_for_digit(tokenizer, prefix, desired)
    if mode == "residual" and source_residual is None:
        raise ValueError("residual intervention lacks a source")
    return {
        "field": field,
        "layer": int(layer),
        "prefix_ids": tuple(ids),
        "source_residual": source_residual,
        "target_id": int(target_id),
        "target_digit": desired,
        "mode": mode,
    }


def _forward_with_optional_patch(model, idx, cache, pos, site):
    handle = None
    if site is not None and site["mode"] == "residual":
        def hook(_module, _inputs, output):
            hidden, past = output
            source = site["source_residual"].to(device=hidden.device, dtype=hidden.dtype)
            if source.shape != hidden[:, -1, :].shape:
                raise ValueError("source residual shape mismatch")
            patched = hidden.clone()
            patched[:, -1, :] = source
            return patched, past

        handle = model.blocks[site["layer"]].register_forward_hook(hook)
    try:
        return model(idx, cache=cache, pos=pos, return_cache=True)
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def greedy_state(model, tokenizer, prompt, device, max_new, sites=(), stop="\nQuestion:"):
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    ids = list(prompt_ids)
    generated = []
    fired = []
    fired_prefixes = set()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    sites_by_prefix = {site["prefix_ids"]: site for site in sites}
    if len(sites_by_prefix) != len(sites):
        raise ValueError("intervention sites have duplicate token prefixes")

    current_site = sites_by_prefix.get(tuple(ids))
    logits, cache = _forward_with_optional_patch(
        model, torch.tensor([ids], dtype=torch.long, device=device), None, 0, current_site
    )
    if current_site is not None and current_site["mode"] == "residual":
        fired_prefixes.add(current_site["prefix_ids"])
        fired.append({"field": current_site["field"], "mode": "residual", "prefix_length": len(ids)})
    position = len(ids)
    for _ in range(max_new):
        site = sites_by_prefix.get(tuple(ids))
        if site is not None and site["mode"] == "token":
            next_id = site["target_id"]
            if site["prefix_ids"] not in fired_prefixes:
                fired_prefixes.add(site["prefix_ids"])
                fired.append({"field": site["field"], "mode": "token", "prefix_length": len(ids)})
        else:
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        if ((stop and stop in text) or has_complete_final_answer(text)
                or (eos_id is not None and next_id == eos_id) or position >= cap):
            break
        ids.append(next_id)
        next_site = sites_by_prefix.get(tuple(ids))
        logits, cache = _forward_with_optional_patch(
            model, torch.tensor([[next_id]], dtype=torch.long, device=device), cache, position, next_site
        )
        if (next_site is not None and next_site["mode"] == "residual"
                and next_site["prefix_ids"] not in fired_prefixes):
            fired_prefixes.add(next_site["prefix_ids"])
            fired.append({"field": next_site["field"], "mode": "residual", "prefix_length": len(ids)})
        position += 1
    returned = generated[:-1] if generated and eos_id is not None and generated[-1] == eos_id else generated
    text = tokenizer.decode(returned, skip_special_tokens=False)
    return {"response": text, "token_ids": returned, "state": parse_state(text), "fired": fired}


def arm_record(result, expected, tokenizer, style):
    parsed = result["state"]
    intended_prompt = microstep_prompt(expected, style=style)
    intended_ids = tokenizer.encode(intended_prompt).ids
    parsed_ids = tokenizer.encode(microstep_prompt(parsed, style=style)).ids if parsed is not None else None
    return {
        "response": result["response"],
        "token_ids": result["token_ids"],
        "token_sha256": token_ids_sha256(result["token_ids"]),
        "parsed": canonical_state(parsed) if parsed is not None else None,
        "expected": canonical_state(expected),
        "exact": parsed == expected,
        "fired": result["fired"],
        "next_prompt_token_transport_exact": parsed_ids == intended_ids if parsed is not None else False,
        "actual_next_prompt_token_sha256": token_ids_sha256(parsed_ids) if parsed_ids is not None else None,
        "intended_next_prompt_token_sha256": token_ids_sha256(intended_ids),
    }


def consumer_branch(model, tokenizer, state, style, layer, device, max_new):
    expected = apply_microstep(state)
    prompt = microstep_prompt(state, style=style)
    fields = {}
    for field in ("carry", "digit"):
        rendered = render_field(model, tokenizer, prompt, expected, field, device)
        fields[field] = field_metrics(rendered, tokenizer)
    generated = greedy_state(model, tokenizer, prompt, device, max_new)
    return {
        "state": canonical_state(state),
        "expected": canonical_state(expected),
        "active_digit": active_digit(expected),
        "fields": fields,
        "greedy": arm_record(generated, expected, tokenizer, style),
        "layer": layer,
    }


@torch.no_grad()
def evaluate_record(model, tokenizer, case, layer, device, max_new):
    episode = case["episode"]
    same_case = case["same_target_donor"]
    base, donor = case["base"], case["counterfactual"]
    base_next, donor_next = case["base_next"], case["counterfactual_next"]
    irrelevant = irrelevant_history_counterfactual(base)
    irrelevant_next = apply_microstep(irrelevant)
    carry_hybrid = hybrid_next(base_next, donor_next, {"carry"})
    digit_hybrid = hybrid_next(base_next, donor_next, {"digit"})
    style = episode["prompt_style"]
    base_prompt = microstep_prompt(base, style=style)
    donor_prompt = microstep_prompt(donor, style=style)
    irrelevant_prompt = microstep_prompt(irrelevant, style=style)
    same_prompt = microstep_prompt(same_case["base"], style=same_case["episode"]["prompt_style"])

    def fields_for(prompt, next_state):
        return {
            field: render_field(model, tokenizer, prompt, next_state, field, device)
            for field in ("carry", "digit")
        }

    base_fields = fields_for(base_prompt, base_next)
    donor_fields = fields_for(donor_prompt, donor_next)
    irrelevant_fields = fields_for(irrelevant_prompt, irrelevant_next)
    same_fields = fields_for(same_prompt, same_case["base_next"])

    field_rows = {}
    identity_exact = True
    for field in ("carry", "digit"):
        tensor = torch.tensor([base_fields[field]["ids"]], dtype=torch.long, device=device)
        identity_logits = patched_logits(
            model, tensor, layer, base_fields[field]["residuals"][layer], alpha=1.0
        )
        irrelevant_patch_logits = patched_logits(
            model, tensor, layer, irrelevant_fields[field]["residuals"][layer], alpha=1.0
        )
        base_metrics = field_metrics(base_fields[field], tokenizer, identity_logits)
        donor_metrics = field_metrics(donor_fields[field], tokenizer)
        irrelevant_metrics = field_metrics(irrelevant_fields[field], tokenizer)
        irrelevant_patch_record = dict(base_fields[field])
        irrelevant_patch_record["logits"] = irrelevant_patch_logits
        irrelevant_patch_metrics = field_metrics(irrelevant_patch_record, tokenizer)
        irrelevant_patch_metrics.update({
            "argmax_invariant": irrelevant_patch_metrics["argmax"] == base_metrics["argmax"],
            "max_abs_digit_logit_delta": max(
                abs(a - b)
                for a, b in zip(base_metrics["digit_logits"], irrelevant_patch_metrics["digit_logits"])
            ),
        })
        same_metrics = field_metrics(same_fields[field], tokenizer)
        identity_exact &= base_metrics["identity_max_abs_logit_delta"] == 0.0
        field_rows[field] = {
            "base": base_metrics,
            "carry_flip": donor_metrics,
            "same_target_donor": same_metrics,
            "irrelevant_history": {
                **irrelevant_metrics,
                "same_target_as_base": irrelevant_metrics["target"] == base_metrics["target"],
                "argmax_invariant": irrelevant_metrics["argmax"] == base_metrics["argmax"],
                "max_abs_digit_logit_delta": max(
                    abs(a - b) for a, b in zip(base_metrics["digit_logits"], irrelevant_metrics["digit_logits"])
                ),
            },
            "irrelevant_transplant_into_base": irrelevant_patch_metrics,
        }

    def residual_sites(desired, sources, fields=("carry", "digit")):
        return [
            intervention_site(
                base_prompt, desired, field, layer, tokenizer, "residual",
                sources[field]["residuals"][layer],
            )
            for field in fields
        ]

    def token_sites(desired):
        return [
            intervention_site(base_prompt, desired, field, layer, tokenizer, "token")
            for field in ("carry", "digit")
        ]

    generated = {
        "baseline": greedy_state(model, tokenizer, base_prompt, device, max_new),
        "identity": greedy_state(model, tokenizer, base_prompt, device, max_new, residual_sites(base_next, base_fields)),
        "same_target": greedy_state(model, tokenizer, base_prompt, device, max_new, residual_sites(base_next, same_fields)),
        "carry_only": greedy_state(
            model, tokenizer, base_prompt, device, max_new,
            residual_sites(carry_hybrid, donor_fields, ("carry",)),
        ),
        "digit_only": greedy_state(
            model, tokenizer, base_prompt, device, max_new,
            residual_sites(digit_hybrid, donor_fields, ("digit",)),
        ),
        "both": greedy_state(model, tokenizer, base_prompt, device, max_new, residual_sites(donor_next, donor_fields)),
        "token_ceiling": greedy_state(model, tokenizer, base_prompt, device, max_new, token_sites(donor_next)),
        "irrelevant_sham": greedy_state(
            model, tokenizer, base_prompt, device, max_new, residual_sites(base_next, irrelevant_fields)
        ),
    }
    expected_by_arm = {
        "baseline": base_next,
        "identity": base_next,
        "same_target": base_next,
        "carry_only": carry_hybrid,
        "digit_only": digit_hybrid,
        "both": donor_next,
        "token_ceiling": donor_next,
        "irrelevant_sham": base_next,
    }
    arms = {
        name: arm_record(result, expected_by_arm[name], tokenizer, style)
        for name, result in generated.items()
    }
    base_consumer = consumer_branch(model, tokenizer, base_next, style, layer, device, max_new)
    donor_consumer = consumer_branch(model, tokenizer, donor_next, style, layer, device, max_new)
    paired_digit_switch = bool(
        base_consumer["active_digit"] != donor_consumer["active_digit"]
        and base_consumer["fields"]["digit"]["correct"]
        and donor_consumer["fields"]["digit"]["correct"]
        and base_consumer["fields"]["digit"]["argmax"] != donor_consumer["fields"]["digit"]["argmax"]
    )
    return {
        "episode_id": episode["id"],
        "same_target_donor_id": same_case["episode"]["id"],
        "regime": episode["split"],
        "transition_index": int(base["p"]),
        "base_state": canonical_state(base),
        "counterfactual_state": canonical_state(donor),
        "base_next": canonical_state(base_next),
        "counterfactual_next": canonical_state(donor_next),
        "boundary_condition": bool(case["boundary"]),
        "teacher_forced_identity_exact": identity_exact,
        "fields": field_rows,
        "arms": arms,
        "consumer": {
            "base": base_consumer,
            "counterfactual": donor_consumer,
            "paired_active_digit_switch_correct": paired_digit_switch,
        },
        "integrated_cycle_exact": bool(arms["both"]["exact"] and donor_consumer["greedy"]["exact"]),
    }


def rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def endpoint_summary(records):
    total = len(records)
    reaches = {}
    for arm in ("identity", "same_target", "carry_only", "digit_only", "both", "token_ceiling", "irrelevant_sham"):
        reaches[arm] = {
            field: sum(any(hit["field"] == field for hit in row["arms"][arm]["fired"]) for row in records)
            for field in ("carry", "digit")
        }
    baseline_exact = sum(row["arms"]["baseline"]["exact"] for row in records)
    same_exact = sum(row["arms"]["same_target"]["exact"] for row in records)
    baseline_failures = total - baseline_exact
    same_rescues = sum(
        not row["arms"]["baseline"]["exact"] and row["arms"]["same_target"]["exact"] for row in records
    )
    both_exact = sum(row["arms"]["both"]["exact"] for row in records)
    token_exact = sum(row["arms"]["token_ceiling"]["exact"] for row in records)
    transport_exact = sum(row["arms"]["both"]["next_prompt_token_transport_exact"] for row in records)
    irrelevant_context_invariant = sum(
        row["fields"]["carry"]["irrelevant_history"]["argmax_invariant"]
        and row["fields"]["digit"]["irrelevant_history"]["argmax_invariant"]
        for row in records
    )
    irrelevant_transplant_invariant = sum(
        row["fields"]["carry"]["irrelevant_transplant_into_base"]["argmax_invariant"]
        and row["fields"]["digit"]["irrelevant_transplant_into_base"]["argmax_invariant"]
        for row in records
    )
    irrelevant_sham_token_equal = sum(
        row["arms"]["irrelevant_sham"]["token_ids"] == row["arms"]["baseline"]["token_ids"]
        for row in records
    )
    irrelevant_sham_exact = sum(row["arms"]["irrelevant_sham"]["exact"] for row in records)
    paired_consumer = sum(row["consumer"]["paired_active_digit_switch_correct"] for row in records)
    base_consumer_exact = sum(row["consumer"]["base"]["greedy"]["exact"] for row in records)
    donor_consumer_exact = sum(row["consumer"]["counterfactual"]["greedy"]["exact"] for row in records)
    cycles = sum(row["integrated_cycle_exact"] for row in records)
    consumer_fields = {
        branch: {
            field: {
                "correct": sum(row["consumer"][branch]["fields"][field]["correct"] for row in records),
                "rate": rate(
                    sum(row["consumer"][branch]["fields"][field]["correct"] for row in records), total
                ),
            }
            for field in ("carry", "digit")
        }
        for branch in ("base", "counterfactual")
    }
    return {
        "records": total,
        "site_reach_by_arm_and_field": reaches,
        "baseline_first_state_exact": baseline_exact,
        "baseline_first_state_exact_rate": rate(baseline_exact, total),
        "same_target_first_state_exact": same_exact,
        "same_target_first_state_exact_rate": rate(same_exact, total),
        "baseline_failures": baseline_failures,
        "same_target_rescues": same_rescues,
        "same_target_rescue_rate_on_baseline_failures": rate(same_rescues, baseline_failures),
        "same_target_minus_baseline_pp": rate(same_exact - baseline_exact, total),
        "counterfactual_first_state_exact": both_exact,
        "counterfactual_first_state_exact_rate": rate(both_exact, total),
        "token_ceiling_first_state_exact": token_exact,
        "token_ceiling_first_state_exact_rate": rate(token_exact, total),
        "intended_next_prompt_transport_exact": transport_exact,
        "intended_next_prompt_transport_exact_rate": rate(transport_exact, total),
        "irrelevant_context_argmax_invariant": irrelevant_context_invariant,
        "irrelevant_context_argmax_invariant_rate": rate(irrelevant_context_invariant, total),
        "irrelevant_transplant_argmax_invariant": irrelevant_transplant_invariant,
        "irrelevant_transplant_argmax_invariant_rate": rate(irrelevant_transplant_invariant, total),
        "irrelevant_sham_token_equal_to_baseline": irrelevant_sham_token_equal,
        "irrelevant_sham_token_equal_to_baseline_rate": rate(irrelevant_sham_token_equal, total),
        "irrelevant_sham_first_state_exact": irrelevant_sham_exact,
        "irrelevant_sham_first_state_exact_rate": rate(irrelevant_sham_exact, total),
        "paired_second_call_active_digit_switch_correct": paired_consumer,
        "paired_second_call_active_digit_switch_correct_rate": rate(paired_consumer, total),
        "second_call_teacher_forced_field_accuracy": consumer_fields,
        "base_second_state_exact": base_consumer_exact,
        "base_second_state_exact_rate": rate(base_consumer_exact, total),
        "counterfactual_second_state_exact": donor_consumer_exact,
        "counterfactual_second_state_exact_rate": rate(donor_consumer_exact, total),
        "integrated_two_call_cycle_exact": cycles,
        "integrated_two_call_cycle_exact_rate": rate(cycles, total),
    }


def summarize(records):
    total = len(records)
    regimes = collections.Counter(row["regime"] for row in records)
    identity_token_mismatch = sum(
        row["arms"]["identity"]["token_ids"] != row["arms"]["baseline"]["token_ids"] for row in records
    )
    identity_teacher_fail = sum(not row["teacher_forced_identity_exact"] for row in records)
    expected_regimes = {regime: 10 for regime in EXPECTED_REGIMES}
    valid = bool(
        total == 50 and dict(regimes) == expected_regimes and identity_token_mismatch == 0
        and identity_teacher_fail == 0 and all(row["boundary_condition"] for row in records)
    )
    result = endpoint_summary(records)
    result.update({
        "records_by_regime": dict(sorted(regimes.items())),
        "identity_token_mismatch": identity_token_mismatch,
        "teacher_forced_identity_failures": identity_teacher_fail,
        "by_regime": {
            regime: endpoint_summary([row for row in records if row["regime"] == regime])
            for regime in EXPECTED_REGIMES
        },
        "valid": valid,
    })
    return result


def publish_json_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.partial".format(path.name, os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _all_regimes_at_least(aggregate, key, floor):
    return all(
        aggregate["by_regime"][regime][key] is not None
        and aggregate["by_regime"][regime][key] >= floor
        for regime in EXPECTED_REGIMES
    )


def _consumer_fields_at_least(summary, floor):
    return all(
        summary["second_call_teacher_forced_field_accuracy"][branch][field]["rate"] is not None
        and summary["second_call_teacher_forced_field_accuracy"][branch][field]["rate"] >= floor
        for branch in ("base", "counterfactual")
        for field in ("carry", "digit")
    )


def build_decision(aggregate, mechanically_valid):
    decision = {
        "mechanically_valid": bool(mechanically_valid),
        "write_serialization_pass": None,
        "token_ceiling_pass": None,
        "consumer_pass": None,
        "irrelevant_history_pass": None,
        "native_residual_rescue_signal": None,
    }
    if not mechanically_valid:
        return decision
    decision.update({
        "write_serialization_pass": (
            aggregate["counterfactual_first_state_exact_rate"] >= 0.50
            and _all_regimes_at_least(aggregate, "counterfactual_first_state_exact_rate", 0.30)
        ),
        "token_ceiling_pass": (
            aggregate["token_ceiling_first_state_exact_rate"] >= 0.80
            and _all_regimes_at_least(aggregate, "token_ceiling_first_state_exact_rate", 0.70)
        ),
        "consumer_pass": (
            aggregate["paired_second_call_active_digit_switch_correct_rate"] >= 0.70
            and _consumer_fields_at_least(aggregate, 0.70)
            and _all_regimes_at_least(
                aggregate, "paired_second_call_active_digit_switch_correct_rate", 0.50
            )
            and all(
                _consumer_fields_at_least(aggregate["by_regime"][regime], 0.50)
                for regime in EXPECTED_REGIMES
            )
        ),
        "irrelevant_history_pass": (
            aggregate["irrelevant_transplant_argmax_invariant_rate"] >= 0.90
            and aggregate["irrelevant_sham_token_equal_to_baseline_rate"] >= 0.90
            and _all_regimes_at_least(
                aggregate, "irrelevant_transplant_argmax_invariant_rate", 0.80
            )
            and _all_regimes_at_least(
                aggregate, "irrelevant_sham_token_equal_to_baseline_rate", 0.80
            )
        ),
        "native_residual_rescue_signal": (
            aggregate["same_target_rescue_rate_on_baseline_failures"] is not None
            and aggregate["same_target_rescue_rate_on_baseline_failures"] >= 0.50
            and aggregate["same_target_minus_baseline_pp"] >= 0.20
        ),
    })
    return decision


def canonical_preflight(args, device):
    frozen_config = (
        args.transition_index == 2 and args.per_regime == 10
        and args.layer == 29 and args.max_new == 96
    )
    snapshot_text = os.environ.get("SHOHIN_DRS_CYCLE_SNAPSHOT_ROOT")
    snapshot_verified = os.environ.get("SHOHIN_DRS_CYCLE_CODE_VERIFIED") == "1"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not frozen_config:
        raise SystemExit("canonical causal-cycle configuration is frozen")
    if device != "cuda":
        raise SystemExit("canonical causal-cycle run requires CUDA BF16")
    if Path(args.out).resolve() != EXPECTED_CANONICAL_OUTPUT:
        raise SystemExit("canonical causal-cycle output path is frozen")
    if not snapshot_text or not snapshot_verified or not slurm_job_id:
        raise SystemExit("canonical causal-cycle run requires a verified private Slurm snapshot")
    snapshot = Path(snapshot_text).resolve()
    scientific_paths = [Path(__file__).resolve(), *(
        Path(value).resolve() for value in (args.ckpt, args.tokenizer, args.episodes)
    )]
    if any(snapshot not in path.parents for path in scientific_paths):
        raise SystemExit("canonical causal-cycle input escaped the private snapshot")
    return {
        "canonical": True,
        "development_only": False,
        "snapshot_private": True,
        "snapshot_root_name": snapshot.name,
        "slurm_job_id": slurm_job_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transition-index", type=int, default=2)
    parser.add_argument("--per-regime", type=int, default=10)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--allow-noncanonical-smoke", action="store_true")
    args = parser.parse_args()
    if args.canonical and args.allow_noncanonical_smoke:
        raise SystemExit("canonical and development-only modes are mutually exclusive")
    if not args.canonical and not args.allow_noncanonical_smoke:
        raise SystemExit("choose --canonical or --allow-noncanonical-smoke explicitly")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    hashes = verify_frozen_inputs(args.ckpt, args.episodes, args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    canonical = canonical_preflight(args, device) if args.canonical else {
        "canonical": False,
        "development_only": True,
        "snapshot_private": False,
        "snapshot_root_name": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    source_hashes = scientific_source_hashes()
    checkpoint, model = load_model(args.ckpt, device)
    if not 0 <= args.layer < len(model.blocks):
        raise SystemExit("intervention layer is outside model")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cases, boundary_inventory = select_cases(args.episodes, args.transition_index, args.per_regime)
    records = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    with autocast:
        for index, case in enumerate(cases, 1):
            records.append(evaluate_record(model, tokenizer, case, args.layer, device, args.max_new))
            print(
                "[causal-cycle] {}/{} {}".format(index, len(cases), case["episode"]["id"]),
                flush=True,
            )
    aggregate = summarize(records)
    mechanically_valid = bool(canonical["canonical"] and aggregate["valid"])
    decision = build_decision(aggregate, mechanically_valid)
    result = {
        "audit": "drs_causal_cycle_post_drs_v2",
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "episodes": args.episodes,
        "frozen_sha256": hashes,
        "scientific_source_sha256": source_hashes,
        "run_mode": canonical,
        "device": device,
        "precision": "cuda_bfloat16_autocast" if device == "cuda" else "native_{}".format(device),
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.machine(),
            "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else None,
        },
        "transition_index": args.transition_index,
        "per_regime": args.per_regime,
        "layer": args.layer,
        "max_new": args.max_new,
        "boundary_inventory": boundary_inventory,
        "aggregate": aggregate,
        "records": records,
        "decision": decision,
        "claim_boundary": (
            "Oracle residuals or target tokens are supplied at exact next-token sites. This localizes DRS "
            "production, serialization, text transport, and next-step response; it is not autonomous reasoning, "
            "latent persistence, or a new primitive."
        ),
    }
    publish_json_exclusive(out, result)
    print(json.dumps({"aggregate": aggregate, "decision": decision}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
