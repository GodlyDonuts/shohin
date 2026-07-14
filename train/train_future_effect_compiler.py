#!/usr/bin/env python3
"""Train the R6 selected-probe effect head from the frozen R4 pointer binder."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from future_effect_algebra import effect_signature, operation_operator, redundant_probe_bank
from future_effect_compiler import ProbeConditionedEffectCompiler
from model import GPT, GPTConfig
from role_equivariant_microcode import IGNORE_ROLE
from train_categorical_microcode import adapter_state, hash_adapter_state
from train_referential_slot_microcode import (
    VIEW_ORDER,
    classify_batch,
    flatten_outputs,
    flatten_targets,
    load_groups,
    lr_scale,
    make_batches,
    masked_cross_entropy,
    mention_loss,
    pad_ids,
)


EFFECT_SCALE = 64.0
HELDOUT_PROBE_INDICES = tuple(
    index for index in range(64) if ((index // 8) + (index % 8)) % 4 == 0
)
TRAIN_PROBE_INDICES = tuple(index for index in range(64) if index not in HELDOUT_PROBE_INDICES)


def selected_probe_indices(step, operation_number, count):
    """Deterministic coverage of the frozen 48-probe training subset."""
    count = int(count)
    if count <= 0 or count > len(TRAIN_PROBE_INDICES):
        raise ValueError("invalid probes-per-operation")
    start = (int(step) * 17 + int(operation_number) * 11) % len(TRAIN_PROBE_INDICES)
    return tuple(
        TRAIN_PROBE_INDICES[(start + offset) % len(TRAIN_PROBE_INDICES)]
        for offset in range(count)
    )


def counterfactual_effect_loss(compiler, results, examples, step, probes_per_operation):
    states, queries = redundant_probe_bank(dtype=torch.float32, device=results[0]["slots"].device)
    losses = []
    operation_number = 0
    for result, example in zip(results, examples):
        for local, output in enumerate(result["operations"]):
            indices = selected_probe_indices(step, operation_number, probes_per_operation)
            index = torch.tensor(indices, dtype=torch.long, device=states.device)
            state_index = index.remainder(states.shape[0])
            query_index = torch.div(index, states.shape[0], rounding_mode="floor")
            selected_states = states.index_select(0, state_index)
            selected_queries = queries.index_select(0, query_index)
            predicted = compiler.predict_effect(output, selected_states, selected_queries)
            operator = operation_operator(
                example.compiled.operation_targets[local],
                example.compiled.operation_values[local],
                dtype=torch.float32,
                device=states.device,
            )
            target = effect_signature(
                operator, states=states, queries=queries,
            ).reshape(-1).index_select(0, index) / EFFECT_SCALE
            losses.append(F.smooth_l1_loss(predicted, target, beta=0.05))
            operation_number += 1
    if not losses:
        raise ValueError("batch contains no operations")
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--pointer-adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--effect-hidden", type=int, default=128)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=48000)
    parser.add_argument("--probes-per-operation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--mention-weight", type=float, default=1.0)
    parser.add_argument("--basis-weight", type=float, default=1.0)
    parser.add_argument("--effect-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("future-effect training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    if args.probes_per_operation <= 0 or args.effect_weight <= 0:
        raise SystemExit("invalid probe count or effect weight")
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    admission = json.load(open(args.admission))
    label_admission = json.load(open(args.label_admission))
    data_sha256 = sha256_file(args.data)
    if not admission.get("all_checks_pass") or admission.get("train_sha256") != data_sha256:
        raise SystemExit("structural admission did not bind effect data")
    if not label_admission.get("all_checks_pass"):
        raise SystemExit("mention-label admission failed")
    if label_admission["datasets"]["train"].get("sha256") != data_sha256:
        raise SystemExit("mention-label admission did not bind effect data")
    if label_admission.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("mention-label admission did not bind tokenizer")

    base_checkpoint = torch.load(args.init, map_location="cpu")
    pointer_checkpoint = torch.load(args.pointer_adapter, map_location="cpu")
    pointer_metadata = pointer_checkpoint.get("categorical_microcode", {})
    if pointer_metadata.get("protocol") != "causal_microcode_referential_slots_v4":
        raise SystemExit("invalid R4 pointer adapter")
    if pointer_metadata.get("role_mode") != "pointer":
        raise SystemExit("R6 requires the R4 pointer arm")
    if pointer_metadata.get("base_sha256") != sha256_file(args.init):
        raise SystemExit("R4 pointer adapter does not bind supplied base")
    if pointer_metadata.get("data_sha256") != data_sha256:
        raise SystemExit("R4 pointer adapter does not bind supplied train data")

    cfg = GPTConfig(**base_checkpoint["cfg"])
    groups = load_groups(args.data, tokenizer, cfg.seq_len)
    batches = make_batches(groups, args.batch_groups, args.max_groups, args.seed)
    if not batches:
        raise SystemExit("selected zero future-effect batches")

    model = GPT(cfg).to("cuda")
    model.load_state_dict(base_checkpoint["model"])
    compiler = ProbeConditionedEffectCompiler(
        model,
        layer=args.layer,
        hidden=args.hidden,
        effect_hidden=args.effect_hidden,
    ).to("cuda")
    missing, unexpected = compiler.load_state_dict(pointer_checkpoint["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    expected_missing = {
        name for name, _ in compiler.named_parameters() if name.startswith("effect_")
    }
    if set(missing) != expected_missing or unexpected:
        raise SystemExit(
            "R4 adapter mismatch missing={} expected={} unexpected={}".format(
                sorted(missing), sorted(expected_missing), sorted(unexpected),
            )
        )
    initial_hash = hash_adapter_state(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(batches)
    metadata = {
        "protocol": "active_counterfactual_distinction_r6",
        "base_checkpoint": os.path.realpath(args.init),
        "base_sha256": sha256_file(args.init),
        "base_step": base_checkpoint.get("step"),
        "pointer_adapter": os.path.realpath(args.pointer_adapter),
        "pointer_adapter_sha256": sha256_file(args.pointer_adapter),
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission_sha256": sha256_file(args.admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "seed": args.seed,
        "layer": args.layer,
        "hidden": args.hidden,
        "effect_hidden": args.effect_hidden,
        "batch_groups": args.batch_groups,
        "selected_groups": len(batches) * args.batch_groups,
        "selected_examples": len(batches) * args.batch_groups * len(VIEW_ORDER),
        "updates": total_steps,
        "probes_per_operation": args.probes_per_operation,
        "train_probe_indices": list(TRAIN_PROBE_INDICES),
        "heldout_probe_indices": list(HELDOUT_PROBE_INDICES),
        "effect_scale": EFFECT_SCALE,
        "learning_rate": args.lr,
        "warmup_updates": args.warmup,
        "gradient_clip": args.clip,
        "classification_weight": args.classification_weight,
        "mention_weight": args.mention_weight,
        "basis_weight": args.basis_weight,
        "effect_weight": args.effect_weight,
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_hash,
        "inference_inputs": (
            "cached token states, formatting-derived spans, dynamic slots, and selected state/query "
            "probe only; no structured operation kind, role, or value"
        ),
        "claim_boundary": (
            "Selected-probe scalar effect learning for a future active-versus-random latent inquiry "
            "test; training loss and oracle mechanics are not reasoning evidence."
        ),
    }
    print(json.dumps({"future_effect_compiler": metadata}, sort_keys=True), flush=True)

    compiler.train()
    started = time.time()
    for step, batch in enumerate(batches):
        examples = []
        for group_index in batch:
            views = groups[group_index][1]
            examples.extend(views[key] for key in VIEW_ORDER)
        ids = pad_ids(examples, "cuda")
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(step, total_steps, args.warmup)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, identity = compiler.encode(ids)
            results = classify_batch(compiler, hidden, identity, examples)
            op_kind_logits, op_role_logits, query_kind_logits, query_role_logits = flatten_outputs(results)
            op_kind_targets, op_role_targets, query_kind_targets, query_role_targets = flatten_targets(
                examples, "cuda",
            )
            classification = (
                F.cross_entropy(op_kind_logits.float(), op_kind_targets)
                + masked_cross_entropy(op_role_logits, op_role_targets)
                + F.cross_entropy(query_kind_logits.float(), query_kind_targets)
                + masked_cross_entropy(query_role_logits, query_role_targets)
            )
            mentions = mention_loss(results, examples)
            basis = compiler.basis_loss()
            effects = counterfactual_effect_loss(
                compiler, results, examples, step, args.probes_per_operation,
            )
            loss = (
                args.classification_weight * classification
                + args.mention_weight * mentions
                + args.basis_weight * basis
                + args.effect_weight * effects
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite future-effect loss at step {}".format(step))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite future-effect gradient at step {}".format(step))
        optimizer.step()
        if step % args.log_every == 0:
            role_valid = op_role_targets.ne(IGNORE_ROLE)
            print(
                "[effect-r6] step={}/{} loss={:.4f} cls={:.4f} mention={:.4f} basis={:.4f} "
                "effect={:.4f} op_kind={:.3f} op_role={:.3f} gnorm={:.3f}".format(
                    step, total_steps, loss.item(), classification.item(), mentions.item(),
                    basis.item(), effects.item(),
                    op_kind_logits.argmax(-1).eq(op_kind_targets).float().mean().item(),
                    op_role_logits.argmax(-1)[role_valid].eq(
                        op_role_targets[role_valid],
                    ).float().mean().item(),
                    float(grad_norm),
                ),
                flush=True,
            )

    output = os.path.join(args.out, "future_effect_adapter_ep1.pt")
    torch.save({
        "adapter_state": adapter_state(compiler),
        "future_effect_compiler": metadata,
        "step": "future_effect_adapter_ep1",
    }, output)
    print("[effect-r6] saved {}".format(output), flush=True)
    print("[effect-r6] done {} updates in {:.0f}s".format(total_steps, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
