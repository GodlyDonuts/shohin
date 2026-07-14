#!/usr/bin/env python3
"""Train matched absolute-role and referential-pointer microcode compilers."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from model import GPT, GPTConfig
from referential_slot_microcode import (
    ReferentialSlotMicrocodeCompiler,
    attention_mass_loss,
    compile_referential_example,
)
from role_equivariant_microcode import (
    IGNORE_ROLE,
    factored_operation_targets,
    factored_query_targets,
)
from train_categorical_microcode import adapter_state, hash_adapter_state


VIEW_ORDER = (
    ("anchor", 0), ("paraphrase_a", 0), ("paraphrase_b", 0),
    ("anchor", 1), ("paraphrase_a", 1), ("paraphrase_b", 1),
)


def load_groups(path, tokenizer, seq_len):
    groups = {}
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = row.get("equivalence_id")
            key = (row.get("semantic_view"), row.get("register_permutation"))
            if not isinstance(group_id, str) or key not in VIEW_ORDER:
                raise ValueError("invalid referential group metadata at row {}".format(line_number))
            group = groups.setdefault(group_id, {})
            if key in group:
                raise ValueError("duplicate referential group view at row {}".format(line_number))
            example = compile_referential_example(row, tokenizer)
            if len(example.compiled.ids) > seq_len:
                raise ValueError("overlength referential row {}".format(line_number))
            group[key] = example
    ordered = []
    for group_id in sorted(groups):
        if set(groups[group_id]) != set(VIEW_ORDER):
            raise ValueError("incomplete referential group {}".format(group_id))
        ordered.append((group_id, groups[group_id]))
    if not ordered:
        raise ValueError("no referential groups")
    return ordered


def make_batches(groups, batch_groups, max_groups, seed):
    indices = list(range(len(groups)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_groups]
    usable = len(indices) // batch_groups * batch_groups
    return [indices[offset:offset + batch_groups] for offset in range(0, usable, batch_groups)]


def pad_ids(examples, device):
    length = max(len(example.compiled.ids) for example in examples)
    return torch.tensor([
        list(example.compiled.ids) + [0] * (length - len(example.compiled.ids))
        for example in examples
    ], dtype=torch.long, device=device)


def lr_scale(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def masked_cross_entropy(logits, targets):
    valid = targets.ne(IGNORE_ROLE)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), targets[valid])


def classify_batch(compiler, hidden, identity, examples):
    results = []
    for local, example in enumerate(examples):
        results.append(compiler.classify_text(
            hidden[local], identity[local], example.intro_positions,
            example.operation_spans, example.query_span,
        ))
    return results


def flatten_outputs(results):
    operation_kind = torch.stack([
        output["kind_logits"] for result in results for output in result["operations"]
    ])
    operation_role = torch.stack([
        output["role_logits"] for result in results for output in result["operations"]
    ])
    query_kind = torch.stack([result["query"]["kind_logits"] for result in results])
    query_role = torch.stack([result["query"]["role_logits"] for result in results])
    return operation_kind, operation_role, query_kind, query_role


def flatten_targets(examples, device):
    operation_targets = [
        target for example in examples for target in example.compiled.operation_targets
    ]
    query_targets = [example.compiled.query_target for example in examples]
    op_kind, op_role = factored_operation_targets(operation_targets, device)
    query_kind, query_role = factored_query_targets(query_targets, device)
    return op_kind, op_role, query_kind, query_role


def mention_loss(results, examples):
    losses = []
    for result, example in zip(results, examples):
        for slot in range(2):
            losses.append(attention_mass_loss(
                result["intro_weights"][:, slot], result["intro_positions"],
                example.intro_slot_targets[slot],
            ))
        for output, targets in zip(result["operations"], example.operation_mention_targets):
            if targets:
                losses.append(attention_mass_loss(
                    output["target_weights"], output["positions"], targets,
                ))
        if example.query_mention_target:
            losses.append(attention_mass_loss(
                result["query"]["target_weights"], result["query"]["positions"],
                example.query_mention_target,
            ))
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--role-mode", choices=("absolute", "pointer"), required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=48000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--basis-weight", type=float, default=1.0)
    parser.add_argument("--mention-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_groups <= 0 or args.max_groups <= 0 or args.mention_weight < 0:
        raise SystemExit("invalid group batch, group limit, or mention weight")
    if not torch.cuda.is_available():
        raise SystemExit("referential slot training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    admission = json.load(open(args.admission))
    label_admission = json.load(open(args.label_admission))
    data_sha256 = sha256_file(args.data)
    if not admission.get("all_checks_pass") or admission.get("train_sha256") != data_sha256:
        raise SystemExit("structural admission did not bind referential data")
    if not label_admission.get("all_checks_pass"):
        raise SystemExit("mention-label admission failed")
    if label_admission["datasets"]["train"].get("sha256") != data_sha256:
        raise SystemExit("mention-label admission did not bind referential data")
    if label_admission.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("mention-label admission did not bind tokenizer")
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    groups = load_groups(args.data, tokenizer, cfg.seq_len)
    batches = make_batches(groups, args.batch_groups, args.max_groups, args.seed)
    if not batches:
        raise SystemExit("selected zero referential batches")
    total_steps = len(batches)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    compiler = ReferentialSlotMicrocodeCompiler(
        model, layer=args.layer, hidden=args.hidden, role_mode=args.role_mode,
    ).to("cuda")
    initial_hash = hash_adapter_state(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    metadata = {
        "protocol": "causal_microcode_referential_slots_v4",
        "base_checkpoint": os.path.realpath(args.init),
        "base_sha256": sha256_file(args.init),
        "base_step": checkpoint.get("step"),
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission": os.path.realpath(args.admission),
        "admission_sha256": sha256_file(args.admission),
        "label_admission": os.path.realpath(args.label_admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "seed": args.seed,
        "layer": args.layer,
        "hidden": args.hidden,
        "batch_groups": args.batch_groups,
        "selected_groups": len(batches) * args.batch_groups,
        "selected_examples": len(batches) * args.batch_groups * len(VIEW_ORDER),
        "updates": total_steps,
        "role_mode": args.role_mode,
        "learning_rate": args.lr,
        "warmup_updates": args.warmup,
        "gradient_clip": args.clip,
        "basis_weight": args.basis_weight,
        "mention_weight": args.mention_weight,
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_hash,
        "view_contract": [[view, permutation] for view, permutation in VIEW_ORDER],
        "inference_inputs": "token states plus formatting-derived intro/event/query spans only",
        "claim_boundary": (
            "Matched text-only test of dynamic entity-slot pointer binding versus an equal-parameter "
            "absolute-role head in the supplied two-register execution substrate."
        ),
    }
    print(json.dumps({"referential_slot_microcode": metadata, "available_groups": len(groups)}, sort_keys=True), flush=True)

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
            classification_loss = (
                F.cross_entropy(op_kind_logits.float(), op_kind_targets)
                + masked_cross_entropy(op_role_logits, op_role_targets)
                + F.cross_entropy(query_kind_logits.float(), query_kind_targets)
                + masked_cross_entropy(query_role_logits, query_role_targets)
            )
            mentions = mention_loss(results, examples)
            basis = compiler.basis_loss()
            loss = classification_loss + args.mention_weight * mentions + args.basis_weight * basis
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite referential loss at step {}".format(step))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite referential gradient at step {}".format(step))
        optimizer.step()
        if step % args.log_every == 0:
            role_valid = op_role_targets.ne(IGNORE_ROLE)
            print(
                "[referential] step={}/{} mode={} loss={:.4f} cls={:.4f} mention={:.4f} "
                "basis={:.4f} op_kind={:.3f} op_role={:.3f} query_kind={:.3f} gnorm={:.3f}".format(
                    step, total_steps, args.role_mode, loss.item(), classification_loss.item(),
                    mentions.item(), basis.item(),
                    op_kind_logits.argmax(-1).eq(op_kind_targets).float().mean().item(),
                    op_role_logits.argmax(-1)[role_valid].eq(op_role_targets[role_valid]).float().mean().item(),
                    query_kind_logits.argmax(-1).eq(query_kind_targets).float().mean().item(),
                    float(grad_norm),
                ), flush=True,
            )

    output = os.path.join(args.out, "microcode_adapter_ep1.pt")
    torch.save({
        "adapter_state": adapter_state(compiler),
        "categorical_microcode": metadata,
        "step": "microcode_adapter_ep1",
    }, output)
    print("[referential] saved {}".format(output), flush=True)
    print("[referential] done {} updates in {:.0f}s".format(total_steps, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
