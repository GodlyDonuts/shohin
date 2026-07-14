#!/usr/bin/env python3
"""Train matched role-factorized microcode compilers with optional causal constraints."""

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

from categorical_microcode import compile_example, sha256_file
from model import GPT, GPTConfig
from role_equivariant_microcode import (
    IGNORE_ROLE,
    RoleEquivariantMicrocodeCompiler,
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
                raise ValueError("invalid group metadata at row {}".format(line_number))
            group = groups.setdefault(group_id, {})
            if key in group:
                raise ValueError("duplicate group view at row {}".format(line_number))
            example = compile_example(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError("overlength role-equivariant row {}".format(line_number))
            group[key] = example
    ordered = []
    for group_id in sorted(groups):
        if set(groups[group_id]) != set(VIEW_ORDER):
            raise ValueError("incomplete role-equivariant group {}".format(group_id))
        ordered.append((group_id, groups[group_id]))
    if not ordered:
        raise ValueError("no role-equivariant groups")
    return ordered


def make_batches(groups, batch_groups, max_groups, seed):
    indices = list(range(len(groups)))
    random.Random(seed).shuffle(indices)
    if max_groups > 0:
        indices = indices[:max_groups]
    usable = len(indices) // batch_groups * batch_groups
    return [indices[offset:offset + batch_groups] for offset in range(0, usable, batch_groups)]


def pad_ids(examples, device):
    length = max(len(example.ids) for example in examples)
    return torch.tensor(
        [list(example.ids) + [0] * (length - len(example.ids)) for example in examples],
        dtype=torch.long, device=device,
    )


def flatten_operations(examples, device):
    batch_indices, positions, targets, slices = [], [], [], []
    cursor = 0
    for local, example in enumerate(examples):
        count = len(example.operation_positions)
        batch_indices.extend([local] * count)
        positions.extend(example.operation_positions)
        targets.extend(example.operation_targets)
        slices.append(slice(cursor, cursor + count))
        cursor += count
    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        slices,
    )


def lr_scale(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def normalized_distance(left, right):
    return (F.normalize(left.float(), dim=-1) - F.normalize(right.float(), dim=-1)).pow(2).sum(-1).mean()


def masked_cross_entropy(logits, targets):
    valid = targets.ne(IGNORE_ROLE)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), targets[valid])


def group_constraints(
    op_kind_features, op_role_features, op_role_targets, op_slices,
    query_kind_features, query_role_features, query_role_targets,
    batch_group_count,
):
    semantic_terms, permutation_terms = [], []
    views_per_group = len(VIEW_ORDER)
    key_index = {key: index for index, key in enumerate(VIEW_ORDER)}

    def example_index(group_local, key):
        return group_local * views_per_group + key_index[key]

    for group_local in range(batch_group_count):
        for permutation in (0, 1):
            anchor = example_index(group_local, ("anchor", permutation))
            for view in ("paraphrase_a", "paraphrase_b"):
                other = example_index(group_local, (view, permutation))
                semantic_terms.append(normalized_distance(
                    op_kind_features[op_slices[anchor]], op_kind_features[op_slices[other]],
                ))
                valid = op_role_targets[op_slices[anchor]].ne(IGNORE_ROLE)
                if bool(valid.any()):
                    semantic_terms.append(normalized_distance(
                        op_role_features[op_slices[anchor]][valid],
                        op_role_features[op_slices[other]][valid],
                    ))
                semantic_terms.append(normalized_distance(
                    query_kind_features[anchor:anchor + 1], query_kind_features[other:other + 1],
                ))
                if int(query_role_targets[anchor]) != IGNORE_ROLE:
                    semantic_terms.append(normalized_distance(
                        query_role_features[anchor:anchor + 1],
                        query_role_features[other:other + 1],
                    ))

        for view in ("anchor", "paraphrase_a", "paraphrase_b"):
            original = example_index(group_local, (view, 0))
            permuted = example_index(group_local, (view, 1))
            original_slice, permuted_slice = op_slices[original], op_slices[permuted]
            permutation_terms.append(normalized_distance(
                op_kind_features[original_slice], op_kind_features[permuted_slice],
            ))
            valid = op_role_targets[original_slice].ne(IGNORE_ROLE)
            if bool(valid.any()):
                permutation_terms.append(normalized_distance(
                    op_role_features[original_slice][valid],
                    -op_role_features[permuted_slice][valid],
                ))
            permutation_terms.append(normalized_distance(
                query_kind_features[original:original + 1], query_kind_features[permuted:permuted + 1],
            ))
            if int(query_role_targets[original]) != IGNORE_ROLE:
                permutation_terms.append(normalized_distance(
                    query_role_features[original:original + 1],
                    -query_role_features[permuted:permuted + 1],
                ))
    semantic = torch.stack(semantic_terms).mean()
    permutation = torch.stack(permutation_terms).mean()
    return semantic, permutation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=48000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--basis-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--permutation-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if (
        args.batch_groups <= 0 or args.max_groups <= 0
        or args.semantic_weight < 0 or args.permutation_weight < 0
    ):
        raise SystemExit("invalid group batch, group limit, or constraint weight")
    if not torch.cuda.is_available():
        raise SystemExit("role-equivariant microcode training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    admission = json.load(open(args.admission))
    data_sha256 = sha256_file(args.data)
    if not admission.get("all_checks_pass") or admission.get("train_sha256") != data_sha256:
        raise SystemExit("role-equivariant admission did not bind training data")
    if admission.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("role-equivariant admission did not bind tokenizer")
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    groups = load_groups(args.data, tokenizer, cfg.seq_len)
    batches = make_batches(groups, args.batch_groups, args.max_groups, args.seed)
    if not batches:
        raise SystemExit("selected zero group batches")
    total_steps = len(batches)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    compiler = RoleEquivariantMicrocodeCompiler(model, layer=args.layer, hidden=args.hidden).to("cuda")
    initial_hash = hash_adapter_state(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    metadata = {
        "protocol": "causal_microcode_role_equivariance_v3",
        "base_checkpoint": os.path.realpath(args.init),
        "base_sha256": sha256_file(args.init),
        "base_step": checkpoint.get("step"),
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission": os.path.realpath(args.admission),
        "admission_sha256": sha256_file(args.admission),
        "admission_eval_sha256": admission["eval_sha256"],
        "seed": args.seed,
        "layer": args.layer,
        "hidden": args.hidden,
        "batch_groups": args.batch_groups,
        "selected_groups": len(batches) * args.batch_groups,
        "selected_examples": len(batches) * args.batch_groups * len(VIEW_ORDER),
        "updates": total_steps,
        "semantic_weight": args.semantic_weight,
        "permutation_weight": args.permutation_weight,
        "learning_rate": args.lr,
        "warmup_updates": args.warmup,
        "gradient_clip": args.clip,
        "basis_weight": args.basis_weight,
        "role_factor_contract": "signed-z2-feature-v1",
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_hash,
        "view_contract": list([view, permutation] for view, permutation in VIEW_ORDER),
        "padding_contract": "right-padding only; causal read positions precede padding",
        "claim_boundary": (
            "Matched test of anchor replay, semantic-view representation alignment, and exact "
            "register-role equivariance in a supplied two-register execution substrate."
        ),
    }
    print(json.dumps({"role_equivariant_microcode": metadata, "available_groups": len(groups)}, sort_keys=True), flush=True)

    compiler.train()
    started = time.time()
    for step, batch in enumerate(batches):
        examples = []
        for group_index in batch:
            views = groups[group_index][1]
            examples.extend(views[key] for key in VIEW_ORDER)
        ids = pad_ids(examples, "cuda")
        op_batch, op_positions, op_targets, op_slices = flatten_operations(examples, "cuda")
        query_positions = torch.tensor([example.query_position for example in examples], dtype=torch.long, device="cuda")
        query_targets = torch.tensor([example.query_target for example in examples], dtype=torch.long, device="cuda")
        op_kind_targets, op_role_targets = factored_operation_targets(op_targets.tolist(), "cuda")
        query_kind_targets, query_role_targets = factored_query_targets(query_targets.tolist(), "cuda")
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(step, total_steps, args.warmup)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = compiler.encode(ids)
            op_features = compiler.position_features(hidden, op_batch, op_positions)
            query_features = compiler.position_features(
                hidden, torch.arange(len(examples), device="cuda"), query_positions,
            )
            op_kind_features, op_role_features = compiler.operation_factor_features(op_features)
            query_kind_features, query_role_features = compiler.query_factor_features(query_features)
            op_kind_logits, op_role_logits = compiler.operation_factor_logits(
                op_kind_features, op_role_features,
            )
            query_kind_logits, query_role_logits = compiler.query_factor_logits(
                query_kind_features, query_role_features,
            )
            op_kind_loss = F.cross_entropy(op_kind_logits.float(), op_kind_targets)
            op_role_loss = masked_cross_entropy(op_role_logits, op_role_targets)
            query_kind_loss = F.cross_entropy(query_kind_logits.float(), query_kind_targets)
            query_role_loss = masked_cross_entropy(query_role_logits, query_role_targets)
            semantic_loss, permutation_loss = group_constraints(
                op_kind_features, op_role_features, op_role_targets, op_slices,
                query_kind_features, query_role_features, query_role_targets,
                len(batch),
            )
            basis_loss = compiler.basis_loss()
            classification_loss = op_kind_loss + op_role_loss + query_kind_loss + query_role_loss
            loss = (
                classification_loss + args.basis_weight * basis_loss
                + args.semantic_weight * semantic_loss
                + args.permutation_weight * permutation_loss
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite role-equivariant loss at step {}".format(step))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite role-equivariant gradient at step {}".format(step))
        optimizer.step()
        if step % args.log_every == 0:
            op_acc = op_kind_logits.argmax(-1).eq(op_kind_targets).float().mean().item()
            role_valid = op_role_targets.ne(IGNORE_ROLE)
            role_acc = op_role_logits.argmax(-1)[role_valid].eq(op_role_targets[role_valid]).float().mean().item()
            query_acc = query_kind_logits.argmax(-1).eq(query_kind_targets).float().mean().item()
            print(
                "[role-equiv] step={}/{} loss={:.4f} cls={:.4f} semantic={:.4f} perm={:.4f} "
                "basis={:.4f} op_kind={:.3f} op_role={:.3f} query_kind={:.3f} gnorm={:.3f}".format(
                    step, total_steps, loss.item(), classification_loss.item(), semantic_loss.item(),
                    permutation_loss.item(), basis_loss.item(), op_acc, role_acc, query_acc,
                    float(grad_norm),
                ), flush=True,
            )

    output = os.path.join(args.out, "microcode_adapter_ep1.pt")
    torch.save({
        "adapter_state": adapter_state(compiler),
        "categorical_microcode": metadata,
        "step": "microcode_adapter_ep1",
    }, output)
    print("[role-equiv] saved {}".format(output), flush=True)
    print("[role-equiv] done {} updates in {:.0f}s".format(total_steps, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
