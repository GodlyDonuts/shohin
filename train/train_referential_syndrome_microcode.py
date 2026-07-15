#!/usr/bin/env python3
"""Train matched R9c text-to-syndrome microcode arms on causal effects."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from model import GPT, GPTConfig
from referential_slot_microcode import ReferentialSlotMicrocodeCompiler
from referential_syndrome_microcode import (
    ReferentialSyndromeBridge,
    directional_effect_losses,
)
from train_referential_slot_microcode import VIEW_ORDER, load_groups, pad_ids


ARM_CONFIGS = {
    "treatment": {"conditioning": "directional", "use_syndrome": True, "shuffle_goal": False},
    "static": {"conditioning": "static", "use_syndrome": False, "shuffle_goal": False},
    "no_syndrome": {"conditioning": "directional", "use_syndrome": False, "shuffle_goal": False},
    "shuffled_goal": {"conditioning": "directional", "use_syndrome": True, "shuffle_goal": True},
}


def depth_matched_batches(groups, batch_groups, max_groups, seed):
    indices = list(range(len(groups)))
    random.Random(seed).shuffle(indices)
    indices = indices[:int(max_groups)]
    buckets = collections.defaultdict(list)
    for index in indices:
        views = groups[index][1]
        depths = {len(example.compiled.operation_targets) for example in views.values()}
        if len(depths) != 1:
            raise ValueError("equivalence group contains multiple event depths")
        buckets[next(iter(depths))].append(index)
    rng = random.Random(seed + 1)
    batches = []
    dropped = 0
    for depth in sorted(buckets):
        selected = buckets[depth]
        usable = len(selected) // int(batch_groups) * int(batch_groups)
        dropped += len(selected) - usable
        batches.extend(
            selected[offset:offset + int(batch_groups)]
            for offset in range(0, usable, int(batch_groups))
        )
    rng.shuffle(batches)
    return batches, {"depth_buckets": len(buckets), "dropped_groups": dropped}


def lr_scale(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def microcode_state(bridge):
    return {
        name: value.detach().cpu()
        for name, value in bridge.microcode.state_dict().items()
    }


def hash_microcode_state(bridge):
    digest = hashlib.sha256()
    for name, tensor in sorted(microcode_state(bridge).items()):
        tensor = tensor.contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_pointer(base_path, adapter_path, device):
    base = torch.load(base_path, map_location="cpu")
    adapter = torch.load(adapter_path, map_location="cpu")
    metadata = adapter.get("categorical_microcode", {})
    if metadata.get("protocol") != "causal_microcode_referential_slots_v4":
        raise SystemExit("pointer adapter has the wrong protocol")
    if metadata.get("role_mode") != "pointer":
        raise SystemExit("R9c requires the admitted R4 pointer adapter")
    if metadata.get("base_sha256") != sha256_file(base_path):
        raise SystemExit("pointer adapter does not bind the supplied base")
    cfg = GPTConfig(**base["cfg"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(base["model"])
    pointer = ReferentialSlotMicrocodeCompiler(
        model, layer=int(metadata["layer"]), hidden=int(metadata["hidden"]), role_mode="pointer",
    ).to(device).eval()
    missing, unexpected = pointer.load_state_dict(adapter["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("pointer mismatch missing={} unexpected={}".format(missing, unexpected))
    pointer.requires_grad_(False)
    return base, metadata, pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pointer-adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_CONFIGS), required=True)
    parser.add_argument("--memory-dim", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("referential syndrome training requires CUDA")
    if args.rounds < 2 or args.batch_groups <= 0 or args.max_groups <= 0:
        raise SystemExit("invalid recurrent or batch schedule")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    data_sha256 = sha256_file(args.data)
    admission = json.load(open(args.admission))
    label_admission = json.load(open(args.label_admission))
    if not admission.get("all_checks_pass") or admission.get("train_sha256") != data_sha256:
        raise SystemExit("structural admission does not bind R9c training data")
    if not label_admission.get("all_checks_pass"):
        raise SystemExit("pointer label admission failed")
    if label_admission["datasets"]["train"].get("sha256") != data_sha256:
        raise SystemExit("pointer label admission does not bind R9c data")
    if label_admission.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("pointer label admission does not bind tokenizer")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    base, pointer_metadata, pointer = load_pointer(args.base, args.pointer_adapter, "cuda")
    groups = load_groups(args.data, tokenizer, GPTConfig(**base["cfg"]).seq_len)
    batches, batch_report = depth_matched_batches(
        groups, args.batch_groups, args.max_groups, args.seed,
    )
    if not batches:
        raise SystemExit("selected zero R9c batches")
    bridge = ReferentialSyndromeBridge(
        pointer, pointer_hidden=int(pointer_metadata["hidden"]), memory_dim=args.memory_dim,
    ).to("cuda")
    trainable = list(bridge.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    arm = ARM_CONFIGS[args.arm]
    metadata = {
        "protocol": "referential_bidirectional_syndrome_microcode_r9c",
        "arm": args.arm,
        "arm_config": arm,
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "base_step": base.get("step"),
        "pointer_adapter": os.path.realpath(args.pointer_adapter),
        "pointer_adapter_sha256": sha256_file(args.pointer_adapter),
        "pointer_protocol": pointer_metadata.get("protocol"),
        "pointer_hidden": int(pointer_metadata["hidden"]),
        "pointer_parameters_trainable": 0,
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "admission_sha256": sha256_file(args.admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "seed": args.seed,
        "memory_dim": args.memory_dim,
        "rounds": args.rounds,
        "batch_groups": args.batch_groups,
        "selected_groups": len(batches) * args.batch_groups,
        "selected_examples": len(batches) * args.batch_groups * len(VIEW_ORDER),
        "updates": len(batches),
        "batch_report": batch_report,
        "learning_rate": args.lr,
        "warmup_updates": args.warmup,
        "gradient_clip": args.clip,
        "adapter_parameters": bridge.adapter_num_params(),
        "initial_adapter_sha256": hash_microcode_state(bridge),
        "supervision": (
            "forward actual-prefix effects plus backward future-goal pullbacks, agreement, endpoint, "
            "and entropy; no opcode cross-entropy"
        ),
        "inference_inputs": (
            "frozen text-derived R4 event features, lexical values and initial quantities, and a "
            "frozen text-derived soft query covector"
        ),
        "claim_boundary": (
            "Matched mechanism-development arm on already-used data. A fit cannot establish reasoning, "
            "fresh transfer, or context scaling without the frozen multi-arm evaluator."
        ),
    }
    print(json.dumps({"referential_syndrome_microcode": metadata}, sort_keys=True), flush=True)

    bridge.train()
    bridge.pointer_compiler.eval()
    started = time.time()
    for step, group_indices in enumerate(batches):
        examples = []
        for group_index in group_indices:
            views = groups[group_index][1]
            examples.extend(views[key] for key in VIEW_ORDER)
        ids = pad_ids(examples, "cuda")
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(step, len(batches), args.warmup)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = bridge.encode_examples(ids, examples)
            input_goals = encoded.query_goals
            if arm["shuffle_goal"]:
                input_goals = input_goals.roll(1, dims=0)
            run = bridge.microcode(
                encoded.event_features, encoded.values, encoded.initial_values, input_goals,
                rounds=args.rounds, conditioning=arm["conditioning"],
                use_syndrome=arm["use_syndrome"], adaptive=False,
            )
            losses = directional_effect_losses(
                run, encoded.values, encoded.initial_values, encoded.query_goals,
                encoded.operation_targets, encoded.answer_targets,
            )
            loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite R9c loss at step {}".format(step))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite R9c gradient at step {}".format(step))
        optimizer.step()
        if step % args.log_every == 0 or step + 1 == len(batches):
            forward_accuracy = run.forward_logits.argmax(-1).eq(encoded.operation_targets).float().mean()
            backward_accuracy = run.backward_logits.argmax(-1).eq(encoded.operation_targets).float().mean()
            agreement = run.forward_logits.argmax(-1).eq(run.backward_logits.argmax(-1)).float().mean()
            query_accuracy = encoded.query_logits.argmax(-1).eq(encoded.query_targets).float().mean()
            print(
                "[r9c] step={}/{} arm={} loss={:.5f} f_effect={:.5f} b_effect={:.5f} "
                "syndrome={:.5f} endpoint={:.5f} entropy={:.5f} f_op={:.3f} b_op={:.3f} "
                "agree={:.3f} query={:.3f} gnorm={:.3f}".format(
                    step, len(batches), args.arm, float(loss), float(losses["forward_effect"]),
                    float(losses["backward_effect"]), float(losses["agreement"]),
                    float(losses["endpoint"]), float(losses["entropy"]), float(forward_accuracy),
                    float(backward_accuracy), float(agreement), float(query_accuracy), float(grad_norm),
                ), flush=True,
            )

    output = os.path.join(args.out, "syndrome_adapter_ep1.pt")
    metadata["final_adapter_sha256"] = hash_microcode_state(bridge)
    torch.save({
        "microcode_state": microcode_state(bridge),
        "referential_syndrome_microcode": metadata,
        "step": "syndrome_adapter_ep1",
    }, output)
    print("[r9c] saved {}".format(output), flush=True)
    print("[r9c] done {} updates in {:.0f}s".format(len(batches), time.time() - started), flush=True)


if __name__ == "__main__":
    main()
