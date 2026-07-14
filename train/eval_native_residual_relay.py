#!/usr/bin/env python3
"""Read-only causal evaluation for Native Residual Relay checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from native_residual_relay import encode_relay, generate_from_relay


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_answer(model, tokenizer, source, suffix, layer, eos_id, max_new):
    source_ids = torch.tensor([tokenizer.encode(source).ids], dtype=torch.long, device="cuda")
    suffix_ids = torch.tensor([tokenizer.encode(suffix).ids], dtype=torch.long, device="cuda")
    relay = encode_relay(model, source_ids, layer)
    tokens = generate_from_relay(model, relay, suffix_ids, layer, eos_id, max_new)
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False).strip(), relay


def decode_with_relay(model, tokenizer, relay, suffix, layer, eos_id, max_new):
    suffix_ids = torch.tensor([tokenizer.encode(suffix).ids], dtype=torch.long, device="cuda")
    tokens = generate_from_relay(model, relay, suffix_ids, layer, eos_id, max_new)
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False).strip()


def decode_direct(model, tokenizer, source, suffix, eos_id, max_new):
    """Full-source bypass for diagnosing a failed relay, never a positive gate."""
    prompt_ids = tokenizer.encode("{}\n{}".format(source, suffix)).ids
    context = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    generated = []
    for _ in range(max_new):
        logits, _ = model(context)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if int(next_id.item()) == int(eos_id) or context.shape[1] + 1 >= model.cfg.seq_len:
            break
        generated.append(int(next_id.item()))
        context = torch.cat((context, next_id), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=False).strip()


def score_result(result):
    """Score one full causal-control row without consulting a model."""
    if result["expected"] == result["counterfactual_expected"]:
        raise ValueError("counterfactual target must differ from the normal target")
    result["direct_correct"] = result["direct"] == result["expected"]
    result["normal_correct"] = result["normal"] == result["expected"]
    result["paraphrase_correct"] = result["paraphrase"] == result["expected"]
    result["counterfactual_correct"] = result["counterfactual"] == result["counterfactual_expected"]
    result["zero_recreates_normal"] = result["zero"] == result["expected"]
    result["shuffle_recreates_normal"] = result["shuffled"] == result["expected"]
    result["strict_causal"] = bool(
        result["normal_correct"]
        and result["paraphrase_correct"]
        and result["counterfactual_correct"]
        and not result["zero_recreates_normal"]
        and not result["shuffle_recreates_normal"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available() or Path(args.out).exists():
        raise SystemExit("CUDA required and output path must be fresh")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    metadata = checkpoint.get("native_residual_relay")
    if not isinstance(metadata, dict) or metadata.get("source_present_at_suffix") is not False or metadata.get("extra_trainable_parameters") != 0:
        raise SystemExit("checkpoint does not certify native source-free relay")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer EOS missing")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    rows = [row for row in rows if row.get("schema") == "native_residual_relay_v1" and row.get("split") == "heldout"][:args.max_examples]
    if not rows:
        raise SystemExit("no held-out NRR rows")
    results, relays = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            normal, relay = decode_answer(model, tokenizer, row["source"], row["suffix_prompt"], int(metadata["layer"]), eos_id, args.max_new)
            paraphrase, same_relay = decode_answer(model, tokenizer, row["paraphrase_source"], row["suffix_prompt"], int(metadata["layer"]), eos_id, args.max_new)
            counter, _ = decode_answer(model, tokenizer, row["counterfactual_source"], row["suffix_prompt"], int(metadata["layer"]), eos_id, args.max_new)
            direct = decode_direct(model, tokenizer, row["source"], row["suffix_prompt"], eos_id, args.max_new)
            zero = decode_with_relay(model, tokenizer, torch.zeros_like(relay), row["suffix_prompt"], int(metadata["layer"]), eos_id, args.max_new)
            relays.append(relay)
            results.append({"episode_id": row["episode_id"], "normal": normal, "paraphrase": paraphrase, "direct": direct,
                            "counterfactual": counter, "zero": zero, "expected": row["response"],
                            "counterfactual_expected": row["counterfactual_response"],
                            "same_cosine": float(torch.nn.functional.cosine_similarity(relay.float(), same_relay.float(), dim=-1).item())})
            print("[nrr-eval] {}/{}".format(index + 1, len(rows)), flush=True)
        for index, result in enumerate(results):
            donor = relays[(index + 1) % len(relays)]
            result["shuffled"] = decode_with_relay(model, tokenizer, donor, rows[index]["suffix_prompt"], int(metadata["layer"]), eos_id, args.max_new)
    for result in results:
        score_result(result)
    summary = {key: sum(bool(row[key]) for row in results) for key in ("direct_correct", "normal_correct", "paraphrase_correct", "counterfactual_correct", "zero_recreates_normal", "shuffle_recreates_normal", "strict_causal")}
    report = {"audit": "native_residual_relay_v1", "checkpoint": args.ckpt, "step": checkpoint.get("step"),
              "checkpoint_metadata": metadata, "data": args.data, "data_sha256": sha256_file(args.data),
              "rows": len(results), "summary": summary, "mean_same_relay_cosine": sum(row["same_cosine"] for row in results) / len(results),
              "results": results, "claim_boundary": "A strict pass only establishes source-free relay use on this synthetic counterfactual suite."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[nrr-eval] summary=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
