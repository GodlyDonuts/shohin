#!/usr/bin/env python3
"""Evaluate active counterfactual distinction on one frozen R6 effect head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import OPCODES, execute_program, sha256_file
from eval_referential_slot_microcode import all_batches, load_examples
from future_distinction_cell import (
    hypothesis_effect_codes,
    legal_operator_hypotheses,
    select_discriminating_probe,
)
from future_effect_algebra import redundant_probe_bank
from future_effect_compiler import ProbeConditionedEffectCompiler
from model import GPT, GPTConfig
from train_future_effect_compiler import EFFECT_SCALE, HELDOUT_PROBE_INDICES, TRAIN_PROBE_INDICES


POLICIES = ("active", "random", "zero", "shuffled", "oracle")


def probe_pair(index, states, queries):
    index = int(index)
    state = states[index % states.shape[0]]
    query = queries[index // states.shape[0]]
    return state, query


def infer_hypothesis(
    compiler, operation_output, codes, states, queries, *, policy, steps,
    seed, target=None, plausible_count=64,
):
    """Infer one operator with an equal-call active/random/control policy."""
    if policy not in POLICIES:
        raise ValueError("unknown policy {}".format(policy))
    if policy == "oracle" and target is None:
        raise ValueError("oracle policy requires a target index")
    scores = torch.zeros(codes.shape[0], dtype=torch.float32, device=codes.device)
    observed = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_order = torch.randperm(codes.shape[1], generator=generator).tolist()
    trace = []
    for latent_step in range(int(steps)):
        if policy == "random":
            probe = random_order[latent_step]
        else:
            if latent_step == 0:
                plausible = tuple(range(codes.shape[0]))
            else:
                count = min(int(plausible_count), codes.shape[0])
                plausible = tuple(torch.topk(scores, count, largest=False).indices.tolist())
            probe = select_discriminating_probe(codes, plausible, observed)
        if policy == "oracle":
            effect = codes[int(target), probe]
            asked_probe = probe
        elif policy == "zero":
            effect = torch.zeros((), dtype=codes.dtype, device=codes.device)
            asked_probe = probe
        else:
            asked_probe = (probe + 17) % codes.shape[1] if policy == "shuffled" else probe
            state, query = probe_pair(asked_probe, states, queries)
            effect = compiler.predict_effect(operation_output, state, query) * EFFECT_SCALE
        scores = scores + (codes[:, probe] - effect.float()).square()
        observed.append(probe)
        top = int(scores.argmin().item())
        trace.append({
            "latent_step": latent_step,
            "selected_probe": int(probe),
            "asked_probe": int(asked_probe),
            "predicted_effect": float(effect.item()),
            "top_hypothesis": top,
            "top_score": float(scores[top].item()),
        })
    return int(scores.argmin().item()), trace


def safe_execute(compiled, hypotheses, selected, query, transition_logits):
    try:
        opcodes = [OPCODES.index(hypotheses[index].opcode) for index in selected]
        values = [hypotheses[index].value for index in selected]
        answer = execute_program(
            compiled.initial_values,
            opcodes,
            values,
            int(query),
            transition_logits,
        )
        return answer, opcodes, values
    except (IndexError, ValueError):
        return None, [], []


def summarize(records):
    regimes = ["all"] + sorted({record["regime"] for record in records})
    output = {}
    for regime in regimes:
        selected = records if regime == "all" else [
            record for record in records if record["regime"] == regime
        ]
        policies = {}
        operation_total = sum(len(record["target_hypotheses"]) for record in selected)
        for policy in POLICIES:
            policies[policy] = {
                "cases": len(selected),
                "answers_correct": sum(record["policies"][policy]["answer_correct"] for record in selected),
                "exact_programs": sum(record["policies"][policy]["exact_program"] for record in selected),
                "operations_correct": sum(
                    sum(record["policies"][policy]["operation_correct"])
                    for record in selected
                ),
                "operations": operation_total,
            }
        output[regime] = {
            "cases": len(selected),
            "policies": policies,
            "train_probe_mse": sum(record["train_probe_squared_error"] for record in selected)
            / sum(record["probe_effect_count"] for record in selected),
            "heldout_probe_mse": sum(record["heldout_probe_squared_error"] for record in selected)
            / sum(record["heldout_effect_count"] for record in selected),
            "query_correct": sum(record["query_correct"] for record in selected),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--evaluation-label-admission")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-steps", type=int, default=3)
    parser.add_argument("--plausible-count", type=int, default=64)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("future-effect evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    if args.latent_steps <= 0 or args.latent_steps > 16:
        raise SystemExit("invalid latent-step budget")

    adapter_checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = adapter_checkpoint.get("future_effect_compiler", {})
    if metadata.get("protocol") != "active_counterfactual_distinction_r6":
        raise SystemExit("invalid R6 adapter metadata")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("R6 adapter does not bind supplied base")
    if metadata.get("train_probe_indices") != list(TRAIN_PROBE_INDICES):
        raise SystemExit("R6 adapter train-probe contract changed")
    if metadata.get("heldout_probe_indices") != list(HELDOUT_PROBE_INDICES):
        raise SystemExit("R6 adapter heldout-probe contract changed")

    admission = json.load(open(args.admission))
    adapter_labels = json.load(open(args.label_admission))
    evaluation_label_path = args.evaluation_label_admission or args.label_admission
    evaluation_labels = json.load(open(evaluation_label_path))
    data_sha256 = sha256_file(args.data)
    if not admission.get("all_checks_pass") or admission.get("eval_sha256") != data_sha256:
        raise SystemExit("structural admission does not bind R6 evaluation data")
    if admission.get("train_sha256") != metadata.get("data_sha256"):
        raise SystemExit("structural admission does not bind R6 train data")
    if not adapter_labels.get("all_checks_pass") or not evaluation_labels.get("all_checks_pass"):
        raise SystemExit("R6 mention-label admission failed")
    if adapter_labels["datasets"]["train"].get("sha256") != metadata.get("data_sha256"):
        raise SystemExit("adapter labels do not bind R6 train data")
    if metadata.get("label_admission_sha256") != sha256_file(args.label_admission):
        raise SystemExit("R6 adapter does not bind label admission")
    if evaluation_labels["datasets"]["eval"].get("sha256") != data_sha256:
        raise SystemExit("evaluation labels do not bind R6 eval data")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    base_checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**base_checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, cfg.seq_len)
    batches = all_batches(examples, args.batch_size)
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_checkpoint["model"])
    compiler = ProbeConditionedEffectCompiler(
        model,
        layer=int(metadata["layer"]),
        hidden=int(metadata["hidden"]),
        effect_hidden=int(metadata["effect_hidden"]),
    ).to("cuda").eval()
    missing, unexpected = compiler.load_state_dict(adapter_checkpoint["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("R6 adapter mismatch missing={} unexpected={}".format(missing, unexpected))

    hypotheses = legal_operator_hypotheses(range(1, args.max_value + 1), dtype=torch.float32)
    states, queries = redundant_probe_bank(dtype=torch.float32, device="cuda")
    codes = hypothesis_effect_codes(hypotheses, states, queries).to("cuda")
    hypothesis_index = {
        (hypothesis.opcode, hypothesis.value): index
        for index, hypothesis in enumerate(hypotheses)
    }
    train_probe_index = torch.tensor(TRAIN_PROBE_INDICES, dtype=torch.long, device="cuda")
    heldout_probe_index = torch.tensor(HELDOUT_PROBE_INDICES, dtype=torch.long, device="cuda")

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            ids = torch.tensor(
                [examples[index].compiled.ids for index in indices],
                dtype=torch.long,
                device="cuda",
            )
            hidden, identity = compiler.encode(ids)
            for local, index in enumerate(indices):
                example = examples[index]
                result = compiler.classify_text(
                    hidden[local], identity[local], example.intro_positions,
                    example.operation_spans, example.query_span,
                )
                query = int(compiler.compose_query_logits(
                    result["query"]["kind_logits"].unsqueeze(0),
                    result["query"]["role_logits"].unsqueeze(0),
                ).argmax().item())
                targets = []
                train_error = 0.0
                heldout_error = 0.0
                for operation_number, output in enumerate(result["operations"]):
                    opcode = OPCODES[example.compiled.operation_targets[operation_number]]
                    value = int(example.compiled.operation_values[operation_number])
                    target = hypothesis_index[(opcode, value)]
                    targets.append(target)
                    predicted_bank = compiler.predict_effect_bank(output, states, queries).reshape(-1) * EFFECT_SCALE
                    squared = (predicted_bank - codes[target]).square()
                    train_error += float(squared.index_select(0, train_probe_index).sum().item())
                    heldout_error += float(squared.index_select(0, heldout_probe_index).sum().item())

                policy_outputs = {}
                for policy in POLICIES:
                    selected_hypotheses = []
                    traces = []
                    for operation_number, output in enumerate(result["operations"]):
                        selected, trace = infer_hypothesis(
                            compiler,
                            output,
                            codes,
                            states,
                            queries,
                            policy=policy,
                            steps=args.latent_steps,
                            seed=args.seed + index * 97 + operation_number,
                            target=targets[operation_number],
                            plausible_count=args.plausible_count,
                        )
                        selected_hypotheses.append(selected)
                        traces.append(trace)
                    answer, opcodes, values = safe_execute(
                        example.compiled,
                        hypotheses,
                        selected_hypotheses,
                        query,
                        compiler.transition_logits,
                    )
                    operation_correct = [
                        predicted == target
                        for predicted, target in zip(selected_hypotheses, targets)
                    ]
                    exact_program = (
                        all(operation_correct)
                        and query == example.compiled.query_target
                    )
                    policy_outputs[policy] = {
                        "answer": answer,
                        "answer_correct": answer == example.compiled.answer,
                        "selected_hypotheses": selected_hypotheses,
                        "operation_correct": operation_correct,
                        "exact_program": exact_program,
                        "opcodes": opcodes,
                        "values": values,
                        "traces": traces,
                    }
                records.append({
                    "index": index,
                    "reference": example.compiled.reference,
                    "regime": example.compiled.regime,
                    "answer": example.compiled.answer,
                    "target_hypotheses": targets,
                    "query_target": example.compiled.query_target,
                    "query_prediction": query,
                    "query_correct": query == example.compiled.query_target,
                    "train_probe_squared_error": train_error,
                    "heldout_probe_squared_error": heldout_error,
                    "probe_effect_count": len(targets) * len(TRAIN_PROBE_INDICES),
                    "heldout_effect_count": len(targets) * len(HELDOUT_PROBE_INDICES),
                    "policies": policy_outputs,
                })
            if batch_number % 10 == 0:
                print("[effect-r6-eval] batches={}/{} cases={}".format(
                    batch_number, len(batches), len(records),
                ), flush=True)

    report = {
        "protocol": "active_counterfactual_distinction_eval_r6",
        "base": str(Path(args.base).resolve()),
        "base_sha256": sha256_file(args.base),
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_sha256": sha256_file(args.adapter),
        "data": str(Path(args.data).resolve()),
        "data_sha256": data_sha256,
        "admission_sha256": sha256_file(args.admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "evaluation_label_admission_sha256": sha256_file(evaluation_label_path),
        "latent_steps": args.latent_steps,
        "plausible_count": args.plausible_count,
        "max_value": args.max_value,
        "hypotheses": len(hypotheses),
        "policies": list(POLICIES),
        "summary": summarize(records),
        "records": records,
        "claim_boundary": (
            "Active versus random uses the byte-identical effect head and equal scalar-probe calls. "
            "Oracle is an upper bound; zero/shuffled are causal controls. This narrow operator board "
            "cannot establish broad language reasoning."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True), flush=True)
    print("[effect-r6-eval] wrote {}".format(args.out), flush=True)


if __name__ == "__main__":
    main()
