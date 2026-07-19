#!/usr/bin/env python3
"""Evaluate learned unit generators behind the frozen S4 v5 parser."""

from __future__ import annotations

import argparse
import collections
import json
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s4_hard_island_soft_interface import decode_hard_island_soft_interface
from s4_set_identity_event_bus import roster_recovery_exact
from s5_learned_generator_executor import (
    GeneratorFactoredS3Executor,
    PERMUTATION_TO_ID,
    decode_v5_program,
    module_state_hash,
    stack_programs,
)
from self_delimiting_event_tape import (
    SelfDelimitingEventTapeParser,
    adapter_hash,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)


FIELDS = (
    "valid", "count_exact", "program_exact", "query_exact", "state_exact",
    "answer_correct", "initial_exact",
)


def summarize(records):
    result = {"rows": len(records)}
    for field in FIELDS:
        correct = sum(bool(record[field]) for record in records)
        result[field] = {"correct": correct, "accuracy": correct / max(1, len(records))}
    return result


def grouped(records, key):
    values = collections.defaultdict(list)
    for record in records:
        values[str(record[key])].append(record)
    return {name: summarize(group) for name, group in sorted(values.items())}


def make_record(example, decoded, initial_exact, final_state=None, answer=None):
    valid = bool(decoded.get("valid")) and final_state is not None and answer is not None
    return {
        "id": example.row_id,
        "depth": example.depth,
        "surface_type": example.surface_type,
        "valid": valid,
        "predicted_event_count": int(decoded.get("event_count", 0)),
        "failure_reason": decoded.get("failure_reason", "unknown"),
        "count_exact": int(decoded.get("event_count", 0)) == example.depth,
        "program_exact": bool(decoded.get("valid")) and decoded.get("program") == example.program,
        "query_exact": bool(decoded.get("valid")) and decoded.get("query") == example.query_target,
        "state_exact": valid and tuple(final_state) == example.final_state,
        "answer_correct": valid and int(answer) == example.answer_identity,
        "initial_exact": bool(initial_exact),
        "amount_two_events": sum(
            int(operation[2] == 2) for operation in decoded.get("program", ())
        ),
    }


def load_executor(path, expected_arm, device):
    bundle = torch.load(path, map_location="cpu")
    metadata = bundle.get("metadata", {})
    if metadata.get("schema") != "r12_s5_learned_unit_generator_checkpoint_v1":
        raise SystemExit("invalid S5 generator checkpoint")
    if metadata.get("arm") != expected_arm:
        raise SystemExit("S5 generator arm mismatch")
    if metadata.get("amount_two_training_examples") != 0:
        raise SystemExit("S5 amount-two supervision is forbidden")
    executor = GeneratorFactoredS3Executor(int(metadata["width"])).to(device).eval()
    executor.generator.load_state_dict(bundle["generator_state"], strict=True)
    if module_state_hash(executor.generator) != metadata["generator_state_sha256"]:
        raise SystemExit("S5 generator state mismatch")
    if module_state_hash(executor) != metadata["executor_state_sha256"]:
        raise SystemExit("S5 executor state mismatch")
    return executor, metadata


def execute_batch(executor, decoded, direction_rotation=False, reset_state=False):
    tensors = stack_programs(decoded, next(executor.parameters()).device, direction_rotation)
    outputs = executor(*tensors, reset_state=reset_state)
    states = outputs["assignment"].argmax(-1).cpu().tolist()
    answers = outputs["answer_probabilities"].argmax(-1).cpu().tolist()
    return states, answers


def exhaustive_closure(executor):
    device = next(executor.parameters()).device
    correct = {1: 0, 2: 0}
    total = {1: 0, 2: 0}
    with torch.inference_mode():
        for state in sorted(PERMUTATION_TO_ID):
            assignment = torch.zeros((1, 3, 3), device=device)
            for position, identity in enumerate(state):
                assignment[0, position, identity] = 1
            for identity in range(3):
                for direction in range(2):
                    for amount in (1, 2):
                        predicted = assignment.clone()
                        for _ in range(amount):
                            predicted, _, _, _ = executor.primitive(
                                predicted,
                                torch.tensor([identity], device=device),
                                torch.tensor([direction], device=device),
                            )
                        expected = list(state)
                        source = expected.index(identity)
                        destination = (
                            max(0, source - amount)
                            if direction == 0 else min(2, source + amount)
                        )
                        expected.insert(destination, expected.pop(source))
                        total[amount] += 1
                        correct[amount] += int(
                            tuple(predicted.argmax(-1)[0].cpu().tolist()) == tuple(expected)
                        )
    return {
        "amount_one": {"correct": correct[1], "total": total[1],
                       "accuracy": correct[1] / total[1]},
        "amount_two_unseen": {"correct": correct[2], "total": total[2],
                              "accuracy": correct[2] / total[2]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--parser", required=True)
    parser.add_argument("--promoted", required=True)
    parser.add_argument("--treatment-generator", required=True)
    parser.add_argument("--shuffled-generator", required=True)
    parser.add_argument("--fit-report", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--board-role", choices=("development", "confirmation"), default="development",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S5 evaluation requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing S5 evaluation")
    report = json.load(open(args.report))
    expected_schema = (
        "r12_s4_hard_island_fresh_development_report_v1"
        if args.board_role == "development"
        else "r12_s5_learned_generator_confirmation_report_v1"
    )
    if report.get("schema") != expected_schema:
        raise SystemExit("invalid S5 {} report".format(args.board_role))
    if not report.get("all_gates_pass") or report.get("confirmation_access") != 0:
        raise SystemExit("S5 {} board is not admitted".format(args.board_role))
    artifact_key = "development" if args.board_role == "development" else "confirmation"
    if report["artifacts"][artifact_key]["sha256"] != sha256_file(args.data):
        raise SystemExit("S5 report does not bind {} data".format(args.board_role))
    fit_report = json.load(open(args.fit_report))
    if not fit_report.get("all_fit_gates_pass"):
        raise SystemExit("S5 generator fit gates failed")
    if fit_report["artifacts"]["treatment"]["sha256"] != sha256_file(
        args.treatment_generator
    ):
        raise SystemExit("S5 fit report does not bind treatment")
    if fit_report["artifacts"]["shuffled"]["sha256"] != sha256_file(
        args.shuffled_generator
    ):
        raise SystemExit("S5 fit report does not bind shuffled control")
    promoted = json.load(open(args.promoted))
    if promoted.get("decision") != "promote_confirmed_s4_v5_bounded_reasoning_baseline":
        raise SystemExit("invalid promoted S4 v5 receipt")
    if promoted.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S5 base does not match promoted receipt")
    if promoted.get("parser_sha256") != sha256_file(args.parser):
        raise SystemExit("S5 parser does not match promoted receipt")

    parser_bundle = torch.load(args.parser, map_location="cpu")
    metadata = parser_bundle.get("parser", {})
    if metadata.get("protocol") != "r12_s4_self_delimiting_event_parser_treatment_v1":
        raise SystemExit("S5 requires the frozen S4 v1 treatment parser")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, "s4_event_tape_development", cfg.seq_len)
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    frozen = SelfDelimitingEventTapeParser(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        encoder_layers=int(metadata["encoder_layers"]),
        ff=int(metadata["ff"]),
    ).to("cuda").eval()
    missing, unexpected = frozen.load_state_dict(parser_bundle["adapter_state"], strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("S5 frozen adapter mismatch")
    if adapter_hash(frozen) != metadata["final_adapter_sha256"]:
        raise SystemExit("S5 frozen adapter hash mismatch")
    treatment, treatment_metadata = load_executor(
        args.treatment_generator, "semantic_unit_generators", "cuda",
    )
    shuffled, shuffled_metadata = load_executor(
        args.shuffled_generator, "fixed_deranged_unit_generators", "cuda",
    )

    arm_records = {name: [] for name in (
        "host_exact", "learned", "shuffled_law", "direction_rotated", "state_reset",
    )}
    parser_parity = 0
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            selected, ids, valid, _ = pad_batch(examples, indices, "cuda")
            outputs = frozen(ids, valid)
            decoded = []
            legacy = []
            initial = []
            for row, example in enumerate(selected):
                initial.append(roster_recovery_exact(
                    example, outputs, row, valid[row], cfg.vocab_size,
                ))
                decoded.append(decode_v5_program(
                    frozen, example, outputs, row, valid[row],
                    parser_bundle["kind_lexicon"],
                ))
                legacy.append(decode_hard_island_soft_interface(
                    frozen, example, outputs, row, valid[row],
                    parser_bundle["kind_lexicon"],
                ))
            for current, old in zip(decoded, legacy):
                parser_parity += int(
                    current.get("valid") == old.get("valid")
                    and current.get("event_count") == old.get("event_count")
                    and current.get("program") == old.get("program")
                    and current.get("query") == old.get("query")
                )
            valid_by_depth = collections.defaultdict(list)
            for local, current in enumerate(decoded):
                if current.get("valid"):
                    valid_by_depth[len(current["program"])].append(local)
            results = {name: {} for name in arm_records if name != "host_exact"}
            for _, local_indices in sorted(valid_by_depth.items()):
                selected_decoded = [decoded[index] for index in local_indices]
                for name, executor, rotate, reset in (
                    ("learned", treatment, False, False),
                    ("shuffled_law", shuffled, False, False),
                    ("direction_rotated", treatment, True, False),
                    ("state_reset", treatment, False, True),
                ):
                    states, answers = execute_batch(executor, selected_decoded, rotate, reset)
                    for index, state, answer in zip(local_indices, states, answers):
                        results[name][index] = (state, answer)
            for local, example in enumerate(selected):
                old = legacy[local]
                arm_records["host_exact"].append(make_record(
                    example,
                    decoded[local],
                    initial[local],
                    old.get("final_state"),
                    old.get("answer_identity"),
                ))
                for name in results:
                    state, answer = results[name].get(local, (None, None))
                    arm_records[name].append(make_record(
                        example, decoded[local], initial[local], state, answer,
                    ))

    result = {
        "schema": "r12_s5_learned_generator_executor_eval_v1",
        "base_sha256": sha256_file(args.base),
        "parser_sha256": sha256_file(args.parser),
        "parser_adapter_sha256": metadata["final_adapter_sha256"],
        "promoted_receipt_sha256": sha256_file(args.promoted),
        "treatment_generator_sha256": sha256_file(args.treatment_generator),
        "shuffled_generator_sha256": sha256_file(args.shuffled_generator),
        "fit_report_sha256": sha256_file(args.fit_report),
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "parser_parity": {"correct": parser_parity, "total": len(examples),
                          "accuracy": parser_parity / len(examples)},
        "closure": {
            "treatment": exhaustive_closure(treatment),
            "shuffled": exhaustive_closure(shuffled),
        },
        "arms": {
            name: {
                "overall": summarize(records),
                "by_depth": grouped(records, "depth"),
                "by_surface": grouped(records, "surface_type"),
                "amount_two_rows": summarize([
                    record for record in records if record["amount_two_events"] > 0
                ]),
            }
            for name, records in arm_records.items()
        },
        "generator_parameters": int(treatment_metadata["generator_parameters"]),
        "parameter_count": int(treatment_metadata["total_parameters"]),
        "training_amount_two_examples": int(
            treatment_metadata["amount_two_training_examples"]
        ),
        "training_recurrent_examples": int(
            treatment_metadata["recurrent_training_examples"]
        ),
        "board_role": args.board_role,
        "development_access": int(args.board_role == "development"),
        "confirmation_access": int(args.board_role == "confirmation"),
        "records": arm_records,
        "claim_boundary": (
            "Fresh-development learned finite generator composition behind frozen S4 v5. "
            "Known operations and structural host recurrence remain; no open-language, "
            "unseen-operation, planning, autonomous-halt, or broad native-reasoning claim."
        ),
    }
    os.makedirs(os.path.dirname(os.path.realpath(args.out)), exist_ok=True)
    with open(args.out, "w") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "out": os.path.realpath(args.out),
        "parser_parity": result["parser_parity"],
        "closure": result["closure"],
        "arms": {name: values["overall"] for name, values in result["arms"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
