#!/usr/bin/env python3
"""Extract provenance-bound R10 categorical scores from frozen R9c no-syndrome."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from categorical_microcode import OPCODES, QUERIES, sha256_file


R9C_PROTOCOL = "referential_bidirectional_syndrome_microcode_r9c"
POINTER_PROTOCOL = "causal_microcode_referential_slots_v4"
STRUCTURAL_ADMISSION_AUDIT = "role_equivariant_microcode_v3"
LABEL_ADMISSION_AUDIT = "referential_slot_label_admission_v1"
NO_SYNDROME_CONFIG = {
    "conditioning": "directional",
    "use_syndrome": False,
    "shuffle_goal": False,
}


def categorical_probabilities(forward_logits, backward_logits, query_logits):
    """Return float32 probabilities for one fixed-replay batch."""
    if forward_logits.ndim != 3 or backward_logits.shape != forward_logits.shape:
        raise ValueError("directional logits must share [batch,events,categories] shape")
    if forward_logits.shape[-1] != len(OPCODES):
        raise ValueError("directional logits have the wrong categorical width")
    if query_logits.ndim != 2 or query_logits.shape != (
        forward_logits.shape[0], len(QUERIES),
    ):
        raise ValueError("query logits must have shape [batch,queries]")
    tensors = (forward_logits, backward_logits, query_logits)
    if not all(torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("all logits must be floating point")
    if not all(tensor.device == forward_logits.device for tensor in tensors):
        raise ValueError("all logits must share a device")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("all logits must be finite")

    forward = forward_logits.float()
    backward = backward_logits.float()
    query = query_logits.float()
    return {
        "joint": (0.5 * (forward + backward)).softmax(dim=-1),
        "forward": forward.softmax(dim=-1),
        "backward": backward.softmax(dim=-1),
        "query": query.softmax(dim=-1),
    }


def _require(condition, message):
    if not condition:
        raise SystemExit(message)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_code_identity(code_revision, expected_extractor_sha256, actual_extractor_sha256):
    _require(_is_git_revision(code_revision), "code revision must be a full lowercase git SHA")
    _require(_is_sha256(expected_extractor_sha256), "expected extractor hash must be SHA-256")
    _require(_is_sha256(actual_extractor_sha256), "actual extractor hash must be SHA-256")
    _require(
        expected_extractor_sha256 == actual_extractor_sha256,
        "executing extractor differs from the frozen extractor hash",
    )


def _load_manifest(path, description):
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    _require(isinstance(manifest, dict), "{} must be a JSON object".format(description))
    return manifest


def validate_hash_bindings(metadata, hashes, admission, label_admission):
    """Fail closed unless the adapter and both admissions form one hash chain."""
    for key in (
        "base_sha256",
        "pointer_adapter_sha256",
        "data_sha256",
        "tokenizer_sha256",
        "admission_sha256",
        "label_admission_sha256",
        "final_adapter_sha256",
    ):
        _require(_is_sha256(metadata.get(key)), "R9c metadata lacks a valid {}".format(key))
    _require(metadata.get("protocol") == R9C_PROTOCOL, "invalid R9c adapter protocol")
    _require(metadata.get("arm") == "no_syndrome", "R10 requires arm=no_syndrome")
    _require(
        metadata.get("arm_config") == NO_SYNDROME_CONFIG,
        "R9c no_syndrome metadata differs from the frozen runtime contract",
    )
    _require(metadata.get("pointer_protocol") == POINTER_PROTOCOL, "invalid R9c pointer protocol")
    _require(
        metadata.get("pointer_parameters_trainable") == 0,
        "R9c metadata does not freeze the pointer adapter",
    )
    rounds = metadata.get("rounds")
    _require(
        isinstance(rounds, int) and not isinstance(rounds, bool) and rounds > 0,
        "R9c metadata has an invalid fixed replay count",
    )
    _require(
        metadata.get("base_sha256") == hashes["base"],
        "R9c adapter does not bind the supplied base",
    )
    _require(
        metadata.get("pointer_adapter_sha256") == hashes["pointer_adapter"],
        "R9c adapter does not bind the supplied pointer",
    )
    _require(
        metadata.get("tokenizer_sha256") == hashes["tokenizer"],
        "R9c adapter does not bind the supplied tokenizer",
    )
    training_sha256 = metadata["data_sha256"]
    _require(
        admission.get("audit") == STRUCTURAL_ADMISSION_AUDIT,
        "invalid structural admission audit",
    )
    _require(admission.get("all_checks_pass") is True, "structural admission failed")
    _require(
        admission.get("eval_sha256") == hashes["data"],
        "structural admission does not bind the evaluation JSONL",
    )
    _require(
        admission.get("train_sha256") == training_sha256,
        "structural admission does not bind the R9c training data",
    )
    _require(
        admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "structural admission does not bind the tokenizer",
    )

    _require(
        label_admission.get("audit") == LABEL_ADMISSION_AUDIT,
        "invalid referential-label admission audit",
    )
    _require(label_admission.get("all_checks_pass") is True, "referential-label admission failed")
    datasets = label_admission.get("datasets")
    _require(isinstance(datasets, dict), "referential-label admission lacks datasets")
    evaluation = datasets.get("eval")
    training = datasets.get("train")
    _require(
        isinstance(evaluation, dict) and evaluation.get("all_checks_pass") is True,
        "referential evaluation labels were not admitted",
    )
    _require(
        isinstance(training, dict) and training.get("all_checks_pass") is True,
        "referential training labels were not admitted",
    )
    _require(
        evaluation.get("sha256") == hashes["data"],
        "referential-label admission does not bind the evaluation JSONL",
    )
    _require(
        training.get("sha256") == training_sha256,
        "referential-label admission does not bind the R9c training data",
    )
    _require(
        label_admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "referential-label admission does not bind the tokenizer",
    )
    _require(
        admission.get("eval_sha256") == evaluation.get("sha256")
        and admission.get("train_sha256") == training.get("sha256")
        and admission.get("tokenizer_sha256") == label_admission.get("tokenizer_sha256"),
        "structural and referential-label admissions do not describe one artifact set",
    )


def validate_pointer_metadata(pointer_metadata, r9c_metadata):
    _require(pointer_metadata.get("protocol") == POINTER_PROTOCOL, "invalid pointer adapter protocol")
    _require(pointer_metadata.get("role_mode") == "pointer", "R10 requires the R4 pointer bridge")
    _require(
        pointer_metadata.get("base_parameters_trainable") == 0,
        "pointer metadata does not freeze the base",
    )
    bindings = (
        ("base_sha256", "base_sha256"),
        ("data_sha256", "data_sha256"),
        ("admission_sha256", "admission_sha256"),
        ("label_admission_sha256", "label_admission_sha256"),
    )
    for pointer_key, r9c_key in bindings:
        _require(
            pointer_metadata.get(pointer_key) == r9c_metadata.get(r9c_key),
            "pointer and R9c metadata disagree on {}".format(pointer_key),
        )
    _require(
        int(pointer_metadata.get("hidden", -1)) == int(r9c_metadata.get("pointer_hidden", -2)),
        "pointer and R9c metadata disagree on hidden width",
    )


def serialize_record(index, wrapped, probabilities, local_index):
    example = wrapped.compiled
    depth = len(example.operation_targets)
    for name in ("joint", "forward", "backward"):
        if probabilities[name].shape[1] != depth:
            raise ValueError("{} score depth differs from compiled example".format(name))
    return {
        "index": int(index),
        "reference": example.reference,
        "regime": example.regime,
        "operation_targets": [int(value) for value in example.operation_targets],
        "operation_values": [int(value) for value in example.operation_values],
        "initial_state": [int(value) for value in example.initial_values],
        "query_target": int(example.query_target),
        "answer": int(example.answer),
        "joint_probabilities": probabilities["joint"][local_index].tolist(),
        "forward_probabilities": probabilities["forward"][local_index].tolist(),
        "backward_probabilities": probabilities["backward"][local_index].tolist(),
        "query_probabilities": probabilities["query"][local_index].tolist(),
    }


def atomic_write_json_no_overwrite(payload, path):
    """Publish complete JSON atomically without an overwrite race."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError("refusing existing output: {}".format(output))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(output.name), suffix=".tmp", dir=str(output.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError("refusing existing output: {}".format(output)) from error
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_hashes(args):
    paths = {
        "base": args.base,
        "pointer_adapter": args.pointer_adapter,
        "adapter": args.adapter,
        "data": args.data,
        "tokenizer": args.tokenizer,
        "structural_admission": args.admission,
        "referential_label_admission": args.label_admission,
    }
    hashes = {}
    for name, path in paths.items():
        _require(Path(path).is_file() and Path(path).stat().st_size > 0, "missing input: {}".format(path))
        hashes[name] = sha256_file(path)
    return paths, hashes


def _require_unchanged(paths, hashes):
    for name, path in paths.items():
        _require(
            Path(path).is_file() and sha256_file(path) == hashes[name],
            "input changed during extraction: {}".format(path),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pointer-adapter", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--extractor-sha256", required=True)
    args = parser.parse_args()

    _require(args.batch_size > 0, "batch size must be positive")
    actual_extractor_sha256 = sha256_file(__file__)
    validate_code_identity(
        args.code_revision, args.extractor_sha256, actual_extractor_sha256,
    )
    _require(not os.path.lexists(args.out), "refusing existing output: {}".format(args.out))
    _require(torch.cuda.is_available(), "R10 score extraction requires CUDA")
    try:
        torch.empty(1024, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as error:
        raise SystemExit("R10 score extraction requires a usable CUDA allocation: {}".format(error))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    paths, hashes = _artifact_hashes(args)
    admission = _load_manifest(args.admission, "structural admission")
    label_admission = _load_manifest(args.label_admission, "referential-label admission")
    checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = checkpoint.get("referential_syndrome_microcode")
    _require(isinstance(metadata, dict), "adapter lacks complete R9c metadata")
    _require(checkpoint.get("step") == "syndrome_adapter_ep1", "invalid R9c adapter step")
    _require(isinstance(checkpoint.get("microcode_state"), dict), "adapter lacks microcode state")
    validate_hash_bindings(metadata, hashes, admission, label_admission)

    from tokenizers import Tokenizer

    from eval_referential_slot_microcode import load_examples
    from eval_referential_syndrome_microcode import matched_batches
    from referential_syndrome_microcode import ReferentialSyndromeBridge
    from train_referential_slot_microcode import pad_ids
    from train_referential_syndrome_microcode import hash_microcode_state, load_pointer

    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    base_checkpoint, pointer_metadata, pointer = load_pointer(
        args.base, args.pointer_adapter, "cuda",
    )
    validate_pointer_metadata(pointer_metadata, metadata)
    examples = load_examples(args.data, tokenizer, pointer.model.cfg.seq_len)
    del base_checkpoint
    batches = matched_batches(examples, args.batch_size)

    bridge = ReferentialSyndromeBridge(
        pointer,
        pointer_hidden=int(pointer_metadata["hidden"]),
        memory_dim=int(metadata["memory_dim"]),
    ).to("cuda").eval()
    incompatible = bridge.microcode.load_state_dict(checkpoint["microcode_state"], strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "R9c microcode state is incompatible with the frozen bridge",
    )
    bridge.requires_grad_(False)
    _require(
        bridge.adapter_num_params() == int(metadata.get("adapter_parameters", -1)),
        "R9c adapter parameter count differs from metadata",
    )
    _require(
        hash_microcode_state(bridge) == metadata.get("final_adapter_sha256"),
        "R9c microcode state does not match its final adapter hash",
    )

    records = [None] * len(examples)
    rounds = int(metadata["rounds"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected = [examples[index] for index in indices]
            encoded = bridge.encode_examples(pad_ids(selected, "cuda"), selected)
            run = bridge.microcode(
                encoded.event_features,
                encoded.values,
                encoded.initial_values,
                encoded.query_goals,
                rounds=rounds,
                conditioning="directional",
                use_syndrome=False,
                adaptive=False,
            )
            probabilities = {
                name: tensor.cpu()
                for name, tensor in categorical_probabilities(
                    run.forward_logits, run.backward_logits, encoded.query_logits,
                ).items()
            }
            for local_index, index in enumerate(indices):
                records[index] = serialize_record(
                    index, examples[index], probabilities, local_index,
                )
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[r10-scores] {}/{} batches".format(batch_number, len(batches)), flush=True)

    _require(all(record is not None for record in records), "R10 extraction left unscored rows")
    _require_unchanged(paths, hashes)
    result = {
        "audit": "referential_version_scores_r10",
        "schema_version": 1,
        "code_revision": args.code_revision,
        "extractor_sha256": actual_extractor_sha256,
        "seed": int(args.seed),
        "base": os.path.realpath(args.base),
        "base_sha256": hashes["base"],
        "pointer_adapter": os.path.realpath(args.pointer_adapter),
        "pointer_adapter_sha256": hashes["pointer_adapter"],
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": hashes["adapter"],
        "adapter_step": checkpoint["step"],
        "adapter_state_sha256": metadata["final_adapter_sha256"],
        "adapter_metadata": metadata,
        "pointer_adapter_metadata": pointer_metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": hashes["data"],
        "tokenizer": os.path.realpath(args.tokenizer),
        "tokenizer_sha256": hashes["tokenizer"],
        "structural_admission": os.path.realpath(args.admission),
        "structural_admission_sha256": hashes["structural_admission"],
        "referential_label_admission": os.path.realpath(args.label_admission),
        "referential_label_admission_sha256": hashes["referential_label_admission"],
        "r9c_training_data": metadata["data"],
        "r9c_training_data_sha256": metadata["data_sha256"],
        "r9c_training_structural_admission_sha256": metadata["admission_sha256"],
        "r9c_training_referential_label_admission_sha256": metadata["label_admission_sha256"],
        "categorical_order": {
            "operations": list(OPCODES),
            "queries": list(QUERIES),
        },
        "replay": {
            "arm": "no_syndrome",
            "mode": "fixed",
            "adaptive": False,
            "rounds": rounds,
            "conditioning": "directional",
            "use_syndrome": False,
            "shuffle_goal": False,
        },
        "cases": len(records),
        "events": sum(len(record["operation_targets"]) for record in records),
        "batches": len(batches),
        "records": records,
        "claim_boundary": (
            "Read-only categorical score extraction from the frozen rejected R9c no-syndrome control. "
            "These probabilities do not establish reasoning, certification, or fresh transfer."
        ),
    }
    try:
        atomic_write_json_no_overwrite(result, args.out)
    except FileExistsError as error:
        raise SystemExit(str(error))
    print("[r10-scores] saved {} cases to {}".format(len(records), args.out), flush=True)


if __name__ == "__main__":
    main()
