#!/usr/bin/env python3
"""CPU-only probability and provenance contracts for the R10 score extractor."""

import copy
import torch

from categorical_microcode import OPCODES, QUERIES
from extract_referential_version_scores import (
    LABEL_ADMISSION_AUDIT,
    NO_SYNDROME_CONFIG,
    POINTER_PROTOCOL,
    R9C_PROTOCOL,
    STRUCTURAL_ADMISSION_AUDIT,
    categorical_probabilities,
    validate_code_identity,
    validate_hash_bindings,
)


def expect_rejected(call, message):
    try:
        call()
    except SystemExit:
        return
    raise AssertionError(message)


def test_hash_bindings():
    hashes = {
        "base": "1" * 64,
        "pointer_adapter": "2" * 64,
        "adapter": "3" * 64,
        "data": "4" * 64,
        "tokenizer": "5" * 64,
        "structural_admission": "6" * 64,
        "referential_label_admission": "7" * 64,
    }
    train_sha256 = "8" * 64
    metadata = {
        "protocol": R9C_PROTOCOL,
        "arm": "no_syndrome",
        "arm_config": NO_SYNDROME_CONFIG,
        "pointer_protocol": POINTER_PROTOCOL,
        "pointer_parameters_trainable": 0,
        "rounds": 3,
        "base_sha256": hashes["base"],
        "pointer_adapter_sha256": hashes["pointer_adapter"],
        "data_sha256": train_sha256,
        "tokenizer_sha256": hashes["tokenizer"],
        "admission_sha256": hashes["structural_admission"],
        "label_admission_sha256": hashes["referential_label_admission"],
        "final_adapter_sha256": "9" * 64,
    }
    admission = {
        "audit": STRUCTURAL_ADMISSION_AUDIT,
        "all_checks_pass": True,
        "eval_sha256": hashes["data"],
        "train_sha256": train_sha256,
        "tokenizer_sha256": hashes["tokenizer"],
    }
    label_admission = {
        "audit": LABEL_ADMISSION_AUDIT,
        "all_checks_pass": True,
        "tokenizer_sha256": hashes["tokenizer"],
        "datasets": {
            "eval": {"all_checks_pass": True, "sha256": hashes["data"]},
            "train": {"all_checks_pass": True, "sha256": train_sha256},
        },
    }
    validate_hash_bindings(metadata, hashes, admission, label_admission)

    for key, hash_key in (
        ("base_sha256", "base"),
        ("pointer_adapter_sha256", "pointer_adapter"),
        ("tokenizer_sha256", "tokenizer"),
    ):
        corrupted = copy.deepcopy(metadata)
        corrupted[key] = "a" * 64
        expect_rejected(
            lambda corrupted=corrupted: validate_hash_bindings(
                corrupted, hashes, admission, label_admission,
            ),
            "mismatched {} was accepted (artifact {})".format(key, hash_key),
        )

    bad_admission = copy.deepcopy(admission)
    bad_admission["eval_sha256"] = "a" * 64
    expect_rejected(
        lambda: validate_hash_bindings(metadata, hashes, bad_admission, label_admission),
        "structural admission with the wrong evaluation hash was accepted",
    )
    bad_labels = copy.deepcopy(label_admission)
    bad_labels["datasets"]["train"]["sha256"] = "a" * 64
    expect_rejected(
        lambda: validate_hash_bindings(metadata, hashes, admission, bad_labels),
        "label admission with the wrong training hash was accepted",
    )


def test_code_identity():
    revision = "a" * 40
    digest = "b" * 64
    validate_code_identity(revision, digest, digest)
    expect_rejected(
        lambda: validate_code_identity(revision, "c" * 64, digest),
        "mismatched extractor content hash was accepted",
    )
    expect_rejected(
        lambda: validate_code_identity("short", digest, digest),
        "abbreviated code revision was accepted",
    )


def main():
    test_hash_bindings()
    test_code_identity()
    generator = torch.Generator().manual_seed(20260714)
    forward = torch.randn(2, 3, len(OPCODES), generator=generator, dtype=torch.float64)
    backward = torch.randn(2, 3, len(OPCODES), generator=generator, dtype=torch.float64)
    query = torch.randn(2, len(QUERIES), generator=generator, dtype=torch.float64)
    probabilities = categorical_probabilities(forward, backward, query)

    assert set(probabilities) == {"joint", "forward", "backward", "query"}
    assert probabilities["joint"].shape == (2, 3, len(OPCODES))
    assert probabilities["query"].shape == (2, len(QUERIES))
    assert all(value.dtype == torch.float32 for value in probabilities.values())
    torch.testing.assert_close(
        probabilities["joint"],
        (0.5 * (forward.float() + backward.float())).softmax(dim=-1),
    )
    torch.testing.assert_close(probabilities["forward"], forward.float().softmax(dim=-1))
    torch.testing.assert_close(probabilities["backward"], backward.float().softmax(dim=-1))
    torch.testing.assert_close(probabilities["query"], query.float().softmax(dim=-1))
    for name, value in probabilities.items():
        torch.testing.assert_close(
            value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)),
            msg=lambda message, name=name: "{}: {}".format(name, message),
        )

    invalid = (
        (forward, backward[:, :2], query),
        (forward[..., :-1], backward[..., :-1], query),
        (forward, backward, query[:, :-1]),
        (forward.long(), backward, query),
    )
    for arguments in invalid:
        try:
            categorical_probabilities(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probability shape or dtype was accepted")

    nonfinite = forward.clone()
    nonfinite[0, 0, 0] = float("nan")
    try:
        categorical_probabilities(nonfinite, backward, query)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite logits were accepted")
    print("R10 referential version score probability tests: passed")


if __name__ == "__main__":
    main()
