#!/usr/bin/env python3
"""Exact contracts for the future-effect affine algebra."""

import json
from pathlib import Path

import torch

from categorical_microcode import OPCODE_TO_ID, QUERY_TO_ID, opcode_for, operation_value, query_for
from future_effect_algebra import (
    compose_operator_codes,
    compose_operators,
    decode_effect_signature,
    decode_operator_code,
    encode_operator_code,
    effect_measurement_matrix,
    effect_signature,
    execute_operator,
    operation_operator,
    program_operator,
    random_orthogonal_measurement_matrix,
    redundant_probe_bank,
    transport_operator_code,
)


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "artifacts/evals/latent_operator_eval_slices_v2_64.jsonl"
    rows = [json.loads(line) for line in data.open() if line.strip()]
    assert len(rows) == 896
    states, queries = redundant_probe_bank()
    singular_values = torch.linalg.svdvals(effect_measurement_matrix(states, queries))
    assert torch.linalg.matrix_rank(effect_measurement_matrix(states, queries)).item() == 9
    assert torch.allclose(singular_values, singular_values[:1].expand_as(singular_values))

    for row in rows:
        keys = row["keys"]
        opcodes = [OPCODE_TO_ID[opcode_for(operation, keys)] for operation in row["operations"]]
        values = [operation_value(operation) for operation in row["operations"]]
        query = QUERY_TO_ID[query_for(row["query"], keys)]
        initial = [row["initial"][key] for key in keys]
        assert execute_operator(initial, opcodes, values, query) == int(row["answer"])

        midpoint = len(opcodes) // 2
        left = program_operator(opcodes[:midpoint], values[:midpoint])
        right = program_operator(opcodes[midpoint:], values[midpoint:])
        full = program_operator(opcodes, values)
        assert torch.equal(compose_operators([left, right]), full)
        decoded = decode_effect_signature(
            effect_signature(full, states=states, queries=queries), states, queries,
        )["operator"]
        assert torch.allclose(decoded, full)

    base = operation_operator("add_0", 3)
    counterfactual = operation_operator("add_1", 3)
    assert not torch.equal(effect_signature(base), effect_signature(counterfactual))
    assert torch.equal(effect_signature(base), base)

    first = operation_operator("merge_0_1")
    second = operation_operator("sub_1", 4)
    third = operation_operator("swap")
    assert torch.equal(
        compose_operators([first, second, third]),
        third @ second @ first,
    )

    # Overcomplete future-effect codes recover a valid operator and expose a
    # sparse corruption without inspecting an arbitrary latent coordinate.
    operator = compose_operators([first, second, third])
    code = effect_signature(operator, states=states, queries=queries)
    clean = decode_effect_signature(code, states, queries)
    assert torch.allclose(clean["operator"], operator)
    assert clean["syndrome"].abs().max().item() < 1e-10

    corrupted = code.clone()
    corrupted[3, 5] += 17
    robust = decode_effect_signature(corrupted, states, queries, max_outliers=1)
    assert torch.allclose(robust["operator"], operator, atol=1e-9, rtol=0)
    assert robust["omitted_index"] == 3 * states.shape[0] + 5
    assert robust["syndrome"].abs().max().item() > 16.9

    left = compose_operators([first, second])
    right = third
    decoded_left = decode_effect_signature(
        effect_signature(left, states=states, queries=queries), states, queries,
    )["operator"]
    decoded_right = decode_effect_signature(
        effect_signature(right, states=states, queries=queries), states, queries,
    )["operator"]
    composed_code = effect_signature(decoded_right @ decoded_left, states=states, queries=queries)
    assert torch.allclose(composed_code, code)

    # A random orthogonal code is only a coordinate change when both arms
    # decode to the same operator. This blocks a meaningless R6 comparison in
    # which the treatment and control differ only by their 64x9 codebook.
    structured = effect_measurement_matrix(states, queries)
    scale = torch.linalg.svdvals(structured)[0].item()
    random_codebook = random_orthogonal_measurement_matrix(scale=scale)
    assert torch.linalg.matrix_rank(random_codebook).item() == 9
    assert torch.allclose(
        torch.linalg.svdvals(random_codebook),
        torch.linalg.svdvals(structured),
        atol=1e-10,
        rtol=0,
    )
    structured_code = encode_operator_code(operator, structured)
    random_code = transport_operator_code(structured_code, structured, random_codebook)
    assert torch.allclose(random_code, encode_operator_code(operator, random_codebook), atol=1e-10)
    assert torch.allclose(decode_operator_code(random_code, random_codebook), operator, atol=1e-10)
    left_structured = encode_operator_code(left, structured)
    right_structured = encode_operator_code(right, structured)
    left_random = transport_operator_code(left_structured, structured, random_codebook)
    right_random = transport_operator_code(right_structured, structured, random_codebook)
    structured_composed = compose_operator_codes(left_structured, right_structured, structured)
    random_composed = compose_operator_codes(left_random, right_random, random_codebook)
    assert torch.allclose(
        transport_operator_code(structured_composed, structured, random_codebook),
        random_composed,
        atol=1e-10,
    )
    print("future-effect algebra tests passed: 896 exact programs")


if __name__ == "__main__":
    main()
