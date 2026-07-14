#!/usr/bin/env python3
"""Exact gates for proof-carrying source-dropped context folding."""

import random

import torch

from counterfactual_context_folding import (
    ProbeObservation,
    empty_context,
    fold_event,
    merge_contexts,
    oracle_event_certificate,
    read_context,
    verify_event_certificate,
)
from future_distinction_cell import (
    hypothesis_effect_codes,
    legal_operator_hypotheses,
)
from future_effect_algebra import compose_operators, query_operator


def main():
    hypotheses = legal_operator_hypotheses(range(1, 100))
    codes = hypothesis_effect_codes(hypotheses)

    certificates = [
        oracle_event_certificate(codes, target, max_probes=3)
        for target in range(len(hypotheses))
    ]
    assert all(certificate.accepted for certificate in certificates)
    assert max(len(certificate.observations) for certificate in certificates) <= 3

    certificate = certificates[137]
    bad_validation = ProbeObservation(
        certificate.validation.probe, certificate.validation.effect + 0.5,
    )
    rejected = verify_event_certificate(
        codes, certificate.observations, bad_validation,
    )
    assert not rejected.accepted
    assert rejected.reason == "validation_mismatch"
    ambiguous = verify_event_certificate(codes, (), certificate.validation)
    assert not ambiguous.accepted
    assert ambiguous.reason == "ambiguous_operator"

    generator = random.Random(20260714)
    targets = [generator.randrange(len(hypotheses)) for _ in range(4096)]
    full = empty_context()
    for target in targets:
        full = fold_event(full, certificates[target], hypotheses)
    expected = compose_operators(hypotheses[target].operator for target in targets)
    assert torch.equal(full.operator, expected)
    assert full.folded_events == len(targets)
    assert full.scalar_payload == 9

    chunks = []
    for start in range(0, len(targets), 257):
        chunk = empty_context()
        for target in targets[start:start + 257]:
            chunk = fold_event(chunk, certificates[target], hypotheses)
        chunks.append(chunk)
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merge_contexts(merged, chunk)
    assert torch.equal(merged.operator, full.operator)
    assert merged.scalar_payload == 9

    initial = (17, 31)
    state = torch.tensor([*initial, 1], dtype=torch.float64)
    for query in range(5):
        expected_answer = int((query_operator(query) @ expected @ state).item())
        assert read_context(full, initial, query) == expected_answer

    print(
        "proof-carrying context folding: passed "
        "(597 certificates, 4096 source-dropped events, 9-scalar carried state)"
    )


if __name__ == "__main__":
    main()
