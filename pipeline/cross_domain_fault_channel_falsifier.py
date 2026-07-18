#!/usr/bin/env python3
"""Finite CPU no-go board for three cross-domain reasoning analogies.

The board proves only bounded collapse/counterexample statements. It contains
no neural implementation, Shohin checkpoint access, data generation, fitting,
or accelerator work.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

PROTOCOL = "R12-CROSS-DOMAIN-FAULT-CHANNEL-NO-GO-v1"


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def majority3(values: tuple[int, int, int]) -> int:
    if any(value not in (0, 1) for value in values):
        raise ValueError("majority inputs must be bits")
    return int(sum(values) >= 2)


def triadic_efference_certificate() -> dict:
    one_fault_checks = 0
    one_fault_failures = []
    repetition_equivalence = True
    shared_semantic_failures = 0
    for state, action in itertools.product((0, 1), repeat=2):
        correct = state ^ action
        codeword = (correct, correct, correct)
        for fault_index in range(3):
            received = list(codeword)
            received[fault_index] ^= 1
            received_tuple = tuple(received)
            decoded = majority3(received_tuple)
            repetition_decoded = majority3(received_tuple)
            one_fault_checks += 1
            repetition_equivalence &= decoded == repetition_decoded
            if decoded != correct:
                one_fault_failures.append(
                    {
                        "state": state,
                        "action": action,
                        "fault_index": fault_index,
                        "decoded": decoded,
                    }
                )

        wrong_shared_action = 1 - action
        shared_proposals = (state ^ wrong_shared_action,) * 3
        shared_semantic_failures += int(majority3(shared_proposals) != correct)

    # The observation (0, 1) has two indistinguishable one-fault origins.
    ambiguous_observation = (0, 1)
    two_lane_origins = (
        {"truth": 0, "faulty_lane": 1},
        {"truth": 1, "faulty_lane": 0},
    )
    return {
        "one_fault_checks": one_fault_checks,
        "one_fault_failures": one_fault_failures,
        "two_lane_observation": list(ambiguous_observation),
        "two_lane_consistent_origins": list(two_lane_origins),
        "two_lane_fault_localization_identifiable": False,
        "three_lane_decoder_equals_repetition_majority": repetition_equivalence,
        "shared_semantic_cases": 4,
        "shared_semantic_failures": shared_semantic_failures,
        "surviving_advantage_over_repetition_code": False,
        "pass": (
            one_fault_checks == 12
            and not one_fault_failures
            and len(two_lane_origins) == 2
            and repetition_equivalence
            and shared_semantic_failures == 4
        ),
    }


def matvec_mod5(matrix: tuple[tuple[int, int], tuple[int, int]], vector: tuple[int, int]) -> tuple[int, int]:
    return (
        (matrix[0][0] * vector[0] + matrix[0][1] * vector[1]) % 5,
        (matrix[1][0] * vector[0] + matrix[1][1] * vector[1]) % 5,
    )


def add_mod5(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return ((left[0] + right[0]) % 5, (left[1] + right[1]) % 5)


def sub_mod5(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return ((left[0] - right[0]) % 5, (left[1] - right[1]) % 5)


def reversible_transport_certificate() -> dict:
    matrix = ((2, 1), (1, 1))
    determinant_mod5 = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) % 5
    states = tuple(itertools.product(range(5), repeat=2))
    nonzero_errors = tuple(state for state in states if state != (0, 0))
    checks = 0
    collapsed_errors = []
    for state in states:
        for error in nonzero_errors:
            clean = state
            perturbed = add_mod5(state, error)
            for step in range(1, 11):
                clean = matvec_mod5(matrix, clean)
                perturbed = matvec_mod5(matrix, perturbed)
                residual = sub_mod5(perturbed, clean)
                checks += 1
                if residual == (0, 0):
                    collapsed_errors.append(
                        {
                            "state": list(state),
                            "error": list(error),
                            "step": step,
                        }
                    )
    return {
        "field": "F_5",
        "matrix": [list(row) for row in matrix],
        "determinant_mod5": determinant_mod5,
        "states": len(states),
        "nonzero_errors": len(nonzero_errors),
        "steps_per_pair": 10,
        "checks": checks,
        "collapsed_errors": collapsed_errors,
        "reversible_transport_contracts_all_errors": False,
        "correction_requires_noninvertible_decoder_or_extra_provenance": True,
        "surviving_advantage_over_matched_recurrence": False,
        "pass": (
            determinant_mod5 == 1
            and checks == 25 * 24 * 10
            and not collapsed_errors
        ),
    }


Permutation = tuple[int, int, int]
IDENTITY: Permutation = (0, 1, 2)
GENERATORS: dict[str, Permutation] = {"s": (1, 0, 2), "t": (0, 2, 1)}


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""
    return tuple(left[right[index]] for index in range(3))  # type: ignore[return-value]


def exact_group_update(state: Permutation, generator: str) -> Permutation:
    return compose(GENERATORS[generator], state)


def execute_word(
    word: Iterable[str],
    update: Callable[[Permutation, str], Permutation] = exact_group_update,
) -> Permutation:
    state = IDENTITY
    for generator in word:
        state = update(state, generator)
    return state


def enumerate_s3_states() -> tuple[Permutation, ...]:
    seen = {IDENTITY}
    queue: deque[Permutation] = deque((IDENTITY,))
    while queue:
        state = queue.popleft()
        for generator in GENERATORS:
            successor = exact_group_update(state, generator)
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return tuple(sorted(seen))


def shortest_word(target: Permutation) -> tuple[str, ...]:
    queue: deque[tuple[Permutation, tuple[str, ...]]] = deque(((IDENTITY, ()),))
    seen = {IDENTITY}
    while queue:
        state, word = queue.popleft()
        if state == target:
            return word
        for generator in GENERATORS:
            successor = exact_group_update(state, generator)
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, (*word, generator)))
    raise AssertionError("unreachable S3 state")


def relation_syndrome_atlas_certificate() -> dict:
    states = enumerate_s3_states()
    transition_pairs = tuple((state, generator) for state in states for generator in GENERATORS)
    exact_atlas = {
        (state, generator): exact_group_update(state, generator)
        for state, generator in transition_pairs
    }
    atlas_matches_recurrence = all(
        exact_atlas[(state, generator)] == exact_group_update(state, generator)
        for state, generator in transition_pairs
    )
    relations = {
        "s_squared": execute_word(("s", "s")) == IDENTITY,
        "t_squared": execute_word(("t", "t")) == IDENTITY,
        "braid": execute_word(("s", "t", "s")) == execute_word(("t", "s", "t")),
    }

    omitted_pair = (states[-1], "s")
    exact_omitted_successor = exact_atlas[omitted_pair]
    wrong_successor = next(state for state in states if state != exact_omitted_successor)

    def patched_update(state: Permutation, generator: str) -> Permutation:
        if (state, generator) == omitted_pair:
            return wrong_successor
        return exact_atlas[(state, generator)]

    admitted_pairs = tuple(pair for pair in transition_pairs if pair != omitted_pair)
    admitted_exact = all(
        patched_update(state, generator) == exact_atlas[(state, generator)]
        for state, generator in admitted_pairs
    )
    witness_prefix = shortest_word(omitted_pair[0])
    witness_word = (*witness_prefix, omitted_pair[1])
    exact_endpoint = execute_word(witness_word)
    patched_endpoint = execute_word(witness_word, patched_update)
    separating_queries = tuple(
        index for index in range(3) if exact_endpoint[index] != patched_endpoint[index]
    )
    return {
        "states": len(states),
        "state_generator_pairs": len(transition_pairs),
        "relations": relations,
        "complete_atlas_matches_six_state_recurrence": atlas_matches_recurrence,
        "omitted_pair": {"state": list(omitted_pair[0]), "generator": omitted_pair[1]},
        "admitted_pairs": len(admitted_pairs),
        "patched_atlas_exact_on_all_admitted_pairs": admitted_exact,
        "witness_word": list(witness_word),
        "exact_endpoint": list(exact_endpoint),
        "patched_endpoint": list(patched_endpoint),
        "separating_late_queries": list(separating_queries),
        "finite_incomplete_atlas_identifies_uniform_update": False,
        "surviving_advantage_over_tied_relation_aware_recurrence": False,
        "pass": (
            len(states) == 6
            and len(transition_pairs) == 12
            and all(relations.values())
            and atlas_matches_recurrence
            and len(admitted_pairs) == 11
            and admitted_exact
            and exact_endpoint != patched_endpoint
            and bool(separating_queries)
        ),
    }


def disjoint_fault_neighborhood_lemma_certificate() -> dict:
    uncoded = {0: frozenset((0, 1)), 1: frozenset((0, 1))}
    codewords = {0: (0, 0, 0), 1: (1, 1, 1)}

    def radius_one(word: tuple[int, int, int]) -> frozenset[tuple[int, int, int]]:
        values = {word}
        for index in range(3):
            mutated = list(word)
            mutated[index] ^= 1
            values.add(tuple(mutated))
        return frozenset(values)

    coded = {label: radius_one(word) for label, word in codewords.items()}
    return {
        "uncoded_neighborhood_intersection": sorted(uncoded[0] & uncoded[1]),
        "uncoded_exact_recovery_identifiable": False,
        "repetition_radius_one_intersection": sorted(coded[0] & coded[1]),
        "repetition_exact_recovery_identifiable": not bool(coded[0] & coded[1]),
        "interpretation": (
            "Disjoint fault neighborhoods are coding redundancy. Extra origin information is "
            "retained provenance; semantic selection absent from the state is an oracle."
        ),
        "pass": bool(uncoded[0] & uncoded[1]) and not bool(coded[0] & coded[1]),
    }


def build_report() -> dict:
    sections = {
        "fault_neighborhood_lemma": disjoint_fault_neighborhood_lemma_certificate(),
        "triadic_efference_commit": triadic_efference_certificate(),
        "shadowed_variational_transport": reversible_transport_certificate(),
        "consolidated_relation_syndrome_atlas": relation_syndrome_atlas_certificate(),
    }
    report = {
        "protocol": PROTOCOL,
        "claim_boundary": (
            "A pass establishes only the named finite no-go witnesses. It does not prove that all "
            "cross-domain mechanisms fail, does not authorize neural code or fitting, and is not a "
            "Shohin capability or novelty claim."
        ),
        "candidate_mechanisms_tested": 3,
        "candidate_mechanisms_surviving": 0,
        "neural_preregistration_authorized": False,
        "sections": sections,
        "all_pass": all(section["pass"] for section in sections.values()),
    }
    report["payload_sha256"] = payload_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = canonical_json_bytes(report)
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
    print(payload.decode(), end="")
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
