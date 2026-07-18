#!/usr/bin/env python3
"""Finite identifiability and collapse board for relation-complete transport."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROTOCOL = "R12-RELATION-COMPLETE-TRANSPORT-v1"
STATE_COUNT = 6
IDENTITY_INDEX = 0
GENERATOR_NAMES = ("s", "t")

Permutation3 = tuple[int, int, int]
Action = tuple[int, ...]
Table = tuple[Action, Action]


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compose3(left: Permutation3, right: Permutation3) -> Permutation3:
    return tuple(left[right[index]] for index in range(3))  # type: ignore[return-value]


S3_STATES: tuple[Permutation3, ...] = tuple(itertools.permutations(range(3)))
S3_INDEX = {state: index for index, state in enumerate(S3_STATES)}
S3_GENERATORS: dict[str, Permutation3] = {"s": (1, 0, 2), "t": (0, 2, 1)}


def canonical_table() -> Table:
    actions = []
    for name in GENERATOR_NAMES:
        generator = S3_GENERATORS[name]
        actions.append(
            tuple(S3_INDEX[compose3(generator, state)] for state in S3_STATES)
        )
    return tuple(actions)  # type: ignore[return-value]


def compose_action(left: Action, right: Action) -> Action:
    return tuple(left[right[index]] for index in range(STATE_COUNT))


IDENTITY_ACTION: Action = tuple(range(STATE_COUNT))


def apply_word(table: Table, start: int, word: Iterable[int]) -> int:
    state = start
    for generator in word:
        state = table[generator][state]
    return state


def relation_violations(table: Table) -> tuple[dict, ...]:
    violations = []
    for state in range(STATE_COUNT):
        s2 = apply_word(table, state, (0, 0))
        t2 = apply_word(table, state, (1, 1))
        sts = apply_word(table, state, (0, 1, 0))
        tst = apply_word(table, state, (1, 0, 1))
        if s2 != state:
            violations.append({"state": state, "relation": "s_squared", "got": s2})
        if t2 != state:
            violations.append({"state": state, "relation": "t_squared", "got": t2})
        if sts != tst:
            violations.append(
                {"state": state, "relation": "braid", "left": sts, "right": tst}
            )
    return tuple(violations)


def orbit(table: Table, start: int = IDENTITY_INDEX) -> frozenset[int]:
    seen = {start}
    queue: deque[int] = deque((start,))
    while queue:
        state = queue.popleft()
        for action in table:
            successor = action[state]
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return frozenset(seen)


@lru_cache(maxsize=1)
def involutions() -> tuple[Action, ...]:
    return tuple(
        action
        for action in itertools.product(range(STATE_COUNT), repeat=STATE_COUNT)
        if compose_action(action, action) == IDENTITY_ACTION
    )


@lru_cache(maxsize=1)
def relation_complete_actions() -> tuple[Table, ...]:
    candidates = []
    for s_action in involutions():
        for t_action in involutions():
            table = (s_action, t_action)
            if compose_action(
                s_action, compose_action(t_action, s_action)
            ) != compose_action(t_action, compose_action(s_action, t_action)):
                continue
            if len(orbit(table)) != STATE_COUNT:
                continue
            candidates.append(table)
    return tuple(candidates)


@lru_cache(maxsize=1)
def regular_action_relabelings() -> tuple[Table, ...]:
    exact = canonical_table()
    tables = set()
    for relabeling in itertools.permutations(range(STATE_COUNT)):
        inverse = tuple(relabeling.index(state) for state in range(STATE_COUNT))
        actions = []
        for action in exact:
            actions.append(
                tuple(
                    relabeling[action[inverse[state]]] for state in range(STATE_COUNT)
                )
            )
        tables.add(tuple(actions))
    return tuple(sorted(tables))  # type: ignore[return-value]


def edge_order() -> tuple[tuple[int, int], ...]:
    return tuple(
        (generator, state)
        for generator in range(len(GENERATOR_NAMES))
        for state in range(STATE_COUNT)
    )


def edge_value(table: Table, edge: tuple[int, int]) -> int:
    generator, state = edge
    return table[generator][state]


def one_erasure_certificate(exact: Table) -> dict:
    omitted = (0, STATE_COUNT - 1)
    completions = []
    for successor in range(STATE_COUNT):
        patched = [list(action) for action in exact]
        patched[omitted[0]][omitted[1]] = successor
        candidate: Table = tuple(tuple(action) for action in patched)  # type: ignore[assignment]
        if not relation_violations(candidate) and len(orbit(candidate)) == STATE_COUNT:
            completions.append(successor)

    wrong_successor = next(
        successor
        for successor in range(STATE_COUNT)
        if successor != edge_value(exact, omitted)
    )
    wrong = [list(action) for action in exact]
    wrong[omitted[0]][omitted[1]] = wrong_successor
    wrong_table: Table = tuple(tuple(action) for action in wrong)  # type: ignore[assignment]
    identity_only = {
        "s_squared": apply_word(wrong_table, IDENTITY_INDEX, (0, 0)) == IDENTITY_INDEX,
        "t_squared": apply_word(wrong_table, IDENTITY_INDEX, (1, 1)) == IDENTITY_INDEX,
        "braid": apply_word(wrong_table, IDENTITY_INDEX, (0, 1, 0))
        == apply_word(wrong_table, IDENTITY_INDEX, (1, 0, 1)),
    }
    global_violations = relation_violations(wrong_table)
    return {
        "omitted_edge": {"generator": GENERATOR_NAMES[omitted[0]], "state": omitted[1]},
        "exact_successor": edge_value(exact, omitted),
        "relation_consistent_successors": completions,
        "unique_exact_completion": completions == [edge_value(exact, omitted)],
        "wrong_patch_successor": wrong_successor,
        "wrong_patch_identity_only_relations": identity_only,
        "wrong_patch_global_violation_count": len(global_violations),
        "wrong_patch_global_violations": list(global_violations),
        "identity_only_checks_are_insufficient": (
            all(identity_only.values()) and bool(global_violations)
        ),
    }


def version_space_certificate(exact: Table, candidates: tuple[Table, ...]) -> dict:
    edges = edge_order()
    profile = []
    minimum_identifying_size = None
    example_identifying_edges = None
    identifying_subset_count = 0
    subsets_checked = 0

    for size in range(len(edges) + 1):
        counts = []
        size_identifying = []
        for subset in itertools.combinations(edges, size):
            subsets_checked += 1
            consistent = sum(
                all(
                    edge_value(candidate, edge) == edge_value(exact, edge)
                    for edge in subset
                )
                for candidate in candidates
            )
            counts.append(consistent)
            if consistent == 1:
                size_identifying.append(subset)
        profile.append(
            {
                "observed_edges": size,
                "subsets": len(counts),
                "minimum_candidates": min(counts),
                "median_candidates": statistics.median(counts),
                "maximum_candidates": max(counts),
                "identifying_subsets": len(size_identifying),
            }
        )
        if size_identifying and minimum_identifying_size is None:
            minimum_identifying_size = size
            example_identifying_edges = size_identifying[0]
            identifying_subset_count = len(size_identifying)

    assert minimum_identifying_size is not None
    assert example_identifying_edges is not None
    return {
        "edge_subsets_checked": subsets_checked,
        "minimum_identifying_edges": minimum_identifying_size,
        "minimum_identifying_subset_count": identifying_subset_count,
        "example_identifying_edges": [
            {
                "generator": GENERATOR_NAMES[generator],
                "state": state,
                "successor": edge_value(exact, (generator, state)),
            }
            for generator, state in example_identifying_edges
        ],
        "profile": profile,
    }


def resource_ledger(presentation: dict) -> dict:
    presentation_bytes = canonical_json_bytes(presentation)
    successor_bits = math.ceil(math.log2(STATE_COUNT))
    return {
        "states": STATE_COUNT,
        "generators": len(GENERATOR_NAMES),
        "directed_edges": STATE_COUNT * len(GENERATOR_NAMES),
        "unconstrained_tables": STATE_COUNT ** (STATE_COUNT * len(GENERATOR_NAMES)),
        "globally_relation_complete_transitive_tables": 120,
        "successor_bits_fixed_width": successor_bits,
        "full_atlas_labeled_target_bits": STATE_COUNT
        * len(GENERATOR_NAMES)
        * successor_bits,
        "minimum_relation_complete_anchor_bits": 4 * successor_bits,
        "labeled_target_bit_reduction_factor": 3,
        "global_relation_endpoint_equations": STATE_COUNT * 3,
        "transition_applications_per_global_relation_check": STATE_COUNT
        * (2 + 2 + 3 + 3),
        "presentation_bytes": len(presentation_bytes),
        "presentation_sha256": hashlib.sha256(presentation_bytes).hexdigest(),
        "state_carrier_and_transitivity_supplied": True,
        "semantic_state_labels_or_decoder_supplied": True,
        "hard_coded_group_action_is_favorable_control": True,
    }


def build_report() -> dict:
    presentation = {
        "carrier_size": STATE_COUNT,
        "generators": list(GENERATOR_NAMES),
        "relations": ["ss=e", "tt=e", "sts=tst"],
        "requirements": ["deterministic", "transitive"],
    }
    exact = canonical_table()
    candidates = relation_complete_actions()
    relabeled_regular_actions = regular_action_relabelings()
    exact_in_candidates = exact in candidates
    one_erasure = one_erasure_certificate(exact)
    version_space = version_space_certificate(exact, candidates)
    ledger = resource_ledger(presentation)

    candidate_set_digest = hashlib.sha256(
        canonical_json_bytes(
            [[list(action) for action in table] for table in candidates]
        )
    ).hexdigest()
    report = {
        "protocol": PROTOCOL,
        "claim_boundary": (
            "Finite exact identifiability and equivalence result only. It does not establish "
            "neural learnability, Shohin capability, autonomous reasoning, or primitive novelty."
        ),
        "presentation": presentation,
        "involutions_on_six_labels": len(involutions()),
        "globally_relation_complete_transitive_actions": len(candidates),
        "regular_action_relabelings": len(relabeled_regular_actions),
        "relation_actions_equal_regular_relabelings": set(candidates)
        == set(relabeled_regular_actions),
        "canonical_action_present": exact_in_candidates,
        "candidate_set_sha256": candidate_set_digest,
        "one_erasure": one_erasure,
        "version_space": version_space,
        "resource_ledger": ledger,
        "equivalence": {
            "relation_atlas_candidate_set_equals_tied_relation_aware_recurrence": set(
                candidates
            )
            == set(relabeled_regular_actions),
            "primitive_novelty_authorized": False,
            "surviving_hypothesis": (
                "matched sample-efficiency and scale-extrapolation advantage from global "
                "relation-syndrome supervision"
            ),
        },
        "neural_preregistration_drafting_authorized": False,
        "all_pass": (
            len(involutions()) == 76
            and len(candidates) == 120
            and len(relabeled_regular_actions) == 120
            and set(candidates) == set(relabeled_regular_actions)
            and exact_in_candidates
            and one_erasure["unique_exact_completion"]
            and one_erasure["identity_only_checks_are_insufficient"]
            and version_space["edge_subsets_checked"] == 2**12
            and version_space["minimum_identifying_edges"] == 4
            and ledger["unconstrained_tables"] == 2_176_782_336
            and ledger["full_atlas_labeled_target_bits"] == 36
            and ledger["minimum_relation_complete_anchor_bits"] == 12
            and ledger["transition_applications_per_global_relation_check"] == 60
        ),
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
