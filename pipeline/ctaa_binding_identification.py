#!/usr/bin/env python3
"""Source-free mechanics audit for causal CTAA binding identification.

This module combines three experiment designs from different fields:

* the alternating group ``A4`` supplies a balanced train/confirmation split;
* persistent excitation from control theory makes every binding observable;
* delayed-match and activity-silent memory protocols motivate a
  write-delete-delay-read barrier and causal register transplantation.

The audit proves only that the proposed finite experiment geometry is coherent.
It does not evaluate a neural model or authorize a scored board.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence


ACTION_COUNT = 4
BINDINGS = tuple(itertools.permutations(range(ACTION_COUNT)))
IDENTITY = tuple(range(ACTION_COUNT))
ADJACENT_TRANSPOSITIONS = (
    (1, 0, 2, 3),
    (0, 2, 1, 3),
    (0, 1, 3, 2),
)
PROBE_TAPE = (0, 1, 2, 3)
DELAYS = (0, 32, 128)
REPORT_SCHEMA = "r12_ctaa_binding_identification_mechanics_v1"


class BindingIdentificationError(ValueError):
    """The finite binding-identification design is malformed."""


def _binding(value: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != ACTION_COUNT or sorted(result) != list(range(ACTION_COUNT)):
        raise BindingIdentificationError("CTAA binding is not a permutation")
    return result  # type: ignore[return-value]


def permutation_parity(value: Sequence[int]) -> int:
    """Return zero for even permutations and one for odd permutations."""

    binding = _binding(value)
    inversions = sum(
        binding[left] > binding[right]
        for left in range(ACTION_COUNT)
        for right in range(left + 1, ACTION_COUNT)
    )
    return inversions % 2


def compose(
    outer: Sequence[int], inner: Sequence[int]
) -> tuple[int, int, int, int]:
    """Compose permutations as ``outer(inner(local_opcode))``."""

    left = _binding(outer)
    right = _binding(inner)
    return tuple(left[right[index]] for index in range(ACTION_COUNT))  # type: ignore[return-value]


def resolve(
    binding: Sequence[int], local_tape: Iterable[int]
) -> tuple[int, ...]:
    checked = _binding(binding)
    tape = tuple(int(item) for item in local_tape)
    if any(item < 0 or item >= ACTION_COUNT for item in tape):
        raise BindingIdentificationError("CTAA local tape differs")
    return tuple(checked[item] for item in tape)


def apply_rebinding(
    binding: Sequence[int], adjacent_card_transposition: Sequence[int]
) -> tuple[int, int, int, int]:
    """Update a local-opcode binding while physical card storage stays fixed."""

    return compose(adjacent_card_transposition, binding)


def alternating_group_audit() -> dict[str, object]:
    even = tuple(binding for binding in BINDINGS if permutation_parity(binding) == 0)
    odd = tuple(binding for binding in BINDINGS if permutation_parity(binding) == 1)
    if len(even) != 12 or len(odd) != 12 or set(even).intersection(odd):
        raise AssertionError("CTAA alternating-group split differs")

    def marginal(partition: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
        result = []
        for opcode in range(ACTION_COUNT):
            counts = Counter(binding[opcode] for binding in partition)
            if set(counts) != set(range(ACTION_COUNT)):
                raise AssertionError("CTAA binding marginal support differs")
            result.append(tuple(counts[card] for card in range(ACTION_COUNT)))
        return tuple(result)

    even_marginal = marginal(even)
    odd_marginal = marginal(odd)
    expected = ((3, 3, 3, 3),) * ACTION_COUNT
    if even_marginal != expected or odd_marginal != expected:
        raise AssertionError("CTAA parity split is not locally balanced")
    return {
        "train_even_count": len(even),
        "confirmation_odd_count": len(odd),
        "train_local_marginals": even_marginal,
        "confirmation_local_marginals": odd_marginal,
    }


@dataclass(frozen=True)
class DelayReadReceipt:
    binding: tuple[int, int, int, int]
    delay: int
    resolved_tape: tuple[int, ...]
    distractor_checksum: int


def delayed_read(
    binding: Sequence[int], delay: int, *, distractor_seed: int
) -> DelayReadReceipt:
    """Read a sealed binding after an unrelated deterministic delay process."""

    sealed = _binding(binding)
    if delay not in DELAYS:
        raise BindingIdentificationError("CTAA delay is not preregistered")
    distractor = int(distractor_seed) & 0xFFFFFFFF
    for step in range(delay):
        distractor = (1_664_525 * distractor + 1_013_904_223 + step) & 0xFFFFFFFF
    return DelayReadReceipt(
        binding=sealed,
        delay=delay,
        resolved_tape=resolve(sealed, PROBE_TAPE),
        distractor_checksum=distractor,
    )


def write_delete_delay_read_audit() -> dict[str, object]:
    receipts = tuple(
        delayed_read(binding, delay, distractor_seed=index + 1)
        for index, binding in enumerate(BINDINGS)
        for delay in DELAYS
    )
    for binding in BINDINGS:
        tapes = {
            receipt.resolved_tape
            for receipt in receipts
            if receipt.binding == binding
        }
        if tapes != {resolve(binding, PROBE_TAPE)}:
            raise AssertionError("CTAA delayed binding read changed with delay")

    identity_reset_differences = sum(
        resolve(binding, PROBE_TAPE) != resolve(IDENTITY, PROBE_TAPE)
        for binding in BINDINGS
    )
    donor_following = 0
    for index, binding in enumerate(BINDINGS):
        donor = BINDINGS[(index + 1) % len(BINDINGS)]
        if resolve(donor, PROBE_TAPE) == resolve(binding, PROBE_TAPE):
            raise AssertionError("CTAA donor register did not differ")
        transplanted = delayed_read(donor, 128, distractor_seed=index + 101)
        donor_following += transplanted.resolved_tape == resolve(donor, PROBE_TAPE)
    if identity_reset_differences != 23 or donor_following != 24:
        raise AssertionError("CTAA delayed binding causal controls differ")
    return {
        "receipt_count": len(receipts),
        "delays": DELAYS,
        "identity_reset_differences": identity_reset_differences,
        "donor_following": donor_following,
    }


def rebinding_dynamics_audit(max_cues: int = 6) -> dict[str, object]:
    if max_cues < 3:
        raise BindingIdentificationError("CTAA rebinding depth is too small")
    reachable = {IDENTITY}
    sequence_count = 0
    prefix_checks = 0
    for length in range(1, max_cues + 1):
        for cue_indices in itertools.product(
            range(len(ADJACENT_TRANSPOSITIONS)), repeat=length
        ):
            sequence_count += 1
            states = [IDENTITY]
            for cue_index in cue_indices:
                states.append(
                    apply_rebinding(
                        states[-1], ADJACENT_TRANSPOSITIONS[cue_index]
                    )
                )
            reachable.update(states)
            for prefix_length in range(length):
                alternate_suffix = (
                    cue_indices[: prefix_length + 1]
                    + tuple(
                        (index + 1) % len(ADJACENT_TRANSPOSITIONS)
                        for index in cue_indices[prefix_length + 1 :]
                    )
                )
                alternate = IDENTITY
                alternate_states = [alternate]
                for cue_index in alternate_suffix:
                    alternate = apply_rebinding(
                        alternate, ADJACENT_TRANSPOSITIONS[cue_index]
                    )
                    alternate_states.append(alternate)
                if (
                    states[: prefix_length + 2]
                    != alternate_states[: prefix_length + 2]
                ):
                    raise AssertionError("CTAA future rebinding changed a prefix")
                prefix_checks += 1
    if reachable != set(BINDINGS):
        raise AssertionError("CTAA adjacent rebinding generators do not span S4")
    cycle_types = Counter(
        (
            permutation_parity(binding),
            sum(binding[index] == index for index in range(ACTION_COUNT)),
        )
        for binding in reachable
    )
    return {
        "max_cues": max_cues,
        "sequence_count": sequence_count,
        "reachable_binding_count": len(reachable),
        "prefix_causality_checks": prefix_checks,
        "parity_fixed_point_counts": {
            f"{parity}:{fixed}": count
            for (parity, fixed), count in sorted(cycle_types.items())
        },
    }


def build_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "claim_boundary": "finite_mechanics_only_no_neural_capability_claim",
        "alternating_group": alternating_group_audit(),
        "write_delete_delay_read": write_delete_delay_read_audit(),
        "rebinding_dynamics": rebinding_dynamics_audit(),
    }
    payload = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    report["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")


if __name__ == "__main__":
    main()
