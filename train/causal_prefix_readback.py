"""Source-free semantic readbacks for every continuous-memory prefix.

Earlier packet experiments allowed an auxiliary probe to read the packet while
the language decoder received answer supervision only at the final source
prefix.  That leaves an avoidable failure mode: a packet can fit a shallow
state probe without being usable by the model's own decoder.

This module defines a deterministic, solver-recomputed readback contract.  At
each source boundary, the decoder receives *only* that continuous packet plus a
fresh question about one register.  It never sees the source chunks again.
The contract is deliberately narrow and is meant to be paired with equal
decoder-work controls before making any retained-state claim.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from prefix_state_supervision import apply_register_operation


def readback_query(key: str) -> str:
    """Return a fixed source-free question for a previously written register."""
    key = str(key)
    if not key or any(character.isspace() for character in key):
        raise ValueError("register key must be one non-whitespace token")
    return (
        "After the updates received so far, what is the current value of "
        "{}? Return only the integer."
    ).format(key)


def prefix_readback_targets(
    initial: Mapping[str, int],
    operations: Iterable[Mapping[str, object]],
    keys: Sequence[str],
    final_state: Sequence[int],
) -> list[dict[str, object]]:
    """Recompute one language readback target after every source write.

    Key selection rotates by prefix position.  Consequently both registers are
    read across the dataset without embedding any source text or target value in
    the decoder question.  The final state is checked against the serialized
    row so trainer-side targets cannot silently drift from the solver data.
    """
    ordered_keys = tuple(str(key) for key in keys)
    if not ordered_keys:
        raise ValueError("readback targets require at least one register key")
    if len(final_state) != len(ordered_keys):
        raise ValueError("final state dimension does not match keys")
    values = {key: int(initial[key]) for key in ordered_keys}
    targets = []
    for index, operation in enumerate(operations):
        values = apply_register_operation(values, operation)
        key = ordered_keys[index % len(ordered_keys)]
        targets.append({
            "prefix_index": index,
            "key": key,
            "query": readback_query(key),
            "answer": str(int(values[key])),
        })
    if not targets:
        raise ValueError("readback targets require at least one source operation")
    expected_final = [int(value) for value in final_state]
    actual_final = [int(values[key]) for key in ordered_keys]
    if actual_final != expected_final:
        raise ValueError("final readback state does not match serialized state")
    return targets


def validate_readback_targets(targets: Sequence[Mapping[str, object]], chunk_count: int) -> None:
    """Verify target shape and ensure questions contain no answer leakage."""
    if len(targets) != int(chunk_count):
        raise ValueError("readback target count does not match source chunks")
    for index, target in enumerate(targets):
        if int(target.get("prefix_index", -1)) != index:
            raise ValueError("readback prefix indexes must be contiguous")
        key, query, answer = str(target.get("key", "")), str(target.get("query", "")), str(target.get("answer", ""))
        if query != readback_query(key):
            raise ValueError("readback query is not canonical")
        if not answer or answer.startswith("+") or any(character not in "-0123456789" for character in answer):
            raise ValueError("readback answer is not an integer")
        if answer in query:
            raise ValueError("readback query leaks the answer")
