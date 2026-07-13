"""Pure local-transition basis for the next Digitwise Recurrent Scratchpad test.

The basis enumerates all *reachable* local decimal configurations for addition
and non-negative subtraction.  It is deliberately defined independently of a
particular train corpus so a generator and audit can agree on what must be
covered without treating numeric magnitude bands as a proxy for algorithmic
support.
"""
from __future__ import annotations

from digitwise_protocol import canonical_state


WIDTHS = (4, 6)
OPERATIONS = ("add", "sub")


def local_context(state):
    """Return the operation-local information needed to rewrite one digit."""
    canonical_state(state)
    if state["z"]:
        return None
    position = int(state["p"])
    return (
        int(state["w"]), str(state["op"]), position, int(state["c"]),
        int(state["a"][position]), int(state["b"][position]),
    )


def reachable_contexts(widths=WIDTHS):
    """Enumerate decimal local contexts compatible with non-negative subtraction."""
    result = set()
    for width in tuple(int(value) for value in widths):
        if width <= 0:
            raise ValueError("basis widths must be positive")
        for operation in OPERATIONS:
            for position in range(width):
                carries = (0,) if position == 0 else (0, 1)
                for carry in carries:
                    for left in range(10):
                        for right in range(10):
                            if operation == "sub" and position == width - 1:
                                if left < right or (left == right and carry):
                                    continue
                            result.add((width, operation, position, carry, left, right))
    return result


def context_label(context):
    width, operation, position, carry, left, right = context
    return "w{}-{}-p{}-c{}-{}{}".format(width, operation, position, carry, left, right)
