#!/usr/bin/env python3
"""Deterministic contract tests for the conditional reflection protocol."""
from __future__ import annotations

from counterfactual_reflection_protocol import (
    NEUTRAL,
    Record,
    consumer_question,
    direct_question,
    exact_answer,
    exact_state,
    expected_answer,
    forward_state,
    neutral_question,
    reflection_question,
    state,
)


def main() -> None:
    record = Record(17, 9, 4, "workshop", "amber", "cobalt")
    carrier = state(record)
    assert carrier == "state:P=21;Q=9;D=4"
    assert expected_answer(record, "sum") == "answer=30"
    assert expected_answer(record, "difference") == "answer=12"

    direct = direct_question(record, "sum", heldout=False)
    reflect = reflection_question(record, heldout=False)
    neutral = neutral_question(record, heldout=False)
    assert "state:P=" not in direct
    assert "answer=" in direct
    assert "answer=" not in reflect and "state:P=" not in reflect
    assert "answer=" not in neutral and "state:P=" not in neutral
    assert NEUTRAL == "note:P=000;Q=000;D=000"
    assert exact_state(NEUTRAL) is None

    consumer = consumer_question(record, "sum", carrier, heldout=True)
    assert "original description has been erased" in consumer
    assert carrier in consumer
    assert "workshop" not in consumer and "amber" not in consumer and "cobalt" not in consumer
    assert forward_state(consumer, carrier, "state:P=22;Q=9;D=4").count("state:P=22;Q=9;D=4") == 1

    assert exact_state(" state:P=21;Q=9;D=4\n") == carrier
    assert exact_state("reasoning\nstate:P=21;Q=9;D=4") is None
    assert exact_state("state:P=21;Q=9") is None
    assert exact_answer("answer=30") == "answer=30"
    assert exact_answer("answer=30 because") is None

    try:
        consumer_question(record, "sum", "state:P=21;Q=9", heldout=False)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed state must not reach a consumer")

    try:
        forward_state("no carrier", carrier, carrier)
    except ValueError:
        pass
    else:
        raise AssertionError("controller must not synthesize an insertion point")

    print("counterfactual reflection protocol contracts passed")


if __name__ == "__main__":
    main()
