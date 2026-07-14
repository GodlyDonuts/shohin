#!/usr/bin/env python3
"""CPU contracts for paired semantic-equivalence rendering."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from categorical_microcode import compile_example  # noqa: E402
from generate_categorical_microcode_equivalence_v2 import make_view  # noqa: E402
from test_categorical_microcode import CharacterTokenizer, row  # noqa: E402


def main():
    source = row(False)
    source["depth"] = len(source["operations"])
    first, second = make_view(source, 7, 0), make_view(source, 7, 1)
    tokenizer = CharacterTokenizer()
    compiled = [compile_example(item, tokenizer) for item in (first, second)]
    assert first["question"] != second["question"]
    assert first["keys"] != second["keys"]
    assert first["equivalence_id"] == second["equivalence_id"]
    assert compiled[0].operation_targets == compiled[1].operation_targets
    assert compiled[0].operation_values == compiled[1].operation_values
    assert compiled[0].query_target == compiled[1].query_target
    assert compiled[0].initial_values == compiled[1].initial_values
    assert compiled[0].answer == compiled[1].answer
    forbidden = ("relocate", "exchange the values assigned", "after all updates")
    assert not any(phrase in first["question"].lower() or phrase in second["question"].lower()
                   for phrase in forbidden)
    print("categorical microcode equivalence tests passed")


if __name__ == "__main__":
    main()
