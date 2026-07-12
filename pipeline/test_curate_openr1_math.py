#!/usr/bin/env python3
from curate_openr1_math import choose_generation, clean_trace, normalized_question


def main():
    row = {
        "generations": ["<think>unverified</think>", "<think>verified</think>"],
        "correctness_math_verify": [False, True],
        "correctness_llama": [True, False],
        "is_reasoning_complete": [True, True],
    }
    assert choose_generation(row) == ("<think>verified</think>", "math_verify")
    fallback = dict(row, correctness_math_verify=[False, False])
    assert choose_generation(fallback) == ("<think>unverified</think>", "llama_judge")
    incomplete = dict(row, is_reasoning_complete=[False, False])
    assert choose_generation(incomplete) is None
    assert clean_trace("<think>work</think>\nfinal") == "work\nfinal"
    assert normalized_question("A!  b\nA") == "a b a"
    print("openr1 curator selection checks: passed")


if __name__ == "__main__":
    main()
