#!/usr/bin/env python3
from curate_openr1_math import (answer_in_text, choose_generation, clean_trace, novel_word_count,
                                normalized_question, verification_stat_key)


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
    assert answer_in_text("\\frac{1}{2}", "Therefore the value is $\\frac{1}{2}$.")
    assert not answer_in_text("13", "The answer is 31.")
    assert novel_word_count("Find x", "Solve x by subtraction") >= 3
    assert verification_stat_key("source_solution") == "source_solution_selected"
    assert verification_stat_key("math_verify") == "math_verify"
    print("openr1 curator selection checks: passed")


if __name__ == "__main__":
    main()
