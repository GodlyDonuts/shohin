#!/usr/bin/env python3
"""Pure-function checks for latent operator held-out evaluation."""

from eval_latent_operator import final_answer, select_rows, summarize


def main():
    rows = [
        {"question": "a", "depth": 5}, {"question": "b", "depth": 5},
        {"question": "c", "depth": 6}, {"question": "d", "depth": 6},
        {"question": "e", "depth": 8}, {"question": "f", "depth": 8},
    ]
    selected = select_rows(rows, per_depth=1, seed=3)
    assert len(selected) == 3 and {item["depth"] for item in selected} == {5, 6, 8}
    assert final_answer("noise The answer is 3. The answer is 8.") == 8
    assert final_answer("no parse") is None
    report = summarize([
        {"latent_steps": 0, "depth": 5, "correct": True},
        {"latent_steps": 0, "depth": 5, "correct": False},
        {"latent_steps": 2, "depth": 5, "correct": True},
    ])
    assert report["0"]["accuracy"] == 0.5
    assert report["2"]["by_depth"]["5"]["correct"] == 1
    print("latent operator evaluator tests passed")


if __name__ == "__main__":
    main()
