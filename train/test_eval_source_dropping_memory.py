#!/usr/bin/env python3
"""Pure deterministic checks for the source-memory held-out evaluator."""

from eval_source_dropping_memory import final_answer, select_rows, shuffled_chunks, summarize


def row(regime, chunks, reference):
    return {
        "chunks": ["one"] * chunks,
        "query": "what?",
        "response": "The answer is 1.",
        "answer": "1",
        "chunk_count": chunks,
        "heldout": True,
        "eval_regime": regime,
        "reference": reference,
    }


def main():
    rows = [
        row("fit_iid", 2, "a"), row("fit_iid", 2, "b"),
        row("full_ood", 5, "c"), row("full_ood", 5, "d"),
    ]
    selected = select_rows(rows, 1, 7)
    assert len(selected) == 2 and {(item["eval_regime"], item["chunk_count"]) for item in selected} == {("fit_iid", 2), ("full_ood", 5)}
    assert final_answer("noise The answer is -12.") == -12
    assert final_answer("12") is None
    shuffled, order = shuffled_chunks(["a", "b", "c"], "ref", 8)
    assert sorted(order) == [0, 1, 2] and sorted(shuffled) == ["a", "b", "c"]
    summary = summarize([
        {"mode": "normal", "eval_regime": "fit_iid", "chunk_count": 2, "correct": True},
        {"mode": "normal", "eval_regime": "full_ood", "chunk_count": 5, "correct": False},
        {"mode": "zero", "eval_regime": "fit_iid", "chunk_count": 2, "correct": False},
    ])
    assert summary["normal"]["correct"] == 1 and summary["zero"]["correct"] == 0
    print("source-memory evaluator tests passed")


if __name__ == "__main__":
    main()
