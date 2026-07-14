#!/usr/bin/env python3
"""Focused contracts for the semantic-bridge held-out evaluator."""

from eval_semantic_bridge import score_response, select_rows, summarize, trace_equation_contract


def row(family, question):
    return {"family": family, "question": question, "answer": "7", "response": "<think>x</think>"}


def main():
    rows = [row("a", "a1"), row("a", "a2"), row("b", "b1"), row("b", "b2")]
    selected = select_rows(rows, per_family=1, seed=7)
    assert len(selected) == 2
    assert {item["family"] for item in selected} == {"a", "b"}
    assert select_rows(rows, per_family=1, seed=7) == selected

    gold = "<think>First 4+3=7. Then 7*2=14.</think> The answer is 14."
    correct = score_response("14", "<think>4 + 3 = 7; 7*2=14.</think> The answer is 14.", gold)
    assert correct == {
        "final": 14,
        "answer_correct": True,
        "trace_present": True,
        "visible_answer_correct": True,
        "trace_contract_correct": True,
    }
    assert trace_equation_contract("<think>4*6^2=144, 2*6^1=12, 5*6^0=5. 144+12+5=161.</think>") == {
        "4*6^2=144", "2*6^1=12", "5*6^0=5", "144+12+5=161",
    }
    decorative = score_response("14", "<think>checked carefully</think> The answer is 14.", gold)
    assert decorative["visible_answer_correct"] and not decorative["trace_contract_correct"]
    no_trace = score_response("7", "The answer is 7.")
    assert no_trace["answer_correct"] and not no_trace["visible_answer_correct"]
    wrong = score_response("7", "<think>4+3=7</think> The answer is 8.", "<think>4+3=7</think> The answer is 7.")
    assert not wrong["answer_correct"]

    result = summarize([dict(row("a", "a1"), **correct)])
    assert result["cases"] == result["answer_correct"] == result["visible_answer_correct"] == result["trace_contract_correct"] == 1
    print("semantic bridge evaluator tests passed")


if __name__ == "__main__":
    main()
