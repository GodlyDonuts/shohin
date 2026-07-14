#!/usr/bin/env python3
"""Pure causal-control contracts for NRR report scoring."""
from eval_native_residual_relay import score_result


def row(**changes):
    value = {
        "direct": "answer=7",
        "normal": "answer=7",
        "paraphrase": "answer=7",
        "counterfactual": "answer=9",
        "zero": "answer=0",
        "shuffled": "answer=3",
        "expected": "answer=7",
        "counterfactual_expected": "answer=9",
    }
    value.update(changes)
    return value


def main():
    assert score_result(row())["strict_causal"]
    assert not score_result(row(shuffled="answer=7"))["strict_causal"]
    assert not score_result(row(counterfactual="answer=7"))["strict_causal"]
    try:
        score_result(row(counterfactual_expected="answer=7"))
    except ValueError:
        pass
    else:
        raise AssertionError("identical normal/counterfactual targets must fail")
    print("native residual relay evaluator checks: passed")


if __name__ == "__main__":
    main()
