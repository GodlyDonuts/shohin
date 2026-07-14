#!/usr/bin/env python3
"""CPU-only result-contract checks for the CRA evaluator."""
from eval_counterfactual_residual_algebra import score_result


def main():
    good = score_result({
        "expected": "answer=2", "counterfactual_expected": "answer=3", "normal": "answer=2",
        "paraphrase": "answer=2", "counterfactual": "answer=3", "zero": "answer=0", "shuffled": "answer=7",
    })
    assert good["strict_causal"]
    bad = score_result({
        "expected": "answer=2", "counterfactual_expected": "answer=3", "normal": "answer=2",
        "paraphrase": "answer=2", "counterfactual": "answer=3", "zero": "answer=2", "shuffled": "answer=7",
    })
    assert not bad["strict_causal"] and bad["zero_recreates_normal"]
    try:
        score_result({"expected": "answer=2", "counterfactual_expected": "answer=2"})
    except ValueError:
        pass
    else:
        raise AssertionError("identical normal/counterfactual target must be rejected")
    print("CRA evaluator checks: passed")


if __name__ == "__main__":
    main()
