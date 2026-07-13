"""CPU-only contracts for causal prefix readback held-out selection/scoring."""

from eval_causal_prefix_readback import last_integer, summarize


def row(mode, correct, prefix=0, key="left"):
    return {
        "mode": mode, "correct": correct, "eval_regime": "fit_iid", "chunk_count": 2,
        "prefix_index": prefix, "key": key,
    }


def main():
    assert last_integer("The value is -12.") == -12
    assert last_integer("nothing") is None
    summary = summarize([
        row("normal", True, 0, "left"), row("normal", False, 1, "right"),
        row("zero", False, 0, "left"), row("zero", False, 1, "right"),
    ])
    assert summary["normal"]["correct"] == 1
    assert summary["normal"]["by_prefix"]["0"]["accuracy"] == 1.0
    assert summary["zero"]["accuracy"] == 0.0
    print("causal prefix readback evaluator contracts passed")


if __name__ == "__main__":
    main()
