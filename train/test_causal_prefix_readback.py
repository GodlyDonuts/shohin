"""CPU-only contracts for deterministic source-free prefix readbacks."""

from causal_prefix_readback import prefix_readback_targets, readback_query, validate_readback_targets


def main():
    targets = prefix_readback_targets(
        {"left": 8, "right": 5},
        [
            {"kind": "add", "target": "left", "value": 2},
            {"kind": "move", "source": "left", "target": "right", "value": 4},
            {"kind": "swap", "left": "left", "right": "right"},
        ],
        ["left", "right"],
        [9, 6],
    )
    assert [(target["key"], target["answer"]) for target in targets] == [
        ("left", "10"), ("right", "9"), ("left", "9"),
    ]
    assert targets[0]["query"] == readback_query("left")
    assert "10" not in targets[0]["query"]
    validate_readback_targets(targets, 3)
    try:
        validate_readback_targets(targets[:-1], 3)
    except ValueError as exc:
        assert "count" in str(exc)
    else:
        raise AssertionError("missing readback target was accepted")
    print("causal prefix readback contracts passed")


if __name__ == "__main__":
    main()
