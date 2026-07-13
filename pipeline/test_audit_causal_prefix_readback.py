"""CPU-only solver contracts for the independent causal readback auditor."""

from audit_causal_prefix_readback import audit_row


def main():
    row = {
        "keys": ["left", "right"], "initial": {"left": 8, "right": 5},
        "operations": [
            {"kind": "add", "target": "left", "value": 2},
            {"kind": "move", "source": "left", "target": "right", "value": 4},
        ],
        "chunks": ["add two", "move four"], "state": [6, 9],
    }
    targets = audit_row(row)
    assert [(target[1], target[3]) for target in targets] == [("left", "10"), ("right", "9")]
    corrupted = dict(row); corrupted["state"] = [7, 9]
    try:
        audit_row(corrupted)
    except ValueError as exc:
        assert "final" in str(exc)
    else:
        raise AssertionError("corrupted final state was accepted")
    print("causal prefix readback auditor contracts passed")


if __name__ == "__main__":
    main()
