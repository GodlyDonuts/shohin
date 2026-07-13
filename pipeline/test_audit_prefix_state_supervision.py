#!/usr/bin/env python3
"""CPU smoke contract for the independent prefix-state auditor."""

from audit_prefix_state_supervision import prefix_states


def main():
    row = {
        "keys": ["left", "right"], "state_scale": 10,
        "initial": {"left": 4, "right": 9},
        "operations": [
            {"kind": "add", "target": "left", "value": 2},
            {"kind": "move", "source": "right", "target": "left", "value": 3},
        ],
        "chunks": ["first", "second"], "state": [9, 6],
    }
    assert prefix_states(row) == [[0.6, 0.9], [0.9, 0.6]]
    row["state"] = [8, 6]
    try:
        prefix_states(row)
    except ValueError as exc:
        assert "final prefix" in str(exc)
    else:
        raise AssertionError("invalid final state was admitted")
    print("prefix-state audit tests passed")


if __name__ == "__main__":
    main()
