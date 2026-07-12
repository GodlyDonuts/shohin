#!/usr/bin/env python3
"""Fast invariant checks for deterministic non-repeating stream handoffs."""
from data import STREAM_SEED_STRIDE, stream_seed


def main():
    base = 777
    seeds = [stream_seed(base, generation) for generation in range(4)]
    assert seeds == [base + STREAM_SEED_STRIDE * generation for generation in range(4)]
    assert len(set(seeds)) == len(seeds)
    try:
        stream_seed(base, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative generation must fail")
    print(f"data-stream handoff invariant passed: {seeds}")


if __name__ == "__main__":
    main()
