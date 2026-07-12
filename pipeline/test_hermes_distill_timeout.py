#!/usr/bin/env python3
"""Regression test: a hung provider child must not stall distillation."""
import time

from hermes_distill import run_hard_timeout


def sleep_forever(send):
    time.sleep(30)
    send.send(("ok", "late"))


def immediate(send):
    send.send(("ok", "ready"))


def main():
    start = time.monotonic()
    status, value = run_hard_timeout(sleep_forever, (), timeout=0.2, grace=0.2)
    elapsed = time.monotonic() - start
    assert (status, value) == ("timeout", None), (status, value)
    assert elapsed < 3, elapsed
    assert run_hard_timeout(immediate, (), timeout=1) == ("ok", "ready")
    print(f"hard provider timeout passed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
