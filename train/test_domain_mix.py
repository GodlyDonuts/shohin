#!/usr/bin/env python3
"""Verify the future language curriculum's *effective* batch composition."""
from data import effective_domain_fractions


def main():
    # fine4, openwebmath, code, finemath3, openmath, fineweb, dclm
    fractions = effective_domain_fractions([2, 4, 28, 4, 6, 32, 24], batch_size=32)
    math = fractions[0] + fractions[1] + fractions[3] + fractions[4]
    code = fractions[2]
    english = fractions[5] + fractions[6]
    assert abs(sum(fractions) - 1.0) < 1e-12
    assert abs(math - 0.25) < 1e-12, math
    assert abs(code - 0.25) < 1e-12, code
    assert abs(english - 0.50) < 1e-12, english
    try:
        effective_domain_fractions([1, 1, 1], batch_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("batch floor must reject too-small batches")
    print(f"effective mix passed: math={math:.3f} code={code:.3f} english={english:.3f}")


if __name__ == "__main__":
    main()
