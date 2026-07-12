#!/usr/bin/env python3
"""Verify the future language curriculum's *effective* batch composition."""
from data import effective_domain_fractions


def main():
    # openwebmath, code, finemath3, openmath, fineweb, dclm. FineMath-4 is
    # intentionally absent because it is contained in FineMath-3.
    fractions = effective_domain_fractions([5, 27, 6, 8, 9, 45], batch_size=32)
    math = fractions[0] + fractions[2] + fractions[3]
    code = fractions[1]
    english = fractions[4] + fractions[5]
    assert abs(sum(fractions) - 1.0) < 1e-12
    # The loader reserves one sequence per active domain, so this six-source
    # curriculum can only approximate the intended 25/25/50 split.
    assert abs(math - 0.25) < 0.003, math
    assert abs(code - 0.25) < 0.003, code
    assert abs(english - 0.50) < 0.003, english
    try:
        effective_domain_fractions([1, 1, 1], batch_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("batch floor must reject too-small batches")
    print(f"effective mix passed: math={math:.3f} code={code:.3f} english={english:.3f}")


if __name__ == "__main__":
    main()
