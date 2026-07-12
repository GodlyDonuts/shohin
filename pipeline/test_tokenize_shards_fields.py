#!/usr/bin/env python3
"""Regression checks for nested quality fields used by future corpus jobs."""
from tokenize_shards import field_value


def main():
    row = {
        "text": "hello",
        "flat": 7,
        "language_id_whole_page_fasttext": {"en": 0.93},
    }
    assert field_value(row, "text") == "hello"
    assert field_value(row, "flat") == 7
    assert field_value(row, "language_id_whole_page_fasttext.en") == 0.93
    assert field_value(row, "language_id_whole_page_fasttext.fr") is None
    assert field_value(row, "missing.path") is None
    print("tokenize field-path checks: passed")


if __name__ == "__main__":
    main()
