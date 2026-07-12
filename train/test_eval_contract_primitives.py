#!/usr/bin/env python3
"""Unit checks for exact-answer extraction under all primitive answer types."""
from eval_contract_primitives import predict, read_rows


assert predict("The answer is 42.", "arithmetic") == "42"
assert predict("state=x\nThe answer is 17.", "state_update") == "17"
assert predict("The answer is [2, 4, 9].", "sort_unique") == "[2,4,9]"
assert predict("The answer is moPQsaic.", "string_insert") == "mopqsaic"
assert predict("Reasoning says no. The answer is no.", "syllogism") == "no"
print("contract primitive evaluator checks: passed")
