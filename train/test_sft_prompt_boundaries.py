#!/usr/bin/env python3
"""Regression tests for inference-aligned SFT completion masking."""
from pathlib import Path

from tokenizers import Tokenizer

from sft import encode_supervised_example


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "artifacts" / "shohin-tok-32k.json"


def check_case(tokenizer, prompt, continuation):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    prompt_ids, token_ids, mask = encode_supervised_example(
        tokenizer, prompt, continuation, eos_id
    )
    completion_ids = tokenizer.encode(continuation).ids

    assert token_ids[:len(prompt_ids)] == prompt_ids
    assert token_ids[len(prompt_ids):-1] == completion_ids
    assert token_ids[-1] == eos_id
    assert mask == [0] * len(prompt_ids) + [1] * (len(completion_ids) + 1)


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_case(
        tokenizer,
        "Question: Return the integer.\nAnswer:",
        " 42",
    )
    # CRLF plus indentation is the boundary that previously allowed BPE merges
    # to invalidate code-completion masks.
    check_case(
        tokenizer,
        "# Task: Return an equilibrium index.\ndef equilibrium_index(arr):\r\n",
        "total_sum = sum(arr)\r\n  left_sum = 0",
    )
    print("sft prompt-boundary tests: passed")


if __name__ == "__main__":
    main()
