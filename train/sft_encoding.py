"""Lightweight, inference-aligned supervised-token encoding.

This module intentionally has no torch/model imports so job admission checks can
verify prompt boundaries without spending a GPU allocation importing the full
training stack.
"""


def encode_supervised_example(tok, prompt, continuation, eos_id):
    """Return independently tokenized prompt/completion ids and loss mask."""
    prompt_ids = tok.encode(prompt).ids
    completion_ids = tok.encode(continuation).ids
    token_ids = prompt_ids + completion_ids + [eos_id]
    completion_mask = [0] * len(prompt_ids) + [1] * (len(completion_ids) + 1)
    return prompt_ids, token_ids, completion_mask
