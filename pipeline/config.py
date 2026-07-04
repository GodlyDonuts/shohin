"""Shohin data-pipeline config: tokenizer spec + sample sources.

Kept dependency-free (plain dicts/lists) so it imports anywhere the env runs.
"""

VOCAB_SIZE = 32768  # master plan §4 — 32k frees ~9.4M params vs SmolLM2's 49k

# Reserved tokens (master plan §4). Order matters — keep this list stable once
# the tokenizer is trained, or ids shift.
SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>",
    "<think>", "</think>",          # short NL chain-of-thought
    "<code>", "</code>",            # program-of-thought (PoT)
    "<answer>", "</answer>",
    "<|user|>", "<|assistant|>", "<|system|>",  # minimal chat template
    "<|correct|>", "<|incorrect|>",             # verifier labels
]

# Tokenizer-training sample sources: (dataset, config, split, text_column, weight).
# Public, ungated, streamable. weight = fraction of the total byte budget
# (≈ master plan's 60% edu / 25% math / 15% code).
SAMPLE_SOURCES = [
    ("HuggingFaceFW/fineweb-edu",         "sample-10BT",    "train", "text",    0.60),
    ("HuggingFaceTB/finemath",            "finemath-4plus", "train", "text",    0.25),
    ("codeparrot/codeparrot-clean-valid", None,             "train", "content", 0.15),
]
