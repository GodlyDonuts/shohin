#!/usr/bin/env python3
"""CPU contracts for prompt-boundary activation exchange mechanics."""
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from eval_paraphrase_state_causality import (
    capture_boundary,
    completion_logprob,
    encoded_prompt,
    evaluate_direction,
    greedy_full_replay,
    patched_logits,
    select_pairs,
)
from model import GPT, GPTConfig


def episode(index):
    p, q = 10 + index, 30 + index
    response = "ledger:P={};Q={}".format(p, q)
    rows = []
    questions = {
        "compile": "source {} has first {} and second {}; emit ledger".format(index, p, q),
        "reflect": "remember source {} values {} and {}; emit retained ledger".format(index, p, q),
        "update": "update source {} ledger".format(index),
        "difference": "subtract source {} ledger".format(index),
        "sum": "sum source {} ledger".format(index),
    }
    for phase, question in questions.items():
        rows.append({
            "schema": "semantic_basis_transport_v2", "split": "factor_language",
            "episode_id": "episode-{}".format(index), "phase": phase, "question": question,
            "response": response,
        })
    return rows


def make_tokenizer(texts):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        texts,
        trainers.WordLevelTrainer(special_tokens=["[UNK]", "<|endoftext|>"]),
    )
    return tokenizer


def main():
    episodes = [episode(index) for index in range(4)]
    pairs = select_pairs(episodes, 2, 20260714)
    assert len(pairs) == 2
    for target, donor in pairs:
        assert target[0]["episode_id"] != donor[0]["episode_id"]
        assert target[0]["response"] != donor[0]["response"]

    texts = []
    for rows in episodes:
        for row in rows:
            texts.extend(("Question: {}\nAnswer:".format(row["question"]), row["response"]))
    tokenizer = make_tokenizer(texts)
    cfg = GPTConfig(vocab_size=tokenizer.get_vocab_size(), n_layer=2, n_head=2, n_kv_head=1,
                    d_model=16, d_ff=32, seq_len=64, zloss=0.0)
    model = GPT(cfg).eval()
    target, donor = pairs[0]
    target_prompt = encoded_prompt(tokenizer, target[0]["question"], cfg.seq_len)
    donor_prompt = encoded_prompt(tokenizer, donor[1]["question"], cfg.seq_len)
    own = capture_boundary(model, target_prompt, 1, "cpu")
    other = capture_boundary(model, donor_prompt, 1, "cpu")
    baseline = patched_logits(model, target_prompt, 1, len(target_prompt) - 1, "cpu")
    identity = patched_logits(model, target_prompt, 1, len(target_prompt) - 1, "cpu", own)
    changed = patched_logits(model, target_prompt, 1, len(target_prompt) - 1, "cpu", other)
    assert torch.allclose(baseline, identity)
    assert not torch.allclose(baseline, changed)
    completion = tokenizer.encode(target[0]["response"]).ids
    assert isinstance(completion_logprob(model, target_prompt, completion, 1, other, 1.0, "cpu"), float)
    generated = greedy_full_replay(model, tokenizer, target_prompt, 1, other, 1.0, "cpu", 3)
    assert isinstance(generated, str)
    result = evaluate_direction(model, tokenizer, target, donor, 1, 1.0, "cpu", 3)
    assert result["target_episode_id"] != result["donor_episode_id"]
    assert result["target_ledger"] != result["donor_ledger"]
    assert "donor_minus_target_logprob" in result["mismatch_state"]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "episodes.jsonl"
        path.write_text("\n".join(json.dumps(row) for rows in episodes for row in rows) + "\n")
        assert path.stat().st_size > 0
    print("paraphrase state causality checks: passed")


if __name__ == "__main__":
    main()
