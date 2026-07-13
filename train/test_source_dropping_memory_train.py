#!/usr/bin/env python3
"""Pure batching and checkpoint-state checks for the source-memory trainer."""

from source_dropping_memory_train import (
    audit_admits_training,
    bucketed_batches,
    limit_complete_batches,
    make_batch,
    non_model_state,
)
from source_dropping_memory import SourceDroppingMemory
from model import GPT, GPTConfig


def main():
    generic_audit = {
        "train_sha256": "data",
        "invalid_train_rows": 0,
        "invalid_eval_rows": 0,
        "duplicate_train_prompts": 0,
        "duplicate_eval_prompts": 0,
        "train_eval_exact_prompt_hits": 0,
        "train_eval_13gram_hits": 0,
    }
    assert audit_admits_training(generic_audit, "data")
    ledger_audit = dict(
        generic_audit,
        audit="certified_latent_ledger_v1",
        counterfactual_train_pairs=2,
        counterfactual_eval_pairs=2,
    )
    assert audit_admits_training(ledger_audit, "data")
    assert not audit_admits_training(dict(ledger_audit, invalid_counterfactual_eval_pairs=1), "data")
    assert not audit_admits_training(dict(ledger_audit, counterfactual_train_pairs=0), "data")
    examples = [
        {"shape": (2, (3, 3), 2, 2)},
        {"shape": (2, (3, 3), 2, 2)},
        {"shape": (2, (3, 3), 2, 2)},
        {"shape": (1, (4,), 2, 2)},
        {"shape": (1, (4,), 2, 2)},
    ]
    first, report = bucketed_batches(examples, batch_size=2, seed=4)
    second, second_report = bucketed_batches(examples, batch_size=2, seed=4)
    assert first == second and report == second_report
    assert report == {"buckets": 2, "full_batches": 2, "dropped_examples": 1}
    assert limit_complete_batches(first, 2, 2) == first[:1]
    for batch in first:
        assert len({examples[index]["shape"] for index in batch}) == 1

    batch_examples = [
        {"chunks": [[1, 2, 3], [4, 5]], "query": [6], "answer": [7]},
        {"chunks": [[8, 9, 10], [11, 12]], "query": [13], "answer": [14]},
    ]
    chunks, query, answer = make_batch(batch_examples, [0, 1], "cpu")
    assert isinstance(chunks, tuple) and [tuple(chunk.shape) for chunk in chunks] == [(2, 3), (2, 2)]
    assert tuple(query.shape) == (2, 1) and tuple(answer.shape) == (2, 1)

    model = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=3, n_kv_head=1, d_model=48, d_ff=96, seq_len=32))
    memory = SourceDroppingMemory(model, slots=2, max_chunks=3)
    state = non_model_state(memory)
    assert set(state) == {"initial_slots", "write_slots", "chunk_bias"}
    print("source-dropping memory trainer tests passed")


if __name__ == "__main__":
    main()
