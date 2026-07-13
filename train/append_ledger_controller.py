"""Transport-only controller for model-authored append-ledger rollouts."""
from __future__ import annotations

from append_ledger_protocol import (canonical_block, canonical_delta, compact_prompt, final_prompt,
                                    parse_answer, parse_block, parse_delta, parse_base, transition_prompt)


def rollout_episode(episode, ask, prompt_style=None):
    """Schedule fixed turns and forward exact model text without arithmetic fallback."""
    base = parse_base(episode["base"])
    if base is None:
        raise ValueError("invalid base")
    style = episode["prompt_style"] if prompt_style is None else prompt_style
    block_size = int(episode["block_size"])
    expected_deltas, expected_blocks = episode["expected_deltas"], episode["expected_blocks"]
    blocks, live, rows, syntactic = [], [], [], True
    block_index = 0
    for step, expected_line in enumerate(expected_deltas):
        prompt = transition_prompt(base, blocks, live, step, style=style)
        response = ask(prompt)
        predicted = parse_delta(response)
        expected = parse_delta(expected_line)
        rows.append({"kind": "delta", "step": step, "prompt": prompt, "response": response,
                     "predicted": predicted, "expected": expected, "correct": predicted == expected})
        if predicted is None:
            syntactic = False
            break
        live.append(canonical_delta(predicted))
        if len(live) == block_size or step + 1 == len(expected_deltas):
            prompt = compact_prompt(base, blocks, live, block_index, style=style)
            response = ask(prompt)
            predicted_block = parse_block(response)
            expected_block = parse_block(expected_blocks[block_index])
            rows.append({"kind": "block", "block": block_index, "prompt": prompt, "response": response,
                         "predicted": predicted_block, "expected": expected_block,
                         "correct": predicted_block == expected_block})
            if predicted_block is None:
                syntactic = False
                break
            blocks.append(canonical_block(predicted_block))
            live = []
            block_index += 1
    final_response = ""
    if syntactic and block_index == len(expected_blocks):
        final_response = ask(final_prompt(base, blocks, style=style))
    exact_chain = syntactic and all(row["correct"] for row in rows)
    return {"rows": rows, "blocks": blocks, "syntactic_closed_loop": syntactic,
            "exact_chain": exact_chain, "final_response": final_response,
            "final_correct": parse_answer(final_response) == int(episode["expected_answer"])}
