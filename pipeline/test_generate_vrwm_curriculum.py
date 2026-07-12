#!/usr/bin/env python3
"""Integrity checks for the VRWM curriculum generator."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from generate_vrwm_curriculum import episode_prompt_signatures, normalized_prompt
from vrwm_protocol import apply_operation, canonical_memory


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    train = directory / "train.jsonl"
    evaluation = directory / "eval.jsonl"
    report = directory / "report.json"
    subprocess.run([
        sys.executable, "pipeline/generate_vrwm_curriculum.py",
        "--train-out", str(train), "--eval-out", str(evaluation), "--report", str(report),
        "--train-episodes", "20", "--eval-per-length", "3", "--seed", "41",
    ], cwd=ROOT, check=True)
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    episodes = [json.loads(line) for line in evaluation.read_text().splitlines()]
    summary = json.loads(report.read_text())
    assert summary["schema"] == "shohin-vrwm-v1"
    assert summary["duplicate_train_prompts_dropped"] >= 0
    assert len(episodes) == 15
    assert {row["training_group"] for row in rows} == {"vrwm"}
    assert {row["source"] for row in rows} == {"vrwm_transition_train", "vrwm_readout_train"}
    assert all(row["question"] == row["completion_prompt"] for row in rows)
    assert all(row["response"] for row in rows)
    prompts = [normalized_prompt(row["completion_prompt"]) for row in rows]
    assert len(prompts) == len(set(prompts))
    assert {row["program_length"] for row in episodes} == {4, 8, 16, 32}
    for episode in episodes:
        memory = episode["initial_memory"]
        for operation, expected in zip(episode["operations"], episode["expected_memories"]):
            assert apply_operation(memory, operation) == expected
            canonical_memory(expected)
            memory = expected
    assert not ({normalized_prompt(row["completion_prompt"]) for row in rows} &
                set().union(*(episode_prompt_signatures(episode) for episode in episodes)))
print("vrwm curriculum generator checks: passed")
