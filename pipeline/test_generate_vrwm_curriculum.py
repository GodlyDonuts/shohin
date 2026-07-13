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

    scratch_train = directory / "scratch_train.jsonl"
    scratch_eval = directory / "scratch_eval.jsonl"
    scratch_report = directory / "scratch_report.json"
    subprocess.run([
        sys.executable, "pipeline/generate_vrwm_curriculum.py",
        "--train-out", str(scratch_train), "--eval-out", str(scratch_eval), "--report", str(scratch_report),
        "--train-episodes", "20", "--train-max-steps", "8", "--train-value-limit", "999",
        "--train-const-limit", "127", "--train-styles", "default", "paraphrase", "--eval-style",
        "semantic", "--response-mode", "scratch", "--repair-examples", "2", "--eval-per-length", "3",
        "--seed", "43",
    ], cwd=ROOT, check=True)
    scratch_rows = [json.loads(line) for line in scratch_train.read_text().splitlines()]
    scratch_episodes = [json.loads(line) for line in scratch_eval.read_text().splitlines()]
    scratch_summary = json.loads(scratch_report.read_text())
    assert scratch_summary["train_styles"] == ["default", "paraphrase"]
    assert scratch_summary["eval_style"] == "semantic"
    assert scratch_summary["response_mode"] == "scratch"
    assert scratch_summary["repair_examples"] == 2
    assert any(row["source"] == "vrwm_transition_scratch_train" for row in scratch_rows)
    assert any(row["source"] == "vrwm_repair_train" for row in scratch_rows)
    assert any(row["response"].startswith("check:") for row in scratch_rows)
    scratch_prompts = {normalized_prompt(row["completion_prompt"]) for row in scratch_rows}
    semantic_eval_prompts = set().union(*(
        episode_prompt_signatures(episode, prompt_style="semantic", include_repair=True)
        for episode in scratch_episodes
    ))
    assert not (scratch_prompts & semantic_eval_prompts)
print("vrwm curriculum generator checks: passed")
