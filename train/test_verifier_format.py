#!/usr/bin/env python3
"""Keep verifier training and inference prompts byte-identical."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.build_verifier_dataset import verifier_question as training_prompt
from eval_verifier import verifier_question as evaluation_prompt


def main():
    question = "What is 12 plus 30?"
    candidate = "12 + 30 = 42. The answer is 42."
    assert training_prompt(question, candidate) == evaluation_prompt(question, candidate)
    assert "42" not in training_prompt(question, candidate).split("Problem:", 1)[0]
    print("verifier prompt contract: passed")


if __name__ == "__main__":
    main()
