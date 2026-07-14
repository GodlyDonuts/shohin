#!/usr/bin/env python3
"""CPU end-to-end contract for the isolated paraphrase-state-alignment trainer."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from model import GPT, GPTConfig


def make_row(episode, phase, response):
    return {
        "schema": "semantic_basis_transport_v2",
        "episode_id": episode,
        "phase": phase,
        "question": "{} source {} ledger".format(phase, episode),
        "response": response,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["<|endoftext|>"])
        tokenizer_path = work / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        cfg = GPTConfig(vocab_size=2, n_layer=2, n_head=2, n_kv_head=1, d_model=16, d_ff=32, seq_len=32, zloss=0.0)
        checkpoint_path = work / "init.pt"
        torch.save({"model": GPT(cfg).state_dict(), "cfg": cfg.__dict__, "step": 0}, checkpoint_path)
        data_path = work / "pairs.jsonl"
        rows = []
        for index in range(12):
            response = "ledger:P={};Q={}".format(index + 1, index + 2)
            rows.extend((
                make_row("episode-{}".format(index), "compile", response),
                make_row("episode-{}".format(index), "reflect", response),
                make_row("episode-{}".format(index), "update", response),
                make_row("episode-{}".format(index), "difference", "answer=1"),
                make_row("episode-{}".format(index), "sum", "answer=2"),
            ))
        data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        for mode, weight, contrastive in (("same", "0.05", "0"), ("mismatch", "0.05", "0"),
                                          ("none", "0", "0"), ("same", "0.05", "0.1")):
            out = work / "out-{}-{}".format(mode, contrastive)
            command = [
                sys.executable, str(root / "train" / "sft_paraphrase_state_alignment.py"),
                "--init", str(checkpoint_path), "--data", str(data_path), "--tokenizer", str(tokenizer_path),
                "--out", str(out), "--epochs", "1", "--batch-size", "1", "--pair-batch-size", "2",
                "--capture-layer", "1", "--align-weight", weight, "--contrastive-weight", contrastive,
                "--pair-mode", mode, "--warmup", "0",
                "--device", "cpu", "--log-every", "1",
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + "\n" + result.stderr
            assert "[psa] saved epoch 1" in result.stdout
            saved = torch.load(out / "psa_ep1.pt", map_location="cpu")
            assert saved["state_alignment"]["mode"] == mode
            assert saved["state_alignment"]["pairs"] == 12
            assert saved["state_alignment"]["contrastive_weight"] == float(contrastive)
    print("paraphrase state alignment e2e checks: passed")


if __name__ == "__main__":
    main()
