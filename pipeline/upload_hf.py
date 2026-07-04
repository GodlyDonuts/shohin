#!/usr/bin/env python
"""Upload the completed Shohin data artifacts to the HF dataset repo.

Uploads only finished, decontaminated, quality-controlled corpora. Raw eval test
sets are intentionally NOT uploaded (re-hosting test sets aids contamination) — we
publish the derived 13-gram decontam artifact instead.

    HF_TOKEN=... python upload_hf.py            # -> Godlydonuts/shohin
"""
import os
from huggingface_hub import HfApi, create_repo

REPO = os.environ.get("SHOHIN_HF_REPO", "Godlydonuts/shohin")
TOKEN = os.environ["HF_TOKEN"]
B = "/lustre/fs1/home/sa305415/shohin/artifacts"
CARD = "/lustre/fs1/home/sa305415/shohin/pipeline/hf_dataset_card.md"

api = HfApi(token=TOKEN)
create_repo(REPO, repo_type="dataset", exist_ok=True, private=True, token=TOKEN)
print("repo ready:", REPO)

uploads = [
    (CARD,                                        "README.md"),
    (f"{B}/shohin-tok-32k.json",                  "tokenizer/shohin-tok-32k.json"),
    (f"{B}/rg_big/rg_train.jsonl",                "reasoning_gym/rg_train.jsonl"),
    (f"{B}/rg_big/rg_traces_train.jsonl",         "reasoning_gym/rg_traces_train.jsonl"),
    (f"{B}/rg_big/rg_eval.jsonl",                 "reasoning_gym/rg_eval.jsonl"),
    (f"{B}/sft/openmath2_concise.clean.jsonl",    "sft/openmath2_concise.clean.jsonl"),
    (f"{B}/sft/openmath2_concise_2M.clean.jsonl", "sft/openmath2_concise_2M.clean.jsonl"),
    (f"{B}/evals/evalgrams.pkl",                  "decontam/evalgrams.pkl"),
]
for local, remote in uploads:
    if os.path.exists(local):
        sz = os.path.getsize(local) / 1e6
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset")
        print(f"uploaded {remote} ({sz:.1f} MB)")
    else:
        print(f"skip (missing) {local}")

print("done -> https://huggingface.co/datasets/" + REPO)
