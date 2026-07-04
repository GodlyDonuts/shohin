# Shohin data pipeline

Storage-lean data build for **Shohin-135M**, runnable under the **1 TB Lustre quota** on Stokes.
Everything streams from Hugging Face (no raw retention) and writes compact artifacts.

## The constraint that shapes this
Personal Lustre quota = **1 TB**. 580B tokens as uint16 = ~1.16 TB, so the full corpus *never fully lands
on Lustre*. Strategy: **build the differentiated, small-footprint data now** (tokenizer, procedural corpus,
SFT curation, decontamination, held-out evals — all CPU-buildable, all < a few hundred GB); **stream the
commodity web bulk later**, straight to the GPU training node's local NVMe.

## Status
- [x] Stokes access (SSH multiplexing) + env (`miniforge3`, conda env `data`)
- [ ] **Tokenizer** — `stream_sample.py` → `train_tokenizer.py` (32k BPE, single-digit, reserved tokens)
- [ ] **Reasoning-Gym procedural corpus** — the differentiator (infinite, verifiable, decontam-by-construction)
- [ ] **stream → zstd uint16 shards** — the storage-lean sharder
- [ ] **13-gram decontamination** + held-out eval freeze

## Run (on Stokes, conda env `data`)
```bash
source /lustre/fs1/home/sa305415/shohin/miniforge3/etc/profile.d/conda.sh && conda activate data
cd /lustre/fs1/home/sa305415/shohin/pipeline

# 1) stream a mixed sample (2 GB smoke test; 20 GB for the final tokenizer)
python stream_sample.py --out ../tok_sample.txt --gb 2

# 2) train the 32k BPE + fertility A/B vs SmolLM2
python train_tokenizer.py --sample ../tok_sample.txt --out ../artifacts/shohin-tok-32k.json
```

## Notes
- **Fertility (bytes/token)** is the *pre-model* tokenizer proxy. True **bits-per-byte** — the other half of
  the master-plan §9 tokenizer A/B — needs the 30M "Mame" proxy model, which waits on GPU.
- Code lives in the repo (source of truth) and is rsync'd to
  `/lustre/fs1/home/sa305415/shohin/pipeline/` on Stokes to run.
