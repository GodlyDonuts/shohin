#!/usr/bin/env python3
"""Create read-only FQRB checkpoint decompositions for causal diagnosis.

The FQRB arm trains a source encoder, residual carrier, and normal decoder at
once. These probes restore selected raw-200k components without changing the
learned carrier checkpoint. They distinguish a carrier that is hidden by
decoder/vocabulary drift from a carrier that was never learned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoder_keys(state: dict[str, torch.Tensor], layer: int) -> set[str]:
    """Return the output-side keys after the FQRB residual injection layer."""
    if layer < 0:
        raise ValueError("layer must be nonnegative")
    keys = {"tok.weight", "head.weight", "norm.w"}
    for key in state:
        if not key.startswith("blocks."):
            continue
        pieces = key.split(".", 2)
        if len(pieces) == 3 and int(pieces[1]) > layer:
            keys.add(key)
    return keys


def make_payload(raw: dict, fqrb: dict, keys: set[str], name: str, layer: int, raw_sha: str, fqrb_sha: str) -> dict:
    if raw.get("cfg") != fqrb.get("cfg"):
        raise ValueError("raw and FQRB checkpoint configs differ")
    raw_state, fqrb_state = raw.get("model"), fqrb.get("model")
    if not isinstance(raw_state, dict) or not isinstance(fqrb_state, dict) or set(raw_state) != set(fqrb_state):
        raise ValueError("raw and FQRB state dictionaries do not match")
    if not keys <= set(raw_state):
        raise ValueError("requested replacement keys are missing")
    state = dict(fqrb_state)
    for key in keys:
        state[key] = raw_state[key].clone()
    return {
        "model": state,
        "cfg": fqrb["cfg"],
        "step": "{}+{}".format(fqrb.get("step", "fqrb"), name),
        "fqrb_checkpoint_probe": {
            "name": name,
            "layer": layer,
            "raw_checkpoint_sha256": raw_sha,
            "fqrb_checkpoint_sha256": fqrb_sha,
            "replaced_key_count": len(keys),
            "replaced_keys": sorted(keys),
            "claim_boundary": "Read-only checkpoint decomposition for FQRB diagnosis; not a trained model or capability result.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--fqrb", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=19)
    args = parser.parse_args()
    raw_path, fqrb_path, out_dir = Path(args.raw), Path(args.fqrb), Path(args.out_dir)
    if not raw_path.is_file() or not fqrb_path.is_file():
        raise SystemExit("raw and FQRB checkpoints must exist")
    if out_dir.exists():
        raise SystemExit("refusing existing probe directory: {}".format(out_dir))
    raw_sha, fqrb_sha = sha256_file(raw_path), sha256_file(fqrb_path)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    fqrb = torch.load(fqrb_path, map_location="cpu", weights_only=False)
    raw_state = raw["model"]
    variants = {
        "raw_lexicon": {"tok.weight", "head.weight"},
        "raw_decoder": decoder_keys(raw_state, args.layer),
    }
    out_dir.mkdir(parents=True)
    manifest = {
        "audit": "fqrb_checkpoint_probe_v1",
        "raw_checkpoint_sha256": raw_sha,
        "fqrb_checkpoint_sha256": fqrb_sha,
        "layer": args.layer,
        "variants": {},
        "claim_boundary": "The generated files are diagnostic swaps. They do not modify either source checkpoint.",
    }
    for name, keys in variants.items():
        path = out_dir / "{}.pt".format(name)
        torch.save(make_payload(raw, fqrb, keys, name, args.layer, raw_sha, fqrb_sha), path)
        manifest["variants"][name] = {"path": str(path), "sha256": sha256_file(path), "replaced_key_count": len(keys)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
