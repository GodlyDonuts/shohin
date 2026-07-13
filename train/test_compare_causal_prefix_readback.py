"""Synthetic contracts for the causal prefix readback promotion gate."""

from compare_causal_prefix_readback import compare


def metadata(mode):
    return {
        "memory": {
            "init": "raw.pt", "data": "data.jsonl", "data_sha256": "smoke-data", "slots": 8,
            "max_chunks": 8, "seed": 17, "updates": 100, "batch_size": 4, "source_present_at_decode": False,
        },
        "readback": {
            "readback_mode": mode, "decoder_readback_at_every_prefix": True,
            "equal_decoder_work_control": mode == "replicated-final",
        },
    }


def report(name, mode, positive):
    rows = []
    for packet_mode in ("normal", "zero", "shuffled"):
        for regime in ("fit_iid", "length_ood", "language_ood"):
            for chunks in (2, 3, 4):
                for prefix in range(chunks):
                    for key in ("left", "right"):
                        correct = packet_mode == "normal" and positive
                        rows.append({
                            "mode": packet_mode, "reference": "{}:{}:{}:{}".format(regime, chunks, prefix, key),
                            "eval_regime": regime, "chunk_count": chunks, "prefix_index": prefix,
                            "key": key, "correct": correct,
                        })
    meta = metadata(mode)
    return {
        "audit": "causal_prefix_readback_heldout_v1", "checkpoint": name, "data_sha256": "smoke-data",
        "seed": 17, "checkpoint_memory_metadata": meta["memory"],
        "checkpoint_readback_metadata": meta["readback"], "rows": rows,
    }


def main():
    verified = report("verified.pt", "verified", True)
    replay = report("replay.pt", "replicated-final", False)
    labels = report("labels.pt", "shuffled", False)
    result = compare(verified, replay, labels)
    assert result["advance_causal_prefix_readback"] and all(result["gates"].values())
    labels = report("labels.pt", "shuffled", True)
    result = compare(verified, replay, labels)
    assert not result["advance_causal_prefix_readback"]
    print("causal prefix readback comparator contracts passed")


if __name__ == "__main__":
    main()
