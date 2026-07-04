#!/usr/bin/env python
"""Probe the reasoning_gym API: version, available datasets, and sample items —
so gen_reasoning_gym.py can be written against the real metadata structure."""
import reasoning_gym as rg

print("reasoning_gym", getattr(rg, "__version__", "?"))

names = None
if hasattr(rg, "list_datasets"):
    try:
        names = list(rg.list_datasets())
    except Exception as e:
        print("list_datasets err:", e)
if names is None:
    try:
        from reasoning_gym.factory import DATASETS
        names = sorted(DATASETS.keys())
    except Exception as e:
        print("factory err:", e)

print("num datasets:", len(names) if names else None)
if names:
    print("names:", names)
    for fam in names[:6]:
        try:
            d = rg.create_dataset(fam, size=2, seed=0)
            it = list(d)[0]
            print("----", fam, "| keys:", list(it.keys()))
            print("   Q:", str(it.get("question"))[:140].replace("\n", " "))
            print("   A:", str(it.get("answer"))[:80])
            md = it.get("metadata") or {}
            print("   meta:", list(md.keys()))
            if hasattr(d, "score_answer"):
                try:
                    print("   self-score:", d.score_answer(answer=it.get("answer"), entry=it))
                except Exception as e:
                    print("   score err:", e)
        except Exception as e:
            print("----", fam, "ERR", repr(e))
