#!/usr/bin/env python3
"""Pure transcript-retention contract for factorized DRS evaluation."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from eval_digitwise_factor import retain_regime_transcript


bucket = {"successes": [], "failures": []}
retain_regime_transcript(bucket, {"id": "good"}, True, 1)
retain_regime_transcript(bucket, {"id": "second-good"}, True, 1)
retain_regime_transcript(bucket, {"id": "bad"}, False, 1)
assert bucket == {"successes": [{"id": "good"}], "failures": [{"id": "bad"}]}
print("digitwise factor evaluator transcript checks: passed")
