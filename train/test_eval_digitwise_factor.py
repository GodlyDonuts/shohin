#!/usr/bin/env python3
"""Pure transcript-retention contract for factorized DRS evaluation."""
import sys
import collections
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from eval_digitwise_factor import failure_mode, retain_regime_transcript


by_regime = collections.defaultdict(lambda: {"successes": [], "failures": []})
retain_regime_transcript(by_regime, "fit", {"id": "good"}, True, 1)
retain_regime_transcript(by_regime, "fit", {"id": "second-good"}, True, 1)
retain_regime_transcript(by_regime, "fit", {"id": "bad"}, False, 1)
assert by_regime["fit"] == {"successes": [{"id": "good"}], "failures": [{"id": "bad"}]}
assert set(by_regime) == {"fit"}
assert failure_mode({"state_closed_loop": False, "rows": [{"index": 2}], "final_correct": False}) == "transition_2"
assert failure_mode({"state_closed_loop": True, "rows": [], "final_correct": False}) == "terminal_answer"
assert failure_mode({"state_closed_loop": True, "rows": [], "final_correct": True}) == "success"
print("digitwise factor evaluator transcript checks: passed")
