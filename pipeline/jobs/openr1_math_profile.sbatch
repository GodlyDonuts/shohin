#!/bin/bash
#SBATCH -J shohin-openr1-profile
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openr1_math_profile_%j.out
#SBATCH --open-mode=append
# Bounded profile of verified OpenR1 trace lengths. This never emits training data.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
LIMIT=${LIMIT:-256}

cd "$BASE/pipeline"
"$PY" - <<PY
import re
from datasets import load_dataset
from tokenizers import Tokenizer
from curate_openr1_math import choose_generation, clean_trace

tokenizer = Tokenizer.from_file("$BASE/artifacts/shohin-tok-32k.json")
stream = load_dataset("open-r1/OpenR1-Math-220k", name="default", split="train", streaming=True)
lengths = []
solution_lengths = []
answer_in_solution = 0
complete = verified = 0
for index, row in enumerate(stream):
    if index >= int("$LIMIT"):
        break
    selected = choose_generation(row)
    if selected is None:
        continue
    verified += 1
    trace, source = selected
    lengths.append(len(tokenizer.encode(clean_trace(trace)).ids))
    solution = str(row.get("solution") or "")
    answer = str(row.get("answer") or "")
    if solution:
        solution_lengths.append(len(tokenizer.encode(solution).ids))
        norm_solution = re.sub(r"\s+", "", solution).replace("$", "")
        norm_answer = re.sub(r"\s+", "", answer).replace("$", "")
        answer_in_solution += int(bool(norm_answer) and norm_answer in norm_solution)
    complete += 1

def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

print({
    "seen": min(index + 1, int("$LIMIT")),
    "verified_complete": verified,
    "lengths": {"min": min(lengths, default=None), "p50": percentile(lengths, .5),
                "p90": percentile(lengths, .9), "p99": percentile(lengths, .99),
                "max": max(lengths, default=None)},
    "under_512": sum(length <= 512 for length in lengths),
    "under_1024": sum(length <= 1024 for length in lengths),
    "under_2048": sum(length <= 2048 for length in lengths),
    "solution_lengths": {"min": min(solution_lengths, default=None), "p50": percentile(solution_lengths, .5),
                         "p90": percentile(solution_lengths, .9), "p99": percentile(solution_lengths, .99),
                         "max": max(solution_lengths, default=None)},
    "solutions_under_512": sum(length <= 512 for length in solution_lengths),
    "answer_textually_in_solution": answer_in_solution,
})
PY
