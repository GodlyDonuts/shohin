#!/usr/bin/env python3
"""Measure verified intermediate reasoning traces on fresh, held-out tasks.

This is deliberately stricter than final-answer accuracy.  Each question asks
for a concise ``<think>`` trace containing one or more hidden numerical
intermediates.  A row receives visible-reasoning credit only when the model
emits the required trace markers *and* the correct final answer.  The markers
are not supplied in the prompt with their values, so they require a real
calculation rather than format imitation.

The audit is evidence for a narrow visible reasoning skill, not a claim that a
model has general intelligence or latent reasoning.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CASES = [
    {
        "id": "product_rejects_27x14",
        "family": "product_then_adjust",
        "question": (
            "A warehouse loads 27 crates with 14 lamps each, then 9 lamps are rejected. "
            "Inside <think> tags, calculate the product and write product=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 369,
        "markers": [["product", 378]],
        "alternate_patterns": [[r"\b27\s*(?:x|\*|times)\s*14\s*=\s*378\b"]],
    },
    {
        "id": "product_bonus_34x16",
        "family": "product_then_adjust",
        "question": (
            "A machine makes 34 trays of 16 parts and receives 7 extra parts. "
            "Inside <think> tags, calculate the product and write product=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 551,
        "markers": [["product", 544]],
        "alternate_patterns": [[r"\b34\s*(?:x|\*|times)\s*16\s*=\s*544\b"]],
    },
    {
        "id": "product_rejects_19x22",
        "family": "product_then_adjust",
        "question": (
            "A studio prints 19 sheets with 22 labels each, then discards 11 labels. "
            "Inside <think> tags, calculate the product and write product=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 407,
        "markers": [["product", 418]],
        "alternate_patterns": [[r"\b19\s*(?:x|\*|times)\s*22\s*=\s*418\b"]],
    },
    {
        "id": "product_bonus_41x13",
        "family": "product_then_adjust",
        "question": (
            "A workshop fills 41 boxes with 13 bolts each and adds 8 spare bolts. "
            "Inside <think> tags, calculate the product and write product=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 541,
        "markers": [["product", 533]],
        "alternate_patterns": [[r"\b41\s*(?:x|\*|times)\s*13\s*=\s*533\b"]],
    },
    {
        "id": "state_add_multiply_subtract_17",
        "family": "state_transition",
        "question": (
            "Start with n=17. Add 8, multiply the result by 3, then subtract 14. "
            "Inside <think> tags, write after_add=<value> and after_multiply=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 61,
        "markers": [["after_add", 25], ["after_multiply", 75]],
        "alternate_patterns": [[r"\b17\s*\+\s*8\s*=\s*25\b"], [r"\b25\s*\*\s*3\s*=\s*75\b"]],
    },
    {
        "id": "state_add_multiply_subtract_23",
        "family": "state_transition",
        "question": (
            "Start with n=23. Add 9, multiply the result by 4, then subtract 7. "
            "Inside <think> tags, write after_add=<value> and after_multiply=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 121,
        "markers": [["after_add", 32], ["after_multiply", 128]],
        "alternate_patterns": [[r"\b23\s*\+\s*9\s*=\s*32\b"], [r"\b32\s*\*\s*4\s*=\s*128\b"]],
    },
    {
        "id": "state_subtract_multiply_add_31",
        "family": "state_transition",
        "question": (
            "Start with n=31. Subtract 12, multiply the result by 5, then add 6. "
            "Inside <think> tags, write after_subtract=<value> and after_multiply=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 101,
        "markers": [["after_subtract", 19], ["after_multiply", 95]],
        "alternate_patterns": [[r"\b31\s*-\s*12\s*=\s*19\b"], [r"\b19\s*\*\s*5\s*=\s*95\b"]],
    },
    {
        "id": "state_double_add_divide_18",
        "family": "state_transition",
        "question": (
            "Start with n=18. Double it, add 12, then divide the result by 4. "
            "Inside <think> tags, write after_double=<value> and after_add=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 12,
        "markers": [["after_double", 36], ["after_add", 48]],
        "alternate_patterns": [[r"\b18\s*\*\s*2\s*=\s*36\b"], [r"\b36\s*\+\s*12\s*=\s*48\b"]],
    },
    {
        "id": "base7_356",
        "family": "base_place_value",
        "question": (
            "Convert the base-7 numeral 356 to base 10. Inside <think> tags, write "
            "hundreds=<value> and tens=<value> for the first two place-value contributions. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 188,
        "markers": [["hundreds", 147], ["tens", 35]],
        "alternate_patterns": [[r"\b3\s*\*\s*(?:7\s*(?:\^|\*\*)\s*2|49)\s*=\s*147\b"], [r"\b5\s*\*\s*7\s*=\s*35\b"]],
    },
    {
        "id": "base8_527",
        "family": "base_place_value",
        "question": (
            "Convert the base-8 numeral 527 to base 10. Inside <think> tags, write "
            "hundreds=<value> and tens=<value> for the first two place-value contributions. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 343,
        "markers": [["hundreds", 320], ["tens", 16]],
        "alternate_patterns": [[r"\b5\s*\*\s*(?:8\s*(?:\^|\*\*)\s*2|64)\s*=\s*320\b"], [r"\b2\s*\*\s*8\s*=\s*16\b"]],
    },
    {
        "id": "repair_product_37x14",
        "family": "trace_repair",
        "question": (
            "A draft incorrectly says that 37 times 14 gives product=512. Check it independently, "
            "then subtract 18. Inside <think> tags, write the corrected product=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 500,
        "markers": [["product", 518]],
        "alternate_patterns": [[r"\b37\s*(?:x|\*|times)\s*14\s*=\s*518\b"]],
    },
    {
        "id": "repair_state_24",
        "family": "trace_repair",
        "question": (
            "A draft says that after adding 11 to n=24, after_add=31. Check the draft, then multiply "
            "the corrected result by 3. Inside <think> tags, write the corrected after_add=<value>. "
            "Then end with 'The answer is <integer>.'."
        ),
        "answer": 105,
        "markers": [["after_add", 35]],
        "alternate_patterns": [[r"\b24\s*\+\s*11\s*=\s*35\b"]],
    },
]


THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
FINAL = re.compile(r"the\s+answer\s+is\s*(-?\d+)\b", re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases_json(path):
    value = json.loads(Path(path).read_text())
    cases = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON must contain a non-empty list")
    required = ("id", "family", "question", "answer", "markers")
    for case in cases:
        if not isinstance(case, dict) or any(field not in case for field in required):
            raise ValueError("every case requires id, family, question, answer, and markers")
        case["answer"] = int(case["answer"])
        if not isinstance(case["markers"], list) or not all(
            isinstance(marker, list) and len(marker) == 2 for marker in case["markers"]
        ):
            raise ValueError("every case marker must be a [name, value] pair")
    return cases


def trace_matches(trace, markers, alternate_patterns=()):
    if trace is None:
        return False
    if alternate_patterns and len(alternate_patterns) != len(markers):
        raise ValueError("alternate_patterns must align with markers")
    for index, (key, value) in enumerate(markers):
        marker = re.search(r"\b{}\s*=\s*{}\b".format(re.escape(key), value), trace, re.IGNORECASE)
        alternatives = alternate_patterns[index] if alternate_patterns else ()
        if not marker and not any(re.search(pattern, trace, re.IGNORECASE) for pattern in alternatives):
            return False
    return True


def score_response(case, response):
    match = THINK.search(response)
    trace = match.group(1).strip() if match else None
    finals = FINAL.findall(response)
    final = int(finals[-1]) if finals else None
    trace_correct = trace_matches(trace, case["markers"], case.get("alternate_patterns", ()))
    answer_correct = final == case["answer"]
    return {
        "trace": trace,
        "final": final,
        "trace_present": trace is not None,
        "trace_correct": trace_correct,
        "answer_correct": answer_correct,
        "correct_trace_and_final": trace_correct and answer_correct,
    }


def load_model(path, device, n_loop=0):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if n_loop:
        cfg.n_loop = n_loop
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def summarize(rows):
    totals = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    for row in rows:
        family = by_family[row["family"]]
        totals["cases"] += 1
        family["cases"] += 1
        for key in ("trace_present", "trace_correct", "answer_correct", "correct_trace_and_final"):
            totals[key] += int(row[key])
            family[key] += int(row[key])
    return {
        **dict(totals),
        "by_family": {name: dict(values) for name, values in sorted(by_family.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--cases-json", default="",
                        help="optional generated JSON case file; default uses the fixed literal audit")
    parser.add_argument("--n-loop", type=int, default=0,
                        help="test-only latent-depth override (0 preserves checkpoint config)")
    args = parser.parse_args()
    if args.n_loop < 0:
        raise SystemExit("--n-loop must be zero or positive")

    if args.cases_json:
        cases = load_cases_json(args.cases_json)
        case_source = args.cases_json
        case_source_sha256 = sha256_file(args.cases_json)
    else:
        cases = CASES
        case_source = __file__
        case_source_sha256 = sha256_file(__file__)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device, n_loop=args.n_loop)
    used_n_loop = model.cfg.n_loop
    rows = []
    for case in cases:
        prompt = "Question: {}\nAnswer:".format(case["question"])
        response = generate(
            model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0,
            skip_special_tokens=False,
        )
        row = dict(case)
        row["prompt"] = prompt
        row["response"] = response
        row.update(score_response(case, response))
        rows.append(row)
        print(
            "[thinking-trace] {} trace={} answer={} both={}".format(
                row["id"], row["trace_correct"], row["answer_correct"], row["correct_trace_and_final"]
            ),
            flush=True,
        )

    result = {
        "audit": "thinking_trace_audit_v1",
        "case_source": case_source,
        "case_source_sha256": case_source_sha256,
        "device": device,
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "n_loop": used_n_loop,
        "cases": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[thinking-trace] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
