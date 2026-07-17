#!/usr/bin/env python3
"""Evaluate Shohin on the frozen, quarantined Researcher Interview v1 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


EXPECTED_IDS = [f"RIV{index:02d}" for index in range(1, 21)]
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    expected = expected.lower()
    if not HEX_SHA256.fullmatch(expected):
        raise ValueError(f"{label} SHA256 must be 64 lowercase hexadecimal characters")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def normalize_exact(value: str) -> str:
    """Normalize encoding/newlines and outer whitespace, but preserve interface syntax."""
    return (
        unicodedata.normalize("NFC", str(value))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _convert(value: str, kind: str) -> Any:
    if kind == "int":
        return int(value)
    if kind == "string":
        return value
    raise ValueError(f"unsupported field type: {kind}")


def _validate_turn(case_id: str, turn: Mapping[str, Any]) -> None:
    required = {
        "id",
        "prompt",
        "expected_output",
        "semantic_pattern",
        "field_types",
        "expected_state",
        "observable_transitions",
        "unchanged_constraints",
    }
    missing = sorted(required - set(turn))
    if missing:
        raise ValueError(f"{case_id}/{turn.get('id', '?')} missing fields: {missing}")
    pattern = re.compile(str(turn["semantic_pattern"]))
    groups = set(pattern.groupindex)
    expected_fields = set(turn["expected_state"])
    if groups != expected_fields or set(turn["field_types"]) != expected_fields:
        raise ValueError(
            f"{case_id}/{turn['id']} parser fields do not match expected state"
        )
    match = pattern.search(normalize_exact(str(turn["expected_output"])))
    if match is None:
        raise ValueError(
            f"{case_id}/{turn['id']} parser does not accept its gold output"
        )
    parsed = {
        name: _convert(match.group(name), turn["field_types"][name]) for name in groups
    }
    if parsed != turn["expected_state"]:
        raise ValueError(
            f"{case_id}/{turn['id']} parser changes its gold semantic state"
        )
    transition_fields = {
        name
        for transition in turn["observable_transitions"]
        for name in transition["fields"]
    }
    if transition_fields != expected_fields:
        raise ValueError(
            f"{case_id}/{turn['id']} transitions do not cover each expected field"
        )


def load_bank(path: str | Path) -> dict[str, Any]:
    bank = json.loads(Path(path).read_text())
    if (
        bank.get("audit") != "researcher_interview_bank_v1"
        or bank.get("schema_version") != 1
    ):
        raise ValueError("unsupported researcher interview bank")
    quarantine = bank.get("quarantine", {})
    if quarantine.get("training_use") != "forbidden" or not quarantine.get(
        "held_out_evaluation_only"
    ):
        raise ValueError("interview bank is not explicitly quarantined from training")
    cases = bank.get("cases")
    if (
        not isinstance(cases, list)
        or [case.get("id") for case in cases] != EXPECTED_IDS
    ):
        raise ValueError("interview bank must contain ordered RIV01-RIV20 exactly once")
    prompts: set[str] = set()
    turn_count = 0
    for index, case in enumerate(cases, 1):
        expected_parity = "odd" if index % 2 else "even"
        expected_class = "local_control" if expected_parity == "odd" else None
        if case.get("parity") != expected_parity:
            raise ValueError(f"{case['id']} parity does not match its case number")
        if expected_class and case.get("class") != expected_class:
            raise ValueError(f"{case['id']} must remain a local control")
        turns = case.get("turns")
        expected_turns = 2 if case["id"] == "RIV20" else 1
        if not isinstance(turns, list) or len(turns) != expected_turns:
            raise ValueError(f"{case['id']} has the wrong number of turns")
        if case["id"] == "RIV20" and [turn["id"] for turn in turns] != [
            "writer",
            "reader",
        ]:
            raise ValueError("RIV20 must freeze writer then reader")
        for turn in turns:
            prompt = str(turn.get("prompt", ""))
            if not prompt or prompt in prompts:
                raise ValueError(
                    f"duplicate or empty interview prompt: {case['id']}/{turn.get('id')}"
                )
            prompts.add(prompt)
            turn_count += 1
            _validate_turn(case["id"], turn)
    if turn_count != 21 or bank.get("freshness_audit", {}).get("prompt_turns") != 21:
        raise ValueError("Researcher Interview v1 must contain exactly 21 prompt turns")
    return bank


def _parse_turn(
    turn: Mapping[str, Any], normalized: str
) -> tuple[dict[str, Any] | None, re.Match[str] | None]:
    match = re.search(str(turn["semantic_pattern"]), normalized)
    if match is None:
        return None, None
    parsed = {
        name: _convert(match.group(name), kind)
        for name, kind in turn["field_types"].items()
    }
    return parsed, match


def _observe_constraint(
    parsed: Mapping[str, Any], constraint: Mapping[str, Any]
) -> Any:
    value = parsed.get(str(constraint["field"]))
    if value is None:
        return None
    if "index" in constraint:
        index = int(constraint["index"])
        if not isinstance(value, str) or index < 0 or index >= len(value):
            return None
        return value[index]
    return value


def score_turn(turn: Mapping[str, Any], response: str) -> dict[str, Any]:
    """Score visible semantics independently from exact interface/termination control."""
    normalized = normalize_exact(response)
    expected_output = normalize_exact(str(turn["expected_output"]))
    parsed, match = _parse_turn(turn, normalized)
    semantic_correct = parsed == turn["expected_state"]
    exact_correct = normalized == expected_output

    transition_rows = []
    first_divergence = None
    for index, transition in enumerate(turn["observable_transitions"]):
        expected = dict(transition["fields"])
        observed = (
            None if parsed is None else {name: parsed.get(name) for name in expected}
        )
        correct = observed == expected
        row = {
            "index": index,
            "name": transition["name"],
            "expected": expected,
            "observed": observed,
            "correct": correct,
        }
        transition_rows.append(row)
        if first_divergence is None and not correct:
            first_divergence = row

    unchanged_details = []
    for constraint in turn["unchanged_constraints"]:
        observed = None if parsed is None else _observe_constraint(parsed, constraint)
        unchanged_details.append(
            {
                **constraint,
                "observed": observed,
                "correct": observed == constraint["expected"],
            }
        )
    unchanged_scorable = bool(unchanged_details) and parsed is not None
    unchanged_corrupted = unchanged_scorable and any(
        not row["correct"] for row in unchanged_details
    )

    candidate = match.group(0) if match is not None else None
    prefix = normalized[: match.start()].strip() if match is not None else None
    suffix = normalized[match.end() :].strip() if match is not None else None
    extra_token_failure = bool(semantic_correct and (prefix or suffix))
    format_only_failure = bool(
        semantic_correct and not exact_correct and not extra_token_failure
    )
    return {
        "prompt": turn["prompt"],
        "expected_output": turn["expected_output"],
        "raw_output": str(response),
        "normalized_output": normalized,
        "normalized_exact_syntax_correct": exact_correct,
        "parsed_state": parsed,
        "expected_state": turn["expected_state"],
        "semantic_state_correct": semantic_correct,
        "candidate": candidate,
        "extra_prefix": prefix,
        "extra_suffix": suffix,
        "extra_token_termination_failure": extra_token_failure,
        "format_only_failure": format_only_failure,
        "observable_transitions": transition_rows,
        "first_divergent_transition": first_divergence,
        "unchanged_field_check": {
            "applicable": bool(unchanged_details),
            "scorable": unchanged_scorable,
            "corrupted": unchanged_corrupted if unchanged_scorable else None,
            "details": unchanged_details,
        },
    }


def _score_standard_case(
    case: Mapping[str, Any], ask: Callable[[str], str]
) -> dict[str, Any]:
    turn = case["turns"][0]
    scored = score_turn(turn, ask(str(turn["prompt"])))
    return {
        "id": case["id"],
        "parity": case["parity"],
        "class": case["class"],
        "domain": case["domain"],
        **scored,
    }


def _source_deleted_prompt(capsule: str, reader_prompt: str) -> str:
    return f"{capsule}\n{reader_prompt}"


def _score_riv20(case: Mapping[str, Any], ask: Callable[[str], str]) -> dict[str, Any]:
    writer_turn, reader_turn = case["turns"]
    writer = score_turn(writer_turn, ask(str(writer_turn["prompt"])))

    gold_prompt = _source_deleted_prompt(
        str(reader_turn["gold_capsule"]), str(reader_turn["prompt"])
    )
    reader_gold = score_turn(reader_turn, ask(gold_prompt))
    reader_gold["prompt"] = gold_prompt
    reader_gold["capsule_source"] = "gold_reader_control"

    authored_capsule = writer["candidate"]
    reader_self = None
    if authored_capsule is not None:
        self_prompt = _source_deleted_prompt(
            str(authored_capsule), str(reader_turn["prompt"])
        )
        reader_self = score_turn(reader_turn, ask(self_prompt))
        reader_self["prompt"] = self_prompt
        reader_self["capsule_source"] = "model_authored_candidate"

    end_to_end_semantic = bool(
        writer["semantic_state_correct"]
        and reader_self is not None
        and reader_self["semantic_state_correct"]
    )
    end_to_end_exact = bool(
        writer["normalized_exact_syntax_correct"]
        and reader_self is not None
        and reader_self["normalized_exact_syntax_correct"]
    )
    first_divergence = None
    if not writer["semantic_state_correct"]:
        first_divergence = {"phase": "writer", **writer["first_divergent_transition"]}
    elif reader_self is None:
        first_divergence = {
            "phase": "reader",
            "index": 0,
            "name": "source_deleted_read_and_update",
            "expected": reader_turn["expected_state"],
            "observed": None,
            "correct": False,
        }
    elif not reader_self["semantic_state_correct"]:
        first_divergence = {
            "phase": "reader",
            **reader_self["first_divergent_transition"],
        }

    self_extra = bool(reader_self and reader_self["extra_token_termination_failure"])
    self_unchanged = reader_self["unchanged_field_check"] if reader_self else None
    return {
        "id": case["id"],
        "parity": case["parity"],
        "class": case["class"],
        "domain": case["domain"],
        "normalized_exact_syntax_correct": end_to_end_exact,
        "semantic_state_correct": end_to_end_semantic,
        "extra_token_termination_failure": bool(
            writer["extra_token_termination_failure"] or self_extra
        ),
        "first_divergent_transition": first_divergence,
        "unchanged_field_check": writer["unchanged_field_check"],
        "writer": writer,
        "reader_gold_capsule": reader_gold,
        "reader_model_capsule": reader_self,
        "source_deleted_contract": {
            "writer_source_prompt_reused": False,
            "gold_reader_receives_only_gold_capsule_and_reader_prompt": True,
            "end_to_end_reader_receives_only_model_capsule_candidate_and_reader_prompt": True,
            "end_to_end_reader_skipped_without_parseable_capsule": reader_self is None,
            "model_authored_capsule": authored_capsule,
            "reader_unchanged_field_check": self_unchanged,
        },
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(bool(row["normalized_exact_syntax_correct"]) for row in rows)
    semantic = sum(bool(row["semantic_state_correct"]) for row in rows)
    extra = sum(bool(row["extra_token_termination_failure"]) for row in rows)
    applicable = sum(bool(row["unchanged_field_check"]["applicable"]) for row in rows)
    corrupted = sum(row["unchanged_field_check"]["corrupted"] is True for row in rows)
    return {
        "cases": total,
        "normalized_exact_syntax_correct": exact,
        "normalized_exact_syntax_accuracy": exact / total if total else 0.0,
        "semantic_state_correct": semantic,
        "semantic_state_accuracy": semantic / total if total else 0.0,
        "extra_token_termination_failures": extra,
        "unchanged_field_cases": applicable,
        "unchanged_field_corruptions": corrupted,
    }


def evaluate_bank(bank: Mapping[str, Any], ask: Callable[[str], str]) -> dict[str, Any]:
    rows = []
    for case in bank["cases"]:
        row = (
            _score_riv20(case, ask)
            if case["id"] == "RIV20"
            else _score_standard_case(case, ask)
        )
        rows.append(row)
        print(
            f"[researcher-interview] {case['id']} exact={row['normalized_exact_syntax_correct']} "
            f"semantic={row['semantic_state_correct']}",
            flush=True,
        )
    odd = [row for row in rows if row["parity"] == "odd"]
    even = [row for row in rows if row["parity"] == "even"]
    riv20 = rows[-1]
    return {
        "summary": {
            "overall": _summary(rows),
            "odd_local": _summary(odd),
            "even_composition": _summary(even),
        },
        "riv20": {
            "writer": {
                "normalized_exact_syntax_correct": riv20["writer"][
                    "normalized_exact_syntax_correct"
                ],
                "semantic_state_correct": riv20["writer"]["semantic_state_correct"],
            },
            "reader_gold_capsule": {
                "normalized_exact_syntax_correct": riv20["reader_gold_capsule"][
                    "normalized_exact_syntax_correct"
                ],
                "semantic_state_correct": riv20["reader_gold_capsule"][
                    "semantic_state_correct"
                ],
            },
            "end_to_end": {
                "normalized_exact_syntax_correct": riv20[
                    "normalized_exact_syntax_correct"
                ],
                "semantic_state_correct": riv20["semantic_state_correct"],
            },
        },
        "rows": rows,
    }


def load_model(path: str | Path, device: str) -> tuple[dict[str, Any], GPT]:
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def write_json_no_replace(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically while refusing an existing canonical path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--interview", required=True)
    parser.add_argument("--interview-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.max_new <= 0:
        raise SystemExit("--max-new must be positive")
    if not HEX_SHA256.fullmatch(args.source_manifest_sha256.lower()):
        raise SystemExit("--source-manifest-sha256 must be a SHA256 digest")

    hashes = {
        "checkpoint": verify_sha256(args.ckpt, args.checkpoint_sha256, "checkpoint"),
        "tokenizer": verify_sha256(args.tokenizer, args.tokenizer_sha256, "tokenizer"),
        "interview": verify_sha256(args.interview, args.interview_sha256, "interview"),
    }
    bank = load_bank(args.interview)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)

    def ask(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new=args.max_new,
            temp=0.0,
            skip_special_tokens=False,
        )

    scored = evaluate_bank(bank, ask)
    result = {
        "audit": "researcher_interview_eval_v1",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "interview": str(Path(args.interview).resolve()),
        "input_sha256": hashes,
        "source_commit": args.source_commit,
        "source_manifest_sha256": args.source_manifest_sha256.lower(),
        "device": device,
        "seed": args.seed,
        "generation": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new,
            "samples": 1,
        },
        "quarantine": bank["quarantine"],
        "claim_boundary": bank["design"]["claim_boundary"],
        **scored,
    }
    write_json_no_replace(args.out, result)
    print(
        json.dumps(
            {"summary": result["summary"], "riv20": result["riv20"]}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
