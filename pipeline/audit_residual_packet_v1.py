#!/usr/bin/env python3
"""Fail-closed independent admission audit for RSP-C1 generation artifacts.

This module intentionally does not import the generator.  It independently
replays every frozen RNG stream, reconstructs the sham permutation, reparses
all protocol channels, and mirrors the exact completion boundary and packing
rule used by ``train/sft.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import stat
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "train" / "residual_packet_protocol.py"
SFT_ENCODING_PATH = ROOT / "train" / "sft_encoding.py"

BOARD_SCHEMA = "residual_packet_board_v1"
MANIFEST_SCHEMA = "residual_packet_generation_manifest_v1"
AUDIT_SCHEMA = "residual_packet_admission_audit_v1"
BOARD_SEED = 2026071503
TRAIN_SEED = 2026071504
OBSERVATION_SEED = 2026071505
SHAM_SEED = 2026071506
PACK_LENGTH = 128
PER_STRATUM = 64
STRATUM_ORDER = ("renderer_ood", "value_ood", "order_ood", "length_ood")
TRAIN_LENGTH_COUNTS = {2: 1024, 3: 2048, 4: 1024}
OPERATION_TYPES = ("add", "multiply", "subtract")
HELD_OUT_BIGRAMS = (("multiply", "add"), ("subtract", "multiply"))
EXPECTED_BOARD_ROWS_SHA256 = "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
EXPECTED_BOARD_SHA256 = "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
EXPECTED_PROTOCOL_SHA256 = "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2"
EXPECTED_SFT_ENCODING_SHA256 = "33d07db19afb7155567cef922dbab616916864eafaeba7fa26f12d2203c0d5b8"
TOKEN_RE = re.compile(r"[a-z]+|\d+")
INTEGER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")
CANONICAL_INTEGER_RE = re.compile(r"(?:0|-?[1-9]\d*)\Z", re.ASCII)

COMPILER_FIELDS = {
    "completion_prompt",
    "id",
    "kind",
    "program_id",
    "question",
    "response",
    "training_group",
}
UPDATER_FIELDS = COMPILER_FIELDS | {"step"}
_PROTOCOL = None
_SFT_ENCODING = None


def _load_hashed_repo_module(
    path: Path, expected_sha256: str, module_label: str
):
    """Compile exact hashed bytes from one non-symlinked repo file."""

    resolved = path.resolve(strict=False)
    if resolved != path or resolved.parent.parent != ROOT:
        raise RuntimeError(f"{module_label} path escaped the repository")
    raw = None
    observed_sha256 = None
    for attempt in range(20):
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"{module_label} is not a regular repository file")
            raw = path.read_bytes()
        except FileNotFoundError:
            if attempt == 19:
                raise
            time.sleep(0.05)
            continue
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_sha256 == expected_sha256:
            break
        if attempt == 19:
            raise RuntimeError(
                f"{module_label} SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
            )
        time.sleep(0.05)
    if raw is None or observed_sha256 != expected_sha256:
        raise RuntimeError(f"could not load frozen {module_label}")
    module_name = f"_shohin_audit_{module_label}_{observed_sha256[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, resolved)
    if specification is None:
        raise ImportError(f"could not create a module specification for {resolved}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(resolved), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if Path(module.__file__).resolve() != resolved:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"{module_label} loaded from the wrong path")
    module.__source_sha256__ = observed_sha256
    return module


def protocol_module():
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = _load_hashed_repo_module(
            PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "residual_packet_protocol"
        )
    return _PROTOCOL


def sft_encoding_module():
    global _SFT_ENCODING
    if _SFT_ENCODING is None:
        _SFT_ENCODING = _load_hashed_repo_module(
            SFT_ENCODING_PATH, EXPECTED_SFT_ENCODING_SHA256, "sft_encoding"
        )
    return _SFT_ENCODING


def training_template_ids() -> tuple[str, ...]:
    protocol = protocol_module()
    for name in (
        "TRAIN_SOURCE_TEMPLATE_IDS",
        "TRAINING_TEMPLATE_IDS",
        "TRAIN_TEMPLATE_IDS",
        "TRAIN_SOURCE_TEMPLATES",
    ):
        if hasattr(protocol, name):
            value = getattr(protocol, name)
            if isinstance(value, dict) or hasattr(value, "keys"):
                value = value.keys()
            result = tuple(str(item) for item in value)
            if result and len(result) == len(set(result)):
                return result
    raise AttributeError("protocol has no unique training source template identifiers")


def reserved_template_id() -> str:
    protocol = protocol_module()
    for name in (
        "RESERVED_SOURCE_TEMPLATE_ID",
        "RESERVED_TEMPLATE_ID",
        "RESERVED_RENDERER_ID",
    ):
        if hasattr(protocol, name):
            return str(getattr(protocol, name))
    raise AttributeError("protocol has no reserved source template identifier")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "ascii"
    )


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_immutable(path: str | Path, label: str) -> bytes:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if info.st_mode & 0o222:
        raise PermissionError(f"{label} is writable")
    return source.read_bytes()


def read_immutable_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_immutable(path, label)
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != pretty_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value, raw


def read_immutable_jsonl(path: str | Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_immutable(path, label)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} is empty or lacks its final newline")
    rows = []
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {label} JSONL row {number}") from error
        if not isinstance(row, dict) or line != canonical_json_bytes(row):
            raise ValueError(f"{label} row {number} is not canonical JSONL")
        rows.append(row)
    return rows, raw


def apply_operation(state: int, operation: Sequence[Any]) -> int:
    kind, operand = str(operation[0]), int(operation[1])
    if kind == "add":
        return state + operand
    if kind == "multiply":
        return state * operand
    if kind == "subtract":
        return state - operand
    raise ValueError(f"invalid operation {kind!r}")


def trajectory(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    result = [int(initial_state)]
    for operation in operations:
        result.append(apply_operation(result[-1], operation))
    return tuple(result)


def semantic_signature(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    return (int(initial_state),) + tuple((str(kind), int(operand)) for kind, operand in operations)


def operation_types(operations: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(operation[0]) for operation in operations)


def digit_width_vector(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    return (len(str(initial_state)),) + tuple(len(str(int(operation[1]))) for operation in operations)


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(str(text).lower()))


def normalized_prompt(text: str) -> str:
    return " ".join(normalized_tokens(text))


def ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    return {tokens[index : index + width] for index in range(len(tokens) - width + 1)}


def integer_occurrences(text: str) -> tuple[int, ...]:
    return tuple(int(match.group(0)) for match in INTEGER_RE.finditer(str(text)))


def _sequence_pool(length: int, required_holdout: tuple[str, str] | None = None) -> tuple[tuple[str, ...], ...]:
    candidates = [()]
    for _ in range(length):
        candidates = [prefix + (operation,) for prefix in candidates for operation in OPERATION_TYPES]
    accepted = []
    for candidate in candidates:
        held = [pair for pair in zip(candidate, candidate[1:]) if pair in HELD_OUT_BIGRAMS]
        if required_holdout is None:
            if not held:
                accepted.append(candidate)
        elif held == [required_holdout]:
            accepted.append(candidate)
    if not accepted:
        raise RuntimeError("empty independently reconstructed sequence pool")
    return tuple(accepted)


def _draw_operand(
    rng: random.Random,
    operation: str,
    *,
    value_ood: bool = False,
    width: int | None = None,
) -> int:
    if value_ood:
        low, high = ((8, 12) if operation == "multiply" else (26, 75))
    else:
        low, high = ((2, 7) if operation == "multiply" else (2, 25))
    if width is not None:
        low = max(low, 10 ** (width - 1) if width > 1 else 0)
        high = min(high, 10**width - 1)
        if low > high:
            raise ValueError("impossible operand width")
    return rng.randint(low, high)


def _draw_program(
    rng: random.Random,
    types: Sequence[str],
    *,
    value_ood: bool = False,
    widths: Sequence[int] | None = None,
) -> tuple[int, list[list[Any]]]:
    if widths is None:
        initial = rng.randint(100, 299) if value_ood else rng.randint(10, 99)
        plan = [[kind, _draw_operand(rng, kind, value_ood=value_ood)] for kind in types]
    else:
        initial = rng.randint(
            max(10, 10 ** (int(widths[0]) - 1)),
            min(99, 10 ** int(widths[0]) - 1),
        )
        plan = [
            [kind, _draw_operand(rng, kind, width=int(width))]
            for kind, width in zip(types, widths[1:])
        ]
    return initial, plan


def _source(initial: int, plan: Sequence[Sequence[Any]], template_id: str) -> str:
    return str(protocol_module().render_source(initial, plan, template_id))


def _packet(state: int, plan: Sequence[Sequence[Any]]) -> str:
    return str(protocol_module().canonical_packet(state, plan))


def independently_rebuild_board() -> dict[str, Any]:
    rng = random.Random(BOARD_SEED)
    templates = training_template_ids()
    reserved = reserved_template_id()
    rows: list[dict[str, Any]] = []
    programs: set[tuple[Any, ...]] = set()
    sources: set[str] = set()
    packets: set[str] = set()
    trajectories: set[tuple[int, ...]] = set()
    answers: set[int] = set()
    for stratum in STRATUM_ORDER:
        for index in range(PER_STRATUM):
            for _ in range(100_000):
                if stratum == "renderer_ood":
                    types = rng.choice(_sequence_pool(3))
                    template_id = reserved
                    initial, plan = _draw_program(rng, types)
                elif stratum == "value_ood":
                    types = rng.choice(_sequence_pool(3))
                    template_id = templates[index % len(templates)]
                    initial, plan = _draw_program(rng, types, value_ood=True)
                elif stratum == "order_ood":
                    held = HELD_OUT_BIGRAMS[0 if index < 32 else 1]
                    length = 3 if index % 32 % 2 == 0 else 4
                    types = rng.choice(_sequence_pool(length, held))
                    template_id = templates[index % len(templates)]
                    initial, plan = _draw_program(rng, types)
                else:
                    types = rng.choice(_sequence_pool(5))
                    template_id = templates[index % len(templates)]
                    initial, plan = _draw_program(rng, types)
                states = trajectory(initial, plan)
                if min(states) <= 0 or states[-1] < 100:
                    continue
                rendered_source = _source(initial, plan, template_id)
                rendered_packet = _packet(initial, plan)
                signature = semantic_signature(initial, plan)
                if (
                    signature in programs
                    or rendered_source in sources
                    or rendered_packet in packets
                    or states in trajectories
                    or states[-1] in answers
                ):
                    continue
                rows.append(
                    {
                        "answer": states[-1],
                        "id": f"{stratum}_{index:03d}",
                        "initial_state": initial,
                        "operations": [list(operation) for operation in plan],
                        "packet": rendered_packet,
                        "source": rendered_source,
                        "stratum": stratum,
                        "template_id": template_id,
                        "trajectory": list(states),
                    }
                )
                programs.add(signature)
                sources.add(rendered_source)
                packets.add(rendered_packet)
                trajectories.add(states)
                answers.add(states[-1])
                break
            else:
                raise RuntimeError("independent board reconstruction exhausted attempts")
    return {
        "case_count": len(rows),
        "per_stratum": PER_STRATUM,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "rows": rows,
        "rows_sha256": digest_bytes(canonical_json_bytes(rows)),
        "schema": BOARD_SCHEMA,
        "seed": BOARD_SEED,
        "stratum_order": list(STRATUM_ORDER),
    }


def _program_record(initial: int, plan: Sequence[Sequence[Any]], template_id: str, tokenizer, pair: int) -> dict[str, Any]:
    states = trajectory(initial, plan)
    source = _source(initial, plan, template_id)
    packet = _packet(initial, plan)
    return {
        "final_answer": states[-1],
        "initial_state": initial,
        "operations": [list(operation) for operation in plan],
        "packet": packet,
        "packet_token_count": len(tokenizer.encode(packet).ids),
        "pair_id": pair,
        "source": source,
        "template_id": template_id,
        "trajectory": states,
    }


def _training_candidate_is_admissible(
    candidate: dict[str, Any],
    board_sets: dict[str, set[Any]],
    used: dict[str, set[Any]],
) -> bool:
    plan = candidate["operations"]
    signature = semantic_signature(candidate["initial_state"], plan)
    states = tuple(candidate["trajectory"])
    packet_values = {candidate["initial_state"]} | {int(operation[1]) for operation in plan}
    return (
        min(states) > 0
        and candidate["final_answer"] not in packet_values
        and not (packet_values & board_sets["answers"])
        and signature not in board_sets["programs"]
        and candidate["source"] not in board_sets["sources"]
        and candidate["packet"] not in board_sets["packets"]
        and states not in board_sets["trajectories"]
        and candidate["final_answer"] not in board_sets["answers"]
        and not (ngrams(candidate["source"]) & board_sets["grams"])
        and signature not in used["programs"]
        and candidate["source"] not in used["sources"]
        and candidate["packet"] not in used["packets"]
        and states not in used["trajectories"]
    )


def independently_rebuild_programs(board_rows: Sequence[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    board_sources = {str(row["source"]) for row in board_rows}
    board_sets: dict[str, set[Any]] = {
        "answers": {int(row["answer"]) for row in board_rows},
        "programs": {
            semantic_signature(int(row["initial_state"]), row["operations"]) for row in board_rows
        },
        "sources": board_sources,
        "packets": {str(row["packet"]) for row in board_rows},
        "trajectories": {tuple(map(int, row["trajectory"])) for row in board_rows},
        "grams": set().union(*(ngrams(source) for source in board_sources)),
    }
    used: dict[str, set[Any]] = {
        "programs": set(),
        "sources": set(),
        "packets": set(),
        "trajectories": set(),
    }
    rng = random.Random(TRAIN_SEED)
    templates = training_template_ids()
    programs: list[dict[str, Any]] = []
    pair = 0
    for length in sorted(TRAIN_LENGTH_COUNTS):
        for local_pair in range(TRAIN_LENGTH_COUNTS[length] // 2):
            for _ in range(10_000):
                types = rng.choice(_sequence_pool(length))
                template_id = templates[local_pair % len(templates)]
                initial, plan = _draw_program(rng, types)
                first = _program_record(initial, plan, template_id, tokenizer, pair)
                if not _training_candidate_is_admissible(first, board_sets, used):
                    continue
                widths = digit_width_vector(initial, plan)
                first_signature = semantic_signature(initial, plan)
                for _ in range(10_000):
                    partner_initial, partner_plan = _draw_program(rng, types, widths=widths)
                    second = _program_record(
                        partner_initial, partner_plan, template_id, tokenizer, pair
                    )
                    if (
                        second["packet_token_count"] != first["packet_token_count"]
                        or len(str(second["final_answer"])) != len(str(first["final_answer"]))
                        or second["final_answer"] == first["final_answer"]
                        or first["final_answer"] in integer_occurrences(second["packet"])
                        or second["final_answer"] in integer_occurrences(first["packet"])
                        or semantic_signature(partner_initial, partner_plan) == first_signature
                    ):
                        continue
                    temporary = {key: set(value) for key, value in used.items()}
                    temporary["programs"].add(first_signature)
                    temporary["sources"].add(first["source"])
                    temporary["packets"].add(first["packet"])
                    temporary["trajectories"].add(tuple(first["trajectory"]))
                    if not _training_candidate_is_admissible(second, board_sets, temporary):
                        continue
                    for program in (first, second):
                        used["programs"].add(
                            semantic_signature(program["initial_state"], program["operations"])
                        )
                        used["sources"].add(program["source"])
                        used["packets"].add(program["packet"])
                        used["trajectories"].add(tuple(program["trajectory"]))
                        programs.append(program)
                    pair += 1
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError("independent training reconstruction exhausted attempts")
    for index, program in enumerate(programs):
        program["id"] = f"train_{index:04d}"
    return programs


def sham_stratum(program: dict[str, Any]) -> tuple[Any, ...]:
    return (
        len(program["operations"]),
        operation_types(program["operations"]),
        program["template_id"],
        digit_width_vector(program["initial_state"], program["operations"]),
        int(program["packet_token_count"]),
        len(str(program["final_answer"])),
    )


def independently_rebuild_sham_mapping(
    programs: Sequence[dict[str, Any]], seed: int = SHAM_SEED
) -> tuple[int, ...]:
    by_stratum: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, program in enumerate(programs):
        by_stratum[sham_stratum(program)].append(index)
    rng = random.Random(seed)
    mapping = [-1] * len(programs)
    for key in sorted(by_stratum):
        order = list(by_stratum[key])
        if len(order) < 2:
            raise ValueError(f"singleton sham stratum {key!r}")
        rng.shuffle(order)
        preferences: dict[int, list[int]] = {}
        for recipient in order:
            choices = [
                donor
                for donor in order
                if donor != recipient
                and programs[recipient]["final_answer"]
                != programs[donor]["final_answer"]
                and programs[recipient]["final_answer"]
                not in integer_occurrences(programs[donor]["packet"])
            ]
            rng.shuffle(choices)
            preferences[recipient] = choices
        owner: dict[int, int] = {}

        def assign(recipient: int, visited: set[int]) -> bool:
            for donor in preferences[recipient]:
                if donor in visited:
                    continue
                visited.add(donor)
                previous = owner.get(donor)
                if previous is None or assign(previous, visited):
                    owner[donor] = recipient
                    return True
            return False

        if any(not assign(recipient, set()) for recipient in order):
            raise ValueError(f"unmatchable sham stratum {key!r}")
        for donor, recipient in owner.items():
            mapping[recipient] = donor
    if sorted(mapping) != list(range(len(programs))) or any(
        index == donor for index, donor in enumerate(mapping)
    ):
        raise ValueError("reconstructed sham mapping is not a derangement")
    return tuple(mapping)


def _draw_observation(
    rng: random.Random,
    operation: Sequence[Any],
    eval_answers: set[int],
    source_answer: int,
) -> tuple[int, int]:
    forbidden = eval_answers | {source_answer}
    for _ in range(100_000):
        state, observed = rng.randint(1000, 9999), rng.randint(1000, 9999)
        if state in forbidden or observed in forbidden:
            continue
        if observed != apply_operation(state, operation):
            return state, observed
    raise RuntimeError("independent observation reconstruction exhausted attempts")


def independently_rebuild_arms(
    programs: Sequence[dict[str, Any]],
    board_rows: Sequence[dict[str, Any]],
    mapping: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = protocol_module()
    rng = random.Random(OBSERVATION_SEED)
    eval_answers = {int(row["answer"]) for row in board_rows}
    treatment: list[dict[str, Any]] = []
    sham: list[dict[str, Any]] = []
    for index, program in enumerate(programs):
        prompt = protocol.compiler_prompt(program["source"])
        common = {
            "completion_prompt": prompt,
            "id": f"{program['id']}_compiler",
            "kind": "compiler",
            "program_id": program["id"],
            "question": prompt,
            "training_group": "residual_packet",
        }
        treatment.append({**common, "response": program["packet"]})
        sham.append({**common, "response": programs[mapping[index]]["packet"]})
        for step, operation in enumerate(program["operations"]):
            state, observed = _draw_observation(
                rng, operation, eval_answers, int(program["final_answer"])
            )
            packet = protocol.canonical_packet(state, program["operations"][step:])
            update_prompt = protocol.update_prompt(packet, observed)
            response = protocol.expected_update(packet, observed)
            row = {
                "completion_prompt": update_prompt,
                "id": f"{program['id']}_updater_{step:02d}",
                "kind": "updater",
                "program_id": program["id"],
                "question": update_prompt,
                "response": response,
                "step": step,
                "training_group": "residual_packet",
            }
            treatment.append(row)
            sham.append(dict(row))
    return treatment, sham


def token_accounting(rows: Sequence[dict[str, Any]], tokenizer, eos_id: int) -> dict[str, int]:
    encode_supervised_example = sft_encoding_module().encode_supervised_example

    totals = Counter()
    for row in rows:
        prompt = str(row.get("completion_prompt", ""))
        response = str(row.get("response", "")).rstrip()
        prompt_ids, token_ids, completion_mask = encode_supervised_example(
            tokenizer, prompt, response, eos_id
        )
        completion_count = len(token_ids) - len(prompt_ids) - 1
        if sum(completion_mask) != completion_count + 1:
            raise ValueError("SFT completion boundary mismatch")
        full_count = len(token_ids)
        if full_count > PACK_LENGTH:
            totals["skipped_too_long"] += 1
            continue
        totals["examples"] += 1
        totals["prompt_token_count"] += len(prompt_ids)
        totals["response_token_count"] += completion_count
        totals["supervised_token_count"] += sum(completion_mask)
        totals["full_token_count"] += full_count
    packed = max(0, (totals["full_token_count"] - 2) // PACK_LENGTH)
    return {
        "compiler_rows": sum(row.get("kind") == "compiler" for row in rows),
        "examples": totals["examples"],
        "forward_token_count": packed * PACK_LENGTH,
        "full_token_count": totals["full_token_count"],
        "packed_sequence_count": packed,
        "prompt_token_count": totals["prompt_token_count"],
        "response_token_count": totals["response_token_count"],
        "skipped_too_long": totals["skipped_too_long"],
        "supervised_token_count": totals["supervised_token_count"],
        "updater_rows": sum(row.get("kind") == "updater" for row in rows),
    }


def parse_updater_prompt(text: str):
    prefix, marker, suffix = "Packet:\n", "\nObserved result: ", "\nNext packet:"
    if not isinstance(text, str) or not text.startswith(prefix) or not text.endswith(suffix):
        return None
    body = text[len(prefix) : -len(suffix)]
    packet_text, separator, observed_text = body.rpartition(marker)
    if not separator or not CANONICAL_INTEGER_RE.fullmatch(observed_text):
        return None
    packet = protocol_module().parse_packet(packet_text)
    if packet is None:
        return None
    return packet_text, packet, int(observed_text)


def _extract_source(prompt: str) -> str | None:
    prefix = "Problem: "
    suffix = "\nCompile only the execution packet.\nPacket:"
    if not isinstance(prompt, str) or not prompt.startswith(prefix) or not prompt.endswith(suffix):
        return None
    return prompt[len(prefix) : -len(suffix)]


def _board_invariant_failures(rows: Sequence[dict[str, Any]]) -> list[str]:
    failures = []
    templates = set(training_template_ids())
    reserved = reserved_template_id()
    if len(rows) != 256:
        failures.append("board_case_count")
        return failures
    if Counter(row.get("stratum") for row in rows) != Counter({name: 64 for name in STRATUM_ORDER}):
        failures.append("board_stratum_counts")
    if [row.get("stratum") for row in rows] != [name for name in STRATUM_ORDER for _ in range(64)]:
        failures.append("board_stratum_order")
    programs, sources, trajectories, answers = set(), set(), set(), set()
    held_counts = Counter()
    for row in rows:
        try:
            initial, plan = int(row["initial_state"]), row["operations"]
            states = trajectory(initial, plan)
            types = operation_types(plan)
            signature = semantic_signature(initial, plan)
            source = _source(initial, plan, str(row["template_id"]))
            packet = _packet(initial, plan)
        except (KeyError, TypeError, ValueError):
            failures.append("malformed_board_rows")
            continue
        if row.get("source") != source or row.get("packet") != packet or row.get("trajectory") != list(states) or row.get("answer") != states[-1]:
            failures.append("malformed_board_rows")
        if min(states) <= 0:
            failures.append("nonpositive_board_intermediate")
        if signature in programs or source in sources or states in trajectories or states[-1] in answers:
            failures.append("duplicate_board_objects")
        programs.add(signature)
        sources.add(source)
        trajectories.add(states)
        answers.add(states[-1])
        stratum = row.get("stratum")
        held = [pair for pair in zip(types, types[1:]) if pair in HELD_OUT_BIGRAMS]
        operands = [int(operation[1]) for operation in plan]
        if stratum == "renderer_ood":
            if len(plan) != 3 or held or row.get("template_id") != reserved:
                failures.append("renderer_ood_contract")
        elif stratum == "value_ood":
            if len(plan) != 3 or held or row.get("template_id") not in templates or not 100 <= initial <= 299:
                failures.append("value_ood_contract")
            if any(not (8 <= operand <= 12) if kind == "multiply" else not (26 <= operand <= 75) for (kind, _), operand in zip(plan, operands)):
                failures.append("value_ood_contract")
        elif stratum == "order_ood":
            if len(plan) not in (3, 4) or len(held) != 1 or row.get("template_id") not in templates:
                failures.append("order_ood_contract")
            elif held:
                held_counts[held[0]] += 1
        elif stratum == "length_ood":
            if len(plan) != 5 or held or row.get("template_id") not in templates:
                failures.append("length_ood_contract")
        if stratum != "value_ood":
            if not 10 <= initial <= 99:
                failures.append("training_domain_board_values")
            for kind, operand in plan:
                low, high = ((2, 7) if kind == "multiply" else (2, 25))
                if not low <= int(operand) <= high:
                    failures.append("training_domain_board_values")
    if held_counts != Counter({HELD_OUT_BIGRAMS[0]: 32, HELD_OUT_BIGRAMS[1]: 32}):
        failures.append("order_ood_holdout_balance")
    return sorted(set(failures))


def _row_invariants(
    treatment: Sequence[dict[str, Any]],
    sham: Sequence[dict[str, Any]],
    programs: Sequence[dict[str, Any]],
    mapping: Sequence[int],
    board_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    protocol = protocol_module()
    program_by_id = {program["id"]: program for program in programs}
    packet_to_index = {program["packet"]: index for index, program in enumerate(programs)}
    eval_answers = {int(row["answer"]) for row in board_rows}
    malformed_compilers = malformed_updaters = correct_transitions = 0
    source_final_occurrences = eval_answer_occurrences = 0
    actual_mapping: dict[int, int] = {}
    updater_pairs_identical = 0
    compiler_controls_matched = 0
    for left, right in zip(treatment, sham):
        if left.get("kind") == "compiler":
            if set(left) != COMPILER_FIELDS or set(right) != COMPILER_FIELDS:
                malformed_compilers += 1
                continue
            program = program_by_id.get(left.get("program_id"))
            parsed = protocol.parse_packet(left.get("response"))
            source = _extract_source(left.get("completion_prompt"))
            if (
                program is None
                or parsed is None
                or source != program["source"]
                or left.get("completion_prompt") != protocol.compiler_prompt(source)
                or left.get("question") != left.get("completion_prompt")
                or left.get("training_group") != "residual_packet"
                or left.get("response") != program["packet"]
            ):
                malformed_compilers += 1
            donor = packet_to_index.get(right.get("response"))
            recipient = int(program["id"].split("_")[1]) if program is not None else -1
            if donor is None or recipient < 0:
                malformed_compilers += 1
            else:
                actual_mapping[recipient] = donor
            left_control = {key: value for key, value in left.items() if key != "response"}
            right_control = {key: value for key, value in right.items() if key != "response"}
            if left_control == right_control:
                compiler_controls_matched += 1
            else:
                malformed_compilers += 1
        elif left.get("kind") == "updater":
            if left == right:
                updater_pairs_identical += 1
            if set(left) != UPDATER_FIELDS or set(right) != UPDATER_FIELDS or left != right:
                malformed_updaters += 1
                continue
            parsed = parse_updater_prompt(left.get("completion_prompt"))
            program = program_by_id.get(left.get("program_id"))
            if parsed is None or program is None:
                malformed_updaters += 1
                continue
            packet_text, packet, observed = parsed
            step = left.get("step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or step >= len(program["operations"])
                or tuple(packet["plan"]) != tuple(tuple(item) for item in program["operations"][step:])
                or left.get("question") != left.get("completion_prompt")
                or left.get("completion_prompt") != protocol.update_prompt(packet_text, observed)
                or left.get("response") != protocol.expected_update(packet_text, observed)
                or left.get("training_group") != "residual_packet"
            ):
                malformed_updaters += 1
            if observed == apply_operation(int(packet["state"]), packet["plan"][0]):
                correct_transitions += 1
        else:
            malformed_compilers += 1
        for row in (left, right):
            values = set(integer_occurrences(str(row.get("response", ""))))
            eval_answer_occurrences += len(values & eval_answers)
            program = program_by_id.get(row.get("program_id"))
            if program is not None and int(program["final_answer"]) in values:
                source_final_occurrences += 1
    expected_mapping = {index: donor for index, donor in enumerate(mapping)}
    return {
        "compiler_controls_matched": compiler_controls_matched,
        "correct_arithmetic_updater_transitions": correct_transitions,
        "evaluation_answer_response_occurrences": eval_answer_occurrences,
        "malformed_compiler_rows": malformed_compilers,
        "malformed_updater_rows": malformed_updaters,
        "reconstructed_sham_mapping_mismatches": sum(
            actual_mapping.get(index) != donor for index, donor in expected_mapping.items()
        ),
        "source_final_answer_response_occurrences": source_final_occurrences,
        "updater_pairs_byte_identical": updater_pairs_identical,
    }


def audit_artifacts(
    board_path: str | Path,
    tokenizer_path: str | Path,
    treatment_path: str | Path,
    sham_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    board, board_raw = read_immutable_json(board_path, "board")
    treatment, treatment_raw = read_immutable_jsonl(treatment_path, "treatment")
    sham, sham_raw = read_immutable_jsonl(sham_path, "sham")
    manifest, manifest_raw = read_immutable_json(manifest_path, "manifest")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer has no <|endoftext|> token")

    failures: list[str] = []
    rebuilt_board = independently_rebuild_board()
    if board != rebuilt_board:
        failures.append("board_seed_reconstruction")
    if digest_bytes(board_raw) != EXPECTED_BOARD_SHA256:
        failures.append("board_artifact_sha256")
    rows_digest = digest_bytes(canonical_json_bytes(board.get("rows")))
    if rows_digest != EXPECTED_BOARD_ROWS_SHA256 or board.get("rows_sha256") != rows_digest:
        failures.append("board_rows_sha256")
    failures.extend(_board_invariant_failures(board.get("rows", ())))

    programs = independently_rebuild_programs(rebuilt_board["rows"], tokenizer)
    mapping = independently_rebuild_sham_mapping(programs)
    expected_treatment, expected_sham = independently_rebuild_arms(
        programs, rebuilt_board["rows"], mapping
    )
    if treatment != expected_treatment:
        failures.append("treatment_seed_reconstruction")
    if sham != expected_sham:
        failures.append("sham_seed_reconstruction")
    if len(treatment) != 16_384 or len(sham) != 16_384:
        failures.append("training_row_counts")

    treatment_tokens = token_accounting(treatment, tokenizer, eos_id)
    sham_tokens = token_accounting(sham, tokenizer, eos_id)
    if treatment_tokens != sham_tokens:
        failures.append("arm_token_accounting")
    if treatment_tokens["skipped_too_long"] or sham_tokens["skipped_too_long"]:
        failures.append("sft_completion_boundary")
    expected_manifest = {
        "artifacts": {
            "board_sha256": EXPECTED_BOARD_SHA256,
            "board_rows_sha256": EXPECTED_BOARD_ROWS_SHA256,
            "sham_rows": len(sham),
            "sham_sha256": digest_bytes(sham_raw),
            "treatment_rows": len(treatment),
            "treatment_sha256": digest_bytes(treatment_raw),
        },
        "length_counts": {str(key): value for key, value in sorted(TRAIN_LENGTH_COUNTS.items())},
        "pack_length": PACK_LENGTH,
        "program_count": len(programs),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "schema": MANIFEST_SCHEMA,
        "seeds": {
            "board": BOARD_SEED,
            "observation": OBSERVATION_SEED,
            "sham": SHAM_SEED,
            "training": TRAIN_SEED,
        },
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "sft_encoding_sha256": EXPECTED_SFT_ENCODING_SHA256,
    }
    if manifest != expected_manifest:
        failures.append("generation_manifest")

    row_metrics = _row_invariants(treatment, sham, programs, mapping, rebuilt_board["rows"])
    required_zero = (
        "correct_arithmetic_updater_transitions",
        "evaluation_answer_response_occurrences",
        "malformed_compiler_rows",
        "malformed_updater_rows",
        "reconstructed_sham_mapping_mismatches",
        "source_final_answer_response_occurrences",
    )
    failures.extend(key for key in required_zero if row_metrics[key])
    if row_metrics["updater_pairs_byte_identical"] != 12_288:
        failures.append("updater_arm_identity")
    if row_metrics["compiler_controls_matched"] != 4_096:
        failures.append("compiler_control_identity")

    length_counts = Counter(len(program["operations"]) for program in programs)
    if dict(length_counts) != TRAIN_LENGTH_COUNTS:
        failures.append("training_length_balance")
    strata = Counter(sham_stratum(program) for program in programs)
    if not strata or min(strata.values()) < 2:
        failures.append("sham_singleton_strata")
    sham_fixed_points = sum(index == donor for index, donor in enumerate(mapping))
    sham_same_program = sum(
        semantic_signature(programs[index]["initial_state"], programs[index]["operations"])
        == semantic_signature(programs[donor]["initial_state"], programs[donor]["operations"])
        for index, donor in enumerate(mapping)
    )
    sham_same_trajectory = sum(
        programs[index]["trajectory"] == programs[donor]["trajectory"]
        for index, donor in enumerate(mapping)
    )
    sham_same_answer = sum(
        programs[index]["final_answer"] == programs[donor]["final_answer"]
        for index, donor in enumerate(mapping)
    )
    if sham_fixed_points or sham_same_program or sham_same_trajectory or sham_same_answer:
        failures.append("sham_derangement_contract")

    treatment_prompts = [normalized_prompt(row["completion_prompt"]) for row in treatment]
    sham_prompts = [normalized_prompt(row["completion_prompt"]) for row in sham]
    duplicate_treatment = len(treatment_prompts) - len(set(treatment_prompts))
    duplicate_sham = len(sham_prompts) - len(set(sham_prompts))
    if duplicate_treatment or duplicate_sham:
        failures.append("duplicate_normalized_training_prompts")

    eval_programs = {
        semantic_signature(row["initial_state"], row["operations"]) for row in rebuilt_board["rows"]
    }
    eval_sources = {row["source"] for row in rebuilt_board["rows"]}
    eval_packets = {row["packet"] for row in rebuilt_board["rows"]}
    eval_trajectories = {tuple(row["trajectory"]) for row in rebuilt_board["rows"]}
    train_programs = {
        semantic_signature(program["initial_state"], program["operations"]) for program in programs
    }
    train_sources = {program["source"] for program in programs}
    train_packets = {program["packet"] for program in programs}
    train_trajectories = {tuple(program["trajectory"]) for program in programs}
    eval_grams: set[tuple[str, ...]] = set()
    train_grams: set[tuple[str, ...]] = set()
    for source in eval_sources:
        eval_grams.update(ngrams(source))
    for source in train_sources:
        train_grams.update(ngrams(source))
    overlap_metrics = {
        "complete_trajectory_overlap": len(eval_trajectories & train_trajectories),
        "exact_packet_overlap": len(eval_packets & train_packets),
        "exact_source_overlap": len(eval_sources & train_sources),
        "normalized_13_token_source_ngram_overlap": len(eval_grams & train_grams),
        "normalized_semantic_program_overlap": len(eval_programs & train_programs),
        "reserved_renderer_training_occurrences": sum(
            program["template_id"] == reserved_template_id() for program in programs
        ),
    }
    failures.extend(key for key, value in overlap_metrics.items() if value)

    report = {
        "admitted": not failures,
        "artifact_sha256": {
            "board": digest_bytes(board_raw),
            "manifest": digest_bytes(manifest_raw),
            "protocol": EXPECTED_PROTOCOL_SHA256,
            "sham": digest_bytes(sham_raw),
            "sft_encoding": EXPECTED_SFT_ENCODING_SHA256,
            "treatment": digest_bytes(treatment_raw),
            "tokenizer": sha256_file(tokenizer_path),
        },
        "board": {
            "case_count": len(board.get("rows", ())),
            "rows_sha256": rows_digest,
            "strata": dict(sorted(Counter(row.get("stratum") for row in board.get("rows", ())).items())),
        },
        "failures": sorted(set(failures)),
        "overlap": overlap_metrics,
        "row_metrics": row_metrics,
        "schema": AUDIT_SCHEMA,
        "sham": {
            "fixed_points": sham_fixed_points,
            "same_answer": sham_same_answer,
            "same_program": sham_same_program,
            "same_trajectory": sham_same_trajectory,
            "strata": len(strata),
            "minimum_stratum_size": min(strata.values()) if strata else 0,
        },
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "training": {
            "duplicate_normalized_sham_prompts": duplicate_sham,
            "duplicate_normalized_treatment_prompts": duplicate_treatment,
            "length_counts": {str(key): value for key, value in sorted(length_counts.items())},
            "programs": len(programs),
            "sham_rows": len(sham),
            "treatment_rows": len(treatment),
        },
    }
    return report


def write_immutable_json(path: str | Path, payload: dict[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = pretty_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating immutable audit")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return digest_bytes(encoded)


def main() -> None:
    raise SystemExit(
        "RSP-C1 is permanently closed; preserve this module as audit evidence only"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--sham", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit_artifacts(
        args.board, args.tokenizer, args.treatment, args.sham, args.manifest
    )
    digest = write_immutable_json(args.out, report)
    print(json.dumps({"admitted": report["admitted"], "audit_sha256": digest, "failures": report["failures"]}, indent=2, sort_keys=True))
    if not report["admitted"]:
        raise SystemExit("residual packet admission audit failed")


if __name__ == "__main__":
    main()
