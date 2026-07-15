#!/usr/bin/env python3
"""Generate the frozen RSP-C1 board and causally matched SFT arms.

Board creation and training-data creation are deliberately separate commands.
The latter accepts only the exact board whose artifact and canonical-row
digests are frozen below.  All outputs are exclusive-create, fsynced, and
read-only.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "train" / "residual_packet_protocol.py"
SFT_ENCODING_PATH = ROOT / "train" / "sft_encoding.py"

BOARD_SCHEMA = "residual_packet_board_v1"
TRAIN_SCHEMA = "residual_packet_training_v1"
MANIFEST_SCHEMA = "residual_packet_generation_manifest_v1"
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
SEEN_BIGRAMS = tuple(
    pair
    for pair in ((left, right) for left in OPERATION_TYPES for right in OPERATION_TYPES)
    if pair not in HELD_OUT_BIGRAMS
)
EXPECTED_BOARD_ROWS_SHA256 = "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
EXPECTED_BOARD_SHA256 = "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
EXPECTED_PROTOCOL_SHA256 = "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2"
EXPECTED_SFT_ENCODING_SHA256 = "33d07db19afb7155567cef922dbab616916864eafaeba7fa26f12d2203c0d5b8"
TOKEN_RE = re.compile(r"[a-z]+|\d+")
INTEGER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")

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
    module_name = f"_shohin_generator_{module_label}_{observed_sha256[:16]}"
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
    """Load the exact concurrently-owned protocol without package resolution."""

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


def _protocol_value(*names: str):
    protocol = protocol_module()
    for name in names:
        if hasattr(protocol, name):
            return getattr(protocol, name)
    raise AttributeError("residual packet protocol is missing " + "/".join(names))


def training_template_ids() -> tuple[str, ...]:
    value = _protocol_value(
        "TRAIN_SOURCE_TEMPLATE_IDS",
        "TRAINING_TEMPLATE_IDS",
        "TRAIN_TEMPLATE_IDS",
        "TRAIN_SOURCE_TEMPLATES",
    )
    if isinstance(value, dict):
        value = value.keys()
    result = tuple(str(item) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError("protocol training template identifiers must be unique")
    return result


def reserved_template_id() -> str:
    return str(
        _protocol_value(
            "RESERVED_TEMPLATE_ID", "RESERVED_RENDERER_ID", "RESERVED_SOURCE_TEMPLATE_ID"
        )
    )


def render_source(initial_state: int, operations: Sequence[Sequence[Any]], template_id: str) -> str:
    function = _protocol_value("render_source", "render_program_source")
    try:
        return str(function(initial_state, operations, template_id))
    except TypeError as first_error:
        try:
            return str(
                function(
                    initial_state=initial_state,
                    operations=operations,
                    template_id=template_id,
                )
            )
        except TypeError:
            raise first_error


def render_packet(state: int, operations: Sequence[Sequence[Any]]) -> str:
    return str(_protocol_value("canonical_packet", "render_packet")(state, operations))


def render_answer(value: int) -> str:
    return str(_protocol_value("canonical_answer", "render_answer")(value))


def compiler_prompt(source: str) -> str:
    return str(_protocol_value("compiler_prompt", "render_compiler_prompt")(source))


def updater_prompt(packet: str, observed_result: int) -> str:
    function = _protocol_value("updater_prompt", "update_prompt", "render_updater_prompt")
    return str(function(packet, observed_result))


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


def apply_operation(state: int, operation: Sequence[Any]) -> int:
    kind, operand = str(operation[0]), int(operation[1])
    if kind == "add":
        return state + operand
    if kind == "multiply":
        return state * operand
    if kind == "subtract":
        return state - operand
    raise ValueError(f"unknown operation {kind!r}")


def mathematical_trajectory(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    states = [int(initial_state)]
    for operation in operations:
        states.append(apply_operation(states[-1], operation))
    return tuple(states)


def semantic_signature(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    return (int(initial_state),) + tuple((str(kind), int(operand)) for kind, operand in operations)


def operation_types(operations: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(operation[0]) for operation in operations)


def digit_width_vector(initial_state: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    return (len(str(initial_state)),) + tuple(len(str(int(operation[1]))) for operation in operations)


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(str(text).lower()))


def ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    return {tokens[index : index + width] for index in range(len(tokens) - width + 1)}


def integer_occurrences(text: str) -> tuple[int, ...]:
    return tuple(int(match.group(0)) for match in INTEGER_RE.finditer(str(text)))


def _sequence_pool(length: int, held_out: tuple[str, str] | None = None) -> tuple[tuple[str, ...], ...]:
    sequences = [()]
    for _ in range(length):
        sequences = [prefix + (kind,) for prefix in sequences for kind in OPERATION_TYPES]
    result = []
    for sequence in sequences:
        bigrams = tuple(zip(sequence, sequence[1:]))
        held = [pair for pair in bigrams if pair in HELD_OUT_BIGRAMS]
        if held_out is None and not held:
            result.append(sequence)
        elif held_out is not None and held == [held_out]:
            result.append(sequence)
    if not result:
        raise RuntimeError("no operation sequence satisfies the requested stratum")
    return tuple(result)


def _operand(rng: random.Random, kind: str, value_ood: bool = False, width: int | None = None) -> int:
    if value_ood:
        low, high = ((8, 12) if kind == "multiply" else (26, 75))
    else:
        low, high = ((2, 7) if kind == "multiply" else (2, 25))
    if width is not None:
        low = max(low, 10 ** (width - 1) if width > 1 else 0)
        high = min(high, 10**width - 1)
        if low > high:
            raise ValueError(f"no {kind} operand with width {width}")
    return rng.randint(low, high)


def _sample_program(
    rng: random.Random,
    types: Sequence[str],
    *,
    value_ood: bool = False,
    widths: Sequence[int] | None = None,
) -> tuple[int, list[list[Any]]]:
    if widths is None:
        initial_state = rng.randint(100, 299) if value_ood else rng.randint(10, 99)
        operations = [[kind, _operand(rng, kind, value_ood)] for kind in types]
    else:
        initial_low = max(10, 10 ** (int(widths[0]) - 1))
        initial_high = min(99, 10 ** int(widths[0]) - 1)
        initial_state = rng.randint(initial_low, initial_high)
        operations = [
            [kind, _operand(rng, kind, False, int(width))]
            for kind, width in zip(types, widths[1:])
        ]
    return initial_state, operations


def _board_row(
    stratum: str,
    index: int,
    initial_state: int,
    operations: Sequence[Sequence[Any]],
    template_id: str,
) -> dict[str, Any]:
    trajectory = mathematical_trajectory(initial_state, operations)
    source = render_source(initial_state, operations, template_id)
    return {
        "answer": trajectory[-1],
        "id": f"{stratum}_{index:03d}",
        "initial_state": initial_state,
        "operations": [list(operation) for operation in operations],
        "packet": render_packet(initial_state, operations),
        "source": source,
        "stratum": stratum,
        "template_id": template_id,
        "trajectory": list(trajectory),
    }


def build_board_rows() -> list[dict[str, Any]]:
    rng = random.Random(BOARD_SEED)
    train_templates = training_template_ids()
    reserved = reserved_template_id()
    if reserved in train_templates:
        raise ValueError("reserved source renderer is present in the training renderer set")
    rows: list[dict[str, Any]] = []
    seen_programs: set[tuple[Any, ...]] = set()
    seen_sources: set[str] = set()
    seen_packets: set[str] = set()
    seen_trajectories: set[tuple[int, ...]] = set()
    seen_answers: set[int] = set()

    for stratum in STRATUM_ORDER:
        for index in range(PER_STRATUM):
            for _attempt in range(100_000):
                if stratum == "renderer_ood":
                    length = 3
                    types = rng.choice(_sequence_pool(length))
                    template_id = reserved
                    initial_state, operations = _sample_program(rng, types)
                elif stratum == "value_ood":
                    length = 3
                    types = rng.choice(_sequence_pool(length))
                    template_id = train_templates[index % len(train_templates)]
                    initial_state, operations = _sample_program(rng, types, value_ood=True)
                elif stratum == "order_ood":
                    local = index % 32
                    held_out = HELD_OUT_BIGRAMS[0 if index < 32 else 1]
                    length = 3 if local % 2 == 0 else 4
                    types = rng.choice(_sequence_pool(length, held_out))
                    template_id = train_templates[index % len(train_templates)]
                    initial_state, operations = _sample_program(rng, types)
                else:
                    length = 5
                    types = rng.choice(_sequence_pool(length))
                    template_id = train_templates[index % len(train_templates)]
                    initial_state, operations = _sample_program(rng, types)
                trajectory = mathematical_trajectory(initial_state, operations)
                if min(trajectory) <= 0 or trajectory[-1] < 100:
                    continue
                row = _board_row(stratum, index, initial_state, operations, template_id)
                signature = semantic_signature(initial_state, operations)
                trajectory_key = tuple(row["trajectory"])
                if (
                    signature in seen_programs
                    or row["source"] in seen_sources
                    or row["packet"] in seen_packets
                    or trajectory_key in seen_trajectories
                    or row["answer"] in seen_answers
                ):
                    continue
                rows.append(row)
                seen_programs.add(signature)
                seen_sources.add(row["source"])
                seen_packets.add(row["packet"])
                seen_trajectories.add(trajectory_key)
                seen_answers.add(row["answer"])
                break
            else:
                raise RuntimeError(f"could not generate board row {stratum}/{index}")
    return rows


def board_payload(rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(build_board_rows() if rows is None else rows)
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


def _assert_frozen_board(payload: dict[str, Any], raw: bytes | None = None) -> None:
    if EXPECTED_BOARD_ROWS_SHA256 == "TO_BE_FROZEN" or EXPECTED_BOARD_SHA256 == "TO_BE_FROZEN":
        raise RuntimeError("board hashes have not been frozen in the generator")
    rows_digest = digest_bytes(canonical_json_bytes(payload.get("rows")))
    artifact_digest = digest_bytes(pretty_json_bytes(payload) if raw is None else raw)
    if rows_digest != EXPECTED_BOARD_ROWS_SHA256:
        raise ValueError("board canonical-row SHA-256 does not match the frozen digest")
    if artifact_digest != EXPECTED_BOARD_SHA256:
        raise ValueError("board artifact SHA-256 does not match the frozen digest")
    if payload.get("rows_sha256") != rows_digest:
        raise ValueError("board embedded canonical-row SHA-256 is wrong")


def _read_frozen_board(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise PermissionError("training requires a regular read-only board")
    raw = source.read_bytes()
    payload = json.loads(raw)
    if raw != pretty_json_bytes(payload):
        raise ValueError("board is not canonical JSON")
    _assert_frozen_board(payload, raw)
    expected = board_payload()
    if payload != expected:
        raise ValueError("board contents do not reproduce from the frozen seed")
    return payload


def _program_record(
    initial_state: int,
    operations: Sequence[Sequence[Any]],
    template_id: str,
    tokenizer,
    pair_id: int,
) -> dict[str, Any]:
    trajectory = mathematical_trajectory(initial_state, operations)
    source = render_source(initial_state, operations, template_id)
    packet = render_packet(initial_state, operations)
    return {
        "final_answer": trajectory[-1],
        "initial_state": initial_state,
        "operations": [list(operation) for operation in operations],
        "packet": packet,
        "packet_token_count": len(tokenizer.encode(packet).ids),
        "pair_id": pair_id,
        "source": source,
        "template_id": template_id,
        "trajectory": trajectory,
    }


def _program_allowed(
    program: dict[str, Any],
    *,
    eval_answers: set[int],
    eval_programs: set[tuple[Any, ...]],
    eval_sources: set[str],
    eval_packets: set[str],
    eval_trajectories: set[tuple[int, ...]],
    eval_grams: set[tuple[str, ...]],
    used_programs: set[tuple[Any, ...]],
    used_sources: set[str],
    used_packets: set[str],
    used_trajectories: set[tuple[int, ...]],
) -> bool:
    operations = program["operations"]
    signature = semantic_signature(program["initial_state"], operations)
    trajectory = tuple(program["trajectory"])
    numeric_packet_fields = {program["initial_state"]} | {
        int(operation[1]) for operation in operations
    }
    return (
        min(trajectory) > 0
        and program["final_answer"] not in numeric_packet_fields
        and not (numeric_packet_fields & eval_answers)
        and signature not in eval_programs
        and program["source"] not in eval_sources
        and program["packet"] not in eval_packets
        and trajectory not in eval_trajectories
        and program["final_answer"] not in eval_answers
        and not (ngrams(program["source"]) & eval_grams)
        and signature not in used_programs
        and program["source"] not in used_sources
        and program["packet"] not in used_packets
        and trajectory not in used_trajectories
    )


def build_training_programs(board_rows: Sequence[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    rng = random.Random(TRAIN_SEED)
    templates = training_template_ids()
    eval_answers = {int(row["answer"]) for row in board_rows}
    eval_programs = {
        semantic_signature(int(row["initial_state"]), row["operations"]) for row in board_rows
    }
    eval_sources = {str(row["source"]) for row in board_rows}
    eval_packets = {str(row["packet"]) for row in board_rows}
    eval_trajectories = {tuple(map(int, row["trajectory"])) for row in board_rows}
    eval_grams: set[tuple[str, ...]] = set()
    for source in eval_sources:
        eval_grams.update(ngrams(source))
    used_programs: set[tuple[Any, ...]] = set()
    used_sources: set[str] = set()
    used_packets: set[str] = set()
    used_trajectories: set[tuple[int, ...]] = set()
    programs: list[dict[str, Any]] = []
    pair_id = 0

    for length in sorted(TRAIN_LENGTH_COUNTS):
        for local_pair in range(TRAIN_LENGTH_COUNTS[length] // 2):
            for _base_attempt in range(10_000):
                types = rng.choice(_sequence_pool(length))
                template_id = templates[local_pair % len(templates)]
                initial_state, operations = _sample_program(rng, types)
                first = _program_record(
                    initial_state, operations, template_id, tokenizer, pair_id
                )
                if not _program_allowed(
                    first,
                    eval_answers=eval_answers,
                    eval_programs=eval_programs,
                    eval_sources=eval_sources,
                    eval_packets=eval_packets,
                    eval_trajectories=eval_trajectories,
                    eval_grams=eval_grams,
                    used_programs=used_programs,
                    used_sources=used_sources,
                    used_packets=used_packets,
                    used_trajectories=used_trajectories,
                ):
                    continue
                widths = digit_width_vector(first["initial_state"], first["operations"])
                first_signature = semantic_signature(first["initial_state"], first["operations"])
                for _partner_attempt in range(10_000):
                    second_initial, second_operations = _sample_program(
                        rng, types, widths=widths
                    )
                    second = _program_record(
                        second_initial, second_operations, template_id, tokenizer, pair_id
                    )
                    if (
                        second["packet_token_count"] != first["packet_token_count"]
                        or len(str(second["final_answer"])) != len(str(first["final_answer"]))
                        or second["final_answer"] == first["final_answer"]
                        or first["final_answer"] in integer_occurrences(second["packet"])
                        or second["final_answer"] in integer_occurrences(first["packet"])
                        or semantic_signature(second_initial, second_operations) == first_signature
                    ):
                        continue
                    if not _program_allowed(
                        second,
                        eval_answers=eval_answers,
                        eval_programs=eval_programs,
                        eval_sources=eval_sources,
                        eval_packets=eval_packets,
                        eval_trajectories=eval_trajectories,
                        eval_grams=eval_grams,
                        used_programs=used_programs | {first_signature},
                        used_sources=used_sources | {first["source"]},
                        used_packets=used_packets | {first["packet"]},
                        used_trajectories=used_trajectories | {tuple(first["trajectory"])},
                    ):
                        continue
                    for program in (first, second):
                        signature = semantic_signature(
                            program["initial_state"], program["operations"]
                        )
                        used_programs.add(signature)
                        used_sources.add(program["source"])
                        used_packets.add(program["packet"])
                        used_trajectories.add(tuple(program["trajectory"]))
                        programs.append(program)
                    pair_id += 1
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError(f"could not generate matched training pair length={length}")

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


def build_sham_permutation(
    programs: Sequence[dict[str, Any]], seed: int = SHAM_SEED
) -> tuple[int, ...]:
    rng = random.Random(seed)
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, program in enumerate(programs):
        strata[sham_stratum(program)].append(index)
    mapping = [-1] * len(programs)
    for key in sorted(strata):
        recipients = list(strata[key])
        if len(recipients) < 2:
            raise RuntimeError(f"singleton sham stratum {key!r}")
        rng.shuffle(recipients)
        preferences: dict[int, list[int]] = {}
        for recipient in recipients:
            choices = [
                donor
                for donor in recipients
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

        if any(not assign(recipient, set()) for recipient in recipients):
            raise RuntimeError(f"no valid sham derangement for {key!r}")
        for donor, recipient in owner.items():
            mapping[recipient] = donor
    if sorted(mapping) != list(range(len(programs))) or any(
        recipient == donor for recipient, donor in enumerate(mapping)
    ):
        raise RuntimeError("sham construction is not a complete derangement")
    return tuple(mapping)


def _observation_pair(
    rng: random.Random,
    operation: Sequence[Any],
    eval_answers: set[int],
    source_answer: int,
) -> tuple[int, int]:
    forbidden = eval_answers | {source_answer}
    for _ in range(100_000):
        state = rng.randint(1000, 9999)
        observed = rng.randint(1000, 9999)
        if state in forbidden or observed in forbidden:
            continue
        if observed == apply_operation(state, operation):
            continue
        return state, observed
    raise RuntimeError("could not sample an arithmetic-false updater observation")


def build_training_arms(
    programs: Sequence[dict[str, Any]],
    board_rows: Sequence[dict[str, Any]],
    sham_mapping: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(OBSERVATION_SEED)
    eval_answers = {int(row["answer"]) for row in board_rows}
    treatment: list[dict[str, Any]] = []
    sham: list[dict[str, Any]] = []
    for index, program in enumerate(programs):
        base = {
            "completion_prompt": compiler_prompt(program["source"]),
            "id": f"{program['id']}_compiler",
            "kind": "compiler",
            "program_id": program["id"],
            "question": compiler_prompt(program["source"]),
            "training_group": "residual_packet",
        }
        treatment.append({**base, "response": program["packet"]})
        sham.append({**base, "response": programs[sham_mapping[index]]["packet"]})
        for step, operation in enumerate(program["operations"]):
            state, observed = _observation_pair(
                rng, operation, eval_answers, int(program["final_answer"])
            )
            prompt = updater_prompt(
                render_packet(state, program["operations"][step:]), observed
            )
            remaining = program["operations"][step + 1 :]
            response = (
                render_packet(observed, remaining) if remaining else render_answer(observed)
            )
            row = {
                "completion_prompt": prompt,
                "id": f"{program['id']}_updater_{step:02d}",
                "kind": "updater",
                "program_id": program["id"],
                "question": prompt,
                "response": response,
                "step": step,
                "training_group": "residual_packet",
            }
            treatment.append(row)
            sham.append(dict(row))
    return treatment, sham


def token_accounting(rows: Sequence[dict[str, Any]], tokenizer, eos_id: int) -> dict[str, int]:
    encode_supervised_example = sft_encoding_module().encode_supervised_example

    prompt_tokens = response_tokens = supervised_tokens = full_tokens = 0
    skipped_too_long = 0
    for row in rows:
        prompt = str(row["completion_prompt"])
        response = str(row["response"]).rstrip()
        prompt_ids, token_ids, completion_mask = encode_supervised_example(
            tokenizer, prompt, response, eos_id
        )
        prompt_count = len(prompt_ids)
        response_count = len(token_ids) - prompt_count - 1
        example_tokens = len(token_ids)
        if sum(completion_mask) != response_count + 1:
            raise RuntimeError("SFT completion boundary no longer matches the frozen contract")
        if example_tokens > PACK_LENGTH:
            skipped_too_long += 1
            continue
        prompt_tokens += prompt_count
        response_tokens += response_count
        supervised_tokens += response_count + 1
        full_tokens += example_tokens
    packed_sequences = max(0, (full_tokens - 2) // PACK_LENGTH)
    return {
        "compiler_rows": sum(row["kind"] == "compiler" for row in rows),
        "examples": len(rows) - skipped_too_long,
        "forward_token_count": packed_sequences * PACK_LENGTH,
        "full_token_count": full_tokens,
        "packed_sequence_count": packed_sequences,
        "prompt_token_count": prompt_tokens,
        "response_token_count": response_tokens,
        "skipped_too_long": skipped_too_long,
        "supervised_token_count": supervised_tokens,
        "updater_rows": sum(row["kind"] == "updater" for row in rows),
    }


def _exclusive_immutable_write(path: str | Path, payload: bytes) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating immutable output")
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
    if destination.stat().st_mode & 0o222:
        raise PermissionError("immutable output retained write bits")
    return digest_bytes(payload)


def write_immutable_json(path: str | Path, payload: Any) -> str:
    return _exclusive_immutable_write(path, pretty_json_bytes(payload))


def write_immutable_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> str:
    return _exclusive_immutable_write(path, jsonl_bytes(rows))


def write_board(path: str | Path) -> dict[str, Any]:
    payload = board_payload()
    _assert_frozen_board(payload)
    digest = write_immutable_json(path, payload)
    return {"board_sha256": digest, "case_count": len(payload["rows"]), "rows_sha256": payload["rows_sha256"]}


def generate_training_outputs(
    board_path: str | Path,
    tokenizer_path: str | Path,
    treatment_path: str | Path,
    sham_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    destinations = tuple(map(Path, (treatment_path, sham_path, manifest_path)))
    if len(set(destinations)) != len(destinations):
        raise ValueError("training output paths must be distinct")
    if any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite a training output")
    board = _read_frozen_board(board_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    programs = build_training_programs(board["rows"], tokenizer)
    mapping = build_sham_permutation(programs)
    treatment, sham = build_training_arms(programs, board["rows"], mapping)
    treatment_raw = jsonl_bytes(treatment)
    sham_raw = jsonl_bytes(sham)
    treatment_tokens = token_accounting(treatment, tokenizer, eos_id)
    sham_tokens = token_accounting(sham, tokenizer, eos_id)
    if treatment_tokens != sham_tokens:
        raise RuntimeError("treatment and sham token accounting differ")
    if treatment_tokens["skipped_too_long"]:
        raise RuntimeError("one or more examples exceed the frozen pack length")
    manifest = {
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
    created: list[Path] = []
    try:
        _exclusive_immutable_write(treatment_path, treatment_raw)
        created.append(Path(treatment_path))
        _exclusive_immutable_write(sham_path, sham_raw)
        created.append(Path(sham_path))
        write_immutable_json(manifest_path, manifest)
        created.append(Path(manifest_path))
    except BaseException:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return manifest


def main() -> None:
    raise SystemExit(
        "RSP-C1 is permanently closed; preserve this module as audit evidence only"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    board_command = commands.add_parser("board", help="create the frozen evaluation board")
    board_command.add_argument("--out", required=True)
    train_command = commands.add_parser("train", help="create treatment/sham SFT JSONL files")
    train_command.add_argument("--board", required=True)
    train_command.add_argument("--tokenizer", required=True)
    train_command.add_argument("--treatment-out", required=True)
    train_command.add_argument("--sham-out", required=True)
    train_command.add_argument("--manifest-out", required=True)
    args = parser.parse_args()
    if args.command == "board":
        result = write_board(args.out)
    else:
        result = generate_training_outputs(
            args.board,
            args.tokenizer,
            args.treatment_out,
            args.sham_out,
            args.manifest_out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
