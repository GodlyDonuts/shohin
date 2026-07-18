#!/usr/bin/env python3
"""Build score-unseen calibration and confirmation boards for R10 ACAW.

This builder has no model, checkpoint, probability, or score inputs.  It
creates two deterministic compiler boards, preflights the shared lexical and
executor contract, and hard-scans prompt and program novelty before writing
immutable JSONL files plus a hash-bound build manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from categorical_microcode import (  # noqa: E402
    OPCODES,
    QUERIES,
    compile_example,
    execute_program,
    transition_basis_targets,
)


SCHEMA = "r10_workspace_board_v2"
NGRAM_WIDTH = 13
EXECUTOR_WIDTH = 8
EXECUTOR_LIMIT = 10 ** EXECUTOR_WIDTH
MAX_ROW_ATTEMPTS = 512
WORD = re.compile(r"\w+")
CANONICAL_GENERATOR_SEEDS = {
    "calibration": 2026071401,
    "confirmation": 2026071402,
}
CANONICAL_R5_NOVELTY_BOARD_SHA256 = (
    "d85f16ff374b0c650cf3603826cc5f3b377842818db62bada3b84e71308b9473"
)
BUILD_MANIFEST_CLAIM_BOUNDARY = (
    "This v2 build proves deterministic score-blind generation on the frozen exact-cell "
    "schedule, compiler/executor preflight, and lexical/program novelty scans only. It "
    "contains no model score and does not authorize an R10 score run until the independent "
    "admission report and frozen gate manifest bind these exact hashes."
)


@dataclass(frozen=True)
class Domain:
    family: str
    keys: tuple[str, str]
    unit: str


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    depths: tuple[int, int]
    numeric_profile: str
    initial_range: tuple[int, int]
    value_range: tuple[int, int]


@dataclass(frozen=True)
class BoardSpec:
    name: str
    reference_prefix: str
    domains: tuple[Domain, Domain, Domain, Domain]
    regimes: tuple[RegimeSpec, RegimeSpec]
    event_prefix: str
    query_prefix: str
    answer_marker: str
    intro_templates: tuple[str, ...]
    operation_templates: dict[str, tuple[str, ...]]
    query_templates: dict[str, tuple[str, ...]]


CALIBRATION_DOMAINS = (
    Domain("map room", ("contour slips", "bearing folios"), "survey entries"),
    Domain("print works", ("proof bundles", "plate packets"), "production records"),
    Domain("signal bureau", ("relay frames", "beacon logs"), "telemetry records"),
    Domain("glass studio", ("anneal trays", "mould tickets"), "work orders"),
)
CONFIRMATION_DOMAINS = (
    Domain("binding hall", ("quire stacks", "cover lots"), "binding materials"),
    Domain("rail control", ("routing chits", "coupler tallies"), "dispatch records"),
    Domain("ceramic works", ("bisque racks", "glaze batches"), "studio pieces"),
    Domain("forecast office", ("radar frames", "pressure charts"), "weather observations"),
)


CALIBRATION_OPERATION_TEMPLATES = {
    "add": (
        "Post an inbound lot of {value} {unit} to {target}.",
        "Credit {target} using a newly recorded {value} {unit}.",
        "Augment the balance named {target} by exactly {value} {unit}.",
    ),
    "sub": (
        "Post an outbound lot of {value} {unit} against {target}.",
        "Debit {target} by a recorded {value} {unit}.",
        "Reduce the balance named {target} by exactly {value} {unit}.",
    ),
    "move": (
        "Reclassify {value} {unit} from {source} under {target}.",
        "Post a paired adjustment of {value} {unit} out of {source} and into {target}.",
        "Debit {source} and credit {target} for the same {value} {unit}.",
    ),
    "merge": (
        "Accumulate the whole balance named {source} into {target} while preserving {source}.",
        "Use the current {source} balance as an additional credit to {target}.",
        "Increase {target} by the entire amount presently recorded under {source}.",
    ),
    "swap": (
        "Transpose the balances attached to {left} and {right}.",
        "Give {left} the prior {right} balance and give {right} the prior {left} balance.",
        "Exchange only the ledger balances named {left} and {right}.",
    ),
}
CONFIRMATION_OPERATION_TEMPLATES = {
    "add": (
        "Apply a positive adjustment of {value} {unit} to {target}.",
        "Enter {value} incoming {unit} on the account for {target}.",
        "Raise {target}'s recorded amount through an addition of {value} {unit}.",
    ),
    "sub": (
        "Apply a negative adjustment of {value} {unit} to {target}.",
        "Enter {value} outgoing {unit} on the account for {target}.",
        "Lower {target}'s recorded amount through a deduction of {value} {unit}.",
    ),
    "move": (
        "Route {value} {unit} away from {source} and onward to {target}.",
        "Record one relocation: {source} loses {value} {unit} as {target} receives them.",
        "Shift a quantity of {value} {unit}; subtract it from {source} and add it to {target}.",
    ),
    "merge": (
        "Append the present amount in {source} to {target}, leaving {source} unchanged.",
        "Treat all of {source}'s current balance as an extra amount for {target}.",
        "Combine {source} into {target} additively without resetting either named account.",
    ),
    "swap": (
        "Let the two accounts {left} and {right} inherit one another's previous amounts.",
        "Replace {left}'s amount with old {right}, and replace {right}'s amount with old {left}.",
        "Perform a two-way balance transposition between {left} and {right}.",
    ),
}


CALIBRATION_QUERY_TEMPLATES = {
    "read": (
        "Return the closing balance associated with {key}.",
        "What closing quantity is posted under {key}?",
        "State the terminal balance for {key}.",
    ),
    "sum": (
        "Return the aggregate closing balance of {left} together with {right}.",
        "What total results when the closing {left} and {right} balances are combined?",
        "State the joint terminal quantity across {left} and {right}.",
    ),
    "difference": (
        "Return the nonnegative closing margin of {high} over {low}.",
        "By what quantity does closing {high} stand above closing {low}?",
        "State the closing {high} balance minus the closing {low} balance.",
    ),
}
CONFIRMATION_QUERY_TEMPLATES = {
    "read": (
        "Provide the final account amount for {key}.",
        "Which ending quantity belongs to {key}?",
        "Report the amount left on {key} after every step.",
    ),
    "sum": (
        "Provide the final combined amount across {left} plus {right}.",
        "Which ending total is obtained by adding {left} and {right}?",
        "Report the sum of the two completed accounts {left} and {right}.",
    ),
    "difference": (
        "Provide the final excess of {high} relative to {low}.",
        "Which ending gap remains when {low} is taken from {high}?",
        "Report the completed {high} amount less the completed {low} amount.",
    ),
}


BOARD_SPECS = {
    "calibration": BoardSpec(
        name="calibration",
        reference_prefix="R10-CAL",
        domains=CALIBRATION_DOMAINS,
        regimes=(
            RegimeSpec("fit_iid", (4, 8), "in_range", (3, 29), (1, 9)),
            RegimeSpec("depth_ood", (16, 32), "shifted", (211, 499), (11, 29)),
        ),
        event_prefix="Event",
        query_prefix="Request",
        answer_marker="Result:",
        intro_templates=(
            "Calibration ledger at {family} opens {left} with {left_value} {unit} and opens "
            "{right} with {right_value} {unit}; every written label is nonnumeric.",
            "For the {family} calibration, the opening balance is {left_value} {unit} under "
            "{left}, while {right} begins at {right_value} {unit}; label words carry no quantity.",
            "A sealed {family} record starts with {left_value} {unit} assigned to {left} and "
            "{right_value} {unit} assigned to {right}; names are textual only.",
        ),
        operation_templates=CALIBRATION_OPERATION_TEMPLATES,
        query_templates=CALIBRATION_QUERY_TEMPLATES,
    ),
    "confirmation": BoardSpec(
        name="confirmation",
        reference_prefix="R10-CON",
        domains=CONFIRMATION_DOMAINS,
        regimes=(
            RegimeSpec("language_ood", (4, 8), "in_range", (3, 29), (1, 9)),
            RegimeSpec("full_ood", (16, 32), "shifted", (701, 1099), (31, 53)),
        ),
        event_prefix="Step",
        query_prefix="Inquiry",
        answer_marker="Answer:",
        intro_templates=(
            "Independent ledger for {family} assigns {left_value} {unit} initially to {left}; "
            "separately, {right} is assigned {right_value} {unit}. All names are text.",
            "At {family}, begin the verification account with {left} holding {left_value} {unit} "
            "and {right} holding {right_value} {unit}; no label denotes a number.",
            "The untouched {family} account records an initial {left_value} {unit} for {left} "
            "versus {right_value} {unit} for {right}; words in names have no numeric role.",
        ),
        operation_templates=CONFIRMATION_OPERATION_TEMPLATES,
        query_templates=CONFIRMATION_QUERY_TEMPLATES,
    ),
}


def regime_by_name(spec: BoardSpec, name: str) -> RegimeSpec:
    try:
        return next(regime for regime in spec.regimes if regime.name == name)
    except StopIteration as error:
        raise ValueError("unknown {} regime {}".format(spec.name, name)) from error


def all_depths(spec: BoardSpec) -> tuple[int, ...]:
    return tuple(depth for regime in spec.regimes for depth in regime.depths)


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text: str, width: int = NGRAM_WIDTH) -> set[str]:
    words = normalized(text).split()
    return {
        " ".join(words[index:index + width])
        for index in range(max(0, len(words) - width + 1))
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def canonical_generation_contract() -> dict:
    return {
        "generator_seeds": dict(CANONICAL_GENERATOR_SEEDS),
        "r5_novelty_board_sha256": CANONICAL_R5_NOVELTY_BOARD_SHA256,
        "seed_variation_forbidden": True,
        "r5_variation_forbidden": True,
    }


def opcode_for_operation(operation, keys) -> str:
    key_index = {key: index for index, key in enumerate(keys)}
    kind = operation["kind"]
    if kind in {"add", "sub"}:
        return "{}_{}".format(kind, key_index[operation["target"]])
    if kind in {"move", "merge"}:
        return "{}_{}_{}".format(
            kind, key_index[operation["source"]], key_index[operation["target"]],
        )
    if kind == "swap":
        return "swap"
    raise ValueError("unknown operation kind {}".format(kind))


def query_opcode(query, keys) -> str:
    key_index = {key: index for index, key in enumerate(keys)}
    kind = query["kind"]
    if kind == "read":
        return "read_{}".format(key_index[query["key"]])
    if kind == "sum":
        return "sum"
    if kind == "difference":
        return "difference_{}_{}".format(
            key_index[query["high"]], key_index[query["low"]],
        )
    raise ValueError("unknown query kind {}".format(kind))


def program_signature(row) -> tuple | None:
    try:
        keys = tuple(row["keys"])
        if len(keys) != 2 or set(row["initial"]) != set(keys):
            return None
        operations = tuple(
            (opcode_for_operation(operation, keys), int(operation.get("value", 0)))
            for operation in row["operations"]
        )
        return (
            tuple(int(row["initial"][key]) for key in keys),
            operations,
            query_opcode(row["query"], keys),
        )
    except (KeyError, TypeError, ValueError):
        return None


def balanced_remainder_plan(totals: dict[tuple[str, int], int], seed: int):
    """Choose per-stratum remainder labels with globally optimal balance."""
    remainders = {stratum: total % len(OPCODES) for stratum, total in totals.items()}
    total_extras = sum(remainders.values())
    base, extra = divmod(total_extras, len(OPCODES))
    labels = list(OPCODES)
    rng = random.Random(seed)
    rng.shuffle(labels)
    remaining = {label: base + int(index < extra) for index, label in enumerate(labels)}
    strata = sorted(remainders, key=lambda item: (-remainders[item], item))

    def assign(position, current):
        if position == len(strata):
            return dict(current) if not any(remaining.values()) else None
        stratum = strata[position]
        width = remainders[stratum]
        candidates = [label for label in labels if remaining[label] > 0]
        combinations = list(itertools.combinations(candidates, width))
        rng.shuffle(combinations)
        future_strata = len(strata) - position - 1
        future_slots = sum(remainders[item] for item in strata[position + 1:])
        for combination in combinations:
            for label in combination:
                remaining[label] -= 1
            feasible = (
                sum(remaining.values()) == future_slots
                and max(remaining.values(), default=0) <= future_strata
            )
            if feasible:
                current[stratum] = tuple(combination)
                result = assign(position + 1, current)
                if result is not None:
                    return result
                del current[stratum]
            for label in combination:
                remaining[label] += 1
        return None

    result = assign(0, {})
    if result is None:
        raise RuntimeError("could not coordinate opcode remainder balance")
    return result


def balanced_opcode_stream(
    total: int,
    rng: random.Random,
    remainder_labels: tuple[str, ...] | None = None,
) -> list[str]:
    stream = []
    full_cycles, remainder = divmod(total, len(OPCODES))
    for _ in range(full_cycles):
        cycle = list(OPCODES)
        rng.shuffle(cycle)
        stream.extend(cycle)
    if remainder:
        selected = list(remainder_labels or ())
        if len(selected) != remainder or len(set(selected)) != remainder:
            raise ValueError("opcode remainder labels do not match stream remainder")
        rng.shuffle(selected)
        stream.extend(selected)
    elif remainder_labels:
        raise ValueError("zero-remainder stream received remainder labels")
    return stream


def operation_from_opcode(opcode: str, values: dict[str, int], keys, regime, rng):
    low, high = regime.value_range
    if opcode.startswith("add_"):
        target = keys[int(opcode[-1])]
        return {"kind": "add", "target": target, "value": rng.randint(low, high)}
    if opcode.startswith("sub_"):
        target = keys[int(opcode[-1])]
        upper = min(high, values[target])
        if upper < low:
            raise ValueError("subtraction source is below the frozen value range")
        return {"kind": "sub", "target": target, "value": rng.randint(low, upper)}
    if opcode.startswith("move_"):
        source_index, target_index = map(int, opcode.split("_")[1:])
        source, target = keys[source_index], keys[target_index]
        upper = min(high, values[source])
        if upper < low:
            raise ValueError("move source is below the frozen value range")
        return {
            "kind": "move", "source": source, "target": target,
            "value": rng.randint(low, upper),
        }
    if opcode.startswith("merge_"):
        source_index, target_index = map(int, opcode.split("_")[1:])
        return {
            "kind": "merge", "source": keys[source_index], "target": keys[target_index],
        }
    if opcode == "swap":
        return {"kind": "swap", "left": keys[0], "right": keys[1]}
    raise ValueError("unknown opcode {}".format(opcode))


def apply_operation(values: dict[str, int], operation) -> dict[str, int]:
    result = dict(values)
    kind = operation["kind"]
    if kind == "add":
        result[operation["target"]] += int(operation["value"])
    elif kind == "sub":
        result[operation["target"]] -= int(operation["value"])
    elif kind == "move":
        value = int(operation["value"])
        result[operation["source"]] -= value
        result[operation["target"]] += value
    elif kind == "merge":
        result[operation["target"]] += result[operation["source"]]
    elif kind == "swap":
        left, right = operation["left"], operation["right"]
        result[left], result[right] = result[right], result[left]
    else:
        raise ValueError("unknown operation {}".format(kind))
    if any(value < 0 or value >= EXECUTOR_LIMIT for value in result.values()):
        raise ValueError("operation leaves the exact executor range")
    return result


def make_query(opcode: str, values: dict[str, int], keys) -> dict:
    left, right = keys
    if opcode == "read_0":
        return {"kind": "read", "key": left, "answer": int(values[left])}
    if opcode == "read_1":
        return {"kind": "read", "key": right, "answer": int(values[right])}
    if opcode == "sum":
        answer = int(values[left] + values[right])
        if answer >= EXECUTOR_LIMIT:
            raise ValueError("sum query leaves the exact executor range")
        return {"kind": "sum", "answer": answer}
    if opcode == "difference_0_1":
        if values[left] < values[right]:
            raise ValueError("difference_0_1 orientation is negative")
        return {
            "kind": "difference", "high": left, "low": right,
            "answer": int(values[left] - values[right]),
        }
    if opcode == "difference_1_0":
        if values[right] < values[left]:
            raise ValueError("difference_1_0 orientation is negative")
        return {
            "kind": "difference", "high": right, "low": left,
            "answer": int(values[right] - values[left]),
        }
    raise ValueError("unknown query opcode {}".format(opcode))


def render_operation(operation, domain: Domain, spec: BoardSpec, template_index: int) -> str:
    template = spec.operation_templates[operation["kind"]][template_index]
    return template.format(unit=domain.unit, **operation)


def render_query(query, domain: Domain, spec: BoardSpec, template_index: int) -> str:
    template = spec.query_templates[query["kind"]][template_index]
    return template.format(left=domain.keys[0], right=domain.keys[1], **query)


def render_question(
    spec: BoardSpec,
    domain: Domain,
    initial,
    operations,
    query,
    intro_template: int,
    operation_templates: list[int],
    query_template: int,
) -> str:
    left, right = domain.keys
    intro = spec.intro_templates[intro_template].format(
        family=domain.family,
        left=left,
        right=right,
        left_value=initial[left],
        right_value=initial[right],
        unit=domain.unit,
    )
    events = "\n".join(
        "{} {}: {}".format(
            spec.event_prefix,
            index + 1,
            render_operation(operation, domain, spec, operation_templates[index]),
        )
        for index, operation in enumerate(operations)
    )
    query_text = render_query(query, domain, spec, query_template)
    return "{}\n{}\n{}: {}\n{}".format(
        intro, events, spec.query_prefix, query_text, spec.answer_marker,
    )


def _row_attempt_rng(seed: int, schedule_id: int, attempt: int) -> random.Random:
    payload = "{}:{}:{}".format(seed, schedule_id, attempt).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def cell_id(regime: str, depth: int, query: str, family: str) -> str:
    return "{}|depth={}|query={}|family={}".format(regime, depth, query, family)


def expected_cells(spec: BoardSpec) -> tuple[str, ...]:
    return tuple(
        cell_id(regime.name, depth, query, domain.family)
        for regime in spec.regimes
        for depth in regime.depths
        for query in QUERIES
        for domain in spec.domains
    )


def make_row(
    spec: BoardSpec,
    index: int,
    schedule_id: int,
    regime: RegimeSpec,
    depth: int,
    target_query: str,
    target_opcodes: list[str],
    domain: Domain,
    cell_index: int,
    seed: int,
    seen_prompts: set[str],
    seen_programs: set[tuple],
) -> dict:
    intro_template = (seed + schedule_id) % len(spec.intro_templates)
    operation_templates = [
        (seed + schedule_id + step) % len(spec.operation_templates[opcode.split("_")[0]])
        for step, opcode in enumerate(target_opcodes)
    ]
    query_kind = "read" if target_query.startswith("read") else (
        "difference" if target_query.startswith("difference") else "sum"
    )
    query_template = (seed + schedule_id) % len(spec.query_templates[query_kind])
    for attempt in range(MAX_ROW_ATTEMPTS):
        rng = _row_attempt_rng(seed, schedule_id, attempt)
        initial = {
            domain.keys[0]: rng.randint(*regime.initial_range),
            domain.keys[1]: rng.randint(*regime.initial_range),
        }
        values = dict(initial)
        operations = []
        try:
            for opcode in target_opcodes:
                operation = operation_from_opcode(opcode, values, domain.keys, regime, rng)
                operations.append(operation)
                values = apply_operation(values, operation)
            query = make_query(target_query, values, domain.keys)
        except ValueError:
            continue
        query["text"] = render_query(query, domain, spec, query_template)
        question = render_question(
            spec, domain, initial, operations, query,
            intro_template, operation_templates, query_template,
        )
        row = {
            "schema": SCHEMA,
            "board": spec.name,
            "question": question,
            "response": "The answer is {}.".format(query["answer"]),
            "answer": str(query["answer"]),
            "source": "r10_workspace_{}_v2".format(spec.name),
            "training_group": "r10_workspace_score_unseen",
            "family": domain.family,
            "unit": domain.unit,
            "depth": int(depth),
            "heldout": True,
            "eval_regime": regime.name,
            "numeric_profile": regime.numeric_profile,
            "cell_id": cell_id(regime.name, depth, target_query, domain.family),
            "cell_index": int(cell_index),
            "reference": "{}-{:06d}".format(spec.reference_prefix, index),
            "generation_seed": int(seed),
            "initial": initial,
            "keys": list(domain.keys),
            "operations": operations,
            "query": query,
            "surface": {
                "intro_template": int(intro_template),
                "operation_templates": list(map(int, operation_templates)),
                "query_template": int(query_template),
            },
        }
        signature = program_signature(row)
        if signature is None:
            raise AssertionError("generated row has no program signature")
        prompt_key = normalized(question)
        if prompt_key in seen_prompts or signature in seen_programs:
            continue
        row["prompt_sha256"] = sha256_bytes(question.encode())
        row["program_sha256"] = canonical_hash(signature)
        seen_prompts.add(prompt_key)
        seen_programs.add(signature)
        return row
    raise RuntimeError("could not generate unique row {} for {}".format(index, spec.name))


def validate_count(name: str, count: int, minimum: int, require_capacity: bool) -> None:
    if count < minimum:
        raise ValueError("{} count {} is below {}".format(name, count, minimum))
    cells = len(expected_cells(BOARD_SPECS[name]))
    if count % cells:
        raise ValueError("{} count must be divisible by {} exact cells".format(name, cells))
    if require_capacity:
        partition_rows = count // len(BOARD_SPECS[name].regimes)
        if math.floor(partition_rows * 0.40) < 368:
            raise ValueError(
                "each confirmation partition must yield at least 368 accepted cases "
                "at 40% coverage"
            )


def build_board(name: str, count: int, seed: int) -> list[dict]:
    if name not in CANONICAL_GENERATOR_SEEDS:
        raise ValueError("unknown board {}".format(name))
    expected_seed = CANONICAL_GENERATOR_SEEDS[name]
    if seed != expected_seed:
        raise ValueError(
            "{} seed must be the frozen canonical seed {}".format(name, expected_seed)
        )
    spec = BOARD_SPECS[name]
    cell_size = count // len(expected_cells(spec))
    row_specs = [
        {
            "schedule_id": schedule_id,
            "regime": regime,
            "depth": depth,
            "query": query,
            "domain": domain,
            "cell_index": cell_index,
        }
        for schedule_id, (regime, depth, query, domain, cell_index) in enumerate(
            (regime, depth, query, domain, cell_index)
            for regime in spec.regimes
            for depth in regime.depths
            for query in QUERIES
            for domain in spec.domains
            for cell_index in range(cell_size)
        )
    ]
    strata = tuple(
        (regime.name, depth)
        for regime in spec.regimes
        for depth in regime.depths
    )
    event_totals = {
        stratum: sum(
            row_spec["depth"]
            for row_spec in row_specs
            if (row_spec["regime"].name, row_spec["depth"]) == stratum
        )
        for stratum in strata
    }
    remainder_plan = balanced_remainder_plan(event_totals, seed + 97_003)
    opcode_chunks = {}
    for regime_name, depth in strata:
        rows_in_stratum = sum(
            (row_spec["regime"].name, row_spec["depth"]) == (regime_name, depth)
            for row_spec in row_specs
        )
        stratum_seed = int.from_bytes(
            hashlib.sha256(
                "{}:{}:{}".format(seed, regime_name, depth).encode()
            ).digest()[:8],
            "big",
        )
        stream = balanced_opcode_stream(
            rows_in_stratum * depth,
            random.Random(stratum_seed),
            remainder_plan[(regime_name, depth)],
        )
        opcode_chunks[(regime_name, depth)] = [
            stream[index:index + depth] for index in range(0, len(stream), depth)
        ]
    seen_prompts: set[str] = set()
    seen_programs: set[tuple] = set()
    rows = [None] * len(row_specs)
    # Allocate directional differences first within each frozen stratum because
    # their sign constraint can make a particular opcode chunk incompatible.
    # Cell membership and every row attribute were fixed before this allocation.
    for stratum in strata:
        indices = [
            index for index, row_spec in enumerate(row_specs)
            if (row_spec["regime"].name, row_spec["depth"]) == stratum
        ]
        indices.sort(key=lambda index: (
            not row_specs[index]["query"].startswith("difference"),
            row_specs[index]["schedule_id"],
        ))
        remaining = list(opcode_chunks[stratum])
        for index in indices:
            row_spec = row_specs[index]
            for chunk_index, target_opcodes in enumerate(remaining):
                try:
                    row = make_row(
                        spec=spec,
                        index=index,
                        schedule_id=row_spec["schedule_id"],
                        regime=row_spec["regime"],
                        depth=row_spec["depth"],
                        target_query=row_spec["query"],
                        target_opcodes=target_opcodes,
                        domain=row_spec["domain"],
                        cell_index=row_spec["cell_index"],
                        seed=seed,
                        seen_prompts=seen_prompts,
                        seen_programs=seen_programs,
                    )
                except RuntimeError:
                    continue
                rows[index] = row
                del remaining[chunk_index]
                break
            else:
                raise RuntimeError(
                    "could not assign a compatible opcode chunk to {} row {}".format(name, index)
                )
        if remaining:
            raise AssertionError("unconsumed opcode chunks for stratum {}".format(stratum))
    if any(row is None for row in rows):
        raise AssertionError("board construction left an empty row")
    return rows


class PredecodedExactTable:
    """Expose the exact ALU argmax without recomputing it for every operation."""

    def __init__(self):
        self.targets = transition_basis_targets()

    def argmax(self, dim=-1):
        if dim != -1:
            raise ValueError("exact transition table only supports the categorical axis")
        return self.targets


def exact_table():
    return PredecodedExactTable()


def board_summary(rows, tokenizer, max_tokens: int) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty board")
    spec = BOARD_SPECS[rows[0]["board"]]
    table = exact_table()
    depths = collections.Counter()
    regimes = collections.Counter()
    numeric_profiles = collections.Counter()
    queries = collections.Counter()
    opcodes = collections.Counter()
    stratum_opcodes = collections.defaultdict(collections.Counter)
    families = collections.Counter()
    cells = collections.Counter()
    cell_indices = collections.defaultdict(set)
    token_lengths = []
    oracle_errors = 0
    for row in rows:
        example = compile_example(row, tokenizer)
        if len(example.ids) > max_tokens:
            raise ValueError("{} exceeds {} tokenizer tokens".format(row["reference"], max_tokens))
        oracle = execute_program(
            example.initial_values,
            example.operation_targets,
            example.operation_values,
            example.query_target,
            table,
            width=EXECUTOR_WIDTH,
        )
        oracle_errors += int(oracle != example.answer)
        depth = len(example.operation_targets)
        depths[str(depth)] += 1
        query_name = QUERIES[example.query_target]
        queries[query_name] += 1
        operation_names = [OPCODES[target] for target in example.operation_targets]
        opcodes.update(operation_names)
        stratum_key = "{}|depth={}".format(row["eval_regime"], depth)
        stratum_opcodes[stratum_key].update(operation_names)
        regimes[row["eval_regime"]] += 1
        numeric_profiles[row["numeric_profile"]] += 1
        families[row["family"]] += 1
        actual_cell = cell_id(row["eval_regime"], depth, query_name, row["family"])
        if row["cell_id"] != actual_cell:
            raise ValueError("{} has a false cell identifier".format(row["reference"]))
        cells[actual_cell] += 1
        cell_indices[actual_cell].add(row["cell_index"])
        token_lengths.append(len(example.ids))
    if oracle_errors:
        raise ValueError("generated board has {} oracle errors".format(oracle_errors))
    expected_depth = len(rows) // 4
    expected_query = len(rows) // len(QUERIES)
    if set(depths.values()) != {expected_depth}:
        raise ValueError("generated board is not depth balanced")
    if set(queries.values()) != {expected_query} or set(queries) != set(QUERIES):
        raise ValueError("generated board is not query balanced")
    expected_regime = len(rows) // len(spec.regimes)
    if set(regimes) != {regime.name for regime in spec.regimes} or set(regimes.values()) != {
        expected_regime
    }:
        raise ValueError("generated board is not regime balanced")
    if set(families) != {domain.family for domain in spec.domains} or set(families.values()) != {
        len(rows) // len(spec.domains)
    }:
        raise ValueError("generated board is not family balanced")
    expected_cell_names = set(expected_cells(spec))
    expected_cell_size = len(rows) // len(expected_cell_names)
    if set(cells) != expected_cell_names or set(cells.values()) != {expected_cell_size}:
        raise ValueError("generated board does not have exact regime/depth/query/family cells")
    if any(
        cell_indices[name] != set(range(expected_cell_size))
        for name in expected_cell_names
    ):
        raise ValueError("generated board cell indices are not exact")
    opcode_values = [opcodes[opcode] for opcode in OPCODES]
    if min(opcode_values) <= 0 or max(opcode_values) - min(opcode_values) > 1:
        raise ValueError("generated board is not globally opcode balanced")
    for stratum, counts in stratum_opcodes.items():
        values = [counts[opcode] for opcode in OPCODES]
        if min(values) <= 0 or max(values) - min(values) > 1:
            raise ValueError("stratum {} is not opcode balanced".format(stratum))
    return {
        "rows": len(rows),
        "depths": dict(sorted(depths.items())),
        "regimes": dict(sorted(regimes.items())),
        "numeric_profiles": dict(sorted(numeric_profiles.items())),
        "queries": dict(sorted(queries.items())),
        "opcodes": dict(sorted(opcodes.items())),
        "opcodes_by_regime_depth": {
            stratum: dict(sorted(counts.items()))
            for stratum, counts in sorted(stratum_opcodes.items())
        },
        "families": dict(sorted(families.items())),
        "expected_cell_count": len(expected_cell_names),
        "rows_per_exact_cell": expected_cell_size,
        "exact_cells": dict(sorted(cells.items())),
        "tokenizer_lengths": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "limit": max_tokens,
        },
        "oracle_errors": oracle_errors,
        "normalized_prompt_duplicates": len(rows) - len({normalized(row["question"]) for row in rows}),
        "program_duplicates": len(rows) - len({program_signature(row) for row in rows}),
    }


def board_index(rows, spec: BoardSpec) -> dict:
    phrases = {
        normalized(phrase)
        for domain in spec.domains
        for phrase in (domain.family, *domain.keys, domain.unit)
    }
    return {
        "exact": {normalized(row["question"]) for row in rows},
        "grams": set().union(*(ngrams(row["question"]) for row in rows)),
        "programs": {program_signature(row) for row in rows},
        "phrases": phrases,
    }


def phrase_hits(words: list[str], phrases: set[str]) -> set[str]:
    padded = " {} ".format(" ".join(words))
    return {phrase for phrase in phrases if " {} ".format(phrase) in padded}


def scan_source(path, role: str, indices: dict[str, dict]) -> dict:
    reports = {
        name: {
            "exact_prompt_rows": 0,
            "ngram13_rows": 0,
            "program_rows": 0,
            "novel_domain_phrase_rows": 0,
            "sample_hits": [],
        }
        for name in indices
    }
    rows = 0
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
                question = row["question"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("{}:{} malformed source row: {}".format(path, line_number, error)) from error
            key = normalized(question)
            words = key.split()
            signature = program_signature(row)
            source_grams = {
                " ".join(words[index:index + NGRAM_WIDTH])
                for index in range(max(0, len(words) - NGRAM_WIDTH + 1))
            }
            for name, index in indices.items():
                exact = key in index["exact"]
                gram = bool(source_grams & index["grams"])
                program = signature is not None and signature in index["programs"]
                phrases = phrase_hits(words, index["phrases"])
                reports[name]["exact_prompt_rows"] += int(exact)
                reports[name]["ngram13_rows"] += int(gram)
                reports[name]["program_rows"] += int(program)
                reports[name]["novel_domain_phrase_rows"] += int(bool(phrases))
                if (exact or gram or program or phrases) and len(reports[name]["sample_hits"]) < 12:
                    reports[name]["sample_hits"].append({
                        "line": line_number,
                        "exact": exact,
                        "ngram13": gram,
                        "program": program,
                        "phrases": sorted(phrases),
                    })
    return {
        "role": role,
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "rows_scanned": rows,
        "boards": reports,
    }


def assert_zero_source_hits(source_reports) -> None:
    failures = []
    for source in source_reports:
        for board, report in source["boards"].items():
            for key in (
                "exact_prompt_rows", "ngram13_rows", "program_rows", "novel_domain_phrase_rows",
            ):
                if report[key]:
                    failures.append("{}:{}:{}={}".format(source["role"], board, key, report[key]))
    if failures:
        raise ValueError("source novelty scan failed: " + ", ".join(failures))


def serialize_jsonl(rows) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def exclusive_write(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as destination:
        destination.write(data)


def build_bundle(
    *,
    training_data,
    r5_board,
    tokenizer_path,
    calibration_out,
    confirmation_out,
    manifest_out,
    calibration_count=800,
    confirmation_count=1840,
    calibration_seed=CANONICAL_GENERATOR_SEEDS["calibration"],
    confirmation_seed=CANONICAL_GENERATOR_SEEDS["confirmation"],
    max_tokens=2048,
    minimum_calibration=800,
    minimum_confirmation=1840,
    require_confirmation_capacity=True,
) -> dict:
    training_data = list(training_data)
    if not training_data:
        raise ValueError("at least one training data path is required")
    inputs = [*training_data, r5_board, tokenizer_path]
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError("missing input {}".format(path))
    output_paths = [calibration_out, confirmation_out, manifest_out]
    if len({str(Path(path).resolve()) for path in output_paths}) != len(output_paths):
        raise ValueError("output paths must be distinct")
    for path in output_paths:
        if Path(path).exists():
            raise FileExistsError("refusing existing output {}".format(path))
    supplied_seeds = {
        "calibration": calibration_seed,
        "confirmation": confirmation_seed,
    }
    if supplied_seeds != CANONICAL_GENERATOR_SEEDS:
        raise ValueError(
            "generator seeds must equal the frozen canonical contract {}".format(
                CANONICAL_GENERATOR_SEEDS
            )
        )
    r5_sha256 = sha256_file(r5_board)
    if r5_sha256 != CANONICAL_R5_NOVELTY_BOARD_SHA256:
        raise ValueError(
            "R5 novelty board SHA256 differs from the frozen canonical artifact"
        )
    validate_count("calibration", calibration_count, minimum_calibration, False)
    validate_count(
        "confirmation", confirmation_count, minimum_confirmation,
        require_confirmation_capacity,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    calibration = build_board("calibration", calibration_count, calibration_seed)
    confirmation = build_board("confirmation", confirmation_count, confirmation_seed)
    summaries = {
        "calibration": board_summary(calibration, tokenizer, max_tokens),
        "confirmation": board_summary(confirmation, tokenizer, max_tokens),
    }
    indices = {
        "calibration": board_index(calibration, BOARD_SPECS["calibration"]),
        "confirmation": board_index(confirmation, BOARD_SPECS["confirmation"]),
    }
    cross_exact = indices["calibration"]["exact"] & indices["confirmation"]["exact"]
    cross_grams = indices["calibration"]["grams"] & indices["confirmation"]["grams"]
    cross_programs = indices["calibration"]["programs"] & indices["confirmation"]["programs"]
    if cross_exact or cross_grams or cross_programs:
        raise ValueError(
            "cross-board overlap exact={} gram13={} program={}".format(
                len(cross_exact), len(cross_grams), len(cross_programs),
            )
        )
    source_reports = [
        scan_source(path, "training_data", indices) for path in training_data
    ] + [scan_source(r5_board, "r5_fresh_board", indices)]
    assert_zero_source_hits(source_reports)

    calibration_bytes = serialize_jsonl(calibration)
    confirmation_bytes = serialize_jsonl(confirmation)
    output_records = {
        "calibration": {
            "path": str(Path(calibration_out).resolve()),
            "sha256": sha256_bytes(calibration_bytes),
            "seed": int(calibration_seed),
            "r5_novelty_board_sha256": r5_sha256,
            **summaries["calibration"],
        },
        "confirmation": {
            "path": str(Path(confirmation_out).resolve()),
            "sha256": sha256_bytes(confirmation_bytes),
            "seed": int(confirmation_seed),
            "r5_novelty_board_sha256": r5_sha256,
            "accepted_capacity_at_40_percent_coverage_by_partition": {
                regime.name: math.floor(
                    summaries["confirmation"]["regimes"][regime.name] * 0.40
                )
                for regime in BOARD_SPECS["confirmation"].regimes
            },
            **summaries["confirmation"],
        },
    }
    manifest = {
        "build": "r10_workspace_boards_v2",
        "schema": SCHEMA,
        "cpu_only": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "ready_for_r10_score_run": False,
        "ngram_width": NGRAM_WIDTH,
        "executor_width": EXECUTOR_WIDTH,
        "generation_contract": canonical_generation_contract(),
        "schedule_contract": {
            name: {
                "cells": len(expected_cells(spec)),
                "regimes": {
                    regime.name: {
                        "depths": list(regime.depths),
                        "numeric_profile": regime.numeric_profile,
                        "initial_range_inclusive": list(regime.initial_range),
                        "event_value_range_inclusive": list(regime.value_range),
                        "families": [domain.family for domain in spec.domains],
                        "queries": list(QUERIES),
                    }
                    for regime in spec.regimes
                },
            }
            for name, spec in BOARD_SPECS.items()
        },
        "tokenizer": {
            "path": str(Path(tokenizer_path).resolve()),
            "sha256": sha256_file(tokenizer_path),
            "max_tokens": int(max_tokens),
        },
        "inputs": source_reports,
        "outputs": output_records,
        "cross_board_scan": {
            "exact_prompt_hits": len(cross_exact),
            "ngram13_hits": len(cross_grams),
            "program_hits": len(cross_programs),
        },
        "claim_boundary": BUILD_MANIFEST_CLAIM_BOUNDARY,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    exclusive_write(calibration_out, calibration_bytes)
    exclusive_write(confirmation_out, confirmation_bytes)
    exclusive_write(manifest_out, manifest_bytes)
    for name, path in (("calibration", calibration_out), ("confirmation", confirmation_out)):
        if sha256_file(path) != output_records[name]["sha256"]:
            raise RuntimeError("post-write hash mismatch for {}".format(name))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", action="append", required=True)
    parser.add_argument("--r5-board", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--calibration-out", required=True)
    parser.add_argument("--confirmation-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--calibration-count", type=int, default=800)
    parser.add_argument("--confirmation-count", type=int, default=1840)
    parser.add_argument(
        "--calibration-seed", type=int,
        default=CANONICAL_GENERATOR_SEEDS["calibration"],
    )
    parser.add_argument(
        "--confirmation-seed", type=int,
        default=CANONICAL_GENERATOR_SEEDS["confirmation"],
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    try:
        manifest = build_bundle(
            training_data=args.training_data,
            r5_board=args.r5_board,
            tokenizer_path=args.tokenizer,
            calibration_out=args.calibration_out,
            confirmation_out=args.confirmation_out,
            manifest_out=args.manifest_out,
            calibration_count=args.calibration_count,
            confirmation_count=args.confirmation_count,
            calibration_seed=args.calibration_seed,
            confirmation_seed=args.confirmation_seed,
            max_tokens=args.max_tokens,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "build": manifest["build"],
        "calibration_sha256": manifest["outputs"]["calibration"]["sha256"],
        "confirmation_sha256": manifest["outputs"]["confirmation"]["sha256"],
        "ready_for_r10_score_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
