#!/usr/bin/env python3
"""Generate the immutable DWEPR Stage-A training acquisition.

The only target calls recorded by this generator are ordinary one-bit READ
answers.  Pair labels, canonical edge targets, and witness masks are rebuilt
from those public answers.  Random choices use the frozen counter PRF and
unbiased finite-bank selection only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence

try:
    from .wgrq_residual_oracle import (
        EVENT_CODE_BITS,
        FLIP,
        ROTATE,
        answer_after_rotations,
        apply_word,
        cancellation_gadget_report,
        cancellation_gadgets,
        canonical_access_word,
        event_counts,
        inverse_rotate,
        rotation_word,
        run_stage_a_symbolic_gates,
        state_mask,
    )
except ImportError:  # Direct execution from the pipeline directory.
    from wgrq_residual_oracle import (
        EVENT_CODE_BITS,
        FLIP,
        ROTATE,
        answer_after_rotations,
        apply_word,
        cancellation_gadget_report,
        cancellation_gadgets,
        canonical_access_word,
        event_counts,
        inverse_rotate,
        rotation_word,
        run_stage_a_symbolic_gates,
        state_mask,
    )


SCHEMA = "wgrq_falsifier_v1"
LEDGER_SCHEMA = "wgrq_ordinary_read_call_v1"
REPORT_SCHEMA = "wgrq_falsifier_v1_report"
FROZEN_PRF_SEED = b"R12-WGRQ-DWEPR-STAGE-A-v1"
FROZEN_GENERATOR_SEED = FROZEN_PRF_SEED
PRF_FORMULA = "SHA256(seed || 0x00 || ASCII(domain) || uint64_be(counter))"
TRAINING_SCALES = (4, 6, 8)
LENGTH_BANDS = ("le_2n", "le_8n")
LENGTH_MULTIPLIERS = {"le_2n": 2, "le_8n": 8}
EPISODES_PER_CELL = 3_072
HISTORIES_PER_EPISODE = 4
PROBES_PER_HISTORY = 8
ORDINARY_CALLS_PER_EPISODE = HISTORIES_PER_EPISODE * PROBES_PER_HISTORY
TOTAL_CELLS = len(TRAINING_SCALES) * len(LENGTH_BANDS)
TOTAL_EPISODES = TOTAL_CELLS * EPISODES_PER_CELL
TOTAL_ORDINARY_CALLS = TOTAL_EPISODES * ORDINARY_CALLS_PER_EPISODE
FROZEN_BATCH_SIZE = 64
FROZEN_GENERATION_CONTRACT_SHA256 = "2800857813631c8d2b597f58f3e6b93646a11cb16340948eb8a563aac2e0618f"
FROZEN_TRANSCRIPT_SHA256 = "ae2849db5d57fda36e2e2fd634ce6e1d0f11eaed7fefe8d9ce722f016f28295a"
FROZEN_ORDINARY_CALL_LEDGER_SHA256 = "251d85432d845c31ce64da1adae132fa8df8f6a63b5db744654b519f2413c9e8"
FROZEN_REPORT_SHA256 = "12c1e54f23b27f3a97a86857b723fec3573f5d558b7528e1615c55746899befb"
FROZEN_PAIRED_SEEDS = (
    17011,
    27103,
    38119,
    49201,
    50311,
    61403,
    72503,
    83609,
    94709,
    105019,
    116027,
    127031,
)
HISTORY_ROLES = ("equivalent_a", "equivalent_b", "different_a", "different_b")


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    else:
        rendered = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    return (rendered + "\n").encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prf_block(seed: bytes, domain: str, counter: int) -> bytes:
    """Evaluate the preregistered SHA-256 counter PRF exactly once."""
    if not isinstance(seed, bytes) or not seed:
        raise ValueError("PRF seed must be nonempty bytes")
    if not isinstance(domain, str):
        raise TypeError("PRF domain must be text")
    domain_bytes = domain.encode("ascii")
    if isinstance(counter, bool) or not isinstance(counter, int) or not 0 <= counter < (1 << 64):
        raise ValueError("PRF counter must fit uint64")
    return hashlib.sha256(
        seed + b"\x00" + domain_bytes + counter.to_bytes(8, "big", signed=False)
    ).digest()


@dataclass
class FixedPRF:
    """Domain-separated PRF stream with rejection only for unbiased selection."""

    seed: bytes = FROZEN_PRF_SEED
    counters: dict[str, int] = field(default_factory=dict)
    blocks_used: int = 0
    rejected_blocks: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, bytes) or not self.seed:
            raise ValueError("PRF seed must be nonempty bytes")
        self.seed = bytes(self.seed)

    def next_block(self, domain: str) -> bytes:
        counter = self.counters.get(domain, 0)
        block = prf_block(self.seed, domain, counter)
        self.counters[domain] = counter + 1
        self.blocks_used += 1
        return block

    def randbelow(self, upper: int, domain: str) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise ValueError("finite bank size must be positive")
        space = 1 << 256
        acceptance_limit = space - (space % upper)
        while True:
            candidate = int.from_bytes(self.next_block(domain), "big")
            if candidate < acceptance_limit:
                return candidate % upper
            self.rejected_blocks += 1

    def choose(self, bank: Sequence[object], domain: str):
        if not bank:
            raise ValueError("cannot select from an empty finite bank")
        return bank[self.randbelow(len(bank), domain)]

    def shuffled(self, values: Iterable[object], domain: str) -> list[object]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            swap = self.randbelow(index + 1, domain)
            result[index], result[swap] = result[swap], result[index]
        return result

    def snapshot(self) -> dict[str, object]:
        counter_payload = canonical_json_bytes(
            [[domain, self.counters[domain]] for domain in sorted(self.counters)]
        )
        return {
            "formula": PRF_FORMULA,
            "seed_ascii": self.seed.decode("ascii"),
            "seed_hex": self.seed.hex(),
            "blocks_used": self.blocks_used,
            "rejected_blocks": self.rejected_blocks,
            "domain_count": len(self.counters),
            "counter_ledger_sha256": sha256_bytes(counter_payload),
        }


PRF = FixedPRF


def probe_rotation_bank(n: int) -> tuple[int, ...]:
    """Return n-1 determining probes, redundant R^(n-1), then repeats to eight."""
    if n not in TRAINING_SCALES:
        raise ValueError("the frozen eight-probe training bank is defined for n in {4,6,8}")
    base = list(range(n))
    probes: list[int] = []
    while len(probes) < PROBES_PER_HISTORY:
        probes.extend(base)
    return tuple(probes[:PROBES_PER_HISTORY])


probe_rotations = probe_rotation_bank


def _short_gadgets(n: int) -> dict[str, tuple[str, ...]]:
    gadgets = cancellation_gadgets(n)
    return {
        "canonical_access": (),
        "ff_local_identity": (FLIP, FLIP),
        "fr_order": gadgets["fr_order"],
        "rf_order": gadgets["rf_order"],
    }


def source_gadgets(n: int, length_band: str) -> dict[str, tuple[str, ...]]:
    if n not in TRAINING_SCALES:
        raise ValueError("unsupported training scale")
    if length_band == "le_2n":
        return _short_gadgets(n)
    if length_band == "le_8n":
        return cancellation_gadgets(n)
    raise ValueError("unknown source-length band")


def _final_difference_for_depth(n: int, depth: int) -> int:
    if not 0 <= depth <= n - 2:
        raise ValueError("witness depth is outside 0..n-2")
    if depth == n - 2:
        return 1 << (n - 1)
    return (1 << (depth + 1)) | (1 << (depth + 2))


def _inverse_rotate_by(state: int, rotations: int, n: int) -> int:
    for _ in range(rotations % n):
        state = inverse_rotate(state, n)
    return state


@lru_cache(maxsize=None)
def endpoint_quad_bank(n: int, depth: int, common_rotations: int) -> tuple[tuple[int, int, int, int], ...]:
    """Build a finite direct bank of four distinct pre-gadget endpoints."""
    if n not in TRAINING_SCALES:
        raise ValueError("unsupported training scale")
    half = n // 2
    mask = state_mask(n)
    balanced_states = tuple(state for state in range(mask + 1) if state.bit_count() == half)
    equivalent_pairs = tuple(
        (state, state ^ mask)
        for state in balanced_states
        if state < (state ^ mask)
    )
    final_difference = _final_difference_for_depth(n, depth)
    base_difference = _inverse_rotate_by(final_difference, common_rotations, n)
    non_equivalent_pairs: list[tuple[int, int]] = []
    if depth < n - 2:
        for left in balanced_states:
            right = left ^ base_difference
            if right.bit_count() == half and left < right:
                non_equivalent_pairs.append((left, right))
    else:
        for left in balanced_states:
            right = left ^ base_difference
            if abs(right.bit_count() - half) != 1:
                raise AssertionError("maximum-depth construction lost the parity obstruction")
            non_equivalent_pairs.append((left, right))
    bank = tuple(
        (eq_left, eq_right, different_left, different_right)
        for eq_left, eq_right in equivalent_pairs
        for different_left, different_right in non_equivalent_pairs
        if len({eq_left, eq_right, different_left, different_right}) == 4
    )
    if not bank:
        raise AssertionError("empty endpoint bank for n={} depth={}".format(n, depth))
    return bank


def _gadget_rotations(word: Sequence[str], n: int) -> int:
    return event_counts(word)[ROTATE] % n


def _base_length_ceiling(n: int, depth: int) -> int:
    return n + n // 2 + (1 if depth == n - 2 else 0)


def feasible_gadget_names(n: int, length_band: str, depth: int) -> tuple[str, ...]:
    capacity = LENGTH_MULTIPLIERS[length_band] * n
    ceiling = _base_length_ceiling(n, depth)
    feasible = tuple(
        name
        for name, word in source_gadgets(n, length_band).items()
        if ceiling + len(word) <= capacity
    )
    if not feasible:
        raise AssertionError("no feasible gadget for the declared cell")
    return feasible


def balanced_depth_schedule(n: int, count: int, prf: FixedPRF, domain: str) -> list[int]:
    if count <= 0:
        raise ValueError("episode count must be positive")
    quotient, remainder = divmod(count, n - 1)
    schedule = [depth for depth in range(n - 1) for _ in range(quotient)]
    schedule.extend(range(remainder))
    return [int(value) for value in prf.shuffled(schedule, domain)]


def cell_schedule(
    n: int,
    length_band: str,
    count: int,
    prf: FixedPRF,
) -> list[tuple[int, str]]:
    cell_domain = "schedule.n{}.{}".format(n, length_band)
    depths = balanced_depth_schedule(n, count, prf, cell_domain + ".depths")
    gadget_orders: dict[int, list[str]] = {}
    occurrences: Counter[int] = Counter()
    result: list[tuple[int, str]] = []
    for depth in depths:
        if depth not in gadget_orders:
            names = feasible_gadget_names(n, length_band, depth)
            gadget_orders[depth] = [
                str(value)
                for value in prf.shuffled(names, cell_domain + ".depth{}.gadgets".format(depth))
            ]
        order = gadget_orders[depth]
        gadget = order[occurrences[depth] % len(order)]
        occurrences[depth] += 1
        result.append((depth, gadget))
    return result


def _identity_padding(
    n: int,
    length: int,
    prf: FixedPRF,
    domain: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    if length < 0 or length % 2:
        raise ValueError("identity padding length must be a nonnegative even number")
    rotation_block_choices = tuple(range(length // n + 1))
    rotation_blocks = int(prf.choose(rotation_block_choices, domain + ".rotation_block_count"))
    ff_blocks = (length - rotation_blocks * n) // 2
    blocks = ["ff"] * ff_blocks + ["rn"] * rotation_blocks
    blocks = [str(value) for value in prf.shuffled(blocks, domain + ".block_order")]
    word: list[str] = []
    for block in blocks:
        word.extend((FLIP, FLIP) if block == "ff" else rotation_word(n))
    result = tuple(word)
    if len(result) != length or apply_word(0, result, n) != 0:
        raise AssertionError("identity padding construction failed")
    return result, {"ff_blocks": ff_blocks, "rn_blocks": rotation_blocks}


def _padding_lengths(
    n: int,
    length_band: str,
    base_max: int,
    gadget_length: int,
) -> tuple[int, ...]:
    capacity = LENGTH_MULTIPLIERS[length_band] * n
    values = tuple(
        padding
        for padding in range(0, capacity - base_max - gadget_length + 1, 2)
        if length_band == "le_2n" or base_max + gadget_length + padding > 2 * n
    )
    if not values:
        raise AssertionError("no identity padding fits the declared length band")
    return values


def _history_digest(events: Sequence[str]) -> str:
    return sha256_bytes("".join(events).encode("ascii"))


def _one_hot(index: int, size: int) -> list[int]:
    if not 0 <= index < size:
        raise ValueError("one-hot index out of range")
    return [int(position == index) for position in range(size)]


def _episode_id(n: int, length_band: str, cell_episode_index: int) -> str:
    compact_band = length_band.replace("_", "")
    return "wgrq-train-n{:02d}-{}-{:06d}".format(n, compact_band, cell_episode_index)


def make_episode(
    *,
    n: int,
    length_band: str,
    cell_episode_index: int,
    global_episode_index: int,
    depth: int,
    gadget_name: str,
    prf: FixedPRF,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Construct one four-history episode and its exact 32-call ledger."""
    if n not in TRAINING_SCALES or length_band not in LENGTH_BANDS:
        raise ValueError("episode is outside the frozen training cells")
    gadgets = source_gadgets(n, length_band)
    if gadget_name not in feasible_gadget_names(n, length_band, depth):
        raise ValueError("gadget does not fit the declared depth/length cell")
    gadget = gadgets[gadget_name]
    episode_id = _episode_id(n, length_band, cell_episode_index)
    episode_domain = "episode.n{}.{}.{}".format(n, length_band, cell_episode_index)
    quad_bank = endpoint_quad_bank(n, depth, _gadget_rotations(gadget, n))
    endpoints = tuple(
        int(value)
        for value in prf.choose(quad_bank, episode_domain + ".endpoint_quad")
    )
    base_words = tuple(canonical_access_word(state, n) for state in endpoints)
    base_max = max(len(word) for word in base_words)
    padding_bank = _padding_lengths(n, length_band, base_max, len(gadget))
    padding_length = int(prf.choose(padding_bank, episode_domain + ".padding_length"))
    padding, padding_plan = _identity_padding(
        n,
        padding_length,
        prf,
        episode_domain + ".padding",
    )
    common_suffix = tuple(gadget) + padding
    words = tuple(tuple(word) + common_suffix for word in base_words)
    capacity = LENGTH_MULTIPLIERS[length_band] * n
    if any(len(word) > capacity for word in words):
        raise AssertionError("source history exceeded its frozen length band")
    if length_band == "le_8n" and max(map(len, words)) <= 2 * n:
        raise AssertionError("long-band episode did not leave the short band")
    physical_endpoints = tuple(apply_word(0, word, n) for word in words)
    probes = probe_rotation_bank(n)
    first_call_id = global_episode_index * ORDINARY_CALLS_PER_EPISODE
    histories: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    for history_index, (role, word, endpoint) in enumerate(
        zip(HISTORY_ROLES, words, physical_endpoints)
    ):
        digest = _history_digest(word)
        history_probes: list[dict[str, object]] = []
        for probe_index, rotations in enumerate(probes):
            call_id = first_call_id + history_index * PROBES_PER_HISTORY + probe_index
            answer = answer_after_rotations(endpoint, rotations, n)
            continuation = list(rotation_word(rotations))
            history_probes.append(
                {
                    "probe_index": probe_index,
                    "continuation_rotations": rotations,
                    "continuation": continuation,
                    "answer": answer,
                    "oracle_call_id": call_id,
                }
            )
            ledger.append(
                {
                    "schema": LEDGER_SCHEMA,
                    "call_id": call_id,
                    "episode_id": episode_id,
                    "global_episode_index": global_episode_index,
                    "history_index": history_index,
                    "history_sha256": digest,
                    "probe_index": probe_index,
                    "continuation_rotations": rotations,
                    "call_kind": "READ",
                    "returned_bits": 1,
                    "answer": answer,
                }
            )
        canonical_bits = [history_probes[index]["answer"] for index in range(n - 1)]
        histories.append(
            {
                "history_index": history_index,
                "history_id": "{}:h{}".format(episode_id, history_index),
                "role": role,
                "events": list(word),
                "source_length": len(word),
                "event_counts": event_counts(word),
                "history_sha256": digest,
                "probes": history_probes,
                "canonical_edge_bits_from_public_answers": canonical_bits,
            }
        )
    signatures = [tuple(history["canonical_edge_bits_from_public_answers"]) for history in histories]
    equivalence_matrix = [
        [int(left_signature == right_signature) for right_signature in signatures]
        for left_signature in signatures
    ]
    if equivalence_matrix[0][1] != 1 or equivalence_matrix[2][3] != 0:
        raise AssertionError("constructed pair roles disagree with public answers")
    differing_probe_indices = [
        probe_index
        for probe_index in range(PROBES_PER_HISTORY)
        if histories[2]["probes"][probe_index]["answer"]
        != histories[3]["probes"][probe_index]["answer"]
    ]
    if not differing_probe_indices:
        raise AssertionError("non-equivalent pair has no public distinguishing answer")
    first_distinguishing_probe_index = differing_probe_indices[0]
    if probes[first_distinguishing_probe_index] != depth:
        raise AssertionError("constructed pair missed its declared shortest witness")
    witness_mask = _one_hot(first_distinguishing_probe_index, PROBES_PER_HISTORY)
    uniform_probe_index = prf.randbelow(
        PROBES_PER_HISTORY,
        episode_domain + ".uniform_probe_index",
    )
    counts_by_history = [history["event_counts"] for history in histories]
    lengths = [int(history["source_length"]) for history in histories]
    parity_obstruction = depth == n - 2
    all_event_counts_matched = all(counts == counts_by_history[0] for counts in counts_by_history)
    all_lengths_matched = len(set(lengths)) == 1
    if parity_obstruction:
        flip_parities = [int(counts[FLIP]) % 2 for counts in counts_by_history[2:4]]
        if flip_parities[0] == flip_parities[1] or all_event_counts_matched:
            raise AssertionError("maximum-depth event-count parity obstruction disappeared")
    elif not all_event_counts_matched or not all_lengths_matched:
        raise AssertionError("a balanceable episode was not event/length matched")
    episode = {
        "schema": SCHEMA,
        "episode_id": episode_id,
        "split": "training",
        "global_episode_index": global_episode_index,
        "batch_index": global_episode_index // FROZEN_BATCH_SIZE,
        "batch_offset": global_episode_index % FROZEN_BATCH_SIZE,
        "cell": {
            "n": n,
            "length_band": length_band,
            "source_length_ceiling": capacity,
            "cell_episode_index": cell_episode_index,
        },
        "event_code_bits": {event: list(EVENT_CODE_BITS[event]) for event in (ROTATE, FLIP)},
        "gadget": {
            "name": gadget_name,
            "events": list(gadget),
            "event_counts": event_counts(gadget),
            "common_identity_padding_events": list(padding),
            "padding_plan": padding_plan,
        },
        "probe_rotations": list(probes),
        "histories": histories,
        "equivalence_label_matrix": equivalence_matrix,
        "pairs": {
            "equivalent": {
                "history_indices": [0, 1],
                "label": 1,
            },
            "non_equivalent": {
                "history_indices": [2, 3],
                "label": 0,
                "shortest_witness_depth": depth,
                "first_distinguishing_probe_index": first_distinguishing_probe_index,
                "first_distinguishing_witness_mask": witness_mask,
            },
        },
        "first_distinguishing_witness_mask": witness_mask,
        "uniform_probe_index": uniform_probe_index,
        "uniform_probe_mask": _one_hot(uniform_probe_index, PROBES_PER_HISTORY),
        "balance": {
            "declared_pair_labels": {"equivalent": 1, "non_equivalent": 1},
            "all_source_lengths_matched": all_lengths_matched,
            "all_event_counts_matched": all_event_counts_matched,
            "maximum_depth_flip_parity_obstruction": parity_obstruction,
        },
        "oracle_call_span": {
            "first_call_id": first_call_id,
            "last_call_id": first_call_id + ORDINARY_CALLS_PER_EPISODE - 1,
            "ordinary_one_bit_read_calls": ORDINARY_CALLS_PER_EPISODE,
        },
    }
    if len(ledger) != ORDINARY_CALLS_PER_EPISODE:
        raise AssertionError("episode ordinary-call ledger is not exactly 32 rows")
    return episode, ledger


def iter_cell_episodes(
    n: int,
    length_band: str,
    *,
    count: int = EPISODES_PER_CELL,
    global_episode_offset: int = 0,
    prf: FixedPRF | None = None,
) -> Iterator[tuple[dict[str, object], list[dict[str, object]]]]:
    prf = prf or FixedPRF()
    schedule = cell_schedule(n, length_band, count, prf)
    for cell_episode_index, (depth, gadget_name) in enumerate(schedule):
        yield make_episode(
            n=n,
            length_band=length_band,
            cell_episode_index=cell_episode_index,
            global_episode_index=global_episode_offset + cell_episode_index,
            depth=depth,
            gadget_name=gadget_name,
            prf=prf,
        )


def generate_preview(
    *,
    episodes_per_cell: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Generate a deterministic in-memory prefix for focused tests only."""
    if episodes_per_cell <= 0:
        raise ValueError("preview count must be positive")
    prf = FixedPRF()
    episodes: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    global_offset = 0
    for n in TRAINING_SCALES:
        for length_band in LENGTH_BANDS:
            for episode, calls in iter_cell_episodes(
                n,
                length_band,
                count=episodes_per_cell,
                global_episode_offset=global_offset,
                prf=prf,
            ):
                episodes.append(episode)
                ledger.extend(calls)
            global_offset += episodes_per_cell
    return episodes, ledger, prf.snapshot()


def records_sha256(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_bytes(record))
    return digest.hexdigest()


def _new_cell_stats(n: int, length_band: str) -> dict[str, object]:
    return {
        "n": n,
        "length_band": length_band,
        "source_length_ceiling": LENGTH_MULTIPLIERS[length_band] * n,
        "episodes": 0,
        "ordinary_one_bit_read_calls": 0,
        "witness_depths": Counter(),
        "gadgets": Counter(),
        "source_lengths": Counter(),
        "probe_rotations": Counter(),
        "answers": Counter(),
        "gadget_pair_labels": defaultdict(Counter),
        "event_count_matched_episodes": 0,
        "length_matched_episodes": 0,
        "parity_obstruction_episodes": 0,
    }


def _update_cell_stats(
    stats: dict[str, object],
    episode: Mapping[str, object],
    calls: Sequence[Mapping[str, object]],
) -> None:
    stats["episodes"] = int(stats["episodes"]) + 1
    stats["ordinary_one_bit_read_calls"] = int(stats["ordinary_one_bit_read_calls"]) + len(calls)
    depth = int(episode["pairs"]["non_equivalent"]["shortest_witness_depth"])
    gadget = str(episode["gadget"]["name"])
    stats["witness_depths"][depth] += 1
    stats["gadgets"][gadget] += 1
    stats["gadget_pair_labels"][gadget]["equivalent"] += 1
    stats["gadget_pair_labels"][gadget]["non_equivalent"] += 1
    for history in episode["histories"]:
        stats["source_lengths"][int(history["source_length"])] += 1
    for rotation in episode["probe_rotations"]:
        stats["probe_rotations"][int(rotation)] += HISTORIES_PER_EPISODE
    for call in calls:
        stats["answers"][int(call["answer"])] += 1
    balance = episode["balance"]
    stats["event_count_matched_episodes"] += int(balance["all_event_counts_matched"])
    stats["length_matched_episodes"] += int(balance["all_source_lengths_matched"])
    stats["parity_obstruction_episodes"] += int(balance["maximum_depth_flip_parity_obstruction"])


def _finalize_cell_stats(stats: Mapping[str, object]) -> dict[str, object]:
    episodes = int(stats["episodes"])
    source_lengths: Counter[int] = stats["source_lengths"]
    witness_depths: Counter[int] = stats["witness_depths"]
    gadgets: Counter[str] = stats["gadgets"]
    labels = stats["gadget_pair_labels"]
    return {
        "n": int(stats["n"]),
        "length_band": str(stats["length_band"]),
        "source_length_ceiling": int(stats["source_length_ceiling"]),
        "episodes": episodes,
        "histories": episodes * HISTORIES_PER_EPISODE,
        "ordinary_one_bit_read_calls": int(stats["ordinary_one_bit_read_calls"]),
        "pair_labels": {"equivalent": episodes, "non_equivalent": episodes},
        "witness_depth_counts": {
            str(depth): witness_depths[depth] for depth in sorted(witness_depths)
        },
        "witness_depth_division_remainder": episodes % (int(stats["n"]) - 1),
        "gadget_counts": {name: gadgets[name] for name in sorted(gadgets)},
        "gadget_pair_label_counts": {
            name: {
                "equivalent": labels[name]["equivalent"],
                "non_equivalent": labels[name]["non_equivalent"],
            }
            for name in sorted(labels)
        },
        "source_length_counts": {
            str(length): source_lengths[length] for length in sorted(source_lengths)
        },
        "minimum_source_length": min(source_lengths),
        "maximum_source_length": max(source_lengths),
        "probe_rotation_call_counts": {
            str(rotation): stats["probe_rotations"][rotation]
            for rotation in sorted(stats["probe_rotations"])
        },
        "answer_counts": {
            str(answer): stats["answers"][answer] for answer in sorted(stats["answers"])
        },
        "event_count_matched_episodes": int(stats["event_count_matched_episodes"]),
        "length_matched_episodes": int(stats["length_matched_episodes"]),
        "parity_obstruction_episodes": int(stats["parity_obstruction_episodes"]),
    }


def generation_contract(symbolic_gates: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "preregistration": "R12_WGRQ_CPU_PREREG.md",
        "prf": {
            "formula": PRF_FORMULA,
            "seed_ascii": FROZEN_PRF_SEED.decode("ascii"),
            "seed_hex": FROZEN_PRF_SEED.hex(),
            "selection": "unbiased finite-bank rejection sampling only",
        },
        "training_scales": list(TRAINING_SCALES),
        "length_bands": [
            {"name": band, "maximum_n_multiple": LENGTH_MULTIPLIERS[band]}
            for band in LENGTH_BANDS
        ],
        "episodes_per_cell": EPISODES_PER_CELL,
        "total_episodes": TOTAL_EPISODES,
        "histories_per_episode": HISTORIES_PER_EPISODE,
        "probes_per_history": PROBES_PER_HISTORY,
        "ordinary_calls_per_episode": ORDINARY_CALLS_PER_EPISODE,
        "total_ordinary_one_bit_read_calls": TOTAL_ORDINARY_CALLS,
        "probe_rotation_banks": {
            str(n): list(probe_rotation_bank(n)) for n in TRAINING_SCALES
        },
        "history_role_order": list(HISTORY_ROLES),
        "batch_size": FROZEN_BATCH_SIZE,
        "paired_initialization_order_seeds": list(FROZEN_PAIRED_SEEDS),
        "cancellation_controls": {
            str(n): cancellation_gadget_report(n) for n in TRAINING_SCALES
        },
        "symbolic_gate_schema": symbolic_gates["schema"],
        "symbolic_gate_scales": [scale["n"] for scale in symbolic_gates["scales"]],
        "no_model_dependent_mining": True,
        "no_target_dependent_rejection": True,
        "no_reseeding_or_seed_search": True,
    }


def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def preflight_immutable_outputs(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    outputs = tuple(Path(path) for path in paths)
    if len({str(path.expanduser().absolute()) for path in outputs}) != len(outputs):
        raise ValueError("transcript, ledger, and report outputs must be distinct")
    occupied = [path for path in outputs if path.exists() or _partial_path(path).exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite immutable WGRQ artifact: {}".format(occupied[0])
        )
    return outputs


@dataclass
class _HashedJsonlWriter:
    output: BinaryIO
    digest: object = field(default_factory=hashlib.sha256)
    rows: int = 0
    byte_count: int = 0

    def write(self, record: Mapping[str, object]) -> None:
        payload = canonical_json_bytes(record)
        self.output.write(payload)
        self.digest.update(payload)
        self.rows += 1
        self.byte_count += len(payload)

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def generate_artifacts(
    transcript_out: str | Path,
    ledger_out: str | Path,
    report_out: str | Path,
) -> dict[str, object]:
    """Generate the one frozen production acquisition and refuse replacement."""
    transcript_path, ledger_path, report_path = preflight_immutable_outputs(
        (transcript_out, ledger_out, report_out)
    )
    symbolic_gates = run_stage_a_symbolic_gates()
    if not symbolic_gates["passed"]:
        raise RuntimeError("symbolic gate rejected WGRQ generation")
    contract = generation_contract(symbolic_gates)
    contract_sha256 = sha256_bytes(canonical_json_bytes(contract))
    if contract_sha256 != FROZEN_GENERATION_CONTRACT_SHA256:
        raise RuntimeError("generation contract differs from its frozen SHA-256")
    for path in (transcript_path, ledger_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    transcript_partial = _partial_path(transcript_path)
    ledger_partial = _partial_path(ledger_path)
    report_partial = _partial_path(report_path)
    created_partials: list[Path] = []
    try:
        prf = FixedPRF()
        cell_stats: list[dict[str, object]] = []
        global_episode_index = 0
        with ExitStack() as stack:
            transcript_file = stack.enter_context(transcript_partial.open("xb"))
            created_partials.append(transcript_partial)
            ledger_file = stack.enter_context(ledger_partial.open("xb"))
            created_partials.append(ledger_partial)
            transcript_writer = _HashedJsonlWriter(transcript_file)
            ledger_writer = _HashedJsonlWriter(ledger_file)
            for n in TRAINING_SCALES:
                for length_band in LENGTH_BANDS:
                    stats = _new_cell_stats(n, length_band)
                    for episode, calls in iter_cell_episodes(
                        n,
                        length_band,
                        count=EPISODES_PER_CELL,
                        global_episode_offset=global_episode_index,
                        prf=prf,
                    ):
                        if episode["global_episode_index"] != global_episode_index:
                            raise AssertionError("episode order ledger is discontinuous")
                        transcript_writer.write(episode)
                        for call in calls:
                            expected_call = ledger_writer.rows
                            if call["call_id"] != expected_call:
                                raise AssertionError("ordinary-call ledger is discontinuous")
                            ledger_writer.write(call)
                        _update_cell_stats(stats, episode, calls)
                        global_episode_index += 1
                    cell_stats.append(_finalize_cell_stats(stats))
            transcript_file.flush()
            ledger_file.flush()
            os.fsync(transcript_file.fileno())
            os.fsync(ledger_file.fileno())
        if transcript_writer.rows != TOTAL_EPISODES:
            raise AssertionError("training transcript does not contain exactly 18,432 episodes")
        if ledger_writer.rows != TOTAL_ORDINARY_CALLS:
            raise AssertionError("ordinary-call ledger does not contain exactly 589,824 calls")
        if transcript_writer.hexdigest() != FROZEN_TRANSCRIPT_SHA256:
            raise RuntimeError("training transcript differs from its frozen SHA-256")
        if ledger_writer.hexdigest() != FROZEN_ORDINARY_CALL_LEDGER_SHA256:
            raise RuntimeError("ordinary-call ledger differs from its frozen SHA-256")
        parity_episodes = sum(cell["parity_obstruction_episodes"] for cell in cell_stats)
        report = {
            "schema": REPORT_SCHEMA,
            "passed": True,
            "generation_contract": contract,
            "symbolic_gates": symbolic_gates,
            "cells": cell_stats,
            "totals": {
                "cells": TOTAL_CELLS,
                "episodes": transcript_writer.rows,
                "histories": transcript_writer.rows * HISTORIES_PER_EPISODE,
                "ordinary_one_bit_read_calls": ledger_writer.rows,
                "returned_answer_bits": ledger_writer.rows,
                "batches": TOTAL_EPISODES // FROZEN_BATCH_SIZE,
            },
            "frozen_call_ledger": {
                "schema": LEDGER_SCHEMA,
                "first_call_id": 0,
                "last_call_id": TOTAL_ORDINARY_CALLS - 1,
                "rows": ledger_writer.rows,
                "one_bit_read_calls": ledger_writer.rows,
                "model_dependent_calls": 0,
                "equivalence_oracle_calls": 0,
                "counterexample_oracle_calls": 0,
            },
            "balance": {
                "pair_labels_per_episode": {"equivalent": 1, "non_equivalent": 1},
                "depth_stratification_rule": "floor/ceiling balanced then fixed-PRF shuffled",
                "gadget_stratification_rule": "balanced within depth over every feasible gadget",
                "maximum_depth_flip_parity_obstruction": {
                    "affected_episodes": parity_episodes,
                    "reported_separately": True,
                    "reason": (
                        "At shortest-witness depth n-2, the edge difference is supported on "
                        "n-2 and n-1, so the physical difference has odd Hamming parity. "
                        "Rotation preserves parity and each F toggles it; the two histories "
                        "therefore cannot have equal F-event-count parity."
                    ),
                },
            },
            "prf_ledger": prf.snapshot(),
            "artifacts": {
                "transcript": {
                    "schema": SCHEMA,
                    "rows": transcript_writer.rows,
                    "bytes": transcript_writer.byte_count,
                    "sha256": transcript_writer.hexdigest(),
                },
                "ordinary_call_ledger": {
                    "schema": LEDGER_SCHEMA,
                    "rows": ledger_writer.rows,
                    "bytes": ledger_writer.byte_count,
                    "sha256": ledger_writer.hexdigest(),
                },
            },
            "hashes": {
                "generation_contract_sha256": contract_sha256,
                "transcript_sha256": transcript_writer.hexdigest(),
                "ordinary_call_ledger_sha256": ledger_writer.hexdigest(),
            },
        }
        report_payload = canonical_json_bytes(report, pretty=True)
        if sha256_bytes(report_payload) != FROZEN_REPORT_SHA256:
            raise RuntimeError("generation report differs from its frozen SHA-256")
        with report_partial.open("xb") as report_file:
            created_partials.append(report_partial)
            report_file.write(report_payload)
            report_file.flush()
            os.fsync(report_file.fileno())
        os.replace(transcript_partial, transcript_path)
        created_partials.remove(transcript_partial)
        os.replace(ledger_partial, ledger_path)
        created_partials.remove(ledger_partial)
        os.replace(report_partial, report_path)
        created_partials.remove(report_partial)
        result = dict(report)
        result["report_sha256"] = sha256_bytes(report_payload)
        return result
    except BaseException:
        for partial in created_partials:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-out", "--out", dest="transcript_out", required=True)
    parser.add_argument("--ledger-out", required=True)
    parser.add_argument("--report-out", "--report", dest="report_out", required=True)
    args = parser.parse_args(argv)
    try:
        report = generate_artifacts(args.transcript_out, args.ledger_out, args.report_out)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "passed": report["passed"],
                "episodes": report["totals"]["episodes"],
                "ordinary_one_bit_read_calls": report["totals"]["ordinary_one_bit_read_calls"],
                "hashes": report["hashes"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
