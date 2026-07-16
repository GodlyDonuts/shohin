"""Generate hash-bound public/oracle data for the ACW hidden-basis falsifier.

Development seeds are public.  Confirmation seed material is retrieved from a
macOS keychain only after model freeze and is never printed or serialized.  The
scored trainer must consume a separate bundle that excludes this module and all
``oracle`` arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FIELD_SIZE = 17
DIMENSION = 3
SOURCE_DIM = 96
EVENT_DIM = 96
EVENTS_PER_ADDRESS = 16
PUBLIC_QUERIES = 24
NEW_QUERIES = 8
TRAIN_HISTORIES = 4096
ADAPTATION_HISTORIES = 1024
EVALUATION_HISTORIES = 2048
TRAIN_MAX_DEPTH = 8
EVALUATION_DEPTHS = (8, 16, 32, 64, 65)
PILOT_SEED = 2026071600
GENERATOR_PROTOCOL = "R12-ACW-HIDDEN-BASIS-v1"
CONFIRMATION_DOMAIN = b"R12-ACW-CONFIRM-v1\x00"
CONFIRMATION_COMMITMENTS = (
    "35102b3974877e8547b9b9c74156c63b71d467820f752301be21721b0f58e9a1",
    "737a6d6a76c3cdbfd07d84c83cfec5491cf13afeb8e077421af789cb652baa7f",
    "0e60eb70f2193ea57710db1f2cf9d6f93cf9b8e310b1b2cf5f4ea2694851854d",
)
KEYCHAIN_SERVICE = "shohin-acw-confirmation-v1"
KEYCHAIN_ACCOUNT = "sairamen"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def confirmation_commitment(seed: bytes) -> str:
    if len(seed) != 32:
        raise ValueError("confirmation seed must contain exactly 32 bytes")
    return sha256_bytes(CONFIRMATION_DOMAIN + seed)


def development_seed_material(seed: int) -> bytes:
    if not 0 <= seed < 2**64:
        raise ValueError("development seed must fit in uint64")
    return b"R12-ACW-DEVELOPMENT-v1\x00" + seed.to_bytes(8, "big")


def _rng(seed_material: bytes, label: str) -> np.random.Generator:
    digest = hashlib.sha256(seed_material + b"\x00" + label.encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def determinant_mod17(matrix: np.ndarray) -> int:
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3 x 3")
    a, b, c = (int(value) for value in matrix[0])
    d, e, f = (int(value) for value in matrix[1])
    g, h, i = (int(value) for value in matrix[2])
    return (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) % FIELD_SIZE


def _invertible_matrix(rng: np.random.Generator) -> np.ndarray:
    while True:
        matrix = rng.integers(0, FIELD_SIZE, size=(3, 3), dtype=np.int16)
        if determinant_mod17(matrix) != 0:
            return matrix.astype(np.int8)


def _normalized_projection(
    rng: np.random.Generator, rows: int, columns: int,
) -> np.ndarray:
    projection = rng.normal(size=(rows, columns)).astype(np.float32)
    norms = np.linalg.norm(projection, axis=1, keepdims=True)
    return projection / np.maximum(norms, np.float32(1e-12))


def _typed_event_vector(event: np.ndarray) -> np.ndarray:
    destination, source, alpha, beta, gamma = (int(value) for value in event)
    vector = np.zeros(2 * DIMENSION + 3 * FIELD_SIZE, dtype=np.float32)
    vector[destination] = 1
    vector[DIMENSION + source] = 1
    offset = 2 * DIMENSION
    for value in (alpha, beta, gamma):
        vector[offset + value] = 1
        offset += FIELD_SIZE
    return vector


def _event_bank(rng: np.random.Generator) -> np.ndarray:
    events: list[tuple[int, int, int, int, int]] = []
    for destination in range(DIMENSION):
        selected: set[tuple[int, int, int, int, int]] = set()
        while len(selected) < EVENTS_PER_ADDRESS:
            source = int(rng.integers(DIMENSION))
            alpha, beta, gamma = (
                int(value) for value in rng.integers(0, FIELD_SIZE, size=3)
            )
            if alpha == 1 and beta == 0 and gamma == 0:
                continue
            selected.add((destination, source, alpha, beta, gamma))
        events.extend(sorted(selected))
    return np.asarray(events, dtype=np.int8)


def _query_bank(
    rng: np.random.Generator,
    count: int,
    *,
    forbidden: Iterable[tuple[int, ...]] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forbidden = {tuple(int(value) for value in row) for row in forbidden}
    while True:
        leading = rng.integers(
            0, FIELD_SIZE, size=(DIMENSION, DIMENSION), dtype=np.int16,
        )
        leading_rows = [tuple(int(value) for value in row) for row in leading]
        if determinant_mod17(leading) != 0 and not (set(leading_rows) & forbidden):
            break
    coefficients = leading_rows
    seen = forbidden | set(coefficients)
    while len(coefficients) < count:
        row = tuple(
            int(value) for value in rng.integers(0, FIELD_SIZE, size=DIMENSION)
        )
        if any(row) and row not in seen:
            coefficients.append(row)
            seen.add(row)
    offsets = rng.integers(0, FIELD_SIZE, size=count, dtype=np.int8)
    permutations = np.stack(
        [rng.permutation(FIELD_SIZE).astype(np.int8) for _ in range(count)],
    )
    return np.asarray(coefficients, dtype=np.int8), offsets, permutations


@dataclass(frozen=True)
class Domain:
    seed_fingerprint: str
    basis: np.ndarray
    source_projection: np.ndarray
    events: np.ndarray
    event_features: np.ndarray
    query_coefficients: np.ndarray
    query_offsets: np.ndarray
    query_permutations: np.ndarray
    new_query_coefficients: np.ndarray
    new_query_offsets: np.ndarray
    new_query_permutations: np.ndarray


def build_domain(seed_material: bytes) -> Domain:
    basis = _invertible_matrix(_rng(seed_material, "basis"))
    source_projection = _normalized_projection(
        _rng(seed_material, "source-projection"), DIMENSION * FIELD_SIZE, SOURCE_DIM,
    )
    events = _event_bank(_rng(seed_material, "events"))
    event_projection = _normalized_projection(
        _rng(seed_material, "event-projection"),
        2 * DIMENSION + 3 * FIELD_SIZE,
        EVENT_DIM,
    )
    event_features = np.stack(
        [_typed_event_vector(event) @ event_projection for event in events],
    ).astype(np.float32)
    queries = _query_bank(_rng(seed_material, "public-queries"), PUBLIC_QUERIES)
    new_queries = _query_bank(
        _rng(seed_material, "new-queries"),
        NEW_QUERIES,
        forbidden=(tuple(int(value) for value in row) for row in queries[0]),
    )
    return Domain(
        seed_fingerprint=sha256_bytes(seed_material),
        basis=basis,
        source_projection=source_projection,
        events=events,
        event_features=event_features,
        query_coefficients=queries[0],
        query_offsets=queries[1],
        query_permutations=queries[2],
        new_query_coefficients=new_queries[0],
        new_query_offsets=new_queries[1],
        new_query_permutations=new_queries[2],
    )


def apply_event(state: np.ndarray, event: np.ndarray) -> np.ndarray:
    destination, source, alpha, beta, gamma = (int(value) for value in event)
    result = state.copy()
    result[destination] = (
        alpha * int(state[destination]) + beta * int(state[source]) + gamma
    ) % FIELD_SIZE
    return result


def state_bucket(seed_material: bytes, state: np.ndarray) -> int:
    canonical = bytes(int(value) for value in state)
    digest = hashlib.sha256(seed_material + b"\x00state\x00" + canonical).digest()
    return int.from_bytes(digest[:8], "big") % 100


def split_name(bucket: int) -> str:
    if not 0 <= bucket < 100:
        raise ValueError("bucket must be in [0,100)")
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "adaptation"
    return "evaluation"


def render_source(domain: Domain, state: np.ndarray) -> np.ndarray:
    hidden = (domain.basis.astype(np.int16) @ state.astype(np.int16)) % FIELD_SIZE
    one_hot = np.zeros(DIMENSION * FIELD_SIZE, dtype=np.float32)
    for coordinate, value in enumerate(hidden):
        one_hot[coordinate * FIELD_SIZE + int(value)] = 1
    return (one_hot @ domain.source_projection).astype(np.float32)


def query_answers(
    state: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    raw = (
        coefficients.astype(np.int16) @ state.astype(np.int16)
        + offsets.astype(np.int16)
    ) % FIELD_SIZE
    return np.asarray(
        [permutations[index, int(value)] for index, value in enumerate(raw)],
        dtype=np.int8,
    )


@dataclass(frozen=True)
class HistorySet:
    source_features: np.ndarray
    source_states: np.ndarray
    event_ids: np.ndarray
    lengths: np.ndarray
    trajectory_states: np.ndarray
    final_states: np.ndarray
    public_answers: np.ndarray
    new_answers: np.ndarray
    visited_buckets: dict[str, int]
    depth_counts: dict[str, int]


def generate_histories(
    domain: Domain,
    seed_material: bytes,
    *,
    count: int,
    target_split: str,
    depths: Iterable[int],
    label: str,
) -> HistorySet:
    depths = tuple(int(depth) for depth in depths)
    if not depths or any(depth < 0 for depth in depths):
        raise ValueError("depths must be nonempty and nonnegative")
    if target_split not in {"train", "adaptation", "evaluation"}:
        raise ValueError("unknown target split")
    rng = _rng(seed_material, label)
    max_depth = max(depths)
    depth_schedule = np.asarray(
        [
            depth
            for index, depth in enumerate(depths)
            for _ in range(count // len(depths) + (index < count % len(depths)))
        ],
        dtype=np.int16,
    )
    if len(depth_schedule) != count:
        raise AssertionError("balanced depth schedule has the wrong length")
    rng.shuffle(depth_schedule)
    sources = np.empty((count, SOURCE_DIM), dtype=np.float32)
    source_states = np.empty((count, DIMENSION), dtype=np.int8)
    event_ids = np.full((count, max_depth), -1, dtype=np.int16)
    lengths = np.empty(count, dtype=np.int16)
    trajectory_states = np.full(
        (count, max_depth + 1, DIMENSION), -1, dtype=np.int8,
    )
    final_states = np.empty((count, DIMENSION), dtype=np.int8)
    public_answers = np.empty((count, PUBLIC_QUERIES), dtype=np.int8)
    new_answers = np.empty((count, NEW_QUERIES), dtype=np.int8)
    visited = {"train": 0, "adaptation": 0, "evaluation": 0}
    accepted = 0
    attempts = 0
    maximum_attempts = max(100_000, count * 2_000)
    while accepted < count:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError("history rejection sampler exceeded its frozen cap")
        source_state = rng.integers(0, FIELD_SIZE, size=DIMENSION, dtype=np.int8)
        if split_name(state_bucket(seed_material, source_state)) != target_split:
            continue
        depth = int(depth_schedule[accepted])
        ids = rng.integers(0, len(domain.events), size=depth, dtype=np.int16)
        state = source_state.copy()
        trajectory = [source_state.copy()]
        for event_id in ids:
            state = apply_event(state, domain.events[int(event_id)])
            trajectory.append(state.copy())
        if split_name(state_bucket(seed_material, state)) != target_split:
            continue
        sources[accepted] = render_source(domain, source_state)
        source_states[accepted] = source_state
        if depth:
            event_ids[accepted, :depth] = ids
        lengths[accepted] = depth
        final_states[accepted] = state
        trajectory_states[accepted, : len(trajectory)] = np.asarray(
            trajectory, dtype=np.int8,
        )
        public_answers[accepted] = query_answers(
            state,
            domain.query_coefficients,
            domain.query_offsets,
            domain.query_permutations,
        )
        new_answers[accepted] = query_answers(
            state,
            domain.new_query_coefficients,
            domain.new_query_offsets,
            domain.new_query_permutations,
        )
        for visited_state in trajectory:
            visited[split_name(state_bucket(seed_material, visited_state))] += 1
        accepted += 1
    return HistorySet(
        source_features=sources,
        source_states=source_states,
        event_ids=event_ids,
        lengths=lengths,
        trajectory_states=trajectory_states,
        final_states=final_states,
        public_answers=public_answers,
        new_answers=new_answers,
        visited_buckets=visited,
        depth_counts={
            str(depth): int(np.count_nonzero(depth_schedule == depth)) for depth in depths
        },
    )


def validate_seed_identity(seed_material: bytes, seed_identity: dict) -> None:
    kind = seed_identity.get("kind")
    if kind == "development":
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("development seed identity has the wrong schema")
        expected = development_seed_material(int(seed_identity["seed"]))
        if not secrets.compare_digest(seed_material, expected):
            raise ValueError("development seed identity does not match seed material")
        return
    if kind == "pilot":
        if seed_identity != {"kind": "pilot", "seed": PILOT_SEED}:
            raise ValueError("pilot seed identity is not the frozen pilot seed")
        if not secrets.compare_digest(seed_material, development_seed_material(PILOT_SEED)):
            raise ValueError("pilot seed material does not match the frozen pilot seed")
        return
    if kind == "confirmation":
        if set(seed_identity) != {"kind", "index", "commitment"}:
            raise ValueError("confirmation seed identity has the wrong schema")
        index = int(seed_identity["index"])
        if not 0 <= index < len(CONFIRMATION_COMMITMENTS):
            raise ValueError("confirmation index is outside the frozen commitments")
        commitment = seed_identity["commitment"]
        if commitment != CONFIRMATION_COMMITMENTS[index]:
            raise ValueError("confirmation identity uses the wrong commitment")
        if not secrets.compare_digest(sha256_bytes(seed_material), commitment):
            raise ValueError("confirmation commitment does not match seed material")
        return
    raise ValueError("unknown seed identity kind")


def initial_train_labels(
    histories: HistorySet, seed_material: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(seed_material, "initial-labels")
    queries = np.empty((len(histories.lengths), 2), dtype=np.int8)
    answers = np.empty_like(queries)
    for index in range(len(histories.lengths)):
        selected = rng.choice(PUBLIC_QUERIES, size=2, replace=False)
        queries[index] = selected
        answers[index] = histories.public_answers[index, selected]
    return queries, answers


def _write_array(root: Path, relative: str, array: np.ndarray, manifest: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    manifest[relative] = {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": file_sha256(path),
    }


def _write_history_set(
    root: Path,
    prefix: str,
    histories: HistorySet,
    arrays: dict,
    *,
    include_truth: bool,
) -> dict:
    _write_array(root, f"{prefix}/source_features.npy", histories.source_features, arrays)
    _write_array(root, f"{prefix}/event_ids.npy", histories.event_ids, arrays)
    _write_array(root, f"{prefix}/lengths.npy", histories.lengths, arrays)
    if include_truth:
        _write_array(root, f"{prefix}/source_states.npy", histories.source_states, arrays)
        _write_array(
            root, f"{prefix}/trajectory_states.npy", histories.trajectory_states, arrays,
        )
        _write_array(root, f"{prefix}/final_states.npy", histories.final_states, arrays)
        _write_array(root, f"{prefix}/public_answers.npy", histories.public_answers, arrays)
        _write_array(root, f"{prefix}/new_answers.npy", histories.new_answers, arrays)
    return histories.visited_buckets


def generate_dataset(
    out: Path,
    seed_material: bytes,
    *,
    seed_identity: dict,
    train_count: int = TRAIN_HISTORIES,
    adaptation_count: int = ADAPTATION_HISTORIES,
    evaluation_count: int = EVALUATION_HISTORIES,
    evaluation_depths: tuple[int, ...] = EVALUATION_DEPTHS,
) -> dict:
    validate_seed_identity(seed_material, seed_identity)
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    domain = build_domain(seed_material)
    arrays: dict[str, dict] = {}
    try:
        _write_array(partial, "public/event_features.npy", domain.event_features, arrays)
        _write_array(
            partial,
            "public/event_addresses.npy",
            domain.events[:, 0].astype(np.int8),
            arrays,
        )
        train = generate_histories(
            domain,
            seed_material,
            count=train_count,
            target_split="train",
            depths=range(TRAIN_MAX_DEPTH + 1),
            label="train-histories",
        )
        initial_queries, initial_answers = initial_train_labels(train, seed_material)
        _write_history_set(
            partial, "public/train", train, arrays, include_truth=False,
        )
        _write_array(partial, "public/train/initial_queries.npy", initial_queries, arrays)
        _write_array(partial, "public/train/initial_answers.npy", initial_answers, arrays)
        _write_array(partial, "oracle/train/final_states.npy", train.final_states, arrays)
        _write_array(partial, "oracle/train/source_states.npy", train.source_states, arrays)
        _write_array(
            partial, "oracle/train/trajectory_states.npy", train.trajectory_states, arrays,
        )
        _write_array(partial, "oracle/train/public_answers.npy", train.public_answers, arrays)

        adaptation = generate_histories(
            domain,
            seed_material,
            count=adaptation_count,
            target_split="adaptation",
            depths=range(TRAIN_MAX_DEPTH + 1),
            label="adaptation-histories",
        )
        _write_history_set(
            partial, "oracle/adaptation", adaptation, arrays, include_truth=True,
        )

        evaluation_visits = {}
        for depth in evaluation_depths:
            histories = generate_histories(
                domain,
                seed_material,
                count=evaluation_count,
                target_split="evaluation",
                depths=(depth,),
                label=f"evaluation-depth-{depth:03d}",
            )
            evaluation_visits[str(depth)] = _write_history_set(
                partial,
                f"oracle/evaluation/depth_{depth:03d}",
                histories,
                arrays,
                include_truth=True,
            )

        _write_array(partial, "oracle/domain/basis.npy", domain.basis, arrays)
        _write_array(partial, "oracle/domain/events.npy", domain.events, arrays)
        _write_array(
            partial, "oracle/domain/query_coefficients.npy",
            domain.query_coefficients, arrays,
        )
        _write_array(
            partial, "oracle/domain/query_offsets.npy", domain.query_offsets, arrays,
        )
        _write_array(
            partial, "oracle/domain/query_permutations.npy",
            domain.query_permutations, arrays,
        )
        _write_array(
            partial, "oracle/domain/new_query_coefficients.npy",
            domain.new_query_coefficients, arrays,
        )
        _write_array(
            partial, "oracle/domain/new_query_offsets.npy",
            domain.new_query_offsets, arrays,
        )
        _write_array(
            partial, "oracle/domain/new_query_permutations.npy",
            domain.new_query_permutations, arrays,
        )
        manifest = {
            "protocol": GENERATOR_PROTOCOL,
            "seed_identity": seed_identity,
            "seed_fingerprint": domain.seed_fingerprint,
            "field_size": FIELD_SIZE,
            "dimension": DIMENSION,
            "source_dim": SOURCE_DIM,
            "event_dim": EVENT_DIM,
            "event_count": len(domain.events),
            "event_address_counts": {
                str(address): int((domain.events[:, 0] == address).sum())
                for address in range(DIMENSION)
            },
            "public_queries": PUBLIC_QUERIES,
            "new_queries": NEW_QUERIES,
            "counts": {
                "train": train_count,
                "adaptation": adaptation_count,
                "evaluation_per_depth": evaluation_count,
            },
            "evaluation_depths": list(evaluation_depths),
            "visited_buckets": {
                "train": train.visited_buckets,
                "adaptation": adaptation.visited_buckets,
                "evaluation": evaluation_visits,
            },
            "depth_counts": {
                "train": train.depth_counts,
                "adaptation": adaptation.depth_counts,
            },
            "arrays": arrays,
        }
        manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        manifest_path = partial / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        partial.replace(out)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def load_confirmation_seed(index: int) -> bytes:
    if not 0 <= index < len(CONFIRMATION_COMMITMENTS):
        raise ValueError("confirmation index is outside the frozen commitments")
    result = subprocess.run(
        [
            "security", "find-generic-password", "-w",
            "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    encoded = json.loads(result.stdout)
    if not isinstance(encoded, list) or len(encoded) != len(CONFIRMATION_COMMITMENTS):
        raise RuntimeError("keychain confirmation payload has the wrong schema")
    seed = bytes.fromhex(encoded[index])
    observed = confirmation_commitment(seed)
    if not secrets.compare_digest(observed, CONFIRMATION_COMMITMENTS[index]):
        raise RuntimeError("confirmation seed does not match its commitment")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--development-seed", type=int)
    group.add_argument("--confirmation-index", type=int)
    group.add_argument("--pilot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pilot:
        seed_material = development_seed_material(PILOT_SEED)
        identity = {"kind": "pilot", "seed": PILOT_SEED}
    elif args.development_seed is not None:
        seed_material = development_seed_material(args.development_seed)
        identity = {"kind": "development", "seed": args.development_seed}
    else:
        seed = load_confirmation_seed(args.confirmation_index)
        seed_material = CONFIRMATION_DOMAIN + seed
        identity = {
            "kind": "confirmation",
            "index": args.confirmation_index,
            "commitment": CONFIRMATION_COMMITMENTS[args.confirmation_index],
        }
    manifest = generate_dataset(args.out, seed_material, seed_identity=identity)
    print(
        f"[acw-generator] kind={identity['kind']} "
        f"payload_sha256={manifest['payload_sha256']}"
    )


if __name__ == "__main__":
    main()
